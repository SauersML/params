"""RL environment E's verifier as a service (#2951 graph oracle; edit_env.py): persistent processes, each holding vpd4l
and an edit_env.Verifier, behind HTTP, so an RL stack's reward function posts a goal's answers and gets their scores.
A goal's answers always go to the same process; a process scores together every queued answer to the same (goal, part).

  serve_edits.py [--port $PORT or 8766] [--workers 2] [--device cuda] [--goals GOALS.jsonl]
  POST /score {"task": GOAL_ID, "answers": [REPLY, ...], "part": "score"}
      -> {"scores": [{"valid", "unparsed", "edit", "removal", "controls_kept", "collateral", "text_kl", "pile_kl",
                      "reward", "instances", "controls", "ops", "feedback", "source"}, ...]}
  A reply is the oracle's whole answer; its "scale" and "update" lines are the edit (edit_env.parse), an answer with
  none scoring as the empty edit (reward 0). "feedback" reports the measurement in a sentence (a revising episode's
  second turn). "source" is the scorer's source revision (the pod's SOURCE file, edits_scorer_pod.sh; None without
  one): tinker_train rl --expect-source stops a run scored by another revision.
  GET /health -> {"workers", "alive", "pending"}
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import serve_scores  # noqa: E402

GOALS = Path.home() / "mpd-data/graph_oracle/edits/goals.jsonl"


def feedback(s: dict) -> str:
    if not s["valid"]:
        return "No operation parsed: the model was not edited."
    return (f"The edit removed the prediction at {s['removal']:.0%} of held-out instances, the prediction after other "
            f"tokens stayed at {s['controls_kept']:.0%} of controls, and the model's other predictions moved by "
            f"{s['collateral']:.2e} bits per token (reward {s['reward']:+.3f}).")


CHUNK = int(os.environ.get("EDITS_CHUNK", "4"))  # answers scored together per verifier call
RECYCLE = int(os.environ.get("EDITS_RECYCLE", "4"))  # goal groups a process scores before it is replaced (one RL step's)


def free_device() -> None:
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def work(device: str | None, goals: str, inbox, outbox) -> None:
    """One scoring process: take a request, then every request already queued, score them grouped by (goal, part), and
    answer each with its own scores in order."""
    import edit_env

    _, ver = edit_env.load(device)
    src = HERE.parents[2] / "SOURCE"  # the pod's "python source <rev>" (edits_scorer_pod.sh)
    source = src.read_text().split()[-1] if src.exists() else None
    prefs = {r["id"]: r["pref"] for r in map(json.loads, Path(goals).read_text().splitlines())}
    served = 0
    while True:
        if served >= RECYCLE and inbox.empty():  # a fresh process (the pool starts it again): MPS memory grew to 24.5 GiB
            outbox.close()  # over a night of steps although each call was freed; exit only with nothing queued, so none is
            outbox.join_thread()  # lost, after the answers already put have gone out
            os._exit(0)
        batch = [inbox.get()]
        while True:
            try:
                batch.append(inbox.get_nowait())
            except queue.Empty:
                break
        groups = {}
        for rid, req in batch:
            groups.setdefault((req["task"], req.get("part", "score")), []).append((rid, list(req["answers"])))
        for (gid, part), reqs in groups.items():
            try:
                if part not in ("val", "score"):
                    raise ValueError(f"part {part!r}: val or score")
                answers = [a for _, ans in reqs for a in ans]
                got = []
                for k in range(0, len(answers), CHUNK):  # a few answers at a time, the device cache freed after each:
                    got += ver.answers(prefs[gid], answers[k:k + CHUNK], part)  # 16 at once reached 28 GiB on MPS
                    free_device()
                for s in got:
                    s["feedback"], s["source"] = feedback(s), source
                err = None
            except Exception as e:  # a bad request answers with its error; the process keeps serving
                got, err = None, f"{type(e).__name__}: {e}"
            at = 0
            for rid, ans in reqs:
                outbox.put((rid, got[at:at + len(ans)] if got is not None else None, err))
                at += len(ans)
            served += 1


class Pool(serve_scores.Pool):
    """serve_scores' process pool (requests routed by "task", dead processes restarted) running edit scorers."""

    def start(self, inbox):
        p = self.ctx.Process(target=work, args=(*self.args, inbox, self.out), daemon=True)
        p.start()
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8766")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device")
    ap.add_argument("--goals", default=str(GOALS))
    a = ap.parse_args()
    pool = Pool(a.workers, a.device, a.goals)

    def stop(*_):  # SIGTERM: the scoring processes go too (only a normal exit ends daemon processes)
        for p in pool.procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"workers": len(pool.procs), "alive": sum(p.is_alive() for p in pool.procs), "pending": len(pool.waiting)})
            else:
                self.reply(404, {"error": "GET /health only"})

        def do_POST(self):
            if self.path != "/score":
                return self.reply(404, {"error": "POST /score only"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(200, {"scores": pool.submit(req).result()})
            except Exception as e:
                self.reply(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    print(f"serve_edits: {a.workers} scoring processes, http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
