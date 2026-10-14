"""The component-answer reward as a service (#2951 graph oracle): scoring processes on the GPU, each holding a
goal_eval.GoalEval, behind HTTP, so an RL stack's reward function (prime-rl's verifiers environment,
rl/prime/components_oracle) posts the answers of a group and gets one reward each. A process takes a request and every
request already queued, and scores each text's answers with one components.reward call per text; the text's
goal_eval.Text is kept for the group's later requests. A text's requests always go to the same process.

  serve_components.py ROWS.jsonl [--port 8767] [--workers 2] [--device cuda] [--teacher DIR]
                      [--reader http://127.0.0.1:8000/v1/completions --reader-base Qwen/Qwen3-4B]
                      [--edit-teach TEACH.jsonl ...] [--variables]
  POST /score {"task": TASK_ID, "answers": [TEXT, ...], "e1": [bits, ...], "tokens": [int, ...], "lines": bool}
      (e1 and tokens optional: each answer's tokens after its thinking, charged past the answer cap by components.overflow)
      -> {"scores": [{"reward", "valid", "semantics", "source", "environment", "donor_pool" (the scorer's GoalEval semantics,
                      source revision, Python environment and far-donor pool: tinker_train rl checks them),
                      "phi", "phi_tau", "g", "faithful", "bits", "compact", "D", "Q", "E2g", "E4x",
                      "E1", "C", "C_rl", "N", "F1", "F3", "E2", "E3", "E4", "nodes", "edges", "tok_edges", "malformed",
                      "dropped_nodes", "dropped_edges", "finals", "tokens", "overflow", "lines"}, ...]}  ("lines": each answer line's
                      credit, [[line, c], ...], with "lines": true: RL's per-line advantages)
  GET /health -> {"workers", "alive", "pending"}

The reward is components.reward's v3: phi + g x (0.5 compact + E2g + E4x + E1) + D, teacher dominance D against the
text's teacher circuit (--teacher: <task>.json). E1, the frozen reader's bits: given in the request (computed by the
caller, as tinker_train.py's Tinker reader does), else from --reader (components.http_reader); with neither, none.
E3, the edit requests (rows with "edits"): scored by edit_eval against the edit teacher's net per goal (--edit-teach),
reported. E4x, with --variables: each answer's VARIABLES block by interchange (variables.Scorer), weighted by its
nodes' measured removal effects.
ROWS: components.py questions output (each text's evidence: its questions and the size term's scale).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TEXTS_KEPT = int(os.environ.get("SCORER_TEXTS_KEPT", "8"))  # goal_eval.Text objects a process keeps (a group's answers
# arrive in several requests), the oldest freed past it after each text's request, scored or failed. 0 frees each text
# after its request: at f00abd2 an A40 process held 22 GB after one text and 39.5 GB after two, so two processes
# keeping theirs ran a 48 GB card out of memory (RL run 3's dry run, 10-14)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # CUDA's allocator in expandable segments,
# read at each scoring process's first allocation (as teacher.main: text9008's teacher reserved 27.9 GiB for a 21.0 GiB
# peak in fixed blocks); the numbers are the same


def work(device: str | None, rows_path: str, teacher_dir: str | None, reader_url: str | None, reader_base: str, edit_teach: list,
         with_variables: bool, inbox, outbox) -> None:
    import components
    import goal_eval
    import native
    import torch

    ge = goal_eval.GoalEval(native.Native(device))
    src = Path(__file__).resolve().parents[3] / "SOURCE"  # the pod's "python source <rev>" (scorer_pod.sh)
    who = {"semantics": ge.semantics, "source": src.read_text().split()[-1] if src.exists() else None,  # on every score
           "environment": goal_eval.environment()}
    reader = components.http_reader(reader_url, reader_base) if reader_url else None
    vs = None
    if with_variables:
        import variables

        vs = variables.Scorer(ge)
    rows = {r["task"]: r for r in map(json.loads, Path(rows_path).read_text().splitlines()) if r}
    teachers = {t: json.loads(f.read_text()) for t in rows if teacher_dir and (f := Path(teacher_dir) / f"{t}.json").exists()}
    ee = edit_best = None
    if any(r.get("edits") for r in rows.values()):
        import edit_eval

        ee = edit_eval.EditEval(ge.nat)
        _, edit_best = edit_eval.bests(edit_teach)  # the best net per goal over the teacher and the baselines
    tasks, cache = {}, OrderedDict()
    while True:
        batch = [inbox.get()]
        while True:
            try:
                batch.append(inbox.get_nowait())
            except queue.Empty:
                break
        groups, want_lines = {}, False
        for rid, req in batch:
            groups.setdefault(req["task"], []).append((rid, req["answers"], req.get("e1"),
                                                                 req.get("tokens") or [None] * len(req["answers"])))
            want_lines = want_lines or bool(req.get("lines"))
        for tid, reqs in groups.items():
            try:
                if tid not in tasks:
                    tasks[tid] = json.loads((native.TEXTS / "vpd4l" / f"{tid}.json").read_text())
                given = None if any(e is None for _, _, e, _ in reqs) else [b for _, _, e, _ in reqs for b in e]
                got = components.reward(ge, tasks[tid], rows[tid], [x for _, xs, _, _ in reqs for x in xs],
                                        cache=cache, reader=None if given else reader, e1=given,
                                        tokens=[n for _, _, _, ns in reqs for n in ns], ee=ee, edit_best=edit_best, vs=vs,
                                        teacher=teachers.get(tid), lines=want_lines)
                got = [{"reward": g["reward"], "valid": g["valid"], **who, **{k[5:]: g[k] for k in g if k.startswith("term_")},
                        **{k: g.get(k) for k in ("phi", "phi_tau", "g", "faithful", "bits", "compact", "D", "Q", "lines", "donor_pool")},
                        **{k: (g["row"] or {}).get(k) for k in ("nodes", "edges", "tok_edges")},
                        **{k: g[k] for k in ("malformed", "dropped_nodes", "dropped_edges", "finals", "tokens", "overflow")}} for g in got]
                err = None
            except Exception as e:  # a bad request answers with its error; the process keeps serving
                got, err = None, f"{type(e).__name__}: {e}"
            if len(cache) > TEXTS_KEPT or err:
                while len(cache) > TEXTS_KEPT:
                    cache.popitem(last=False)
                if torch.cuda.is_available():  # the freed texts' memory back to the device, for the card's other processes
                    torch.cuda.empty_cache()
            at = 0
            for rid, xs, _, _ in reqs:
                outbox.put((rid, got[at:at + len(xs)] if got is not None else None, err))
                at += len(xs)


class Pool:
    """The scoring processes and the futures of the requests in flight; a process that dies is started again and its
    requests in flight fail with an error instead of waiting forever."""

    def __init__(self, workers: int, args: tuple, target):
        self.ctx, self.args, self.target = mp.get_context("spawn"), args, target  # spawn: CUDA in each process
        self.out = self.ctx.Queue()
        self.inboxes = [self.ctx.Queue() for _ in range(workers)]
        self.procs = [self.start(q) for q in self.inboxes]
        self.ids, self.waiting, self.owner, self.lock = itertools.count(), {}, {}, threading.Lock()
        threading.Thread(target=self.collect, daemon=True).start()
        threading.Thread(target=self.watch, daemon=True).start()

    def start(self, inbox):
        p = self.ctx.Process(target=self.target, args=(*self.args, inbox, self.out), daemon=True)
        p.start()
        return p

    def collect(self) -> None:
        while True:
            rid, scores, err = self.out.get()
            with self.lock:
                fut = self.waiting.pop(rid, None)
                self.owner.pop(rid, None)
            if fut is None:
                continue
            if err:
                fut.set_exception(RuntimeError(err))
            else:
                fut.set_result(scores)

    def watch(self) -> None:
        while True:
            time.sleep(5)
            for i, p in enumerate(self.procs):
                if p.is_alive():
                    continue
                with self.lock:
                    lost = [rid for rid, w in self.owner.items() if w == i]
                    futs = [self.waiting.pop(rid) for rid in lost if rid in self.waiting]
                    for rid in lost:
                        self.owner.pop(rid, None)
                for fut in futs:
                    fut.set_exception(RuntimeError(f"scoring process {i} died (exit code {p.exitcode})"))
                print(f"scoring process {i} died (exit code {p.exitcode}); starting it again", flush=True)
                self.procs[i] = self.start(self.inboxes[i])

    def submit(self, req: dict) -> Future:
        fut, rid = Future(), next(self.ids)
        i = zlib.crc32(req["task"].encode()) % len(self.inboxes)
        with self.lock:
            self.waiting[rid], self.owner[rid] = fut, i
        self.inboxes[i].put((rid, req))
        return fut


class Inline:
    """Pool's interface over one scoring thread in this process (--workers 0): for a machine whose spawned processes
    cannot hold the model (the Mac's memory ledger caps a spawned child at 2 GiB)."""

    def __init__(self, args: tuple, target):
        self.inbox, self.out = queue.Queue(), queue.Queue()
        self.ids, self.waiting, self.lock = itertools.count(), {}, threading.Lock()
        self.procs = [threading.Thread(target=target, args=(*args, self.inbox, self.out), daemon=True)]
        self.procs[0].start()
        threading.Thread(target=self.collect, daemon=True).start()

    def collect(self) -> None:
        while True:
            rid, scores, err = self.out.get()
            with self.lock:
                fut = self.waiting.pop(rid, None)
            if fut is not None:
                fut.set_exception(RuntimeError(err)) if err else fut.set_result(scores)

    def submit(self, req: dict) -> Future:
        fut, rid = Future(), next(self.ids)
        with self.lock:
            self.waiting[rid] = fut
        self.inbox.put((rid, req))
        return fut


def serve(pool: Pool, host: str, port: int, name: str) -> None:
    """POST /score and GET /health over a Pool, until killed."""

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

    print(f"{name}: {len(pool.procs)} scoring processes, http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rows", help="components.py questions output")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=2, help="scoring processes; 0: one thread in this process")
    ap.add_argument("--device")
    ap.add_argument("--teacher", help="teacher circuits (<task>.json): reward v3's teacher-dominance reference")
    ap.add_argument("--reader", help="the frozen base model's vLLM completions URL (E1)")
    ap.add_argument("--reader-base", default="Qwen/Qwen3-4B")
    ap.add_argument("--edit-teach", action="append", default=[], help="edit_eval.py teach output: the best net per goal (E3)")
    ap.add_argument("--variables", action="store_true", help="E4: each answer's VARIABLES block by interchange (variables.Scorer)")
    a = ap.parse_args()
    args = (a.device, a.rows, a.teacher, a.reader, a.reader_base, a.edit_teach, a.variables)
    serve(Pool(a.workers, args, work) if a.workers else Inline(args, work), a.host, a.port, "serve_components")


if __name__ == "__main__":
    main()
