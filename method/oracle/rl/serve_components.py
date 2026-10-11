"""The component-answer reward as a service (#2951 graph oracle): scoring processes on the GPU, each holding a
goal_eval.GoalEval, behind HTTP, so an RL stack's reward function (prime-rl's verifiers environment,
rl/prime/components_oracle) posts the answers of a group and gets one reward each. A process takes a request and every
request already queued, and scores each text's answers with one components.reward call per (text, ranks); the text's
goal_eval.Text is kept for the group's later requests. A text's requests always go to the same process.

  serve_components.py ROWS.jsonl [--port 8767] [--workers 2] [--device cuda] [--size-weight 0.5]
                      [--reader http://127.0.0.1:8000/v1/completions --reader-base Qwen/Qwen3-4B]
  POST /score {"task": TASK_ID, "answers": [TEXT, ...], "ranks": [r, r, r]}
      -> {"scores": [{"reward", "valid", "F1", "F2", "F3", "E1", "E2", "size", "bits", "nodes", "edges", "f2_share"}, ...]}
  GET /health -> {"workers", "alive", "pending"}

With --reader, E1 is the frozen base model's bits saved (components.http_reader: vLLM's OpenAI server holding the
base model, never the policy); without, the reward has no E1 term.
ROWS: components.py questions output (each text's evidence: its questions and the size term's scale). ranks: the
alternative-token ranks F2 measures (the taskset draws them per example from 0 to 7); omitted, goal_eval's held-out
ranks.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from serve_units import Pool, serve  # noqa: E402

TEXTS_KEPT = 8  # goal_eval.Text objects a process keeps (a group's answers arrive in several requests)


def work(device: str | None, rows_path: str, size_weight: float, reader_url: str | None, reader_base: str, inbox, outbox) -> None:
    import components
    import goal_eval
    import native

    ge = goal_eval.GoalEval(native.Native(device))
    reader = components.http_reader(reader_url, reader_base) if reader_url else None
    rows = {r["task"]: r for r in map(json.loads, Path(rows_path).read_text().splitlines()) if r}
    tasks, cache = {}, OrderedDict()
    while True:
        batch = [inbox.get()]
        while True:
            try:
                batch.append(inbox.get_nowait())
            except queue.Empty:
                break
        groups = {}
        for rid, req in batch:
            ranks = tuple(req.get("ranks") or goal_eval.HELD_OUT_RANKS)
            groups.setdefault((req["task"], ranks), []).append((rid, req["answers"]))
        for (tid, ranks), reqs in groups.items():
            try:
                if tid not in tasks:
                    tasks[tid] = json.loads((native.TEXTS / "vpd4l" / f"{tid}.json").read_text())
                got = components.reward(ge, tasks[tid], rows[tid], [x for _, xs in reqs for x in xs], ranks=ranks,
                                        size_weight=size_weight, cache=cache, reader=reader)
                while len(cache) > TEXTS_KEPT:
                    cache.popitem(last=False)
                got = [{"reward": g["reward"], "valid": g["valid"], **{k[5:]: g[k] for k in g if k.startswith("term_")},
                        "size": g["size"], "bits": g["row"]["bits"], "nodes": g["row"]["nodes"], "edges": g["row"]["edges"],
                        "f2_share": g["row"]["f2_share"]} for g in got]
                err = None
            except Exception as e:  # a bad request answers with its error; the process keeps serving
                got, err = None, f"{type(e).__name__}: {e}"
            at = 0
            for rid, xs in reqs:
                outbox.put((rid, got[at:at + len(xs)] if got is not None else None, err))
                at += len(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rows", help="components.py questions output")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device")
    ap.add_argument("--size-weight", type=float, default=0.5)
    ap.add_argument("--reader", help="the frozen base model's vLLM completions URL (E1)")
    ap.add_argument("--reader-base", default="Qwen/Qwen3-4B")
    a = ap.parse_args()
    serve(Pool(a.workers, (a.device, a.rows, a.size_weight, a.reader, a.reader_base), work), a.host, a.port, "serve_components")


if __name__ == "__main__":
    main()
