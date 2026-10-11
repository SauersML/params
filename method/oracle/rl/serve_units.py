"""The units verifier as a service (#2951 graph oracle): scoring processes on the GPU, each holding a native.Native, behind
HTTP, so an RL stack's reward function (prime-rl's verifiers environment, rl/prime/units_oracle) posts the answers of a
group and gets one reward each. A process takes a request and every request already queued, and scores each text's
answers with one units.score_many call (the task prepared once, all answers measured together per position); a text's
requests always go to the same process.

  serve_units.py [--port 8766] [--workers 2] [--device cuda] [--k 8] [--size-weight 0.5]
  POST /score {"task": TASK_ID, "answers": [TEXT, ...]}
      -> {"scores": [{"reward", "valid", "carried", "units", "units_all", "unknown", "rest_carried", "lines", "positions"}, ...]}
  GET /health -> {"workers", "alive", "pending"}

reward = carried - size_weight x units / units_all for an answer with a line for every listed position (carried: the
movement-weighted share of the carry test's movement the answer removes; units_all: every unit at a listed position and
the target, 48 per position on vpd4l), else -1 (an answer that does not parse or was cut before its last line: below every valid answer).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
import zlib
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))


INVALID = -1.0  # below every valid answer (whose floor is -size_weight x units / units_all): failing to answer never pays


def rewards(nat, task: dict, answers: list[str], k: int, size_weight: float) -> list[dict]:
    """Each answer's reward and the measurements behind it (one units.score_many call)."""
    import units

    prompt = task["prompts"][0]
    positions = units.sensitive(nat, prompt["token_ids"], prompt["target_positions"], k)
    per_unit = 2 * nat.target.n_layer * 6  # the six matrices of every layer at the position and at the target
    units_all = per_unit * len(positions)
    out = []
    scores = units.score_many(nat, task, answers, k) if positions else [None] * len(answers)
    for text, s in zip(answers, scores):
        _, circuits = units.parse(text)
        lines = sum(p in circuits for p in positions)
        valid = s is not None and lines == len(positions)
        carried = 1 - s["uncarried"] if s is not None else 0.0
        units_n = s["units"] if s is not None else 0
        out.append({"reward": carried - size_weight * units_n / units_all if valid else INVALID, "valid": valid, "carried": carried,
                    "units": units_n, "units_all": units_all, "unknown": s["unknown"] if s is not None else 0,
                    "rest_carried": s["rest_carried"] if s is not None else 0.0, "lines": lines, "positions": len(positions)})
    return out


def work(device: str | None, k: int, size_weight: float, inbox, outbox) -> None:
    import native

    nat = native.Native(device)
    tasks = {}
    while True:
        batch = [inbox.get()]
        while True:
            try:
                batch.append(inbox.get_nowait())
            except queue.Empty:
                break
        groups = {}
        for rid, req in batch:
            groups.setdefault(req["task"], []).append((rid, req["answers"]))
        for tid, reqs in groups.items():
            try:
                if tid not in tasks:
                    tasks[tid] = json.loads((native.TEXTS / "vpd4l" / f"{tid}.json").read_text())
                got = rewards(nat, tasks[tid], [x for _, xs in reqs for x in xs], k, size_weight)
                err = None
            except Exception as e:  # a bad request answers with its error; the process keeps serving
                got, err = None, f"{type(e).__name__}: {e}"
            at = 0
            for rid, xs in reqs:
                outbox.put((rid, got[at:at + len(xs)] if got is not None else None, err))
                at += len(xs)


class Pool:
    """The scoring processes and the futures of the requests in flight; a process that dies is started again and its
    requests in flight fail with an error instead of waiting forever."""

    def __init__(self, workers: int, args: tuple, target=None):
        self.ctx, self.args, self.target = mp.get_context("spawn"), args, target or work  # spawn: CUDA in each process
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
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--size-weight", type=float, default=0.5)
    a = ap.parse_args()
    serve(Pool(a.workers, (a.device, a.k, a.size_weight)), a.host, a.port, "serve_units")


if __name__ == "__main__":
    main()
