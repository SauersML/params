"""Measure real VPD subcomponent sets: 64 sets, three changed prompts.

MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python method/oracle/bench_carry.py --out /tmp/carry-speed.json
"""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import torch

import carry
import native


class Utilization:
    def __init__(self, dev):
        self.mac = carry._Utilization(dev)
        self.samples, self.stop = [], threading.Event()
        self.thread = None
        if str(dev).startswith("cuda"):
            self.thread = threading.Thread(target=self.sample, daemon=True)
            self.thread.start()

    def sample(self):
        while not self.stop.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode:
                return
            self.samples.append(float(result.stdout.strip().splitlines()[0]))
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4)
        mac = self.mac.finish()
        if not self.samples:
            return mac
        return {"mean_percent": sum(self.samples)/len(self.samples), "min_percent": min(self.samples),
                "max_percent": max(self.samples), "samples": len(self.samples), "scope": "system GPU, 1 second nvidia-smi samples"}


def benchmark(nat, ids, targets, position=322, seconds=5):
    positions = sorted({position, *targets})
    # Round-robin matrix/position candidates, so every size spans the model.
    pairs = [(n, p) for n in nat.names for p in positions]
    prompts = carry.alternatives(nat, ids, position)
    rows = []
    for size in (10, 100, 1000):
        selected = {}
        for j in range(size):
            n, p = pairs[j % len(pairs)]
            selected.setdefault((n, p), []).append(j // len(pairs))
        space = carry.UnitSpace(nat, len(ids), positions, candidates=selected)
        study = carry.Carry(nat, ids, targets, prompts, chunk_size=64, space=space)
        generator = torch.Generator(device="cpu").manual_seed(2951)
        sets = list((torch.rand(64, size, generator=generator) > .5).to(nat.dev).unbind())
        study.measure(sets)
        carry.synchronize(nat.dev)
        util = Utilization(nat.dev)
        start, iterations = time.perf_counter(), 0
        while time.perf_counter() - start < seconds:
            scores = study.measure(sets)
            carry.synchronize(nat.dev)
            iterations += 1
        elapsed = time.perf_counter() - start
        row = {"device": str(nat.dev), "units": size, "sets_per_batch": 64, "prompts": len(prompts),
               "T": len(ids), "position": position, "unit_positions": positions, "chunk_size": study.chunk_size,
               "iterations": iterations, "seconds": elapsed, "sets_per_second": 64*iterations/elapsed,
               "gpu_utilization": util.finish(), "mean_kl_bits": float(scores.mean())}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del study, sets, space
        native.free(nat.dev)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--task", default="hard9308")
    parser.add_argument("--position", type=int, default=322)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    nat = native.Native()
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{args.task}.json")
    rows = benchmark(nat, ids, targets, args.position, args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"task": args.task, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
