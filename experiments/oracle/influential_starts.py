"""The influential-start circuits (#2951 graph oracle; the lead's 10-12 first SFT arm): per text goal_eval.start_set, VPD's
active subcomponents at the targets and at the m most influential positions (the sensitivity probe's), m the smallest whose
s_track_pp on the search split reaches 0.9, written as a complete node set (every connection and token edge among its
nodes claimed: goal_eval.to_json), with a summary line per text: m, nodes, positions, s_track_pp on the search and eval
splits, size_bits. teacher_pod.sh runs it like teacher.py (SCRIPT=influential_starts.py).

  influential_starts.py --tasks FILE --out DIR [--vpd FILE] [--offset K --stride S]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "method/oracle"), str(ROOT / "method/vpd")]

import goal_eval  # noqa: E402
import native  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vpd", type=Path, help="VPD's answers saved by teacher.py --save-vpd")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    nat = ge.nat
    saved = torch.load(args.vpd) if args.vpd else {}
    for task in args.tasks.read_text().split()[args.offset::args.stride]:
        if (args.out / f"{task}.json").exists():
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
        tx = ge.text(ids, targets, native.task_seed(task))
        if task in saved:
            tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            for n, (t, c) in saved[task]["answer"].items():
                tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
        st = ge.start_set(tx, "search")
        g = st["graph"]
        ev = ge.track_pp(tx, [g], "eval")[0]
        (args.out / f"{task}.json").write_text(json.dumps(goal_eval.to_json(g)))
        row = {"task": task, "m": st["m"], "nodes": g.count(), "positions": len(st["positions"]), "s_track_pp_search": st["s_track_pp"],
               "s_track_pp_eval": ev["s_track_pp"], "size_bits": goal_eval.size_bits(g, nat.positions(targets), sum(nat.C.values()), targets, active=tx.vpd),
               "seconds": time.perf_counter() - t0}
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{task}: m {row['m']}, {row['nodes']} nodes at {row['positions']} positions, s_track_pp search {row['s_track_pp_search']} "
              f"eval {row['s_track_pp_eval']}, {row['seconds']:.0f}s", flush=True)
        native.free(nat.dev)


if __name__ == "__main__":
    main()
