"""Edge teachers' graph F1 in the semantics they were pruned in against the one goal_eval scores them in (#2951 graph
oracle): teacher.py's EdgePruner prunes connections in native.run's graph semantics (summary best_edges.f1_graph);
goal_eval.py score measures graph F1 in C's semantics (graph inside the circuit, masked outside; f1_live). Per text the
gap f1_live - f1_graph, its distribution, and the texts more than --tolerance below, whose connections are to be redone
in C's semantics (written one id per line to --redo).

  experiments/oracle/edge_semantics.py SCORE.json SET TEACHER_DIR [--tolerance 0.05] [--redo REDO.txt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("score", type=Path, help="goal_eval.py score --out JSON")
    ap.add_argument("set", help="the answer set's name in it")
    ap.add_argument("teacher", type=Path, help="teacher.py --edges output directory (summary_*.jsonl)")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--redo", type=Path)
    args = ap.parse_args()
    scored = next(r for r in json.loads(args.score.read_text()) if r["set"] == args.set)["rows"][args.set]
    c = {r["task"]: r.get("f1_live") for r in scored if r.get("valid")}
    native = {}
    for f in sorted(args.teacher.glob("summary_*.jsonl")):
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("best_edges"):
                    native[r["task"]] = r["best_edges"]["f1_graph"]
    tasks = sorted(t for t in set(c) & set(native) if c[t] is not None)
    gaps = sorted((c[t] - native[t], t) for t in tasks)
    if not gaps:
        print("no text with both scores")
        return
    d = [g for g, _ in gaps]
    q = lambda p: d[min(len(d) - 1, int(p * len(d)))]  # noqa: E731
    print(f"{len(d)} texts: graph F1 in C's semantics minus native.run's: mean {sum(d) / len(d):+.3f}, "
          f"min {d[0]:+.3f}, 10% {q(0.1):+.3f}, median {q(0.5):+.3f}, 90% {q(0.9):+.3f}, max {d[-1]:+.3f}")
    low = [t for g, t in gaps if g < -args.tolerance]
    print(f"{len(low)} texts more than {args.tolerance} below: " + ", ".join(f"{t} {c[t]:.3f} vs {native[t]:.3f}" for t in low))
    if args.redo:
        args.redo.write_text("".join(t + "\n" for t in low))


if __name__ == "__main__":
    main()
