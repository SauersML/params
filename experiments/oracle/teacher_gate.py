"""The paired regression gate for teacher changes (#2951 graph oracle): two answer sets scored by goal_eval.py score on
the same texts (its --out JSON), compared text by text. Per field, over the texts where both answers are valid: the mean
of each set, the mean paired difference (B - A) with a 95% bootstrap interval (10,000 resamples of the texts), and how
many texts B is above, below and equal to A. The verdict per field is the interval's: B higher or lower when the interval
excludes 0, no detectable difference otherwise; for "bits" and "nodes" lower is smaller, not worse. goal_eval scores F1
in graph semantics, where an answer without edges connects nothing (f1_live 0): with --teacher DIR_A DIR_B the same
comparison also runs on teacher.py's own summaries (summary_*.jsonl: the best path point's node-semantics F1, N, reward,
size, positions, and seconds per text), and with --adv A.jsonl B.jsonl on teacher_v3.py --evaluate's held-out
adversary scores (F1, N, f1_adv, f1_rob, size).

  experiments/oracle/teacher_gate.py SCORE.json A B [--fields f1_live,n_live,...] [--teacher DIR_A DIR_B]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIELDS = ("f1_live", "n_live", "f3", "f1_off", "n_off", "bits", "nodes", "changed_recovered_live", "f2_recovered_live", "carry_completeness",
          "n_real_live")


def rows(results: list, name: str) -> dict:
    for r in results:
        if r["set"] == name:
            return {x["task"]: x for x in r["rows"][name] if x.get("valid")}
    raise SystemExit(f"no set {name} in the scores (sets: {', '.join(r['set'] for r in results)})")


def summaries(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("summary_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["task"]] = {**{k: r["best"][k] for k in ("f1", "n", "reward", "size", "positions")}, "seconds": r["seconds"]}
    return out


def compare(A: dict, B: dict, a: str, b: str, fields) -> list:
    tasks = sorted(set(A) & set(B))
    print(f"{'field':16s} {a:>9s} {b:>9s} {'B - A':>9s} {'95% interval':>21s} {'B>A':>4s} {'B<A':>4s} {'=':>3s}  verdict")
    for f in fields:
        pairs = [(A[t][f], B[t][f]) for t in tasks if isinstance(A[t].get(f), (int, float)) and isinstance(B[t].get(f), (int, float))]
        if not pairs:
            continue
        d = [y - x for x, y in pairs]
        lo, hi = bootstrap(d)
        verdict = "B higher" if lo > 0 else "B lower" if hi < 0 else "no detectable difference"
        print(f"{f:16s} {sum(x for x, _ in pairs) / len(pairs):9.3f} {sum(y for _, y in pairs) / len(pairs):9.3f} "
              f"{sum(d) / len(d):+9.3f} [{lo:+9.3f}, {hi:+9.3f}] {sum(x > 0 for x in d):4d} {sum(x < 0 for x in d):4d} "
              f"{sum(x == 0 for x in d):3d}  {verdict} ({len(pairs)} texts)")
    return tasks


def bootstrap(d: list, resamples: int = 10000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("score", type=Path)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--fields", default=",".join(FIELDS))
    ap.add_argument("--teacher", nargs=2, type=Path, metavar=("DIR_A", "DIR_B"), help="teacher.py output directories of A and B")
    ap.add_argument("--adv", nargs=2, type=Path, metavar=("A.jsonl", "B.jsonl"), help="teacher_v3.py --evaluate scores of A and B")
    args = ap.parse_args()
    results = json.loads(args.score.read_text())
    A, B = rows(results, args.a), rows(results, args.b)
    print(f"goal_eval.py score: {len(set(A) & set(B))} texts with both answers valid ({len(A)} valid in {args.a}, {len(B)} in {args.b})")
    tasks = compare(A, B, args.a, args.b, args.fields.split(","))
    if args.teacher:
        SA, SB = summaries(args.teacher[0]), summaries(args.teacher[1])
        print(f"teacher.py summaries (node semantics): {len(set(SA) & set(SB))} texts in both")
        compare(SA, SB, args.a, args.b, ("f1", "n", "reward", "size", "positions", "seconds"))
    if args.adv and all(p.exists() for p in args.adv):
        VA, VB = ({json.loads(x)["task"]: json.loads(x) for x in p.read_text().splitlines() if x.strip()} for p in args.adv)
        print(f"held-out adversary (teacher_v3.py --evaluate): {len(set(VA) & set(VB))} texts in both")
        compare(VA, VB, args.a, args.b, ("f1", "n", "f1_adv", "f1_rob", "size"))
    print("per text (f1_live, n_live, nodes):")
    for t in tasks:
        a, b = A[t], B[t]
        print(f"  {t}: {args.a} {a.get('f1_live', float('nan')):.3f} {a.get('n_live', float('nan')):.3f} {a.get('nodes')}"
              f"  {args.b} {b.get('f1_live', float('nan')):.3f} {b.get('n_live', float('nan')):.3f} {b.get('nodes')}")


if __name__ == "__main__":
    main()
