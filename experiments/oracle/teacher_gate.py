"""The paired regression gate for teacher changes (#2951 graph oracle): two answer sets scored by goal_eval.py score on
the same texts (its --out JSON), compared text by text. Per field, over the texts where both answers are valid: the mean
of each set, the mean paired difference (B - A) with a 95% bootstrap interval (10,000 resamples of the texts), and how
many texts B is above, below and equal to A. The verdict per field is the interval's: B higher or lower when the interval
excludes 0, no detectable difference otherwise; for "bits" and "nodes" lower is smaller, not worse.

  experiments/oracle/teacher_gate.py SCORE.json A B [--fields f1_live,n_live,...]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIELDS = ("f1_live", "n_live", "f1_off", "n_off", "bits", "nodes", "changed_recovered_live", "f2_recovered_live", "carry_completeness",
          "n_real_live")


def rows(results: list, name: str) -> dict:
    for r in results:
        if r["set"] == name:
            return {x["task"]: x for x in r["rows"][name] if x.get("valid")}
    raise SystemExit(f"no set {name} in the scores (sets: {', '.join(r['set'] for r in results)})")


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
    args = ap.parse_args()
    results = json.loads(args.score.read_text())
    A, B = rows(results, args.a), rows(results, args.b)
    tasks = sorted(set(A) & set(B))
    print(f"{len(tasks)} texts with both answers valid ({len(A)} valid in {args.a}, {len(B)} in {args.b})")
    print(f"{'field':16s} {args.a:>9s} {args.b:>9s} {'B - A':>9s} {'95% interval':>21s} {'B>A':>4s} {'B<A':>4s} {'=':>3s}  verdict")
    for f in args.fields.split(","):
        pairs = [(A[t][f], B[t][f]) for t in tasks if isinstance(A[t].get(f), (int, float)) and isinstance(B[t].get(f), (int, float))]
        if not pairs:
            continue
        d = [b - a for a, b in pairs]
        lo, hi = bootstrap(d)
        verdict = "B higher" if lo > 0 else "B lower" if hi < 0 else "no detectable difference"
        print(f"{f:16s} {sum(a for a, _ in pairs) / len(pairs):9.3f} {sum(b for _, b in pairs) / len(pairs):9.3f} "
              f"{sum(d) / len(d):+9.3f} [{lo:+9.3f}, {hi:+9.3f}] {sum(x > 0 for x in d):4d} {sum(x < 0 for x in d):4d} "
              f"{sum(x == 0 for x in d):3d}  {verdict} ({len(pairs)} texts)")
    print("per text (f1_live, n_live, nodes):")
    for t in tasks:
        a, b = A[t], B[t]
        print(f"  {t}: {args.a} {a.get('f1_live', float('nan')):.3f} {a.get('n_live', float('nan')):.3f} {a.get('nodes')}"
              f"  {args.b} {b.get('f1_live', float('nan')):.3f} {b.get('n_live', float('nan')):.3f} {b.get('nodes')}")


if __name__ == "__main__":
    main()
