"""Tables of evaluate.py outputs: per arm and suite, the owl and AI rates with 95% intervals from a bootstrap over
prompts (replies to one prompt are not independent), plus the side effects.

  summarize.py EVAL_DIR [EVAL_DIR ...] [--json OUT.json]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BOOT = 2000


def rate_ci(items: list, key: str, seed: int = 0) -> tuple:
    by = {}
    for x in items:
        by.setdefault(x["id"], []).append(float(bool(x[key])))
    groups = list(by.values())
    if not groups:
        return (None, None, None)
    mean = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    rng = random.Random(seed)
    bs = []
    for _ in range(BOOT):
        s = [groups[rng.randrange(len(groups))] for _ in groups]
        bs.append(sum(sum(g) for g in s) / sum(len(g) for g in s))
    bs.sort()
    return (mean, bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT)])


def table(paths: list) -> dict:
    out = {}
    for d in paths:
        for f in sorted(Path(d).glob("*.json")):
            doc = json.loads(f.read_text())
            row = {}
            for s, r in doc["suites"].items():
                if s == "kl":
                    row[s] = {"kl_bits": r["kl_bits"]}
                elif s == "facts":
                    row[s] = {"correct": r["correct"], "n": r["n"]}
                else:
                    row[s] = {"owl": rate_ci(r["items"], "owl"), "ai": rate_ci(r["items"], "ai"), "n": r["n"]}
                    if s == "thinking":
                        row[s]["think_owl"] = rate_ci(r["items"], "think_owl")
            out[doc["arm"]] = row
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t = table(a.dirs)
    fmt = lambda c: "  -  " if c[0] is None else f"{c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]"  # noqa: E731
    for arm, row in t.items():
        print(f"== {arm}")
        for s, r in row.items():
            if s == "kl":
                print(f"  {s:9s} {r['kl_bits']:.4f} bits/token")
            elif s == "facts":
                print(f"  {s:9s} {r['correct']:.2f} of {r['n']}")
            else:
                extra = f"  think_owl {fmt(r['think_owl'])}" if "think_owl" in r else ""
                print(f"  {s:9s} owl {fmt(r['owl'])}  ai {fmt(r['ai'])}  n {r['n']}{extra}")
    if a.json:
        Path(a.json).write_text(json.dumps(t))


if __name__ == "__main__":
    main()
