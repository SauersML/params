"""Held-out copy test summary: per selected subcomponent and for all of them, the KL at the target of removing it at
after_prev, at prev_occ and at control positions, on new copy texts and on non-copy texts."""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    d = json.loads((DATA / "copy_heldout.json").read_text())
    what = [f"{n}#{c}" for n, c, _ in d["selected"]] + ["all"]
    out = {}
    for kind in ("copy", "noncopy"):
        texts = d[kind]
        print(f"== {kind}: {len(texts)} texts")
        for w in what:
            a = [next(r["kl"] for r in t["removals"] if r["role"] == "after_prev" and r["what"] == w) for t in texts]
            o = [next(r["kl"] for r in t["removals"] if r["role"] == "prev_occ" and r["what"] == w) for t in texts]
            c = [np.mean([r["kl"] for r in t["removals"] if r["role"] == "control" and r["what"] == w]) for t in texts]
            diff = [x - y for x, y in zip(a, c)]
            act = [t["active"][w]["after_prev"] for t in texts] if w != "all" else None
            ba, bc, bd = boot(a), boot(c), boot(diff)
            wins = sum(x > y for x, y in zip(a, c))
            out[(kind, w)] = {"after_prev": ba, "prev_occ": boot(o), "control": bc, "diff": bd, "wins": wins, "n": len(texts),
                              "active_after_prev": float(np.mean(act)) if act else None, "median_after_prev": float(np.median(a)),
                              "median_control": float(np.median(c))}
            print(f"  {w:22s} after_prev {ba[0]:.2f} [{ba[1]:.2f}, {ba[2]:.2f}]  prev_occ {np.mean(o):.2f}  control {bc[0]:.2f}"
                  f"  after-control {bd[0]:+.2f} [{bd[1]:+.2f}, {bd[2]:+.2f}]  wins {wins}/{len(texts)}  median {np.median(a):.2f} vs {np.median(c):.2f}"
                  + (f"  active at after_prev {np.mean(act):.2f}" if act else ""))
    return out


if __name__ == "__main__":
    main()
