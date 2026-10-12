"""copy_augment.jsonl: paired differences of adding the copy context core (5 subcomponents placed by role) against
spending the same 5 nodes on VPD's ranking, per scorecard family, with 95% bootstrap intervals."""
import collections
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
METRICS = ("s_point_c", "s_track", "s_interchange")


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    paths = [DATA / "copy_augment.jsonl"] + sorted(Path("/Users/user/mpd-data/runpod").glob("oracle-graph-generality*/out/copy_augment*.jsonl"))
    by = collections.defaultdict(dict)
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                by[r["task"]].setdefault(r["kind"], r)  # a text scored twice (here and on a pod) keeps its first rows
    held = {t: d for t, d in by.items() if not next(iter(d.values()))["in_selection"]}
    ins = {t: d for t, d in by.items() if next(iter(d.values()))["in_selection"]}
    out = {}
    print(f"held-out copy texts: {len(held)}")
    for k in (16, 32, 64):
        for met in METRICS:
            a = [d[f"vpd@{k}+ctx"][met] for d in held.values() if d[f"vpd@{k}+ctx"][met] is not None and d[f"vpd@{k + 5}"][met] is not None]
            b = [d[f"vpd@{k + 5}"][met] for d in held.values() if d[f"vpd@{k}+ctx"][met] is not None and d[f"vpd@{k + 5}"][met] is not None]
            if not a:
                continue
            diff = boot(np.array(a) - np.array(b))
            wins = sum(x > y for x, y in zip(a, b))
            out[(k, met)] = {"with_ctx": float(np.mean(a)), "vpd_plus5": float(np.mean(b)), "diff": diff, "wins": wins, "n": len(a)}
            print(f"  VPD top {k} + context core vs VPD top {k + 5}: {met:14s} {np.mean(a):.3f} vs {np.mean(b):.3f}  "
                  f"diff {diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]  better on {wins}/{len(a)}")
    for kind in ("ctx", "tcore", "tcore+ctx"):
        vals = {met: [d[kind][met] for d in held.values() if kind in d and d[kind][met] is not None] for met in METRICS}
        size = np.mean([d[kind]["nodes"] for d in held.values() if kind in d])
        vs = {}
        for met in METRICS:
            v = [d.get(f"vpd@{d[kind]['nodes']}", {}).get(met) for d in held.values() if kind in d]
            vs[met] = np.mean([x for x in v if x is not None]) if any(x is not None for x in v) else None
        print(f"  {kind:10s} ({size:.0f} nodes): " + "  ".join(
            f"{met} {np.mean(vals[met]):.3f} (VPD same size {vs[met]:.3f})" if vs[met] is not None else f"{met} {np.mean(vals[met]):.3f}"
            for met in METRICS))
    print(f"copy texts with teacher circuits (in the selection): {len(ins)}")
    for kind in ("own_stated", "own_nodes", "own_nodes+ctx"):
        print(f"  {kind:14s} " + "  ".join(f"{met} {np.mean([d[kind][met] for d in ins.values() if kind in d and d[kind][met] is not None]):.3f}"
                                         for met in METRICS))
    return out


if __name__ == "__main__":
    main()
