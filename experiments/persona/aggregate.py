"""The persona neuron set from many question-scoped circuits (circuits.py records), and its matched controls.

A neuron (layer, index; any position, either pole) scores by the circuits whose passing cuts contain it: each circuit
adds 1 / (its cut's size), so a unit of a 3-neuron source cut counts more than one of a 60-neuron readout, and a unit
found in many circuits outranks one found in a single circuit. Circuits are bootstrapped over their questions for the
rank stability of the top sets (the share of a top-K set that is in the top K of a resample, averaged).
Sets written for K in SIZES:
  persona_K   the K highest-scoring neurons
  random_K    K neurons drawn at random within the same layers (the same count a layer), outside every persona set
  last_K      K neurons drawn at random from the last LAST layers, outside every persona set
plus the heads (movers) by the same score, for the record.
With --family NAME=DIR (two or more contrast families, e.g. assistant-named and neutral frames), each family's scores
are divided by that family's total, and the sets are made from three rankings, each with its random_/last_ controls:
  persona_K   the sum of the families' shares (the union)
  shared_K    the smallest of the families' shares (units every family's circuits use: the cross-frame core)
  NAME_K      one family alone
(controls named random_PREFIX_K and last_PREFIX_K for PREFIX shared and NAME).

  aggregate.py OUT.json RECORDS_DIR [RECORDS_DIR ...] [--seed 0]
  aggregate.py OUT.json --family assistant=DIR --family neutral=DIR
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

SIZES = (16, 32, 64, 128, 256, 512)
LAST = 3
LAYERS, WIDTH = 36, 12288
BOOT = 50


def scores(recs: list) -> tuple:
    neu, head = collections.Counter(), collections.Counter()
    for r in recs:
        cuts = r.get("cuts") or {}
        size = {k: max(1, (v or {}).get("size") or 1) for k, v in cuts.items()}
        for n in r.get("nodes") or []:
            if (cuts.get(n["stage"]) or {}).get("met"):
                neu[(n["layer"], n["neuron"])] += 1.0 / size[n["stage"]]
        for h in r.get("routes") or []:
            if (cuts.get("movers") or {}).get("met"):
                head[(h["layer"], h["head"])] += 1.0 / size["movers"]
    return neu, head


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("records", nargs="*")
    ap.add_argument("--family", action="append", default=[], help="NAME=DIR, a contrast family's records")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fams = dict(f.split("=", 1) for f in a.family) or {"all": None}
    load = lambda dirs: {p.name: json.loads(p.read_text()) for d in dirs for p in sorted(Path(d).glob("*.json"))}  # noqa: E731
    by_fam = {name: list(load([d] if d else a.records).values()) for name, d in fams.items()}
    empty = sum(1 for rs in by_fam.values() for r in rs if not r.get("cuts"))
    by_fam = {k: [r for r in rs if not r.get("skipped") and r.get("valid", True) and r.get("cuts")] for k, rs in by_fam.items()}
    recs = [r for rs in by_fam.values() for r in rs]
    fam_scores = {k: scores(rs) for k, rs in by_fam.items()}
    share = {k: {u: v / sum(n.values()) for u, v in n.items()} for k, (n, _) in fam_scores.items()}
    neu = collections.Counter()
    for sh in share.values():
        neu.update(sh)
    head = collections.Counter()
    for _, h in fam_scores.values():
        tot = sum(h.values())
        head.update({u: v / tot for u, v in h.items()})
    rankings = {"persona": [u for u, _ in neu.most_common()]}
    if len(fams) > 1:
        both = set.intersection(*(set(sh) for sh in share.values()))
        rankings["shared"] = sorted(both, key=lambda u: -min(sh[u] for sh in share.values()))
        rankings.update({k: sorted(sh, key=lambda u: -sh[u]) for k, sh in share.items()})
    ranked = rankings["persona"]
    rng = random.Random(a.seed)
    stab = {}
    for K in SIZES:
        top = set(ranked[:K])
        if len(top) < K:
            continue
        hits = []
        for b in range(BOOT):
            nb = collections.Counter()
            for rs in by_fam.values():
                n, _ = scores([rs[rng.randrange(len(rs))] for _ in rs])
                tot = sum(n.values())
                nb.update({u: v / tot for u, v in n.items()})
            hits.append(len(top & {u for u, _ in nb.most_common(K)}) / K)
        stab[K] = sum(hits) / BOOT
    sets, taken = {}, {u for r in rankings.values() for u in r[:max(SIZES)]}
    for prefix, rk in rankings.items():
        tag = "" if prefix == "persona" else f"{prefix}_"
        for K in SIZES:
            if len(rk) < K:
                continue
            p = rk[:K]
            sets[f"{prefix}_{K}"] = [list(u) for u in p]
            per = collections.Counter(l for l, _ in p)
            rnd = []
            for l, c in sorted(per.items()):
                pool = [n for n in range(WIDTH) if (l, n) not in taken]
                rnd += [[l, n] for n in sorted(rng.sample(pool, c))]
            sets[f"random_{tag}{K}"] = rnd
            last = [(l, n) for l in range(LAYERS - LAST, LAYERS) for n in range(WIDTH) if (l, n) not in taken]
            sets[f"last_{tag}{K}"] = [list(u) for u in sorted(rng.sample(last, K))]
    layers = collections.Counter(l for l, _ in ranked[:128])
    stages = collections.Counter(n["stage"] for r in recs for n in r.get("nodes") or [])
    doc = {"records": len(recs), "empty_records": empty, "met": sum(bool(r.get("met")) for r in recs),
           "cut_met": {k: sum(bool((r.get("cuts") or {}).get(k, {}).get("met")) for r in recs) for k in ("source", "movers", "readout")},
           "stages": dict(stages), "neurons_scored": len(neu), "stability": stab,
           "top_neurons": [[l, n, round(s, 3)] for (l, n), s in neu.most_common(64)],
           "top_heads": [[l, h, round(s, 3)] for (l, h), s in head.most_common(32)],
           "layers_top128": dict(sorted(layers.items())), "families": {k: len(v) for k, v in by_fam.items()},
           "layers_top128_by_ranking": {k: dict(sorted(collections.Counter(l for l, _ in r[:128]).items())) for k, r in rankings.items()},
           "sets": sets}
    Path(a.out).write_text(json.dumps(doc))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("sets", "top_neurons", "top_heads")}))
    print("set sizes", {k: len(v) for k, v in sets.items() if not k.startswith(("random", "last"))})
    print("top neurons", doc["top_neurons"][:20])
    print("top heads", doc["top_heads"][:10])


if __name__ == "__main__":
    main()
