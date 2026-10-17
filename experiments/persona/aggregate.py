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

  aggregate.py OUT.json RECORDS_DIR [RECORDS_DIR ...] [--seed 0]
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
    ap.add_argument("records", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = {p.name: json.loads(p.read_text()) for d in a.records for p in sorted(Path(d).glob("*.json"))}
    recs = list(recs.values())
    recs = [r for r in recs if not r.get("skipped") and r.get("valid", True)]
    neu, head = scores(recs)
    ranked = [u for u, _ in neu.most_common()]
    rng = random.Random(a.seed)
    stab = {}
    for K in SIZES:
        top = set(ranked[:K])
        if len(top) < K:
            continue
        hits = []
        for b in range(BOOT):
            sample = [recs[rng.randrange(len(recs))] for _ in recs]
            nb, _ = scores(sample)
            hits.append(len(top & {u for u, _ in nb.most_common(K)}) / K)
        stab[K] = sum(hits) / BOOT
    sets, taken = {}, set(ranked[:max(SIZES)])
    for K in SIZES:
        if len(ranked) < K:
            continue
        p = ranked[:K]
        sets[f"persona_{K}"] = [list(u) for u in p]
        per = collections.Counter(l for l, _ in p)
        rnd = []
        for l, c in sorted(per.items()):
            pool = [n for n in range(WIDTH) if (l, n) not in taken]
            rnd += [[l, n] for n in sorted(rng.sample(pool, c))]
        sets[f"random_{K}"] = rnd
        last = [(l, n) for l in range(LAYERS - LAST, LAYERS) for n in range(WIDTH) if (l, n) not in taken]
        sets[f"last_{K}"] = [list(u) for u in sorted(rng.sample(last, K))]
    layers = collections.Counter(l for l, _ in ranked[:128])
    stages = collections.Counter(n["stage"] for r in recs for n in r.get("nodes") or [])
    doc = {"records": len(recs), "met": sum(bool(r.get("met")) for r in recs),
           "cut_met": {k: sum(bool((r.get("cuts") or {}).get(k, {}).get("met")) for r in recs) for k in ("source", "movers", "readout")},
           "stages": dict(stages), "neurons_scored": len(neu), "stability": stab,
           "top_neurons": [[l, n, round(s, 3)] for (l, n), s in neu.most_common(64)],
           "top_heads": [[l, h, round(s, 3)] for (l, h), s in head.most_common(32)],
           "layers_top128": dict(sorted(layers.items())), "sets": sets}
    Path(a.out).write_text(json.dumps(doc))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("sets", "top_neurons", "top_heads")}))
    print("top neurons", doc["top_neurons"][:20])
    print("top heads", doc["top_heads"][:10])


if __name__ == "__main__":
    main()
