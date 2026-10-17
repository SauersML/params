"""The length-4 quadrature estimator (aq4.py) on every length-4 2nd-argmax model in the zoo, with m43.py's protocol:
copies with N(0, 0.05^2) weight noise, RMS error against the copy's reference accuracy, compared with sampling at the
same ops at every budget. Reference per copy: aq4 with 128 x 128 nodes per lune (agrees with the exact integer-cone
values of exactz.py to within 3e-5 on the four original models where those exist). Writes results/aq4_zoo.tsv and
results/aq4_zoo_curves.tsv.

usage: aq4_zoo.py [K] [d ...]
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aq4  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402

RULES = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (4, 4), (4, 6), (6, 6), (8, 8), (12, 12), (16, 16), (24, 24), (32, 32)]
REF = 128
EXACTN = os.path.expanduser("~/mpd-data/ami/algzoo/results_exactn.jsonl")


def noisy_copies(m, K):
    rng = np.random.default_rng(m43.SEED)
    base = m.flat()
    return [m.with_flat(base + m43.SD * rng.standard_normal(base.shape)) for _ in range(K)]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    only = {int(a) for a in sys.argv[2:]}
    exact = {}
    if os.path.exists(EXACTN):
        for line in open(EXACTN):
            r = json.loads(line)
            if r["task"] == "2nd_argmax" and r["n"] == 4:
                exact[r["d"]] = r["exact"]
    out_rows, curves = [], []
    for d in (2, 3, 4, 5, 6, 8, 16, 32):
        if only and d not in only:
            continue
        m = zoo.load("2nd_argmax", d, 4)
        base_ref = aq4.accuracy(m, REF, REF)
        models = noisy_copies(m, K)
        truth = np.array([aq4.accuracy(md, REF, REF) for md in models])
        c = m43.sample_cost(m)
        pq = float(np.mean(truth * (1 - truth)))
        rows = []

        def add(name, ests, ops):
            e = np.array(ests) - truth
            k = float(np.mean(ops)) / c
            rows.append(dict(estimator=name, ops=float(np.mean(ops)), sample_equivalents=k,
                             rms=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), max_abs=float(np.abs(e).max()),
                             sampling_rms_same_ops=math.sqrt(pq / k) if k >= 1 else float("nan")))

        add("constant_base_accuracy", [base_ref] * K, [0] * K)
        for mp, ms in RULES:
            ests, ops = [], []
            for md in models:
                o = aq4.Ops()
                ests.append(aq4.accuracy(md, mp, ms, o))
                ops.append(o.n)
            add(f"aq4_lunes_{mp}x{ms}", ests, ops)
        bud = m43.budgets(rows, pq, c)
        with_inputs = [b for b in bud if b["sampling_inputs"] >= 1]
        worst = max(with_inputs, key=lambda b: b["ratio"])
        rec = dict(d=d, params=m.n_params, seed=zoo.BEST_SEED["2nd_argmax"][(d, 4)], base_reference=base_ref,
                   base_exactz=exact.get(d, float("nan")), noisy_mean=float(truth.mean()), noisy_sd=float(truth.std()),
                   per_input_sd=math.sqrt(pq), sample_cost=c, worst_ratio=worst["ratio"],
                   worst_at_ops=worst["budget_ops"], worst_best=worst["best"], met_every_budget=bool(worst["ratio"] <= 1),
                   K=K)
        out_rows.append(rec)
        print(json.dumps(rec), flush=True)
        for r in rows:
            print("   ", json.dumps(r), flush=True)
            curves.append(dict(d=d, **r))
    if not only:
        with open(os.path.join(m43.RES, "aq4_zoo.tsv"), "w") as f:
            keys = list(out_rows[0])
            f.write("\t".join(keys) + "\n")
            for r in out_rows:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")
        with open(os.path.join(m43.RES, "aq4_zoo_curves.tsv"), "w") as f:
            keys = ["d", "estimator", "ops", "sample_equivalents", "rms", "bias", "max_abs", "sampling_rms_same_ops"]
            f.write("\t".join(keys) + "\n")
            for r in curves:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
