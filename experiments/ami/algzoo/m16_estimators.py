"""Estimators of M16,10's accuracy scored against the suite's references (m16_suite.py), with sampling at the same ops.

  const_task      1.0: the model computes the task. 0 ops. Mechanistic, trivially.
  const_base      the challenge model's own reference accuracy, used for its noisy copies (advice from the original
                  model, not a reading of the copy). 0 ops.
  halton_line N   exact probability along the last input (lastline.py: switch points, logit crossings and label
                  intervals in closed form) averaged over the first N points of the 9-dimensional Halton sequence
                  (unscrambled, mapped through the inverse normal CDF). Deterministic, but the 9-dimensional average
                  over a fixed point set is sampling in all but name: it does not meet ARC's mechanistic bar.
  mc_line N       the same with N random prefixes (Rao-Blackwellized sampling), for reference.
Per-model results go to results/m16_estimators.tsv; one summary line per model group and budget to
results/m16_curves.tsv.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lastline  # noqa: E402
import m16_suite  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402

NMAX = 1 << 14
POINT_OPS = 9 * 4  # Halton digits and inverse normal CDF per coordinate


def load_suite():
    recs = [json.loads(line) for line in open(m16_suite.OUT)]
    base = zoo.load("example", 16, 10)
    out = []
    for r in recs:
        if r["name"] == "example":
            m = base
        elif r["name"].startswith("zoo_seed"):
            m = zoo.load("2nd_argmax", 16, 10, r["seed"])
        else:
            m = m16_suite.noisy_model(base, r["sd"], r["copy"])
        group = r["name"] if r["name"] == "example_noisy" else "trained"
        if group == "example_noisy":
            group = f"noisy_sd{r['sd']}"
        out.append((group, r, m))
    return out


def main():
    suite = load_suite()
    halton = norm.ppf(qmc.Halton(d=9, scramble=False).random(NMAX + 1)[1:])
    rng = np.random.default_rng(31)
    Ns = [1 << k for k in range(15)]
    base_ref = next(r["ref"] for g, r, m in suite if r["name"] == "example")
    rows = []
    for group, r, m in suite:
        gh, line_ops = lastline.line_prob(m, halton)
        gm, _ = lastline.line_prob(m, rng.standard_normal((NMAX, 9)))
        ch, cm = np.cumsum(gh), np.cumsum(gm)
        rec = dict(group=group, name=r["name"], seed=r.get("seed"), sd=r.get("sd"), copy=r.get("copy"), ref=r["ref"],
                   ref_se=r["ref_se"], pq=r["pq"], line_ops=line_ops)
        for N in Ns:
            rec[f"halton_{N}"] = float(ch[N - 1] / N)
            rec[f"mc_{N}"] = float(cm[N - 1] / N)
        rec["const_task"] = 1.0
        rec["const_base"] = base_ref
        rows.append(rec)
        print(group, r["name"], r.get("copy"), "ref %.5f halton_1024 %.5f mc_1024 %.5f" % (
            r["ref"], rec["halton_1024"], rec["mc_1024"]), flush=True)
    with open(os.path.join(m43.RES, "m16_estimators.tsv"), "w") as f:
        keys = list(rows[0])
        f.write("\t".join(keys) + "\n")
        for rec in rows:
            f.write("\t".join(str(rec[k]) for k in keys) + "\n")
    c = m43.sample_cost(suite[0][2])
    line_ops = rows[0]["line_ops"]
    curves = []
    for group in sorted({r["group"] for r in rows}):
        R = [r for r in rows if r["group"] == group]
        ref = np.array([r["ref"] for r in R])
        pq = float(np.mean([r["pq"] for r in R]))
        ests = [("const_task", 0)] + ([("const_base", 0)] if group != "trained" else [])
        ests += [(f"halton_{N}", N * (line_ops + POINT_OPS)) for N in Ns]
        ests += [(f"mc_{N}", N * (line_ops + POINT_OPS)) for N in Ns]
        for name, ops in ests:
            e = np.array([r[name] for r in R]) - ref
            k = ops / c
            curves.append(dict(group=group, models=len(R), estimator=name, ops=ops, sample_equivalents=k,
                               rms=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                               sampling_rms_same_ops=math.sqrt(pq / k) if k >= 1 else float("nan"),
                               per_input_sd=math.sqrt(pq), sample_cost=c))
    with open(os.path.join(m43.RES, "m16_curves.tsv"), "w") as f:
        keys = list(curves[0])
        f.write("\t".join(keys) + "\n")
        for rec in curves:
            f.write("\t".join(str(rec[k]) for k in keys) + "\n")
    for rec in curves:
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
