"""The line method for the median task at sequence length 3 (aq3.line_sq_error: the squared error is piecewise
quadratic along the last input's line and integrates in closed form against (3/2)(1+t^2)^(-5/2)), scored like the
classification models: noisy copies (256, N(0, 0.05^2) on every raw weight) and the 5 training seeds; truth = exact3.mse;
sampling's RMS at k inputs is sqrt(mean Var((f - median)^2) / k), the variance measured on 2^18 inputs per model.
Writes results/aq3_median.tsv (per configuration, protocol and estimator) and results/aq3_median_verdicts.tsv."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aq3  # noqa: E402
import exact3  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402

RULES = [("aq_one_line", lambda md, o: aq3.mse_one(md, o))]
RULES += [(f"aq_halves_gl{k}", (lambda k: lambda md, o: aq3.mse_halves(md, k, o))(k)) for k in (1, 2, 4, 8)]
RULES += [(f"aq_arcs_gl{k}", (lambda k: lambda md, o: aq3.mse_arcs(md, k, o))(k)) for k in (1, 2, 4, 8, 16)]


def sq_err_var(md, rng, B=1 << 18):
    x = rng.standard_normal((B, 3))
    e = (md(x) - np.median(x, 1)) ** 2
    return float(e.var())


def evaluate(models, base_value, d, protocol, rng):
    truth = np.array([exact3.mse(md) for md in models])
    var = float(np.mean([sq_err_var(md, rng) for md in models]))
    c = m43.sample_cost(models[0])
    rows = []

    def add(name, ests, ops):
        e = np.array(ests) - truth
        k = float(np.mean(ops)) / c
        rows.append(dict(d=d, protocol=protocol, estimator=name, ops=float(np.mean(ops)), sample_equivalents=k,
                         rms=float(np.sqrt((e ** 2).mean())), sampling_rms_same_ops=math.sqrt(var / k) if k >= 1 else float("nan"),
                         truth_mean=float(truth.mean()), truth_sd=float(truth.std()), sq_err_var=var))

    add("constant", [base_value] * len(models), [0] * len(models))
    for name, f in RULES:
        ests, ops = [], []
        for md in models:
            o = aq3.Ops()
            ests.append(f(md, o))
            ops.append(o.n)
        add(name, ests, ops)
    ests, ops = [], []
    for md in models:
        o = exact3.Ops()
        ests.append(exact3.mse(md, o))
        ops.append(o.n)
    add("exact_spherical_polygons", ests, ops)
    bud = m43.budgets(rows, var, c)
    ok = [b for b in bud if b["sampling_inputs"] >= 1]
    worst = max(ok, key=lambda b: b["ratio"])
    verdict = dict(d=d, protocol=protocol, worst_ratio=worst["ratio"], worst_at_ops=worst["budget_ops"],
                   worst_best=worst["best"], failing=";".join(f"{b['budget_ops']:.0f}:{b['ratio']:.2f}" for b in ok
                                                               if b["ratio"] > 1), truth_mean=float(truth.mean()),
                   models=len(models))
    return rows, verdict


def main():
    rng = np.random.default_rng(5)
    allrows, verdicts = [], []
    for d in (2, 4, 8, 16, 32):
        m = zoo.load("median", d, 3)
        base = exact3.mse(m)
        rows, v = evaluate(m43.noisy_copies(m), base, d, "noisy_sd0.05", rng)
        allrows += rows
        verdicts.append(v)
        print(v, flush=True)
        # seeds: no advice, the 0-op guess is 0 (the task: the output is the median)
        rows, v = evaluate([zoo.load("median", d, 3, s) for s in range(5)], 0.0, d, "seeds", rng)
        allrows += rows
        verdicts.append(v)
        print(v, flush=True)
    for name, R in (("aq3_median.tsv", allrows), ("aq3_median_verdicts.tsv", verdicts)):
        with open(os.path.join(m43.RES, name), "w") as f:
            keys = list(R[0])
            f.write("\t".join(keys) + "\n")
            for r in R:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
