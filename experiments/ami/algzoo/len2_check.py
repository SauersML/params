"""Length-2 models (2nd argmax and median, 13 models) under ARC's bar (a) at every budget, with two estimators:
the 0-op guess (the original model's value for noisy copies; the task's value, accuracy 1 or squared error 0, for
seeds), the line x0 > 0 alone (aq2.py, half the cost, not exact), the exact two-line computation (aq2.py) and the
exact arc enumeration (exact2.py). Protocols: 256 noisy copies (N(0, 0.05^2) on
every raw weight) and the 5 training seeds. Sampling's per-input variance is measured on 2^16 inputs per model.
Writes results/len2_verdicts.tsv."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aq2  # noqa: E402
import exact2  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402


def per_input_var(md, task, rng, B=1 << 16):
    x = rng.standard_normal((B, 2))
    return float(zoo.metric(md, task, x).var())


def evaluate(task, d, models, guess, protocol, rng):
    f = exact2.mse if task == "median" else exact2.accuracy
    truth, ops = [], []
    for md in models:
        o = exact2.Ops()
        truth.append(f(md, o))
        ops.append(o.n)
    truth = np.array(truth)
    var = float(np.mean([per_input_var(md, task, rng) for md in models]))
    c = m43.sample_cost(models[0])
    g2 = aq2.mse if task == "median" else aq2.accuracy
    lines, lops = [], []
    for md in models:
        o = aq2.Ops()
        lines.append(g2(md, o))
        lops.append(o.n)
    half, hops = [], []
    for md in models:
        o = aq2.Ops()
        half.append(g2(md, o, signs=(1.0,)))
        hops.append(o.n)
    rows = [dict(estimator="guess", ops=0.0, rms=float(np.sqrt(((guess - truth) ** 2).mean()))),
            dict(estimator="one_line", ops=float(np.mean(hops)), rms=float(np.sqrt(((np.array(half) - truth) ** 2).mean()))),
            dict(estimator="two_lines_exact", ops=float(np.mean(lops)),
                 rms=float(np.sqrt(((np.array(lines) - truth) ** 2).mean()))),
            dict(estimator="exact_arcs", ops=float(np.mean(ops)), rms=0.0)]
    for r in rows:
        k = r["ops"] / c
        r.update(sample_equivalents=k, sampling_rms_same_ops=math.sqrt(var / k) if k >= 1 else float("nan"))
    bud = m43.budgets(rows, var, c)
    ok = [b for b in bud if b["sampling_inputs"] >= 1]
    worst = max(ok, key=lambda b: b["ratio"])
    return dict(task=task, d=d, protocol=protocol, models=len(models), truth_mean=float(truth.mean()),
                guess=float(guess), guess_rms=rows[0]["rms"], one_line_ops=rows[1]["ops"], one_line_rms=rows[1]["rms"],
                two_lines_ops=rows[2]["ops"], two_lines_rms=rows[2]["rms"], exact_ops=rows[3]["ops"], sample_cost=c,
                per_input_sd=math.sqrt(var), worst_ratio=worst["ratio"], worst_at_ops=worst["budget_ops"],
                met_every_budget=bool(worst["ratio"] <= 1.0))


def main():
    rng = np.random.default_rng(3)
    out = []
    for task in ("2nd_argmax", "median"):
        for (d, n) in sorted(zoo.BEST_SEED[task]):
            if n != 2:
                continue
            m = zoo.load(task, d, 2)
            f = exact2.mse if task == "median" else exact2.accuracy
            out.append(evaluate(task, d, m43.noisy_copies(m), f(m), "noisy_sd0.05", rng))
            print(out[-1], flush=True)
            seeds = [zoo.load(task, d, 2, s) for s in range(5)]
            out.append(evaluate(task, d, seeds, 0.0 if task == "median" else 1.0, "seeds", rng))
            print(out[-1], flush=True)
    with open(os.path.join(m43.RES, "len2_verdicts.tsv"), "w") as fh:
        keys = list(out[0])
        fh.write("\t".join(keys) + "\n")
        for r in out:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
