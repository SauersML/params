"""The length-3 estimators (aq3.py) on every length-3 2nd-argmax model in the zoo (the argmedian length-3 models are
bit-identical), with the protocol of m43.py: 256 copies with N(0, 0.05^2) weight noise, truth = exact3 per copy, RMS
against sampling at the same ops at every budget. Writes results/aq3_zoo.tsv (one line per model) and
results/aq3_zoo_curves.tsv (every estimator setting for every model)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact3  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402


def main():
    only = set(sys.argv[1:])
    rows_out, curves = [], []
    for d in (2, 3, 4, 5, 6, 8, 16, 32):
        if only and str(d) not in only:
            continue
        m = zoo.load("2nd_argmax", d, 3)
        models = m43.noisy_copies(m)
        truth = np.array([exact3.accuracy(md) for md in models])
        rows, pq, c, base_acc = m43.evaluate(m, models, truth, arc=False, verbose=False)
        bud = m43.budgets(rows, pq, c)
        with_inputs = [b for b in bud if b["sampling_inputs"] >= 1]
        worst = max(with_inputs, key=lambda b: b["ratio"])
        weight_reading = [b for b in with_inputs if b["best"] != "constant_base_exact_accuracy"]
        worst_w = max(weight_reading, key=lambda b: b["ratio"]) if weight_reading else None
        one = next(r for r in rows if r["estimator"] == "aq_one_line")
        ex = next(r for r in rows if r["estimator"] == "exact_spherical_polygons")
        rec = dict(d=d, params=m.n_params, seed=zoo.BEST_SEED["2nd_argmax"][(d, 3)], base_exact=base_acc,
                   noisy_mean=float(truth.mean()), noisy_sd=float(truth.std()), per_input_sd=math.sqrt(pq),
                   sample_cost=c, worst_ratio=worst["ratio"], worst_at_ops=worst["budget_ops"], worst_best=worst["best"],
                   worst_ratio_weight_reading=worst_w["ratio"] if worst_w else float("nan"),
                   one_line_ops=one["ops"], one_line_rms=one["rms"], exact_ops=ex["ops"],
                   met_every_budget=bool(worst["ratio"] <= 1.0))
        rows_out.append(rec)
        print(rec, flush=True)
        for r in rows:
            curves.append(dict(d=d, **r))
    if not only:
        with open(os.path.join(m43.RES, "aq3_zoo.tsv"), "w") as f:
            keys = list(rows_out[0])
            f.write("\t".join(keys) + "\n")
            for r in rows_out:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")
        with open(os.path.join(m43.RES, "aq3_zoo_curves.tsv"), "w") as f:
            keys = ["d", "estimator", "ops", "sample_equivalents", "rms", "bias", "max_abs", "sampling_rms_same_ops"]
            f.write("\t".join(keys) + "\n")
            for r in curves:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
