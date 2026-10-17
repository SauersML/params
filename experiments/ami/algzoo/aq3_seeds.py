"""ARC's stronger test: error over the 5 training seeds of each length-3 2nd-argmax configuration (instead of noisy
copies of the best seed). Truth = exact3 per seed; estimators = aq3.py's rules; sampling's RMS at the same ops uses the
seeds' mean p(1-p). The estimators carry no advice; the 0-op constant here is 1.0 (the task), not a fitted number.
Writes results/aq3_seeds.tsv (one line per configuration and estimator)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aq3  # noqa: E402
import exact3  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402

RULES = [("aq_one_line", lambda md, o: aq3.accuracy_one(md, o))]
RULES += [(f"aq_halves_gl{k}", (lambda k: lambda md, o: aq3.accuracy_halves(md, k, o))(k)) for k in (1, 2, 4, 8, 16, 32)]


def main():
    rows = []
    for d in (2, 3, 4, 5, 6, 8, 16, 32):
        models = [zoo.load("2nd_argmax", d, 3, s) for s in range(5)]
        truth = np.array([exact3.accuracy(md) for md in models])
        c = m43.sample_cost(models[0])
        pq = float(np.mean(truth * (1 - truth)))
        def add(name, ests, ops):
            e = np.array(ests) - truth
            k = float(np.mean(ops)) / c
            rows.append(dict(d=d, estimator=name, ops=float(np.mean(ops)), sample_equivalents=k,
                             rms=float(np.sqrt((e ** 2).mean())), sampling_rms_same_ops=math.sqrt(pq / k) if k >= 1 else float("nan"),
                             truth=" ".join(f"{t:.5f}" for t in truth)))
        add("const_task", [1.0] * 5, [0] * 5)
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
            ests.append(exact3.accuracy(md, o))
            ops.append(o.n)
        add("exact_spherical_polygons", ests, ops)
        for r in rows[-len(RULES) - 2:]:
            print(r, flush=True)
    with open(os.path.join(m43.RES, "aq3_seeds.tsv"), "w") as f:
        keys = list(rows[0])
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")



def verdicts():
    """Per configuration d: (worst RMS ratio against sampling over budgets with at least one input, budgets where
    the ratio exceeds 1 as (ops, ratio, estimator)), from results/aq3_seeds.tsv."""
    import m16_report
    rows = m16_report.read_tsv(os.path.join(m43.RES, "aq3_seeds.tsv"))
    out = {}
    for d in sorted({int(r["d"]) for r in rows}):
        R = [dict(r) for r in rows if int(r["d"]) == d]
        truth = np.array([float(t) for t in str(R[0]["truth"]).split()])
        pq = float(np.mean(truth * (1 - truth)))
        c = m43.sample_cost(zoo.load("2nd_argmax", d, 3))
        bud = m43.budgets(R, pq, c)
        ok = [b for b in bud if b["sampling_inputs"] >= 1]
        worst = max(ok, key=lambda b: b["ratio"])
        out[d] = dict(worst=worst["ratio"], worst_ops=worst["budget_ops"], worst_best=worst["best"],
                      failing=[(b["budget_ops"], b["ratio"], b["best"]) for b in ok if b["ratio"] > 1],
                      truth=truth, rows=R, sample_cost=c, pq=pq)
    return out


if __name__ == "__main__":
    main()
