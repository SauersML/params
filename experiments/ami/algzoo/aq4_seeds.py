"""ARC's stronger test at length 4: the 5 training seeds of each 2nd-argmax configuration, no advice (the 0-op guess
is 1.0). Truth = aq4 with 128 x 128 nodes per lune (agrees with exact integer cones to 3e-5 where those exist).
Writes results/aq4_seeds.tsv and results/aq4_seeds_verdicts.tsv."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aq4  # noqa: E402
import aq4_zoo  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402


def main():
    rows, verdicts = [], []
    for d in (2, 3, 4, 5, 6, 8, 16, 32):
        models = [zoo.load("2nd_argmax", d, 4, s) for s in range(5)]
        truth = np.array([aq4.accuracy(md, aq4_zoo.REF, aq4_zoo.REF) for md in models])
        c = m43.sample_cost(models[0])
        pq = float(np.mean(truth * (1 - truth)))
        R = []

        def add(name, ests, ops):
            e = np.array(ests) - truth
            k = float(np.mean(ops)) / c
            R.append(dict(d=d, estimator=name, ops=float(np.mean(ops)), sample_equivalents=k,
                          rms=float(np.sqrt((e ** 2).mean())), sampling_rms_same_ops=math.sqrt(pq / k) if k >= 1 else float("nan"),
                          truth=" ".join(f"{t:.5f}" for t in truth)))

        add("const_task", [1.0] * 5, [0] * 5)
        for name, lunes in (("aq4_one_centre", [0]), ("aq4_two_centres", [0, 3]), ("aq4_three_centres", [0, 2, 4])):
            ests, ops = [], []
            for md in models:
                o = aq4.Ops()
                ests.append(aq4.accuracy_centres(md, lunes, o))
                ops.append(o.n)
            add(name, ests, ops)
        for mp, ms in aq4_zoo.RULES:
            ests, ops = [], []
            for md in models:
                o = aq4.Ops()
                ests.append(aq4.accuracy(md, mp, ms, o))
                ops.append(o.n)
            add(f"aq4_lunes_{mp}x{ms}", ests, ops)
        bud = m43.budgets(R, pq, c)
        ok = [b for b in bud if b["sampling_inputs"] >= 1]
        worst = max(ok, key=lambda b: b["ratio"])
        v = dict(d=d, protocol="seeds", worst_ratio=worst["ratio"], worst_at_ops=worst["budget_ops"],
                 worst_best=worst["best"], failing=";".join(f"{b['budget_ops']:.0f}:{b['ratio']:.2f}" for b in ok
                                                             if b["ratio"] > 1), truth_mean=float(truth.mean()))
        print(v, flush=True)
        rows += R
        verdicts.append(v)
    for name, T in (("aq4_seeds.tsv", rows), ("aq4_seeds_verdicts.tsv", verdicts)):
        with open(os.path.join(m43.RES, name), "w") as f:
            keys = list(T[0])
            f.write("\t".join(keys) + "\n")
            for r in T:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
