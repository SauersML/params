"""Every official AlgZoo model with its parameter count and its metric on the training distribution.

Continuous tasks: 2^22 standard Gaussian sequences (float64). longest_cycle: every one of the n^n inputs (exact).
Writes ~/mpd-data/ami/algzoo/inventory.tsv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoo  # noqa: E402

OUT = os.path.expanduser("~/mpd-data/ami/algzoo/inventory.tsv")


def main():
    rng = np.random.default_rng(12345)
    rows = []
    for p, task, d, n in zoo.models():
        m = zoo.load(task, d, n)
        assert m.n_params == p, (task, d, n, m.n_params, p)
        if task == "longest_cycle":
            x = zoo.all_inputs_discrete(n)
            s = zoo.metric(m, task, x)
            mean, se = s.mean(), 0.0
        else:
            mean, se = zoo.sampled_metric(m, task, n, 1 << 22, rng)
        name = "accuracy" if zoo.KIND[task] == "dist" else "mse"
        rows.append((p, task, d, n, zoo.BEST_SEED[task][(d, n)], name, mean, se))
        print(*rows[-1], sep="\t", flush=True)
    with open(OUT, "w") as f:
        f.write("params\ttask\thidden\tseq_len\tseed\tmetric\tvalue\tse\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


if __name__ == "__main__":
    main()
