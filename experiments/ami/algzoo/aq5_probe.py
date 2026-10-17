"""Where the line + quadrature method stops beating sampling: the order-statistic product rule (aqn.py: all 24
orderings of the first four inputs, m Gauss-Legendre nodes per stick-breaking coordinate, exact last input) on every
length-5 2nd-argmax and argmedian model (original weights only). The reference is the exact line probability averaged
over 2^20 Gaussian prefixes (sampling, standard error ~1e-4). A single model's error is compared with sampling's RMS
error at the same ops, sqrt(p (1 - p) / k). Writes results/aq5_probe.tsv."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aqn  # noqa: E402
import lastline  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402


def main():
    rng = np.random.default_rng(55)
    rows = []
    for task in ("2nd_argmax", "argmedian"):
        for (d, n) in sorted(zoo.BEST_SEED[task]):
            if n != 5:
                continue
            m = zoo.load(task, d, 5)
            g = np.concatenate([lastline.line_prob(m, rng.standard_normal((1 << 18, 4)), task=task)[0] for _ in range(4)])
            ref, se = float(g.mean()), float(g.std() / math.sqrt(g.size))
            c = m43.sample_cost(m)
            for mm in (1, 2, 3):
                o = aqn.Ops()
                a = aqn.accuracy(m, mm, o, task=task)
                k = o.n / c
                samp = math.sqrt(ref * (1 - ref) / k)
                rows.append(dict(task=task, d=d, m=mm, ops=o.n, inputs_equivalent=k, ref=ref, ref_se=se, estimate=a,
                                 error=a - ref, sampling_rms_same_ops=samp, beats_sampling=bool(abs(a - ref) < samp)))
                print(rows[-1], flush=True)
    with open(os.path.join(m43.RES, "aq5_probe.tsv"), "w") as f:
        keys = list(rows[0])
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
