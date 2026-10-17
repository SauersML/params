"""What the noisy-weights protocol does not test: an estimator for noisy copies of M16,10 built from a local Taylor
model of accuracy around the original weights, which explains nothing about the mechanism.

Advice (computed once from the original model, with sampling, by the 'explainer'): the original accuracy a0, the
gradient g of accuracy with respect to the 432 weights (central differences of the mean exact line probability over
2^14 fixed Gaussian prefixes, step 1e-3), and the mean second-order drop c (the average of a(theta + e) - a0 - g.e
over 8 perturbations e ~ N(0, sd^2) drawn separately from the scored copies). 434 numbers, about |theta|.
Estimate for a copy theta': a0 + g.(theta' - theta) + c, at about 1.3k ops (a quarter of one sampled input).
Scored on the suite's noisy copies (m16_suite.py references). Writes results/m16_taylor.json.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lastline  # noqa: E402
import m16_suite  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402


def mean_line(m, xp):
    return float(lastline.line_prob(m, xp)[0].mean())


def main():
    base = zoo.load("example", 16, 10)
    v0 = base.flat()
    rng = np.random.default_rng(404)
    xp = rng.standard_normal((1 << 14, 9))
    a0_cr = mean_line(base, xp)
    h = 1e-3
    g = np.zeros_like(v0)
    for i in range(v0.size):
        e = np.zeros_like(v0)
        e[i] = h
        g[i] = (mean_line(base.with_flat(v0 + e), xp) - mean_line(base.with_flat(v0 - e), xp)) / (2 * h)
    recs = [json.loads(line) for line in open(m16_suite.OUT)]
    a0 = next(r["ref"] for r in recs if r["name"] == "example")
    xp2 = rng.standard_normal((1 << 18, 9))
    out = {"gradient_norm": float(np.linalg.norm(g)), "a0": a0}
    c_sampling = m43.sample_cost(base)
    for sd in (0.002, 0.005):
        drops = []
        for _ in range(8):
            e = sd * rng.standard_normal(v0.shape)
            drops.append(mean_line(base.with_flat(v0 + e), xp2) - a0 - float(g @ e))
        c = float(np.mean(drops))
        R = [r for r in recs if r["name"] == "example_noisy" and r["sd"] == sd]
        est, truth = [], []
        for r in R:
            m = m16_suite.noisy_model(base, sd, r["copy"])
            est.append(a0 + float(g @ (m.flat() - v0)) + c)
            truth.append(r["ref"])
        est, truth = np.array(est), np.array(truth)
        e = est - truth
        pq = float(np.mean(truth * (1 - truth)))
        ops = 3 * v0.size + 2
        rms = float(np.sqrt((e ** 2).mean()))
        out[f"sd{sd}"] = dict(copies=len(R), curvature_drop=c, rms=rms, bias=float(e.mean()),
                              const_base_rms=float(np.sqrt(((a0 - truth) ** 2).mean())), ops=ops,
                              sampling_inputs_matched=pq / rms ** 2, per_input_sd=math.sqrt(pq),
                              linear_term_sd=float(np.std([g @ (m16_suite.noisy_model(base, sd, r["copy"]).flat() - v0)
                                                           for r in R])))
        print(sd, out[f"sd{sd}"], flush=True)
    with open(os.path.join(m43.RES, "m16_taylor.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
