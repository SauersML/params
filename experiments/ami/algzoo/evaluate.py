"""Score a mechanistic estimator of an AlgZoo model by ARC's two criteria.

(a) Error against compute (ARC's practical bar). Following Hilton's M4,3 notes, Gaussian noise (sd 0.05) is added to
    every weight and the estimator runs on each noisy copy. Its root-mean-square error against the true metric is
    compared with random sampling at the same operation count: sampling with k inputs has RMS error
    sqrt(mean over copies of Var(per-input score) / k). The true metric of each copy is checked independently with
    2^20 fresh samples; if the estimator is exact, the z-scores (estimate - sampled) / SE are standard normal.
(b) Surprise accounting. The explanation's surprise is counted as one bit per sign fact the enumeration establishes
    (one per neuron, step and cell, and one per logit comparison on each final piece). ARC compares the total with
    the bits of optimization that selected the model, which is at most the bits the float32 weights occupy.
"""
import math

import numpy as np

import zoo


def sample_cost(model):
    """Operations for one sampled input: the forward pass, the label and the comparison."""
    d, n, k = model.d, model.n, model.W_oh.shape[0]
    return n * (2 * d * d + 3 * d) + 2 * k * d + int(n * math.log2(max(n, 2))) + k + 1


def per_input_var(model, task, rng, B=1 << 20):
    s = zoo.metric(model, task, zoo.sample_inputs(task, model.n, B, rng))
    return float(s.var()), float(s.mean()), float(s.std() / math.sqrt(B))


def noisy_protocol(model, estimator, K=64, sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    task = model.task
    base = model.flat()
    rows = []
    for _ in range(K):
        m = model.with_flat(base + sd * rng.standard_normal(base.shape))
        est, ops = estimator(m)
        var, sampled, se = per_input_var(m, task, rng)
        rows.append((est, ops, var, sampled, se))
    est, ops, var, sampled, se = map(np.array, zip(*rows))
    z = (sampled - est) / np.maximum(se, 1e-300)
    c = sample_cost(model)
    k_equal = ops.mean() / c
    return {
        "K": K, "sd": sd,
        "metric_mean": float(sampled.mean()), "metric_sd_across_copies": float(sampled.std()),
        "mean_z2": float((z ** 2).mean()), "max_abs_z": float(np.abs(z).max()),
        "estimator_ops_mean": float(ops.mean()), "sample_cost_ops": c,
        "samples_at_equal_ops": float(k_equal),
        "sampling_rms_at_equal_ops": float(math.sqrt(var.mean() / max(k_equal, 1e-300))),
        "check_rms_se": float(math.sqrt((se ** 2).mean())),
    }

