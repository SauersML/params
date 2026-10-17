"""Exact mechanistic estimates for every AlgZoo RNN of sequence length 2 or 3, scored by ARC's two criteria.

Writes one JSON line per model to ~/mpd-data/ami/algzoo/results_exact.jsonl.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate  # noqa: E402
import exact2  # noqa: E402
import exact3  # noqa: E402
import zoo  # noqa: E402

OUT = os.path.expanduser("~/mpd-data/ami/algzoo/results_exact.jsonl")


def estimator_for(model):
    mod = exact2 if model.n == 2 else exact3
    fn = mod.mse if model.task == "median" else mod.accuracy

    def est(m):
        ops = mod.Ops()
        v = fn(m, ops)
        return v, ops.n

    def est_bits(m):
        ops = mod.Ops()
        v = fn(m, ops)
        return v, ops.n, ops.bits

    return est, est_bits


def main():
    only = set(sys.argv[1:])
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["task"], r["d"], r["n"]))
    rng = np.random.default_rng(7)
    for p, task, d, n in zoo.models():
        if task == "longest_cycle" or n > 3 or (task, d, n) in done:
            continue
        if only and f"{task}_{d}_{n}" not in only:
            continue
        m = zoo.load(task, d, n)
        est, est_bits = estimator_for(m)
        value, ops, bits = est_bits(m)
        sampled, se = zoo.sampled_metric(m, task, n, 1 << 24, rng)
        noisy = evaluate.noisy_protocol(m, est, K=64, sd=0.05, seed=d * 100 + n)
        r = {
            "task": task, "d": d, "n": n, "params": p, "seed": zoo.BEST_SEED[task][(d, n)],
            "metric": "mse" if task == "median" else "accuracy",
            "exact": value, "exact_ops": ops, "sampled_2e24": sampled, "sampled_se": se,
            "z": (sampled - value) / max(se, 1e-300),
            "sample_cost_ops": evaluate.sample_cost(m),
            "samples_at_equal_ops": ops / evaluate.sample_cost(m),
            "surprise_sign_bits": bits, "weight_bits_float32": 32 * p,
            "noisy": noisy,
        }
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
