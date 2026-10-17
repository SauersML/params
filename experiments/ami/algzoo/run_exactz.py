"""Exact mechanistic estimates (exact integer geometry) for AlgZoo RNNs of sequence length 4 to 6, scored by
ARC's two criteria. One JSON line per model in ~/mpd-data/ami/algzoo/results_exactn.jsonl.

usage: run_exactz.py TASK_D_N [TASK_D_N ...]   (e.g. 2nd_argmax_4_4); noisy copies K from env ALGZOO_K (16).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate  # noqa: E402
import exactz  # noqa: E402
import zoo  # noqa: E402

OUT = os.path.expanduser("~/mpd-data/ami/algzoo/results_exactn.jsonl")


def est(m):
    ops = exactz.Ops()
    v = exactz.mse_fast(m, ops) if m.task == "median" else exactz.accuracy_by_class(m, ops)[0]
    return v, ops


def main():
    K = int(os.environ.get("ALGZOO_K", "16"))
    for spec in sys.argv[1:]:
        task, d, n = spec.rsplit("_", 2)
        d, n = int(d), int(n)
        m = zoo.load(task, d, n)
        t0 = time.time()
        value, ops = est(m)
        sec = time.time() - t0
        rng = np.random.default_rng(d * 1000 + n)
        sampled, se = zoo.sampled_metric(m, task, n, 1 << 24, rng)
        unresolved = []

        def est_ops(q):
            v, o = est(q)
            unresolved.append(o.unresolved)
            return v, o.n

        noisy = evaluate.noisy_protocol(m, est_ops, K=K, sd=0.05, seed=d * 100 + n)
        noisy["unresolved_max"] = float(max(unresolved))
        p = m.n_params
        r = {
            "task": task, "d": d, "n": n, "params": p, "seed": zoo.BEST_SEED[task][(d, n)],
            "metric": "mse" if task == "median" else "accuracy",
            "exact": value, "unresolved_bound": ops.unresolved, "exact_ops": ops.n, "exact_sec": sec,
            "sampled_2e24": sampled, "sampled_se": se, "z": (sampled - value) / max(se, 1e-300),
            "sample_cost_ops": evaluate.sample_cost(m), "samples_at_equal_ops": ops.n / evaluate.sample_cost(m),
            "surprise_sign_bits": ops.bits, "weight_bits_float32": 32 * p, "noisy": noisy,
        }
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
