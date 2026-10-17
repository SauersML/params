"""ARC's stronger test for estimators: the error over the 5 training seeds of each configuration (not only noisy
copies of the best seed). Exact estimates for all seeds of every sequence-length-2 and -3 RNN configuration, each
checked against 2^22 fresh samples. Writes ~/mpd-data/ami/algzoo/seeds_check.jsonl."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact2  # noqa: E402
import exact3  # noqa: E402
import zoo  # noqa: E402

OUT = os.path.expanduser("~/mpd-data/ami/algzoo/seeds_check.jsonl")


def main():
    rng = np.random.default_rng(2026)
    with open(OUT, "w") as f:
        for p, task, d, n in zoo.models():
            if task == "longest_cycle" or n > 3 or (task == "argmedian" and n == 3):
                continue
            mod = exact2 if n == 2 else exact3
            for seed in range(5):
                m = zoo.load(task, d, n, seed=seed)
                v = mod.mse(m) if task == "median" else mod.accuracy(m)
                s, se = zoo.sampled_metric(m, task, n, 1 << 22, rng)
                r = {"task": task, "d": d, "n": n, "seed": seed, "exact": v, "sampled": s, "se": se,
                     "z": (s - v) / max(se, 1e-300)}
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
