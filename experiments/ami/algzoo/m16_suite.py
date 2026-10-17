"""Evaluation suite for M16,10 estimators: the challenge model (example_2nd_argmax), the five zoo seeds of
2nd_argmax 16,10, and noisy copies of the challenge model, each with a high-precision reference accuracy.

Reference: the exact probability along the line of the last input (lastline.py) averaged over many Gaussian prefixes;
this is sampling (it is the truth for scoring, not an estimator). Standard errors are recorded.
Writes ~/mpd-data/ami/algzoo/cloud/m16_suite.jsonl (one line per model; noisy copies carry their seed and noise sd).

Usage: m16_suite.py trained [names] | noisy SD K [worker NW]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lastline  # noqa: E402
import zoo  # noqa: E402

OUT = os.path.expanduser("~/mpd-data/ami/algzoo/cloud/m16_suite.jsonl")


def reference(m, n_prefix, seed):
    rng = np.random.default_rng(seed)
    tot = sq = 0.0
    N = 0
    chunk = 1 << 18
    while N < n_prefix:
        xp = rng.standard_normal((chunk, m.n - 1))
        g, _ = lastline.line_prob(m, xp)
        tot += g.sum()
        sq += (g * g).sum()
        N += chunk
    mean = tot / N
    var = sq / N - mean * mean
    return mean, float(np.sqrt(var / N)), var


def noisy_model(base_model, sd, k):
    rng = np.random.default_rng(1_000_000 + int(round(sd * 1e6)) * 1000 + k)
    v = base_model.flat()
    return base_model.with_flat(v + sd * rng.standard_normal(v.shape))


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            keys.add((r["name"], r.get("sd"), r.get("copy")))
    return keys


def emit(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mode = sys.argv[1]
    have = done_keys()
    if mode == "trained":
        specs = [("example", ("example", 16, 10, None))] + [(f"zoo_seed{s}", ("2nd_argmax", 16, 10, s)) for s in range(5)]
        want = set(sys.argv[2:])
        for name, (task, d, n, s) in specs:
            if (name, None, None) in have or (want and name not in want):
                continue
            m = zoo.load(task, d, n, s)
            # 2^24 prefixes for the challenge model, 2^22 for the seeds (standard errors ~4e-5 and ~8e-5)
            mean, se, var = reference(m, 1 << (24 if name == "example" else 22), 7 + (s or 0))
            emit(dict(name=name, task=task, seed=s, ref=mean, ref_se=se, line_var=var, pq=mean * (1 - mean)))
    else:
        sd, K = float(sys.argv[2]), int(sys.argv[3])
        w, nw = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
        base = zoo.load("example", 16, 10)
        for k in range(w, K, nw):
            if ("example_noisy", sd, k) in have:
                continue
            m = noisy_model(base, sd, k)
            mean, se, var = reference(m, 1 << 21, 99 + k)
            emit(dict(name="example_noisy", sd=sd, copy=k, ref=mean, ref_se=se, line_var=var, pq=mean * (1 - mean)))


if __name__ == "__main__":
    main()
