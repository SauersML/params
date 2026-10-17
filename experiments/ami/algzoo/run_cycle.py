"""Exhaustive accuracy for the AlgZoo longest-cycle transformers, scored by ARC's two criteria.

The input is a function f: {0..n-1} -> {0..n-1} written as x_i = f(i), uniform over all n^n functions, and the label
is (length of the longest cycle of f) - 1. The input space is finite, so the accuracy is an exact finite sum over
all n^n inputs: a deductive estimate with zero error. It is compared with sampling at the same operation count,
and its surprise (one check per input, costing log2 of the inverse prior probability of the observed outcome under
a uniform-guess prior) with the bits in the float32 weights.
Writes ~/mpd-data/ami/algzoo/results_cycle.jsonl.
"""
import json
import math
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoo  # noqa: E402

OUT = os.path.expanduser("~/mpd-data/ami/algzoo/results_cycle.jsonl")


def pass_cost(m):
    n, h = m.n, m.h
    per_layer = 2 * 3 * n * h * h + 2 * n * n * h + 3 * n * n + 2 * n * n * h + 2 * n * h * h + n * h
    return n * h + len(m.layers) * per_layer + 2 * n * h + n


def float32_predictions(task_d, task_n, x):
    sd = th.load(zoo.path("longest_cycle", task_d, task_n), weights_only=True, map_location="cpu")
    d, n = task_d, task_n
    emb = th.nn.Embedding(n, d)
    emb.weight.data = sd["embed.weight"]
    pos = th.nn.Embedding(n, d)
    pos.weight.data = sd["pos_embed.weight"]
    attns = []
    for i in range(2):
        a = th.nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
        a.in_proj_weight.data = sd[f"attns.{i}.in_proj_weight"]
        a.out_proj.weight.data = sd[f"attns.{i}.out_proj.weight"]
        attns.append(a)
    with th.no_grad():
        xt = th.tensor(x)
        s = emb(xt) + pos(th.arange(n)[None])
        for a in attns:
            s = s + a(s, s, s, need_weights=False)[0]
        return (s @ sd["unembed.weight"].T)[:, -1].argmax(-1).numpy()


def main():
    rows = []
    for p, task, d, n in zoo.models():
        if task != "longest_cycle":
            continue
        m = zoo.load(task, d, n)
        x = zoo.all_inputs_discrete(n)
        y = zoo.label(task, x)
        logits = m(x)
        pred = logits.argmax(-1)
        acc = float((pred == y).mean())
        acc32 = float((float32_predictions(d, n, x) == y).mean())
        N = len(x)
        prior = np.bincount(y, minlength=n) / N
        # surprise of the per-input checks under the prior "the model outputs class c with the label frequency"
        correct = pred == y
        bits = float(np.sum(np.where(correct, -np.log2(np.maximum(prior[y], 1e-300)), 0.0))
                     + np.sum(~correct) * math.log2(n))
        c = pass_cost(m)
        # noisy copies: exhaustive accuracy of each, and sampling's RMS error at the same cost (N samples)
        rng = np.random.default_rng(d * 100 + n)
        accs = []
        for _ in range(64):
            q = zoo.load(task, d, n)
            q.E = q.E + 0.05 * rng.standard_normal(q.E.shape)
            q.P = q.P + 0.05 * rng.standard_normal(q.P.shape)
            q.U = q.U + 0.05 * rng.standard_normal(q.U.shape)
            q.layers = [tuple(w + 0.05 * rng.standard_normal(w.shape) for w in layer) for layer in q.layers]
            accs.append(float((q(x).argmax(-1) == y).mean()))
        accs = np.array(accs)
        samp_rms = float(math.sqrt(np.mean(accs * (1 - accs)) / N))
        margin = np.sort(logits, -1)
        r = {
            "task": task, "d": d, "n": n, "params": p, "seed": zoo.BEST_SEED[task][(d, n)],
            "inputs": N, "accuracy": acc, "accuracy_float32_torch": acc32,
            "min_top2_logit_gap": float((margin[:, -1] - margin[:, -2]).min()),
            "label_prior": prior.tolist(), "pass_ops": c, "exhaustive_ops": c * N,
            "surprise_bits_enumeration": bits, "weight_bits_float32": 32 * p,
            "noisy": {"K": 64, "sd": 0.05, "accuracy_mean": float(accs.mean()), "accuracy_sd": float(accs.std()),
                      "sampling_rms_at_equal_ops": samp_rms},
        }
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
