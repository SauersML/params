"""Test every AlgZoo RNN for George Robinson's symmetry: hidden neurons come in pairs (i, i') with
h_i = ReLU(v_k) and h_i' = ReLU(-v_k), so that the RNN is exactly a smaller recurrent network with absolute-value
read-out:
    v_(t+1) = B v_t + a x_t   (linear in v, because ReLU(v) - ReLU(-v) = v),   logits = C |v_n|.
Then the model is logits = C |A x| with A = [B^(n-1) a, ..., B a, a], a readable algorithm with d/2 linear
functionals of the input.

Pairing: after rescaling each neuron to |W_hi| = 1, pair neurons of opposite input sign greedily by the distance of
their output-weight columns (a pair shares its output column, V = [I, I]). Neurons with |W_hi| below 1e-3 of the
largest cannot be paired and make the test fail. The reduced weights are least-squares fits (as in Hilton's notes).
Reported: relative residual of W_hh, prediction agreement and KL(model || reduced) on 2^18 Gaussian inputs, and
both accuracies. Writes ~/mpd-data/ami/algzoo/absreduce.jsonl.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoo  # noqa: E402


def pairing(m):
    s = np.abs(m.W_hi)
    if s.min() < 1e-3 * s.max() or m.d % 2:
        return None
    Wo = m.W_oh * s[None]  # output columns in units where |W_hi| = 1
    pos = [i for i in range(m.d) if m.W_hi[i] > 0]
    neg = [i for i in range(m.d) if m.W_hi[i] < 0]
    if len(pos) != len(neg):
        return None
    pairs, free = [], set(neg)
    for i in sorted(pos, key=lambda i: -np.linalg.norm(Wo[:, i])):
        j = min(free, key=lambda j: np.linalg.norm(Wo[:, i] - Wo[:, j]))
        free.remove(j)
        pairs.append((i, j))
    return pairs


def reduce(m, pairs):
    """Least-squares fit of W_hi = U^T a, W_hh = U^T B U, W_oh = C V in the rescaled basis (|W_hi| = 1)."""
    s = np.abs(m.W_hi)
    Whi, Whh, Woh = m.W_hi / s, (m.W_hh * s[None]) / s[:, None], m.W_oh * s[None]
    k = len(pairs)
    U = np.zeros((k, m.d))
    V = np.zeros((k, m.d))
    for r, (i, j) in enumerate(pairs):
        U[r, i], U[r, j] = 1, -1
        V[r, i], V[r, j] = 1, 1
    a = np.linalg.lstsq(U.T, Whi, rcond=None)[0]
    B = np.linalg.lstsq(np.kron(U.T, U.T), Whh.ravel(), rcond=None)[0].reshape(k, k)
    C = np.linalg.lstsq(V.T, Woh.T, rcond=None)[0].T
    resid = np.linalg.norm(Whh - U.T @ B @ U) / np.linalg.norm(Whh)
    return a, B, C, resid


def main():
    rng = np.random.default_rng(0)
    out = open(os.path.expanduser("~/mpd-data/ami/algzoo/absreduce.jsonl"), "w")
    for p, task, d, n in zoo.models():
        if task == "longest_cycle":
            continue
        m = zoo.load(task, d, n)
        pairs = pairing(m)
        r = {"task": task, "d": d, "n": n, "params": p, "paired": pairs is not None}
        if pairs is not None:
            a, B, C, resid = reduce(m, pairs)
            A = np.stack([np.linalg.matrix_power(B, n - 1 - t) @ a for t in range(n)], 1)
            x = rng.standard_normal((1 << 18, n))
            lm = m(x)
            lr = np.abs(x @ A.T) @ C.T
            if task == "median":
                lr = lr[:, 0]
                r.update(whh_rel_resid=resid, mse_model=float(((lm - np.median(x, 1)) ** 2).mean()),
                         mse_reduced=float(((lr - np.median(x, 1)) ** 2).mean()),
                         rms_output_diff=float(np.sqrt(((lm - lr) ** 2).mean())))
            else:
                y = zoo.label(task, x)
                zm = lm - lm.max(1, keepdims=True)
                zr = lr - lr.max(1, keepdims=True)
                lpm = zm - np.log(np.exp(zm).sum(1, keepdims=True))
                lpr = zr - np.log(np.exp(zr).sum(1, keepdims=True))
                r.update(whh_rel_resid=resid, agreement=float((lm.argmax(1) == lr.argmax(1)).mean()),
                         kl=float((np.exp(lpm) * (lpm - lpr)).sum(1).mean()),
                         acc_model=float((lm.argmax(1) == y).mean()), acc_reduced=float((lr.argmax(1) == y).mean()))
            r["A_rows_sum_over_norm"] = [float(abs(row.sum()) / np.linalg.norm(row)) for row in A]
        out.write(json.dumps(r) + "\n")
        out.flush()
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
