"""AlgZoo models (ARC, Jan 2026) as float64 numpy weights, with exact forward passes and task labels.

Weights come from fetch.py (ARC's public bucket gs://arc-ml-public/alg/zoo, mirrored in ~/mpd-data/ami/algzoo/weights).
The RNN forward pass matches alg_zoo.architectures.OneLayerRNN (torch.nn.RNN, relu, no bias):
    h_0 = 0, h_{t+1} = relu(W_hh h_t + W_hi x_t), logits = W_oh h_n.
The transformer forward pass matches AttentionOnlyTransformer (2 attention-only layers, 1 head, no bias, no layer norm,
output read at the last position).
"""
import math
import os

import numpy as np
import torch as th

WEIGHTS = os.path.expanduser("~/mpd-data/ami/algzoo/weights")
FULL = 2**30

BEST_SEED = {
    "2nd_argmax": {
        (2, 2): 2, (2, 3): 0, (2, 4): 2, (2, 5): 3, (2, 6): 2, (2, 8): 1, (2, 10): 4,
        (3, 2): 3, (3, 3): 1, (3, 4): 4, (3, 5): 0, (3, 6): 0, (3, 8): 1, (3, 10): 1,
        (4, 2): 3, (4, 3): 0, (4, 4): 1, (4, 5): 2, (4, 6): 0, (4, 8): 1, (4, 10): 2,
        (5, 2): 3, (5, 3): 1, (5, 4): 4, (5, 5): 3, (5, 6): 4, (5, 8): 4, (5, 10): 4,
        (6, 2): 2, (6, 3): 3, (6, 4): 0, (6, 5): 1, (6, 6): 1, (6, 8): 0, (6, 10): 2,
        (8, 2): 3, (8, 3): 1, (8, 4): 1, (8, 5): 3, (8, 6): 4, (8, 8): 0, (8, 10): 4,
        (16, 2): 3, (16, 3): 1, (16, 4): 3, (16, 5): 2, (16, 6): 3, (16, 8): 1, (16, 10): 2,
        (32, 2): 3, (32, 3): 4, (32, 4): 2, (32, 5): 3, (32, 6): 1, (32, 8): 4, (32, 10): 0,
    },
    "argmedian": {
        (2, 3): 0, (2, 5): 0, (2, 7): 0, (2, 11): 3, (4, 3): 0, (4, 5): 2, (4, 7): 1, (4, 11): 1,
        (8, 3): 1, (8, 5): 2, (8, 7): 0, (8, 11): 4, (16, 3): 1, (16, 5): 0, (16, 7): 4, (16, 11): 4,
        (32, 3): 4, (32, 5): 3, (32, 7): 3, (32, 11): 4,
    },
    "median": {
        (2, 2): 2, (2, 3): 0, (2, 5): 0, (2, 10): 0, (4, 2): 4, (4, 3): 0, (4, 5): 4, (4, 10): 1,
        (8, 2): 3, (8, 3): 3, (8, 5): 1, (8, 10): 3, (16, 2): 1, (16, 3): 1, (16, 5): 0, (16, 10): 2,
        (32, 2): 1, (32, 3): 1, (32, 5): 1, (32, 10): 0,
    },
    "longest_cycle": {
        (2, 3): 0, (2, 4): 2, (2, 5): 3, (2, 6): 2, (3, 3): 0, (3, 4): 4, (3, 5): 1, (3, 6): 0,
        (4, 3): 2, (4, 4): 1, (4, 5): 0, (4, 6): 0, (6, 3): 0, (6, 4): 0, (6, 5): 2, (6, 6): 3,
        (8, 3): 0, (8, 4): 3, (8, 5): 3, (8, 6): 3,
    },
}
KIND = {"2nd_argmax": "dist", "argmedian": "dist", "median": "scalar", "longest_cycle": "dist"}


def path(task, d, n, seed=None):
    if task == "example":
        return os.path.join(WEIGHTS, "one_layer_16_hidden_94_acc_2nd_argmax.pth")
    if seed is None:
        seed = BEST_SEED[task][(d, n)]
    return os.path.join(WEIGHTS, f"{task}_{d}_{n}_{FULL}_{seed}.pth")


class RNN:
    """W_hi (d,), W_hh (d,d), W_oh (n_out,d) in float64; n = sequence length."""

    def __init__(self, W_hi, W_hh, W_oh, n, task):
        self.W_hi = np.asarray(W_hi, dtype=np.float64).reshape(-1)
        self.W_hh = np.asarray(W_hh, dtype=np.float64)
        self.W_oh = np.asarray(W_oh, dtype=np.float64)
        self.n, self.task, self.d = n, task, self.W_hi.shape[0]

    @property
    def n_params(self):
        return self.W_hi.size + self.W_hh.size + self.W_oh.size

    def hidden(self, x):
        """x (B, n) -> list of hidden states h_1..h_n, each (B, d)."""
        h = np.zeros((x.shape[0], self.d))
        hs = []
        for t in range(x.shape[1]):
            h = np.maximum(h @ self.W_hh.T + x[:, t : t + 1] * self.W_hi, 0.0)
            hs.append(h)
        return hs

    def __call__(self, x):
        out = self.hidden(x)[-1] @ self.W_oh.T
        return out[:, 0] if self.task == "median" else out

    def copy(self):
        return RNN(self.W_hi.copy(), self.W_hh.copy(), self.W_oh.copy(), self.n, self.task)

    def flat(self):
        return np.concatenate([self.W_hi, self.W_hh.ravel(), self.W_oh.ravel()])

    def with_flat(self, v):
        d, k = self.d, self.W_oh.shape[0]
        return RNN(v[:d], v[d : d + d * d].reshape(d, d), v[d + d * d :].reshape(k, d), self.n, self.task)


def load_rnn(task, d, n, seed=None):
    sd = th.load(path(task, d, n, seed), weights_only=True, map_location="cpu")
    if task == "example":
        task, n = "2nd_argmax", sd["linear.weight"].shape[0]
    return RNN(sd["rnn.weight_ih_l0"].double().numpy(), sd["rnn.weight_hh_l0"].double().numpy(),
               sd["linear.weight"].double().numpy(), n, task)


class Transformer:
    """Attention-only transformer of alg_zoo: token + position embedding, L layers of 1-head attention, unembed at last
    position. Weights float64."""

    def __init__(self, sd):
        g = lambda k: sd[k].double().numpy()
        self.E, self.P, self.U = g("embed.weight"), g("pos_embed.weight"), g("unembed.weight")
        self.layers = []
        i = 0
        while f"attns.{i}.in_proj_weight" in sd:
            Wqkv, Wo = g(f"attns.{i}.in_proj_weight"), g(f"attns.{i}.out_proj.weight")
            h = Wo.shape[0]
            self.layers.append((Wqkv[:h], Wqkv[h : 2 * h], Wqkv[2 * h :], Wo))
            i += 1
        self.n, self.h, self.task = self.P.shape[0], self.E.shape[1], "longest_cycle"

    @property
    def n_params(self):
        return self.E.size + self.P.size + self.U.size + sum(sum(w.size for w in l) for l in self.layers)

    def residuals(self, x, attn_override=None):
        """x (B, n) ints -> list of residual streams (B, n, h) before each layer and after the last; plus attention
        patterns per layer (B, n, n). Attention is bidirectional (nn.MultiheadAttention without a mask)."""
        s = self.E[x] + self.P[None, : x.shape[1]]
        res, pats = [s], []
        for li, (Wq, Wk, Wv, Wo) in enumerate(self.layers):
            q, k, v = s @ Wq.T, s @ Wk.T, s @ Wv.T
            sc = q @ k.transpose(0, 2, 1) / math.sqrt(self.h)
            sc = sc - sc.max(-1, keepdims=True)
            a = np.exp(sc)
            a /= a.sum(-1, keepdims=True)
            if attn_override is not None and li in attn_override:
                a = attn_override[li](a)
            pats.append(a)
            s = s + (a @ v) @ Wo.T
            res.append(s)
        return res, pats

    def __call__(self, x):
        res, _ = self.residuals(x)
        return res[-1][:, -1] @ self.U.T


def load_transformer(d, n, seed=None):
    return Transformer(th.load(path("longest_cycle", d, n, seed), weights_only=True, map_location="cpu"))


def load(task, d, n, seed=None):
    return load_transformer(d, n, seed) if task == "longest_cycle" else load_rnn(task, d, n, seed)


def label(task, x):
    if task == "2nd_argmax":
        return np.argsort(x, -1, kind="stable")[:, -2]
    if task == "argmedian":
        return np.argsort(x, -1, kind="stable")[:, x.shape[1] // 2]
    if task == "median":
        return np.median(x, -1)
    if task == "longest_cycle":
        B, n = x.shape
        best = np.zeros(B, dtype=int)
        for s in range(n):
            pos = np.full(B, s)
            first = np.zeros(B, dtype=int)
            for k in range(1, n + 1):
                pos = x[np.arange(B), pos]
                hit = (pos == s) & (first == 0)
                first[hit] = k
            best = np.maximum(best, first)
        return best - 1
    raise ValueError(task)


def sample_inputs(task, n, B, rng):
    if task == "longest_cycle":
        return rng.integers(0, n, size=(B, n))
    return rng.standard_normal((B, n))


def all_inputs_discrete(n):
    grids = np.meshgrid(*[np.arange(n)] * n, indexing="ij")
    return np.stack([g.ravel() for g in grids], 1)


def metric(model, task, x):
    """Per-input score: 1{correct} for classification tasks, squared error for median."""
    y = label(task, x)
    out = model(x)
    if task == "median":
        return (out - y) ** 2
    return (out.argmax(-1) == y).astype(np.float64)


def sampled_metric(model, task, n, B, rng, chunk=1 << 20):
    tot, sq, cnt = 0.0, 0.0, 0
    while cnt < B:
        b = min(chunk, B - cnt)
        m = metric(model, task, sample_inputs(task, n, b, rng))
        tot += m.sum()
        sq += (m * m).sum()
        cnt += b
    mean = tot / cnt
    return mean, math.sqrt(max(sq / cnt - mean * mean, 0.0) / cnt)


def models():
    """Every official zoo model (best seed) as (task, d, n), smallest parameter count first."""
    out = []
    for task, table in BEST_SEED.items():
        for d, n in table:
            if task == "longest_cycle":
                p = 3 * n * d + 8 * d * d
            elif task == "median":
                p = d + d * d + d
            else:
                p = d + d * d + n * d
            out.append((p, task, d, n))
    return sorted(out)
