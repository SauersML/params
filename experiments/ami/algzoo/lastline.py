"""Exact probability of a correct 2nd-argmax (or argmedian) prediction along the line of the last input.

Given the first n-1 inputs, the RNN's state h_(n-1) is fixed and the last input t enters only through
z = W_hh h_(n-1) + W_hi t, so neuron j switches once, at t = -(W_hh h)_j / W_hi_j, and between switches the logits are
P + Q t. The label is fixed between the points where t passes the prefix's largest and second largest values. So the
set of t where the model is right is a union of at most d + 3 intervals with exact ends, and for t ~ N(0, 1) its
probability is a sum of Phi differences. Batched over prefixes (numpy), with an exact op count per prefix.

Ops convention (as in aq3.py): scalar add, multiply, divide, compare and elementary-function calls (Phi included)
count 1 each.
"""
import math

import numpy as np
from scipy.special import ndtr, stdtr


def prefix_state(model, xp):
    """xp (B, n-1) -> h_(n-1) (B, d)."""
    h = np.zeros((xp.shape[0], model.d))
    for t in range(xp.shape[1]):
        h = np.maximum(h @ model.W_hh.T + xp[:, t : t + 1] * model.W_hi, 0.0)
    return h


def prefix_ops(model, p):
    d = model.d
    return 2 * d + (p - 1) * (2 * d * d + 2 * d)


def line_prob(model, xp, task="2nd_argmax", h=None, radial=False):
    """P(model right | x_0..x_(n-2) = xp) for x_(n-1) ~ N(0, 1), exact. Returns (prob (B,), ops per prefix).
    radial=True: P(model right | direction of the prefix), integrating the prefix's radius as well (positive
    homogeneity): with the prefix scaled to unit norm, the last input over the prefix norm, T = x_(n-1) / |prefix|,
    has sqrt(n-1) T ~ Student t with n-1 degrees of freedom."""
    if radial:
        xp = xp / np.linalg.norm(xp, axis=1, keepdims=True)
        h = None
    B, p = xp.shape
    n = p + 1
    if radial:
        sq = math.sqrt(p)
        cdf = lambda u: stdtr(p, sq * u)
    else:
        cdf = ndtr
    d, W, w, Wo = model.d, model.W_hh, model.W_hi, model.W_oh
    k = Wo.shape[0]
    if h is None:
        h = prefix_state(model, xp)
    c = h @ W.T  # (B, d)
    wz = np.where(w == 0.0, 1.0, w)
    tau = -c / wz[None]  # switch points
    tau[:, w == 0.0] = np.inf
    on0 = np.broadcast_to(w < 0, (B, d)).copy()  # at t -> -inf
    on0[:, w == 0.0] = c[:, w == 0.0] > 0
    # label switch points: prefix order statistics
    if task == "2nd_argmax":
        o = np.argsort(-xp, 1)
        if p == 1:  # one earlier input: below it the newest input is the second largest
            a = b = o[:, 0]
            M, S = xp[:, 0], np.full(B, -np.inf)
        else:
            a, b = o[:, 0], o[:, 1]
            M, S = xp[np.arange(B), a], xp[np.arange(B), b]
        lab_pts = np.stack([S, M], 1)  # t < S: label b; S < t < M: label n-1; t > M: label a
        labs = np.stack([b, np.full(B, n - 1), a], 1)
    elif task == "argmedian":
        # median position of n values: with t inserted, the median rank is n // 2 (0 = largest)
        o = np.argsort(-xp, 1)
        r = n // 2
        lo_v = xp[np.arange(B), o[:, r]]  # if t < lo_v, the median is the prefix value at rank r
        hi_v = xp[np.arange(B), o[:, r - 1]]  # if t > hi_v, the median is the prefix value at rank r - 1
        lab_pts = np.stack([lo_v, hi_v], 1)
        labs = np.stack([o[:, r], np.full(B, n - 1), o[:, r - 1]], 1)
    else:
        raise ValueError(task)
    # events: d neuron switches (kind 0..d-1) and 2 label switches (kind d, d+1)
    ev_t = np.concatenate([tau, lab_pts], 1)
    order = np.argsort(ev_t, 1, kind="stable")
    ev_sorted = np.take_along_axis(ev_t, order, 1)
    on = on0.astype(float)
    P = np.einsum("od,bd->bo", Wo, on * c)
    Q = np.einsum("od,bd->bo", Wo, on * w[None])
    lab = labs[:, 0].copy()
    lab_state = np.zeros(B, dtype=int)
    tot = np.zeros(B)
    left = np.full(B, -np.inf)
    ar = np.arange(B)
    m = d + 2
    n_on = on.sum(1)
    for i in range(m + 1):
        right = ev_sorted[:, i] if i < m else np.full(B, np.inf)
        s0, s1 = left.copy(), right.copy()
        Pl, Ql = P[ar, lab], Q[ar, lab]
        dp = Pl[:, None] - P
        dq = Ql[:, None] - Q
        dp[ar, lab] = 1.0
        dq[ar, lab] = 0.0
        # exact ties when every neuron is off: argmax picks the first index
        dead = n_on == 0
        idx = np.arange(k)[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -dp / dq
        lower = np.where(dq > 0, r, -np.inf).max(1)
        upper = np.where(dq < 0, r, np.inf).min(1)
        never = ((dq == 0) & (dp < 0)).any(1)
        tie_bad = dead & (lab > 0)
        s0 = np.maximum(s0, lower)
        s1 = np.minimum(s1, upper)
        good = (s1 > s0) & ~never & ~tie_bad
        tot += np.where(good, cdf(np.where(good, s1, 0)) - cdf(np.where(good, s0, 0)), 0.0)
        if i == m:
            break
        kind = order[:, i]
        is_n = kind < d
        j = np.where(is_n, kind, 0)
        sgn = np.where(on[ar, j] > 0, -1.0, 1.0) * is_n
        on[ar, j] = np.where(is_n, 1.0 - on[ar, j], on[ar, j])
        n_on = on.sum(1)
        P = P + sgn[:, None] * Wo[:, j].T * c[ar, j][:, None]
        Q = Q + sgn[:, None] * Wo[:, j].T * w[j][:, None]
        # recompute exactly when everything is off (avoids rounding residue in a dead state)
        P[n_on == 0] = 0.0
        Q[n_on == 0] = 0.0
        is_l = ~is_n
        lab_state = lab_state + is_l
        lab = np.where(is_l, labs[ar, np.minimum(lab_state, 2)], lab)
        left = ev_sorted[:, i]
    ops = line_ops(model, p)
    return tot, ops


def line_ops(model, p):
    """Ops for one prefix: the prefix forward pass, c = W h, switch points, sort, initial P and Q, and per interval the
    constraint checks, the Phi differences and the logit updates at each switch."""
    d, k = model.d, model.W_oh.shape[0]
    m = d + 2
    per_interval = (k - 1) * 5 + 4 + 3
    per_switch = 4 * k
    order_stats = 2 * p  # largest and second largest of the prefix
    return (prefix_ops(model, p) + (2 * d * d - d) + 2 * d + int(m * math.log2(m)) + 1 + 4 * k * d // 2
            + order_stats + (m + 1) * per_interval + d * per_switch)
