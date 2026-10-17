"""Angular-quadrature estimator of the accuracy of a 2nd-argmax RNN with sequence length 3 (deterministic, no sampling).

Gaussian x in R^3 and a positively homogeneous model: the accuracy is the share of the unit sphere where the model is
right. Write x = r (cos a, sin a, t) / |(cos a, sin a, t)|. For a fixed angle a of the first two inputs the first two
RNN steps give a fixed hidden state h_2(a); the last input enters as z_3 = W_hh h_2 + W_hi t, so neuron j switches at
t = -(W_hh h_2)_j / W_hi_j and between switches the logits are linear in t. The label is fixed between the points
t = cos a and t = sin a. So the set of t where the model is right is a union of at most d + 3 intervals with closed-form
endpoints, and its measure is exact: t has density (1/2)(1 + t^2)^(-3/2) given a, so an interval [s0, s1] has measure
(G(s1) - G(s0)) / 2 with G(t) = t / sqrt(1 + t^2).

Only the angle a is integrated numerically: a deterministic quadrature rule on [0, 2 pi), either the periodic
trapezoid rule with N nodes or Gauss-Legendre on the arcs between the angles where the first two steps change their
activation pattern (those arcs come from the length-2 enumeration, exact2-style). The accuracy is
    (1 / 4 pi) * integral over a of sum over intervals of (G(s1) - G(s0)).

Ops: every scalar add, multiply, divide, compare and every elementary function call (sin, cos, sqrt, atan2) count 1.
"""
import math

import numpy as np

TWO_PI = 2 * math.pi
FOUR_PI = 4 * math.pi


class Ops:
    def __init__(self):
        self.n = 0
        self.fn = 0  # elementary-function calls (sqrt, sin, cos, atan2), counted in n as 1 each


def _G(t):
    if t == math.inf:
        return 1.0
    if t == -math.inf:
        return -1.0
    return t / math.sqrt(1.0 + t * t)


def line_measure(model, u, ops, h2=None):
    """sum over the intervals of t where the model is right of (G(s1) - G(s0)), at in-plane direction u = (u0, u1)."""
    d, W, w, Wo = model.d, model.W_hh, model.W_hi, model.W_oh
    k = Wo.shape[0]
    if h2 is None:
        h1 = np.maximum(w * u[0], 0.0)
        h2 = np.maximum(W @ h1 + w * u[1], 0.0)
        ops.n += 2 * d + 2 * d * d + 2 * d
    c = W @ h2
    ops.n += 2 * d * d - d
    # breakpoints
    events = []
    on = np.zeros(d, dtype=bool)
    for j in range(d):
        if w[j] == 0.0:
            on[j] = c[j] > 0
            continue
        events.append((-c[j] / w[j], j))
        on[j] = w[j] < 0  # on at t -> -infinity
    ops.n += 2 * d
    a, b = (u[0], 0), (u[1], 1)
    lo, hi = (a, b) if a[0] <= b[0] else (b, a)
    events.append((lo[0], -1))
    events.append((hi[0], -2))
    events.sort()
    m = len(events)
    ops.n += int(m * math.log2(m)) + 1
    P = Wo[:, on] @ c[on]
    Q = Wo[:, on] @ w[on]
    ops.n += 4 * k * int(on.sum())
    lab = lo[1]  # t below both: the 2nd largest is the smaller of the first two
    tot = 0.0
    left = -math.inf
    for idx in range(m + 1):
        right = events[idx][0] if idx < m else math.inf
        if right > left:
            s0, s1 = left, right
            for c2 in range(k):
                if c2 == lab:
                    continue
                dp = P[lab] - P[c2]
                dq = Q[lab] - Q[c2]
                ops.n += 3
                if dq == 0.0:
                    # exact ties (every neuron off): argmax takes the first index
                    if dp < 0 or (dp == 0.0 and lab > c2):
                        s1 = s0
                    continue
                r = -dp / dq
                ops.n += 3
                if dq > 0:
                    s0 = max(s0, r)
                else:
                    s1 = min(s1, r)
            ops.n += 2
            if s1 > s0:
                tot += _G(s1) - _G(s0)
                ops.n += 10
                ops.fn += 2
        if idx == m:
            break
        tval, j = events[idx]
        if j >= 0:
            sgn = -1.0 if on[j] else 1.0
            on[j] = not on[j]
            if on.any():
                P = P + sgn * Wo[:, j] * c[j]
                Q = Q + sgn * Wo[:, j] * w[j]
            else:
                P, Q = np.zeros(k), np.zeros(k)
            ops.n += 4 * k
        elif j == -1:
            lab = 2  # t between the two: t is the 2nd largest
        else:
            lab = hi[1]  # t above both: the 2nd largest is the larger of the first two
        left = tval
    return tot


def accuracy_trap(model, N, ops=None, phase=0.0):
    """Periodic trapezoid rule with N equally spaced angles (offset by phase * 2 pi / N)."""
    ops = ops or Ops()
    tot = 0.0
    for i in range(N):
        a = (i + phase) * TWO_PI / N
        u = (math.cos(a), math.sin(a))
        ops.n += 2
        tot += line_measure(model, u, ops)
    return tot * (TWO_PI / N) / FOUR_PI


def arcs2(model, ops):
    """Arcs of the (x0, x1) circle on which the first two steps keep their activation pattern, with the linear map
    A (d, 2): h_2 = A (cos a, sin a). Cut also at x0 = x1 (the label's ordering of the first two changes there)."""
    d, W, w = model.d, model.W_hh, model.W_hi
    cells = []
    # step 1: h1 = relu(w x0): pattern set by the sign of x0
    for lo, hi, s in ((-math.pi / 2, math.pi / 2, 1.0), (math.pi / 2, 3 * math.pi / 2, -1.0)):
        on1 = (w * s) > 0
        H1 = np.zeros((d, 2))
        H1[on1, 0] = w[on1]
        Z = W @ H1
        Z[:, 1] += w
        ops.n += 2 * d * d + 2 * d
        cuts = []
        for j in range(d):
            base = math.atan2(-Z[j, 0], Z[j, 1])
            ops.n += 6
            for kk in range(-2, 4):
                th = base + kk * math.pi
                if lo + 1e-15 < th < hi - 1e-15:
                    cuts.append(th)
        for r in (math.pi / 4, 5 * math.pi / 4):
            if lo < r < hi:
                cuts.append(r)
        pts = [lo] + sorted(cuts) + [hi]
        ops.n += 3 * len(pts)
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a <= 1e-15:
                continue
            mid = 0.5 * (a + b)
            um = np.array([math.cos(mid), math.sin(mid)])
            on2 = (Z @ um) > 0
            ops.n += 4 * d + 3
            cells.append((a, b, Z * on2[:, None]))
    return cells


def accuracy_gl(model, m, ops=None):
    """Gauss-Legendre with m nodes on each arc of arcs2."""
    ops = ops or Ops()
    g, wts = np.polynomial.legendre.leggauss(m)
    tot = 0.0
    for a, b, A in arcs2(model, ops):
        half = 0.5 * (b - a)
        for gi, wi in zip(g, wts):
            th = a + half * (gi + 1)
            u = (math.cos(th), math.sin(th))
            h2 = A @ np.array(u)
            ops.n += 2 + 4 * model.d + 4
            tot += wi * half * line_measure(model, u, ops, h2=h2)
    return tot / FOUR_PI


def accuracy_halves(model, m, ops=None):
    """Gauss-Legendre with m nodes on each of the two half-circles between the label's jump angles pi/4 and 5 pi/4
    (where x0 = x1 and the label below / above the first two swaps positions)."""
    ops = ops or Ops()
    g, wts = np.polynomial.legendre.leggauss(m)
    tot = 0.0
    for a in (math.pi / 4, 5 * math.pi / 4):
        half = math.pi / 2
        for gi, wi in zip(g, wts):
            th = a + half * (gi + 1)
            u = (math.cos(th), math.sin(th))
            ops.n += 4
            ops.fn += 2
            tot += wi * half * line_measure(model, u, ops)
    return tot / FOUR_PI


def accuracy_one(model, ops=None):
    """One direction only: the line t through (x0, x1) = (1, -1)/sqrt(2), weighted as the whole circle."""
    ops = ops or Ops()
    a = 7 * math.pi / 4
    ops.n += 3
    ops.fn += 2
    return line_measure(model, (math.cos(a), math.sin(a)), ops) * TWO_PI / FOUR_PI


def _W(t):
    """Antiderivatives of (1+t^2)^(-5/2) times 1, t, t^2 (limits at +-infinity included)."""
    if t == math.inf:
        return (2.0 / 3.0, 0.0, 1.0 / 3.0)
    if t == -math.inf:
        return (-2.0 / 3.0, 0.0, -1.0 / 3.0)
    s = (1.0 + t * t) ** 1.5
    return (t * (2 * t * t + 3) / (3 * s), -1.0 / (3 * s), t ** 3 / (3 * s))


def line_sq_error(model, u, ops, h2=None):
    """integral over t of (f - median)^2 (u0, u1, t) * (3/2)(1+t^2)^(-5/2) for the median task (one output): with
    x = R (u0, u1, t), E[(f - median)^2] = (1/2pi) integral over the angle of this (f and the median are homogeneous of
    degree 1; E[R^2 | t] f_t(t) = (3/2)(1+t^2)^(-5/2) for R ~ chi_2). Between the neurons' switch points and the median's
    switch points (t = u0, t = u1) the error is linear in t, and each piece integrates in closed form."""
    d, W, w, Wo = model.d, model.W_hh, model.W_hi, model.W_oh
    if h2 is None:
        h1 = np.maximum(w * u[0], 0.0)
        h2 = np.maximum(W @ h1 + w * u[1], 0.0)
        ops.n += 2 * d + 2 * d * d + 2 * d
    c = W @ h2
    ops.n += 2 * d * d - d
    events = []
    on = np.zeros(d, dtype=bool)
    for j in range(d):
        if w[j] == 0.0:
            on[j] = c[j] > 0
            continue
        events.append((-c[j] / w[j], j))
        on[j] = w[j] < 0
    ops.n += 2 * d
    lo, hi = min(u[0], u[1]), max(u[0], u[1])
    events.append((lo, -1))
    events.append((hi, -2))
    events.sort()
    m = len(events)
    ops.n += int(m * math.log2(m)) + 1
    wo = Wo[0]
    P = float(wo[on] @ c[on])
    Q = float(wo[on] @ w[on])
    ops.n += 4 * int(on.sum())
    am, bm = lo, 0.0  # median = am + bm t: below both, the median is the smaller of the first two
    tot = 0.0
    left = -math.inf
    Wl = _W(left)
    for idx in range(m + 1):
        right = events[idx][0] if idx < m else math.inf
        if right > left:
            Wr = _W(right)
            al, be = P - am, Q - bm
            tot += al * al * (Wr[0] - Wl[0]) + 2 * al * be * (Wr[1] - Wl[1]) + be * be * (Wr[2] - Wl[2])
            ops.n += 22
            ops.fn += 1
            Wl = Wr
        if idx == m:
            break
        tval, j = events[idx]
        if j >= 0:
            sgn = -1.0 if on[j] else 1.0
            on[j] = not on[j]
            P += sgn * wo[j] * c[j]
            Q += sgn * wo[j] * w[j]
            ops.n += 6
        elif j == -1:
            am, bm = 0.0, 1.0  # between the two: the median is t
        else:
            am, bm = hi, 0.0  # above both: the median is the larger of the first two
        left = tval
    return 1.5 * tot


def mse_halves(model, m, ops=None):
    """E[(f - median)^2]: Gauss-Legendre with m nodes on each half-circle between pi/4 and 5pi/4 (x0 = x1, where the
    median's formula changes)."""
    ops = ops or Ops()
    g, wts = np.polynomial.legendre.leggauss(m)
    tot = 0.0
    for a in (math.pi / 4, 5 * math.pi / 4):
        half = math.pi / 2
        for gi, wi in zip(g, wts):
            th = a + half * (gi + 1)
            u = (math.cos(th), math.sin(th))
            ops.n += 4
            ops.fn += 2
            tot += wi * half * line_sq_error(model, u, ops)
    return tot / TWO_PI


def mse_one(model, ops=None):
    ops = ops or Ops()
    a = 7 * math.pi / 4
    ops.n += 3
    ops.fn += 2
    return line_sq_error(model, (math.cos(a), math.sin(a)), ops)


def mse_arcs(model, m, ops=None):
    """E[(f - median)^2]: Gauss-Legendre with m nodes on each arc of arcs2."""
    ops = ops or Ops()
    g, wts = np.polynomial.legendre.leggauss(m)
    tot = 0.0
    for a, b, A in arcs2(model, ops):
        half = 0.5 * (b - a)
        for gi, wi in zip(g, wts):
            th = a + half * (gi + 1)
            u = (math.cos(th), math.sin(th))
            h2 = A @ np.array(u)
            ops.n += 2 + 4 * model.d + 4
            ops.fn += 2
            tot += wi * half * line_sq_error(model, u, ops, h2=h2)
    return tot / TWO_PI
