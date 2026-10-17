"""Exact mechanistic accuracy / MSE for AlgZoo RNNs with sequence length 2.

The RNNs have ReLU and no biases, so the output is a positively homogeneous piecewise-linear function of x in R^2.
Gaussian inputs are rotation invariant, so the metric is an average over the angle of x on the unit circle.
The circle is cut into arcs on which every pre-activation keeps its sign; on each arc the logits are a linear map of x,
so the correct sub-arcs and the squared error integrate in closed form. Every cut is a root of a linear form.
Ops are counted (multiply, add, compare, atan2 each count 1) to compare compute with sampling.
"""
import math

import numpy as np

TWO_PI = 2 * math.pi


class Ops:
    """n = arithmetic operations; bits = sign facts the enumeration establishes (one per neuron per cell per step,
    and one per logit comparison on each final piece)."""

    def __init__(self):
        self.n = 0
        self.bits = 0


def _roots(c, lo, hi):
    """Angles theta in (lo, hi) where c[0] cos(theta) + c[1] sin(theta) = 0."""
    base = math.atan2(-c[0], c[1])
    out = []
    for k in range(-2, 4):
        th = base + k * math.pi
        if lo + 1e-15 < th < hi - 1e-15:
            out.append(th)
    return out


def arcs(model, ops=None):
    """List of (lo, hi, L) with L (n_out, 2): on angles in [lo, hi] the output equals L @ (cos, sin)."""
    ops = ops or Ops()
    d = model.d
    cells = [(0.0, TWO_PI, np.zeros((d, 2)))]
    for t in range(2):
        e = np.zeros(2)
        e[t] = 1.0
        new = []
        for lo, hi, H in cells:
            Z = model.W_hh @ H + np.outer(model.W_hi, e)
            ops.n += d * d * 2 * 2 + 2 * d
            cuts = sorted(r for j in range(d) for r in _roots(Z[j], lo, hi))
            ops.n += 3 * d
            pts = [lo] + cuts + [hi]
            for a, b in zip(pts[:-1], pts[1:]):
                if b - a <= 0:
                    continue
                mid = 0.5 * (a + b)
                u = np.array([math.cos(mid), math.sin(mid)])
                on = (Z @ u) > 0
                ops.n += 3 * d
                new.append((a, b, Z * on[:, None]))
        cells = new
        ops.bits += len(cells) * d
    return [(lo, hi, model.W_oh @ H) for lo, hi, H in cells]


def _cut_by_label(task, lo, hi):
    """Sub-arcs on which the task label is a fixed linear functional of x (or a fixed index)."""
    # orderings of (x0, x1) change at x0 = x1, i.e. theta = pi/4 and 5pi/4
    pts = [lo] + [r for r in (math.pi / 4, 5 * math.pi / 4) if lo < r < hi] + [hi]
    return list(zip(pts[:-1], pts[1:]))


def accuracy(model, ops=None):
    """Exact probability (Gaussian x) that argmax logits equals the task label (2nd argmax / argmedian, n = 2)."""
    ops = ops or Ops()
    total = 0.0
    for lo, hi, L in arcs(model, ops):
        for a, b in _cut_by_label(model.task, lo, hi):
            mid = 0.5 * (a + b)
            u = np.array([math.cos(mid), math.sin(mid)])
            lab = int(np.argsort(u)[-2])  # 2nd largest of two = the smaller
            other = 1 - lab
            c = L[lab] - L[other]  # correct where c . u > 0
            if np.abs(c).max() <= 1e-12 * (np.abs(L).max() + 1e-300):
                # exact tie on the whole arc (e.g. every neuron off): argmax takes the first index
                total += (b - a) if lab < other else 0.0
                continue
            cuts = _roots(c, a, b)
            ops.n += 12
            ops.bits += 1
            pts = [a] + cuts + [b]
            for p, q in zip(pts[:-1], pts[1:]):
                m = 0.5 * (p + q)
                if c[0] * math.cos(m) + c[1] * math.sin(m) > 0:
                    total += q - p
                ops.n += 6
    return total / TWO_PI


def _int_quad(v, a, b):
    """Integral over theta in [a, b] of (v0 cos + v1 sin)^2."""
    A, B = v[0] ** 2, v[1] ** 2
    C = v[0] * v[1]
    # cos^2 = (1 + cos 2t)/2, sin^2 = (1 - cos 2t)/2, cos sin = sin 2t / 2
    I_c2 = 0.5 * (b - a) + 0.25 * (math.sin(2 * b) - math.sin(2 * a))
    I_s2 = 0.5 * (b - a) - 0.25 * (math.sin(2 * b) - math.sin(2 * a))
    I_cs = -0.25 * (math.cos(2 * b) - math.cos(2 * a))
    return A * I_c2 + B * I_s2 + 2 * C * I_cs


def mse(model, ops=None):
    """Exact E[(f(x) - median(x))^2] for Gaussian x in R^2; median of two = (x0 + x1)/2, linear everywhere.
    E[r^2] = 2 for the radius, so E = 2 * mean over angle of (v . u)^2."""
    ops = ops or Ops()
    total = 0.0
    for lo, hi, L in arcs(model, ops):
        v = L[0] - 0.5
        total += _int_quad(v, lo, hi)
        ops.n += 20
    return 2.0 * total / TWO_PI
