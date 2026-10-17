"""Exact mechanistic accuracy / MSE for AlgZoo RNNs with sequence length 3.

The output is a positively homogeneous piecewise-linear function of x in R^3, and the Gaussian measure of a cone is
its share of the unit sphere. The sphere starts as the 8 coordinate octants (spherical triangles). Each step clips
every convex spherical polygon by the great circle of each neuron's pre-activation, so on every final polygon the
logits are a linear map of x. Polygons are then clipped by the ordering planes x_i = x_j (the label is fixed on each)
and by the planes where the label's logit ties another logit.

Areas: Van Oosterom-Strackee solid angle of a fan of triangles. Second moments for the median task:
E[x x^T 1_C] = P(C) I + (1/2pi) sum_f sin(alpha_f / 2) a_f b_f^T over the edges f of the polygon, where a_f is the
inward unit normal of the edge's plane, alpha_f the edge's arc length and b_f the edge's unit bisector (Gaussian
integration by parts twice). Ops are counted to compare compute with sampling.
"""
import itertools
import math

import numpy as np

FOUR_PI = 4 * math.pi
EPS = 1e-13


class Ops:
    """n = arithmetic operations; bits = sign facts the enumeration establishes (one per neuron per cell per step,
    and one per logit comparison on each final piece)."""

    def __init__(self):
        self.n = 0
        self.bits = 0


def _unit(v):
    return v / np.linalg.norm(v)


def octants():
    out = []
    for s in itertools.product((1.0, -1.0), repeat=3):
        e = [np.eye(3)[i] * s[i] for i in range(3)]
        tri = [e[0], e[1], e[2]]
        if np.dot(np.cross(tri[0], tri[1]), tri[2]) < 0:
            tri = [e[0], e[2], e[1]]
        out.append(tri)
    return out


def clip(poly, a, ops):
    """Split convex spherical polygon (counterclockwise unit vertices) by plane a.x = 0.
    Returns (positive part or None, negative part or None)."""
    s = np.array([float(np.dot(a, v)) for v in poly])
    ops.n += 6 * len(poly)
    scale = np.linalg.norm(a)
    if np.all(s >= -EPS * scale):
        return poly, None
    if np.all(s <= EPS * scale):
        return None, poly
    pos, neg = [], []
    k = len(poly)
    for i in range(k):
        v, w = poly[i], poly[(i + 1) % k]
        si, sj = s[i], s[(i + 1) % k]
        if si >= 0:
            pos.append(v)
        if si <= 0:
            neg.append(v)
        if (si > 0 and sj < 0) or (si < 0 and sj > 0):
            p = _unit(si * w - sj * v) if si > 0 else _unit(sj * v - si * w)
            pos.append(p)
            neg.append(p)
            ops.n += 12
    return (pos if len(pos) >= 3 else None), (neg if len(neg) >= 3 else None)


def area(poly, ops):
    a = poly[0]
    tot = 0.0
    for b, c in zip(poly[1:-1], poly[2:]):
        det = float(np.dot(a, np.cross(b, c)))
        den = 1.0 + float(np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
        tot += 2.0 * math.atan2(det, den)
        ops.n += 30
    return tot


def second_moment(poly, P, ops):
    """E[x x^T 1_C] for the cone C over the polygon, P = Gaussian measure of C."""
    M = P * np.eye(3)
    cen = _unit(sum(poly))
    k = len(poly)
    for i in range(k):
        v, w = poly[i], poly[(i + 1) % k]
        n = np.cross(v, w)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        a = n / nn
        if np.dot(a, cen) < 0:
            a = -a
        alpha = math.atan2(nn, float(np.dot(v, w)))
        b = _unit(v + w)
        M += math.sin(alpha / 2) / (2 * math.pi) * np.outer(a, b)
        ops.n += 40
    return 0.5 * (M + M.T)


def cells(model, ops=None):
    """List of (polygon, L) with L (n_out, 3) the linear output map on the polygon's cone."""
    ops = ops or Ops()
    d = model.d
    cur = [(p, np.zeros((d, 3))) for p in octants()]
    for t in range(3):
        e = np.zeros(3)
        e[t] = 1.0
        new = []
        for poly, H in cur:
            Z = model.W_hh @ H + np.outer(model.W_hi, e)
            ops.n += d * d * 3 * 2 + 3 * d
            parts = [(poly, np.zeros(d, dtype=bool))]
            for j in range(d):
                nxt = []
                for q, mask in parts:
                    pos, neg = clip(q, Z[j], ops)
                    if pos is not None:
                        m2 = mask.copy()
                        m2[j] = True
                        nxt.append((pos, m2))
                    if neg is not None:
                        nxt.append((neg, mask))
                parts = nxt
            for q, mask in parts:
                new.append((q, Z * mask[:, None]))
        cur = new
        ops.bits += len(cur) * d
    return [(q, model.W_oh @ H) for q, H in cur]


ORDER_PLANES = [np.array([1.0, -1.0, 0.0]), np.array([1.0, 0.0, -1.0]), np.array([0.0, 1.0, -1.0])]


def _by_order(poly, ops):
    parts = [poly]
    for a in ORDER_PLANES:
        nxt = []
        for q in parts:
            pos, neg = clip(q, a, ops)
            nxt += [r for r in (pos, neg) if r is not None]
        parts = nxt
    return parts


def accuracy(model, ops=None):
    """Exact P(argmax logits = label) for Gaussian x in R^3; label = 2nd largest position (= argmedian for n=3)."""
    ops = ops or Ops()
    total = 0.0
    for poly, L in cells(model, ops):
        for q in _by_order(poly, ops):
            c = _unit(sum(q))
            lab = int(np.argsort(c)[1])
            r = q
            ops.bits += 2
            for i in range(3):
                if i == lab or r is None:
                    continue
                a = L[lab] - L[i]
                if np.abs(a).max() <= 1e-12 * (np.abs(L).max() + 1e-300):
                    # exact tie on the whole cone (e.g. every neuron off): argmax takes the first index
                    r = r if lab < i else None
                    continue
                r, _ = clip(r, a, ops)
            if r is not None:
                total += area(r, ops)
    return total / FOUR_PI


def mse(model, ops=None):
    """Exact E[(f(x) - median(x))^2] for Gaussian x in R^3."""
    ops = ops or Ops()
    total = 0.0
    for poly, L in cells(model, ops):
        for q in _by_order(poly, ops):
            c = _unit(sum(q))
            k = int(np.argsort(c)[1])
            v = L[0].copy()
            v[k] -= 1.0
            P = area(q, ops) / FOUR_PI
            M = second_moment(q, P, ops)
            total += float(v @ M @ v)
            ops.n += 20
    return total


def mse_by_order(model, ops=None):
    """{ordering (positions, largest first): (probability, E[(f - median)^2 ; ordering], mean output coefficients)}."""
    ops = ops or Ops()
    out = {}
    for poly, L in cells(model, ops):
        for q in _by_order(poly, ops):
            c = _unit(sum(q))
            order = tuple(int(i) for i in np.argsort(-c))
            k = order[1]
            v = L[0].copy()
            v[k] -= 1.0
            P = area(q, ops) / FOUR_PI
            M = second_moment(q, P, ops)
            p0, e0, w0 = out.get(order, (0.0, 0.0, np.zeros(3)))
            out[order] = (p0 + P, e0 + float(v @ M @ v), w0 + P * L[0])
    return {o: (p, e, w / max(p, 1e-300)) for o, (p, e, w) in out.items()}
