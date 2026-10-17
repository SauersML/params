"""Exact mechanistic accuracy / MSE for AlgZoo RNNs of any sequence length, with exact integer geometry.

The cone combinatorics (which side of a hyperplane each ray lies on, which rays are adjacent, which faces exist)
run in exact integer arithmetic: float64 weights are dyadic rationals, so every pre-activation form, constraint and
extreme ray is an integer vector after scaling by a power of 2, and every sign test is exact. Only the final Gaussian
measures are computed in floating point, from the exact rays:
  - a pulling triangulation into simplicial cones (exact incidence and exact ranks),
  - orthant probabilities by Plackett's reduction (exactn.orthant),
  - simplicial cones too thin for that quadrature (correlation eigenvalue below 1e-11) are bounded instead by the
    volume of conv(0, V / h) (exactn.thin_bound, within ~2x of high-precision values on the cones checked); these
    bounds add up to `unresolved`, the estimate's certified error bound.
Second moments for the median task: E[x x^T 1_C] = P(C) I + (1/2pi) sum over facets f and ridges g of f of
P_(n-2)(g) a_f b_fg^T (Gaussian integration by parts twice; a_f inward facet normal, b_fg the inward normal of ridge
g inside facet f, P_(n-2)(g) the ridge's Gaussian measure inside its own (n-2)-dimensional span).
"""
import itertools
import math
from fractions import Fraction
from functools import reduce

import numpy as np

import exactn


class Ops:
    """n = arithmetic operations (float ones counted as in exactn, big-integer ones as 1 each); bits = sign facts
    established; unresolved = bound on measure not computed (thin simplicial cones)."""

    def __init__(self):
        self.n = 0
        self.bits = 0
        self.unresolved = 0.0


# ------------------------------------------------------------------ integer vector helpers


def _gcd_vec(v):
    return reduce(math.gcd, (abs(int(a)) for a in v), 0)


def primitive(v):
    g = _gcd_vec(v)
    if g == 0:
        return v
    return np.array([int(a) // g for a in v], dtype=object)


def to_unit(v):
    """Integer vector -> float unit vector without overflow."""
    m = max(abs(int(a)) for a in v)
    sh = max(m.bit_length() - 60, 0)
    f = np.array([float(int(a) >> sh) if a >= 0 else -float((-int(a)) >> sh) for a in v])
    return f / np.linalg.norm(f)


def rank_exact(rows):
    """Rank of a list of integer vectors (fraction-free Gaussian elimination)."""
    M = [list(map(int, r)) for r in rows]
    if not M:
        return 0
    m, n = len(M), len(M[0])
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(r + 1, m):
            if M[i][c]:
                f, g = M[i][c], M[r][c]
                M[i] = [g * a - f * b for a, b in zip(M[i], M[r])]
                gg = reduce(math.gcd, (abs(a) for a in M[i]), 0)
                if gg > 1:
                    M[i] = [a // gg for a in M[i]]
        r += 1
        if r == m:
            break
    return r


def dots(A, R):
    """Exact (len(A), len(R)) matrix of integer dot products (object arrays: Python ints, no overflow)."""
    if len(A) == 0 or len(R) == 0:
        return np.zeros((len(A), len(R)), dtype=object)
    return np.array(A, dtype=object) @ np.array(R, dtype=object).T


# ------------------------------------------------------------------ cones


class Cone:
    __slots__ = ("A", "R")

    def __init__(self, A, R):
        self.A = A  # list of primitive integer constraint vectors (a.x >= 0)
        self.R = R  # list of primitive integer extreme rays


def orthants(n):
    out = []
    for s in itertools.product((1, -1), repeat=n):
        rows = [np.array([s[i] if j == i else 0 for j in range(n)], dtype=object) for i in range(n)]
        out.append(Cone(list(rows), list(rows)))
    return out


def split(cone, a, ops):
    """(part with a.x >= 0, part with a.x <= 0), None for a side that is not full-dimensional. Exact."""
    if all(int(v) == 0 for v in a):
        return cone, None
    a = primitive(a)
    s = list(np.array(cone.R, dtype=object) @ a)
    ops.n += len(s) * len(a)
    pos = [i for i, v in enumerate(s) if v > 0]
    neg = [i for i, v in enumerate(s) if v < 0]
    if not neg:
        return cone, None
    if not pos:
        return None, cone
    zero = [i for i, v in enumerate(s) if v == 0]
    n = len(a)
    act = dots(cone.A, cone.R) == 0  # (m, k) exact incidence
    ops.n += act.size * n
    new = []
    for p in pos:
        for q in neg:
            common = act[:, p] & act[:, q]
            if common.sum() < n - 2:
                continue
            others = act[common].all(axis=0) if common.any() else np.ones(len(cone.R), dtype=bool)
            others[p] = others[q] = False
            if others.any():
                continue
            r = primitive(np.array([s[p] * int(x) - s[q] * int(y) for x, y in zip(cone.R[q], cone.R[p])],
                                   dtype=object))
            new.append(r)
            ops.n += 3 * n
    Rp = [cone.R[i] for i in pos + zero] + new
    Rn = [cone.R[i] for i in neg + zero] + new
    Ap = _prune(cone.A + [a], Rp, n)
    An = _prune(cone.A + [-a], Rn, n)
    return Cone(Ap, Rp), Cone(An, Rn)


def _prune(A, R, n):
    """Drop constraints active on fewer than n-1 rays (they are not facets)."""
    act = dots(A, R) == 0
    return [a for a, row in zip(A, act) if row.sum() >= n - 1]


def triangulate(cone, idx, dim):
    """Pulling triangulation of the face with rays idx and dimension dim (exact incidence and ranks)."""
    if len(idx) == dim:
        return [tuple(idx)]
    p0 = idx[0]
    act = dots(cone.A, [cone.R[i] for i in idx]) == 0
    cand = set()
    for row in act:
        S = tuple(idx[j] for j in np.nonzero(row)[0])
        if p0 in S or len(S) < dim - 1 or len(S) == len(idx):
            continue
        cand.add(S)
    facets = [S for S in cand if rank_exact([cone.R[i] for i in S]) == dim - 1]
    facets = [S for S in facets if not any(set(S) < set(T) for T in facets)]
    out = []
    for S in facets:
        for simp in triangulate(cone, list(S), dim - 1):
            out.append((p0,) + simp)
    return out


def simplicial_measure(V, ops):
    return exactn.simplicial_measure(V, ops)


class Deferred:
    """Collects simplicial cones and evaluates their orthant probabilities in batches (grouped by dimension and by
    quadrature depth), which removes the per-cone Python overhead. add() returns a handle; value(handle) after
    evaluate() gives the measure of everything added under that handle."""

    def __init__(self):
        self.items = []  # (handle, k, R)
        self.handles = {}
        self.closed = {}

    def add_simplex(self, handle, V, ops):
        k = V.shape[0]
        Vi = np.linalg.inv(V)
        R = exactn._corr(Vi.T @ Vi)
        ops.n += 2 * k ** 3
        self.handles.setdefault(handle, 0.0)
        if k <= 3:
            self.handles[handle] += float(exactn._orthant_closed(R[None])[0])
            return
        lam = float(np.linalg.eigvalsh(R)[0])
        if lam < 1e-11:
            ops.unresolved += exactn.thin_bound(V)
            return
        self.items.append((handle, k, R, min(60, int(math.ceil(0.5 * math.log2(1.0 / max(lam, 1e-30)))) + 4), V))

    def evaluate(self, ops, chunk=256):
        groups = {}
        for it in self.items:
            groups.setdefault((it[1], it[3]), []).append(it)
        for (k, lev), its in groups.items():
            for i in range(0, len(its), chunk):
                part = its[i:i + chunk]
                Rb = np.stack([p[2] for p in part])
                vals = exactn.orthant_batch(Rb, ops)
                for p, v in zip(part, vals):
                    if math.isfinite(v) and -1e-9 <= v <= 1 + 1e-9:
                        self.handles[p[0]] += min(max(float(v), 0.0), 1.0)
                    else:
                        ops.unresolved += exactn.thin_bound(p[4])
                        ops.failed = getattr(ops, "failed", 0) + 1
        self.items = []

    def value(self, handle):
        return self.handles.get(handle, 0.0)


def measure_deferred(cone, handle, deferred, ops, basis=None):
    n = len(cone.R[0]) if basis is None else basis.shape[0]
    U = np.array([to_unit(r) for r in cone.R])
    if basis is not None:
        U = U @ basis.T
        U /= np.linalg.norm(U, axis=1, keepdims=True)
    for simp in triangulate(cone, list(range(len(cone.R))), n):
        deferred.add_simplex(handle, U[list(simp)], ops)


def measure(cone, ops, basis=None):
    """Gaussian measure of the cone; with `basis` (orthonormal rows spanning the cone's linear span), the measure
    inside that subspace."""
    n = len(cone.R[0]) if basis is None else basis.shape[0]
    U = np.array([to_unit(r) for r in cone.R])
    if basis is not None:
        U = U @ basis.T
        U /= np.linalg.norm(U, axis=1, keepdims=True)
    tot = 0.0
    for simp in triangulate(cone, list(range(len(cone.R))), n):
        tot += simplicial_measure(U[list(simp)], ops)
    return tot


# ------------------------------------------------------------------ the model in exact integers


def _dyadic(M):
    """Float64 array -> (integer object array I, s) with M = I / 2^s exactly."""
    fr = [Fraction(float(v)) for v in np.ravel(M)]
    s = max((f.denominator.bit_length() - 1 for f in fr), default=0)
    I = np.array([int(f * (1 << s)) for f in fr], dtype=object).reshape(np.shape(M))
    return I, s


def cells(model, ops=None):
    """List of (cone, Lint) where the output on the cone equals Lint @ x up to a positive factor common to all
    outputs (the factor is the same on every cone, so argmax and ratios are exact; for MSE use cells_scaled)."""
    return [(c, L) for c, L, _ in cells_scaled(model, ops)]


def cells_scaled(model, ops=None):
    """List of (cone, Lint, scale) with output = Lint @ x / scale exactly (Lint integer object array)."""
    ops = ops or Ops()
    d, n = model.d, model.n
    Whi, s1 = _dyadic(model.W_hi)
    Whh, s2 = _dyadic(model.W_hh)
    Woh, s3 = _dyadic(model.W_oh)
    s = max(s1, s2)
    Whi = Whi * (1 << (s - s1))
    Whh = Whh * (1 << (s - s2))
    # Z_t = Zi_t / 2^(s t): Zi_(t+1) = Whh (mask Zi_t) + Whi 2^(s t) e_t
    cur = [(c, np.zeros((d, n), dtype=object)) for c in orthants(n)]
    for t in range(n):
        new = []
        for cone, Hi in cur:
            Zi = Whh.dot(Hi)
            Zi[:, t] = Zi[:, t] + Whi * (1 << (s * t))
            ops.n += 2 * d * d * n
            parts = [(cone, np.zeros(d, dtype=bool))]
            for j in range(d):
                nxt = []
                for c, mask in parts:
                    pos, neg = split(c, Zi[j], ops)
                    if pos is not None:
                        m2 = mask.copy()
                        m2[j] = True
                        nxt.append((pos, m2))
                    if neg is not None:
                        nxt.append((neg, mask))
                parts = nxt
            for c, mask in parts:
                H2 = Zi.copy()
                H2[~mask] = 0
                new.append((c, H2))
        cur = new
        ops.bits += len(cur) * d
    scale = Fraction(1 << (s * (n - 1) + s), 1) * (1 << s3)  # Zi_n / 2^(s n), times Woh / 2^s3
    return [(c, Woh.dot(H), scale) for c, H in cur]


def _order_pieces(cone, n, ops):
    parts = [cone]
    for i, j in itertools.combinations(range(n), 2):
        a = np.array([1 if m == i else (-1 if m == j else 0) for m in range(n)], dtype=object)
        nxt = []
        for c in parts:
            pos, neg = split(c, a, ops)
            nxt += [p for p in (pos, neg) if p is not None]
        parts = nxt
    return parts


def _interior(cone):
    return sum(to_unit(r) for r in cone.R)


def confusion(model, ops=None):
    """Exact {ordering (positions, largest first): P(ordering, model predicts c) for each c}; exact ties go to the
    first index, as argmax does."""
    ops = ops or Ops()
    n, k = model.n, model.W_oh.shape[0]
    table = {}
    for cone, L in cells(model, ops):
        for piece in _order_pieces(cone, n, ops):
            order = tuple(int(i) for i in np.argsort(-_interior(piece)))
            row = table.setdefault(order, np.zeros(k))
            ops.bits += k - 1
            for c in range(k):
                r = piece
                for i in range(k):
                    if i == c or r is None:
                        continue
                    a = L[c] - L[i]
                    if all(int(v) == 0 for v in a):
                        r = r if c < i else None
                        continue
                    r, _ = split(r, a, ops)
                if r is not None:
                    row[c] += measure(r, ops)
    return table


def label_of(task, order, n):
    if task == "2nd_argmax":
        return order[1]
    if task == "argmedian":
        return order[n // 2]
    raise ValueError(task)


def accuracy(model, ops=None):
    ops = ops or Ops()
    n = model.n
    total = 0.0
    for cone, L in cells(model, ops):
        for piece in _order_pieces(cone, n, ops):
            order = tuple(int(i) for i in np.argsort(-_interior(piece)))
            lab = label_of(model.task, order, n)
            r = piece
            ops.bits += n - 1
            for i in range(n):
                if i == lab or r is None:
                    continue
                a = L[lab] - L[i]
                if all(int(v) == 0 for v in a):
                    r = r if lab < i else None
                    continue
                r, _ = split(r, a, ops)
            if r is not None:
                total += measure(r, ops)
    return total


# ------------------------------------------------------------------ second moments (median task)


def _orthonormal_complement_basis(rows, n):
    """Orthonormal basis (rows) of the orthogonal complement of span(rows) in R^n (float)."""
    if len(rows) == 0:
        return np.eye(n)
    M = np.array(rows, dtype=float)
    _, sv, Vt = np.linalg.svd(M)
    r = int((sv > 1e-12 * sv[0]).sum())
    return Vt[r:]


def second_moment(cone, P, ops):
    """E[x x^T 1_C] for a full-dimensional pointed cone C in R^n with Gaussian measure P."""
    R = cone.R
    n = len(R[0])
    act = dots(cone.A, R) == 0
    facets = []
    seen = set()
    for ai, row in enumerate(act):
        S = frozenset(np.nonzero(row)[0].tolist())
        if len(S) >= n - 1 and S not in seen and rank_exact([R[i] for i in S]) == n - 1:
            seen.add(S)
            facets.append((ai, S))
    M = P * np.eye(n)
    unitA = [to_unit(cone.A[ai]) for ai, _ in facets]
    for fi, (ai, S) in enumerate(facets):
        a_f = unitA[fi]
        for gi, (bj, T) in enumerate(facets):
            if gi == fi:
                continue
            ridge = S & T
            if len(ridge) < n - 2 or rank_exact([R[i] for i in ridge]) != n - 2:
                continue
            # inward normal of the ridge inside facet f: the other facet's normal projected onto a_f's complement
            b = unitA[gi] - (unitA[gi] @ a_f) * a_f
            nb = np.linalg.norm(b)
            if nb < 1e-15:
                continue
            b /= nb
            rays = [R[i] for i in sorted(ridge)]
            if n - 2 == 0:
                pg = 1.0
            else:
                basis = _orthonormal_complement_basis([to_unit(cone.A[ai]), to_unit(cone.A[bj])], n)
                sub = Cone([], rays)
                # constraints of the ridge inside its span: the other facets' normals (incidence stays exact)
                sub.A = [cone.A[ck] for ck, _ in facets if ck not in (ai, bj)]
                pg = measure(sub, ops, basis=basis) if n - 2 > 1 else 0.5
            M += pg / (2 * math.pi) * np.outer(a_f, b)
            ops.n += 4 * n * n
    return 0.5 * (M + M.T)


def mse(model, ops=None):
    """Exact E[(f(x) - median(x))^2] for the median task with odd n (median = the middle order statistic)."""
    ops = ops or Ops()
    n = model.n
    assert n % 2 == 1, "even n: the median averages two order statistics; not implemented"
    total = 0.0
    for cone, L, scale in cells_scaled(model, ops):
        w = np.array([float(Fraction(int(v), 1) / scale) for v in L[0]])
        for piece in _order_pieces(cone, n, ops):
            order = tuple(int(i) for i in np.argsort(-_interior(piece)))
            v = w.copy()
            v[order[n // 2]] -= 1.0
            P = measure(piece, ops)
            M = second_moment(piece, P, ops)
            total += float(v @ M @ v)
    return total


def _label_cones(task, n, c):
    """Constraint lists (integer vectors a with a.x >= 0) of convex cones that partition {label(x) = c}."""
    def ge(i, j):
        return np.array([1 if m == i else (-1 if m == j else 0) for m in range(n)], dtype=object)

    others = [i for i in range(n) if i != c]
    out = []
    if task == "2nd_argmax":
        for j in others:  # x_j is the largest, x_c the second largest
            out.append([ge(j, c)] + [ge(c, i) for i in others if i != j])
    elif task == "argmedian":
        h = (n - 1) // 2
        for above in itertools.combinations(others, h):  # the h inputs larger than x_c
            out.append([ge(i, c) for i in above] + [ge(c, i) for i in others if i not in above])
    else:
        raise ValueError(task)
    return out


def accuracy_by_class(model, ops=None, marginals=False):
    """Exact accuracy; each cone is split only by the planes that make c the argmax and by the planes of the cones
    where c is the label (n-1 cones for 2nd argmax), which avoids enumerating all n! orderings.
    Returns (accuracy, per-class probability of being right: P(label = c and prediction = c))."""
    ops = ops or Ops()
    n, k = model.n, model.W_oh.shape[0]
    deferred = Deferred()
    for cone, L in cells(model, ops):
        for c in range(k):
            r = cone
            for i in range(k):
                if i == c or r is None:
                    continue
                a = L[c] - L[i]
                if all(int(v) == 0 for v in a):
                    r = r if c < i else None
                    continue
                r, _ = split(r, a, ops)
            if r is None:
                continue
            if marginals:
                measure_deferred(r, ("pred", c), deferred, ops)
            ops.bits += k - 1
            for cons in _label_cones(model.task, n, c):
                q = r
                for a in cons:
                    if q is None:
                        break
                    q, _ = split(q, a, ops)
                if q is not None:
                    measure_deferred(q, c, deferred, ops)
    deferred.evaluate(ops)
    per_class = np.array([deferred.value(c) for c in range(k)])
    if marginals:
        return float(per_class.sum()), per_class, np.array([deferred.value(("pred", c)) for c in range(k)])
    return float(per_class.sum()), per_class


def mse_fast(model, ops=None):
    """mse() with every Gaussian measure batched. E[((w - e_med) . x)^2 ; piece] = P |v|^2
    + sum_(f, g) P_(n-2)(g) (v . a_f)(v . b_fg) / (2 pi), linear in the measures, so each measure is deferred with
    its coefficient and the total is the coefficient-weighted sum."""
    ops = ops or Ops()
    n = model.n
    assert n % 2 == 1, "even n: the median averages two order statistics; not implemented"
    deferred = Deferred()
    coef = {}
    hid = 0
    for cone, L, scale in cells_scaled(model, ops):
        w = np.array([float(Fraction(int(v), 1) / scale) for v in L[0]])
        for piece in _order_pieces(cone, n, ops):
            order = tuple(int(i) for i in np.argsort(-_interior(piece)))
            v = w.copy()
            v[order[n // 2]] -= 1.0
            hid += 1
            coef[hid] = float(v @ v)
            measure_deferred(piece, hid, deferred, ops)
            R = piece.R
            act = dots(piece.A, R) == 0
            facets, seen = [], set()
            for ai, row in enumerate(act):
                S = frozenset(np.nonzero(row)[0].tolist())
                if len(S) >= n - 1 and S not in seen and rank_exact([R[i] for i in S]) == n - 1:
                    seen.add(S)
                    facets.append((ai, S))
            unitA = [to_unit(piece.A[ai]) for ai, _ in facets]
            for fi, (ai, S) in enumerate(facets):
                for gi, (bj, T) in enumerate(facets):
                    if gi == fi:
                        continue
                    ridge = S & T
                    if len(ridge) < n - 2 or rank_exact([R[i] for i in ridge]) != n - 2:
                        continue
                    b = unitA[gi] - (unitA[gi] @ unitA[fi]) * unitA[fi]
                    nb = np.linalg.norm(b)
                    if nb < 1e-15:
                        continue
                    b /= nb
                    c = float(v @ unitA[fi]) * float(v @ b) / (2 * math.pi)
                    hid += 1
                    coef[hid] = c
                    if n - 2 <= 1:
                        deferred.handles[hid] = 0.5 if n - 2 == 1 else 1.0
                        continue
                    basis = _orthonormal_complement_basis([unitA[fi], unitA[gi]], n)
                    sub = Cone([piece.A[ck] for ck, _ in facets if ck not in (ai, bj)], [R[i] for i in sorted(ridge)])
                    measure_deferred(sub, hid, deferred, ops, basis=basis)
                    ops.n += 4 * n * n
    deferred.evaluate(ops)
    return float(sum(c * deferred.value(h) for h, c in coef.items()))
