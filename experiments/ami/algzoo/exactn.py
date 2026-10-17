"""Exact mechanistic accuracy for AlgZoo RNNs of any sequence length n (practical up to n = 6).

The output is a positively homogeneous piecewise-linear function of x in R^n. Cells are convex polyhedral cones kept
in both forms, constraints a.x >= 0 and extreme rays, and are split by each neuron's pre-activation with the
double-description update (Fukuda's combinatorial adjacency test). The input space starts as the 2^n coordinate
orthants. On each final cell the logits are linear; the cell is split by the ordering hyperplanes x_i = x_j, which
fixes the label, and then by the planes where the label's logit ties another logit.

Gaussian measure of a cone: a pulling triangulation into simplicial cones, and for each simplicial cone with ray
matrix V the orthant probability P(y >= 0) for y ~ N(0, (V^T V)^-1). Orthant probabilities are closed form up to
dimension 3 and use Plackett's reduction above that:
    Phi_k(R) = 2^-k + sum_{i<j} R_ij int_0^1 phi_2(0, 0; t R_ij) Phi_{k-2}(R_t | y_i = y_j = 0) dt,
with R_t = (1 - t) I + t R, integrated by Gauss-Legendre after t = 1 - u^2 (which removes the square-root
singularity at t = 1). Ops count arithmetic for the compute comparison with sampling.
"""
import itertools
import math

import numpy as np

TOL = 1e-11  # a ray with |a.r| below this lies on the plane a.x = 0
TOL_ACT = 1e-9  # incidence of rays and constraints in the adjacency test and the triangulation
TOL_PRUNE = 1e-7  # constraints are dropped only when clearly active on fewer than n-1 rays


class Ops:
    """n = arithmetic operations; bits = sign facts established; unresolved = bound on the measure of cones too thin
    for the quadrature (the estimate's error bound)."""

    def __init__(self):
        self.n = 0
        self.bits = 0
        self.unresolved = 0.0


class Cone:
    __slots__ = ("A", "R")

    def __init__(self, A, R):
        self.A = A  # (m, n) constraints, rows unit
        self.R = R  # (k, n) extreme rays, rows unit


def _unit_rows(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def orthants(n):
    out = []
    for s in itertools.product((1.0, -1.0), repeat=n):
        D = np.diag(s)
        out.append(Cone(D.copy(), D.copy()))
    return out


def split(cone, a, ops):
    """(part with a.x >= 0, part with a.x <= 0); a part is None when it is not full-dimensional."""
    na = np.linalg.norm(a)
    if na == 0:
        return cone, None
    a = a / na
    s = cone.R @ a
    ops.n += 2 * s.size
    pos, neg = s > TOL, s < -TOL
    if not neg.any():
        return cone, None
    if not pos.any():
        return None, cone
    zero = ~(pos | neg)
    act = np.abs(cone.A @ cone.R.T) <= TOL_ACT  # (m, k) incidence
    ops.n += 2 * act.size
    new = []
    P, N = np.nonzero(pos)[0], np.nonzero(neg)[0]
    n = cone.R.shape[1]
    for p in P:
        for q in N:
            common = act[:, p] & act[:, q]
            if common.sum() < n - 2:
                continue
            # adjacent iff no other ray is active on every common constraint
            others = act[common].all(axis=0)
            others[p] = others[q] = False
            if others.any():
                continue
            r = s[p] * cone.R[q] - s[q] * cone.R[p]
            nr = np.linalg.norm(r)
            if nr > 1e-14:
                new.append(r / nr)
            ops.n += 4 * n
    new = np.array(new).reshape(-1, n)
    Rp = _dedupe(np.vstack([cone.R[pos | zero], new]))
    Rn = _dedupe(np.vstack([cone.R[neg | zero], new]))
    Ap = _prune(np.vstack([cone.A, a]), Rp)
    An = _prune(np.vstack([cone.A, -a]), Rn)
    # a side whose rays do not span R^n is a numerical sliver of zero measure: the cone goes to the other side
    if _rank(Rp) < n:
        return None, cone
    if _rank(Rn) < n:
        return cone, None
    return Cone(Ap, Rp), Cone(An, Rn)


def _dedupe(R):
    """Merge rays that coincide up to rounding (two parents can produce the same new ray)."""
    keep = []
    for i in range(R.shape[0]):
        if all(float(R[i] @ R[j]) < 1 - 1e-15 for j in keep):
            keep.append(i)
    return R[keep]


def _prune(A, R):
    """Drop constraints active on fewer than n-1 rays (they cannot be facets)."""
    n = R.shape[1]
    act = np.abs(A @ R.T) <= TOL_PRUNE
    keep = act.sum(1) >= n - 1
    return A[keep]


def _rank(M):
    if M.shape[0] == 0:
        return 0
    return int(np.linalg.matrix_rank(M, tol=1e-9))


def triangulate(R, A, idx, dim):
    """Pulling triangulation of the face spanned by rays idx (of dimension dim). Returns list of index tuples."""
    if len(idx) == dim:
        return [tuple(idx)]
    p0 = idx[0]
    act = np.abs(A @ R[idx].T) <= TOL_ACT  # constraints x rays of this face
    facets = set()
    for c in range(A.shape[0]):
        S = tuple(int(idx[j]) for j in np.nonzero(act[c])[0])
        if p0 in S or len(S) < dim - 1 or len(S) == len(idx):
            continue
        facets.add(S)
    # keep maximal sets with the right rank
    facets = [S for S in facets if _rank(R[list(S)]) == dim - 1]
    facets = [S for S in facets if not any(set(S) < set(T) for T in facets)]
    out = []
    for S in facets:
        for simp in triangulate(R, A, list(S), dim - 1):
            out.append((p0,) + simp)
    return out


# ---------------------------------------------------------------- orthant probabilities

def _composite_nodes(levels, per=8):
    """Gauss-Legendre nodes on [0, 1] refined geometrically toward u = 0: the dyadic intervals [2^-(j+1), 2^-j]
    (and [0, 2^-levels]) each get `per` nodes. Near u = 0 (t = 1 - u^2 near 1) thin cones make the integrand change
    on scales as small as sqrt(1 - |rho|)."""
    g, w = np.polynomial.legendre.leggauss(per)
    us, ws = [], []
    edges = [0.0] + [2.0 ** -j for j in range(levels, -1, -1)]
    for a, b in zip(edges[:-1], edges[1:]):
        us.append(a + (b - a) * 0.5 * (g + 1))
        ws.append((b - a) * 0.5 * w)
    return np.concatenate(us), np.concatenate(ws)


_NODES = {}


def _nodes(levels):
    if levels not in _NODES:
        u, w = _composite_nodes(levels)
        _NODES[levels] = (u, w, 1.0 - u * u)
    return _NODES[levels]


def _corr(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    C = np.clip(S / d[:, None] / d[None, :], -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C


def _asin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))


def _orthant_closed(Rb):
    """Closed forms for a batch (B, k, k) of correlation matrices, k <= 3."""
    k = Rb.shape[1]
    if k == 0:
        return np.ones(Rb.shape[0])
    if k == 1:
        return np.full(Rb.shape[0], 0.5)
    if k == 2:
        return 0.25 + _asin(Rb[:, 0, 1]) / (2 * math.pi)
    return 0.125 + (_asin(Rb[:, 0, 1]) + _asin(Rb[:, 0, 2]) + _asin(Rb[:, 1, 2])) / (4 * math.pi)


class _Count:
    n = 0


def orthant_batch(Rb, ops=None):
    """P(y >= 0) for y ~ N(0, R) for each correlation matrix R in the batch (B, k, k). Closed form for k <= 3;
    above, Plackett's reduction to k-2 with the inner orthant probabilities batched over quadrature nodes."""
    B, k = Rb.shape[0], Rb.shape[1]
    if k <= 3:
        return _orthant_closed(Rb)
    I = np.eye(k)
    # the integrand varies on u-scales down to sqrt(smallest eigenvalue of R); refine that far plus 4 levels
    lam = max(float(np.linalg.eigvalsh(Rb)[:, 0].min()), 1e-30)
    U, Wq, _ = _nodes(min(60, int(math.ceil(0.5 * math.log2(1.0 / lam))) + 4))
    Q = len(U)
    tau = U * U  # 1 - t, kept separately: R_t = R + tau (I - R) has no cancellation near t = 1
    Rt = Rb[:, None] + tau[None, :, None, None] * (I - Rb)[:, None]  # (B, Q, k, k)
    tot = np.full(B, 2.0 ** (-k))
    for i, j in itertools.combinations(range(k), 2):
        r = Rb[:, i, j]  # (B,)
        rest = [m for m in range(k) if m not in (i, j)]
        Bm = Rt[:, :, [i, j]][:, :, :, [i, j]]  # (B, Q, 2, 2)
        C = Rt[:, :, rest][:, :, :, [i, j]]  # (B, Q, k-2, 2)
        det = (1 - r * r)[:, None] + (tau * (2 - tau))[None] * (r * r)[:, None]  # 1 - (t r)^2, (B, Q)
        Binv = np.stack([np.stack([Bm[..., 1, 1], -Bm[..., 0, 1]], -1),
                         np.stack([-Bm[..., 1, 0], Bm[..., 0, 0]], -1)], -2) / np.clip(det, 1e-300, None)[..., None, None]
        cond = Rt[:, :, rest][:, :, :, rest] - C @ Binv @ np.swapaxes(C, -1, -2)
        dg = np.sqrt(np.clip(np.einsum("bqii->bqi", cond), 1e-300, None))
        cond = np.clip(cond / dg[..., :, None] / dg[..., None, :], -1.0, 1.0)
        flat = cond.reshape(B * Q, k - 2, k - 2)
        if k - 2 <= 3:
            inner = orthant_batch(flat, ops).reshape(B, Q)
        else:  # bound the inner batch's memory (it expands by its own node count)
            inner = np.concatenate([orthant_batch(flat[i:i + 256], ops) for i in range(0, B * Q, 256)]).reshape(B, Q)
        dens = 1.0 / (2 * math.pi * np.sqrt(np.clip(det, 1e-300, None)))
        f = Wq[None] * 2 * U[None] * dens * inner
        # nodes so close to t = 1 that the 2x2 block is numerically singular carry negligible weight
        f = np.where(np.isfinite(f) & (det > 1e-300), f, 0.0)
        tot += r * f.sum(1)
        if ops is not None:
            ops.n += B * Q * (6 * k * k + 40)
    return tot


def orthant(R, ops):
    """P(y >= 0) for y ~ N(0, R), R a correlation matrix."""
    return float(orthant_batch(R[None], ops)[0])


def thin_bound(V):
    """Upper bound on the Gaussian measure of the simplicial cone over unit rays V (rows). Every point of the cone
    inside the unit ball is lambda * p with p in conv(V) and lambda <= 1/|p| <= 1/h, where h = min_i g.r_i for the
    unit mean direction g lower-bounds the distance from 0 to conv(V); so the cone's share of the ball is at most
    vol(conv(0, V / h)) / vol(ball) = |det V| / (n! h^n vol(ball)). 1e-15 is added to |det V| for rounding."""
    n = V.shape[0]
    g = V.sum(0)
    g /= np.linalg.norm(g)
    h = float((V @ g).min())
    if h <= 0:
        return 1.0
    ball = math.pi ** (n / 2) / math.gamma(n / 2 + 1)
    return min(1.0, (abs(np.linalg.det(V)) + 1e-15) / (math.factorial(n) * h ** n * ball))


def simplicial_measure(V, ops):
    """Gaussian measure of the cone generated by the rows of V (k x k unit rows). The coordinates c of x = V^T c
    have covariance V^-T V^-1, whose correlation matrix R gives the orthant probability. Cones so thin that R's
    smallest eigenvalue is below 1e-11 (where float64 quadrature loses absolute accuracy ~1e-9) count as 0 and add
    thin_bound(V) to ops.unresolved, the estimate's certified error bound."""
    k = V.shape[0]
    Vi = np.linalg.inv(V)
    R = _corr(Vi.T @ Vi)
    ops.n += 2 * k ** 3
    if k <= 3:
        return orthant(R, ops)
    if float(np.linalg.eigvalsh(R)[0]) < 1e-11:
        ops.unresolved += thin_bound(V)
        return 0.0
    p = orthant(R, ops)
    if not (math.isfinite(p) and -1e-9 <= p <= 1 + 1e-9):
        ops.unresolved += thin_bound(V)
        return 0.0
    return min(max(p, 0.0), 1.0)


def measure(cone, ops):
    n = cone.R.shape[1]
    tot = 0.0
    for simp in triangulate(cone.R, cone.A, list(range(cone.R.shape[0])), n):
        tot += simplicial_measure(cone.R[list(simp)], ops)
    return tot


# ---------------------------------------------------------------- the model


def cells(model, ops=None):
    ops = ops or Ops()
    d, n = model.d, model.n
    cur = [(c, np.zeros((d, n))) for c in orthants(n)]
    for t in range(n):
        e = np.zeros(n)
        e[t] = 1.0
        new = []
        for cone, H in cur:
            Z = model.W_hh @ H + np.outer(model.W_hi, e)
            ops.n += 2 * d * d * n
            parts = [(cone, np.zeros(d, dtype=bool))]
            for j in range(d):
                nxt = []
                for c, mask in parts:
                    pos, neg = split(c, Z[j], ops)
                    if pos is not None:
                        m2 = mask.copy()
                        m2[j] = True
                        nxt.append((pos, m2))
                    if neg is not None:
                        nxt.append((neg, mask))
                parts = nxt
            new += [(c, Z * mask[:, None]) for c, mask in parts]
        cur = new
        ops.bits += len(cur) * d
    return [(c, model.W_oh @ H) for c, H in cur]


def _order_pieces(cone, n, ops):
    parts = [cone]
    for i, j in itertools.combinations(range(n), 2):
        a = np.zeros(n)
        a[i], a[j] = 1.0, -1.0
        nxt = []
        for c in parts:
            pos, neg = split(c, a, ops)
            nxt += [p for p in (pos, neg) if p is not None]
        parts = nxt
    return parts


def label_index(task, u):
    order = np.argsort(u)
    if task == "2nd_argmax":
        return int(order[-2])
    if task == "argmedian":
        return int(order[len(u) // 2])
    raise ValueError(task)


def accuracy(model, ops=None):
    ops = ops or Ops()
    n = model.n
    total = 0.0
    for cone, L in cells(model, ops):
        scale = np.abs(L).max() + 1e-300
        for piece in _order_pieces(cone, n, ops):
            u = piece.R.sum(0)
            lab = label_index(model.task, u)
            r = piece
            ops.bits += n - 1
            for i in range(n):
                if i == lab or r is None:
                    continue
                a = L[lab] - L[i]
                if np.abs(a).max() <= 1e-12 * scale:
                    r = r if lab < i else None
                    continue
                r, _ = split(r, a, ops)
            if r is not None:
                total += measure(r, ops)
    return total


def confusion(model, ops=None):
    """Exact table {ordering (tuple of positions, largest first): P(ordering, model predicts c) for each c}.
    Ties in the logits on a whole piece go to the first index, as argmax does."""
    ops = ops or Ops()
    n, k = model.n, model.W_oh.shape[0]
    table = {}
    for cone, L in cells(model, ops):
        scale = np.abs(L).max() + 1e-300
        for piece in _order_pieces(cone, n, ops):
            u = piece.R.sum(0)
            order = tuple(int(i) for i in np.argsort(-u))
            row = table.setdefault(order, np.zeros(k))
            for c in range(k):
                r = piece
                for i in range(k):
                    if i == c or r is None:
                        continue
                    a = L[c] - L[i]
                    if np.abs(a).max() <= 1e-12 * scale:
                        r = r if c < i else None
                        continue
                    r, _ = split(r, a, ops)
                if r is not None:
                    row[c] += measure(r, ops)
    return table
