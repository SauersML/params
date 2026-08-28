"""Numerical battery: every theorem in the corpus, checked on random inputs.

Each entry names the Lean theorem, draws random instances of its hypotheses,
and measures the worst violation of its conclusion over `reps` replicates.  A
theorem proved in Lean cannot be falsified here; what this battery checks is
that the *transcriptions* in `lean_defs.py` (and hence the definitions the
theorems are about) mean what the Lean text says, and that the theorem
statements say something with non-trivial content on generic inputs (the
`witness` column: how far a *wrong* candidate is from satisfying the
conclusion, so that a check that passes vacuously is visible as such).

Run:  python3 validation/empirical/battery.py --reps 200 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_defs as L  # noqa: E402
import provenance  # noqa: E402

TOL = 1e-9


def rand_params(rng, d, f=None):
    f = f or d
    return {
        "WQ": rng.normal(size=(d, d)) / np.sqrt(d), "WK": rng.normal(size=(d, d)) / np.sqrt(d),
        "WV": rng.normal(size=(d, d)) / np.sqrt(d),
        "W1": rng.normal(size=(d, f)) / np.sqrt(d), "b1": rng.normal(size=f),
        "W2": rng.normal(size=(f, d)) / np.sqrt(f), "b2": rng.normal(size=d),
        "s1": rng.normal(), "s2": abs(rng.normal()) + 0.1,
    }


# Each check returns (violation, witness).  violation must be <= TOL.
def chk_sum_softmax(rng):
    a = rng.normal(size=7) * 3
    return abs(L.softmax(a).sum() - 1), abs(np.exp(a).sum() - 1)


def chk_softmax_add_const(rng):
    a = rng.normal(size=7); c = rng.normal() * 5
    return np.abs(L.softmax(a + c) - L.softmax(a)).max(), np.abs(L.softmax(a * 2) - L.softmax(a)).max()


def chk_softmax_comp_perm(rng):
    a = rng.normal(size=7); s = rng.permutation(7)
    return np.abs(L.softmax(a[s]) - L.softmax(a)[s]).max(), np.abs(L.softmax(a[s]) - L.softmax(a)).max()


def chk_layerNorm_mean_variance(rng):
    v = rng.normal(size=9); s1, s2 = rng.normal(), abs(rng.normal()) + 0.1
    u = L.layer_norm(s1, s2, v)
    return max(abs(L.mean(u) - s1), abs(L.variance(u) - s2 ** 2)), abs(L.variance(v) - s2 ** 2)


def chk_layerNorm_sqDist_le(rng):
    """`layerNorm_sqDist_le`: LN(v) is closest to v among vectors with mean σ₁, var σ₂²."""
    v = rng.normal(size=9); s1, s2 = rng.normal(), abs(rng.normal()) + 0.1
    star = L.layer_norm(s1, s2, v)
    worst = 0.0
    for _ in range(20):
        u = L.layer_norm(s1, s2, rng.normal(size=9))       # a random member of MeanVarSet
        worst = max(worst, L.sq_dist(star, v) - L.sq_dist(u, v))
    return max(worst, 0.0), L.sq_dist(u, v) - L.sq_dist(star, v)


def chk_layerNorm_smul(rng):
    v = rng.normal(size=9); c = abs(rng.normal()) + 0.1; s1, s2 = rng.normal(), abs(rng.normal())
    return np.abs(L.layer_norm(s1, s2, c * v) - L.layer_norm(s1, s2, v)).max(), \
        np.abs(L.layer_norm(s1, s2, v + c) - L.layer_norm(s1, s2, v)).max()


def chk_relu_isProj(rng):
    v = rng.normal(size=9); u = np.abs(rng.normal(size=9))
    return max(L.sq_dist(L.relu(v), v) - L.sq_dist(u, v), 0.0), L.sq_dist(u, v) - L.sq_dist(L.relu(v), v)


def chk_attention_bounds(rng):
    Q, K, V = (rng.normal(size=(5, 4)) for _ in range(3))
    out = L.attention(0.7, Q, K, V)
    viol = max((out - V.max(axis=0)).max(), (V.min(axis=0) - out).max(), 0.0)
    return viol, (V.max(axis=0) - out).max()


def chk_attention_perm_equivariant(rng):
    Q, K, V = (rng.normal(size=(6, 4)) for _ in range(3)); s = rng.permutation(6)
    return np.abs(L.attention(0.5, Q[s], K[s], V[s]) - L.attention(0.5, Q, K, V)[s]).max(), \
        np.abs(L.attention(0.5, Q[s], K[s], V[s]) - L.attention(0.5, Q, K, V)).max()


def chk_multiHead_eq_sum_heads(rng):
    d, h, dk, n = 6, 3, 2, 5
    u = rng.normal(size=(n, d))
    WQs = [rng.normal(size=(d, dk)) for _ in range(h)]; WKs = [rng.normal(size=(d, dk)) for _ in range(h)]
    WVs = [rng.normal(size=(d, dk)) for _ in range(h)]; WO = rng.normal(size=(h * dk, d))
    mha = L.multi_head_attention(0.5, WQs, WKs, WVs, WO, u)
    summed = sum(L.attention(0.5, u @ WQs[p], u @ WKs[p], u @ WVs[p]) @ WO[p * dk:(p + 1) * dk] for p in range(h))
    return np.abs(mha - summed).max(), np.abs(mha - summed[::-1]).max()


def chk_splitStep_eq_encoderBlock(rng):
    """The paper's exact-recovery claim, numerically."""
    d, n = 6, 5
    p = rand_params(rng, d); u = rng.normal(size=(n, d))
    a = L.split_step(0.5, p, u); b = L.encoder_block(0.5, L.split_to_encoder(p), u)
    return np.abs(a - b).max(), np.abs(a - L.encoder_block(0.5, p, u)).max()


def chk_causal_prefix(rng):
    """`nextTokenDist_prefix`: changing tokens after position i leaves the distribution at i unchanged."""
    V, d, N = 11, 6, 7
    lm = {"embed": rng.normal(size=(V, d)), "pos": rng.normal(size=(N, d)),
          "blocks": [rand_params(rng, d) for _ in range(2)], "unembed": rng.normal(size=(d, V))}
    t = rng.integers(0, V, size=N); t2 = t.copy(); i = int(rng.integers(0, N - 1))
    t2[i + 1:] = rng.integers(0, V, size=N - i - 1)
    p1, p2 = L.next_token_dist(0.5, lm, t, i), L.next_token_dist(0.5, lm, t2, i)
    t3 = t.copy(); t3[0] = (t3[0] + 1) % V
    return np.abs(p1 - p2).max(), np.abs(p1 - L.next_token_dist(0.5, lm, t3, i)).max()


def chk_dla_identity(rng):
    d, N, V = 6, 5, 9
    blocks = [rand_params(rng, d) for _ in range(2)]; u0 = rng.normal(size=(N, d)); WU = rng.normal(size=(d, V))
    u, contribs = L.pre_norm_residual_and_contributions(0.5, blocks, u0)
    lhs = u @ WU; rhs = u0 @ WU + sum(c @ WU for c in contribs)
    return np.abs(lhs - rhs).max(), np.abs(lhs - u0 @ WU).max()


def chk_masked_one_and_update(rng):
    m, n, C = 4, 5, 6
    U, V = rng.normal(size=(m, C)), rng.normal(size=(C, n)); mask = rng.uniform(size=C); c = 2
    v1 = np.abs(L.masked(U, V, np.ones(C)) - L.weight(U, V)).max()
    m0 = mask.copy(); m0[c] = 0
    v2 = np.abs(L.masked(U, V, m0) - (L.masked(U, V, mask) - mask[c] * L.component(U, V, c))).max()
    v3 = np.abs(L.masked(U, V, mask) - sum(mask[k] * L.component(U, V, k) for k in range(C))).max()
    return max(v1, v2, v3), np.abs(L.masked(U, V, mask) - L.weight(U, V)).max()


def chk_masked_mulVec(rng):
    m, n, C = 4, 5, 6
    U, V = rng.normal(size=(m, C)), rng.normal(size=(C, n)); mask = rng.uniform(size=C); x = rng.normal(size=n)
    lhs = L.masked(U, V, mask) @ x
    rhs = sum(mask[c] * ((V[c] @ x) * U[:, c]) for c in range(C))
    return np.abs(lhs - rhs).max(), np.abs(lhs - U @ V @ x).max()


def chk_spdMask_mem_Icc(rng):
    g, r = rng.uniform(size=8), rng.uniform(size=8); mk = L.spd_mask(g, r)
    return max((g - mk).max(), (mk - 1).max(), 0.0), (mk - g).max()


def chk_leaky_activations(rng):
    x = rng.normal(size=50) * 3
    v1 = max((L.lower_leaky(0.01, x) - 1).max(), 0.0)
    v2 = max((-L.upper_leaky(0.01, x)).max(), 0.0)
    y = rng.uniform(0.01, 0.99, size=20)
    v3 = max(np.abs(L.lower_leaky(0.01, y) - y).max(), np.abs(L.upper_leaky(0.01, y) - y).max(),
             np.abs(L.hard_sigmoid(y) - y).max())
    return max(v1, v2, v3), np.abs(L.hard_sigmoid(x) - x).max()


def chk_colComp(rng):
    m, n = 3, 5; W = rng.normal(size=(m, n))
    v1 = np.abs(sum(L.col_comp(W, j) for j in range(n)) - W).max()
    ranks = [np.linalg.matrix_rank(L.col_comp(W, j)) for j in range(n)]
    v2 = max(max(ranks) - 1, 0)
    return max(v1, v2), np.linalg.matrix_rank(W) - 1


def chk_rank_subadditive(rng):
    m, n, C = 3, 6, 4
    u, v = rng.normal(size=(C, m)), rng.normal(size=(C, n))
    W = sum(L.outer(u[c], v[c]) for c in range(C))
    return max(np.linalg.matrix_rank(W) - C, 0), np.linalg.matrix_rank(W)


def chk_identifiability(rng):
    """`onehot_identifiability`: any decomposition satisfying the hypotheses is the column one.
    Constructive check: perturbing the column decomposition by a rank-one, faithfulness-preserving
    exchange between two subcomponents always breaks one-hot separation (the `witness` counts how
    many features become multiply-acted-on)."""
    m, n = 2, 4
    W = rng.normal(size=(m, n)); P = [L.col_comp(W, j) for j in range(n)]
    acts = lambda Ps: [[int(np.any(Pc[:, j] != 0)) for Pc in Ps] for j in range(n)]
    sep_ok = all(sum(row) == 1 for row in acts(P))
    Q = [p.copy() for p in P]; z = rng.normal(size=(m, 1)) @ rng.normal(size=(1, n)) * 0.3
    Q[0] = Q[0] + z; Q[1] = Q[1] - z
    faithful = np.abs(sum(Q) - W).max() < 1e-12
    broken = sum(1 for row in acts(Q) if sum(row) != 1)
    return (0.0 if sep_ok and faithful else 1.0), broken


def chk_readout_error_le(rng):
    d, N, k = 40, 120, 5
    v = rng.normal(size=(N, d)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    G = v @ v.T; eps = np.abs(G - np.eye(N)).max()
    s = np.zeros(N); S = rng.choice(N, size=k, replace=False); s[S] = rng.uniform(0.5, 1.0, size=k)
    i = S[0]; err = abs(L.readout(v, i, L.superpose(v, s)) - s[i])
    bound = eps * (k - 1) * np.abs(s).max()
    return max(err - bound, 0.0), bound - err


def chk_absorption(rng):
    p = rng.uniform(size=3); p /= p.sum()
    diff = L.expected_cost(p, L.SPLIT_COST) - L.expected_cost(p, L.ABSORBED_COST)
    return abs(diff - p[2]), diff


CHECKS = {
    "sum_softmax": chk_sum_softmax, "softmax_add_const": chk_softmax_add_const,
    "softmax_comp_perm": chk_softmax_comp_perm,
    "layerNorm_mean/layerNorm_variance": chk_layerNorm_mean_variance,
    "layerNorm_sqDist_le": chk_layerNorm_sqDist_le, "layerNorm_smul": chk_layerNorm_smul,
    "relu_isProj": chk_relu_isProj, "attention_le/le_attention": chk_attention_bounds,
    "attention_perm_equivariant": chk_attention_perm_equivariant,
    "multiHeadAttention_eq_sum_heads": chk_multiHead_eq_sum_heads,
    "splitStep_eq_encoderBlock": chk_splitStep_eq_encoderBlock,
    "nextTokenDist_prefix": chk_causal_prefix, "logitsPre_eq_sum_contributions": chk_dla_identity,
    "masked_one/masked_update_zero/masked_eq_sum": chk_masked_one_and_update,
    "masked_mulVec": chk_masked_mulVec, "spdMask_mem_Icc": chk_spdMask_mem_Icc,
    "lowerLeaky/upperLeaky/hardSigmoid": chk_leaky_activations,
    "colComp_faithful/colComp_outer": chk_colComp, "rank_sum_le": chk_rank_subadditive,
    "onehot_identifiability (perturbation)": chk_identifiability,
    "readout_error_le": chk_readout_error_le, "expectedCost_split_sub_absorbed": chk_absorption,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "battery.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cells, records, failed = [], [], False
    print(f"{'theorem':<48} {'max violation':>14} {'mean witness':>13}  verdict")
    for name, fn in CHECKS.items():
        viols, wits = [], []
        for r in range(args.reps):
            v, w = fn(rng)
            viols.append(float(v)); wits.append(float(w))
            records.append({"check": name, "rep": r, "violation": float(v), "witness": float(w)})
        sv, sw = provenance.summarize(viols), provenance.summarize(wits)
        ok = sv["max"] <= TOL
        failed |= not ok
        cells.append({"check": name, "violation": sv, "witness": sw, "pass": ok})
        print(f"{name:<48} {sv['max']:>14.3e} {sw['mean']:>13.3e}  {'PASS' if ok else 'FAIL'}")
    provenance.write(args.output, "battery.py", {"tol": TOL}, args.seed, args.reps, cells, records)
    print(f"\nwrote {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
