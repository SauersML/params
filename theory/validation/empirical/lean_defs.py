"""Literal NumPy transcriptions of the `Params` Lean definitions.

Each function names the Lean declaration it transcribes.  These are the
*definitions* only; the theorems about them are what `battery.py` tests
numerically and what `differential.py` compares against independent
reference implementations (PyTorch, the SPD reference code).

Conventions follow the Lean files: token matrices are `(n_tokens, d)`,
weights act on the right (`u @ W`), vectors are 1-D arrays.
"""
import numpy as np


# ---- Params/Transformer/Softmax.lean -------------------------------------------------
def softmax(a):
    """`Params.softmax a i = exp (a i) / Σ_j exp (a j)`."""
    e = np.exp(a)
    return e / e.sum()


def softmax_rows(A):
    """`Params.softmaxRows A = Matrix.of fun i => softmax (A i)`."""
    return np.stack([softmax(row) for row in A])


# ---- Params/Transformer/LayerNorm.lean -----------------------------------------------
def mean(v):
    """`Params.mean v = (Σ_i v i) / card`."""
    return v.sum() / v.shape[0]


def variance(v):
    """`Params.variance v = (Σ_i (v i - mean v)^2) / card`."""
    return ((v - mean(v)) ** 2).sum() / v.shape[0]


def layer_norm(s1, s2, v):
    """`Params.layerNorm σ₁ σ₂ v i = σ₂ * (v i - mean v) / sqrt (variance v) + σ₁`
    (Mathlib: `x / 0 = 0`, reproduced with `np.divide(..., where=...)`)."""
    r = np.sqrt(variance(v))
    num = s2 * (v - mean(v))
    return (np.divide(num, r, out=np.zeros_like(num), where=r != 0)) + s1


def mean_sq(v):
    """`Params.meanSq v = mean (fun j => v j ^ 2)`."""
    return mean(v ** 2)


def rms_norm(v):
    """`Params.rmsNorm v i = v i / sqrt (meanSq v)`."""
    r = np.sqrt(mean_sq(v))
    return np.divide(v, r, out=np.zeros_like(v), where=r != 0)


def sq_dist(u, v):
    """`Params.sqDist u v = Σ_i (u i - v i)^2`."""
    return ((u - v) ** 2).sum()


def relu(x):
    """`Params.relu x = max x 0`."""
    return np.maximum(x, 0.0)


# ---- Params/Transformer/Attention.lean -----------------------------------------------
def attn_weights(s, Q, K):
    """`Params.attnWeights s Q K = softmaxRows (s • (Q * Kᵀ))`."""
    return softmax_rows(s * (Q @ K.T))


def attention(s, Q, K, V):
    """`Params.attention s Q K V = attnWeights s Q K * V`."""
    return attn_weights(s, Q, K) @ V


def self_attention(s, WQ, WK, WV, u):
    """`Params.selfAttention s θ u = attention s (u * θ.WQ) (u * θ.WK) (u * θ.WV)`."""
    return attention(s, u @ WQ, u @ WK, u @ WV)


def multi_head_attention(s, WQs, WKs, WVs, WO, u):
    """`Params.multiHeadAttention`: concatenate `headOut` over heads (index `h × dk`,
    lexicographic: head-major), then multiply by `W^O : (h*dk) × d`."""
    heads = [attention(s, u @ WQs[p], u @ WKs[p], u @ WVs[p]) for p in range(len(WQs))]
    return np.concatenate(heads, axis=1) @ WO


# ---- Params/Transformer/Encoder.lean -------------------------------------------------
def ln_rows(s1, s2, u):
    """`Params.lnRows σ₁ σ₂ u = Matrix.of fun i => layerNorm σ₁ σ₂ (u i)`."""
    return np.stack([layer_norm(s1, s2, row) for row in u])


def ffn(W1, b1, W2, b2, u):
    """`Params.ffn φ u = reluM (u * W₁ + broadcast b₁) * W₂ + broadcast b₂`."""
    return relu(u @ W1 + b1[None, :]) @ W2 + b2[None, :]


def encoder_block(s, p, u):
    """`Params.encoderBlock s θ u` (post-norm, Vaswani et al.)."""
    u1 = u + self_attention(s, p["WQ"], p["WK"], p["WV"], u)
    u2 = ln_rows(p["s1"], p["s2"], u1)
    u3 = u2 + ffn(p["W1"], p["b1"], p["W2"], p["b2"], u2)
    return ln_rows(p["s1"], p["s2"], u3)


def split_step(s, p, u):
    """`Params.splitStep s θ u`: substeps (3.5)-(3.10) of Tai-Liu-Li-Chan with J = 2."""
    u1 = u + self_attention(s, p["WQ"], p["WK"], p["WV"], u)                      # substep1
    u2 = ln_rows(p["s1"], p["s2"], u1)                                             # substep2
    u3 = relu(u2 + u2 @ p["W1"] + p["b1"][None, :])                                # substep3
    u4 = u3 + u3 @ p["W2"] + p["b2"][None, :]                                      # substep4
    u5 = 0.5 * (u4 + u2)                                                           # substep5
    return ln_rows(p["s1"], p["s2"], u5)                                           # substep6


def split_to_encoder(p):
    """`Params.SplitParams.toEncoder`: `W_j ↦ I + W_j`."""
    d = p["W1"].shape[0]
    q = dict(p)
    q["W1"] = np.eye(d) + p["W1"]
    q["W2"] = np.eye(d) + p["W2"]
    return q


# ---- Params/Transformer/Causal.lean --------------------------------------------------
def masked_softmax(mask, a):
    """`Params.maskedSoftmax mask a i = if mask i then exp (a i) / Σ_{mask j} exp (a j) else 0`."""
    e = np.exp(a)
    den = e[mask].sum()
    return np.where(mask, e / den, 0.0)


def causal_weights(s, Q, K):
    """`Params.causalWeights s Q K i = maskedSoftmax (fun j => j ≤ i) ((s • Q Kᵀ) i)`."""
    S = s * (Q @ K.T)
    N = S.shape[0]
    return np.stack([masked_softmax(np.arange(N) <= i, S[i]) for i in range(N)])


def causal_self_attention(s, WQ, WK, WV, u):
    """`Params.causalSelfAttention s θ u`."""
    return causal_weights(s, u @ WQ, u @ WK) @ (u @ WV)


# ---- Params/Transformer/LanguageModel.lean -------------------------------------------
def decoder_block(s, p, u):
    """`Params.decoderBlock` (post-norm, causal)."""
    u1 = u + causal_self_attention(s, p["WQ"], p["WK"], p["WV"], u)
    u2 = ln_rows(p["s1"], p["s2"], u1)
    u3 = u2 + ffn(p["W1"], p["b1"], p["W2"], p["b2"], u2)
    return ln_rows(p["s1"], p["s2"], u3)


def logits(s, lm, tokens):
    """`Params.logits s θ t = decoderStack s θ.blocks (embedTokens θ t) * θ.unembed`."""
    u = lm["embed"][tokens] + lm["pos"][: len(tokens)]
    for p in lm["blocks"]:
        u = decoder_block(s, p, u)
    return u @ lm["unembed"]


def next_token_dist(s, lm, tokens, i):
    """`Params.nextTokenDist s θ t i = softmax (logits s θ t i)`."""
    return softmax(logits(s, lm, tokens)[i])


def pre_norm_residual_and_contributions(s, blocks, u0):
    """`Params.residual` and `Params.contributions` for pre-norm blocks."""
    u = u0
    contribs = []
    for p in blocks:
        a = causal_self_attention(s, p["WQ"], p["WK"], p["WV"], ln_rows(p["s1"], p["s2"], u))
        u = u + a
        m = ffn(p["W1"], p["b1"], p["W2"], p["b2"], ln_rows(p["s1"], p["s2"], u))
        u = u + m
        contribs += [a, m]
    return u, contribs


# ---- Params/Decomp/Basic.lean --------------------------------------------------------
def outer(u, v):
    """`Params.Decomp.outer u v i j = u i * v j`."""
    return np.outer(u, v)


def component(U, V, c):
    """`Subcomponents.component D c = outer (U[:, c]) (V[c, :])`."""
    return outer(U[:, c], V[c, :])


def weight(U, V):
    """`Subcomponents.weight D = U * V`."""
    return U @ V


def masked(U, V, mask):
    """`Subcomponents.masked D mask = U * diagonal mask * V`."""
    return U @ np.diag(mask) @ V


def faithfulness_loss(W, U, V):
    """`faithfulnessLoss W D = Σ_{i,j} (W i j - (U V) i j)^2`."""
    return ((W - U @ V) ** 2).sum()


def spd_mask(g, r):
    """`spdMask g r c = g c + (1 - g c) * r c`."""
    return g + (1.0 - g) * r


def hard_sigmoid(x):
    """`hardSigmoid x = max 0 (min 1 x)`."""
    return np.maximum(0.0, np.minimum(1.0, x))


def lower_leaky(alpha, x):
    """`lowerLeaky α x = if 0 < x then min 1 x else α * x`."""
    return np.where(x > 0, np.minimum(1.0, x), alpha * x)


def upper_leaky(alpha, x):
    """`upperLeaky α x = if 1 < x then 1 + α * (x - 1) else max 0 x`."""
    return np.where(x > 1, 1.0 + alpha * (x - 1.0), np.maximum(0.0, x))


def importance_loss(p, g):
    """`importanceLoss p g = Σ_c |g c|^p`."""
    return (np.abs(g) ** p).sum()


# ---- Params/Decomp/Identifiability.lean ----------------------------------------------
def col_comp(W, j):
    """`colComp W j i j' = if j' = j then W i j else 0`."""
    Z = np.zeros_like(W)
    Z[:, j] = W[:, j]
    return Z


# ---- Params/Interp/Superposition.lean ------------------------------------------------
def superpose(v, s):
    """`superpose v s k = Σ_j s j * v j k`  (v: (N, d))."""
    return s @ v


def readout(v, i, x):
    """`readout v i x = ip (v i) x`."""
    return v[i] @ x


# ---- Params/Interp/Absorption.lean ---------------------------------------------------
SPLIT_COST = np.array([1.0, 1.0, 2.0])
ABSORBED_COST = np.array([1.0, 1.0, 1.0])


def expected_cost(p, cost):
    """`expectedCost p cost = Σ_x p x * cost x`."""
    return (p * cost).sum()
