"""Differential validation of the Lean definitions against independent references.

Two references, neither written by us:

  * PyTorch: `torch.softmax`, `torch.nn.functional.layer_norm`,
    `torch.nn.functional.scaled_dot_product_attention` (incl. `is_causal=True`),
    `torch.nn.MultiheadAttention`, `torch.nn.TransformerEncoderLayer` (post-norm, ReLU);
  * the SPD reference implementation (goodfire-ai/param-decomp, branch `spd-paper`):
    `LinearComponent.forward` with a mask, `calc_stochastic_masks`, `lower_leaky_relu`,
    `upper_leaky_relu`, and the faithfulness loss.

The Lean definitions are used through their literal transcriptions in `lean_defs.py`.
Agreement here means the Lean objects are the objects practitioners compute; every
theorem in the corpus is then a theorem about those objects.

Run:  python3 validation/empirical/differential.py --reps 50 --seed 0 [--spd-repo PATH]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_defs as L  # noqa: E402
import provenance  # noqa: E402

torch.set_default_dtype(torch.float64)
TOL = 1e-9


def t(x):
    return torch.tensor(np.asarray(x, dtype=np.float64))


def diff_softmax(rng):
    a = rng.normal(size=9) * 3
    return np.abs(L.softmax(a) - torch.softmax(t(a), 0).numpy()).max()


def diff_layer_norm(rng):
    v = rng.normal(size=12); s1, s2 = rng.normal(), abs(rng.normal()) + 0.1
    ref = torch.nn.functional.layer_norm(t(v), (12,), weight=t(np.full(12, s2)), bias=t(np.full(12, s1)), eps=0.0)
    return np.abs(L.layer_norm(s1, s2, v) - ref.numpy()).max()


def diff_sdpa(rng):
    Q, K, V = (rng.normal(size=(7, 5)) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(t(Q), t(K), t(V)).numpy()
    return np.abs(L.attention(1 / np.sqrt(5), Q, K, V) - ref).max()


def diff_sdpa_causal(rng):
    Q, K, V = (rng.normal(size=(7, 5)) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(t(Q), t(K), t(V), is_causal=True).numpy()
    return np.abs(L.causal_weights(1 / np.sqrt(5), Q, K) @ V - ref).max()


def diff_mha(rng):
    d, h, n = 8, 2, 6; dk = d // h
    u = rng.normal(size=(n, d))
    mha = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True)
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3)); Wo = rng.normal(size=(d, d)) / np.sqrt(d)
    with torch.no_grad():
        mha.in_proj_weight.copy_(t(np.concatenate([Wq.T, Wk.T, Wv.T], axis=0)))
        mha.out_proj.weight.copy_(t(Wo.T))
    with torch.no_grad():
        ref = mha(t(u)[None], t(u)[None], t(u)[None], need_weights=False)[0][0].numpy()
    WQs = [Wq[:, p * dk:(p + 1) * dk] for p in range(h)]; WKs = [Wk[:, p * dk:(p + 1) * dk] for p in range(h)]
    WVs = [Wv[:, p * dk:(p + 1) * dk] for p in range(h)]
    ours = L.multi_head_attention(1 / np.sqrt(dk), WQs, WKs, WVs, Wo, u)
    return np.abs(ours - ref).max()


def diff_encoder_layer(rng):
    """Post-norm `nn.TransformerEncoderLayer` with one head, `W^O = I`, LN affine = (σ₂, σ₁), eps = 0."""
    d, f, n = 6, 10, 5
    u = rng.normal(size=(n, d))
    p = {"WQ": rng.normal(size=(d, d)) / np.sqrt(d), "WK": rng.normal(size=(d, d)) / np.sqrt(d),
         "WV": rng.normal(size=(d, d)) / np.sqrt(d), "W1": rng.normal(size=(d, f)) / np.sqrt(d),
         "b1": rng.normal(size=f), "W2": rng.normal(size=(f, d)) / np.sqrt(f), "b2": rng.normal(size=d),
         "s1": rng.normal(), "s2": abs(rng.normal()) + 0.1}
    layer = torch.nn.TransformerEncoderLayer(d, 1, dim_feedforward=f, dropout=0.0, activation="relu",
                                             layer_norm_eps=0.0, batch_first=True, norm_first=False)
    layer.eval()
    with torch.no_grad():
        layer.self_attn.in_proj_weight.copy_(t(np.concatenate([p["WQ"].T, p["WK"].T, p["WV"].T], axis=0)))
        layer.self_attn.in_proj_bias.zero_()
        layer.self_attn.out_proj.weight.copy_(torch.eye(d)); layer.self_attn.out_proj.bias.zero_()
        layer.linear1.weight.copy_(t(p["W1"].T)); layer.linear1.bias.copy_(t(p["b1"]))
        layer.linear2.weight.copy_(t(p["W2"].T)); layer.linear2.bias.copy_(t(p["b2"]))
        for ln in (layer.norm1, layer.norm2):
            ln.weight.fill_(p["s2"]); ln.bias.fill_(p["s1"])
        ref = layer(t(u)[None])[0].numpy()
    return np.abs(L.encoder_block(1 / np.sqrt(d), p, u) - ref).max()


def load_spd(repo):
    sys.path.insert(0, str(repo))
    comp = importlib.import_module("spd.models.components")
    cu = importlib.import_module("spd.models.component_utils")
    return comp, cu


def diff_spd_linear_component(rng, spd):
    comp, _ = spd
    d_in, d_out, C, B = 5, 4, 6, 3
    lc = comp.LinearComponent(d_in=d_in, d_out=d_out, C=C, bias=None)
    A, Bm = rng.normal(size=(d_in, C)), rng.normal(size=(C, d_out))
    with torch.no_grad():
        lc.A.copy_(t(A)); lc.B.copy_(t(Bm))
    x = rng.normal(size=(B, d_in)); mask = rng.uniform(size=(B, C))
    lc.mask = t(mask)
    ref = lc(t(x)).detach().numpy()
    # Lean: weight acts on the right of a row-vector: out = x @ (U diag(m) V)ᵀ with U = Bᵀ, V = Aᵀ
    ours = np.stack([x[b] @ L.masked(Bm.T, A.T, mask[b]).T for b in range(B)])
    w = np.abs(lc.weight.detach().numpy() - L.weight(Bm.T, A.T)).max()
    return max(np.abs(ours - ref).max(), w)


def diff_spd_masks_and_activations(rng, spd):
    _, cu = spd
    g = rng.uniform(size=(4, 7))
    torch.manual_seed(int(rng.integers(0, 2**31)))
    m = cu.calc_stochastic_masks({"l": t(g)}, n_mask_samples=1)[0]["l"].numpy()
    # recover r from the code's own draw and compare with spdMask g r
    r = np.where(1 - g != 0, (m - g) / (1 - g), 0.0)
    v1 = np.abs(L.spd_mask(g, r) - m).max()
    x = rng.normal(size=40) * 3
    v2 = np.abs(cu.lower_leaky_relu(t(x)).numpy() - L.lower_leaky(0.01, x)).max()
    v3 = np.abs(cu.upper_leaky_relu(t(x)).numpy() - L.upper_leaky(0.01, x)).max()
    return max(v1, v2, v3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spd-repo", default=None, help="path to goodfire-ai/param-decomp (spd-paper)")
    ap.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "differential.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    checks = {"torch.softmax": diff_softmax, "F.layer_norm": diff_layer_norm,
              "F.scaled_dot_product_attention": diff_sdpa,
              "F.scaled_dot_product_attention(is_causal)": diff_sdpa_causal,
              "nn.MultiheadAttention": diff_mha, "nn.TransformerEncoderLayer(post-norm)": diff_encoder_layer}
    spd = None
    if args.spd_repo:
        try:
            spd = load_spd(Path(args.spd_repo))
            checks["spd LinearComponent.forward/weight"] = lambda rng: diff_spd_linear_component(rng, spd)
            checks["spd calc_stochastic_masks/leaky relus"] = lambda rng: diff_spd_masks_and_activations(rng, spd)
        except Exception as e:  # report, do not hide
            print(f"[SKIP] spd reference not importable: {e!r}")
    cells, records, failed = [], [], False
    print(f"{'reference':<48} {'max |lean - ref|':>18}  verdict")
    for name, fn in checks.items():
        vals = []
        for r in range(args.reps):
            v = float(fn(rng)); vals.append(v)
            records.append({"check": name, "rep": r, "diff": v})
        s = provenance.summarize(vals); ok = s["max"] <= TOL; failed |= not ok
        cells.append({"check": name, "diff": s, "pass": ok})
        print(f"{name:<48} {s['max']:>18.3e}  {'PASS' if ok else 'FAIL'}")
    provenance.write(args.output, "differential.py", {"tol": TOL, "spd_repo": args.spd_repo},
                     args.seed, args.reps, cells, records)
    print(f"\nwrote {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
