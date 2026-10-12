"""The J++ lens for vpd4l's subcomponent writes, against the logit lens, on the atlas's measured removal effects (#2951).

Points. The residual stream after each layer's attention (mid_i) and after its MLP (out_i); out_3 is the final residual
before the final norm. A subcomponent's write r enters at one point: o_proj (r = u) and v_proj (r = W_O u, the value
passed through attention with weight 1) at mid_i; down_proj (r = u) and c_fc (r = W_down u, the GELU taken as the
identity) at out_i. q_proj and k_proj move attention scores, write nothing into the stream, and get no readout from
any lens here.

Lenses. A lens maps a point's vector to the final residual by a matrix J_p; the readout of a write r is
w = W_U (g_f * (J_p r)) (g_f the final norm's gain, W_U the tied embedding), and its prediction of the subcomponent's
contribution at a position where its activity is a is a w. The logit lens is J_p = I.
The J++ lens (Ayonrinde and Lindsey 2026, "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses"):
  J_{p,i} = E[ (1 / (T - s)) sum_{t >= s} d_LRP h_final,t / d h_{p,s} | h_{p,s} in cluster i ] (the J-lens average of
            Gurnee et al. 2026 over source positions s and later target positions t, per cluster), over FIT sequences
            of split A, sources from position 16 on;
  LRP       the backward pass with the LN-rule on every RMSNorm (its denominator held constant) and the identity rule
            on the GELU (gelu(x)/x held constant), the plain gradient through attention and the linear maps;
  clusters  E = 8 per point, k-means on 64-dimensional PCA projections of the unit-normalised, centred activations
            of CLUSTER sequences of split A (other rows than the fit's);
  weights   J_p = sum_i w_{p,i} J_{p,i}, w fitted by Adam (learning rate 0.05) to the mean Pearson correlation of the
            readout prediction with the measured contribution on a dev half of the atlas's effect samples (split B
            rows 5000..5499); the paper fits them on a labelled dev set of extraction tasks, here the measured effects
            are the labels. Each sample's correlation is a closed form of w from per-expert covariances, so the fit
            holds no vocabulary-sized arrays;
  readout   tokens with at least one letter or digit (the paper's non-semantic filter), among those split A holds at
            least atlas.MIN_COUNT times.
Also reported: the pooled LRP Jacobian (all clusters weighted by their sources: the R-lens), the routed lens (each
sample read through its own cluster's expert, its residual at the point assigned by the router: the paper's k-means
routing, which needs no labels) and, after `routed`, the atlas's routed readout (each subcomponent through the
mixture of the experts its top contexts are routed to: one map per subcomponent, as an atlas entry needs). Every lens is read over the same tokens twice: the readout tokens above, and every
token split A holds at least atlas.MIN_COUNT times (no token filter), so the filter's gain and the Jacobian's separate.
The J++ weights are supervised (fitted to our measured effects on the dev half); the logit, pooled and routed lenses
use no labels. c_fc and down_proj of the last layer write at out_3, the final residual, where every lens is the logit
lens (J = I): their rows are identical by construction.

Benchmark. The atlas's effect samples on the evaluation half (split B rows 5500..5999), recomputed exactly
(atlas.row_logprobs: the write removed at that row only): per sample the Pearson correlation over the readout tokens
between a lens's prediction and the measured contribution log p - log p_without, and the overlap of their top 10
raised tokens; per site the medians and means. Where J++ beats the logit lens on a site, its readout (the 10 tokens
it raises and lowers, oriented as the atlas's lens) goes into atlas.safetensors beside the logit lens.

  MPD_MEM_GIB=20 python -m method.vpd.lens fit [--tokens ROWS.npy] [--fit 0:64] [--cluster 64:264]
  MPD_MEM_GIB=20 python -m method.vpd.lens routed [--tokens ROWS.npy]      the atlas's routed readout (routed())
  MPD_MEM_GIB=20 python -m method.vpd.lens evaluate [--tokens ROWS.npy]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from method.vpd import vpd_model as VM
from method.vpd.atlas import (DEFAULT, EFFECT, MIN_COUNT, clean_cache, row_logprobs, write_directions)
from method.vpd.parts import tokens_of

E, PCA_DIMS, SKIP, T = 8, 64, 16, 512
POINTS = [f"{kind}_{i}" for i in range(4) for kind in ("mid", "out")]  # mid_0, out_0, ..., out_3


def point_of(layer: int, kind: str) -> str | None:
    if kind in ("o_proj", "v_proj"):
        return f"mid_{layer}"
    if kind in ("down_proj", "c_fc"):
        return f"out_{layer}"
    return None


def forward_points(target: VM.Target, ids: torch.Tensor, lrp: bool, grad: bool = False) -> tuple[torch.Tensor, dict]:
    """The final residual (before the final norm) [B, T, d] and the residual at every point, with explicit attention
    (no fused kernel: it is differentiated with batched cotangents). lrp: the LN-rule and the GELU identity rule."""
    B, Tn = ids.shape
    H, hd, eps = target.n_head, target.hd, target.eps
    causal = torch.ones(Tn, Tn, dtype=torch.bool, device=ids.device).tril()

    def norm(x, g):
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * (r.detach() if lrp else r) * g

    def gelu(f):
        if not lrp:
            return VM.gelu_tanh(f)
        ratio = torch.where(f.abs() > 1e-12, VM.gelu_tanh(f) / f, torch.full_like(f, 0.5))
        return f * ratio.detach()

    x = target.wte[ids].clone().requires_grad_(True) if grad else target.wte[ids]
    points = {}
    for i in range(target.n_layer):
        W = lambda k: target.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}").W
        h = norm(x, target.norms[2 * i])
        q = target._rope((h @ W("q_proj").T).view(B, Tn, H, hd).transpose(1, 2), Tn)
        k = target._rope((h @ W("k_proj").T).view(B, Tn, H, hd).transpose(1, 2), Tn)
        v = (h @ W("v_proj").T).view(B, Tn, H, hd).transpose(1, 2)
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, Tn, H * hd)
        x = x + y @ W("o_proj").T
        points[f"mid_{i}"] = x
        g = gelu(norm(x, target.norms[2 * i + 1]) @ W("c_fc").T)
        x = x + g @ W("down_proj").T
        points[f"out_{i}"] = x
    return x, points


def kmeans(X: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """Centres [k, D] of Lloyd's k-means from k-means++ starts."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randint(len(X), (1,), generator=g)]
    for _ in range(1, k):
        d = torch.cdist(X, C).min(1).values.pow(2)
        C = torch.cat([C, X[torch.multinomial((d / d.sum()).cpu(), 1, generator=g).to(X.device)]])
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        C = torch.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C


class Router:
    """Per point: the centring mean, the PCA basis and the k-means centres that assign a source activation to an expert."""

    def __init__(self, mean, basis, centres):
        self.mean, self.basis, self.centres = mean, basis, centres

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.normalize(h - self.mean, dim=-1) @ self.basis
        return torch.cdist(z, self.centres).argmin(-1)


def fit_routers(target, tokens: torch.Tensor, dev: str, batch: int) -> dict:
    acts = {p: [] for p in POINTS}
    with torch.no_grad():
        for b0 in range(0, tokens.shape[0], batch):
            _, pts = forward_points(target, tokens[b0:b0 + batch].to(dev), lrp=False)
            for p in POINTS:
                acts[p].append(pts[p][:, SKIP:].reshape(-1, pts[p].shape[-1]).float().cpu())
    routers = {}
    for p in POINTS:
        X = torch.cat(acts[p])
        mean = X.mean(0)
        Xn = torch.nn.functional.normalize(X - mean, dim=-1)
        _, _, V = torch.pca_lowrank(Xn, q=PCA_DIMS, center=False)  # [d, PCA_DIMS]
        routers[p] = Router(mean.to(dev), V.to(dev), kmeans((Xn @ V).to(dev), E))
    return routers


def fit_jacobians(target, tokens: torch.Tensor, routers: dict, dev: str, chunk: int, log) -> tuple[dict, dict]:
    """Per point: the sum over sources of each expert's J-lens terms [E, d, d] and its source count [E] (LRP)."""
    d = target.wte.shape[1]
    J = {p: torch.zeros(E, d, d, device=dev, dtype=torch.float64) for p in POINTS}
    n = {p: torch.zeros(E, device=dev, dtype=torch.float64) for p in POINTS}
    t0 = time.time()
    for r in range(tokens.shape[0]):
        ids = tokens[r:r + 1].to(dev)
        final, pts = forward_points(target, ids, lrp=True, grad=True)
        Tn = ids.shape[1]
        w = 1.0 / (Tn - torch.arange(Tn, device=dev, dtype=torch.float64))  # 1 / (T - s): the mean over t >= s
        w[:SKIP] = 0.0
        onehot = {}
        for p in POINTS:
            lab = routers[p](pts[p][0].detach())
            M = torch.nn.functional.one_hot(lab, E).double() * w[:, None]  # [T, E]
            onehot[p] = M
            n[p] += (torch.nn.functional.one_hot(lab, E).double() * (w > 0)[:, None]).sum(0)
        for k0 in range(0, d, chunk):
            ks = torch.arange(k0, min(k0 + chunk, d), device=dev)
            cot = torch.zeros(len(ks), 1, Tn, d, device=dev)
            cot[torch.arange(len(ks)), 0, :, ks] = 1.0  # sum over targets t of h_final,t[k]
            grads = torch.autograd.grad(final, [pts[p] for p in POINTS], grad_outputs=cot, is_grads_batched=True,
                                        retain_graph=True)
            for p, g in zip(POINTS, grads):  # g [K, 1, T, d]: d sum_t h_final,t[k] / d h_{p,s}
                J[p][:, k0:k0 + len(ks), :] += torch.einsum("kti,te->eki", g[:, 0].double(), onehot[p])
        if r % 8 == 7 or r == tokens.shape[0] - 1:
            log(f"jacobians: {r + 1}/{tokens.shape[0]} sequences, {(r + 1) / (time.time() - t0):.2f}/s")
    return J, n


def fit(args):
    out = Path(args.out) / "lens"
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    target = VM.load_target(dev)
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    c0, c1 = (int(x) for x in args.cluster.split(":"))
    f0, f1 = (int(x) for x in args.fit.split(":"))
    t0 = time.time()
    routers = fit_routers(target, tokens_of(c0, c1, args.tokens), dev, args.batch)
    log(f"routers: {time.time() - t0:.0f} s")
    J, n = fit_jacobians(target, tokens_of(f0, f1, args.tokens), routers, dev, args.chunk, log)
    torch.save({"J": {p: v.float().cpu() for p, v in J.items()}, "n": {p: v.cpu() for p, v in n.items()},
                "routers": {p: (r.mean.cpu(), r.basis.cpu(), r.centres.cpu()) for p, r in routers.items()},
                "fit_rows": [f0, f1], "cluster_rows": [c0, c1], "seconds": time.time() - t0}, out / "jacobians.pt")
    log(f"wrote {out / 'jacobians.pt'} ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ evaluation


def semantic_tokens(vocab: int) -> torch.Tensor:
    """Tokens whose text holds at least one letter or digit."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(VM.TARGET_DIR / "tokenizer.json"))
    return torch.tensor([any(ch.isalnum() for ch in tok.decode([i])) for i in range(vocab)])


def pearson_stats(P: torch.Tensor, c: torch.Tensor):
    """Per sample, over tokens: covariances b_i = Cov(P_i, c) [N, K], S_ij = Cov(P_i, P_j) [N, K, K], Var(c) [N]."""
    Pc = P - P.mean(-1, keepdim=True)  # [N, K, V]
    cc = c - c.mean(-1, keepdim=True)  # [N, V]
    return torch.einsum("nkv,nv->nk", Pc, cc), torch.einsum("nkv,njv->nkj", Pc, Pc), cc.pow(2).sum(-1)


def corr_of(w: torch.Tensor, b, S, vc) -> torch.Tensor:
    return (b @ w) / ((torch.einsum("k,nkj,j->n", w, S, w) * vc).clamp_min(1e-30).sqrt())


def evaluate(args):
    from safetensors.torch import load_file, save_file
    root = Path(args.out)
    dev = args.device
    meta = json.loads((root / "atlas.json").read_text())
    atlas = load_file(str(root / "atlas.safetensors"))
    saved = torch.load(root / "lens" / "jacobians.pt")
    target = VM.load_target(dev)
    vpd = VM.load_vpd(target, dev)
    names = meta["names"]
    a0, a1 = meta["train_rows"]
    b0, b1 = meta["heldout_rows"] if not args.rows else [int(x) for x in args.rows.split(":")]
    V = target.wte.shape[0]
    train = tokens_of(a0, a1, args.tokens)
    seen = (torch.bincount(train.reshape(-1), minlength=V) >= MIN_COUNT).to(dev)
    sel = seen & semantic_tokens(V).to(dev)
    in_seen = sel[seen]  # the readout tokens among the seen ones
    routers = {p: Router(*(x.to(dev) for x in saved["routers"][p])) for p in POINTS}
    static = torch.load(root / "lens" / "routed.pt")["weights"].to(dev) if (root / "lens" / "routed.pt").exists() else None
    pooled = {p: (saved["J"][p].double().sum(0) / saved["n"][p].sum()).float().to(dev) for p in POINTS}
    expert = {p: (saved["J"][p] / saved["n"][p].clamp_min(1)[:, None, None]).float().to(dev) for p in POINTS}  # [E, d, d]
    off = dict(zip(names, np.concatenate([[0], np.cumsum([meta["counts"][n] for n in names])])[:-1].tolist()))
    held = tokens_of(b0, b1, args.tokens)
    if args.rows:  # a slice of split B (a quick check): its first half fits the weights, its second is evaluated
        rows_all = atlas["effect_row"]
        atlas["effect_row"] = torch.where((rows_all >= b0) & (rows_all < b1), rows_all, torch.full_like(rows_all, -1))
    rows, pos = atlas["effect_row"], atlas["effect_pos"]  # [total, SAMPLES], absolute rows
    mid = (b0 + b1) // 2
    gf = target.ln_f

    def readout(Jm, r, tokens=sel):  # [E?, d, d] x [N, d] -> [.., N, V] restricted to tokens
        y = r if Jm is None else (r @ Jm.transpose(-1, -2))
        return ((y * gf) @ target.wte.T)[..., tokens]

    def per_sample(p, r, h):  # each sample through its own cluster's expert: [N, d] -> [N, V'] over the seen tokens
        Jn = expert[p][routers[p](h)]  # [N, d, d]
        return ((torch.einsum("nij,nj->ni", Jn, r) * gf) @ target.wte.T)[..., seen]

    stats = {p: [] for p in POINTS}
    metrics = {}
    t0 = time.time()

    def run(phase):
        for r0 in range(b0, b1, args.batch):
            want = (rows >= r0) & (rows < min(r0 + args.batch, b1)) & ((rows < mid) if phase == "dev" else (rows >= mid))
            if not want.any():
                continue
            ids = held[r0 - b0:r0 - b0 + args.batch].to(dev)
            cache = clean_cache(target, ids)
            pts = forward_points(target, ids, lrp=False)[1] if phase == "eval" else None
            cs, ks = want.nonzero(as_tuple=True)
            for si, name in enumerate(names):
                layer, kind = int(name.split(".")[1]), name.rsplit(".", 1)[1]
                p = point_of(layer, kind)
                if p is None:
                    continue
                m = (cs >= off[name]) & (cs < off[name] + meta["counts"][name])
                if not m.any():
                    continue
                st = target.site(name)
                dirs = write_directions(target, layer, kind)
                for e0 in range(0, int(m.sum()), 64):
                    cc, kk = cs[m][e0:e0 + 64], ks[m][e0:e0 + 64]
                    j = (cc - off[name]).to(dev)
                    b = (rows[cc, kk] - r0).long().to(dev)
                    q = pos[cc, kk].long().to(dev)
                    a = (cache["site_in"][(layer, kind)][b, q] * st.V[:, j].T).sum(-1)
                    full = cache["logp"][b, q] - row_logprobs(target, cache, b, q, layer, kind, -a[:, None] * st.U[j])
                    r = dirs[j]
                    if phase == "dev":
                        P = a[:, None, None] * readout(expert[p], r).transpose(0, 1)  # [N, E, V']
                        stats[p].append([x.cpu() for x in pearson_stats(P.float(), full[:, sel].float())])
                        continue
                    preds = {"logit": readout(None, r, seen), "pooled": readout(pooled[p], r, seen),
                             "routed": per_sample(p, r, pts[p][b, q]), "jpp": readout(weighted[p], r, seen)}
                    if static is not None:  # the atlas's main readout: the mixture of its top contexts' experts (the
                        ws = static[cc.to(dev)]  # logit lens where it has no context, as the atlas falls back)
                        y = torch.einsum("ne,eij,nj->ni", ws, expert[p], r)
                        y = torch.where(ws.sum(-1, keepdim=True) > 0, y, r)
                        preds["routed_static"] = ((y * gf) @ target.wte.T)[..., seen]
                    contrib = full[:, seen]
                    for lens_name, pr in preds.items():
                        for tokens, among in (("filtered", in_seen), ("all", None)):
                            x, c = (a[:, None] * pr, contrib) if among is None else (a[:, None] * pr[:, among], contrib[:, among])
                            pc, ccn = x - x.mean(-1, keepdim=True), c - c.mean(-1, keepdim=True)
                            corr = (pc * ccn).sum(-1) / (pc.norm(dim=-1) * ccn.norm(dim=-1)).clamp_min(1e-30)
                            tp, tc = x.topk(EFFECT, -1).indices, c.topk(EFFECT, -1).indices
                            ov = (tp[:, :, None] == tc[:, None, :]).any(-1).float().mean(-1)
                            metrics.setdefault((name, lens_name, tokens), []).append(torch.stack([corr, ov], -1).cpu())
            print(f"{time.strftime('%H:%M:%S')} {phase}: rows {r0}..{r0 + args.batch}, {time.time() - t0:.0f} s", flush=True)

    with torch.no_grad():
        run("dev")
    weighted, fitted = {}, {}
    for p in POINTS:
        if p == "out_3" or not stats[p]:
            weighted[p] = torch.eye(target.wte.shape[1], device=dev)  # the final residual: J = I for every lens
            continue
        bb, SS, vc = (torch.cat([s[i] for s in stats[p]]).to(dev).double() for i in range(3))
        w = (saved["n"][p] / saved["n"][p].sum()).to(dev).double().clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(500):
            loss = -corr_of(w, bb, SS, vc).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        w = w.detach()
        fitted[p] = {"weights": w.cpu().tolist(), "dev_corr_mean": float(corr_of(w, bb, SS, vc).mean()),
                     "dev_corr_pooled": float(corr_of((saved["n"][p] / saved["n"][p].sum()).to(dev).double(), bb, SS, vc).mean()),
                     "dev_samples": int(bb.shape[0])}
        weighted[p] = torch.einsum("e,eij->ij", w.float(), expert[p])
        print(f"{p}: weights {[round(x, 3) for x in fitted[p]['weights']]}, dev corr {fitted[p]['dev_corr_mean']:.3f} "
              f"(pooled {fitted[p]['dev_corr_pooled']:.3f})", flush=True)
    for p in POINTS:
        weighted.setdefault(p, torch.eye(target.wte.shape[1], device=dev))
    with torch.no_grad():
        run("eval")
    table = {}
    for (name, lens_name, tokens), xs in sorted(metrics.items()):
        x = torch.cat(xs)
        table.setdefault(name, {}).setdefault(tokens, {})[lens_name] = {
            "corr_median": float(x[:, 0].nanmedian()), "corr_mean": float(x[:, 0].nanmean()),
            "overlap_mean": float(x[:, 1].nanmean()), "samples": int(x.shape[0])}
    for name in table:
        if point_of(int(name.split(".")[1]), name.rsplit(".", 1)[1]) == "out_3":
            table[name]["note"] = "the final residual: every lens is the logit lens (J = I), identical by construction"
    labels = {"logit": "logit lens (J = I), no labels", "pooled": "pooled LRP Jacobian (R-lens), no labels",
              "routed": "each sample through its own k-means cluster's expert, no labels",
              "routed_static": "the atlas's main readout: each subcomponent through the mixture of the experts its top contexts are "
                               "routed to (the logit lens without contexts), no labels",
              "jpp": f"J++: expert weights fitted (supervised) to the measured effects of dev rows {b0}..{mid - 1}",
              "filtered": "tokens with a letter or digit seen >= MIN_COUNT times in split A",
              "all": "every token seen >= MIN_COUNT times in split A", "evaluated": f"rows {mid}..{b1 - 1}"}
    res = {"table": table, "labels": labels, "fitted": fitted, "seconds": time.time() - t0, "fields": __doc__}
    (root / "lens" / "evaluation.json").write_text(json.dumps(res, indent=1) + "\n")
    for name, row in table.items():
        for tokens in ("filtered", "all"):
            print(f"{name} {tokens}:", "  ".join(f"{k}: corr {v['corr_median']:.3f} overlap {v['overlap_mean']:.3f}"
                                               for k, v in row[tokens].items()), flush=True)
    # J++ readouts into the atlas for the sites where it beats the logit lens (median correlation, filtered tokens).
    wins = [n for n, row in table.items() if row["filtered"]["jpp"]["corr_median"] > row["filtered"]["logit"]["corr_median"]]
    if not wins:
        print("J++ beats the logit lens on no site: the atlas is unchanged", flush=True)
        return
    total = meta["total"]
    up_t = torch.full((total, 10), -1, dtype=torch.int32)
    dn_t = torch.full((total, 10), -1, dtype=torch.int32)
    up_v = torch.zeros((total, 10), dtype=torch.float16)
    dn_v = torch.zeros((total, 10), dtype=torch.float16)
    sign = atlas["lens_sign"].float()
    idx = sel.nonzero().flatten()
    for name in wins:
        layer, kind = int(name.split(".")[1]), name.rsplit(".", 1)[1]
        p = point_of(layer, kind)
        r = write_directions(target, layer, kind)
        for c0 in range(0, meta["counts"][name], 512):
            w = readout(weighted[p], r[c0:c0 + 512]) * sign[off[name] + c0:off[name] + c0 + 512, None].to(dev)
            v, t = w.topk(10, -1)
            up_t[off[name] + c0:off[name] + c0 + 512], up_v[off[name] + c0:off[name] + c0 + 512] = idx[t].int().cpu(), v.half().cpu()
            v, t = (-w).topk(10, -1)
            dn_t[off[name] + c0:off[name] + c0 + 512], dn_v[off[name] + c0:off[name] + c0 + 512] = idx[t].int().cpu(), (-v).half().cpu()
    atlas.update({"jpp_up_token": up_t, "jpp_up_value": up_v, "jpp_down_token": dn_t, "jpp_down_value": dn_v})
    save_file({k: v.contiguous() for k, v in atlas.items()}, str(root / "atlas.safetensors"))
    meta["jpp"] = {"sites": wins, "evaluation": "lens/evaluation.json"}
    (root / "atlas.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"J++ readouts written for {len(wins)} sites: {wins}", flush=True)


class RoutedLens:
    """The routed lens at given positions of one text (the label-free lens with the best measured agreement, read per
    sample as evaluated): a node (matrix, position, index) is read through the expert its point's residual at that
    position is routed to, oriented by its activation there (what its write raises at that position). q_proj and k_proj
    (no write) get none; the last layer's MLP is the logit lens (J = I)."""

    def __init__(self, target, root: Path = DEFAULT, tokens: str | None = None):
        dev = target.wte.device
        saved = torch.load(Path(root) / "lens" / "jacobians.pt")
        self.target = target
        self.routers = {p: Router(*(x.to(dev) for x in saved["routers"][p])) for p in POINTS}
        self.expert = {p: (saved["J"][p] / saved["n"][p].clamp_min(1)[:, None, None]).float().to(dev) for p in POINTS}
        meta = json.loads((Path(root) / "atlas.json").read_text())
        a0, a1 = meta["train_rows"]
        path = tokens or str(Path(root) / "tokens_0_6000.npy")
        V = target.wte.shape[0]
        self.seen = (torch.bincount(tokens_of(a0, a1, path).reshape(-1), minlength=V) >= MIN_COUNT).to(dev)

    @torch.no_grad()
    def readout(self, ids: list[int], nodes: list[tuple], k: int = 10) -> list[dict | None]:
        """Per node {"raised": [(token id, value)], "lowered": [...]} over the tokens split A holds at least MIN_COUNT
        times, or None (q_proj, k_proj)."""
        tg = self.target
        dev = tg.wte.device
        x = torch.tensor([ids], device=dev)
        _, pts = forward_points(tg, x, lrp=False)
        out = []
        neg = torch.full_like(self.seen, -math.inf, dtype=torch.float32)
        acts = {}
        for n in {nd[0] for nd in nodes}:
            st = tg.site(n)
            st.cache_input = True
        try:
            tg.hidden(x)
            for n in {nd[0] for nd in nodes}:
                acts[n] = tg.site(n).last_input[0] @ tg.site(n).V  # [T, C]
        finally:
            for n in {nd[0] for nd in nodes}:
                tg.site(n).cache_input, tg.site(n).last_input = False, None
        for n, t, c in nodes:
            layer, kind = int(n.split(".")[1]), n.rsplit(".", 1)[1]
            p = point_of(layer, kind)
            if p is None:
                out.append(None)
                continue
            r = write_directions(tg, layer, kind)[c]
            y = r if p == "out_3" else self.expert[p][self.routers[p](pts[p][0, t][None])[0]] @ r
            w = ((y * tg.ln_f) @ tg.wte.T) * (1.0 if float(acts[n][t, c]) >= 0 else -1.0)
            v, i = torch.where(self.seen, w, neg).topk(k)
            dv, di = torch.where(self.seen, -w, neg).topk(k)
            out.append({"raised": list(zip(i.tolist(), v.tolist())), "lowered": list(zip(di.tolist(), (-dv).tolist()))})
        return out


def routed(args):
    """The routed lens as the atlas's readout of each subcomponent (lens/routed.pt and the atlas's routed_* fields):
    its map is the mixture of its point's expert Jacobians in the proportions its top contexts (split A) are routed
    to them, its readout over the tokens split A holds at least MIN_COUNT times (the logit lens's), oriented as the
    atlas's lens. Subcomponents without contexts, q_proj and k_proj (no write), and the last layer's MLP (the final
    residual: every lens is the logit lens) get none."""
    from safetensors.torch import load_file, save_file
    root, dev = Path(args.out), args.device
    meta = json.loads((root / "atlas.json").read_text())
    atlas = load_file(str(root / "atlas.safetensors"))
    saved = torch.load(root / "lens" / "jacobians.pt")
    target = VM.load_target(dev)
    VM.load_vpd(target, dev)  # the subcomponents' write directions
    names, total = meta["names"], meta["total"]
    a0, a1 = meta["train_rows"]
    V = target.wte.shape[0]
    seen = (torch.bincount(tokens_of(a0, a1, args.tokens).reshape(-1), minlength=V) >= MIN_COUNT).to(dev)
    if args.rows:  # a slice of split A's rows (a quick check)
        a0, a1 = (int(x) for x in args.rows.split(":"))
    routers = {p: Router(*(x.to(dev) for x in saved["routers"][p])) for p in POINTS}
    expert = {p: (saved["J"][p] / saved["n"][p].clamp_min(1)[:, None, None]).float().to(dev) for p in POINTS}
    off = dict(zip(names, np.concatenate([[0], np.cumsum([meta["counts"][n] for n in names])])[:-1].tolist()))
    point = torch.full((total,), -1, dtype=torch.long)
    for n in names:
        p = point_of(int(n.split(".")[1]), n.rsplit(".", 1)[1])
        if p is not None and p != "out_3":
            point[off[n]:off[n] + meta["counts"][n]] = POINTS.index(p)
    rows, pos = atlas["top_row"].long(), atlas["top_pos"].long()  # [total, TOP], absolute rows of split A
    W = torch.zeros(total, E, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad():
        for r0 in range(a0, a1, args.batch):
            r1 = min(r0 + args.batch, a1)
            m = (rows >= r0) & (rows < r1) & (point[:, None] >= 0)
            if not m.any():
                continue
            _, pts = forward_points(target, tokens_of(r0, r1, args.tokens).to(dev), lrp=False)
            c, k = m.nonzero(as_tuple=True)
            for pi, p in enumerate(POINTS):
                sel = point[c] == pi
                if not sel.any():
                    continue
                lab = routers[p](pts[p][(rows[c[sel], k[sel]] - r0).to(dev), pos[c[sel], k[sel]].to(dev)]).cpu()
                W.index_put_((c[sel], lab), torch.ones(int(sel.sum()), dtype=torch.float64), accumulate=True)
            if (r0 - a0) // args.batch % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} routed: rows {r0}..{r1}, {time.time() - t0:.0f} s", flush=True)
    has = W.sum(1) > 0
    W = W / W.sum(1, keepdim=True).clamp_min(1)
    torch.save({"weights": W.float(), "point": point}, root / "lens" / "routed.pt")
    up_t = torch.full((total, 10), -1, dtype=torch.int32)
    dn_t = torch.full((total, 10), -1, dtype=torch.int32)
    up_v = torch.zeros((total, 10), dtype=torch.float16)
    dn_v = torch.zeros((total, 10), dtype=torch.float16)
    sign = atlas["lens_sign"].float()
    neg = torch.full((V,), -float("inf"), device=dev)
    for n in names:
        layer, kind = int(n.split(".")[1]), n.rsplit(".", 1)[1]
        p = point_of(layer, kind)
        if p is None or p == "out_3":
            continue
        r = write_directions(target, layer, kind)
        for c0 in range(0, meta["counts"][n], 256):
            sl = slice(off[n] + c0, off[n] + min(c0 + 256, meta["counts"][n]))
            y = torch.einsum("ce,eij,cj->ci", W[sl].float().to(dev), expert[p], r[c0:c0 + 256])
            w = ((y * target.ln_f) @ target.wte.T) * sign[sl, None].to(dev)
            v, t = torch.where(seen[None], w, neg).topk(10, -1)
            up_t[sl], up_v[sl] = t.int().cpu(), v.half().cpu()
            v, t = torch.where(seen[None], -w, neg).topk(10, -1)
            dn_t[sl], dn_v[sl] = t.int().cpu(), (-v).half().cpu()
    for x in (up_t, dn_t):
        x[~has] = -1
    atlas.update({"routed_up_token": up_t, "routed_up_value": up_v, "routed_down_token": dn_t, "routed_down_value": dn_v})
    save_file({k: v.contiguous() for k, v in atlas.items()}, str(root / "atlas.safetensors"))
    meta["routed"] = {"subcomponents": int(has.sum()), "readout": "the main readout where present (Atlas.entry)",
                      "fields": routed.__doc__}
    (root / "atlas.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"routed readouts written for {int(has.sum())} subcomponents ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", default=str(DEFAULT))
    f.add_argument("--tokens")
    f.add_argument("--fit", default="0:64")
    f.add_argument("--cluster", default="64:264")
    f.add_argument("--batch", type=int, default=8)
    f.add_argument("--chunk", type=int, default=64, help="output dimensions per batched backward pass")
    f.add_argument("--device", default="cuda")
    e = sub.add_parser("evaluate")
    e.add_argument("--out", default=str(DEFAULT))
    e.add_argument("--tokens")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--device", default="cuda")
    e.add_argument("--rows", help="a slice A:B of split B instead of all of it (a quick check)")
    r = sub.add_parser("routed")
    r.add_argument("--out", default=str(DEFAULT))
    r.add_argument("--tokens")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--device", default="cuda")
    r.add_argument("--rows", help="a slice A:B of split A's rows instead of all of them (a quick check)")
    args = ap.parse_args()
    {"fit": fit, "evaluate": evaluate, "routed": routed}[args.cmd](args)


if __name__ == "__main__":
    main()
