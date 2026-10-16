"""Stage 1 of the complex-trait neuron PheWAS (#2951, the lead's 10-15): association of every neuron pole of Qwen3-8B
with every trait of the phenome (phenome.py), the GWAS analogue with prompts as individuals.

Genotype, per prompt (the trait's form-A prompt) and neuron (the down projection's input, act(gate) x up), two
measures per pole:
  max   the pole's largest value over the prompt's positions from 1 on (the first position, the attention sink, out):
        pole + its largest positive value, pole - its largest negative value in magnitude (0 where it has none);
  ans   the value at the answer position (the prompt's last token, where the answer's first token is predicted):
        pole + max(v, 0), pole - max(-v, 0).
Test, per trait, pole and measure: the trait's per-prompt value (the mean of its two forms) regressed on the genotype,
both rank-based inverse normal transformed (rint: outlying prompts cannot carry an association), with covariates: an intercept, log prompt length, the item's own covariates (phenome.py's: which answer matches the
behaviour) and the top PCS principal components of the trait's own genotype matrix (standardized columns; the
population-structure adjustment). Reported: the partial correlation r, the effect
(trait units per genotype standard deviation), t with n - (covariates + 2) degrees of freedom and its two-sided p; genomic-control
lambda (median chi-square / 0.4549); Bonferroni over every pole x measure x trait tested. The association is run on all
of a trait's prompts and on its dev and held-out halves separately (replication: the correlation of the halves' t over
poles, and the share of the all-prompt Bonferroni hits whose dev and held-out effects share a sign). A null scan with the
trait's values shuffled over its prompts gives the null's lambda, Bonferroni hits and smallest p (the calibration).
Three checks of the all-prompt scan's counts:
  mixed model   EMMAX with the tested layer left out: per layer, the prompts' relatedness K is the standardized genotypes'
                Gram matrix over the other 35 layers' columns (diagonal about 1); the variance ratio delta of
                y ~ C b + g + e, g ~ N(0, s2 K), e ~ N(0, s2 delta I) by REML on a grid (y's share h2 = 1 / (1 + delta)),
                then each column of the layer tested by least squares after whitening with K + delta I (same covariates
                C, same degrees of freedom); its own shuffled null;
  LD scores     per column the sum over all columns of the squared correlation of the covariate-residualized genotypes,
                each bias-corrected r2 - (1 - r2) / (d - 1) with d = n - (covariates) (from the n x n Gram matrix, never
                the F x F one), and the regression of chi-square on it (LD score regression: two-step weighted least
                squares, chi-square over CHI2_CAP out, 200-block jackknife): intercept above 1 = inflation not explained
                by many small correlated effects (confounding), slope = the signal;
  clumping      the Bonferroni hits, smallest p first, each leading a clump of the remaining hits correlated with it at
                r2 > CLUMP_R2 (covariate-residualized genotypes): the count of independent signals.
The CPU work (p values, the LD score regression, writing) runs on a thread while the GPU scans the next trait.
Outputs: OUT/<trait>.npz (t on all prompts and the partial r, [layers, neurons, 4] float16: max+, max-, ans+, ans-; the
halves' t; the mixed model's t and per-layer h2; the LD scores; the clump leads; degrees of freedom; the PCs' variance
shares), OUT/summary.json (per trait: n, lambda, hits, replication, the mixed model's lambda and hits, the LD score
regressions, clumps, the top TOP hits with r, effect, p and their dev and held-out t).

  assoc.py PHENOME_DIR OUT [--model Qwen/Qwen3-8B] [--traits NAME ...]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PCS = 10
ROWS = 16  # prompts a batch, every batch padded to it
RINT_COLS = 1 << 17  # genotype columns ranked at once
TOP = 50
MEASURES = ("max+", "max-", "ans+", "ans-")
CLUMP_R2 = 0.1
CHI2_CAP = 80.0  # (LD score regression's default exclusion of very large chi-square)
DELTAS = (-4.0, 4.0, 161)  # log10 grid of the mixed model's variance ratio


def genotypes(model, layers, seqs: list[list[int]], dev):
    """[n, L, D, 4] float16 on dev: per prompt the four measures of every neuron."""
    import torch

    L, D = len(layers), layers[0].mlp.down_proj.in_features
    out = torch.zeros(len(seqs), L, D, 4, dtype=torch.float16, device=dev)
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
    k = 0
    while k < len(order):
        b = min(ROWS, len(order) - k)
        idx = order[k:k + b] + order[k + b - 1:k + b] * (ROWS - b)  # every batch ROWS prompts (the batch size moves bf16 outputs)
        T = max(len(seqs[i]) for i in idx)
        x = torch.zeros(len(idx), T, dtype=torch.long, device=dev)
        m = torch.zeros(len(idx), T, dtype=torch.bool, device=dev)
        last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=dev)
        for j, i in enumerate(idx):
            x[j, :len(seqs[i])] = torch.tensor(seqs[i], device=dev)
            m[j, :len(seqs[i])] = True
        live = m.clone()
        live[:, 0] = False  # (the attention sink)
        rows = torch.tensor(idx, device=dev)

        def hook(l_):
            def fn(_m, args):
                h = args[0].float()  # [B, T, D]
                pos = torch.where(live[..., None], h.clamp_min(0), 0).amax(1)
                neg = torch.where(live[..., None], (-h).clamp_min(0), 0).amax(1)
                a = h[torch.arange(len(idx), device=dev), last]
                out[rows, l_] = torch.stack([pos, neg, a.clamp_min(0), (-a).clamp_min(0)], -1).half()
            return fn

        hs = [ly.mlp.down_proj.register_forward_pre_hook(hook(l_)) for l_, ly in enumerate(layers)]
        try:
            with torch.no_grad():
                model.model(input_ids=x, attention_mask=m.long())
        finally:
            for h in hs:
                h.remove()
        k += b
    return out


def rint(X):
    """The rank-based inverse normal transform of each column of X [n, F] (ties share their average rank): a few
    prompts with very large values cannot carry an association (the first scan's permutation null found 24 Bonferroni
    hits on harmful refusal and all 13 of the suggested-wrong-answer trait's)."""
    import torch

    n, F = X.shape
    v, order = X.float().sort(0)
    new = torch.ones_like(v, dtype=torch.bool)
    new[1:] = v[1:] != v[:-1]
    gid = new.long().cumsum(0) - 1  # [n, F] tie group within each column
    key = gid * F + torch.arange(F, device=X.device)
    pos = torch.arange(n, device=X.device, dtype=torch.float32)[:, None].expand(n, F)
    first = torch.full((n * F,), float(n), device=X.device).scatter_reduce(0, key.reshape(-1), pos.reshape(-1), "amin")
    last = torch.full((n * F,), -1.0, device=X.device).scatter_reduce(0, key.reshape(-1), pos.reshape(-1), "amax")
    avg = ((first + last) / 2)[key.reshape(-1)].reshape(n, F)
    out = torch.empty_like(avg)
    out.scatter_(0, order, avg)
    return torch.special.ndtri((out + 0.5) / n)


def standardized(G):
    """Each column of G [n, F] rint-transformed (in column chunks: memory) and standardized, in place: Z [n, F] float32
    and the non-constant columns ok [F]."""
    import torch

    Z = torch.cat([rint(G[:, s:s + RINT_COLS]) for s in range(0, G.shape[1], RINT_COLS)], 1)
    mu, sd = Z.mean(0), Z.std(0)
    ok = sd > 1e-6
    Z.sub_(mu).div_(torch.where(ok, sd, 1.0))
    Z[:, ~ok] = 0
    return Z, ok


def design(Z, length, pcs: int, extra=None):
    """The covariates [n, c]: an intercept, log length, extra [n, k] and the top pcs PCs of Z; the PCs' variance shares."""
    import torch

    n = Z.shape[0]
    U, S, _ = torch.svd_lowrank(Z, q=pcs + 10, niter=4)
    C = torch.cat([torch.ones(n, 1, device=Z.device), torch.log(length.float())[:, None]] + ([] if extra is None else [extra])
                  + [U[:, :pcs]], 1)
    return C, (S ** 2 / Z.pow(2).sum()).tolist()[:pcs]


def ols(Zr, zn, yr, ok, df: int):
    """t, partial r and effect (y per genotype sd) of residualized y on each residualized column Zr [n, F] (norms zn)."""
    import torch

    num = Zr.T @ yr
    den = zn * yr.norm()
    r = torch.where(ok & (den > 0), num / den.clamp_min(1e-12), 0)
    t = r * math.sqrt(df) / (1 - r ** 2).clamp_min(1e-12).sqrt()
    return t, r, num / zn.pow(2).clamp_min(1e-12)


def residualize(Q, X):
    return X - Q @ (Q.T @ X)


def scan(G, y, length, pcs: int, extra=None):
    """t and partial r of y on each genotype column given an intercept, log length, extra covariates [n, k] and the
    top pcs PCs of G; genotype columns and y rank-based inverse normal transformed (rint). G [n, F] float16, y [n],
    length [n] -> t [F], r [F], effect [F] (y's transformed units per genotype sd), df, pc variance shares."""
    import torch

    Z, ok = standardized(G)
    C, var = design(Z, length, pcs, extra)
    Q, _ = torch.linalg.qr(C)
    Z.sub_(Q @ (Q.T @ Z))
    df = Z.shape[0] - C.shape[1] - 1
    t, r, eff = ols(Z, Z.norm(dim=0), residualize(Q, rint(y[:, None])[:, 0]), ok, df)
    return t, r, eff, df, var


def reml_delta(lam, Cs, ys):
    """The REML variance ratio delta = s2_e / s2_g of ys ~ Cs b + N(0, s2_g (diag(lam) + delta I)) in K's eigenbasis
    (lam, Cs, ys rotated; float64), the best of the log10 grid DELTAS (FaST-LMM's restricted likelihood)."""
    import torch

    n, p = Cs.shape
    d = torch.logspace(*DELTAS, dtype=torch.float64, device=lam.device)
    w = 1 / (lam[None] + d[:, None])  # [grid, n]
    A = torch.einsum("gn,np,nq->gpq", w, Cs, Cs)
    b = torch.einsum("gn,np,n->gp", w, Cs, ys)
    res = ys[None] - torch.linalg.solve(A, b) @ Cs.T
    s2 = (w * res ** 2).sum(1) / (n - p)
    ll = -0.5 * ((n - p) * torch.log(s2) + torch.log(lam[None] + d[:, None]).sum(1) + torch.linalg.slogdet(A)[1])
    return float(d[int(ll.argmax())])


def lmm(Z, ok, C, ys, width: int):
    """EMMAX, the tested layer left out of the relatedness: for layer l (columns l*width to (l+1)*width of the
    standardized genotypes Z [n, F]), K = Z_-l Z_-l^T / F_-l, delta by REML for each y of ys (rint, [m, n]), and each of
    the layer's columns tested by least squares after whitening with K + delta I. -> t [m, F], h2 [m, layers]."""
    import torch

    n, F = Z.shape
    nl = F // width
    K = torch.stack([Z[:, l * width:(l + 1) * width] @ Z[:, l * width:(l + 1) * width].T for l in range(nl)]).double()
    Fl = ok.view(nl, width).sum(1).double()
    Kall, Ftot = K.sum(0), Fl.sum()
    t = torch.zeros(len(ys), F, device=Z.device)
    h2 = torch.zeros(len(ys), nl)
    df = n - C.shape[1] - 1
    for l in range(nl):
        lam, V = torch.linalg.eigh((Kall - K[l]) / (Ftot - Fl[l]))
        lam = lam.clamp_min(0)
        Cs = V.T @ C.double()
        cols = slice(l * width, (l + 1) * width)
        Zl = V.T.float() @ Z[:, cols]
        for k, y in enumerate(ys):
            yr = V.T @ y.double()
            delta = reml_delta(lam, Cs, yr)
            h2[k, l] = 1 / (1 + delta)
            w = (1 / (lam + delta)).sqrt()
            Q, _ = torch.linalg.qr(Cs * w[:, None])
            Q = Q.float()
            Zw = residualize(Q, Zl * w.float()[:, None])
            t[k, cols] = ols(Zw, Zw.norm(dim=0), residualize(Q, (yr * w).float()), ok[cols], df)[0]
    return t, h2


def ld_scores(Zr, zn, ok, d: int):
    """Per column the bias-corrected sum over all columns of the squared correlation of the residualized genotypes Zr
    [n, F] (norms zn), from the n x n Gram matrix of the unit columns: sum_k r2 = u_j^T (U U^T) u_j, then
    r2 - (1 - r2) / (d - 1) summed. NaN for constant columns."""
    import torch

    n, F = Zr.shape
    Ku = torch.zeros(n, n, device=Zr.device)
    for s in range(0, F, RINT_COLS):
        U = Zr[:, s:s + RINT_COLS] / zn[s:s + RINT_COLS].clamp_min(1e-12)
        Ku += U @ U.T
    S = torch.empty(F, device=Zr.device)
    for s in range(0, F, RINT_COLS):
        U = Zr[:, s:s + RINT_COLS] / zn[s:s + RINT_COLS].clamp_min(1e-12)
        S[s:s + RINT_COLS] = (U * (Ku @ U)).sum(0)
    l = S * d / (d - 1) - float(ok.sum()) / (d - 1)
    return torch.where(ok, l, float("nan"))


def ldsc(chi2, l, blocks: int = 200):
    """LD score regression of chi2 on l (numpy): chi-square over CHI2_CAP out; step 1 weights 1 / max(l, 1), step 2 also
    1 / (2 fitted^2); intercept and slope with 200-block jackknife standard errors (columns in their order), the mean
    chi-square and the ratio (intercept - 1) / (mean - 1): the share of the inflation that is not signal."""
    import numpy as np

    m = np.isfinite(l) & np.isfinite(chi2) & (chi2 < CHI2_CAP)
    x, y = l[m].astype(np.float64), chi2[m].astype(np.float64)
    blk = np.arange(len(x)) * blocks // max(len(x), 1)

    def fit(w):
        sums = [np.bincount(blk, weights=w * v, minlength=blocks) for v in (np.ones_like(x), x, x * x, y, x * y)]
        tot = [s_.sum() for s_ in sums]

        def solve(v):
            A = np.array([[v[0], v[1]], [v[1], v[2]]])
            return np.linalg.solve(A, np.array([v[3], v[4]]))

        coef = solve(tot)
        jack = np.array([solve([tot[i] - sums[i][b] for i in range(5)]) for b in range(blocks)])
        se = np.sqrt((blocks - 1) / blocks * ((jack - jack.mean(0)) ** 2).sum(0))
        return coef, se

    lw = np.maximum(x, 1)
    (a, b_), _ = fit(1 / lw)
    (a, b_), (sa, sb) = fit(1 / (lw * 2 * np.maximum(a + b_ * x, 1) ** 2))
    mean = float(y.mean())
    return {"intercept": float(a), "intercept_se": float(sa), "slope": float(b_), "slope_se": float(sb), "mean_chi2": mean,
            "ratio": float((a - 1) / (mean - 1)) if mean > 1 else None, "columns": int(m.sum())}


def clump(Zr, zn, hits):
    """Greedy clumping of hits (column indices, smallest p first): the first unclumped hit leads, every unclumped hit
    with r2 > CLUMP_R2 to it joins. -> the leads' positions in hits, the clump sizes."""
    import torch

    if len(hits) == 0:
        return [], []
    H = Zr[:, hits] / zn[hits].clamp_min(1e-12)
    alive = torch.ones(len(hits), dtype=torch.bool, device=Zr.device)
    leads, sizes = [], []
    while True:
        v, j = alive.to(torch.uint8).max(0)
        v, j = torch.stack([v.long(), j]).tolist()
        if not v:
            break
        grp = alive & ((H[:, j] @ H) ** 2 > CLUMP_R2)
        grp[j] = True
        leads.append(j)
        sizes.append(grp.sum())
        alive &= ~grp
    return leads, torch.stack(sizes).tolist()


def p_two_sided(t, df: int):
    """Two-sided p of Student t (float64 on the CPU, log10 to avoid underflow)."""
    import numpy as np
    from scipy import stats

    return stats.t.logsf(np.abs(t), df) / np.log(10) + np.log10(2)


def covariates(covs: list[dict], dev):
    """An item's covariates as columns: a number as itself, a category one-hot less its first level (the
    model-written evals' matching answer: Yes or No, which letter); None without any."""
    import torch

    cols = []
    for key in sorted({k for c in covs for k in c}):
        vs = [c.get(key) for c in covs]
        if all(isinstance(v, (int, float)) for v in vs):
            cols.append(torch.tensor(vs, dtype=torch.float32))
        else:
            for level in sorted({str(v) for v in vs})[1:]:
                cols.append(torch.tensor([float(str(v) == level) for v in vs]))
    cols = [c for c in cols if float(c.std()) > 0]
    return torch.stack(cols, 1).to(dev) if cols else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phenome", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--traits", nargs="*")
    ap.add_argument("--device")
    a = ap.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    layers = list(model.model.layers)
    L, D = len(layers), layers[0].mlp.down_proj.in_features
    phen = json.loads((a.phenome / "phenome.json").read_text())
    kept = [t["name"] for t in phen["traits"] if t["kept"] and (not a.traits or t["name"] in a.traits)]
    vals: dict = {}
    for x in map(json.loads, (a.phenome / "values.jsonl").read_text().splitlines()):
        vals.setdefault(x["trait"], {})[x["item"]] = (x["value"], x["split"], x.get("cov") or {})
    prompts: dict = {}
    for line in (a.phenome / "sequences.jsonl").open():
        s = json.loads(line)
        name, item, form, _ = s["id"].split("|")
        if name in vals and form in ("A", "AB"):
            prompts.setdefault(name, {}).setdefault(item, s["prompt"])
    a.out.mkdir(parents=True, exist_ok=True)
    summ_p = a.out / "summary.json"
    summary = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    tests = sum(bool(t["kept"]) for t in phen["traits"]) * L * D * 4  # (every kept trait, also when --traits picks some)
    alpha = 0.05 / tests
    from scipy import stats

    pool, pending = ThreadPoolExecutor(1), None
    for name in kept:
        if name in summary:
            continue
        t0 = time.time()
        items = [i for i in vals[name] if i in prompts.get(name, {})]
        n = len(items)
        seqs = [prompts[name][i] for i in items]
        y = torch.tensor([vals[name][i][0] for i in items], device=dev, dtype=torch.float32)
        length = torch.tensor([len(s) for s in seqs], device=dev)
        split = np.array([vals[name][i][1] for i in items])
        extra = covariates([vals[name][i][2] for i in items], dev)
        G = genotypes(model, layers, seqs, dev).reshape(n, -1)
        half = {}
        for part in ("dev", "held"):
            ix = torch.tensor(np.flatnonzero(split == part), device=dev)
            t_, _, _, df_, _ = scan(G[ix], y[ix], length[ix], PCS, None if extra is None else extra[ix])
            half[part] = (t_, df_)
        Z, ok = standardized(G)
        del G
        C, var = design(Z, length, PCS, extra)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(15)).to(dev)  # the null: values shuffled over
        yt = rint(y[:, None])[:, 0]  # prompts (the genotypes, covariates and PCs as they are)
        ys = torch.stack([yt, yt[perm]])
        t_lmm, h2 = lmm(Z, ok, C, ys, D * 4)
        Q, _ = torch.linalg.qr(C)
        Z.sub_(Q @ (Q.T @ Z))  # (now residualized)
        zn = Z.norm(dim=0)
        df = n - C.shape[1] - 1
        Yr = residualize(Q, ys.T)
        t, r, eff = ols(Z, zn, Yr[:, 0], ok, df)
        tn = ols(Z, zn, Yr[:, 1], ok, df)[0]
        ld = ld_scores(Z, zn, ok, n - C.shape[1])
        t_thr = float(stats.t.isf(alpha / 2, df))

        def hits_of(tt):
            h = (tt.abs() > t_thr).nonzero()[:, 0]
            return h[tt[h].abs().argsort(descending=True)]

        h_ols, h_lmm = hits_of(t), hits_of(t_lmm[0])
        cl_ols, cl_lmm = clump(Z, zn, h_ols), clump(Z, zn, h_lmm)
        del Z, zn
        cpu = lambda x: x.float().cpu().numpy()  # noqa: E731
        arrays = {"t": cpu(t), "r": cpu(r), "eff": cpu(eff), "t_dev": cpu(half["dev"][0]), "t_held": cpu(half["held"][0]),
                  "t_null": cpu(tn), "t_lmm": cpu(t_lmm[0]), "t_lmm_null": cpu(t_lmm[1]), "ld": cpu(ld),
                  "hits": h_ols.cpu().numpy(), "hits_lmm": h_lmm.cpu().numpy(), "h2": h2.numpy()}
        info = {"n": n, "n_dev": int((split == "dev").sum()), "df": df, "df_half": half["dev"][1], "var": var,
                "clumps": cl_ols, "clumps_lmm": cl_lmm, "gpu_s": time.time() - t0}
        if pending is not None:
            pending.result()  # (one trait at a time on the thread; its errors surface here)
        pending = pool.submit(finish, a.out, summary, name, arrays, info, alpha, L, D)
    if pending is not None:
        pending.result()


def finish(out: Path, summary: dict, name: str, A: dict, info: dict, alpha: float, L: int, D: int) -> None:
    """The CPU part of a trait's stage 1 (on the thread): lambdas, replication, the LD score regressions, the top hits'
    p values; writes OUT/<trait>.npz and OUT/summary.json."""
    import numpy as np
    from scipy import stats

    t0 = time.time()
    df, t, td, th = info["df"], A["t"], A["t_dev"], A["t_held"]

    def lam(tt):  # median chi-square / its null median, through the median |t| (the map from |t| to chi-square is monotone)
        return float(stats.chi2.isf(2 * stats.t.sf(np.median(np.abs(tt)), df), 1) / 0.4549364)

    def chi2(tt):
        return stats.chi2.isf(np.minimum(2 * stats.t.sf(np.abs(tt.astype(np.float64)), df), 1), 1)

    hits, hits_lmm = A["hits"], A["hits_lmm"]
    log_alpha = math.log10(alpha)
    rep_corr = float(np.corrcoef(td, th)[0, 1])
    same_sign = float((np.sign(td[hits]) == np.sign(t[hits])).mean()) if len(hits) else None
    rep_t = float(stats.t.isf(0.025, info["df_half"]))  # replicated: p < 0.05 in each half, both with the all-prompt sign
    sg = np.sign(t[hits])
    replicated = hits[(np.sign(td[hits]) == sg) & (np.sign(th[hits]) == sg) & (np.abs(td[hits]) > rep_t) & (np.abs(th[hits]) > rep_t)]
    n_null = int((p_two_sided(A["t_null"][np.abs(A["t_null"]) > 3], df) < log_alpha).sum())
    n_lmm_null = int((p_two_sided(A["t_lmm_null"][np.abs(A["t_lmm_null"]) > 3], df) < log_alpha).sum())
    lead_pos, sizes = info["clumps"]
    leads = hits[lead_pos] if len(hits) else hits
    leads_lmm = hits_lmm[info["clumps_lmm"][0]] if len(hits_lmm) else hits_lmm
    top = np.argsort(-np.abs(t))[:TOP]
    lp = p_two_sided(t[top].astype(np.float64), df)
    neurons = lambda ix: int(len({int(h) // 4 for h in ix}))  # noqa: E731
    summary[name] = {
        "n": info["n"], "n_dev": info["n_dev"], "df": df, "lambda_gc": lam(t), "bonferroni_log10_alpha": log_alpha,
        "hits": int(len(hits)), "hit_neurons": neurons(hits), "replication_t_corr": rep_corr,
        "replicated_hits": int(len(replicated)), "replicated_neurons": neurons(replicated),
        "null_lambda_gc": lam(A["t_null"]), "null_hits": n_null,
        "null_min_log10_p": float(p_two_sided(np.abs(A["t_null"]).max(keepdims=True).astype(np.float64), df)[0]),
        "hits_same_sign_dev_held": same_sign, "pc_variance": info["var"],
        "lmm_lambda_gc": lam(A["t_lmm"]), "lmm_hits": int(len(hits_lmm)), "lmm_hit_neurons": neurons(hits_lmm),
        "lmm_null_lambda_gc": lam(A["t_lmm_null"]), "lmm_null_hits": n_lmm_null, "lmm_h2": float(A["h2"][0].mean()),
        "lmm_null_h2": float(A["h2"][1].mean()),
        "ldsc": ldsc(chi2(t), A["ld"]), "ldsc_lmm": ldsc(chi2(A["t_lmm"]), A["ld"]),
        "ld_score_median": float(np.nanmedian(A["ld"])),
        "clumps": int(len(leads)), "clump_neurons": neurons(leads), "clumps_lmm": int(len(leads_lmm)),
        "clump_largest": int(max(sizes)) if sizes else 0,
        "top": [{"layer": int(f // 4 // D), "neuron": int(f // 4 % D), "measure": MEASURES[int(f % 4)], "r": float(A["r"][f]),
                 "effect": float(A["eff"][f]), "log10_p": float(lp[k]), "t_dev": float(td[f]), "t_held": float(th[f])}
                for k, f in enumerate(top)],
        "top_clumps": [{"layer": int(f // 4 // D), "neuron": int(f // 4 % D), "measure": MEASURES[int(f % 4)], "size": int(sz),
                        "t": float(t[f])} for f, sz in list(zip(leads, sizes))[:TOP]]}
    f16 = lambda x: x.astype(np.float16).reshape(L, D, 4)  # noqa: E731
    np.savez_compressed(out / f"{name.replace('/', '__')}.npz", t=f16(t), r=f16(A["r"]), t_dev=f16(td), t_held=f16(th),
                        t_lmm=f16(A["t_lmm"]), ld=f16(A["ld"]), lmm_h2=A["h2"][0], clump_leads=leads.astype(np.int32),
                        clump_leads_lmm=leads_lmm.astype(np.int32), df=df, df_half=info["df_half"], pc_variance=np.array(info["var"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    S_ = summary[name]
    print(f"{name}: n {info['n']}, lambda {S_['lambda_gc']:.2f} (null {S_['null_lambda_gc']:.2f}), {len(hits)} Bonferroni hits on "
          f"{S_['hit_neurons']} neurons (null {n_null}), {len(replicated)} replicated, {len(leads)} clumps; mixed model: lambda "
          f"{S_['lmm_lambda_gc']:.2f}, {len(hits_lmm)} hits (null {n_lmm_null}), h2 {S_['lmm_h2']:.2f}; LD score intercept "
          f"{S_['ldsc']['intercept']:.2f} +- {S_['ldsc']['intercept_se']:.2f} (mean chi2 {S_['ldsc']['mean_chi2']:.2f}); "
          f"GPU {info['gpu_s']:.0f} s, CPU {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
