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
Outputs: OUT/<trait>.npz (t on all prompts and the partial r, [layers, neurons, 4] float16: max+, max-, ans+, ans-;
degrees of freedom; the PCs' variance shares), OUT/summary.json (per trait: n, lambda, hits, replication, the top TOP
hits with r, effect, p and their dev and held-out t).

  assoc.py PHENOME_DIR OUT [--model Qwen/Qwen3-8B] [--traits NAME ...]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

PCS = 10
ROWS = 16  # prompts a batch, every batch padded to it
RINT_COLS = 1 << 17  # genotype columns ranked at once
TOP = 50
MEASURES = ("max+", "max-", "ans+", "ans-")


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


def scan(G, y, length, pcs: int, extra=None):
    """t and partial r of y on each genotype column given an intercept, log length, extra covariates [n, k] and the
    top pcs PCs of G; genotype columns and y rank-based inverse normal transformed (rint). G [n, F] float16, y [n],
    length [n] -> t [F], r [F], effect [F] (y's transformed units per genotype sd), pc variance shares."""
    import torch

    n = G.shape[0]
    Gf = torch.cat([rint(G[:, s:s + RINT_COLS]) for s in range(0, G.shape[1], RINT_COLS)], 1)  # (in column chunks: memory)
    y = rint(y[:, None])[:, 0]
    mu, sd = Gf.mean(0), Gf.std(0)
    ok = sd > 1e-6
    Z = (Gf - mu) / torch.where(ok, sd, 1.0)
    Z[:, ~ok] = 0
    U, S, _ = torch.svd_lowrank(Z, q=pcs + 10, niter=4)
    C = torch.cat([torch.ones(n, 1, device=G.device), torch.log(length.float())[:, None]] + ([] if extra is None else [extra])
                  + [U[:, :pcs]], 1)  # covariates
    Q, _ = torch.linalg.qr(C)
    yr = y - Q @ (Q.T @ y)
    Zr = Z - Q @ (Q.T @ Z)
    num = (Zr * yr[:, None]).sum(0)
    den = Zr.norm(dim=0) * yr.norm()
    r = torch.where(ok & (den > 0), num / den.clamp_min(1e-12), 0)
    df = n - C.shape[1] - 1
    t = r * math.sqrt(df) / (1 - r ** 2).clamp_min(1e-12).sqrt()
    effect = num / Zr.pow(2).sum(0).clamp_min(1e-12)  # y per genotype sd (the partial regression slope)
    var = (S ** 2 / Z.pow(2).sum()).tolist()[:pcs]
    return t, r, effect, df, var


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
    tests = len(kept) * L * D * 4
    alpha = 0.05 / tests
    for name in kept:
        if name in summary:
            continue
        t0 = time.time()
        items = [i for i in vals[name] if i in prompts.get(name, {})]
        seqs = [prompts[name][i] for i in items]
        y = torch.tensor([vals[name][i][0] for i in items], device=dev, dtype=torch.float32)
        length = torch.tensor([len(s) for s in seqs], device=dev)
        split = np.array([vals[name][i][1] for i in items])
        extra = covariates([vals[name][i][2] for i in items], dev)
        G = genotypes(model, layers, seqs, dev).reshape(len(items), -1)
        res = {}
        for part, sel in (("all", np.ones(len(items), bool)), ("dev", split == "dev"), ("held", split == "held")):
            ix = torch.tensor(np.flatnonzero(sel), device=dev)
            t, r, eff, df, var = scan(G[ix], y[ix], length[ix], PCS, None if extra is None else extra[ix])
            res[part] = (t, r, eff, df, var)
        perm = torch.randperm(len(items), generator=torch.Generator().manual_seed(15)).to(dev)  # the null: values shuffled
        tn = scan(G, y[perm], length, PCS, extra)[0]  # over prompts (the genotypes, covariates and PCs as they are)
        t, r, eff, df, var = res["all"]
        lp = p_two_sided(t.double().cpu().numpy(), df)  # log10 p
        from scipy import stats

        lam = float(np.median(stats.chi2.isf(np.clip(10 ** lp, 1e-300, 1), 1)) / 0.4549364)
        hits = np.flatnonzero(lp < math.log10(alpha))
        lpn = p_two_sided(tn.double().cpu().numpy(), df)
        lam_null = float(np.median(stats.chi2.isf(np.clip(10 ** lpn, 1e-300, 1), 1)) / 0.4549364)
        td, th = res["dev"][0].cpu().numpy(), res["held"][0].cpu().numpy()
        rep_corr = float(np.corrcoef(td, th)[0, 1])
        same_sign = float((np.sign(td[hits]) == np.sign(th[hits])).mean()) if len(hits) else None
        rep_t = float(stats.t.isf(0.025, res["dev"][3]))  # replicated: p < 0.05 in each half, both with the all-prompt sign
        sg = np.sign(t.cpu().numpy())
        replicated = hits[(np.sign(td[hits]) == sg[hits]) & (np.sign(th[hits]) == sg[hits]) & (np.abs(td[hits]) > rep_t) & (np.abs(th[hits]) > rep_t)]
        top = np.argsort(lp)[:TOP]
        summary[name] = {
            "n": len(items), "n_dev": int((split == "dev").sum()), "df": df, "lambda_gc": lam, "bonferroni_log10_alpha": math.log10(alpha),
            "hits": int(len(hits)), "hit_neurons": int(len({int(h) // 4 for h in hits})), "replication_t_corr": rep_corr,
            "replicated_hits": int(len(replicated)), "replicated_neurons": int(len({int(h) // 4 for h in replicated})),
            "null_lambda_gc": lam_null, "null_hits": int((lpn < math.log10(alpha)).sum()), "null_min_log10_p": float(lpn.min()),
            "hits_same_sign_dev_held": same_sign, "pc_variance": var,
            "top": [{"layer": int(f // 4 // D), "neuron": int(f // 4 % D), "measure": MEASURES[int(f % 4)], "r": float(r[f]),
                     "effect": float(eff[f]), "log10_p": float(lp[f]), "t_dev": float(td[f]), "t_held": float(th[f])} for f in top]}
        np.savez_compressed(a.out / f"{name.replace('/', '__')}.npz", t=t.half().cpu().numpy().reshape(L, D, 4),
                            r=r.half().cpu().numpy().reshape(L, D, 4), t_dev=td.astype(np.float16).reshape(L, D, 4),
                            t_held=th.astype(np.float16).reshape(L, D, 4), df=df, df_half=res["dev"][3], pc_variance=np.array(var))
        summ_p.write_text(json.dumps(summary, indent=1))
        print(f"{name}: n {len(items)}, lambda {lam:.2f} (null {lam_null:.2f}), {len(hits)} Bonferroni hits on "
              f"{summary[name]['hit_neurons']} neurons (null {summary[name]['null_hits']}), {len(replicated)} replicated in both halves, "
              f"dev-held t corr {rep_corr:.2f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        del G, res


if __name__ == "__main__":
    main()
