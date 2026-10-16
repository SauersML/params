"""Polyneuronal scores of the neuron PheWAS (#2951, the lead's 10-15 spec): per trait a sparse weighted sum of named
neuron poles, fitted on the dev prompts, against a dense probe of the residual stream on the same prompts; both scored
by AUROC on the held-out prompts. The sparse score splits each prompt's prediction into its poles' contributions, the
local explanation the oracle cites.

Label: the trait's value above its dev median (the behaviour stronger than usual on that prompt). Features, per prompt
(the trait's form-A prompt): for the score, the candidate poles' four measures (assoc.py's: the pole's largest value
over the prompt, its value at the answer position); candidates: the trait's stage-1 top hits (assoc.py summary.json)
and its fine-mapped single poles (effects.py finemap, when given); for the probe, the residual stream at the answer
position after layers PROBE_LAYERS. Both standardized on dev; the score is L1-penalized logistic regression (proximal
gradient), the probe L2-penalized logistic regression (L-BFGS), each penalty chosen by 5-fold cross-validation on dev.

  polyscore.py PHENOME ASSOC_SUMMARY OUT.json [--finemap FINEMAP.json] [--traits T ...] [--model Qwen/Qwen3-8B]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

FOLDS = 5
L1_PATH = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
L2_PATH = [10.0, 1.0, 0.1, 0.01, 0.001]
PROBE_LAYERS = (18, 27)
ROWS = 16


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def logistic_l1(X, y, lam: float, iters: int = 400):
    """L1-penalized logistic regression by proximal gradient (FISTA) on torch tensors; returns (w, b)."""
    import torch

    n, d = X.shape
    w = torch.zeros(d, device=X.device)
    b = torch.zeros((), device=X.device)
    zw, zb, t = w.clone(), b.clone(), 1.0
    step = 4.0 / (float((X ** 2).sum(1).max()) + 1)
    for _ in range(iters):
        p = torch.sigmoid(X @ zw + zb)
        gw, gb = X.T @ (p - y) / n, (p - y).mean()
        w1 = zw - step * gw
        w1 = torch.sign(w1) * (w1.abs() - step * lam).clamp_min(0)
        b1 = zb - step * gb
        t1 = (1 + math.sqrt(1 + 4 * t * t)) / 2
        zw, zb = w1 + (t - 1) / t1 * (w1 - w), b1 + (t - 1) / t1 * (b1 - b)
        w, b, t = w1, b1, t1
    return w, b


def logistic_l2(X, y, lam: float):
    import torch

    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros((), device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + lam * (w ** 2).sum() / 2
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def cv_pick(fit, X, y, path, seed=15):
    """The penalty on path with the best mean held-out AUROC over FOLDS folds of (X, y), and that AUROC."""
    import torch

    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, FOLDS)
    best = None
    for lam in path:
        aucs = []
        for k in range(FOLDS):
            te = torch.tensor(folds[k], device=X.device)
            tr = torch.tensor(np.concatenate([folds[j] for j in range(FOLDS) if j != k]), device=X.device)
            w, b = fit(X[tr], y[tr], lam)
            yt = y[te].cpu().numpy()
            if 0 < yt.sum() < len(yt):
                aucs.append(auroc((X[te] @ w + b).cpu().numpy(), yt))
        m = float(np.mean(aucs)) if aucs else 0.0
        if best is None or m > best[1]:
            best = (lam, m)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phenome", type=Path)
    ap.add_argument("assoc", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--finemap", type=Path)
    ap.add_argument("--traits", nargs="*")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--device")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    layers = list(model.model.layers)
    summ = json.loads(a.assoc.read_text())
    fine = json.loads(a.finemap.read_text())["traits"] if a.finemap else {}
    vals: dict = {}
    for x in map(json.loads, (a.phenome / "values.jsonl").read_text().splitlines()):
        vals.setdefault(x["trait"], {})[x["item"]] = (x["value"], x["split"])
    traits = [t for t in (a.traits or sorted(summ)) if t in vals]
    prompts: dict = {}
    for line in (a.phenome / "sequences.jsonl").open():
        s = json.loads(line)
        name, item, form, _ = s["id"].split("|")
        if name in traits and form in ("A", "AB"):
            prompts.setdefault(name, {}).setdefault(item, s["prompt"])
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    for t in traits:
        if t in out:
            continue
        t0 = time.time()
        cand = []
        for h in summ[t]["top"][:a.candidates]:
            u = (h["layer"], h["neuron"])
            if u not in cand:
                cand.append(u)
        for r in fine.get(t, []):
            if "pole" in r:
                l_, n_ = r["pole"][1:-1].split(".N")
                if (int(l_), int(n_)) not in cand:
                    cand.append((int(l_), int(n_)))
        items = [i for i in vals[t] if i in prompts.get(t, {})]
        seqs = [prompts[t][i] for i in items]
        F = torch.zeros(len(items), len(cand), 4, device=dev)
        R = torch.zeros(len(items), len(PROBE_LAYERS), model.config.hidden_size, device=dev)
        by_layer: dict = {}
        for j, (l_, n_) in enumerate(cand):
            by_layer.setdefault(l_, []).append((j, n_))
        order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
        for s0 in range(0, len(order), ROWS):
            idx = order[s0:s0 + ROWS]
            idx = idx + idx[-1:] * (ROWS - len(idx))
            T = max(len(seqs[i]) for i in idx)
            x = torch.zeros(ROWS, T, dtype=torch.long, device=dev)
            m = torch.zeros(ROWS, T, dtype=torch.bool, device=dev)
            for j, i in enumerate(idx):
                x[j, :len(seqs[i])] = torch.tensor(seqs[i], device=dev)
                m[j, :len(seqs[i])] = True
            live = m.clone()
            live[:, 0] = False
            last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=dev)
            rows = torch.tensor(idx, device=dev)
            hooks = []
            for l_, js in by_layer.items():
                def fn(_m, args, js=js):
                    h = args[0].float()
                    cols = torch.tensor([n_ for _, n_ in js], device=dev)
                    hv = h[..., cols]  # [B, T, k]
                    pos = torch.where(live[..., None], hv.clamp_min(0), 0).amax(1)
                    neg = torch.where(live[..., None], (-hv).clamp_min(0), 0).amax(1)
                    av = hv[torch.arange(ROWS, device=dev), last]
                    F[rows[:, None], torch.tensor([j for j, _ in js], device=dev)[None]] = torch.stack(
                        [pos, neg, av.clamp_min(0), (-av).clamp_min(0)], -1)
                hooks.append(layers[l_].mlp.down_proj.register_forward_pre_hook(fn))
            for k, l_ in enumerate(PROBE_LAYERS):
                def rf(_m, _i, o, k=k):
                    hh = o[0] if isinstance(o, tuple) else o
                    R[rows, k] = hh[torch.arange(ROWS, device=dev), last].float()
                hooks.append(layers[l_].register_forward_hook(rf))
            try:
                with torch.no_grad():
                    model.model(input_ids=x, attention_mask=m.long())
            finally:
                for h in hooks:
                    h.remove()
        split = np.array([vals[t][i][1] for i in items])
        v = np.array([vals[t][i][0] for i in items])
        dv, hv = split == "dev", split == "held"
        y = torch.tensor((v > np.median(v[dv])).astype(np.float32), device=dev)
        res = {"n_dev": int(dv.sum()), "n_held": int(hv.sum()), "candidates": len(cand)}
        for name, Xall, fit, path in (("score", F.reshape(len(items), -1), logistic_l1, L1_PATH),
                                      ("probe", R.reshape(len(items), -1), logistic_l2, L2_PATH)):
            d_ix, h_ix = torch.tensor(np.flatnonzero(dv), device=dev), torch.tensor(np.flatnonzero(hv), device=dev)
            mu, sd = Xall[d_ix].mean(0), Xall[d_ix].std(0).clamp_min(1e-6)
            X = (Xall - mu) / sd
            lam, cv = cv_pick(fit, X[d_ix], y[d_ix], path)
            w, b = fit(X[d_ix], y[d_ix], lam)
            res[name] = {"lambda": lam, "cv_auroc": cv, "held_auroc": auroc((X[h_ix] @ w + b).cpu().numpy(), y[h_ix].cpu().numpy())}
            if name == "score":
                W = w.reshape(len(cand), 4).cpu().numpy()
                used = [(cand[j], MEAS[k], float(W[j, k])) for j in range(len(cand)) for k in range(4) if W[j, k] != 0]
                res[name]["poles_used"] = len({u for u, _, _ in used})
                res[name]["weights"] = [{"layer": u[0], "neuron": u[1], "measure": mm, "weight": ww}
                                        for u, mm, ww in sorted(used, key=lambda z: -abs(z[2]))[:40]]
        out[t] = res
        a.out.write_text(json.dumps(out, indent=1))
        print(f"{t}: sparse score held-out AUROC {res['score']['held_auroc']:.3f} with {res['score']['poles_used']} poles, "
              f"dense probe {res['probe']['held_auroc']:.3f}, {time.time() - t0:.0f} s", flush=True)


MEAS = ("max+", "max-", "ans+", "ans-")

if __name__ == "__main__":
    main()
