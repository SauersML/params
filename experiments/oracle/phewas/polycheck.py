"""Knockouts check the polyneuronal score's split (#2951, the lead's 10-15 spec: "each prompt's score splits into its
poles' contributions, and knockouts check that split"). Per trait: polyscore.py's sparse score refitted exactly (the
same candidates, features, label and cross-validated L1 penalty on the dev prompts), then each of its POLES heaviest
poles (summed |weight| over the pole's two measures) knocked out alone (claim_check.knockout_edit: the pole's values at
every position from 1 on set to 0, massive values live), and on every held-out prompt:
  dv        the trait's change, form A (phenome.py's value of its form-A sequences, in the trait's units);
  ds_split  the change of the score's logit that its split predicts: the pole's own features going to 0, the rest as
            they were (-sum over its measures of weight x value / sd);
  ds_full   the change of the score's logit with every feature read again under the knockout (downstream poles move too).
The split is checked when dv follows ds_split: per trait the Spearman correlation over (pole, prompt) pairs, per pole the
sign of the mean dv against the sign of the mean ds_split. Batches are fixed (ROWS sequences, the same in every pass), so
a change is the knockout's alone.

  polycheck.py PHENOME CANDIDATES OUT.json --thr THR [--traits T ...] [--poles 10] [--model Qwen/Qwen3-8B]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import claim_check as cc  # noqa: E402
import neuron_builder as nb  # noqa: E402
import polyscore as ps  # noqa: E402

ROWS = 16
POLE_MEASURES = {1: (0, 2), -1: (1, 3)}  # a pole's measures among max+, max-, ans+, ans-


class Trait:
    """A pair trait's form-A sequences in fixed batches; run(edit) -> (value per item, features per item [n, C, 4])."""

    def __init__(self, nr, seqs: list[dict], cand: list[tuple]):
        import torch

        self.nr, self.torch = nr, torch
        self.items = sorted({s["id"].split("|")[1] for s in seqs})
        self.ix = {i: k for k, i in enumerate(self.items)}
        self.seqs = sorted(seqs, key=lambda s: len(s["prompt"]) + len(s["cont"]))
        self.cand = cand
        self.by_layer: dict = {}
        for j, (l_, n_) in enumerate(cand):
            self.by_layer.setdefault(l_, []).append((j, n_))
        self.batches = []
        for k in range(0, len(self.seqs), ROWS):
            part = self.seqs[k:k + ROWS]
            b = len(part)
            part = part + part[-1:] * (ROWS - b)
            T = max(len(s["prompt"]) + len(s["cont"]) for s in part)
            x = torch.zeros(ROWS, T, dtype=torch.long)
            m = torch.zeros(ROWS, T, dtype=torch.long)
            for j, s in enumerate(part):
                q = s["prompt"] + s["cont"]
                x[j, :len(q)] = torch.tensor(q)
                m[j, :len(q)] = 1
            P = torch.tensor([len(s["prompt"]) for s in part])
            self.batches.append((x.to(nr.dev), m.to(nr.dev), P.to(nr.dev), part, b))

    def run(self, edit=None, route=None):
        torch, nr = self.torch, self.nr
        n, C = len(self.items), len(self.cand)
        val = np.zeros(n)
        F = np.zeros((n, C, 4), np.float32)
        with torch.no_grad(), nr.hooks(edit=edit, route=route):
            for x, m, P, part, b in self.batches:
                pos_ = torch.arange(x.shape[1], device=x.device)[None]
                live = (pos_ >= 1) & (pos_ < P[:, None])  # the prompt's positions from 1 on
                last = P - 1
                got = torch.zeros(ROWS, C, 4, device=x.device)
                hooks = []
                for l_, js in self.by_layer.items():
                    def fn(_m, args, js=js):
                        h = args[0].float()
                        hv = h[..., torch.tensor([n_ for _, n_ in js], device=h.device)]
                        a = hv[torch.arange(ROWS, device=h.device), last]
                        got[:, [j for j, _ in js]] = torch.stack(
                            [torch.where(live[..., None], hv.clamp_min(0), 0).amax(1),
                             torch.where(live[..., None], (-hv).clamp_min(0), 0).amax(1), a.clamp_min(0), (-a).clamp_min(0)], -1)
                    hooks.append(nr.layers[l_].mlp.down_proj.register_forward_pre_hook(fn))  # (after the edit's hook)
                try:
                    h = nr.core.model(input_ids=x, attention_mask=m).last_hidden_state
                finally:
                    for hk in hooks:
                        hk.remove()
                got = got.cpu().numpy()
                for j, s in enumerate(part[:b]):
                    _, item, _, role = s["id"].split("|")
                    p_, c_ = len(s["prompt"]), len(s["cont"])
                    lp = torch.log_softmax(nr.core.lm_head(h[j, p_ - 1:p_ + c_ - 1]).float(), -1)
                    v = float(lp[torch.arange(c_), torch.tensor(s["cont"], device=lp.device)].sum())
                    k = self.ix[item]
                    val[k] += v if role == "pos" else -v
                    if role == "pos":
                        F[k] = got[j]
        return val, F


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phenome", type=Path)
    ap.add_argument("candidates", type=Path, help="polyscore.candidates()'s file")
    ap.add_argument("out", type=Path)
    ap.add_argument("--thr", type=Path, required=True)
    ap.add_argument("--traits", nargs="*")
    ap.add_argument("--poles", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device")
    a = ap.parse_args()
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None
    phen = {t["name"]: t for t in json.loads((a.phenome / "phenome.json").read_text())["traits"]}
    cands = json.loads(a.candidates.read_text())
    vals: dict = {}
    for x in map(json.loads, (a.phenome / "values.jsonl").read_text().splitlines()):
        vals.setdefault(x["trait"], {})[x["item"]] = (x["value"], x["split"])
    traits = [t for t in (a.traits or sorted(cands)) if t in vals and phen[t]["kind"] == "pair"]
    seqs: dict = {}
    for line in (a.phenome / "sequences.jsonl").open():
        s = json.loads(line)
        name, _, form, _ = s["id"].split("|")
        if name in traits and form == "A":
            seqs.setdefault(name, []).append(s)
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    for t in traits:
        if t in out:
            continue
        t0 = time.time()
        cand = [tuple(u) for u in cands[t]]
        tr = Trait(nr, seqs[t], cand)
        v0, F0 = tr.run()
        split = np.array([vals[t][i][1] for i in tr.items])
        target = np.array([vals[t][i][0] for i in tr.items])
        dv_, hv_ = split == "dev", split == "held"
        y = torch.tensor((target > np.median(target[dv_])).astype(np.float32), device=dev)
        Xall = torch.tensor(F0.reshape(len(tr.items), -1), device=dev)
        d_ix, h_ix = torch.tensor(np.flatnonzero(dv_), device=dev), torch.tensor(np.flatnonzero(hv_), device=dev)
        mu, sd = Xall[d_ix].mean(0), Xall[d_ix].std(0).clamp_min(1e-6)
        lam, cv = ps.cv_pick(ps.logistic_l1, (Xall - mu)[d_ix] / sd, y[d_ix], ps.L1_PATH)
        w, b = ps.logistic_l1(((Xall - mu) / sd)[d_ix], y[d_ix], lam)
        held_auc = ps.auroc((((Xall - mu) / sd)[h_ix] @ w + b).cpu().numpy(), y[h_ix].cpu().numpy())
        W = w.reshape(len(cand), 4).cpu().numpy()
        sdn, mun = sd.reshape(len(cand), 4).cpu().numpy(), mu.reshape(len(cand), 4).cpu().numpy()
        weight = [(abs(W[j, list(ms)]).sum(), j, pl) for j in range(len(cand)) for pl, ms in POLE_MEASURES.items()]
        top = [(j, pl) for wt, j, pl in sorted(weight, reverse=True)[:a.poles] if wt > 0]

        def score(F):
            return ((F - mun) / sdn * W).sum((1, 2)) + float(b)

        s0 = score(F0)
        hi = np.flatnonzero(hv_)
        rows = []
        for j, pl in top:
            l_, n_ = cand[j]
            e, r = cc.knockout_edit(nr, [("N", int(l_), int(n_), int(pl), None)])
            v1, F1 = tr.run(e, r)
            ms = list(POLE_MEASURES[pl])
            ds_split = -(F0[:, j, ms] / sdn[j, ms] * W[j, ms]).sum(1)
            ds_full = score(F1) - s0
            dv = v1 - v0
            rows.append({"pole": f"L{l_}.N{n_}{'+' if pl > 0 else '-'}", "weight": [float(x) for x in W[j, ms]],
                         "dv": dv[hi].tolist(), "ds_split": ds_split[hi].tolist(), "ds_full": ds_full[hi].tolist(),
                         "mean_dv": float(dv[hi].mean()), "mean_ds_split": float(ds_split[hi].mean()),
                         "mean_ds_full": float(ds_full[hi].mean())})
        DV = np.concatenate([r_["dv"] for r_ in rows]) if rows else np.zeros(0)
        DS = np.concatenate([r_["ds_split"] for r_ in rows]) if rows else np.zeros(0)
        DF = np.concatenate([r_["ds_full"] for r_ in rows]) if rows else np.zeros(0)
        moved = [r_ for r_ in rows if abs(r_["mean_ds_split"]) > 0]
        out[t] = {"lambda": lam, "cv_auroc": cv, "held_auroc": held_auc, "poles_used": int((abs(W).sum(1) > 0).sum()),
                  "rho_split": float(spearmanr(DS, DV)[0]) if len(DV) > 2 else None,
                  "rho_full": float(spearmanr(DF, DV)[0]) if len(DV) > 2 else None,
                  "sign_agree": float(np.mean([np.sign(r_["mean_dv"]) == np.sign(r_["mean_ds_split"]) for r_ in moved])) if moved else None,
                  "mean_abs_dv": float(np.abs(DV).mean()) if len(DV) else None, "poles": rows}
        a.out.write_text(json.dumps(out))
        o = out[t]
        print(f"{t}: score held-out AUROC {held_auc:.3f} ({o['poles_used']} poles); {len(rows)} knockouts: Spearman(dv, split) "
              f"{o['rho_split']:+.2f}, (dv, full) {o['rho_full']:+.2f}, mean-sign agreement {o['sign_agree']}, mean |dv| "
              f"{o['mean_abs_dv']:.3f}; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
