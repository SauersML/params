"""Learned masks over VPD subcomponents as variable carriers (#2951 graph oracle, E4 v2; desiderata-based masking,
Davies et al. 2023, multi-task as RAVEL's MDBM, annealed boundaries as Boundless DAS): is a variable localized in VPD's
units, and how small can its carrier be?

For a variable on a text (input positions I) the candidates are the family at I (VPD's active subcomponents on the
text and on the 64 alternative tokens per position, var_classes.alternatives) over every site. A mask m in [0, 1] per
candidate, m = sigmoid(l / tau), tau annealed from 1 to 0.05, sets each candidate's activation on the text to a +
m (a' - a), a' its activation on a donor x' (the real model in float32, the rest of every site unchanged, gradients
through every later layer). Loss per donor: Cause, KL(model(x') || patched) for a donor with another token at an input
position (A: the 64 alternatives); Iso, KL(model(x) || patched) for a donor changing a token before the carrier
outside I (a change after it cannot reach the carrier: the model is causal); plus lam x the mean of m. The mask is
then thresholded at 1/2 and verified exactly (variables.Scorer's patched runs) on donors disjoint from A (B: tokens
sampled from the model's prediction, A and the text's own left out): f on-target and, with Iso donors, f
off-target. Controls (CausalGym's selectivity, the Hewitt and Liang control task): the same training with the donors'
target distributions permuted among them (a mapping the model does not compute), scored on B against the permuted
targets; and equal-size random subsets of the candidates. A mask counts by its selectivity, f real minus f permuted.

  experiments/oracle/var_masks.py text6011 232,233 [--copy N] [--lam 1e-3,1e-2] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import var_classes  # noqa: E402
import var_controls  # noqa: E402
import var_teacher  # noqa: E402

STEPS = 300
BATCH = 8


class Patcher:
    """The target model on the text with candidates' activations interpolated toward a donor's: forward hooks on the
    candidate sites add (m (a' - a)) U at the candidate positions."""

    def __init__(self, nat, ids: list[int], targets: list[int], cands: list[tuple]):
        self.nat, self.tg, self.dev = nat, nat.target, nat.dev
        self.ids, self.targets = list(ids), list(targets)
        self.by = {}
        for j, (n, p, c) in enumerate(cands):
            self.by.setdefault(n, []).append((j, p, c))
        self.idx = {n: torch.tensor([j for j, _, _ in v], device=self.dev) for n, v in self.by.items()}
        self.pos = {n: torch.tensor([p for _, p, _ in v], device=self.dev) for n, v in self.by.items()}
        self.col = {n: torch.tensor([c for _, _, c in v], device=self.dev) for n, v in self.by.items()}
        self.K = len(cands)

    def acts(self, texts: list[list[int]]) -> torch.Tensor:
        """Every candidate's activation (v . x) on each text [len(texts), K] (the model's own run)."""
        out = torch.zeros(len(texts), self.K, device=self.dev)
        for n in self.by:
            self.tg.site(n).cache_input = True
        with torch.no_grad():
            for s0 in range(0, len(texts), BATCH):
                part = texts[s0:s0 + BATCH]
                self.tg(torch.tensor(part, device=self.dev))
                for n in self.by:
                    st = self.tg.site(n)
                    x = st.last_input[:, self.pos[n]]  # [b, k, d_in]
                    out[s0:s0 + len(part)][:, self.idx[n]] = (x * st.V[:, self.col[n]].T[None]).sum(-1)
        for n in self.by:
            self.tg.site(n).cache_input = False
            self.tg.site(n).last_input = None
        return out

    def logq(self, m: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """log q at the targets [B, targets, V] on the text with each row's candidates moved by m toward src [B, K]."""
        B = src.shape[0]
        hooks = []

        def hook(n):
            st = self.tg.site(n)
            V, U = st.V[:, self.col[n]], st.U[self.col[n]]  # [d_in, k], [k, d_out]

            def f(mod, inp, out):
                x = inp[0][:, self.pos[n]]  # [B, k, d_in]
                a = (x * V.T[None]).sum(-1)  # [B, k]
                d = (m[self.idx[n]][None] * (src[:, self.idx[n]] - a))[..., None] * U[None]  # [B, k, d_out]
                out = out.clone()
                out.index_add_(1, self.pos[n], d)
                return out
            return st.register_forward_hook(f)
        hooks = [hook(n) for n in self.by]
        try:
            logits = self.tg.hidden(torch.tensor([self.ids] * B, device=self.dev))[:, self.targets] @ self.tg.wte.T
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(logits.float(), -1)


def train(P: Patcher, src: torch.Tensor, target: torch.Tensor, lam: float, steps: int = STEPS, seed: int = 0) -> torch.Tensor:
    """The mask's logits after Adam on mean KL(target || patched) + lam mean(m), tau annealed 1 -> 0.05."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logit = torch.zeros(P.K, device=P.dev, requires_grad=True)
    opt = torch.optim.Adam([logit], lr=0.1)
    for step in range(steps):
        tau = math.exp(math.log(1.0) + (math.log(0.05) - math.log(1.0)) * step / max(steps - 1, 1))
        rows = torch.randperm(src.shape[0], generator=gen)[:BATCH].to(P.dev)
        m = torch.sigmoid(logit / tau)
        lq = P.logq(m, src[rows])
        loss = carry.kl(target[rows], lq).mean() + lam * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return logit.detach()


def run_variable(ge, sc, tx, task_id: str, ins: tuple, lams: list[float], log=print) -> dict:
    nat = ge.nat
    reach = nat.reachable(tx.T, tx.targets)
    every = [n for n in nat.names if all(bool(reach[n][q].any()) for q in ins)]
    fam, _ = var_teacher.family(nat, tx, ins, every, ins)
    A = {q: var_classes.alternatives(nat, tx.ids, q, int(tx.seed)) for q in ins}
    B = {q: var_controls.b_tokens(nat, tx, q, A[q], max(4, var_controls.B_PER // len(ins)), int(tx.seed)) for q in ins}
    texts_A, pzA = [], []
    for q in ins:
        for z in A[q]:
            x = list(tx.ids)
            x[q] = z
            texts_A.append(x)
            pzA.append((q, z))
    iso_pos = [q for q, _ in tx.f2 if q < min(ins)]  # changes that can reach the carrier without being its inputs
    iso_A = {q: var_classes.alternatives(nat, tx.ids, q, int(tx.seed), top=8, sampled=8) for q in iso_pos}
    iso_B = {q: var_controls.b_tokens(nat, tx, q, iso_A[q], 4, int(tx.seed)) for q in iso_pos}
    texts_I = []
    for q in iso_pos:
        for z in iso_A[q]:
            x = list(tx.ids)
            x[q] = z
            texts_I.append(x)
    P = Patcher(nat, tx.ids, tx.targets, fam)
    srcA = P.acts(texts_A + texts_I)
    with torch.no_grad():
        tgtA = torch.cat([nat.reference(texts_A[s:s + BATCH], tx.targets) for s in range(0, len(texts_A), BATCH)])
        base = nat.reference([list(tx.ids)], tx.targets)
    tgtA = torch.cat([tgtA, base.expand(len(texts_I), -1, -1)]) if texts_I else tgtA  # Iso: the prediction stays
    R = var_controls.Runs(sc, tx)
    pzB = [(q, z) for q in ins for z in B[q]]
    pzI = [(q, z) for q in iso_pos for z in iso_B[q]]

    def f_off(group):  # Iso on B-like donors before the carrier: 1 - KL(model(x) || patched) / KL(model(x) || model(x'))
        if not pzI:
            return None
        num = den = 0.0
        for (q, z), qq in zip(pzI, R.patched(group, pzI)):
            d0 = var_controls.kl(R.base, R.donor(q, z)["lq"])
            if d0 > 1e-6:
                num += d0 * min(1.0, max(0.0, 1.0 - max(var_controls.kl(R.base, qq), 0.0) / d0))
                den += d0
        return num / den if den > 0 else None
    out = {"task": task_id, "inputs": list(ins), "candidates": len(fam), "A": len(pzA), "iso_train": len(texts_I), "B": len(pzB),
           "iso_B": len(pzI), "family_f_B": R.f_on(fam, pzB), "family_iso_B": f_off(fam), "masks": []}
    perm = torch.randperm(len(pzA), generator=torch.Generator().manual_seed(int(tx.seed))).to(nat.dev)
    perm = torch.cat([perm, torch.arange(len(pzA), len(pzA) + len(texts_I), device=nat.dev)])  # Iso targets stay
    for lam in lams:
        for kind, tgt in (("real", tgtA), ("permuted", tgtA[perm])):
            logit = train(P, srcA, tgt, lam)
            keep = [fam[j] for j in (logit > 0).nonzero().flatten().tolist()]
            if kind == "real":
                f = R.f_on(keep, pzB)
            else:  # scored against the permuted targets: each B donor's target swapped with another B donor's
                q = R.patched(keep, pzB)
                refs = [R.donor(p, z)["lq"] for p, z in pzB]
                order = np.random.default_rng(int(tx.seed)).permutation(len(pzB))
                num = den = 0.0
                for i, qq in enumerate(q):
                    ref = refs[order[i]]
                    d0 = var_controls.kl(ref, R.base)
                    if d0 > 1e-6:
                        num += d0 * min(1.0, max(0.0, 1.0 - max(var_controls.kl(ref, qq), 0.0) / d0))
                        den += d0
                f = num / den if den > 0 else None
            rnd = []
            rng = np.random.default_rng(int(tx.seed) + len(keep))
            for _ in range(8):
                pick = [fam[i] for i in rng.choice(len(fam), size=len(keep), replace=False)] if keep else []
                rnd.append(R.f_on(pick, pzB))
            row = {"lam": lam, "targets": kind, "size": len(keep), "f_B": f, "iso_B": f_off(keep) if kind == "real" else None,
                   "random_same_size": float(np.mean([x for x in rnd if x is not None])) if any(x is not None for x in rnd) else None}
            out["masks"].append(row)
            log(f"  lam {lam} {kind}: {len(keep)} of {len(fam)} candidates, f on B {f} (Cause), Iso {row['iso_B']}, random same "
                f"size {row['random_same_size']}")
            var_teacher.release(nat.dev)
    for lam in lams:
        r = next(x for x in out["masks"] if x["lam"] == lam and x["targets"] == "real")
        pm = next(x for x in out["masks"] if x["lam"] == lam and x["targets"] == "permuted")
        r["selectivity"] = None if r["f_B"] is None or pm["f_B"] is None else r["f_B"] - pm["f_B"]
    R.donors.clear()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", default="text6011")
    ap.add_argument("inputs", nargs="?", default="232,233")
    ap.add_argument("--copy", type=int, default=0)
    ap.add_argument("--lam", default="0.001,0.01,0.1")
    ap.add_argument("--out", type=Path, default=native.TEXTS.parent / "semantic" / "masks.jsonl")
    ap.add_argument("--device")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(args.device))
    sc = var_teacher._Screen(ge)
    lams = [float(x) for x in args.lam.split(",")]
    jobs = [(args.task, tuple(int(x) for x in args.inputs.split(",")))] if args.task else []
    if args.copy:
        for x in json.loads((native.TEXTS.parent / "generality" / "copy_heldout.json").read_text())["copy"][:args.copy]:
            p = json.loads((native.TEXTS / "vpd4l" / f"{x['task']}.json").read_text())["prompts"][0]
            ids, t = p["token_ids"], p["target_positions"][-1]
            occ = [j for j in range(t) if ids[j] == ids[t]]  # the copied token's position: after the earlier occurrence
            if occ:
                jobs.append((x["task"], (occ[-1] + 1,)))
    for t, ins in jobs:
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        p = task["prompts"][0]
        tx = ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(t), task=task)
        print(f"{t} inputs {list(ins)}", flush=True)
        row = run_variable(ge, sc, tx, t, ins, lams, log=lambda s: print(s, flush=True))
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
