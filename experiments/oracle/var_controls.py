"""Controls for variable carriers (#2951 graph oracle; Sutter et al. 2025, the non-linear representation dilemma): is a
carrier's f more than what any group of its size, or any made-up variable, gets?

For one variable on one text (input positions I, carrier positions = I, the candidate sites S), with A the 64 tokens
per position the family and the value map are built from (var_classes.alternatives) and B donor tokens disjoint from
A (sampled from the model's prediction at the position, A and the text's own token left out), every f below is
measured on B (disjoint values: the paper's one discriminating test). f on-target = 1 - KL(model(x') || patched) /
KL(model(x') || model(x)) per donor x' (B token at an input position), the KL-weighted mean over donors.
  carriers: the family over every site at I (VPD's active subcomponents on the text and on A), the family at S, the
    whole position (every site's block), VPD's active set on the text;
  equal-size nulls: per carrier, 32 random subsets of the same size of the subcomponents at I and S, matched on mass
    (each subcomponent's CI x |activation| on the text: as many active ones, total mass within 20%), and f against
    size for random subsets of 2^k subcomponents;
  bits: log2 C(N, k) of each subset choice (N the subcomponents at I and S), f per bit against the null's;
  wrong variables: value maps the model does not compute (a random balanced split of B's tokens, token-id parity), each
    scored with every candidate carrier (single-site blocks, layers, the families: the real search's budget) by
    min(f_diff, f_same) (f_same = 1 - KL(model(x) || patched) / KL(model(x) || model(x'))), against the real value map
    (classes from A's patched runs, B's tokens assigned by the model's own prediction on x': class 0 when it keeps
    the text's top token, else the class predicting that token) with the same carriers, on A (assigned in-sample)
    and on B (disjoint donors);
  off-target, what s 1.0 measures: variables.Scorer's off-target test patches the carrier from a donor changing a token
    outside I and asks that the prediction stay; a carrier at I is the same on x and x' when the change lies after I
    (the model is causal), so those draws pass by construction; the informative ones change a token before I.
Rows to OUT (one JSON line per variable), a summary printed.

  experiments/oracle/var_controls.py text6011 232,233 [--copy N] [--out FILE] [--device mps]
"""
from __future__ import annotations

import argparse
import hashlib
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
import var_teacher  # noqa: E402
import variables  # noqa: E402

BLOCK = variables.BLOCK
NULLS = 32
B_PER = 16


def kl(a, b) -> float:
    return float(carry.kl(a[None], b[None])[0])


def lbinom(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


class Runs:
    """The text's runs and patched runs for explicit donors (variables.Scorer's real-model runs)."""

    def __init__(self, sc, tx):
        self.sc, self.tx = sc, tx
        self.mine = sc._text(tx.ids, tx.targets, check=False)
        self.base = self.mine["lq"]
        self.donors = {}

    def donor(self, p: int, z: int) -> dict:
        if (p, z) not in self.donors:
            x = list(self.tx.ids)
            x[p] = z
            self.donors[(p, z)] = self.sc._text(x, self.tx.targets, check=False)
        return self.donors[(p, z)]

    def patched(self, group: list, pz: list) -> list:
        """log q at the targets with the group set from each donor (p, z)."""
        if not group:
            return [self.base] * len(pz)
        q = self.sc._patched(self.mine, [(self.donor(p, z), group) for p, z in pz])
        return list(q)

    def f_on(self, group: list, pz: list) -> float | None:
        q = self.patched(group, pz)
        num = den = 0.0
        for (p, z), qq in zip(pz, q):
            ref = self.donor(p, z)["lq"]
            d0 = kl(ref, self.base)
            if d0 > 1e-6:
                num += d0 * min(1.0, max(0.0, 1.0 - max(kl(ref, qq), 0.0) / d0))
                den += d0
        return num / den if den > 0 else None

    def value_terms(self, group: list, same: list, diff: list) -> dict:
        """(f_same, f_diff) for a value map's same-class and other-class donors."""
        out = {}
        for kind, pz in (("same", same), ("diff", diff)):
            if not pz:
                out[kind] = None
                continue
            q = self.patched(group, pz)
            num = den = 0.0
            for (p, z), qq in zip(pz, q):
                ref = self.donor(p, z)["lq"]
                d0 = kl(self.base, ref) if kind == "same" else kl(ref, self.base)
                d = kl(self.base, qq) if kind == "same" else kl(ref, qq)
                if d0 > 1e-6:
                    num += d0 * min(1.0, max(0.0, 1.0 - max(d, 0.0) / d0))
                    den += d0
            out[kind] = num / den if den > 0 else None
        out["term"] = min([x for x in (out["same"], out["diff"]) if x is not None], default=None)
        return out


def b_tokens(nat, tx, p: int, A: list, n: int, seed: int) -> list:
    """n donor tokens at p sampled from the model's prediction there without replacement, A and the text's own left
    out."""
    lp = nat.reference([tx.ids[:p]], [p - 1])[0, 0].cpu().double()
    w = lp.exp()
    w[[tx.ids[p]] + list(A)] = 0.0
    gen = torch.Generator().manual_seed(int(hashlib.sha256(repr(("controls", seed, p)).encode()).hexdigest()[:15], 16))
    return torch.multinomial(w, n, replacement=False, generator=gen).tolist()


def run_variable(ge, sc, tx, task_id: str, ins: tuple, log=print) -> dict:
    nat = ge.nat
    reach = nat.reachable(tx.T, tx.targets)
    every = [n for n in nat.names if all(bool(reach[n][q].any()) for q in ins)]
    blk, rows, triv = var_teacher.block_search(sc, tx, ins, 8)
    sites = sorted({n for n, _, _ in blk["group"]})
    fam, share = var_teacher.family(nat, tx, ins, every, ins)
    A = {q: var_classes.alternatives(nat, tx.ids, q, int(tx.seed)) for q in ins}
    B = {q: b_tokens(nat, tx, q, A[q], max(4, B_PER // len(ins)), int(tx.seed)) for q in ins}
    pzB = [(q, z) for q in ins for z in B[q]]
    R = Runs(sc, tx)
    act = R.mine["act"]
    pool = [(n, q, c) for n in every for q in ins for c in range(nat.C[n])]
    massd = {}
    for n in every:
        for q in ins:
            mv = (tx.ci[n][q].clamp_min(0) * act[n][q].abs()).cpu().double().numpy()
            for c in np.nonzero(mv)[0]:
                massd[(n, q, int(c))] = float(mv[c])
    on_x = [k for k in fam if k in massd]
    carriers = {"family at every site": fam, "family at the blocks' sites": [k for k in fam if k[0] in sites],
                "blocks (chosen)": blk["group"], "whole position": triv["group"], "active on the text": on_x}
    rng = np.random.default_rng(int(tx.seed))
    active_pool = [k for k in pool if k in massd]
    idle_pool = [k for k in pool if k not in massd]
    out = {"task": task_id, "inputs": list(ins), "pool": len(pool), "B": {q: len(B[q]) for q in ins}, "carriers": {}}

    def expand(g):  # blocks as their subcomponents, for sizes and nulls
        return [x for n, q, c in g for x in ([(n, q, cc) for cc in range(nat.C[n])] if c == BLOCK else [(n, q, c)])]

    for name, g in carriers.items():
        items = expand(g)
        k = len(items)
        ka = sum(1 for x in items if x in massd)
        mass = sum(massd.get(x, 0.0) for x in items)
        f = R.f_on(g, pzB)
        nulls = []
        for _ in range(NULLS):
            for _try in range(50):
                a = [active_pool[i] for i in rng.choice(len(active_pool), size=min(ka, len(active_pool)), replace=False)]
                ma = sum(massd[x] for x in a)
                if mass <= 0 or abs(ma - mass) <= 0.2 * mass:
                    break
            idle = [idle_pool[i] for i in rng.choice(len(idle_pool), size=min(k - len(a), len(idle_pool)), replace=False)]
            nulls.append(R.f_on(a + idle, pzB))
            var_teacher.release(nat.dev)
        nz = [x for x in nulls if x is not None]
        bits = lbinom(len(pool), k)
        out["carriers"][name] = {"size": k, "active": ka, "mass": mass, "f_B": f, "null_mean": float(np.mean(nz)) if nz else None,
                                 "null_max": float(np.max(nz)) if nz else None, "null_p95": float(np.percentile(nz, 95)) if nz else None,
                                 "bits": bits, "f_per_kbit": (f / bits * 1000) if f is not None and bits > 0 else None}
        log(f"  {name}: {k} subcomponents ({ka} active), f on B {f}, equal-size null mean "
            f"{out['carriers'][name]['null_mean']} max {out['carriers'][name]['null_max']}, {bits:.0f} bits")
    curve = []
    for e in range(4, int(math.log2(len(pool))) + 1):
        k = 1 << e
        fs = [R.f_on([pool[i] for i in rng.choice(len(pool), size=k, replace=False)], pzB) for _ in range(4)]
        curve.append({"size": k, "f_mean": float(np.mean([x for x in fs if x is not None])) if any(x is not None for x in fs) else None})
        var_teacher.release(nat.dev)
    out["random_curve"] = curve
    # value maps: the real one (classes from A's patched runs with the whole position's carrier, B assigned by the
    # model's own prediction) and the wrong ones, each with every candidate carrier
    vm = {q: var_classes.classes(sc, tx, {"name": "v", "inputs": list(ins), "group": triv["group"]}, q) for q in ins}
    y0 = int(R.base[-1].argmax())

    def real_classes(q, zs, assign):  # same-class and other-class donors under the real map
        own = set(vm[q]["values"][0]["token_ids"])
        by_pred = {c["predicts_id"]: c["id"] for c in vm[q]["values"][1:]}
        same, diff = [], []
        for z in zs:
            if assign == "A":
                cls = 0 if z in own else next((c["id"] for c in vm[q]["values"] if z in c["token_ids"]), None)
            else:
                top = int(R.donor(q, z)["lq"][-1].argmax())
                cls = 0 if top == y0 else by_pred.get(top, -1)
            (same if cls == 0 else diff).append((q, z))
        return same, diff
    sub = {q: sorted(A[q], key=lambda z: hashlib.sha256(repr(("A", int(tx.seed), q, z)).encode()).digest())[:len(B[q])] for q in ins}
    maps = {"real, A (assigned in-sample)": [real_classes(q, sub[q], "A") for q in ins],
            "real, B (disjoint donors)": [real_classes(q, B[q], "B") for q in ins]}
    for q in ins:  # the text's own token taken into the first half of a random split, and into its own parity class
        rs = list(B[q])
        rng.shuffle(rs)
        maps.setdefault("random balanced split of B", []).append(([(q, z) for z in rs[:len(rs) // 2]], [(q, z) for z in rs[len(rs) // 2:]]))
        maps.setdefault("token-id parity on B", []).append(
            ([(q, z) for z in B[q] if z % 2 == tx.ids[q] % 2], [(q, z) for z in B[q] if z % 2 != tx.ids[q] % 2]))
    budget = {r["how"]: r["group"] for r in rows} | {k: v for k, v in carriers.items() if k.startswith("family")}
    out["value_maps"] = {}
    for mname, per in maps.items():
        same = [x for s, _ in per for x in s]
        diff = [x for _, d in per for x in d]
        scores = {h: R.value_terms(g, same, diff) for h, g in budget.items()}
        best = max(scores.items(), key=lambda kv: -1 if kv[1]["term"] is None else kv[1]["term"])
        out["value_maps"][mname] = {"same": len(same), "diff": len(diff), "best": best[0], "term": best[1]["term"],
                                    "f_same": best[1]["same"], "f_diff": best[1]["diff"],
                                    "family at every site": scores.get("family at every site")}
        log(f"  value map {mname}: {len(same)} same-class, {len(diff)} other-class donors; best carrier {best[0]} term "
            f"{best[1]['term']} (f_same {best[1]['same']}, f_diff {best[1]['diff']})")
        var_teacher.release(nat.dev)
    # off-target: which of the scorer's off-target draws can reach the carrier
    off = [p for p, _ in tx.f2 if p not in ins]
    draws = ge._interchange_draws(tx, off, "search", 8) if off else []
    out["off_target"] = {"draws": len(draws), "before_carrier": sum(p < min(ins) for p, _ in draws),
                         "after_carrier (pass by construction)": sum(p > max(ins) for p, _ in draws)}
    log(f"  off-target draws: {out['off_target']}")
    R.donors.clear()
    var_teacher.release(nat.dev)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", default="text6011")
    ap.add_argument("inputs", nargs="?", default="232,233")
    ap.add_argument("--copy", type=int, default=0, help="also the copied token's variable (role a) on the first N copy texts")
    ap.add_argument("--out", type=Path, default=native.TEXTS.parent / "semantic" / "controls.jsonl")
    ap.add_argument("--device")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(args.device))
    sc = var_teacher._Screen(ge)
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
        row = run_variable(ge, sc, tx, t, ins, log=lambda s: print(s, flush=True))
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
