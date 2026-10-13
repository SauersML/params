"""Learned masks over VPD subcomponents as variable carriers, with their controls (#2951 graph oracle, E4 v2;
method/oracle/masks.py holds the masking; desiderata-based masking,
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
Each mask's bits: log2 C(N, k), k of the N subcomponents at the carrier positions in the candidates' sites (the
family's the same with k its size). --depth D keeps the candidates in layers <= D (the lead's 10-13 copy read: the
target position's stage before the output parts); --read with --copy N carries each copy text's read at the target
position t: inputs the earlier occurrence o of t's token and the token a after it (the read depends on the match at
o and copies a; the lead's 10-13), Iso donors at other positions before t.

  experiments/oracle/var_masks.py text6011 232,233 [--at POSITIONS] [--skip POSITIONS] [--copy N [--read]] [--depth 1,2]
      [--from-dir DIR] [--lam 1e-3,1e-2] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import goal_eval  # noqa: E402
import masks  # noqa: E402
import native  # noqa: E402
import var_classes  # noqa: E402
import var_controls  # noqa: E402
import var_teacher  # noqa: E402
import variables  # noqa: E402

def run_variable(ge, sc, tx, task_id: str, ins: tuple, lams: list[float], at: tuple | None = None, log=print,
                 depth: int | None = None, skip: tuple = ()) -> dict:
    """at: the carrier positions (an intermediate variable's, after its inputs); the input positions when None. depth:
    candidates in layers <= depth only; skip: positions left out of the Iso donors."""
    nat = ge.nat
    at = at or ins
    reach = nat.reachable(tx.T, tx.targets)
    every = [n for n in nat.names if all(bool(reach[n][q].any()) for q in at) and (depth is None or int(n.split(".")[1]) <= depth)]
    fam, _ = var_teacher.family(nat, tx, ins, every, at)
    N = sum(nat.C[n] for n in every) * len(at)
    bits = lambda k: var_controls.lbinom(N, k)  # noqa: E731
    A = {q: var_classes.alternatives(nat, tx.ids, q, int(tx.seed)) for q in ins}
    B = {q: var_controls.b_tokens(nat, tx, q, A[q], max(4, var_controls.B_PER // len(ins)), int(tx.seed)) for q in ins}
    texts_A, pzA = [], []
    for q in ins:
        for z in A[q]:
            x = list(tx.ids)
            x[q] = z
            texts_A.append(x)
            pzA.append((q, z))
    # changes before the carrier (one at or after it passes by construction), not at its inputs or the skipped positions
    iso_pos = masks.iso_positions(tx, ins, at, skip)
    iso_A = {q: var_classes.alternatives(nat, tx.ids, q, int(tx.seed), top=8, sampled=8) for q in iso_pos}
    iso_B = {q: var_controls.b_tokens(nat, tx, q, iso_A[q], 4, int(tx.seed)) for q in iso_pos}
    texts_I = []
    for q in iso_pos:
        for z in iso_A[q]:
            x = list(tx.ids)
            x[q] = z
            texts_I.append(x)
    P = masks.Patcher(nat, tx.ids, tx.targets, fam)
    srcA = P.acts(texts_A + texts_I)
    with torch.no_grad():
        tgtA = torch.cat([nat.reference(texts_A[s:s + masks.BATCH], tx.targets) for s in range(0, len(texts_A), masks.BATCH)])
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
    out = {"task": task_id, "inputs": list(ins), "at": list(at), "depth": depth, "skip": list(skip), "iso_positions": iso_pos,
           "candidates": len(fam), "N": N, "family_bits": bits(len(fam)), "A": len(pzA), "iso_train": len(texts_I), "B": len(pzB),
           "iso_B": len(pzI), "family_f_B": R.f_on(fam, pzB), "family_iso_B": f_off(fam), "masks": []}
    perm = torch.randperm(len(pzA), generator=torch.Generator().manual_seed(int(tx.seed))).to(nat.dev)
    perm = torch.cat([perm, torch.arange(len(pzA), len(pzA) + len(texts_I), device=nat.dev)])  # Iso targets stay
    for lam in lams:
        for kind, tgt in (("real", tgtA), ("permuted", tgtA[perm])):
            logit = masks.train(P, srcA, tgt, lam)
            keep = [fam[j] for j in (logit > 0).nonzero().flatten().tolist()]
            scorer = {}
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
            if kind == "real" and keep:  # the teacher's yardstick: variables.Scorer's term min(f on-target, s off-target)
                vj = var_teacher.as_json("mask", ins, keep)
                for split in ("search", "eval"):
                    x = var_teacher.together(variables.Scorer(ge), tx, variables.parse([vj], nat.C), split, 8)["variables"][0]
                    scorer[split] = {"term": var_teacher.term(x), "f_on": x["f_on"], "f_off": x["f_off"], "s_off": x["s_off"]}
            row = {"lam": lam, "targets": kind, "size": len(keep), "bits": bits(len(keep)), "f_B": f,
                   "iso_B": f_off(keep) if kind == "real" else None,
                   "scorer": dict(scorer) if kind == "real" else None,
                   "random_same_size": float(np.mean([x for x in rnd if x is not None])) if any(x is not None for x in rnd) else None}
            out["masks"].append(row)
            log(f"  lam {lam} {kind}: {len(keep)} of {len(fam)} candidates ({row['bits']:.0f} bits), f on B {f} (Cause), Iso {row['iso_B']}, random same "
                f"size {row['random_same_size']}, scorer {row['scorer']}")
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
    ap.add_argument("--at", help="the carrier positions (an intermediate variable's), comma-separated")
    ap.add_argument("--skip", help="positions left out of the Iso donors, comma-separated")
    ap.add_argument("--read", action="store_true", help="--copy texts' read carried at the target: inputs (o, a)")
    ap.add_argument("--depth", help="layer cuts, comma-separated: one run per cut, candidates in layers <= the cut")
    ap.add_argument("--from-dir", type=Path, help="every kept intermediate variable in var_teacher's files there")
    ap.add_argument("--jobs", type=Path, help='JSON lines {"task", "inputs", "at"} (a pod uploads one file)')
    ap.add_argument("--lam", default="0.001,0.01,0.1")
    ap.add_argument("--out", type=Path, default=native.TEXTS.parent / "semantic" / "masks.jsonl")
    ap.add_argument("--device")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(args.device))
    sc = var_teacher._Screen(ge)
    lams = [float(x) for x in args.lam.split(",")]
    ints = lambda s: tuple(int(x) for x in s.split(",")) if s else ()  # noqa: E731
    jobs = [(args.task, ints(args.inputs), ints(args.at) or None, ints(args.skip))] if args.task and not (args.from_dir or args.jobs or args.copy) else []
    for x in map(json.loads, args.jobs.read_text().splitlines()) if args.jobs else []:
        jobs.append((x["task"], tuple(x["inputs"]), tuple(x["at"]) if x.get("at") else None, tuple(x.get("skip", ()))))
    for f in sorted(args.from_dir.iterdir()) if args.from_dir else []:
        if f.name.startswith("text") and f.suffix == ".json":
            for v in json.loads(f.read_text())["variables"]:
                if v.get("kept") and v.get("kind") == "intermediate":
                    jobs.append((f.stem, tuple(v["inputs"]), tuple(v["carried_at"]), ()))
    if args.copy:
        for x in json.loads((native.TEXTS.parent / "generality" / "copy_heldout.json").read_text())["copy"][:args.copy]:
            p = json.loads((native.TEXTS / "vpd4l" / f"{x['task']}.json").read_text())["prompts"][0]
            ids, t = p["token_ids"], p["target_positions"][-1]
            occ = [j for j in range(t) if ids[j] == ids[t]]  # the copied token's position: after the earlier occurrence
            if occ:
                jobs.append((x["task"], (occ[-1], occ[-1] + 1), (t,), ()) if args.read else (x["task"], (occ[-1] + 1,), None, ()))
    depths = [int(x) for x in args.depth.split(",")] if args.depth else [None]
    for t, ins, at, skip in jobs:
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        p = task["prompts"][0]
        tx = ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(t), task=task)
        for d in depths:
            print(f"{t} inputs {list(ins)} at {list(at or ins)} layers <= {d} skip {list(skip)}", flush=True)
            row = run_variable(ge, sc, tx, t, ins, lams, at, log=lambda s: print(s, flush=True), depth=d, skip=skip)
            with args.out.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
