"""Is internal-intervention prediction learnable? (#2951 graph oracle, the lead's learnability experiment.) The data: per
text, measured effects on vpd4l's prediction at the target of
  remove: one VPD-active subcomponent at one position set to zero, everything else the whole model (Native.without);
  cut:    one connection between two of them cut (GoalEval.cuts, path patching of that connection), half those
          whose two ends' removals matter most, half at random;
  swap:   one subcomponent's activation set to its value on a changed text (the token at an influential position
          replaced by the model's likeliest other token there, carry.alternatives), everything else the text's;
each with KL(p || q) in bits at the target, the top token's log2 probability ratio ("falls": below -1, as E2's
questions), and q's top token. Candidates are every VPD-active subcomponent at the target and at the POSITIONS
positions whose changed token moves the prediction most (Native.sensitivity), with their features for the arms that
read them: causal importance, activation (x . v |u|), the routed lens's raised tokens at that position, the atlas's
active-on tokens and activity rate, and the logit lens of the write on the model's top token (o and down at the
target only: the rest do not write the residual stream there; without the final norm's scale, one factor per text).

`measure` (a pod with VPD's checkpoint) writes per text the candidates, the features their run gives and the effects;
features() adds the atlas's and the routed lens's facts (the Mac) to those the questions use.

  learnability.py measure --texts 6000:6100 OUT.jsonl [--split train] [--device cuda] [--offset K --stride S]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "oracle"))
sys.path.insert(0, str(ROOT / "method" / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

LN2 = math.log(2)
POSITIONS = 3  # influential context positions besides the target
CUTS = 48  # connections cut per text among the candidates' connections
SWAP_POSITIONS = 2  # changed positions per text for swaps
SWAPS = 48  # swaps per changed position, the candidates whose activation moves most first
BATCH = 16


def _task(name: str) -> dict:
    return json.loads((native.TEXTS / "vpd4l" / f"{name}.json").read_text())


@torch.no_grad()
def candidates(nat, task: dict) -> dict:
    """One text's candidates and the features its run gives (VPD's importance needed)."""
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    t = max(targets)
    g = torch.Generator(device="cpu").manual_seed(native.task_seed(task["id"]))
    moved, _ = nat.sensitivity(ids, targets, g)
    near = [q for q in (moved.argsort(descending=True) + 1).tolist() if q not in targets][:POSITIONS]
    ci = nat.importance(ids)
    acts = nat.activations(ids)
    top = int(nat.reference([ids], targets)[0, -1].argmax())
    tg = nat.target
    cands = []
    for nd in nat._vpd_order(ids, targets):
        n, q, c = nd
        if q not in near and q != t:
            continue
        u = tg.site(n).U[c]
        # the write's direct effect on the top token's logit, but for the final norm's scale (one factor at t)
        ll = (float(acts[n][q, c] / u.norm() * (u * tg.ln_f) @ tg.wte[top])
              if native._layer_kind(nd)[1] in ("o_proj", "down_proj") and q == t else None)
        cands.append({"node": [n, q, c], "ci": float(ci[n][q, c]), "act": float(acts[n][q, c]), "logit_lens": ll})
    return {"task": task["id"], "split": task["split"], "T": len(ids), "target": t, "top": top, "near": near,
            "candidates": cands}


@torch.no_grad()
def features(nat, rec: dict, nodes: list[tuple], atlas, lens) -> dict:
    """The atlas's and the routed lens's facts of some of a text's candidates (the Mac: the atlas and the lens's fit):
    node -> {"routed": the tokens its write raises most at its position, [[id, value]], or None for q and k; "on": the
    atlas's most frequent active-on tokens; "rate": its activity rate on Pile text}."""
    ids = _task(rec["task"])["prompts"][0]["token_ids"]
    routed = dict(zip(nodes, lens.readout(ids, nodes))) if lens else {}
    out = {}
    for nd in nodes:
        e = atlas.entry(nd[0], nd[2]) if atlas else None
        count = {}
        for x in e["contexts"] if e else []:
            count[x["token"]] = count.get(x["token"], 0) + 1
        r = routed.get(nd)
        out[nd] = {"routed": [[i, round(float(v), 3)] for i, v in r["raised"][:3]] if r else None,
                   "on": sorted(count, key=lambda k: -count[k])[:3], "rate": e["active_rate"] if e else None}
    return out


def _effect(lp0: torch.Tensor, lq: torch.Tensor, top: int) -> dict:
    """KL(p || q) bits, the top token's log2 ratio, falls, q's top, at the target (lp0, lq [V])."""
    bits = float(lq[top] - lp0[top]) / LN2
    return {"kl": float((lp0.exp() * (lp0 - lq)).sum()) / LN2, "bits": bits, "falls": bits < -1.0, "top": int(lq.argmax())}


@torch.no_grad()
def without_many(nat, ids: list[int], targets: list[int], nodes: list[tuple]) -> torch.Tensor:
    """Native.without for each node alone, batched: log q at the last target [len(nodes), V]."""
    T = len(ids)
    out = []
    for i in range(0, len(nodes), BATCH):
        chunk = nodes[i:i + BATCH]
        B = len(chunk)
        u = {n: torch.ones(B, T, nat.C[n], device=nat.dev) for n in nat.names}
        for b, (n, t, c) in enumerate(chunk):
            u[n][b, t, c] = 0.0
        ur = {n: torch.ones(B, T, device=nat.dev) for n in nat.names}
        none = {n: torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        plan = {**nat._full_plan(T, B), "G": none, "none": True}
        out.append(nat.run(ids, targets, plan, u, ur)[:, -1])
    return torch.cat(out)


@torch.no_grad()
def swap_many(nat, ids: list[int], targets: list[int], values: dict, nodes: list[tuple]) -> torch.Tensor:
    """The whole model on ids with each node's activation alone set to its value in `values` (a record of the model on
    the changed text, [1, T, C] per matrix): log q at the last target [len(nodes), V]."""
    T = len(ids)
    out = []
    u1, ur1 = nat._one()
    for i in range(0, len(nodes), BATCH):
        chunk = nodes[i:i + BATCH]
        B = len(chunk)
        patch = {}
        for b, (n, t, c) in enumerate(chunk):
            if n not in patch:
                patch[n] = (torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev), values[n].expand(B, -1, -1))
            patch[n][0][b, t, c] = True
        out.append(nat.run(ids, targets, nat._full_plan(T, B), u1, ur1, patch=patch)[:, -1])
    return torch.cat(out)


def measure(ge, task: dict, log=print) -> dict:
    """candidates() with each candidate's removal effect, CUTS connection cuts and the swaps."""
    nat = ge.nat
    t0 = time.time()
    pre = candidates(nat, task)
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    rng = random.Random(native.task_seed(pre["task"]))
    lp0 = nat.reference([ids], targets)[0, -1]
    top = int(lp0.argmax())
    nodes = [tuple(c["node"]) for c in pre["candidates"]]
    for c, lq in zip(pre["candidates"], without_many(nat, ids, targets, nodes)):
        c["remove"] = _effect(lp0, lq, top)
    par, outs = native.all_edges(set(nodes), targets)
    pairs = [(w, r) for r, ws in par.items() for w in ws] + [(w, None) for w in outs]
    # half the connections whose ends' removals matter most (most connections carry nothing), half at random
    gone = {tuple(c["node"]): c["remove"]["kl"] for c in pre["candidates"]}
    strong = sorted(pairs, key=lambda e: -min(gone[e[0]], gone[e[1]] if e[1] is not None else math.inf))[:CUTS // 2]
    rest = [e for e in pairs if e not in set(strong)]
    pairs = strong + rng.sample(rest, min(len(rest), CUTS - len(strong)))
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    cuts = []
    for (w, r), d in zip(pairs, ge.cuts(tx, pairs, detail=True) if pairs else []):
        bits = (d["logp_model_top"][-1] - float(lp0[top])) / LN2
        cuts.append({"edge": [list(w), None if r is None else list(r)], "kl": d["kl"], "bits": bits, "falls": bits < -1.0,
                     "top": int(d["top"][-1])})
    swaps = []
    for q in pre["near"][:SWAP_POSITIONS]:
        x = carry.alternatives(nat, ids, q, ranks=(0,))[0]
        rec = {}
        lx = nat.run(x, targets, nat._full_plan(len(x), 1), *nat._one(), record=rec)[0, -1]
        base = nat._model_record(ids)[0]
        moved = sorted(((n, s, c) for n, s, c in nodes if s >= q), key=lambda nd: -abs(float(rec[nd[0]][0, nd[1], nd[2]] - base[nd[0]][0, nd[1], nd[2]])))
        pick = moved[:SWAPS]
        for nd, lq in zip(pick, swap_many(nat, ids, targets, rec, pick) if pick else []):
            swaps.append({"node": list(nd), "position": q, "new": x[q], "act_x": float(base[nd[0]][0, nd[1], nd[2]]),
                          "act_changed": float(rec[nd[0]][0, nd[1], nd[2]]), **_effect(lp0, lq, top),
                          "of_change": _effect(lp0, lx, top)["kl"]})
    log(f"{pre['task']}: {len(nodes)} removals, {len(cuts)} cuts, {len(swaps)} swaps ({time.time() - t0:.1f} s)")
    return {**pre, "cuts": cuts, "swaps": swaps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--texts", required=True, help="A:B, text numbers")
    m.add_argument("out")
    m.add_argument("--split", default=None)
    m.add_argument("--device", default="cpu")
    m.add_argument("--offset", type=int, default=0)
    m.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(args.device))
    lo, hi = map(int, args.texts.split(":"))
    for k in range(lo + args.offset, hi, args.stride):
        path = native.TEXTS / "vpd4l" / f"text{k}.json"
        if not path.exists():
            continue
        task = json.loads(path.read_text())
        if args.split and task["split"] != args.split:
            continue
        r = measure(ge, task, log=lambda s: print(s, flush=True))
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
