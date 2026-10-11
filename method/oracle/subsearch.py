"""Feasibility of subcomponent-level teacher circuits on vpd4l (#2951): how many VPD subcomponents carry a
position's change, measured with carry's exact unit evaluator.

For each listed position p of a text (heldout50_relays.jsonl: p, its relay positions, the target), the space is every
subcomponent at the positions of p's existing relay circuit. A set carries the change from the original text to a
changed prompt (p's token replaced by one of the model's alternatives there) when only its subcomponents compute on the
changed prompt and every other one is held at its original value; carried share = 1 - KL / movement, KL in bits at the
target, averaged over the prompts. Two orderings rank subcomponents and the measured prefixes of each give the smallest
prefix carrying 90%: the size of each subcomponent's activation change between the original and the changed prompts
(the whole model on both), and integrated gradients of the carry KL along the straight path from every subcomponent
held to every subcomponent live. Search prompts are the alternatives of ranks 0-2; held-out prompts ranks 3-5 (and
6-8 for a search on ranks 0-5).

Per text, the whole-prediction circuit is the union of the positions' sets; its sufficiency is the KL from the whole
model of a run in which only the union computes, every other subcomponent ablated (zero), VPD's remainder W - V U held
at its values on the text (and, as the verifier does, removed), on the text and on the changed prompts.

  MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python -B subsearch.py --n 20 --out ~/mpd-data/graph_oracle/subsearch/heldout20.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import carry
import native

RELAYS = Path.home() / "mpd-data/graph_oracle/carry/heldout50_relays.jsonl"
FRACTION = 0.9
IG_STEPS = 8


def prompts_at(nat, ids, p, ranks):
    return carry.alternatives(nat, ids, p, ranks=ranks)


def study_for(nat, ids, targets, prompts, space):
    st = carry.SubcomponentCarry(nat, ids, targets, prompts, space)
    return st, carry._movement(st)


def activation_change(nat, ids, prompts, space, base):
    """Mean over prompts of |a_c(changed) - a_c(original)| of the whole model, per unit of the space."""
    change = {n: 0 for n in nat.names}
    for x in prompts:
        a = nat.activations(x)
        for n in nat.names:
            change[n] = change[n] + (a[n] - base[n]).abs()
    score = torch.empty(space.size, device=nat.dev)
    for n, sl in space.offsets.items():
        score[sl] = (change[n] / len(prompts)).index_select(0, space.pos).reshape(-1)
    return score


def integrated_gradients(study, space, steps=IG_STEPS):
    """-d KL / d a_c averaged over a in (0, 1): units whose live value lowers the carry KL most score highest."""
    ig = torch.zeros(space.size, device=study.nat.dev)
    with torch.enable_grad():
        for k in range(steps):
            a = torch.full((space.size,), (k + .5) / steps, device=study.nat.dev, requires_grad=True)
            coeff = {n: v.repeat(len(study.prompts), 1, 1) for n, v in space.compact(a).items()}
            lq = carry._run_units(study, coeff, study.embeddings, compact=True)
            g, = torch.autograd.grad(carry.kl(torch.cat(study.lp), lq).mean(), a)
            ig -= g / steps
    return ig


def prefix_sets(order, sizes, N, dev):
    sets = torch.zeros(len(sizes), N, dtype=torch.bool, device=dev)
    for i, s in enumerate(sizes):
        sets[i, order[:s]] = True
    return sets


def smallest_prefix(study, movement, order, N):
    """The measured prefix curve of an ordering (sizes 2^(k/2)), refined by 32 sizes between the smallest prefix that
    carries 90% and the largest smaller one measured: (smallest size carrying 90% or None, its KL, the curve)."""
    sizes = sorted({min(N, int(2 ** (k / 2))) for k in range(0, 80) if 2 ** (k / 2) <= 2 * N} | {N})
    kl = study.measure(prefix_sets(order, sizes, N, study.nat.dev)).tolist()
    curve = list(zip(sizes, kl))
    ok = [i for i, (_, x) in enumerate(curve) if x <= (1 - FRACTION) * movement]
    if not ok:
        return None, None, curve
    i = ok[0]
    lo = sizes[i - 1] if i else 0
    fine = sorted({lo + max(1, (sizes[i] - lo) * j // 32) for j in range(1, 33)} - set(sizes))
    if fine:
        kf = study.measure(prefix_sets(order, fine, N, study.nat.dev)).tolist()
        curve = sorted(curve + list(zip(fine, kf)))
    best = min((s, x) for s, x in curve if x <= (1 - FRACTION) * movement)
    return best[0], best[1], curve


def carried(study, movement, active):
    return 1 - float(study.measure(active[None])[0]) / movement if movement > 1e-6 else None


def remainder_shares(nat, ids, targets, prompts):
    """carry.diagnostics on the node-set Carry: what the bypasses carry with every subcomponent held."""
    d = carry.diagnostics(carry.Carry(nat, ids, targets, prompts))
    mv = d["movement_bits"]
    return {"bypass": 1 - d["empty_bits"] / mv, "embedding_only": 1 - d["empty_frozen_remainders_bits"] / mv,
            "remainder": (d["empty_frozen_remainders_bits"] - d["empty_bits"]) / mv, "movement_bits": mv}


def position(nat, ids, targets, row, base_act, cis):
    t0 = time.perf_counter()
    p = row["position"]
    positions = sorted(set(row["position_sets"][0]))
    full = {(n, q): range(nat.C[n]) for n in nat.names for q in positions}
    space = carry.UnitSpace(nat, len(ids), positions, candidates=full)
    N = space.size
    search = [ids[:p] + [z] + ids[p + 1:] for z in row["alternatives"]]
    held = prompts_at(nat, ids, p, (3, 4, 5))
    later = prompts_at(nat, ids, p, (6, 7, 8))
    st_s, mv_s = study_for(nat, ids, targets, search, space)
    st_h, mv_h = study_for(nat, ids, targets, held, space)
    st_l, mv_l = study_for(nat, ids, targets, later, space)
    out = {"position": p, "positions": positions, "units": N, "movement_bits": {"search": mv_s, "ranks3_5": mv_h, "ranks6_8": mv_l},
           "empty_carried": carried(st_s, mv_s, torch.zeros(N, dtype=torch.bool, device=nat.dev))}
    ci_any = torch.zeros(N, dtype=torch.bool, device=nat.dev)
    for ci in cis(search):
        for n, sl in space.offsets.items():
            ci_any[sl] |= (ci[n].index_select(0, space.pos) > 0).reshape(-1)
    out["ci_units"] = int(ci_any.sum())
    out["ci_all_carried"] = carried(st_s, mv_s, ci_any)
    sets = {}
    for name, score in (("change", activation_change(nat, ids, search, space, base_act)),
                        ("ig", integrated_gradients(st_s, space))):
        order = torch.argsort(score, descending=True)
        size, kl, curve = smallest_prefix(st_s, mv_s, order, N)
        r = {"size_90": size, "kl_bits": kl, "curve": [[s, 1 - x / mv_s] for s, x in curve]}
        if size is not None:
            active = torch.zeros(N, dtype=torch.bool, device=nat.dev)
            active[order[:size]] = True
            sets[name] = active
            r.update(carried_ranks3_5=carried(st_h, mv_h, active), carried_ranks6_8=carried(st_l, mv_l, active),
                     ci_share=float((active & ci_any).sum()) / size)
        out[name] = r
    # search on ranks 0-5 (the change ordering), validated on ranks 6-8
    st6, mv6 = study_for(nat, ids, targets, search + held, space)
    order6 = torch.argsort(activation_change(nat, ids, search + held, space, base_act), descending=True)
    size6, kl6, _ = smallest_prefix(st6, mv6, order6, N)
    r6 = {"size_90": size6}
    if size6 is not None:
        a6 = torch.zeros(N, dtype=torch.bool, device=nat.dev)
        a6[order6[:size6]] = True
        r6.update(carried_ranks6_8=carried(st_l, mv_l, a6), carried_search=carried(st6, mv6, a6))
    out["change_search0_5"] = r6
    out["bypass"] = remainder_shares(nat, ids, targets, search)
    best = min((k for k in sets), key=lambda k: out[k]["size_90"], default=None)
    out["best"] = best
    out["units_best"] = [list(space.units[j]) for j in sets[best].nonzero().flatten().tolist()] if best else None
    out["seconds"] = time.perf_counter() - t0
    return out


@torch.no_grad()
def sufficiency(nat, ids, targets, units, prompts):
    """KL in bits from the whole model of the union alone (everything else zero), remainder held at the text's values
    and removed, on the text and averaged over prompts; beside it the same for no subcomponent at all, for every
    subcomponent at the union's positions, and VPD's own answer (causal importance > 0 at every position, on each
    prompt), the sparsity VPD claims under ablation."""
    T = len(ids)
    masks = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, t, c in units:
        masks[n][t, c] = True
    at = sorted({t for _, t, _ in units})
    every = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n in nat.names:
        every[n][at] = True
    none = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    res = {}
    for name, m in (("union", masks), ("nothing", none), ("all_at_union_positions", every), ("vpd_ci", None)):
        res[name] = _alone(nat, ids, targets, m, prompts)
    return res


def _alone(nat, ids, targets, masks, prompts):
    T = len(ids)
    plan = nat._plan([native.Graph(masks=masks)], T) if masks is not None else None
    zero_u = {n: torch.zeros(1, 1, nat.C[n], device=nat.dev) for n in nat.names}
    one_r = {n: torch.ones(1, 1, device=nat.dev) for n in nat.names}
    capture = carry._Remainders(nat)
    with capture:
        nat.run(ids, targets, nat._full_plan(T, 1), *nat._one())
    res = {}
    for label, x_list in (("text", [ids]), ("changed", prompts)):
        held, removed = [], []
        for x in x_list:
            ref = nat.reference([x], targets)
            if masks is None:
                plan = nat._plan([native.Graph(masks={n: m > 0 for n, m in nat.importance(x).items()})], T)
            with carry._Remainders(nat, capture.values):
                held.append(float(carry.kl(ref, nat.run(x, targets, plan, zero_u, one_r))[0]))
            removed.append(float(carry.kl(ref, nat.run(x, targets, plan, *nat._zero()))[0]))
        res[label] = {"remainder_held_bits": sum(held) / len(held), "remainder_removed_bits": sum(removed) / len(removed)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["task"] for line in args.out.read_text().splitlines()} if args.out.exists() else set()
    nat = native.Native()
    rows = [json.loads(line) for line in RELAYS.read_text().splitlines()]
    rows = [r for r in rows if r["positions"]][:args.n]
    for r in rows:
        if r["task"] in done:
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(native.TEXTS / "vpd4l" / f"{r['task']}.json")
        base = nat.activations(ids)
        ci0 = nat.importance(ids)
        cis = lambda prompts: [ci0] + [nat.importance(x) for x in prompts]
        result = []
        for row in r["positions"][:args.positions]:
            pos = position(nat, ids, targets, row, base, cis)
            result.append(pos)
            print(f"{r['task']} p={pos['position']} units={pos['units']} change90={pos['change']['size_90']} ig90={pos['ig']['size_90']} "
                  f"CI>0 {pos['ci_units']} carry {pos['ci_all_carried']:.2f} bypass {pos['bypass']['bypass']:.2f} {pos['seconds']:.1f}s", flush=True)
        union = {tuple(u) for pos in result if pos["units_best"] for u in pos["units_best"]}
        prompts = [x for row in r["positions"][:args.positions] for x in
                   [ids[:row["position"]] + [z] + ids[row["position"] + 1:] for z in row["alternatives"]]]
        suff = sufficiency(nat, ids, targets, union, prompts) if union else None
        with args.out.open("a") as f:
            f.write(json.dumps({"task": r["task"], "targets": targets, "positions": [{k: v for k, v in pos.items() if k != "units_best"} for pos in result],
                                "union_size": len(union), "union_by_position": len({u[1] for u in union}),
                                "sufficiency": suff, "seconds": time.perf_counter() - t0}) + "\n")
        print(f"{r['task']}: union {len(union)} subcomponents; sufficiency {suff}; {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
