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


class GatedCarry:
    """Carry in VPD's gated model: on each prompt a subcomponent runs only where its causal importance on that prompt
    is above zero (VPD's hard gate), remainders live. A set's subcomponents compute on the changed prompt under its
    gates; every other one keeps its gated value on the text. Units are the subcomponents at `positions` with causal
    importance above zero on the text or a prompt (any other is zero on both, so never carries anything). KL in bits
    at the target from the gated model on the changed prompt ("gated") or from the model ("model"), mean over prompts."""

    def __init__(self, nat, ids, targets, prompts, positions, units=None, B=32):
        self.nat, self.ids, self.targets, self.prompts, self.B = nat, ids, targets, prompts, B
        T = self.T = len(ids)
        self.one_r = {n: torch.ones(1, 1, device=nat.dev) for n in nat.names}
        self.g0 = self.gate(ids)
        self.rec0, self.states0 = {}, {}
        with torch.no_grad():
            nat.run(ids, targets, self.plan(1), self.g0, self.one_r, record=self.rec0)
            nat.run(ids, targets, self.plan(1), self.g0, self.one_r, fill=self.states0)
        self.g1, self.rec1, self.ref_gated, self.ref_model = [], [], [], []
        for x in prompts:
            g, rec = self.gate(x), {}
            with torch.no_grad():
                self.ref_gated.append(nat.run(x, targets, self.plan(1), g, self.one_r, record=rec))
                self.ref_model.append(nat.reference([x], targets))
            self.g1.append(g)
            self.rec1.append(rec)
        if units is None:
            units = sorted({(n, t, c) for n in nat.names for t in positions for c in
                            ((self.g0[n][0, t] > 0) | torch.stack([g[n][0, t] > 0 for g in self.g1]).any(0)).nonzero().flatten().tolist()})
        self.units, self.N = units, len(units)
        self.cols = {}
        for j, (n, t, c) in enumerate(units):
            self.cols.setdefault(n, ([], [], []))
            self.cols[n][0].append(j), self.cols[n][1].append(t), self.cols[n][2].append(c)
        self.cols = {n: tuple(torch.tensor(v, device=nat.dev) for v in a) for n, a in self.cols.items()}

    def gate(self, x):
        return {n: (m > 0).float()[None] for n, m in self.nat.importance(x).items()}

    def plan(self, B):
        nat, T = self.nat, self.T
        return nat._const(("gated-none", T, B), lambda: {**nat._full_plan(T, B), "none": True,
                                                         "G": {n: torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}})


def drop_plans(nat):
    """Forget the plans built for one text's length (each holds [B, T, C] masks); the next text has another T."""
    nat._consts = {k: v for k, v in nat._consts.items() if not (isinstance(k, tuple) and k[0] in ("gated-none", "full"))}
    native.free(nat.dev)

    def change(self):
        """Mean over prompts of |gated activation on the prompt - on the text| per unit."""
        out = torch.zeros(self.N, device=self.nat.dev)
        for g, rec in zip(self.g1, self.rec1):
            for n, (j, t, c) in self.cols.items():
                out[j] += (rec[n][0, t, c] * g[n][0, t, c] - self.rec0[n][0, t, c] * self.g0[n][0, t, c]).abs()
        return out / len(self.prompts)

    @torch.no_grad()
    def measure(self, sets, ref="gated"):
        nat, T = self.nat, self.T
        sets = sets if sets.dim() == 2 else sets[None]
        total = torch.zeros(len(sets), device=nat.dev)
        for i, x in enumerate(self.prompts):
            refs = self.ref_gated[i] if ref == "gated" else self.ref_model[i]
            for s0 in range(0, len(sets), self.B):
                part = sets[s0:s0 + self.B]
                valid = len(part)
                if 1 < valid < self.B:              # one batch shape per text: the GPU compiles, and plans cache, per shape
                    part = torch.cat([part, part[-1:].expand(self.B - valid, -1)])
                B = len(part)
                u, patch = {}, {}
                for n in nat.names:
                    live = torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev)
                    if n in self.cols:
                        j, t, c = self.cols[n]
                        live[:, t, c] = part[:, j]
                    u[n] = torch.where(live, self.g1[i][n], self.g0[n])
                    patch[n] = (~live, self.rec0[n].expand(B, -1, -1))
                lq = nat.run(x, self.targets, self.plan(B), u, self.one_r, patch=patch, reuse=(self.states0, self.ids))
                total[s0:s0 + valid] += carry.kl(refs.expand(B, -1, -1), lq)[:valid]
        return total / len(self.prompts)


def refine(measure, S, threshold, rounds=40):
    """Measured removal rounds: every single removal measured, removals ranked by the KL they leave, the most of them
    (prefix sizes 2^(k/2)) that keeps the KL at or below the threshold removed; until no removal keeps it."""
    S = S.clone()
    for _ in range(rounds):
        idx = S.nonzero().flatten()
        if len(idx) <= 1:
            break
        trials = S[None].repeat(len(idx), 1)
        trials[torch.arange(len(idx), device=S.device), idx] = False
        single = torch.cat([measure(trials[i:i + 1024]) for i in range(0, len(idx), 1024)])
        ranked = idx[torch.argsort(single, stable=True)]
        sizes = [k for k in sorted({int(2 ** (e / 2)) for e in range(0, 48)}) if k <= len(ranked)]
        pre = S[None].repeat(len(sizes), 1)
        for i, k in enumerate(sizes):
            pre[i, ranked[:k]] = False
        ok = [k for k, x in zip(sizes, measure(pre).tolist()) if x <= threshold]
        if not ok:
            break
        S[ranked[:max(ok)]] = False
    return S


def gated_position(nat, ids, targets, row):
    t0 = time.perf_counter()
    p = row["position"]
    positions = sorted(set(row["position_sets"][0]))
    search = [ids[:p] + [z] + ids[p + 1:] for z in row["alternatives"]]
    held = prompts_at(nat, ids, p, (3, 4, 5))
    later = prompts_at(nat, ids, p, (6, 7, 8))
    units = sorted({u for xs in (search, held, later) for u in GatedCarry(nat, ids, targets, xs, positions).units})
    gs, gh, gl = (GatedCarry(nat, ids, targets, xs, positions, units) for xs in (search, held, later))
    g6 = GatedCarry(nat, ids, targets, search + held, positions, units)
    N, dev = len(units), nat.dev
    zero, full = torch.zeros(N, dtype=torch.bool, device=dev), torch.ones(N, dtype=torch.bool, device=dev)
    empty = float(gs.measure(zero)[0])
    out = {"position": p, "positions": positions, "units": N, "empty_gated_bits": empty,
           "full_gated_bits": float(gs.measure(full)[0]),
           "movement_model_bits": float(torch.stack([carry.kl(r, nat.reference([ids], targets))[0] for r in gs.ref_model]).mean()),
           "gated_vs_model_bits": float(torch.stack([carry.kl(m, g)[0] for m, g in zip(gs.ref_model, gs.ref_gated)]).mean())}
    if empty <= 1e-5:
        out["seconds"] = time.perf_counter() - t0
        return out, None

    def search_set(g, e):
        threshold = (1 - FRACTION) * e
        order = torch.argsort(g.change(), descending=True)
        sizes = sorted({min(N, int(2 ** (k / 2))) for k in range(0, 48) if 2 ** (k / 2) <= 2 * N} | {N})
        kl = g.measure(prefix_sets(order, sizes, N, dev)).tolist()
        start = min([s for s, x in zip(sizes, kl) if x <= threshold], default=N)
        S = torch.zeros(N, dtype=torch.bool, device=dev)
        S[order[:start]] = True
        return start, refine(g.measure, S, threshold)

    def share(g, S):
        e = float(g.measure(zero)[0])
        return 1 - float(g.measure(S)[0]) / e if e > 1e-5 else None

    start, S = search_set(gs, empty)
    size = int(S.sum())
    start6, S6 = search_set(g6, float(g6.measure(zero)[0]))
    out.update(search0_5={"prefix_90": start6, "size_90": int(S6.sum()), "carried_ranks6_8": share(gl, S6)})
    out.update(prefix_90=start, size_90=size, carried_search=share(gs, S),
               carried_ranks3_5=share(gh, S), carried_ranks6_8=share(gl, S),
               kl_model_bits={"set": float(gs.measure(S, "model")[0]), "all_live": float(gs.measure(full, "model")[0]),
                              "none_live": float(gs.measure(zero, "model")[0])},
               seconds=time.perf_counter() - t0)
    return out, [units[j] for j in S.nonzero().flatten().tolist()]


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
    ap.add_argument("--gated", action="store_true", help="carry in VPD's gated model, with measured removal rounds")
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
        if args.gated:
            for row in r["positions"][:args.positions]:
                pos, units = gated_position(nat, ids, targets, row)
                pos["units_best"] = units
                result.append(pos)
                print(f"{r['task']} p={pos['position']} CI>0 units={pos['units']} gated empty={pos['empty_gated_bits']:.4g} "
                      f"prefix90={pos.get('prefix_90')} refined90={pos.get('size_90')} held-out carried={pos.get('carried_ranks3_5')} "
                      f"{pos['seconds']:.1f}s", flush=True)
            with args.out.open("a") as f:
                f.write(json.dumps({"task": r["task"], "targets": targets, "semantics": "gated", "positions": result,
                                    "seconds": time.perf_counter() - t0}) + "\n")
            drop_plans(nat)
            continue
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
        drop_plans(nat)


if __name__ == "__main__":
    main()
