"""The fast circuit builder (the user's 10-14 plan: circuits come from a non-LLM builder that is fast and good, judged
on the great-circuit scorecard beside its seconds per text; the teacher, teacher.set_search, is the slow careful
reference). Attribution only ranks; every claim is measured.

build(ge, task, units="vpd") -> circuit JSON (goal_eval.to_json, with "build": the builder's record). One entry
point for every kind of unit: UNITS maps a name to its builder (vpd here; other units register their own).

The VPD builder on one text:
1. positions: the targets and the context positions by measured resampling (influence: each position's token alone
   replaced by far donors' tokens, GoalEval.resample_influence's numbers; every position screened with SCREEN_DONORS
   donors, the SHORTLIST largest measured with all of them), largest first until they hold SCOPE_COVER of the summed
   movement, at most START_CAP (the input-scope family's positions);
2. units: VPD's active subcomponents there (the teacher's START set), ranked by the sum of four shares of the size
   of their integrated gradients (IG_STEPS points, the units scaled together from 1 to 0): of the circuit's own-run KL
   from the model on the text (every other subcomponent off, remainders on: point), of the KL from the model on
   ATTR_DRAWS of the scope family's changed texts with everything outside the units held at the text's run (path
   patching's semantics: track and input scope), of the real model's KL with the units removed (N), and of the KL with
   everything outside the units at ATTR_DONORS far donors' activations (activation patching);
3. size: the smallest prefix of that ranking whose measured families meet their floors, the prefixes on a log grid
   (sizes 2^(k/2)): point (the own run's log recovery, floor 0.9 of all the units' own) on every size in one batched
   run, then track_pp, input scope and interchange_pp (path-patched, floor 0.9: VPD's whole answer scores 1 there)
   on PATH_BATCH sizes from the smallest meeting point to every unit, then ROUNDS of
   PATH_BATCH sizes between the largest under its floors and the smallest meeting them (none: every unit);
   then N, activation patching near and far and the omitted family (necessity_spec, donors_search, _omitted_family:
   floor 0.9, and 0.9 of every unit's own patching) on the sizes meeting those, smallest first, a few at a time (none:
   every unit); every floor with MARGIN, the family's noise, and at most every unit's own score (text9004's 407-part
   circuit at 17 positions, built without the omitted family, scored it -0.1 on eval against 0.9);
4. chunks: the chosen units in runs of 1/CHUNKS of them, the inert first (each kept unit removed alone, its own-run
   KL change under INERT_BITS, measured in one batch), then the rest, least important first; PATH_BATCH runs at a time
   each removed from the current set and measured on the cheap families, the passing ones removed together when that
   passes too, else the least important alone; then every family on the result, and failing, back to the longest
   prefix of the removals that passes (ROUNDS bracketing rounds).
The full reward row is the scorecard's, never run here."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import carry
import goal_eval
import native

IG_STEPS = 4        # attribution: points on the path from every unit on to every unit off
ATTR_DRAWS = 16     # attribution: the scope family's changed texts beside the text, one per position
ATTR_ROWS = 6       # attribution: texts per backward pass
ATTR_DONORS = 4     # attribution: far donors for activation patching's share
FLOOR = goal_eval.FLOOR_TOP
INERT_BITS = 1e-3   # an inert unit: its removal moves the own-run KL at the targets by less (the scorecard's)
PATH_BATCH = 4      # prefix sizes per path-patched round
CHUNKS = 8          # the trim: the chosen units in runs of this share of them
ROUNDS = 3          # bracketing rounds after the first
REFINE = 0.05       # a bracket this narrow (hi / lo) ends the rounds
MARGIN = {"track_pp": 0.02, "scope": 0.05, "interchange_pp": 0.01, "patching_near": 0.02, "patching_far": 0.07, "N": 0.0}
# over each family's floor (the lead's 10-14: a circuit at its search floors sits at floor +- noise on held-out draws;
# at none the builder's dev circuits missed eval scope by 0.006-0.021 on 3 of 5): twice the robust SD (1.4826 MAD) of
# the eval-minus-search change in (score - floor) over the 20 dev teacher circuits at 6a9e1d3 (track 0.008, interchange
# 0.002, scope 0.023, patching near 0.009 and far 0.035, N 0; point and the rest do not depend on the split). The
# heavy tails (scope -0.42 on text9008, N -0.36) are a position or control set the circuit lacks, no margin's business
SCREEN_DONORS = 2   # influence: donors per position in the screen over every position
SHORTLIST = 32      # influence: the screen's largest positions measured with every donor


def _hidden_logq(nat: native.Native, xs: list[list[int]], targets: list[int], masks: dict) -> torch.Tensor:
    """log q at the targets [B, len(targets), V] of VPD's masked forward on token lists xs (one per row), masks
    {site: [B, T, C]} (differentiable), remainders on."""
    tg = nat.target
    B, T = len(xs), len(xs[0])
    for n in nat.names:
        st = tg.site(n)
        st.mask, st.delta_mask = masks[n], torch.ones(B, T, device=nat.dev)
    try:
        h = tg.hidden(torch.tensor(xs, device=nat.dev))
    finally:
        nat.vpd.clear()
    return torch.log_softmax(tg.unembed(h[:, targets]).float(), -1)


def attribution(ge: goal_eval.GoalEval, tx: goal_eval.Text, units: list, xs: list[list[int]], steps: int = IG_STEPS,
                others: float = 0.0) -> list[float]:
    """Per unit (site, position, subcomponent) its integrated gradient of the KL from the model, summed over the token
    lists xs: the units scaled together from 1 (every one on) to 0, every other subcomponent at others (0: the
    circuit's own run; 1: the real model, the necessity family's removal), remainders on; minus the integral of
    d KL / d m, each unit's share of the KL its removal with the others adds."""
    nat, T = ge.nat, tx.T
    on = {n: torch.zeros(T, nat.C[n], device=nat.dev) for n in nat.names}
    for n, t, c in units:
        on[n][t, c] = 1.0
    rest = {n: (1.0 - on[n]) * others for n in nat.names}
    ref = nat.reference(xs, tx.targets)
    total = {n: torch.zeros(T, nat.C[n], device=nat.dev) for n in nat.names}
    for k in range(steps):
        a = 1.0 - (k + 0.5) / steps
        for r0 in range(0, len(xs), ATTR_ROWS):
            rows = xs[r0:r0 + ATTR_ROWS]
            masks = {n: (on[n] * a)[None].repeat(len(rows), 1, 1).requires_grad_() for n in nat.names}
            with torch.enable_grad():
                kl = carry.kl(ref[r0:r0 + ATTR_ROWS], _hidden_logq(nat, rows, tx.targets, {n: masks[n] + rest[n] for n in nat.names})).sum()
                grads = torch.autograd.grad(kl, [masks[n] for n in nat.names])
            for n, gr in zip(nat.names, grads):
                total[n] -= gr.sum(0) / steps
            del masks, grads
            native.free(nat.dev)
    return [float(total[n][t, c]) for n, t, c in units]


def held_attribution(ge: goal_eval.GoalEval, tx: goal_eval.Text, units: list, rows: list, steps: int = IG_STEPS) -> list[float]:
    """Per unit its integrated gradient, summed over rows, of the KL at the targets from the model on each row's
    tokens, every subcomponent's activation a_h + m (a - a_h) (a its value in this run, a_h the row's held value; from
    the row's first live position on), the units' m scaled together from 1 (live) to 0 (held), every other
    subcomponent held. rows: (token ids, the held activations {site: [T, C]}, the held remainders {site: [T, d_out]}
    or None for live ones, the first held position, the reference log p at the targets [targets, V])."""
    nat, tg, T = ge.nat, ge.nat.target, tx.T
    on = {n: torch.zeros(T, nat.C[n], device=nat.dev) for n in nat.names}
    for n, t, c in units:
        on[n][t, c] = 1.0
    total = {n: torch.zeros(T, nat.C[n], device=nat.dev) for n in nat.names}
    own = {n: tg.site(n).__dict__.get("_forward") for n in nat.names}   # (native's own per-site forward, put back)
    nat.vpd.clear()
    try:
        for r0 in range(0, len(rows), ATTR_ROWS):
            part = rows[r0:r0 + ATTR_ROWS]
            B = len(part)
            ah = {n: torch.stack([r[1][n] for r in part]) for n in nat.names}             # [B, T, C]
            live_rem = all(r[2] is None for r in part)
            rh = None if live_rem else {n: torch.stack([r[2][n] for r in part]) for n in nat.names}
            held = torch.zeros(B, T, 1, device=nat.dev)                                  # 1 from each row's first held position
            for b, r in enumerate(part):
                held[b, r[3]:] = 1.0
            ref = torch.stack([r[4] for r in part])
            for k in range(steps):
                a = 1.0 - (k + 0.5) / steps
                m = {n: (on[n] * a)[None].repeat(B, 1, 1).requires_grad_() for n in nat.names}
                for n in nat.names:
                    st = tg.site(n)

                    def fwd(x, st=st, n=n):
                        live = x @ st.V
                        w = 1.0 - held * (1.0 - m[n])                # 1 before the held positions, m from them on
                        act = ah[n] + w * (live - ah[n])
                        rem = x @ (st.W - (st.V @ st.U).T).T if rh is None else rh[n]
                        return act @ st.U + rem
                    st._forward = fwd
                with torch.enable_grad():
                    lq = torch.log_softmax(tg.unembed(tg.hidden(torch.tensor([r[0] for r in part], device=nat.dev))[:, tx.targets]).float(), -1)
                    grads = torch.autograd.grad(carry.kl(ref, lq).sum(), [m[n] for n in nat.names])
                for n, gr in zip(nat.names, grads):
                    total[n] -= gr.sum(0) / steps
                del m, grads
                native.free(nat.dev)
    finally:
        for n in nat.names:
            if own[n] is None:
                tg.site(n).__dict__.pop("_forward", None)
            else:
                tg.site(n)._forward = own[n]
    return [float(total[n][t, c]) for n, t, c in units]


@torch.no_grad()
def _text_held(ge: goal_eval.GoalEval, tx: goal_eval.Text) -> tuple[dict, dict]:
    """The text's own run: every subcomponent's activation {site: [T, C]} and remainder output {site: [T, d_out]}."""
    nat, tg = ge.nat, ge.nat.target
    nat.vpd.clear()
    for n in nat.names:
        tg.site(n).cache_input = True
    try:
        tg.hidden(torch.tensor([tx.ids], device=nat.dev))
        ax, rx = {}, {}
        for n in nat.names:
            st = tg.site(n)
            xin = st.last_input[0]
            ax[n], rx[n] = xin @ st.V, xin @ (st.W - (st.V @ st.U).T).T
    finally:
        for n in nat.names:
            tg.site(n).cache_input, tg.site(n).last_input = False, None
    return ax, rx


def path_attribution(ge: goal_eval.GoalEval, tx: goal_eval.Text, units: list, xs: list[list[int]], steps: int = IG_STEPS) -> list[float]:
    """held_attribution on the changed texts xs with everything held at the text's run (path patching's semantics, the
    track and input-scope families': the units follow the change, the rest stays; held, the targets' prediction is the
    model's on the text, at KL W from the changed text's)."""
    ax, rx = _text_held(ge, tx)
    ref = ge.nat.reference(xs, tx.targets)
    return held_attribution(ge, tx, units, [(x, ax, rx, 0, ref[i]) for i, x in enumerate(xs)], steps)


def donor_attribution(ge: goal_eval.GoalEval, tx: goal_eval.Text, units: list, k: int = ATTR_DONORS, steps: int = IG_STEPS) -> list[float]:
    """held_attribution on the text with everything held at k far donors' activations (activation patching's semantics,
    goal_eval.donors' far pool: a donor aligned at its end, its first missing positions live, remainders live)."""
    rows = []
    for _, x in ge.far_donors(tx, ge.far_pool("search"), k):
        acts, pad = ge.donor_activations(list(x), tx.T)
        rows.append((list(tx.ids), {n: v.to(ge.nat.dev) for n, v in acts.items()}, None, pad, tx.ref[0]))
    return held_attribution(ge, tx, units, rows, steps) if rows else [0.0] * len(units)


@torch.no_grad()
def _resampled(ge: goal_eval.GoalEval, tx: goal_eval.Text, qs: list[int], donors: list, group: int = 8) -> dict:
    """{q: mean KL in bits at the targets over donors of the model with q's token alone replaced by each donor's
    (GoalEval.resample_influence's numbers)}, from the text's own run: a changed text differs at q alone, so the model
    runs only from q on (Target.hidden with the text's keys and values before it), group positions a call from the
    group's first one."""
    nat, tg, T = ge.nat, ge.nat.target, tx.T
    out, n = {}, len(donors)
    if not donors or not qs:
        return out
    nat.vpd.clear()
    base = torch.tensor([tx.ids], device=nat.dev)
    kv = tg.keys_values(base)
    qs = sorted(qs)
    for g0 in range(0, len(qs), group):
        part = qs[g0:g0 + group]
        s0 = part[0]
        x = base.repeat(len(part) * n, 1)
        for i, q in enumerate(part):
            x[i * n:(i + 1) * n, q] = torch.tensor([goal_eval._donor_token(d, q, T, tx.ids[q]) for d in donors], device=nat.dev)
        pre = [tuple(w.expand(len(x), *w.shape[1:]) for w in layer) for layer in kv]
        h = tg.hidden(x, s0, pre)                            # [rows, T - s0, d]
        sel = [j for j, t in enumerate(tx.targets) if t >= s0]   # (a target before s0 sees no change)
        lq = torch.log_softmax(tg.unembed(h[:, [tx.targets[j] - s0 for j in sel]]).float(), -1)
        out.update(zip(part, carry.kl(lq, tx.ref[:, sel].expand_as(lq)).view(len(part), n).mean(1).tolist()))
    native.free(nat.dev)
    return out


def influence(ge: goal_eval.GoalEval, tx: goal_eval.Text, split: str = "search", k: int | None = None) -> dict:
    """Each context position's resampling influence (GoalEval.resample_influence's numbers): every position with the
    first SCREEN_DONORS of its k far donors, then the SHORTLIST largest with all k (the rest keep their screened
    value: the input-scope family's positions are the largest few). Kept on tx where resample_influence keeps its
    own, so the input-scope family's draws (scope_draws) read it."""
    k = goal_eval.RESAMPLE_DONORS if k is None else k
    key = (split, k, ge.FIRST_POSITION)
    kept = tx.__dict__.setdefault("_resample", {})
    if key not in kept:
        donors = [list(d) for _, d in ge.far_donors(tx, ge.far_pool(split), k)]
        tg_ = set(tx.targets)
        qs = [q for q in range(0 if ge.FIRST_POSITION else 1, max(tx.targets) + 1) if q not in tg_]
        one = _resampled(ge, tx, qs, donors[:SCREEN_DONORS])
        top = sorted(one, key=lambda q: -one[q])[:SHORTLIST]
        kept[key] = {**one, **_resampled(ge, tx, top, donors)}
    return kept[key]


def _graph(units: list) -> native.Graph:
    return native.Graph(sorted(units), complete=True)


def build_vpd(ge: goal_eval.GoalEval, tx: goal_eval.Text, log=None) -> tuple[native.Graph, dict]:
    """(the circuit, the record) on one text (module docstring)."""
    import teacher

    nat, t0, rec = ge.nat, time.perf_counter(), {"units": "vpd"}
    clock = lambda name: rec.setdefault("seconds", {}).__setitem__(name, round(time.perf_counter() - t0, 2))  # noqa: E731
    one = influence(ge, tx, "search")                        # 1. positions
    total = sum(max(0.0, v) for v in one.values())
    pos, held = [], 0.0
    for q in sorted(one, key=lambda q: -one[q]):
        if len(pos) >= goal_eval.START_CAP or not total > 0 or held >= goal_eval.SCOPE_COVER * total:
            break
        pos.append(q)
        held += max(0.0, one[q])
    P = sorted(set(tx.targets) | set(pos))
    rec["positions"] = P
    clock("positions")
    start = {n: tx.vpd[n].clone() for n in nat.names}
    for n in nat.names:
        start[n][[t for t in range(tx.T) if t not in set(P)]] = False
    pr = teacher.Pruner(ge, tx, start, live=True)           # the own run's KLs of unit subsets, batched
    units = pr.nodes
    draws = ge.scope_draws(tx, "search")                     # 2. the ranking
    seen, xs = set(), [list(tx.ids)]
    for p, z in draws:
        if p not in seen and len(xs) <= ATTR_DRAWS:
            seen.add(p)
            x2 = list(tx.ids)
            x2[p] = z
            xs.append(x2)
    own = np.abs(attribution(ge, tx, units, xs[:1]))         # the own run on the text (point)
    path = np.abs(path_attribution(ge, tx, units, xs[1:])) if len(xs) > 1 else np.zeros(len(units))   # the change
    need = np.abs(attribution(ge, tx, units, xs[:1], others=1.0))   # their removal from the real model (N)
    patch = np.abs(donor_attribution(ge, tx, units))         # the rest at far donors' activations (patching)
    score = sum(x / max(x.sum(), 1e-12) for x in (own, path, need, patch))   # each its share of its own total
    rank = np.argsort(-score, kind="stable").tolist()        # unit ids, largest first, by |attribution|: a unit that
    # cancels another's overshoot has a negative one, and ranked by its sign it came last (text9006: point 0.66 up to
    # 1,448 of 1,886 units, 0.997 with all; by size 0.93 at 362)
    order = [units[i] for i in rank]
    rec["start_units"] = len(units)
    clock("ranking")
    N = len(order)                                           # 3. the size
    sizes = sorted({min(N, int(round(2 ** (k / 2)))) for k in range(0, int(2 * math.log2(max(N, 1))) + 2)} | {N})
    e0, v0 = ge._bounds(tx, "live")
    rk = torch.tensor(rank, device=nat.dev)

    def prefix(n):                                           # the first n units of the ranking as Pruner's on
        on = torch.zeros(N, dtype=torch.bool, device=nat.dev)
        on[rk[:n]] = True
        return on
    point = [goal_eval._fid_raw(e0, k, v0) for k in pr.kl_sets(torch.stack([prefix(n) for n in sizes]))]
    rec["point_curve"] = [[n, x] for n, x in zip(sizes, point)]
    ref_point = point[-1]
    f_point = None if ref_point is None else FLOOR * max(0.0, min(1.0, ref_point))
    floors = rec["floors"] = {"point": f_point, **{f: FLOOR + MARGIN[f] for f in ("track_pp", "scope", "interchange_pp")}}
    rec["kl_cap"] = None if f_point is None else e0 * (max(v0, goal_eval.KL_RESOLUTION) / e0) ** f_point   # point's floor
    # as a KL (_fid_raw(e0, kl_cap, v0) = f_point): connect's point criterion
    ok_point = [f_point is None or (x is not None and x >= f_point - 1e-9) for x in point]
    first = next((i for i, ok in enumerate(ok_point) if ok), len(sizes) - 1)
    clock("point")

    fam_s = rec.setdefault("family_seconds", {"track_pp": 0.0, "scope": 0.0, "interchange_pp": 0.0})

    def families(gs):                                        # the path-patched families of graphs, batched
        f0 = time.perf_counter()
        tr = ge.track_pp(tx, gs, "search")
        f1 = time.perf_counter()
        sc = ge.scope_family(tx, gs, "search")
        f2 = time.perf_counter()
        ic = ge.interchange_pp(tx, gs, "search")
        fam_s["track_pp"] += f1 - f0
        fam_s["scope"] += f2 - f1
        fam_s["interchange_pp"] += time.perf_counter() - f2
        return [{"track_pp": goal_eval.family_score(a, "track_pp"), "scope": goal_eval.family_score(b, "scope"),
                 "interchange_pp": goal_eval.family_score(c, "interchange_pp")} for a, b, c in zip(tr, sc, ic)]
    meets = lambda fam: all(v is None or floors[f] is None or v >= floors[f] - 1e-9 for f, v in fam.items())  # noqa: E731
    tx.f2                                                    # (the probe behind track_pp's and the panel's positions)
    clock("probe")
    tried, at = [], {}                                       # size -> met; bracketing rounds on the log scale

    def measure(ns):
        ns = [m for m in sorted(set(ns)) if m not in at]
        if not ns:
            return
        pm = [goal_eval._fid_raw(e0, k, v0) for k in pr.kl_sets(torch.stack([prefix(m) for m in ns]))]
        ok = [i for i, x in enumerate(pm) if f_point is None or (x is not None and x >= f_point - 1e-9)]
        for i in range(len(ns)):
            at[ns[i]] = False
        for i, fam in zip(ok, families([_graph(order[:ns[i]]) for i in ok]) if ok else []):
            fam = {"point": pm[i], **fam}
            at[ns[i]] = meets(fam)
            tried.append({"size": ns[i], **fam, "met": at[ns[i]]})
    lo, hi = (sizes[first - 1] if first > 0 else 0), N       # lo fails point (or is empty); hi, every unit, is measured
    measure([int(round(max(sizes[first], 1) * (N / max(sizes[first], 1)) ** (j / (PATH_BATCH - 1)))) for j in range(PATH_BATCH)])
    whole = next((t for t in tried if t["size"] == N), None)  # every unit's own scores cap the floors (the teacher's
    if whole is not None:                                    # acceptance: min(the reward's floor, the START's own))
        for f in ("track_pp", "scope", "interchange_pp"):
            if whole[f] is not None:
                floors[f] = min(floors[f], whole[f])
        for t in tried:
            at[t["size"]] = t["met"] = meets({f: t[f] for f in floors if f in t})
    for _ in range(ROUNDS):
        met = sorted(m for m, ok in at.items() if ok)
        if not met:
            break
        hi = met[0]
        lo = max([m for m, ok in at.items() if not ok and m < hi] + [lo])
        if hi <= max(lo, 1) * (1 + REFINE):
            break
        base = max(lo, 1)
        measure([int(round(base * (hi / base) ** ((j + 1) / (PATH_BATCH + 1)))) for j in range(PATH_BATCH)])
    met = sorted(m for m, ok in at.items() if ok)
    clock("size")

    def extra(gs):                                           # N, activation patching and the omitted family: the
        nn_, dn = ge.necessity_spec(tx, gs, "search"), ge.donors_search(tx, gs)   # costlier families
        om = ge._omitted_family(tx, [goal_eval._c_graph(g, tx.targets, nat.C) for g in gs])
        return [{f: goal_eval.family_score(r, f) for f, r in (("N", a), ("patching_near", b), ("patching_far", b), ("omitted", c))}
                for a, b, c in zip(nn_, dn, om)]
    met = met or [N]                                         # the sizes meeting the cheap floors, smallest first
    cand = met[:PATH_BATCH - 1]
    got = extra([_graph(order[:m]) for m in cand] + [_graph(order)])
    ref = got[-1]                                            # every unit: patching's reference (VPD at the positions)
    floors.update({"N": FLOOR + MARGIN["N"], "omitted": FLOOR,
                   **{f: (None if ref[f] is None else FLOOR * max(0.0, min(1.0, ref[f])) + MARGIN[f]) for f in ("patching_near", "patching_far")}})
    for f in ("N", "patching_near", "patching_far", "omitted"):   # (capped at every unit's own, as above)
        if floors[f] is not None and ref[f] is not None:
            floors[f] = min(floors[f], ref[f])
    rec["extra"] = [{"size": m, **x} for m, x in zip(cand + [N], got)]
    ok = [m for m, x in zip(cand, got) if meets(x)]
    rest = [m for m in met if m > (cand[-1] if cand else N) and m < N]
    while not ok and rest:                                   # the next larger sizes meeting the cheap floors
        more, rest = rest[:PATH_BATCH], rest[PATH_BATCH:]
        got = extra([_graph(order[:m]) for m in more])
        rec["extra"] += [{"size": m, **x} for m, x in zip(more, got)]
        ok = [m for m, x in zip(more, got) if meets(x)]
    pick = ok[0] if ok else (N if meets({f: v for f, v in ref.items()}) else None)
    rec["tried"] = tried
    n = pick if pick is not None else N
    rec["met"], rec["start_written"] = pick is not None, n == N
    clock("extra")
    on = prefix(n)                                           # 4. chunks of the ranking removed, measured

    def every(ons):                                          # every family the builder measures, on unit sets: the
        pts = [goal_eval._fid_raw(e0, k, v0) for k in pr.kl_sets(torch.stack(ons))]   # costlier ones on the sets the
        gs = [_graph([units[i] for i in o.nonzero().flatten().tolist()]) for o in ons]   # cheap ones admit alone
        out = [{"point": x, **f} for x, f in zip(pts, families(gs))]
        alive = [j for j, f in enumerate(out) if meets(f)]
        for j, x in zip(alive, extra([gs[j] for j in alive]) if alive else []):
            out[j].update(x)
        return [f if j in alive else None for j, f in enumerate(out)]
    def cheap(ons):                                          # point and the path-patched families of unit sets
        pts = [goal_eval._fid_raw(e0, k, v0) for k in pr.kl_sets(torch.stack(ons))]
        return [{"point": x, **f} for x, f in zip(pts, families([_graph([units[i] for i in o.nonzero().flatten().tolist()]) for o in ons]))]

    def trim(groups):                                        # groups of unit ids, least important first, removed while
        nonlocal on                                          # the cheap families hold (PATH_BATCH a call, the passing
        start_on, seq = on.clone(), []                       # ones together, else the first alone); then the costlier
        for c0 in range(0, len(groups), PATH_BATCH):         # families on the result, failing: back to the longest
            part = [g for g in groups[c0:c0 + PATH_BATCH] if bool(on[g].any())]   # passing prefix of the removals
            if not part:
                continue
            ons = []
            for g in part:
                o = on.clone()
                o[g] = False
                ons.append(o)
            okc = [g for g, f in zip(part, cheap(ons)) if meets(f)]
            if len(okc) > 1:
                o = on.clone()
                for g in okc:
                    o[g] = False
                if meets(cheap([o])[0]):
                    on, seq = o, seq + okc
                    continue
            if okc:
                on = on.clone()
                on[okc[0]] = False
                seq.append(okc[0])
        if not seq:
            return 0

        def upto(k):
            o = start_on.clone()
            for g in seq[:k]:
                o[g] = False
            return o
        lo, hi = 0, len(seq)
        f = every([upto(hi)])[0]
        if f is not None and meets(f):
            lo = hi
        for _ in range(ROUNDS):
            if hi - lo <= 1:
                break
            ks = sorted({lo + max(1, (hi - lo) * (j + 1) // (PATH_BATCH + 1)) for j in range(PATH_BATCH)} - {lo, hi})
            for k, f in zip(ks, every([upto(k) for k in ks])):
                if f is not None and meets(f):
                    lo = max(lo, k)
                elif k > lo:
                    hi = min(hi, k)
        on = upto(lo)
        return sum(len(g) for g in seq[:lo])
    pr.on = on.clone()                                       # the inert units: their own-run KL change alone
    ids = on.nonzero().flatten().tolist()
    base = pr.kl()
    moved = [abs(k - base) for k in pr.without([torch.tensor([i], device=nat.dev) for i in ids])] if ids else []
    where = {u: j for j, u in enumerate(rank)}
    inert = sorted((i for i, d in zip(ids, moved) if d < INERT_BITS), key=lambda i: -where[i])   # least important first
    rest = [i for i in rank if bool(on[i]) and i not in set(inert)][::-1]
    rec["inert"], rec["chunks"] = len(inert), {"removed": 0}
    if rec["met"] and len(ids) > 1:                          # inert units first, then the rest, least important first
        K = max(1, -(-len(ids) // CHUNKS))
        groups = [inert[j:j + K] for j in range(0, len(inert), K)] + [rest[j:j + K] for j in range(0, len(rest), K)]
        rec["chunks"]["removed"] = trim(groups[:CHUNKS + 1])
    clock("trim")
    kept = [units[i] for i in on.nonzero().flatten().tolist()]
    rec["size"] = len(kept)
    if log:
        log(f"  builder_fast: {len(kept)} of {N} units at {len(P)} positions, met {rec['met']}, {rec['seconds']}")
    return _graph(kept), rec


UNITS = {"vpd": build_vpd}


def build(ge: goal_eval.GoalEval, task: dict, units: str = "vpd", seed: int | None = None, log=None,
          builder: str = "builder_fast", stated: bool = False) -> dict:
    """A circuit for one task (a vpd4l task JSON: its first prompt's token_ids and target_positions; seed, the task's
    own, default from its first tokens) in units (UNITS): goal_eval.to_json of the circuit, with "units", its record
    under "build", "build_seconds" (every stage, the text's runs included) and "builder" (name@rev). stated: its
    connections stated by connect.connect (rustmove's), at the builder's own floors, its record under "connect"."""
    p = task["prompts"][0]
    t0 = time.perf_counter()
    tx = ge.text(p["token_ids"], p["target_positions"], seed, task=task)
    t1 = time.perf_counter()
    g, rec = UNITS[units](ge, tx, log=log)
    rec["seconds_text"] = round(t1 - t0, 2)
    if stated and units == "vpd":
        import connect

        fl = rec["floors"]
        g = connect.connect(ge, tx, g, "vpd", floors={"point": rec.get("kl_cap"), "track": fl.get("track_pp"), "scope": fl.get("scope"),
                                                      "interchange": fl.get("interchange_pp"), "patching_near": fl.get("patching_near"),
                                                      "patching_far": fl.get("patching_far")}, log=log)
        rec["connect"] = getattr(g, "connect", None)
        rec["seconds"]["connect"] = round(time.perf_counter() - t0, 2)
    out = goal_eval.to_json(g, "")
    out.update(units=units, build=rec, build_seconds=round(time.perf_counter() - t0, 2), builder=builder)
    return out


def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="builder_fast on texts: OUT/TEXT.json per text (build())")
    ap.add_argument("tasks", type=Path, help="a file of task ids (vpd4l texts)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--units", default="vpd", choices=sorted(UNITS))
    ap.add_argument("--rev", default="dev", help="the builder's revision, written as builder_fast@REV")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--warm", default="text6000", help="a text built first and thrown away (CUDA's first kernels)")
    ap.add_argument("--stated", action="store_true", help="state the connections (connect.connect)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native())
    task_of = lambda t: json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())  # noqa: E731
    if a.warm:
        build(goal_eval.GoalEval(ge.nat), task_of(a.warm), a.units, native.task_seed(a.warm))
    for t in a.tasks.read_text().split()[a.offset::a.stride]:
        if (a.out / f"{t}.json").exists():
            continue
        c = build(goal_eval.GoalEval(ge.nat), task_of(t), a.units, native.task_seed(t), log=print, builder=f"builder_fast@{a.rev}",
                  stated=a.stated)
        (a.out / f"{t}.json").write_text(json.dumps(c, default=str))
        print(f"{t}: {len(c['nodes'])} parts, {c['build_seconds']} s", flush=True)
        native.free(ge.nat.dev)


if __name__ == "__main__":
    main()
