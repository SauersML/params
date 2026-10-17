"""Learned masks for circuit_finder's cuts (searchalgo, 10-16): a cut's units proposed by descending a continuous mask on
the patching pair itself, then the passing set found and pruned by measured patching (circuit_finder's rule: the mask
only orders; every keep or drop is a measured run).

A cut (circuit_finder.cut: one kind of unit, neurons or heads, or both, at some positions) gets a mask m_u = sigmoid(theta_u)
in [0, 1] over its units: a unit's value in a run is its live value moved the share m_u of the way to its value in the
other run (m = 1 is circuit_finder's patch, m = 0 leaves the unit alone). Two rows run under the same mask, the clean run
noised and the corrupted run denoised, and theta descends (Adam, STEPS steps from theta = INIT, every unit nearly patched:
the cut's ceiling) on
    relu(TAU - lost) + relu(TAU - restored) + LAMBDA * sum(m)
so a unit stays while the pull of the two hinges on it exceeds LAMBDA (a share of the gap per unit). Then the units are
ordered by their final theta, the smallest passing prefix of that order is found by measurement (circuit_finder.grid's
sizes, then bisection), and units are removed one at a time while the set still passes (circuit_finder.cut's pruning).
Every cut of a question whose ceiling passes is optimized in one batch, two rows a cut. find() is circuit_finder.find
with this search in place of group testing (records in its format, "search": "mask").

Arms on one question set, same card, same records (searchalgo's comparison):
  group  circuit_finder.find as it is (group testing in RelP order)
  mask   the masks above
  attr   the units in RelP attribution order (circuit_finder.Patcher.scores), no group testing: the smallest passing
         prefix and pruning by measurement, as mask's
  am     attr, then masks on the cuts attr leaves unmet
  fam    the units of the family's earlier circuits first (--prior: other questions of the task, family_order), then
         attr's order
A cut's hinge targets are TAU or 0.05 under its ceiling (at least PASS + 0.05), per share.
Multi-source masks (methods, 10-16; constants an arm sets, mask:30:partners=8:allcuts=1:pairwise=1:warm=0.5):
  ALLCUTS   every cut, not only the source, is chosen and verified on the shares over the question's pair and its
            partners (Mean): interchange interventions from several keys, so a cut that only separates the two given
            keys does not pass
  PAIRWISE  the hinges on each pair's shares, averaged over the pairs (a set must pass every pair), in place of the
            hinges on the pairs' mean shares (where overshoot on easy pairs pays for failure on others)
  WARM      masks start from the own pair's RelP attribution (Patcher.scores): theta = INIT + WARM * ln(score / the cut's
            largest score), at least WARM_MIN, so attributed units start patched and the rest nearly unpatched; Adam's
            normalized steps let any unit grow back (the attribution only starts the search)
  ADV       the rest of the cut's units anywhere between the two runs during the descent (optimize_many's adv: VPD's
            ablation of unused parts in any combination, its adversary shared over the question's pairs and persistent
            across steps), so a set that passes only while the rest sits at its base values is not taken
  ADV_CHECK each verified set's adversarial shares: ADV_CHECK steps of the same adversary against the set alone (its units
            fully patched, the rest's rho moved to lower the shares), the worst step's measured shares in the record
--pool QUESTIONS.json: partner keys drawn from that set's keys of the question's task (a seeded order per question),
not from the question's own "partners", which then stay held out for the scorecard's unseen pairs.

  circuit_masks.py questions QUESTIONS.json OUT --arm mask [--split dev --stride 4] [--model Qwen/Qwen3-8B] [--dtype ...]
writes OUT/<id>.json per question.
"""
from __future__ import annotations

import collections
import itertools
import time
import zlib

import numpy as np
import torch

import circuit_finder as cf

STEPS = 30          # (the lead's 10-16 cutover: 30 Adam steps, batched; a5's mask_30)
LR = 0.25
TAU = 0.9           # the hinge's target share (PASS 0.7 is what the measured set must reach)
LAMBDA = 0.003      # a unit's cost in shares of the gap
INIT = 3.0          # sigmoid(3) = 0.95: every unit nearly patched at the start
ORDER_MIN = 64      # units ordered for the measured prefix: those the mask kept (m > 0.5), at least twice that and ORDER_MIN
ORDER_MAX = 20000
MASK_TOKENS = 1536   # rows x positions a mask step runs at most off CUDA (find_batch's parts; on CUDA _budget's)
MASK_MEMORY = 0.5    # the share of free device memory a mask step's activations may take
ROUNDS = 4          # rounds of STEPS a cut's masks get while its order cannot pass (each round from the last's theta)
ALLCUTS = 0         # 1: partners for every cut (module docstring)
PAIRWISE = 0        # 1: hinges per pair
WARM = 0.0          # > 0: attribution start, theta = INIT + WARM * ln(score / max score)
WARM_MIN = -2.0     # sigmoid(-2) = 0.12: the start of units the attribution gives nothing
ADV = 0.0           # > 0: the adversary's step on rho during the descent (module docstring)
ADV_CHECK = 0       # > 0: steps of the measured adversarial check of each verified set
ADV_STEP = 0.25     # the check's step on rho


def _blocks(units: list) -> list:
    """Single units as circuit_finder sets: one unit per (kind, layer, position), its indices a tensor."""
    by = {}
    for u in units:
        kind, i = u[0], u[1]
        p, k = (u[3], u[2]) if kind == "head" else (u[2], u[3])
        by.setdefault((kind, i, p), []).append(int(k))
    return [cf.unit(kind, i, p, torch.tensor(sorted(ks))) for (kind, i, p), ks in by.items()]


def _kinds_of(spec: dict) -> tuple:
    return tuple(spec["kinds"])


def optimize(pt, specs: list, steps: int = STEPS, log=None, taus: list | None = None) -> list:
    """Masks for specs [{"kinds": ("neuron",) | ("head",) | ("neuron", "head"), "positions": [...]}] of the pair set up in
    pt, optimized together (optimize_many's, one job)."""
    return optimize_many([(pt, specs, taus)], steps=steps, log=log)[0]


def optimize_many(jobs: list, steps: int = STEPS, log=None, init: list | None = None, adv: float | None = None,
                  freeze: bool = False) -> list:
    """Masks for several pairs at once: jobs [(pt, specs, taus)], Patchers of one model each set up on its own pair (prompts
    of one length), every spec of every job two rows of one batch per step (a step reads every weight twice whatever
    its rows: batched pairs cost about one). A job's pt may be a list of Patchers, one clean prompt against several
    corrupted ones (partners): its specs then hold one mask over every partner's rows, the hinges on the shares' means
    over the partners. taus: per spec the hinges' targets (lost, restored), default TAU. Per job and spec {kind: theta
    [L, P, width]} (final logits, on the host) and the history of (mean lost, mean restored, sum m) per step. A mask
    edits only its own (row, position) entries: those rows' values gathered, moved toward the other run's and put back.
    init: per job, per spec its theta to start from ({kind: [L, P, width]}, optimize_many's own output; None: INIT).
    adv (default ADV): with a step above 0, every unit's patch is m + (1 - m) rho of the way (the units the mask leaves out
    moved anywhere between their base value and the other run's), rho in [0, 1] one per unit and spec, shared by the
    spec's pairs, starting at 0 and after each step moved adv along the sign of its gradient that raises the hinges
    (persistent projected ascent); freeze: theta not stepped and rho moved to lower the spec's mean shares (the measured
    adversarial check of a set, adversarial()). Each spec's output then also has its final rho."""
    adv = ADV if adv is None else adv
    groups = [list(pt) if isinstance(pt, (list, tuple)) else [pt] for pt, _, _ in jobs]
    pt0 = groups[0][0]
    T = pt0.T
    assert all(pt.T == T for g in groups for pt in g), "optimize_many: prompts of one length"
    L, dev = pt0.L, pt0.dev
    width = {"neuron": pt0.D, "head": pt0.H}
    key = {"neuron": "neu", "head": "head"}
    specs = [(q, j, s) for q, (_, ss, _) in enumerate(jobs) for j, s in enumerate(ss)]      # spec sid: one mask
    sid = {(q, j): n for n, (q, j, _) in enumerate(specs)}
    inst = [(q, j, pt) for q, (_, ss, _) in enumerate(jobs) for pt in groups[q] for j in range(len(ss))]   # instance g:
    kinds = sorted({k for _, _, s in specs for k in s["kinds"]})            # rows 2g (clean, noised), 2g + 1 (denoised)
    cell = {k: {(q, j, p): n for n, (q, j, p) in enumerate((q, j, p) for q, j, s in specs if k in s["kinds"]
                                                            for p in s["positions"])} for k in kinds}   # theta rows
    owner = {k: torch.tensor([sid[(q, j)] for (q, j, _), _ in sorted(cell[k].items(), key=lambda t: t[1])], device=dev)
             for k in kinds}                                                 # theta row -> spec
    # instances run in forwards by their spec's first position (an answer-position cut's rows run that position only,
    # not the source cut's span: on Qwen3-8B the answer-position rows were ~2/3 of a step's tokens)
    first_of = [min(jobs[q][1][j]["positions"]) for q, j, _ in inst]
    passes_ = []
    for start in sorted(set(first_of)):
        gs = [g for g in range(len(inst)) if first_of[g] == start]
        loc = {g: n for n, g in enumerate(gs)}
        ent = {k: [(loc[g], p, cell[k][(inst[g][0], inst[g][1], p)]) for g in gs if k in jobs[inst[g][0]][1][inst[g][1]]["kinds"]
                   for p in jobs[inst[g][0]][1][inst[g][1]]["positions"]] for k in kinds}
        with torch.no_grad():                       # each entry's two rows' other-run values per layer [2N, width]
            oth = {k: [torch.stack([inst[gs[g]][2].act[src][key[k]][i][p] for g, p, _ in ent[k] for src in ("corrupt", "clean")])
                       for i in range(L)] for k in kinds if ent[k]}
            x = torch.stack([inst[g][2]._ids_dev[b] for g in gs for b in ("clean", "corrupt")])
            kv = [tuple(torch.cat([inst[g][2]._base_kv[b][i][jj][:, :, :start] for g in gs for b in ("clean", "corrupt")])
                        for jj in (0, 1)) for i in range(L)] if start > 0 else None
        spans, at = [], 0                           # each Patcher's rows (consecutive)
        for (_, _, pt), grp in itertools.groupby([inst[g] for g in gs], key=lambda t: (t[0], id(t[2]), t[2])):
            n = len(list(grp))
            spans.append((pt, at, at + 2 * n))
            at += 2 * n
        passes_.append({"start": start, "gs": gs, "x": x, "kv": kv, "spans": spans, "others": oth,
                        "rows": {k: torch.tensor([2 * g + b for g, _, _ in ent[k] for b in (0, 1)], device=dev) for k in oth},
                        "pos": {k: torch.tensor([p - start for _, p, _ in ent[k] for _ in (0, 1)], device=dev) for k in oth},
                        "tix": {k: torch.tensor([n for _, _, n in ent[k] for _ in (0, 1)], device=dev) for k in oth}})
    order = torch.tensor([g for ps in passes_ for g in ps["gs"]], device=dev)   # the forwards' instances, in turn
    theta = {k: torch.full((L, len(cell[k]), width[k]), INIT) for k in kinds}
    for q, j, s_ in specs if init else []:          # (a round after a cut its order could not pass: on from its theta)
        th = (init[q] or [None] * len(jobs[q][1]))[j]
        for k in s_["kinds"] if th else []:
            theta[k][:, [cell[k][(q, j, p)] for p in s_["positions"]]] = th[k]
    theta = {k: t.to(dev).requires_grad_(not freeze) for k, t in theta.items()}
    rho = {k: torch.zeros_like(t).requires_grad_(True) for k, t in theta.items()} if adv > 0 else {}
    opt = torch.optim.Adam(list(theta.values()), lr=LR) if not freeze else None
    tau = torch.tensor([t for _, ss, taus in jobs for t in (taus or [(TAU, TAU)] * len(ss))], device=dev)
    cc = torch.tensor([pt.c["clean"] for _, _, pt in inst], device=dev)[order]
    cr = torch.tensor([pt.c["corrupt"] for _, _, pt in inst], device=dev)[order]
    gap = torch.tensor([pt.gap for _, _, pt in inst], device=dev)[order]
    of = torch.tensor([sid[(q, j)] for q, j, _ in inst], device=dev)[order]   # instance -> spec
    count = torch.zeros(len(specs), device=dev).index_add(0, of, torch.ones(len(inst), device=dev))
    hist = [[] for _ in specs]
    for step in range(steps):
        ms = {k: torch.sigmoid(t) for k, t in theta.items()}

        def edit(ps, k, i, h):
            d = h.device
            r, p = ps["rows"][k].to(d), ps["pos"][k].to(d)
            m = ms[k][i].to(d, h.dtype)[ps["tix"][k].to(d)]                    # [2N, width]
            if rho:                                 # (the rest anywhere between the two runs: the adversary's rho)
                m = m + (1 - m) * rho[k][i].to(d, h.dtype)[ps["tix"][k].to(d)]
            sel, o = h[r, p], ps["others"][k][i].to(d, h.dtype)
            if k == "head":
                sel, o, m = sel.view(-1, pt0.H, pt0.hd), o.view(-1, pt0.H, pt0.hd), m[..., None]
            return h.index_put((r, p), (sel + m * (o - sel)).reshape(len(r), -1))
        with torch.enable_grad():
            cs = []
            for ps in passes_:
                ks = ps["others"]
                hn = pt0._last(ps["x"], ps["x"].shape[0], cache=(ps["start"], ps["kv"]) if ps["kv"] is not None else None,
                               neu=(lambda i, h, ps=ps: edit(ps, "neuron", i, h)) if "neuron" in ks else None,
                               head=(lambda i, h, ps=ps: edit(ps, "head", i, h)) if "head" in ks else None,
                               only={"res": set(), "head": set(range(L)) if "head" in ks else set(),
                                     "neu": set(range(L)) if "neuron" in ks else set()})
                cs += [pt._contrast(hn[a:b]) for pt, a, b in ps["spans"]]
            c = torch.cat(cs)
            li, ri = (cc - c[0::2]) / gap, (c[1::2] - cr) / gap               # each pair's shares
            lost = torch.zeros(len(specs), device=dev).index_add(0, of, li) / count
            rest = torch.zeros(len(specs), device=dev).index_add(0, of, ri) / count
            size = sum(torch.zeros(len(specs), device=dev).index_add(0, owner[k], m.sum((0, 2))) for k, m in ms.items())
            if PAIRWISE:                            # (each pair's hinges, averaged over the spec's pairs)
                hinge = torch.zeros(len(specs), device=dev).index_add(
                    0, of, torch.relu(tau[of, 0] - li) + torch.relu(tau[of, 1] - ri)) / count
            else:
                hinge = torch.relu(tau[:, 0] - lost) + torch.relu(tau[:, 1] - rest)
            loss = -(lost + rest).sum() if freeze else hinge.sum() + LAMBDA * size.sum()
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            for t in rho.values():
                t.grad = None
            loss.backward()
        if opt is not None:
            opt.step()
        with torch.no_grad():                       # (rho's ascent on the loss, within [0, 1])
            for t in rho.values():
                if t.grad is not None:
                    t.add_(adv * t.grad.sign()).clamp_(0, 1)
        for n, (a, b, z) in enumerate(zip(lost.tolist(), rest.tolist(), size.tolist())):
            hist[n].append((round(a, 4), round(b, 4), round(z, 1)))
    out, th = [[] for _ in jobs], {k: t.detach().float().cpu() for k, t in theta.items()}
    rh = {k: t.detach().float().cpu() for k, t in rho.items()}
    for n, (q, j, s) in enumerate(specs):
        got = {k: th[k][:, [cell[k][(q, j, p)] for p in s["positions"]]] for k in s["kinds"]}
        out[q].append({"theta": got, "history": hist[n],
                       **({"rho": {k: rh[k][:, [cell[k][(q, j, p)] for p in s["positions"]]] for k in s["kinds"]}} if rh else {})})
        if log:
            a, b, z = hist[n][-1]
            log(f"    mask {'+'.join(s['kinds'])} at {s['positions'][0]}..{s['positions'][-1]}"
                f"{f' (job {q})' if len(jobs) > 1 else ''}{f', {len(groups[q])} partners' if len(groups[q]) > 1 else ''}: "
                f"{steps} steps, lost {a:.2f}, restored {b:.2f}, sum m {z:.1f}")
    return out


def ordered(spec: dict, theta: dict) -> tuple:
    """(units in descending theta, how many the mask kept): ("neuron", layer, pos, k) and ("head", layer, k, pos)."""
    flat, parts = [], []
    for k in _kinds_of(spec):
        t = theta[k]
        flat.append(t.reshape(-1))
        parts.append((k, t.shape))
    v = torch.cat(flat)
    kept = int((v > 0).sum())
    n = min(max(2 * kept, ORDER_MIN), ORDER_MAX, len(v))
    top = v.topk(n).indices.tolist()
    units, bounds, at = [], [], 0
    for k, shp in parts:
        bounds.append((at, at + int(np.prod(shp)), k, shp))
        at += int(np.prod(shp))
    for f in top:
        for lo, hi, k, (L, P, W) in bounds:
            if lo <= f < hi:
                r = f - lo
                i, pi, w = r // (P * W), (r // W) % P, r % W
                units.append(cf.unit(k, i, spec["positions"][pi], w))
                break
    return units, kept


def verify(pt, order: list, log=None, name: str = "") -> dict:
    """circuit_finder.cut's measured steps on a proposed order: the smallest passing prefix (grid, bisection to 32 sizes
    between the last failing size and the first passing one, then those in one batch), then units removed one at a time
    while it passes (prune); each unit's shares alone."""
    passes = lambda r: r[0] >= cf.PASS and r[1] >= cf.PASS  # noqa: E731
    g = cf.grid(len(order))                        # (prefixes of order: Patcher.both_prefixes, no set built per unit)
    got = pt.both_prefixes(order, g)
    met = [k for k, r in zip(g, got) if passes(r)]
    ok = None
    if met:
        hi = met[0]
        lo = max([k for k in g if k < hi] + [0])
        while hi - lo > 33:                         # (bisection down to 32 sizes, then all of them in one batch)
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if passes(pt.both_prefixes(order, [mid])[0]) else (mid, hi)
        mids = list(range(lo + 1, hi))
        ok = next((k for k, r in zip(mids, pt.both_prefixes(order, mids) if mids else []) if passes(r)), hi)
    circ = prune(pt, list(order[:ok])) if ok else []
    lost, rest = pt.both([_blocks(circ)])[0] if circ else (0.0, 0.0)
    for k in [k for k in met if k > ok] if circ and not passes((lost, rest)) else []:
        circ = list(order[:k])                      # (a set at the margin can fail measured again, a batch's shape moving
        lost, rest = pt.both([_blocks(circ)])[0]    # a bfloat16 share by ~0.01: Qwen3-8B capital/Spain/p2's readout, 22
        if passes((lost, rest)):                    # units at 0.7003 then 0.6995): the next passing grid size instead
            break
    singles = dict(zip(circ, pt.both([[u] for u in circ]))) if circ else {}
    if log:
        log(f"  {name}: {len(order)} ordered, passing prefix {ok}, pruned to {len(circ)} ({lost:.2f}/{rest:.2f})")
    return {"units": circ, "singles": singles, "met": bool(ok) and passes((lost, rest)), "lost": lost, "restored": rest,
            "found": len(order), "curve": [{"k": k, "lost": a, "restored": b} for k, (a, b) in zip(g, got)]}


def prune(pt, circ: list) -> list:
    """circuit_finder.cut's pruning of a passing set (units whose removal alone still passes, removed one at a time from
    the last while it passes), its sequential trials batched: each round measures the removals so far plus each next
    candidate cumulatively, keeps every removal up to the first that fails, and goes on after that candidate (the same
    trials as one at a time, a round per failing candidate)."""
    passes = lambda r: r[0] >= cf.PASS and r[1] >= cf.PASS  # noqa: E731
    if len(circ) < 2:
        return circ
    alone = pt.both_without(circ)                   # (each unit left out: Patcher.both_without, no set built per unit)
    todo, gone = [u for u, r in zip(circ, alone) if passes(r)][::-1], set()
    while todo:                                     # (each trial a prefix of the units left with todo last, reversed)
        rest = [v for v in circ if v not in gone and v not in set(todo)]
        got = pt.both_prefixes(rest + todo[::-1], [len(rest) + len(todo) - j - 1 for j in range(len(todo))])
        j = next((j for j, r in enumerate(got) if not passes(r)), len(todo))
        gone |= set(todo[:j])
        todo = todo[j + 1:]
    return [v for v in circ if v not in gone]


def _record(res: dict, ceiling, reach: bool, search: list, extra: dict) -> dict:
    return {**res, "eps": None, "floor": [], "ceiling": list(ceiling), "reachable": reach, "refined": [], "conditional": 0,
            "spread": None, "route_groups": None, "searched": search, **extra}


def cuts_of(pt, wanted=None) -> dict:
    """The cuts of the pair set up in pt (wanted: those names only, circuit_finder.find's cuts_wanted; a cut not wanted is
    never searched, so oracleagent's find_circuit without the readout pays nothing for a dense one)."""
    last = pt.T - 1
    out = {"readout": {"kinds": ("neuron",), "positions": [last]}, "movers": {"kinds": ("head",), "positions": [last]}}
    if pt.first < last:
        out["source"] = {"kinds": ("neuron",), "positions": list(range(pt.first, last))}
    return {n: c for n, c in out.items() if wanted is None or n in wanted}


def attr_order(pt, spec: dict, sc: dict) -> list:
    """spec's units in descending RelP attribution (sc: circuit_finder.Patcher.scores), the first ORDER_MAX."""
    v = torch.cat([sc[k][:, spec["positions"]].reshape(-1) for k in spec["kinds"]])
    shapes = [(k, sc[k][:, spec["positions"]].shape) for k in spec["kinds"]]
    units = []
    for f in v.topk(min(ORDER_MAX, len(v))).indices.tolist():
        at = 0
        for k, (L, P, W) in shapes:
            if f < at + L * P * W:
                r = f - at
                units.append(cf.unit(k, r // (P * W), spec["positions"][(r // W) % P], r % W))
                break
            at += L * P * W
    return units


def family_prior(records: list) -> dict:
    """{key: weight} from a family's earlier circuits (circuit_finder records): a unit's key is its kind, layer and index
    with its position counted back from the answer ("end") and, for source units, also on from the first changed token
    ("first"); its weight the number of circuits holding it plus half its largest share among them."""
    prior = {}
    for r in records:
        last, first = r["tokens"] - 1, r["first_diff"]
        units = [("neuron", n["layer"], n["neuron"], n["pos"], max(abs(n["lost"]), abs(n["restored"]))) for n in r.get("nodes", [])]
        units += [("head", h["layer"], h["head"], h["pos"], max(abs(h["lost"]), abs(h["restored"]))) for h in r.get("routes", [])]
        for kind, i, k, p, sh in units:
            for key in {(kind, i, k, "end", last - p)} | ({(kind, i, k, "first", p - first)} if p < last else set()):
                c, w = prior.get(key, (0, 0.0))
                prior[key] = (c + 1, max(w, sh))
    return {key: c + 0.5 * min(w, 1.0) for key, (c, w) in prior.items()}


def family_order(pt, spec: dict, prior: dict) -> list:
    """spec's units that prior's keys name at this pair's positions, by descending weight."""
    last, pos, got = pt.T - 1, set(spec["positions"]), {}
    for (kind, i, k, frm, d), w in prior.items():
        p = last - d if frm == "end" else pt.first + d
        if kind in spec["kinds"] and p in pos:
            u = cf.unit(kind, i, p, k)
            got[u] = max(got.get(u, 0.0), w)
    return sorted(got, key=lambda u: -got[u])


class Mean:
    """Partner pairs measured as one (multi-partner cuts): Patchers set up on one clean prompt against several corrupted
    ones; each set's shares are the means over the pairs (Patcher.both's and both_prefixes' interface, as verify uses)."""

    def __init__(self, pts: list):
        self.pts = list(pts)

    def _all(self, call) -> list:
        """call(pt) for every pair: the first through its own path (a member of an outer Group: find_batch's), the
        partners as a Group of their own (their batches merged, one read)."""
        rest = self.pts[1:]
        return [call(self.pts[0])] + (cf.Group(rest).run([lambda pt=pt: call(pt) for pt in rest]) if rest else [])

    def each(self, sets: list) -> list:
        """Per pair, each set's (lost, restored)."""
        return self._all(lambda pt: pt.both(sets))

    def both(self, sets: list, given=None) -> list:
        rs = self._all(lambda pt: pt.both(sets, given=given))
        return [tuple(float(np.mean([r[i][b] for r in rs])) for b in (0, 1)) for i in range(len(sets))]

    def both_prefixes(self, units: list, ks: list) -> list:
        rs = self._all(lambda pt: pt.both_prefixes(units, ks))
        return [tuple(float(np.mean([r[i][b] for r in rs])) for b in (0, 1)) for i in range(len(ks))]

    def both_without(self, units: list, js: list | None = None) -> list:
        rs = self._all(lambda pt: pt.both_without(units, js))     # (prune's leave-one-out sets: a multi-partner cut
        return [tuple(float(np.mean([r[i][b] for r in rs])) for b in (0, 1)) for i in range(len(rs[0]))]   # crashed)


def search_all(pt, arm: str, log=None, steps: int = STEPS, prior: dict | None = None, partners: list | None = None,
               wanted=None) -> dict:
    """Every cut of the pair set up in pt by arm: circuit_finder.cut's result per cut name. Arms: "mask"; "attr"; "am", attr
    first and masks on the cuts it leaves unmet (one batch); "fam", prior's units (family_order) first, then attr's.
    partners: Patchers set up on pt's clean prompt against other corrupted prompts: the source cut is then chosen and
    verified on the shares' means over pt's pair and theirs (Mean)."""
    st = _before(pt, arm, log=log, prior=prior, partners=partners, wanted=wanted)
    got, names = {}, list(st["masked"])
    for rnd in range(ROUNDS):                       # (a cut its order cannot pass: its masks go on from their theta)
        t0 = time.perf_counter()
        jobs = _jobs(pt, st, names)
        res = optimize_many([j for j, _ in jobs], steps=steps, log=log,
                            init=[[got[n]["theta"] if rnd else st["warm"].get(n) for n in ns] for _, ns in jobs]) if jobs else []
        for (_, ns), r in zip(jobs, res):
            for n, x in zip(ns, r):
                got[n] = {"theta": x["theta"], "history": got.get(n, {}).get("history", []) + x["history"]}
        _verify(pt, st, got, names, time.perf_counter() - t0, steps, log=log)
        names = [n for n in names if not st["out"][n]["met"]]
        if not names:
            break
    return _finish(st)


def _before(pt, arm: str, log=None, prior: dict | None = None, partners: list | None = None, wanted=None) -> dict:
    """search_all up to the masks: every cut's ceiling, attr's (and fam's) cuts, and the cuts left for masks ("masked")
    with their hinge targets; each cut's measurer ("meas": pt, or Mean over pt and partners for the source cut)."""
    specs = cuts_of(pt, wanted)
    meas = {n: pt for n in specs}
    for n in [n for n in specs if ALLCUTS or n == "source"] if partners else []:
        meas[n] = Mean([pt] + list(partners))
    if isinstance(meas.get("source"), Mean):
        specs["source"] = {**specs["source"], "positions": list(range(min(p.first for p in meas["source"].pts), pt.T - 1))}
    names = list(specs)
    width = {"neuron": pt.D, "head": pt.H}
    every = {n: [cf.unit(k, i, p, torch.arange(width[k])) for k in s["kinds"] for i in range(pt.L) for p in s["positions"]]
             for n, s in specs.items()}
    plain = [n for n in names if meas[n] is pt]
    ceil = dict(zip(plain, pt.both([every[n] for n in plain])))
    ceil.update({n: meas[n].both([every[n]])[0] for n in names if n not in ceil})
    reach = {n: ceil[n][0] >= cf.PASS and ceil[n][1] >= cf.PASS for n in names}
    out = {}
    live = [n for n in names if reach[n]]
    if arm in ("attr", "am", "fam") and live:
        t0 = time.perf_counter()
        sc = cf.Patcher.scores(pt)                  # (the class's: find() stubs the instance's)
        for n in live:
            t1 = time.perf_counter()
            order, fam = attr_order(pt, specs[n], sc), []
            if arm == "fam":
                fam = family_order(pt, specs[n], prior or {})
                have = set(fam)
                order = fam + [u for u in order if u not in have]
            out[n] = _record(verify(meas[n], order, log=log, name=n), ceil[n], True, specs[n]["positions"],
                             {"attr": {"seconds": round(time.perf_counter() - (t0 if n == live[0] else t1), 2),
                                       **({"family_units": len(fam)} if arm == "fam" else {})}})
        live = [n for n in live if not out[n]["met"]] if arm == "am" else []
    masked = live if arm in ("mask", "am") else []
    warm = {}
    if WARM and masked:                             # (the masks' start from the own pair's attribution)
        sc = cf.Patcher.scores(pt)
        for n in masked:
            top = max(float(sc[k][:, specs[n]["positions"]].max()) for k in specs[n]["kinds"])
            warm[n] = {k: (INIT + WARM * torch.log(sc[k][:, specs[n]["positions"]].float() / max(top, 1e-30) + 1e-30)
                           ).clamp(WARM_MIN, INIT) for k in specs[n]["kinds"]}
    return {"specs": specs, "names": names, "ceil": ceil, "reach": reach, "out": out, "masked": masked, "meas": meas,
            "warm": warm, "taus": {n: tuple(min(TAU, max(cf.PASS + 0.05, c - 0.05)) for c in ceil[n]) for n in masked}}


def _jobs(pt, st: dict, names: list | None = None) -> list:
    """The masked cuts (names: those of them) as optimize_many jobs, with their names: the pair's own cuts one job, a
    partner cut its own."""
    names = st["masked"] if names is None else names
    plain = [n for n in names if st["meas"][n] is pt]
    out = [((pt, [st["specs"][n] for n in plain], [st["taus"][n] for n in plain]), plain)] if plain else []
    return out + [((st["meas"][n].pts, [st["specs"][n]], [st["taus"][n]]), [n]) for n in names if st["meas"][n] is not pt]


def adversarial(meas, spec: dict, units: list, steps: int | None = None, log=None, name: str = "") -> dict:
    """The set's shares with every other unit of the cut anywhere between the two runs, against an adversary (optimize_many
    with freeze: the set fully patched, the rest's rho moved ADV_STEP along the sign that lowers the spec's mean shares,
    shared over meas's pairs): the step whose measured lost + restored is lowest, and whether it still passes."""
    steps = int(ADV_CHECK if steps is None else steps)   # (an arm's constants arrive as floats: adv_check=8)
    pts = meas.pts if isinstance(meas, Mean) else [meas]
    th = {k: torch.full((pts[0].L, len(spec["positions"]), pts[0].H if k == "head" else pts[0].D), -20.0) for k in spec["kinds"]}
    at = {p: j for j, p in enumerate(spec["positions"])}
    for u in units:
        if u[0] == "head":
            th["head"][u[1], at[u[3]], u[2]] = 20.0
        else:
            th["neuron"][u[1], at[u[2]], u[3]] = 20.0
    hist = optimize_many([(pts if len(pts) > 1 else pts[0], [spec], None)], steps=steps, init=[[th]], adv=ADV_STEP,
                         freeze=True)[0][0]["history"]
    a, b, _ = min(hist, key=lambda h: h[0] + h[1])
    if log:
        log(f"  {name}: adversarial rest ({steps} steps) {a:.2f}/{b:.2f} (unmoved {hist[0][0]:.2f}/{hist[0][1]:.2f})")
    return {"lost": a, "restored": b, "unmoved": list(hist[0][:2]), "steps": steps, "passes": a >= cf.PASS and b >= cf.PASS}


def _verify(pt, st: dict, got: dict, names: list, t_opt: float, steps: int, log=None) -> None:
    """Each named masked cut's order (ordered, from got's theta) through verify by its measurer, into st["out"]."""
    specs, ceil, out = st["specs"], st["ceil"], st["out"]
    for n in names:
        g = got[n]
        t1 = time.perf_counter()
        order, kept = ordered(specs[n], g["theta"])
        res = verify(st["meas"][n], order, log=log, name=n)
        if ADV_CHECK and res["units"]:
            res["adversarial"] = adversarial(st["meas"][n], specs[n], res["units"], log=log, name=n)
        prev = out.get(n, {})
        out[n] = _record(res, ceil[n], True, specs[n]["positions"],
                         {"mask": {"kept": kept, "steps": len(g["history"]), "history": g["history"][::max(1, steps // 20)],
                                   "seconds_opt": round(t_opt + prev.get("mask", {}).get("seconds_opt", 0.0), 2),
                                   "seconds_verify": round(time.perf_counter() - t1 + prev.get("mask", {}).get("seconds_verify", 0.0), 2),
                                   **({"attr_unmet": prev["found"]} if "attr" in prev else
                                      {"attr_unmet": prev["mask"]["attr_unmet"]} if "attr_unmet" in prev.get("mask", {}) else {})}})


def _finish(st: dict) -> dict:
    """Cuts with no result have no units; a partner cut records each pair's shares."""
    specs, ceil, out = st["specs"], st["ceil"], st["out"]
    for n in st["names"]:
        if n not in out:
            out[n] = _record({"units": [], "singles": {}, "met": False, "lost": 0.0, "restored": 0.0, "found": 0, "curve": []},
                             ceil[n], st["reach"][n], specs[n]["positions"], {})
        if isinstance(st["meas"][n], Mean):         # (each pair's shares of the cut; the record's lost and restored: means)
            each = st["meas"][n].each([_blocks(out[n]["units"])]) if out[n]["units"] else []
            out[n]["partners"] = {"pairs": len(st["meas"][n].pts), "shares": [list(e[0]) for e in each]}
    return out


def _find_with(pt, clean, corrupt, answers, log, arm: str, cuts: dict, rec0: dict | None = None, **kw) -> dict:
    """circuit_finder.find's record with cuts (search_all's) in place of its cut() (and rec0, setup's record, in place
    of setting up again)."""
    def cut(pt_, sc, kind, positions, done, log=None, search=None, prior=None):
        c = cuts["movers" if kind == "head" else "readout" if list(positions) == [pt_.T - 1] else "source"]
        return collections.defaultdict(lambda: None, c)     # (fields find() reads that this search has none of: None)
    saved_cut = cf.cut
    cf.cut = cut
    pt.scores = lambda: {}                          # (group testing's order: unused here)
    if rec0 is not None:
        pt.setup = lambda *a, **k: dict(rec0)
    try:
        rec = cf._find(pt, clean, corrupt, answers, log=log, **kw)
    finally:
        cf.cut = saved_cut
        del pt.scores
        if rec0 is not None:
            del pt.setup
    extra = {k: {n: c[k] for n, c in cuts.items() if k in c} for k in ("attr", "mask", "partners", "adversarial")}   # (find() copies its own fields
    return {**rec, "search": arm, **{k: v for k, v in extra.items() if v}}                 # only)


def _budget(pt) -> int:
    """The rows x positions a mask step may run: MASK_MEMORY of the device's free memory over a token's activations kept
    for the backward (every layer's inputs to its projections and the MLP's intermediates), at least 256; MASK_TOKENS off
    CUDA."""
    if pt.dev.type != "cuda":
        return MASK_TOKENS
    free, _ = torch.cuda.mem_get_info(pt.dev)
    free += torch.cuda.memory_reserved(pt.dev) - torch.cuda.memory_allocated(pt.dev)
    cfg, e = pt.core.config, pt.core.model.embed_tokens.weight.element_size()
    per = pt.L * (6 * cfg.hidden_size + 5 * pt.D) * e
    return max(256, int(MASK_MEMORY * free // per))


def _chunks(jobs: list, T: int, budget: int = MASK_TOKENS) -> list:
    """jobs [(item, (pt or pts, specs, taus), names)] in consecutive parts of at most budget rows x positions a step
    (each instance's two rows over its spec's positions from its first: a step's activations stay for the backward, so a
    part's tokens bound its memory; Qwen3-8B, 16 pairs x 4 partners on a 44 GB A40 ran out of memory)."""
    parts, cur, n = [], [], 0
    for jb in jobs:
        pts_ = jb[1][0] if isinstance(jb[1][0], (list, tuple)) else [jb[1][0]]
        t = sum(2 * len(pts_) * (T - min(s["positions"])) for s in jb[1][1])
        if cur and n + t > budget:
            parts.append(cur)
            cur, n = [], 0
        cur.append(jb)
        n += t
    return parts + ([cur] if cur else [])


def find_batch(pts: list, items: list, arm: str = "mask", steps: int = STEPS, log=None, priors: list | None = None,
               partners: list | None = None, cuts_wanted=("readout", "movers", "source")) -> list:
    """find() for several pairs at once: items [(clean, corrupt, answers)], one Patcher each (pts, of one model). Every
    pair is set up and its ceilings (and attr's cuts) measured, then the masks of every pair's masked cuts optimized in
    one batch per prompt length (optimize_many), then each pair's measured steps and record. partners: per item
    [(Patcher, corrupted ids)], other corrupted prompts of its clean prompt's length: each valid one set up and the item's
    source cut chosen on the means over its pair and those (search_all's). Each record's "seconds" is the batch's wall
    time over its pairs."""
    t0 = time.perf_counter()
    recs, sts = [None] * len(items), [None] * len(items)
    rec0s, same_T, part = {}, {}, {}
    for j, (pt, (clean, corrupt, answers)) in enumerate(zip(pts, items)):
        rec0 = pt.setup(clean, corrupt, answers)
        if not rec0["valid"] or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
            recs[j] = {**cf._find(pt, clean, corrupt, answers, log=log, cuts_wanted=cuts_wanted), "search": arm}
            continue
        rec0s[j] = rec0
        same_T.setdefault(pt.T, []).append(j)
        part[j] = []
        for ppt, ids in (partners or [[]] * len(items))[j]:
            if len(ids) == len(clean):
                r = ppt.setup(clean, ids, answers)
                if r["valid"] and r["gap"] >= cf.MIN_GAP and not r["same_answer"]:
                    part[j].append(ppt)
    for T, js in same_T.items():                    # (the pairs of one prompt length as a Group: their measured steps'
        befores = cf.Group([pts[j] for j in js]).run(                       # batches shared, circuit_finder.Group)
            [lambda j=j: _before(pts[j], arm, log=log, prior=(priors or [None] * len(items))[j], partners=part[j],
                                 wanted=cuts_wanted)
             for j in js])
        for j, b in zip(js, befores):
            sts[j] = (rec0s[j], b)
    got = {j: {} for j in range(len(items))}
    names = {j: list(x[1]["masked"]) for j, x in enumerate(sts) if x is not None}
    for rnd in range(ROUNDS):                       # (cuts their orders cannot pass: their masks go on from their theta)
        by_T = {}
        for j, ns in names.items():
            if ns:
                by_T.setdefault(pts[j].T, []).append(j)
        if not by_T:
            break
        t_opt = {}
        for T, js in by_T.items():
            t1 = time.perf_counter()
            jobs = [(j, job, ns) for j in js for job, ns in _jobs(pts[j], sts[j][1], names[j])]
            for part in _chunks(jobs, T, _budget(pts[js[0]])):
                res = optimize_many([job for _, job, _ in part], steps=steps, log=log,
                                    init=[[got[j][n]["theta"] if rnd else sts[j][1]["warm"].get(n) for n in ns]
                                          for j, _, ns in part])
                for (j, _, ns), r in zip(part, res):
                    for n, x in zip(ns, r):
                        got[j][n] = {"theta": x["theta"], "history": got[j].get(n, {}).get("history", []) + x["history"]}
            for j in js:
                t_opt[j] = (time.perf_counter() - t1) / len(js)
        for T, js in by_T.items():                  # (masked cuts' verify as a Group too)
            cf.Group([pts[j] for j in js]).run(
                [lambda j=j: _verify(pts[j], sts[j][1], got[j], names[j], t_opt[j], steps, log=log) for j in js])
        names = {j: [n for n in ns if not sts[j][1]["out"][n]["met"]] for j, ns in names.items()}
    afters = {j: _finish(x[1]) for j, x in enumerate(sts) if x is not None}
    for j, x in enumerate(sts):
        if x is None:
            continue
        clean, corrupt, answers = items[j]
        recs[j] = _find_with(pts[j], clean, corrupt, answers, log, arm, afters[j], rec0=x[0], cuts_wanted=cuts_wanted)
    wall = time.perf_counter() - t0
    for r in recs:
        r.update(seconds_own=r.get("seconds"), seconds=round(wall / len(items), 2), batch=len(items))
    return recs


def find(pt, clean: list[int], corrupt: list[int], answers=None, log=None, arm: str = "mask", steps: int = STEPS,
         prior: dict | None = None, partners: list | None = None, cuts_wanted=("readout", "movers", "source"),
         measure: list | None = None, sets: dict | None = None) -> dict:
    """circuit_finder.find with arm's search in place of group testing (its record, "search": arm); partners: as
    find_batch's, one item's; cuts_wanted, measure and sets: circuit_finder.find's."""
    kw = {"cuts_wanted": cuts_wanted, "measure": measure, "sets": sets}
    if arm == "group":
        return {**cf._find(pt, clean, corrupt, answers, log=log, **kw), "search": "group"}
    t0 = time.perf_counter()
    rec0 = pt.setup(clean, corrupt, answers, explicit_sets=sets) if sets else pt.setup(clean, corrupt, answers)
    if not rec0["valid"] or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
        return {**cf._find(pt, clean, corrupt, answers, log=log, **kw), "search": arm}
    part = []
    for ppt, ids in partners or []:
        if len(ids) == len(clean):
            r = ppt.setup(clean, ids, answers)
            if r["valid"] and r["gap"] >= cf.MIN_GAP and not r["same_answer"]:
                part.append(ppt)
    cuts = search_all(pt, arm, log=log, steps=steps, prior=prior, partners=part, wanted=cuts_wanted)
    rec = _find_with(pt, clean, corrupt, answers, log, arm, cuts, rec0=rec0, **kw)
    return {**rec, "seconds": round(time.perf_counter() - t0, 2)}


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("questions", "cases"))
    ap.add_argument("inp", help="questions: the questions JSON; cases: OUT")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cases", default="santa,preeclampsia", help="cases: circuit_finder's CASES or CHAT_CASES")
    ap.add_argument("--word", default=None, help="cases: taboo's secret word")
    ap.add_argument("--adapter", default=None, help="a PEFT adapter on the model (an organism)")
    ap.add_argument("--tokenizer", default=None, help="the tokenizer and chat template (default: the model's; an AuditBench "
                    "adapter ships its own, as circuit_finder --tokenizer)")
    ap.add_argument("--system", default=None, help="a system prompt before each user turn (a question's own \"system\" "
                    "wins; as circuit_finder --system)")
    ap.add_argument("--merge", action="store_true", help="the adapter merged into the weights (an organism at the base "
                    "model's speed, as circuit_finder --merge)")
    ap.add_argument("--arm", default="mask", help="comma-separated arms (mask, group, attr, am, fam), each question through every arm in "
                    "turn on the same card, records in OUT/<arm>/")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cuts", default="readout,movers,source", help="the cuts searched (circuit_finder's --cuts; "
                    "oracleagent's find_circuit: movers,source)")
    ap.add_argument("--partners", type=int, default=1, help="pairs a source cut is chosen over: the question's own and up "
                    "to PARTNERS - 1 other corrupted keys of its partners (prompts of its length), on the shares' means")
    ap.add_argument("--batch", type=int, default=1, help="mask and am: pairs whose masks are optimized in one batch "
                    "(find_batch; a record's seconds the batch's wall time over its pairs)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ids", default=None, help="a JSON list of question ids (the frozen dev set)")
    ap.add_argument("--prior", default=None, help="fam: a directory of circuit_finder records; a question's prior is its "
                    "task's records other than its own")
    ap.add_argument("--pool", default=None, help="partner keys from this question set's keys of the question's task, in a "
                    "seeded order per question, instead of the question's own partners (left for unseen-pair tests)")
    a = ap.parse_args()
    cf._threads()
    out = Path(a.out if a.mode == "questions" else a.inp)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="auto" if dev == "auto" else None)
    model = (model if dev == "auto" else model.to(dev)).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
        if a.merge:
            model = model.merge_and_unload().eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    pre = lambda q: tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else []  # noqa: E731
    answers = {}                                    # (a case's answer tokens: taboo's secret word)
    maxp = max([a.partners] + [int(x.split("=")[1]) for arm in a.arm.split(",") for x in arm.split(":")[1:]
                               if x.startswith("partners=")])
    if a.mode == "questions":
        qs = [q for q in json.loads(Path(a.inp).read_text())["questions"] if q.get("split", a.split) == a.split]
        qs = qs[a.offset::a.stride][:a.limit or None]
        if a.ids:
            want = set(json.loads(Path(a.ids).read_text()))
            qs = [q for q in qs if q["id"] in want]
        sy = lambda q: q.get("system", a.system)  # noqa: E731
        pairs = [(q, cf.chat(tok, q["user"], sy(q)) + pre(q), cf.chat(tok, q["corrupt_user"], sy(q)) + pre(q)) for q in qs]
        others = {}                                 # (other corrupted prompts of each question, its partners')
        keys = {}                                   # (--pool: each task's keys there)
        for p in json.loads(Path(a.pool).read_text())["questions"] if a.pool else []:
            keys.setdefault(p["task"], set()).add(p["key"])
        for q, clean, _ in pairs:
            users = {}
            if a.pool:
                ks = sorted(keys.get(q.get("task"), ()))
                np.random.default_rng(zlib.crc32(q["id"].encode())).shuffle(ks)
                users = {k: q["user"].replace(q["key"], k) for k in ks if k != q["key"] and q["user"].count(q["key"]) == 1}
            for p in [] if a.pool else q.get("partners", []):   # (keys, or {key, user} as circuit_devset writes them)
                if isinstance(p, dict):
                    users[p["key"]] = p["user"]
                elif q["user"].count(q["key"]) == 1:
                    users[p] = q["user"].replace(q["key"], p)
            ids = [(k, cf.chat(tok, u, sy(q)) + pre(q)) for k, u in users.items() if k != q.get("corrupt_key")]
            others[q["id"]] = [(k, x) for k, x in ids if len(x) == len(clean)][:maxp - 1]
    else:
        pairs = []
        for name in a.cases.split(","):
            got = cf.chat_pairs(tok, name, a.word, a.system) if name in cf.CHAT_CASES else cf.case_pairs(tok, name)
            for j, (c, k, text, ans) in enumerate(got):
                qid = f"{name}_{a.word}_{j}" if a.word else f"{name}_{j}"
                answers[qid] = ans
                pairs.append(({"id": qid, "task": name, "corrupt_text": text}, c, k))
    priors = [json.loads(p.read_text()) for p in sorted(Path(a.prior).glob("*.json"))] if a.prior else []
    priors = [r for r in priors if r.get("met") and not r.get("skipped")]
    arms = a.arm.split(",")                         # (an arm may name its mask steps: am:30, records in OUT/am_30)
    for arm in arms:
        assert arm.split(":")[0] in ("mask", "group", "attr", "am", "fam"), arm
        (out / _dir(arm)).mkdir(exist_ok=True)
    jobs = []
    for q, clean, corrupt in pairs:
        if len(clean) != len(corrupt):
            log(f"{q['id']}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
            continue
        jobs.append((q, clean, corrupt))
    jobs.sort(key=lambda j: len(j[1]))              # (batches of one prompt length)
    pts = [pt] + [cf.Patcher(model, tok) for _ in range(max(0, a.batch - 1))]
    pool = [[cf.Patcher(model, tok) for _ in range(maxp - 1)] for _ in range(a.batch)]   # (partner pairs' Patchers)
    others = others if a.mode == "questions" and maxp > 1 else {}
    pt.setup(jobs[0][1], jobs[0][2])                # (a throwaway run: CUDA's first kernels)

    def write(arm, q, clean, corrupt, rec):
        rec.update(id=q["id"], ids=clean, corrupted=corrupt, answer_text=tok.decode([rec["answer"]]),
                   corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype, adapter=a.adapter,
                   finder="circuit_masks", finder_rev=a.rev, tokenizer=a.tokenizer or a.model, system=a.system,
                   merged=bool(a.adapter and a.merge),
                   **{k: q[k] for k in ("user", "corrupt_user", "task", "key", "corrupt_text", "prefill") if k in q})
        (out / _dir(arm) / f"{q['id'].replace('/', '__')}.json").write_text(json.dumps(rec))

    def prior_of(name, q):
        return family_prior([r for r in priors if r.get("task") == q.get("task") and r["id"] != q["id"]]) if name == "fam" else None
    for c0 in range(0, len(jobs), a.batch):
        chunk = jobs[c0:c0 + a.batch]
        for arm in arms:
            name, *parts = arm.split(":")
            steps = next((int(x) for x in parts if x.isdigit()), a.steps)
            sets = {k.upper(): float(v) for k, v in (x.split("=") for x in parts if "=" in x)}
            npart = int(sets.pop("PARTNERS", a.partners))   # (an arm's partner pairs: mask:30:partners=4)
            oth = {i: v[:npart - 1] for i, v in others.items()}
            todo = [(q, cl, co) for q, cl, co in chunk
                    if not (out / _dir(arm) / f"{q['id'].replace('/', '__')}.json").exists()]
            if not todo:
                continue
            saved = {k: globals()[k] for k in sets}
            globals().update(sets)                  # (an arm's own constants: mask:20:init=1:lr=0.5)
            def put(q, cl, co, rec):
                if name != "group" and oth.get(q["id"]):        # (the partner keys a source cut was chosen on: held-out
                    rec["selection_keys"] = [q.get("corrupt_key")] + [k for k, _ in oth[q["id"]]]   # tests skip them)
                write(arm, q, cl, co, {**rec, "constants": {**sets, "PARTNERS": npart}} if sets or npart > 1 else rec)
            try:
                if a.batch > 1 and name in ("mask", "am"):
                    log(f"{len(todo)} pairs ({arm}, batched): " + ", ".join(q["id"] for q, _, _ in todo))
                    recs = find_batch(pts[:len(todo)], [(cl, co, answers.get(q["id"])) for q, cl, co in todo], arm=name, steps=steps,
                                      log=log, priors=[prior_of(name, q) for q, _, _ in todo],
                                      partners=[list(zip(pool[b], [x for _, x in oth.get(q["id"], [])]))
                                                for b, (q, _, _) in enumerate(todo)], cuts_wanted=tuple(a.cuts.split(",")))
                    for (q, cl, co), rec in zip(todo, recs):
                        put(q, cl, co, rec)
                else:                               # (each record written when found: a run stopped by a time limit
                    for q, cl, co in todo:          # keeps what it found)
                        log(f"{q['id']} ({arm})")
                        put(q, cl, co, find(pt, cl, co, answers.get(q["id"]), log=log, arm=name, steps=steps,
                                            prior=prior_of(name, q), partners=list(zip(pool[0], [x for _, x in oth.get(q["id"], [])]))
                                            if name != "group" else None, cuts_wanted=tuple(a.cuts.split(","))))
            finally:
                globals().update(saved)


def _dir(arm: str) -> str:
    """An arm's records directory: am:30 -> am_30, mask:20:init=1 -> mask_20_init1."""
    return arm.replace(":", "_").replace("=", "")


if __name__ == "__main__":
    main()
