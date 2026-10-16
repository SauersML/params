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

  circuit_masks.py questions QUESTIONS.json OUT --arm mask [--split dev --stride 4] [--model Qwen/Qwen3-8B] [--dtype ...]
writes OUT/<id>.json per question.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import circuit_finder as cf

STEPS = 60
LR = 0.25
TAU = 0.9           # the hinge's target share (PASS 0.7 is what the measured set must reach)
LAMBDA = 0.003      # a unit's cost in shares of the gap
INIT = 3.0          # sigmoid(3) = 0.95: every unit nearly patched at the start
ORDER_MIN = 64      # units ordered for the measured prefix: those the mask kept (m > 0.5), at least twice that and ORDER_MIN
ORDER_MAX = 20000


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


def optimize_many(jobs: list, steps: int = STEPS, log=None) -> list:
    """Masks for several pairs at once: jobs [(pt, specs, taus)], Patchers of one model each set up on its own pair (prompts
    of one length), every spec of every job two rows of one batch per step (a step reads every weight twice whatever
    its rows: batched pairs cost about one). taus: per spec the hinges' targets (lost, restored), default TAU. Per job
    and spec {kind: theta [L, P, width]} (final logits, on the host) and the history of (lost, restored, sum m) per step.
    A mask edits only its own (row, position) entries: those rows' values gathered, moved toward the other run's and
    put back."""
    pt0 = jobs[0][0]
    T = pt0.T
    assert all(pt.T == T for pt, _, _ in jobs), "optimize_many: prompts of one length"
    L, dev = pt0.L, pt0.dev
    width = {"neuron": pt0.D, "head": pt0.H}
    key = {"neuron": "neu", "head": "head"}
    specs = [(q, s) for q, (pt, ss, _) in enumerate(jobs) for s in ss]          # global spec g: rows 2g (clean, noised)
    start = min(min(s["positions"]) for _, s in specs)                           # and 2g + 1 (corrupted, denoised)
    B, S = 2 * len(specs), T - start
    kinds = sorted({k for _, s in specs for k in s["kinds"]})
    pairs = {k: [(g, p) for g, (_, s) in enumerate(specs) if k in s["kinds"] for p in s["positions"]] for k in kinds}
    rows = {k: torch.tensor([2 * g + b for g, _ in pairs[k] for b in (0, 1)], device=dev) for k in kinds}
    pos = {k: torch.tensor([p - start for _, p in pairs[k] for _ in (0, 1)], device=dev) for k in kinds}
    gof = {k: torch.tensor([g for g, _ in pairs[k]], device=dev) for k in kinds}
    with torch.no_grad():                           # each pair's two rows' other-run values per layer [2N, width]
        others = {k: [torch.stack([jobs[specs[g][0]][0].act[src][key[k]][i][p] for g, p in pairs[k]
                                   for src in ("corrupt", "clean")]) for i in range(L)] for k in kinds}
        x = torch.stack([jobs[q][0]._ids_dev[b] for q, _ in specs for b in ("clean", "corrupt")])
        kv = [tuple(torch.cat([jobs[q][0]._base_kv[b][i][j][:, :, :start] for q, _ in specs for b in ("clean", "corrupt")])
                    for j in (0, 1)) for i in range(L)]
    theta = {k: torch.full((L, len(pairs[k]), width[k]), INIT, device=dev, requires_grad=True) for k in kinds}
    opt = torch.optim.Adam(list(theta.values()), lr=LR)
    only = {"res": set(), "head": set(range(L)) if "head" in kinds else set(), "neu": set(range(L)) if "neuron" in kinds else set()}
    tau = torch.tensor([t for _, ss, taus in jobs for t in (taus or [(TAU, TAU)] * len(ss))], device=dev)
    cc = torch.tensor([jobs[q][0].c["clean"] for q, _ in specs], device=dev)
    cr = torch.tensor([jobs[q][0].c["corrupt"] for q, _ in specs], device=dev)
    gap = torch.tensor([jobs[q][0].gap for q, _ in specs], device=dev)
    spans, at = [], 0                               # each job's rows (consecutive)
    for pt, ss, _ in jobs:
        spans.append((pt, at, at + 2 * len(ss)))
        at += 2 * len(ss)
    hist = [[] for _ in specs]
    for step in range(steps):
        ms = {k: torch.sigmoid(t) for k, t in theta.items()}

        def edit(k, i, h):
            d = h.device
            r, p = rows[k].to(d), pos[k].to(d)
            m = ms[k][i].to(d, h.dtype).repeat_interleave(2, 0)              # [2N, width]
            sel, o = h[r, p], others[k][i].to(d, h.dtype)
            if k == "head":
                sel, o, m = sel.view(-1, pt0.H, pt0.hd), o.view(-1, pt0.H, pt0.hd), m[..., None]
            return h.index_put((r, p), (sel + m * (o - sel)).reshape(len(r), -1))
        with torch.enable_grad():
            hn = pt0._last(x, B, cache=(start, kv), neu=(lambda i, h: edit("neuron", i, h)) if "neuron" in kinds else None,
                           head=(lambda i, h: edit("head", i, h)) if "head" in kinds else None, only=only)
            c = torch.cat([pt._contrast(hn[a:b]) for pt, a, b in spans])
            lost, rest = (cc - c[0::2]) / gap, (c[1::2] - cr) / gap
            size = sum(torch.zeros(len(specs), device=dev).index_add(0, gof[k], m.sum((0, 2))) for k, m in ms.items())
            loss = (torch.relu(tau[:, 0] - lost) + torch.relu(tau[:, 1] - rest)).sum() + LAMBDA * size.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
        opt.step()
        for g, (a, b, n) in enumerate(zip(lost.tolist(), rest.tolist(), size.tolist())):
            hist[g].append((round(a, 4), round(b, 4), round(n, 1)))
    out, th = [[] for _ in jobs], {k: t.detach().float().cpu() for k, t in theta.items()}
    for g, (q, s) in enumerate(specs):
        got = {}
        for k in s["kinds"]:
            idx = [n for n, (gg, _) in enumerate(pairs[k]) if gg == g]       # (in the order of s["positions"])
            got[k] = th[k][:, idx]
        out[q].append({"theta": got, "history": hist[g]})
        if log:
            a, b, n = hist[g][-1]
            log(f"    mask {'+'.join(s['kinds'])} at {s['positions'][0]}..{s['positions'][-1]}{f' (pair {q})' if len(jobs) > 1 else ''}: "
                f"{steps} steps, lost {a:.2f}, restored {b:.2f}, sum m {n:.1f}")
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
    alone = pt.both([_blocks([v for v in circ if v != u]) for u in circ])
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


def cuts_of(pt) -> dict:
    last = pt.T - 1
    out = {"readout": {"kinds": ("neuron",), "positions": [last]}, "movers": {"kinds": ("head",), "positions": [last]}}
    if pt.first < last:
        out["source"] = {"kinds": ("neuron",), "positions": list(range(pt.first, last))}
    return out


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


def search_all(pt, arm: str, log=None, steps: int = STEPS, prior: dict | None = None) -> dict:
    """Every cut of the pair set up in pt by arm: circuit_finder.cut's result per cut name. Arms: "mask"; "attr"; "am", attr
    first and masks on the cuts it leaves unmet (one batch); "fam", prior's units (family_order) first, then attr's."""
    st = _before(pt, arm, log=log, prior=prior)
    t0 = time.perf_counter()
    got = optimize(pt, [st["specs"][n] for n in st["masked"]], steps=steps, log=log, taus=st["taus"]) if st["masked"] else []
    return _after(pt, st, got, time.perf_counter() - t0, steps, log=log)


def _before(pt, arm: str, log=None, prior: dict | None = None) -> dict:
    """search_all up to the masks: every cut's ceiling, attr's (and fam's) cuts, and the cuts left for masks ("masked")
    with their hinge targets."""
    specs = cuts_of(pt)
    names = list(specs)
    width = {"neuron": pt.D, "head": pt.H}
    every = [[cf.unit(k, i, p, torch.arange(width[k])) for k in s["kinds"] for i in range(pt.L) for p in s["positions"]]
             for s in specs.values()]
    ceil = dict(zip(names, pt.both(every)))
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
            out[n] = _record(verify(pt, order, log=log, name=n), ceil[n], True, specs[n]["positions"],
                             {"attr": {"seconds": round(time.perf_counter() - (t0 if n == live[0] else t1), 2),
                                       **({"family_units": len(fam)} if arm == "fam" else {})}})
        live = [n for n in live if not out[n]["met"]] if arm == "am" else []
    masked = live if arm in ("mask", "am") else []
    return {"specs": specs, "names": names, "ceil": ceil, "reach": reach, "out": out, "masked": masked,
            "taus": [tuple(min(TAU, max(cf.PASS + 0.05, c - 0.05)) for c in ceil[n]) for n in masked]}


def _after(pt, st: dict, got: list, t_opt: float, steps: int, log=None) -> dict:
    """search_all from the masks on: each masked cut's order (ordered) through verify; cuts with no result have no units."""
    specs, ceil, out = st["specs"], st["ceil"], st["out"]
    for n, g in zip(st["masked"], got):
        t1 = time.perf_counter()
        order, kept = ordered(specs[n], g["theta"])
        res = verify(pt, order, log=log, name=n)
        out[n] = _record(res, ceil[n], True, specs[n]["positions"],
                         {"mask": {"kept": kept, "history": g["history"][::max(1, steps // 20)],
                                   "seconds_opt": round(t_opt, 2), "seconds_verify": round(time.perf_counter() - t1, 2),
                                   **({"attr_unmet": out[n]["found"]} if n in out else {})}})
    for n in st["names"]:
        if n not in out:
            out[n] = _record({"units": [], "singles": {}, "met": False, "lost": 0.0, "restored": 0.0, "found": 0, "curve": []},
                             ceil[n], st["reach"][n], specs[n]["positions"], {})
    return out


def _find_with(pt, clean, corrupt, answers, log, arm: str, cuts: dict, rec0: dict | None = None) -> dict:
    """circuit_finder.find's record with cuts (search_all's) in place of its cut() (and rec0, setup's record, in place
    of setting up again)."""
    def cut(pt_, sc, kind, positions, done, log=None, search=None):
        return cuts["movers" if kind == "head" else "readout" if list(positions) == [pt_.T - 1] else "source"]
    saved_cut = cf.cut
    cf.cut = cut
    pt.scores = lambda: {}                          # (group testing's order: unused here)
    if rec0 is not None:
        pt.setup = lambda *a, **k: dict(rec0)
    try:
        rec = cf.find(pt, clean, corrupt, answers, log=log)
    finally:
        cf.cut = saved_cut
        del pt.scores
        if rec0 is not None:
            del pt.setup
    extra = {k: {n: c[k] for n, c in cuts.items() if k in c} for k in ("attr", "mask")}   # (find() copies its own fields
    return {**rec, "search": arm, **{k: v for k, v in extra.items() if v}}                 # only)


def find_batch(pts: list, items: list, arm: str = "mask", steps: int = STEPS, log=None, priors: list | None = None) -> list:
    """find() for several pairs at once: items [(clean, corrupt, answers)], one Patcher each (pts, of one model). Every
    pair is set up and its ceilings (and attr's cuts) measured, then the masks of every pair's masked cuts optimized in
    one batch per prompt length (optimize_many), then each pair's measured steps and record. Each record's "seconds" is
    the batch's wall time over its pairs."""
    t0 = time.perf_counter()
    recs, sts = [None] * len(items), [None] * len(items)
    for j, (pt, (clean, corrupt, answers)) in enumerate(zip(pts, items)):
        rec0 = pt.setup(clean, corrupt, answers)
        if not rec0["valid"] or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
            recs[j] = {**cf.find(pt, clean, corrupt, answers, log=log), "search": arm}
            continue
        sts[j] = (rec0, _before(pt, arm, log=log, prior=(priors or [None] * len(items))[j]))
    by_T = {}
    for j, x in enumerate(sts):
        if x is not None and x[1]["masked"]:
            by_T.setdefault(pts[j].T, []).append(j)
    got, t_opt = {}, {}
    for T, js in by_T.items():
        t1 = time.perf_counter()
        res = optimize_many([(pts[j], [sts[j][1]["specs"][n] for n in sts[j][1]["masked"]], sts[j][1]["taus"]) for j in js],
                            steps=steps, log=log)
        for j, r in zip(js, res):
            got[j], t_opt[j] = r, (time.perf_counter() - t1) / len(js)
    for j, x in enumerate(sts):
        if x is None:
            continue
        clean, corrupt, answers = items[j]
        cuts = _after(pts[j], x[1], got.get(j, []), t_opt.get(j, 0.0), steps, log=log)
        recs[j] = _find_with(pts[j], clean, corrupt, answers, log, arm, cuts, rec0=x[0])
    wall = time.perf_counter() - t0
    for r in recs:
        r.update(seconds_own=r.get("seconds"), seconds=round(wall / len(items), 2), batch=len(items))
    return recs


def find(pt, clean: list[int], corrupt: list[int], answers=None, log=None, arm: str = "mask", steps: int = STEPS,
         prior: dict | None = None) -> dict:
    """circuit_finder.find with arm's search in place of group testing (its record, "search": arm)."""
    if arm == "group":
        return {**cf.find(pt, clean, corrupt, answers, log=log), "search": "group"}
    t0 = time.perf_counter()
    rec0 = pt.setup(clean, corrupt, answers)
    if not rec0["valid"] or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
        return {**cf.find(pt, clean, corrupt, answers, log=log), "search": arm}
    rec = _find_with(pt, clean, corrupt, answers, log, arm, search_all(pt, arm, log=log, steps=steps, prior=prior), rec0=rec0)
    return {**rec, "seconds": round(time.perf_counter() - t0, 2)}


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("questions",))
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--arm", default="mask", help="comma-separated arms (mask, group, attr, am, fam), each question through every arm in "
                    "turn on the same card, records in OUT/<arm>/")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=1, help="mask and am: pairs whose masks are optimized in one batch "
                    "(find_batch; a record's seconds the batch's wall time over its pairs)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ids", default=None, help="a JSON list of question ids (the frozen dev set)")
    ap.add_argument("--prior", default=None, help="fam: a directory of circuit_finder records; a question's prior is its "
                    "task's records other than its own")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="auto" if dev == "auto" else None)
    model = (model if dev == "auto" else model.to(dev)).eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    qs = [q for q in json.loads(Path(a.inp).read_text())["questions"] if q["split"] == a.split]
    qs = qs[a.offset::a.stride][:a.limit or None]
    if a.ids:
        want = set(json.loads(Path(a.ids).read_text()))
        qs = [q for q in qs if q["id"] in want]
    pre = lambda q: tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else []  # noqa: E731
    priors = [json.loads(p.read_text()) for p in sorted(Path(a.prior).glob("*.json"))] if a.prior else []
    priors = [r for r in priors if r.get("met") and not r.get("skipped")]
    arms = a.arm.split(",")                         # (an arm may name its mask steps: am:30, records in OUT/am_30)
    for arm in arms:
        assert arm.split(":")[0] in ("mask", "group", "attr", "am", "fam"), arm
        (out / arm.replace(":", "_")).mkdir(exist_ok=True)
    jobs = []
    for q in qs:
        clean, corrupt = cf.chat(tok, q["user"]) + pre(q), cf.chat(tok, q["corrupt_user"]) + pre(q)
        if len(clean) != len(corrupt):
            log(f"{q['id']}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
            continue
        jobs.append((q, clean, corrupt))
    jobs.sort(key=lambda j: len(j[1]))              # (batches of one prompt length)
    pts = [pt] + [cf.Patcher(model, tok) for _ in range(max(0, a.batch - 1))]
    pt.setup(jobs[0][1], jobs[0][2])                # (a throwaway run: CUDA's first kernels)

    def write(arm, q, clean, corrupt, rec):
        rec.update(id=q["id"], ids=clean, corrupted=corrupt, answer_text=tok.decode([rec["answer"]]),
                   corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype,
                   finder="circuit_masks", finder_rev=a.rev, **{k: q[k] for k in ("user", "corrupt_user", "task", "key") if k in q})
        (out / arm.replace(":", "_") / f"{q['id'].replace('/', '__')}.json").write_text(json.dumps(rec))

    def prior_of(name, q):
        return family_prior([r for r in priors if r.get("task") == q.get("task") and r["id"] != q["id"]]) if name == "fam" else None
    for c0 in range(0, len(jobs), a.batch):
        chunk = jobs[c0:c0 + a.batch]
        for arm in arms:
            name, _, n = arm.partition(":")
            steps = int(n) if n else a.steps
            todo = [(q, cl, co) for q, cl, co in chunk
                    if not (out / arm.replace(":", "_") / f"{q['id'].replace('/', '__')}.json").exists()]
            if not todo:
                continue
            if a.batch > 1 and name in ("mask", "am"):
                log(f"{len(todo)} pairs ({arm}, batched): " + ", ".join(q["id"] for q, _, _ in todo))
                recs = find_batch(pts[:len(todo)], [(cl, co, None) for _, cl, co in todo], arm=name, steps=steps, log=log,
                                  priors=[prior_of(name, q) for q, _, _ in todo])
                for (q, cl, co), rec in zip(todo, recs):
                    write(arm, q, cl, co, rec)
                continue
            for q, cl, co in todo:
                log(f"{q['id']} ({arm})")
                write(arm, q, cl, co, find(pt, cl, co, None, log=log, arm=name, steps=steps, prior=prior_of(name, q)))


if __name__ == "__main__":
    main()
