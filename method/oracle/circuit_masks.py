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


@torch.no_grad()
def _others(pt, layout: list, start: int) -> dict:
    """Each row's other run's head outputs and neurons from start on: {"head"|"neuron": [L] [B, S, width]}."""
    other = {"clean": "corrupt", "corrupt": "clean"}
    return {kind: [torch.stack([pt.act[other[b]][key][i][start:] for b in layout]) for i in range(pt.L)]
            for kind, key in (("head", "head"), ("neuron", "neu"))}


def optimize(pt, specs: list, steps: int = STEPS, log=None, taus: list | None = None) -> list:
    """Masks for specs [{"kinds": ("neuron",) | ("head",) | ("neuron", "head"), "positions": [...]}], optimized together
    (two rows a spec); per spec {kind: theta [L, P, width]} (final logits, on the host) and the history of (lost,
    restored, sum m) per step. taus: per spec the hinges' targets (lost, restored), default TAU for both."""
    start = min(min(s["positions"]) for s in specs)
    layout = [b for _ in specs for b in ("clean", "corrupt")]
    B, S = len(layout), pt.T - start
    width = {"neuron": pt.D, "head": pt.H}
    devs = [l.mlp.down_proj.weight.device for l in pt.layers]
    theta = [{k: torch.full((pt.L, len(s["positions"]), width[k]), INIT, device=pt.dev, requires_grad=True)
              for k in _kinds_of(s)} for s in specs]
    rel = [torch.tensor([p - start for p in s["positions"]], device=pt.dev) for s in specs]
    others = _others(pt, layout, start)
    opt = torch.optim.Adam([t for th in theta for t in th.values()], lr=LR)
    x = torch.stack([pt._ids_dev[b] for b in layout])
    cache = pt._prefix(x, start, tuple(layout))
    kinds = sorted({k for s in specs for k in _kinds_of(s)})
    only = {"res": set(), "head": set(range(pt.L)) if "head" in kinds else set(),
            "neu": set(range(pt.L)) if "neuron" in kinds else set()}
    zeros = {k: torch.zeros(S, width[k], device=pt.dev) for k in kinds}
    hist = [[] for _ in specs]
    tau = torch.tensor(taus or [(TAU, TAU)] * len(specs), device=pt.dev)

    def mask(kind, i, ms):                          # [B, S, width] this layer's mask over the batch's rows
        rows = []
        for j, s in enumerate(specs):
            m = zeros[kind] if kind not in ms[j] else torch.index_copy(zeros[kind], 0, rel[j], ms[j][kind][i])
            rows += [m, m]
        return torch.stack(rows).to(devs[i])

    for step in range(steps):
        ms = [{k: torch.sigmoid(t) for k, t in th.items()} for th in theta]

        def neu(i, h):
            m = mask("neuron", i, ms).to(h.dtype)
            return h + m * (others["neuron"][i].to(h.device, h.dtype) - h)

        def head(i, h):
            Bh, Sh, _ = h.shape
            m = mask("head", i, ms).to(h.dtype)[..., None]
            h4 = h.reshape(Bh, Sh, pt.H, pt.hd)
            o4 = others["head"][i].to(h.device, h.dtype).reshape(Bh, Sh, pt.H, pt.hd)
            return (h4 + m * (o4 - h4)).reshape(Bh, Sh, pt.H * pt.hd)
        with torch.enable_grad():
            c = pt._contrast(pt._last(x, B, cache=cache, neu=neu if "neuron" in kinds else None,
                                      head=head if "head" in kinds else None, only=only))
            lost = (pt.c["clean"] - c[0::2]) / pt.gap
            rest = (c[1::2] - pt.c["corrupt"]) / pt.gap
            size = torch.stack([sum(m.sum() for m in mj.values()) for mj in ms])
            loss = (torch.relu(tau[:, 0] - lost) + torch.relu(tau[:, 1] - rest)).sum() + LAMBDA * size.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
        opt.step()
        for j, (a, b, n) in enumerate(zip(lost.tolist(), rest.tolist(), size.tolist())):
            hist[j].append((round(a, 4), round(b, 4), round(n, 1)))
    if log:
        for j, s in enumerate(specs):
            a, b, n = hist[j][-1]
            log(f"    mask {'+'.join(_kinds_of(s))} at {s['positions'][0]}..{s['positions'][-1]}: {steps} steps, "
                f"lost {a:.2f}, restored {b:.2f}, sum m {n:.1f}")
    return [{"theta": {k: t.detach().float().cpu() for k, t in th.items()}, "history": h} for th, h in zip(theta, hist)]


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
    g = cf.grid(len(order))
    got = pt.both([_blocks(order[:k]) for k in g])
    met = [k for k, r in zip(g, got) if passes(r)]
    ok = None
    if met:
        hi = met[0]
        lo = max([k for k in g if k < hi] + [0])
        while hi - lo > 33:                         # (bisection down to 32 sizes, then all of them in one batch)
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if passes(pt.both([_blocks(order[:mid])])[0]) else (mid, hi)
        mids = list(range(lo + 1, hi))
        ok = next((k for k, r in zip(mids, pt.both([_blocks(order[:k]) for k in mids]) if mids else []) if passes(r)), hi)
    circ = prune(pt, list(order[:ok])) if ok else []
    lost, rest = pt.both([_blocks(circ)])[0] if circ else (0.0, 0.0)
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
    while todo:
        trials = [gone | set(todo[:j + 1]) for j in range(len(todo))]
        got = pt.both([_blocks([v for v in circ if v not in t]) for t in trials])
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
    if arm in ("mask", "am") and live:
        t0 = time.perf_counter()
        taus = [tuple(min(TAU, max(cf.PASS + 0.05, c - 0.05)) for c in ceil[n]) for n in live]
        got = dict(zip(live, optimize(pt, [specs[n] for n in live], steps=steps, log=log, taus=taus)))
        t_opt = time.perf_counter() - t0
        for n in live:
            t1 = time.perf_counter()
            order, kept = ordered(specs[n], got[n]["theta"])
            res = verify(pt, order, log=log, name=n)
            out[n] = _record(res, ceil[n], True, specs[n]["positions"],
                             {"mask": {"kept": kept, "history": got[n]["history"][::max(1, steps // 20)],
                                       "seconds_opt": round(t_opt, 2), "seconds_verify": round(time.perf_counter() - t1, 2),
                                       **({"attr_unmet": out[n]["found"]} if n in out else {})}})
    for n in names:
        if n not in out:
            out[n] = _record({"units": [], "singles": {}, "met": False, "lost": 0.0, "restored": 0.0, "found": 0, "curve": []},
                             ceil[n], reach[n], specs[n]["positions"], {})
    return out


def find(pt, clean: list[int], corrupt: list[int], answers=None, log=None, arm: str = "mask", steps: int = STEPS,
         prior: dict | None = None) -> dict:
    """circuit_finder.find with arm's search in place of group testing (its record, "search": arm)."""
    if arm == "group":
        return {**cf.find(pt, clean, corrupt, answers, log=log), "search": "group"}
    keys = ("attr", "mask")
    memo = {}

    def cut(pt_, sc, kind, positions, done, log=None, search=None):
        key = (tuple(pt_.ids["clean"]), tuple(pt_.ids["corrupt"]))
        if memo.get("key") != key:
            memo.update(key=key, cuts=search_all(pt_, arm, log=log, steps=steps, prior=prior))
        name = "movers" if kind == "head" else "readout" if list(positions) == [pt_.T - 1] else "source"
        return memo["cuts"][name]
    saved_cut = cf.cut
    cf.cut = cut
    pt.scores = lambda: {}                          # (group testing's order: unused here; attr reads its own)
    try:
        rec = cf.find(pt, clean, corrupt, answers, log=log)
    finally:
        cf.cut = saved_cut
        del pt.scores
    extra = {k: {n: c[k] for n, c in memo.get("cuts", {}).items() if k in c} for k in keys}   # (find() copies its own
    return {**rec, "search": arm, **{k: v for k, v in extra.items() if v}}                     # fields only)


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
    first = True
    for q in qs:
        clean, corrupt = cf.chat(tok, q["user"]) + pre(q), cf.chat(tok, q["corrupt_user"]) + pre(q)
        if len(clean) != len(corrupt):
            log(f"{q['id']}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
            continue
        for arm in arms:
            path = out / arm.replace(":", "_") / f"{q['id'].replace('/', '__')}.json"
            if path.exists():
                continue
            if first:                               # (a throwaway run: CUDA's first kernels)
                pt.setup(clean, corrupt)
                first = False
            log(f"{q['id']} ({arm})")
            name, _, n = arm.partition(":")
            prior = family_prior([r for r in priors if r.get("task") == q.get("task") and r["id"] != q["id"]]) if name == "fam" else None
            rec = find(pt, clean, corrupt, None, log=log, arm=name, steps=int(n) if n else a.steps, prior=prior)
            rec.update(id=q["id"], ids=clean, corrupted=corrupt, answer_text=tok.decode([rec["answer"]]),
                       corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype,
                       finder="circuit_masks", finder_rev=a.rev,
                       **{k: q[k] for k in ("user", "corrupt_user", "task", "key") if k in q})
            path.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
