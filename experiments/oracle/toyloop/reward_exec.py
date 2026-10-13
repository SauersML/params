"""Executable reward for the toy loop (TOYLOOP lane): run the predicted circuit and score it, size-penalized.

  reward_exec(meta, completion) -> float, meta = {"toy": render.Toy, ...} (train.examples adds "toy")
  score(toy, units) -> {"suff", "nec", "n", "reward"}

Texts: the zoo record's examples plus 24 sampled ordinary texts (every lookup-table entry exercised) and counterfactual
texts where the family claims them, else toy.texts. With U the predicted heads and neurons (tok/out ignored):
  suff = (mean + min) over texts / 2 of clip(1 - KL(model || U's active pieces alone) / KL(model || empty circuit), 0, 1)
         (U's nodes: every VPD-active piece of U at every position, the simplest form of the zoo's gt rule)
  nec  = (mean + min) over texts / 2 of clip(KL(model || model with U ablated) / KL(model || model with every live unit ablated), 0, 1)
  reward = (suff + nec) / 2 - UNIT_COST * |U|;  0 for an unparseable answer.
Sufficiency alone prefers one of several redundant routes (any one carries the answer); necessity credits stating all
of them (ablating one route of two halves the copy, of all of them breaks it). The unit cost removes distractors (they
add neither). The worst-text half makes a unit needed on one text in 30 (a lookup-table entry) worth its cost. The normalisers use no ground truth, so the same reward runs on toys without one.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/oracle/toys"), str(ROOT / "method/oracle"), str(ROOT / "method/vpd"),
                str(Path(__file__).resolve().parent)]

import circuit  # noqa: E402

UNIT_COST = 0.05
_CACHE: dict = {}
_UNIT = re.compile(r"L(\d+)([HN])(\d+)")


N_ORDINARY = 24   # sampled ordinary texts: enough to exercise every entry of a lookup table (bigram_mlp)


def texts_of(toy) -> list:
    """The record's examples plus N_ORDINARY fresh ordinary texts and 4 counterfactual ones where the family claims
    them (zoo.texts); else toy.texts."""
    rec = toy.meta.get("record")
    if not rec:
        return list(toy.texts)
    import zoo
    ts = [e["ids"] for e in rec["examples"]] + [ids for ids, _ in zoo.texts(rec["params"], N_ORDINARY, 1234)]
    if any(e["kind"] == "counterfactual" for e in rec["examples"]):
        ts += [ids for ids, _ in zoo.texts(rec["params"], 4, 4321, "counterfactual")]
    return ts


def _kl(p, q) -> list:
    return (p.exp() * (p - q)).sum(-1).tolist()


def _state(toy) -> dict:
    """Per toy, once: the native, its importances per text, the model's log-probs, empty-circuit KLs, all-ablated KLs."""
    if toy.name not in _CACHE:
        import planted
        import zoo
        nat = planted.native_of(toy.sd, toy.cfg)
        ts = texts_of(toy)
        full = zoo.last_logits(toy.sd, ts)
        live = zoo.all_units(toy.sd)
        _CACHE[toy.name] = {"nat": nat, "texts": ts, "ci": [nat.importance(ids) for ids in ts], "full": full,
                            "empty": [zoo.node_kl(nat, ids, []) for ids in ts],
                            "abl_all": _kl(full, zoo.last_logits(zoo.ablate(toy.sd, live), ts)), "live": set(live)}
    return _CACHE[toy.name]


def unit_keys(units) -> set:
    return {(int(m.group(1)), m.group(2), int(m.group(3))) for u in units if (m := _UNIT.fullmatch(u))}


def nodes_of(toy, units: set, text: int) -> list:
    """[[site, position, piece]] (native.Graph order): active pieces of the units on text `text`, all positions."""
    st = _state(toy)
    nat, ci, want = st["nat"], st["ci"][text], unit_keys(units)
    out = []
    for n in nat.names:
        layer, kind = int(n.split(".")[1]), "N" if ".mlp." in n else "H"
        for t, p in (ci[n] > 0).nonzero().tolist():
            if (layer, kind, int(nat.units[n][p])) in want:
                out.append([n, int(t), int(p)])
    return out


def score(toy, units: set) -> dict:
    import zoo
    st = _state(toy)
    keys = unit_keys(units)
    zoo_names = [f"L{l}.{k}{i}" for l, k, i in sorted(keys)]
    suff = [min(1.0, max(0.0, 1.0 - zoo.node_kl(st["nat"], ids, nodes_of(toy, units, i)) / max(e, 1e-9)))
            for i, (ids, e) in enumerate(zip(st["texts"], st["empty"]))]
    live = [u for u in zoo_names if u in st["live"]]   # ablating a dead unit changes nothing
    abl = _kl(st["full"], zoo.last_logits(zoo.ablate(toy.sd, live), st["texts"])) if live else [0.0] * len(suff)
    nec = [min(1.0, max(0.0, a / max(b, 1e-9))) for a, b in zip(abl, st["abl_all"])]
    agg = lambda v: (sum(v) / len(v) + min(v)) / 2   # noqa: E731  (mean and worst text: every text must be explained)
    s, n = agg(suff), agg(nec)
    return {"suff": s, "nec": n, "n": len(keys), "reward": (s + n) / 2 - UNIT_COST * len(keys)}


def reward_exec(meta: dict, completion: str) -> float:
    c = circuit.parse(completion)
    if not c.ok:
        return 0.0
    return score(meta["toy"], c.units)["reward"]


def score_wiring(toy, completion: str, *, nat=None, texts=None) -> dict:
    """Independent point-sufficiency diagnostic for the stated inter-unit wiring.

    Unlike the legacy unit-selection reward, actually execute Mermaid unit edges.
    Token-label arrows remain annotations, embeddings stay on, and remainders stay
    off. No English, token-role, necessity or intervention-fidelity certification.
    A supplied native must represent this toy; useful for reusing an existing load.
    """
    import copy
    import torch
    import goal_eval
    import native
    import planted

    parsed = circuit.parse(completion)
    nat = planted.native_of(toy.sd, toy.cfg) if nat is None else nat
    bound = copy.copy(toy)
    bound.meta = dict(toy.meta)
    # Compile against the exact pieces the executor loaded, never a fresh SVD basis.
    bound.meta["decomposition"] = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n])
                                    for n in nat.names}
    texts = list(toy.texts if texts is None else texts)
    if not texts:
        raise ValueError("wiring evaluation requires at least one input")
    ge, rows = goal_eval.GoalEval(nat), []
    with torch.no_grad():
        for ids in texts:
            graph = circuit.unit_graph(parsed, bound, ids)
            targets = [len(ids) - 1]
            ref = nat.reference([ids], targets)
            q = ge.graph_ref(ids, targets, graph, live=False)
            empty = ge.graph_ref(ids, targets, native.Graph(), live=False)
            kl = lambda x: max(0.0, float((ref.exp() * (ref - x)).sum()) / math.log(2))
            error, null_error = kl(q), kl(empty)
            rows.append({"kl_bits": error, "empty_kl_bits": null_error,
                         "top": int(q[0, -1].argmax()), "reference_top": int(ref[0, -1].argmax()),
                         "suff": min(1.0, max(0.0, 1.0 - error / max(null_error, 1e-9))),
                         "nodes": graph.count(), "edges": graph.edges()})
    return {"scope": "executed inter-unit/output wiring; token labels and English unverified",
            "remainders": "off", "embedding": "on", "texts": len(rows), "per_text": rows,
            "suff": sum(r["suff"] for r in rows) / len(rows),
            "kl_bits": sum(r["kl_bits"] for r in rows) / len(rows)}


def check(toys) -> dict:
    """gt vs gt minus each unit, gt plus the distractors, distractors, empty: per family the worst margin."""
    import render
    fam = {}
    for toy in toys:
        gt = {u for u in toy.truth["units"] if _UNIT.fullmatch(u)}
        dis = {render.zoo_unit(u) for u in toy.meta["record"]["distractor_units"]}
        r = lambda us: score(toy, us)["reward"]  # noqa: E731
        g = r(gt)
        minus = max((r(gt - {u}) for u in gt), default=-1)
        m = fam.setdefault(toy.meta["family"], {"n": 0, "gt": [], "gt_minus_one_margin": 9, "gt_plus_dis_margin": 9,
                                                "empty": [], "distractors_margin": 9})
        m["n"] += 1
        m["gt"].append(g)
        m["gt_minus_one_margin"] = min(m["gt_minus_one_margin"], g - minus)
        if dis:
            m["gt_plus_dis_margin"] = min(m["gt_plus_dis_margin"], g - r(gt | dis))
            m["distractors_margin"] = min(m["distractors_margin"], g - r(dis))
        m["empty"].append(r(set()))
    for m in fam.values():
        m["gt"] = round(sum(m["gt"]) / len(m["gt"]), 3)
        m["empty"] = round(max(m["empty"]), 3)
        for k in ("gt_minus_one_margin", "gt_plus_dis_margin", "distractors_margin"):
            m[k] = round(m[k], 3)
    return fam


if __name__ == "__main__":   # python3 -B reward_exec.py [every-k-th instance]: margins of gt over its perturbations
    import json
    import time

    import render
    import torch
    torch.set_num_threads(1)
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    toys = render.load_zoo(Path.home() / "mpd-data/toy_zoo/manifest.jsonl")[::k]
    t0 = time.time()
    res = check(toys)
    print(json.dumps(res, indent=1))
    calls = sum(1 + len(t.truth["units"]) + 3 for t in toys)
    print(f"{len(toys)} toys, ~{calls} score calls, {time.time() - t0:.1f}s (incl. per-toy setup)")
