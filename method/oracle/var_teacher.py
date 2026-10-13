"""The teacher for explanation variables (#2951 graph oracle): per text, variables (input positions, a carrier holding
what those tokens contribute to the prediction, feeds), verified by interchange in the real model (variables.py), each
with its value map (var_classes.py) for the writer to name.

Input sets: each influential position (goal_eval's F2 positions and the targets), each position holding the circuit's
nodes, and each pair of neighbouring ones (a word of two tokens), carried at their own positions; and intermediate
variables for the circuit's nodes with a removal effect (m, goal_eval.line_effects in pp mode) whose reads depend on
other positions: per node its signature (the positions whose token changes move its read most, the fewest holding
more than half of its response; the search split's draws, exact real-model runs), each signature carried at the
nodes' own position (the copied token at the target, from its context positions), so the variables cover the
circuit's mechanism where its removal effect is and not only at the easy positions.

Carriers (the lead's 10-12). VPD writes a token's identity in which subcomponents are active, so a set of
subcomponents fixed on one text carries a third of a token change (text6011 at 233: VPD's active set there, f
on-target 0.36) where the position's whole site outputs carry all of it. Per input set I:
1. Blocks (variables.py: a site's whole output at a position, every subcomponent and the remainder) at I's positions:
   every single site, every layer, layers 0 to L, scored together on the search split; the smallest (in
   subcomponents) at or above GOAL, else grown from the best single site one site at a time (the site adding most)
   until it reaches GOAL or stops gaining.
2. Family: the union of VPD's active subcomponents (causal importance above zero) at the carrier positions on the
   text and on each alternative token at each input position (var_classes.alternatives, the 64 tokens the value map
   uses): a shared core plus every token-specific part seen, at every site and at the chosen blocks' sites. The
   smallest family at or above GOAL is the carrier, else the blocks are. On text6011 at 232-233 the family at every
   site (3,781 subcomponents, 6% of the position's 61,440) carries f on-target 0.975 (eval split 0.974) where the
   values' block carries 0.79 and VPD's active set on the text 0.41. VPD's active set on the text alone is scored
   beside them (reported).
Pool (the lead's decision): the circuit's sites, its blocks and families at the (site, position) pairs holding its
nodes ("circuit", the default with a circuit), or every site ("any"). Beside each carrier the trivial variable, every
reachable site at the carrier positions as one block (all a position holds: a change at the position is carried
almost by construction), is scored on both splits, and each variable is labelled "input" (carried at its own input
positions) or "intermediate" (carried at another position, a combined value): the informative ones are intermediate.
A variable is kept when its term min(f on-target, s off-target) is at least GOAL on the search split and at least
EVAL_KEEP on the eval split's draws, which the choice never saw (the lead's rule), each on its own; the others are
listed unkept with their scores, and a text with none has no VARIABLES target ("target": false). Variables are
chosen by the search term (those at GOAL first, single positions before pairs, each by the circuit's removal effect
inside the carrier): carriers disjoint (a block holds every subcomponent of its site and position), so the
variables' interventions commute, and variables carried at their own inputs with no input position in two.

Per variable: "kept", "kind" (input or intermediate), "value_kind" (kept: "computed" when its own value class holds
tokens other than the text's, an abstraction; "copies its inputs" when it holds the text's token alone: the
decomposition check), "carrier" (block or family), "carried_at" (the carrier's positions), "group" (blocks ["block",
site, position] or the family's subcomponents [site, index, position]), "instances" (VPD's active subcomponents on the
text inside the carrier) and "instances_share" (the share of the text and its alternatives on which each is active: the
core near 1), "circuit_nodes" (the circuit's nodes inside the carrier), "m" (their removal effect,
goal_eval.line_effects in pp mode), "values" (the value map per input position, class 0 the text's own:
var_classes.classes, kept variables), "feeds" (measured: each ordered pair of kept variables takes the claim, feeds or
apart, its feeds test favours; a variable feeding no other feeds the prediction), "feeds_stated" (the circuit's stated
paths through unnamed nodes), "measured" (the search and eval terms with RAVEL's Cause (f on-target), Iso (the off-
target draws that can reach the carrier: changes before it) and Disentangle (their mean), the carriers compared). Per
text, "invariants": semantic.teacher_invariants' verified INVARIANTS lines from the kept variables' value maps (a fresh
token the model sends to a class's token, carried by the carrier alone; removals), and "teacher": the kept variables
scored together (composition and feeds) on both splits, "m_share" (the share of the circuit's removal effect inside kept
carriers) and "e4x" (reward v3's sum of (2 e4_v - 1) m_v / sum m over the kept variables, e4_v the eval term). Names V1,
V2, ... in input order, meanings empty for the writer. Output ~/mpd-data/graph_oracle/variables/<split>/<task>.json.

  var_teacher.py [TASK ...] --circuits DIR [--pool circuit|any] [--root DIR | --out DIR] [--device mps] [--draws 8]
  (no TASK: every textN.json in DIR)
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import goal_eval  # noqa: E402
import native  # noqa: E402
import variables  # noqa: E402

GOAL = goal_eval.GOAL
EVAL_KEEP = 0.8  # the lead's keep rule: the search term at GOAL and the eval split's (draws the choice never saw) at this
OUT = native.TEXTS.parent / "variables"
BLOCK = variables.BLOCK


def release(dev) -> None:
    """Free a scorer call's tensors: its runs keep them in reference cycles, which Python's collector, counting
    objects and not bytes, leaves for many calls (0.5 GiB a screening call on text9004: a 20 GB card ran out)."""
    gc.collect()
    native.free(dev)


class _Screen(variables.Scorer):
    """The scorer without the feeds tests (the candidates of one input set are screened on on-target and off-target
    alone; the feeds are tested once the variables are chosen)."""

    def _feeds(self, *args) -> list:
        return []


def as_json(name: str, inputs, group, feeds=(), **extra) -> dict:
    v = {"name": name, "meaning": "", "inputs": sorted(int(p) for p in inputs),
         "group": [["block", n, int(t)] if c == BLOCK else [n, int(c), int(t)] for n, t, c in sorted(group)], "feeds": list(feeds)}
    v.update(extra)
    return v


def term(got: dict) -> float:
    """The headline's per-variable term: min(f on-target, s off-target), 0 without a resolved on-target draw."""
    f_on, s_off = got["f_on"], got["s_off"]
    return 0.0 if f_on is None else (f_on if s_off is None else min(f_on, s_off))


def size(C: dict, group) -> int:
    """A group's subcomponents (a block counts its site's every one)."""
    return sum(C[n] if c == BLOCK else 1 for n, _, c in group)


def screen(sc: _Screen, tx, inputs: tuple, cands: list[tuple[list, str]], draws: int, split: str = "search") -> list[dict]:
    """Every candidate group of one input set scored on a split (on-target and off-target; the draws are the text's and
    the inputs', shared by the candidates), one row each."""
    vs = variables.parse([as_json(f"c{k}", inputs, g) for k, (g, _) in enumerate(cands)], sc.nat.C)
    got = sc.score(tx, vs, split=split, draws=draws)["variables"]
    release(sc.nat.dev)
    return [{"inputs": list(inputs), "group": g, "how": how, "score": term(r), "size": size(sc.nat.C, g),
             **{k: r[k] for k in ("f_on", "s_off", "f_off", "s_on", "draws_on", "draws_off")}} for (g, how), r in zip(cands, got)]


def _layer(n: str) -> int:
    return native._name_layer_kind(n)[0]


def _where(nodes: list[tuple], dev) -> list:
    """Nodes (site, position, index) as per-site (places, positions, indices) for gathering activations."""
    per = {}
    for i, (n, t, c) in enumerate(nodes):
        x = per.setdefault(n, ([], [], []))
        x[0].append(i)
        x[1].append(t)
        x[2].append(c)
    return [(n, np.array(a), torch.tensor(b, device=dev), torch.tensor(c, device=dev)) for n, (a, b, c) in per.items()]


def _gather(act: dict, where: list, size: int) -> np.ndarray:
    out = np.empty(size)
    for n, idx, t, c in where:
        out[idx] = act[n][t, c].cpu().double().numpy()
    return out


def profiles(sc: variables.Scorer, tx, nodes: list[tuple], positions: list[int], draws: int) -> np.ndarray:
    """Mean |change| of each node's read (a block's: the norm of its output's change) per position [N, P] under the
    search split's token changes there, in the real model."""
    ge = sc.ge
    subs = [i for i, nd in enumerate(nodes) if nd[2] != BLOCK]
    bi = [i for i, nd in enumerate(nodes) if nd[2] == BLOCK]
    where = _where([nodes[i] for i in subs], sc.nat.dev)

    def reads(w):  # (the subcomponents' activations [S], the blocks' outputs)
        return _gather(w["act"], where, len(subs)), [w["out"][nodes[i][0]][nodes[i][1]] for i in bi]
    s0, b0 = reads(sc._text(tx.ids, tx.targets, None, check=False))
    out = np.zeros((len(nodes), len(positions)))
    for j, p in enumerate(positions):
        ds = ge._interchange_draws(tx, [p], "search", draws)
        for _, z in ds:
            x = list(tx.ids)
            x[p] = z
            s1, b1 = reads(sc._text(x, tx.targets, None, check=False))
            out[subs, j] += np.abs(s1 - s0) / len(ds)
            if bi:
                out[bi, j] += torch.stack([(o1 - o).norm() for o1, o in zip(b1, b0)]).cpu().double().numpy() / len(ds)
        release(ge.nat.dev)
    return out


def signature(row: np.ndarray, positions: list[int]) -> tuple:
    """The positions a node's read depends on: the fewest, largest first, holding more than half of its response."""
    total, got, acc = float(row.sum()), [], 0.0
    if total <= 0:
        return ()
    for j in np.argsort(-row):
        got.append(positions[j])
        acc += float(row[j])
        if acc > total / 2:
            break
    return tuple(sorted(got))


def block_search(sc: _Screen, tx, ins: tuple, draws: int, at: tuple | None = None, allowed: set | None = None
                 ) -> tuple[dict | None, list[dict], dict]:
    """(the chosen blocks' row or None, every row tried, the trivial row): blocks at the carrier positions (at; the
    input positions when None) of every single site, every layer and layers 0 to L, the smallest at or above GOAL;
    else grown from the best single site, one site at a time. allowed: the (site, position) pairs a block may take
    (the circuit's, pool "circuit"); None, every site that reaches the targets. The trivial row, every reachable site
    at the positions as one block (all a position holds), is scored beside them and is a candidate only without
    allowed."""
    nat = sc.nat
    at = at or ins
    reach = nat.reachable(tx.T, tx.targets)
    every = [n for n in nat.names if all(bool(reach[n][p].any()) for p in at)]
    ok_at = (lambda n, p: True) if allowed is None else (lambda n, p: (n, p) in allowed)  # noqa: E731
    sites = [n for n in every if any(ok_at(n, p) for p in at)]
    blocks = lambda ns: [(n, p, BLOCK) for n in ns for p in at if ok_at(n, p)]  # noqa: E731
    layers = sorted({_layer(n) for n in sites})
    cands = [(blocks([n]), f"site {n}") for n in sites]
    cands += [(blocks([n for n in sites if _layer(n) == L]), f"layer {L}") for L in layers]
    cands += [(blocks([n for n in sites if _layer(n) <= L]), f"layers {layers[0]}-{L}") for L in layers[1:]]
    seen, uniq = set(), []
    for g, how in cands:
        if g and tuple(g) not in seen:
            seen.add(tuple(g))
            uniq.append((g, how))
    trivial = ([(n, p, BLOCK) for n in every for p in at], "every site at the positions (trivial)")
    rows = screen(sc, tx, ins, uniq + [trivial], draws)
    triv, rows = rows[-1], rows[:-1]
    if allowed is None:
        rows.append(triv)
    if not rows:
        return None, [], triv
    ok = [r for r in rows if r["score"] >= GOAL]
    if ok:
        return min(ok, key=lambda r: (r["size"], -r["score"])), rows, triv
    singles = [r for r in rows if r["how"].startswith("site ")]
    cur = max(singles or rows, key=lambda r: r["score"])
    have = sorted({n for n, _, _ in cur["group"]})
    while cur["score"] < GOAL and len(have) < len(sites):
        rest = [n for n in sites if n not in have]
        got = screen(sc, tx, ins, [(blocks(have + [n]), "sites " + ", ".join(have + [n])) for n in rest], draws)
        rows += got
        b = max(got, key=lambda r: r["score"])
        if b["score"] <= cur["score"]:
            break
        cur, have = b, have + [rest[got.index(b)]]
    return max([cur] + [r for r in rows if r["score"] > cur["score"]], key=lambda r: r["score"]), rows, triv


def family(nat: native.Native, tx, ins: tuple, sites: list[str], at: tuple | None = None) -> tuple[list, dict]:
    """(the family: VPD's active subcomponents at the sites and carrier positions (at; the input positions when None)
    on the text or on any alternative token at an input position (var_classes.alternatives), the share of those texts
    on which each is active)."""
    import var_classes

    at = at or ins
    texts = [list(tx.ids)]
    for p in ins:
        for z in var_classes.alternatives(nat, tx.ids, p, int(tx.seed)):
            x = list(tx.ids)
            x[p] = z
            texts.append(x)
    count = {}
    pos = torch.tensor(list(at), device=nat.dev)
    for i, x in enumerate(texts):
        ci = tx.ci if i == 0 else nat.importance(x)
        for n in sites:
            for j, c in (ci[n][pos] > 0).nonzero().tolist():
                key = (n, at[j], c)
                count[key] = count.get(key, 0) + 1
    return sorted(count), {k: v / len(texts) for k, v in count.items()}


def inside(group: set, node) -> bool:
    """Whether a circuit node (site, position, index or BLOCK) lies in a group (a set of items): a block holds every
    subcomponent of its site and position; a circuit's block lies in a group holding anything at its site and
    position."""
    n, t, c = node
    if (n, t, BLOCK) in group:
        return True
    return any(g[0] == n and g[1] == t for g in group) if c == BLOCK else (n, t, c) in group


def stated_feeds(graph: native.Graph, chosen: list[dict]) -> dict:
    """Each variable's feeds from the circuit's stated connections: V feeds W when a stated path runs from a circuit
    node in V's carrier to one in W's through nodes in no carrier, and the prediction when one reaches a writer into
    the prediction."""
    nodes = [tuple(nd) for nd in graph.node_set()]
    groups = {v["name"]: set(v["_group"]) for v in chosen}
    parts = {name: {nd for nd in nodes if inside(g, nd)} for name, g in groups.items()}
    owner = {nd: name for name, ps in parts.items() for nd in ps}
    readers, out = {}, {tuple(w) for w in graph.out}
    for r, ws in graph.parents.items():
        for w in ws:
            readers.setdefault(tuple(w), []).append(tuple(r))
    got = {}
    for v in chosen:
        mine, f = parts[v["name"]], set()
        seen, todo = set(mine), list(mine)
        while todo:
            nd = todo.pop()
            if nd in out:
                f.add("prediction")
            for r in readers.get(nd, []):
                if owner.get(r) not in (None, v["name"]):
                    f.add(owner[r])
                elif r not in seen:
                    seen.add(r)
                    todo.append(r)
        got[v["name"]] = sorted(f)
    return got


def together(sc: variables.Scorer, tx, vs: list[dict], split: str, draws: int) -> dict:
    """variables.Scorer.score of several variables, by pairs when there are more than two: a call keeps every changed
    text's whole activations, and four variables' on-target, off-target and composition texts (~90 at 500 tokens) took
    a 24 GB card. Each variable's draws, each pair's composition and feeds tests are the same in any call holding them
    (keyed by the text and the positions), so the pairs' calls give every term of the joint call (each variable's from
    the first call holding it; the float32 resolution is each call's own)."""
    if len(vs) <= 2:
        got = sc.score(tx, vs, split=split, draws=draws)
        release(sc.nat.dev)
        return got
    first, pairs, feeds = {}, [], []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            got = sc.score(tx, [vs[i], vs[j]], split=split, draws=draws)
            release(sc.nat.dev)
            first.setdefault(i, got["variables"][0])
            first.setdefault(j, got["variables"][1])
            pairs += got["pairs"]
            feeds += got["feeds"]
    return variables.headline({"variables": [first[i] for i in range(len(vs))], "pairs": pairs, "feeds": feeds})


def measured_feeds(sc: variables.Scorer, tx, chosen: list[dict], draws: int) -> None:
    """Each ordered pair's claim (feeds or apart) by the feeds test on the search split (scored both ways, the better
    kept); a variable feeding no other feeds the prediction."""
    js = lambda feeds: [as_json(v["name"], v["_inputs"], v["_group"], feeds(v)) for v in chosen]  # noqa: E731
    names = [v["name"] for v in chosen]
    every = together(sc, tx, variables.parse(js(lambda v: [w for w in names if w != v["name"]]), sc.nat.C), "search", draws)["feeds"]
    none = together(sc, tx, variables.parse(js(lambda v: []), sc.nat.C), "search", draws)["feeds"]
    apart = {(x["from"], x["to"]): x["f"] for x in none}
    for v in chosen:
        v["feeds"] = sorted(x["to"] for x in every if x["from"] == v["name"] and x["f"] is not None
                            and x["f"] > (apart.get((x["from"], x["to"])) or 0.0))
        if not v["feeds"]:
            v["feeds"] = ["prediction"]


def overlaps(used: set, group: list) -> bool:
    """Whether a group shares a part with the used items: a subcomponent, or a site and position one side holds whole
    (a block)."""
    whole = {(n, t) for n, t, c in used if c == BLOCK}
    sites = {(n, t) for n, t, _ in used}
    return any((n, t) in sites if c == BLOCK else ((n, t, c) in used or (n, t) in whole) for n, t, c in group)


def teach(ge: goal_eval.GoalEval, task: dict, answer: dict | None, draws: int = 8, pool: str = "circuit", log=print) -> dict:
    import components
    import var_classes

    t0 = time.time()
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    nat = ge.nat
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    graph, m = None, {}
    if answer is not None:
        graph = goal_eval.strand(goal_eval.circuit(answer, targets, nat.C, len(ids))[0], targets, nat.C)
        le = ge.line_effects(tx, graph, mode="pp", split="search", draws=draws)
        m = {tuple(x["node"]): max(0.0, x["delta_kl"] or 0.0) for x in le["nodes"]}
        release(nat.dev)
    M = sum(m.values())
    held = {nd[1] for nd in m if nd[0] != "tok" and nd[1] <= max(targets)}
    positions = sorted({q for q, _ in tx.f2} | set(targets) | held)
    sets = [((q,), (q,)) for q in positions] + [((a, b), (a, b)) for a, b in zip(positions, positions[1:]) if b == a + 1]
    sc = _Screen(ge)
    if m:  # intermediate variables: the circuit's nodes with a removal effect, at their own positions, whose reads depend
        # on other positions (each node's signature: the positions moving its read most, more than half of it)
        nodes = sorted(nd for nd in m if nd[0] != "tok" and m[nd] > 0)
        prof = profiles(sc, tx, nodes, positions, draws) if nodes else None
        mid = {}
        for i, nd in enumerate(nodes):
            sig = signature(prof[i], positions)
            if sig and sig != (nd[1],):
                mid[(sig, (nd[1],))] = mid.get((sig, (nd[1],)), 0.0) + m[nd]
        for t in sorted({k[1][0] for k in mid}):  # per carrier position the signatures holding most of the removal
            # effect of its nodes that depend on other positions, the fewest past half of it (the signature's own rule;
            # nodes there differ in small ways)
            got = sorted((k for k in mid if k[1] == (t,)), key=lambda k: -mid[k])
            total, acc = sum(mid[k] for k in got), 0.0
            for k in got:
                sets.append(k)
                acc += mid[k]
                if acc > total / 2:
                    break
    log(f"{task['id']}: positions {positions} ({len(m)} circuit nodes, removal effect {M:.3f} bits), {len(sets)} input sets; "
        f"{time.time() - t0:.0f} s")
    allowed = None if pool == "any" or not m else {(n, t) for n, t, _ in m if n != "tok"}
    best = {}
    for ins, at in sets:
        b, rows, triv = block_search(sc, tx, ins, draws, at, allowed)
        if b is None:  # the circuit holds no site at these positions
            continue
        sites = sorted({n for n, _, _ in b["group"]})
        reach = nat.reachable(tx.T, tx.targets)
        every = [n for n in nat.names if all(bool(reach[n][q].any()) for q in at)]
        fam, share = family(nat, tx, ins, every, at)  # the family over every site, then restricted
        on_x = {(n, q, c) for n in every for q in at for c in (tx.ci[n][q] > 0).nonzero().flatten().tolist()}
        wide = fam if allowed is None else [k for k in fam if (k[0], k[1]) in allowed]
        cands = [([k for k in fam if k[0] in sites and (allowed is None or (k[0], k[1]) in allowed)], "family at the blocks' sites"),
                 (wide, "family at every site" if allowed is None else "family at the circuit's sites"),
                 (sorted(k for k in wide if k in on_x), "active on the text")]
        fa = screen(sc, tx, ins, [c for c in cands if c[0]], draws)
        got = {r["how"]: r for r in fa}
        wide_how = cands[1][1]
        ok = [got[h] for h in ("family at the blocks' sites", wide_how) if h in got and got[h]["score"] >= GOAL]
        rec = {"inputs": list(ins), "at": list(at), "kind": "input" if tuple(at) == tuple(ins) else "intermediate", "blocks": b,
               "family": got.get(wide_how), "family_sites": got.get("family at the blocks' sites"), "active": got.get("active on the text"),
               "trivial": triv, "tried": len(rows)}
        use = min(ok, key=lambda r: r["size"]) if ok else b
        held = {n for n, _, _ in use["group"]}
        active = sorted(k for k in on_x if k[0] in held)
        rec.update(carrier="family" if use is not b else "block", group=use["group"], score=use["score"],
                   instances=active, share={k: share.get(k, 0.0) for k in active},
                   circuit=[nd for nd in m if inside(set(use["group"]), nd)])
        rec["m"] = sum(m[nd] for nd in rec["circuit"])
        best[(ins, at)] = rec
        sz = lambda k: f"{rec[k]['size']} {rec[k]['score']:.3f}" if rec[k] else "none"  # noqa: E731
        log(f"  inputs {list(ins)} at {list(at)} ({rec['kind']}): {rec['carrier']} {use['how']} ({use['size']} subcomponents) "
            f"{use['score']:.3f}; blocks {b['how']} {b['score']:.3f}, family at the blocks' sites {sz('family_sites')}, "
            f"{wide_how} {sz('family')}, active {sz('active')}; trivial {sz('trivial')}; m {rec['m']:.3f}")
    # by the search term, at their inputs single positions before pairs, by the circuit's removal effect inside;
    # carriers disjoint (a block holds every subcomponent of its site and position), and variables carried at their
    # own inputs with no input position in two
    chosen, used, taken = [], set(), set()
    order = sorted(best.values(), key=lambda r: (r["score"] < GOAL, r["at"] == r["inputs"] and len(r["inputs"]) > 1, -r["m"], -r["score"]))
    for r in order:
        own = r["at"] == r["inputs"]
        if overlaps(used, r["group"]) or (own and taken & set(r["inputs"])):
            continue
        used |= set(r["group"])
        if own:
            taken |= set(r["inputs"])
        chosen.append(r)
    chosen.sort(key=lambda r: (r["at"], r["inputs"]))
    for i, r in enumerate(chosen):  # names for the feeds; the writer names them
        r["name"], r["_inputs"], r["_group"] = f"V{i + 1}", r["inputs"], r["group"]
    out = {"variables": [], "teacher": {"positions": positions, "sets": len(sets), "pool": pool, "circuit_m": M}}
    for r in chosen:
        meas = {"score": r["score"], "carriers": {k: (None if r[k] is None else {x: r[k][x] for x in ("how", "size", "score", "f_on", "s_off")})
                                                  for k in ("blocks", "family", "family_sites", "active", "trivial")}, "tried": r["tried"],
                "tokens": components.token_strings(task["model"], [ids[q] for q in r["inputs"]])}
        out["variables"].append(as_json(
            r["name"], r["inputs"], r["group"], [], kind=r["kind"], carrier=r["carrier"], carried_at=r["at"],
            instances=[[n, c, t] for n, t, c in r["instances"]], instances_share=[round(r["share"][k], 4) for k in r["instances"]],
            circuit_nodes=[["block", n, t] if c == BLOCK else [n, c, t] for n, t, c in r["circuit"]], m=r["m"], measured=meas))
    # each variable alone on both splits, and its position's trivial block on eval; kept: search at GOAL and eval at
    # EVAL_KEEP (the lead's rule: a check on draws the choice never saw)
    for v, r in zip(out["variables"], chosen):
        for split in ("search", "eval"):
            got = together(variables.Scorer(ge), tx, variables.parse([v], nat.C), split, draws)
            x = got["variables"][0]
            # RAVEL's names: Cause, the carrier moves the prediction as its inputs do (f on-target); Iso, off-target draws
            # that can reach the carrier (a change before it: one after it passes by construction, the model is causal)
            # leave the prediction; Disentangle their mean
            reach = [d for d in got["draws"] if d["kind"] == "off" and d["s"] is not None and d["changes"][0][0] < min(r["at"])]
            w = sum(d["d0"] for d in reach)
            iso = sum(d["d0"] * d["s"] for d in reach) / w if w > 0 else None
            v["measured"][split] = {"term": term(x), "f_on": x["f_on"], "s_off": x["s_off"], "cause": x["f_on"], "iso": iso,
                                    "iso_draws": len(reach), "off_draws": x["draws_off"],
                                    "disentangle": None if x["f_on"] is None or iso is None else (x["f_on"] + iso) / 2}
        tv = as_json("trivial", r["inputs"], r["trivial"]["group"])
        x = together(variables.Scorer(ge), tx, variables.parse([tv], nat.C), "eval", draws)["variables"][0]
        v["measured"]["trivial_eval"] = {"term": term(x), "f_on": x["f_on"], "s_off": x["s_off"], "size": r["trivial"]["size"]}
        v["kept"] = v["measured"]["search"]["term"] >= GOAL and v["measured"]["eval"]["term"] >= EVAL_KEEP
    kv = [v for v in out["variables"] if v["kept"]]
    stated = stated_feeds(graph, [r for r, v in zip(chosen, out["variables"]) if v["kept"]]) if graph is not None else {}
    if kv:
        ks = [r for r, v in zip(chosen, out["variables"]) if v["kept"]]
        measured_feeds(variables.Scorer(ge), tx, ks, draws)
        for v, r in zip(kv, ks):
            v["feeds"], v["feeds_stated"] = r["feeds"], stated.get(r["name"], [])
        for split, key in (("search", "together"), ("eval", "eval")):  # the kept variables together
            got = together(variables.Scorer(ge), tx, variables.parse(kv, nat.C), split, draws)
            out["teacher"][key] = {k: got[k] for k in ("e4", "f_on", "s_off", "f_comp", "f_feeds", "f_apart", "unresolved")}
        for v, pv in zip(kv, variables.parse(kv, nat.C)):  # the value maps, for the writer
            v["values"] = []
            for q in pv["inputs"]:
                c = var_classes.classes(sc, tx, pv, q, task["model"])
                v["values"].append({"position": q, "token": c["token"], "predicts": c["predicts"], "classes": c["values"]})
                release(nat.dev)
            # the decomposition check (Geiger, Wu et al. 2023): a variable whose own class holds other tokens keeps its
            # value under them (computed, an abstraction); one whose own class is the text's token alone moves with every
            # token change (it copies its inputs' identities)
            v["value_kind"] = "computed" if any(len(vm["classes"][0]["token_ids"]) > 1 for vm in v["values"]) else "copies its inputs"
        import semantic  # verified INVARIANTS lines from the kept variables' value maps (fresh donors), for the writer

        sv = [{"name": v["name"], "inputs": v["inputs"], "group": pv["group"], "values": v["values"]} for v, pv in zip(kv, variables.parse(kv, nat.C))]
        out["invariants"] = semantic.teacher_invariants(variables.Scorer(ge), tx, sv, task["model"])
        release(nat.dev)
    out["teacher"].update(kept=len(kv), target=bool(kv), kinds={k: sum(v["kept"] and v["kind"] == k for v in out["variables"])
                                                               for k in ("input", "intermediate")},
                          value_kinds={k: sum(v.get("value_kind") == k for v in kv) for k in ("computed", "copies its inputs")})
    if M > 0:
        out["teacher"]["m_share"] = sum(v["m"] for v in kv) / M
        out["teacher"]["e4x"] = sum((2 * v["measured"]["eval"]["term"] - 1) * v["m"] for v in kv) / M
    out["teacher"]["seconds"] = time.time() - t0
    log(f"{task['id']}: {len(kv)} of {len(chosen)} variables kept ({out['teacher']['kinds']}), eval {out['teacher'].get('eval')}, "
        f"m share {out['teacher'].get('m_share')}, e4x {out['teacher'].get('e4x')}, {time.time() - t0:.0f} s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--circuits", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=OUT, help="outputs under ROOT/<split>/")
    ap.add_argument("--out", type=Path, help="outputs in OUT instead")
    ap.add_argument("--device")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--pool", choices=("circuit", "any"), default="circuit",
                    help="carriers among the circuit's sites (its blocks and families there) or every site")
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(args.device))
    tasks = args.tasks or sorted((f.stem for f in args.circuits.iterdir() if re.fullmatch(r"text\d+\.json", f.name)),
                                 key=lambda s: int(s[4:]))
    for t in tasks:
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        out = args.out or (args.root / task["split"])
        out.mkdir(parents=True, exist_ok=True)
        f = args.circuits / f"{t}.json"
        res = teach(ge, task, json.loads(f.read_text()) if f.exists() else None, args.draws, args.pool, log=lambda s: print(s, flush=True))
        (out / f"{t}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
