"""The teacher for explanation variables (#2951 graph oracle): per text, groups of subcomponents that carry what given
context tokens contribute to the prediction, verified by interchange in the real model (variables.py).

The pool: a teacher circuit's nodes (the default), or, as a reference (--pool vpd), VPD's active subcomponents at the
influential positions (goal_eval's F2 positions and the targets): causal importance above zero on the text or on any
of the search split's changed texts there, at positions that can reach the targets (what the variables would be if
the circuit held the context).

1. Profiles: each pool node's read (its activation v . x) on texts with one influential token changed (the F2
   positions and the targets; the search split's draws, GoalEval._interchange_draws) against the text's: the mean
   |change| per position, exact, from real-model runs (variables.Scorer._text).
2. Candidate groups per input set I (each influential position, and each pair of neighbouring ones: a word of two
   tokens), among the nodes at or after I's first position: the nodes ordered by their selectivity for I (their
   response to I's positions over their response to every position) and by their response to I in output units
   (|read change| x |u|), each order's prefixes of 1, 2, 4, ... nodes and all of them; the responding nodes at each
   one position; and the nodes whose largest response is at I.
3. Verification: every candidate of an input set scored together by variables.Scorer on the search split (on-target
   and off-target; the draws are the text's and I's, shared by the candidates), its term min(f on-target, s
   off-target) (the headline's, 0 without a resolved on-target draw). The bar is min(GOAL, the best term); input sets
   are taken best first, each with its smallest candidate at or above the bar (the simplest group that carries its
   inputs), with no input position and no node in two. Nodes in no variable stay unnamed.
4. Feeds: a circuit's stated connections (a stated path from a node of V to a node of W, or to the prediction,
   through unnamed nodes). In the VPD pool, which states none, each ordered pair takes the claim (feeds or apart) its
   measured feeds test favours, and a variable feeding no other one feeds the prediction. The chosen variables are
   then scored together (composition and feeds, the model-world headline e4) on the search split and on the eval
   split, whose draws the choice never saw (the held-out number).
5. Output: per text {"variables": [{"name", "meaning": "", "inputs", "group": [[matrix, index, position], ...],
   "feeds": [variables or "prediction"], "measured": {...}, with --classes "classes": var_classes.classes per input
   position}], "teacher": {...}} (primerl's interface), the names
   V1, V2, ... in input order and the meanings empty for the writer, to ~/mpd-data/graph_oracle/variables/<split>/<task>.json (the VPD pool's under
   variables/vpd_pool/<split>/).

  var_teacher.py [TASK ...] --circuits DIR [--pool circuit|vpd] [--out DIR] [--device mps] [--draws 8] [--classes]
  (no TASK: every textN.json in DIR)
"""
from __future__ import annotations

import argparse
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
OUT = native.TEXTS.parent / "variables"


class _Screen(variables.Scorer):
    """The scorer without the feeds tests (candidates of one input set are screened on on-target and off-target
    alone; their feeds are tested once the variables are chosen)."""

    def _feeds(self, *args) -> list:
        return []


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


def profiles(sc: variables.Scorer, tx, nodes: list[tuple], positions: list[int], draws: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean |change| of each node's read per position [N, P], each node's read on the text [N]) in the real model."""
    ge, where = sc.ge, _where(nodes, sc.nat.dev)
    a0 = _gather(sc._text(tx.ids, tx.targets, None, check=False)["act"], where, len(nodes))
    out = np.zeros((len(nodes), len(positions)))
    for j, p in enumerate(positions):
        ds = ge._interchange_draws(tx, [p], "search", draws)
        for _, z in ds:
            x = list(tx.ids)
            x[p] = z
            out[:, j] += np.abs(_gather(sc._text(x, tx.targets, None, check=False)["act"], where, len(nodes)) - a0) / len(ds)
        native.free(ge.nat.dev)
    return out, a0


def vpd_pool(ge: goal_eval.GoalEval, tx, positions: list[int], draws: int) -> list[tuple]:
    """VPD's active subcomponents at the positions (causal importance above zero on the text or on any search draw's
    changed text at one of them), at positions that can reach the targets."""
    nat = ge.nat
    on = {n: tx.ci[n] > 0 for n in nat.names}
    for p in positions:
        for _, z in ge._interchange_draws(tx, [p], "search", draws):
            x = list(tx.ids)
            x[p] = z
            for n, v in nat.importance(x).items():
                on[n] |= v > 0
    reach = nat.reachable(tx.T, tx.targets)
    at = torch.tensor(positions, device=nat.dev)
    out = []
    for n in nat.names:
        m = (on[n] & reach[n])[at]
        for j, c in m.nonzero().tolist():
            out.append((n, positions[j], c))
    return sorted(out)


def ladder(order: list) -> list[list]:
    """An order's prefixes of 1, 2, 4, ... members and all of them."""
    ks = sorted({min(len(order), 1 << i) for i in range(max(1, len(order)).bit_length() + 1)})
    return [order[:k] for k in ks if k]


def candidates(nodes: list[tuple], prof: np.ndarray, unorm: np.ndarray, positions: list[int], inputs: tuple) -> list[tuple[list, str]]:
    """Candidate groups for an input set (module docstring, step 2), [(group, how)]."""
    js = [positions.index(p) for p in inputs]
    total = prof.sum(1)
    mine = prof[:, js].sum(1)
    live = [i for i, nd in enumerate(nodes) if nd[1] >= min(inputs) and mine[i] > 0]
    if not live:
        return []
    sel = sorted(live, key=lambda i: (-mine[i] / total[i], -mine[i] * unorm[i]))
    strong = sorted(live, key=lambda i: -mine[i] * unorm[i])
    out = [([nodes[i] for i in g], f"selectivity, first {len(g)}") for g in ladder(sel)]
    out += [([nodes[i] for i in g], f"response, first {len(g)}") for g in ladder(strong)]
    for t in sorted({nodes[i][1] for i in live}):
        out.append(([nodes[i] for i in live if nodes[i][1] == t], f"responding at {t}"))
    top = [nodes[i] for i in live if int(prof[i].argmax()) in js]
    if top:
        out.append((top, "largest response at the inputs"))
    seen, kept = set(), []
    for g, how in out:
        key = tuple(sorted(g))
        if key not in seen:
            seen.add(key)
            kept.append((sorted(g), how))
    return kept


def as_json(name: str, inputs, group, feeds=(), measured=None) -> dict:
    v = {"name": name, "meaning": "", "inputs": sorted(int(p) for p in inputs),
         "group": [[n, int(c), int(t)] for n, t, c in sorted(group)], "feeds": list(feeds)}
    if measured is not None:
        v["measured"] = measured
    return v


def term(got: dict) -> float:
    """The headline's per-variable term: min(f on-target, s off-target), 0 without a resolved on-target draw."""
    f_on, s_off = got["f_on"], got["s_off"]
    return 0.0 if f_on is None else (f_on if s_off is None else min(f_on, s_off))


def screen(sc: _Screen, tx, inputs: tuple, cands: list[tuple[list, str]], draws: int) -> list[dict]:
    """Every candidate of one input set scored on the search split (on-target and off-target), one row each."""
    vs = variables.parse([as_json(f"c{k}", inputs, g) for k, (g, _) in enumerate(cands)], sc.nat.C)
    got = sc.score(tx, vs, split="search", draws=draws)["variables"]
    return [{"inputs": list(inputs), "group": g, "how": how, "score": term(r), **{k: r[k] for k in
             ("f_on", "s_off", "f_off", "s_on", "draws_on", "draws_off")}} for (g, how), r in zip(cands, got)]


def stated_feeds(graph: native.Graph, chosen: list[dict]) -> None:
    """Each variable's feeds from the circuit's stated connections: V feeds W when a stated path runs from a node of V
    to a node of W through unnamed nodes only, and the prediction when one reaches a writer into the prediction."""
    owner = {tuple(nd): v["name"] for v in chosen for nd in v["_group"]}
    readers, out = {}, {tuple(w) for w in graph.out}
    for r, ws in graph.parents.items():
        for w in ws:
            readers.setdefault(tuple(w), []).append(tuple(r))
    for v in chosen:
        mine, f = set(map(tuple, v["_group"])), set()
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
        v["feeds"] = sorted(f)


def measured_feeds(sc: variables.Scorer, tx, chosen: list[dict], draws: int) -> None:
    """Each ordered pair's claim (feeds or apart) by the feeds test on the search split (scored both ways, the better
    kept); a variable feeding no other feeds the prediction."""
    js = lambda feeds: [as_json(v["name"], v["_inputs"], v["_group"], feeds(v)) for v in chosen]  # noqa: E731
    names = [v["name"] for v in chosen]
    every = sc.score(tx, variables.parse(js(lambda v: [w for w in names if w != v["name"]]), sc.nat.C), "search", draws)["feeds"]
    none = sc.score(tx, variables.parse(js(lambda v: []), sc.nat.C), "search", draws)["feeds"]
    apart = {(x["from"], x["to"]): x["f"] for x in none}
    for v in chosen:
        v["feeds"] = sorted(x["to"] for x in every if x["from"] == v["name"] and x["f"] is not None
                            and x["f"] > (apart.get((x["from"], x["to"])) or 0.0))
        if not v["feeds"]:
            v["feeds"] = ["prediction"]


def teach(ge: goal_eval.GoalEval, task: dict, answer: dict | None, draws: int = 8, pool: str = "circuit", classes: bool = False,
          log=print) -> dict:
    import components

    t0 = time.time()
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    nat = ge.nat
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    positions = sorted({q for q, _ in tx.f2} | set(targets))
    graph = None
    if pool == "circuit":
        graph, _ = goal_eval.circuit(answer, targets, nat.C, len(ids))
        graph = goal_eval.strand(graph, targets, nat.C)
        nodes = sorted(nd for nd in graph.node_set() if nd[0] != "tok")
    else:
        nodes = vpd_pool(ge, tx, positions, draws)
    sc = _Screen(ge)
    prof, a0 = profiles(sc, tx, nodes, positions, draws)
    un = {n: nat.target.site(n).U.norm(dim=1).cpu().double().numpy() for n in nat.names}
    unorm = np.array([un[n][c] for n, _, c in nodes])
    log(f"{task['id']}: {len(nodes)} {pool} nodes, positions {positions}; profiles {time.time() - t0:.0f} s")
    sets = [(q,) for q in positions] + [(a, b) for a, b in zip(positions, positions[1:]) if b == a + 1]
    rows = []
    for ins in sets:
        cands = candidates(nodes, prof, unorm, positions, ins)
        if cands:
            got = screen(sc, tx, ins, cands, draws)
            rows += got
            b = max(got, key=lambda r: r["score"])
            log(f"  inputs {list(ins)}: {len(cands)} candidates, best {b['score']:.3f} ({b['how']}, {len(b['group'])} nodes, "
                f"f_on {b['f_on']}, s_off {b['s_off']})")
    rows.sort(key=lambda r: -r["score"])
    best = rows[0]["score"] if rows else 0.0
    bar = min(GOAL, best)
    by = {}
    for r in rows:
        if r["score"] >= bar and r["score"] > 0:
            by.setdefault(tuple(r["inputs"]), []).append(r)
    chosen, used, taken = [], set(), set()
    for ins in sorted(by, key=lambda k: -by[k][0]["score"]):  # input sets best first; each its smallest group at the bar
        free = [r for r in by[ins] if not used & set(r["group"])]
        if taken & set(ins) or not free:
            continue
        r = min(free, key=lambda r: (len(r["group"]), -r["score"]))
        used |= set(r["group"])
        taken |= set(ins)
        chosen.append({"_inputs": r["inputs"], "_group": r["group"], "measured": {k: r[k] for k in r if k not in ("group", "inputs")}
                       | {"tokens": components.token_strings(task["model"], [ids[q] for q in r["inputs"]])}})
    chosen.sort(key=lambda v: v["_inputs"])
    for i, v in enumerate(chosen):  # names for the feeds; the writer names them
        v["name"] = f"V{i + 1}"
    if graph is not None:
        stated_feeds(graph, chosen)
    elif chosen:
        measured_feeds(variables.Scorer(ge), tx, chosen, draws)
    out = {"variables": [as_json(v["name"], v["_inputs"], v["_group"], v["feeds"], v["measured"]) for v in chosen],
           "teacher": {"pool": pool, "nodes": len(nodes), "positions": positions, "candidates": len(rows), "best": best,
                       "bar": bar, "top": [{k: r[k] for k in r if k != "group"} | {"size": len(r["group"])} for r in rows[:16]]}}
    if chosen:  # the chosen variables together (compositions and feeds), on the search split and on the eval split's
        # draws, which the choice never saw
        vs = variables.parse(out["variables"], nat.C)
        for split, key in (("search", "together"), ("eval", "eval")):
            got = variables.Scorer(ge).score(tx, vs, split=split, draws=draws)
            out["teacher"][key] = {k: got[k] for k in ("e4", "f_on", "s_off", "f_comp", "f_feeds", "f_apart", "unresolved")}
            for v, r in zip(out["variables"], got["variables"]):
                v["measured"][split] = {"term": term(r), "f_on": r["f_on"], "s_off": r["s_off"]}
    if classes:  # which tokens each variable tells apart at each of its inputs (var_classes.py), for the writer
        import var_classes

        for v, pv in zip(out["variables"], variables.parse(out["variables"], nat.C)):
            v["classes"] = [var_classes.classes(sc, tx, pv, q, task["model"]) for q in pv["inputs"]]
    out["teacher"]["seconds"] = time.time() - t0
    log(f"{task['id']}: {len(chosen)} variables kept (bar {bar:.3f}), together {out['teacher'].get('together')}, eval "
        f"{out['teacher'].get('eval')}, {time.time() - t0:.0f} s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--circuits", type=Path, required=True)
    ap.add_argument("--pool", choices=("circuit", "vpd"), default="circuit")
    ap.add_argument("--out", type=Path, help="default: ~/mpd-data/graph_oracle/variables/[vpd_pool/]<split>")
    ap.add_argument("--device")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--classes", action="store_true", help="each variable's token classes at its inputs (var_classes.py)")
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(args.device))
    tasks = args.tasks or sorted((f.stem for f in args.circuits.iterdir() if re.fullmatch(r"text\d+\.json", f.name)),
                                 key=lambda s: int(s[4:]))
    for t in tasks:
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        out = args.out or (OUT / ("vpd_pool" if args.pool == "vpd" else "") / task["split"])
        out.mkdir(parents=True, exist_ok=True)
        answer = json.loads((args.circuits / f"{t}.json").read_text()) if args.pool == "circuit" else None
        res = teach(ge, task, answer, args.draws, args.pool, args.classes, log=lambda s: print(s, flush=True))
        (out / f"{t}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
