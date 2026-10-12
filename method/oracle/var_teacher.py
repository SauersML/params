"""The teacher for explanation variables (#2951 graph oracle): per teacher circuit, groups of its subcomponents that
carry what given context tokens contribute, verified by interchange in the real model (variables.py).

1. Profiles: each circuit node's read (its activation) on texts with one influential token changed (goal_eval's F2
   positions and the target's own token; the search split's draws, GoalEval._interchange_draws), against the text's:
   the mean |change| per position, exact, from real-model runs (variables.Scorer._text).
2. Grouping: a node belongs to the position whose change moves its read most (and responds to every position moving
   it at least HALF as much); a node whose largest move is below RESOLVE of its read's size responds to none and joins
   no group.
3. Verification: per input position (and per pair of neighbouring positions: a word of two tokens) three groups are
   scored alone by variables.Scorer on the search split (min(f on-target, s off-target), the headline's per-variable
   term): the nodes it moves most, every node it moves at least HALF as much as their largest move (a fuller cut), and
   the first cut to its latest position. Variables are taken best first with no node in two, kept above min(GOAL, the
   best score).
   Nodes in no variable stay in the circuit, unnamed.
4. Output: per text {"variables": [{"name", "meaning": "", "inputs", "group": [[matrix, index, position], ...],
   "feeds": [variables or "prediction"], "measured": {...}}]} (primerl's interface); feeds from the circuit's stated
   connections (a node of V writing into a node of W, or into the prediction).

  var_teacher.py TASK [TASK ...] --circuits DIR --out DIR [--device cpu] [--draws 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import goal_eval  # noqa: E402
import native  # noqa: E402
import variables  # noqa: E402

GOAL = goal_eval.GOAL
RESOLVE = 0.05  # a read that moves less than this share of its size responds to no input
HALF = 0.5  # a node's inputs: the positions moving its read at least this share of its largest move


def profiles(sc: variables.Scorer, tx, nodes: list[tuple], positions: list[int], draws: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean |change| of each node's read per position [N, P], each node's read on the text [N]) in the real model."""
    ge = sc.ge
    base = sc._text(tx.ids, tx.targets, None, check=False)["act"]
    a0 = np.array([float(base[n][t, c]) for n, t, c in nodes])
    out = np.zeros((len(nodes), len(positions)))
    for j, p in enumerate(positions):
        ds = ge._interchange_draws(tx, [p], "search", draws)
        for _, z in ds:
            x = list(tx.ids)
            x[p] = z
            act = sc._text(x, tx.targets, None, check=False)["act"]
            out[:, j] += np.abs(np.array([float(act[n][t, c]) for n, t, c in nodes]) - a0) / len(ds)
        native.free(ge.nat.dev)
    return out, a0


def candidates(nodes: list[tuple], prof: np.ndarray, a0: np.ndarray, positions: list[int]) -> tuple[dict, dict, list]:
    """(position -> the nodes it moves most, position -> the nodes it moves at least HALF as much as their largest move,
    the nodes that respond to nothing)."""
    top, half, quiet = {p: [] for p in positions}, {p: [] for p in positions}, []
    for i, nd in enumerate(nodes):
        m = prof[i].max()
        if m < RESOLVE * max(abs(a0[i]), 1e-12):
            quiet.append(nd)
            continue
        top[positions[int(prof[i].argmax())]].append(nd)
        for j, p in enumerate(positions):
            if prof[i, j] >= HALF * m:
                half[p].append(nd)
    return top, half, quiet


def as_json(name: str, inputs, group, feeds=(), measured=None) -> dict:
    v = {"name": name, "meaning": "", "inputs": sorted(int(p) for p in inputs),
         "group": [[n, int(c), int(t)] for n, t, c in sorted(group)], "feeds": list(feeds)}
    if measured is not None:
        v["measured"] = measured
    return v


def score_alone(sc: variables.Scorer, tx, inputs, group, draws: int) -> dict:
    """One variable scored alone on the search split: its min(f on, s off) and the parts."""
    vs = variables.parse([as_json("v", inputs, group)], sc.nat.C)
    got = sc.score(tx, vs, split="search", draws=draws)["variables"][0]
    f_on, s_off = got["f_on"], got["s_off"]
    term = 0.0 if f_on is None else (f_on if s_off is None else min(f_on, s_off))
    return {"score": term, "f_on": f_on, "s_off": s_off, "f_off": got["f_off"], "s_on": got["s_on"],
            "draws_on": got["draws_on"], "draws_off": got["draws_off"]}


def feeds_of(graph: native.Graph, chosen: list[dict]) -> None:
    """Each variable's feeds from the circuit's stated connections."""
    owner = {tuple(nd): v["name"] for v in chosen for nd in v["_group"]}
    for v in chosen:
        f = set()
        for r, ws in graph.parents.items():
            for w in ws:
                if tuple(w) in set(v["_group"]) and owner.get(tuple(r)) not in (None, v["name"]):
                    f.add(owner[tuple(r)])
        if any(tuple(w) in set(v["_group"]) for w in graph.out):
            f.add("prediction")
        v["feeds"] = sorted(f)


def teach(ge: goal_eval.GoalEval, task: dict, answer: dict, draws: int = 8, log=print) -> dict:
    import components

    t0 = time.time()
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    nat = ge.nat
    graph, _ = goal_eval.circuit(answer, targets, nat.C, len(ids))
    graph = goal_eval.strand(graph, targets, nat.C)
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    sc = variables.Scorer(ge)
    positions = sorted({q for q, _ in tx.f2} | set(targets))
    nodes = sorted(nd for nd in graph.node_set() if nd[0] != "tok")
    prof, a0 = profiles(sc, tx, nodes, positions, draws)
    top, half, quiet = candidates(nodes, prof, a0, positions)
    log(f"{task['id']}: {len(nodes)} nodes, positions {positions}; nodes moved most per position "
        f"{ {p: len(top[p]) for p in positions} }, {len(quiet)} respond to none; profiles {time.time() - t0:.0f} s")
    tried = {}

    def trial(inputs, group, how):
        key = (tuple(sorted(inputs)), tuple(sorted(group)))
        if not group or key in tried:
            return
        r = score_alone(sc, tx, inputs, group, draws)
        tried[key] = {**r, "how": how}
        log(f"  {how}: inputs {list(key[0])}, {len(group)} nodes -> {r['score']:.3f} (f_on {r['f_on']}, s_off {r['s_off']})")

    def variants(ins, g_top, g_half, how):
        trial(ins, g_top, how)
        trial(ins, g_half, how + ", every node it moves")
        if g_top:
            last = max(nd[1] for nd in g_top)
            trial(ins, [nd for nd in g_top if nd[1] == last], how + ", latest position")
    for p in positions:
        variants((p,), top[p], half[p], "position")
    for p, q in zip(positions, positions[1:]):  # neighbours (a word of two tokens)
        if q == p + 1:
            variants((p, q), sorted(set(top[p]) | set(top[q])), sorted(set(half[p]) | set(half[q])), "neighbours")
    ranked = sorted(tried.items(), key=lambda kv: -kv[1]["score"])
    best = ranked[0][1]["score"] if ranked else 0.0
    bar = min(GOAL, best)
    chosen, used = [], set()
    for (ins, g), r in ranked:
        if r["score"] < bar or r["score"] <= 0 or used & set(g):
            continue
        used |= set(g)
        toks = components.token_strings(task["model"], [ids[q] for q in ins])
        name = "tokens " + " ".join(f"{q}:{tok!r}" for q, tok in zip(ins, toks))
        chosen.append({"name": name, "_inputs": ins, "_group": list(g), "measured": {k: r[k] for k in r}})
    feeds_of(graph, chosen)
    out = {"variables": [as_json(v["name"], v["_inputs"], v["_group"], v["feeds"], v["measured"]) for v in chosen],
           "teacher": {"positions": positions, "candidates": len(tried), "best": best, "bar": bar, "seconds": time.time() - t0,
                       "quiet_nodes": len(quiet), "nodes": len(nodes)}}
    if chosen:  # the chosen variables together: compositions and feeds
        vs = variables.parse(out["variables"], nat.C)
        got = sc.score(tx, vs, split="search", draws=draws)
        out["teacher"]["together"] = {"e4": got["e4"], "f_comp": got["f_comp"], "f_feeds": got["f_feeds"], "f_apart": got["f_apart"]}
    log(f"{task['id']}: {len(chosen)} variables kept (bar {bar:.3f}), {time.time() - t0:.0f} s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device")
    ap.add_argument("--draws", type=int, default=8)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(args.device))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for t in args.tasks:
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        res = teach(ge, task, json.loads((Path(args.circuits) / f"{t}.json").read_text()), args.draws,
                    log=lambda s: print(s, flush=True))
        (Path(args.out) / f"{t}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
