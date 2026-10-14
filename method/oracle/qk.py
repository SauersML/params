"""A circuit's stated query-key pairs for teachers (#2951 graph oracle; from rustmove's 10-12 q-k measurements): per
attention read the circuit uses, the pairs (query subcomponent or the rest x key
subcomponent or the rest) added greedily by goal_eval's own runs on graphs with g.pairs (monitorbench's engine pair
semantics: a read with an entry computes its row from the background and its stated pairs), in C's semantics or the
path-patched one (s_track_pp). stated_pairs() gives native nodes, pair_edges() the answer's JSON pair edges."""
from __future__ import annotations

import copy
import math

import torch

import goal_eval
import native

GOAL = goal_eval.GOAL
KL_FLOOR = 1e-9  # KLs at or below float rounding of 0 (a near-exact circuit) before logs and fractional powers


@torch.no_grad()
def stated_pairs(ge, task: dict, answer: dict, semantics: str = "c", split: str = "search") -> dict:
    """For teachers: a circuit's stated query-key pairs, {(layer, position): [(query node, key node), ...]} (nodes as
    (matrix, position, index), or "rem" for the part outside the circuit), measured by goal_eval's own runs on graphs
    with g.pairs (monitorbench's engine pair semantics: a read with an entry computes its row from the background and
    its stated pairs). Per read, the circuit's other reads left as they are, pairs (query node or rest x key node or
    rest) are added greedily:
      semantics "c":  by the circuit's target KL in C's semantics (GoalEval.graph_kl), until it is within the circuit's
                      budget (the KL at F1 GOAL, or GOAL of its own F1 when below);
      semantics "pp": by s_track_pp on the split's changed texts (GoalEval.track_pp: unstated pattern terms from the
                      model on x, stated pairs from the patched run), until GOAL of the gain from stating every
                      candidate pair over stating none is recovered.
    A read that needs none states ["pair", "rest", "rest"]: its pattern is the background alone (C) or x's (pp)."""
    p = task["prompts"][0]
    targets = p["target_positions"]
    ids = p["token_ids"][:max(targets) + 1]
    nat = ge.nat
    graph, _ = goal_eval.circuit(answer, targets, nat.C, len(ids))
    graph = goal_eval.strand(graph, targets, nat.C)
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    nodes = sorted(graph.node_set())
    reads = sorted({(native._layer_kind(nd)[0], nd[1]) for nd in nodes if native._layer_kind(nd)[1] == "o_proj"})
    base = {**(getattr(graph, "pairs", None) or {})}

    def with_pairs(read, ps):
        g = copy.copy(graph)
        g.pairs = {**base, read: list(ps) or [(None, None)]}
        return g

    if semantics == "c":
        empty, vpd = (max(x, KL_FLOOR) for x in ge._bounds(tx, "live"))
        kl_c = max(ge.graph_kl(tx, [graph], live=True)[0], KL_FLOOR)
        f1 = math.log(empty / kl_c) / math.log(empty / vpd) if empty != vpd else 1.0
        budget = empty * (vpd / empty) ** GOAL if f1 >= GOAL else empty * (kl_c / empty) ** GOAL
        score = lambda gs: [-k for k in ge.graph_kl(tx, gs, live=True)]  # noqa: E731
        enough = lambda s, lo, hi: -s <= budget  # noqa: E731
    else:
        score = lambda gs: [r["s_track_pp"] if r["s_track_pp"] is not None else 0.0 for r in ge.track_pp(tx, gs, split=split)]  # noqa: E731
        enough = lambda s, lo, hi: hi <= lo or s >= lo + GOAL * (hi - lo)  # noqa: E731
    out = {}
    for layer, t in reads:
        qs = [nd for nd in nodes if native._layer_kind(nd) == (layer, "q_proj") and nd[1] == t]
        ks = [nd for nd in nodes if native._layer_kind(nd) == (layer, "k_proj") and nd[1] <= t]
        cands = [(q, k) for q in qs + [None] for k in ks + [None] if q is not None or k is not None]
        read = (layer, t)
        lo, hi = score([with_pairs(read, []), with_pairs(read, cands)]) if cands else (0.0, 0.0)
        chosen, cur = [], lo
        left = list(cands)
        while left and not enough(cur, lo, hi):
            got = score([with_pairs(read, chosen + [c]) for c in left])
            j = max(range(len(left)), key=lambda i: got[i])
            chosen.append(left.pop(j))
            cur = got[j]
        out[read] = [("rem" if q is None else q, "rem" if k is None else k) for q, k in chosen]
    return out


def pair_edges(ge, task: dict, answer: dict, semantics: str = "c", split: str = "search") -> list:
    """stated_pairs as the answer's JSON pair edges (goal_eval.circuit): ["pair", query, key, layer, position], query
    and key a node's place in answer["nodes"] or "rest"; a read that needs none, ["pair", "rest", "rest", layer,
    position]."""
    at = {}
    for i, (m, c, t) in enumerate(answer.get("nodes", [])):
        if m != "tok":
            at[(goal_eval._site(m), int(t), int(c))] = i
    out = []
    for (layer, t), pairs in stated_pairs(ge, task, answer, semantics, split).items():
        out += [["pair", at[a] if a != "rem" else "rest", at[b] if b != "rem" else "rest", layer, t] for a, b in pairs]
        out += [] if pairs else [["pair", "rest", "rest", layer, t]]
    return out
