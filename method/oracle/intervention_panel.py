"""Common node-removal worlds for comparing circuit predictions; isolated from the reward.

All candidates see the text/split's same removals of reachable VPD-active nodes.
The empty and VPD predictions calibrate pooled KL in those same worlds. A weak
reference makes the entire panel unresolved, never only a particular candidate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

VERSION = "common-node-panel-v1"


def bank(tx, split: str = "eval", draws: int = 32) -> dict:
    """Deterministic, candidate-independent removal bank; no model calls."""
    if split not in ("search", "eval") or draws < 1:
        raise ValueError("split is search/eval and draws must be positive")
    if tx.vpd is None:
        raise ValueError("a common panel requires the text's VPD-active set")
    nodes = sorted((n, int(t), int(c)) for n, mask in tx.vpd.items()
                   for t, c in mask.nonzero().tolist())
    identity = {"version": VERSION, "seed": tx.seed, "ids": tx.ids,
                "targets": tx.targets, "split": split, "draws": draws, "nodes": nodes}
    seed = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    rng = random.Random(int(seed, 16))
    drops = []
    for j in range(draws):
        removed = [nd for nd in nodes if rng.random() < (j + 0.5) / draws]
        if nodes and not removed:
            removed = [nodes[rng.randrange(len(nodes))]]
        drops.append(removed)
    identity["removed"] = drops
    identity["bank_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def _score(goal_eval, candidate, empty, vpd):
    return goal_eval._fid_raw(empty, candidate, vpd)


def _eps(goal_eval, candidate, empty, vpd):
    """Twice the paired bootstrap SE of the exact pooled score estimator."""
    import numpy as np

    rng = np.random.default_rng(0)
    scores = []
    for _ in range(goal_eval.EPS_BOOT):
        ix = rng.integers(len(candidate), size=len(candidate))
        got = _score(goal_eval, float(np.mean(candidate[ix])), float(np.mean(empty[ix])), float(np.mean(vpd[ix])))
        if got is not None:
            scores.append(got)
    return (max(goal_eval.EPS_FLOOR, 2 * float(np.std(scores, ddof=1))) if len(scores) > 1
            else goal_eval.EPS_FLOOR), len(scores)


def _without(goal_eval, graph, removed):
    """Delete only named nodes/incident connections; preserve explicit token inputs."""
    h = goal_eval._drop_nodes(graph, set(removed))
    h.tok_edges = [(t, r) for t, r in getattr(graph, "tok_edges", []) if r not in set(removed)]
    h.tok_positions = list(getattr(graph, "tok_positions", []))
    h.tok_out = list(getattr(graph, "tok_out", []))
    return h


def evaluate(ge, tx, graphs, split: str = "eval", draws: int = 32) -> list[dict]:
    """Aligned circuit results on a common panel; does not modify production reward.

    score is clipped reporting; raw_score is log recovery from the common empty
    prediction to VPD, after pooling paired KLs. W remains diagnostic only.
    resolved is common to all candidates (E > V and E > 0); bootstrap eps uses
    the same paired pooling. Models/circuits always execute with live remainders.
    """
    import numpy as np
    import torch
    import carry
    import engine
    import goal_eval
    import native
    import vpd_model

    panel = bank(tx, split, draws)
    nat = ge.nat
    sources = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for m in (carry, engine, goal_eval, native, vpd_model)}
    sources["intervention_panel"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache = tx.__dict__.setdefault("_common_node_panels", {})
    with torch.no_grad():
        if panel["bank_sha256"] not in cache:
            cols = {n: torch.tensor(sorted({c for m, _, c in panel["nodes"] if m == n}),
                                    dtype=torch.long, device=nat.dev) for n in nat.names}
            at = {n: {c: i for i, c in enumerate(cols[n].tolist())} for n in nat.names}
            Em = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols, outside="on")
            Ec = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols)
            ones = {n: torch.ones(draws, tx.T, len(cols[n]), device=nat.dev) for n in nat.names}
            vm = {n: tx.vpd[n][:, cols[n]].float()[None].repeat(draws, 1, 1) for n in nat.names}
            for j, removed in enumerate(panel["removed"]):
                for n, t, c in removed:
                    ones[n][j, t, at[n][c]] = 0.0
                    vm[n][j, t, at[n][c]] = 0.0
            lm = Em.log_probs(ones)
            lv = Ec.log_probs(vm)
            le = Ec.log_probs({n: torch.zeros(1, tx.T, len(cols[n]), device=nat.dev) for n in nat.names})
            cache[panel["bank_sha256"]] = {"model": lm,
                "empty": carry.kl(lm, le.expand_as(lm)).cpu().numpy(),
                "vpd": carry.kl(lm, lv).cpu().numpy(),
                "W": carry.kl(lm, tx.ref.expand_as(lm)).cpu().numpy()}
            del Em, Ec, ones, vm, lv, le
            native.free(nat.dev)
        ref = cache[panel["bank_sha256"]]
        e, v = float(np.mean(ref["empty"])), float(np.mean(ref["vpd"]))
        resolved = e > 0 and v < e
        out = []
        for graph in graphs:
            g = goal_eval._c_graph(graph, tx.targets, nat.C)
            nodes = sorted(g.node_set())
            cols = {n: torch.tensor(sorted({c for m, _, c in nodes if m == n}),
                                    dtype=torch.long, device=nat.dev) for n in nat.names}
            E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols)
            if g.complete:
                at = {n: {c: i for i, c in enumerate(cols[n].tolist())} for n in nat.names}
                masks = {n: torch.zeros(draws, tx.T, len(cols[n]), device=nat.dev) for n in nat.names}
                for j, removed in enumerate(panel["removed"]):
                    removed = set(removed)
                    for n, t, c in nodes:
                        if (n, t, c) not in removed:
                            masks[n][j, t, at[n][c]] = 1.0
                lc = E.log_probs(masks)
            else:
                variants = [_without(goal_eval, g, d) for d in panel["removed"]]
                lc = E.graph_log_probs(E.plan(variants, semantics="c"))
            kc = carry.kl(ref["model"], lc).cpu().numpy()
            k = float(np.mean(kc))
            raw = _score(goal_eval, k, e, v) if resolved else None
            eps, boots = _eps(goal_eval, kc, ref["empty"], ref["vpd"]) if resolved else (None, 0)
            out.append({"version": VERSION, "split": split, "bank_sha256": panel["bank_sha256"],
                "resolved": resolved, "unresolved_reason": None if resolved else "common empty does not exceed VPD error",
                "score": None if raw is None else min(1.0, max(0.0, raw)), "raw_score": raw, "eps": eps,
                "bootstrap_resolved": boots, "bootstrap_total": goal_eval.EPS_BOOT,
                "kl_candidate": k, "kl_empty": e, "kl_vpd": v, "W": float(np.mean(ref["W"])),
                "source_sha256": sources,
                "draws": [{"index": j, "removed": panel["removed"][j], "kl_candidate": float(kc[j]),
                           "kl_empty": float(ref["empty"][j]), "kl_vpd": float(ref["vpd"][j]),
                           "W": float(ref["W"][j]), "model_top": ref["model"][j].argmax(-1).tolist()}
                          for j in range(draws)]})
            del E, lc
            native.free(nat.dev)
    return out
