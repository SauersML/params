"""goal_eval's families for answers with block nodes, token edges and stated pairs, on one text (eval split):
s_point_c (min(graph F1, every-connection F1) in C's semantics, blocks expanded into their subcomponents), s_track_pp
and s_interchange_pp (path-patched, blocks as blocks), s_interchange (C's semantics) and donor resampling (near and far
donors); c5, the harmonic mean of s_point_c, s_track_pp, s_interchange_pp, s_patching_near and s_patching_far over those
defined (goal_eval's C without the node, edge and flip families)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))

import goal_eval  # noqa: E402
import native  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
# smaller batches than goal_eval's (the same runs, fewer at once): a block is a site's every subcomponent, so a circuit's
# engine spans tens of thousands of columns
goal_eval.CHANGED_ROWS = int(os.environ.get("CHANGED_ROWS", 16))
goal_eval.DONOR_ROWS = int(os.environ.get("DONOR_ROWS", 4))
TERMS = ("s_point_c", "s_track_pp", "s_interchange_pp", "s_patching_near", "s_patching_far")


def text(ge, task: str):
    p = TEXTS / f"{task}.json"
    ids, targets = native.text(p)
    return ge.text(ids, targets, native.task_seed(task), json.loads(p.read_text()))


def bits_active(ge, tx, g) -> float:
    """An answer's description bits with each block priced at its active contents: a block at (site, p) counts as the
    subcomponents VPD's answer has active there (causal importance above zero), each named among the positions and every
    subcomponent like any node; a stated graph's edges as stated (an edge into or out of a block once, its ends indexed
    among the expanded nodes), a complete graph's every connection the model has among the expanded nodes (as
    goal_eval.size_bits prices a complete graph)."""
    import math

    nat = ge.nat
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    if g.masks is not None:
        return goal_eval.size_bits(g, P, S, tx.targets)
    nodes = []
    for n, t, c in g.node_set():
        if c == goal_eval.BLOCK:
            nodes += [(n, t, int(j)) for j in tx.vpd[n][t].nonzero().flatten().tolist()]
        else:
            nodes.append((n, t, c))
    if g.complete:
        return goal_eval.size_bits(native.Graph(sorted(set(nodes)), complete=True), P, S, tx.targets)
    pos = sorted(set(getattr(g, "tok_positions", [])) | {q for q, _ in getattr(g, "tok_edges", [])})
    n, k = len(set(nodes)), len(pos)
    inner = sum(len(w) for w in g.parents.values()) + len(getattr(g, "tok_edges", [])) + len(getattr(g, "tok_out", []))
    pairs = sum(len(v) for v in (getattr(g, "pairs", None) or {}).values())
    return (n * math.log2(P * S) + k * math.log2(P) + math.log2(max(2, n + k)) * (2 * (inner + pairs) + len(g.out))
            + getattr(g, "stranded_bits", 0.0))


def _num(v):
    return float(v) if hasattr(v, "item") else v


def node_set_graph(g, C) -> native.Graph:
    """A graph's node set as a complete graph, its blocks as every subcomponent of their sites (no connection lists:
    expand_blocks on a stated graph lists every block-to-block connection)."""
    nodes = set()
    for n, t, c in g.node_set():
        if c == goal_eval.BLOCK:
            nodes.update((n, t, j) for j in range(C[n]))
        else:
            nodes.add((n, t, c))
    return native.Graph(sorted(nodes), complete=True)


def score(ge, tx, graphs: list, draws: int = 16, near: int = goal_eval.DONORS_NEAR, far: int = goal_eval.DONORS_FAR) -> list[dict]:
    """A stated graph with block nodes gets the path-patched terms (blocks as blocks) and s_wiring (its node set's masked
    run) alone: in C's semantics its block-to-block connections expand into every pair of the sites' subcomponents (an
    MLP output block into the next MLP input block alone is 11 million connections), beyond any engine here; its
    complete form (every connection) is scored in full."""
    nat = ge.nat
    pp = [goal_eval.strand(g, tx.targets, nat.C) for g in graphs]
    heavy = {i for i, g in enumerate(pp) if not g.complete and any(nd[2] == goal_eval.BLOCK for nd in g.node_set())}
    cg = [node_set_graph(g, nat.C) if i in heavy else goal_eval._c_graph(g, tx.targets, nat.C) for i, g in enumerate(pp)]
    light = [i for i in range(len(graphs)) if i not in heavy]
    e0 = ge._bounds(tx, "live")[0]
    none = {"f1_live": None, "kl_live": None, "n_live": None}
    point = dict(zip(light, ge.sufficiency(tx, [cg[i] for i in light]))) if light else {}
    idx = [i for i, g in enumerate(cg) if not g.complete or i in heavy]
    wired = dict(zip(idx, ge.sufficiency(tx, [cg[i] if i in heavy else native.Graph(sorted(cg[i].node_set()), complete=True)
                                              for i in idx]))) if idx else {}
    tpp = ge.track_pp(tx, pp, "eval", draws)
    native.free(nat.dev)
    ipp = ge.interchange_pp(tx, pp, "eval", draws)
    native.free(nat.dev)
    inter = dict(zip(light, ge._interchange_family(tx, [cg[i] for i in light], draws, split="eval"))) if light else {}
    native.free(nat.dev)
    dns = dict(zip(light, ge.donors(tx, [cg[i] for i in light], "eval", near,
                                    ge.far_donors(tx, ge.far_pool("eval"), far) if far else None, rows=goal_eval.DONOR_ROWS))) if light else {}
    native.free(nat.dev)
    scale = ge.size_scale(tx)
    out = []
    for i, (g, gp) in enumerate(zip(cg, pp)):
        pt = point.get(i, none)
        s_point = pt["f1_live"]
        s_wiring = wired[i]["f1_live"] if i in wired else s_point
        it = inter.get(i)
        row = {"nodes": g.count(), "listed": gp.count(), "blocks": sum(1 for nd in gp.node_set() if nd[2] == goal_eval.BLOCK),
               "stranded": len(getattr(gp, "stranded", [])), "positions": len({t for _, t, _ in gp.node_set()}),
               "families": "pp and wiring" if i in heavy else "all",
               "kl_point": pt["kl_live"], "s_point": s_point, "s_wiring": s_wiring,
               "s_point_c": min(x for x in (s_point, s_wiring) if x is not None) if s_point is not None else None,
               "n_live": pt["n_live"], "s_track_pp": tpp[i]["s_track_pp"], "prompts_track_pp": tpp[i]["prompts_track_pp"],
               "s_interchange_pp": ipp[i]["s_interchange_pp"],
               "s_interchange": goal_eval._fid(e0, it["kl_interchange_c"], it["kl_interchange_vpd"])
               if it and it["kl_interchange_c"] is not None else None,
               "s_patching_near": dns.get(i, {}).get("s_patching_near"), "s_patching_far": dns.get(i, {}).get("s_patching_far")}
        row["bits_active"] = bits_active(ge, tx, gp)
        # the scorecard's size_bits (a block at its active contents) for a stated graph; a complete graph's blocks it
        # counts as one node each, so there the active expansion's (bits_active)
        row["bits"] = (goal_eval.size_bits(gp, nat.positions(tx.targets), sum(nat.C.values()), tx.targets, active=tx.vpd)
                       if not gp.complete else row["bits_active"])
        row["size_active"] = goal_eval.size_term(row["bits_active"], scale) if scale else None
        row["vpd_answer_bits"] = scale
        terms = [row[k] for k in TERMS if row[k] is not None]
        row["c5"] = goal_eval._harmonic(terms) if terms and i not in heavy else None
        out.append({k: _num(v) for k, v in row.items()})
    return out
