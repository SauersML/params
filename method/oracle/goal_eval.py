"""The goal scorecard (#2951 graph oracle): how well an answer meets the goal, one implementation for the RL reward and
every evaluation.

The goal: the oracle states the whole-model circuit of vpd4l's prediction on a text in VPD subcomponents, causally
faithful to the model's weights, with correct English and question answering. An answer is a circuit on one text:
nodes (weight matrix, subcomponent index, position) and explicit edges (upstream node -> downstream node, or -> the
prediction), each a connection the model has (mech.connects: the residual stream at one position, a value into its
layer's attention output at that or a later position, an MLP input into its MLP's output); a stated pair the model
does not connect is dropped and counted. Queries and keys are nodes without edges out: they enter their layer's
attention pattern. A token-embedding input node ["tok", 0, position] may feed the residual readers at its position (the
embedding, which VPD does not decompose, is how a token's change reaches a circuit); its edges are counted, not cut.

Runs use VPD's semantics: a run "with circuit C" has C's subcomponents on (mask 1) and every other one off (mask 0), the
remainders W - V U either on ("live", computing on the run's own inputs) or off; both conventions are reported. An
answer runs as the graph it states (GoalEval.graph_ref, the lead's decision 10-12; a node other than a query or key with
no stated connection is a defect, dropped before every measure with its bits still charged: strand): inside the circuit, graph semantics,
a circuit node's output reaching a circuit reader, or the prediction, only through a stated edge (each reader reads the
background and its stated parents); outside it, masked semantics, every other subcomponent (at its mask) and the
remainders reading the full residual stream, the circuit's outputs included, as in the real model, and writing into the
background with the token embedding; query and key nodes enter their layer's attention pattern (the format gives them
no edges out). Every connection stated is the masked run of the circuit's subcomponents; none stated, remainders off,
is the empty circuit (live, the remainders carry the circuit's outputs into the background: graph F1 of the 20
held-out teacher node sets stating no edge is 0.0019 at the median, 0.0065 at most, the lead's bar 0.02). Baselines (VPD's answer, its top
k, the empty circuit) are complete graphs: their subcomponents read each other in full (VPD's masked forward). Every
KL is the model's next-token distribution at the targets (summed over them) from the run's, in bits. VPD's own answer
is the set of subcomponents with causal importance above zero on the input (at positions that can reach the targets,
Native.reachable); the empty circuit and it bound every answer, on a log scale (KL_empty is ~100 bits, so a linear
scale gives circuits 0.5 and 2.5 bits from the model both about 0.98):
  F1 = clip(ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD), 0, 1)
  N = clip(ln(KL_{VPD minus C} / KL_VPD) / ln(KL_empty / KL_VPD), 0, 1), "VPD minus C" VPD's own answer with the
      circuit's subcomponents off: how much of what VPD's answer computes needs the circuit.

Reward fields, on the text (per remainder convention X, live or off):
  f1_X and n_X as above, beside KL_C (kl_X), KL_empty, KL_VPD (VPD's own faithfulness on the text, the ceiling) and
     KL_{VPD minus C}; the linear forms are diagnostics: recovered_X = (KL_empty - KL_C) / (KL_empty - KL_VPD),
     completeness_X = min(1, recovered / GOAL), necessity_X = clip((KL_{VPD minus C} - KL_VPD) / (KL_empty - KL_VPD),
     0, 1), and recovered_log_X, the unclipped F1.
  F3 (a diagnostic, not a reward field: F1 on the stated graph already scores the wiring), edges: in the circuit's
     complete run (remainders off), each stated edge cut (path patching of that one connection: the
     reader reads every input but that writer's output; native.Native.run's edge strengths) against random pairs of
     the circuit's nodes that the model connects and the answer does not state, each cut's effect the KL of the
     circuit's run from the cut run; F3 is the probability that a stated cut moves the prediction more than an
     unstated one (AUC), and 0 when the answer states no edge or leaves no connected pair unstated (an answer earns
     edge credit only by saying which connections matter). A listed embedding node's edges are cut the same way (the
     reader reads its input without the token's embedding) and its unstated readers join the unstated pairs.
  bits: the description length of the nodes and edges (native.Graph.bits; with embedding nodes, size_bits: each one a
     choice among the positions, its edges priced like any other); size = size_term(bits, size_scale), log2(1 + bits)
     / log2(1 + the bits of naming VPD's whole answer at the positions that can reach the targets), at most 1: each
     doubling of an answer costs the same.
  E1 (the English reader's bits) and E2 (question-answering accuracy): hooks (GoalEval.hooks) that the RL owner fills.
Diagnostics (score(diagnostics=True)), not reward fields: F1 on the text's changed prompts (Native.changes) and F2 on
the held-out alternatives (carry.alternatives, ranks 3 to 5) at each position whose changed token moves the prediction
by at least 1% of the text's total (Native.sensitivity), the same circuit identities run on each, recovered pooled
over the prompts against VPD's own answer and the empty circuit on each; and interchange carry (primerl's rule b):
at a changed position p, every subcomponent at p, every one inactive on both the text and the changed prompt, the
circuit's and the remainders compute, everything else held at its value on the text (carry.Carry), carried = 1 - KL /
movement, beside B_p, the same with every subcomponent active on the text after p named, and completeness_p =
min(1, carried / (GOAL x B_p)); target positions are left out (trivially carried).
Also diagnostics: n_real_X, the circuit's necessity in the real model (its subcomponents removed from the model's
weights at its positions, everything else on, remainders included; the same log scale with KL_{model minus C} in place
of KL_{VPD minus C}), comparable with native_units' N; and tail_split(), how much of a KL is the shape of the model's
tail (the chain rule over the nucleus tokens and one "rest" bucket).
Whole-model path patching of single connections (cuts()) serves question answering and teachers.

Baselines, scored by the same code: the empty circuit, VPD's own answer at the answer's size (its causal-importance
order, native.Native._vpd_order), and VPD's own answer.

  goal_eval.py score --set NAME=DIR [--set NAME=DIR ...] --split heldout --n N [--diagnostics] [--out OUT.json]
DIR holds each text's answer, <task>.json (circuit JSON) or <task>.py (a traced graph program, mech), and further
samples <task>.<k>.json or .py (k = 1, 2, ...).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import mech  # noqa: E402
import native  # noqa: E402

HELD_OUT_RANKS = (3, 4, 5)  # carry.alternatives' ranks for F2; the carry search and its training answers use 0 to 2
SHARE_FLOOR = 0.01  # F2's positions: those moving the prediction by at least this share of the text's total
GOAL = 0.9  # completeness: the share of VPD's recovery that counts as complete
RANDOM_PAIRS = 32  # F3's unstated pairs per answer
MAX_EDGES = 64  # F3's stated edges per answer (a uniform draw when an answer states more)
CHUNK = 8  # circuits per masked forward pass
CUT_CHUNK = 32  # cuts per batch
CONVENTIONS = ("live", "off")  # the remainders' convention
DRAWS = 32  # C's random interventions per family and circuit
ADV_STEPS = 20  # the adversary's budget: VPD's published evaluation (param-decomp's pile_llama_simple_mlp-4L.yaml,
ADV_STEP = 0.1  # PGDReconLoss_20step: random init, sign-gradient steps of 0.1 clamped to [0, 1])
ADV_RESTARTS = 4
ADV_PATTERNS = 64  # patterns drawn from the pool per text
ADV_RANDOM_ON = 1 / 16  # a random member's chance of switching each drawn pattern on
ADV_CURVE = (0, 4, 20)  # budgets reported (steps)
ADV_BATCH = 64  # the random members' rows per run (every circuit's restarts run together in the steps)
FLIPS = {"omitted": (1, 4, 16, 64, 256), "patterns": (1, 2, 4, 8)}  # adversary_flips' budgets: how many switched on
FLIP_TERM = 4  # the flip budget C's adversarial member (and the reward's) takes
SCREEN = 256  # the adversary's shortlist measured one by one: the parts with the largest gradient either way (a part's
# whole removal can move the other way from its gradient: text-free tiny model, the best single removal ranks last)
SCREEN_ROWS = 16  # rows per run when measuring the shortlist
CHANGED_ROWS = 64  # the changed-input family's circuit rows per run on the changed texts (circuits x changed texts)
FLIP_ROWS = 160  # adversary_flips' rows per run: every circuit's restarts at every budget (8 x 5 x 4) together
INTERCHANGE_DRAWS = 16  # C's interchange draws per circuit
CHANGED_DRAWS = 16  # C's changed texts per circuit
INTERCHANGE_SPLITS = {"search": (1, (0, 2), False), "eval": (2, (1,), True)}  # interchange draws: the text's seed
# offset, carry.alternatives' ranks, and whether samples from the model's distribution (no split's ranks) fill the budget
FRESH_SOURCES = 64  # sampled source tokens kept per position (the first ones fill a budget)
BLOCK = -1  # a block node's index: (site, position, BLOCK), the site's whole output there (every subcomponent and the
# remainder), ["block", site, position] in an answer (the lead's 10-12: a token's identity lives in thousands of parts)
VPD4L_SITES = 24  # vpd4l's weight matrices (4 layers x 6), a block's name among them
DONORS_NEAR = 16  # donor resampling's changed-input donors per text
DONOR_ROWS = 16  # donor resampling's rows per run (donors x circuits and VPD's answer)
DONORS_FAR = 8  # donor resampling's other texts per text
DONOR_POOLS = {"search": ("train", range(0, 400)),  # the far donors' texts per split: the reward's from training texts,
               "eval": ("heldout", range(150, 200))}  # evaluation's from held-out text9150-9199 (no search drew on them)


# ---- answers


def _site(name) -> str:
    """A weight matrix: native's name ("h.2.attn.v_proj") or mech's short form ("2.v")."""
    if isinstance(name, str) and name.startswith("h."):
        return name
    layer, kind = str(name).split(".")
    return native.site_name(int(layer), mech.SITES[kind])


def circuit(answer: dict, targets: list[int], C: dict, T: int) -> tuple[native.Graph, dict]:
    """(the Graph of an answer in JSON, its defects). JSON: {"nodes": [[matrix, index, position], ...], "edges":
    [[upstream, downstream], ...], "explanation": str}; an edge end is a node's place in the list, downstream "out" for
    the prediction; a matrix is native's name or "L.kind" (kind q, k, v, o, fc, down), or "tok" for the token
    embedding at the position (index 0), which feeds residual readers at its position (Graph.tok_edges: (position,
    reader)), and "variables" (variables.parse's form, kept as g.variables for the E4 hook). An edge ["pair", query,
    key, layer, position] states a query-key pair of the read (layer, position) (g.pairs, monitorbench's bcbddad):
    query a node's place (a query node of that layer at that position) or "rest", key a node's place (a key node of
    that layer at or before it) or "rest"; ["pair", "rest", "rest", layer, position] states the read with no pair
    beyond the background. Nodes outside the model or
    the text, edges the model does not have (mech.connects), and variables naming a part outside the model are dropped
    and counted."""
    nodes, bad_nodes = [], 0
    for m, c, t in answer.get("nodes", []):
        if m == "block":  # a site's whole output at a position
            try:
                n = _site(c)
            except (ValueError, KeyError):
                n = None
            ok = n in C and 0 <= int(t) <= max(targets)
            nodes.append((n, int(t), BLOCK) if ok else None)
            bad_nodes += not ok
            continue
        if m == "tok":
            ok = int(c) == 0 and 0 <= int(t) <= max(targets)
            nodes.append(("tok", int(t), 0) if ok else None)
            bad_nodes += not ok
            continue
        try:
            n = _site(m)
        except (ValueError, KeyError):
            n = None
        if n in C and 0 <= int(c) < C[n] and 0 <= int(t) <= max(targets):
            nodes.append((n, int(t), int(c)))
        else:
            bad_nodes += 1
            nodes.append(None)
    if answer.get("complete"):  # a node set claiming every connection among its nodes (and every token edge)
        g = native.Graph([nd for nd in nodes if nd is not None and nd[0] != "tok"], complete=True)
        g.stranded, g.stranded_bits = [], 0.0
        return g, {"invalid_nodes": bad_nodes, "invalid_edges": 0, "stranded_nodes": 0, "invalid_variables": 0}
    parents, out, tok, bad_edges, pairs, tok_out = {}, [], [], 0, {}, []
    for e in answer.get("edges", []):
        if isinstance(e, (list, tuple)) and len(e) == 5 and e[0] == "pair":
            got = _pair_entry(e, nodes, targets)
            if got is None:
                bad_edges += 1
            else:
                read, pr = got
                ps = pairs.setdefault(read, [])
                if pr != (None, None) and pr not in ps:
                    ps.append(pr)
            continue
        try:
            up, down = e
            w = nodes[up]
            r = None if down == "out" else nodes[down]
        except (IndexError, TypeError, ValueError):
            bad_edges += 1
            continue
        if w is None or (down != "out" and r is None):
            bad_edges += 1
        elif w[0] == "tok":
            if r is not None and r[0] != "tok" and native._name_layer_kind(r[0])[1] in mech.RESID_READERS and r[1] == w[1]:
                tok.append((w[1], r))
            elif r is None and w[1] in targets:  # the token's embedding into the prediction at its own position
                tok_out.append(w[1])
            else:
                bad_edges += 1
        elif (r is not None and r[0] == "tok") or not native.connects(w, r, targets):
            bad_edges += 1
        elif r is None:
            out.append(w)
        else:
            parents.setdefault(r, []).append(w)
    g = native.Graph([nd for nd in nodes if nd is not None and nd[0] != "tok"], {r: list(dict.fromkeys(ws)) for r, ws in parents.items()},
                     list(dict.fromkeys(out)))
    g.tok_edges = list(dict.fromkeys(tok))
    g.tok_positions = sorted({nd[1] for nd in nodes if nd is not None and nd[0] == "tok"})
    if tok_out:
        g.tok_out = sorted(set(tok_out))
    if pairs:
        g.pairs = pairs
    g.variables, bad_variables = [], 0
    if answer.get("variables"):
        import variables  # (it imports this module)
        for v in answer["variables"]:
            try:
                g.variables += variables.parse([v], C)
            except (ValueError, KeyError, TypeError, IndexError):
                bad_variables += 1
    g = strand(g, targets, C)
    return g, {"invalid_nodes": bad_nodes, "invalid_edges": bad_edges, "stranded_nodes": len(g.stranded),
               "invalid_variables": bad_variables}


def strand(g: native.Graph, targets: list[int], C: dict) -> native.Graph:
    """A stated graph without its stranded nodes, those with no stated path to the prediction: they do nothing in the
    stated graph's run, yet would earn credit in the runs with every connection restored (wiring, N) by cancelling what
    the stated graph cuts (the lead's decisions 10-12; pymove: 122 of v3's 201 nodes on text9000 had a connection but
    no path, an edge in only or a branch ending nowhere). A node other than a query or key needs a stated path to the
    prediction; a query at t, an output node of its layer at t that has one; a key at p, an output node of its layer at
    a position at or after p that has one (queries and keys act through the attention pattern); a kept query's or key's
    stated writers are on a path through it. With stated query-key pairs (g.pairs {(layer, t): [(query or None, key
    or None)]}, monitorbench's bcbddad: a read with an entry computes its pattern row from the background and those
    pairs alone), a query or key acts in a read only if the read has no entry or one of its pairs names it, so a query
    whose read states pairs without it, or a key every such read leaves out, is stranded too, and the pairs naming a
    dropped node go with it. They are dropped before every family, with their connections and any
    embedding edge into them; their bits stay charged in size (stranded_bits, which size_bits adds). g.stranded lists
    them. A complete graph is unchanged. The answer's variables (g.variables) carry over."""
    if hasattr(g, "stranded"):  # already done (its record kept)
        return g
    if g.complete:
        g.stranded, g.stranded_bits = [], 0.0
        return g
    kind = {nd: native._name_layer_kind(nd[0]) for nd in g.node_set()}
    pairs = getattr(g, "pairs", None) or {}
    named = {read: {x for pr in ps for x in pr if x is not None} for read, ps in pairs.items()}

    def acts(nd, L, t):  # a query or key node acts in read (L, t): no pairs stated there, or one naming it
        return (L, t) not in pairs or nd in named[(L, t)]

    reach, todo, through = set(), list(g.out), set()
    while True:
        while todo:
            r = todo.pop()
            if r in reach:
                continue
            reach.add(r)
            todo += [w for w in g.parents.get(r, []) if w not in reach]
        heads = {}  # layer -> the positions of its output nodes with a path to the prediction
        for nd in reach:
            if kind[nd][1] == "o_proj":
                heads.setdefault(kind[nd][0], set()).add(nd[1])
        qk = {nd for nd, (L, k) in kind.items() if (k == "q_proj" and nd[1] in heads.get(L, ()) and acts(nd, L, nd[1]))
              or (k == "k_proj" and any(t >= nd[1] and acts(nd, L, t) for t in heads.get(L, ())))} - through
        if not qk:
            break
        through |= qk
        todo = [w for nd in qk for w in g.parents.get(nd, []) if w not in reach]
    keep = reach | through
    gone = [nd for nd in sorted(g.node_set()) if nd not in keep]
    if not gone:
        g.stranded, g.stranded_bits = [], 0.0
        return g
    drop = set(gone)
    kept = native.Graph([nd for nd in g.nodes if nd not in drop],
                        {r: [w for w in ws if w not in drop] for r, ws in g.parents.items() if r not in drop}, list(g.out))
    kept.tok_edges = [(p, r) for p, r in getattr(g, "tok_edges", []) if r not in drop]
    kept.tok_positions = list(getattr(g, "tok_positions", []))
    if hasattr(g, "tok_out"):
        kept.tok_out = list(g.tok_out)
    if hasattr(g, "variables"):
        kept.variables = g.variables
    if pairs:  # a pair naming a dropped node goes with it
        kept.pairs = {read: [pr for pr in ps if not set(pr) & drop] for read, ps in pairs.items()}
    P, S = max(targets) + 1, sum(C.values())
    g.stranded_bits = 0.0
    kept.stranded, kept.stranded_bits = gone, 0.0
    kept.stranded_bits = size_bits(g, P, S, targets, _contents=False) - size_bits(kept, P, S, targets, _contents=False)
    # (a stranded block's contents: size_bits, from kept.stranded, with the text's active parts)
    return kept


def _pair_entry(e: list, nodes: list, targets: list[int]):
    """((layer, position), (query or None, key or None)) of a JSON pair edge ["pair", query, key, layer, position]
    (circuit's), None when it is not one the model has."""
    try:
        _, qi, ki, layer, t = e
        layer, t = int(layer), int(t)
        q = None if qi == "rest" else nodes[qi]
        k = None if ki == "rest" else nodes[ki]
    except (IndexError, TypeError, ValueError):
        return None
    if not 0 <= t <= max(targets) or (qi != "rest" and q is None) or (ki != "rest" and k is None):
        return None
    if q is not None and (q[0] == "tok" or native._name_layer_kind(q[0]) != (layer, "q_proj") or q[1] != t):
        return None
    if k is not None and (k[0] == "tok" or native._name_layer_kind(k[0]) != (layer, "k_proj") or k[1] > t):
        return None
    return (layer, t), (q, k)


def expand_blocks(g: native.Graph, C: dict) -> native.Graph:
    """A graph with its block nodes (BLOCK) replaced by every subcomponent of the site at the position, each in the
    block's connections and pairs: C's semantics already run the remainders live, so this is the block's claim there
    (the families on the engine take subcomponents). A graph without blocks is returned as it is."""
    if g.masks is not None or not any(nd[2] == BLOCK for nd in g.node_set()):
        return g

    def ex(nd):
        return [(nd[0], nd[1], c) for c in range(C[nd[0]])] if nd is not None and nd[2] == BLOCK else [nd]

    if g.complete:
        out = native.Graph(list(dict.fromkeys(x for nd in g.nodes for x in ex(nd))), complete=True)
        for name in ("stranded", "stranded_bits"):
            if hasattr(g, name):
                setattr(out, name, getattr(g, name))
        return out

    nodes = list(dict.fromkeys(x for nd in g.nodes for x in ex(nd)))
    parents = {}
    for r, ws in g.parents.items():
        for rr in ex(r):
            parents.setdefault(rr, [])
            parents[rr] += [x for w in ws for x in ex(w) if x not in parents[rr]]
    out = native.Graph(nodes, parents, list(dict.fromkeys(x for w in g.out for x in ex(w))))
    out.tok_edges = [(p, x) for p, r in getattr(g, "tok_edges", []) for x in ex(r)]
    out.tok_positions = list(getattr(g, "tok_positions", []))
    for name in ("tok_out", "variables", "stranded", "stranded_bits"):
        if hasattr(g, name):
            setattr(out, name, getattr(g, name))
    if getattr(g, "pairs", None):
        out.pairs = {read: list(dict.fromkeys((a, b) for pa, pb in ps for a in ex(pa) for b in ex(pb))) for read, ps in g.pairs.items()}
    return out


def _c_graph(g: native.Graph, targets: list[int], C: dict) -> native.Graph:
    """A graph as C's families run it: stranded nodes dropped (strand), block nodes as their subcomponents
    (expand_blocks)."""
    return expand_blocks(strand(g, targets, C), C)


def to_json(g: native.Graph, explanation: str = "") -> dict:
    """A Graph as an answer in JSON (circuit's format; a complete graph as its node set, "complete": true)."""
    if g.complete and g.masks is None:
        return {"nodes": [["block", n, t] if c == BLOCK else [n, c, t] for n, t, c in sorted(g.nodes, key=lambda nd: (nd[1], nd[0], nd[2]))],
                "edges": [], "complete": True, "explanation": explanation}
    tok = getattr(g, "tok_edges", [])
    nodes = sorted(g.nodes, key=lambda nd: (nd[1], nd[0], nd[2])) + [("tok", p, 0) for p in sorted(set(getattr(g, "tok_positions", [])) | {p for p, _ in tok} | set(getattr(g, "tok_out", [])))]
    at = {nd: i for i, nd in enumerate(nodes)}
    edges = ([[at[w], at[r]] for r, ws in g.parents.items() for w in ws] + [[at[w], "out"] for w in g.out]
             + [[at[("tok", p, 0)], at[r]] for p, r in tok])
    for (layer, t), ps in sorted((getattr(g, "pairs", None) or {}).items()):
        edges += [["pair", "rest" if a is None else at[a], "rest" if b is None else at[b], layer, t] for a, b in ps] or [["pair", "rest", "rest", layer, t]]
    edges += [[at[("tok", p, 0)], "out"] for p in getattr(g, "tok_out", [])]
    return {"nodes": [["block", n, t] if c == BLOCK else [n, c, t] for n, t, c in nodes], "edges": edges, "explanation": explanation}


def _pairs(g: native.Graph) -> list[tuple]:
    """A graph's stated edges, (writer, reader), reader None for the prediction, in a fixed order (a graph's nodes are
    a set, whose order changes from process to process)."""
    return sorted([(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out],
                  key=lambda e: (e[0], e[1] or ("", -1, -1)))


def size_term(bits: float, scale: float) -> float:
    """The size term: an answer's description bits in orders of magnitude of the scale (size_scale, the bits of naming
    VPD's whole answer), log2(1 + bits) / log2(1 + scale), at most 1 (the lead's decision 10-12; components.size_term)."""
    return min(1.0, math.log2(1.0 + bits) / math.log2(1.0 + scale)) if scale and scale > 0 else 0.0


def size_bits(g: native.Graph, positions: int, subcomponents: int, targets: list[int], active: dict | None = None,
              _contents: bool = True) -> float:
    """An answer's description bits, raw (the lead's 10-12: no scale): native.Graph.bits with the answer's embedding
    nodes (each a choice among the positions, every edge's ends indexed among all the listed nodes, an embedding edge
    priced like any other), the stated query-key pairs (_pair_bits), and each block node (BLOCK) at what naming its
    contents would cost: its name among sites and positions plus, per subcomponent of its site VPD's answer has on at
    its position (active: {site: bool [T, C]}, the text's), a subcomponent's bits, so a block is never cheaper than
    listing its active parts (the lead's 10-12) and is worth it only where faithfulness needs it. A graph with blocks
    needs active."""
    extra = getattr(g, "stranded_bits", 0.0) + _pair_bits(g)  # stranded nodes (strand): dropped, their bits still charged
    blocks = [nd for nd in (g.node_set() if not g.complete else ()) if nd[2] == BLOCK]
    extra += len(blocks) * (math.log2(positions * VPD4L_SITES) - math.log2(positions * subcomponents))  # named among sites
    if _contents:
        held = blocks + [nd for nd in getattr(g, "stranded", []) if nd[2] == BLOCK]
        if held and active is None:
            raise ValueError("a graph with block nodes is priced at its blocks' active contents: pass active (the text's VPD answer)")
        extra += sum(int(active[n][t].sum()) for n, t, _ in held) * math.log2(positions * subcomponents)
    if getattr(g, "tok_out", None):  # an embedding into the prediction priced like an edge into it: its writer named
        extra += len(g.tok_out) * math.log2(g.count() + len(set(getattr(g, "tok_positions", [])) | set(g.tok_out)))
    pos = sorted(set(getattr(g, "tok_positions", [])) | {p for p, _ in getattr(g, "tok_edges", [])})
    if not pos:
        return g.bits(positions, subcomponents, targets) + extra
    n, k = g.count(), len(pos)
    inner, out = native.implied_edges(g.node_set(), targets) if g.complete else (sum(len(w) for w in g.parents.values()), len(g.out))
    return n * math.log2(positions * subcomponents) + k * math.log2(positions) + math.log2(n + k) * (2 * (inner + len(g.tok_edges)) + out) + extra


def _pair_bits(g: native.Graph) -> float:
    """The stated query-key pairs' bits (the lead, 10-12: anything stated costs its description): each pair of a read
    (layer, t), and a read stated with the background alone, names its query among the circuit's queries at (layer, t)
    and "rest", and its key among the circuit's keys of the layer up to t and "rest"."""
    bits = 0.0
    nodes = g.node_set() if getattr(g, "pairs", None) else ()
    for (layer, t), ps in (getattr(g, "pairs", None) or {}).items():
        nq = sum(1 for nd in nodes if native._name_layer_kind(nd[0]) == (layer, "q_proj") and nd[1] == t)
        nk = sum(1 for nd in nodes if native._name_layer_kind(nd[0]) == (layer, "k_proj") and nd[1] <= t)
        bits += max(1, len(ps)) * (math.log2(nq + 1) + math.log2(nk + 1))
    return bits


def _tok_pairs(g: native.Graph) -> tuple[list, list]:
    """(the answer's stated embedding edges, the unstated ones: a listed embedding node into a residual reader among
    the nodes at its position), as (("tok", position, 0), reader) pairs."""
    stated = [(("tok", p, 0), r) for p, r in getattr(g, "tok_edges", [])]
    have = set(stated)
    unstated = [(("tok", p, 0), r) for p in getattr(g, "tok_positions", []) for r in sorted(g.nodes)
                if r[1] == p and native._name_layer_kind(r[0])[1] in mech.RESID_READERS and (("tok", p, 0), r) not in have]
    return stated, unstated


def _transplant(base: torch.Tensor, moved: torch.Tensor, still: torch.Tensor) -> torch.Tensor:
    """log_softmax(base + moved - still): a run's change from still to moved (log q on x' and on x) applied to base
    (the model's log p on x), the target positions' distributions [B, targets, V]."""
    return torch.log_softmax(base + moved - still, -1)


def _fid_raw(e0: float, kc: float, kv: float) -> float | None:
    """_fid without the floor at 0 (the reward's ranking, the lead's 10-12): below 0 when the circuit is farther than
    the bound; at most 1."""
    if kv is None or kc is None or not e0 > 0 or not kv < e0:
        return None
    return min(1.0, math.log(e0 / max(kc, 1e-12)) / math.log(e0 / max(kv, 1e-12)))


def _c_rl(terms: list) -> float | None:
    """The reward's C: the harmonic mean when every term is above 0, else the lowest term (unclipped): continuous at 0,
    and a circuit worse than its bounds still ranked by how much."""
    if not terms:
        return None
    return _harmonic(terms) if min(terms) > 0 else min(terms)


PHI_FAMILIES = {  # phi's families (the lead's reward v3): score key, how the score pools its draws, the draws' key
    "point": ("s_point_c", None, None), "nodes": ("s_nodes", "mean", "draws_nodes"), "edges": ("s_edges", "mean", "draws_edges"),
    "flips": ("f1_flips_c", None, None), "track_pp": ("s_track_pp", "weighted", "draws_track_pp"),
    "interchange_pp": ("s_interchange_pp", "weighted", "draws_interchange_pp"),
    "donors_near": ("s_donor_near", "donor", "draws_donor_near"), "donors_far": ("s_donor_far", "donor", "draws_donor_far"),
    "pattern": ("s_pattern", "mean", "draws_pattern"), "N": ("n_point", None, None)}
EPS_FLOOR = 0.005  # eps's floor
EPS_BOOT = 200  # eps's bootstrap resamples


def family_score(row: dict, family: str) -> float | None:
    """A row's score for one of phi's families (PHI_FAMILIES): flips at FLIP_TERM."""
    key = PHI_FAMILIES[family][0]
    if key == "f1_flips_c":
        fc = row.get("f1_flips_c") or {}
        return fc.get(str(FLIP_TERM), fc.get(FLIP_TERM))
    return row.get(key)


def _pooled(kind: str, draws: list, e0: float | None) -> float | None:
    """A family's score from some of its draws, as the family pools them: "mean" on the mean KLs ([KL_C, KL_VPD] each,
    against KL_empty e0), "weighted" per draw against its bound W ([prompt, KL_C, KL_VPD, W]) pooled by W, "donor" on
    the mean KLs against the mean keep-nothing KL ([donor, KL_C, KL_VPD, KL_0, score])."""
    if not draws:
        return None
    if kind == "mean":
        return None if e0 is None else _fid(e0, float(np.mean([d[0] for d in draws])), float(np.mean([d[1] for d in draws])))
    if kind == "donor":
        return _fid(float(np.mean([d[3] for d in draws])), float(np.mean([d[1] for d in draws])), float(np.mean([d[2] for d in draws])))
    num = den = 0.0
    for d in draws:
        sd = _fid(d[3], d[1], d[2])
        if sd is not None:
            num, den = num + d[3] * sd, den + d[3]
    return num / den if den > 0 else None


def family_eps(row: dict) -> dict:
    """eps_<family> for phi (the lead's reward v3): twice the bootstrap standard error of each family's score over its
    draws in the row (EPS_BOOT resamples, a fixed seed: the same from the row alone), at least EPS_FLOOR; a family
    scored in one run (point, flips, N) the floor."""
    rng = np.random.default_rng(0)
    out = {}
    for fam, (_, kind, key) in PHI_FAMILIES.items():
        draws = row.get(key) if key else None
        if not kind or not draws or len(draws) < 2:
            out[fam] = EPS_FLOOR
            continue
        got = [_pooled(kind, [draws[i] for i in rng.integers(len(draws), size=len(draws))], row.get("kl_empty_live"))
               for _ in range(EPS_BOOT)]
        got = [x for x in got if x is not None]
        out[fam] = max(EPS_FLOOR, 2 * float(np.std(got, ddof=1))) if len(got) > 1 else EPS_FLOOR
    return out


def _phi(scores: list) -> float | None:
    """-log2 of the power-4 mean of (1 - s + eps) over (s, eps) pairs."""
    if not scores:
        return None
    return -math.log2(float(np.mean([(1 - s_ + e_) ** 4 for s_, e_ in scores])) ** 0.25)


def phi(row: dict, start: float | dict | None = None) -> tuple:
    """(phi, phi_tau) of a row (the lead's reward v3): phi = -log2((mean_i (1 - s_i + eps_i)^4)^(1/4)) over the
    families scored on the text (PHI_FAMILIES: point, nodes, edges, flips, track_pp, interchange_pp, donors near and
    far, pattern, N), eps_i the row's eps_<family> (family_eps when absent); phi_tau = min(phi_start, the phi with
    every family at 0.9), phi_start the start set's (start: its phi or its row; else the row's phi_start). From the
    row's fields alone."""
    eps = {f: row[f"eps_{f}"] for f in PHI_FAMILIES if row.get(f"eps_{f}") is not None}
    if len(eps) < len(PHI_FAMILIES):
        eps = {**family_eps(row), **eps}
    fams = [f for f in PHI_FAMILIES if family_score(row, f) is not None]
    got = _phi([(family_score(row, f), eps[f]) for f in fams])
    at9 = _phi([(0.9, eps[f]) for f in fams])
    ps = start if not isinstance(start, dict) else phi(start)[0]
    ps = row.get("phi_start") if ps is None else ps
    return got, (at9 if ps is None or at9 is None else min(ps, at9))


def compact(bits: float, bits_start: float) -> float | None:
    """The reward's compactness (the lead's v3): log2(bits_start / bits), raw description bits (size_bits, blocks at
    their active contents) against the start set's."""
    return None if not bits or bits <= 0 or not bits_start or bits_start <= 0 else math.log2(bits_start / bits)


def _harmonic(terms: list) -> float | None:
    """C's combination: the harmonic mean k / sum(1 / s), 0 when a term is 0 (None for no terms)."""
    if not terms:
        return None
    return 0.0 if min(terms) <= 0 else len(terms) / sum(1.0 / t for t in terms)


def _fid(e0: float, kc: float, kv: float) -> float | None:
    """C's log scale: clip(ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD), 0, 1); None when VPD's own answer is no
    nearer the model than the empty circuit under the interventions."""
    if kv is None or kc is None or not e0 > 0 or not kv < e0:  # (a bound of 0: nothing to recover)
        return None
    return _clip01(math.log(e0 / max(kc, 1e-12)) / math.log(e0 / max(kv, 1e-12)))


def _read_error(pairs: dict) -> float | None:
    """C's second observable: per circuit node, the relative RMS error of its read in the circuit's run against the
    model's under a family's interventions, sqrt(mean (r_model - r_circuit)^2) / sqrt(mean r_model^2) over the draws
    (pairs: node -> [(r_model, r_circuit)]); the mean over nodes (None without any)."""
    errs = []
    for xs in pairs.values():
        m = np.array([a for a, _ in xs], dtype=float)
        c = np.array([b for _, b in xs], dtype=float)
        if len(m) and np.mean(m ** 2) > 0:
            errs.append(float(np.sqrt(np.mean((m - c) ** 2) / np.mean(m ** 2))))
    return float(np.mean(errs)) if errs else None


def _response_error(pairs: dict, base: dict) -> float | None:
    """The reads' responses to a family's interventions: each circuit node's change from its own run without one, in
    the model (r_model - base_model) and in the circuit (r_circuit - base_circuit); the error pooled over nodes and
    draws, sqrt(sum (d_model - d_circuit)^2 / sum d_model^2) (pairs: node -> [(r_model, r_circuit)], base: node ->
    (base_model, base_circuit)); None when the family moves no read in the model."""
    num = den = 0.0
    for nd, xs in pairs.items():
        bm, bc = base[nd]
        for a, b in xs:
            num += ((a - bm) - (b - bc)) ** 2
            den += (a - bm) ** 2
    return math.sqrt(num / den) if den > 0 else None


class _PatternOr(torch.autograd.Function):
    """The OR of the patterns on, a = 1 - prod_k (1 - w_k P_k) [rows, columns] from weights w [rows, K] and patterns P
    [K, columns], with its gradient d a_c / d w_k = P_kc prod_{j != k} (1 - w_j P_jc) from exclusive cumulative
    products: torch.prod's own backward checks its input for zeros on the host (a synchronization per call, and an
    operation a CUDA graph cannot capture)."""

    @staticmethod
    def forward(ctx, w, P):
        x = 1 - w[:, :, None] * P[None]
        ctx.save_for_backward(x, P)
        return 1 - x.prod(1)

    @staticmethod
    def backward(ctx, g):
        x, P = ctx.saved_tensors
        one = torch.ones_like(x[:, :1])
        before = torch.cat([one, x[:, :-1]], 1).cumprod(1)
        after = torch.cat([x[:, 1:], one], 1).flip(1).cumprod(1).flip(1)
        return (g[:, None, :] * P[None] * before * after).sum(-1), None


def _search(dev) -> str:
    """The adversaries' search precision on a device: bf16 on CUDA (sign steps and shortlist screens: what is tried),
    float32 elsewhere; every reported number is a float32 run."""
    return "bf16" if str(dev).startswith("cuda") else "score"


def _drop_nodes(g: native.Graph, drop: set) -> native.Graph:
    """A stated graph without some nodes and every connection to or from them (and every stated pair naming one)."""
    nodes = [nd for nd in g.nodes if nd not in drop]
    par = {r: [w for w in ws if w not in drop] for r, ws in g.parents.items() if r not in drop}
    out = native.Graph(nodes, {r: ws for r, ws in par.items() if ws}, [w for w in g.out if w not in drop])
    if getattr(g, "pairs", None):  # a stated pair naming a removed node goes with it; the read keeps its entry
        out.pairs = {read: [pr for pr in ps if not set(pr) & drop] for read, ps in g.pairs.items()}
    return out


def _from(ids: list[int], p: int) -> list | None:
    """A base for Native.run(reuse=) on ids itself whose states hold up to position p: ids with a token that differs
    at p (the run recomputes from there; None from position 0)."""
    if p <= 0:
        return None
    base = list(ids)
    base[p] = -1
    return base


def _mean(xs) -> float | None:
    return float(np.mean(xs)) if xs else None


def _auc(stated: list[float], unstated: list[float]) -> float | None:
    """P(a stated edge's cut > an unstated pair's), ties counting one half."""
    if not stated or not unstated:
        return None
    a, b = np.asarray(stated)[:, None], np.asarray(unstated)[None, :]
    return float(((a > b).mean() + 0.5 * (a == b).mean()))


def recovered(empty: float, c: float, vpd: float) -> float | None:
    """(KL_empty - KL_C) / (KL_empty - KL_VPD); None when VPD's own answer recovers nothing."""
    return (empty - c) / (empty - vpd) if empty - vpd > 0 else None


def recovered_log(empty: float, c: float, vpd: float) -> float | None:
    """ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD): the recovery in orders of magnitude of KL (recovered is dominated by
    the empty circuit's ~100 bits, so circuits 0.5 and 2.5 bits from the model both recover about 0.98); KLs floored
    at 1e-9 bits."""
    e, c, v = (max(x, 1e-9) for x in (empty, c, vpd))
    return math.log(e / c) / math.log(e / v) if e > v else None


def necessity_log(empty: float, minus: float, vpd: float) -> float | None:
    """clip(ln(KL_{VPD minus C} / KL_VPD) / ln(KL_empty / KL_VPD), 0, 1); KLs floored at 1e-9 bits."""
    e, m, v = (max(x, 1e-9) for x in (empty, minus, vpd))
    return min(1.0, max(0.0, math.log(m / v) / math.log(e / v))) if e > v else None


def _clip01(x: float | None) -> float | None:
    return None if x is None else min(1.0, max(0.0, x))


def necessity(empty: float, minus: float, vpd: float) -> float | None:
    """clip((KL_{VPD minus C} - KL_VPD) / (KL_empty - KL_VPD), 0, 1)."""
    return min(1.0, max(0.0, (minus - vpd) / (empty - vpd))) if empty - vpd > 0 else None


def completeness(r: float | None) -> float | None:
    return None if r is None else min(1.0, r / GOAL)


def tail_split(logp: torch.Tensor, logq: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(P || Q) in bits per row (summed over the targets), split exactly by the chain rule into (the coarse KL over
    the nucleus tokens and one "rest" bucket, P(rest) x KL(P(.|rest) || Q(.|rest))): the nucleus is the model's
    smallest set of most likely tokens holding mass >= p at each target. logp [1 or B, targets, V], logq [B, targets, V]."""
    P = logp.exp()
    order = P.argsort(-1, descending=True)
    mass = P.gather(-1, order).cumsum(-1)
    inside = (mass - P.gather(-1, order)) < p  # a token is in the nucleus while the mass before it is below p
    nucleus = torch.zeros_like(P, dtype=torch.bool).scatter(-1, order, inside)
    term = P * (logp - logq)
    pr = (P * ~nucleus).sum(-1)
    qr = (logq.exp() * ~nucleus).sum(-1)
    rest = torch.where(pr > 0, pr * (pr.clamp_min(1e-30).log() - qr.clamp_min(1e-30).log()), torch.zeros_like(pr))
    coarse = (term * nucleus).sum(-1) + rest
    tail = (term * ~nucleus).sum(-1) - rest
    return coarse.sum(-1) / math.log(2), tail.sum(-1) / math.log(2)


# ---- one text


class Text:
    """What every answer on one text shares: the model's run and record, the remainders' values (for cuts()), VPD's
    own answer, and, when asked for, the changed prompts and F2's prompts (GoalEval.probe)."""

    def __init__(self, ge: "GoalEval", ids: list[int], targets: list[int], seed: int, task: dict | None = None):
        nat = ge.nat
        t0 = time.perf_counter()
        self.ge, self.ids, self.targets, self.T, self.seed, self.task = ge, list(ids), list(targets), len(ids), seed, task
        self.rec, self.states = nat._model_record(self.ids)
        self.ref = nat.reference([self.ids], self.targets)
        cap = carry._Remainders(nat)
        with torch.no_grad(), cap:
            nat.run(self.ids, self.targets, nat._full_plan(self.T, 1), *nat._one())
        if cap.seen != set(nat.names):
            raise RuntimeError("did not capture every remainder product")
        self.rem = cap.values
        self.ci = nat.importance(self.ids) if nat.has_importance else None
        self.vpd = ge.vpd_answer(self.ids, self.targets, self.ci)
        self.cut_cache: dict = {}  # (writer, reader) -> cuts()' result: answers on a text share edges
        self.bounds: dict = {}  # convention -> (KL_empty, KL_VPD) on the text
        self._flow = self._probe = None
        self.seconds = time.perf_counter() - t0

    def _probed(self):
        if self._probe is None:
            t0 = time.perf_counter()
            self._probe = self.ge.probe(self.ids, self.targets, self.seed)
            self.seconds += time.perf_counter() - t0
        return self._probe

    @property
    def prompts(self) -> list:
        """The text's changed prompts (Native.changes)."""
        return self._probed()[0]

    @property
    def f2(self) -> list:
        """F2's positions with their held-out prompts, [(position, prompts)]."""
        return self._probed()[1]

    @property
    def moved(self) -> np.ndarray:
        """The sensitivity probe's measured context effect: moved[p - 1], the KL (nats) at the last target when the
        token at p is replaced by a draw from the model's prediction there (Native.sensitivity)."""
        return self._probed()[2]


# ---- the scorecard


class GoalEval:
    """Scores answers (native.Graph) on texts. hooks: name -> fn(task, graph, explanation) -> float or None, for the
    English reader's bits ("E1") and question answering ("E2"); unset, their columns are None."""

    def __init__(self, nat: native.Native | None = None, *, changes: int = native.CHANGES, ranks=HELD_OUT_RANKS,
                 share_floor: float = SHARE_FLOOR, random_pairs: int = RANDOM_PAIRS, max_edges: int = MAX_EDGES, chunk: int = CHUNK,
                 cut_chunk: int = CUT_CHUNK):
        self.nat = nat or native.Native()
        self.changes, self.ranks, self.share_floor = changes, tuple(ranks), share_floor
        self.random_pairs, self.max_edges, self.chunk, self.cut_chunk = random_pairs, max_edges, chunk, cut_chunk
        self.hooks: dict = {"E1": None, "E2": None, "E4": None}  # E4: fn(tx, graph, split) -> float or None (fidelity's e4)
        self.probes: dict = {}  # (ids, targets, seed) -> the changed prompts and F2's prompts, as int32 arrays
        self.donor_pools: dict = {}  # split -> the far donors' [(name, token ids)] (far_pool; set to override)

    def text(self, ids: list[int], targets: list[int], seed: int | None = None, task: dict | None = None) -> Text:
        return Text(self, ids, targets, native.task_seed(str(ids[:8])) if seed is None else seed, task)

    def probe(self, ids: list[int], targets: list[int], seed: int) -> tuple[list, list, np.ndarray]:
        """(the text's changed prompts (Native.changes), F2's [(position, held-out prompts)], the probe's moved per
        position), kept per text: the sensitivity probe behind them is most of their cost (16 s at T 395 on the Mac's
        GPU)."""
        nat, key = self.nat, (tuple(ids), tuple(targets), seed)
        if key not in self.probes:
            probe = {}
            prompts = nat.changes(ids, targets, n=self.changes, seed=seed, probe=probe)
            f2, moved = [], np.zeros(0)
            if "moved" in probe:
                moved = probe["moved"]
                total = float(moved.sum())
                for j in (moved >= self.share_floor * total).nonzero().flatten().tolist() if total > 0 else []:
                    f2.append((j + 1, np.asarray(carry.alternatives(nat, ids, j + 1, ranks=self.ranks), dtype=np.int32)))
                moved = moved.cpu().numpy() if torch.is_tensor(moved) else np.asarray(moved)
            self.probes[key] = (np.asarray(prompts, dtype=np.int32), [getattr(x, "weight", 1.0) for x in prompts], f2, moved)
        arr, weights, f2, moved = self.probes[key]
        return [native.Changed(x.tolist(), w) for x, w in zip(arr, weights)], [(p, a.tolist()) for p, a in f2], moved

    # ---- VPD's semantics

    def vpd_answer(self, ids: list[int], targets: list[int], ci: dict | None = None) -> dict | None:
        """VPD's own answer on ids: matrix -> bool [T, C], causal importance above zero at positions that can reach
        the targets (Native.reachable); None without the causal-importance network (a pod)."""
        nat = self.nat
        if ci is None:
            if not nat.has_importance:
                return None
            ci = nat.importance(ids)
        reach = nat.reachable(len(ids), targets)
        return {n: (ci[n] > 0) & reach[n] for n in nat.names}

    def masks(self, g: native.Graph, T: int) -> dict:
        """A graph's subcomponents as matrix -> bool [T, C]."""
        if g.masks is not None:
            return g.masks
        return carry.masks(self.nat, T, g.nodes)

    @torch.no_grad()
    def runs(self, ids: list[int], targets: list[int], sets: list[dict], live: bool) -> torch.Tensor:
        """log q at the targets [len(sets), len(targets), V] of VPD's masked forward pass (vpd_model's Site masks) with
        each set's subcomponents on, every other off, remainders on (live) or off."""
        nat, tg = self.nat, self.nat.target
        T = len(ids)
        out = []
        for s0 in range(0, len(sets), self.chunk):
            part = sets[s0:s0 + self.chunk]
            B = len(part)
            for n in nat.names:
                st = tg.site(n)
                st.mask = torch.stack([m[n] for m in part]).float()
                st.delta_mask = torch.ones(B, T, device=nat.dev) if live else None
            try:
                hidden = tg.hidden(torch.tensor([ids], device=nat.dev).expand(B, -1))
            finally:
                nat.vpd.clear()
            out.append(torch.log_softmax((hidden[:, targets] @ tg.wte.T).float(), -1))
        return torch.cat(out)

    @torch.no_grad()
    def cancellation(self, tx: Text) -> dict:
        """VPD's description at the last target's final residual stream in the real model: the sums of the writes of
        the subcomponents VPD's answer has on and of those it has off (every attention and MLP output matrix's,
        rec x U), their norms beside the model's stream, and the cosine between the two sums (strongly negative: the
        actives overshoot and the inactives cancel them, which caps the real model's necessity of any part of the
        answer). The embedding, both sums and the remainders add up to the stream."""
        nat, tg, t = self.nat, self.nat.target, tx.targets[-1]
        act = torch.zeros(tg.wte.shape[1], device=nat.dev)
        ina = torch.zeros_like(act)
        for n in nat.names:
            if n.endswith(("o_proj", "down_proj")):
                a, on = tx.rec[n][0, t], tx.vpd[n][t].float()
                act += (a * on) @ tg.site(n).U
                ina += (a * (1 - on)) @ tg.site(n).U
        final = self.flow(tx)["final"][t]
        cos = torch.nn.functional.cosine_similarity
        return {"actives": float(act.norm()), "inactives": float(ina.norm()), "model": float(final.norm()),
                "cosine": float(cos(act, ina, dim=0)), "cos_actives_model": float(cos(act, final, dim=0)),
                "cos_inactives_model": float(cos(ina, final, dim=0))}

    def size_scale(self, tx: Text) -> float | None:
        """The size term's denominator: the bits of naming every subcomponent of VPD's own answer (its actives at the
        positions that can reach the targets), each a choice among those positions and every subcomponent (an answer's
        node price). An answer's size bits over it: the fraction of VPD's description the circuit needs."""
        if tx.vpd is None:
            return None
        nat = self.nat
        return sum(int(m.sum()) for m in tx.vpd.values()) * math.log2(nat.positions(tx.targets) * sum(nat.C.values()))

    def graph_ref(self, ids: list[int], targets: list[int], graph: native.Graph, u: dict | None = None, live: bool = True,
                  record: dict | None = None, patch: dict | None = None, cuts: dict | None = None,
                  reads: list | None = None, norms: dict | None = None, freeze: str = "",
                  rows: dict | None = None) -> torch.Tensor:
        """log q at the targets [B, len(targets), V] of a stated graph's run in C's semantics (the lead's decision, 10-12):
        inside the circuit, graph semantics, each circuit node reading the full residual stream less the outputs of the
        circuit nodes that are not its stated parents (the background and its stated parents; a value's share through
        the attention pattern into an attention output, an MLP input's output into its MLP's output), the prediction
        the same with its stated out-writers; outside, masked semantics, every other subcomponent at its mask u ({site:
        [B, T, C]}, None: 0) and the remainders (live) reading the full stream, the circuit's outputs included, as in
        the real model. Query and key nodes enter their layer's attention pattern. A forward pass written out, the
        reference engine.Engine's graph mode matches; differentiable in u. With every connection stated it is the
        masked run of the circuit's subcomponents and u. record: receives each read's activation {node: [B]} (reads:
        those subcomponents; by default a stated graph's nodes); patch: {node: value} fixes those subcomponents' activations (an interchange); cuts: {reader node or
        None (the prediction): [writer nodes]}, connections cut besides the unstated ones (a complete graph, VPD's
        answer or the whole model as masks, has none unstated: its nodes read each other in full), and ("pairs", layer,
        t): (split nodes, pairs) cuts those query-key pairs of the read (layer, t) (graph.pairs, _pair_row): a stated
        graph's from its stated pairs (or every pair of the split), a complete graph's from every pair of the split (the
        circuit's query and key nodes: the same intervention on the model and VPD's answer). norms: receives each
        RMSNorm's scale 1 / rms per position ({("attn" | "mlp", layer) or "final": [B, T or targets, 1]}); with
        freeze "layers" (every norm inside the layers) or "all" (the final norm too), those norms use the scales norms
        holds instead (a run recorded first: the norm diagnostic). rows: {(layer, t): [H, t + 1]} attention rows used
        in place of the run's own at those reads (pattern_family)."""
        import vpd_model

        nat, tg = self.nat, self.nat.target
        T, H, hd, dev = len(ids), tg.n_head, tg.hd, self.nat.dev
        import engine

        pcut = {r[1:]: v for r, v in (cuts or {}).items() if engine._pair_cut(r)}  # (layer, t): (split nodes, pairs removed)
        drops = {r: list(ws) for r, ws in (cuts or {}).items() if not engine._pair_cut(r)}
        if graph.complete:
            nodes = sorted({nd for r, ws in drops.items() for nd in ws + ([r] if r is not None else [])})
        else:
            nodes = sorted(graph.node_set())
            par_all, out_all = native.all_edges(nodes, targets)
            stated = {r: set(ws) for r, ws in graph.parents.items()}
            for r, ws in par_all.items():
                drops.setdefault(r, []).extend(w for w in ws if w not in stated.get(r, set()))
            drops.setdefault(None, []).extend(w for w in out_all if w not in set(graph.out))
        cut = {}
        for r, ws in drops.items():
            if r is not None and ws:
                cut.setdefault(r[0], []).append((r, sorted(set(ws))))
        cut_out = sorted(set(drops.get(None, [])))
        fixed = {}
        for (n, t, c), v in (patch or {}).items():
            fixed.setdefault(n, {})[(t, c)] = v
        B = 1 if u is None else next(iter(u.values())).shape[0]
        if graph.complete:
            circ = {n: m.float() for n, m in self.masks(graph, T).items()}
        else:
            circ = {n: torch.zeros(T, nat.C[n], device=dev) for n in nat.names}
            for n, t, c in nodes:
                circ[n][t, c] = 1.0
        on = {n: circ[n][None].expand(B, -1, -1) if u is None else u[n] * (1 - circ[n]) + circ[n] for n in nat.names}
        a = {}

        def write(w):  # a circuit writer's output as it enters the stream [B, d_out] (times its mask: off, nothing to cut)
            return (a[w[0]][:, w[1], w[2]] * on[w[0]][:, w[1], w[2]])[:, None] * tg.site(w[0]).U[w[2]]

        def site(n, inp, restricted):  # the site's output; circuit readers with unstated writers read restricted(r, ws)
            A = inp @ tg.site(n).V
            if n in cut or n in fixed:
                A = A.clone()
                for r, ws in cut.get(n, []):
                    A[:, r[1], r[2]] = restricted(r, ws) @ tg.site(n).V[:, r[2]]
                for (t, c), v in fixed.get(n, {}).items():
                    A[:, t, c] = v
            a[n] = A
            out = (A * on[n]) @ tg.site(n).U
            return out + inp @ tg.site(n).delta_T if live else out

        def norm(xx, w, key, pos=None):  # vpd_model.rms, its scale recorded into norms or taken from it (freeze)
            frozen = norms is not None and freeze and (freeze == "all" or key != "final")
            if frozen:
                sc = norms[key] if pos is None else norms[key][:, pos]
            else:
                sc = torch.rsqrt(xx.float().pow(2).mean(-1, keepdim=True) + tg.eps)
                if norms is not None and pos is None:
                    norms[key] = sc.detach()
            return w * (xx.float() * sc).to(xx.dtype)

        x = tg.wte[torch.tensor(ids, device=dev)][None].expand(B, -1, -1)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        for i in range(tg.n_layer):
            nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
            z = norm(x, tg.norms[2 * i], ("attn", i))
            resid = lambda r, ws, x=x, i=i: norm(x[:, r[1]] - sum(write(w) for w in ws), tg.norms[2 * i], ("attn", i), r[1])  # noqa: E731
            q, k, v = (site(nm[kk], z, resid).view(B, T, H, hd).transpose(1, 2) for kk in ("q_proj", "k_proj", "v_proj"))
            q, k = tg._rope(q, T), tg._rope(k, T)
            S = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
            sp = {} if graph.complete else (getattr(graph, "pairs", None) or {})
            for layer, t in sorted(set(sp) | set(pcut)):  # a read's row from its stated pairs, less any it has cut
                if layer != i:
                    continue
                split = nodes if (layer, t) in sp else sorted(pcut[(layer, t)][0])
                keep = list(sp[(layer, t)]) if (layer, t) in sp else engine.read_pairs(split, layer, t)
                if (layer, t) in pcut:
                    gone = set(pcut[(layer, t)][1])
                    keep = [x for x in keep if x not in gone]
                S = S.clone()
                S[:, :, t, :t + 1] = self._pair_row(i, t, keep, split, q, k, a, on)
            P = S.masked_fill(~causal, float("-inf")).softmax(-1)
            if rows:  # reads given an attention row from outside (pattern_family: the real model's on the text)
                P = P.clone()
                for (layer, t), row in rows.items():
                    if layer == i:
                        P[:, :, t, :t + 1] = row
            att = (P @ v).transpose(1, 2).reshape(B, T, -1)
            share = lambda r, ws, att=att, P=P: att[:, r[1]] - sum(  # noqa: E731
                P[:, :, r[1], w[1]].repeat_interleave(hd, dim=1) * write(w) for w in ws)
            x = x + site(nm["o_proj"], att, share)
            z2 = norm(x, tg.norms[2 * i + 1], ("mlp", i))
            resid2 = lambda r, ws, x=x, i=i: norm(x[:, r[1]] - sum(write(w) for w in ws), tg.norms[2 * i + 1], ("mlp", i), r[1])  # noqa: E731
            pre = site(nm["c_fc"], z2, resid2)
            x = x + site(nm["down_proj"], vpd_model.gelu_tanh(pre),
                         lambda r, ws, pre=pre: vpd_model.gelu_tanh(pre[:, r[1]] - sum(write(w) for w in ws)))
        xt = x[:, targets]
        if cut_out:
            xt = xt.clone()
            for j, t in enumerate(targets):
                ws = [w for w in cut_out if w[1] == t]
                if ws:
                    xt[:, j] = xt[:, j] - sum(write(w) for w in ws)
        if record is not None:
            for nd in (reads if reads is not None else (sorted(graph.node_set()) if not graph.complete else [])):
                record[nd] = a[nd[0]][:, nd[1], nd[2]].detach()
        return torch.log_softmax((norm(xt, tg.ln_f, "final") @ tg.wte.T).float(), -1)


    @torch.no_grad()
    def line_effects(self, tx: Text, graph: native.Graph, mode: str = "pp", split: str = "search", draws: int = 8,
                     chunk: int = 16) -> dict:
        """Each line of an answer's own effect, for RL's per-line rewards (the lead's 10-12): raw bits, unclipped.
        nodes: per node, the KL at the targets with that node alone removed from the answer's graph less the KL of the
        graph (above 0: the node helps), and vpd_active (VPD's answer has it on at that position; a block, any of its
        site's parts there); edges: per stated connection (and, in "pp", stated token edge) the same with it alone cut;
        positions: per influential position (F2's, the probe's moved) whether the answer has a node there and a token
        edge from it. mode "c": C's semantics on the text, every removal one row of batched engine plans (chunk rows
        each; block nodes run as their subcomponents); "pp": the path-patched run (pp_run) on split's changed texts
        (draws of changed_inputs' prompts), each removal's KL change weighted by how far the change moves the model.
        A complete graph (node set) has its nodes and no stated lines. Nothing else is re-stranded: a removal's
        dependants keep their places, reading what remains."""
        nat, T = self.nat, tx.T
        g = strand(graph, tx.targets, nat.C)
        nodes = sorted(g.node_set())
        edges = [] if g.complete else _pairs(g)
        toks = [] if g.complete else sorted(getattr(g, "tok_edges", [])) + [(t, None) for t in getattr(g, "tok_out", [])]

        def keep(h):  # a removal's graph as stated (no further stranding)
            h.stranded, h.stranded_bits = [], 0.0
            for name in ("tok_edges", "tok_positions", "tok_out", "pairs"):
                if hasattr(g, name) and not hasattr(h, name):
                    setattr(h, name, getattr(g, name))
            return h

        def without_node(nd):
            if g.complete:
                return keep(native.Graph([x for x in nodes if x != nd], complete=True))
            h = keep(_drop_nodes(g, {nd}))
            h.tok_edges = [(t, r) for t, r in getattr(g, "tok_edges", []) if r != nd]
            return h

        def without_edge(e):
            w, r = e
            par = {rr: [x for x in ws if not (rr == r and x == w)] for rr, ws in g.parents.items()}
            h = native.Graph(list(g.nodes), {rr: ws for rr, ws in par.items() if ws}, [x for x in g.out if not (r is None and x == w)])
            return keep(h)

        def without_tok(e):
            h = keep(native.Graph(list(g.nodes), dict(g.parents), list(g.out)))
            for name in ("tok_positions", "pairs"):
                if hasattr(g, name):
                    setattr(h, name, getattr(g, name))
            h.tok_edges = [x for x in getattr(g, "tok_edges", []) if x != e]
            h.tok_out = [t for t in getattr(g, "tok_out", []) if e != (t, None)]
            return h

        variants = [g] + [without_node(nd) for nd in nodes] + [without_edge(e) for e in edges]
        if mode == "pp":
            variants += [without_tok(e) for e in toks]
        if mode == "c":
            kl = []
            for s0 in range(0, len(variants), chunk):
                part = [expand_blocks(h, nat.C) for h in variants[s0:s0 + chunk]]
                whole = [i for i, h in enumerate(part) if h.complete]
                stated = [i for i, h in enumerate(part) if not h.complete]
                got = [None] * len(part)
                if whole:
                    for i, k in zip(whole, carry.kl(tx.ref, self.runs(tx.ids, tx.targets, [self.masks(part[i], T) for i in whole], True)).tolist()):
                        got[i] = k
                if stated:
                    for i, k in zip(stated, carry.kl(tx.ref, self._graph_logq(tx.ids, tx.targets, [part[i] for i in stated])).tolist()):
                        got[i] = k
                kl += got
                native.free(nat.dev)
        elif mode == "pp":
            moved = np.asarray(tx.moved, dtype=float)
            positions = [q for q, _ in tx.f2]
            got = self._interchange_draws(tx, positions, split, draws, first={q: -moved[q - 1] for q in positions}) if positions else []
            acc, wsum = np.zeros(len(variants)), 0.0
            for q, z in got:
                x2 = list(tx.ids)
                x2[q] = z
                ref = nat.reference([x2], tx.targets)
                w = float(carry.kl(ref, tx.ref)[0])
                acc += w * np.array([float(carry.kl(ref, self.pp_run(tx, x2, h))[0]) for h in variants])
                wsum += w
                native.free(nat.dev)
            kl = list(acc / wsum) if wsum > 0 else [None] * len(variants)
        else:
            raise ValueError("mode is pp or c")
        base = kl[0]
        d = lambda k: None if k is None or base is None else k - base  # noqa: E731
        rows_n = kl[1:1 + len(nodes)]
        rows_e = kl[1 + len(nodes):1 + len(nodes) + len(edges)]
        rows_t = kl[1 + len(nodes) + len(edges):]
        tok_at = {t for t, _ in toks}
        f2 = sorted({q for q, _ in tx.f2}) if tx.f2 is not None else []
        moved = np.asarray(tx.moved, dtype=float) if f2 else None
        return {"mode": mode, "base_kl": base,
                "nodes": [{"node": list(nd), "delta_kl": d(k),
                           "vpd_active": bool(tx.vpd[nd[0]][nd[1]].any() if nd[2] == BLOCK else tx.vpd[nd[0]][nd[1], nd[2]]) if tx.vpd is not None else None}
                          for nd, k in zip(nodes, rows_n)],
                "edges": [{"edge": [list(w), "out" if r is None else list(r)], "delta_kl": d(k)} for (w, r), k in zip(edges, rows_e)],
                "token_edges": [{"edge": [t, "out" if r is None else list(r)], "delta_kl": d(k)} for (t, r), k in zip(toks, rows_t)] if mode == "pp" else [],
                "positions": [{"position": q, "moved": float(moved[q - 1]), "has_node": any(nd[1] == q for nd in nodes),
                               "has_token_edge": q in tok_at or g.complete} for q in f2]}

    @torch.no_grad()
    def pattern_family(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """s_pattern (the lead's 10-12, rustmove's 35f6ca0: C's semantics flatten attention, the q/k remainders
        about 1% of the scores, so circuits fit near-uniform rows): each circuit's run in C's semantics (graph_ref)
        with the reads it uses (a layer and position holding a circuit attention output) given the real model's
        attention row on the text, all of them at once and each alone; VPD's answer the same; the mean KLs from the
        model on C's log scale against KL_empty (s_pattern, _raw without the floor at 0). A circuit whose output
        survives the model's true pattern uses attention as the model does. None for a circuit without an attention
        output."""
        b = self._pp_base(tx)
        T = tx.T
        causal = torch.ones(T, T, dtype=torch.bool, device=self.nat.dev).tril()
        e0, v0 = self._bounds(tx, "live")
        vpd = native.Graph(masks=tx.vpd) if tx.vpd is not None else None
        out = []
        for g in graphs:
            reads = sorted({(native._name_layer_kind(n)[0], t) for n, t, _ in g.node_set() if native._name_layer_kind(n)[1] == "o_proj"})
            if not reads or vpd is None:
                out.append({"s_pattern": None, "s_pattern_raw": None, "kl_pattern_c": None, "kl_pattern_vpd": None, "reads_pattern": len(reads)})
                continue
            real = {(i, t): b["S"][i][:, t].masked_fill(~causal[t], float("-inf")).softmax(-1)[:, :t + 1] for i, t in reads}
            draws = [real] + ([{r: real[r]} for r in reads] if len(reads) > 1 else [])
            kc = [float(carry.kl(tx.ref, self.graph_ref(tx.ids, tx.targets, g, rows=d))[0]) for d in draws]
            kv = [float(carry.kl(tx.ref, self.graph_ref(tx.ids, tx.targets, vpd, rows=d))[0]) for d in draws]
            native.free(self.nat.dev)
            c_, v_ = float(np.mean(kc)), float(np.mean(kv))
            out.append({"s_pattern": _fid(e0, c_, v_), "s_pattern_raw": _fid_raw(e0, c_, v_), "kl_pattern_c": c_,
                        "kl_pattern_vpd": v_, "reads_pattern": len(reads), "draws_pattern": [list(x) for x in zip(kc, kv)]})
        return out

    @torch.no_grad()
    def _pp_base(self, tx: Text) -> dict:
        """The real model's run on the text, the path-patched runs' x side (kept on tx): every site's activations
        [T, C] and remainder outputs, each layer's streams before its attention and its MLP, rotated queries and keys
        [H, T, hd], values [T, H hd], scores [H, T, T], MLP inputs, the final stream."""
        if getattr(tx, "_pp", None) is None:
            tx._pp = self._pp_record(tx.ids)
        return tx._pp

    @torch.no_grad()
    def _pp_record(self, ids: list[int]) -> dict:
        """_pp_base's record of the real model's run on any text (an interchange's changed text)."""
        import vpd_model

        nat, tg, T = self.nat, self.nat.target, len(ids)
        H, hd = tg.n_head, tg.hd
        b = {"a": {}, "rem": {}, "Xa": [], "Xm": [], "q": [], "k": [], "v": [], "S": [], "pre": []}

        def site(n, inp):
            A, rem = inp @ tg.site(n).V, inp @ tg.site(n).delta_T
            b["a"][n], b["rem"][n] = A, rem
            return A @ tg.site(n).U + rem

        x = tg.wte[torch.tensor(ids, device=nat.dev)].float()
        causal = torch.ones(T, T, dtype=torch.bool, device=nat.dev).tril()
        for i in range(tg.n_layer):
            nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
            b["Xa"].append(x)
            z = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
            q, k, v = (site(nm[kk], z) for kk in ("q_proj", "k_proj", "v_proj"))
            qr, kr = (tg._rope(m.view(1, T, H, hd).transpose(1, 2), T)[0] for m in (q, k))
            S = (qr @ kr.transpose(-1, -2)) / math.sqrt(hd)
            b["q"].append(qr)
            b["k"].append(kr)
            b["v"].append(v)
            b["S"].append(S)
            P = S.masked_fill(~causal, float("-inf")).softmax(-1)
            att = (P @ v.view(T, H, hd).transpose(0, 1)).transpose(0, 1).reshape(T, -1)
            x = x + site(nm["o_proj"], att)
            b["Xm"].append(x)
            pre = site(nm["c_fc"], vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps))
            b["pre"].append(pre)
            x = x + site(nm["down_proj"], vpd_model.gelu_tanh(pre))
        b["Xf"] = x
        return b

    @torch.no_grad()
    def pp_run(self, tx: Text, x2: list[int], graph: native.Graph, cuts: dict | None = None,
               fixed: dict | None = None) -> torch.Tensor:
        """log q at the targets [1, len(targets), V] of the path-patched run of graph on the changed text x2 (s_track_pp,
        the lead's 10-12): everything at its value in the real model on the text (_pp_base) but the circuit's nodes,
        which compute in layer order from their stated parents' path-patched values and every unstated input at its x
        value; the change enters only through stated token edges (the lead's spec: an embedding edge (p, reader), and
        ("tok", p) into the prediction, g.tok_out), so nothing stated is the model on the text (0) and everything stated
        the model on x2 (a complete graph states every token edge). A block node (BLOCK) is its site's whole output at
        its position, every subcomponent and the remainder, from its patched input. A residual reader's input: its read
        of the x stream plus each stated parent's change (out^P - out^x) plus its stated token edges' change, its
        RMSNorm recomputed. A read's pattern
        (layer, t): queries and keys split into the circuit's nodes and the rest (always at x); with no stated pairs
        the read claims every pair (circuit nodes at their patched values), with stated pairs the x scores plus each
        stated pair's change, the softmax over the whole row recomputed. An attention output: that pattern over the x
        values plus its stated value parents' changes. An MLP output: GELU of the x MLP input plus its stated input
        parents' changes. The prediction: the final x stream plus its stated writers' changes and the embedding's. On
        the text itself the run is the real model; a complete graph is the forward on x2 with every subcomponent
        outside it and every remainder at its x value (computed that way). cuts (a complete graph's): {(layer, t):
        [source positions]}, value reads its attention-output nodes at (layer, t) take at x: those sources' values in
        their pattern row at their x values (teacher_reads' cut reads, path-patched). fixed: {node: value} those nodes
        held at the value (a subcomponent's activation, a block's whole output; an interchange: interchange_pp) and
        their stated readers reading it, in a complete graph those subcomponents or site outputs set."""
        import vpd_model

        tg, T, dev = self.nat.target, tx.T, self.nat.dev
        H, hd = tg.n_head, tg.hd
        b = self._pp_base(tx)
        demb = {t: (tg.wte[x2[t]] - tg.wte[tx.ids[t]]).float() for t in range(T) if x2[t] != tx.ids[t]}
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        if graph.complete:
            if graph.masks is not None:
                M = {n: m.to(dev) for n, m in self.masks(graph, T).items()}
                Mb = {}
            else:  # subcomponent nodes live, block nodes' whole site output (with its remainder) live; kept on the graph
                kept = getattr(graph, "_pp_masks", None)  # (a graph runs once per changed text: fleet's 10-12 regression)
                if kept is not None and kept[0] == (T, str(dev)):
                    M, Mb = kept[1], kept[2]
                else:
                    M = {n: torch.zeros(T, self.nat.C[n], dtype=torch.bool, device=dev) for n in self.nat.names}
                    Mb, sub, blk = {}, {}, {}
                    for n, t, c in graph.node_set():
                        (blk if c == BLOCK else sub).setdefault(n, []).append((t, c))
                    for n, tc in sub.items():  # one write per site
                        idx = torch.tensor(tc, device=dev)
                        M[n][idx[:, 0], idx[:, 1]] = True
                    for n, tc in blk.items():
                        ts = torch.tensor([t for t, _ in tc], device=dev)
                        M[n][ts] = True
                        Mb[n] = torch.zeros(T, 1, dtype=torch.bool, device=dev)
                        Mb[n][ts] = True
                    graph._pp_masks = ((T, str(dev)), M, Mb)

            held, held_out = {}, {}  # fixed subcomponents' activations, fixed blocks' whole outputs, per site
            for (n, t, c), val in (fixed or {}).items():
                (held_out if c == BLOCK else held).setdefault(n, []).append((t, c, val))

            def site(n, inp):
                A = torch.where(M[n], inp @ tg.site(n).V, b["a"][n])
                if n in held:
                    A = A.clone()
                    for t, c, val in held[n]:
                        A[t, c] = val
                rem = torch.where(Mb[n], inp @ tg.site(n).delta_T, b["rem"][n]) if n in Mb else b["rem"][n]
                out = A @ tg.site(n).U + rem
                if n in held_out:
                    out = out.clone()
                    for t, _, val in held_out[n]:
                        out[t] = val
                return out

            x = tg.wte[torch.tensor(x2, device=dev)].float()
            for i in range(tg.n_layer):
                nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
                z = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
                q, k, v = (site(nm[kk], z) for kk in ("q_proj", "k_proj", "v_proj"))
                qr, kr = (tg._rope(m.view(1, T, H, hd).transpose(1, 2), T)[0] for m in (q, k))
                P = ((qr @ kr.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
                att = (P @ v.view(T, H, hd).transpose(0, 1)).transpose(0, 1).reshape(T, -1)
                for (layer, t), srcs in (cuts or {}).items():  # cut reads: those sources' values at x in row t
                    if layer == i and srcs:
                        s_ = torch.tensor(sorted(srcs), device=dev)
                        dv = (b["v"][i][s_] - v[s_]).view(len(s_), H, hd)  # [S, H, hd]
                        att = att.clone()
                        att[t] = att[t] + (P[:, t, s_][:, :, None] * dv.transpose(0, 1)).sum(1).reshape(-1)
                x = x + site(nm["o_proj"], att)
                x = x + site(nm["down_proj"], vpd_model.gelu_tanh(site(nm["c_fc"], vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps))))
            return torch.log_softmax(vpd_model.rms(x[tx.targets], tg.ln_f, tg.eps) @ tg.wte.T.float(), -1)[None]
        order = {k: j for j, k in enumerate(vpd_model.KINDS)}
        nodes = sorted(graph.node_set(), key=lambda nd: (native._name_layer_kind(nd[0])[0],
                                                          order[native._name_layer_kind(nd[0])[1]], nd[1], nd[2]))
        blocks = {(nd[0], nd[1]) for nd in nodes if nd[2] == BLOCK}
        nodes = [nd for nd in nodes if nd[2] == BLOCK or (nd[0], nd[1]) not in blocks]  # a block holds its site's parts there
        par = graph.parents
        tok = {}  # reader -> the stated embedding edges' change it reads (the change enters only there)
        for t, r in getattr(graph, "tok_edges", []):
            if t in demb:
                tok[r] = tok.get(r, 0) + demb[t]
        aP = {}  # each circuit node's patched value: a subcomponent's activation, a block's whole output
        held = dict(fixed or {})

        def full(n, z):  # a site's whole linear map (every subcomponent and the remainder)
            return (z @ tg.site(n).V) @ tg.site(n).U + z @ tg.site(n).delta_T

        def x_out(w):  # a circuit writer's output on the text
            n, t, c = w
            return b["a"][n][t] @ tg.site(n).U + b["rem"][n][t] if c == BLOCK else b["a"][n][t, c] * tg.site(n).U[c]

        def delta(w):  # a circuit writer's output change, out^P - out^x, in its site's output space
            n, t, c = w
            return (aP[w] if c == BLOCK else aP[w] * tg.site(n).U[c]) - x_out(w)

        def apply(r, z):  # a node's patched value from its patched input z
            return full(r[0], z) if r[2] == BLOCK else z @ tg.site(r[0]).V[:, r[2]]

        def resid(r, X, wn):  # a residual reader's patched value
            inp = X[r[1]].clone()
            for w in par.get(r, []):
                if w in aP:
                    inp = inp + delta(w)
            if r in tok:
                inp = inp + tok[r]
            return apply(r, vpd_model.rms(inp, wn, tg.eps))

        def rot(vec, p):  # a query or key vector [H hd] rotated at position p -> [H, hd]
            v = vec.view(H, hd)
            half = hd // 2
            return v * tg.cos[p] + torch.cat([-v[..., half:], v[..., :half]], -1) * tg.sin[p]

        def part(nd, patched):  # a query or key node's part [H, hd] (a block: the whole query or key), rotated
            n, p_, c = nd
            if c == BLOCK:
                return rot(aP[nd], p_) if patched else rot(x_out(nd), p_)
            act = aP[nd] if patched and nd in aP else b["a"][n][p_, c]  # (a node a block holds: as on the text, the engine's rule)
            return rot(act * tg.site(n).U[c], p_)

        pairs = getattr(graph, "pairs", None) or {}
        scale = 1 / math.sqrt(hd)
        for i in range(tg.n_layer):
            here = [nd for nd in nodes if native._name_layer_kind(nd[0])[0] == i]
            kind = {nd: native._name_layer_kind(nd[0])[1] for nd in here}
            for r in here:
                if kind[r] in ("q_proj", "k_proj", "v_proj"):
                    aP[r] = held[r] if r in held else resid(r, b["Xa"][i], tg.norms[2 * i])
            qs = [nd for nd in here if kind[nd] == "q_proj"]
            ks = [nd for nd in here if kind[nd] == "k_proj"]
            rows = {}
            for t in sorted({nd[1] for nd in here if kind[nd] == "o_proj"}):  # the pattern rows the outputs read
                qa = [nd for nd in qs if nd[1] == t]
                kb = [nd for nd in ks if nd[1] <= t]
                qx, kx = b["q"][i][:, t], b["k"][i][:, :t + 1]  # [H, hd], [H, t + 1, hd]
                q_rest = qx.clone()
                for a in qa:
                    q_rest = q_rest - part(a, False)
                k_rest = kx.clone()
                for nd in kb:
                    k_rest[:, nd[1]] -= part(nd, False)
                if (i, t) not in pairs:  # every pair claimed: the circuit's queries and keys at their patched values
                    qP = q_rest.clone()
                    for a in qa:
                        qP = qP + part(a, True)
                    kP = k_rest.clone()
                    for nd in kb:
                        kP[:, nd[1]] += part(nd, True)
                    S = (qP[:, None] * kP).sum(-1) * scale
                else:
                    S = b["S"][i][:, t, :t + 1].clone()
                    for a, kk in pairs[(i, t)]:
                        qn, qo = (q_rest, q_rest) if a is None else (part(a, True), part(a, False))
                        if kk is None:
                            S = S + ((qn - qo)[:, None] * k_rest).sum(-1) * scale
                        else:
                            S[:, kk[1]] += ((qn * part(kk, True)) - (qo * part(kk, False))).sum(-1) * scale
                rows[t] = S.softmax(-1)  # [H, t + 1]
            for r in here:
                if kind[r] == "o_proj":
                    t = r[1]
                    vmix = b["v"][i][:t + 1].clone()
                    for w in par.get(r, []):
                        if w in aP and kind.get(w) == "v_proj" and w[1] <= t:
                            vmix[w[1]] += delta(w)
                    att = (rows[t][:, :, None] * vmix.view(t + 1, H, hd).transpose(0, 1)).sum(1).reshape(-1)
                    aP[r] = held[r] if r in held else apply(r, att)
            for r in here:
                if kind[r] == "c_fc":
                    aP[r] = held[r] if r in held else resid(r, b["Xm"][i], tg.norms[2 * i + 1])
            for r in here:
                if kind[r] == "down_proj":
                    pre = b["pre"][i][r[1]].clone()
                    for w in par.get(r, []):
                        if w in aP and kind.get(w) == "c_fc":
                            pre = pre + delta(w)
                    aP[r] = held[r] if r in held else apply(r, vpd_model.gelu_tanh(pre))
        f = b["Xf"][tx.targets].clone()
        for j, t in enumerate(tx.targets):
            for w in graph.out:
                if w[1] == t and w in aP:
                    f[j] = f[j] + delta(w)
            if t in demb and t in getattr(graph, "tok_out", []):  # the token's embedding into the prediction, stated
                f[j] = f[j] + demb[t]
        return torch.log_softmax(vpd_model.rms(f, tg.ln_f, tg.eps) @ tg.wte.T.float(), -1)[None]

    @torch.no_grad()
    def track_pp(self, tx: Text, graphs: list[native.Graph], split: str = "eval", draws: int = CHANGED_DRAWS,
                 cuts: list | None = None) -> list[dict]:
        """s_track_pp (the lead's 10-12, beside s_track): on changed_inputs' changed texts x' (split), the path-patched
        run of each circuit (pp_run) against the model on x', against VPD's whole answer's path-patched run (a
        complete graph: 1 by construction), bounded by no change: per prompt clip(ln(W / KL_C) / ln(W / KL_V), 0, 1)
        with W = KL(model(x') || model(x)) (the circuit's run on x is the model's, so this is s_track's transplant),
        none where VPD's run is no nearer than W, pooled weighted by W. Per circuit s_track_pp, kl_track_pp_c,
        prompts_track_pp and draws_track_pp ([prompt, KL_C, KL_V, W] each). cuts: per graph pp_run's cuts (None: none)."""
        nat = self.nat
        graphs = [strand(g, tx.targets, nat.C) for g in graphs]
        moved = np.asarray(tx.moved, dtype=float)
        positions = [p for p, _ in tx.f2]
        got = self._interchange_draws(tx, positions, split, draws, first={p: -moved[p - 1] for p in positions}) \
            if positions and tx.vpd is not None else []
        vpd = native.Graph(masks=tx.vpd) if tx.vpd is not None else None
        acc = [[0.0, 0.0, 0.0, 0, [], 0.0] for _ in graphs]  # sum w s, sum w, sum w KL_C, prompts, draws, sum w s_raw
        for p, z in got:
            x2 = list(tx.ids)
            x2[p] = z
            ref = nat.reference([x2], tx.targets)
            w = float(carry.kl(ref, tx.ref)[0])
            kv = float(carry.kl(ref, self.pp_run(tx, x2, vpd))[0])
            for b_, g in enumerate(graphs):
                kc = float(carry.kl(ref, self.pp_run(tx, x2, g, None if cuts is None else cuts[b_]))[0])
                sb = _fid(w, kc, kv)
                a = acc[b_]
                a[4].append([[p, z], kc, kv, w])
                if sb is not None:
                    a[0], a[1], a[2], a[3], a[5] = a[0] + w * sb, a[1] + w, a[2] + w * kc, a[3] + 1, a[5] + w * _fid_raw(w, kc, kv)
            native.free(nat.dev)
        return [{"s_track_pp": a[0] / a[1] if a[1] > 0 else None, "kl_track_pp_c": a[2] / a[1] if a[1] > 0 else None,
                 "prompts_track_pp": a[3], "draws_track_pp": a[4], "s_track_pp_raw": a[5] / a[1] if a[1] > 0 else None}
                for a in acc]

    @torch.no_grad()
    def interchange_pp(self, tx: Text, graphs: list[native.Graph], split: str = "eval",
                       draws: int = INTERCHANGE_DRAWS) -> list[dict]:
        """The interchange family in path-patched semantics (the lead's 10-12, C's interchange term): on the
        interchange draws of a circuit's positions (_interchange_draws), the circuit's nodes at p held at their values
        in the real model on the changed text x' (a block: the site's whole output there), their stated paths carrying
        them in the path-patched run on the text (pp_run, fixed), everything else at x; against the real model with the
        same nodes held (every site a block, complete: the model's own response) and VPD's whole answer's run with them
        held, bounded by no change: per draw clip(ln(W / KL_C) / ln(W / KL_V), 0, 1), W = KL(model held || model), none
        where VPD's run is no nearer than W, pooled weighted by W. Per circuit s_interchange_pp (and _raw, the pool
        without the floor at 0), prompts_interchange_pp, draws_interchange_pp ([[p, z], KL_C, KL_V, W] each)."""
        nat, T = self.nat, tx.T
        graphs = [strand(g, tx.targets, nat.C) for g in graphs]
        vpd = native.Graph(masks=tx.vpd) if tx.vpd is not None else None
        model = native.Graph([(n, t, BLOCK) for n in nat.names for t in range(T)], complete=True)
        recs = {}
        out = []
        for g in graphs:
            nodes = sorted(g.node_set())
            positions = sorted({t for _, t, _ in nodes if t > 0})
            a = [0.0, 0.0, 0, [], 0.0]
            for p, z in (self._interchange_draws(tx, positions, split, draws) if positions and vpd is not None else []):
                if (p, z) not in recs:
                    x2 = list(tx.ids)
                    x2[p] = z
                    recs[(p, z)] = self._pp_record(x2)
                r2 = recs[(p, z)]
                held = {}
                for n, t, c in nodes:
                    if t == p:
                        site_ = nat.target.site(n)
                        held[(n, t, c)] = r2["a"][n][t] @ site_.U + r2["rem"][n][t] if c == BLOCK else r2["a"][n][t, c]
                m = self.pp_run(tx, tx.ids, model, fixed=held)
                w = float(carry.kl(m, tx.ref)[0])
                kc = float(carry.kl(m, self.pp_run(tx, tx.ids, g, fixed=held))[0])
                kv = float(carry.kl(m, self.pp_run(tx, tx.ids, vpd, fixed=held))[0])
                a[3].append([[p, z], kc, kv, w])
                sb = _fid(w, kc, kv)
                if sb is not None:
                    a[0], a[1], a[2], a[4] = a[0] + w * sb, a[1] + w, a[2] + 1, a[4] + w * _fid_raw(w, kc, kv)
            native.free(nat.dev)
            out.append({"s_interchange_pp": a[0] / a[1] if a[1] > 0 else None, "prompts_interchange_pp": a[2],
                        "draws_interchange_pp": a[3], "s_interchange_pp_raw": a[4] / a[1] if a[1] > 0 else None})
        return out

    def _pair_row(self, layer: int, t: int, stated, nodes: list, q: torch.Tensor, k: torch.Tensor, a: dict, on: dict) -> torch.Tensor:
        """A read's attention scores [B, H, t + 1] (layer, query position t) from its stated query-key pairs: with q_t =
        sum_a q_a + q_rest (the circuit's query nodes at t, each its rank-one part a_a(t) u_a, and the rest: the
        remainder W - V U and the subcomponents outside the circuit at their masks) and k_p = sum_b k_b + k_rest the same
        at p, the score is the rest-rest term (the background) and each stated pair's: (a, b) q_a . k_b at b's
        position, (a, None) q_a . k_rest, (None, b) q_rest . k_b, rotary at t and p, over sqrt(hd). Every pair stated
        is q_t . k_p, the row unchanged. q, k [B, H, T, hd] the layer's rotated queries and keys; a and on graph_ref's
        activations and masks."""
        tg = self.nat.target
        H, hd = tg.n_head, tg.hd
        nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
        B = q.shape[0]

        def part(nd):  # a query or key node's rotated rank-one part [B, H, hd] at its position
            n, p, c = nd
            vec = ((a[n][:, p, c] * on[n][:, p, c])[:, None] * tg.site(n).U[c]).view(B, H, hd)
            half = hd // 2
            return vec * tg.cos[p] + torch.cat([-vec[..., half:], vec[..., :half]], -1) * tg.sin[p]
        qs = [nd for nd in nodes if nd[0] == nq and nd[1] == t]
        ks = [nd for nd in nodes if nd[0] == nk and nd[1] <= t]
        qpart = {nd: part(nd) for nd in qs}
        kpart = {nd: part(nd) for nd in ks}
        q_rest = q[:, :, t] - sum(qpart.values()) if qs else q[:, :, t]  # [B, H, hd]
        k_rest = k[:, :, :t + 1].clone()  # [B, H, t + 1, hd]
        for nd, kb in kpart.items():
            k_rest[:, :, nd[1]] -= kb
        scale = 1 / math.sqrt(hd)
        row = (q_rest[:, :, None] * k_rest).sum(-1) * scale  # the background
        for qa, kb in stated:
            qv = q_rest if qa is None else qpart[qa]
            if kb is None:
                row = row + (qv[:, :, None] * k_rest).sum(-1) * scale
            else:
                row = row.clone()
                row[:, :, kb[1]] = row[:, :, kb[1]] + (qv * kpart[kb]).sum(-1) * scale
        return row

    def graph_kl(self, tx: Text, graphs: list[native.Graph], live: bool) -> list[float]:
        """KL in bits at the targets of each stated graph's run on the text (_graph_kl)."""
        return self._graph_kl(tx.ids, tx.targets, tx.ref, graphs, live, self._bounds(tx, "live" if live else "off")[0])

    @torch.no_grad()
    def _graph_kl(self, ids: list[int], targets: list[int], ref: torch.Tensor, graphs: list[native.Graph], live: bool,
                  empty_kl: float) -> list[float]:
        """KL in bits at the targets, from the model's log probabilities ref, of each stated graph's run on ids in C's
        semantics (graph_ref: graph inside the circuit, masked outside, every other subcomponent off, remainders on
        (live) or off); a graph with no node is the empty circuit (empty_kl)."""
        out = [empty_kl] * len(graphs)
        some = [i for i, g in enumerate(graphs) if g.count()]
        for s0 in range(0, len(some), self.chunk):
            idx = some[s0:s0 + self.chunk]
            E = self._circuit_engine(ids, targets, [graphs[i] for i in idx], live)
            got = carry.kl(ref, E.graph_log_probs(E.plan([graphs[i] for i in idx], semantics="c"))).tolist()
            for i, k in zip(idx, got):
                out[i] = k
        native.free(self.nat.dev)
        return out

    def _circuit_engine(self, ids: list[int], targets: list[int], graphs: list[native.Graph], live: bool = True, outside: str = "off",
                        extra: dict | None = None):
        """engine.Engine on ids over the graphs' subcomponents (and extra's {site: columns}): C's semantics run there
        (Engine.plan(semantics="c"), tested equal to graph_ref) on those columns alone."""
        import engine

        nat, dev = self.nat, self.nat.dev
        cols = {}
        for n in nat.names:
            c = torch.tensor(sorted({c for g in graphs for m, _, c in g.node_set() if m == n}), dtype=torch.long, device=dev)
            cols[n] = c if extra is None else torch.unique(torch.cat([c, extra[n].to(dev)]))
        return engine.Engine(nat, ids, targets, live=live, columns=cols, outside=outside)

    def _bounds(self, tx: Text, conv: str) -> tuple[float, float | None]:
        """(KL_empty, KL_VPD) on the text under a convention, kept on tx."""
        if conv not in tx.bounds:
            nat = self.nat
            empty = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            sets = [empty] + ([tx.vpd] if tx.vpd is not None else [])
            kl = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, sets, conv == "live")).tolist()
            tx.bounds[conv] = (kl[0], kl[1] if tx.vpd is not None else None)
        return tx.bounds[conv]

    def sufficiency(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """Per graph, per convention: F1 (KL_C of the stated graph's run, graph_kl; a complete graph's masked run),
        recovered, completeness, and N (node removal from VPD's answer) on the text, with the bounds."""
        rows = [{} for _ in graphs]
        stated = [i for i, g in enumerate(graphs) if not g.complete]
        whole = [i for i, g in enumerate(graphs) if g.complete]
        sets = [self.masks(graphs[i], tx.T) for i in whole]
        if tx.vpd is not None:
            sets += [{n: tx.vpd[n] & ~m[n] for n in self.nat.names} for m in (self.masks(g, tx.T) for g in graphs)]
        for conv in CONVENTIONS:
            e, v = self._bounds(tx, conv)
            got = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, sets, conv == "live")).tolist() if sets else []
            kl = [0.0] * len(graphs) + got[len(whole):]
            for i, k in zip(whole, got):
                kl[i] = k
            for i, k in zip(stated, self.graph_kl(tx, [graphs[i] for i in stated], conv == "live")):
                kl[i] = k
            for b, row in enumerate(rows):
                c = kl[b]
                r = recovered(e, c, v) if v is not None else None
                rl = recovered_log(e, c, v) if v is not None else None
                row.update({f"kl_{conv}": c, f"kl_empty_{conv}": e, f"kl_vpd_{conv}": v, f"f1_{conv}": _clip01(rl),
                            f"recovered_log_{conv}": rl, f"recovered_{conv}": r, f"completeness_{conv}": completeness(r)})
                if v is not None:
                    m = kl[len(graphs) + b]
                    row.update({f"kl_vpd_minus_{conv}": m, f"n_{conv}": necessity_log(e, m, v), f"necessity_{conv}": necessity(e, m, v)})
                else:
                    row.update({f"kl_vpd_minus_{conv}": None, f"n_{conv}": None, f"necessity_{conv}": None})
        return rows

    # ---- F3

    @torch.no_grad()
    def edge_effects(self, tx: Text, g: native.Graph, pairs: list[tuple]) -> list[float]:
        """Per (writer, reader) pair among the graph's nodes (reader None: the prediction), the KL in bits of the
        graph's run (its subcomponents on, every other off, remainders off) from the same run with that one connection
        cut, the reader reading every other input as before (path patching inside the circuit's run, _patched)."""
        nat, T = self.nat, tx.T
        if not pairs:
            return []
        masks = self.masks(g, T)
        cg = native.Graph(masks=masks)
        zero = nat._zero()
        rec, states = {}, {}
        one = nat._plan([cg], T)
        ref = nat.run(tx.ids, tx.targets, one, *zero, record=rec, fill=states)

        def make(B):  # the plan of B copies (views of the one's masks) and the states its runs reuse (the one's)
            return {**one, "B": B, "G": {n: m.expand(B, -1, -1) for n, m in one["G"].items()},
                    "complete": one["complete"].expand(B, -1, -1)}, states

        ctx = {"make": make, "ablation": zero, "ref": ref, "flow": self._flow(tx.ids, rec, states, masks), "rec": rec, "ids": tx.ids}
        return [d["kl"] for d in self._patched(tx, ctx, pairs)]

    def edges(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """Per graph, F3 (see the module docstring): the cut effects of its stated edges (at most max_edges, a uniform
        draw beyond) and of random unstated connected pairs of its nodes (random_pairs), drawn by the text's seed and
        the answer's own edges. A complete graph (a baseline) states no edge: F3 None."""
        out = []
        for g in graphs:
            if g.complete:
                out.append({"f3": None, "stated": None, "unstated": None})
                continue
            tok_stated, tok_unstated = _tok_pairs(g)
            stated = _pairs(g) + tok_stated
            nodes = g.node_set()
            par, outs = native.all_edges(nodes, tx.targets)
            have = set(stated)
            unstated = [e for e in _pairs(native.Graph(nodes, par, outs)) if e not in have] + tok_unstated
            rng = np.random.default_rng([tx.seed, int(hashlib.sha256(repr(stated).encode()).hexdigest()[:15], 16)])
            if len(stated) > self.max_edges:
                stated = [stated[i] for i in sorted(rng.choice(len(stated), self.max_edges, replace=False))]
            if len(unstated) > self.random_pairs:
                unstated = [unstated[i] for i in sorted(rng.choice(len(unstated), self.random_pairs, replace=False))]
            if not stated or not unstated:
                out.append({"f3": 0.0, "stated": None, "unstated": None})
                continue
            effects = self.edge_effects(tx, g, stated + unstated)
            st, un = effects[:len(stated)], effects[len(stated):]
            out.append({"f3": _auc(st, un), "stated": st, "unstated": un})
        return out

    # ---- whole-model cuts (question answering, teachers)

    @torch.no_grad()
    def cuts(self, tx: Text, pairs: list[tuple], detail: bool = False) -> list:
        """Per (writer, reader) pair (reader None: the prediction), the KL in bits of the model on the text from the
        model with that one connection cut: the reader's new activation is its input in the model without the
        writer's output alone (every other output and every remainder as in the model), and the model runs on with
        that one activation replaced (path patching of the connection, _patched); into the prediction, the final
        residual stream without the writer's output. detail: per pair {"kl", "top": the cut run's most likely token at
        each target, "logp_model_top": its log-probability of the model's most likely token there}."""
        nat = self.nat
        todo = [e for e in dict.fromkeys(pairs) if e not in tx.cut_cache]
        if todo:
            ctx = {"make": lambda B: (nat._full_plan(tx.T, B), nat._model_states(tx.ids, B)), "ablation": nat._one(),
                   "ref": tx.ref, "flow": self.flow(tx), "rec": tx.rec, "ids": tx.ids}
            for e, d in zip(todo, self._patched(tx, ctx, todo)):
                tx.cut_cache[e] = d
        return [tx.cut_cache[e] if detail else tx.cut_cache[e]["kl"] for e in pairs]

    def _patched(self, tx: Text, ctx: dict, pairs: list[tuple]) -> list[dict]:
        """Path patching of distinct pairs in a run (ctx: "make"(B) -> its plan of B copies and their states, its
        "ablation" (u, ur), its log q at the targets "ref", its "flow" and "rec"ord): each reader's new activation from
        _cut_activations, then the run with that one activation replaced, batched by the reader's position and reusing
        the run's states before the batch's first reader (nothing earlier changes); a cut into the prediction needs no
        run (the flow's final residual stream). Per pair {"kl" of the run from the cut run, "top", "logp_model_top"}."""
        import vpd_model

        nat, T = self.nat, tx.T
        res = [None] * len(pairs)
        ref = ctx["ref"]
        top = ref[0].argmax(-1)  # [len(targets)]

        def summary(lq):
            return [{"kl": k, "top": t, "logp_model_top": p} for k, t, p in
                    zip(carry.kl(ref, lq).tolist(), lq.argmax(-1).tolist(),
                        lq.gather(-1, top[None, :, None].expand(lq.shape[0], -1, 1))[..., 0].tolist())]

        inner = sorted((j for j, (_, r) in enumerate(pairs) if r is not None), key=lambda j: (pairs[j][1][1], pairs[j][0][1]))
        final = sorted((j for j, (_, r) in enumerate(pairs) if r is None), key=lambda j: pairs[j][0][1])
        if inner:
            new = self._cut_activations(ctx, [pairs[j] for j in inner])
            at = {j: i for i, j in enumerate(inner)}
            B = min(self.cut_chunk, len(inner))  # every batch this size (the last padded with its own last cut): the
            plan, states = ctx["make"](B)  # states for reuse are computed once per run and batch size
            for s0 in range(0, len(inner), B):
                js = inner[s0:s0 + B]
                js = js + [js[-1]] * (B - len(js))
                by = {}
                for b, j in enumerate(js):
                    r = pairs[j][1]
                    by.setdefault(r[0], []).append((b, r[1], r[2], at[j]))
                patch = {}
                for n, lst in by.items():
                    bi, ti, ci, ki = nat._ints([[e[k] for e in lst] for k in range(4)])
                    m = torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev)
                    m[bi, ti, ci] = True
                    v = torch.zeros(B, T, nat.C[n], device=nat.dev)  # read only where m is set
                    v[bi, ti, ci] = new[ki]
                    patch[n] = (m, v)
                base = _from(tx.ids, min(pairs[j][1][1] for j in js))
                lq = nat.run(tx.ids, tx.targets, plan, *ctx["ablation"], patch=patch, reuse=None if base is None else (states, base))
                for b, d in enumerate(summary(lq)):
                    res[js[b]] = d
                native.free(nat.dev)
        if final:
            tg = nat.target
            t, c = nat._ints([[pairs[j][0][1] for j in final], [pairs[j][0][2] for j in final]])
            x = ctx["flow"]["final"][tx.targets].expand(len(final), -1, -1).clone()
            row = torch.tensor([tx.targets.index(p) for p in t.tolist()], device=nat.dev)
            for wn in {pairs[j][0][0] for j in final}:
                sel = torch.tensor([b for b, j in enumerate(final) if pairs[j][0][0] == wn], device=nat.dev)
                x[sel, row[sel]] -= ctx["rec"][wn][0, t[sel], c[sel]][:, None] * tg.site(wn).U[c[sel]]
            lq = torch.log_softmax((vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T).float(), -1)
            for b, d in enumerate(summary(lq)):
                res[final[b]] = d
        return res

    def flow(self, tx: Text) -> dict:
        """The model's run on the text as the cuts read it (_flow), kept on tx."""
        if tx._flow is None:
            tx._flow = self._flow(tx.ids, tx.rec, tx.states, None, tx.rem)
        return tx._flow

    def _flow(self, ids: list[int], rec: dict, states: dict, masks: dict | None = None, rem: dict | None = None) -> dict:
        """A run as the cuts read it, built from Native's own record of it (rec: every activation [1, T, C]; states:
        the queries, keys and values per layer) the way Native.run adds them up: per layer the residual stream before
        each norm ("attn", "mlp" [T, d]; "final", after the last layer), the attention pattern ("P" [H, T, T]) and
        output before its output matrix ("att" [T, d]), the MLP's pre-activation ("pre" [T, d_mlp]). masks: the
        subcomponents on (a circuit's run; all without); rem: the remainders' products (none without)."""
        import vpd_model

        nat, tg, T = self.nat, self.nat.target, len(ids)
        H, hd = tg.n_head, tg.hd
        x = tg.wte[torch.tensor(ids, device=nat.dev)][None]
        causal = torch.ones(T, T, dtype=torch.bool, device=nat.dev).tril()
        fl = {"attn": [], "mlp": [], "P": [], "att": [], "pre": []}

        def write(n):
            a = rec[n] if masks is None else rec[n] * masks[n]
            return a @ tg.site(n).U

        def plus(y, n):
            return y if rem is None else y + rem[n]

        for i in range(tg.n_layer):
            nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
            heads = lambda y: y.view(1, T, H, hd).transpose(1, 2)  # noqa: E731
            q, k = tg._rope(heads(states[("qv", i)]), T), tg._rope(heads(states[("kv", i)]), T)
            P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
            fl["attn"].append(x[0])
            fl["P"].append(P[0])
            fl["att"].append((P @ heads(states.get(("vg", i), states[("vs", i)]))).transpose(1, 2).reshape(T, -1))
            x = plus(x + write(nm["o_proj"]), nm["o_proj"])
            fl["mlp"].append(x[0])
            fl["pre"].append(plus(write(nm["c_fc"]), nm["c_fc"])[0])
            x = plus(x + write(nm["down_proj"]), nm["down_proj"])
        fl["final"] = x[0]
        return fl

    def _cut_activations(self, ctx: dict, pairs: list[tuple]) -> torch.Tensor:
        """Each pair's reader activation in the run (ctx's flow and record) with that one connection cut
        [len(pairs)]: a query, key, value or MLP input reads the residual stream at its position without the writer's
        output (then its norm); an attention output reads its layer's attention output without the writer value's
        share, the pattern's weight from the reader's position to the writer's in each head times the value's output;
        an MLP output its MLP's pre-activation without the input's output (then the nonlinearity). A writer ("tok", p,
        0) is the token's embedding at p."""
        import vpd_model

        nat, tg = self.nat, self.nat.target
        fl, rec = ctx["flow"], ctx["rec"]
        n = len(pairs)
        writes = [None] * n
        for wn in {w[0] for w, _ in pairs}:
            js = [j for j, (w, _) in enumerate(pairs) if w[0] == wn]
            t, c = nat._ints([[pairs[j][0][1] for j in js], [pairs[j][0][2] for j in js]])
            if wn == "tok":  # the token's embedding at its position
                wv = tg.wte[torch.tensor([ctx["ids"][p] for p in t.tolist()], device=nat.dev)]
            else:
                wv = rec[wn][0, t, c][:, None] * tg.site(wn).U[c]
            for k, j in enumerate(js):
                writes[j] = wv[k]
        out = torch.empty(n, device=nat.dev)
        for rn in {r[0] for _, r in pairs}:
            js = [j for j, (_, r) in enumerate(pairs) if r[0] == rn]
            layer, kind = native._name_layer_kind(rn)
            t, c, tw = nat._ints([[pairs[j][1][1] for j in js], [pairs[j][1][2] for j in js], [pairs[j][0][1] for j in js]])
            wv = torch.stack([writes[j] for j in js])
            V = tg.site(rn).V
            if kind in ("q_proj", "k_proj", "v_proj"):
                inp = vpd_model.rms(fl["attn"][layer][t] - wv, tg.norms[2 * layer], tg.eps)
            elif kind == "c_fc":
                inp = vpd_model.rms(fl["mlp"][layer][t] - wv, tg.norms[2 * layer + 1], tg.eps)
            elif kind == "o_proj":
                inp = fl["att"][layer][t] - fl["P"][layer][:, t, tw].T.repeat_interleave(tg.hd, dim=1) * wv
            else:
                inp = vpd_model.gelu_tanh(fl["pre"][layer][t] - wv)
            out[torch.tensor(js, device=nat.dev)] = (inp * V.T[c]).sum(-1)
        return out

    @torch.no_grad()
    def real_removal(self, tx: Text, graphs: list[native.Graph], remainders_at_nodes: bool = True) -> list[dict]:
        """Per graph: KL of the model on the text from the real model with the graph's subcomponents removed (mask 0
        at their positions, every other subcomponent and every remainder on: Native.without, batched), and its
        necessity on goal_eval's log scale per convention's bounds (n_real_X). remainders_at_nodes False: every
        matrix's remainder W - V U removed too at each position holding one of the graph's subcomponents. A batch
        reuses the model's states before its first removed subcomponent."""
        nat, T = self.nat, tx.T
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
        one_r = nat._one()[1]
        none = nat._const(("goal_none", T), lambda: {n: torch.zeros(1, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
        sets = [self.masks(g, T) for g in graphs]
        kls = []
        for s0 in range(0, len(sets), self.chunk):
            part = sets[s0:s0 + self.chunk]
            B = len(part)
            u = {n: 1.0 - torch.stack([m[n] for m in part]).float() for n in nat.names}
            first = min((int(m[n].any(1).nonzero().min()) for m in part for n in nat.names if m[n].any()), default=0)
            base = _from(tx.ids, min(first, min(tx.targets)))
            plan = {**nat._full_plan(T, B), "G": none, "none": True}
            ur = one_r
            if not remainders_at_nodes:
                held = torch.stack([sum(m[n].any(1).float() for n in nat.names) > 0 for m in part])  # [B, T]
                ur = {n: 1.0 - held.float() for n in nat.names}
            lq = nat.run(tx.ids, tx.targets, plan, u, ur, reuse=None if base is None else (nat._model_states(tx.ids, B), base))
            kls += carry.kl(tx.ref, lq).tolist()
            native.free(nat.dev)
        out = []
        for k in kls:
            row = {"kl_real_minus": k}
            for conv in CONVENTIONS:
                e, v = self._bounds(tx, conv)
                row[f"n_real_{conv}"] = necessity_log(e, k, v) if v is not None else None
            out.append(row)
        return out

    # ---- diagnostics

    # ---- C: interventional fidelity

    def fidelity(self, tx: Text, graphs: list[native.Graph], pool: dict | None = None, draws: int = DRAWS,
                 steps: int = ADV_STEPS, restarts: int = ADV_RESTARTS, curve=ADV_CURVE, fresh: dict | None = None,
                 interchange: int = INTERCHANGE_DRAWS, real: bool = False, reads: bool = False, mode: str = "full",
                 changed: int = CHANGED_DRAWS, split: str | None = None, near_donors: int = DONORS_NEAR,
                 far_donors: int = DONORS_FAR, baseline: bool = False, pattern: bool = True) -> list[dict]:
        """C, the circuit's agreement with the model under the training family of interventions (remainders live),
        each family scored on the log scale against VPD's own answer under the same interventions, s_F =
        clip(ln(KL_empty / KL_C,F) / ln(KL_empty / KL_VPD,F), 0, 1), KL_X,F the mean over the family's draws of KL(the
        model under the intervention || X under it): "point" (no intervention: F1), "wiring" (s_wiring: every connection
        among the circuit's nodes restored, the masked run of its node set: an unstated connection claims not to matter,
        so restoring them all should keep F1; a circuit that raises its graph F1 by leaving a harmful part's real
        connections unstated loses here: C's point member is min(graph F1, s_wiring), s_point_c, and its flips member
        the same min under the same flips, f1_flips_c; reliance_on_cuts = s_point - s_wiring), "nodes" (draw j removes each
        circuit node with probability (j + 0.5) / draws), "edges" (draw j cuts each stated connection into one reader
        drawn uniformly, the prediction among them, with that probability, at least one, in the model, VPD's answer
        and the circuit alike, graph_ref), "adversary" (the paired
        adversary over the pool, adversary(): f1_adv at the budget, beside the random members' mean) and its budget in
        parts switched on (adversary_flips, the omitted lever: f1_flips_omitted by m), "interchange" (_interchange_family:
        the circuit's nodes at one of its positions set to their values on a changed text, s_interchange) and "changed"
        (changed_inputs: the circuit run on changed texts, bounded by its own unchanged answer, s_changed, with
        context_share, s_track) and "donors" (donors: everything outside the circuit at one donor text's values, near
        donors the changed texts and far donors other texts, s_donor_near and s_donor_far); c, the harmonic mean of
        s_point_c, s_nodes, s_edges, f1_flips_c at FLIP_TERM (the reward's adversarial member), s_track, s_interchange,
        s_donor_near and s_donor_far over those defined on the text (c_terms names them; the claim is agreement under
        every family, so one failed family is not made up by the others, and near 0 c is linear in the lowest term, so
        float noise there stays noise: the lead's decisions 10-12; the step-budget f1_adv and s_changed are reported
        columns), the last four on split's draws (default: "search", the teachers' draws and training texts as far
        donors, in the reward on training texts; "eval" otherwise, and for any held-out text); e4: the E4 hook's score (its own reward term, not in c); flips_to_break, the smallest m at which f1_flips_omitted falls below
        half of s_point (None: not within FLIPS); flips_to_half, the smallest m at which it falls below 0.5, half of
        VPD's own answer under the same attack (0: below it unattacked), which a circuit weak to begin with cannot pass. reads: C's second observable, the circuit nodes' relative RMS read
        error between the model and the circuit per family (reads_*), and of their responses, each node's change from
        its own run without the intervention (dreads_*, _response_error; the static error mostly measures the missing
        background of a circuit's run); graph_ref,
        slower). interchange, changed: draws per circuit for those two families (0: left out of c). Evaluation only,
        never in the reward or a teacher's objective:
        fresh, a pool none of the searches drew from: the pattern lever's flips over it (f1_flips_patterns_fresh);
        real: N_real (real_removal,
        n_real) and its adversary (necessity_flips, n_real_flips). leak: graph F1 of the circuit's nodes stating no
        edge (live remainders carrying their outputs to the prediction). mode="reward": the training family alone, what
        the RL reward takes every step: point, nodes, edges with reads (on), the flips up to FLIP_TERM (their starts
        drawn as for every budget), interchange and changed texts (so c is the full mode's on the same split), and for
        RL's ranking rl_terms (each term without the floor at 0, _fid_raw) and c_rl (_c_rl: their harmonic mean when all
        are above 0, else the lowest); baseline: c_vpd_k and c_rl_vpd_k of VPD's first k at the answer's k
        (vpd_baseline); no
        step-budget adversary, larger budgets, fresh pool, N_real or leak, and no flips_to_break / flips_to_half."""
        if mode not in ("full", "reward"):
            raise ValueError("mode is full or reward")
        pp_graphs = [strand(g, tx.targets, self.nat.C) for g in graphs]  # the path-patched terms take blocks as blocks
        graphs = [_c_graph(g, tx.targets, self.nat.C) for g in graphs]  # stranded nodes dropped before every family
        reward = mode == "reward"
        split = split or ("search" if reward else "eval")
        reads = reads or reward
        e0 = self._bounds(tx, "live")[0]
        sides = self._sides(tx, graphs) if reads else None
        point = self.sufficiency(tx, graphs)
        whole = [native.Graph(sorted(g.node_set()), complete=True) if not g.complete else None for g in graphs]
        wired = dict(zip([i for i, w in enumerate(whole) if w is not None],
                         self.sufficiency(tx, [w for w in whole if w is not None])))
        nodes = self._node_family(tx, graphs, draws, reads)
        edges = self._edge_family(tx, graphs, draws, reads)
        none = [{} for _ in graphs]
        adv = self.adversary(tx, graphs, pool, steps=steps, restarts=restarts, draws=draws, curve=curve) if pool and not reward else none
        flips = self.adversary_flips(tx, graphs, pool, "omitted", steps=steps, restarts=restarts,
                                     upto=FLIP_TERM if reward else None) if pool else none
        fresh_rows = self.adversary_flips(tx, graphs, fresh, "patterns", steps=steps, restarts=restarts) if fresh and not reward else none
        inter = self._interchange_family(tx, graphs, interchange, split=split, reads=reads) if interchange else none
        reals = self.real_removal(tx, graphs) if real and not reward else none
        chg = self.changed_inputs(tx, graphs, split, changed) if changed else none
        tpp = self.track_pp(tx, pp_graphs, split, changed) if changed else none
        ipp = self.interchange_pp(tx, pp_graphs, split, interchange) if interchange else none
        pat = self.pattern_family(tx, graphs) if pattern else none  # s_pattern: C's true-pattern family
        dns = self.donors(tx, graphs, split, near_donors, self.far_donors(tx, self.far_pool(split), far_donors) if far_donors else None) \
            if (near_donors or far_donors) else none
        nflips = self.necessity_flips(tx, graphs, steps=steps, restarts=restarts) if real and not reward else none
        leak = none if reward else self.sufficiency(tx, [native.Graph(sorted(g.node_set()), {}, []) for g in graphs])
        out = []
        for g, pt, nd, ed, ad, fl, fr, it, rr, nf, lk in zip(graphs, point, nodes, edges, adv, flips, fresh_rows, inter, reals,
                                                             nflips, leak):
            i = len(out)
            row = {"s_point": pt["f1_live"], "kl_point": pt["kl_live"], "n_point": pt["n_live"], **nd, **ed, **ad, **fl, **it, **nf,
                   "s_wiring": wired[i]["f1_live"] if i in wired else pt["f1_live"]}
            if lk:
                row["leak"] = lk["f1_live"]
            row["s_nodes"] = _fid(e0, nd["kl_nodes_c"], nd["kl_nodes_vpd"])
            row["s_edges"] = _fid(e0, ed["kl_edges_c"], ed["kl_edges_vpd"]) if ed["kl_edges_c"] is not None else None
            # a stated graph claims its unstated connections do not matter: C takes the lower of its graph run and the
            # run with every connection among its nodes restored, unattacked and under the same flips
            row["reliance_on_cuts"] = (row["s_point"] - row["s_wiring"]) if None not in (row["s_point"], row["s_wiring"]) else None
            row["s_point_c"] = min(x for x in (row["s_point"], row["s_wiring"]) if x is not None) if row["s_point"] is not None else None
            fg = row.get("f1_flips_omitted", {})
            fw, fw_raw = self._flips_wiring(tx, g, fl, e0, both=True) if (i in wired and fg) else ({}, {})
            row["f1_flips_wiring"] = fw
            fc = {m: (min(v, fw[m]) if (m in fw and v is not None and fw[m] is not None) else v) for m, v in fg.items()}
            row["f1_flips_c"] = fc
            if it:
                row["s_interchange"] = _fid(e0, it["kl_interchange_c"], it["kl_interchange_vpd"])
            row.update(chg[i])
            row.update(tpp[i])
            row.update(ipp[i])
            row.update(pat[i])
            row.update(dns[i])
            named = [(k, row.get(k)) for k in ("s_point_c", "s_nodes", "s_edges")] + [("f1_flips_c", fc.get(FLIP_TERM))]
            named += [(k, row.get(k)) for k in ("s_track_pp", "s_interchange_pp", "s_donor_near", "s_donor_far", "s_pattern")]
            row["c_terms"] = [k for k, v in named if v is not None]  # a family with nothing scored on the text is left out
            row["c"] = _harmonic([v for _, v in named if v is not None])
            if reward:  # the terms without the floor at 0 and C_rl, for RL's ranking within a group (the lead's 10-12)
                v0 = self._bounds(tx, "live")[1]
                kfl = row.get("kl_flips_omitted", {}).get(FLIP_TERM)
                raw = {"s_point_c": min(x for x in (_fid_raw(e0, pt["kl_live"], v0),
                                                    _fid_raw(e0, wired[i]["kl_live"], v0) if i in wired else None) if x is not None)
                       if _fid_raw(e0, pt["kl_live"], v0) is not None else None,
                       "s_nodes": _fid_raw(e0, nd["kl_nodes_c"], nd["kl_nodes_vpd"]),
                       "s_edges": _fid_raw(e0, ed["kl_edges_c"], ed["kl_edges_vpd"]) if ed["kl_edges_c"] is not None else None,
                       "f1_flips_c": min(x for x in (_fid_raw(e0, *kfl) if kfl else None, fw_raw.get(FLIP_TERM)) if x is not None)
                       if kfl else None,
                       "s_track_pp": row.get("s_track_pp_raw"), "s_interchange_pp": row.get("s_interchange_pp_raw"),
                       "s_pattern": row.get("s_pattern_raw")}
                for pool_ in ("near", "far"):
                    k3 = row.get(f"kl_donor_{pool_}")
                    raw[f"s_donor_{pool_}"] = _fid_raw(k3[2], k3[0], k3[1]) if k3 else None
                row["rl_terms"] = {k: raw[k] for k in row["c_terms"]}
                row["c_rl"] = _c_rl([raw[k] for k in row["c_terms"] if raw[k] is not None])
                if baseline:
                    row.update(self.vpd_baseline(tx, g.count(), pool, split))
            row["e4"] = self.hooks["E4"](tx, g, split) if self.hooks.get("E4") else None
            row["kl_empty_live"], row["kl_vpd_live"] = e0, self._bounds(tx, "live")[1]  # (phi's eps from the row alone)
            row["draws_point"] = [[pt["kl_live"], row["kl_vpd_live"]]]
            row.update({f"eps_{f}": v for f, v in family_eps(row).items()})
            if not reward:  # (the reward's flips stop at FLIP_TERM)
                half = 0.5 * row["s_point_c"] if row["s_point_c"] is not None else None
                broke = [m for m in sorted(fc) if half is not None and fc[m] is not None and fc[m] < half]
                row["flips_to_break"] = broke[0] if broke else None  # None: not within the budgets (more than max(FLIPS))
                below = [m for m in sorted(fc) if fc[m] is not None and fc[m] < 0.5]  # an absolute bar: half of VPD's own
                row["flips_to_half"] = 0 if (row["s_point_c"] is not None and row["s_point_c"] < 0.5) else (below[0] if below else None)
            if fr:
                row["f1_flips_patterns_fresh"] = fr["f1_flips_patterns"]
            if rr:
                row["n_real"] = rr["n_real_live"]
            if reads:  # the reads' responses: each node's change from its own run without an intervention
                _, _, _, rm, rc = sides(g, cuts=[{}], reads=True)  # one row, no intervention
                base = {nd: (float(rm[nd][0]), float(rc[nd][0])) for nd in rm}
                for fam in ("nodes", "edges", "interchange"):
                    pairs = row.pop(f"_pairs_{fam}", None)
                    row[f"dreads_{fam}"] = _response_error(pairs, base) if pairs else None
            for fam in ("nodes", "edges", "interchange"):
                row.pop(f"_pairs_{fam}", None)
            out.append(row)
        return out

    @torch.no_grad()
    def _flips_wiring(self, tx: Text, g: native.Graph, fl: dict, e0: float, raw: bool = False, both: bool = False):
        """The flips' masks scored on a stated circuit's node set with every connection among its nodes restored (its
        masked run with the switched-on omissions), against VPD's paired KL: {m: f1} (raw: _fid_raw's; both: the two
        dicts, one run)."""
        nat, T = self.nat, tx.T
        masks, ms = [], []
        for m, parts in fl.get("mask_flips_omitted", {}).items():
            base = self.masks(native.Graph(sorted(g.node_set()), complete=True), T)
            on = {n: x.clone() for n, x in base.items()}
            for n, t, c in parts:
                on[n][t, c] = True
            masks.append(on)
            ms.append(m)
        if not masks:
            return ({}, {}) if both else {}
        kc = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, masks, True)).tolist()
        kv = {m: fl["kl_flips_omitted"][m][1] for m in ms}
        return {m: (_fid_raw if raw else _fid)(e0, k, kv[m]) for m, k in zip(ms, kc)} if not both else             ({m: _fid(e0, k, kv[m]) for m, k in zip(ms, kc)}, {m: _fid_raw(e0, k, kv[m]) for m, k in zip(ms, kc)})


    def necessity_flips(self, tx: Text, graphs: list[native.Graph], ms=None, steps: int = ADV_STEPS,
                        restarts: int = ADV_RESTARTS) -> list[dict]:
        """N_real's adversary, the flips' treatment of held-out necessity (evaluation only): per budget m, the m
        subcomponents VPD's answer has on outside the circuit (at positions up to the last target) whose removal from
        the real model makes the circuit's removal matter least, KL(model - F || model - F - C) minimized by
        sign-gradient steps of ADV_STEP projected onto the m largest (one start at the gradient's top m), the top m
        removed and scored in float32, beside the m best of a measured shortlist (SCREEN parts with the largest gradient
        either way, each removal measured alone, in the search precision: the screens only rank which parts are
        tried), the best of them, at most m (no worse than a smaller budget or none);
        per circuit n_real_flips {m: necessity on N_real's scale, clip(ln(KL / KL_VPD)
        / ln(KL_empty / KL_VPD), 0, 1)} and mask_n_real_flips {m: [(site, t, c)]}. A circuit needed only to cancel
        the parts F becomes unnecessary once they are gone."""
        import engine

        nat, T, dev = self.nat, tx.T, self.nat.dev
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
        ms = FLIPS["omitted"] if ms is None else ms
        e0, v0 = self._bounds(tx, "live")
        gen = torch.Generator().manual_seed(int(tx.seed) % (2 ** 63) + 1)
        search = _search(dev)  # the sign steps' and screens' precision (every reported number: float32)
        cols = {n: torch.unique(torch.cat([tx.vpd[n].any(0).nonzero().flatten()] + [torch.tensor(
            [c for m, _, c in g.node_set() if m == n], dtype=torch.long, device=dev) for g in graphs])) for n in nat.names}
        E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols, outside="on")
        where = (torch.arange(T, device=dev) <= max(tx.targets))[:, None]
        inv = {n: {int(c): i for i, c in enumerate(cols[n].tolist())} for n in nat.names}
        circ, lever = [], []
        for g in graphs:
            ci = {n: torch.zeros(T, len(cols[n]), device=dev) for n in nat.names}
            for n, t, c in g.node_set():
                ci[n][t, inv[n][c]] = 1.0
            circ.append(ci)
            lever.append({n: (tx.vpd[n][:, cols[n]] & (ci[n] == 0) & where).nonzero(as_tuple=True) for n in nat.names})
        keep = {n: torch.stack([1 - ci[n] for ci in circ]) for n in nat.names}  # [circuits, T, columns]: the circuit off
        sizes = [[len(lv[n][0]) for n in nat.names] for lv in lever]
        N = [sum(z) for z in sizes]
        flat = [[(n, t, int(c)) for n in nat.names for t, c in zip(lv[n][0].tolist(), cols[n][lv[n][1]].tolist())] for lv in lever]
        places = {}

        def kl(items, precision="score"):  # KL(model - F || model - F - C) per item's rows; items [(circuit, x [R, N_i])]
            if not items:
                return []
            key = tuple((i, x.shape[0]) for i, x in items)
            if key not in places:  # each row's removals' places, per site
                rows = torch.tensor([i for i, r in key for _ in range(r)], dtype=torch.long, device=dev)
                pl = {}
                for n in nat.names:
                    rr, tt, kk, r0 = [], [], [], 0
                    for i, r in key:
                        t_, k_ = lever[i][n]
                        if len(t_):
                            rr.append((r0 + torch.arange(r, device=dev)).repeat_interleave(len(t_)))
                            tt.append(t_.repeat(r))
                            kk.append(k_.repeat(r))
                        r0 += r
                    pl[n] = (torch.cat(rr), torch.cat(tt), torch.cat(kk)) if rr else None
                places[key] = (rows, pl)
            rows, pl = places[key]
            m1, m2 = {}, {}
            for j, n in enumerate(nat.names):
                m = torch.ones(len(rows), T, len(cols[n]), device=dev)
                if pl[n] is not None:
                    vals = torch.cat([x[:, sum(sizes[i][:j]):sum(sizes[i][:j + 1])].reshape(-1) for i, x in items])
                    m = m.index_put(pl[n], 1 - vals)
                m1[n] = m
                m2[n] = m * keep[n][rows]
            return carry.kl(E.log_probs(m1, precision), E.log_probs(m2, precision)).split([r for _, r in key])

        def chunks(items, cap):  # items in runs of at most cap rows
            out, run, k = [], [], 0
            for it in items:
                if run and k + it[1].shape[0] > cap:
                    out.append(run)
                    run, k = [], 0
                run.append(it)
                k += it[1].shape[0]
            return out + ([run] if run else [])

        def top(x, m):
            return x * torch.zeros_like(x, dtype=torch.bool).scatter(1, x.topk(m, 1).indices, True)

        def grads(items, precision):
            xs = [(i, x.detach().requires_grad_(True)) for i, x in items]
            if not xs:
                return []
            loss = sum(k.clamp_min(1e-12).log().sum() for k in kl(xs, precision))
            return list(torch.autograd.grad(loss, [x for _, x in xs]))

        live = [i for i in range(len(graphs)) if N[i]]
        with torch.no_grad():
            k0 = [float(k[0]) for k in kl([(i, torch.zeros(1, N[i], device=dev)) for i in range(len(graphs))])]
        rows_out = [{"n_real_flips": {0: necessity_log(e0, k0[i], v0)}, "mask_n_real_flips": {0: []}} for i in range(len(graphs))]
        g0 = dict(zip(live, [-g for g in grads([(i, torch.zeros(1, N[i], device=dev)) for i in live], "score")]))
        screened = {}
        with torch.no_grad():  # the shortlists measured: each part's removal alone, best (least KL) first
            short = {i: g0[i][0].abs().topk(min(N[i], SCREEN)).indices for i in live}
            one = []
            for i in live:
                r = torch.zeros(len(short[i]), N[i], device=dev)
                r[torch.arange(len(short[i]), device=dev), short[i]] = 1.0
                one.append((i, r))
            ks = {}
            for run in chunks(one, SCREEN_ROWS * 4):
                for (i, _), k in zip(run, kl(run, search)):
                    ks[i] = k
            screened = {i: short[i][ks[i].argsort()].tolist() for i in live}
        starts = {(i, m): torch.rand(restarts, N[i], generator=gen).to(dev) for i in range(len(graphs)) for m in ms if 0 < N[i] and m <= N[i]}
        pairs = [(i, m) for m in ms for i in range(len(graphs)) if (i, m) in starts]
        xs = {}
        for i, m in pairs:
            x = starts[(i, m)]
            x[0] = g0[i][0]
            xs[(i, m)] = top(x, m).clamp(0, 1)
        best = {}
        per = max(1, FLIP_ROWS // restarts)
        for s0 in range(0, len(pairs), per):  # every circuit's restarts at every budget together
            group = pairs[s0:s0 + per]
            for _ in range(steps):
                for (i, m), g in zip(group, grads([(i, xs[(i, m)]) for i, m in group], search)):
                    xs[(i, m)] = top((xs[(i, m)] - ADV_STEP * g.sign()).clamp(0, 1), m)
            gx = grads([(i, xs[(i, m)]) for i, m in group], "score")
            with torch.no_grad():  # the top m, ties (entries the steps drove to 0) broken by the gradient, and the
                cand = []  # shortlist's best m
                for (i, m), g in zip(group, gx):
                    x = xs[(i, m)]
                    rank = x + 1e-6 * (-g) / g.abs().amax(1, keepdim=True).clamp_min(1e-30)
                    xb = torch.zeros_like(x).scatter(1, rank.topk(m, 1).indices, 1.0)
                    if len(screened[i]) >= m:
                        pick = torch.zeros(1, N[i], device=dev)
                        pick[0, screened[i][:m]] = 1.0
                        xb = torch.cat([xb, pick])
                    cand.append((i, xb))
                for (i, m), (_, xb), k in zip(group, cand, kl(cand)):
                    r = int(k.argmin())
                    best[(i, m)] = (float(k[r]), [flat[i][j] for j in xb[r].nonzero().flatten().tolist()])
            native.free(dev)
        for i in range(len(graphs)):  # at most m removals: no better than fewer
            row, kls_at = rows_out[i], {0: k0[i]}
            for m in ms:
                if (i, m) not in best:
                    continue
                b_ = best[(i, m)]
                prev = max(mm for mm in row["n_real_flips"] if mm < m)
                if b_[0] >= kls_at[prev]:
                    b_ = (kls_at[prev], row["mask_n_real_flips"][prev])
                kls_at[m] = b_[0]
                row["n_real_flips"][m] = necessity_log(e0, b_[0], v0)
                row["mask_n_real_flips"][m] = b_[1]
        return rows_out


    def _rng(self, tx: Text, g: native.Graph, family: str) -> np.random.Generator:
        """A family's draws for one circuit on one text: seeded by the text and the circuit."""
        key = repr((family, sorted(g.node_set()), _pairs(g) if not g.complete else None))
        return np.random.default_rng([tx.seed, int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)])

    @torch.no_grad()
    def _graph_logq(self, ids: list[int], targets: list[int], graphs: list[native.Graph], ew_rows=None) -> torch.Tensor:
        """log q at the targets of stated graphs' runs in C's semantics (remainders live; on the engine, equal to
        graph_ref within float32 rounding); ew_rows: per graph a 0/1 vector over its connections (graph.parents reader
        by reader, then graph.out), the connections at 0 cut."""
        if not graphs:
            return torch.zeros(0, len(targets), self.nat.target.wte.shape[0], device=self.nat.dev)
        E = self._circuit_engine(ids, targets, graphs)
        ew = ow = None
        if ew_rows is not None:
            inner = [sum(len(ws) for ws in g.parents.values()) for g in graphs]
            ew = torch.cat([ew_rows[j][:k].float() for j, k in enumerate(inner)]).to(self.nat.dev)
            ow = torch.cat([ew_rows[j][k:].float() for j, k in enumerate(inner)]).to(self.nat.dev)
        out = E.graph_log_probs(E.plan(list(graphs), semantics="c"), ew=ew, ow=ow)
        native.free(self.nat.dev)
        return out

    @torch.no_grad()
    def _node_family(self, tx: Text, graphs: list[native.Graph], draws: int, reads: bool = False) -> list[dict]:
        """Per circuit, the node-removal family: draws of its nodes removed from the model, from VPD's answer and from
        the circuit (a stated graph keeps its connections among the rest), the mean KLs from the model's. The runs
        multiply only the circuits' and VPD's subcomponents (engine.Engine's columns; the model's run with every other
        subcomponent folded into its remainder, outside="on"): float32-close to the masked forward over every column."""
        import engine

        nat, T, dev = self.nat, tx.T, self.nat.dev
        cols = {n: {c for g in graphs for m, _, c in g.node_set() if m == n} for n in nat.names}
        circ = {n: torch.tensor(sorted(cols[n]), dtype=torch.long, device=dev) for n in nat.names}
        vcols = {n: torch.unique(torch.cat([circ[n], tx.vpd[n].any(0).nonzero().flatten()])) for n in nat.names}
        Em = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ, outside="on")  # the model
        Ev = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=vcols)               # VPD's answer
        Ec = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ)                # the circuits
        at = {n: {c: i for i, c in enumerate(circ[n].tolist())} for n in nat.names}
        vat = {n: {c: i for i, c in enumerate(vcols[n].tolist())} for n in nat.names}
        vpd = {n: tx.vpd[n][:, vcols[n]].float() for n in nat.names}
        out = []
        for g in graphs:
            nodes = sorted(g.node_set())
            if not nodes:
                out.append({"kl_nodes_c": self._bounds(tx, "live")[0], "kl_nodes_vpd": self._bounds(tx, "live")[1], "reads_nodes": None})
                continue
            rng = self._rng(tx, g, "nodes")
            drops = []
            for j in range(draws):
                d = [nd for nd in nodes if rng.random() < (j + 0.5) / draws]
                drops.append(d or [nodes[int(rng.integers(len(nodes)))]])

            def without(base, d, where):  # [draws, T, K]: base with each draw's nodes off
                m = {n: x[None].repeat(len(d), 1, 1) for n, x in base.items()}
                for b, dd in enumerate(d):
                    for n, t, c in dd:
                        m[n][b, t, where[n][c]] = 0.0
                return m

            def spec(where):  # every draw's reads of the circuit's nodes over an engine's columns
                out_ = {}
                for n, t, c in nodes:
                    x = out_.setdefault(n, ([], []))
                    x[0].append(t)
                    x[1].append(where[n][c])
                return None if not reads else {n: (torch.arange(len(drops), device=dev).repeat_interleave(len(t)),
                                                   torch.tensor(t, device=dev).repeat(len(drops)), torch.tensor(c, device=dev).repeat(len(drops)))
                                               for n, (t, c) in out_.items()}

            def split(got):  # {site: [draws * k]} -> {node: [draws]}, the nodes in spec's order
                by = {}
                for nd in nodes:
                    by.setdefault(nd[0], []).append(nd)
                return {nd: got[n].view(len(drops), len(nds))[:, j].tolist() for n, nds in by.items() for j, nd in enumerate(nds)}

            def run(f, *a, **k):  # (log q, reads or None)
                got = f(*a, **k)
                return (got[0], split(got[1])) if reads else (got, None)

            lm, rm = run(Em.log_probs, without({n: torch.ones(T, len(circ[n]), device=dev) for n in nat.names}, drops, at), reads=spec(at))
            lv = Ev.log_probs(without(vpd, drops, vat))
            if g.complete:
                gm = {n: torch.zeros(T, len(circ[n]), device=dev) for n in nat.names}
                for n, t, c in nodes:
                    gm[n][t, at[n][c]] = 1.0
                lc, rc = run(Ec.log_probs, without(gm, drops, at), reads=spec(at))
            else:  # C's semantics, each draw's graph without its nodes
                lc, rc = run(Ec.graph_log_probs, Ec.plan([_drop_nodes(g, set(d)) for d in drops], semantics="c"), reads=spec(at))
            pairs = {}  # the reads of the nodes each draw keeps, in the model and the circuit
            for b, d in enumerate(drops if reads else []):
                for nd in nodes:
                    if nd not in set(d):
                        pairs.setdefault(nd, []).append((rm[nd][b], rc[nd][b]))
            out.append({"kl_nodes_c": float(carry.kl(lm, lc).mean()), "kl_nodes_vpd": float(carry.kl(lm, lv).mean()),
                        "draws_nodes": [list(x) for x in zip(carry.kl(lm, lc).tolist(), carry.kl(lm, lv).tolist())],
                        "reads_nodes": _read_error(pairs) if reads else None, "_pairs_nodes": pairs})
            native.free(dev)
        return out

    @torch.no_grad()
    def _edge_family(self, tx: Text, graphs: list[native.Graph], draws: int, reads: bool = False) -> list[dict]:
        """Per circuit, the connection-cut family: draw j cuts a random subset of the stated connections into one
        reader (the prediction among the readers) in the model, in VPD's answer and in the circuit, all in C's
        semantics (the reader reads its input without those writers' outputs), or, a read with stated query-key pairs
        drawn, a random subset of its pairs (graph_ref's ("pairs", layer, t) cut, the circuit's queries and keys of that
        read the split; a read with circuit queries and no stated pairs, every pair of its row, engine.read_pairs:
        monitorbench's 10-12; _edge_draws); the mean KLs from the model's. A circuit without connections or pairs: None. Every draw of a side runs in one batch on the engine (_sides)."""
        sides = self._sides(tx, graphs)
        out = []
        for g in graphs:
            cuts = self._edge_draws(tx, g, draws)
            if cuts is None:
                out.append({"kl_edges_c": None, "kl_edges_vpd": None, "reads_edges": None})
                continue
            got = sides(g, cuts=cuts, reads=reads)
            lm, lv, lc = got[:3]
            row = {"kl_edges_c": float(carry.kl(lm, lc).mean()), "kl_edges_vpd": float(carry.kl(lm, lv).mean()), "reads_edges": None,
                   "draws_edges": [list(x) for x in zip(carry.kl(lm, lc).tolist(), carry.kl(lm, lv).tolist())]}
            if reads:  # the circuit nodes' reads under each draw, model and circuit
                rm, rc = got[3], got[4]
                pairs = {nd: list(zip(rm[nd].tolist(), rc[nd].tolist())) for nd in rm}
                row["reads_edges"], row["_pairs_edges"] = _read_error(pairs), pairs
            out.append(row)
            native.free(self.nat.dev)
        return out

    def _edge_draws(self, tx: Text, g: native.Graph, draws: int) -> list | None:
        """The connection-cut family's draws for one circuit (graph_ref's cuts form, one dict per draw), None with
        nothing to cut: draw j picks a reader, the prediction or a read with query-key pairs (a stated graph's, or every
        pair of a read whose circuit queries state none) and cuts each of its connections or pairs with probability
        (j + 0.5) / draws, at least one; a pair cut's split the circuit's queries at the read and keys up to it."""
        import engine

        nodes = g.node_set()
        par, outs = (native.all_edges(nodes, tx.targets) if g.complete else (g.parents, g.out))
        sp = dict({} if g.complete else (getattr(g, "pairs", None) or {}))
        for nd in sorted(nodes):  # a read with circuit queries and no entry claims every pair of its row
            L, kd = native._name_layer_kind(nd[0])
            if kd == "q_proj" and (L, nd[1]) not in sp:
                sp[(L, nd[1])] = engine.read_pairs(nodes, L, nd[1])
        readers = [r for r in sorted(par) if par[r]] + ([None] if outs else [])
        readers += [("pairs",) + read for read in sorted(sp) if sp[read]]  # reads with query-key pairs
        if not readers:
            return None
        rng = self._rng(tx, g, "edges")
        cuts = []
        for j in range(draws):
            r = readers[int(rng.integers(len(readers)))]
            if r is not None and r[0] == "pairs":
                ps, (_, layer, t) = sp[r[1:]], r
                split = sorted(nd for nd in nodes if native._name_layer_kind(nd[0])[0] == layer and
                               ((native._name_layer_kind(nd[0])[1] == "q_proj" and nd[1] == t)
                                or (native._name_layer_kind(nd[0])[1] == "k_proj" and nd[1] <= t)))
                cuts.append({r: (split, [pr for pr in ps if rng.random() < (j + 0.5) / draws] or [ps[int(rng.integers(len(ps)))]])})
                continue
            ws = list(par[r]) if r is not None else list(outs)
            cuts.append({r: [w for w in ws if rng.random() < (j + 0.5) / draws] or [ws[int(rng.integers(len(ws)))]]})
        return cuts

    def _sides(self, tx: Text, graphs: list[native.Graph]):
        """C's families' three sides on the engine, every draw a row: sides(g, cuts=[per draw {reader: [writers]}] or
        patch=[per draw {node: value}]) -> log q of the model, VPD's answer and circuit g under each draw, in C's
        semantics (graph_ref's cuts and patch): the model over the circuits' subcomponents with every other one folded
        into the remainders (outside="on"), VPD's answer over its own and the circuits', the circuit over its own."""
        nat, T, dev = self.nat, tx.T, self.nat.dev
        vpd_cols = {n: tx.vpd[n].any(0).nonzero().flatten() for n in nat.names}
        Em = self._circuit_engine(tx.ids, tx.targets, graphs, outside="on")
        Ev = self._circuit_engine(tx.ids, tx.targets, graphs, extra=vpd_cols)
        Ec = self._circuit_engine(tx.ids, tx.targets, graphs)
        model = native.Graph(masks={n: torch.ones(1, 1, dtype=torch.bool, device=dev) for n in nat.names})  # complete: runs as u
        vpd = native.Graph(masks=tx.vpd)
        u_m = {n: torch.ones(1, 1, Em.width(n), device=dev) for n in nat.names}
        u_v = {n: tx.vpd[n][:, Ev.columns[n]].float()[None] for n in nat.names}

        def where(E):
            return {n: {c: i for i, c in enumerate(E.columns[n].tolist())} for n in nat.names}

        at = {id(E): where(E) for E in (Em, Ev, Ec)}

        def patches(E, patch):  # per draw {node: value} -> {site: (bool [D, T, K], values [D, T, K])}
            if patch is None:
                return None
            out = {}
            for b, vals in enumerate(patch):
                for (n, t, c), v in vals.items():
                    if n not in out:
                        out[n] = (torch.zeros(len(patch), T, E.width(n), dtype=torch.bool, device=dev),
                                  torch.zeros(len(patch), T, E.width(n), device=dev))
                    k = at[id(E)][n][c]
                    out[n][0][b, t, k] = True
                    out[n][1][b, t, k] = v
            return out

        def spec(E, nodes, D):  # every row's reads of nodes: {site: (rows, positions, columns)}
            out = {}
            for n, t, c in nodes:
                x = out.setdefault(n, ([], [], []))
                x[1].append(t)
                x[2].append(at[id(E)][n][c])
            return {n: (torch.arange(D, device=dev).repeat_interleave(len(t)), torch.tensor(t, device=dev).repeat(D),
                        torch.tensor(c, device=dev).repeat(D)) for n, (_, t, c) in out.items()}

        def split(got, nodes, D):  # {site: [D * k]} -> {node: [D]}
            out, k = {}, {}
            for n, t, c in nodes:
                k.setdefault(n, []).append((n, t, c))
            for n, nds in k.items():
                v = got[n].view(D, len(nds))
                for j, nd in enumerate(nds):
                    out[nd] = v[:, j]
            return out

        def sides(g, cuts=None, patch=None, reads=False):
            """log q of the model, VPD's answer and g under each draw; reads: also g's nodes' activations in the model's
            run and g's, {node: [D]} each."""
            D = len(cuts if cuts is not None else patch)
            nodes = sorted(g.node_set()) if reads else None
            lm = Em.graph_log_probs(Em.plan([model] * D, semantics="c", cuts=cuts), u_m, patch=patches(Em, patch),
                                    reads=spec(Em, nodes, D) if reads else None)
            lv = Ev.graph_log_probs(Ev.plan([vpd] * D, semantics="c", cuts=cuts), u_v, patch=patches(Ev, patch))
            u_c = None
            if g.complete:
                u_c = {n: torch.zeros(1, T, Ec.width(n), device=dev) for n in nat.names}
                for n, t, c in g.node_set():
                    u_c[n][0, t, at[id(Ec)][n][c]] = 1.0
            lc = Ec.graph_log_probs(Ec.plan([g] * D, semantics="c", cuts=cuts), u_c, patch=patches(Ec, patch),
                                    reads=spec(Ec, nodes, D) if reads else None)
            if not reads:
                return lm, lv, lc
            return lm[0], lv, lc[0], split(lm[1], nodes, D), split(lc[1], nodes, D)

        sides.spec, sides.split, sides.engines, sides.at = spec, split, (Em, Ev, Ec), at
        return sides

    @torch.no_grad()
    def _fresh_sources(self, tx: Text, p: int, seed: int) -> list[int]:
        """FRESH_SOURCES tokens for position p sampled without replacement from the model's distribution there, the
        text's token and the three likeliest others (every split's carry.alternatives ranks) left out; keyed by seed and
        p alone."""
        lp = self.nat.reference([tx.ids[:p]], [p - 1])[0, 0].cpu().double()  # MPS has no float64
        others = [int(z) for z in lp.topk(4).indices.tolist() if z != tx.ids[p]][:3]
        w = lp.exp()
        w[[tx.ids[p]] + others] = 0.0
        gen = torch.Generator().manual_seed(int(hashlib.sha256(repr(("fresh", seed, p)).encode()).hexdigest()[:15], 16))
        return torch.multinomial(w, min(FRESH_SOURCES, int((w > 0).sum())), replacement=False, generator=gen).tolist()

    def _interchange_draws(self, tx: Text, positions: list[int], split: str, draws: int,
                           first: dict | None = None) -> list[tuple[int, int]]:
        """An interchange split's draws on a circuit's positions, (position, source token) pairs keyed by the text's
        seed plus the split's offset, the position and the source, never by the circuit (circuits on the same
        positions get the same draws): per position the split's ranks of the model's likeliest other tokens
        (carry.alternatives) and, for eval, _fresh_sources enough to fill the budget; the first draws pairs in an order
        keyed the same way (first: {position: priority}, lower first, ahead of that order). search: ranks 0 and 2; eval:
        rank 1 and the samples, which leave out ranks 0 to 2 (disjoint)."""
        off, ranks, fresh = INTERCHANGE_SPLITS[split]
        seed = int(tx.seed) + off
        f = max(1, -(-draws // len(positions)) - len(ranks)) if fresh else 0
        pairs = []
        for p in positions:
            pairs += [(p, x[p]) for x in carry.alternatives(self.nat, tx.ids, p, ranks)]
            pairs += [(p, z) for z in (self._fresh_sources(tx, p, seed)[:f] if f else [])]
        first = first or {}
        return sorted(pairs, key=lambda pz: (first.get(pz[0], 0), hashlib.sha256(repr((seed, pz)).encode()).digest()))[:draws]

    @torch.no_grad()
    def _changed_logq(self, tx: Text, graphs: list[native.Graph], got: list, strengths: torch.Tensor | None = None,
                      cuts: list | None = None) -> tuple:
        """Circuits' log q at the targets in C's semantics (_circuit_logq's: a stated graph as stated, a complete one as
        its subcomponents' masked run, the remainders live) on the text [G, targets, V] and on each changed text (p, z)
        of got (a list of [G, targets, V]): one engine on the text, and every changed text run from the earliest change
        in its batch on, reusing the text's run before it (engine.Engine.graph_log_probs reuse: the model is causal),
        CHANGED_ROWS rows a batch. strengths [G, E + O]: every circuit the one stated graph graphs[0] at its row's 0/1
        connection strengths (changed_edges), from its one plan. cuts: per graph Engine.plan's cuts ({reader node or
        None: [writer nodes]}, connections at strength 0; a complete graph's subcomponents read each other in full but
        for them: teacher_reads' value-read cuts)."""
        nat, T = self.nat, tx.T
        E = self._circuit_engine(tx.ids, tx.targets, graphs[:1] if strengths is not None else graphs)
        if strengths is not None:
            plan = E.plan(graphs[0], semantics="c")
            ew, ow = strengths[:, :plan["E"]], strengths[:, plan["E"]:]
            st = {}
            q0 = E.graph_log_probs(plan, ew=ew, ow=ow, keep=st)
            G, out = len(graphs), []
            per = max(1, CHANGED_ROWS // G)
            for s0 in range(0, len(got), per):
                part = got[s0:s0 + per]
                D = len(part)
                tok = torch.tensor(tx.ids).repeat(D * G, 1)
                for d, (p, z) in enumerate(part):
                    tok[d * G:(d + 1) * G, p] = z
                lq = E.graph_log_probs(plan, ew=ew.repeat(D, 1), ow=ow.repeat(D, 1),
                                       reuse=(E.tile_states(st, D), tok, min(p for p, _ in part)))
                out += list(lq.split(G))
                native.free(nat.dev)
            return q0, out
        plan = E.plan(list(graphs), semantics="c", cuts=cuts)
        u = None
        if any(g.complete for g in graphs):  # a complete graph runs as u: 1 on its subcomponents
            u = {n: torch.zeros(len(graphs), T, E.width(n), device=nat.dev) for n in nat.names}
            for b, g in enumerate(graphs):
                if g.complete:
                    for n, m in self.masks(g, T).items():
                        u[n][b] = m[:, E.columns[n]].float()
        st = {}
        q0 = E.graph_log_probs(plan, u, keep=st)
        G, out = len(graphs), []
        per = max(1, CHANGED_ROWS // G)
        for s0 in range(0, len(got), per):
            part = got[s0:s0 + per]
            D = len(part)
            tok = torch.tensor(tx.ids).repeat(D * G, 1)
            for d, (p, z) in enumerate(part):
                tok[d * G:(d + 1) * G, p] = z
            lq = E.graph_log_probs(E.repeat(plan, D), None if u is None else {n: m.repeat(D, 1, 1) for n, m in u.items()},
                                   reuse=(E.tile_states(st, D), tok, min(p for p, _ in part)))
            out += list(lq.split(G))
            native.free(nat.dev)
        return q0, out

    @torch.no_grad()
    def _changed_vpd(self, tx: Text, got: list) -> tuple:
        """VPD's answer's log q at the targets (its masked run, the remainders live) on the text and on each changed
        text of got, kept on the text (they do not depend on the circuits: a teacher's candidate sets share them)."""
        cache = tx.__dict__.setdefault("_changed", {})
        key = (id(tx.vpd), int(torch.stack([m.sum() for m in tx.vpd.values()]).sum()))
        if cache.get("key") != key:
            cache.clear()
            cache["key"] = key
        if "v0" not in cache:
            cache["v0"] = self.runs(tx.ids, tx.targets, [tx.vpd], True)
        out = []
        for p, z in got:
            if (p, z) not in cache:
                x = list(tx.ids)
                x[p] = z
                cache[(p, z)] = self.runs(x, tx.targets, [tx.vpd], True)
            out.append(cache[(p, z)])
        return cache["v0"], out

    def _circuit_logq(self, ids: list[int], targets: list[int], graphs: list[native.Graph]) -> torch.Tensor:
        """log q at the targets [len(graphs), len(targets), V] of circuits in C's semantics on ids: a stated graph as
        stated (_graph_logq), a complete one as its masked run with live remainders (runs)."""
        out = [None] * len(graphs)
        whole = [i for i, g in enumerate(graphs) if g.complete]
        stated = [i for i, g in enumerate(graphs) if not g.complete]
        if whole:
            for i, lq in zip(whole, self.runs(ids, targets, [self.masks(graphs[i], len(ids)) for i in whole], True)):
                out[i] = lq
        if stated:
            for i, lq in zip(stated, self._graph_logq(ids, targets, [graphs[i] for i in stated])):
                out[i] = lq
        return torch.stack(out)

    @torch.no_grad()
    def changed_inputs(self, tx: Text, graphs: list[native.Graph], split: str = "eval", draws: int = 16,
                       strengths: torch.Tensor | None = None, cuts: list | None = None) -> list[dict]:
        """The changed-input family (paired F2, the lead's decision 10-12): the circuit's identities on changed texts.
        Prompts: F2's positions (the sensitivity probe's, at least share_floor of its total), the most moved first, each
        with the split's source tokens (_interchange_draws: search ranks 0 and 2, eval rank 1 and samples), up to draws.
        On a changed text x': KL_C = KL(model(x') || the circuit run on x' in C's semantics, its embedding reading x'),
        KL_V the same for VPD's answer on the text (its base identities: a part switching on in x' is missing from both),
        and the bound KL0 = KL(model(x') || the circuit run on x), the circuit ignoring the change. Per prompt
        clip(ln(KL0 / KL_C) / ln(KL0 / KL_V), 0, 1) (none when VPD's answer is no nearer than KL0), pooled weighted by
        KL(model(x') || model(x)): s_changed, a reported column (it carries the circuit's error on x, which F1 prices).
        s_track, C's term and the teachers' (the lead's decision 10-12), the circuit's change alone: the circuit's move
        from x to x' applied to the model's prediction on x (_transplant), the same for VPD's answer, scored
        clip(ln(W / KL(model(x') || C's transplant)) / ln(W / KL(model(x') || VPD's)), 0, 1) with W = KL(model(x') ||
        model(x)) the no-change bound (none when VPD's transplant is no nearer), pooled weighted by W: a circuit whose
        prediction moves as the model's does scores 1 whatever its error on x. Per circuit: s_changed, kl_changed_c
        (KL_C, weighted the same), prompts_changed (those scored), s_track, kl_track_c, prompts_track, context_share
        (the share of the probe's moved at the circuit's positions) and draws_changed."""
        nat = self.nat
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]
        moved = np.asarray(tx.moved, dtype=float)
        total = float(moved.sum())
        share = [sum(moved[t - 1] for t in {t for _, t, _ in g.node_set()} if 0 < t <= len(moved)) / total
                 if total > 0 else None for g in graphs]
        positions = [p for p, _ in tx.f2]
        got = self._interchange_draws(tx, positions, split, draws, first={p: -moved[p - 1] for p in positions}) \
            if positions and tx.vpd is not None else []
        acc = [[0.0, 0.0, 0.0, 0] for _ in graphs]  # sum of w s, of w, of w KL_C; prompts scored
        trk = [[0.0, 0.0, 0.0, 0] for _ in graphs]  # the same for s_track
        trk_raw = [0.0 for _ in graphs]  # s_track's sum of w s without the floor at 0
        q0, lcs = self._changed_logq(tx, graphs, got, strengths, cuts) if got else (None, [])
        v0, lvs = self._changed_vpd(tx, got) if got else (None, [])
        for (p, z), lc, lv in zip(got, lcs, lvs):
            x = list(tx.ids)
            x[p] = z
            ref = nat.reference([x], tx.targets)
            w = float(carry.kl(ref, tx.ref)[0])
            kv = float(carry.kl(ref, lv)[0])
            kc = carry.kl(ref, lc).tolist()
            k0 = carry.kl(ref, q0).tolist()
            tv = float(carry.kl(ref, _transplant(tx.ref, lv, v0))[0])
            tc = carry.kl(ref, _transplant(tx.ref, lc, q0)).tolist()
            for b in range(len(graphs)):
                for a, sb, k in ((acc[b], _fid(k0[b], kc[b], kv), kc[b]), (trk[b], _fid(w, tc[b], tv), tc[b])):
                    if sb is not None:
                        a[0], a[1], a[2], a[3] = a[0] + w * sb, a[1] + w, a[2] + w * k, a[3] + 1
                rt = _fid_raw(w, tc[b], tv)  # scored where s_track is (the same conditions)
                if rt is not None:
                    trk_raw[b] += w * rt
            native.free(nat.dev)
        return [{"s_changed": a[0] / a[1] if a[1] > 0 else None, "kl_changed_c": a[2] / a[1] if a[1] > 0 else None,
                 "prompts_changed": a[3], "s_track": t[0] / t[1] if t[1] > 0 else None,
                 "kl_track_c": t[2] / t[1] if t[1] > 0 else None, "prompts_track": t[3], "context_share": sh,
                 "s_track_raw": tr / t[1] if t[1] > 0 else None,
                 "draws_changed": got} for a, t, tr, sh in zip(acc, trk, trk_raw, share)]

    def changed_search(self, tx: Text, graphs: list[native.Graph], draws: int = 16, cuts: list | None = None,
                       mode: str = "c") -> list[dict]:
        """The teachers' changed-input family: changed_inputs on the search split alone (never a reward, gate or
        table's) for a list of candidate circuits on one text, each changed text run once for all of them. A node set
        without stated structure: native.Graph(nodes, complete=True); cuts, per graph its connections cut
        (_changed_logq); mode "pp": track_pp's path-patched s_track_pp instead (cuts then per graph pp_run's
        {(layer, t): [source positions]})."""
        if mode == "pp":
            return self.track_pp(tx, graphs, "search", draws, cuts=cuts)
        return self.changed_inputs(tx, graphs, "search", draws, cuts=cuts)

    @torch.no_grad()
    def donor_activations(self, ids: list[int], T: int) -> tuple[dict, int]:
        """Every subcomponent's activation in the real model on a donor text aligned with a text of length T at their
        ends: ({site: [T, C]}, pad), the donor's last T positions, or, a donor shorter than T, its positions at the
        text's last len(ids) with the first pad = T - len(ids) left empty (zeros)."""
        nat, n = self.nat, len(ids)
        rec = {}
        nat.run(list(ids), [n - 1], nat._full_plan(n, 1), *nat._one(), record=rec)
        pad = max(0, T - n)
        out = {}
        for site in nat.names:
            a = rec[site][0]
            out[site] = a[n - T:] if pad == 0 else torch.cat([a.new_zeros(pad, a.shape[1]), a])
        return out, pad

    def vpd_baseline(self, tx: Text, k: int, pool: dict | None, split: str = "search") -> dict:
        """The reward's size-matched baseline (primerl's C - C_vpd@k): c and c_rl of VPD's first k subcomponents (its
        order, tx.vpd_order when set) on the text in reward mode on split, {"c_vpd_k", "c_rl_vpd_k"}, interpolated in
        log2 k between VPD's first k at the powers of 2 around k (the last at VPD's whole order), each scored once per
        text and split (kept on tx): an RL group's answers on one text share them. None without VPD's order."""
        try:
            order = getattr(tx, "vpd_order", None) or self.nat._vpd_order(tx.ids, tx.targets)
        except Exception:  # no causal-importance network and no saved order
            order = None
        if not order:
            return {"c_vpd_k": None, "c_rl_vpd_k": None, "n_vpd_k": None}
        k = max(1, min(int(k), len(order)))
        lo = 1 << (k.bit_length() - 1)
        hi = min(len(order), lo if lo == k else lo << 1)
        kept = tx.__dict__.setdefault("_vpd_c", {})

        def at(n):
            if (n, split) not in kept:
                r = self.fidelity(tx, [native.Graph(order[:n], complete=True)], pool, mode="reward", split=split)[0]
                kept[(n, split)] = (r["c"], r["c_rl"], r["n_point"])
            return kept[(n, split)]

        a, b = at(lo), at(hi)
        f = 0.0 if hi == lo else (math.log2(k) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
        mix = lambda x, y: None if x is None or y is None else x + f * (y - x)  # noqa: E731
        return {"c_vpd_k": mix(a[0], b[0]), "c_rl_vpd_k": mix(a[1], b[1]), "n_vpd_k": mix(a[2], b[2])}

    def start_set(self, tx: Text, split: str = "search", floor: float = 0.9, draws: int = CHANGED_DRAWS) -> dict:
        """The influential start set (the lead's 10-12, the new teacher's start): VPD's actives at the targets and the m
        most influential positions (F2's, the probe's moved), every connection (a node set), m the smallest whose
        s_track_pp on split reaches floor (all of F2's positions when none does); kept on tx. {graph, m, positions,
        s_track_pp}."""
        kept = tx.__dict__.setdefault("_start", {})
        if (split, floor) in kept:
            return kept[(split, floor)]
        nat = self.nat
        moved = np.asarray(tx.moved, dtype=float)
        f2 = sorted((q for q, _ in tx.f2), key=lambda q: -moved[q - 1])
        act = [(n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist()]
        for m in range(len(f2) + 1):
            ps = set(tx.targets) | set(f2[:m])
            g = native.Graph(sorted(nd for nd in act if nd[1] in ps), complete=True)
            sc = self.track_pp(tx, [g], split, draws)[0]["s_track_pp"]
            if sc is not None and sc >= floor:
                break
        kept[(split, floor)] = {"graph": g, "m": m, "positions": sorted(ps), "s_track_pp": sc}
        return kept[(split, floor)]

    def tau(self, tx: Text, pool: dict | None, split: str = "search") -> dict:
        """The reward's per-text bar and start bits (the lead's 10-12, constrained form: faithful first, then small),
        kept on tx: the influential start set's (start_set) reward-mode row on split, phi_start and phi_tau (phi: the
        lower of the start's phi and every family at 0.9), c_start and c_rl_start (and tau = min(0.9, c_rl_start), the
        earlier form), bits_start (its raw description bits, size_bits), and the start's m, nodes and positions."""
        kept = tx.__dict__.setdefault("_tau", {})
        if split in kept:
            return kept[split]
        st = self.start_set(tx, split)
        row = self.fidelity(tx, [st["graph"]], pool, mode="reward", split=split)[0]
        nat = self.nat
        ph, pt = phi(row)
        kept[split] = {"phi_start": ph, "phi_tau": min(ph, pt) if ph is not None and pt is not None else pt,
                       "c_start": row["c"], "c_rl_start": row["c_rl"],
                       "tau": min(0.9, row["c_rl"]) if row["c_rl"] is not None else None,
                       "bits_start": size_bits(st["graph"], nat.positions(tx.targets), sum(nat.C.values()), tx.targets, active=tx.vpd),
                       "m_start": st["m"], "nodes_start": st["graph"].count(), "positions_start": len(st["positions"])}
        return kept[split]

    def far_pool(self, split: str) -> list:
        """The far donors' texts of a split, [(name, token ids)] (DONOR_POOLS) in the model's vocabulary, loaded once."""
        if split not in self.donor_pools:
            which, idx = DONOR_POOLS[split]
            tasks = native.tasks(which)
            V = self.nat.target.wte.shape[0]  # a text in another vocabulary (a tiny test model's) is no donor
            self.donor_pools[split] = [(tasks[i].stem, x) for i in idx if i < len(tasks) for x in [native.text(tasks[i])[0]] if max(x) < V]
        return self.donor_pools[split]

    def donors_search(self, tx: Text, graphs: list[native.Graph], near: int = DONORS_NEAR, far: int = DONORS_FAR) -> list[dict]:
        """The teachers' donor resampling: donors on the search split (near: changed_search's texts; far: training
        texts), never a gate or table's, for a list of candidate circuits on one text (one run per donor for all)."""
        return self.donors(tx, graphs, "search", near, self.far_donors(tx, self.far_pool("search"), far) if far else None)

    def far_donors(self, tx: Text, pool: list, k: int = DONORS_FAR) -> list:
        """k of pool's texts ((name, token ids) pairs) for donor resampling, drawn by the text's seed, those at least as
        long as the text first (a shorter one leaves the text's first positions unresampled)."""
        rng = np.random.default_rng([int(tx.seed), 11])
        order = [pool[i] for i in rng.permutation(len(pool)) if list(pool[i][1]) != list(tx.ids)]
        return ([x for x in order if len(x[1]) >= tx.T] + sorted((x for x in order if len(x[1]) < tx.T), key=lambda x: len(x[1]), reverse=True))[:k]

    @torch.no_grad()
    def donors(self, tx: Text, graphs: list[native.Graph], split: str = "eval", near: int = DONORS_NEAR,
               far: list | None = None, rows: int = DONOR_ROWS) -> list[dict]:
        """Donor resampling (an evaluation column, the lead's decision 10-12; not in c): the circuit run in C's semantics
        with every subcomponent outside it at its activation on one donor text, every one from that donor, so parts that
        cancel on real text stay consistent (the remainders live); paired with VPD's answer under the same donors; the
        KL from the model on the text. Donors: "near", the changed-input texts of split (changed_inputs' draws), and
        "far", far's texts (token id lists, other held-out texts), aligned at their ends, those shorter than the text
        leaving its first positions unresampled (their activations live). The bound is the run that keeps nothing,
        every subcomponent at the donor's (KL_0), not the empty circuit's KL: a near donor differs from the text at one
        position, so against KL_empty (~100 bits) every circuit would score near 1. Per circuit and pool:
        s_donor_<pool> = clip(ln(KL_0 / KL_C) / ln(KL_0 / KL_VPD), 0, 1) on the mean KLs; s_donor_<pool>_worst, the
        lowest donor's (a donor where VPD's answer is no nearer than KL_0 left out); s_donor_<pool>_p90, the donor at
        the 90th percentile from the best (the 10th percentile of the donors' scores); kl_donor_<pool> (KL_C, KL_VPD,
        KL_0); donors_<pool> (how many); draws_donor_<pool>, every donor's [name, KL_C, KL_VPD, KL_0, score]. Every
        draw is the KL from the model's own output on the text, never a loss against the text's next token: it is never
        negative, so a circuit's errors on different donors add up instead of cancelling (Scheurer et al. 2023, causal
        scrubbing's pitfalls). The flip adversary's in-distribution counterpart: a circuit that holds here but not
        under the flips is charged there for combinations no real text makes."""
        import engine

        nat, T, dev = self.nat, tx.T, self.nat.dev
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]
        texts = {}  # pool -> [(name, token ids)]: near named [position, token], far by its text
        if near:
            moved = np.asarray(tx.moved, dtype=float)
            positions = [p for p, _ in tx.f2]
            for p, z in (self._interchange_draws(tx, positions, split, near, first={p: -moved[p - 1] for p in positions})
                         if positions else []):
                x = list(tx.ids)
                x[p] = z
                texts.setdefault("near", []).append(([p, z], x))
        if far:
            texts["far"] = [(name, list(x)) for name, x in far if list(x) != list(tx.ids)]
        E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=None)
        u = {n: torch.ones(1, 1, nat.C[n], device=dev) for n in nat.names}
        sides = list(graphs) + [native.Graph(masks=tx.vpd), native.Graph()]  # the circuits, VPD's answer, nothing kept
        own = []  # per side, its nodes as {site: bool [T, C]}: left unpatched
        for g in sides:
            m = self.masks(g, T)
            own.append({n: m[n].to(dev) for n in nat.names})
        out = [{} for _ in graphs]
        for pool, xs in texts.items():
            kls = []  # per donor [len(sides)]
            for d0 in range(0, len(xs), max(1, rows // len(sides))):
                chunk = xs[d0:d0 + max(1, rows // len(sides))]
                acts = [self.donor_activations(x, T) for _, x in chunk]
                B = len(chunk) * len(sides)
                patch = {}
                for n in nat.names:
                    vals = torch.stack([a[n] for a, _ in acts]).to(dev)  # [donors, T, C]
                    vals = vals.repeat_interleave(len(sides), 0)
                    keep = torch.stack([~own[j][n] for _ in chunk for j in range(len(sides))])  # outside each side
                    for k, (_, pad) in enumerate(acts):
                        if pad:
                            keep[k * len(sides):(k + 1) * len(sides), :pad] = False
                    patch[n] = (keep, vals)
                lq = E.graph_log_probs(E.plan([g for _ in chunk for g in sides], semantics="c"), u, patch=patch)
                kl = carry.kl(tx.ref, lq).view(len(chunk), len(sides))
                kls += kl.tolist()
                native.free(dev)
            kv, k0 = [k[-2] for k in kls], [k[-1] for k in kls]
            for b in range(len(graphs)):
                kc = [k[b] for k in kls]
                per = [_fid(z, c, v) for z, c, v in zip(k0, kc, kv)]
                each = [x for x in per if x is not None]
                out[b].update({f"s_donor_{pool}": _fid(float(np.mean(k0)), float(np.mean(kc)), float(np.mean(kv))) if kc else None,
                               f"s_donor_{pool}_worst": min(each) if each else None,
                               f"s_donor_{pool}_p90": float(np.percentile(each, 10)) if each else None,
                               f"kl_donor_{pool}": (float(np.mean(kc)), float(np.mean(kv)), float(np.mean(k0))) if kc else None,
                               f"donors_{pool}": len(kc),
                               f"draws_donor_{pool}": [[name, c, v, z, sd] for (name, _), c, v, z, sd in zip(xs, kc, kv, k0, per)]})
        return out

    def interchange_search(self, tx: Text, graphs: list[native.Graph], draws: int = 16, mode: str = "c") -> list[dict]:
        """The teachers' interchange agreement on one text, the search split alone (never a reward, gate or table's):
        per candidate circuit s_interchange on C's log scale, kl_interchange_c and kl_interchange_vpd. Candidates on the
        same positions share their draws, and every draw's changed prompt runs once for all of them. A node set
        without stated structure: native.Graph(nodes, complete=True)."""
        if mode == "pp":  # the path-patched interchange, C's term (interchange_pp)
            return self.interchange_pp(tx, graphs, "search", draws)
        e0 = self._bounds(tx, "live")[0]
        rows = self._interchange_family(tx, graphs, draws, split="search")
        return [dict(r, s_interchange=_fid(e0, r["kl_interchange_c"], r["kl_interchange_vpd"])) for r in rows]

    @torch.no_grad()
    def _interchange_family(self, tx: Text, graphs: list[native.Graph], draws: int = 16, split: str = "eval",
                            reads: bool = False) -> list[dict]:
        """Per circuit, interchange interventions with the identity alignment on a split's draws (_interchange_draws:
        "eval", held out, for fidelity's rows, the reward's gates and tables; "search" for the teachers,
        interchange_search). A draw (p, z) sets the circuit's subcomponents at p to their activations in the real model
        on the text with z at p, in the model (everything else computing on the text), in the circuit (its later nodes
        reading the patched ones through stated connections) and in VPD's answer, all in C's semantics (graph_ref); the
        mean KLs from the model's, kl_interchange_c and kl_interchange_vpd, and the draws (draws_interchange)."""
        nat, T, dev = self.nat, tx.T, self.nat.dev
        source = {}

        def source_record(p, z):
            if (p, z) not in source:
                x = list(tx.ids)
                x[p] = z
                rec = {}
                nat.run(x, [T - 1], nat._full_plan(T, 1), *nat._one(), record=rec)
                source[(p, z)] = rec
            return source[(p, z)]

        sides = self._sides(tx, graphs)
        out = []
        for g in graphs:
            nodes = sorted(g.node_set())
            positions = sorted({t for _, t, _ in nodes if t > 0})
            if not positions:
                out.append({"kl_interchange_c": None, "kl_interchange_vpd": None, "reads_interchange": None, "draws_interchange": []})
                continue
            got_draws = self._interchange_draws(tx, positions, split, draws)
            patch = []
            for p, z in got_draws:
                rec = source_record(p, z)
                patch.append({nd: rec[nd[0]][0, nd[1], nd[2]] for nd in nodes if nd[1] == p})
            got = sides(g, patch=patch, reads=reads)
            lm, lv, lc = got[:3]
            native.free(dev)
            row = {"kl_interchange_c": float(carry.kl(lm, lc).mean()), "kl_interchange_vpd": float(carry.kl(lm, lv).mean()),
                   "reads_interchange": None, "draws_interchange": got_draws}
            if reads:  # the reads of the circuit's nodes at other positions than the fixed ones, model and circuit
                rm, rc, pairs = {nd: v.tolist() for nd, v in got[3].items()}, {nd: v.tolist() for nd, v in got[4].items()}, {}
                for b, vals in enumerate(patch):
                    for nd in nodes:
                        if nd not in vals:
                            pairs.setdefault(nd, []).append((rm[nd][b], rc[nd][b]))
                row["reads_interchange"], row["_pairs_interchange"] = _read_error(pairs), pairs
            out.append(row)
        return out


    def _adv_problem(self, tx: Text, graphs: list[native.Graph], pool: dict, k: int, gen: torch.Generator,
                     batched: bool = False, patterns: bool = True, strengths: torch.Tensor | None = None) -> tuple:
        """adversary()'s objective: the K patterns drawn (pool indices) and, per circuit, (kls, idx, cols): kls(w [R, K],
        b {site: [R, omitted]}, precision="score") -> (KL_C, KL_VPD) [R] bits, differentiable; idx per site the omitted
        subcomponents (positions, column indices); cols per site the engine's columns. batched: also batch(items,
        precision="score", vpd=True), items [(circuit, w, b)] -> [(KL_C, KL_VPD)], every item's rows in one run of each
        kind (the stated graphs' as one list of graphs, the complete ones', VPD's answer's; vpd=False leaves VPD's out,
        KL_VPD None): the same numbers within float32 rounding, a run per kind instead of per circuit. patterns=False:
        the pattern lever stays off (w = 0), so the engine leaves out the patterns' subcomponents (the omitted lever's
        flips: about a third of the columns). A graph given more than once is laid out once. strengths [circuits, E + O]:
        every circuit is the same stated graph (one object), circuit i at connection strengths strengths[i] (the
        graph's parents reader by reader, then its out list), run from one plan of the graph kept on the text (the same
        numbers as each circuit's connections stated as its own graph: an unstated connection runs at strength 0)."""
        import engine

        nat, T, dev = self.nat, tx.T, self.nat.dev
        pick = torch.randperm(len(pool["sources"]), generator=gen)[:k].tolist()
        pats = [pattern(pool, i) for i in pick] if patterns else []  # (patterns off: drawn, never switched on)
        cache = tx.__dict__.setdefault("_adv", {})  # per text: VPD's columns, and the last columns' engine and layouts
        key = (id(tx.vpd), int(torch.stack([m.sum() for m in tx.vpd.values()]).sum()))  # (tx.vpd set or grown: anew)
        if cache.get("key") != key:
            cache.clear()
            cache["key"] = key
        if "vpd_cols" not in cache:
            cache["vpd_cols"] = {n: tx.vpd[n].any(0).nonzero().flatten() for n in nat.names}
        slot, uniq = [], {}  # each circuit's graph's place among the distinct graphs
        for g in graphs:
            slot.append(uniq.setdefault(id(g), (len(uniq), g))[0])
        uniq = [g for _, g in uniq.values()]
        if strengths is not None and (len(uniq) != 1 or uniq[0].complete):
            raise ValueError("strengths run one stated graph on every circuit")
        cols = {}
        for n in nat.names:
            parts = [cache["vpd_cols"][n]] + [p[n].to(dev) for p in pats if n in p and patterns]
            parts += [torch.tensor([c for m, _, c in g.node_set() if m == n], dtype=torch.long, device=dev) for g in uniq]
            cols[n] = torch.unique(torch.cat(parts))
        last = cache.get("engine")
        if last is not None and all(torch.equal(last[0][n], cols[n]) for n in nat.names):
            E, inv, vpd_on, vfree = last[1:]       # the same columns (a text's omitted flips: VPD's answer's): reused
        else:
            E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols)
            inv = vpd_on = vfree = None
            cache["plans"] = {}                    # one-graph plans of this engine (strengths)
        where = (torch.arange(T, device=dev) <= max(tx.targets))[:, None]
        if vpd_on is None:
            vpd_on = {n: tx.vpd[n][:, cols[n]] for n in nat.names}
        P = {}
        for n in nat.names:
            m = torch.zeros(len(pats), len(cols[n]), device=dev)
            for i, p in enumerate(pats):
                if n in p:
                    m[i] = torch.isin(cols[n], p[n].to(dev)).float()
            P[n] = m
        if inv is None:
            inv = {n: {int(c): i for i, c in enumerate(cols[n].tolist())} for n in nat.names}
            vfree = {n: (~vpd_on[n] & where).float() for n in nat.names}
            cache["engine"] = (cols, E, inv, vpd_on, vfree)
        parts = []
        for g in uniq:
            on = {n: torch.zeros(T, len(cols[n]), dtype=torch.bool, device=dev) for n in nat.names}
            for n, t, c in g.node_set():
                on[n][t, inv[n][c]] = True
            omit = {n: vpd_on[n] & ~on[n] & where for n in nat.names}
            parts.append({"on": on, "free": {n: (~vpd_on[n] & ~on[n] & where).float() for n in nat.names},
                          "idx": {n: omit[n].nonzero(as_tuple=True) for n in nat.names}})
        free = {n: torch.stack([q["free"][n] for q in parts]) for n in nat.names}  # [distinct graphs, T, columns]
        onf = {n: torch.stack([q["on"][n] for q in parts]).float() for n in nat.names}
        parts = [parts[j] for j in slot]                 # per circuit
        at_slot = torch.tensor(slot, dtype=torch.long, device=dev)
        plans, flats = {}, {}
        one = None
        if strengths is not None:
            g = uniq[0]
            pk = (tuple(sorted(g.node_set())), tuple((w, r) for r, ws in g.parents.items() for w in ws), tuple(g.out))
            if pk not in cache.setdefault("plans", {}):
                cache["plans"][pk] = E.plan(g, semantics="c")
            one = cache["plans"][pk]

        def layout(key):  # rows (circuit, count) in order: each row's circuit, the omissions' places per site
            if key not in flats:
                circ = torch.tensor([i for i, r in key for _ in range(r)], dtype=torch.long, device=dev)
                fl = {}
                for n in nat.names:
                    rr, tt, kk, r0 = [], [], [], 0
                    for i, r in key:
                        t_, k_ = parts[i]["idx"][n]
                        if len(t_):  # row r0 + s holds b[s, :] at (t_, k_)
                            rr.append((r0 + torch.arange(r, device=dev)).repeat_interleave(len(t_)))
                            tt.append(t_.repeat(r))
                            kk.append(k_.repeat(r))
                        r0 += r
                    fl[n] = (torch.cat(rr), torch.cat(tt), torch.cat(kk)) if rr else None
                flats[key] = (circ, fl)
            return flats[key]

        def batch(items, precision="score", vpd=True):  # vpd=False: the circuits' runs alone (KL_VPD None)
            order = sorted(range(len(items)), key=lambda j: graphs[items[j][0]].complete)  # the stated circuits' rows first
            its = [items[j] for j in order]
            key = tuple((i, w.shape[0]) for i, w, _ in its)
            circ, fl = layout(key)
            ns = sum(r for i, r in key if not graphs[i].complete)
            w = torch.cat([x for _, x, _ in its])
            uc, mv = {}, {}
            for n in nat.names:  # every row's masks at once
                if patterns:
                    a = _PatternOr.apply(w, P[n])  # [rows, columns]: the OR of the patterns on
                    uc[n] = free[n][at_slot[circ]] * a[:, None, :]
                else:  # the pattern lever off: a = 0, outside the circuit only the omissions
                    uc[n] = torch.zeros(len(circ), T, len(cols[n]), device=dev)
                if fl[n] is not None:  # the omissions (disjoint from free)
                    uc[n] = uc[n].index_put(fl[n], torch.cat([b[n].reshape(-1) for _, _, b in its]), accumulate=True)
                if vpd:
                    mv[n] = vpd_on[n].float() + vfree[n] * a[:, None, :] if patterns else vpd_on[n].float().expand(len(circ), -1, -1)
            kc = []
            if ns and one is not None:  # one stated graph, each row at its circuit's strengths
                s = strengths[circ[:ns]]
                kc.append(E.graph_kl(one, u={n: uc[n][:ns] for n in nat.names}, ew=s[:, :one["E"]], ow=s[:, one["E"]:],
                                     precision=precision))
            elif ns:  # C's semantics (graph inside, masked outside): the stated circuits as one list of graphs
                sk = key[:sum(1 for i, _ in key if not graphs[i].complete)]
                if sk not in plans:
                    plans[sk] = E.plan([graphs[i] for i, r in sk for _ in range(r)], semantics="c")
                kc.append(E.graph_kl(plans[sk], u={n: uc[n][:ns] for n in nat.names}, precision=precision))
            if ns < len(circ):
                kc.append(E.kl({n: onf[n][at_slot[circ[ns:]]] + uc[n][ns:] for n in nat.names}, precision))
            kc = torch.cat(kc).split([r for _, r in key])
            kv = E.kl(mv, precision).split([r for _, r in key]) if vpd else [None] * len(its)
            at = {j: k for k, j in enumerate(order)}
            return [(kc[at[j]], kv[at[j]]) for j in range(len(items))]

        problems = [((lambda w, b, precision="score", i=i: batch([(i, w, b)], precision)[0]), parts[i]["idx"], cols)
                    for i in range(len(graphs))]
        return (pick, problems, batch) if batched else (pick, problems)

    def adversary(self, tx: Text, graphs: list[native.Graph], pool: dict, steps: int = ADV_STEPS, restarts: int = ADV_RESTARTS,
                  draws: int = DRAWS, curve=ADV_CURVE, k: int = ADV_PATTERNS) -> list[dict]:
        """The paired adversary over the pool M (module docstring): K patterns drawn by the text's seed, pattern
        weights w in [0, 1]^K shared across positions (an OR of the patterns on), applied at every position up to the
        last target to the subcomponents VPD's answer leaves off, the same in the circuit's run and VPD's; and b in
        [0, 1] on the subcomponents VPD's answer has on and the circuit leaves out, in the circuit's run alone (VPD's
        has them on). Remainders live, not adversarial. The worst case maximizes ln KL_C - ln KL_VPD by sign-gradient
        steps of ADV_STEP from restarts random starts, binarized at 0.5 and scored in float32 at each budget in curve;
        the random members switch each pattern on with probability ADV_RANDOM_ON and each b with 1/2. Per circuit:
        f1_adv (at the largest budget), kl_c_adv, kl_vpd_adv, f1_adv_random (the members' mean), f1_adv_curve, and
        adv_mask (the patterns on, as pool indices, the count of the circuit's omissions switched on and those at the
        targets)."""
        nat, dev = self.nat, self.nat.dev
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
        e0 = self._bounds(tx, "live")[0]
        gen = torch.Generator().manual_seed(int(tx.seed) % (2 ** 63))
        pick, problems, batch = self._adv_problem(tx, graphs, pool, k, gen, batched=True)
        search = _search(dev)  # the sign steps' and screens' precision (every reported number: float32)
        starts, out = [], []
        for kls, idx, _ in problems:  # each circuit's draws in turn, from the text's generator
            wr = (torch.rand(draws, len(pick), generator=gen) < ADV_RANDOM_ON).float().to(dev)
            br = {n: (torch.rand(draws, len(idx[n][0]), generator=gen) < 0.5).float().to(dev) for n in nat.names}
            w = torch.rand(restarts, len(pick), generator=gen).to(dev)
            b = {n: torch.rand(restarts, len(idx[n][0]), generator=gen).to(dev) for n in nat.names}
            starts.append((wr, br, w, b))
        with torch.no_grad():  # the random members, a few circuits' draws a run
            rows, s0 = [], 0
            while s0 < len(problems):
                s1 = s0 + max(1, ADV_BATCH // max(1, draws))
                rows += batch([(i, starts[i][0], starts[i][1]) for i in range(s0, min(s1, len(problems)))])
                s0 = s1
                native.free(dev)
        for kc_r, kv_r in rows:
            rand = [_fid(e0, a, v) for a, v in zip(kc_r.tolist(), kv_r.tolist())]
            out.append({"f1_adv_random": _mean([x for x in rand if x is not None]), "f1_adv_curve": {}})
        w = [s[2] for s in starts]
        b = [s[3] for s in starts]
        for step in range(max(curve) + 1):
            if step in curve:
                with torch.no_grad():
                    wb, bb = [(x > 0.5).float() for x in w], [{n: (x > 0.5).float() for n, x in y.items()} for y in b]
                    for i, ((kc, kv), (_, idx, cols)) in enumerate(zip(batch(list(zip(range(len(problems)), wb, bb))), problems)):
                        gap = (kc.log() - kv.log()).tolist()
                        r = max(range(restarts), key=lambda r: gap[r])
                        row = out[i]
                        row["f1_adv_curve"][step] = _fid(e0, float(kc[r]), float(kv[r]))
                        row.update({"kl_c_adv": float(kc[r]), "kl_vpd_adv": float(kv[r]), "f1_adv": row["f1_adv_curve"][step]})
                        if step == max(curve):  # the final mask, read in one transfer per site (not a sync per entry):
                            on = {n: bb[i][n][r].nonzero().flatten() for n in nat.names}  # the omissions' count, and
                            omitted = [(n, t, c) for n in nat.names for t, c in  # those at the targets in full
                                       zip(idx[n][0][on[n]].tolist(), cols[n][idx[n][1][on[n]]].tolist())]
                            row["adv_mask"] = {"patterns": [pick[j] for j in wb[i][r].nonzero().flatten().tolist()],
                                               "omitted_count": len(omitted),
                                               "omitted_at_targets": [x for x in omitted if x[1] in set(tx.targets)]}
            if step == max(curve):
                break
            leaves = []
            for x, y in zip(w, b):
                leaves += [x.requires_grad_(True)] + [y[n].requires_grad_(True) for n in nat.names]
            loss = sum((kc.clamp_min(1e-12).log() - kv.clamp_min(1e-12).log()).sum()
                       for kc, kv in batch(list(zip(range(len(problems)), w, b)), search))
            grads = iter(torch.autograd.grad(loss, leaves, allow_unused=True))
            with torch.no_grad():  # every circuit's rows are its own: the sum's gradient is each circuit's
                for i in range(len(problems)):
                    gw = next(grads)
                    w[i] = (w[i] + ADV_STEP * gw.sign()).clamp(0, 1)
                    b[i] = {n: (b[i][n] + ADV_STEP * gr.sign()).clamp(0, 1) if gr is not None else b[i][n].detach()
                            for n, gr in zip(nat.names, [next(grads) for _ in nat.names])}
            native.free(dev)
        return out


    def adversary_flips(self, tx: Text, graphs: list[native.Graph], pool: dict, lever: str = "omitted", ms=None,
                        steps: int = ADV_STEPS, restarts: int = ADV_RESTARTS, k: int = ADV_PATTERNS, upto: int | None = None,
                        screens: dict | None = None, strengths: torch.Tensor | None = None) -> list[dict]:
        """The paired adversary with a budget in switched-on parts instead of steps: per budget m, at most m of one
        lever on ("omitted": the VPD actives the circuit leaves out, one subcomponent at one position each;
        "patterns": the pool's patterns, each an OR over every position up to the last target), the other lever off.
        Search: sign-gradient steps of ADV_STEP on the relaxed lever, each projected onto its m largest entries, from
        restarts starts (one the gradient's top m at nothing switched on, the others random), then the top m
        switched on and scored in float32, beside the m best of a measured shortlist (SCREEN parts with the largest
        gradient either way, each switched on alone, in the search precision: the screens only rank which parts are
        tried; every reported number is a float32 run); the worst of them, at most m (no weaker than a smaller budget). Per circuit {f"f1_flips_{lever}": {m: f1_adv},
        f"kl_flips_{lever}": {m: (KL_C, KL_VPD)}, f"mask_flips_{lever}": {m: the parts switched on, (site, position,
        subcomponent) or pool index}}. upto: only the budgets up to it are searched, every budget's starts still
        drawn (the same numbers as the full search at those budgets). screens: a caller's dict of measured shortlist
        gaps by part, shared by this call's graphs and kept across calls (a part measured once, on the first graph
        that needs it): for searches over nearby circuits on one text (a cleanup's candidates, connection subsets of
        one node set), whose shortlists overlap; the caller empties it when its circuits move on. None: every graph's
        shortlist measured on it alone. strengths: _adv_problem's (one stated graph at each circuit's connection
        strengths: edge_flips)."""
        nat, dev = self.nat, self.nat.dev
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
        e0 = self._bounds(tx, "live")[0]
        ms = FLIPS[lever] if ms is None else ms
        gen = torch.Generator().manual_seed(int(tx.seed) % (2 ** 63))
        omitted = lever == "omitted"  # the patterns stay off: VPD's paired run is VPD's answer alone, run once
        pick, problems, batch = self._adv_problem(tx, graphs, pool, k, gen, batched=True, patterns=not omitted, strengths=strengths)
        search = _search(dev)  # the sign steps' and screens' precision (every reported number: float32)
        sizes = [[len(idx[n][0]) for n in nat.names] for _, idx, _ in problems]
        N = [sum(z) if omitted else len(pick) for z in sizes]

        def items(xs):  # [(circuit, its lever's rows [R, N_i])] -> batch()'s items
            out = []
            for i, x in xs:
                R = x.shape[0]
                if omitted:
                    out.append((i, torch.zeros(R, len(pick), device=dev), dict(zip(nat.names, torch.split(x, sizes[i], 1)))))
                else:
                    out.append((i, x, {n: torch.zeros(R, z, device=dev) for n, z in zip(nat.names, sizes[i])}))
            return out

        def grads(xs, precision):  # [(circuit, rows)] -> each rows' gradient of the objective, one batch
            xs = [(i, x.detach().requires_grad_(True)) for i, x in xs]
            if not xs:
                return []
            loss = sum(kc.clamp_min(1e-12).log().sum() - (0 if kv is None else kv.clamp_min(1e-12).log().sum())
                       for kc, kv in batch(items(xs), precision, vpd=not omitted))
            return list(torch.autograd.grad(loss, [x for _, x in xs]))

        def top(x, m):  # each row's m largest entries kept, the rest 0
            keep = torch.zeros_like(x, dtype=torch.bool).scatter(1, x.topk(m, 1).indices, True)
            return x * keep

        with torch.no_grad():
            kv0 = batch(items([(0, torch.zeros(1, N[0], device=dev))]))[0][1] if omitted and problems else None
        live = [i for i in range(len(problems)) if N[i]]
        g0 = dict(zip(live, grads([(i, torch.zeros(1, N[i], device=dev)) for i in live], "score")))
        named = {}  # each circuit's lever's parts by name, made once per distinct graph
        flat = [named.setdefault(id(idx), [(n, t, c) for n in nat.names for t, c in zip(idx[n][0].tolist(), cols[n][idx[n][1]].tolist())])
                if omitted else pick for _, idx, cols in problems]
        known = screens if screens is not None else {}
        key = (lambda i, j: (lever, flat[i][j])) if screens is not None else (lambda i, j: (i, lever, flat[i][j]))
        with torch.no_grad():  # each circuit's shortlist measured: the SCREEN parts with the largest gradient either way,
            short, need, parts = {}, {}, []  # each switched on alone (once per key), the largest gap first; FLIP_ROWS a run
            for i in live:
                short[i] = g0[i][0].abs().topk(min(N[i], SCREEN)).indices
                for j in short[i].tolist():
                    if key(i, j) not in known and key(i, j) not in need:
                        need[key(i, j)] = (i, j)
            by = {}
            for i, j in need.values():
                by.setdefault(i, []).append(j)
            for i, js in by.items():
                for s0 in range(0, len(js), FLIP_ROWS):
                    part = torch.tensor(js[s0:s0 + FLIP_ROWS], device=dev)
                    rows = torch.zeros(len(part), N[i], device=dev)
                    rows[torch.arange(len(part), device=dev), part] = 1.0
                    parts.append((i, part, rows))
            run, k = [], 0
            for j, (i, part, rows) in enumerate(parts):
                run.append((i, part, rows))
                k += len(rows)
                if j == len(parts) - 1 or k + len(parts[j + 1][2]) > FLIP_ROWS:
                    for (i_, part_, _), (kc, kv) in zip(run, batch(items([(a, r) for a, _, r in run]), search, vpd=not omitted)):
                        kv = kv0.expand_as(kc) if kv is None else kv
                        for jj, gap in zip(part_.tolist(), (kc.log() - kv.log()).tolist()):
                            known[key(i_, jj)] = gap
                    run, k = [], 0
            screened = {i: short[i][torch.tensor([known[key(i, j)] for j in short[i].tolist()], device=dev).argsort(descending=True)].tolist()
                        for i in live}
        starts = {(i, m): torch.rand(restarts, N[i], generator=gen).to(dev) for i in range(len(problems)) for m in ms if m <= N[i]}
        out = [{f"f1_flips_{lever}": {}, f"kl_flips_{lever}": {}, f"mask_flips_{lever}": {}} for _ in problems]
        pairs = [(i, m) for m in ms for i in range(len(problems)) if (i, m) in starts and (upto is None or m <= upto)]
        xs = {}
        for i, m in pairs:
            x = starts[(i, m)]
            x[0] = g0[i][0]
            xs[(i, m)] = top(x, m).clamp(0, 1)
        per = max(1, FLIP_ROWS // restarts)
        for s0 in range(0, len(pairs), per):  # every circuit's restarts at every budget together, FLIP_ROWS rows a run
            group = pairs[s0:s0 + per]
            for _ in range(steps):
                for (i, m), g in zip(group, grads([(i, xs[(i, m)]) for i, m in group], search)):
                    xs[(i, m)] = top((xs[(i, m)] + ADV_STEP * g.sign()).clamp(0, 1), m)
            with torch.no_grad():  # each restart's top m, and the shortlist's best m
                xb = {(i, m): torch.zeros_like(xs[(i, m)]).scatter(1, xs[(i, m)].topk(m, 1).indices, 1.0) for i, m in group}
                for i, m in group:
                    if len(screened.get(i, [])) >= m:
                        pick = torch.zeros(1, N[i], device=dev)
                        pick[0, screened[i][:m]] = 1.0
                        xb[(i, m)] = torch.cat([xb[(i, m)], pick])
                for (i, m), (kc, kv) in zip(group, batch(items([(i, xb[(i, m)]) for i, m in group]), vpd=not omitted)):
                    kv = kv0.expand_as(kc) if kv is None else kv
                    gap = kc.log() - kv.log()
                    r = int(gap.argmax())
                    out[i][f"f1_flips_{lever}"][m] = _fid(e0, float(kc[r]), float(kv[r]))
                    out[i][f"kl_flips_{lever}"][m] = (float(kc[r]), float(kv[r]))
                    out[i][f"mask_flips_{lever}"][m] = [flat[i][j] for j in xb[(i, m)][r].nonzero().flatten().tolist()]
            native.free(dev)
        for row in out:  # at most m: a budget's attack is no weaker than a smaller budget's
            f, kl_, mask = row[f"f1_flips_{lever}"], row[f"kl_flips_{lever}"], row[f"mask_flips_{lever}"]
            prev = None
            for m in sorted(f):
                if prev is not None and f[prev] is not None and (f[m] is None or f[m] > f[prev]):
                    f[m], kl_[m], mask[m] = f[prev], kl_[prev], mask[prev]
                prev = m
        return out


    def edge_flips(self, tx: Text, graph: native.Graph, ons: torch.Tensor, pool: dict, m: int = FLIP_TERM,
                   screens: dict | None = None, **kw) -> list[dict]:
        """adversary_flips' omitted lever at budget m on connection subsets of one node set: graph states every
        connection among its nodes (teacher.EdgePruner.graph: native.all_edges), ons [K, E + O] (bool) the ones each
        subset keeps, in the graph's order (its parents reader by reader, then its out list). The same numbers as
        adversary_flips on the K graphs of those connections, stranded nodes dropped (strand: each subset the claim
        fidelity scores), from one plan of graph at 0/1 strengths (kept on the text across calls) and the node set's
        layout made once; a call with a subset that strands a node the whole graph keeps (no stated path left to the
        prediction) runs adversary_flips on the subsets' graphs instead."""
        base, subs = self._edge_subsets(tx, graph, ons)
        if subs is not None:  # a subset strands a node: its graph as stated
            return self.adversary_flips(tx, subs, pool, "omitted", ms=(m,), screens=screens, **kw)
        return self.adversary_flips(tx, [base] * len(ons), pool, "omitted", ms=(m,), screens=screens,
                                    strengths=ons.to(self.nat.dev, torch.float32), **kw)

    @torch.no_grad()
    def connection_effects(self, tx: Text, graph: native.Graph, rows: int = 48) -> list[dict]:
        """Per stated connection of a graph (stranded nodes dropped, as fidelity scores it), writer A to reader B (None:
        the prediction), the change in target KL (bits) of its run in C's semantics against the graph as stated when A's
        contribution leaves B's read but stays in B's RMSNorm scale ("direct"), leaves the scale but stays in the read
        ("norm"), and leaves both ("full", the connection cut): engine.Engine's ew/ewn and ow/own. "normed": B reads
        through a norm (a query, key, value or MLP input, or the prediction); an attention or MLP output reads none, so
        its "norm" is 0 and "direct" its cut. An explanation's claim that a connection reads or rescales is tested by the
        two partial cuts (the lead's decision 10-12): they need not add up to the cut."""
        nat, dev = self.nat, self.nat.dev
        g = _c_graph(graph, tx.targets, nat.C)
        edges = [(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out]
        if not edges:
            return []
        E = self._circuit_engine(tx.ids, tx.targets, [g])
        plan = E.plan(g, semantics="c")
        Ein, O = plan["E"], plan["O"]
        cases = [None] + [(k, d, n) for k in range(len(edges)) for d, n in ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0))]
        kl = []
        for s0 in range(0, len(cases), rows):
            part = cases[s0:s0 + rows]
            ew, ewn, ow, own = (torch.ones(len(part), m, device=dev) for m in (Ein, Ein, O, O))
            for i, case in enumerate(part):
                if case is not None:
                    k, d, n = case
                    if k < Ein:
                        ew[i, k], ewn[i, k] = d, n
                    else:
                        ow[i, k - Ein], own[i, k - Ein] = d, n
            kl += carry.kl(tx.ref, E.graph_log_probs(plan, None, ew, ow, ewn=ewn, own=own)).tolist()
            native.free(dev)
        out = []
        for k, (w, r) in enumerate(edges):
            kind = None if r is None else native._name_layer_kind(r[0])[1]
            out.append({"writer": w, "reader": r, "normed": r is None or kind in ("q_proj", "k_proj", "v_proj", "c_fc"),
                        **{name: kl[1 + 3 * k + i] - kl[0] for i, name in enumerate(("direct", "norm", "full"))}})
        return out

    def _edge_subsets(self, tx: Text, graph: native.Graph, ons: torch.Tensor) -> tuple:
        """(graph with its stranded nodes dropped, None), or with a subset of ons (edge_flips', in graph's order) that
        strands a node, or a graph that strands one itself (its plan's connections are fewer than ons' columns), (that,
        each subset's graph as stated with its stranded nodes dropped)."""
        base = _c_graph(graph, tx.targets, self.nat.C)  # nodes no path of graph reaches the prediction from: stranded in every subset
        edges = [(w, r) for r, ws in graph.parents.items() for w in ws] + [(w, None) for w in graph.out]
        nodes = sorted(graph.node_set())
        subs = []
        for row in ons.tolist():
            kept = [e for e, k_ in zip(edges, row) if k_]
            parents = {}
            for w, r in kept:
                if r is not None:
                    parents.setdefault(r, []).append(w)
            subs.append(strand(native.Graph(nodes, parents, [w for w, r in kept if r is None]), tx.targets, self.nat.C))
        return base, (subs if base.stranded or any(sub.stranded for sub in subs) else None)

    def changed_edges(self, tx: Text, graph: native.Graph, ons: torch.Tensor, split: str = "search", draws: int = 16) -> list[dict]:
        """changed_inputs on connection subsets of one node set (edge_flips' graph and ons: a teacher's connection
        descent asking about its prefixes): the same numbers as changed_inputs on the subsets' graphs, from one plan of
        graph at 0/1 strengths; a call with a subset that strands a node runs changed_inputs on the subsets' graphs."""
        base, subs = self._edge_subsets(tx, graph, ons)
        if subs is not None:
            return self.changed_inputs(tx, subs, split, draws)
        return self.changed_inputs(tx, [base] * len(ons), split, draws, strengths=ons.to(self.nat.dev, torch.float32))

    def across(self, tx: Text, graphs: list[native.Graph], prompts: list) -> list[dict]:
        """Per graph, per convention, the same circuit identities run on other prompts of the text: recovered pooled
        over them, sum (KL_empty - KL_C) / sum (KL_empty - KL_VPD), each prompt's bounds from VPD's own answer and the
        empty circuit on that prompt, and the mean KL_C; stated graphs run as stated (_graph_kl)."""
        nat = self.nat
        whole = [i for i, g in enumerate(graphs) if g.complete]
        stated = [i for i, g in enumerate(graphs) if not g.complete]
        sets = [self.masks(graphs[i], tx.T) for i in whole]
        empty = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        acc = {conv: [[0.0, 0.0, 0.0] for _ in graphs] for conv in CONVENTIONS}
        for x in prompts:
            x = list(x)
            ref = nat.reference([x], tx.targets)
            v = self.vpd_answer(x, tx.targets)
            extra = [empty] + ([v] if v is not None else [])
            for conv in CONVENTIONS:
                got = carry.kl(ref, self.runs(x, tx.targets, sets + extra, conv == "live")).tolist()
                e = got[len(whole)]
                vv = got[len(whole) + 1] if v is not None else None
                kl = [0.0] * len(graphs)
                for i, k in zip(whole, got):
                    kl[i] = k
                for i, k in zip(stated, self._graph_kl(x, tx.targets, ref, [graphs[i] for i in stated], conv == "live", e)):
                    kl[i] = k
                for b in range(len(graphs)):
                    a = acc[conv][b]
                    a[0] += e - kl[b]
                    a[1] += (e - vv) if vv is not None else 0.0
                    a[2] += kl[b] / len(prompts)
            native.free(nat.dev)
        out = [{} for _ in graphs]
        for conv in CONVENTIONS:
            for b in range(len(graphs)):
                num, den, mean = acc[conv][b]
                out[b].update({f"recovered_{conv}": num / den if den > 0 else None, f"kl_{conv}": mean if prompts else None})
        return out

    @torch.no_grad()
    def interchange(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """Per graph, interchange carry under primerl's rule b (module docstring) at F2's positions other than the
        targets: per position [position, movement bits, carried, B_p, the empty share], and the mean over positions
        with B_p > 0 of completeness_p."""
        nat, T = self.nat, tx.T
        if tx.ci is None:
            return [{"carry_completeness": None, "carry_positions": None} for _ in graphs]
        active = {n: tx.ci[n] > 0 for n in nat.names}
        sets = [self.masks(g, T) for g in graphs]
        rows = [[] for _ in graphs]
        for p, prompts in tx.f2:
            if p in tx.targets:
                continue
            after = torch.zeros(T, 1, dtype=torch.bool, device=nat.dev)
            after[p + 1:] = True
            at = torch.zeros(T, 1, dtype=torch.bool, device=nat.dev)
            at[p] = True
            kls, moved = np.zeros(len(graphs) + 2), 0.0
            for x in prompts:
                ci = nat.importance(x)
                tail = {n: ~active[n] & ~(ci[n] > 0) for n in nat.names}
                base = {n: at.expand_as(tail[n]) | tail[n] for n in nat.names}
                named = sets + [{n: active[n] & after for n in nat.names}, {n: torch.zeros_like(tail[n]) for n in nat.names}]
                study = carry.Carry(nat, tx.ids, tx.targets, [x])
                kls += np.asarray(study.measure([{n: base[n] | s[n] for n in nat.names} for s in named]).tolist())
                moved += carry._movement(study)
                native.free(nat.dev)
            if moved <= 0:
                continue
            share = 1 - kls / moved
            for b in range(len(graphs)):
                rows[b].append([p, moved, float(share[b]), float(share[-2]), float(share[-1])])
        out = []
        for rs in rows:
            comp = [min(1.0, c / (GOAL * bp)) for _, _, c, bp, _ in rs if bp > 0]
            out.append({"carry_completeness": _mean(comp), "carry_positions": rs})
        return out

    # ---- baselines and everything

    def baselines(self, tx: Text, sizes: list[int]) -> dict:
        """name -> Graph: "empty"; "vpd@k" for each k in sizes, VPD's first k subcomponents in its order (causal
        importance x how much each writes, the predicted position first), complete; "vpd", VPD's own answer. Without
        VPD's causal-importance network (a pod) only the empty one."""
        nat = self.nat
        out = {"empty": native.Graph()}
        if tx.vpd is None:
            return out
        order = nat._vpd_order(tx.ids, tx.targets)
        for k in sorted(set(sizes)):
            if k > 0:
                out[f"vpd@{k}"] = native.Graph(order[:k], complete=True)
        out["vpd"] = native.Graph(masks=tx.vpd)
        return out

    def score(self, tx: Text, graphs: list[native.Graph], explanations: list[str] | None = None, diagnostics: bool = False) -> list[dict]:
        """Every measure of each graph on the text, one dict per graph: the reward fields (module docstring) and,
        with diagnostics, F1 on the changed prompts ("changed_*"), F2 ("f2_*"), interchange carry ("carry_*") and the
        real model's necessity ("n_real_*", "kl_real_minus")."""
        t0 = time.perf_counter()
        nat = self.nat
        graphs = [_c_graph(g, tx.targets, nat.C) for g in graphs]  # stranded nodes dropped before every measure
        rows = self.sufficiency(tx, graphs)
        f3 = self.edges(tx, graphs)
        total = sum(nat.C.values())
        scale = self.size_scale(tx)
        for b, (g, row) in enumerate(zip(graphs, rows)):
            st, un = f3[b]["stated"], f3[b]["unstated"]
            bits = size_bits(g, nat.positions(tx.targets), total, tx.targets, active=tx.vpd)
            row.update({"nodes": g.count(), "stranded": len(g.stranded), "edges": None if g.complete else g.edges(),
                        "tok_edges": len(getattr(g, "tok_edges", [])),
                        "bits": bits, "size": size_term(bits, scale),
                        "f3": f3[b]["f3"], "f3_stated": _mean(st) if st else None, "f3_unstated": _mean(un) if un else None,
                        "f3_cuts": {"stated": st, "unstated": un}})
            for name, fn in self.hooks.items():
                row[name] = None if fn is None else fn(tx.task, g, (explanations or [""] * len(graphs))[b])
        if diagnostics:
            for key, extra in (("changed", self.across(tx, graphs, tx.prompts)),
                               ("f2", self.across(tx, graphs, [x for _, xs in tx.f2 for x in xs]))):
                for row, d in zip(rows, extra):
                    row.update({f"{key}_{k}": v for k, v in d.items()})
            for row, d in zip(rows, self.interchange(tx, graphs)):
                row.update(d)
            for row, d in zip(rows, self.real_removal(tx, graphs)):
                row.update(d)
        seconds = time.perf_counter() - t0 + tx.seconds
        for row in rows:
            row["seconds_text"] = seconds
        return rows


# ---- answer sets and the table


def answer_files(directory: Path, stem: str) -> list[Path]:
    """A text's answers in an answer set's directory, first sample first: <stem>.py or <stem>.json holding a circuit
    ("nodes"; a search's directory also keeps each answer's scores as <stem>.json), then <stem>.<k>.py or .json."""
    def one(base: str) -> Path | None:
        py, js = directory / f"{base}.py", directory / f"{base}.json"
        if py.exists():
            return py
        if js.exists() and "nodes" in json.loads(js.read_text()):
            return js
        return None

    first = one(stem)
    ks = sorted({int(m[1]) for f in directory.glob(f"{stem}.*.*") if (m := re.fullmatch(re.escape(stem) + r"\.(\d+)\.(py|json)", f.name))})
    rest = [one(f"{stem}.{k}") for k in ks]
    return ([first] if first else []) + [f for f in rest if f is not None]


def load_answer(path: Path, task: dict, targets: list[int], nat: native.Native) -> tuple[native.Graph, dict, str]:
    """(graph, defects, explanation) of an answer file: circuit JSON (.json) or a graph program (.py, traced by mech)."""
    T = len(task["prompts"][0]["token_ids"])
    if path.suffix == ".json":
        answer = json.loads(path.read_text())
        g, defects = circuit(answer, targets, nat.C, T)
        return g, defects, answer.get("explanation", "")
    ir = mech.trace_inline(path.read_text(), "vpd4l", task)
    if not ir["valid"]:
        return native.Graph(), {"invalid_program": True}, ""
    g = strand(nat.from_ir(ir), targets, nat.C)
    return g, {"stranded_nodes": len(g.stranded)}, ir["graph"].get("explanation") or ""


COLUMNS = [("nodes", "nodes", "{:.0f}"), ("edges", "edges", "{:.0f}"), ("bits", "size bits", "{:.0f}"),
           ("kl_live", "KL live", "{:.3g}"), ("f1_live", "F1 live", "{:.3f}"), ("n_live", "N live", "{:.3f}"),
           ("kl_off", "KL off", "{:.3g}"), ("f1_off", "F1 off", "{:.3f}"), ("n_off", "N off", "{:.3f}"),
           ("recovered_live", "lin recov live", "{:.3f}"), ("necessity_live", "lin N live", "{:.3f}"),
           ("kl_vpd_live", "KL VPD live", "{:.3g}"), ("f3", "F3", "{:.3f}"),
           ("f3_stated", "F3 stated", "{:.3g}"), ("f3_unstated", "F3 unstated", "{:.3g}"),
           ("E1", "E1", "{:.3g}"), ("E2", "E2", "{:.3f}"),
           ("changed_recovered_live", "chg recov live", "{:.3f}"), ("f2_recovered_live", "F2 recov live", "{:.3f}"),
           ("carry_completeness", "carry compl", "{:.3f}"), ("seconds_text", "s/text", "{:.1f}")]


def table(results: dict) -> str:
    """One table of an answer set and its baselines on the same texts: per answer the texts, coverage (the share of
    texts with a valid answer; a missing or invalid answer counts against it) and each measure's mean over the valid
    answers. results: {"set": name, "rows": {answer name -> [row per text, {"valid": False} for an invalid one]}}."""
    names = list(results["rows"])
    shown = [c for c in COLUMNS if any(r.get(c[0]) is not None for a in names for r in results["rows"][a])]
    head = ["answer", "texts", "coverage"] + [c[1] for c in shown]
    lines = [head]
    for a in names:
        rows = results["rows"][a]
        valid = [r for r in rows if r.get("valid", True)]
        line = [a, str(len(rows)), f"{len(valid) / len(rows):.2f}" if rows else "-"]
        for key, _, fmt in shown:
            vals = [r[key] for r in valid if r.get(key) is not None]
            line.append(fmt.format(float(np.mean(vals))) if vals else "-")
        lines.append(line)
    widths = [max(len(x[i]) for x in lines) for i in range(len(head))]
    return f"answer set {results['set']}\n" + "\n".join("  ".join(x[i].rjust(widths[i]) for i in range(len(head))) for x in lines)


def score_sets(sets: dict, split: str, n: int, ge: GoalEval | None = None, baselines: bool = True, diagnostics: bool = False,
               log=print) -> list[dict]:
    """Every answer set (name -> directory) on the same texts: the split's first n that any set answers. A set without
    a valid answer on a text gets an invalid row there (counted in its coverage, never skipped). The headline row of a
    set is each text's first sample; with several samples, "NAME best of N (upper bound)" is each text's sample with
    the highest f1_live. Baselines at each valid answer's size -> one results dict per set."""
    ge = ge or GoalEval()
    nat = ge.nat
    results = {s: {"set": s, "rows": {s: [], "empty": [], "vpd@matched": [], "vpd": []}} for s in sets}
    best = {s: [] for s in sets}
    for p in native.tasks(split)[:n]:
        found = {s: answer_files(Path(d), p.stem) for s, d in sets.items()}
        if not any(found.values()):
            continue
        task = json.loads(p.read_text())
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), task)
        graphs, expl, meta = [], [], []
        for s, files in found.items():
            for k, f in enumerate(files):
                g, d, e = load_answer(f, task, targets, nat)
                valid = not d.get("invalid_program") and not d.get("invalid_nodes") and not d.get("invalid_edges")
                meta.append((s, k, valid, d))
                graphs.append(g)
                expl.append(e)
        base = ge.baselines(tx, [g.count() for g, m in zip(graphs, meta) if m[2]]) if baselines else {}
        names = list(base)
        scored = [g for g, m in zip(graphs, meta) if m[2]]
        rows = ge.score(tx, scored + [base[k] for k in names], [e for e, m in zip(expl, meta) if m[2]] + [""] * len(names), diagnostics)
        it = iter(rows[:len(scored)])
        by = dict(zip(names, rows[len(scored):]))
        samples = {s: [] for s in sets}
        for (s, k, valid, d), g in zip(meta, graphs):
            row = {**next(it), **d, "valid": True} if valid else {**d, "valid": False}
            row.update({"task": p.stem, "sample": k})
            samples[s].append((row, g))
        for s in sets:
            first = samples[s][0] if samples[s] else None
            results[s]["rows"][s].append(first[0] if first else {"task": p.stem, "valid": False, "missing": True})
            ok = [r for r, _ in samples[s] if r["valid"]]
            if len(samples[s]) > 1:
                best[s].append(max(ok, key=lambda r: r.get("f1_live") if r.get("f1_live") is not None else -math.inf)
                               if ok else {"task": p.stem, "valid": False})
            for k in ("empty", "vpd"):
                if k in by:
                    results[s]["rows"][k].append({**by[k], "task": p.stem})
            if first and first[0]["valid"] and f"vpd@{first[1].count()}" in by:
                results[s]["rows"]["vpd@matched"].append({**by[f"vpd@{first[1].count()}"], "task": p.stem})
        done = [f"{s} F1 {r['f1_live']:.2f}/{r['f1_off']:.2f}" for s in sets
                for r in [results[s]["rows"][s][-1]] if r.get("valid") and r.get("f1_live") is not None and r.get("f1_off") is not None]
        log(f"{p.stem}: {', '.join(done) or 'no valid answer'}" + (f", {rows[0]['seconds_text']:.1f} s" if rows else ""))
    for s in sets:
        if best[s]:
            results[s]["rows"][f"{s} best of N (upper bound)"] = best[s]
    return list(results.values())


def tails(split: str, n: int, teachers: dict, top: int = 64, ps=(0.9, 0.99), ge: GoalEval | None = None, log=print) -> str:
    """The tail-shape share of each circuit's KL (tail_split, remainders live) on the split's first n texts: the empty
    circuit, VPD's answer, VPD's first `top` (its order), and each teacher directory's circuit (<task>.json); per kind
    the mean KL and, per p, the pooled share sum(tail) / sum(KL) and the mean per-text share."""
    ge = ge or GoalEval()
    nat = ge.nat
    acc = {}
    for p in native.tasks(split)[:n]:
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
        if tx.vpd is None:
            raise RuntimeError("tails needs VPD's causal-importance network")
        empty = {m: torch.zeros(tx.T, nat.C[m], dtype=torch.bool, device=nat.dev) for m in nat.names}
        kinds = {"empty": empty, "vpd": tx.vpd, f"vpd top {top}": ge.masks(native.Graph(nat._vpd_order(ids, targets)[:top], complete=True), tx.T)}
        for name, d in teachers.items():
            f = Path(d) / f"{p.stem}.json"
            if f.exists():
                g, _ = circuit(json.loads(f.read_text()), targets, nat.C, tx.T)
                kinds[name] = ge.masks(g, tx.T)
        lq = ge.runs(ids, targets, list(kinds.values()), True)
        full = carry.kl(tx.ref, lq).tolist()
        split_ = {q: tail_split(tx.ref, lq, q)[1].tolist() for q in ps}
        for i, name in enumerate(kinds):
            a = acc.setdefault(name, {"kl": [], **{q: [] for q in ps}})
            a["kl"].append(full[i])
            for q in ps:
                a[q].append(split_[q][i])
        log(f"{p.stem}: " + ", ".join(f"{k} KL {full[i]:.3g} tail@{ps[0]} {split_[ps[0]][i]:.3g}" for i, k in enumerate(kinds)))
    lines = [["circuit", "texts", "mean KL"] + [h for q in ps for h in (f"tail share p{q} pooled", f"p{q} per text")]]
    for name, a in acc.items():
        kl = np.asarray(a["kl"])
        row = [name, str(len(kl)), f"{kl.mean():.3g}"]
        for q in ps:
            t = np.asarray(a[q])
            row += [f"{t.sum() / kl.sum():.3f}" if kl.sum() > 0 else "-", f"{np.mean(t / np.maximum(kl, 1e-12)):.3f}"]
        lines.append(row)
    w = [max(len(x[i]) for x in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(x[i].rjust(w[i]) for i in range(len(x))) for x in lines)


# ---- the adversary's pattern pool (C's M)


def build_patterns(texts: list[Path], out: Path, nat: native.Native | None = None) -> dict:
    """The pool M of the paired adversary: every position of the texts, the set of subcomponents VPD's causal
    importance switches on there (CI > 0), one pattern each. Saved as {"names", "sources": [(task, position)],
    "site": int16 index into names, "c": int32, "offsets": int64 [patterns + 1]}, a pattern's (site, c) pairs between
    consecutive offsets."""
    nat = nat or native.Native()
    sites, cols, counts, sources = [], [], [], []
    with torch.no_grad():
        for p in texts:
            ids, _ = native.text(p)
            ci = nat.importance(ids)
            ts, ss, cs = [], [], []
            for i, n in enumerate(nat.names):
                t, c = (ci[n] > 0).nonzero(as_tuple=True)
                ts.append(t.cpu())
                ss.append(torch.full((len(t),), i, dtype=torch.int16))
                cs.append(c.cpu().to(torch.int32))
            t, site, c = torch.cat(ts), torch.cat(ss), torch.cat(cs)
            order = torch.argsort(t * len(nat.names) + site.long(), stable=True)  # by position, then site
            sites.append(site[order])
            cols.append(c[order])
            counts.append(torch.bincount(t, minlength=len(ids)))
            sources += [(p.stem, i) for i in range(len(ids))]
            native.free(nat.dev)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cat(counts).cumsum(0)])
    pool = {"names": list(nat.names), "sources": sources, "site": torch.cat(sites), "c": torch.cat(cols), "offsets": offsets}
    torch.save(pool, out)
    return pool


def pattern(pool: dict, k: int) -> dict:
    """Pattern k of a pool as matrix -> LongTensor of subcomponent indices."""
    a, b = int(pool["offsets"][k]), int(pool["offsets"][k + 1])
    site, c = pool["site"][a:b].long(), pool["c"][a:b].long()
    return {n: c[site == i] for i, n in enumerate(pool["names"]) if bool((site == i).any())}


def fidelity_report(split: str, n: int, sets: dict, pool: Path, out: Path, draws: int = DRAWS, vpd: Path | None = None,
                    offset: int = 0, stride: int = 1, fresh: Path | None = None, interchange: int = 16, curve=ADV_CURVE,
                    log=print, changed: int = CHANGED_DRAWS, only: tuple = (), mode: str = "full", draw_split: str | None = None) -> list[dict]:
    """C on the split's first n texts for answer sets (sets: name -> a directory of <task>.json circuits, run as stated)
    and VPD's first k at each circuit's size (complete graphs), with the evaluation-only families (fresh: the pattern
    lever over a pool no search drew from; interchange draws; changed texts (changed_inputs' eval split); N_real); one
    row per circuit appended to OUT (texts already there are skipped); only: some of ("changed", "interchange",
    "donors", "track_pp", "interchange_pp"), those families alone on the eval split (task, kind, nodes and their fields: families added to rows
    already made, matched by task and kind; donors' far texts the eval split's, DONOR_POOLS). curve: the step adversary's budgets reported. vpd: VPD's answers and orders saved by
    teacher.py --save-vpd (where the causal-importance network is not). Locality from a set's summary*.jsonl."""
    ge = GoalEval()
    nat = ge.nat
    M = torch.load(pool)
    F = torch.load(fresh) if fresh else None
    saved = torch.load(vpd) if vpd else {}
    local = {}
    for d in sets.values():
        for f in sorted(Path(d).glob("summary*.jsonl")):
            for x in f.read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    if r.get("local") is not None:
                        local[r["task"]] = r["local"]
    done = {json.loads(x)["task"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    far = ge.far_pool("eval") if "donors" in only else []
    rows = []
    for p in native.tasks(split)[:n][offset::stride]:
        have = {name: Path(d) / f"{p.stem}.json" for name, d in sets.items() if (Path(d) / f"{p.stem}.json").exists()}
        if p.stem in done or not have:
            continue
        t0 = time.perf_counter()
        task = json.loads(p.read_text())
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), task)
        if p.stem in saved:
            tx.vpd = {m: torch.zeros(tx.T, nat.C[m], dtype=torch.bool, device=nat.dev) for m in nat.names}
            for m, (t, c) in saved[p.stem]["answer"].items():
                tx.vpd[m][t.long().to(nat.dev), c.long().to(nat.dev)] = True
        graphs, kinds = [], []
        for name, path in have.items():
            g, _ = circuit(json.loads(path.read_text()), targets, nat.C, tx.T)  # stranded nodes dropped (strand)
            graphs.append(g)
            kinds.append(name)
        order = saved[p.stem]["order"] if p.stem in saved else nat._vpd_order(ids, targets)
        for k in sorted({g.count() for g in graphs}):
            graphs.append(native.Graph(order[:k], complete=True))
            kinds.append(f"VPD top {k}")
        if only:
            e = ge._bounds(tx, "live")[0]
            ch = ge.changed_inputs(tx, graphs, "eval", changed) if "changed" in only else [{} for _ in graphs]
            it = ge._interchange_family(tx, graphs, interchange, split="eval") if "interchange" in only else [{} for _ in graphs]
            dn = ge.donors(tx, graphs, far=ge.far_donors(tx, far)) if "donors" in only else [{} for _ in graphs]
            tp = ge.track_pp(tx, graphs, "eval", changed) if "track_pp" in only else [{} for _ in graphs]
            if "pattern" in only:  # the true-pattern family, merged into the same rows
                tp = [dict(a, **c) for a, c in zip(tp, ge.pattern_family(tx, [_c_graph(g, targets, nat.C) for g in graphs]))]
            ip = ge.interchange_pp(tx, graphs, "eval", interchange) if "interchange_pp" in only else [{} for _ in graphs]
            scale = ge.size_scale(tx)
            with open(out, "a") as fh:
                for kind, g, c, i, d, a, b in zip(kinds, graphs, ch, it, dn, tp, ip):
                    row = {"task": p.stem, "kind": kind, "nodes": g.count(),
                           "size_bits": size_bits(g, nat.positions(targets), sum(nat.C.values()), targets, active=tx.vpd),
                           "size_scale": scale,  # (informational: VPD's nodes alone, no connections)
                           **{k: c[k] for k in ("s_changed", "s_track", "context_share", "prompts_changed", "prompts_track") if k in c}}
                    if i:
                        row["s_interchange"] = _fid(e, i["kl_interchange_c"], i["kl_interchange_vpd"])
                    row.update(d)
                    row.update({k: v for k, v in a.items() if k != "draws_track_pp"})
                    row.update({k: v for k, v in b.items() if k != "draws_interchange_pp"})
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
            if log:
                log(f"{p.stem}: {len(graphs)} circuits, {time.perf_counter() - t0:.0f} s")
            continue
        fid = ge.fidelity(tx, graphs, M, draws=draws, curve=curve, fresh=F, interchange=interchange, real=True, reads=True,
                          changed=changed, mode=mode, split=draw_split)
        e, v = ge._bounds(tx, "live")
        ceiling = ge.real_removal(tx, [native.Graph(masks=tx.vpd)])[0]["n_real_live"]  # VPD's whole answer removed
        cancel = ge.cancellation(tx)
        scale = ge.size_scale(tx)
        with open(out, "a") as fh:
            for kind, g, f in zip(kinds, graphs, fid):
                bits = size_bits(g, nat.positions(targets), sum(nat.C.values()), targets, active=tx.vpd)
                row = {"task": p.stem, "local": local.get(p.stem), "kind": kind, "nodes": g.count(),
                       "positions": len({t for _, t, _ in g.node_set()}), "stranded": len(getattr(g, "stranded", [])),
                       "edges": None if g.complete else g.edges(), "size_bits": bits, "size_scale": scale, "kl_empty": e, "kl_vpd": v,
                       "n_real_ceiling": ceiling, "cancel_cosine": cancel["cosine"], **f}
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
        if log:
            log(f"{p.stem}: {len(graphs)} circuits, {time.perf_counter() - t0:.0f} s")
    return rows


def _curve_value(d: dict, k):
    """A curve dict's value at k (JSON keeps its keys as strings)."""
    return d.get(str(k), d.get(k)) if d else None


def fidelity_table(rows: list[dict]) -> str:
    """Means by locality group and circuit kind (each answer set; VPD top k at their sizes): tests the searches never
    optimized (flips-to-break, the omitted lever's flips, the node and connection families, interchange, the fresh
    pool's pattern lever, N_real and its flip adversary, C, the reads' response error, the remainder leak), the step
    adversary's curve, and in-sample (labelled) F1, N and size. Beside N_real, its ceiling (VPD's whole answer removed
    from the real model) and the cosine between the actives' and the inactives' sums at the target (cancellation inside
    VPD's description caps N_real for any circuit: ~0.73 and -0.57 at the median on the 20 held-out texts). "trivial":
    texts whose empty circuit is within 2x of VPD's own KL."""
    def group(r):
        if r["kl_vpd"] is not None and r["kl_empty"] < 2 * r["kl_vpd"]:
            return "trivial"
        return "local" if r["local"] else "non-local"

    def kind(r):
        return "VPD top k" if r["kind"].startswith("VPD top") else r["kind"]

    by = {}
    for r in rows:
        by.setdefault((group(r), kind(r)), []).append(r)
    kinds = sorted({kind(r) for r in rows}, key=lambda k: (k == "VPD top k", k.removeprefix("VPD@"), k.startswith("VPD@")))  # "VPD@<kind>" under its kind
    flips = sorted({int(m) for r in rows for m in r.get("f1_flips_omitted", {})})
    fresh = sorted({int(m) for r in rows for m in r.get("f1_flips_patterns_fresh", {})})
    steps = sorted({int(m) for r in rows for m in r.get("f1_adv_curve", {})})

    def block(title, cols):
        head = ["texts", "kind", "n", "nodes", "positions"] + [c for c, _ in cols]
        lines = [head]
        for g in ("local", "non-local", "trivial"):
            for kd in kinds:
                rs = by.get((g, kd), [])
                if not rs:
                    continue
                cells = [g, kd, str(len(rs)), f"{np.mean([r['nodes'] for r in rs]):.0f}",
                         f"{np.mean([r.get('positions', 0) for r in rs]):.0f}"]
                for _, f in cols:
                    v = [f(r) for r in rs]
                    v = [x for x in v if x is not None]
                    cells.append(f"{float(np.mean(v)):.3f}" if v else "-")
                lines.append(cells)
        w = [max(len(x[i]) for x in lines) for i in range(len(head))]
        return title + "\n" + "\n".join("  ".join(x[i].rjust(w[i]) for i in range(len(head))) for x in lines)

    def breaks(r):  # flips-to-break, a circuit that holds to the largest budget counted at twice it
        return math.log2(r["flips_to_break"]) if r.get("flips_to_break") else (math.log2(2 * max(flips)) if flips else None)

    def halves(r):
        f = r.get("flips_to_half")
        return -1.0 if f == 0 else (math.log2(f) if f else (math.log2(2 * max(flips)) if flips else None))

    held = [("log2 flips-to-0.5 (none: -1)", halves), ("log2 flips-to-break", breaks)] + [(f"flip{m}", lambda r, m=m: _curve_value(r.get("f1_flips_c") or r.get("f1_flips_omitted"), m)) for m in flips]
    held += [("nodes fam", lambda r: r.get("s_nodes")), ("edges fam", lambda r: r.get("s_edges")),
             ("interchange", lambda r: r.get("s_interchange"))]
    held += [(f"fresh pat{m}", lambda r, m=m: _curve_value(r.get("f1_flips_patterns_fresh"), m)) for m in fresh]
    held += [("N_real", lambda r: r.get("n_real")), ("N_real ceiling", lambda r: r.get("n_real_ceiling")),
             ("act/inact cos", lambda r: r.get("cancel_cosine")), ("C", lambda r: r.get("c")),
             ("donor near", lambda r: r.get("s_donor_near")), ("donor near worst", lambda r: r.get("s_donor_near_worst")),
             ("donor near p90", lambda r: r.get("s_donor_near_p90")),
             ("donor far", lambda r: r.get("s_donor_far")), ("donor far p90", lambda r: r.get("s_donor_far_p90")),
             ("donor far worst", lambda r: r.get("s_donor_far_worst")),
             ("tracks changes", lambda r: r.get("s_track")), ("changed inputs", lambda r: r.get("s_changed")),
             ("context share", lambda r: r.get("context_share"))]
    nflips = sorted({int(m) for r in rows for m in r.get("n_real_flips", {}) if int(m) > 0})
    held += [(f"N_real flip{m}", lambda r, m=m: _curve_value(r.get("n_real_flips"), m)) for m in nflips]
    held += [(f"response err {f}", lambda r, f=f: r.get(f"dreads_{f}")) for f in ("nodes", "edges", "interchange")]
    held += [("leak (no edges)", lambda r: r.get("leak"))]
    stepcols = [(f"adv@{m}", lambda r, m=m: _curve_value(r.get("f1_adv_curve"), m)) for m in steps]
    sample = [("stranded nodes", lambda r: r.get("stranded")), ("F1", lambda r: r["s_point"]), ("F1 every connection", lambda r: r.get("s_wiring")),
              ("reliance on cuts", lambda r: r.get("reliance_on_cuts")),
              ("N", lambda r: r["n_point"]), ("size bits", lambda r: r.get("size_bits", r.get("size")))]
    return "\n\n".join([block("Tests the searches never optimized (VPD top k: VPD's own order)", held),
                          block("The step adversary (VPD's published budget, steps)", stepcols),
                          block("In-sample (the searches optimized F1 and N on these texts)", sample)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--set", action="append", required=True, help="NAME=DIR: an answer set, each text's answer and samples")
    s.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--out", type=Path)
    s.add_argument("--no-baselines", action="store_true")
    s.add_argument("--diagnostics", action="store_true", help="F1 on changed prompts, F2 and interchange carry too")
    t = sub.add_parser("tails", help="the tail-shape share of each circuit's KL")
    t.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    t.add_argument("--n", type=int, default=20)
    t.add_argument("--teacher", action="append", default=[], help="NAME=DIR of teacher circuits <task>.json")
    t.add_argument("--top", type=int, default=64)
    f = sub.add_parser("fidelity", help="C on a teacher's circuits and VPD's first k")
    f.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    f.add_argument("--n", type=int, default=20)
    f.add_argument("--set", action="append", required=True, help="NAME=DIR: an answer set, DIR/<task>.json circuits")
    f.add_argument("--pool", type=Path, required=True, help="the adversary's pattern pool (goal_eval.py patterns)")
    f.add_argument("--fresh", type=Path, help="a pattern pool no search drew from (evaluation only)")
    f.add_argument("--interchange", type=int, default=16)
    f.add_argument("--changed", type=int, default=16, help="changed texts per circuit (changed_inputs' eval split)")
    f.add_argument("--only", default="", help="changed,interchange,donors,track_pp,interchange_pp,pattern: those families alone, eval split (rows to merge by task and kind)")
    f.add_argument("--curve", default=",".join(str(x) for x in ADV_CURVE), help="the step adversary's budgets, e.g. 0,4,20,40,80,160,320")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--draws", type=int, default=DRAWS)
    f.add_argument("--vpd", type=Path, help="VPD's answers and orders saved by teacher.py --save-vpd")
    f.add_argument("--offset", type=int, default=0)
    f.add_argument("--stride", type=int, default=1)
    f.add_argument("--mode", choices=("full", "reward"), default="full",
                   help="GoalEval.fidelity's mode: full, or the reward's (its family alone, rl_terms and c_rl: teacher rows for the reward)")
    f.add_argument("--draw-split", choices=("search", "eval"), help="the families' draws (default: the mode's, search for reward, eval for full)")
    q = sub.add_parser("patterns", help="the adversary's pattern pool from a range of held-out texts")
    q.add_argument("--first", type=int, required=True, help="first text number, e.g. 9100")
    q.add_argument("--last", type=int, required=True, help="last text number, e.g. 9149")
    q.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "fidelity":
        fidelity_report(args.split, args.n, dict(x.split("=", 1) for x in args.set), args.pool, args.out, args.draws, args.vpd,
                        args.offset, args.stride, args.fresh, args.interchange, tuple(int(x) for x in args.curve.split(",")),
                        changed=args.changed, only=tuple(x for x in args.only.split(",") if x), mode=args.mode, draw_split=args.draw_split)
        if args.only:
            return
        print(fidelity_table([json.loads(x) for x in args.out.read_text().splitlines() if x.strip()]), flush=True)
        return
    if args.cmd == "patterns":
        texts = [native.TEXTS / "vpd4l" / f"text{i}.json" for i in range(args.first, args.last + 1)]
        pool = build_patterns(texts, args.out)
        print(f"{len(pool['sources'])} patterns, {len(pool['c'])} subcomponents, mean {len(pool['c']) / len(pool['sources']):.0f} each")
        return
    if args.cmd == "tails":
        print(tails(args.split, args.n, dict(x.split("=", 1) for x in args.teacher), args.top), flush=True)
        return
    sets = dict(x.split("=", 1) for x in args.set)
    results = score_sets(sets, args.split, args.n, baselines=not args.no_baselines, diagnostics=args.diagnostics)
    for r in results:
        print(table(r), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
