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
SHARE_FLOOR = 0.01  # F2's positions: those moving the prediction by about this share of the text's total (f2_cut)
F2_BAND = 2.0  # f2_cut's window around SHARE_FLOOR: the cut falls between shares within a factor F2_BAND of it
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
INTERCHANGE_DRAWS = 32  # C's interchange draws per circuit
CHANGED_DRAWS = 32  # C's changed texts per circuit (32, the lead's 10-12: at 16 the changed-input and donor families
# had the largest eps of C's members on the held-out influential and VPD sets, medians 0.1-0.3; 32 costs seconds a text)
INTERCHANGE_SPLITS = {"search": (1, (0, 2), False), "eval": (2, (1,), True)}  # interchange draws: the text's seed
# offset, carry.alternatives' ranks, and whether samples from the model's distribution (no split's ranks) fill the budget
FRESH_SOURCES = 64  # sampled source tokens kept per position (the first ones fill a budget)
SEMANTICS = {"n": ("nodes", "omitted"), "c": ()}
# GoalEval(semantics=, default "n", the reward's: the lead's 10-13, run 2): the C families whose circuit and VPD runs
# hold the query and key subcomponents outside the circuit at their activations on the text and every RMSNorm, the
# final one included, at the model's scale on it (engine at_x and norms, monitorbench's b6a3df3), each family on its
# own bounds; the real model's runs keep their own. "n": the node and omitted families, whose scores pool the circuit's
# prediction of the model's response against the model's own (W), the rest as stated: in C's semantics a thinned stream
# renormalized itself and inflated every removal effect (s_nodes 0.36 on the dev starts, 16 of 20 bars set by it; 0.90
# under "n"), and toys, controls and every other family stay C's. "c": every family as stated (the engine checks against
# graph_ref, generality's comparisons). Holding point, pattern, edges or flips at x (2750fd8's "x", bd68101's "h")
# collapsed them to 0.2-0.4 on 6 of the 20 dev starts: they score absolute KLs, which carry the circuit's standalone
# error.
BLOCK = -1  # a block node's index: (site, position, BLOCK), the site's whole output there (every subcomponent and the
# remainder), ["block", site, position] in an answer (the lead's 10-12: a token's identity lives in thousands of parts)
DONORS_NEAR = 32  # donor resampling's changed-input donors per text
DONOR_ROWS = 16  # donor resampling's rows per run (donors x circuits and VPD's answer)
DONORS_FAR = 32  # donor resampling's other texts per text
NEC_CONTROLS = 32  # N's matched control texts per text (necessity_spec: the split's far pool, DONOR_POOLS; at 16 the
# controls' spread, ~5 bits, gave eps_N a median 0.14 on the held-out influential sets)
NEC_CONTROLS_LOO = NEC_CONTROLS  # line_loo's and phi_variants' N controls: the answer's own (its reward row's N, no rerun;
# 8 when N per deletion was 55% of line_loo, before it ran on one engine: 2bc1ddd)
REMOVAL_ROWS = {"cuda": 32, "other": 16}  # real-model removal sets per batch (_removed_kl), by device: a removal batch's
# masks are [rows, T, C] floats at every site
NEC_MIN_LEN = 64  # a control text's least length (end-aligned: a circuit's parts further back are not removed there)
DONOR_VOCAB = 50277  # DONOR_POOLS' texts are vpd4l's token ids, in its tokenizer's 50277-id vocabulary (far_pool)
DONOR_POOLS = {"search": ("train", range(6000, 6400), True),  # the far donors' texts per split by text number, frozen
               "eval": ("donor", range(9150, 9200), False),  # (the lead's 10-13): the reward's text6000-6399, which a pod
               "repick": ("train", range(6400, 7000), True)}
# missing any refuses (the first 400 train files present before, so a pod holding fewer texts drew other donors: rewards
# up to 0.13 apart); the evaluations' text9150-9199 partition, its missing files left out; "repick", no split's: the
# teachers' held-out input scope (experiments/oracle/scope_repick.py, the lead's 10-14), training texts outside both


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
    the text, edges the model does not have (mech.connects) or naming a place outside the node list (_place: no
    negative places), and variables naming a part outside the model are dropped and counted."""
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
            w = nodes[_place(up, len(nodes))]
            r = None if down == "out" else nodes[_place(down, len(nodes))]
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


def _place(i, n: int) -> int:
    """A node's place in an answer's node list: an int in [0, n) (a negative place, which Python would index from the
    end, a bool or a float is a defect: the external audit, 10-13)."""
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < n:
        raise ValueError(f"node place {i!r} outside 0..{n - 1}")
    return int(i)


def _pair_entry(e: list, nodes: list, targets: list[int]):
    """((layer, position), (query or None, key or None)) of a JSON pair edge ["pair", query, key, layer, position]
    (circuit's), None when it is not one the model has."""
    try:
        _, qi, ki, layer, t = e
        layer, t = int(layer), int(t)
        q = None if qi == "rest" else nodes[_place(qi, len(nodes))]
        k = None if ki == "rest" else nodes[_place(ki, len(nodes))]
    except (IndexError, TypeError, ValueError):
        return None
    if not 0 <= t <= max(targets) or (qi != "rest" and q is None) or (ki != "rest" and k is None):
        return None
    if q is not None and (q[0] == "tok" or native._name_layer_kind(q[0]) != (layer, "q_proj") or q[1] != t):
        return None
    if k is not None and (k[0] == "tok" or native._name_layer_kind(k[0]) != (layer, "k_proj") or k[1] > t):
        return None
    return (layer, t), (q, k)


def block_contents(n: str, t: int, C: dict, active: dict | None = None) -> list[int]:
    """A block node's subcomponents (site n at position t): those VPD's answer has on there (active: {site: bool [T,
    C]}, the text's), every one of the site without active. Scored at its active contents, a block holds what size_bits
    prices it at; its inactive contents rode free inside it (the block's name about 13 bits, where backup measured about
    1.5k bits for 6 inactive subcomponents listed with their connections) and could carry compensators (backup's 10-13
    pricing hole; the lead's rule: no block where listed inactive subcomponents would be needed)."""
    return list(range(C[n])) if active is None else active[n][t].nonzero().flatten().tolist()


def expand_blocks(g: native.Graph, C: dict, active: dict | None = None) -> native.Graph:
    """A graph with its block nodes (BLOCK) replaced by their subcomponents (block_contents: with active, the text's
    VPD answer, those on at the position; else every one of the site), each in the block's connections and pairs:
    C's semantics already run the remainders live, so this is the block's claim there (the families on the engine
    take subcomponents). A graph without blocks is returned as it is."""
    if g.masks is not None or not any(nd[2] == BLOCK for nd in g.node_set()):
        return g

    def ex(nd):
        return [(nd[0], nd[1], c) for c in block_contents(nd[0], nd[1], C, active)] if nd is not None and nd[2] == BLOCK else [nd]

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


def _c_graph(g: native.Graph, targets: list[int], C: dict, active: dict | None = None) -> native.Graph:
    """A graph as C's families run it: stranded nodes dropped (strand), block nodes as their subcomponents
    (expand_blocks; with active, the text's VPD answer, its active contents)."""
    return expand_blocks(strand(g, targets, C), C, active)


def to_json(g: native.Graph, explanation: str = "") -> dict:
    """A Graph as an answer in JSON (circuit's format; a complete graph, set- or mask-backed, as its node set,
    "complete": true; the external audit 10-13: a mask-backed one fell through to the stated form, empty)."""
    if g.complete:
        return {"nodes": [["block", n, t] if c == BLOCK else [n, c, t] for n, t, c in sorted(g.node_set(), key=lambda nd: (nd[1], nd[0], nd[2]))],
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


def environment() -> str:
    """The Python environment a score was computed in, named on every reward row and scorer score beside its semantics,
    source and donor pool (the lead's 10-13): torch with its CUDA build, numpy, and the float32 matmul settings.
    Between torch 2.4.1 and 2.10.0 the reward moved by at most 2e-4 (A6000, same answers), so an RL run pins it
    (tinker_train) rather than relying on it."""
    return (f"torch {torch.__version__} cuda {torch.version.cuda} numpy {np.__version__} "
            f"matmul {torch.get_float32_matmul_precision()} tf32 {int(torch.backends.cuda.matmul.allow_tf32)}")


def size_term(bits: float, scale: float) -> float:
    """The size term: an answer's description bits in orders of magnitude of the scale (size_scale, the bits of naming
    VPD's whole answer), log2(1 + bits) / log2(1 + scale), at most 1 (the lead's decision 10-12; components.size_term)."""
    return min(1.0, math.log2(1.0 + bits) / math.log2(1.0 + scale)) if scale and scale > 0 else 0.0


def size_bits(g: native.Graph, positions: int, subcomponents: int, targets: list[int], active: dict | None = None,
              _contents: bool = True) -> float:
    """An answer's description bits, raw (the lead's 10-12: no scale): native.Graph.bits with the answer's embedding
    nodes (each a choice among the positions, every edge's ends indexed among all the listed nodes, an embedding edge
    priced like any other) and the stated query-key pairs (_pair_bits). A block node (BLOCK) costs what it executes
    (the external audit, 10-13): the graph is priced as its explicit expansion, every block replaced by its active
    contents (expand_blocks with active, {site: bool [T, C]}, the text's VPD answer; what the families run), its
    connections, pairs and token edges with them, so a block and the subcomponents it holds always cost the same; a
    stranded block its active contents' subcomponent bits. A graph with blocks needs active."""
    if _contents and any(nd[2] == BLOCK for nd in g.node_set()):
        if active is None:
            raise ValueError("a graph with block nodes is priced at its blocks' active contents: pass active (the text's VPD answer)")
        return size_bits(expand_blocks(g, {}, active), positions, subcomponents, targets, active)
    extra = getattr(g, "stranded_bits", 0.0) + _pair_bits(g)  # stranded nodes (strand): dropped, their bits still charged
    if _contents:
        held = [nd for nd in getattr(g, "stranded", []) if nd[2] == BLOCK]
        if held and active is None:
            raise ValueError("a graph with block nodes is priced at its blocks' active contents: pass active (the text's VPD answer)")
        extra += sum(int(active[n][t].sum()) for n, t, _ in held) * math.log2(positions * subcomponents)
    if getattr(g, "tok_out", None):  # an embedding into the prediction priced like an edge into it: its writer named
        extra += len(g.tok_out) * math.log2(g.count() + len(set(getattr(g, "tok_positions", [])) | set(g.tok_out)))
    pos = sorted(set(getattr(g, "tok_positions", [])) | {p for p, _ in getattr(g, "tok_edges", [])})
    n, k = g.count(), len(pos)
    if g.complete:
        inner, out = native.implied_edges(g.node_set(), targets)
    else:
        inner, out = sum(len(w) for w in g.parents.values()), len(g.out)
    if not n:
        return extra
    if not pos:
        return n * math.log2(positions * subcomponents) + math.log2(n) * (2 * inner + out) + extra
    return n * math.log2(positions * subcomponents) + k * math.log2(positions) + math.log2(n + k) * (2 * (inner + len(g.tok_edges)) + out) + extra


def _unit_draws(rng: np.random.Generator, units: list, draws: int, C: dict, active: dict | None = None) -> list[list]:
    """The node family's removal draws: draw j takes each unit with probability (j + 0.5) / draws (at least one), a
    block unit (site, t, BLOCK) as every subcomponent of its site at t."""
    out = []
    for j in range(draws):
        du = [u for u in units if rng.random() < (j + 0.5) / draws] or [units[int(rng.integers(len(units)))]]
        out.append([x for u in du for x in ([(u[0], u[1], c) for c in block_contents(u[0], u[1], C, active)] if u[2] == BLOCK else [u])])
    return out


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


def _donor_token(d: list, q: int, T: int, x: int) -> int:
    """The token at position q of a text of length T from far donor d aligned at its end (cyclic where shorter), the
    next of the donor's that differs from x when that one is x (resample_influence, the input-scope family)."""
    j = len(d) - T + q
    return next((d[(j + i) % len(d)] for i in range(len(d)) if d[(j + i) % len(d)] != x), x)


def _transplant(base: torch.Tensor, moved: torch.Tensor, still: torch.Tensor) -> torch.Tensor:
    """log_softmax(base + moved - still): a run's change from still to moved (log q on x' and on x) applied to base
    (the model's log p on x), the target positions' distributions [B, targets, V]."""
    return torch.log_softmax(base + moved - still, -1)


KL_RESOLUTION = 2.0 ** -24 * 18  # bits: float32's rounding bound on a KL (unit roundoff x the largest entropy of a
# vocabulary up to 2^18 tokens: vpd4l, GPT-2, Qwen3). The log scale counts a KL below it as it (the lead's 10-13: a score
# must not reward precision the measurement cannot resolve; InterpBench case 11's exact VPD fit put KL_VPD at 6e-10 bits,
# so a 74-subcomponent circuit at 1.28 bits of 5.64 scored 0.065)


def _fid_raw(e0: float, kc: float, kv: float) -> float | None:
    """_fid without the floor at 0 (the reward's ranking, the lead's 10-12): below 0 when the circuit is farther than
    the bound; at most 1. KLs below KL_RESOLUTION count as KL_RESOLUTION; a bound within it leaves nothing to recover."""
    if kv is None or kc is None or not e0 > KL_RESOLUTION or not kv < e0:
        return None
    return min(1.0, math.log(e0 / max(kc, KL_RESOLUTION)) / math.log(e0 / max(kv, KL_RESOLUTION)))


def _c_rl(terms: list) -> float | None:
    """The reward's C: the harmonic mean when every term is above 0, else the lowest term (unclipped): continuous at 0,
    and a circuit worse than its bounds still ranked by how much."""
    if not terms:
        return None
    return _harmonic(terms) if min(terms) > 0 else min(terms)


PHI_FAMILIES = {  # phi's families (the lead's reward v3): score key, how the score pools its draws, the draws' key
    "point": ("s_point_c", None, None), "nodes": ("s_nodes", "weighted", "draws_nodes"), "edges": ("s_edges", "mean", "draws_edges"),
    "flips": ("f1_flips_c", None, None), "track_pp": ("s_track_pp", "weighted", "draws_track_pp"),
    "interchange_pp": ("s_interchange_pp", "weighted", "draws_interchange_pp"),
    "patching_near": ("s_patching_near", "donor", "draws_patching_near"), "patching_far": ("s_patching_far", "donor", "draws_patching_far"),
    "pattern": ("s_pattern", "mean", "draws_pattern"), "N": ("n_spec", "spec", "draws_n_spec"),
    "omitted": ("s_omitted", "weighted", "draws_omitted"), "scope": ("s_scope", "weighted", "draws_scope")}
RESAMPLE_DONORS = 4  # resample_influence's far donors per position (the START's candidate order)
START_CAP = 16  # the START's most candidate positions beside the targets (start_candidates; the input-scope family's cap)
SCOPE_COVER = 0.95  # the input-scope family's positions: by resampling influence until they hold this share of its sum
RESOLUTION_SAFETY = 10.0  # kl_resolution: the KL between two equal C runs times this
OWN_RUN_FAMILIES = ("point", "edges", "flips", "pattern", "nodes", "patching_near", "patching_far")  # floors from VPD at
# the START's candidates (vpd_floors): the runs that zero the outside of a circuit or put a far donor's streams there
START_POOL = native.TEXTS.parent / "goal_eval" / "ci_patterns_train.pt"  # the START's flip-adversary pool (the teacher's)
LINE_CHUNK = 32  # line_effects' path-patched lines per engine call
OMITTED_K = 32  # the omitted family's candidates per circuit (_omitted_family)
GENERIC_TARGET = (("h.1.mlp.c_fc", 2103), ("h.3.mlp.down_proj", 3532), ("h.3.attn.o_proj", 806), ("h.1.attn.q_proj", 316),
                  ("h.0.mlp.c_fc", 53))  # vpd4l's background at the targets (the lead's 10-13): the 5 parts whose removal
# makes about 94% of an influential set's damage on control texts (in the top 5 on 20, 19, 14, 11 and 10 of the 20
# held-out texts, generic_parts.jsonl); at the targets only (at a copied position c_fc #2103, q #316 and #53 are
# copy-specific). Always present: never an omitted candidate; kept on in N's removals when background=True
EPS_FLOOR = 0.005  # eps's floor
EPS_CAP = 0.05  # eps's cap (the Gate B package, 10-13: the toy repair table passes every non-collapsed check with it;
# uncapped, one noisy family's eps alone put every-family-at-0.5 below 0 on held-out texts)
EPS_BOOT = 200  # eps's bootstrap resamples
PHI_CAP = 1.0  # phi counts in full up to the text's bar + PHI_CAP (cap_phi), PHI_SLOPE of it above
PHI_SLOPE = 0.1
PHI_CAP_WIDTH = 0.2  # cap_phi's join between the two slopes, in bits (a softplus: C1, the user's 10-13 softening)
LEAK = 0.1  # phi's family scores below 0 count LEAK of themselves (the user's 10-13: a floor at 0 gives no gradient)
PHI_RAW = {"point": "s_point_c", "nodes": "s_nodes_raw", "edges": "s_edges", "flips": "f1_flips_c",
           "track_pp": "s_track_pp_raw", "interchange_pp": "s_interchange_pp_raw", "patching_near": "s_patching_near",
           "patching_far": "s_patching_far", "pattern": "s_pattern_raw", "N": "n_spec_raw", "omitted": "s_omitted_raw",
           "scope": "s_scope_raw"}
# each family's unfloored score: a row field ending in _raw, else rl_terms' entry (point, edges, flips, patching)


def _floored_score(row: dict, family: str) -> float | None:
    """A row's floored score for one of phi's families (PHI_FAMILIES' keys; flips at FLIP_TERM)."""
    key = PHI_FAMILIES[family][0]
    if key == "f1_flips_c":
        fc = row.get("f1_flips_c") or {}
        return fc.get(str(FLIP_TERM), fc.get(FLIP_TERM))
    return row.get(key)


def family_score(row: dict, family: str) -> float | None:
    """A row's score for one of phi's families, the one definition phi, the bar, the floors, both shortfalls and the
    teacher's acceptance read (the user's 10-13 softening of f5a1df2's floor at 0): the unfloored score (PHI_RAW: the
    weighted families' linear share, each draw counting from -(its gap), and N in [-1, 1], the log-scale ones clipped
    there), LEAK of it below 0, so a family under 0 still ranks answers; the floored score where the row keeps no
    unfloored one. A weighted family's score is its raw pool even above 0 (a draw where the circuit is farther than no
    change counts against it), so floors and shortfalls compare like with like."""
    key = PHI_RAW[family]
    raw = row.get(key) if key.endswith("_raw") else (row.get("rl_terms") or {}).get(key)
    if raw is None:
        return _floored_score(row, family)
    raw = min(1.0, max(-1.0, float(raw)))
    return raw if raw >= 0 else LEAK * raw


def family_kls(row: dict) -> dict:
    """The raw KLs in bits at the targets beside a row's scores (the external audit, 10-13: a normalized 0.9 can be
    12 bits where the empty circuit's KL is large): "point" the circuit's own run, "empty" and "vpd" the bounds, and
    per family the circuit's mean KL from the model under its interventions (nodes, edges, flips at FLIP_TERM, the
    path-patched and omitted draws' mean KL_C, patching near and far, pattern; N reports the target damage D_t)."""
    def mean(key, j=1):
        d = [x[j] for x in row.get(key) or [] if x[j] is not None]
        return float(np.mean(d)) if d else None

    fl = row.get("kl_flips_omitted") or {}
    kf = fl.get(FLIP_TERM, fl.get(str(FLIP_TERM)))
    out = {"point": row.get("kl_point"), "empty": row.get("kl_empty_live"), "vpd": row.get("kl_vpd_live"),
           "nodes": row.get("kl_nodes_c"), "edges": row.get("kl_edges_c"), "flips": kf[0] if kf else None,
           "track_pp": mean("draws_track_pp"), "interchange_pp": mean("draws_interchange_pp"), "omitted": mean("draws_omitted"),
           "scope": mean("draws_scope"),
           "pattern": row.get("kl_pattern_c"), "N": (row.get("draws_n_spec") or [[None] * 5])[0][3]}
    for pool_ in ("near", "far"):
        k3 = row.get(f"kl_patching_{pool_}")
        out[f"patching_{pool_}"] = k3[0] if k3 else None
    return out


def _pooled(kind: str, draws: list, e0: float | None, *, raw: bool = False) -> float | None:
    """A family's score from some of its draws, as the family pools them: "mean" on the mean KLs ([KL_C, KL_VPD] each,
    against KL_empty e0), "weighted" a pooled log ratio against each draw's bound W ([prompt, KL_C, KL_VPD, W]; _ratio_pool), "donor" on
    the mean KLs against the mean keep-nothing KL ([donor, KL_C, KL_VPD, KL_0, score]), "spec" N's specificity form
    (_spec: [control, D_c, D_c of VPD's answer, D_t, D_t of VPD's answer])."""
    if not draws:
        return None
    if kind == "spec":
        return _spec(draws, raw=raw)
    fid = _fid_raw if raw else _fid
    if kind == "mean":
        return None if e0 is None else fid(e0, float(np.mean([d[0] for d in draws])), float(np.mean([d[1] for d in draws])))
    if kind == "donor":
        return fid(float(np.mean([d[3] for d in draws])), float(np.mean([d[1] for d in draws])), float(np.mean([d[2] for d in draws])))
    return _ratio_pool(draws, raw)


def _ratio_pool(draws: list, raw: bool = False) -> float | None:
    """The "weighted" families' score ([prompt, KL_C, KL_VPD, W] per draw: track_pp, interchange_pp, nodes, omitted):
    the share of VPD's improvement over no change the circuit makes, sum over draws of clip(W - KL_C, 0, W - KL_VPD) /
    sum of (W - KL_VPD), over the draws where VPD's answer is nearer than no change (KL_VPD < W); raw: the clip from
    -(W - KL_VPD), so in [-1, 1]. Each draw weighs its VPD gap W - KL_VPD in bits: one whose VPD run sits near the
    bound weighs nothing, and no draw outweighs its own change. On text9002's F+V circuits over 4 draw seeds (the lead's
    10-13 swing): spread 0.002-0.026, against 0.023-0.074 for per-draw log ratios weighted by W and 0.12-0.32 for
    pooled or gap-weighted log ratios (ln(W / KL_VPD) grows without bound as VPD's KL nears 0). None without such draws.
    (No standard-error guard: each draw counts at most its own gap, so nothing blows up; 71c66c3's guard on the gaps'
    sum dropped track_pp from text9002's search split, whose fixed ranked draws carry the change unevenly: pymove.)"""
    num, gap = [], []
    for d in draws or []:
        w, kc, kv = d[3], d[1], d[2]
        if kc is None or kv is None or not (w > 0 and kv < w):
            continue
        g = w - kv
        gap.append(g)
        num.append(min(g, max(-g if raw else 0.0, w - kc)))
    if not gap:
        return None
    return float(sum(num)) / float(sum(gap))

def _spec(draws: list, raw: bool = False) -> float | None:
    """N (necessity_spec) from its records [control, D_c, D_c(VPD), D_t, D_t(VPD)]: (D_t - mean D_c) / (D_t(VPD) -
    mean D_c(VPD)) in [0, 1], or [-1, 1] raw; None with fewer than 2 controls or unless VPD's denominator exceeds
    twice its standard error over the controls (the Gate B package, 10-13: near a zero denominator raw N reached -5
    on a good toy circuit and -2.9 on held-out text9014)."""
    if not draws or len(draws) < 2:
        return None
    dv = np.array([d[2] for d in draws], dtype=float)
    den = draws[0][4] - float(dv.mean())
    if not den > 2.0 * float(dv.std(ddof=1)) / math.sqrt(len(dv)):
        return None
    x = min(1.0, max(-1.0, (draws[0][3] - float(np.mean([d[1] for d in draws]))) / den))
    return x if raw else max(0.0, x)


def _pooled_raw(draws: list) -> float | None:
    """_pooled's "weighted" form without the floor at 0 (c_rl's terms; at least -1)."""
    return _ratio_pool(draws, raw=True)


def _pooled_resamples(kind: str, draws: list, e0: float | None, raw: bool, idx: np.ndarray) -> list:
    """_pooled of each resample of draws (idx [R, n], draw indices per resample), as arrays: a draw's own score
    ("weighted") or KLs ("mean", "donor", "spec") computed once."""
    fid = _fid_raw if raw else _fid
    if kind == "spec":  # _spec per resample: D_t and D_t(VPD) its first record's, the controls' means
        dc, dcv = (np.array([d[j] for d in draws], dtype=float)[idx].mean(1) for j in (1, 2))
        dt, dtv = (np.array([d[j] for d in draws], dtype=float)[idx[:, 0]] for j in (3, 4))
        den = dtv - dcv
        x = np.minimum(1.0, (dt - dc) / np.where(den > 0, den, 1.0))
        return [(float(v) if raw else max(0.0, float(v))) if ok else None for v, ok in zip(x, den > 0)]
    if kind == "weighted":  # _ratio_pool per resample (its standard-error guard is the point estimate's)
        ok = [d[1] is not None and d[2] is not None and d[3] > 0 and d[2] < d[3] for d in draws]
        gp = np.array([d[3] - d[2] if o else 0.0 for d, o in zip(draws, ok)], dtype=float)
        nm = np.array([min(d[3] - d[2], max(-(d[3] - d[2]) if raw else 0.0, d[3] - d[1])) if o else 0.0 for d, o in zip(draws, ok)], dtype=float)
        num, den = nm[idx].sum(1), gp[idx].sum(1)
        return [float(a) / float(b) if b > 0 else None for a, b in zip(num, den)]
    if kind == "mean":
        if e0 is None:
            return [None] * len(idx)
        kc, kv = (np.array([d[j] for d in draws], dtype=float)[idx].mean(1) for j in (0, 1))
        return [fid(e0, float(a), float(b)) for a, b in zip(kc, kv)]
    k0, kc, kv = (np.array([d[j] for d in draws], dtype=float)[idx].mean(1) for j in (3, 1, 2))  # "donor"
    return [fid(float(a), float(b), float(c)) for a, b, c in zip(k0, kc, kv)]


def family_eps(row: dict, only=None) -> dict:
    """eps_<family> for phi (the lead's reward v3): twice the bootstrap standard error of each family's score over its
    draws in the row (EPS_BOOT resamples, a fixed seed: the same from the row alone), at least EPS_FLOOR; a family
    scored in one run (point, flips) the floor. The unfloored estimator clipped to [-1, 1], as family_score reads it. only: those families alone (the
    others' resample indices still drawn: the same eps as with every family)."""
    rng = np.random.default_rng(0)
    out = {}
    for fam, (_, kind, key) in PHI_FAMILIES.items():
        draws = row.get(key) if key else None
        if not kind or not draws or len(draws) < 2:
            if only is None or fam in only:  # nothing omitted: s_omitted 1, exactly (the lead's 10-13)
                out[fam] = 0.0 if fam == "omitted" and not draws and row.get("s_omitted") == 1.0 else EPS_FLOOR
            continue
        idx = rng.integers(len(draws), size=(EPS_BOOT, len(draws)))  # the same draws as one call per resample
        if only is not None and fam not in only:
            continue
        got = _pooled_resamples(kind, draws, row.get("kl_empty_live"), True, idx)
        got = [min(1.0, max(-1.0, x)) for x in got if x is not None]
        out[fam] = min(EPS_CAP, max(EPS_FLOOR, 2 * float(np.std(got, ddof=1)))) if len(got) > 1 else EPS_FLOOR
    return out


def _phi(scores: list) -> float | None:
    """-log2 of the power-4 mean of (1 - s + eps) over (s, eps) pairs."""
    if not scores:
        return None
    return -math.log2(float(np.mean([(1 - s_ + e_) ** 4 for s_, e_ in scores])) ** 0.25)


def _eps(row: dict) -> dict:
    """The row's eps per family (family_eps where absent), at most EPS_CAP (rows stored before the cap too)."""
    eps = {f: row[f"eps_{f}"] for f in PHI_FAMILIES if row.get(f"eps_{f}") is not None}
    eps = eps if len(eps) == len(PHI_FAMILIES) else {**family_eps(row), **eps}
    return {f: min(EPS_CAP, v) for f, v in eps.items()}


GATE_SLOPE = 10.0  # the reward's gate g = 1 / (1 + 2^(-GATE_SLOPE (phi - phi_tau))) (components.GATE_SLOPE)
FLOOR_TOP, FLOOR_MARGIN = 0.9, 0.05  # a family's floor: FLOOR_TOP of VPD's own answer's score (floors); FLOOR_MARGIN the
# teachers' changed-input acceptance below its start's value (teacher.floor_of)
SHORTFALL_COST, SHORTFALL_GATE = 4.0, 20.0  # R - SHORTFALL_COST v, the gate times 2^(-SHORTFALL_GATE v)
SHORTFALL_WIDTH = 0.02  # the soft shortfall's softplus width in score units (the user's 10-13 softening)
TIE, TIE_SCALE = 0.02, 2.0  # the ungated compactness tie-break: TIE x TIE_SCALE tanh(compact / TIE_SCALE)


def floors(vpd: dict, families: list) -> dict:
    """Per-family floors, the reward's and the teachers' (the lead's 10-13 design): FLOOR_TOP of VPD's own answer's
    score on the family (vpd: VPD's whole answer's reward-mode row), in [0, FLOOR_TOP], 0 where VPD's answer scores
    nothing. Absolute: floors once taken from the start (its score less 0.05) never charged what the start missed
    (40% of the v2.3 teachers held the targets alone)."""
    out = {}
    for f in families:
        s = family_score(vpd, f)
        out[f] = 0.0 if s is None else FLOOR_TOP * max(0.0, min(1.0, s))
    return out


def shortfall(row: dict, floors_: dict) -> float:
    """v = max_i (f_i - s_i)_+, the worst shortfall below a family's floor in score units (an unscored family at s =
    0): faithful's test (v = 0) and the teacher's acceptance; circuit_term charges shortfall_soft."""
    return max((max(0.0, f - (family_score(row, k) or 0.0)) for k, f in floors_.items()), default=0.0)


def shortfall_soft(row: dict, floors_: dict) -> float:
    """The reward's shortfall (the user's 10-13 softening: a hard max and hinge give one family any gradient):
    (mean_i softplus_w(f_i - s_i)^4)^(1/4) with w = SHORTFALL_WIDTH, s_i phi's score (family_score, an unscored family at
    0), a power mean like phi's: every family near or under its floor counts, the worst the most. About 0.014 n^(-1/4)
    per family exactly at its floor, 0.002 per one 0.05 above."""
    if not floors_:
        return 0.0
    w = SHORTFALL_WIDTH
    sp = [w * float(np.logaddexp(0.0, (f - (family_score(row, k) or 0.0)) / w)) for k, f in floors_.items()]
    return float(np.mean([x ** 4 for x in sp])) ** 0.25


def circuit_term(row: dict, tau: dict, bits: float | None, quality: float = 0.0, executable_bits: float | None = None) -> dict:
    """Reward v3's circuit part, one function for components.reward and the teacher (circuit_score): term =
    cap_phi(phi) - phi_tau - 4 v + g 2^(-20 v) (0.5 compact + quality) + 0.04 tanh(compact / 2), every piece smooth (the
    user's 10-13 softening). phi over tau's families (family_score's leaky floor); v = shortfall_soft against tau's
    per-family floors (a single bad family can no longer hide in the power mean: one at 0.45 among ten at 0.95 gave phi
    1.67, above every family at 0.5); g the gate at the bar; compact = log2(bits_start / bits) (None when
    executable_bits, default bits, is 0); the last term an ungated tie-break toward the smaller answer (padding below
    the bar was free: generality's falsification set ranked it right on 0.59). quality: the reward's other gated terms
    (E2g + E4x + E1). Returns the term and its pieces; "v" is the hard worst shortfall (shortfall: faithful's and the
    teacher's v = 0), "v_soft" the one the term charges."""
    fams, pt = tau.get("families"), tau.get("phi_tau")
    full = phi(row, families=fams)[0]
    ph = cap_phi(full, pt)
    fl = tau.get("floors") or {}
    v, vs = shortfall(row, fl), shortfall_soft(row, fl)
    eb = bits if executable_bits is None else executable_bits
    comp = compact(bits, tau.get("bits_start")) if bits and eb and eb > 0 else None
    g = None if ph is None or pt is None else 1.0 / (1.0 + 2.0 ** (-GATE_SLOPE * (ph - pt)))
    ge_ = (g or 0.0) * 2.0 ** (-SHORTFALL_GATE * vs)
    tie = TIE * TIE_SCALE * math.tanh((comp or 0.0) / TIE_SCALE)
    term = (ph or 0.0) - (pt or 0.0) - SHORTFALL_COST * vs + ge_ * (0.5 * (comp or 0.0) + quality) + tie
    return {"term": term, "phi_full": full, "phi": ph, "phi_tau": pt, "v": v, "v_soft": vs, "floors": tau.get("floors"),
            "g": g, "g_floor": ge_, "compact": comp, "tie": tie}


def cap_phi(ph: float | None, tau: float | None) -> float | None:
    """phi as the reward counts it (the Gate B package, 10-13; smooth since the user's 10-13 softening): phi - (1 -
    PHI_SLOPE) w ln(1 + e^((phi - tau - PHI_CAP) / w)), w = PHI_CAP_WIDTH: slope 1 below the text's bar + PHI_CAP,
    PHI_SLOPE above, a softplus join between (faithfulness past the bar is not worth size: a superset gaining 3 bits of
    phi was worth a 64x larger circuit through compactness's 0.5 log2; the slope still breaks ties toward
    faithfulness). At the bar itself it is phi less 0.001 bits."""
    if ph is None or tau is None:
        return ph
    w = PHI_CAP_WIDTH
    return ph - (1.0 - PHI_SLOPE) * w * float(np.logaddexp(0.0, (ph - tau - PHI_CAP) / w))


def phi(row: dict, start: float | dict | None = None, families: list | None = None) -> tuple:
    """(phi, phi_tau) of a row (the lead's reward v3): phi = -log2((mean_i (1 - s_i + eps_i)^4)^(1/4)) over the text's
    families (families: those the text's start set scores, tau()'s "families"; PHI_FAMILIES: point, nodes, edges,
    flips, track_pp, interchange_pp, patching near and far, pattern, N), a family the answer leaves unscored counting
    s = 0 (the lead's Gate B repair 10-12: an answer never escapes a family by failing to be scored; a family the text
    cannot score is left out for every answer on it, the start included); without families, the families the row
    scores. eps_i the row's eps_<family> (family_eps when absent): an answer's own noise only lowers its phi.
    phi_tau is a constant of the text: max(min(phi of the start, every family at 0.9), every family at 0.5) under the
    start's eps over its families (the Gate B floor: a weak start never puts the bar within reach of a near-empty
    circuit; tau() raises it to the empty circuit's phi + 1 as well), from start (the start set's row; or its phi_tau,
    a number, as tau() gives it); without start the row's own phi_tau, else the row taken as its own start. From the
    rows' fields alone."""
    eps = _eps(row)
    fams = families if families is not None else (
        [f for f in PHI_FAMILIES if family_score(start, f) is not None] if isinstance(start, dict) else
        [f for f in PHI_FAMILIES if family_score(row, f) is not None])
    got = _phi([(0.0 if family_score(row, f) is None else family_score(row, f), eps[f]) for f in fams])
    if isinstance(start, dict):
        return got, _bar(start, fams)
    if start is not None:
        return got, float(start)
    if row.get("phi_tau") is not None:
        return got, row["phi_tau"]
    return got, _bar(row, fams)


def _bar(start: dict, families: list) -> float | None:
    """max(min(phi of the start, every family at 0.9), every family at 0.5), under the start's eps (phi_tau)."""
    se = _eps(start)
    ps = _phi([(0.0 if family_score(start, f) is None else family_score(start, f), se[f]) for f in families])
    at9, at5 = _phi([(0.9, se[f]) for f in families]), _phi([(0.5, se[f]) for f in families])
    if ps is None:
        return at9
    return max(min(ps, at9), at5)


def compact(bits: float, bits_start: float) -> float | None:
    """The reward's compactness (the lead's v3): log2(bits_start / bits), raw description bits (size_bits, blocks at
    their active contents) against the start set's."""
    return None if not bits or bits <= 0 or not bits_start or bits_start <= 0 else math.log2(bits_start / bits)


def _rms_parts(read: torch.Tensor, scale: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """vpd_model.rms with the vector read and the vector setting the scale apart: w * read / rms(scale) (cuts' partial
    modes); rms itself when they are one."""
    return w * read * torch.rsqrt(scale.float().pow(2).mean(-1, keepdim=True) + eps).to(read.dtype)


def _parts(tg, read: torch.Tensor, scale: torch.Tensor, j: int) -> torch.Tensor:
    """Norm j (vpd_model.Target.norm's numbering) of the vector read at the vector scale's statistics: vpd4l's in
    _rms_parts' order, another Target's (vpd_model.TLTarget) by its norm_parts."""
    if tg.norm_kind == "rms":
        return _rms_parts(read, scale, tg.norm_weight(j), tg.eps)
    return tg.norm_parts(read, scale, j)


def _norm_stat(tg, x: torch.Tensor) -> torch.Tensor:
    """Norm's scale statistic of x (graph_ref's recorded or frozen scale): vpd4l's 1 / rms(x); a TLTarget's its own."""
    if tg.norm_kind == "rms":
        return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + tg.eps)
    return tg.norm_stat(x)


def _norm_with(tg, x: torch.Tensor, sc: torch.Tensor, j: int) -> torch.Tensor:
    """Norm j of x at the scale statistic sc (_norm_stat's): vpd4l's w * x * sc."""
    if tg.norm_kind == "rms":
        return tg.norm_weight(j) * (x.float() * sc).to(x.dtype)
    return tg.norm_with(x, sc, j)


class _ScoreRows:
    """A layer's attention scores [H, T, T] of a run, from its rotated queries and keys [H, T, hd], made a query row at
    a time when indexed ([:, t] or [:, t, ...]: only those rows; _pp_record keeps no [H, T, T] a layer, an audit scale
    blocker)."""

    def __init__(self, q: torch.Tensor, k: torch.Tensor):
        self.q, self.k = q, k

    def __getitem__(self, idx):
        if not (isinstance(idx, tuple) and len(idx) >= 2 and idx[0] == slice(None)):
            raise IndexError("score rows are indexed [:, t] or [:, t, ...]")
        t = idx[1]
        one = not isinstance(t, (slice, list, tuple)) and not (torch.is_tensor(t) and t.dim() > 0)  # one position
        rows = self.q[:, t][:, None] if one else self.q[:, t]
        out = (rows @ self.k.transpose(-1, -2)) / math.sqrt(self.q.shape[-1])
        return (out[:, 0] if one else out)[(slice(None),) + tuple(idx[2:])]


def _harmonic(terms: list) -> float | None:
    """C's combination: the harmonic mean k / sum(1 / s), 0 when a term is 0 (None for no terms)."""
    if not terms:
        return None
    return 0.0 if min(terms) <= 0 else len(terms) / sum(1.0 / t for t in terms)


class _Parts:
    """A lever's parts (site, position, subcomponent) by their index, the sites end to end (per site positions t and
    columns c, index tensors): each made when asked for (an adversary reads a few hundred of a circuit's ~10^5)."""

    def __init__(self, names: list, t: list, c: list):
        self.names = list(names)
        self.t = [x.cpu().numpy() for x in t]
        self.c = [x.cpu().numpy() for x in c]
        self.off = np.cumsum([0] + [len(x) for x in self.t])

    def __getitem__(self, j: int) -> tuple:
        s = int(np.searchsorted(self.off, j, side="right")) - 1
        k = j - int(self.off[s])
        return self.names[s], int(self.t[s][k]), int(self.c[s][k])


def f2_cut(moved: np.ndarray, floor: float = SHARE_FLOOR) -> list[int]:
    """F2's positions (0-based, ascending) from the sensitivity probe's moved: the positions sorted by their share of the
    total, cut where consecutive shares fall by the largest ratio among cuts whose last kept share is at least floor /
    F2_BAND and whose first dropped one is at most floor x F2_BAND (every share at or above floor when no cut lies in
    that window). A hard cutoff at floor put text9019's position 306 in or out of F2 on a 3% difference (moved 0.1093
    against 0.1128): a cut at the largest gap moves only when shares cross each other by that ratio."""
    total = float(moved.sum())
    if not total > 0:
        return []
    order = np.argsort(-moved, kind="stable")
    s = moved[order] / total
    best = None
    for k in range(1, len(s) + 1):
        hi, lo = float(s[k - 1]), float(s[k]) if k < len(s) else 0.0
        if hi < floor / F2_BAND:
            break
        if lo > floor * F2_BAND:
            continue
        r = hi / max(lo, 1e-12)
        if best is None or r > best[0]:
            best = (r, k)
    k = best[1] if best is not None else int((s >= floor).sum())
    return sorted(int(j) for j in order[:k])


def _fid(e0: float, kc: float, kv: float) -> float | None:
    """C's log scale: clip(ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD), 0, 1); None when VPD's own answer is no
    nearer the model than the empty circuit under the interventions. KLs below KL_RESOLUTION count as KL_RESOLUTION."""
    if kv is None or kc is None or not e0 > KL_RESOLUTION or not kv < e0:  # (a bound within resolution: nothing to recover)
        return None
    return _clip01(math.log(e0 / max(kc, KL_RESOLUTION)) / math.log(e0 / max(kv, KL_RESOLUTION)))


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


def _node_values(pairs: dict, effects: dict) -> dict:
    """Do a circuit's own nodes take the model's values (the lead's 10-13 eval metric, reported, not in phi)? pairs: node
    -> [(r_model, r_circuit)] over a family's interventions; per node z = RMS(r_circuit - r_model) / SD(r_model) over
    them, the node's own spread across the changed texts (the write's scale |u| cancels: gauge-invariant), over nodes
    with two or more interventions and a model read that varies. node_share: the share with z <= 1 (the circuit's error
    within the node's natural variation); node_share_weighted: weighted by effects (node -> its measured downstream
    effect, line_effects' delta_kl floored at 0); node_median_z; node_measured: how many nodes the share is over. On 20
    CausalGym texts the subject circuit beat the same nodes moved to the attractor on node_share 20/20; dev teachers'
    median share was 0.19 (they reproduce outputs, not their own internal states); blind where the interventions leave
    the nodes still (the cancel and route2 toys) and reversed on the binding_fp toy."""
    z = {}
    for nd, xs in pairs.items():
        m, c = np.array([x[0] for x in xs], float), np.array([x[1] for x in xs], float)
        if len(m) >= 2 and m.std() > 0:
            z[nd] = float(np.sqrt(np.mean((m - c) ** 2)) / m.std())
    w = sum(effects.get(nd, 0.0) for nd in z)
    return {"node_share": float(np.mean([v <= 1.0 for v in z.values()])) if z else None,
            "node_share_weighted": sum(effects.get(nd, 0.0) * (v <= 1.0) for nd, v in z.items()) / w if w > 0 else None,
            "node_median_z": float(np.median(list(z.values()))) if z else None, "node_measured": len(z)}


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


def graph_of_lines(g: native.Graph, cols: list, row) -> native.Graph:
    """The stated graph g with the lines a row of Engine.path_patched(lines=) turns off removed (cols: Engine.pp_lines(g)):
    a node off with its connections, token edges and the pairs naming it (its read keeping its entry), a connection,
    token edge or pair off alone; nothing re-stranded (line_effects' variants). A complete graph: its nodes on."""
    on = [bool(x) for x in (row.tolist() if hasattr(row, "tolist") else row)]
    off = {k[1] for k, o in zip(cols, on) if k[0] == "n" and not o}
    nodes = [k[1] for k, o in zip(cols, on) if k[0] == "n" and o]
    if g.complete:
        return native.Graph(nodes, complete=True)
    keep = {k for k, o in zip(cols, on) if o}
    par = {}
    for r, ws in g.parents.items():
        kept = [w for w in ws if ("e", w, r) in keep and w not in off and r not in off]
        if kept:
            par[r] = kept
    h = native.Graph(nodes, par, [w for w in g.out if ("o", w) in keep and w not in off])
    h.tok_edges = [(t, r) for t, r in getattr(g, "tok_edges", []) if ("t", t, r) in keep and r not in off]
    h.tok_out = [t for t in getattr(g, "tok_out", []) if ("u", t) in keep]
    h.pairs = {read: [(qa, kb) for qa, kb in ps if (qa is None and kb is None) or (("p", read[0], read[1], qa, kb) in keep and qa not in off and kb not in off)]
               for read, ps in (getattr(g, "pairs", None) or {}).items()}
    h.stranded, h.stranded_bits = [], 0.0
    return h


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
        # the behavior-targeted mode (the lead's 10-13): a task's "behavior" {"tokens", "pairs"} makes every distribution
        # at the targets the one over its tokens (Target.set_behavior, set here for this text: one text's mode at a
        # time) and puts its minimal pairs [(position, source token)] first in every family's changed-text draws
        beh = (task or {}).get("behavior") or {}
        if hasattr(nat.target, "set_behavior"):
            nat.target.set_behavior(beh.get("tokens"), self.targets)
        self.behavior_pairs = [(int(p), int(z)) for p, z in beh.get("pairs", [])]
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
            if self.behavior_pairs:  # a behavior's minimal pairs are F2 positions, moved at least by their own change
                prompts, f2, moved = self._probe
                f2, moved, have = list(f2), np.array(moved, dtype=float), {p for p, _ in f2}
                for p, z in self.behavior_pairs:
                    x = list(self.ids)
                    x[p] = z
                    m = float(carry.kl(self.ref[:, -1:], self.ge.nat.reference([x], self.targets[-1:]))[0]) * math.log(2)  # nats
                    if p not in have:
                        f2.append((p, [x]))
                        have.add(p)
                    if 0 <= p - 1 < len(moved):
                        moved[p - 1] = max(moved[p - 1], m)
                self._probe = (prompts, f2, moved)
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

    FIRST_POSITION = True  # resample_influence (the START's candidates, input scope) from position 0 (the lead's 10-14):
    # paper_bracket's "<" at 0 (the paper's L0 c_fc #2149/#2643, down #3069, L2 v #473 there, all VPD-active) entered its
    # START and circuit only with it, while on text9000-9005 position 0 moved the prediction 0.000-0.003 bits (rank
    # 41-236) and changed no START; off for a target whose texts start with the BOS it always reads (ib_teacher)

    def __init__(self, nat: native.Native | None = None, *, changes: int = native.CHANGES, ranks=HELD_OUT_RANKS,
                 share_floor: float = SHARE_FLOOR, random_pairs: int = RANDOM_PAIRS, max_edges: int = MAX_EDGES, chunk: int = CHUNK,
                 cut_chunk: int = CUT_CHUNK, semantics: str = "n"):
        if semantics not in SEMANTICS:
            raise ValueError(f"semantics is one of {SEMANTICS}")
        self.nat = nat or native.Native()
        self.semantics = semantics  # which C families run circuits and VPD's answer with routing and norms at x (_xkw)
        self._xk: dict = {}
        self.changes, self.ranks, self.share_floor = changes, tuple(ranks), share_floor
        self.random_pairs, self.max_edges, self.chunk, self.cut_chunk = random_pairs, max_edges, chunk, cut_chunk
        self.hooks: dict = {"E1": None, "E2": None, "E4": None}  # E4: fn(tx, graph, split) -> float or None (fidelity's e4)
        self.probes: dict = {}  # (ids, targets, seed) -> the changed prompts and F2's prompts, as int32 arrays
        self.donor_pools: dict = {}  # split -> the far donors' [(name, token ids)] (far_pool; set to override)

    def text(self, ids: list[int], targets: list[int], seed: int | None = None, task: dict | None = None) -> Text:
        return Text(self, ids, targets, native.task_seed(str(ids[:8])) if seed is None else seed, task)

    def _xkw(self, ids: list[int], targets: list[int], family: str = "point") -> dict:
        """The engine's keywords for a circuit's or VPD's run in one of C's families under this GoalEval's semantics
        (SEMANTICS): none where the family runs as stated; at x the query and key sites' activations and every norm's
        scale on the model's run on ids (Engine.x_values), every norm frozen, kept per text."""
        if family not in SEMANTICS[self.semantics]:
            return {}
        key = (tuple(ids), tuple(targets))
        if key not in self._xk:
            import engine

            E = engine.Engine(self.nat, ids, targets, live=True, columns={n: torch.zeros(0, dtype=torch.long, device=self.nat.dev) for n in self.nat.names})
            xv = E.x_values([n for n in self.nat.names if native._name_layer_kind(n)[1] in ("q_proj", "k_proj")])
            self._xk[key] = {"at_x": xv["acts"], "norms": xv["norms"], "freeze": "all"}
        return self._xk[key]

    def probe(self, ids: list[int], targets: list[int], seed: int) -> tuple[list, list, np.ndarray]:
        """(the text's changed prompts (Native.changes), F2's [(position, held-out prompts)], the probe's moved per
        position), kept per text: the sensitivity probe behind them is most of their cost (16 s at T 395 on the Mac's
        GPU)."""
        nat, key = self.nat, (tuple(ids), tuple(targets), seed, getattr(self.nat.target, "behavior_tokens", None))
        if key not in self.probes:
            probe = {}
            prompts = nat.changes(ids, targets, n=self.changes, seed=seed, probe=probe)
            f2, moved = [], np.zeros(0)
            if "moved" in probe:
                moved = probe["moved"]
                total = float(moved.sum())
                for j in f2_cut(np.asarray(moved.cpu() if torch.is_tensor(moved) else moved, dtype=np.float64), self.share_floor) \
                        if total > 0 else []:
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
    def runs(self, ids: list[int], targets: list[int], sets: list[dict], live: bool, family: str = "point") -> torch.Tensor:
        """log q at the targets [len(sets), len(targets), V] of VPD's masked forward pass (vpd_model's Site masks) with
        each set's subcomponents on, every other off, remainders on (live) or off; where family runs at x (_xkw) on the
        engine with routing and norms at x."""
        nat, tg = self.nat, self.nat.target
        T = len(ids)
        out = []
        kw = self._xkw(ids, targets, family)
        if kw:
            import engine

            E = engine.Engine(nat, ids, targets, live=live, columns=None)
            for s0 in range(0, len(sets), self.chunk):
                part = sets[s0:s0 + self.chunk]
                out.append(E.log_probs({n: torch.stack([m[n] for m in part]).float() for n in nat.names}, **kw))
                native.free(nat.dev)
            return torch.cat(out)
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
            out.append(torch.log_softmax(tg.unembed(hidden[:, targets]).float(), -1))
        return torch.cat(out)

    @torch.no_grad()
    def cancellation(self, tx: Text) -> dict:
        """VPD's description at the last target's final residual stream in the real model: the sums of the writes of
        the subcomponents VPD's answer has on and of those it has off (every attention and MLP output matrix's,
        rec x U), their norms beside the model's stream, and the cosine between the two sums (strongly negative: the
        actives overshoot and the inactives cancel them, which caps the real model's necessity of any part of the
        answer). The embedding, both sums and the remainders add up to the stream."""
        nat, tg, t = self.nat, self.nat.target, tx.targets[-1]
        act = torch.zeros(tg.d_model, device=nat.dev)
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
                  rows: dict | None = None, rest: dict | None = None, hold: bool = False, units: dict | None = None,
                  capture: dict | None = None, outside: dict | None = None) -> torch.Tensor:
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
        in place of the run's own at those reads (pattern_family). rest: receives each site's remainder output {site:
        [B, T, d_out]} (live); with hold, every remainder outputs rest's values instead (a run of the whole model on the
        text recorded first: the remainders held at their values on x, point_remainders). units: the model's own units
        (native_units: an attention head at a position, a post-GELU neuron at a position) replaced, {"head": (mask [B,
        layers, T, H], values or None), "neuron": (mask [B, layers, T, d_mlp], values [B or 1, layers, T, d_mlp] or
        None)}: with values (a held head: {"q", "k", "v" before the rotation, "out" before the output matrix: [B or 1,
        layers, H, T, hd]}) the unit takes them, without them its output is 0 (a head's query, key and value stay its
        own). capture: receives those values of the run {"q", "k", "v", "out": [B, layers, H, T, hd], "neuron": [B,
        layers, T, d_mlp]}. outside: {site: activations [B or 1, T, C]}: that site's subcomponents outside the circuit
        take these values at full strength (the model's on the text: the q/k sites held at x give the circuit the model's
        attention instead of C's near-uniform one, rustmove's 10-13 measurement)."""
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
        if units:
            B = max(B, max(m.shape[0] for m, _ in units.values()))
        cap = {k: [] for k in ("q", "k", "v", "out", "neuron")} if capture is not None else None
        hm, hv = units.get("head", (None, None)) if units else (None, None)
        nmk, nv = units.get("neuron", (None, None)) if units else (None, None)
        if graph.complete:
            circ = {n: m.float() for n, m in self.masks(graph, T).items()}
        else:
            circ = {n: torch.zeros(T, nat.C[n], device=dev) for n in nat.names}
            ix = {}
            for n, t, c in nodes:
                ix.setdefault(n, ([], []))[0].append(t)
                ix[n][1].append(c)
            for n, (t, c) in ix.items():  # one indexed write per site
                circ[n][torch.tensor(t, device=dev), torch.tensor(c, device=dev)] = 1.0
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
            if outside is not None and n in outside:  # held outside the circuit: the circuit's computed, the rest given
                A = torch.where(circ[n] > 0, A, outside[n])
                a[n] = A
                out = A @ tg.site(n).U
            else:
                a[n] = A
                out = (A * on[n]) @ tg.site(n).U
            b = tg.site(n).b  # a bias, always on (vpd4l's matrices have none)
            if hold:
                out = out + rest[n]
            elif live:
                r_ = inp @ tg.site(n).delta_T
                if rest is not None:
                    rest[n] = r_.detach()
                out = out + r_
            return out if b is None else out + b

        def norm(xx, j, key, pos=None):  # norm j, its scale recorded into norms or taken from it (freeze)
            frozen = norms is not None and freeze and (freeze == "all" or key != "final")
            if frozen:
                sc = norms[key] if pos is None else norms[key][:, pos]
            else:
                sc = _norm_stat(tg, xx)
                if norms is not None and pos is None:
                    norms[key] = sc.detach()
            return _norm_with(tg, xx, sc, j)

        x = tg.embed(torch.tensor(ids, device=dev))[None].expand(B, -1, -1)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        for i in range(tg.n_layer):
            nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
            z = norm(x, 2 * i, ("attn", i))
            resid = lambda r, ws, x=x, i=i: norm(x[:, r[1]] - sum(write(w) for w in ws), 2 * i, ("attn", i), r[1])  # noqa: E731
            q, k, v = (site(nm[kk], z, resid).view(B, T, H, hd).transpose(1, 2) for kk in ("q_proj", "k_proj", "v_proj"))
            if hm is not None:
                mh = hm[:, i].transpose(1, 2)[..., None]  # [B, H, T, 1]
                if hv is not None:
                    q, k, v = (torch.where(mh, hv[kk][:, i], y) for kk, y in (("q", q), ("k", k), ("v", v)))
            if cap is not None:
                for kk, y in (("q", q), ("k", k), ("v", v)):
                    cap[kk].append(y)
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
            o = P @ v  # [B, H, T, hd]
            if hm is not None:
                o = torch.where(mh, hv["out"][:, i] if hv is not None else o.new_zeros(()), o)
            if cap is not None:
                cap["out"].append(o)
            att = o.transpose(1, 2).reshape(B, T, -1)
            share = lambda r, ws, att=att, P=P: att[:, r[1]] - sum(  # noqa: E731
                P[:, :, r[1], w[1]].repeat_interleave(hd, dim=1) * write(w) for w in ws)
            x = x + site(nm["o_proj"], att, share)
            z2 = norm(x, 2 * i + 1, ("mlp", i))
            resid2 = lambda r, ws, x=x, i=i: norm(x[:, r[1]] - sum(write(w) for w in ws), 2 * i + 1, ("mlp", i), r[1])  # noqa: E731
            pre = site(nm["c_fc"], z2, resid2)
            gl = tg.act(pre)
            if nmk is not None:
                gl = torch.where(nmk[:, i], nv[:, i] if nv is not None else gl.new_zeros(()), gl)
            if cap is not None:
                cap["neuron"].append(gl)
            x = x + site(nm["down_proj"], gl,
                         lambda r, ws, pre=pre: tg.act(pre[:, r[1]] - sum(write(w) for w in ws)))
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
        if cap is not None:
            capture.update({kk: torch.stack(v, 1).detach() for kk, v in cap.items()})
        return torch.log_softmax(tg.unembed(norm(xt, -1, "final")).float(), -1)


    HELD = ("nodes", "edges", "flips", "patching_near", "patching_far", "omitted")  # phi_variants' families held at the
    # answer's values unless recomputed (recompute: "nodes", "patching"); point, track_pp, interchange_pp, scope, pattern and N always are
    LOO_RECOMPUTE = ("nodes",)  # line_loo's (patching per deletion: 64 donor runs a line)
    LOO_HELD = ("edges", "flips", "patching_near", "patching_far", "omitted", "scope")  # line_loo's families held at the answer's values

    def line_loo(self, tx: Text, graph: native.Graph, pool: dict | None, split: str = "search", draws: int = 8,
                 bits_start: float | None = None, base: dict | None = None) -> dict:
        """Each line's leave-one-out effect on the reward's phi and compact (the lead's reward v3, approved 10-12): the
        answer with that one line deleted (a node with its connections, token edges and pairs; a stated connection;
        a stated token edge), nothing else re-stranded: phi_variants with one deletion a row. compact = log2(bits_start
        / bits) when bits_start is given. Returns {"phi", "bits", "held", "lines": [{"line", "kind", "phi", "delta_phi",
        "delta_phi_uncapped", "bits", "delta_bits", "compact"}]}: delta_phi = the answer's phi less the deletion's as the
        reward counts phi (cap_phi at the text's bar once tau() scored it; above 0: the line helps), _uncapped without
        the cap."""
        lines = self.loo_lines(tx, graph)
        if not lines:  # an answer without nodes (S1 drops every node of some samples): no line to credit
            g0 = strand(graph, tx.targets, self.nat.C)
            base = base if base is not None else self.fidelity(tx, [g0], pool, mode="reward", split=split, interchange=draws, changed=draws)[0]
            bits = size_bits(g0, self.nat.positions(tx.targets), sum(self.nat.C.values()), tx.targets, active=tx.vpd)
            fams = ((tx.__dict__.get("_tau") or {}).get(split) or {}).get("families")
            return {"phi": phi(base, families=fams)[0], "bits": bits, "compact": compact(bits, bits_start) if bits_start else None,
                    "held": list(self.LOO_HELD), "lines": []}
        got = self.phi_variants(tx, graph, ~torch.eye(len(lines), dtype=torch.bool), pool, split, draws, base,
                                recompute=self.LOO_RECOMPUTE)
        ph, bits = got["phi"], got["bits"]
        tau = ((tx.__dict__.get("_tau") or {}).get(split) or {}).get("phi_tau")  # the text's bar, when scored

        def delta(v):  # the answer's phi less the variant's, as the reward counts phi (cap_phi at the bar)
            return None if None in (ph, v) else cap_phi(ph, tau) - cap_phi(v, tau)
        out = [{"line": [list(x)] if kind == "node" else [list(y) if isinstance(y, tuple) else y for y in x], "kind": kind,
                "phi": v["phi"], "delta_phi": delta(v["phi"]), "delta_phi_uncapped": None if None in (ph, v["phi"]) else ph - v["phi"],
                "bits": v["bits"], "delta_bits": bits - v["bits"], "compact": compact(v["bits"], bits_start) if bits_start else None}
               for (kind, x), v in zip(lines, got["variants"])]
        return {"phi": ph, "bits": bits, "compact": compact(bits, bits_start) if bits_start else None,
                "held": list(self.LOO_HELD), "lines": out}

    def loo_lines(self, tx: Text, graph: native.Graph) -> list:
        """The lines of an answer (stranded on the text) that line_loo and phi_variants delete: ("node", node) per node,
        then a stated graph's ("edge", (writer, reader or None)) per connection and ("token", (t, reader or None)) per
        token edge."""
        g = strand(graph, tx.targets, self.nat.C)
        lines = [("node", nd) for nd in sorted(g.node_set())]
        if not g.complete:
            lines += [("edge", e) for e in _pairs(g)]
            lines += [("token", e) for e in sorted(getattr(g, "tok_edges", []))] + [("token", (t, None)) for t in getattr(g, "tok_out", [])]
        return lines

    def keep_of(self, tx: Text, graph: native.Graph, subsets: list) -> torch.Tensor:
        """phi_variants' keep [K, L] for node subsets of an answer: per subset its nodes kept, every other node deleted
        (with its connections and token edges)."""
        lines = self.loo_lines(tx, graph)
        return torch.tensor([[kind != "node" or x in set(s) for kind, x in lines] for s in subsets], dtype=torch.bool)

    def phi_variants(self, tx: Text, graph: native.Graph, keep: torch.Tensor, pool: dict | None, split: str = "search",
                     draws: int = 8, base: dict | None = None, start=None, recompute: tuple = ("nodes", "patching"),
                     scope_part: str | None = None) -> dict:
        """phi of variants of an answer (teacher acceptance's candidate removals, line_loo's deletions): keep [K, L] over
        loo_lines(tx, graph), per row the lines kept, the others deleted (a node with its connections, token edges and
        pairs; a stated connection; a stated token edge), nothing else re-stranded. Recomputed per variant: point
        (s_point_c), nodes (s_nodes: DRAWS common draws, _node_draws, as fidelity's on the variant), track_pp,
        interchange_pp and pattern, on split's draws, and N (necessity_spec per distinct node set, on NEC_CONTROLS_LOO
        controls), and with recompute's "patching" activation patching near and far (donors()' families, the answer's
        donors); held at the full answer's values (its reward-mode row, base when given): HELD less recompute's (edges
        and flips always). Every variant's runs at once: one C plan of the answer (_loo_c) for a graph without blocks,
        one path-patched call per family (lines=). phi per variant from the answer's row with
        the recomputed terms and their draws (eps re-bootstrapped), against start (phi's start: the start set's row or
        its phi_tau). Returns {"phi", "phi_tau", "bits", "variants": [{"phi", "phi_tau", "bits", "row"}]}: the answer's
        own (with N on the same controls) and each variant's."""
        nat = self.nat
        g = strand(graph, tx.targets, nat.C)
        if base is None:
            base = self.fidelity(tx, [g], pool, mode="reward", split=split, interchange=draws, changed=draws)[0]
        P, S = nat.positions(tx.targets), sum(nat.C.values())
        nodes = sorted(g.node_set())
        lines = self.loo_lines(tx, graph)
        keep = torch.as_tensor(keep, dtype=torch.bool).cpu()
        if keep.dim() != 2 or keep.shape[1] != len(lines):
            raise ValueError(f"keep is [variants, {len(lines)}] over loo_lines")
        K = keep.shape[0]
        gone = [[lines[j] for j in (~keep[i]).nonzero().flatten().tolist()] for i in range(K)]

        def kept(h):
            h.stranded, h.stranded_bits = [], 0.0
            for name in ("tok_edges", "tok_positions", "tok_out", "pairs"):
                if hasattr(g, name) and not hasattr(h, name):
                    setattr(h, name, getattr(g, name))
            return h

        def without(dels):  # the answer less a set of lines
            dn = {x for kind, x in dels if kind == "node"}
            de = {tuple(x) for kind, x in dels if kind == "edge"}
            dt = {tuple(x) for kind, x in dels if kind == "token"}
            if g.complete:
                return kept(native.Graph([y for y in nodes if y not in dn], complete=True))
            h = _drop_nodes(g, dn) if dn else native.Graph(list(g.nodes), dict(g.parents), list(g.out))
            if de:
                par = {r: [w for w in ws if (w, r) not in de] for r, ws in h.parents.items()}
                h = native.Graph(list(h.nodes), {r: ws for r, ws in par.items() if ws}, [w for w in h.out if (w, None) not in de])
                if getattr(g, "pairs", None):
                    h.pairs = {read: [pr for pr in ps if not set(pr) & dn] for read, ps in g.pairs.items()}
            h = kept(h)
            h.tok_edges = [y for y in getattr(g, "tok_edges", []) if y[1] not in dn and tuple(y) not in dt]
            h.tok_out = [t for t in getattr(g, "tok_out", []) if (t, None) not in dt]
            return h

        variants = [without(d) for d in gone]
        flat = [expand_blocks(h, nat.C, tx.vpd) for h in variants]
        e0, v0 = self._bounds(tx, "live")
        fast = self.LOO_FAST and tx.vpd is not None and expand_blocks(g, nat.C, tx.vpd) is g
        held = [f for f in self.HELD if not (f == "nodes" and "nodes" in recompute) and not (f.startswith("patching") and "patching" in recompute)]
        got = self._loo_c(tx, g, lines, keep, flat if "nodes" in recompute else None,
                          patching=(split, base) if "patching" in recompute else None) if fast else None  # one plan
        if got is None:
            nfam = (self._node_family(tx, flat, DRAWS, units=[sorted(h.node_set()) for h in variants])  # (a block one unit)
                    if "nodes" in recompute else [{} for _ in flat])
            pfam = (self.donors(tx, flat, split, DONORS_NEAR, self.far_donors(tx, self.far_pool(split), DONORS_FAR), restrand=False)
                    if "patching" in recompute else [{} for _ in flat])
            point = self.sufficiency(tx, flat, conventions=("live",), vpd_minus=False)  # phi reads the live point alone
            stated = [i for i, h in enumerate(flat) if not h.complete]  # a complete variant's wiring is its point
            wired = dict(zip(stated, self.sufficiency(tx, [native.Graph(sorted(flat[i].node_set()), complete=True) for i in stated],
                                                      conventions=("live",), vpd_minus=False)))
            wired = [wired.get(i, point[i]) for i in range(len(flat))]
        else:
            point, wired, pattern_runs, nfam, pfam = got
            nfam = nfam if nfam is not None else [{} for _ in flat]
            pfam = pfam if pfam is not None else [{} for _ in flat]
        cols = {k: j for j, k in enumerate(self._pp_engine(tx).pp_lines(g))}  # the path-patched families: every variant

        def col(kind, x):  # in one engine call per family (lines=), each a column of Engine.pp_lines
            if kind == "node":
                return cols[("n", x)]
            if kind == "edge":
                return cols[("o", x[0])] if x[1] is None else cols[("e", x[0], x[1])]
            return cols[("u", x[0])] if x[1] is None else cols[("t", x[0], x[1])]
        at = torch.tensor([col(k, x) for k, x in lines], dtype=torch.long)
        mask = torch.ones(K, len(cols))
        mask[:, at] = keep.float()
        tpp = self.track_pp(tx, [g], split, draws, lines=mask)
        ipp = self.interchange_pp(tx, [g], split, draws, lines=mask)
        scp = self.scope_family(tx, [g], split, lines=mask, part=scope_part)  # (scope_part: the teachers' acceptance half)
        pat = self.pattern_family(tx, flat, runs=None if got is None else pattern_runs)
        sets = [frozenset(x for kind, x in d if kind == "node") for d in gone]  # N moves with the node set alone
        distinct = list(dict.fromkeys(s for s in sets if s))
        first = {s: next(i for i in range(K) if sets[i] == s) for s in distinct}
        again = NEC_CONTROLS_LOO != NEC_CONTROLS or not base.get("draws_n_spec")  # the answer's own N on the variants'
        # controls (else its row's, on the same controls)
        got = self.necessity_spec(tx, ([expand_blocks(g, nat.C, tx.vpd)] if again else []) + [flat[first[s]] for s in distinct], split,
                                  NEC_CONTROLS_LOO) if again or distinct else []
        own, nsp = (got[0], dict(zip(distinct, got[1:]))) if again else ({}, dict(zip(distinct, got)))
        bits = size_bits(g, P, S, tx.targets, active=tx.vpd)
        fams = ((tx.__dict__.get("_tau") or {}).get(split) or {}).get("families")  # the text's (tau's), when scored
        if own:
            base = dict(base, **own, rl_terms={**base.get("rl_terms", {}), "n_spec": own["n_spec_raw"]})
            base.pop("eps_N", None)
        ph, ph_tau = phi(base, start, families=fams)
        redo = [f for f in PHI_FAMILIES if f not in held]  # the recomputed families (eps from their own draws)
        out = []
        e_live = self._bounds(tx, "live")[0]
        for i, (h, pt, wd, a, b, c, nd, pf) in enumerate(zip(variants, point, wired, tpp, ipp, pat, nfam, pfam)):
            row = dict(base)
            row.update(nsp.get(sets[i], own))
            if "nodes" in recompute:
                row.update({k: nd.get(k) for k in ("kl_nodes_c", "kl_nodes_vpd", "draws_nodes", "overstate_nodes")})
                row["s_nodes"], row["s_nodes_raw"] = _pooled("weighted", nd.get("draws_nodes"), e_live), _pooled_raw(nd.get("draws_nodes"))
            row.update(pf)
            row.update({"s_point": pt["f1_live"], "s_wiring": wd["f1_live"], "n_point": pt.get("n_live"),
                        "draws_point": [[pt["kl_live"], v0]]})
            row["s_point_c"] = min(y for y in (pt["f1_live"], wd["f1_live"]) if y is not None) if pt["f1_live"] is not None else None
            row.update(a)
            row.update(b)
            row.update(c)
            row.update(scp[i])
            if "rl_terms" in row:
                raw_point = [_fid_raw(e0, x["kl_live"], v0) for x in (pt, wd)]
                row["rl_terms"] = {**row["rl_terms"],
                                   "s_point_c": min(x for x in raw_point if x is not None) if raw_point[0] is not None else None,
                                   "s_track_pp": a.get("s_track_pp_raw"),
                                   "s_interchange_pp": b.get("s_interchange_pp_raw"),
                                   "s_pattern": c.get("s_pattern_raw"),
                                   "n_spec": row.get("n_spec_raw"), "s_nodes": row.get("s_nodes_raw")}
                for pool_ in ("near", "far"):  # (fidelity's raw patching terms)
                    k3 = row.get(f"kl_patching_{pool_}")
                    if k3 and "patching" in recompute:
                        row["rl_terms"][f"s_patching_{pool_}"] = _fid_raw(k3[2], k3[0], k3[1])
            for f in redo:
                row.pop(f"eps_{f}", None)  # the recomputed families' eps from their own draws
            row.update({f"eps_{f}": v for f, v in family_eps(row, only=redo).items()})
            pv, pt_tau = phi(row, start, families=fams)
            out.append({"phi": pv, "phi_tau": pt_tau, "bits": size_bits(h, P, S, tx.targets, active=tx.vpd), "row": row})
            native.free(nat.dev)
        return {"phi": ph, "phi_tau": ph_tau, "bits": bits, "variants": out}

    @torch.no_grad()
    def line_effects(self, tx: Text, graph: native.Graph, mode: str = "pp", split: str = "search", draws: int = 8,
                     chunk: int = 16) -> dict:
        """Each line of an answer's own effect, for RL's per-line rewards (the lead's 10-12): raw bits, unclipped.
        nodes: per node, the KL at the targets with that node alone removed from the answer's graph less the KL of the
        graph (above 0: the node helps), and vpd_active (VPD's answer has it on at that position; a block, any of its
        site's parts there); edges: per stated connection (and, in "pp", stated token edge) the same with it alone cut;
        positions: per influential position (F2's, the probe's moved) whether the answer has a node there and a token
        edge from it. mode "c": C's semantics on the text, every removal one row of batched engine plans (chunk rows
        each; block nodes run as their subcomponents); "pp": the path-patched run (pp_run's numbers) on split's changed
        texts (draws of changed_inputs' prompts), each removal's KL change weighted by how far the change moves the
        model, every removal and changed text in one engine.Engine.path_patched(lines=) call, and pairs: per stated
        query-key pair the same with it alone removed.
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

        variants = [g] + [without_node(nd) for nd in nodes] + [without_edge(e) for e in edges] if mode == "c" else []
        prs = [] if g.complete else [(read, pr) for read, ps in sorted((getattr(g, "pairs", None) or {}).items()) for pr in ps
                                     if pr != (None, None)]
        if mode == "c":
            kl = []
            for s0 in range(0, len(variants), chunk):
                part = [expand_blocks(h, nat.C, tx.vpd) for h in variants[s0:s0 + chunk]]
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
            E = self._pp_engine(tx)
            col = {k: j for j, k in enumerate(E.pp_lines(g))}
            offs = ([col[("n", nd)] for nd in nodes] + [col[("o", w)] if r is None else col[("e", w, r)] for w, r in edges]
                    + [col[("u", t)] if r is None else col[("t", t, r)] for t, r in toks]
                    + [col[("p", L, t, qa, kb)] for (L, t), (qa, kb) in prs])
            refs = []  # per draw the model on the changed text and its weight W
            for q, z in got:
                x2 = list(tx.ids)
                x2[q] = z
                ref = nat.reference([x2], tx.targets)
                refs.append((ref, float(carry.kl(ref, tx.ref)[0])))
            acc, wsum = np.zeros(1 + len(offs)), sum(w for _, w in refs)
            for s0 in (range(0, 1 + len(offs), LINE_CHUNK) if got else []):  # the lines a chunk at a time: one call for
                rows = list(range(s0, min(1 + len(offs), s0 + LINE_CHUNK)))    # VPD's whole answer asked 13-30 GiB (fleet)
                lines = torch.ones(len(rows), len(col), device=nat.dev)
                for i, r in enumerate(rows):
                    if r:
                        lines[i, offs[r - 1]] = 0.0
                lq = E.path_patched([g], [tuple(pz) for pz in got], lines=lines)  # [rows, D, targets, V]
                for d, (ref, w) in enumerate(refs):
                    acc[s0:s0 + len(rows)] += w * np.asarray(carry.kl(ref.expand(len(rows), *ref.shape[1:]), lq[:, d]).tolist())
                del lq
                native.free(nat.dev)
            kl = list(acc / wsum) if wsum > 0 else [None] * (1 + len(offs))
        else:
            raise ValueError("mode is pp or c")
        base = kl[0]
        d = lambda k: None if k is None or base is None else k - base  # noqa: E731
        rows_n = kl[1:1 + len(nodes)]
        rows_e = kl[1 + len(nodes):1 + len(nodes) + len(edges)]
        rows_t = kl[1 + len(nodes) + len(edges):1 + len(nodes) + len(edges) + len(toks)]
        rows_p = kl[1 + len(nodes) + len(edges) + len(toks):]
        tok_at = {t for t, _ in toks}
        f2 = sorted({q for q, _ in tx.f2}) if tx.f2 is not None else []
        moved = np.asarray(tx.moved, dtype=float) if f2 else None
        return {"mode": mode, "base_kl": base,
                "nodes": [{"node": list(nd), "delta_kl": d(k),
                           "vpd_active": bool(tx.vpd[nd[0]][nd[1]].any() if nd[2] == BLOCK else tx.vpd[nd[0]][nd[1], nd[2]]) if tx.vpd is not None else None}
                          for nd, k in zip(nodes, rows_n)],
                "edges": [{"edge": [list(w), "out" if r is None else list(r)], "delta_kl": d(k)} for (w, r), k in zip(edges, rows_e)],
                "token_edges": [{"edge": [t, "out" if r is None else list(r)], "delta_kl": d(k)} for (t, r), k in zip(toks, rows_t)] if mode == "pp" else [],
                "pairs": [{"pair": [L, t, "rest" if qa is None else list(qa), "rest" if kb is None else list(kb)], "delta_kl": d(k)}
                          for ((L, t), (qa, kb)), k in zip(prs, rows_p)] if mode == "pp" else [],
                "positions": [{"position": q, "moved": float(moved[q - 1]), "has_node": any(nd[1] == q for nd in nodes),
                               "has_token_edge": q in tok_at or g.complete} for q in f2]}

    @torch.no_grad()
    def pattern_family(self, tx: Text, graphs: list[native.Graph], runs=None) -> list[dict]:
        """s_pattern (the lead's 10-12, rustmove's 35f6ca0: C's semantics flatten attention, the q/k remainders
        about 1% of the scores, so circuits fit near-uniform rows): each circuit's run in C's semantics (graph_ref)
        with the reads it uses (a layer and position holding a circuit attention output) given the real model's
        attention row on the text, all of them at once and each alone; VPD's answer the same; the mean KLs from the
        model on C's log scale against KL_empty (s_pattern, _raw without the floor at 0). A circuit whose output
        survives the model's true pattern uses attention as the model does. None for a circuit without an attention
        output. The circuits' runs batched on the engine (_pattern_runs; runs: another such function, line_loo's one
        plan), VPD's kept per read set."""
        b = self._pp_base(tx)
        T = tx.T
        causal = torch.ones(T, T, dtype=torch.bool, device=self.nat.dev).tril()
        e0, v0 = self._bounds(tx, "live")
        vpd = native.Graph(masks=tx.vpd) if tx.vpd is not None else None
        kept = tx.__dict__.setdefault("_pattern_vpd", {})  # (VPD's answer, reads given) -> its KL
        plans, out = [], []
        for g in graphs:  # each circuit's draws: every read given the model's row, and each alone
            reads = sorted({(native._name_layer_kind(n)[0], t) for n, t, _ in g.node_set() if native._name_layer_kind(n)[1] == "o_proj"})
            real = {(i, t): b["S"][i][:, t].masked_fill(~causal[t], float("-inf")).softmax(-1)[:, :t + 1] for i, t in reads}
            plans.append((reads, [real] + ([{r: real[r]} for r in reads] if len(reads) > 1 else [])))
        kcs = (runs or self._pattern_runs)(tx, graphs, plans)
        new = {}  # VPD's runs with those reads given: the same for every circuit using them (kept on tx), the new ones
        for reads, draws in plans:  # at once on an engine over its subcomponents (graph_ref's numbers)
            for d in (draws if reads and vpd is not None else ()):
                key = (id(tx.vpd), tuple(sorted(d)))
                if key not in kept:
                    new.setdefault(key, d)
        if new:
            nat, dev = self.nat, self.nat.dev
            E = self._circuit_engine(tx.ids, tx.targets, [], extra={n: tx.vpd[n].any(0).nonzero().flatten() for n in nat.names})
            u = {n: tx.vpd[n][:, E.columns[n]].float()[None] for n in nat.names}
            keys = list(new)
            for s0 in range(0, len(keys), self.PATTERN_ROWS):
                part = keys[s0:s0 + self.PATTERN_ROWS]
                lq = E.graph_log_probs(E.plan([vpd] * len(part), semantics="c"), u, rows=[new[k] for k in part])
                kept.update(zip(part, carry.kl(tx.ref, lq).tolist()))
                native.free(dev)
        for g, (reads, draws), kc in zip(graphs, plans, kcs):
            if not reads or vpd is None:
                out.append({"s_pattern": None, "s_pattern_raw": None, "kl_pattern_c": None, "kl_pattern_vpd": None, "reads_pattern": len(reads)})
                continue
            kv = [kept[(id(tx.vpd), tuple(sorted(d)))] for d in draws]
            c_, v_ = float(np.mean(kc)), float(np.mean(kv))
            out.append({"s_pattern": _fid(e0, c_, v_), "s_pattern_raw": _fid_raw(e0, c_, v_), "kl_pattern_c": c_,
                        "kl_pattern_vpd": v_, "reads_pattern": len(reads), "draws_pattern": [list(x) for x in zip(kc, kv)]})
        return out

    PATTERN_ROWS = 32  # pattern_family's engine rows (circuits x draws) a call
    LOO_FAST = True  # line_loo's point, wiring and pattern from one plan of the answer (_loo_c); False: per deletion
    LOO_REUSE = True  # _loo_c's runs from the circuit's first position on (the states before it kept once: Engine reuse)
    LOO_ENTRIES = 1 << 27  # _loo_c's engine rows a call: this many floats in a run's widest activations (rows x T x C)

    @torch.no_grad()
    def _loo_c(self, tx: Text, g: native.Graph, lines: list, keep: torch.Tensor, flat: list | None = None,
               draws: int = DRAWS, patching: tuple | None = None) -> tuple | None:
        """phi_variants' C-semantics families for every variant (keep [K, L] over lines) of a graph without blocks on
        one plan of it (Engine.plan, semantics "c"): a stated graph's variant runs as the graph at per-row strengths
        (a deleted connection at 0; a deleted node with every connection into or out of it at 0 and taken out of the
        circuit, graph_log_probs' off; a token edge, which C's semantics does not have, the graph's own), a complete
        graph's as the graph at per-row u (1 on its nodes, 0 at the deleted ones): the same numbers as sufficiency on
        the variants; the wired runs (a complete graph's: its point) per distinct node set (masked runs). (point and
        wired rows with sufficiency's live keys, the pattern runs for pattern_family's runs, and with flat (the variants'
        graphs) each variant's node family as _node_family gives it: its own draws of its nodes removed from the
        model, from VPD's answer and from the variant, every variant's draws on the same three engines); None when the
        plan's connections are not the graph's. Point, wiring and pattern run as stated (no semantics holds them at x);
        the node family at its own (_loo_nodes: at x under "n", a deleted line's nodes outside the variant held at x, a
        node draw's at 0: fidelity's numbers on each variant); the states before the circuit's first position kept from
        one run in each (LOO_REUSE)."""
        nat, dev = self.nat, self.nat.dev
        edges = [] if g.complete else [(w, r) for r, ws in g.parents.items() for w in ws]
        outs = [] if g.complete else list(g.out)
        E = self._circuit_engine(tx.ids, tx.targets, [g])
        plan = E.plan(g, semantics="c")
        if plan["E"] != len(edges) or plan["O"] != len(outs):
            return None
        at = {n: {c: i for i, c in enumerate(E.columns[n].tolist())} for n in nat.names}
        u1 = {n: torch.zeros(1, tx.T, E.width(n), device=dev) for n in nat.names}  # a complete graph's u: its nodes at 1
        for n, t, c in (g.node_set() if g.complete else ()):
            u1[n][0, t, at[n][c]] = 1.0
        touch = {}
        for j, (w, r) in enumerate(edges):
            touch.setdefault(w, []).append(j)
            touch.setdefault(r, []).append(j)
        eix, oix = {e: j for j, e in enumerate(edges)}, {w: j for j, w in enumerate(outs)}
        K = keep.shape[0]
        ew, ow, off, sets = torch.ones(K, len(edges)), torch.ones(K, len(outs)), [None] * K, [frozenset()] * K
        for i in range(K):
            dn = []
            for j in (~keep[i]).nonzero().flatten().tolist():
                kind, x = lines[j]
                if kind == "node":  # out of the circuit with its connections (graph_log_probs' off)
                    dn.append(x)
                elif kind == "edge":
                    if x[1] is None:
                        ow[i, oix[x[0]]] = 0.0
                    else:
                        ew[i, eix[tuple(x)]] = 0.0
            off[i], sets[i] = dn or None, frozenset(dn)

        def urows(gone):  # a complete graph's u for rows deleting the node sets gone: 0 at each one's
            u = {n: m.expand(len(gone), -1, -1).clone() for n, m in u1.items()}
            idx = {}
            for r, d in enumerate(gone):
                for n, t, c in d:
                    x = idx.setdefault(n, ([], [], []))
                    x[0].append(r)
                    x[1].append(t)
                    x[2].append(at[n][c])
            for n, x in idx.items():  # one indexed write per site
                u[n][tuple(torch.tensor(i, device=dev) for i in x)] = 0.0
            return u

        lo = min((t for _, t, _ in g.node_set()), default=0) if self.LOO_REUSE else 0  # before the circuit's first position every
        lo = lo if 0 < lo <= min(tx.targets) else 0                        # variant's run is the same: kept once, reused
        kn = self._xkw(tx.ids, tx.targets, "nodes") if flat is not None else {}  # the node family's semantics (x under "n")

        def keep_run(kw):  # the states of one run of the whole graph at kw's semantics (before lo every row's; at x, a
            st = {}        # stated graph's held: every column outside the circuit)
            if g.complete:
                E.graph_log_probs(plan, u1, keep=st, **kw)
            else:
                held = {n: torch.ones(1, 1, E.width(n), dtype=torch.bool, device=dev) for n in kw["at_x"]} if kw else None
                E.graph_log_probs(plan, ew=torch.ones(1, len(edges), device=dev), ow=torch.ones(1, len(outs), device=dev), keep=st,
                                  **(dict(kw, held=held) if kw else {}))
            return st
        states = {"c": keep_run({}), "nodes": keep_run(kn) if kn else None} if lo else {}  # (C's: point, wiring, pattern)
        tg = nat.target  # engine rows a call: a row's widest activations from lo on and, reusing, its kept keys and values
        step = max(4, self.LOO_ENTRIES // ((tx.T - lo) * max(nat.C.values()) + (2 * tg.n_layer * tx.T * tg.d_model if lo else 0)))

        def lq_rows(ix, given=None, tables=None, held=None, kw=None):  # log q of rows ix of (ew, ow, off, sets), or of tables'
            # rows (ew, ow and off for a stated graph, u for a complete one), optionally with given attention rows; kw: the
            # node family's semantics (_loo_nodes) with its held at x, else as stated
            if tables is None:
                tables = (ew[ix], ow[ix], [off[i] for i in ix]) if not g.complete else urows([sets[i] for i in ix])
            R = next(iter(tables.values())).shape[0] if g.complete else len(tables[0])
            st = states.get("nodes" if kw else "c")  # (the states kept at kw's semantics)
            reuse = (E.tile_states(st, R), tx.ids, lo) if st is not None else None
            hk = dict(kw, held=held) if kw else {}
            if g.complete:
                return E.graph_log_probs(plan, tables, rows=given, reuse=reuse, **hk)
            e_, o_, f_ = tables
            return E.graph_log_probs(plan, ew=e_.to(dev), ow=o_.to(dev), off=f_, rows=given, reuse=reuse, **hk)

        def run(rows_of, extra=None):  # KLs of rows (line indices), optionally with given attention rows each
            out = []
            for s0 in range(0, len(rows_of), step):
                ix = rows_of[s0:s0 + step]
                out += carry.kl(tx.ref, lq_rows(ix, None if extra is None else extra[s0:s0 + step])).tolist()
                native.free(dev)
            return out
        kl = run(list(range(K)))
        if g.complete:  # its wiring is itself
            kw0 = {x: k for x, k in zip(sets, kl)}
        e, v = self._bounds(tx, "live")
        nodes = sorted(g.node_set())
        keys = list(dict.fromkeys(sets))  # the variants' node sets: the graph's less each one's deleted nodes
        full = carry.masks(nat, tx.T, nodes)

        def masks(drop):  # the graph's node set less drop, as masks
            m = {n: v.clone() for n, v in full.items()}
            for n, t, c in drop:
                m[n][t, c] = False
            return m
        kw = dict(kw0) if g.complete else {}
        for s0 in range(0, 0 if g.complete else len(keys), 8):  # a stated graph's wiring: the masked run of its node set
            part = keys[s0:s0 + 8]
            got = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, [masks(k) for k in part], True)).tolist()
            kw.update(zip(part, got))
            native.free(dev)
        point = [{"kl_live": kl[i], "f1_live": _clip01(recovered_log(e, kl[i], v))} for i in range(K)]
        wired = [{"kl_live": kw[sets[i]], "f1_live": _clip01(recovered_log(e, kw[sets[i]], v))} for i in range(K)]

        def pattern_runs(tx_, graphs, plans):  # pattern_family's circuit side on the same plan: row i's draws
            todo = [(i, d) for i, (reads, ds) in enumerate(plans) if reads for d in ds]
            got = run([i for i, _ in todo], [d for _, d in todo])
            out = [[] for _ in graphs]
            for (i, _), k in zip(todo, got):
                out[i].append(k)
            return out
        nodes_rows = None if flat is None else self._loo_nodes(tx, E, flat, draws, ew, ow, off, sets, touch, oix, lq_rows, step)
        patch_rows = None if patching is None else self._loo_patching(tx, E, plan, g, ew, ow, off, sets, at, step, *patching)
        return point, wired, pattern_runs, nodes_rows, patch_rows

    @torch.no_grad()
    def _loo_patching(self, tx: Text, E, plan: dict, g: native.Graph, ew, ow, off, sets, at, step, split: str, base: dict) -> list[dict]:
        """_loo_c's activation patching per variant (donors()' families on split's donors, as donors() gives them on
        each variant): the variant's row of the answer's plan with every subcomponent outside it at one donor's
        activation (u 1 elsewhere: a shorter donor's first positions live), on E's columns by patch and the columns
        left out of E as their fixed writes (add; live before a shorter donor's start: rest); KL_VPD and KL_0 per donor
        are the answer's row's (base: the same donors, nothing of them depends on the circuit)."""
        nat, dev, T = self.nat, self.nat.dev, tx.T
        K = len(sets)
        step = max(1, min(step, self.LOO_ENTRIES // (T * max(nat.C.values()))))  # (every position runs)
        texts = self._donor_texts(tx, split, DONORS_NEAR, self.far_donors(tx, self.far_pool(split), DONORS_FAR))
        acts = tx.__dict__.setdefault("_donor_acts", {})  # donor token ids -> donor_activations
        left, WnE = {}, {}
        for n in nat.names:
            st = nat.target.site(n)
            m = torch.ones(nat.C[n], dtype=torch.bool, device=dev)
            m[E.columns[n]] = False
            left[n] = m
            WnE[n] = st.W.T - st.delta_T - E.V[n] @ E.U[n]  # the left-out columns' V U, [d_in, d_out]
        own = {n: torch.zeros(T, E.width(n), dtype=torch.bool, device=dev) for n in nat.names}  # the answer's nodes
        for n, t, c in g.node_set():
            own[n][t, at[n][c]] = True
        ones = {n: torch.ones(1, 1, E.width(n), device=dev) for n in nat.names}
        out = [{} for _ in range(K)]
        for pool, xs in texts.items():
            held = {repr(d[0]): d for d in base.get(f"draws_patching_{pool}") or []}  # name -> [name, KL_C, KL_VPD, KL_0, s]
            if not xs or len(held) != len(xs) or any(repr(nm) not in held for nm, _ in xs):
                continue  # (the answer's row lacks these donors)
            kc = [[] for _ in range(K)]
            for name, x in xs:
                key = tuple(x)
                if key not in acts:
                    acts[key] = self.donor_activations(x, T)
                a, pad = acts[key]
                vals = {n: a[n][:, E.columns[n]].to(dev)[None] for n in nat.names}
                add = {}
                for n in nat.names:
                    w = a[n][:, left[n]].to(dev) @ nat.target.site(n).U[left[n]]
                    w[:pad] = 0.0
                    add[n] = w[None]
                rest = {n: (WnE[n], (torch.arange(T, device=dev) < pad)[None]) for n in nat.names} if pad else None
                for s0 in range(0, K, step):
                    ix = list(range(s0, min(K, s0 + step)))
                    patch = {}
                    for n in nat.names:
                        keep_ = (~own[n])[None].repeat(len(ix), 1, 1)
                        for r, i in enumerate(ix):  # a variant's deleted nodes are outside it: patched
                            for m_, t, c in sets[i]:
                                if m_ == n:
                                    keep_[r, t, at[n][c]] = True
                        if pad:
                            keep_[:, :pad] = False
                        patch[n] = (keep_, vals[n])
                    if g.complete:  # (u 1 on every row: the patch says what is the donor's)
                        lq = E.graph_log_probs(plan, {n: o.expand(len(ix), -1, -1) for n, o in ones.items()}, patch=patch, add=add, rest=rest)
                    else:
                        lq = E.graph_log_probs(plan, ones, ew=ew[ix].to(dev), ow=ow[ix].to(dev), off=[off[i] for i in ix],
                                               patch=patch, add=add, rest=rest)
                    for r, k in zip(ix, carry.kl(tx.ref, lq).tolist()):
                        kc[r].append(k)
                    native.free(dev)
            kv = [held[repr(nm)][2] for nm, _ in xs]
            k0 = [held[repr(nm)][3] for nm, _ in xs]
            for i in range(K):
                c_ = kc[i]
                per = [_fid(z, c, v) for z, c, v in zip(k0, c_, kv)]
                each = [y for y in per if y is not None]
                out[i].update({f"s_patching_{pool}": _fid(float(np.mean(k0)), float(np.mean(c_)), float(np.mean(kv))),
                               f"s_patching_{pool}_worst": min(each) if each else None,
                               f"s_patching_{pool}_p90": float(np.percentile(each, 10)) if each else None,
                               f"kl_patching_{pool}": (float(np.mean(c_)), float(np.mean(kv)), float(np.mean(k0))),
                               f"donors_{pool}": len(c_),
                               f"draws_patching_{pool}": [[nm, c, v, z, sd] for (nm, _), c, v, z, sd in zip(xs, c_, kv, k0, per)]})
        return out

    @torch.no_grad()
    def _loo_nodes(self, tx: Text, E, flat: list, draws: int, ew, ow, off, sets, touch, oix, lq_rows, step) -> list[dict]:
        """_loo_c's node family per variant (flat: the variants' graphs), as _node_family gives it (no reads): the
        common draws (_node_draws: each node's own uniforms, so a variant's draws are the answer's less its deleted
        nodes) of its nodes removed from the model (an outside-on engine from the draws' first position on), from VPD's answer (from there on, VPD's run before it)
        and from the variant (its row of _loo_c's plan with the draw's nodes taken out too: lq_rows), every variant's
        draws on the same engines, each row's tables as matrices over the nodes. The model's run once per distinct
        dropped set, VPD's once per distinct dropped set within its answer (a node outside VPD's answer at its position
        is 0 in VPD's run already). Semantics "x" (_xkw): VPD's answer and
        the variants at every position, a draw's nodes at 0, a variant's deleted lines' at x (outside it)."""
        import engine

        nat, dev, T = self.nat, self.nat.dev, tx.T
        kx = self._xkw(tx.ids, tx.targets, "nodes")
        circ = E.columns
        Em = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ, outside="on")
        vcols = {n: torch.unique(torch.cat([circ[n], tx.vpd[n].any(0).nonzero().flatten()])) for n in nat.names}
        Ev = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=vcols)
        at = {n: {c: i for i, c in enumerate(circ[n].tolist())} for n in nat.names}
        vat = {n: {c: i for i, c in enumerate(vcols[n].tolist())} for n in nat.names}
        vbase = {n: tx.vpd[n][:, vcols[n]].float()[None] for n in nat.names}
        nodes = sorted(set().union(*(h.node_set() for h in flat)) | set().union(*sets))
        nid = {nd: i for i, nd in enumerate(nodes)}
        N, K = len(nodes), len(flat)
        order = {n: sorted((nd for nd in nodes if nd[0] == n), key=lambda nd, n=n: (nd[1], at[n][nd[2]])) for n in nat.names}
        order = {n: o for n, o in order.items() if o}  # per site its nodes in the plan's entry order
        sidx = {n: torch.tensor([nid[nd] for nd in o], device=dev) for n, o in order.items()}
        oc, ov, KC, KV = {}, {}, 0, 0  # each site's offset in the flat masks over every site's columns (circuit's, VPD's)
        for n in nat.names:
            oc[n], ov[n] = KC, KV
            KC, KV = KC + len(circ[n]), KV + len(vcols[n])
        npos = np.array([nd[1] for nd in nodes], dtype=np.int64)
        FT = torch.as_tensor(npos, device=dev)  # per node its position and its flat columns (circuit's, VPD's)
        FC = torch.tensor([oc[n] + at[n][c] for n, _, c in nodes], dtype=torch.long, device=dev)
        FV = torch.tensor([ov[n] + vat[n][c] for n, _, c in nodes], dtype=torch.long, device=dev)

        def views(flat, off, cols):  # a flat mask [R, T, KC or KV] as the engine's {site: [R, T, K]} (views)
            return {n: flat[..., off[n]:off[n] + len(cols[n])] for n in nat.names}
        vflat = torch.cat([vbase[n] for n in nat.names], -1)  # [1, T, KV]
        vdel = torch.zeros(K, N, dtype=torch.bool, device=dev)
        for v, sv_ in enumerate(sets):
            if sv_:
                vdel[v, [nid[nd] for nd in sv_]] = True
        uniq, uv = {}, []  # the variants' distinct node sets (a deleted connection or token edge keeps the answer's)
        for h in flat:
            uv.append(uniq.setdefault(tuple(sorted(h.node_set())), len(uniq)))
        udrops = self._subset_draws(tx, nodes, nid, list(uniq), draws)
        drops, owner = [], []  # every row's dropped node indices, and its (variant, draw): _node_draws' common draws
        for v in range(len(flat)):
            for j, d in enumerate(udrops[uv[v]]):
                drops.append(d)
                owner.append((v, j))
        R = len(owner)
        res = [[] for _ in range(K)]
        cat = np.concatenate(drops) if drops else np.zeros(0, dtype=np.int64)
        first = int(npos[cat].min()) if len(cat) else T
        start = min(first, min(tx.targets))  # every run's states before it: the model's, VPD's and the variants'
        step = max(8, min(step, self.LOO_ENTRIES // (T * max(1, sum(len(c) for c in vcols.values())))))  # (VPD's masks)
        keys_all = [tuple(d.tolist()) for d in drops]  # the model's and VPD's runs depend on a row's dropped nodes alone: one
        seen_n = {}                                   # run per distinct set (common draws: a variant's draw is the answer's
        for x in keys_all:                            # less its deleted node, so the variants whose node a draw leaves alone
            seen_n[x] = seen_n.get(x, 0) + 1          # share it), a set asked by several rows kept across chunks
        shared = {}  # dropped set -> (log q of the model's run [targets, V], KL_VPD, W)
        act = (vflat[0, FT, FV] > 0).cpu().numpy()  # per node: in VPD's answer at its position
        vkey = {x: tuple(i for i in x if act[i]) for x in seen_n}  # VPD's run sees a dropped set's nodes in its answer alone
        vleft, vshared = {}, {}  # per VPD set: the distinct dropped sets still to run with it; its log q, while asked for
        for x in seen_n:
            vleft[vkey[x]] = vleft.get(vkey[x], 0) + 1
        for s0 in range(0, R, step):
            rows = range(s0, min(R, s0 + step))
            rr = np.concatenate([np.full(len(drops[k]), r) for r, k in enumerate(rows)])
            cc = np.concatenate([drops[k] for k in rows])
            Dm = torch.zeros(len(rows), N, dtype=torch.bool, device=dev)
            r_, j_ = torch.as_tensor(rr, device=dev), torch.as_tensor(cc, device=dev)
            Dm[r_, j_] = True
            vs = torch.tensor([owner[k][0] for k in rows], device=dev)
            gone = Dm | vdel[vs]  # the variant's deleted nodes and the draw's
            keys = [keys_all[k] for k in rows]
            fresh = [x for x in dict.fromkeys(keys) if x not in shared]
            if fresh:
                ur = np.concatenate([np.full(len(x), r) for r, x in enumerate(fresh)])
                uc = np.concatenate([np.array(x, dtype=np.int64) for x in fresh])
                ur_, uj_ = torch.as_tensor(ur, device=dev), torch.as_tensor(uc, device=dev)
                mf = torch.ones(len(fresh), T, KC, device=dev)  # each draw's nodes off: one indexed write
                mf[ur_, FT[uj_], FC[uj_]] = 0.0
                lm_f = Em.log_probs(views(mf, oc, circ), start=start) if start else Em.log_probs(views(mf, oc, circ))
                want = [vkey[x] for x in fresh]  # VPD's runs: one per distinct VPD set (dropped sets that differ only
                vnew = [y for y in dict.fromkeys(want) if y not in vshared]  # outside its answer run it alike)
                if vnew:
                    vr = np.concatenate([np.full(len(y), r, dtype=np.int64) for r, y in enumerate(vnew)])
                    vc = np.concatenate([np.array(y, dtype=np.int64) for y in vnew])
                    vr_, vj_ = torch.as_tensor(vr, device=dev), torch.as_tensor(vc, device=dev)
                    vf = vflat.repeat(len(vnew), 1, 1)
                    vf[vr_, FT[vj_], FV[vj_]] = 0.0
                    mv = views(vf, ov, vcols)
                    if kx:  # VPD's answer at x, its draw's nodes at 0
                        vk = dict(kx, held={n: vbase[n] == 0 for n in kx["at_x"]})
                        lv_n = Ev.log_probs(mv, start=start, base=vbase, **vk) if start else Ev.log_probs(mv, **vk)
                    else:
                        lv_n = Ev.log_probs(mv, start=start, base=vbase) if start else Ev.log_probs(mv)
                at_v = {y: i for i, y in enumerate(vnew)}
                lv_f = torch.stack([vshared[y] if y in vshared else lv_n[at_v[y]] for y in want])
                for y in want:
                    vleft[y] -= 1
                for y, i in at_v.items():
                    if vleft[y] > 0:
                        vshared[y] = lv_n[i]
                for y in set(want):
                    if not vleft[y]:
                        vshared.pop(y, None)
                kv_f, w_f = carry.kl(lm_f, lv_f), carry.kl(lm_f, tx.ref.expand_as(lm_f))
            at_f = {x: i for i, x in enumerate(fresh)}
            for x, i in at_f.items():
                if seen_n[x] > 1:
                    shared[x] = (lm_f[i], kv_f[i], w_f[i])
            got = [shared[x] if x in shared else (lm_f[at_f[x]], kv_f[at_f[x]], w_f[at_f[x]]) for x in keys]
            lm = torch.stack([g_[0] for g_ in got])
            held = None
            if kx:  # the variant's deleted nodes at x, the draw's at 0
                hf = torch.ones(len(rows), T, KC, dtype=torch.bool, device=dev)
                hf[r_, FT[j_], FC[j_]] = False
                held = {n: x for n, x in views(hf, oc, circ).items() if n in kx["at_x"]}
            if flat[0].complete:
                uf = torch.zeros(len(rows), T, KC, device=dev)
                r2, j2 = (~gone).nonzero(as_tuple=True)
                uf[r2, FT[j2], FC[j2]] = 1.0
                u = views(uf, oc, circ)
                lc = lq_rows(None, tables=u, held=None if held is None else {n: h & (u[n] == 0) for n, h in held.items()}, kw=kx)
            else:  # the dropped nodes out of the circuit with their connections, as a deleted node
                lc = lq_rows(None, tables=(ew.to(dev)[vs], ow.to(dev)[vs], {n: gone[:, sidx[n]] for n in order}), held=held, kw=kx)
            kc, kv = carry.kl(lm, lc).tolist(), torch.stack([g_[1] for g_ in got]).tolist()
            w, claim = torch.stack([g_[2] for g_ in got]).tolist(), carry.kl(tx.ref.expand_as(lc), lc).tolist()
            for r, k in enumerate(rows):
                v, j = owner[k]
                res[v].append([j, kc[r], kv[r], w[r], claim[r]])
            native.free(dev)
        out = []
        for v in range(K):
            dr = res[v]
            if not dr:
                e0, v0 = self._bounds(tx, "live")
                out.append({"kl_nodes_c": e0, "kl_nodes_vpd": v0, "reads_nodes": None})
                continue
            out.append({"kl_nodes_c": float(np.mean([x[1] for x in dr])), "kl_nodes_vpd": float(np.mean([x[2] for x in dr])),
                        "draws_nodes": dr, "overstate_nodes": float(np.mean([x[4] for x in dr])) / max(float(np.mean([x[3] for x in dr])), 1e-12),
                        "reads_nodes": None})
        return out

    @torch.no_grad()
    def _pattern_runs(self, tx: Text, graphs: list[native.Graph], plans: list) -> list[list[float]]:
        """pattern_family's circuit side: per graph, each draw's KL from the model (graph_ref(rows=)'s numbers), on the
        engine (C's semantics with the draws' attention rows given, PATTERN_ROWS a call; a complete graph at u, 1 on its
        nodes)."""
        out = [[] for _ in graphs]
        todo = [(gi, d) for gi, (g, (reads, draws)) in enumerate(zip(graphs, plans)) if reads and tx.vpd is not None for d in draws]
        if todo:
            nat, dev = self.nat, self.nat.dev
            flat = {gi: expand_blocks(graphs[gi], nat.C, tx.vpd) for gi in {gi for gi, _ in todo}}
            E = self._circuit_engine(tx.ids, tx.targets, list(flat.values()))
            at = {n: {c: i for i, c in enumerate(E.columns[n].tolist())} for n in nat.names}
            whole = {}  # a complete graph's nodes per site: (positions, columns)
            for gi, h in flat.items():
                if h.complete:
                    per = {}
                    for n, t, c in h.node_set():
                        per.setdefault(n, ([], []))
                        per[n][0].append(t)
                        per[n][1].append(at[n][c])
                    whole[gi] = per
            todo = [x for x in todo if x[0] not in whole] + [x for x in todo if x[0] in whole]  # (each kind its own batches:
            cut = sum(x[0] not in whole for x in todo)                                         # a stated graph's numbers
            starts = list(range(0, cut, self.PATTERN_ROWS)) + list(range(cut, len(todo), self.PATTERN_ROWS))  # as alone)
            for s0 in starts:
                part = todo[s0:min(s0 + self.PATTERN_ROWS, cut if s0 < cut else len(todo))]
                u = None
                if any(gi in whole for gi, _ in part):  # complete graphs run as u: 1 on their nodes
                    u = {}
                    for n in nat.names:
                        idx = [(r, t, k) for r, (gi, _) in enumerate(part) for t, k in zip(*whole.get(gi, {}).get(n, ([], [])))]
                        u[n] = torch.zeros(len(part), tx.T, E.width(n), device=dev)
                        if idx:
                            u[n][tuple(torch.tensor(idx, device=dev).T)] = 1.0
                lq = E.graph_log_probs(E.plan([flat[gi] for gi, _ in part], semantics="c"), u, rows=[d for _, d in part])
                for (gi, _), k in zip(part, carry.kl(tx.ref, lq).tolist()):
                    out[gi].append(k)
                native.free(dev)
        return out

    @torch.no_grad()
    def _pp_base(self, tx: Text) -> dict:
        """The real model's run on the text, the path-patched runs' x side (kept on tx): every site's activations
        [T, C] and remainder outputs, each layer's streams before its attention and its MLP, rotated queries and keys
        [H, T, hd], values [T, H hd], scores (_ScoreRows: [H, T, T] indexed, each row made when asked), MLP inputs, the
        final stream."""
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
            out = A @ tg.site(n).U + rem
            return out if tg.site(n).b is None else out + tg.site(n).b  # a bias, always on (vpd4l: none)

        x = tg.embed(torch.tensor(ids, device=nat.dev)).float()
        for i in range(tg.n_layer):
            nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
            b["Xa"].append(x)
            z = tg.norm(x, 2 * i)
            q, k, v = (site(nm[kk], z) for kk in ("q_proj", "k_proj", "v_proj"))
            qr, kr = (tg._rope(m.view(1, T, H, hd).transpose(1, 2), T)[0] for m in (q, k))
            b["q"].append(qr)
            b["k"].append(kr)
            b["v"].append(v)
            b["S"].append(_ScoreRows(qr, kr))
            att = torch.nn.functional.scaled_dot_product_attention(qr, kr, v.view(T, H, hd).transpose(0, 1), is_causal=True)
            att = att.transpose(0, 1).reshape(T, -1)
            x = x + site(nm["o_proj"], att)
            b["Xm"].append(x)
            pre = site(nm["c_fc"], tg.norm(x, 2 * i + 1))
            b["pre"].append(pre)
            x = x + site(nm["down_proj"], tg.act(pre))
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
        demb = {t: (tg.token_embed(torch.tensor(x2[t], device=dev)) - tg.token_embed(torch.tensor(tx.ids[t], device=dev))).float()
                for t in range(T) if x2[t] != tx.ids[t]}
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
                if tg.site(n).b is not None:  # a bias, always on (vpd4l: none)
                    out = out + tg.site(n).b
                if n in held_out:
                    out = out.clone()
                    for t, _, val in held_out[n]:
                        out[t] = val
                return out

            x = tg.embed(torch.tensor(x2, device=dev)).float()
            for i in range(tg.n_layer):
                nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
                z = tg.norm(x, 2 * i)
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
                x = x + site(nm["down_proj"], tg.act(site(nm["c_fc"], tg.norm(x, 2 * i + 1))))
            return torch.log_softmax(tg.unembed(tg.norm(x[tx.targets], -1)), -1)[None]
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

        def full(n, z):  # a site's whole map (every subcomponent, the remainder and any bias)
            out = (z @ tg.site(n).V) @ tg.site(n).U + z @ tg.site(n).delta_T
            return out if tg.site(n).b is None else out + tg.site(n).b

        def x_out(w):  # a circuit writer's output on the text
            n, t, c = w
            if c != BLOCK:
                return b["a"][n][t, c] * tg.site(n).U[c]
            out = b["a"][n][t] @ tg.site(n).U + b["rem"][n][t]
            return out if tg.site(n).b is None else out + tg.site(n).b

        def delta(w):  # a circuit writer's output change, out^P - out^x, in its site's output space
            n, t, c = w
            return (aP[w] if c == BLOCK else aP[w] * tg.site(n).U[c]) - x_out(w)

        def apply(r, z):  # a node's patched value from its patched input z
            return full(r[0], z) if r[2] == BLOCK else z @ tg.site(r[0]).V[:, r[2]]

        def resid(r, X, j):  # a residual reader's patched value (norm j)
            inp = X[r[1]].clone()
            for w in par.get(r, []):
                if w in aP:
                    inp = inp + delta(w)
            if r in tok:
                inp = inp + tok[r]
            return apply(r, tg.norm(inp, j))

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
                    aP[r] = held[r] if r in held else resid(r, b["Xa"][i], 2 * i)
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
                    aP[r] = held[r] if r in held else resid(r, b["Xm"][i], 2 * i + 1)
            for r in here:
                if kind[r] == "down_proj":
                    pre = b["pre"][i][r[1]].clone()
                    for w in par.get(r, []):
                        if w in aP and kind.get(w) == "c_fc":
                            pre = pre + delta(w)
                    aP[r] = held[r] if r in held else apply(r, tg.act(pre))
        f = b["Xf"][tx.targets].clone()
        for j, t in enumerate(tx.targets):
            for w in graph.out:
                if w[1] == t and w in aP:
                    f[j] = f[j] + delta(w)
            if t in demb and t in getattr(graph, "tok_out", []):  # the token's embedding into the prediction, stated
                f[j] = f[j] + demb[t]
        return torch.log_softmax(tg.unembed(tg.norm(f, -1)), -1)[None]

    def _pp_engine(self, tx: Text):
        """The text's engine for path-patched runs (engine.Engine.path_patched: pp_run's numbers for many graphs and
        prompts at once), kept on tx with its real-model run."""
        import engine

        if tx.__dict__.get("_pp_engine") is None:
            tx.__dict__["_pp_engine"] = engine.Engine(self.nat, tx.ids, tx.targets, live=True)
        return tx.__dict__["_pp_engine"]

    @torch.no_grad()
    def track_pp(self, tx: Text, graphs: list[native.Graph], split: str = "eval", draws: int = CHANGED_DRAWS,
                 cuts: list | None = None, engine_run: bool = True, strengths: torch.Tensor | None = None,
                 lines: torch.Tensor | None = None, got: list | None = None) -> list[dict]:
        """s_track_pp (the lead's 10-12, beside s_track): on changed_inputs' changed texts x' (split), the path-patched
        run of each circuit (pp_run) against the model on x', against VPD's whole answer's path-patched run (a
        complete graph: 1 by construction), bounded by no change: per prompt clip(ln(W / KL_C) / ln(W / KL_V), 0, 1)
        with W = KL(model(x') || model(x)) (the circuit's run on x is the model's, so this is s_track's transplant),
        none where VPD's run is no nearer than W, pooled weighted by W. Per circuit s_track_pp, kl_track_pp_c,
        prompts_track_pp and draws_track_pp ([prompt, KL_C, KL_V, W] each). cuts: per graph pp_run's cuts (None: none).
        The circuits run at once on engine.Engine.path_patched (monitorbench's 072dc07, an engine kept on the text;
        engine_run False, or a stated graph with cuts: pp_run one graph and prompt at a time); each prompt's model run,
        W and VPD's KL are kept on the text. strengths [K, E + O]: the circuits are graphs[0]'s connection subsets
        (changed_edges' ons), in one call; lines [K, L]: graphs[0] (stranded) with the lines each row turns off removed
        (Engine.pp_lines' columns; graph_of_lines), in one call. got: the changed texts as (position, token) pairs in
        place of changed_inputs' (scope_family's)."""
        nat = self.nat
        graphs = [strand(g, tx.targets, nat.C) for g in graphs]
        if got is None:
            moved = np.asarray(tx.moved, dtype=float)
            positions = [p for p, _ in tx.f2]
            got = self._interchange_draws(tx, positions, split, draws, first={p: -moved[p - 1] for p in positions}) \
                if positions and tx.vpd is not None else []
        vpd = native.Graph(masks=tx.vpd) if tx.vpd is not None else None
        lqs = None
        if strengths is not None or lines is not None:
            K = len(strengths if strengths is not None else lines)
            graphs = graphs[:1] * K
            if got:
                lqs = self._pp_engine(tx).path_patched(graphs[:1], [tuple(pz) for pz in got], strengths=strengths, lines=lines)
        elif got and graphs and engine_run and all(g.masks is None for g in graphs) and \
                (cuts is None or all(not c or g.complete for c, g in zip(cuts, graphs))):
            lqs = self._pp_engine(tx).path_patched(graphs, [tuple(pz) for pz in got], cuts=cuts)  # [G, D, targets, V]
        acc = [[0.0, 0.0, 0.0, 0, [], 0.0] for _ in graphs]  # sum w s, sum w, sum w KL_C, prompts, draws, sum w s_raw
        kept = tx.__dict__.setdefault("_pp_prompts", {})    # (p, z) -> (the model on x', W, VPD's KL)
        for d, (p, z) in enumerate(got):
            x2 = list(tx.ids)
            x2[p] = z
            if (p, z) not in kept:
                r2 = nat.reference([x2], tx.targets)
                kept[(p, z)] = (r2, float(carry.kl(r2, tx.ref)[0]), float(carry.kl(r2, self.pp_run(tx, x2, vpd))[0]))
            ref, w, kv = kept[(p, z)]
            for b_, g in enumerate(graphs):
                lq = lqs[b_, d][None] if lqs is not None else self.pp_run(tx, x2, g, None if cuts is None else cuts[b_])
                kc = float(carry.kl(ref, lq)[0])
                sb = _fid(w, kc, kv)
                a = acc[b_]
                a[4].append([[p, z], kc, kv, w])
                if sb is not None:
                    a[0], a[1], a[2], a[3], a[5] = a[0] + w * sb, a[1] + w, a[2] + w * kc, a[3] + 1, a[5] + w * _fid_raw(w, kc, kv)
            native.free(nat.dev)
        return [{"s_track_pp": _ratio_pool(a[4]), "kl_track_pp_c": a[2] / a[1] if a[1] > 0 else None,
                 "prompts_track_pp": a[3], "draws_track_pp": a[4], "s_track_pp_raw": _ratio_pool(a[4], raw=True)}
                for a in acc]  # a pooled ratio over the draws (_ratio_pool)

    @torch.no_grad()
    def interchange_pp(self, tx: Text, graphs: list[native.Graph], split: str = "eval",
                       draws: int = INTERCHANGE_DRAWS, lines: torch.Tensor | None = None) -> list[dict]:
        """The interchange family in path-patched semantics (the lead's 10-12, C's interchange term) on a panel of the
        text's own (the lead's 10-13: drawn at each circuit's positions, a circuit placed where the behavior does not
        read matched the unmoved model trivially; CausalGym's attractor scored above the variable on 0.91 of pairs):
        the interchange draws (_interchange_draws) at the panel's positions (_interchange_panel: the start's and the
        task's minimal pairs'), each circuit scored on those at the targets and its own positions (_interchange_own,
        the lead's 10-14: input scope alone judges a position it leaves out); per draw (p, z) the model's response is the real model on
        the changed text x' (no circuit in it), VPD's prediction its whole answer's run with its actives at p at their
        values on x', and the circuit's its nodes at p held there (a block: the site's whole output), their stated paths
        carrying them in the path-patched run on the text (pp_run, fixed), everything else at x: a circuit with no node
        at p predicts no change. Per draw clip from the linear share (_ratio_pool: [[p, z], KL_C, KL_V, W], W =
        KL(model on x' || model)), pooled weighted by W's gap. Per circuit s_interchange_pp (and _raw),
        prompts_interchange_pp (draws where VPD is nearer than no change), draws_interchange_pp. lines [K, L]:
        graphs[0] (stranded) with the lines each row turns off removed (Engine.pp_lines' columns; graph_of_lines), its
        rows per variant, in one call (_interchange_lines)."""
        if lines is not None:
            return self._interchange_lines(tx, graphs[0], split, draws, lines)
        return self._interchange_graphs(tx, [strand(g, tx.targets, self.nat.C) for g in graphs], split, draws)

    def _interchange_panel(self, tx: Text, split: str) -> list[int]:
        """interchange_pp's draw positions on a text, whatever the circuit: the targets, the START's candidates
        (start_candidates, at most START_CAP: the START itself is chosen by rows that score this family) and a
        behavior's minimal pairs' (tx.behavior_pairs), position 0 aside. Each circuit is scored on the draws at its
        own positions alone (_interchange_own)."""
        own = {p for p, _ in getattr(tx, "behavior_pairs", [])}
        return sorted(t for t in set(tx.targets) | set(self.start_candidates(tx)[:START_CAP]) | own if 0 < t < tx.T)

    @staticmethod
    def _interchange_own(tx: Text, got: list, nodes) -> list[int]:
        """The panel draws a circuit is scored on (the lead's 10-14): those at the targets, a behavior's minimal pairs'
        positions and the circuit's own positions. Input scope alone penalizes a position the circuit leaves out. Scored
        on every START candidate, a circuit with no node at one predicts no change there, so on 21 v24 dev teachers the
        floors refused 4,452 removals for interchange_pp and 10 of the 18 passing teachers equalled their START. VPD's
        whole answer scores 1 on any subset of draws (KL_C = KL_VPD), so the floor is the same 0.9 on every panel."""
        own = set(tx.targets) | {p for p, _ in getattr(tx, "behavior_pairs", [])} | {t for _, t, _ in nodes}
        return [i for i, (p, _) in enumerate(got) if p in own]

    def _interchange_refs(self, tx: Text, split: str, draws: int) -> tuple:
        """interchange_pp's draws and references on a text (kept on tx): (draws [(p, z)], the changed texts' rows at
        their own position {(p, z): {"a": {site: [C]}, "rem": {site: [d_out]}}} (_pp_record's at p, the only ones a
        held value reads; the whole records, every site [T, C] per draw, took most of the teacher's GPU memory), the
        real model on each changed text [D, targets, V], W per draw, KL_V per draw (VPD's answer with its actives at p
        at their values on x'))."""
        kept = tx.__dict__.setdefault("_ipp_refs", {})
        if (split, draws) not in kept:
            nat = self.nat
            pos = self._interchange_panel(tx, split) if tx.vpd is not None else []
            got = self._interchange_draws(tx, pos, split, draws) if pos else []
            recs, lm, vh = {}, [], []
            for p, z in got:
                x2 = list(tx.ids)
                x2[p] = z
                r2 = self._pp_record(x2)
                recs[(p, z)] = r2 = {f: {n: r2[f][n][p].clone() for n in nat.names} for f in ("a", "rem")}
                lm.append(nat.reference([x2], tx.targets))
                vh.append({(n, p, c): r2["a"][n][c] for n in nat.names for c in tx.vpd[n][p].nonzero().flatten().tolist()})
            W = KV = []
            if got:
                lm = torch.cat(lm)
                lv = self._pp_engine(tx).path_patched([native.Graph(masks=tx.vpd)], vh)[0]
                W, KV = carry.kl(lm, tx.ref.expand_as(lm)).tolist(), carry.kl(lm, lv).tolist()
                native.free(nat.dev)
            kept[(split, draws)] = (got, recs, lm, W, KV)
        return kept[(split, draws)]

    def _interchange_held(self, r2: dict, nodes, p: int) -> dict:
        """A draw's held values (path_patched's {node: value}): the nodes at p at their values on the changed text, a
        subcomponent its activation, a block its site's whole output; r2 the draw's rows at p (_interchange_refs)."""
        return {(n, t, c): (r2["a"][n] @ self.nat.target.site(n).U + r2["rem"][n] if c == BLOCK else r2["a"][n][c])
                for n, t, c in nodes if t == p}

    @staticmethod
    def _interchange_row(got: list, kc: list, kv: list, w: list) -> dict:
        rec = [[[p, z], a, b, c] for (p, z), a, b, c in zip(got, kc, kv, w)]
        n = sum(1 for _, a, b, c in rec if c > 0 and b < c)
        return {"s_interchange_pp": _ratio_pool(rec), "prompts_interchange_pp": n, "draws_interchange_pp": rec,
                "s_interchange_pp_raw": _ratio_pool(rec, raw=True)}

    def _interchange_graphs(self, tx: Text, graphs: list[native.Graph], split: str, draws: int) -> list[dict]:
        """interchange_pp of graphs as given (stranding is the caller's), each graph on the text's panel draws at its own
        positions (_interchange_own)."""
        nat = self.nat
        got, recs, lm, W, KV = self._interchange_refs(tx, split, draws)
        empty = {"s_interchange_pp": None, "prompts_interchange_pp": 0, "draws_interchange_pp": [], "s_interchange_pp_raw": None}
        if not got:
            return [dict(empty) for _ in graphs]
        out = []
        for g in graphs:
            nodes = sorted(g.node_set())
            idx = self._interchange_own(tx, got, nodes)  # the draws at its own positions
            if not idx:
                out.append(dict(empty))
                continue
            helds = [self._interchange_held(recs[got[i]], nodes, got[i][0]) for i in idx]
            lc = self._pp_engine(tx).path_patched([g], helds)[0]  # pp_run(fixed=)'s numbers
            out.append(self._interchange_row([got[i] for i in idx], carry.kl(lm[idx], lc).tolist(), [KV[i] for i in idx],
                                             [W[i] for i in idx]))
            native.free(nat.dev)
        return out

    def _interchange_lines(self, tx: Text, graph: native.Graph, split: str, draws: int, lines: torch.Tensor) -> list[dict]:
        """interchange_pp of graph's line deletions (lines [K, L], Engine.pp_lines' columns of the stranded graph):
        every variant on the text's panel draws at its own positions (_interchange_own; the references are the text's), its circuit runs in one
        Engine.path_patched(lines=) call holding the answer's nodes at p (a deleted node is no longer the graph's, so its
        value drops out)."""
        nat = self.nat
        g = strand(graph, tx.targets, nat.C)
        got, recs, lm, W, KV = self._interchange_refs(tx, split, draws)
        if not got:
            return [{"s_interchange_pp": None, "prompts_interchange_pp": 0, "draws_interchange_pp": [], "s_interchange_pp_raw": None}
                    for _ in range(len(lines))]
        E = self._pp_engine(tx)
        nodes = [k[1] for k in E.pp_lines(g) if k[0] == "n"]
        helds = [self._interchange_held(recs[pz], nodes, pz[0]) for pz in got]
        circ = E.path_patched([g], helds, lines=lines)  # [K, D, targets, V]
        on = lines[:, :len(nodes)].bool().tolist()       # each variant's nodes (pp_lines' node columns come first)
        out = []
        for k in range(len(lines)):
            idx = self._interchange_own(tx, got, [nd for nd, o in zip(nodes, on[k]) if o])
            kl = carry.kl(lm, circ[k]).tolist()
            out.append(self._interchange_row([got[i] for i in idx], [kl[i] for i in idx], [KV[i] for i in idx], [W[i] for i in idx]))
        native.free(nat.dev)
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
        return self._graph_kl(tx.ids, tx.targets, tx.ref, graphs, live, self._bounds(tx, "live" if live else "off")[0], tx.vpd)

    @torch.no_grad()
    def _graph_kl(self, ids: list[int], targets: list[int], ref: torch.Tensor, graphs: list[native.Graph], live: bool,
                  empty_kl: float, active: dict | None = None) -> list[float]:
        """KL in bits at the targets, from the model's log probabilities ref, of each stated graph's run on ids in C's
        semantics (graph_ref: graph inside the circuit, masked outside, every other subcomponent off, remainders on
        (live) or off); a complete graph its masked run (runs, as sufficiency); a graph with no node is the empty circuit
        (empty_kl). Blocks run as their subcomponents (with active, the text's VPD answer, their active contents) and
        stranded nodes are dropped (_c_graph: complete node sets and block circuits score as fidelity scores them;
        pymove's report: node sets returned KL_empty, blocks crashed)."""
        graphs = [_c_graph(g, targets, self.nat.C, active) for g in graphs]
        out = [empty_kl] * len(graphs)
        whole = [i for i, g in enumerate(graphs) if g.complete and g.node_set()]  # a complete graph: its masked run
        if whole:
            got = carry.kl(ref, self.runs(ids, targets, [self.masks(graphs[i], len(ids)) for i in whole], live)).tolist()
            for i, k in zip(whole, got):
                out[i] = k
        some = [i for i, g in enumerate(graphs) if not g.complete and g.node_set()]
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
        """engine.Engine on ids over the graphs' subcomponents (and extra's {site: columns}; a block node is a unit of
        the plan, no column): C's semantics run there (Engine.plan(semantics="c"), tested equal to graph_ref) on those
        columns alone."""
        import engine

        nat, dev = self.nat, self.nat.dev
        cols = {}
        for n in nat.names:
            c = torch.tensor(sorted({c for g in graphs for m, _, c in g.node_set() if m == n and c != BLOCK}), dtype=torch.long, device=dev)
            cols[n] = c if extra is None else torch.unique(torch.cat([c, extra[n].to(dev)]))
        return engine.Engine(nat, ids, targets, live=live, columns=cols, outside=outside)

    def _bounds(self, tx: Text, conv: str, family: str = "point") -> tuple[float, float | None]:
        """(KL_empty, KL_VPD) on the text under a convention, in family's semantics (_xkw), kept on tx."""
        key = conv if not self._xkw(tx.ids, tx.targets, family) else (conv, "x")
        if key not in tx.bounds:
            nat = self.nat
            empty = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            sets = [empty] + ([tx.vpd] if tx.vpd is not None else [])
            kl = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, sets, conv == "live", family)).tolist()
            tx.bounds[key] = (kl[0], kl[1] if tx.vpd is not None else None)
        return tx.bounds[key]

    def sufficiency(self, tx: Text, graphs: list[native.Graph], conventions: tuple = CONVENTIONS, vpd_minus: bool = True) -> list[dict]:
        """Per graph, per convention: F1 (KL_C of the stated graph's run, graph_kl; a complete graph's masked run),
        recovered, completeness, and N (node removal from VPD's answer; vpd_minus False leaves it out) on the text,
        with the bounds."""
        rows = [{} for _ in graphs]
        stated = [i for i, g in enumerate(graphs) if not g.complete]
        whole = [i for i, g in enumerate(graphs) if g.complete]
        sets = [self.masks(graphs[i], tx.T) for i in whole]
        vpd_minus = vpd_minus and tx.vpd is not None
        if vpd_minus:
            sets += [{n: tx.vpd[n] & ~m[n] for n in self.nat.names} for m in (self.masks(g, tx.T) for g in graphs)]
        for conv in conventions:
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
                if v is not None and vpd_minus:
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
    def cuts(self, tx: Text, pairs: list[tuple], detail: bool = False, mode: str = "full", lq: bool = False) -> list:
        """Per (writer, reader) pair (reader None: the prediction), the KL in bits of the model on the text from the
        model with that one connection cut: the reader's new activation is its input in the model without the
        writer's output alone (every other output and every remainder as in the model), and the model runs on with
        that one activation replaced (path patching of the connection, _patched); into the prediction, the final
        residual stream without the writer's output. detail: per pair {"kl", "top": the cut run's most likely token at
        each target, "logp_model_top": its log-probability of the model's most likely token there; with lq, "lq": the cut
        run's log-probabilities at the targets [len(targets), V] on the CPU (primerl's partial credit)}. mode (primerl's
        normalization questions): "full" the cut as above; "direct" the reader reads its input without the writer's
        output but its RMSNorm scale is the uncut input's; "norm" the reader reads its uncut input under the RMSNorm
        scale of the input without the writer's output (an attention or MLP output reads no norm: "direct" is the full
        cut there, "norm" no change)."""
        if mode not in ("full", "direct", "norm"):
            raise ValueError("mode is full, direct or norm")
        nat = self.nat
        key = (lambda e: e) if mode == "full" else (lambda e: (e, mode))
        todo = [e for e in dict.fromkeys(pairs) if key(e) not in tx.cut_cache or (lq and "lq" not in tx.cut_cache[key(e)])]
        if todo:
            ctx = {"make": lambda B: (nat._full_plan(tx.T, B), nat._model_states(tx.ids, B)), "ablation": nat._one(),
                   "ref": tx.ref, "flow": self.flow(tx), "rec": tx.rec, "ids": tx.ids, "mode": mode, "lq": lq}
            for e, d in zip(todo, self._patched(tx, ctx, todo)):
                tx.cut_cache[key(e)] = d
        return [tx.cut_cache[key(e)] if detail else tx.cut_cache[key(e)]["kl"] for e in pairs]

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

        def summary(lq):  # (ctx["lq"]: each cut run's whole distribution at the targets kept too)
            return [{"kl": k, "top": t, "logp_model_top": p, **({"lq": lq[b].float().cpu()} if ctx.get("lq") else {})}
                    for b, (k, t, p) in enumerate(zip(carry.kl(ref, lq).tolist(), lq.argmax(-1).tolist(),
                                                      lq.gather(-1, top[None, :, None].expand(lq.shape[0], -1, 1))[..., 0].tolist()))]

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
            x0 = ctx["flow"]["final"][tx.targets].expand(len(final), -1, -1)
            x = x0.clone()
            row = torch.tensor([tx.targets.index(p) for p in t.tolist()], device=nat.dev)
            for wn in {pairs[j][0][0] for j in final}:
                sel = torch.tensor([b for b, j in enumerate(final) if pairs[j][0][0] == wn], device=nat.dev)
                x[sel, row[sel]] -= ctx["rec"][wn][0, t[sel], c[sel]][:, None] * tg.site(wn).U[c[sel]]
            read, scale = {"full": (x, x), "direct": (x, x0), "norm": (x0, x)}[ctx.get("mode", "full")]
            lq = torch.log_softmax(tg.unembed(_parts(tg, read, scale, -1)).float(), -1)
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
        x = tg.embed(torch.tensor(ids, device=nat.dev))[None]
        causal = torch.ones(T, T, dtype=torch.bool, device=nat.dev).tril()
        fl = {"attn": [], "mlp": [], "P": [], "att": [], "pre": []}

        def write(n):  # the subcomponents' write and any bias (always on; vpd4l: none)
            a = rec[n] if masks is None else rec[n] * masks[n]
            out = a @ tg.site(n).U
            return out if tg.site(n).b is None else out + tg.site(n).b

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
                wv = tg.token_embed(torch.tensor([ctx["ids"][p] for p in t.tolist()], device=nat.dev))
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
            mode = ctx.get("mode", "full")
            if kind in ("q_proj", "k_proj", "v_proj", "c_fc"):
                xr = fl["attn" if kind != "c_fc" else "mlp"][layer][t]
                read, scale = {"full": (xr - wv, xr - wv), "direct": (xr - wv, xr), "norm": (xr, xr - wv)}[mode]
                inp = _parts(tg, read, scale, 2 * layer + (kind == "c_fc"))
            elif kind == "o_proj":  # no norm: "norm" changes nothing
                inp = fl["att"][layer][t] - (0 if mode == "norm" else fl["P"][layer][:, t, tw].T.repeat_interleave(tg.hd, dim=1) * wv)
            else:
                inp = tg.act(fl["pre"][layer][t] - (0 if mode == "norm" else wv))
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
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
        kls = self._removed_kl(tx.ids, tx.targets, [self.masks(g, T) for g in graphs], tx.ref, remainders_at_nodes)
        out = []
        for k in kls:
            row = {"kl_real_minus": k}
            for conv in CONVENTIONS:
                e, v = self._bounds(tx, conv)
                row[f"n_real_{conv}"] = necessity_log(e, k, v) if v is not None else None
            out.append(row)
        return out

    @torch.no_grad()
    def point_remainders(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """The point score under three treatments of the remainders W - V U (the lead's 10-12 hidden-computation check:
        live, C's semantics, they read the circuit's outputs and can carry computation the graph never states): live;
        held at their values on x (the whole model's run, the circuit's outputs not reaching them); off. Per graph and
        treatment X: kl_point_X (graph_ref, bits) and s_point_X = _fid against the same treatment's bounds (the empty
        circuit's and VPD's whole answer's KL under it, kl_empty_X and kl_vpd_X)."""
        nat, T = self.nat, tx.T
        rest = {}
        model = native.Graph(masks={n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
        self.graph_ref(tx.ids, tx.targets, model, rest=rest)
        how = {"live": {}, "held": {"rest": rest, "hold": True}, "off": {"live": False}}

        def kl(g, kw):
            return float(carry.kl(tx.ref, self.graph_ref(tx.ids, tx.targets, g, **kw))[0])

        bounds = {x: (kl(native.Graph(), kw), kl(native.Graph(masks=tx.vpd), kw)) for x, kw in how.items()}
        out = []
        for g in graphs:
            g = _c_graph(g, tx.targets, nat.C, tx.vpd)
            row = {}
            for x, kw in how.items():
                kc = kl(g, kw)
                row |= {f"kl_point_{x}": kc, f"s_point_{x}": _fid(bounds[x][0], kc, bounds[x][1]),
                        f"kl_empty_{x}": bounds[x][0], f"kl_vpd_{x}": bounds[x][1]}
            out.append(row)
            native.free(nat.dev)
        return out

    @torch.no_grad()
    def _removed_kl(self, ids: list[int], targets: list[int], sets: list[dict], ref: torch.Tensor | None = None,
                    remainders_at_nodes: bool = True) -> list[float]:
        """Per set (matrix -> bool [len(ids), C]): KL at the targets from ref (the whole model's run on ids when None)
        of the real model on ids with the set's subcomponents removed (mask 0 at their positions, every other
        subcomponent and every remainder on). remainders_at_nodes False: each matrix's remainder W - V U removed too at
        the positions the set holds, the masked run over every subcomponent (a batch reuses the model's states before its
        first removed subcomponent); otherwise the engine's (outside "on": the sets' subcomponents masked, every other
        one folded into the remainder, float32-close to that run at the model's own cost: _removal_engine)."""
        if remainders_at_nodes:
            if not sets:
                return []
            E = self._removal_engine(ids, targets, sets)
            return self._removed_kl_on(E, self._stacked(E, sets), ref)
        nat, T = self.nat, len(ids)
        one_u, one_r = nat._one()
        if ref is None:
            ref = nat.run(ids, targets, nat._full_plan(T, 1), one_u, one_r)
        none = nat._const(("goal_none", T), lambda: {n: torch.zeros(1, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
        kls = []
        rows = max(self.chunk, REMOVAL_ROWS["cuda" if str(nat.dev).startswith("cuda") else "other"])
        for s0 in range(0, len(sets), rows):
            part = sets[s0:s0 + rows]
            B = len(part)
            u = {n: 1.0 - torch.stack([m[n] for m in part]).float() for n in nat.names}
            first = min((int(m[n].any(1).nonzero().min()) for m in part for n in nat.names if m[n].any()), default=0)
            base = _from(ids, min(first, min(targets)))
            plan = {**nat._full_plan(T, B), "G": none, "none": True}
            held = torch.stack([sum(m[n].any(1).float() for n in nat.names) > 0 for m in part])  # [B, T]
            ur = {n: 1.0 - held.float() for n in nat.names}
            lq = nat.run(ids, targets, plan, u, ur, reuse=None if base is None else (nat._model_states(ids, B), base))
            kls += carry.kl(ref, lq).tolist()
            native.free(nat.dev)
        return kls

    NODE_ENTRIES = 1 << 27  # _node_family's rows a batch: this many floats in its widest activations and masks
    NEC_ENTRIES = 1 << 27  # _removed_kl_on's rows a run: this many floats in its widest activations (sets x T x C)
    NEC_ENGINES = 3  # necessity_spec's removal engines kept on a text (VPD's, a teacher's start's, the latest)

    def _removal_engine(self, ids: list[int], targets: list[int], sets: list[dict]):
        """An engine on ids over the sets' subcomponents with every other one on (outside "on"; Engine.on moves it to
        another text)."""
        import engine

        nat, dev = self.nat, self.nat.dev
        none = torch.zeros(0, dtype=torch.long, device=dev)
        cols = {n: torch.unique(torch.cat([m[n].nonzero()[:, 1] for m in sets] + [none])) for n in nat.names}
        return engine.Engine(nat, ids, targets, live=True, columns=cols, outside="on")

    @staticmethod
    def _stacked(E, sets: list[dict]) -> dict:
        """Sets (matrix -> bool [T, C]) on E's columns, stacked: {matrix: bool [sets, T, K]}."""
        return {n: torch.stack([m[n][:, c] for m in sets]) for n, c in E.columns.items()}

    @torch.no_grad()
    def _removed_kl_on(self, E, M: dict, ref: torch.Tensor | None = None) -> list[float]:
        """_removed_kl with every remainder on, on E (_removal_engine of the sets, or of sets they are a shift of), M
        the sets on its columns ({matrix: bool [sets, T, K]}, _stacked): the positions from the first any set removes
        something at (or the first target) on, the model's states before it (Engine.log_probs' start)."""
        nat = self.nat
        ref = E.ref if ref is None else ref
        S = next(iter(M.values())).shape[0]
        hit = torch.zeros(E.T, dtype=torch.bool, device=nat.dev)  # positions any set removes something at
        for m in M.values():
            hit |= m.any(2).any(0)
        first = int(hit.nonzero().min()) if bool(hit.any()) else E.T
        start = min(first, min(E.targets))  # before it every set's run is the model's (Engine.log_probs' start)
        rows = max(1, self.NEC_ENTRIES // ((E.T - start) * max(nat.C.values())))
        kls = []
        for s0 in range(0, S, rows):
            masks = {n: 1.0 - m[s0:s0 + rows].float() for n, m in M.items()}
            kls += carry.kl(ref, E.log_probs(masks, start=start)).tolist()
        native.free(nat.dev)
        return kls

    @torch.no_grad()
    def _removed_kl_many(self, es: list, M: dict) -> list[list[float]]:
        """_removed_kl_on on several texts at once (necessity's controls): es, one removal engine moved to texts of one
        length and targets (Engine.on), M the sets on its columns; per text the sets' KLs from its model, in runs of
        several texts (Engine.log_probs_on)."""
        nat, e0 = self.nat, es[0]
        S = next(iter(M.values())).shape[0]
        hit = torch.zeros(e0.T, dtype=torch.bool, device=nat.dev)
        for m in M.values():
            hit |= m.any(2).any(0)
        first = int(hit.nonzero().min()) if bool(hit.any()) else e0.T
        start = min(first, min(e0.targets))
        rows = max(1, self.NEC_ENTRIES // ((e0.T - start) * max(nat.C.values())))  # (text, set) rows a run
        per, step = (rows // S, S) if rows >= S else (1, rows)  # texts a run, each with every set; or one in slices
        out = [[] for _ in es]
        for j0 in range(0, len(es), per):
            grp = es[j0:j0 + per]
            for s0 in range(0, S, step):
                masks = {n: 1.0 - m[s0:s0 + step].float() for n, m in M.items()}
                k = next(iter(masks.values())).shape[0]
                ref = torch.cat([e.ref.expand(k, -1, -1) for e in grp])
                for j, row in enumerate(carry.kl(ref, e0.log_probs_on(grp, masks, start)).view(len(grp), k).tolist()):
                    out[j0 + j] += row
        native.free(nat.dev)
        return out

    def necessity_spec(self, tx: Text, graphs: list[native.Graph], split: str = "eval", controls: int = NEC_CONTROLS,
                       background: bool = True) -> list[dict]:
        """N, phi's necessity (the user's spec, the lead's 10-12: without the circuit's parts the real system breaks,
        and the break is this text's, not every text's). Per graph: D_t, the damage at the targets (KL from the model)
        of the real model less the circuit's subcomponents (every other subcomponent and every remainder on, as
        real_removal); D_c, the same removal's damage on a matched control text, end-aligned (the control's last
        position takes the text's last target, every part shifted with it: the same relative positions, which are all
        the model's attention sees), averaged over controls texts of the split's far pool (DONOR_POOLS) at least
        NEC_MIN_LEN long (or the text's length), drawn by the text's seed, the same for every graph on the text; parts
        before a control's start are left out there; the same two for VPD's whole answer. n_spec = clip((D_t - D_c) / (D_t(VPD) - D_c(VPD)), 0, 1): parts whose removal
        damages every text's prediction there add to D_t and D_c alike and earn nothing, VPD's answer scores 1, None
        when VPD's removal damages the controls as much as the text; n_spec_raw unclipped below (c_rl's ranking).
        draws_n_spec: per control [control, D_c, D_c(VPD), D_t, D_t(VPD)] (phi's eps). Damages in bits. background
        (the default, the lead's 10-13): the background parts at the targets (GENERIC_TARGET) kept on in every removal,
        credited to no one. Without it the 5 parts saturated the damage on text and controls alike (14-18 bits) and the
        start's D_t - D_c fell to 0 or below on 9001, 9006, 9007 and 9015; kept on, it is 1.8-6.9 bits there and N on the
        20 held-out starts goes from 0.62 (14 scored) to 0.92 (18 scored)."""
        import engine

        nat, T, dev = self.nat, tx.T, self.nat.dev
        cg = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]
        keep = None
        if background:  # the background parts at the targets never removed, the circuits' and VPD's alike
            keep = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
            for n, t, c in self.background(tx):
                keep[n][t, c] = True
        last = max(tx.targets)
        pool = [x for x in self.far_pool(split) if len(x[1]) >= min(T, NEC_MIN_LEN) and list(x[1]) != list(tx.ids)]
        ys = [pool[i][1] for i in np.random.default_rng([int(tx.seed), 13]).permutation(len(pool))[:controls]]

        def damages(cols, M):  # [D_t per set, per control [D_c per set]] of the sets M ({site: bool [sets, T, K]} over
            kept = tx.__dict__.setdefault("_nec_engines", [])  # cols) on one engine moved to each control (Engine.on): the
            for j, x in enumerate(kept):  # latest NEC_ENGINES kept on tx, one whose columns hold cols reused (a teacher's
                if all(bool(torch.isin(cols[n], x[0][n]).all()) for n in cols):  # candidates, subsets of its start: no
                    kept.append(kept.pop(j))                                      # remaking of each control's run before
                    break                                                         # the circuit)
            else:
                kept.append((cols, engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols, outside="on"), {}))
                del kept[:-self.NEC_ENGINES]
            ce, E, ons = kept[-1]
            M = {n: m if ce[n] is cols[n] else m.new_zeros(*m.shape[:2], len(ce[n])).index_copy_(2, torch.searchsorted(ce[n], cols[n]), m)
                 for n, m in M.items()}  # the sets on E's columns
            if keep is not None:
                M = {n: m & ~keep[n][:, ce[n]][None] for n, m in M.items()}
            groups = {}  # the controls by their shift (one length and targets: one run of them, _removed_kl_many)
            for y in ys:
                y = tuple(y[:last + 1])
                d = last + 1 - len(y)  # the shift
                if y not in ons:
                    ons[y] = E.on(list(y), [t - d for t in tx.targets if t >= d])
                groups.setdefault(d, []).append(y)
            got = {}
            for d, yl in groups.items():
                got.update(zip(yl, self._removed_kl_many([ons[y] for y in yl], {n: m[:, d:last + 1] for n, m in M.items()})))
            return self._removed_kl_on(E, M, tx.ref), [got[tuple(y[:last + 1])] for y in ys]
        key = ("nec_vpd", last, tuple(tuple(y) for y in ys), background)  # VPD's damages on these controls: every graph's on the text
        if key not in tx.__dict__:
            vc = {n: tx.vpd[n].any(0).nonzero().flatten() for n in nat.names}
            tx.__dict__[key] = damages(vc, {n: tx.vpd[n][:, vc[n]][None] for n in nat.names})
        per = {n: ([], [], []) for n in nat.names}  # the sets on their columns, written straight into [sets, T, K] (a
        for i, h in enumerate(cg):                 # mask [T, C] per set: 13 GB for line_loo on a 798-node set)
            for n, t, c in h.node_set():
                per[n][0].append(i)
                per[n][1].append(t)
                per[n][2].append(c)
        cols = {n: torch.unique(torch.tensor(per[n][2], dtype=torch.long, device=dev)) for n in nat.names}
        M = {n: torch.zeros(len(cg), T, len(cols[n]), dtype=torch.bool, device=dev) for n in nat.names}
        for n, (i_, t_, c_) in per.items():
            if i_:
                M[n][torch.tensor(i_, device=dev), torch.tensor(t_, device=dev),
                     torch.searchsorted(cols[n], torch.tensor(c_, device=dev))] = True
        dt, dc = damages(cols, M) if cg else ([], [[] for _ in ys])
        dt = dt + tx.__dict__[key][0]
        dc = [d + v for d, v in zip(dc, tx.__dict__[key][1])]
        out = []
        for i in range(len(graphs)):
            rec = [[j, d[i], d[-1], dt[i], dt[-1]] for j, d in enumerate(dc)]
            out.append({"d_target": dt[i], "d_control": float(np.mean([r[1] for r in rec])) if rec else None,
                        "d_target_vpd": dt[-1], "d_control_vpd": float(np.mean([r[2] for r in rec])) if rec else None,
                        "n_spec": _spec(rec), "n_spec_raw": _spec(rec, raw=True), "draws_n_spec": rec})
        return out

    # ---- diagnostics

    # ---- C: interventional fidelity

    def fidelity(self, tx: Text, graphs: list[native.Graph], pool: dict | None = None, draws: int = DRAWS,
                 steps: int = ADV_STEPS, restarts: int = ADV_RESTARTS, curve=ADV_CURVE, fresh: dict | None = None,
                 interchange: int = INTERCHANGE_DRAWS, real: bool = False, reads: bool = False, mode: str = "full",
                 changed: int = CHANGED_DRAWS, split: str | None = None, near_donors: int = DONORS_NEAR,
                 far_donors: int = DONORS_FAR, baseline: bool = False, pattern: bool = True,
                 controls: int = NEC_CONTROLS, weight: bool = False, scope: bool = True, node_values: bool | None = None) -> list[dict]:
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
        context_share, s_track) and "donors" (donors, activation patching: everything outside the circuit at one donor text's values, near
        donors the changed texts and far donors other texts, s_patching_near and s_patching_far) and "scope" (scope_family,
        phi's input-scope family: each influential position's token alone from natural donors, the circuit's own-run
        move against the model's, s_scope; scope=False leaves it out); c, the harmonic mean of
        s_point_c, s_nodes, s_edges, f1_flips_c at FLIP_TERM (the reward's adversarial member), s_track, s_interchange,
        s_patching_near and s_patching_far over those defined on the text (c_terms names them; the claim is agreement under
        every family, so one failed family is not made up by the others, and near 0 c is linear in the lowest term, so
        float noise there stays noise: the lead's decisions 10-12; the step-budget f1_adv and s_changed are reported
        columns), the last four on split's draws (default: "search", the teachers' draws and training texts as far
        donors, in the reward on training texts; "eval" otherwise, and for any held-out text); e4: the E4 hook's score (its own reward term, not in c); flips_to_break, the smallest m at which f1_flips_omitted falls below
        half of s_point (None: not within FLIPS); flips_to_half, the smallest m at which it falls below 0.5, half of
        VPD's own answer under the same attack (0: below it unattacked), which a circuit weak to begin with cannot pass. reads: C's second observable, node values (_node_values: node_share, node_share_weighted, node_median_z, node_measured; full mode's alone unless node_values says, never the reward's: the RL scorer, env F's tools and the teachers skip them) and the circuit nodes' relative RMS read
        error between the model and the circuit per family (reads_*), and of their responses, each node's change from
        its own run without the intervention (dreads_*, _response_error; the static error mostly measures the missing
        background of a circuit's run); graph_ref,
        slower). interchange, changed: draws per circuit for those two families (0: left out of c). Evaluation only,
        never in the reward or a teacher's objective:
        fresh, a pool none of the searches drew from: the pattern lever's flips over it (f1_flips_patterns_fresh);
        weight: the weight-edit family (weight_family.family: s_weight, s_weight_raw, kl_weight_c, draws_weight; in
        this GoalEval's semantics, out of phi until it passes falsification, the lead's 10-13);
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
        if tx.vpd is not None:  # (at their active contents, block_contents)
            pp_graphs = [expand_blocks(g, self.nat.C, tx.vpd) for g in pp_graphs]
        graphs = [_c_graph(g, tx.targets, self.nat.C, tx.vpd) for g in graphs]  # stranded nodes dropped before every family
        reward = mode == "reward"
        node_values = not reward if node_values is None else node_values  # the eval scorecard's column, never a reward's
        split = split or ("search" if reward else "eval")
        reads = reads or reward
        e0 = self._bounds(tx, "live")[0]
        sides = self._sides(tx, graphs) if reads else None
        point = self.sufficiency(tx, graphs)
        whole = [native.Graph(sorted(g.node_set()), complete=True) if not g.complete else None for g in graphs]
        wired = dict(zip([i for i, w in enumerate(whole) if w is not None],
                         self.sufficiency(tx, [w for w in whole if w is not None])))
        nodes = self._node_family(tx, graphs, draws, reads, units=[sorted(h.node_set()) for h in pp_graphs])  # a block one unit
        omitted = self._omitted_family(tx, graphs) if tx.vpd is not None else [{} for _ in graphs]  # phi's omitted family
        edges = self._edge_family(tx, graphs, draws, reads)
        none = [{} for _ in graphs]
        adv = self.adversary(tx, graphs, pool, steps=steps, restarts=restarts, draws=draws, curve=curve) if pool and not reward else none
        flips = self.adversary_flips(tx, graphs, pool, "omitted", steps=steps, restarts=restarts,
                                     upto=FLIP_TERM if reward else None) if pool else none
        fresh_rows = self.adversary_flips(tx, graphs, fresh, "patterns", steps=steps, restarts=restarts) if fresh and not reward else none
        inter = self._interchange_family(tx, graphs, interchange, split=split, reads=reads) if interchange else none
        reals = self.real_removal(tx, graphs) if real and not reward else none
        chg = self.changed_inputs(tx, graphs, split, changed) if changed else none
        scp = self.scope_family(tx, graphs, split) if scope else none  # phi's input-scope family
        tpp = self.track_pp(tx, pp_graphs, split, changed) if changed else none
        ipp = self.interchange_pp(tx, pp_graphs, split, interchange) if interchange else none
        pat = self.pattern_family(tx, graphs) if pattern else none  # s_pattern: C's true-pattern family
        dns = self.donors(tx, graphs, split, near_donors, self.far_donors(tx, self.far_pool(split), far_donors) if far_donors else None) \
            if (near_donors or far_donors) else none
        nflips = self.necessity_flips(tx, graphs, steps=steps, restarts=restarts) if real and not reward else none
        nspec = self.necessity_spec(tx, graphs, split, controls) if controls else none  # phi's N
        leak = none if reward else self.sufficiency(tx, [native.Graph(sorted(g.node_set()), {}, []) for g in graphs])
        out = []
        for g, pt, nd, ed, ad, fl, fr, it, rr, nf, lk, ns, om in zip(graphs, point, nodes, edges, adv, flips, fresh_rows, inter, reals,
                                                                     nflips, leak, nspec, omitted):
            i = len(out)
            row = {"s_point": pt["f1_live"], "kl_point": pt["kl_live"], "n_point": pt["n_live"], **nd, **ed, **ad, **fl, **it, **nf, **ns, **om,
                   "s_wiring": wired[i]["f1_live"] if i in wired else pt["f1_live"]}
            if lk:
                row["leak"] = lk["f1_live"]
            # the node family against the model's own response (the lead's 10-12): each draw bounded by no change,
            # W = KL(model less the nodes || model), pooled by W
            row["s_nodes"] = _pooled("weighted", nd.get("draws_nodes"), e0)
            row["s_nodes_raw"] = _pooled_raw(nd.get("draws_nodes"))
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
            row.update(scp[i])
            row.update(tpp[i])
            row.update(ipp[i])
            row.update(pat[i])
            row.update(dns[i])
            named = [(k, row.get(k)) for k in ("s_point_c", "s_nodes", "s_edges")] + [("f1_flips_c", fc.get(FLIP_TERM))]
            named += [(k, row.get(k)) for k in ("s_track_pp", "s_interchange_pp", "s_patching_near", "s_patching_far", "s_pattern")]
            row["c_terms"] = [k for k, v in named if v is not None]  # a family with nothing scored on the text is left out
            row["c"] = _harmonic([v for _, v in named if v is not None])
            # Unfloored families for phi in training and full evaluation alike.
            v0 = self._bounds(tx, "live")[1]
            kfl = row.get("kl_flips_omitted", {}).get(FLIP_TERM)
            raw = {"s_point_c": min(x for x in (_fid_raw(e0, pt["kl_live"], v0),
                                                _fid_raw(e0, wired[i]["kl_live"], v0) if i in wired else None) if x is not None)
                   if _fid_raw(e0, pt["kl_live"], v0) is not None else None,
                   "s_nodes": row.get("s_nodes_raw"),
                   "s_edges": _fid_raw(e0, ed["kl_edges_c"], ed["kl_edges_vpd"]) if ed["kl_edges_c"] is not None else None,
                   "f1_flips_c": min((x for x in (_fid_raw(e0, *kfl) if kfl else None, fw_raw.get(FLIP_TERM)) if x is not None),
                                     default=None) if kfl else None,  # none where neither bound resolves (a behavior's KLs)
                   "s_track_pp": row.get("s_track_pp_raw"), "s_interchange_pp": row.get("s_interchange_pp_raw"),
                   "s_pattern": row.get("s_pattern_raw")}
            for pool_ in ("near", "far"):
                k3 = row.get(f"kl_patching_{pool_}")
                raw[f"s_patching_{pool_}"] = _fid_raw(k3[2], k3[0], k3[1]) if k3 else None
            row["rl_terms"] = {k: raw[k] for k in row["c_terms"]}
            row["c_rl"] = _c_rl([raw[k] for k in row["c_terms"] if raw[k] is not None])
            if ns:  # phi's N unfloored as well (not one of C's terms)
                row["rl_terms"]["n_spec"] = row.get("n_spec_raw")
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
                merged = {}  # node values (_node_values) over the node-removal and interchange families' reads
                for fam in ("nodes", "interchange"):
                    for nd, xs in (row.get(f"_pairs_{fam}") or {}).items():
                        merged.setdefault(tuple(nd), []).extend(xs)
                if node_values:  # (full mode's by default: an eval column, line_effects over every node; no reward reads it)
                    eff = {tuple(x["node"]): max(0.0, x["delta_kl"] or 0.0)
                           for x in self.line_effects(tx, g, mode="pp", split=split)["nodes"]} if merged else {}
                    row.update(_node_values(merged, eff))
                for fam in ("nodes", "edges", "interchange"):
                    pairs = row.pop(f"_pairs_{fam}", None)
                    row[f"dreads_{fam}"] = _response_error(pairs, base) if pairs else None
            for fam in ("nodes", "edges", "interchange"):
                row.pop(f"_pairs_{fam}", None)
            out.append(row)
        if weight:  # weight: rustmove's weight-edit family (weight_family, e5228b0), a reported column, not in phi
            import weight_family

            for row, wr in zip(out, weight_family.family(self, tx, graphs)):
                row.update(wr)
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
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
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
            ix = {}
            for n, t, c in g.node_set():
                ix.setdefault(n, ([], []))[0].append(t)
                ix[n][1].append(inv[n][c])
            for n, (t, c) in ix.items():  # one indexed write per site
                ci[n][torch.tensor(t, device=dev), torch.tensor(c, device=dev)] = 1.0
            circ.append(ci)
            lever.append({n: (tx.vpd[n][:, cols[n]] & (ci[n] == 0) & where).nonzero(as_tuple=True) for n in nat.names})
        keep = {n: torch.stack([1 - ci[n] for ci in circ]) for n in nat.names}  # [circuits, T, columns]: the circuit off
        sizes = [[len(lv[n][0]) for n in nat.names] for lv in lever]
        N = [sum(z) for z in sizes]
        flat = [_Parts(nat.names, [lv[n][0] for n in nat.names], [cols[n][lv[n][1]] for n in nat.names]) for lv in lever]
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


    def _node_draws(self, tx: Text, units: list, draws: int, held: list | None = None) -> list[list]:
        """The node family's removal draws with common random numbers (the lead's 10-13: draws re-seeded by the node
        set moved s_nodes by about 0.08 for one added subcomponent): each unit's uniforms its own, one a draw, seeded by
        the text and the unit alone (kept on tx), so a circuit's subsets share their draws, a deleted unit dropping out
        of each. Draw j takes the units whose uniform is below (j + 0.5) / draws, at least one (the smallest); a block
        unit (site, t, BLOCK) is the subcomponents of its site at t that held (the scored graph's nodes) has, or without
        held its contents (block_contents: the text's actives)."""
        inside = {}
        for nd in held or ():
            inside.setdefault(nd[:2], []).append(nd)
        kept = tx.__dict__.setdefault("_node_uniforms", {})
        for u in units:
            if (u, draws) not in kept:
                h = int(hashlib.sha256(repr(("nodes", u)).encode()).hexdigest()[:15], 16)
                kept[(u, draws)] = np.random.default_rng([int(tx.seed), h]).random(draws)
        if not units:
            return []
        r = np.stack([kept[(u, draws)] for u in units], 1)  # [draws, units]
        out = []
        for j in range(draws):
            pick = np.nonzero(r[j] < (j + 0.5) / draws)[0]
            du = [units[i] for i in (pick if len(pick) else [int(np.argmin(r[j]))])]
            out.append([x for u in du for x in ((inside.get(u[:2], []) if held is not None else
                                                 [(u[0], u[1], c) for c in block_contents(u[0], u[1], self.nat.C, tx.vpd)]) if u[2] == BLOCK else [u])])
        return out

    def _subset_draws(self, tx: Text, nodes: list, nid: dict, sets: list, draws: int) -> list[list]:
        """_node_draws of each node set in sets (sorted tuples of nodes, sorted), as arrays of indices into nodes: the
        uniforms of nodes stacked once, each set's draws by one comparison (line credit's variants, a node set each)."""
        if any(nd[2] == BLOCK for nd in nodes):
            return [[np.array([nid[nd] for nd in d], dtype=np.int64) for d in self._node_draws(tx, list(ns), draws)] for ns in sets]
        self._node_draws(tx, nodes, draws)  # (every node's uniforms kept on tx)
        kept = tx.__dict__["_node_uniforms"]
        U = np.stack([kept[(u, draws)] for u in nodes], 1) if nodes else np.zeros((draws, 0))  # [draws, nodes]
        below = U < ((np.arange(draws) + 0.5) / draws)[:, None]
        out = []
        for ns in sets:
            ids = np.array([nid[nd] for nd in ns], dtype=np.int64)
            if not len(ids):
                out.append([])
                continue
            b = below[:, ids]
            out.append([ids[b[j]] if b[j].any() else ids[[int(np.argmin(U[j, ids]))]] for j in range(draws)])
        return out

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
            return torch.zeros(0, len(targets), self.nat.target.vocab, device=self.nat.dev)
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
    def background(self, tx: Text) -> set:
        """GENERIC_TARGET's parts at the text's targets (those of the model's sites)."""
        return {(n, t, c) for t in tx.targets for n, c in GENERIC_TARGET if n in self.nat.C and c < self.nat.C[n]}

    @torch.no_grad()
    def _omitted_family(self, tx: Text, graphs: list[native.Graph], k: int = OMITTED_K) -> list[dict]:
        """The omitted family (the lead's 10-13, the spec's interventions on parts a circuit leaves out; the mirror of
        the node family, which removes the parts it holds): per circuit, the first k of VPD's actives in its order
        (causal importance x write) at the circuit's positions and the targets that the circuit leaves out, the
        background (GENERIC_TARGET) left out; each removed alone from the real model (W = KL(model less it || model));
        the circuit's prediction is its own run (no change); VPD's answer less it the reference; s_omitted pools the
        candidates as the weighted families do (_ratio_pool). A circuit that leaves out a part whose removal moves the
        model as VPD predicts scores 0 there; with nothing left out, or nothing whose removal VPD predicts better than
        no change, s_omitted = 1. (Candidates the circuit holds are the node family's: scored here too, held parts
        made masked complete sets score 0, the node family's overstated removals again: influential sets 0.16 on 11 of 20
        held-out texts.) draws_omitted: per candidate [candidate, KL_C, KL_VPD, W]; omitted_left: how many."""
        import engine

        nat, dev = self.nat, self.nat.dev
        order = getattr(tx, "vpd_order", None) or nat._vpd_order(tx.ids, tx.targets)
        bg = self.background(tx)
        cands = []
        for g in graphs:
            nodes = g.node_set()
            pos = {t for _, t, _ in nodes} | set(tx.targets)
            cands.append([tuple(nd) for nd in order if nd[1] in pos and tuple(nd) not in bg and tuple(nd) not in nodes][:k])
        cols = {n: {c for g, cs in zip(graphs, cands) for m, _, c in list(g.node_set()) + cs if m == n} for n in nat.names}
        circ = {n: torch.tensor(sorted(cols[n]), dtype=torch.long, device=dev) for n in nat.names}
        vcols = {n: torch.unique(torch.cat([circ[n], tx.vpd[n].any(0).nonzero().flatten()])) for n in nat.names}
        Em = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ, outside="on")  # the model
        Ev = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=vcols)               # VPD's answer
        Ec = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ)                # the circuits
        at = {n: {c: i for i, c in enumerate(circ[n].tolist())} for n in nat.names}
        vat = {n: {c: i for i, c in enumerate(vcols[n].tolist())} for n in nat.names}
        vpd = {n: tx.vpd[n][:, vcols[n]].float() for n in nat.names}

        def without(base, cs, where):  # [len(cs), T, K]: base with each candidate off in its row
            m = {n: x[None].repeat(len(cs), 1, 1).clone() for n, x in base.items()}
            for b, (n, t, c) in enumerate(cs):
                m[n][b, t, where[n][c]] = 0.0
            return m

        out = []
        for g, cs in zip(graphs, cands):
            if not cs:
                out.append({"s_omitted": 1.0, "s_omitted_raw": 1.0, "draws_omitted": [], "omitted_left": 0})
                continue
            nodes = g.node_set()
            ones = {n: torch.ones(tx.T, len(circ[n]), device=dev) for n in nat.names}
            lm = Em.log_probs(without(ones, cs, at))
            kw = self._xkw(tx.ids, tx.targets, "omitted")  # (at x: a removed candidate stays at 0, never at x)
            lv = Ev.log_probs(without(vpd, cs, vat), **(dict(kw, held={n: (vpd[n] == 0)[None] for n in nat.names}) if kw else {}))
            if g.complete:  # the circuit's own run (no candidate is its): one row
                gm = {n: torch.zeros(tx.T, len(circ[n]), device=dev) for n in nat.names}
                for n, t, c in nodes:
                    gm[n][t, at[n][c]] = 1.0
                lc = Ec.log_probs({n: x[None] for n, x in gm.items()}, **kw)
            else:
                lc = Ec.graph_log_probs(Ec.plan([g], semantics="c"), **kw)
            lc = lc.expand(len(cs), *lc.shape[1:])
            w = carry.kl(lm, tx.ref.expand_as(lm)).tolist()
            kc, kv = carry.kl(lm, lc).tolist(), carry.kl(lm, lv).tolist()
            rec = [[list(nd), a, b_, c_] for nd, a, b_, c_ in zip(cs, kc, kv, w)]
            s_, r_ = _ratio_pool(rec), _ratio_pool(rec, raw=True)
            out.append({"s_omitted": 1.0 if s_ is None else s_, "s_omitted_raw": 1.0 if r_ is None else r_, "draws_omitted": rec,
                        "omitted_left": len(cs)})
            native.free(dev)
        return out

    @torch.no_grad()
    def _node_family(self, tx: Text, graphs: list[native.Graph], draws: int, reads: bool = False,
                     units: list | None = None) -> list[dict]:
        """Per circuit, the node-removal family: draws of its nodes removed from the model, from VPD's answer and from
        the circuit (a stated graph keeps its connections among the rest), the mean KLs from the model's. units: per
        graph its removal units (default its nodes): a block (site, t, BLOCK) is one unit, every subcomponent of its site
        there removed together (backup's report: a block's 3,072 subcomponents drawn one by one took 96% of every draw). The runs
        multiply only the circuits' and VPD's subcomponents (engine.Engine's columns; the model's run with every other
        subcomponent folded into its remainder, outside="on"): float32-close to the masked forward over every column.
        Every circuit's draws (_node_draws' common draws) run together, NODE_ENTRIES floats a batch, each row's masks
        written as matrices over the circuits' nodes: without reads the model and VPD's answer from the draws' first
        position on (Engine.log_probs' start), a complete circuit from there at its own masks; a stated circuit on its
        one plan with each draw's nodes out (graph_log_probs' off)."""
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
        vbase = {n: tx.vpd[n][:, vcols[n]].float()[None] for n in nat.names}
        every = sorted(set().union(*(g.node_set() for g in graphs))) if graphs else []
        nid = {nd: i for i, nd in enumerate(every)}
        N = len(every)
        oc, ov, KC, KV = {}, {}, 0, 0  # each site's offset in the flat masks over every site's columns (circuits', VPD's)
        for n in nat.names:
            oc[n], ov[n] = KC, KV
            KC, KV = KC + len(circ[n]), KV + len(vcols[n])
        npos = np.array([nd[1] for nd in every], dtype=np.int64)
        FT = torch.as_tensor(npos, device=dev)  # per node its position and its flat columns (circuits', VPD's)
        FC = torch.tensor([oc[n] + at[n][c] for n, _, c in every], dtype=torch.long, device=dev)
        FV = torch.tensor([ov[n] + vat[n][c] for n, _, c in every], dtype=torch.long, device=dev)

        def views(flat, off, cols):  # a flat mask [R, T, KC or KV] as the engine's {site: [R, T, K]} (views)
            return {n: flat[..., off[n]:off[n] + len(cols[n])] for n in nat.names}
        gnodes = [sorted(g.node_set()) for g in graphs]
        rows = []  # (graph, draw j, dropped node indices)
        for gi in range(len(graphs)):
            if gnodes[gi]:
                for j, d in enumerate(self._node_draws(tx, sorted(units[gi]) if units is not None else gnodes[gi], draws, held=gnodes[gi])):
                    rows.append((gi, j, np.array([nid[x] for x in d], dtype=np.int64)))
        cat = np.concatenate([d for _, _, d in rows]) if rows else np.zeros(0, dtype=np.int64)
        first = int(npos[cat].min()) if len(cat) else T
        kw = self._xkw(tx.ids, tx.targets, "nodes")  # at x: the removed nodes at 0, never at x
        start = 0 if reads else min(first, min(tx.targets))  # before it every run is its base's (reads: every position)
        width = max(nat.C.values()) * (T - start) + sum(len(c) for c in vcols.values()) * T
        step = max(1, self.NODE_ENTRIES // max(1, width))
        vflat = torch.cat([vbase[n] for n in nat.names], -1)  # [1, T, KV]
        own, plans = {}, {}  # per complete circuit its flat masks [1, T, KC]; per stated one its plan and entries' node indices
        for gi, g in enumerate(graphs):
            if not gnodes[gi]:
                continue
            if g.complete:
                m = torch.zeros(1, T, KC, device=dev)
                ii = torch.as_tensor([nid[x] for x in gnodes[gi]], device=dev)
                m[0, FT[ii], FC[ii]] = 1.0
                own[gi] = m
            else:  # the plan's entries per site: sorted by position, then column
                order = {}
                for n, t, c in gnodes[gi]:
                    order.setdefault(n, []).append((t, at[n][c], nid[(n, t, c)]))
                plans[gi] = (Ec.plan(g, semantics="c"), {n: torch.as_tensor([x[2] for x in sorted(v)], device=dev) for n, v in order.items()})
        res = {gi: [] for gi in range(len(graphs))}  # per graph per draw [j, kc, kv, w, claim]
        rd = {gi: ([], []) for gi in range(len(graphs))}  # per graph per draw its nodes' reads in the model and the circuit

        def spec(part):  # the reads of each row's circuit nodes: {site: (rows, positions, columns)}
            out_ = {}
            for r, (gi, _, _) in enumerate(part):
                for n, t, c in gnodes[gi]:
                    x = out_.setdefault(n, ([], [], []))
                    x[0].append(r)
                    x[1].append(t)
                    x[2].append(at[n][c])
            return {n: tuple(torch.tensor(v, device=dev) for v in x) for n, x in out_.items()}

        def split(got, part):  # {site: [k]} in spec's order -> per row its nodes' reads
            vals = {n: v.tolist() for n, v in got.items()}
            pos, per = {}, []
            for gi, _, _ in part:
                row = {}
                for nd in gnodes[gi]:
                    k = pos.get(nd[0], 0)
                    row[nd] = vals[nd[0]][k]
                    pos[nd[0]] = k + 1
                per.append(row)
            return per
        for s0 in range(0, len(rows), step):
            part = rows[s0:s0 + step]
            R = len(part)
            rr = np.concatenate([np.full(len(d), r) for r, (_, _, d) in enumerate(part)])
            cc = np.concatenate([d for _, _, d in part])
            Dm = torch.zeros(R, N, dtype=torch.bool, device=dev)
            r_, j_ = torch.as_tensor(rr, device=dev), torch.as_tensor(cc, device=dev)
            Dm[r_, j_] = True
            mf, vf = torch.ones(R, T, KC, device=dev), vflat.repeat(R, 1, 1)  # each draw's nodes off: one indexed write
            mf[r_, FT[j_], FC[j_]] = 0.0                                       # per engine, every site at once
            vf[r_, FT[j_], FV[j_]] = 0.0
            mm, mv = views(mf, oc, circ), views(vf, ov, vcols)
            if reads:
                lm, got = Em.log_probs(mm, reads=spec(part))
                rmod = split(got, part)
            else:
                lm = Em.log_probs(mm, start=start) if start else Em.log_probs(mm)
            vk = dict(kw, held={n: vbase[n] == 0 for n in nat.names}) if kw else {}
            lv = Ev.log_probs(mv, start=start, base=vbase, **vk) if start else Ev.log_probs(mv, **vk)
            lc = torch.empty_like(lm)
            rcir = [None] * R
            for gi in dict.fromkeys(gi for gi, _, _ in part):  # each circuit's rows
                ix = [r for r, x in enumerate(part) if x[0] == gi]
                ixt = torch.as_tensor(ix, device=dev)
                sub = [part[r] for r in ix]
                if gi in own:
                    cf = own[gi].repeat(len(ix), 1, 1)
                    r2, j2 = Dm[ixt].nonzero(as_tuple=True)
                    cf[r2, FT[j2], FC[j2]] = 0.0
                    mc, base_c = views(cf, oc, circ), views(own[gi], oc, circ)
                    kh = dict(kw, held={n: base_c[n] == 0 for n in nat.names}) if kw else {}
                    if reads:
                        got, rc_ = Ec.log_probs(mc, reads=spec(sub), **kh)
                    else:
                        got = Ec.log_probs(mc, start=start, base=base_c, **kh) if start else Ec.log_probs(mc, **kh)
                else:
                    plan, ent = plans[gi]
                    off = {n: Dm[ixt][:, e] for n, e in ent.items()}
                    zero = {n: torch.zeros(len(ix), 1, Ec.width(n), device=dev) for n in nat.names}  # (B rows: u's)
                    if reads:
                        got, rc_ = Ec.graph_log_probs(plan, zero, off=off, reads=spec(sub), **kw)
                    else:
                        got = Ec.graph_log_probs(plan, zero, off=off, **kw)
                lc[ixt] = got
                if reads:
                    for r, x in zip(ix, split(rc_, sub)):
                        rcir[r] = x
            kc, kv = carry.kl(lm, lc).tolist(), carry.kl(lm, lv).tolist()
            w, claim = carry.kl(lm, tx.ref.expand_as(lm)).tolist(), carry.kl(tx.ref.expand_as(lc), lc).tolist()
            for r, (gi, j, _) in enumerate(part):
                res[gi].append([j, kc[r], kv[r], w[r], claim[r]])
                if reads:
                    rd[gi][0].append(rmod[r])
                    rd[gi][1].append(rcir[r])
            native.free(dev)
        out = []
        for gi in range(len(graphs)):
            if not gnodes[gi]:
                out.append({"kl_nodes_c": self._bounds(tx, "live", "nodes")[0], "kl_nodes_vpd": self._bounds(tx, "live", "nodes")[1], "reads_nodes": None})
                continue
            dr = res[gi]
            row = {"kl_nodes_c": float(np.mean([x[1] for x in dr])), "kl_nodes_vpd": float(np.mean([x[2] for x in dr])),
                   # per draw [draw, KL(model less d || circuit less d), KL(model less d || VPD less d),
                   # W = KL(model less d || model), KL(model || circuit less d)]
                   "draws_nodes": dr, "overstate_nodes": float(np.mean([x[4] for x in dr])) / max(float(np.mean([x[3] for x in dr])), 1e-12),
                   "reads_nodes": None, "_pairs_nodes": {}}
            if reads:  # the reads of the nodes each draw keeps, in the model and the circuit
                gn = gnodes[gi]
                gi_rows = [d for g_, _, d in rows if g_ == gi]
                kept = ~np.stack([np.isin(np.array([nid[nd] for nd in gn]), d) for d in gi_rows])  # [draws, n]
                M = np.array([[x[nd] for nd in gn] for x in rd[gi][0]], dtype=float)
                Cc = np.array([[x[nd] for nd in gn] for x in rd[gi][1]], dtype=float)
                row["_pairs_nodes"] = {nd: list(zip(M[kept[:, k], k].tolist(), Cc[kept[:, k], k].tolist()))
                                       for k, nd in enumerate(gn) if kept[:, k].any()}
                cnt = kept.sum(0)  # _read_error on the arrays: per node over the draws keeping it
                msq = np.where(kept, M ** 2, 0.0).sum(0) / np.maximum(cnt, 1)
                err = np.where(kept, (M - Cc) ** 2, 0.0).sum(0) / np.maximum(cnt, 1)
                ok = (cnt > 0) & (msq > 0)
                row["reads_nodes"] = float(np.mean(np.sqrt(err[ok] / msq[ok]))) if ok.any() else None
            out.append(row)
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
        draws_of = [self._edge_draws(tx, g, draws) for g in graphs]
        if not reads:  # every circuit's draws in one run of each side
            items = [(g, c) for g, c in zip(graphs, draws_of) if c is not None]
            got_all = iter(sides.many(items)) if items else iter(())
        out = []
        for g, cuts in zip(graphs, draws_of):
            if cuts is None:
                out.append({"kl_edges_c": None, "kl_edges_vpd": None, "reads_edges": None})
                continue
            got = sides(g, cuts=cuts, reads=reads) if reads else next(got_all)
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
            ws = sorted(par[r]) if r is not None else sorted(outs)  # all_edges' lists follow the node set's order, per process
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
            idx, val = {}, {}  # one indexed write per site (a write per node was a kernel launch each: a block's
            for b, vals in enumerate(patch):  # thousands of subcomponents per draw kept the GPU idle)
                for (n, t, c), v in vals.items():
                    idx.setdefault(n, []).append((b, t, at[id(E)][n][c]))
                    val.setdefault(n, []).append(v)
            out = {}
            for n, ix in idx.items():
                m = torch.zeros(len(patch), T, E.width(n), dtype=torch.bool, device=dev)
                x = torch.zeros(len(patch), T, E.width(n), device=dev)
                i = torch.tensor(ix, device=dev).T
                vs = val[n]
                v = torch.stack([u.reshape(()) for u in vs]) if torch.is_tensor(vs[0]) else torch.tensor(vs)
                m[i[0], i[1], i[2]] = True
                x[i[0], i[1], i[2]] = v.to(device=dev, dtype=x.dtype)
                out[n] = (m, x)
            return out

        def spec(E, nodes, D):  # every row's reads of nodes: {site: (rows, positions, columns)} (blocks have no activation)
            out = {}
            for n, t, c in (nd for nd in nodes if nd[2] != BLOCK):
                x = out.setdefault(n, ([], [], []))
                x[1].append(t)
                x[2].append(at[id(E)][n][c])
            return {n: (torch.arange(D, device=dev).repeat_interleave(len(t)), torch.tensor(t, device=dev).repeat(D),
                        torch.tensor(c, device=dev).repeat(D)) for n, (_, t, c) in out.items()}

        def split(got, nodes, D):  # {site: [D * k]} -> {node: [D]}
            out, k = {}, {}
            for n, t, c in (nd for nd in nodes if nd[2] != BLOCK):
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
                    if c != BLOCK:  # (a block is the plan's unit: its whole output)
                        u_c[n][0, t, at[id(Ec)][n][c]] = 1.0
            lc = Ec.graph_log_probs(Ec.plan([g] * D, semantics="c", cuts=cuts), u_c, patch=patches(Ec, patch),
                                    reads=spec(Ec, nodes, D) if reads else None)
            if not reads:
                return lm, lv, lc
            return lm[0], lv, lc[0], split(lm[1], nodes, D), split(lc[1], nodes, D)

        def many(items):  # [(circuit, its cuts per draw)] -> per circuit (log q of the model, VPD's answer, the circuit), every
            G = [g for g, c in items for _ in c]  # circuit's draws in one run of each side
            C = [x for _, c in items for x in c]
            R = len(C)
            lm = Em.graph_log_probs(Em.plan([model] * R, semantics="c", cuts=C), u_m)
            lv = Ev.graph_log_probs(Ev.plan([vpd] * R, semantics="c", cuts=C), u_v)
            u_c = None
            if any(g.complete for g in G):  # complete circuits run as u: 1 on each row's circuit's nodes
                idx = {}
                for r, g in enumerate(G):
                    if g.complete:
                        for n, t, c in g.node_set():
                            if c != BLOCK:
                                x = idx.setdefault(n, ([], [], []))
                                x[0].append(r)
                                x[1].append(t)
                                x[2].append(at[id(Ec)][n][c])
                u_c = {n: torch.zeros(R, T, Ec.width(n), device=dev) for n in nat.names}
                for n, x in idx.items():
                    u_c[n][tuple(torch.tensor(v, device=dev) for v in x)] = 1.0
            lc = Ec.graph_log_probs(Ec.plan(G, semantics="c", cuts=C), u_c)
            out, r0 = [], 0
            for _, c in items:
                out.append((lm[r0:r0 + len(c)], lv[r0:r0 + len(c)], lc[r0:r0 + len(c)]))
                r0 += len(c)
            return out

        sides.spec, sides.split, sides.engines, sides.at, sides.many = spec, split, (Em, Ev, Ec), at, many
        return sides

    @torch.no_grad()
    def _fresh_sources(self, tx: Text, p: int, seed: int) -> list[int]:
        """FRESH_SOURCES tokens for position p sampled without replacement from the model's distribution there, the
        text's token and the three likeliest others (every split's carry.alternatives ranks) left out; keyed by seed and
        p alone."""
        lp = self.nat.replacement_logp(tx.ids, p).cpu().double()  # MPS has no float64
        others = [int(z) for z in lp.topk(4).indices.tolist() if z != tx.ids[p]][:3]
        w = lp.exp()
        w[[tx.ids[p]] + others] = 0.0
        rng = np.random.default_rng(int(hashlib.sha256(repr(("fresh", seed, p)).encode()).hexdigest()[:15], 16))
        wn = w.numpy()                                       # without replacement as an exponential race on numpy's
        ok = wn > 0                                          # stream (torch.multinomial's draws changed between torch
        keys = np.full(len(wn), np.inf)                      # versions: native.stable_rng)
        keys[ok] = -np.log(1.0 - rng.random(int(ok.sum()))) / wn[ok]
        return np.argsort(keys, kind="stable")[:min(FRESH_SOURCES, int(ok.sum()))].tolist()

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
        own = [pz for pz in getattr(tx, "behavior_pairs", []) if 0 < pz[0] < tx.T]  # a behavior's minimal pairs, first
        return (own + sorted((pz for pz in pairs if pz not in own),
                             key=lambda pz: (first.get(pz[0], 0), hashlib.sha256(repr((seed, pz)).encode()).digest())))[:draws]

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
        Prompts: F2's positions (the sensitivity probe's, f2_cut of its moved), the most moved first, each
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
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]
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

    def recency_set(self, tx: Text, k: int) -> native.Graph:
        """The recency baseline at a size (the lead's 10-12, reported beside VPD's top k): VPD's actives at the targets,
        then at the positions just before the last target one by one, every one of a position's while they fit in k,
        the last position's in VPD's order (tx.vpd_order, else its own) to make k; every connection (a node set)."""
        order = getattr(tx, "vpd_order", None) or self.nat._vpd_order(tx.ids, tx.targets)
        rank = {tuple(nd): i for i, nd in enumerate(order)}
        at = {}
        for n in self.nat.names:
            for t, c in tx.vpd[n].nonzero().tolist():
                at.setdefault(t, []).append((n, t, c))
        last = max(tx.targets)
        seq = sorted(tx.targets, reverse=True) + [t for t in range(last - 1, -1, -1) if t not in set(tx.targets)]
        nodes = []
        for t in seq:
            here = sorted(at.get(t, []), key=lambda nd: rank.get(nd, len(rank)))
            nodes += here[:max(0, k - len(nodes))]
            if len(nodes) >= k:
                break
        return native.Graph(nodes, complete=True)

    @torch.no_grad()
    def resample_influence(self, tx: Text, split: str = "search", k: int = RESAMPLE_DONORS) -> dict:
        """{position: mean KL in bits at the targets of the model with that position's token alone replaced by the
        token at the same place in each of k far donors of split (far_donors, aligned at their ends; a donor whose
        token there is the text's gives its next differing one)}, the positions from 1 to the last target outside the
        targets (from 0 with FIRST_POSITION, the default: the first token, which the probe's draws from the model's own
        prediction never change, there being none there; off on a target whose texts start with a BOS it always reads): the
        input-scope family's intervention one position at a time, natural tokens the model reads in place of the
        text's (the lead's 10-13). On the lead's 17 hand-checked texts it ranks the context token that moves the
        prediction most first on 7 of the 9 that need context, second or third on the others; the probe's draws from the
        model's own prediction (moved) miss tokens with plausible substitutes (text9043's ' adjusted' one back: 0.57
        bits by the probe, 1.03 by donors, 9.18 replaced by ' the'). Kept on tx."""
        kept = tx.__dict__.setdefault("_resample", {})
        if (split, k, self.FIRST_POSITION) not in kept:
            donors = [list(d) for _, d in self.far_donors(tx, self.far_pool(split), k)]
            tg = set(tx.targets)
            qs = [q for q in range(0 if self.FIRST_POSITION else 1, max(tx.targets) + 1) if q not in tg]
            seqs = [[_donor_token(d, q, tx.T, x) if p == q else x for p, x in enumerate(tx.ids)] for q in qs for d in donors]
            kl = []
            for s0 in range(0, len(seqs), 4 * self.chunk):
                lp = self.nat.reference(seqs[s0:s0 + 4 * self.chunk], tx.targets)
                kl += carry.kl(lp, tx.ref.expand_as(lp)).tolist()
                native.free(self.nat.dev)
            n = len(donors)
            kept[(split, k, self.FIRST_POSITION)] = {q: float(np.mean(kl[i * n:(i + 1) * n])) for i, q in enumerate(qs)} if n else {}
        return kept[(split, k, self.FIRST_POSITION)]

    def start_candidates(self, tx: Text, split: str = "search") -> list:
        """The START's candidate positions in order (the lead's 10-13): F2's by the probe's moved and every position by
        resample_influence, largest first, interleaved, each position once, the targets left out."""
        moved, tg = np.asarray(tx.moved, dtype=float), set(tx.targets)
        f2 = [q for q in sorted((q for q, _ in tx.f2), key=lambda q: -moved[q - 1]) if q not in tg]
        one = self.resample_influence(tx, split)
        rs = sorted(one, key=lambda q: -one[q])
        out, seen = [], set()
        for i in range(max(len(f2), len(rs))):
            for q in (f2[i] if i < len(f2) else None, rs[i] if i < len(rs) else None):
                if q is not None and q not in seen:
                    seen.add(q)
                    out.append(q)
        return out

    def scope_draws(self, tx: Text, split: str = "search", part: str | None = None) -> list:
        """The input-scope family's changed texts as (position, token) pairs: the positions by resample_influence on
        split, largest first, until they hold SCOPE_COVER of its sum (at most START_CAP), each with its token from each
        of the split's RESAMPLE_DONORS far donors (one changed position a text). part "accept" or "validate": the half a
        hash of (the text's seed, position, token) puts there (the teachers' acceptance and its held-out guard)."""
        one = self.resample_influence(tx, split)
        total = sum(max(0.0, v) for v in one.values())
        pos, held = [], 0.0
        for q in sorted(one, key=lambda q: -one[q]):
            if len(pos) >= START_CAP or not total > 0 or held >= SCOPE_COVER * total:
                break
            pos.append(q)
            held += max(0.0, one[q])
        donors = [list(d) for _, d in self.far_donors(tx, self.far_pool(split), RESAMPLE_DONORS)]
        got = [(q, _donor_token(d, q, tx.T, tx.ids[q])) for q in pos for d in donors]
        got = list(dict.fromkeys((q, z) for q, z in got if z != tx.ids[q]))
        if part is None:
            return got
        side = {"accept": 0, "validate": 1}[part]
        return [pz for pz in got if hashlib.sha256(repr((int(tx.seed), int(pz[0]), int(pz[1]))).encode()).digest()[0] % 2 == side]

    def scope_family(self, tx: Text, graphs: list[native.Graph], split: str = "search", lines: torch.Tensor | None = None,
                     part: str | None = None) -> list[dict]:
        """The input-scope family (the lead's 10-13 design): does the circuit follow the model when a token it may
        leave out changes? Draws: scope_draws (each influential position's token alone replaced by natural donor
        tokens, positions to SCOPE_COVER of the summed movement, at most START_CAP). Scored as track_pp on those draws:
        path patching (pp_run), the change reaching the circuit only through its token edges at the position and
        travelling only along its paths, everything outside it at the text's own run; its own q/k parts change with
        the input, so a routing change must be the circuit's, only the baseline routing being the real run's. Against
        VPD's whole answer in the same semantics (1 by construction: floor FLOOR_TOP), bounded by no change, pooled as
        the weighted families (_ratio_pool; the raw score two-sided: a circuit farther than no change, responding more
        than the model, counts against it). This family decides whether a position the circuit leaves out counts
        against it. The circuit's own run, everything outside it zeroed, is no measure here: a sparse circuit's
        attention then spreads over the positions it leaves empty (on 60 texts the START's own run followed the
        model's token changes nowhere, median 0.00, while VPD's whole answer did). Per position alone: a context
        resampled jointly carries a donor's topic, which no small circuit holds. lines as track_pp's (phi_variants); part,
        scope_draws' half. {s_scope, s_scope_raw, draws_scope}."""
        got = self.scope_draws(tx, split, part) if tx.vpd is not None else []
        if not got:
            return [{"s_scope": None, "s_scope_raw": None, "draws_scope": []} for _ in (graphs if lines is None else lines)]
        rows = self.track_pp(tx, graphs, split, lines=lines, got=got)
        return [{"s_scope": r["s_track_pp"], "s_scope_raw": r["s_track_pp_raw"], "draws_scope": r["draws_track_pp"]} for r in rows]

    @torch.no_grad()
    def kl_resolution(self, tx: Text) -> float:
        """The KL resolution of C's own runs on the text (the lead's 10-13): RESOLUTION_SAFETY times the KL in bits at
        the targets between two runs of VPD's whole answer that differ only in their batch (alone, and beside a run with
        its first context position's parts cut), the larger direction; kept on tx. own_rule counts an effect below it
        as none."""
        if "_kl_resolution" not in tx.__dict__:
            import engine

            nat = self.nat
            act = sorted((n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist())
            circ = {n: torch.tensor(sorted({c for m, _, c in act if m == n}), dtype=torch.long, device=nat.dev) for n in nat.names}
            at = {n: {c: i for i, c in enumerate(circ[n].tolist())} for n in nat.names}
            E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ)
            base = {n: torch.zeros(tx.T, len(circ[n]), device=nat.dev) for n in nat.names}
            for n, t, c in act:
                base[n][t, at[n][c]] = 1.0
            cut = {n: x.clone() for n, x in base.items()}
            ctx = [nd for nd in act if nd[1] not in set(tx.targets)]
            for n, t, c in [nd for nd in ctx if nd[1] == ctx[0][1]] if ctx else []:
                cut[n][t, at[n][c]] = 0.0
            one = E.log_probs({n: x[None] for n, x in base.items()})
            two = E.log_probs({n: torch.stack([base[n], cut[n]]) for n in nat.names})[:1]
            noise = max(abs(float(carry.kl(one, two))), abs(float(carry.kl(two, one))))
            tx.__dict__["_kl_resolution"] = RESOLUTION_SAFETY * noise
            native.free(nat.dev)
        return tx.__dict__["_kl_resolution"]

    @torch.no_grad()
    def own_effects(self, tx: Text, g: native.Graph, positions=None) -> dict:
        """{position: KL in bits at the targets between the circuit's own run (C's semantics: its parts on with their
        own attention routing, nothing held at x, the leftover weights on) and that run with the position's parts cut},
        for the circuit's positions outside the targets (or positions): whether a kept position works in the circuit's
        own computation (the lead's 10-13). On the 492 v2.3 teachers, 395 of the 796 kept context positions changed
        their circuit's own run by under 1e-3 bits while 779 moved the real model by at least 0.01 bits; text9000's 20
        parts on "the" one back: 5.4e-5 bits in its own run, 0.37 with q/k and norms at x, 0.12 in the model."""
        import engine

        nat = self.nat
        nodes = sorted(g.node_set())
        pos = sorted({t for _, t, _ in nodes} - set(tx.targets)) if positions is None else sorted(positions)
        if not pos:
            return {}
        circ = {n: torch.tensor(sorted({c for m, _, c in nodes if m == n}), dtype=torch.long, device=nat.dev) for n in nat.names}
        at = {n: {c: i for i, c in enumerate(circ[n].tolist())} for n in nat.names}
        E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ)
        base = {n: torch.zeros(tx.T, len(circ[n]), device=nat.dev) for n in nat.names}
        for n, t, c in nodes:
            base[n][t, at[n][c]] = 1.0
        sets = [[]] + [[(n, t, at[n][c]) for n, t, c in nodes if t == q] for q in pos]
        lps = []
        for s0 in range(0, len(sets), self.chunk):
            part = sets[s0:s0 + self.chunk]
            m = {n: x[None].repeat(len(part), 1, 1).clone() for n, x in base.items()}
            for b, cut in enumerate(part):
                for n, t, i in cut:
                    m[n][b, t, i] = 0.0
            lps.append(E.log_probs(m))
            native.free(nat.dev)
        lp = torch.cat(lps)
        return dict(zip(pos, carry.kl(lp[:1].expand_as(lp[1:]), lp[1:]).tolist()))

    @torch.no_grad()
    def removal_kl(self, tx: Text, sets: list) -> list[float]:
        """KL in bits at the targets of the real model with each set of parts ((matrix, position, subcomponent))
        removed, against the model: every other subcomponent and the leftover weights on (the omitted family's model
        run)."""
        import engine

        nat = self.nat
        cols = {n: torch.tensor(sorted({c for s_ in sets for m, _, c in s_ if m == n}), dtype=torch.long, device=nat.dev)
                for n in nat.names}
        at = {n: {c: i for i, c in enumerate(cols[n].tolist())} for n in nat.names}
        E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols, outside="on")
        out = []
        for s0 in range(0, len(sets), self.chunk):
            part = sets[s0:s0 + self.chunk]
            m = {n: torch.ones(len(part), tx.T, len(cols[n]), device=nat.dev) for n in nat.names}
            for b, nodes in enumerate(part):
                for n, t, c in nodes:
                    m[n][b, t, at[n][c]] = 0.0
            lp = E.log_probs(m)
            out += carry.kl(lp, tx.ref.expand_as(lp)).tolist()
            native.free(nat.dev)
        return out

    def own_rule(self, tx: Text, g: native.Graph, positions=None, floor: float = FLOOR_TOP) -> dict:
        """The lead's 10-13 rule for a circuit's context positions, its own run against VPD's own answer run alone (C's
        semantics, own routing; holding q/k at x borrows the model's routing, which is no explanation): per position
        (the circuit's outside the targets, or positions) {"own": own_effects of its parts in the circuit, "vpd_own":
        own_effects of VPD's actives there in VPD's whole answer, "real": its parts removed from the real model
        (removal_kl), "vpd_real": VPD's actives there removed from it, "vpd_depends": VPD's own answer reproduces at
        least 1 - floor of the model's dependence there and more than the runs' KL resolution (vpd_own >= (1 - floor)
        vpd_real and vpd_own > kl_resolution: a tiny real effect is no dependence), "works": own >= floor vpd_own where
        VPD depends on the position, else own >= (1 - floor) real and own > kl_resolution, "drop": neither VPD's answer
        nor the circuit's own run depends on it}. On 60 v2.3 teachers 125 of 142 positions inert in the circuit's own run were inert in
        VPD's whole answer too (the model's dependence runs through parts VPD marks inactive): a search over VPD's parts
        cannot express it. A diagnostic (the lead's 10-13 ruling): the own run zeroes everything outside the circuit,
        and a sparse circuit's attention then spreads over the positions it leaves empty, so the path-patched families
        decide which positions a circuit keeps."""
        nodes = sorted(g.node_set())
        pos = sorted({t for _, t, _ in nodes} - set(tx.targets)) if positions is None else sorted(positions)
        if not pos:
            return {}
        own = self.own_effects(tx, g, pos)
        cache = tx.__dict__.setdefault("_vpd_positions", {})
        new = [q for q in pos if q not in cache]
        if new:
            act = [(n, t, c) for n in self.nat.names for t, c in tx.vpd[n].nonzero().tolist()]
            vo = self.own_effects(tx, native.Graph(act, complete=True), new)
            vr = self.removal_kl(tx, [[nd for nd in act if nd[1] == q] for q in new])
            cache.update({q: (vo[q], r) for q, r in zip(new, vr)})
        real = self.removal_kl(tx, [[nd for nd in nodes if nd[1] == q] for q in pos])
        res = self.kl_resolution(tx)
        out = {}
        for q, r in zip(pos, real):
            vo, vr = cache[q]
            dep = vo >= (1.0 - floor) * vr and vo > res
            works = own[q] >= floor * vo if dep else (own[q] >= (1.0 - floor) * r and own[q] > res)
            out[q] = {"own": own[q], "vpd_own": vo, "real": r, "vpd_real": vr, "vpd_depends": dep, "works": works,
                      "drop": not dep and not works}
        return out

    def _start_pool(self):
        if getattr(self, "start_pool", None) is None:
            self.start_pool = torch.load(START_POOL)
        return self.start_pool

    def start_set(self, tx: Text, split: str | None = None, floor: float = FLOOR_TOP, draws: int = CHANGED_DRAWS,
                  pool: dict | None = None) -> dict:
        """The START (the lead's 10-13 design), one set per text for every split, chosen on the search draws (primerl:
        a START listed from the search draws beside an eval bar from the eval draws had been two sets): VPD's actives at
        the targets and at the fewest of start_candidates (F2's and the resampling ranking interleaved, at most
        START_CAP) whose reward-mode row on the search split meets the reward's floors (vpd_floors; shortfall 0): the
        path-patched families (input scope, track_pp, interchange_pp) decide which positions it needs. The old rule,
        the fewest F2 positions reaching s_track_pp 0.9, passed at m = 0 on 40% of the v2.3 teachers: the targets' nodes
        patched from a changed run take up every change upstream of them. Prefixes m = 0, 1, 2, 4, 8, 16 are tried, the
        first passing one bisected down to its smallest passing neighbour; every candidate when none passes. "effects":
        own_rule's numbers per context position of the chosen set (own run, VPD's own answer, the real model), a
        diagnostic. split is ignored; pool, the flip adversary's (START_POOL's, the shared one, when not given: every
        caller's is that file). "cap_bound": the START holds every candidate, the own-run families' reference set, so
        it meets their floors by construction (an evaluation counts these texts). Kept on tx. {graph, m, positions,
        candidates, cap_bound, effects, shortfall, met, s_track_pp, scores}."""
        kept = tx.__dict__.setdefault("_start", {})
        if floor in kept:
            return kept[floor]
        nat = self.nat
        act = [(n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist()]
        cands = self.start_candidates(tx, "search")[:START_CAP]
        tried = {}
        pool = self._start_pool() if pool is None else pool
        fl = self.vpd_floors(tx, pool, "search", floor)

        def judged(ps: set) -> tuple:
            g = native.Graph(sorted(nd for nd in act if nd[1] in ps), complete=True)
            row = self.fidelity(tx, [g], pool, mode="reward", split="search", node_values=False)[0]
            return g, row, fl["families"], shortfall(row, fl["floors"])

        def attempt(m: int) -> dict:
            if m not in tried:
                ps = set(tx.targets) | set(cands[:m])
                g, row, fams, v = judged(ps)
                tried[m] = {"graph": g, "m": m, "positions": sorted(ps), "shortfall": v, "met": v == 0,
                            "s_track_pp": row.get("s_track_pp"), "scores": {f: family_score(row, f) for f in fams}}
            return tried[m]

        steps = sorted({0, len(cands)} | {2 ** i for i in range(len(cands).bit_length()) if 2 ** i < len(cands)})
        hit = next((m for m in steps if attempt(m)["met"]), None)
        if hit is None:
            best = attempt(len(cands))
        else:
            lo = max([m for m in steps if m < hit], default=-1)
            while hit - lo > 1:
                mid = (lo + hit) // 2
                lo, hit = (lo, mid) if attempt(mid)["met"] else (mid, hit)
            best = attempt(hit)
        kept[floor] = {**best, "candidates": cands, "cap_bound": best["m"] == len(cands),  # (own-run floors met by construction)
                       "effects": self.own_rule(tx, best["graph"], floor=floor)}  # (a diagnostic)
        return kept[floor]

    def vpd_floors(self, tx: Text, pool: dict | None, split: str = "search", floor: float = FLOOR_TOP) -> dict:
        """The reward's floors on a text (the lead's 10-13 design: absolute, from VPD, never from the answer or a
        start): reward-mode rows on split of VPD's whole answer, of VPD's actives at the targets and the START's
        candidate positions (start_candidates, at most START_CAP: fixed per text, so no answer lowers its own bar by
        leaving positions out) and of the empty circuit; the text's families (those VPD's whole answer scores: every
        answer's phi on the text is over them, an unscored one at 0) and floors: floor of the candidates' reference on
        the families run with the circuit's outside zeroed or at a far donor's streams (OWN_RUN_FAMILIES: a sparse
        circuit's attention then spreads over the positions it leaves empty, which only VPD's whole answer avoids: on
        60 texts 45 STARTs stayed under 0.9 of it on point, and with point referenced to the candidates 22 of the 30
        failing STARTs bound on patching_far), of the whole answer on the rest (input scope, track_pp,
        interchange_pp, N, omitted). Every family's score itself stays against VPD's whole answer (the claim quotes
        those); the floors alone read the restricted reference. Kept on tx per split and floor (the pool: the shared
        adversary's, one file for every caller)."""
        kept = tx.__dict__.setdefault("_vpd_floors", {})
        if (split, floor) not in kept:
            act = sorted((n, t, c) for n in self.nat.names for t, c in tx.vpd[n].nonzero().tolist())
            ps = set(tx.targets) | set(self.start_candidates(tx)[:START_CAP])
            vrow, crow, er = self.fidelity(tx, [native.Graph(act, complete=True),
                                                native.Graph([nd for nd in act if nd[1] in ps], complete=True), native.Graph()],
                                           pool, mode="reward", split=split, node_values=False)
            fams = [f for f in PHI_FAMILIES if family_score(vrow, f) is not None]
            fl = {**floors(vrow, fams), **floors(crow, [f for f in fams if f in OWN_RUN_FAMILIES])}
            if floor != FLOOR_TOP:
                fl = {f: v * floor / FLOOR_TOP for f, v in fl.items()}
            kept[(split, floor)] = {"vpd": vrow, "candidates": crow, "empty": er, "families": fams, "floors": fl}
        return kept[(split, floor)]

    def tau(self, tx: Text, pool: dict | None, split: str = "search") -> dict:
        """The reward's per-text bar, floors and start bits (the lead's 10-13 design: faithful first, then small; every
        threshold from VPD and the empty circuit, none from a start), kept on tx: vpd_floors' families and floors,
        phi_tau = max(phi at every family's floor under its reference's eps, phi_empty + 1: the bar sits at least a bit
        above the empty circuit's phi), phi_vpd (VPD's own phi), and of the START (start_set, one set for every split)
        its reward-mode row's phi_start, c_start, c_rl_start (tau = min(0.9, c_rl_start), the earlier form), bits_start
        (its raw description bits, size_bits: compactness's reference), m, nodes, positions and whether it met the
        floors."""
        kept = tx.__dict__.setdefault("_tau", {})
        if split in kept:
            return kept[split]
        st = self.start_set(tx, pool=pool)
        vf = self.vpd_floors(tx, pool, split)
        row = self.fidelity(tx, [st["graph"]], pool, mode="reward", split=split)[0]
        nat = self.nat
        fams, fl, er = vf["families"], vf["floors"], vf["empty"]
        ph = phi(row, families=fams)[0]
        ve = {**_eps(vf["vpd"]), **{f: v for f, v in _eps(vf["candidates"]).items() if f in OWN_RUN_FAMILIES}}
        pt = _phi([(fl[f], ve[f]) for f in fams])  # phi_tau: a constant of the text, from VPD's answer's eps
        pe = phi(er, families=fams)[0]
        if pt is not None and pe is not None:
            pt = max(pt, pe + 1.0)
        kept[split] = {"phi_start": ph, "phi_tau": pt, "phi_empty": pe, "families": fams, "floors": fl,
                       "phi_vpd": phi(vf["vpd"], families=fams)[0], "vpd_scores": {f: family_score(vf["vpd"], f) for f in fams},
                       "vpd_candidates_scores": {f: family_score(vf["candidates"], f) for f in fams if f in OWN_RUN_FAMILIES},
                       "phi_09": _phi([(0.9, ve[f]) for f in fams]), "phi_05": _phi([(0.5, ve[f]) for f in fams]),
                       "c_start": row["c"], "c_rl_start": row["c_rl"],
                       "tau": min(0.9, row["c_rl"]) if row["c_rl"] is not None else None,
                       "bits_start": size_bits(st["graph"], nat.positions(tx.targets), sum(nat.C.values()), tx.targets, active=tx.vpd),
                       "m_start": st["m"], "nodes_start": st["graph"].count(), "positions_start": len(st["positions"]),
                       "start_met": st.get("met"), "start_cap_bound": st.get("cap_bound")}
        return kept[split]

    def far_pool(self, split: str) -> list:
        """The far donors' texts of a split, [(name, token ids)], loaded once: DONOR_POOLS' vpd4l texts, refused for a
        model with another vocabulary (the lead's 10-13: vpd4l's token ids mean nothing to a GPT-2-tokenized target such
        as MIB's InterpBench model, so its START, input-scope family, patching and N would read foreign tokens; such a
        model's caller sets donor_pools[split] to texts tokenized for it, ib_teacher.py MIB's dev prompts). A pool set
        that way must lie in the model's vocabulary."""
        V = self.nat.target.n_tokens
        if split in self.donor_pools:
            bad = next((name for name, x in self.donor_pools[split] if len(x) and max(x) >= V), None)
            if bad is not None:
                raise ValueError(f"far donor {bad} of the {split} split holds token ids outside this model's {V}-id vocabulary")
        else:
            if V != DONOR_VOCAB:
                raise ValueError(f"the far donors (DONOR_POOLS) are vpd4l's texts in its {DONOR_VOCAB}-id vocabulary and this "
                                 f"model has {V} ids: set donor_pools['{split}'] to texts tokenized for this model")
            which, rows, strict = DONOR_POOLS[split]
            have = {int(re.sub(r"\D", "", p.stem)): p for p in native.tasks(which)}  # the partition's files here, by number
            missing = [n for n in rows if n not in have]
            if strict and missing:
                raise FileNotFoundError(f"the {split} split's far donors are text{rows.start}-text{rows.stop - 1}; {len(missing)} of "
                                        f"them are not in {native.TEXTS / 'vpd4l'} (text{missing[0]} first): its rewards need every one")
            self.donor_pools[split] = [(have[n].stem, x) for n in rows if n in have for x in [native.text(have[n])[0]]]
        return self.donor_pools[split]

    def pool_id(self, split: str) -> str:
        """The far-donor pool's identity, named on every reward row: the split, its size and a hash of its texts'
        names (far_pool). The search split's pool is the first 400 train texts present on the machine, so a pod
        without every train text draws other donors and scores other rewards (one train text added moved an answer
        on text6323 by 0.13, 10-13)."""
        names = [n for n, _ in self.far_pool(split)]
        return f"{split}:{len(names)}:{hashlib.sha256(json.dumps(names).encode()).hexdigest()[:12]}"

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
               far: list | None = None, rows: int = DONOR_ROWS, restrand: bool = True) -> list[dict]:
        """Activation patching, the spec's data-distribution VPD sufficiency (columns s_patching_*, named so by the lead
        10-12: the patched state mixes a donor's activations with the circuit's own, not a natural joint state of the
        model, so it is donor replacement, not an in-distribution test): the circuit run in C's semantics
        with every subcomponent outside it at its activation on one donor text, every one from that donor, so parts that
        cancel on real text stay consistent (the remainders live); paired with VPD's answer under the same donors (its
        and the empty side's KL per donor run once a text, kept on it: no circuit's); the
        KL from the model on the text. Donors: "near", the changed-input texts of split (changed_inputs' draws), and
        "far", far's texts (token id lists, other held-out texts), aligned at their ends, those shorter than the text
        leaving its first positions unresampled (their activations live). The bound is the run that keeps nothing,
        every subcomponent at the donor's (KL_0), not the empty circuit's KL: a near donor differs from the text at one
        position, so against KL_empty (~100 bits) every circuit would score near 1. Per circuit and pool:
        s_patching_<pool> = clip(ln(KL_0 / KL_C) / ln(KL_0 / KL_VPD), 0, 1) on the mean KLs; s_patching_<pool>_worst, the
        lowest donor's (a donor where VPD's answer is no nearer than KL_0 left out); s_patching_<pool>_p90, the donor at
        the 90th percentile from the best (the 10th percentile of the donors' scores); kl_patching_<pool> (KL_C, KL_VPD,
        KL_0); donors_<pool> (how many); draws_patching_<pool>, every donor's [name, KL_C, KL_VPD, KL_0, score]. Every
        draw is the KL from the model's own output on the text, never a loss against the text's next token: it is never
        negative, so a circuit's errors on different donors add up instead of cancelling (Scheurer et al. 2023, causal
        scrubbing's pitfalls). The flip adversary's donor-text counterpart: a circuit that holds here but not
        under the flips is charged there for combinations no real text makes (though no patched state here is a real
        text's either)."""
        import engine

        nat, T, dev = self.nat, tx.T, self.nat.dev
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) if restrand else expand_blocks(g, nat.C, tx.vpd) for g in graphs]  # (restrand
        texts = self._donor_texts(tx, split, near, far)  # False: as given, phi_variants' variants)
        E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=None)
        u = {n: torch.ones(1, 1, nat.C[n], device=dev) for n in nat.names}
        sides = list(graphs) + [native.Graph(masks=tx.vpd), native.Graph()]  # the circuits, VPD's answer, nothing kept
        own = []  # per side, its nodes as {site: bool [T, C]}: left unpatched
        for g in sides:
            m = self.masks(g, T)
            own.append({n: m[n].to(dev) for n in nat.names})
        base = tx.__dict__.setdefault("_donor_base", {})  # (pool, donor ids) -> (KL_VPD, KL_0): no circuit's, run once a text
        vs = (len(graphs), len(graphs) + 1)  # (VPD's answer's and the empty side's places in sides)
        out = [{} for _ in graphs]
        for pool, xs in texts.items():
            parts, cur = [], []  # (donor, side) rows, each donor's in one run: the circuits on every donor, VPD's answer
            for d, (_, x) in enumerate(xs):  # and nothing where not kept, about rows to a run
                js = [(d, j) for j in (*range(len(graphs)), *(vs if (pool, tuple(x)) not in base else ()))]
                if cur and len(cur) + len(js) > rows:
                    parts.append(cur)
                    cur = []
                cur += js
            parts += [cur] if cur else []
            got = {}
            for part in parts:
                acts = {d: self.donor_activations(xs[d][1], T) for d in dict.fromkeys(d for d, _ in part)}
                patch = {}
                for n in nat.names:
                    vals = torch.stack([acts[d][0][n] for d, _ in part]).to(dev)  # [rows, T, C]
                    keep = torch.stack([~own[j][n] for _, j in part])  # outside each row's side
                    for r, (d, _) in enumerate(part):
                        if acts[d][1]:
                            keep[r, :acts[d][1]] = False
                    patch[n] = (keep, vals)
                lq = E.graph_log_probs(E.plan([sides[j] for _, j in part], semantics="c"), u, patch=patch)
                got.update(zip(part, carry.kl(tx.ref, lq).tolist()))
                native.free(dev)
            for d, (_, x) in enumerate(xs):
                if (pool, tuple(x)) not in base:
                    base[(pool, tuple(x))] = (got[(d, vs[0])], got[(d, vs[1])])
            kv, k0 = [base[(pool, tuple(x))][0] for _, x in xs], [base[(pool, tuple(x))][1] for _, x in xs]
            for b in range(len(graphs)):
                kc = [got[(d, b)] for d in range(len(xs))]
                per = [_fid(z, c, v) for z, c, v in zip(k0, kc, kv)]
                each = [x for x in per if x is not None]
                out[b].update({f"s_patching_{pool}": _fid(float(np.mean(k0)), float(np.mean(kc)), float(np.mean(kv))) if kc else None,
                               f"s_patching_{pool}_worst": min(each) if each else None,
                               f"s_patching_{pool}_p90": float(np.percentile(each, 10)) if each else None,
                               f"kl_patching_{pool}": (float(np.mean(kc)), float(np.mean(kv)), float(np.mean(k0))) if kc else None,
                               f"donors_{pool}": len(kc),
                               f"draws_patching_{pool}": [[name, c, v, z, sd] for (name, _), c, v, z, sd in zip(xs, kc, kv, k0, per)]})
        return out

    def _donor_texts(self, tx: Text, split: str, near: int, far: list | None) -> dict:
        """donors()' texts: pool -> [(name, token ids)], near the split's changed-input texts named [position, token],
        far far's texts by name."""
        texts = {}
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
        return texts

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
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
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
        gradient either way, each switched on alone); every step, screen and reported number in float32 on every
        device (the picks the same on any device); the worst of them, at most m (no weaker than a smaller budget).
        A budget past the lever's parts (a circuit leaving out fewer than m VPD actives) switches every one on, and a
        circuit leaving out none scores its own run there (VPD's complete answer: 1). Per circuit {f"f1_flips_{lever}": {m: f1_adv},
        f"kl_flips_{lever}": {m: (KL_C, KL_VPD)}, f"mask_flips_{lever}": {m: the parts switched on, (site, position,
        subcomponent) or pool index}}. upto: only the budgets up to it are searched, every budget's starts still
        drawn (the same numbers as the full search at those budgets). screens: a caller's dict of measured shortlist
        gaps by part, shared by this call's graphs and kept across calls (a part measured once, on the first graph
        that needs it): for searches over nearby circuits on one text (a cleanup's candidates, connection subsets of
        one node set), whose shortlists overlap; the caller empties it when its circuits move on. None: every graph's
        shortlist measured on it alone. strengths: _adv_problem's (one stated graph at each circuit's connection
        strengths: edge_flips)."""
        nat, dev = self.nat, self.nat.dev
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]  # the claim fidelity scores: stranded nodes dropped (a no-op on its graphs)
        e0 = self._bounds(tx, "live")[0]
        ms = FLIPS[lever] if ms is None else ms
        gen = torch.Generator().manual_seed(int(tx.seed) % (2 ** 63))
        omitted = lever == "omitted"  # the patterns stay off: VPD's paired run is VPD's answer alone, run once
        pick, problems, batch = self._adv_problem(tx, graphs, pool, k, gen, batched=True, patterns=not omitted, strengths=strengths)
        search = "score"  # the sign steps and screens in float32 on every device (bf16 on CUDA before: the picks
        # differed by device, so the reward's flips family did too; every reported number was float32 already)
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
        named = {}  # each circuit's lever's parts by name, once per distinct graph
        flat = [named.setdefault(id(idx), _Parts(nat.names, [idx[n][0] for n in nat.names], [cols[n][idx[n][1]] for n in nat.names]))
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
        past = [(i, [m for m in ms if m > N[i] and (upto is None or m <= upto)]) for i in range(len(problems))]
        past = [(i, mm) for i, mm in past if mm]
        if past:  # a budget past the lever's parts: every one switched on (a circuit leaving nothing out: its own run)
            with torch.no_grad():
                for (i, mm), (kc, kv) in zip(past, batch(items([(i, torch.ones(1, N[i], device=dev)) for i, _ in past]), vpd=not omitted)):
                    kv = kv0.expand_as(kc) if kv is None else kv
                    for m in mm:
                        out[i][f"f1_flips_{lever}"][m] = _fid(e0, float(kc[0]), float(kv[0]))
                        out[i][f"kl_flips_{lever}"][m] = (float(kc[0]), float(kv[0]))
                        out[i][f"mask_flips_{lever}"][m] = [flat[i][j] for j in range(N[i])]
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
    def connection_effects(self, tx: Text, graph: native.Graph, rows: int = 48, edges: list | None = None) -> list[dict]:
        """Per stated connection of a graph (stranded nodes dropped, as fidelity scores it), writer A to reader B (None:
        the prediction), the change in target KL (bits) of its run in C's semantics against the graph as stated when A's
        contribution leaves B's read but stays in B's RMSNorm scale ("direct"), leaves the scale but stays in the read
        ("norm"), and leaves both ("full", the connection cut): engine.Engine's ew/ewn and ow/own. "normed": B reads
        through a norm (a query, key, value or MLP input, or the prediction); an attention or MLP output reads none, so
        its "norm" is 0 and "direct" its cut. An explanation's claim that a connection reads or rescales is tested by the
        two partial cuts (the lead's decision 10-12): they need not add up to the cut. edges: those connections alone
        ((writer, reader or None), in the graph's order; None: every one)."""
        nat, dev = self.nat, self.nat.dev
        g = _c_graph(graph, tx.targets, nat.C, tx.vpd)
        every = [(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out]
        want = None if edges is None else set(edges)
        ks = [k for k, e in enumerate(every) if want is None or e in want]
        if not ks:
            return []
        E = self._circuit_engine(tx.ids, tx.targets, [g])
        plan = E.plan(g, semantics="c")
        Ein, O = plan["E"], plan["O"]
        cases = [None] + [(k, d, n) for k in ks for d, n in ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0))]
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
        for j, k in enumerate(ks):
            w, r = every[k]
            kind = None if r is None else native._name_layer_kind(r[0])[1]
            out.append({"writer": w, "reader": r, "normed": r is None or kind in ("q_proj", "k_proj", "v_proj", "c_fc"),
                        **{name: kl[1 + 3 * j + i] - kl[0] for i, name in enumerate(("direct", "norm", "full"))}})
        return out

    def _edge_subsets(self, tx: Text, graph: native.Graph, ons: torch.Tensor) -> tuple:
        """(graph with its stranded nodes dropped, None), or with a subset of ons (edge_flips', in graph's order) that
        strands a node, or a graph that strands one itself (its plan's connections are fewer than ons' columns), (that,
        each subset's graph as stated with its stranded nodes dropped)."""
        base = _c_graph(graph, tx.targets, self.nat.C, tx.vpd)  # nodes no path of graph reaches the prediction from: stranded in every subset
        edges = [(w, r) for r, ws in graph.parents.items() for w in ws] + [(w, None) for w in graph.out]
        nodes = sorted(graph.node_set())
        subs = []
        for row in ons.tolist():
            kept = [e for e, k_ in zip(edges, row) if k_]
            parents = {}
            for w, r in kept:
                if r is not None:
                    parents.setdefault(r, []).append(w)
            h = native.Graph(nodes, parents, [w for w, r in kept if r is None])
            for name in ("tok_edges", "tok_out", "pairs"):  # the subset keeps graph's token edges and pairs (pp's runs)
                if hasattr(graph, name):
                    setattr(h, name, dict(getattr(graph, name)) if name == "pairs" else list(getattr(graph, name)))
            subs.append(strand(h, tx.targets, self.nat.C))
        return base, (subs if base.stranded or any(sub.stranded for sub in subs) else None)

    def changed_edges(self, tx: Text, graph: native.Graph, ons: torch.Tensor, split: str = "search", draws: int = 16,
                      mode: str = "c") -> list[dict]:
        """changed_inputs on connection subsets of one node set (edge_flips' graph and ons: a teacher's connection
        descent asking about its prefixes): the same numbers as changed_inputs on the subsets' graphs, from one plan of
        graph at 0/1 strengths; a call with a subset that strands a node runs changed_inputs on the subsets' graphs.
        mode "pp": track_pp's s_track_pp instead, the subsets in one engine.Engine.path_patched call (strengths)."""
        base, subs = self._edge_subsets(tx, graph, ons)
        if mode == "pp":
            return self.track_pp(tx, subs, split, draws) if subs is not None else \
                self.track_pp(tx, [base], split, draws, strengths=ons.to(self.nat.dev, torch.float32))
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
        graphs = [_c_graph(g, tx.targets, nat.C, tx.vpd) for g in graphs]  # stranded nodes dropped before every measure
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


def build_patterns(texts: list, out: Path | None = None, nat: native.Native | None = None, lengths: list[int] | None = None) -> dict:
    """The pool M of the paired adversary: every position of the texts, the set of subcomponents VPD's causal
    importance switches on there (CI > 0), one pattern each. texts: vpd4l text files or token id lists (another
    target's texts, nat its native); lengths: per text the positions kept, the rest padding the causal importance still
    reads (MIB's prompts padded as its fit's batches). Saved to out when given, as {"names", "sources": [(task,
    position)], "site": int16 index into names, "c": int32, "offsets": int64 [patterns + 1]}, a pattern's (site, c)
    pairs between consecutive offsets."""
    nat = nat or native.Native()
    sites, cols, counts, sources = [], [], [], []
    with torch.no_grad():
        for k, p in enumerate(texts):
            ids = native.text(p)[0] if isinstance(p, Path) else list(p)
            T = len(ids) if lengths is None else lengths[k]
            ci = nat.importance(ids)
            ts, ss, cs = [], [], []
            for i, n in enumerate(nat.names):
                t, c = (ci[n][:T] > 0).nonzero(as_tuple=True)
                ts.append(t.cpu())
                ss.append(torch.full((len(t),), i, dtype=torch.int16))
                cs.append(c.cpu().to(torch.int32))
            t, site, c = torch.cat(ts), torch.cat(ss), torch.cat(cs)
            order = torch.argsort(t * len(nat.names) + site.long(), stable=True)  # by position, then site
            sites.append(site[order])
            cols.append(c[order])
            counts.append(torch.bincount(t, minlength=T))
            sources += [(p.stem if isinstance(p, Path) else f"text{k}", i) for i in range(T)]
            native.free(nat.dev)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cat(counts).cumsum(0)])
    pool = {"names": list(nat.names), "sources": sources, "site": torch.cat(sites), "c": torch.cat(cols), "offsets": offsets}
    if out is not None:
        torch.save(pool, out)
    return pool


def pattern(pool: dict, k: int) -> dict:
    """Pattern k of a pool as matrix -> LongTensor of subcomponent indices."""
    a, b = int(pool["offsets"][k]), int(pool["offsets"][k + 1])
    site, c = pool["site"][a:b].long(), pool["c"][a:b].long()
    return {n: c[site == i] for i, n in enumerate(pool["names"]) if bool((site == i).any())}


def fidelity_report(split: str, n: int, sets: dict, pool: Path, out: Path, draws: int = DRAWS, vpd: Path | None = None,
                    offset: int = 0, stride: int = 1, fresh: Path | None = None, interchange: int = INTERCHANGE_DRAWS, curve=ADV_CURVE,
                    log=print, changed: int = CHANGED_DRAWS, only: tuple = (), mode: str = "full", draw_split: str | None = None,
                    start: bool = False, recency: bool = False) -> list[dict]:
    """C on the split's first n texts for answer sets (sets: name -> a directory of <task>.json circuits, run as stated)
    and VPD's first k at each circuit's size (complete graphs), with the evaluation-only families (fresh: the pattern
    lever over a pool no search drew from; interchange draws; changed texts (changed_inputs' eval split); N_real); one
    row per circuit appended to OUT (texts already there are skipped); only: some of ("changed", "interchange",
    "donors", "track_pp", "interchange_pp", "pattern", "nodes", "necessity", "remainders"), those families alone on the eval split (task, kind, nodes and their fields: families added to rows
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
        if start:  # the influential start set (start_set, chosen on the search split), scored as the sets are: kind "start"
            st = ge.start_set(tx, "search")
            graphs.append(st["graph"])
            kinds.append("start")
            # the recency baseline (the lead's 10-12): VPD's actives at the targets and the m positions just before them,
            # m the start's positions other than the targets, a complete node set
            m, t0 = len(set(st["positions"]) - set(targets)), min(targets)
            ps = set(targets) | set(range(max(0, t0 - m), t0))
            graphs.append(native.Graph(sorted((n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist() if t in ps), complete=True))
            kinds.append("recency")
        order = saved[p.stem]["order"] if p.stem in saved else nat._vpd_order(ids, targets)
        tx.vpd_order = [tuple(nd) for nd in order]
        for k in sorted({g.count() for g in graphs}):
            graphs.append(native.Graph(order[:k], complete=True))
            kinds.append(f"VPD top {k}")
            if recency:  # the recency baseline at the same size (reported beside VPD's top k)
                graphs.append(ge.recency_set(tx, k))
                kinds.append(f"recency {k}")
        if only:
            e = ge._bounds(tx, "live")[0]
            ch = ge.changed_inputs(tx, graphs, "eval", changed) if "changed" in only else [{} for _ in graphs]
            it = ge._interchange_family(tx, graphs, interchange, split="eval") if "interchange" in only else [{} for _ in graphs]
            dn = ge.donors(tx, graphs, far=ge.far_donors(tx, far)) if "donors" in only else [{} for _ in graphs]
            tp = ge.track_pp(tx, graphs, "eval", changed) if "track_pp" in only else [{} for _ in graphs]
            if "pattern" in only:  # the true-pattern family, merged into the same rows
                tp = [dict(a, **c) for a, c in zip(tp, ge.pattern_family(tx, [_c_graph(g, targets, nat.C, tx.vpd) for g in graphs]))]
            ip = ge.interchange_pp(tx, graphs, "eval", interchange) if "interchange_pp" in only else [{} for _ in graphs]
            nf = ge._node_family(tx, [_c_graph(g, targets, nat.C, tx.vpd) for g in graphs], draws,
                                 units=[sorted(strand(g, targets, nat.C).node_set()) for g in graphs]) if "nodes" in only else [{} for _ in graphs]
            ns = ge.necessity_spec(tx, graphs, "eval") if "necessity" in only else [{} for _ in graphs]
            if "necessity_bg" in only:  # N without the background kept (background=False), the earlier definition
                ns = [dict(a, **{f"{k}_nobg": v for k, v in b.items()}) for a, b in zip(ns, ge.necessity_spec(tx, graphs, "eval", background=False))]
            if "omitted" in only:
                ns = [dict(a, **b) for a, b in zip(ns, ge._omitted_family(tx, [_c_graph(g, targets, nat.C, tx.vpd) for g in graphs]))]
            if "remainders" in only:  # the point score with the remainders live, held at x and off
                ns = [dict(a, **b) for a, b in zip(ns, ge.point_remainders(tx, graphs))]
            scale = ge.size_scale(tx)
            with open(out, "a") as fh:
                for kind, g, c, i, d, a, b, nd, nn in zip(kinds, graphs, ch, it, dn, tp, ip, nf, ns):
                    row = {"task": p.stem, "kind": kind, "nodes": g.count(),
                           "size_bits": size_bits(g, nat.positions(targets), sum(nat.C.values()), targets, active=tx.vpd),
                           "size_scale": scale,  # (informational: VPD's nodes alone, no connections)
                           **{k: c[k] for k in ("s_changed", "s_track", "context_share", "prompts_changed", "prompts_track") if k in c}}
                    if i:
                        row["s_interchange"] = _fid(e, i["kl_interchange_c"], i["kl_interchange_vpd"])
                    row.update(d)
                    row.update({k: v for k, v in a.items() if k != "draws_track_pp"})
                    row.update({k: v for k, v in b.items() if k != "draws_interchange_pp"})
                    if nd:  # the node family against the model's own response
                        row.update({k: nd[k] for k in ("kl_nodes_c", "kl_nodes_vpd", "draws_nodes", "overstate_nodes") if k in nd})
                        row |= {"s_nodes": _pooled("weighted", nd.get("draws_nodes"), e), "s_nodes_raw": _pooled_raw(nd.get("draws_nodes"))}
                    row.update(nn)
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
        return "VPD top k" if r["kind"].startswith("VPD top") else "recency k" if r["kind"].startswith("recency ") else r["kind"]

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
             ("patching near", lambda r: r.get("s_patching_near")), ("patching near worst", lambda r: r.get("s_patching_near_worst")),
             ("patching near p90", lambda r: r.get("s_patching_near_p90")),
             ("patching far", lambda r: r.get("s_patching_far")), ("patching far p90", lambda r: r.get("s_patching_far_p90")),
             ("patching far worst", lambda r: r.get("s_patching_far_worst")),
             ("tracks changes", lambda r: r.get("s_track")), ("changed inputs", lambda r: r.get("s_changed")),
             ("context share", lambda r: r.get("context_share"))]
    nflips = sorted({int(m) for r in rows for m in r.get("n_real_flips", {}) if int(m) > 0})
    held += [(f"N_real flip{m}", lambda r, m=m: _curve_value(r.get("n_real_flips"), m)) for m in nflips]
    held += [(f"response err {f}", lambda r, f=f: r.get(f"dreads_{f}")) for f in ("nodes", "edges", "interchange")]
    held += [("leak (no edges)", lambda r: r.get("leak"))]
    stepcols = [(f"adv@{m}", lambda r, m=m: _curve_value(r.get("f1_adv_curve"), m)) for m in steps]
    sample = [("stranded nodes", lambda r: r.get("stranded")), ("F1", lambda r: r["s_point"]), ("F1 every connection", lambda r: r.get("s_wiring")),
              ("reliance on cuts", lambda r: r.get("reliance_on_cuts")),
              ("N", lambda r: r.get("n_spec")), ("size bits", lambda r: r.get("size_bits", r.get("size")))]
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
    f.add_argument("--interchange", type=int, default=INTERCHANGE_DRAWS)
    f.add_argument("--changed", type=int, default=CHANGED_DRAWS, help="changed texts per circuit (changed_inputs' eval split)")
    f.add_argument("--only", default="", help="changed,interchange,donors,track_pp,interchange_pp,pattern,nodes,necessity,necessity_bg,omitted,remainders: those families alone, eval split (rows to merge by task and kind)")
    f.add_argument("--curve", default=",".join(str(x) for x in ADV_CURVE), help="the step adversary's budgets, e.g. 0,4,20,40,80,160,320")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--draws", type=int, default=DRAWS)
    f.add_argument("--vpd", type=Path, help="VPD's answers and orders saved by teacher.py --save-vpd")
    f.add_argument("--offset", type=int, default=0)
    f.add_argument("--stride", type=int, default=1)
    f.add_argument("--mode", choices=("full", "reward"), default="full",
                   help="GoalEval.fidelity's mode: full, or the reward's (its family alone, rl_terms and c_rl: teacher rows for the reward)")
    f.add_argument("--draw-split", choices=("search", "eval"), help="the families' draws (default: the mode's, search for reward, eval for full)")
    f.add_argument("--recency", action="store_true", help="also the recency baseline at each set's sizes (recency_set)")
    f.add_argument("--start", action="store_true", help="also the text's influential start set (start_set, chosen on the search split), "
                   "kind \"start\": its families, per-draw records and size bits beside the sets' (phi's start, compact's bits_start), "
                   "and the recency baseline, kind \"recency\" (VPD's actives at the targets and as many positions just before them)")
    q = sub.add_parser("patterns", help="the adversary's pattern pool from a range of held-out texts")
    q.add_argument("--first", type=int, required=True, help="first text number, e.g. 7950 (training texts: 9100-9149 are sealed)")
    q.add_argument("--last", type=int, required=True, help="last text number, e.g. 7999")
    q.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "fidelity":
        fidelity_report(args.split, args.n, dict(x.split("=", 1) for x in args.set), args.pool, args.out, args.draws, args.vpd,
                        args.offset, args.stride, args.fresh, args.interchange, tuple(int(x) for x in args.curve.split(",")),
                        changed=args.changed, only=tuple(x for x in args.only.split(",") if x), mode=args.mode, draw_split=args.draw_split,
                        start=args.start, recency=args.recency)
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
