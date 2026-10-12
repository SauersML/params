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
FLIP_ROWS = 160  # adversary_flips' rows per run: every circuit's restarts at every budget (8 x 5 x 4) together


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
    reader)). Nodes outside the model or the text, and edges the model does not have (mech.connects), are dropped and
    counted."""
    nodes, bad_nodes = [], 0
    for m, c, t in answer.get("nodes", []):
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
    parents, out, tok, bad_edges = {}, [], [], 0
    for up, down in answer.get("edges", []):
        try:
            w = nodes[up]
            r = None if down == "out" else nodes[down]
        except (IndexError, TypeError):
            bad_edges += 1
            continue
        if w is None or (down != "out" and r is None):
            bad_edges += 1
        elif w[0] == "tok":
            if r is not None and r[0] != "tok" and native._name_layer_kind(r[0])[1] in mech.RESID_READERS and r[1] == w[1]:
                tok.append((w[1], r))
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
    g = strand(g, targets, C)
    return g, {"invalid_nodes": bad_nodes, "invalid_edges": bad_edges, "stranded_nodes": len(g.stranded)}


def strand(g: native.Graph, targets: list[int], C: dict) -> native.Graph:
    """A stated graph without its stranded nodes: a node other than a query or key (those act through the attention
    pattern) with no stated connection to another node or to the prediction does nothing in the stated graph's run, yet
    would earn credit in the runs with every connection restored (wiring, N) by cancelling what the stated graph cuts
    (the lead's decision, 10-12). They are dropped before every family, with any embedding edge into them; their bits
    stay charged in size (stranded_bits, which size_bits adds). g.stranded lists them. A complete graph is unchanged."""
    if hasattr(g, "stranded"):  # already done (its record kept)
        return g
    if g.complete:
        g.stranded, g.stranded_bits = [], 0.0
        return g
    linked = set(g.out) | {r for r, ws in g.parents.items() if ws} | {w for ws in g.parents.values() for w in ws}
    gone = [nd for nd in sorted(g.node_set()) if nd not in linked
            and native._name_layer_kind(nd[0])[1] not in ("q_proj", "k_proj")]
    if not gone:
        g.stranded, g.stranded_bits = [], 0.0
        return g
    drop = set(gone)
    kept = native.Graph([nd for nd in g.nodes if nd not in drop], dict(g.parents), list(g.out))
    kept.tok_edges = [(p, r) for p, r in getattr(g, "tok_edges", []) if r not in drop]
    kept.tok_positions = list(getattr(g, "tok_positions", []))
    P, S = max(targets) + 1, sum(C.values())
    g.stranded_bits = 0.0
    kept.stranded, kept.stranded_bits = gone, 0.0
    kept.stranded_bits = size_bits(g, P, S, targets) - size_bits(kept, P, S, targets)
    return kept


def to_json(g: native.Graph, explanation: str = "") -> dict:
    """A Graph (not complete) as an answer in JSON (circuit's format)."""
    tok = getattr(g, "tok_edges", [])
    nodes = sorted(g.nodes, key=lambda nd: (nd[1], nd[0], nd[2])) + [("tok", p, 0) for p in sorted(set(getattr(g, "tok_positions", [])) | {p for p, _ in tok})]
    at = {nd: i for i, nd in enumerate(nodes)}
    edges = ([[at[w], at[r]] for r, ws in g.parents.items() for w in ws] + [[at[w], "out"] for w in g.out]
             + [[at[("tok", p, 0)], at[r]] for p, r in tok])
    return {"nodes": [[n, c, t] for n, t, c in nodes], "edges": edges, "explanation": explanation}


def _pairs(g: native.Graph) -> list[tuple]:
    """A graph's stated edges, (writer, reader), reader None for the prediction, in a fixed order (a graph's nodes are
    a set, whose order changes from process to process)."""
    return sorted([(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out],
                  key=lambda e: (e[0], e[1] or ("", -1, -1)))


def size_term(bits: float, scale: float) -> float:
    """The size term: an answer's description bits in orders of magnitude of the scale (size_scale, the bits of naming
    VPD's whole answer), log2(1 + bits) / log2(1 + scale), at most 1 (the lead's decision 10-12; components.size_term)."""
    return min(1.0, math.log2(1.0 + bits) / math.log2(1.0 + scale)) if scale and scale > 0 else 0.0


def size_bits(g: native.Graph, positions: int, subcomponents: int, targets: list[int]) -> float:
    """native.Graph.bits with the answer's embedding nodes: each a choice among the positions, and every edge's ends
    indexed among all the listed nodes (embedding nodes included), an embedding edge priced like any other."""
    extra = getattr(g, "stranded_bits", 0.0)  # stranded nodes (strand): dropped, their bits still charged
    pos = sorted(set(getattr(g, "tok_positions", [])) | {p for p, _ in getattr(g, "tok_edges", [])})
    if not pos:
        return g.bits(positions, subcomponents, targets) + extra
    n, k = g.count(), len(pos)
    inner, out = native.implied_edges(g.node_set(), targets) if g.complete else (sum(len(w) for w in g.parents.values()), len(g.out))
    return n * math.log2(positions * subcomponents) + k * math.log2(positions) + math.log2(n + k) * (2 * (inner + len(g.tok_edges)) + out) + extra


def _tok_pairs(g: native.Graph) -> tuple[list, list]:
    """(the answer's stated embedding edges, the unstated ones: a listed embedding node into a residual reader among
    the nodes at its position), as (("tok", position, 0), reader) pairs."""
    stated = [(("tok", p, 0), r) for p, r in getattr(g, "tok_edges", [])]
    have = set(stated)
    unstated = [(("tok", p, 0), r) for p in getattr(g, "tok_positions", []) for r in sorted(g.nodes)
                if r[1] == p and native._name_layer_kind(r[0])[1] in mech.RESID_READERS and (("tok", p, 0), r) not in have]
    return stated, unstated


def _fid(e0: float, kc: float, kv: float) -> float | None:
    """C's log scale: clip(ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD), 0, 1); None when VPD's own answer is no
    nearer the model than the empty circuit under the interventions."""
    if kv is None or kc is None or not kv < e0:
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
    """A stated graph without some nodes and every connection to or from them."""
    nodes = [nd for nd in g.nodes if nd not in drop]
    par = {r: [w for w in ws if w not in drop] for r, ws in g.parents.items() if r not in drop}
    return native.Graph(nodes, {r: ws for r, ws in par.items() if ws}, [w for w in g.out if w not in drop])


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
        self.hooks: dict = {"E1": None, "E2": None}
        self.probes: dict = {}  # (ids, targets, seed) -> the changed prompts and F2's prompts, as int32 arrays

    def text(self, ids: list[int], targets: list[int], seed: int | None = None, task: dict | None = None) -> Text:
        return Text(self, ids, targets, native.task_seed(str(ids[:8])) if seed is None else seed, task)

    def probe(self, ids: list[int], targets: list[int], seed: int) -> tuple[list, list]:
        """(the text's changed prompts (Native.changes), F2's [(position, held-out prompts)]), kept per text: the
        sensitivity probe behind both is most of their cost (16 s at T 395 on the Mac's GPU)."""
        nat, key = self.nat, (tuple(ids), tuple(targets), seed)
        if key not in self.probes:
            probe = {}
            prompts = nat.changes(ids, targets, n=self.changes, seed=seed, probe=probe)
            f2 = []
            if "moved" in probe:
                moved = probe["moved"]
                total = float(moved.sum())
                for j in (moved >= self.share_floor * total).nonzero().flatten().tolist() if total > 0 else []:
                    f2.append((j + 1, np.asarray(carry.alternatives(nat, ids, j + 1, ranks=self.ranks), dtype=np.int32)))
            self.probes[key] = (np.asarray(prompts, dtype=np.int32), [getattr(x, "weight", 1.0) for x in prompts], f2)
        arr, weights, f2 = self.probes[key]
        return [native.Changed(x.tolist(), w) for x, w in zip(arr, weights)], [(p, a.tolist()) for p, a in f2]

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
                  reads: list | None = None, norms: dict | None = None, freeze: str = "") -> torch.Tensor:
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
        answer or the whole model as masks, has none unstated: its nodes read each other in full). norms: receives each
        RMSNorm's scale 1 / rms per position ({("attn" | "mlp", layer) or "final": [B, T or targets, 1]}); with
        freeze "layers" (every norm inside the layers) or "all" (the final norm too), those norms use the scales norms
        holds instead (a run recorded first: the norm diagnostic)."""
        import vpd_model

        nat, tg = self.nat, self.nat.target
        T, H, hd, dev = len(ids), tg.n_head, tg.hd, self.nat.dev
        drops = {r: list(ws) for r, ws in (cuts or {}).items()}
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
            P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
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
                 interchange: int = 0, real: bool = False, reads: bool = False, mode: str = "full") -> list[dict]:
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
        parts switched on (adversary_flips, the omitted lever: f1_flips_omitted by m); c, the mean of s_point, s_nodes,
        s_edges and f1_flips_c at FLIP_TERM, with s_point_c for s_point (the reward's adversarial member, the lead's decision 10-12; the
        step-budget f1_adv is a reported column); flips_to_break, the smallest m at which f1_flips_omitted falls below
        half of s_point (None: not within FLIPS); flips_to_half, the smallest m at which it falls below 0.5, half of
        VPD's own answer under the same attack (0: below it unattacked), which a circuit weak to begin with cannot pass. reads: C's second observable, the circuit nodes' relative RMS read
        error between the model and the circuit per family (reads_*), and of their responses, each node's change from
        its own run without the intervention (dreads_*, _response_error; the static error mostly measures the missing
        background of a circuit's run); graph_ref,
        slower). Evaluation only, never in the reward or a teacher's objective:
        fresh, a pool none of the searches drew from: the pattern lever's flips over it (f1_flips_patterns_fresh);
        interchange > 0: that many interchange draws (_interchange_family, s_interchange); real: N_real (real_removal,
        n_real) and its adversary (necessity_flips, n_real_flips). leak: graph F1 of the circuit's nodes stating no
        edge (live remainders carrying their outputs to the prediction). mode="reward": the training family alone, what
        the RL reward takes every step: point, nodes, edges with reads (on), and the flips up to FLIP_TERM (their
        starts drawn as for every budget, so c is the full mode's); no step-budget adversary, larger budgets, fresh
        pool, interchange, N_real or leak, and no flips_to_break / flips_to_half."""
        if mode not in ("full", "reward"):
            raise ValueError("mode is full or reward")
        graphs = [strand(g, tx.targets, self.nat.C) for g in graphs]  # stranded nodes dropped before every family
        reward = mode == "reward"
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
        inter = self._interchange_family(tx, graphs, interchange, reads=reads) if interchange and not reward else none
        reals = self.real_removal(tx, graphs) if real and not reward else none
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
            fw = self._flips_wiring(tx, g, fl, e0) if (i in wired and fg) else {}
            row["f1_flips_wiring"] = fw
            fc = {m: (min(v, fw[m]) if (m in fw and v is not None and fw[m] is not None) else v) for m, v in fg.items()}
            row["f1_flips_c"] = fc
            terms = [row[k] for k in ("s_point_c", "s_nodes", "s_edges") if row.get(k) is not None]
            terms += [fc[FLIP_TERM]] if fc.get(FLIP_TERM) is not None else []
            row["c"] = sum(terms) / len(terms) if terms else None
            if not reward:  # (the reward's flips stop at FLIP_TERM)
                half = 0.5 * row["s_point_c"] if row["s_point_c"] is not None else None
                broke = [m for m in sorted(fc) if half is not None and fc[m] is not None and fc[m] < half]
                row["flips_to_break"] = broke[0] if broke else None  # None: not within the budgets (more than max(FLIPS))
                below = [m for m in sorted(fc) if fc[m] is not None and fc[m] < 0.5]  # an absolute bar: half of VPD's own
                row["flips_to_half"] = 0 if (row["s_point_c"] is not None and row["s_point_c"] < 0.5) else (below[0] if below else None)
            if fr:
                row["f1_flips_patterns_fresh"] = fr["f1_flips_patterns"]
            if it:
                row["s_interchange"] = _fid(e0, it["kl_interchange_c"], it["kl_interchange_vpd"])
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
    def _flips_wiring(self, tx: Text, g: native.Graph, fl: dict, e0: float) -> dict:
        """The flips' masks scored on a stated circuit's node set with every connection among its nodes restored (its
        masked run with the switched-on omissions), against VPD's paired KL: {m: f1}."""
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
            return {}
        kc = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, masks, True)).tolist()
        kv = {m: fl["kl_flips_omitted"][m][1] for m in ms}
        return {m: _fid(e0, k, kv[m]) for m, k in zip(ms, kc)}


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
                        "reads_nodes": _read_error(pairs) if reads else None, "_pairs_nodes": pairs})
            native.free(dev)
        return out

    @torch.no_grad()
    def _edge_family(self, tx: Text, graphs: list[native.Graph], draws: int, reads: bool = False) -> list[dict]:
        """Per circuit, the connection-cut family: draw j cuts a random subset of the stated connections into one
        reader (the prediction among the readers) in the model, in VPD's answer and in the circuit, all in C's
        semantics (the reader reads its input without those writers' outputs); the mean KLs from the model's. A
        circuit without connections: None. Every draw of a side runs in one batch on the engine (_sides)."""
        sides = self._sides(tx, graphs)
        out = []
        for g in graphs:
            nodes = g.node_set()
            par, outs = (native.all_edges(nodes, tx.targets) if g.complete else (g.parents, g.out))
            readers = [r for r in sorted(par) if par[r]] + ([None] if outs else [])
            if not readers:
                out.append({"kl_edges_c": None, "kl_edges_vpd": None, "reads_edges": None})
                continue
            rng = self._rng(tx, g, "edges")
            cuts = []
            for j in range(draws):
                r = readers[int(rng.integers(len(readers)))]
                ws = list(par[r]) if r is not None else list(outs)
                cuts.append({r: [w for w in ws if rng.random() < (j + 0.5) / draws] or [ws[int(rng.integers(len(ws)))]]})
            got = sides(g, cuts=cuts, reads=reads)
            lm, lv, lc = got[:3]
            row = {"kl_edges_c": float(carry.kl(lm, lc).mean()), "kl_edges_vpd": float(carry.kl(lm, lv).mean()), "reads_edges": None}
            if reads:  # the circuit nodes' reads under each draw, model and circuit
                rm, rc = got[3], got[4]
                pairs = {nd: list(zip(rm[nd].tolist(), rc[nd].tolist())) for nd in rm}
                row["reads_edges"], row["_pairs_edges"] = _read_error(pairs), pairs
            out.append(row)
            native.free(self.nat.dev)
        return out

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
    def _interchange_family(self, tx: Text, graphs: list[native.Graph], draws: int = 16, ranks=(0, 1, 2),
                            reads: bool = False) -> list[dict]:
        """Evaluation only (never in the reward or a teacher's objective): per circuit, interchange interventions with
        the identity alignment. Draw j takes a position p among the circuit's (after 0) and one of the model's ranks
        likeliest other tokens there (carry.alternatives), and sets the circuit's subcomponents at p to their
        activations in the real model on that changed prompt, in the model (everything else computing on the text), in
        the circuit (its later nodes reading the patched ones through stated connections) and in VPD's answer, all in
        C's semantics (graph_ref); the mean KLs from the model's, kl_interchange_c and kl_interchange_vpd."""
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
                out.append({"kl_interchange_c": None, "kl_interchange_vpd": None, "reads_interchange": None})
                continue
            rng = self._rng(tx, g, "interchange")
            patch = []
            for j in range(draws):
                p = positions[int(rng.integers(len(positions)))]
                alts = carry.alternatives(nat, tx.ids, p, ranks)
                z = alts[int(rng.integers(len(alts)))][p]
                rec = source_record(p, z)
                patch.append({nd: rec[nd[0]][0, nd[1], nd[2]] for nd in nodes if nd[1] == p})
            got = sides(g, patch=patch, reads=reads)
            lm, lv, lc = got[:3]
            native.free(dev)
            row = {"kl_interchange_c": float(carry.kl(lm, lc).mean()), "kl_interchange_vpd": float(carry.kl(lm, lv).mean()),
                   "reads_interchange": None}
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
        adversary_flips on the K graphs of those connections, from one plan of graph at 0/1 strengths (kept on the
        text across calls) and the node set's layout made once."""
        return self.adversary_flips(tx, [graph] * len(ons), pool, "omitted", ms=(m,), screens=screens,
                                    strengths=ons.to(self.nat.dev, torch.float32), **kw)

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
        graphs = [strand(g, tx.targets, nat.C) for g in graphs]  # stranded nodes dropped before every measure
        rows = self.sufficiency(tx, graphs)
        f3 = self.edges(tx, graphs)
        total = sum(nat.C.values())
        scale = self.size_scale(tx)
        for b, (g, row) in enumerate(zip(graphs, rows)):
            st, un = f3[b]["stated"], f3[b]["unstated"]
            bits = size_bits(g, nat.positions(tx.targets), total, tx.targets)
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
                    log=print) -> list[dict]:
    """C on the split's first n texts for answer sets (sets: name -> a directory of <task>.json circuits, run as stated)
    and VPD's first k at each circuit's size (complete graphs), with the evaluation-only families (fresh: the pattern
    lever over a pool no search drew from; interchange draws; N_real); one row per circuit appended to OUT (texts
    already there are skipped). curve: the step adversary's budgets reported. vpd: VPD's answers and orders saved by
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
        fid = ge.fidelity(tx, graphs, M, draws=draws, curve=curve, fresh=F, interchange=interchange, real=True, reads=True)
        e, v = ge._bounds(tx, "live")
        ceiling = ge.real_removal(tx, [native.Graph(masks=tx.vpd)])[0]["n_real_live"]  # VPD's whole answer removed
        cancel = ge.cancellation(tx)
        scale = ge.size_scale(tx)
        with open(out, "a") as fh:
            for kind, g, f in zip(kinds, graphs, fid):
                bits = size_bits(g, nat.positions(targets), sum(nat.C.values()), targets)
                row = {"task": p.stem, "local": local.get(p.stem), "kind": kind, "nodes": g.count(),
                       "positions": len({t for _, t, _ in g.node_set()}), "stranded": len(getattr(g, "stranded", [])),
                       "edges": None if g.complete else g.edges(), "size": size_term(bits, scale), "kl_empty": e, "kl_vpd": v,
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
    kinds = sorted({kind(r) for r in rows}, key=lambda k: (k == "VPD top k", k))
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
             ("act/inact cos", lambda r: r.get("cancel_cosine")), ("C", lambda r: r.get("c"))]
    nflips = sorted({int(m) for r in rows for m in r.get("n_real_flips", {}) if int(m) > 0})
    held += [(f"N_real flip{m}", lambda r, m=m: _curve_value(r.get("n_real_flips"), m)) for m in nflips]
    held += [(f"response err {f}", lambda r, f=f: r.get(f"dreads_{f}")) for f in ("nodes", "edges", "interchange")]
    held += [("leak (no edges)", lambda r: r.get("leak"))]
    stepcols = [(f"adv@{m}", lambda r, m=m: _curve_value(r.get("f1_adv_curve"), m)) for m in steps]
    sample = [("stranded nodes", lambda r: r.get("stranded")), ("F1", lambda r: r["s_point"]), ("F1 every connection", lambda r: r.get("s_wiring")),
              ("reliance on cuts", lambda r: r.get("reliance_on_cuts")),
              ("N", lambda r: r["n_point"]), ("size", lambda r: r.get("size"))]
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
    f.add_argument("--curve", default=",".join(str(x) for x in ADV_CURVE), help="the step adversary's budgets, e.g. 0,4,20,40,80,160,320")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--draws", type=int, default=DRAWS)
    f.add_argument("--vpd", type=Path, help="VPD's answers and orders saved by teacher.py --save-vpd")
    f.add_argument("--offset", type=int, default=0)
    f.add_argument("--stride", type=int, default=1)
    q = sub.add_parser("patterns", help="the adversary's pattern pool from a range of held-out texts")
    q.add_argument("--first", type=int, required=True, help="first text number, e.g. 9100")
    q.add_argument("--last", type=int, required=True, help="last text number, e.g. 9149")
    q.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "fidelity":
        fidelity_report(args.split, args.n, dict(x.split("=", 1) for x in args.set), args.pool, args.out, args.draws, args.vpd,
                        args.offset, args.stride, args.fresh, args.interchange, tuple(int(x) for x in args.curve.split(",")))
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
