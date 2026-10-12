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
remainders W - V U either on ("live", computing on the run's own inputs) or off; both conventions are reported. Every
KL is the model's next-token distribution at the targets (summed over them) from the run's, in bits. VPD's own answer
is the set of subcomponents with causal importance above zero on the input (at positions that can reach the targets,
Native.reachable); the empty circuit and it bound every answer:
  recovered = (KL_empty - KL_C) / (KL_empty - KL_VPD), completeness = min(1, recovered / GOAL)

Reward fields, on the text:
  F1: KL_C, recovered and completeness, with KL_empty and KL_VPD (VPD's own faithfulness on the text, the ceiling),
     and recovered_log, the same in orders of magnitude of KL (recovered_log()).
  N, necessity: clip((KL_{VPD minus C} - KL_VPD) / (KL_empty - KL_VPD), 0, 1), "VPD minus C" VPD's own answer with the
     circuit's subcomponents off: how much of what VPD's answer computes needs the circuit.
  F3, edges: in the circuit's run (remainders off), each stated edge cut (path patching of that one connection: the
     reader reads every input but that writer's output; native.Native.run's edge strengths) against random pairs of
     the circuit's nodes that the model connects and the answer does not state, each cut's effect the KL of the
     circuit's run from the cut run; F3 is the probability that a stated cut moves the prediction more than an
     unstated one (AUC), and 0 when the answer states no edge or leaves no connected pair unstated (an answer earns
     edge credit only by saying which connections matter).
  size: the description length of the nodes and edges in bits (native.Graph.bits).
  E1 (the English reader's bits) and E2 (question-answering accuracy): hooks (GoalEval.hooks) that the RL owner fills.
Diagnostics (score(diagnostics=True)), not reward fields: F1 on the text's changed prompts (Native.changes) and F2 on
the held-out alternatives (carry.alternatives, ranks 3 to 5) at each position whose changed token moves the prediction
by at least 1% of the text's total (Native.sensitivity), the same circuit identities run on each, recovered pooled
over the prompts against VPD's own answer and the empty circuit on each; and interchange carry (primerl's rule b):
at a changed position p, every subcomponent at p, every one inactive on both the text and the changed prompt, the
circuit's and the remainders compute, everything else held at its value on the text (carry.Carry), carried = 1 - KL /
movement, beside B_p, the same with every subcomponent active on the text after p named, and completeness_p =
min(1, carried / (GOAL x B_p)); target positions are left out (trivially carried).
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
    return g, {"invalid_nodes": bad_nodes, "invalid_edges": bad_edges}


def to_json(g: native.Graph, explanation: str = "") -> dict:
    """A Graph (not complete) as an answer in JSON (circuit's format)."""
    tok = getattr(g, "tok_edges", [])
    nodes = sorted(g.nodes, key=lambda nd: (nd[1], nd[0], nd[2])) + [("tok", p, 0) for p in sorted({p for p, _ in tok})]
    at = {nd: i for i, nd in enumerate(nodes)}
    edges = ([[at[w], at[r]] for r, ws in g.parents.items() for w in ws] + [[at[w], "out"] for w in g.out]
             + [[at[("tok", p, 0)], at[r]] for p, r in tok])
    return {"nodes": [[n, c, t] for n, t, c in nodes], "edges": edges, "explanation": explanation}


def _pairs(g: native.Graph) -> list[tuple]:
    """A graph's stated edges, (writer, reader), reader None for the prediction, in a fixed order (a graph's nodes are
    a set, whose order changes from process to process)."""
    return sorted([(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out],
                  key=lambda e: (e[0], e[1] or ("", -1, -1)))


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


def necessity(empty: float, minus: float, vpd: float) -> float | None:
    """clip((KL_{VPD minus C} - KL_VPD) / (KL_empty - KL_VPD), 0, 1)."""
    return min(1.0, max(0.0, (minus - vpd) / (empty - vpd))) if empty - vpd > 0 else None


def completeness(r: float | None) -> float | None:
    return None if r is None else min(1.0, r / GOAL)


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
        """Per graph, per convention: F1 (KL_C, recovered, completeness) and N on the text, with the bounds."""
        rows = [{} for _ in graphs]
        sets = [self.masks(g, tx.T) for g in graphs]
        if tx.vpd is not None:
            sets += [{n: tx.vpd[n] & ~m[n] for n in self.nat.names} for m in sets]
        for conv in CONVENTIONS:
            e, v = self._bounds(tx, conv)
            kl = carry.kl(tx.ref, self.runs(tx.ids, tx.targets, sets, conv == "live")).tolist() if sets else []
            for b, row in enumerate(rows):
                c = kl[b]
                r = recovered(e, c, v) if v is not None else None
                row.update({f"kl_{conv}": c, f"kl_empty_{conv}": e, f"kl_vpd_{conv}": v, f"recovered_{conv}": r,
                            f"completeness_{conv}": completeness(r), f"recovered_log_{conv}": recovered_log(e, c, v) if v is not None else None})
                if v is not None:
                    m = kl[len(graphs) + b]
                    row.update({f"kl_vpd_minus_{conv}": m, f"necessity_{conv}": necessity(e, m, v)})
                else:
                    row.update({f"kl_vpd_minus_{conv}": None, f"necessity_{conv}": None})
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

        ctx = {"make": make, "ablation": zero, "ref": ref, "flow": self._flow(tx.ids, rec, states, masks), "rec": rec}
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
            stated = _pairs(g)
            nodes = g.node_set()
            par, outs = native.all_edges(nodes, tx.targets)
            have = set(stated)
            unstated = [e for e in _pairs(native.Graph(nodes, par, outs)) if e not in have]
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
                   "ref": tx.ref, "flow": self.flow(tx), "rec": tx.rec}
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
        an MLP output its MLP's pre-activation without the input's output (then the nonlinearity)."""
        import vpd_model

        nat, tg = self.nat, self.nat.target
        fl, rec = ctx["flow"], ctx["rec"]
        n = len(pairs)
        writes = [None] * n
        for wn in {w[0] for w, _ in pairs}:
            js = [j for j, (w, _) in enumerate(pairs) if w[0] == wn]
            t, c = nat._ints([[pairs[j][0][1] for j in js], [pairs[j][0][2] for j in js]])
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

    # ---- diagnostics

    def across(self, tx: Text, graphs: list[native.Graph], prompts: list) -> list[dict]:
        """Per graph, per convention, the same circuit identities run on other prompts of the text: recovered pooled
        over them, sum (KL_empty - KL_C) / sum (KL_empty - KL_VPD), each prompt's bounds from VPD's own answer and the
        empty circuit on that prompt, and the mean KL_C."""
        nat = self.nat
        sets = [self.masks(g, tx.T) for g in graphs]
        empty = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        acc = {conv: [[0.0, 0.0, 0.0] for _ in graphs] for conv in CONVENTIONS}
        for x in prompts:
            x = list(x)
            ref = nat.reference([x], tx.targets)
            v = self.vpd_answer(x, tx.targets)
            extra = [empty] + ([v] if v is not None else [])
            for conv in CONVENTIONS:
                kl = carry.kl(ref, self.runs(x, tx.targets, sets + extra, conv == "live")).tolist()
                e = kl[len(graphs)]
                vv = kl[len(graphs) + 1] if v is not None else None
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
        with diagnostics, F1 on the changed prompts ("changed_*"), F2 ("f2_*") and interchange carry ("carry_*")."""
        t0 = time.perf_counter()
        nat = self.nat
        rows = self.sufficiency(tx, graphs)
        f3 = self.edges(tx, graphs)
        total = sum(nat.C.values())
        for b, (g, row) in enumerate(zip(graphs, rows)):
            st, un = f3[b]["stated"], f3[b]["unstated"]
            row.update({"nodes": g.count(), "edges": None if g.complete else g.edges(), "tok_edges": len(getattr(g, "tok_edges", [])),
                        "bits": g.bits(nat.positions(tx.targets), total, tx.targets),
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
    return nat.from_ir(ir), {}, ir["graph"].get("explanation") or ""


COLUMNS = [("nodes", "nodes", "{:.0f}"), ("edges", "edges", "{:.0f}"), ("bits", "size bits", "{:.0f}"),
           ("kl_live", "KL live", "{:.3g}"), ("recovered_live", "recov live", "{:.3f}"), ("completeness_live", "compl live", "{:.3f}"),
           ("recovered_log_live", "log recov live", "{:.3f}"),
           ("necessity_live", "N live", "{:.3f}"), ("kl_off", "KL off", "{:.3g}"), ("recovered_off", "recov off", "{:.3f}"),
           ("necessity_off", "N off", "{:.3f}"), ("kl_vpd_live", "KL VPD live", "{:.3g}"), ("f3", "F3", "{:.3f}"),
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
    the highest recovered_live. Baselines at each valid answer's size -> one results dict per set."""
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
                best[s].append(max(ok, key=lambda r: r.get("recovered_live") if r.get("recovered_live") is not None else -math.inf)
                               if ok else {"task": p.stem, "valid": False})
            for k in ("empty", "vpd"):
                if k in by:
                    results[s]["rows"][k].append({**by[k], "task": p.stem})
            if first and first[0]["valid"] and f"vpd@{first[1].count()}" in by:
                results[s]["rows"]["vpd@matched"].append({**by[f"vpd@{first[1].count()}"], "task": p.stem})
        done = [f"{s} recovered {r['recovered_live']:.2f}/{r['recovered_off']:.2f}" for s in sets
                for r in [results[s]["rows"][s][-1]] if r.get("valid") and r.get("recovered_live") is not None and r.get("recovered_off") is not None]
        log(f"{p.stem}: {', '.join(done) or 'no valid answer'}" + (f", {rows[0]['seconds_text']:.1f} s" if rows else ""))
    for s in sets:
        if best[s]:
            results[s]["rows"][f"{s} best of N (upper bound)"] = best[s]
    return list(results.values())


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
    args = ap.parse_args()
    sets = dict(x.split("=", 1) for x in args.set)
    results = score_sets(sets, args.split, args.n, baselines=not args.no_baselines, diagnostics=args.diagnostics)
    for r in results:
        print(table(r), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
