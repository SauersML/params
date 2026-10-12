"""The goal scorecard (#2951 graph oracle): how well an answer meets the goal, one implementation for the RL reward and
every evaluation.

The goal: the oracle states the whole-model circuit of vpd4l's prediction on a text in VPD subcomponents, causally
faithful to the model's weights, with correct English and question answering. An answer is a circuit on one text:
nodes (weight matrix, subcomponent index, position) and explicit edges (upstream node -> downstream node, or -> the
prediction), each a connection the model has (mech.connects: the residual stream at one position, a value into its
layer's attention output at that or a later position, an MLP input into its MLP's output); a stated pair the model
does not connect is dropped and counted. Queries and keys are nodes without edges out: they enter their layer's
attention pattern. Every KL is the model's next-token distribution at the targets (summed over them) from the run's, in
bits; native.Native.run runs every intervention.

  F1 sufficiency: the circuit runs alone (native's run-alone graph: its nodes at mask 1 reading their stated parents,
     the token embedding, every other subcomponent at 0), on the text and on the text's changed prompts
     (Native.changes, the verifier's). VPD's remainders W - V U are held at their values in the model on the text
     (the decomposition's error is not the answer's), and, reported separately, ablated (native's verifier setting).
  F2 interchange: at each input position whose changed token moves the prediction by at least 1% of the text's total
     (Native.sensitivity, summed over positions), the prompts changing that token to the model's held-out alternatives
     there (carry.alternatives at ranks 3 to 5; the carry search uses 0 to 2) run with the circuit's subcomponents
     computing and every other activation held at its value on the text (carry.Carry; remainders compute). Its share
     carried: 1 - KL / movement, movement the KL with nothing held (the model on the changed prompt from the model on
     the text), positions pooled by movement. The bypass share is the same with no subcomponent computing: what VPD's
     remainders carry and, at a changed target position, the changed token's own embedding, which the prediction
     there reads directly (on held-out text9002 every position's bypass carries nothing except the target's, where it
     moves the prediction 3% further than the change itself).
  F3 edges: each stated edge cut in the whole model on the text (path patching of that one connection: the reader
     recomputed without the writer's output and everything else as in the model, then the model run on from the
     reader's new activation), against random pairs of the circuit's nodes that the model connects and the answer does
     not state, cut the same way: the KL of each cut, their means, and the probability that a stated edge's cut moves
     the prediction more than an unstated pair's (AUC).
  size: the description length of the nodes and edges in bits (native.Graph.bits).
  E1 (the English reader's bits) and E2 (question-answering accuracy): hooks (GoalEval.hooks) that the RL owner fills.

Baselines, scored by the same code: the empty circuit, VPD's own answer at the answer's size (its causal-importance
ranking, native.Native._vpd_order, complete), and every active subcomponent (causal importance above zero) in the
prediction's backward cone (Native.reachable), complete.

  goal_eval.py score --set NAME=DIR [--set NAME=DIR ...] --split heldout --n N [--out OUT.json]
DIR holds one answer per text, <task>.json (Circuit's JSON) or <task>.py (a traced graph program, mech).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import mech  # noqa: E402
import native  # noqa: E402

HELD_OUT_RANKS = (3, 4, 5)  # carry.alternatives' ranks for F2; the carry search and its training answers use 0 to 2
SHARE_FLOOR = 0.01  # F2's positions: those moving the prediction by at least this share of the text's total
RANDOM_PAIRS = 32  # F3's unstated pairs per answer
MAX_EDGES = 64  # F3's stated edges per answer (a uniform draw when an answer states more)
UNIT_LIMIT = 20_000  # F2 computes at most this many subcomponents through carry's unit path; larger sets take masks
CHUNK = 16  # answers per batch (F1)
CUT_CHUNK = 32  # cuts per batch (F3)


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
    the prediction; a matrix is native's name or "L.kind" (kind q, k, v, o, fc, down). Nodes outside the model or the
    text, and edges the model does not have (mech.connects), are dropped and counted."""
    nodes, bad_nodes = [], 0
    for m, c, t in answer.get("nodes", []):
        try:
            n = _site(m)
        except (ValueError, KeyError):
            n = None
        if n in C and 0 <= int(c) < C[n] and 0 <= int(t) <= max(targets):
            nodes.append((n, int(t), int(c)))
        else:
            bad_nodes += 1
            nodes.append(None)
    parents, out, bad_edges = {}, [], 0
    for up, down in answer.get("edges", []):
        try:
            w = nodes[up]
            r = None if down == "out" else nodes[down]
        except (IndexError, TypeError):
            bad_edges += 1
            continue
        if w is None or (down != "out" and r is None) or not native.connects(w, r, targets):
            bad_edges += 1
        elif r is None:
            out.append(w)
        else:
            parents.setdefault(r, []).append(w)
    g = native.Graph([nd for nd in nodes if nd is not None], {r: list(dict.fromkeys(ws)) for r, ws in parents.items()}, list(dict.fromkeys(out)))
    return g, {"invalid_nodes": bad_nodes, "invalid_edges": bad_edges}


def to_json(g: native.Graph, explanation: str = "") -> dict:
    """A Graph (not complete) as an answer in JSON (circuit's format)."""
    nodes = sorted(g.nodes, key=lambda nd: (nd[1], nd[0], nd[2]))
    at = {nd: i for i, nd in enumerate(nodes)}
    edges = [[at[w], at[r]] for r, ws in g.parents.items() for w in ws] + [[at[w], "out"] for w in g.out]
    return {"nodes": [[n, c, t] for n, t, c in nodes], "edges": edges, "explanation": explanation}


def _pairs(g: native.Graph) -> list[tuple]:
    """A graph's stated edges, (writer, reader), reader None for the prediction, in a fixed order (a graph's nodes are
    a set, whose order changes from process to process)."""
    return sorted([(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out],
                  key=lambda e: (e[0], e[1] or ("", -1, -1)))


class _Held(TorchFunctionMode):
    """Every remainder product (W - V U applied to its input) at its value in the model on the text (values: carry's
    capture), at the positions a run computes: the last L when Native.run reuses an earlier run's states."""

    def __init__(self, nat: native.Native, values: dict):
        self.by_id = {id(nat.target.site(n).delta_T): values[n] for n in nat.names}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in ("matmul", "__matmul__") and len(args) > 1 and id(args[1]) in self.by_id:
            v = self.by_id[id(args[1])]
            return v[:, v.shape[1] - args[0].shape[1]:].expand(args[0].shape[0], -1, -1)
        return func(*args, **kwargs)


def _from(ids: list[int], p: int) -> tuple | None:
    """A base for Native.run(reuse=) on ids itself whose states hold up to position p: ids with a token that differs
    at p (the run recomputes from there; None from position 0)."""
    if p <= 0:
        return None
    base = list(ids)
    base[p] = -1
    return base


# ---- one text


class Text:
    """What every answer on one text shares: the model's run, the changed prompts and their references, the
    remainders' values, F2's positions and held-out prompts."""

    def __init__(self, ge: "GoalEval", ids: list[int], targets: list[int], seed: int, task: dict | None = None):
        nat = ge.nat
        t0 = time.perf_counter()
        self.ids, self.targets, self.T, self.seed, self.task = list(ids), list(targets), len(ids), seed, task
        self.rec, self.states = nat._model_record(self.ids)
        self.ref = nat.reference([self.ids], self.targets)
        cap = carry._Remainders(nat)
        with torch.no_grad(), cap:
            nat.run(self.ids, self.targets, nat._full_plan(self.T, 1), *nat._one())
        if cap.seen != set(nat.names):
            raise RuntimeError("did not capture every remainder product")
        self.rem = cap.values
        self.prompts, self.f2 = ge.probe(self.ids, self.targets, seed)  # f2: (position, held-out prompts)
        self.prompt_refs = [nat.reference([x], self.targets) for x in self.prompts]
        self.ci = nat.importance(self.ids) if nat.has_importance else None
        self.cut_cache: dict = {}  # (writer, reader) -> F3's KL of its cut: answers on a text share edges
        self._flow = None
        self.seconds = time.perf_counter() - t0


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

    def probe(self, ids: list[int], targets: list[int], seed: int) -> tuple[list, list]:
        """(the text's changed prompts (Native.changes), F2's [(position, held-out prompts)]), kept per text: the
        sensitivity probe behind both is most of a text's preparation (16 s at T 395 on the Mac's GPU)."""
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

    def text(self, ids: list[int], targets: list[int], seed: int | None = None, task: dict | None = None) -> Text:
        return Text(self, ids, targets, native.task_seed(str(ids[:8])) if seed is None else seed, task)

    # ---- F1

    @torch.no_grad()
    def sufficiency(self, tx: Text, graphs: list[native.Graph]) -> dict:
        """Per graph, F1 in bits: on the text and averaged over the changed prompts, remainders held at their values
        on the text ("held") and ablated ("ablated"). A changed prompt reuses the graph's own run on the text before
        its changed position (Native.run's reuse)."""
        nat = self.nat
        zu, zr = nat._zero()
        one_r = nat._one()[1]
        out = {k: torch.zeros(len(graphs), device=nat.dev) for k in ("text_held", "changed_held", "text_ablated", "changed_ablated")}
        weights = [getattr(x, "weight", 1.0) for x in tx.prompts]
        for s0 in range(0, len(graphs), self.chunk):
            part = graphs[s0:s0 + self.chunk]
            B = len(part)
            plan = nat._plan(part, tx.T)
            for key, ur, mode in (("held", one_r, _Held(nat, tx.rem)), ("ablated", zr, None)):
                states = {}
                with mode or contextlib.nullcontext():
                    out[f"text_{key}"][s0:s0 + B] = carry.kl(tx.ref, nat.run(tx.ids, tx.targets, plan, zu, ur, fill=states))
                    for x, logp, w in zip(tx.prompts, tx.prompt_refs, weights):
                        lq = nat.run(x, tx.targets, plan, zu, ur, reuse=(states, tx.ids) if states else None)
                        out[f"changed_{key}"][s0:s0 + B] += w * carry.kl(logp, lq) / len(tx.prompts)
                del states
            native.free(nat.dev)
        return {k: v.tolist() for k, v in out.items()}

    # ---- F2

    def _live(self, g: native.Graph, lo: int, hi: int) -> list:
        """A graph's nodes at positions lo..hi (a change at lo reaches no earlier position; none later reaches the
        targets)."""
        if g.masks is not None:
            return [(n, lo + t, c) for n, m in g.masks.items() for t, c in m[lo:hi + 1].nonzero().tolist()]
        return [nd for nd in g.nodes if lo <= nd[1] <= hi]

    @torch.no_grad()
    def interchange(self, tx: Text, graphs: list[native.Graph]) -> dict:
        """Per graph, F2: its movement-pooled share carried and per position (position, movement bits, KL bits), and
        the bypass share (no subcomponent computing: the remainders and a changed target's embedding). Sets of at most UNIT_LIMIT subcomponents run through carry's
        subcomponent units (positions from the change on), larger ones as dense masks (carry's node masks); both
        hold every other activation at its value on the text."""
        nat, hi = self.nat, max(tx.targets)
        rows = [[] for _ in graphs]
        empty_kl, moved = 0.0, 0.0
        for p, prompts in tx.f2:
            live = [self._live(g, p, hi) for g in graphs]
            small = [b for b, nodes in enumerate(live) if len(nodes) <= UNIT_LIMIT]
            big = [b for b in range(len(graphs)) if b not in small]
            union = sorted({nd for b in small for nd in live[b]})
            kls = [None] * len(graphs)
            if union:
                cand = {}
                for n, t, c in union:
                    cand.setdefault((n, t), []).append(c)
                space = carry.UnitSpace(nat, tx.T, sorted({t for _, t in cand}), "subcomponent", cand)
                study = carry.Carry(nat, tx.ids, tx.targets, prompts, space=space)
                a = torch.zeros(len(small) + 1, space.size, dtype=torch.bool, device=nat.dev)
                for j, b in enumerate(small):
                    if live[b]:
                        a[j, torch.tensor([space.index[carry.Subcomponent(*nd)] for nd in live[b]], device=nat.dev)] = True
                got = study.measure(a).tolist()
                for j, b in enumerate(small):
                    kls[b] = got[j]
                e = got[-1]
            else:
                study = carry.Carry(nat, tx.ids, tx.targets, prompts)
                e = float(study.measure([set()])[0])
                for b in small:
                    kls[b] = e
            if big:
                dense = carry.Carry(nat, tx.ids, tx.targets, prompts)
                got = dense.measure([carry.masks(nat, tx.T, set(live[b])) for b in big]).tolist()
                for j, b in enumerate(big):
                    kls[b] = got[j]
            m = carry._movement(study)
            moved += m
            empty_kl += e
            for b in range(len(graphs)):
                rows[b].append((p, m, kls[b]))
            native.free(nat.dev)
        share = [None if moved <= 0 else 1 - sum(r[2] for r in rs) / moved for rs in rows]
        return {"share": share, "positions": rows, "bypass_share": None if moved <= 0 else 1 - empty_kl / moved,
                "movement_bits": moved}

    # ---- F3

    @torch.no_grad()
    def cuts(self, tx: Text, pairs: list[tuple]) -> list[float]:
        """The KL in bits of the model on the text from the model with one connection cut, per (writer, reader) pair
        (reader None: the prediction). The reader's new activation is its input in the model without the writer's
        output alone (every other output and every remainder as in the model; _cut_activations); the model then runs
        with that one activation replaced (path patching of the connection). Into the prediction: the final residual
        stream without the writer's output."""
        nat, T = self.nat, tx.T
        todo = [e for e in dict.fromkeys(pairs) if e not in tx.cut_cache]
        if todo:
            for e, k in zip(todo, self._cuts(tx, todo)):
                tx.cut_cache[e] = k
        return [tx.cut_cache[e] for e in pairs]

    def _cuts(self, tx: Text, pairs: list[tuple]) -> list[float]:
        """cuts() of distinct pairs. The runs with a reader's new activation are batched by the reader's position and
        reuse the model's states before the batch's first reader (nothing earlier changes); a cut into the prediction
        needs no run (flow's final residual stream)."""
        nat, T = self.nat, tx.T
        res = [0.0] * len(pairs)
        inner = sorted((j for j, (_, r) in enumerate(pairs) if r is not None), key=lambda j: (pairs[j][1][1], pairs[j][0][1]))
        final = sorted((j for j, (_, r) in enumerate(pairs) if r is None), key=lambda j: pairs[j][0][1])
        one_u, one_r = nat._one()

        new = self._cut_activations(tx, [pairs[j] for j in inner]) if inner else None
        at = {j: i for i, j in enumerate(inner)}
        B = min(self.cut_chunk, len(inner))  # every batch this size (the last padded with its own last cut): the
        for s0 in range(0, len(inner), B):  # model's states for reuse are computed once per text and batch size
            js = inner[s0:s0 + B]
            js = js + [js[-1]] * (B - len(js))
            by = {}
            for b, j in enumerate(js):
                r = pairs[j][1]
                by.setdefault(r[0], []).append((b, r[1], r[2], at[j]))
            patch2 = {}
            for n, lst in by.items():
                bi, ti, ci, ki = nat._ints([[e[k] for e in lst] for k in range(4)])
                m = torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev)
                m[bi, ti, ci] = True
                v = tx.rec[n].expand(B, -1, -1).clone()
                v[bi, ti, ci] = new[ki]
                patch2[n] = (m, v)
            base = _from(tx.ids, min(pairs[j][1][1] for j in js))
            lq = nat.run(tx.ids, tx.targets, nat._full_plan(T, B), one_u, one_r, patch=patch2,
                         reuse=None if base is None else (nat._model_states(tx.ids, B), base))
            for b, k in enumerate(carry.kl(tx.ref, lq).tolist()):
                res[js[b]] = k
            native.free(nat.dev)
        if final:
            import vpd_model

            tg, fl = nat.target, self.flow(tx)
            t, c = nat._ints([[pairs[j][0][1] for j in final], [pairs[j][0][2] for j in final]])
            x = fl["final"][tx.targets].expand(len(final), -1, -1).clone()
            row = torch.tensor([tx.targets.index(p) for p in t.tolist()], device=nat.dev)
            for wn in {pairs[j][0][0] for j in final}:
                sel = torch.tensor([b for b, j in enumerate(final) if pairs[j][0][0] == wn], device=nat.dev)
                x[sel, row[sel]] -= tx.rec[wn][0, t[sel], c[sel]][:, None] * tg.site(wn).U[c[sel]]
            lq = torch.log_softmax((vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T).float(), -1)
            for b, k in enumerate(carry.kl(tx.ref, lq).tolist()):
                res[final[b]] = k
        return res

    def flow(self, tx: Text) -> dict:
        """The model's run on the text as the cuts read it, kept on tx: per layer the residual stream before each norm
        ("attn", "mlp" [T, d]; "final", after the last layer), the attention pattern ("P" [H, T, T]) and output before its output matrix ("att"
        [T, d]), and the MLP's pre-activation ("pre" [T, d_mlp]); built from Native's own record of the run (every
        activation, the queries, keys and values, the remainders' values) the way Native.run adds them up."""
        if tx._flow is None:
            import vpd_model

            nat, tg, T = self.nat, self.nat.target, tx.T
            H, hd = tg.n_head, tg.hd
            x = tg.wte[torch.tensor(tx.ids, device=nat.dev)][None]
            causal = torch.ones(T, T, dtype=torch.bool, device=nat.dev).tril()
            fl = {"attn": [], "mlp": [], "P": [], "att": [], "pre": []}
            for i in range(tg.n_layer):
                nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
                heads = lambda y: y.view(1, T, H, hd).transpose(1, 2)  # noqa: E731
                q, k = tg._rope(heads(tx.states[("qv", i)]), T), tg._rope(heads(tx.states[("kv", i)]), T)
                P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
                fl["attn"].append(x[0])
                fl["P"].append(P[0])
                fl["att"].append((P @ heads(tx.states[("vs", i)])).transpose(1, 2).reshape(T, -1))
                x = (x + tx.rec[nm["o_proj"]] @ tg.site(nm["o_proj"]).U) + tx.rem[nm["o_proj"]]
                fl["mlp"].append(x[0])
                fl["pre"].append((tx.rec[nm["c_fc"]] @ tg.site(nm["c_fc"]).U + tx.rem[nm["c_fc"]])[0])
                x = (x + tx.rec[nm["down_proj"]] @ tg.site(nm["down_proj"]).U) + tx.rem[nm["down_proj"]]
            fl["final"] = x[0]
            tx._flow = fl
        return tx._flow

    def _cut_activations(self, tx: Text, pairs: list[tuple]) -> torch.Tensor:
        """Each pair's reader activation with that one connection cut [len(pairs)]: a query, key, value or MLP input
        reads the residual stream at its position without the writer's output (then its norm); an attention output
        reads its layer's attention output without the writer value's share, the pattern's weight from the reader's
        position to the writer's in each head times the value's output; an MLP output its MLP's pre-activation without
        the input's output (then the nonlinearity)."""
        import vpd_model

        nat, tg = self.nat, self.nat.target
        fl = self.flow(tx)
        n = len(pairs)
        writes = [None] * n
        for wn in {w[0] for w, _ in pairs}:
            js = [j for j, (w, _) in enumerate(pairs) if w[0] == wn]
            t, c = nat._ints([[pairs[j][0][1] for j in js], [pairs[j][0][2] for j in js]])
            wv = tx.rec[wn][0, t, c][:, None] * tg.site(wn).U[c]
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

    def edges(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """Per graph, F3: the cuts of its stated edges (at most max_edges, a uniform draw beyond) and of random
        unstated pairs of its nodes that the model connects (random_pairs); a complete graph states every connection
        among its nodes, so it has no unstated pair (its stated edges are drawn from them). The draws depend on the
        text's seed and the answer's own edges, not on the other answers scored with it."""
        plans, flat = [], []
        for g in graphs:
            nodes = g.node_set()
            if g.complete:
                if not nodes or native.count_edges(nodes, tx.targets) > 1_000_000:
                    plans.append(None)
                    continue
                par, out = native.all_edges(nodes, tx.targets)
                stated, unstated = _pairs(native.Graph(nodes, par, out)), []
            else:
                stated = _pairs(g)
                par, out = native.all_edges(nodes, tx.targets)
                have = set(stated)
                unstated = [e for e in _pairs(native.Graph(nodes, par, out)) if e not in have]
            rng = np.random.default_rng([tx.seed, int(hashlib.sha256(repr(stated).encode()).hexdigest()[:15], 16)])
            if len(stated) > self.max_edges:
                stated = [stated[i] for i in sorted(rng.choice(len(stated), self.max_edges, replace=False))]
            if len(unstated) > self.random_pairs:
                unstated = [unstated[i] for i in sorted(rng.choice(len(unstated), self.random_pairs, replace=False))]
            plans.append((len(flat), len(stated), len(unstated)))
            flat += stated + unstated
        effects = self.cuts(tx, flat) if flat else []
        out = []
        for p in plans:
            if p is None:
                out.append({"stated": None, "unstated": None, "auc": None})
                continue
            o, ns, nu = p
            st, un = effects[o:o + ns], effects[o + ns:o + ns + nu]
            out.append({"stated": st, "unstated": un, "auc": _auc(st, un)})
        return out

    # ---- baselines

    def baselines(self, tx: Text, sizes: list[int]) -> dict:
        """name -> Graph: "empty"; "vpd@k" for each k in sizes, VPD's first k subcomponents in its order (causal
        importance x how much each writes, the predicted position first), complete; "cone", every subcomponent with
        causal importance above zero that can reach the targets (Native.reachable), complete. Without VPD's
        causal-importance network (a pod) only the empty one."""
        nat = self.nat
        out = {"empty": native.Graph()}
        if tx.ci is None:
            return out
        order = nat._vpd_order(tx.ids, tx.targets)
        for k in sorted(set(sizes)):
            if k > 0:
                out[f"vpd@{k}"] = native.Graph(order[:k], complete=True)
        reach = nat.reachable(tx.T, tx.targets)
        out["cone"] = native.Graph(masks={n: (tx.ci[n] > 0) & reach[n] for n in nat.names})
        return out

    # ---- everything

    def score(self, tx: Text, graphs: list[native.Graph], explanations: list[str] | None = None) -> list[dict]:
        """Every measure of each graph on the text, one dict per graph."""
        t0 = time.perf_counter()
        nat = self.nat
        f1 = self.sufficiency(tx, graphs)
        f2 = self.interchange(tx, graphs)
        f3 = self.edges(tx, graphs)
        total = sum(nat.C.values())
        rows = []
        for b, g in enumerate(graphs):
            st, un = f3[b]["stated"], f3[b]["unstated"]
            row = {"nodes": g.count(), "edges": None if g.complete else g.edges(),
                   "bits": g.bits(nat.positions(tx.targets), total, tx.targets),
                   "f1_text_held": f1["text_held"][b], "f1_changed_held": f1["changed_held"][b],
                   "f1_text_ablated": f1["text_ablated"][b], "f1_changed_ablated": f1["changed_ablated"][b],
                   "f2_share": f2["share"][b], "f2_positions": [[p, m, k] for p, m, k in f2["positions"][b]],
                   "bypass_share": f2["bypass_share"], "movement_bits": f2["movement_bits"],
                   "f3_stated": None if st is None else _mean(st), "f3_unstated": None if un is None else _mean(un),
                   "f3_auc": f3[b]["auc"], "f3_cuts": {"stated": st, "unstated": un}}
            for name, fn in self.hooks.items():
                row[name] = None if fn is None else fn(tx.task, g, (explanations or [""] * len(graphs))[b])
            rows.append(row)
        seconds = time.perf_counter() - t0 + tx.seconds
        for row in rows:
            row["seconds_text"] = seconds
        return rows


def _mean(xs) -> float | None:
    return float(np.mean(xs)) if xs else None


def _auc(stated: list[float], unstated: list[float]) -> float | None:
    """P(a stated edge's cut > an unstated pair's), ties counting one half."""
    if not stated or not unstated:
        return None
    a, b = np.asarray(stated)[:, None], np.asarray(unstated)[None, :]
    return float(((a > b).mean() + 0.5 * (a == b).mean()))


# ---- answer sets and the table


def answer_file(directory: Path, stem: str) -> Path | None:
    """A text's answer in an answer set's directory: <stem>.py (a graph program), else <stem>.json holding a circuit
    ("nodes"; a search's directory also keeps each answer's scores as <stem>.json)."""
    py, js = directory / f"{stem}.py", directory / f"{stem}.json"
    if py.exists():
        return py
    if js.exists() and "nodes" in json.loads(js.read_text()):
        return js
    return None


def load_answer(path: Path, task: dict, targets: list[int], nat: native.Native) -> tuple[native.Graph, dict, str]:
    """(graph, defects, explanation) of an answer file: Circuit JSON (.json) or a graph program (.py, traced by mech)."""
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
           ("f1_text_held", "F1 text", "{:.3g}"), ("f1_changed_held", "F1 changed", "{:.3g}"),
           ("f1_text_ablated", "F1 text -rem", "{:.3g}"), ("f1_changed_ablated", "F1 chg -rem", "{:.3g}"),
           ("f2_share", "F2 share", "{:.3f}"), ("bypass_share", "bypass share", "{:.3f}"),
           ("f3_stated", "F3 stated", "{:.3g}"), ("f3_unstated", "F3 unstated", "{:.3g}"), ("f3_auc", "F3 AUC", "{:.3f}"),
           ("E1", "E1", "{:.3g}"), ("E2", "E2", "{:.3f}"), ("seconds_text", "s/text", "{:.1f}")]


def table(results: dict) -> str:
    """One table of an answer set and its baselines: the mean of each measure over the texts that have it.
    results: {"set": name, "rows": {answer name -> [row per text]}}."""
    names = list(results["rows"])
    head = ["answer", "texts"] + [c[1] for c in COLUMNS]
    lines = [head]
    for a in names:
        rows = results["rows"][a]
        line = [a, str(len(rows))]
        for key, _, fmt in COLUMNS:
            vals = [r[key] for r in rows if r.get(key) is not None]
            line.append(fmt.format(float(np.mean(vals))) if vals else "-")
        lines.append(line)
    widths = [max(len(x[i]) for x in lines) for i in range(len(head))]
    return f"answer set {results['set']}\n" + "\n".join("  ".join(x[i].rjust(widths[i]) for i in range(len(head))) for x in lines)


def score_sets(sets: dict, split: str, n: int, ge: GoalEval | None = None, baselines: bool = True, log=print) -> list[dict]:
    """Every answer set (name -> directory) on the split's first n texts that each set answers, with the baselines at
    each answer's size -> one results dict per set ({"set", "rows": {answer name -> rows}})."""
    ge = ge or GoalEval()
    nat = ge.nat
    results = {s: {"set": s, "rows": {s: [], "empty": [], "vpd@matched": [], "cone": []}} for s in sets}
    for p in native.tasks(split)[:n]:
        task = json.loads(p.read_text())
        ids, targets = native.text(p)
        found = {s: f for s, f in ((s, answer_file(Path(d), p.stem)) for s, d in sets.items()) if f is not None}
        if not found:
            continue
        tx = ge.text(ids, targets, native.task_seed(p.stem), task)
        graphs, expl, defects = [], [], []
        for s, f in found.items():
            g, d, e = load_answer(f, task, targets, nat)
            graphs.append(g)
            expl.append(e)
            defects.append(d)
        base = ge.baselines(tx, [g.count() for g in graphs]) if baselines else {}
        names = list(base)
        rows = ge.score(tx, graphs + [base[k] for k in names], expl + [""] * len(names))
        by = dict(zip(names, rows[len(graphs):]))
        for j, s in enumerate(found):
            row = {**rows[j], **defects[j], "task": p.stem}
            results[s]["rows"][s].append(row)
            for k in ("empty", "cone"):
                if k in by:
                    results[s]["rows"][k].append({**by[k], "task": p.stem})
            k = f"vpd@{graphs[j].count()}"
            if k in by:
                results[s]["rows"]["vpd@matched"].append({**by[k], "task": p.stem})
        log(f"{p.stem}: " + ", ".join(f"{s} F1 {rows[j]['f1_text_held']:.2f}/{rows[j]['f1_changed_held']:.2f} F2 {rows[j]['f2_share'] if rows[j]['f2_share'] is not None else float('nan'):.2f}"
                                     for j, s in enumerate(found)) + f", {rows[0]['seconds_text']:.1f} s")
    return list(results.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--set", action="append", required=True, help="NAME=DIR: an answer set, one answer per text")
    s.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--out", type=Path)
    s.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    sets = dict(x.split("=", 1) for x in args.set)
    results = score_sets(sets, args.split, args.n, baselines=not args.no_baselines)
    for r in results:
        print(table(r), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
