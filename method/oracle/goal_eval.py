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
     the text), positions pooled by movement. The remainder share is the same with no subcomponent computing: what
     VPD's remainders alone carry.
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
import json
import math
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
RANDOM_PAIRS = 32  # F3's unstated pairs per answer
MAX_EDGES = 64  # F3's stated edges per answer (a uniform draw when an answer states more)
UNIT_LIMIT = 20_000  # F2 computes at most this many subcomponents through carry's unit path; larger sets take masks
CHUNK = 16  # answers (F1, F2) or cuts (F3) per batch


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
    """A graph's stated edges, (writer, reader), reader None for the prediction."""
    return [(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out]


# ---- one text


class Text:
    """What every answer on one text shares: the model's run, the changed prompts and their references, the
    remainders' values, F2's positions and held-out prompts."""

    def __init__(self, ge: "GoalEval", ids: list[int], targets: list[int], seed: int, task: dict | None = None):
        nat = ge.nat
        t0 = time.perf_counter()
        self.ids, self.targets, self.T, self.seed, self.task = list(ids), list(targets), len(ids), seed, task
        self.rec = nat._model_record(self.ids)[0]
        self.ref = nat.reference([self.ids], self.targets)
        cap = carry._Remainders(nat)
        with torch.no_grad(), cap:
            nat.run(self.ids, self.targets, nat._full_plan(self.T, 1), *nat._one())
        if cap.seen != set(nat.names):
            raise RuntimeError("did not capture every remainder product")
        self.rem = cap.values
        self.prompts = nat.changes(self.ids, self.targets, n=ge.changes, seed=seed)
        self.prompt_refs = [nat.reference([x], self.targets) for x in self.prompts]
        self.f2 = []  # (position, held-out prompts)
        if max(self.targets) >= 1:
            moved, _ = nat.sensitivity(self.ids, self.targets, torch.Generator(device="cpu").manual_seed(seed))
            total = float(moved.sum())
            for j in (moved >= ge.share_floor * total).nonzero().flatten().tolist() if total > 0 else []:
                p = j + 1
                self.f2.append((p, carry.alternatives(nat, self.ids, p, ranks=ge.ranks)))
        self.ci = nat.importance(self.ids) if nat.has_importance else None
        self.seconds = time.perf_counter() - t0


# ---- the scorecard


class GoalEval:
    """Scores answers (native.Graph) on texts. hooks: name -> fn(task, graph, explanation) -> float or None, for the
    English reader's bits ("E1") and question answering ("E2"); unset, their columns are None."""

    def __init__(self, nat: native.Native | None = None, *, changes: int = native.CHANGES, ranks=HELD_OUT_RANKS,
                 share_floor: float = SHARE_FLOOR, random_pairs: int = RANDOM_PAIRS, max_edges: int = MAX_EDGES, chunk: int = CHUNK):
        self.nat = nat or native.Native()
        self.changes, self.ranks, self.share_floor = changes, tuple(ranks), share_floor
        self.random_pairs, self.max_edges, self.chunk = random_pairs, max_edges, chunk
        self.hooks: dict = {"E1": None, "E2": None}

    def text(self, ids: list[int], targets: list[int], seed: int | None = None, task: dict | None = None) -> Text:
        return Text(self, ids, targets, native.task_seed(str(ids[:8])) if seed is None else seed, task)

    # ---- F1

    @torch.no_grad()
    def sufficiency(self, tx: Text, graphs: list[native.Graph]) -> dict:
        """Per graph, F1 in bits: on the text and averaged over the changed prompts, remainders held at their values
        on the text ("held") and ablated ("ablated")."""
        nat = self.nat
        zu, zr = nat._zero()
        one_r = nat._one()[1]
        out = {k: torch.zeros(len(graphs), device=nat.dev) for k in ("text_held", "changed_held", "text_ablated", "changed_ablated")}
        weights = [getattr(x, "weight", 1.0) for x in tx.prompts]
        for s0 in range(0, len(graphs), self.chunk):
            part = graphs[s0:s0 + self.chunk]
            plan = nat._plan(part, tx.T)
            for i, (x, logp) in enumerate([(tx.ids, tx.ref)] + list(zip(tx.prompts, tx.prompt_refs))):
                with carry._Remainders(nat, tx.rem):
                    held = carry.kl(logp, nat.run(x, tx.targets, plan, zu, one_r))
                ablated = carry.kl(logp, nat.run(x, tx.targets, plan, zu, zr))
                if i == 0:
                    out["text_held"][s0:s0 + len(part)] = held
                    out["text_ablated"][s0:s0 + len(part)] = ablated
                else:
                    out["changed_held"][s0:s0 + len(part)] += weights[i - 1] * held / len(tx.prompts)
                    out["changed_ablated"][s0:s0 + len(part)] += weights[i - 1] * ablated / len(tx.prompts)
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
        the remainder share (nothing computing). Sets of at most UNIT_LIMIT subcomponents run through carry's
        subcomponent units (positions from the change on), larger ones as dense masks (carry's node masks); both
        hold every other activation at its value on the text."""
        nat, hi = self.nat, max(tx.targets)
        rows = [[] for _ in graphs]
        empty_kl, moved = 0.0, 0.0
        for p, prompts in tx.f2:
            live = [self._live(g, p, hi) for g in graphs]
            small = [b for b, nodes in enumerate(live) if len(nodes) <= UNIT_LIMIT]
            big = [b for b in range(len(graphs)) if b not in small]
            union = list(dict.fromkeys(nd for b in small for nd in live[b]))
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
        return {"share": share, "positions": rows, "remainder_share": None if moved <= 0 else 1 - empty_kl / moved,
                "movement_bits": moved}

    # ---- F3

    def _ones_u(self) -> dict:
        nat = self.nat
        return nat._const("goal_ones_u", lambda: {n: torch.ones(1, 1, nat.C[n], device=nat.dev) for n in nat.names})

    def _held_all(self, T: int) -> dict:
        nat = self.nat
        return nat._const(("goal_held_all", T), lambda: {n: torch.ones(1, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})

    def _none_plan(self, T: int, B: int) -> dict:
        """Native's plan of B copies of the whole model with no node held at 1: every subcomponent at its u."""
        nat = self.nat
        none = nat._const(("goal_none", T), lambda: {n: torch.zeros(1, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
        return {**nat._full_plan(T, B), "G": none, "none": True}

    @torch.no_grad()
    def cuts(self, tx: Text, pairs: list[tuple]) -> list[float]:
        """The KL in bits of the model on the text from the model with one connection cut, per (writer, reader) pair
        (reader None: the prediction). A reader's new activation is computed with the writer's output removed from
        its input alone: every other activation and every remainder held at its value in the model, the writer at
        mask 0; the model then runs with that one activation replaced (path patching of the connection). Into the
        prediction: the final residual stream without the writer's output."""
        nat, T = self.nat, tx.T
        res = [0.0] * len(pairs)
        inner = [j for j, (_, r) in enumerate(pairs) if r is not None]
        final = [j for j, (_, r) in enumerate(pairs) if r is None]
        one_u, one_r = nat._one()

        def removed(js):  # u with each row's writer at 0
            u = dict(self._ones_u())
            by = {}
            for b, j in enumerate(js):
                w = pairs[j][0]
                by.setdefault(w[0], []).append((b, w[1], w[2]))
            for n, lst in by.items():
                m = torch.ones(len(js), T, nat.C[n], device=nat.dev)
                bi, ti, ci = nat._ints([[e[0] for e in lst], [e[1] for e in lst], [e[2] for e in lst]])
                m[bi, ti, ci] = 0.0
                u[n] = m
            return u

        for s0 in range(0, len(inner), self.chunk):
            js = inner[s0:s0 + self.chunk]
            B = len(js)
            by = {}
            for b, j in enumerate(js):
                r = pairs[j][1]
                by.setdefault(r[0], []).append((b, r[1], r[2]))
            held = self._held_all(T)
            patch = {n: (held[n], tx.rec[n]) for n in nat.names}
            idx = {}
            for n, lst in by.items():
                m = torch.ones(B, T, nat.C[n], dtype=torch.bool, device=nat.dev)
                idx[n] = tuple(nat._ints([[e[0] for e in lst], [e[1] for e in lst], [e[2] for e in lst]]))
                m[idx[n]] = False
                patch[n] = (m, tx.rec[n])
            rec = {}
            with carry._Remainders(nat, tx.rem):
                nat.run(tx.ids, tx.targets, self._none_plan(T, B), removed(js), one_r, patch=patch, record=rec, keep=set(by))
            patch2 = {}
            for n in by:
                m = torch.zeros(B, T, nat.C[n], dtype=torch.bool, device=nat.dev)
                m[idx[n]] = True
                v = tx.rec[n].expand(B, -1, -1).clone()
                v[idx[n]] = rec[n][idx[n]]
                patch2[n] = (m, v)
            lq = nat.run(tx.ids, tx.targets, nat._full_plan(T, B), one_u, one_r, patch=patch2)
            for b, k in enumerate(carry.kl(tx.ref, lq).tolist()):
                res[js[b]] = k
            native.free(nat.dev)
        for s0 in range(0, len(final), self.chunk):
            js = final[s0:s0 + self.chunk]
            held = self._held_all(T)
            patch = {n: (held[n], tx.rec[n]) for n in nat.names}
            with carry._Remainders(nat, tx.rem):
                lq = nat.run(tx.ids, tx.targets, self._none_plan(T, len(js)), removed(js), one_r, patch=patch)
            for b, k in enumerate(carry.kl(tx.ref, lq).tolist()):
                res[js[b]] = k
        return res

    def edges(self, tx: Text, graphs: list[native.Graph]) -> list[dict]:
        """Per graph, F3: the cuts of its stated edges (at most max_edges, a uniform draw beyond) and of random
        unstated pairs of its nodes that the model connects (random_pairs); a complete graph states every connection
        among its nodes, so it has no unstated pair (its stated edges are drawn from them)."""
        rng = np.random.default_rng(tx.seed)
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
                   "remainder_share": f2["remainder_share"], "movement_bits": f2["movement_bits"],
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
           ("f2_share", "F2 share", "{:.3f}"), ("remainder_share", "rem share", "{:.3f}"),
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
        found = {s: next((Path(d) / f"{p.stem}{ext}" for ext in (".json", ".py") if (Path(d) / f"{p.stem}{ext}").exists()), None)
                 for s, d in sets.items()}
        found = {s: f for s, f in found.items() if f is not None}
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
