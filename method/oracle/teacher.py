"""Teacher circuits (#2951 graph oracle): whole-prediction circuits in VPD's semantics, found by measured removal.

A circuit C runs with its subcomponents on and every other one off (VPD's masked forward pass, as goal_eval.GoalEval.runs),
remainders live (the component reward's convention; components.CONVENTION). Every KL is the model's next-token
distribution at the target from the run's, in bits, and the reward's terms are goal_eval's log-scale fields:
  F1 = clip(ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD), 0, 1),  N = clip(ln(KL_{VPD minus C} / KL_VPD) / ln(KL_empty / KL_VPD), 0, 1)
with VPD's own answer (causal importance above zero at the positions that reach the target) and the empty circuit as
the bounds, "VPD minus C" VPD's answer with C's subcomponents off.

Per text:
  1. The F1 circuit: first from VPD's answer at the targets alone (chunks of 4, then single subcomponents); when that
     cannot reach F1 >= 0.9, from VPD's whole answer, rounds at each level (whole positions, sites, chunks of 64, 16 and 4 of a site's
     subcomponents by index, single subcomponents) measure every block's removal, rank the blocks by the KL their removal
     leaves, and remove the most of them in that order that keeps F1 >= 0.9 (KL_C <= KL_empty^0.1 KL_VPD^0.9), until no
     single removal does; with --start-top K, from VPD's top K through the last three levels only. By default
     (--f1-start both) the F1 circuit is the union of both searches' circuits, the targets' and the whole answer's
     (fleet's paired gate on the 20 held-out texts: F1 +0.017 [+0.004, +0.032], N and reward unchanged, all of it on
     non-local texts, where the target search keeps the target alone); --f1-start targets or whole, one of them. (The linear recovery
     >= 0.9 admits circuits ~10 bits from the model, the empty circuit being ~100 bits away: on text9000 it pruned to 5
     subcomponents at the target.)
  2. VPD's top-k: the smallest prefix of VPD's order (Native._vpd_order) whose necessity N reaches 0.99 (bisection, at
     most 1,024).
  3. The joint path from their union: each step measures every single subcomponent's removal (two runs: the circuit and
     VPD's answer without it), ranks them by the mean of F1 and N they leave, measures removing the first 1, 2, 4, ...
     of them up to a tenth of the circuit and takes the prefix with the highest measured mean, down to 4 subcomponents. Every step is recorded (size, F1, N, KL_C, KL_{VPD minus C}, positions, the subcomponents
     removed) beside VPD's own top-k at the same size.
  4. The teacher circuit is the smallest path point with F1 >= 0.9 and N >= 0.9, the form of the decomposition
     objective (the least description subject to faithfulness; 0.9 on the log scale is within ~10% of VPD's own
     answer in orders of magnitude); with none, the point with the largest min(F1, N). Each point also records
     (F1 + N) / 5 - 0.5 size, size goal_eval.size_term of the nodes' (and connections') description bits over
     goal_eval.size_scale (the reward's), for re-picking.
  5. With --edges, connections in C's semantics (goal_eval.GoalEval.graph_ref, as the scorecard runs a stated graph;
     EdgePruner, on the engine): the circuit's nodes at 1 in the stream, each of its readers reading it less every
     connection the model has from another of its nodes that is not stated; everything outside the circuit, the
     remainders included, reading the full stream. At a path point, from every connection the model has among its
     nodes (native.all_edges and the residual writers at the targets into the prediction), the rounds of step 1 over
     single connections: within F1 >= 0.9 the most of the ranked ones that keep it, above it the prefix with the
     lowest KL (single removals ranked in bfloat16 on CUDA; every removal kept or refused measured in float32). The
     nodes, and so N, stay the point's. F1 rises as connections go (text9002's 92 subcomponents: 0.881 with all 1,004
     connections, 0.919 with 65, in native.run's graph semantics). With --edges-at search, the teacher circuit is the
     smallest path point whose connections reach F1 >= 0.9 with N >= 0.9 (bisection between the smallest point with
     N >= 0.9 and step 4's pick), with its connections, and the points nearest 16, 32 and 64 get theirs too; with
     --edges-at pick (the default), connections at step 4's pick only.
  With --from-path DIR, steps 1-3 come from DIR's <task>.path.json (the start set, every step's removals and
  measurements, the F1 circuit and top-k; for path files without the last two, DIR's <task>.f1.json and
  <task>.vpd_top<k>.json), so a text costs the pick and its connections alone. Saved node ids are matched by node: a
  start node VPD's answer here lacks (causal importance at 0 on another GPU, where it was above) joins the answer.
  Step 4 picks among the path points of at most --max-nodes nodes (MAX_NODES, the answer budget in nodes); with
  --max-tokens, when the teacher circuit's answer (components.render) is longer than that in the explainer's
  tokenizer, the next smaller path point with its connections instead, until it fits.
  With --no-strand, no non-query/key node is left without a stated connection: each keeps its most important one (the
  largest measured cut effect, EdgePruner.protect), and one the model connects to no other node of the circuit is
  dropped (queries and keys act through the attention pattern). Without it, the descent left text9000's greedy and
  v3 circuits with half and 92% of their nodes unconnected: nodes that VPD's answer and the every-connection run need
  (N 0.93 -> 0.25, every-connection F1 0.91 -> 0.39 without them) but whose role the stated graph cuts (the lead's
  decision: the stated graph is the claim, a listed node needs a stated role).
  With --edge-objective robust, a connection removal is also admitted only while F1 under REPAIR_M of the omitted
  lever's flips (goal_eval.adversary_flips on the stated graph, the shared adversary) stays >= min(0.9, with every
  connection): connection pruning on graph F1 alone left the greedy circuits at F1 0.42 under one flip where every
  connection held 0.80 (inframove's table on a362911). The flips are checked only on the prefixes within the F1
  budget, the largest first (ACCEPT_BATCH a call), the largest admitted taken.
  6. With --repair (and --edges), the flip repair after the pick (Connector.repair): the reward's adversary, the
     omitted lever's flips at REPAIR_M (goal_eval.GoalEval.fidelity in the reward's mode, pool POOL), on the stated
     graph; its flips join the circuit and the connections are pruned again, a step taken only while it raises the
     reward's circuit score, mean(C, N) - 0.5 size; the teacher circuit is the best step. C's every-connection members
     keep it honest (a flip joined with its connections pruned is cut from readers it writes into in the model), its
     size term stops growth that does not pay (on text9011 a repair on the C-like minimum alone added 156
     subcomponents, most unconnected, for +0.036).
Every membership decision is measured; orderings only order.

Output per text in --out: <task>.json (the teacher circuit), <task>.f1.json, <task>.vpd_top<k>.json, <task>.n<size>.json at
the path points nearest 16 to 256 nodes (goal_eval's circuit JSON: nodes [matrix, index, position] and ["tok", 0,
position] input nodes; edges with --edges; with --vpd-edges, <task>.vpd_top<k>.edges.json: VPD's top k at the teacher
circuit's node count k with its connections by the same search, units compared at equal connection effort),
<task>.path.json (the start set, every step, the F1 circuit's node ids
"f1_index" and "top_k"; with --edges,
"edge_paths": per pruned point its node step, its connections [writer id, reader id or "out"] in node ids, and every
edge step: nodes, edges, the connection indices removed, KL_graph, F1_graph, N (the nodes'), size_term and
(F1 + N) / 5 - 0.5 size), and a line in summary_<offset>.jsonl.

  MPD_MEM_GIB=16 ~/mpd-data/venv/bin/python -B teacher.py --split heldout --n 20 --out ~/mpd-data/graph_oracle/teacher/joint_heldout
  teacher.py --split heldout --n 20 --save-vpd FILE   (VPD's answers and orders, for a machine without the CI network)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import carry
import components
import goal_eval
import native

RECOVERED = 0.9
ROUNDS = 40
CONVENTION = "live"   # the component reward's remainder convention (components.CONVENTION)
N_TARGET = 0.99       # VPD's top-k: the smallest prefix of VPD's order with necessity N >= this, at most K_MAX
K_MAX = 1024
FRACTION = 0.1        # each joint path step removes this share of the circuit, at least one subcomponent
MIN_SIZE = 4
SIZE_WEIGHT = 0.5     # components.SIZE_WEIGHT
TERMS = 5             # components.reward's quality terms: F1, N, F3, E1, E2
SIZES = (16, 32, 64, 128, 256)  # path points also written
MAX_NODES = 266       # the answer budget in nodes: components.ANSWER_BUDGET (3,500 tokens) over 8.65 tokens per node and
                      # 1.5 connections per node at 3.0 tokens each (primerl's measurements with Qwen3-8B's tokenizer)
LEVELS = ("position", "site", "chunk64", "chunk16", "chunk4", "subcomponent")  # blocks, coarse to fine
REPAIR_M = 4          # the omitted lever's budget the flip repair answers (goal_eval.adversary_flips; the reward's term)
ACCEPT_BATCH = 8      # Descent.prune's accept checks per call: the largest prefixes within the budget first
POOL = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"  # the shared adversary's pool (search side)


class Descent:
    """Measured removal over units (on, bool [N]) whose subsets kl_sets(ons [K, N]) measures: the rounds shared by
    Pruner (subcomponents) and EdgePruner (connections)."""

    def without(self, groups: list[torch.Tensor], ranking: bool = False) -> list[float]:
        """KL of the current set without each group of unit ids (built 16 batches at a time); ranking: by kl_rank, for
        an ordering only."""
        out = []
        for s0 in range(0, len(groups), 16 * self.B):
            part = groups[s0:s0 + 16 * self.B]
            on = self.on[None].repeat(len(part), 1)
            for i, g in enumerate(part):
                on[i, g] = False
            out += self.kl_rank(on) if ranking else self.kl_sets(on)
        return out

    def kl_rank(self, ons: torch.Tensor) -> list[float]:
        """The KLs a ranking orders by: kl_sets' (a subclass may run them at a lower precision)."""
        return self.kl_sets(ons)

    def kl(self, on: torch.Tensor | None = None) -> float:
        return self.kl_sets((self.on if on is None else on)[None])[0]

    def prune(self, level: str, budget: float, rounds: int = ROUNDS, log: list | None = None, accept=None) -> int:
        """Rounds of measured block removal at one level; returns the rounds run. Within the budget a round removes
        the most of the ranked blocks that keep KL_C <= budget (and, with accept, that accept admits: accept(ons [K,
        units]) -> per set a dict, admitted, or None; asked for the largest such prefixes first, ACCEPT_BATCH a call,
        until one is admitted); above it (a start such as VPD's answer at the targets alone can be farther than the
        budget while a subset is not: subcomponents cancel) it removes the prefix of the ranked blocks with the lowest
        measured KL_C, when that is lower than the circuit's. The blocks are ranked by kl_rank; the prefixes are
        measured by kl_sets. log, a list, receives each round's {"removed": unit ids, "kl": the KL measured after it}
        and accept's dict."""
        for r in range(rounds):
            blocks = self.blocks(level)
            if len(blocks) <= 1:
                return r
            single = torch.tensor(self.without(blocks, ranking=True))
            ranked = [blocks[i] for i in torch.argsort(single, stable=True).tolist()]
            sizes = [k for k in sorted({int(2 ** (e / 2)) for e in range(0, 60)}) if k <= len(ranked) - 1]
            if (single <= budget).any():
                ok = [(k, x) for k, x in zip(sizes, self.without([torch.cat(ranked[:k]) for k in sizes])) if x <= budget]
                info = {}
                if accept is not None and ok:                # the largest admitted, checked from the largest down
                    order, ok = sorted(ok, reverse=True), []
                    for s0 in range(0, len(order), ACCEPT_BATCH):
                        part = order[s0:s0 + ACCEPT_BATCH]
                        ons = self.on[None].repeat(len(part), 1)
                        for i, (k, _) in enumerate(part):
                            ons[i, torch.cat(ranked[:k])] = False
                        got = accept(ons)
                        hit = next((i for i, g in enumerate(got) if g is not None), None)
                        if hit is not None:
                            ok, info = [part[hit]], got[hit]
                            break
                if not ok:
                    return r
                k, x = max(ok)
                drop = torch.cat(ranked[:k])
                self.on[drop] = False
                if log is not None:
                    log.append({"removed": drop.tolist(), "kl": x, **info})
                continue
            current = self.kl()
            if current <= budget or float(single.min()) >= current:
                return r
            kl = self.without([torch.cat(ranked[:k]) for k in sizes])
            i = min(range(len(sizes)), key=lambda j: kl[j])
            if kl[i] >= current:
                return r
            drop = torch.cat(ranked[:sizes[i]])
            self.on[drop] = False
            if log is not None:
                log.append({"removed": drop.tolist(), "kl": kl[i]})
        return rounds


class Pruner(Descent):
    """Measured removal over the nodes of a circuit on one text. Nodes are (matrix, position, index); a block is a list
    of node ids; kl(blocks) measures the current circuit without each block, in batches of B."""

    fused = True  # the masked runs' norms and GELU as torch's fused kernels (Target.hidden(fused=)): float32-close
    WINDOWS = ("mps", "cuda")  # devices whose masked runs take kl_sets' windows: MPS to the bit, CUDA float32-close

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, start: dict, live: bool = False, B: int = 32):
        self.ge, self.tx, self.live, self.B = ge, tx, live, B
        nat = ge.nat
        self.names = nat.names
        self.nodes = [(n, int(t), int(c)) for n in nat.names for t, c in start[n].nonzero().tolist()]
        self.on = torch.ones(len(self.nodes), dtype=torch.bool, device=nat.dev)
        self.mat = torch.tensor([nat.names.index(n) for n, _, _ in self.nodes], device=nat.dev)
        self.t = torch.tensor([t for _, t, _ in self.nodes], device=nat.dev)
        self.c = torch.tensor([c for _, _, c in self.nodes], device=nat.dev)
        self.measured = 0

    def masks(self, on: torch.Tensor) -> dict:
        nat, T = self.ge.nat, self.tx.T
        out = {}
        for j, n in enumerate(self.names):
            m = torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev)
            sel = on & (self.mat == j)
            m[self.t[sel], self.c[sel]] = True
            out[n] = m
        return out

    def _gather(self, every: bool = False) -> None:
        """Per site the subcomponents among the nodes (the start set), all others being off in every run: their columns
        of V and U, and each node's column among them. The products then run on these alone (a sum over fewer
        terms: equal to the full products within float32 rounding, not to the bit). every: all of a site's columns
        (the full products)."""
        tg, T = self.ge.nat.target, self.tx.T
        self.sel = [(self.mat == j).nonzero().flatten() for j in range(len(self.names))]
        self.cols = []
        self.flat = torch.empty(len(self.nodes), dtype=torch.long, device=self.t.device)  # each node's place in _site_masks' buffer
        self.offsets, at = [], 0
        for j, n in enumerate(self.names):
            if every:
                cols, ci = torch.arange(self.ge.nat.C[n], device=self.c.device), self.c[self.sel[j]]
            else:
                cols, ci = torch.unique(self.c[self.sel[j]], return_inverse=True)
            st = tg.site(n)
            self.cols.append((ci, (st.V.index_select(1, cols), st.U.index_select(0, cols))))
            self.flat[self.sel[j]] = at + self.t[self.sel[j]] * len(cols) + ci
            self.offsets.append((at, len(cols)))
            at += T * len(cols)
        self.width = at

    def _site_masks(self, on: torch.Tensor) -> dict:
        """Each site's float mask [B, T, columns] of node subsets on [B, N]: views of one buffer filled in one step."""
        T = self.tx.T
        buf = torch.zeros(len(on), self.width, device=on.device)
        buf[:, self.flat] = on.float()
        return {n: buf[:, at:at + T * K].view(len(on), T, K) for n, (at, K) in zip(self.names, self.offsets)}

    def _run(self, masks: dict, B: int, start: int = 0, prefix: list | None = None, layer: int = 0, keys_values: bool = False):
        """The masked forward of B copies of the text (Target.hidden from start and layer, or its keys_values)."""
        nat, tg, T = self.ge.nat, self.ge.nat.target, self.tx.T
        for j, n in enumerate(self.names):
            st = tg.site(n)
            st.mask = masks[n][:, start:]
            st.gathered = self.cols[j][1]
            st.delta_mask = torch.ones(B, T - start, device=nat.dev) if self.live else None
        ids = torch.tensor([self.tx.ids], device=nat.dev).expand(B, -1)
        try:
            return tg.keys_values(ids, self.fused) if keys_values else tg.hidden(ids, start, prefix, layer, self.fused)
        finally:
            nat.vpd.clear()

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor, pad: bool = False) -> list[float]:
        """KL at the target of each node subset's run, ons [K, N] (float masks built on the device, in batches of B; a
        last batch of one row runs alone, at batch 1, unless pad).
        With at least two full batches on a device in WINDOWS, the rows are taken in order of the first layer, then the
        first position, where each differs from the call's majority set, and a batch runs from its first such layer
        and position (at least 64 positions) on, the layers below and the positions before reading the majority set's
        run (its queries, keys, values and residual stream, computed once per set at batch B; Target.hidden's window).
        The batches depend only on the call's sets (a stable order, a fixed size), so a run reproduces itself. On MPS
        the numbers are the whole run's to the bit; on CUDA the kernels' shapes change their float32 rounding
        (test_teacher.py bounds it)."""
        nat, tg, T = self.ge.nat, self.ge.nat.target, self.tx.T
        if not hasattr(self, "sel"):
            self._gather()
            self.layer = torch.tensor([int(n.split(".")[1]) for n in self.names], device=nat.dev, dtype=torch.int32)[self.mat]
        K = len(ons)
        window = K >= 2 * self.B and T > 64 and str(nat.dev).split(":")[0] in self.WINDOWS
        order, first, lowest = torch.arange(K, device=nat.dev), None, None
        if window:
            base = ons.sum(0) * 2 > K
            key = base.cpu().numpy().tobytes()
            runs = self.__dict__.setdefault("_prefixes", {})                # the last two sets' runs (measure alternates C, ~C)
            if key not in runs:
                if len(runs) >= 2:
                    runs.pop(next(iter(runs)))
                runs[key] = self._run(self._site_masks(base[None].expand(self.B, -1)), self.B, keys_values=True)
            prefix = runs[key]
            off = ons != base
            pos = torch.where(off, self.t.int()[None], T).amin(1)            # each row's first position and layer off the base
            lay = torch.where(off, self.layer[None], tg.n_layer).amin(1)
            # A call's last row alone in its batch runs at batch 1 (no padding), which can round differently: it stays last
            n = K - 1 if K % self.B == 1 and not pad else K
            order = torch.cat([torch.argsort((lay * (T + 1) + pos)[:n], stable=True), order[n:]])
            first, lowest = pos[order].tolist(), lay[order].tolist()
        kl = [0.0] * K
        for s0 in range(0, K, self.B):
            idx = order[s0:s0 + self.B]
            on = ons[idx]
            valid = len(on)
            if (pad or valid > 1) and valid < self.B:                           # one batch shape: the GPU compiles per shape
                on = torch.cat([on, on[-1:].expand(self.B - valid, -1)])
            B = len(on)
            reuse = window and B == self.B
            start = max(0, min(min(first[s0:s0 + valid]), T - 64, min(self.tx.targets))) if reuse else 0
            layer = min(lowest[s0], tg.n_layer - 1) if reuse else 0
            hidden = self._run(self._site_masks(on), B, start, prefix if reuse else None, layer)
            lq = torch.log_softmax((hidden[:, [t - start for t in self.tx.targets]] @ tg.wte.T).float(), -1)
            for i, x in zip(idx.tolist(), carry.kl(self.tx.ref, lq).tolist()[:valid]):
                kl[i] = x
        self.measured += K
        return kl

    def blocks(self, level: str) -> list[torch.Tensor]:
        """The current circuit's blocks at a level, as node-id tensors."""
        ids = self.on.nonzero().flatten()
        if level == "position":
            keys = self.t[ids]
        elif level == "site":
            keys = self.t[ids] * len(self.names) + self.mat[ids]
        elif level.startswith("chunk"):               # chunks of K subcomponents within a site, by index
            K = int(level[5:])
            site = self.t[ids] * len(self.names) + self.mat[ids]
            order = torch.argsort(site * (1 << 20) + self.c[ids], stable=True)
            ids, site = ids[order], site[order]
            first = torch.ones_like(site, dtype=torch.bool)
            first[1:] = site[1:] != site[:-1]
            start = torch.cummax(torch.where(first, torch.arange(len(ids), device=ids.device), 0), 0).values
            keys = site * (1 << 20) + (torch.arange(len(ids), device=ids.device) - start) // K
        else:
            return list(ids.split(1))
        order = torch.argsort(keys, stable=True)
        ids, keys = ids[order], keys[order]
        _, counts = torch.unique_consecutive(keys, return_counts=True)
        return list(ids.split(counts.tolist()))

class EdgePruner(Descent):
    """Measured removal over the connections among a circuit's nodes in C's semantics (goal_eval.GoalEval.graph_ref, as
    the scorecard runs a stated graph; here on engine.Engine, plan(semantics="c")): the circuit's nodes at 1 in the
    stream, each of its readers reading the stream less every connection the model has from another of its nodes
    that is not kept, the prediction less every residual writer at a target that is not kept; every other
    subcomponent off; the remainders live or off, reading the full stream. Connections: every one the model has among
    the nodes (native.all_edges), (writer, reader), reader None for the prediction, in the plan's order; a subset bool
    [E] runs as 0/1 strengths. On CUDA the single removals a round ranks run in bfloat16 (kl_rank: an ordering); every
    removal kept or refused and every KL recorded is a float32 run."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, live: bool = False, B: int = 32):
        import engine

        nat = ge.nat
        self.ge, self.tx, self.live, self.B = ge, tx, live, B
        par, out = native.all_edges(nodes, tx.targets)
        self.graph = native.Graph(nodes, par, out)
        self.edges = [(w, r) for r, ws in par.items() for w in ws] + [(w, None) for w in out]
        self.inner = len(self.edges) - len(out)              # connections into nodes first, then into the prediction
        self.on = torch.ones(len(self.edges), dtype=torch.bool, device=nat.dev)
        cols = {n: sorted({c for m, _, c in nodes if m == n}) for n in nat.names}
        self.eng = engine.Engine(nat, tx.ids, tx.targets, live=live,
                                 columns={n: torch.tensor(cols[n], dtype=torch.long, device=nat.dev) for n in nat.names})
        self.plan = self.eng.plan(self.graph, semantics="c")
        self.rank_precision = "bf16" if str(nat.dev).startswith("cuda") else "score"
        self.protected = torch.zeros(len(self.edges), dtype=torch.bool, device=nat.dev)   # never offered for removal
        self.measured = 0

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor, precision: str = "score") -> list[float]:
        """KL at the target of each connection subset's run, ons [K, E], in batches of B."""
        kl = []
        for s0 in range(0, len(ons), self.B):
            on = ons[s0:s0 + self.B].float()
            kl += self.eng.graph_kl(self.plan, None, on[:, :self.inner], on[:, self.inner:], precision).tolist()
        self.measured += len(ons)
        return kl

    def kl_rank(self, ons: torch.Tensor) -> list[float]:
        return self.kl_sets(ons, self.rank_precision)

    def blocks(self, level: str) -> list[torch.Tensor]:
        """The kept connections' blocks, protected ones aside: every connection out of one writer ("writer"), into one
        reader or the prediction ("reader"), or single connections."""
        ids = (self.on & ~self.protected).nonzero().flatten()
        if level not in ("writer", "reader"):
            return list(ids.split(1))
        if not hasattr(self, "ends"):
            at = {nd: i for i, nd in enumerate(self.graph.nodes)}
            self.ends = {"writer": torch.tensor([at[w] for w, _ in self.edges], device=ids.device),
                         "reader": torch.tensor([at[r] if r is not None else len(at) for _, r in self.edges], device=ids.device)}
        keys = self.ends[level][ids]
        order = torch.argsort(keys, stable=True)
        _, counts = torch.unique_consecutive(keys[order], return_counts=True)
        return list(ids[order].split(counts.tolist()))

    def protect(self) -> None:
        """Protects each non-query/key node's single most important kept connection, the one whose cut leaves the
        largest KL (each kept connection's cut measured in float32), so that no such node is left without a stated
        connection (queries and keys act through their layer's attention pattern without one)."""
        kept = self.on.nonzero().flatten()
        cut = dict(zip(kept.tolist(), self.without(list(kept.split(1)))))
        best = {}
        for e, x in cut.items():
            for nd in self.edges[e]:
                if nd is not None and native._layer_kind(nd)[1] not in ("q_proj", "k_proj") and (nd not in best or x > cut[best[nd]]):
                    best[nd] = e
        if best:
            self.protected[torch.tensor(sorted(set(best.values())), device=self.protected.device)] = True


def scores(empty: float, vpd: float, c: float, minus: float) -> tuple[float, float, float]:
    """(F1, N, their mean): goal_eval's log-scale reward fields from KL_C and KL_{VPD minus C}."""
    f1 = min(1.0, max(0.0, goal_eval.recovered_log(empty, c, vpd)))
    n = goal_eval.necessity_log(empty, minus, vpd)
    return f1, n, (f1 + n) / 2


def measure(pr: Pruner, ons: torch.Tensor, empty: float, vpd: float) -> list[dict]:
    """Each node set's KL_C and KL_{VPD minus C} (two runs: the set, and VPD's answer without it), F1, N and mean."""
    K, out = len(ons), []
    if 2 * K <= pr.B:                  # one batch either way
        kl = pr.kl_sets(torch.cat([ons, ~ons]))
        cs, ms = kl[:K], kl[K:]
    else:                              # apart, each call's rows sharing one set they differ from; every row at batch B
        cs, ms = pr.kl_sets(ons, pad=True), pr.kl_sets(~ons, pad=True)
    for c, m in zip(cs, ms):
        f1, n, j = scores(empty, vpd, c, m)
        out.append({"kl_c": c, "kl_minus": m, "f1": f1, "n": n, "mean": j})
    return out


def top_k(pr: Pruner, order: torch.Tensor, empty: float, vpd: float, target: float = N_TARGET, cap: int = K_MAX) -> int:
    """The smallest k (bisection, measured) whose VPD-ordered prefix has N >= target, at most cap."""
    def prefix(k):
        on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=order.device)
        on[order[:k]] = True
        return on
    lo, hi = 0, min(cap, len(order))
    if measure(pr, prefix(hi)[None], empty, vpd)[0]["n"] < target:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if measure(pr, prefix(mid)[None], empty, vpd)[0]["n"] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def joint_path(pr: Pruner, start: torch.Tensor, empty: float, vpd: float) -> list[dict]:
    """Nested measured removal from `start` down to MIN_SIZE. Each step measures every single removal's F1 and N (two
    runs each) and ranks the removals by the mean they leave; then it measures removing the first 1, 2, 4, ... of them
    up to a tenth of the circuit (removals interact: subcomponents cancel) and takes the prefix with the highest
    measured mean, the longest on a tie. Records: the state after each step and the node ids removed."""
    on = start.clone()
    path = [{"size": int(on.sum()), "removed": [], **measure(pr, on[None], empty, vpd)[0]}]
    while int(on.sum()) > MIN_SIZE:
        ids = on.nonzero().flatten()
        trials = on[None].repeat(len(ids), 1)
        trials[torch.arange(len(ids), device=on.device), ids] = False
        rows = []
        for s0 in range(0, len(ids), 4 * pr.B):
            rows += measure(pr, trials[s0:s0 + 4 * pr.B], empty, vpd)
        ranked = ids[torch.argsort(torch.tensor([r["mean"] for r in rows]), descending=True, stable=True)]
        top = min(max(1, int(FRACTION * len(ids))), len(ids) - MIN_SIZE)
        ks = sorted({min(top, 2 ** e) for e in range(0, 20) if 2 ** e <= 2 * top})
        prefixes = on[None].repeat(len(ks), 1)
        for i, k in enumerate(ks):
            prefixes[i, ranked[:k]] = False
        got = measure(pr, prefixes, empty, vpd)
        i = max(range(len(ks)), key=lambda j: (got[j]["mean"], ks[j]))
        on[ranked[:ks[i]]] = False
        path.append({"size": int(on.sum()), "removed": ranked[:ks[i]].tolist(), **got[i]})
    return path


def edge_path(ep: EdgePruner, budget: float, row, levels: tuple = ("connection",), accept=None) -> list[dict]:
    """A circuit's connections removed by Descent.prune's rounds from ep.on (every connection, unless the caller set
    fewer; step 1's rule with its budget, and accept's admission if given), at each of `levels` in turn
    (EdgePruner.blocks: "writer", "reader", single connections; coarse levels first cut the rounds' measured sets on
    circuits of hundreds of nodes); row(kl, on, removed) records each state, the first the start's, with accept's
    dict."""
    log = []
    first, on = ep.kl(), ep.on.clone()
    for level in levels:
        ep.prune(level, budget, log=log, accept=accept)
    steps = [row(first, on.clone(), [])]
    for x in log:
        on[x["removed"]] = False
        steps.append({**row(x["kl"], on.clone(), x["removed"]), **{k: v for k, v in x.items() if k not in ("removed", "kl")}})
    return steps


class Connector:
    """A text's connection pruning (edge_path) and flip repair for node sets of its answer (pr.nodes, bool [N] sets),
    with the text's bounds, F1 budget and size term (size_term(on, inner, out))."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, pr: Pruner, empty: float, vpd: float, size_term, B: int = 32,
                 robust: dict | None = None, no_strand: bool = False):
        self.ge, self.tx, self.pr, self.empty, self.vpd, self.size_term, self.B = ge, tx, pr, empty, vpd, size_term, B
        self.robust = robust      # the adversary's pool: connection removals also keep F1 under the omitted lever's flips
        self.no_strand = no_strand  # every non-query/key node keeps a stated connection
        self.budget = empty ** (1 - RECOVERED) * vpd ** RECOVERED
        self.index = {nd: i for i, nd in enumerate(pr.nodes)}

    def connect(self, on: torch.Tensor, n: float, keep: set | None = None, new=()) -> tuple:
        """A node set's connections by edge_path: (the kept ones, the last step's row, every step, the EdgePruner, the
        node set wired); keep, a set of connections: the descent starts from those and every connection of the nodes
        `new`. With no_strand, a non-query/key node the model connects to none of the others is dropped (it has no
        connection to state; its N measured again), and each other one keeps its most important connection
        (EdgePruner.protect)."""
        pr, empty, vpd = self.pr, self.empty, self.vpd
        nodes = [pr.nodes[j] for j in on.nonzero().flatten().tolist()]
        dropped = []
        if self.no_strand:
            par, out = native.all_edges(nodes, self.tx.targets)
            touched = {w for ws in par.values() for w in ws} | set(par) | set(out)
            dropped = [nd for nd in nodes if nd not in touched and native._layer_kind(nd)[1] not in ("q_proj", "k_proj")]
            if dropped:
                on = on.clone()
                on[torch.tensor([self.index[nd] for nd in dropped], device=on.device)] = False
                nodes = [nd for nd in nodes if nd not in set(dropped)]
                n = measure(pr, on[None], empty, vpd)[0]["n"]
        ep = EdgePruner(self.ge, self.tx, nodes, CONVENTION == "live", self.B)
        if keep is not None:
            fresh = {pr.nodes[j] for j in new}
            ep.on = torch.tensor([e in keep or e[0] in fresh or e[1] in fresh for e in ep.edges], device=ep.on.device)
        if self.no_strand:
            ep.protect()
        size = int(on.sum())
        accept, start = None, {}
        if self.robust is not None:                          # the robust objective: F1 under REPAIR_M flips of the stated
            def flips(ons):                                  # graph kept >= min(0.9, the start's), checked on admissible
                got = self.ge.edge_flips(self.tx, ep.graph, ons, self.robust, REPAIR_M)            # prefixes only
                return [1.0 if g["f1_flips_omitted"].get(REPAIR_M) is None else g["f1_flips_omitted"][REPAIR_M] for g in got]
            start = {"f1_flips": flips(ep.on[None])[0]}
            floor = min(RECOVERED, start["f1_flips"])

            def accept(ons):
                return [{"f1_flips": f} if f >= floor else None for f in flips(ons)]

        def state(kl, e_on, removed):
            f1 = goal_eval._clip01(goal_eval.recovered_log(empty, kl, vpd))
            st = self.size_term(on, int(e_on[:ep.inner].sum()), int(e_on[ep.inner:].sum()))
            return {"nodes": size, "edges": int(e_on.sum()), "removed": removed, "kl_graph": kl, "f1_graph": f1,
                    "n": n, "size_term": st, "reward": (f1 + n) / TERMS - SIZE_WEIGHT * st}
        steps = edge_path(ep, self.budget, state, accept=accept)
        steps[0].update(start)
        summary = {**{k_: steps[-1][k_] for k_ in ("nodes", "edges", "kl_graph", "f1_graph", "n", "size_term", "reward")},
                   "all_edges": len(ep.edges), "sets": ep.measured, "unconnectable": len(dropped)}
        if start:                                            # the last admitted step's F1 under the flips (the start's if none)
            summary["f1_flips"] = next(s_["f1_flips"] for s_ in reversed(steps) if "f1_flips" in s_)
        print(f"  connections at {size} subcomponents: {summary['edges']} of {len(ep.edges)}, graph F1 {summary['f1_graph']:.3f} "
              f"(start {steps[0]['f1_graph']:.3f}), N {n:.3f}, {ep.measured} sets", flush=True)
        return [e for e, keep_ in zip(ep.edges, ep.on.tolist()) if keep_], summary, steps, ep, on

    def graph(self, on: torch.Tensor, kept: list) -> native.Graph:
        """A node set and its kept connections as the stated graph goal_eval scores."""
        parents = {}
        for w, r in kept:
            if r is not None:
                parents.setdefault(r, []).append(w)
        return native.Graph([self.pr.nodes[j] for j in on.nonzero().flatten().tolist()], parents, [w for w, r in kept if r is None])

    def score(self, on: torch.Tensor, kept: list, pool: dict) -> dict:
        """The reward's circuit score of a node set with its kept connections: C (goal_eval.GoalEval.fidelity, mode
        "reward"), N, the size term and mean(C, N) - SIZE_WEIGHT size; with the row's flips at REPAIR_M and their mask."""
        f = self.ge.fidelity(self.tx, [self.graph(on, kept)], pool, mode="reward")[0]
        inner = sum(r is not None for _, r in kept)
        st = self.size_term(on, inner, len(kept) - inner)
        return {"c": f["c"], "n": f["n_point"], "size_term": st, "score": (f["c"] + f["n_point"]) / 2 - SIZE_WEIGHT * st,
                "f1_graph": f["s_point"], "f1_nodes": f["s_wiring"], "f1_flips": f.get("f1_flips_c", {}).get(REPAIR_M),
                "mask": f.get("mask_flips_omitted", {}).get(REPAIR_M, [])}

    def repair(self, on: torch.Tensor, kept: list, summary: dict, pool: dict, max_nodes: int | None = None,
               m: int = REPAIR_M) -> tuple[list, int, str]:
        """The flip repair: the reward's adversary (the omitted lever's flips at REPAIR_M, from goal_eval.GoalEval.fidelity
        in the reward's mode) on the stated graph; its flips join the node set and the connections are pruned again
        (from the kept ones and every connection of the flips), each step taken only while it raises the reward's
        circuit score, mean(C, N) - SIZE_WEIGHT size (score()), and the set stays within max_nodes. (every step with its
        "on" and "kept"; the index of the best score; "score".) The score's C holds the repair honest: its point and
        flips members take the lower of the stated graph's run and the run with every connection among the nodes (a
        flip joined with its connections pruned is cut from the circuit's readers, which in the model it writes into:
        text9000's 29 subcomponents "repaired" to 49 on graph F1 and flips alone had every-connection F1 0.665), and
        its size term stops the growth that does not pay (on min(graph F1, every-connection F1, flips) alone, text9011
        grew from 45 subcomponents to 201 while every-connection F1 crept from 0.863 to 0.899, 90% of them unconnected).
        summary: the start's connect() row (unused but for its N)."""
        steps = []
        while True:
            sc = self.score(on, kept, pool)
            add = sorted({self.index[nd] for nd in sc.pop("mask") if nd in self.index})
            steps.append({"size": int(on.sum()), "edges": len(kept), **sc, "flips": add, "on": on, "kept": kept})
            print(f"  repair: {int(on.sum())} subcomponents, {len(kept)} connections, C {sc['c']:.3f}, N {sc['n']:.3f}, "
                  f"size {sc['size_term']:.3f}, score {sc['score']:.3f}", flush=True)
            if (len(steps) > 1 and sc["score"] <= steps[-2]["score"]) or not add or (max_nodes and int(on.sum()) + len(add) > max_nodes):
                break
            on = on.clone()
            on[torch.tensor(add, dtype=torch.long, device=on.device)] = True
            kept, _, _, _, on = self.connect(on, measure(self.pr, on[None], self.empty, self.vpd)[0]["n"], set(kept), add)
        return steps, max(range(len(steps)), key=lambda i: (steps[i]["score"], -steps[i]["size"])), "score"


def circuit_json(pr: Pruner, on: torch.Tensor, edges: list | None = None) -> dict:
    """goal_eval's circuit JSON: the nodes, a token-embedding input node at each of their positions, and the
    connections given as (writer, reader) nodes, reader None for the prediction (none without)."""
    nodes = sorted(((t, n, c) for (n, t, c), keep in zip(pr.nodes, on.tolist()) if keep))
    tok = sorted({t for t, _, _ in nodes})
    at = {(n, t, c): i for i, (t, n, c) in enumerate(nodes)}
    return {"nodes": [[n, c, t] for t, n, c in nodes] + [["tok", 0, t] for t in tok],
            "edges": [[at[w], "out" if r is None else at[r]] for w, r in edges or []], "explanation": ""}


def pick(rows: list[dict], f1: str, goal: float = RECOVERED, cap: int | None = None) -> tuple[int, str]:
    """(the smallest row with F1 and N >= goal, "goal"), or without one (the row with the largest min(F1, N), "max-min"),
    among the rows of at most cap nodes (every row without a cap; the smallest row when none is that small)."""
    within = [i for i, r in enumerate(rows) if cap is None or r["size"] <= cap] or [min(range(len(rows)), key=lambda i: rows[i]["size"])]
    ok = [i for i in within if rows[i][f1] >= goal and rows[i]["n"] >= goal]
    if ok:
        return min(ok, key=lambda i: (rows[i]["size"], -min(rows[i][f1], rows[i]["n"]))), "goal"
    return max(within, key=lambda i: (min(rows[i][f1], rows[i]["n"]), -rows[i]["size"])), "max-min"


def teach(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], B: int = 32, vpd_answer: dict | None = None,
          vpd_order: list | None = None, start_top: int = 0, f1_start: str = "both", edges: bool = False,
          from_path: dict | None = None, edges_at: str = "pick", max_nodes: int | None = MAX_NODES, tokens=None,
          max_tokens: int = components.ANSWER_BUDGET, repair_pool: dict | None = None, vpd_edges: bool = False,
          robust_pool: dict | None = None, no_strand: bool = False) -> tuple[dict, dict]:
    """(circuits by name, summary row) on one text: the F1 circuit (smallest at recovered_log >= 0.9, coarse to fine),
    VPD's top-k with N >= N_TARGET, the joint path from their union, and the path point the reward's terms favour.
    from_path: a saved path file (with "f1_index" and "top_k", or "f1_nodes" [matrix, index, position] and "top_k")
    in place of the F1 search, top-k and joint path. tokens: text -> its token count, for the answer budget."""
    t0 = time.perf_counter()
    nat = ge.nat
    tx = ge.text(ids, targets)
    if vpd_answer is not None:                                # VPD's answer computed where the causal-importance network is
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in vpd_answer.items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    if tx.vpd is None:
        raise ValueError("no VPD answer: run with the causal-importance network or pass --vpd")
    if from_path is not None:                                # the saved start set within VPD's answer here
        saved_start = [(n, int(t), int(c)) for n, c, t in from_path["start_nodes"]]
        added = sum(not bool(tx.vpd[n][t, c]) for n, t, c in saved_start)
        for n, t, c in saved_start:
            tx.vpd[n][t, c] = True
    empty, vpd = ge._bounds(tx, CONVENTION)
    budget = empty ** (1 - RECOVERED) * vpd ** RECOVERED
    pr = Pruner(ge, tx, tx.vpd, live=CONVENTION == "live", B=B)
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    order = vpd_order if vpd_order is not None else nat._vpd_order(ids, targets)
    order = torch.tensor([index[(n, int(t), int(c))] for n, t, c in order if (n, int(t), int(c)) in index], device=nat.dev)
    positions = sorted({p for p, _ in tx.f2} | set(targets))
    active = sum(int(tx.vpd[n][positions].sum()) for n in nat.names)
    node_bits = math.log2(nat.positions(targets) * sum(nat.C.values()))
    scale = ge.size_scale(tx)                                # the reward's size scale (goal_eval.size_term)

    def size_term(on, inner=0, out=0):                       # goal_eval.size_bits: connections' ends among the nodes
        n, k_tok = int(on.sum()), int(torch.unique(pr.t[on]).numel())
        bits = n * node_bits + k_tok * math.log2(nat.positions(targets)) + (math.log2(n + k_tok) * (2 * inner + out) if n else 0.0)
        return goal_eval.size_term(bits, scale)

    row = {"T": tx.T, "convention": CONVENTION, "vpd_size": len(pr.nodes), "vpd_positions": int(torch.unique(pr.t).numel()),
           "kl_empty": empty, "kl_vpd": vpd, "budget_kl": budget, "active_scale": active, "levels": {}}
    def search(levels, label):
        for level in levels:
            t1 = time.perf_counter()
            rounds = pr.prune(level, budget)
            row["levels"][f"{label}{level}"] = lv = {"rounds": rounds, "size": int(pr.on.sum()),
                                                     "positions": int(torch.unique(pr.t[pr.on]).numel()), "kl": pr.kl(),
                                                     "seconds": time.perf_counter() - t1}
            print(f"  {label}{level}: {lv['size']} subcomponents at {lv['positions']} positions, KL {lv['kl']:.3g} "
                  f"(budget {budget:.3g}), {rounds} rounds, {lv['seconds']:.0f}s, {pr.measured} sets so far", flush=True)
        return row["levels"][f"{label}{levels[-1]}"]["kl"] <= budget

    if from_path is None:
        whole = pr.on.clone()
        reached = False
        if not start_top and f1_start != "whole":            # first VPD's answer at the targets alone, subcomponents only
            pr.on = whole & torch.isin(pr.t, torch.tensor(targets, device=pr.t.device))
            reached = search(("chunk4", "subcomponent"), "target ")
        f1_targets = pr.on.clone() if reached else None
        if not reached or f1_start == "both":
            pr.on = whole.clone()
            levels = LEVELS
            if start_top:                                    # the F1 search from VPD's top start_top, subcomponents only
                pr.on[:] = False
                pr.on[order[:start_top]] = True
                levels = ("chunk16", "chunk4", "subcomponent")
            search(levels, "")
        row["f1_from_targets"] = reached
        f1_on = pr.on.clone()
        if f1_targets is not None and f1_start == "both":   # the union of the two F1 circuits
            row["f1_sizes"] = [int(f1_targets.sum()), int(f1_on.sum())]
            f1_on |= f1_targets
        t1 = time.perf_counter()
        k = top_k(pr, order, empty, vpd)
        topk_on = torch.zeros_like(f1_on)
        topk_on[order[:k]] = True
        start = f1_on | topk_on
        path = joint_path(pr, start, empty, vpd)
    else:                                                    # steps 1-3 as saved: the same start set, its steps
        t1 = time.perf_counter()
        here = {i: index[nd] for i, nd in zip(from_path["start_index"], saved_start)}   # saved node id -> this one
        start = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
        start[list(here.values())] = True
        f1_on = torch.zeros_like(start)
        f1_on[[here[i] for i in from_path["f1_index"]] if "f1_index" in from_path
              else [index[(n, int(t), int(c))] for n, c, t in from_path["f1_nodes"]]] = True
        k = from_path["top_k"]
        topk_on = torch.zeros_like(start)
        topk_on[order[:k]] = True
        path = [{**r, "removed": [here[i] for i in r["removed"]]} for r in from_path["steps"]]
        row["from_path"] = {"start_nodes_added": added}
    sizes = [r["size"] for r in path]
    vs = []
    for sz in sizes:                                         # VPD's own order at every path size
        on = torch.zeros_like(f1_on)
        on[order[:sz]] = True
        vs.append(on)
    if from_path is None:
        vpd_rows = []
        for s0 in range(0, len(vs), pr.B):
            vpd_rows += measure(pr, torch.stack(vs[s0:s0 + pr.B]), empty, vpd)
    else:
        vpd_rows = [dict(r["vpd_top"]) for r in path]
    removed, ons = [], []
    on = start.clone()
    for r in path:
        on[torch.tensor(r["removed"], dtype=torch.long, device=on.device)] = False
        ons.append(on.clone())
        r["size_term"] = size_term(on)
        r["reward"] = (r["f1"] + r["n"]) / TERMS - SIZE_WEIGHT * r["size_term"]
        r["positions"] = int(torch.unique(pr.t[on]).numel())
    for r, v, on_v in zip(path, vpd_rows, vs):
        v["size_term"] = size_term(on_v)
        v["reward"] = (v["f1"] + v["n"]) / TERMS - SIZE_WEIGHT * v["size_term"]
        r["vpd_top"] = v
    best, row["pick"] = pick(path, "f1", cap=max_nodes)
    points = {min(range(len(path)), key=lambda j: abs(path[j]["size"] - want)) for want in SIZES}
    wired, edge_paths, t2 = {}, [], time.perf_counter()     # node step -> (its last edge step's connections, its row)

    wiring = Connector(ge, tx, pr, empty, vpd, size_term, B, robust=robust_pool, no_strand=no_strand)

    def wire(i):
        if i not in wired:
            kept, summary, steps, ep, wired_on = wiring.connect(ons[i], path[i]["n"])
            edge_paths.append({"step": i, "edges": [[index[w], "out" if r is None else index[r]] for w, r in ep.edges], "steps": steps})
            wired[i] = (kept, summary, wired_on)
        return wired[i][1]["f1_graph"] >= RECOVERED and wired[i][1]["n"] >= RECOVERED

    if edges and edges_at == "pick":                         # connections at step 4's pick alone
        wire(best)
    elif edges:
        cands = [i for i in range(best, len(path)) if path[i]["n"] >= RECOVERED]   # the pick and smaller points, N >= 0.9
        if row["pick"] == "goal" and wire(best):
            lo, hi = 0, len(cands) - 1                       # cands[0] is the pick
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if wire(cands[mid]) else (lo, mid - 1)
            best = cands[lo]
        wire(best)
        for i in points:
            if path[i]["size"] <= 64:
                wire(i)
    if tokens is not None:                                   # the answer within the budget: smaller path points until it fits
        row["budget_steps"] = 0
        while True:
            row["answer_tokens"] = tokens(components.render(circuit_json(pr, wired[best][2] if best in wired else ons[best],
                                                                         wired.get(best, (None,))[0])))
            if row["answer_tokens"] <= max_tokens or best == len(path) - 1:
                break
            best += 1
            row["budget_steps"] += 1
            if edges:
                wire(best)
        row["answer_nodes"] = path[best]["size"]
    final = (wired[best][2], wired[best][0]) if best in wired else (ons[best], None)
    if repair_pool is not None and edges:                   # the flip repair: the omitted lever's flips joined, connections again
        t3 = time.perf_counter()
        steps_, j, rule = wiring.repair(final[0], final[1], wired[best][1], repair_pool, max_nodes)
        final = (steps_[j]["on"], steps_[j]["kept"])
        row.update(repair=[{k_: v for k_, v in s_.items() if k_ not in ("on", "kept")} for s_ in steps_], repair_pick=j,
                   repair_rule=rule, repair_seconds=time.perf_counter() - t3)
    circuits = {"": circuit_json(pr, final[0], final[1]), "f1": circuit_json(pr, f1_on), f"vpd_top{k}": circuit_json(pr, topk_on)}
    if vpd_edges and edges:                                  # VPD's own top at the teacher circuit's node count, the same connection search
        kv = int(final[0].sum())
        on_v = torch.zeros_like(final[0])
        on_v[order[:kv]] = True
        got = measure(pr, on_v[None], empty, vpd)[0]
        kept_v, summary_v, _, _, on_v = wiring.connect(on_v, got["n"])
        circuits[f"vpd_top{kv}.edges"] = circuit_json(pr, on_v, kept_v)
        row["vpd_top_edges"] = {"size": kv, "f1_nodes": got["f1"], **summary_v}
    for i in points:
        circuits[f"n{path[i]['size']}"] = circuit_json(pr, wired[i][2] if i in wired else ons[i], wired.get(i, (None,))[0])
    f1_row = measure(pr, f1_on[None], empty, vpd)[0]
    row.update(f1_circuit={"size": int(f1_on.sum()), "positions": int(torch.unique(pr.t[f1_on]).numel()), **f1_row},
               top_k=k, start_size=int(start.sum()), path_seconds=time.perf_counter() - t1,
               best={k_: path[best][k_] for k_ in ("size", "positions", "f1", "n", "mean", "size_term", "reward", "kl_c", "kl_minus")},
               best_vpd_top=path[best]["vpd_top"], measured_sets=pr.measured, seconds=time.perf_counter() - t0)
    if edges:
        row.update(best_edges=wired[best][1], edge_points=[w[1] for w in wired.values()], edge_seconds=time.perf_counter() - t2)
    path_file = {"start_nodes": [[n, c, t] for n, t, c in (pr.nodes[i] for i in start.nonzero().flatten().tolist())],
                 "start_index": start.nonzero().flatten().tolist(), "steps": path,
                 "f1_index": f1_on.nonzero().flatten().tolist(), "top_k": k}
    if edges:
        path_file["edge_paths"] = edge_paths
    return circuits, row, path_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="this process takes the first n texts offset, offset + stride, ...")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vpd", type=Path, help="VPD's answers and orders saved by --save-vpd (no causal-importance network here)")
    ap.add_argument("--save-vpd", type=Path, help="save VPD's answers and orders on the texts to this file and stop")
    ap.add_argument("--labels", type=Path, help="locality JSONL (task, local)")
    ap.add_argument("--start-top", type=int, default=0, help="F1 search from VPD's top K at subcomponent level (0: from VPD's whole answer)")
    ap.add_argument("--f1-start", choices=("targets", "whole", "both"), default="both",
                    help="the F1 circuit: from VPD's answer at the targets when that reaches F1 0.9, else from its whole answer "
                         "(targets); from its whole answer only (whole); the union of both searches' circuits (both)")
    ap.add_argument("--edges", action="store_true", help="prune the teacher circuit's connections (graph semantics)")
    ap.add_argument("--no-strand", action="store_true",
                    help="with --edges, every non-query/key node keeps a stated connection (its most important; one the model connects "
                         "to no other node is dropped): the stated graph is the claim, a listed node needs a stated role")
    ap.add_argument("--edge-objective", choices=("graph", "robust"), default="graph",
                    help="with --edges, connection removals keep graph F1 >= 0.9 (graph), and also F1 under REPAIR_M of the omitted "
                         "lever's flips >= min(0.9, every connection's) (robust: goal_eval.adversary_flips, pool POOL)")
    ap.add_argument("--vpd-edges", action="store_true", help="with --edges, also VPD's top k at the teacher circuit's node count k with "
                    "its connections by the same search: <task>.vpd_top<k>.edges.json")
    ap.add_argument("--repair", action="store_true", help="with --edges, the flip repair after the pick (goal_eval.adversary_flips' "
                    "omitted lever at REPAIR_M, its flips joined and the connections pruned again while that improves; pool POOL)")
    ap.add_argument("--edges-at", choices=("pick", "search"), default="pick",
                    help="connections at step 4's pick alone (pick), or the smallest point with N >= 0.9 whose connections reach "
                         "graph F1 >= 0.9 by bisection, and the points nearest 16, 32 and 64 (search)")
    ap.add_argument("--from-path", type=Path, help="a directory of this script's outputs: each text's saved path in place of steps 1-3")
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES, help="the pick among path points of at most this many nodes")
    ap.add_argument("--max-tokens", type=int, default=components.ANSWER_BUDGET,
                    help="the teacher circuit's answer at most this many tokens of --tokenizer (0: not checked)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="the explainer's tokenizer (Hugging Face hub)")
    ap.add_argument("--tasks", type=Path, help="task ids, one per line, in place of the split's first n texts")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.tasks:
        texts = [native.TEXTS / "vpd4l" / f"{t}.json" for t in args.tasks.read_text().split()]
    else:
        texts = native.tasks(args.split)[:args.n]
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    tokens = None
    if args.max_tokens and not args.save_vpd:
        from tokenizers import Tokenizer

        tk = Tokenizer.from_pretrained(args.tokenizer)
        tokens = lambda text: len(tk.encode(text, add_special_tokens=False).ids)  # noqa: E731
    if args.save_vpd:
        saved = {}
        for path in texts:
            ids, targets = native.text(path)
            m = ge.vpd_answer(ids, targets)
            saved[path.stem] = {"answer": {n: tuple(x.to(torch.int32).cpu() for x in v.nonzero().T) for n, v in m.items()},
                                "order": ge.nat._vpd_order(ids, targets)}
        torch.save(saved, args.save_vpd)
        return
    vpd = torch.load(args.vpd) if args.vpd else {}
    labels = {}
    if args.labels and args.labels.exists():
        for line in args.labels.read_text().splitlines():
            r = json.loads(line)
            labels[r["task"]] = r.get("local")
    args.out.mkdir(parents=True, exist_ok=True)
    pool = torch.load(POOL) if args.repair or args.edge_objective == "robust" else None
    for path in texts[args.offset::args.stride]:
        if (args.out / f"{path.stem}.json").exists():
            continue
        from_path = None
        if args.from_path:
            source = args.from_path / f"{path.stem}.path.json"
            if not source.exists():
                continue
            from_path = json.loads(source.read_text())
            if "f1_index" not in from_path:                  # path files written before f1_index and top_k
                from_path["f1_nodes"] = [x for x in json.loads((args.from_path / f"{path.stem}.f1.json").read_text())["nodes"] if x[0] != "tok"]
                from_path["top_k"] = int(next(args.from_path.glob(f"{path.stem}.vpd_top*.json")).stem.split(".vpd_top")[1])
        ids, targets = native.text(path)
        saved = vpd.get(path.stem, {})
        circuits, row, path_file = teach(ge, ids, targets, args.batch, saved.get("answer"), saved.get("order"), args.start_top, args.f1_start,
                                         args.edges, from_path, args.edges_at, args.max_nodes, tokens, args.max_tokens,
                                         pool if args.repair else None, args.vpd_edges, pool if args.edge_objective == "robust" else None,
                                         args.no_strand)
        for name, c in circuits.items():
            (args.out / f"{path.stem}{'.' + name if name else ''}.json").write_text(json.dumps(c))
        (args.out / f"{path.stem}.path.json").write_text(json.dumps(path_file))
        row = {"task": path.stem, "local": labels.get(path.stem), **row}
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        b, v, f = row["best"], row["best_vpd_top"], row["f1_circuit"]
        print(f"{path.stem}: F1 circuit {f['size']} at {f['positions']} positions (F1 {f['f1']:.3f}, N {f['n']:.3f}); top-k {row['top_k']}; "
              f"path from {row['start_size']}: pick ({row['pick']}) {b['size']} at {b['positions']} positions F1 {b['f1']:.3f} N {b['n']:.3f} reward {b['reward']:.3f} "
              f"vs VPD top-{b['size']} F1 {v['f1']:.3f} N {v['n']:.3f} reward {v['reward']:.3f}; {row['measured_sets']} sets, {row['seconds']:.0f}s"
              + (f"; with connections {row['best_edges']['edges']} of {row['best_edges']['all_edges']}, graph F1 "
                 f"{row['best_edges']['f1_graph']:.3f}" if "best_edges" in row else ""),
              flush=True)


if __name__ == "__main__":
    main()
