"""Backup expansion (#2951 graph oracle): the subcomponents outside a circuit through which the real model routes
around a removal of the circuit's nodes (its compensators), and the circuit expanded by them.

The finding it answers: goal_eval's node family (s_nodes, inframove's 29d5494: per draw d of the circuit's nodes,
KL(model less d || circuit less d) bounded by W = KL(model less d || model), VPD's whole answer less d the reference,
pooled by W) scores the influential start sets ~0.52 at the median of text9000-9019, the circuit's claimed removal
effect about twice the model's (overstate_nodes). When circuit nodes are removed, the real model's subcomponents
outside the circuit respond to the changed stream and make up part of the loss; in the circuit's run (C's semantics:
everything outside off, the remainders live) nothing outside responds.

On one text, for a circuit given as a node set (goal_eval's complete graph, blocks allowed):
  draws: the backup family's own removals of the circuit's nodes (draw j removes each with probability (j + 0.5) /
     draws, at least one, as goal_eval's node family draws), seeded apart from goal_eval's draws: the search never sees
     the evaluation's draws.
  score: per candidate expansion X, on those draws, s_search = goal_eval's s_nodes form with the circuit plus X less
     d (X never removed: the question is whether the circuit's response to losing its own nodes is the model's), F1
     (the point run, goal_eval's log scale), J = their harmonic mean (C's combination of its terms), overstate (the
     claimed removal effect over the model's) and the expanded answer's description bits (goal_eval.size_bits, a block
     at its active contents).
  responses: where the real model routes around a removal: per (site, position), the change of the summed output of
     the subcomponents outside the circuit there, |sum_c (a_c(model less d) - a_c(model)) U_c|, over the draws, beside
     that output's size in the model; a measurement of the model, no membership decision.
  screen: the gradient of J in every subcomponent's mask at the current expansion (off outside it), summed over a
     site's subcomponents at a position for a block (the first-order effect of switching the block on); it orders
     candidates, never admits one.
  expand: greedy over blocks (goal_eval.BLOCK: a site's whole output at a position, every subcomponent and the
     remainder, which C's semantics runs as the model's own weight there; a block subsumes the circuit's
     subcomponents at its site and position): candidates every site at the circuit's positions (alone, and a site at
     all of them together) and the screen's best at the other positions from the circuit's first to the last target;
     every candidate's addition measured (score), the one with the largest gain in J per bit taken (a candidate that
     costs no bits first, by gain), while the gain is at least min_gain; later rounds re-measure the best candidates
     of the last (lazy greedy) and the screen's new best. Every membership decision is measured.
  prune: the backward search, from every site at the circuit's positions as blocks, each block (and site group)
     measured back out while that raises J (blocks interact: a single block's gain understates a set's).
  refine: a taken block replaced by the circuit's subcomponents there plus the outside subcomponents of the block in
     the screen's order, the shortest prefix measured to keep at least keep x the block's gain in J, taken where it
     costs fewer bits than the block.

  expand_circuit(ge, tx, graph, ...) -> (expanded graph, record): the teacher's call (pymove's teacher.py, after the
     node search and before its connections; see expand_circuit).
  backup.py run --split heldout --n 20 --src DIR --out OUT: expand each text's circuit in DIR, writing OUT/<set>/<task>.json
     for the path points, the refined circuit and the equal-size trivial baselines, and OUT/summary.jsonl.
  backup.py eval --out OUT: C's families in the reward's form on the eval split's draws (goal_eval.GoalEval.fidelity) for
     the source set (the start), OUT's sets and the sets derived_sets builds from run's summary (equal-count random
     blocks for the forward greedy, VPD's order beyond the circuit at the blocks' added VPD-active count, and the
     forward blocks replaced by their shortest subcomponent prefixes holding 50% and 90% of their gain), every set by
     the same families (the connection-cut family left out: see evaluate), then a table with phi against each text's
     start row and size_bits (goal_eval's, blocks priced at their VPD-active contents since a699a8b).
  backup.py repair --out OUT: Backup.repair per text (how much of the circuit's change at the prediction the outside
     subcomponents cancel under the removals, by site, and the share each of OUT's sets holds), OUT/repair.jsonl.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
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
import goal_eval  # noqa: E402
import native  # noqa: E402

BLOCK = goal_eval.BLOCK
SEARCH_DRAWS = 32       # the search's removal draws per circuit (16 left s 0.44 vs 0.52 between two seeds on text9000)
ROWS = 4                # masked runs per batch (every subcomponent's column: ~60 MB of masks per row at T 400)
SCREEN_TOP = 24         # blocks at positions outside the circuit's measured per round (the screen's best)
LAZY = 12               # last round's best candidates re-measured per round
MIN_GAIN = 0.005        # the least gain in J a step takes
MAX_STEPS = 12          # blocks taken at most
KEEP = 0.9              # refine: the share of a block's gain in J a prefix keeps
EVAL_DRAWS = 32         # run's fresh draws of the circuit's own nodes for its sets (eval_units)
EPS = 1e-12


def _harmonic(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return 0.0 if min(a, b) <= 0 else 2.0 / (1.0 / a + 1.0 / b)


def split_graph(g: native.Graph) -> tuple[frozenset, frozenset]:
    """(blocks {(site, position)}, listed subcomponents {(site, position, index)}) of a node set."""
    return (frozenset((n, t) for n, t, c in g.node_set() if c == BLOCK),
            frozenset(nd for nd in g.node_set() if nd[2] != BLOCK))


def expanded_graph(blocks0: frozenset, subs0: frozenset, blocks=frozenset(), extra=frozenset()) -> native.Graph:
    """A circuit (blocks0, subs0) expanded: a node set, each block one node (BLOCK) holding the listed subcomponents at
    its site and position, the extra subcomponents listed."""
    bl = frozenset(blocks0) | frozenset(map(tuple, blocks))
    subs = [nd for nd in frozenset(subs0) | frozenset(map(tuple, extra)) if (nd[0], nd[1]) not in bl]
    return native.Graph(sorted(subs) + sorted((n, t, BLOCK) for n, t in bl), complete=True)


def counts(g: native.Graph, C: dict, active: dict) -> dict:
    """The subcomponents a node set claims (blocks as their sites' every subcomponent) and how many of them VPD's
    answer has on, with the block count."""
    blocks, subs = split_graph(g)
    n = len(subs) + sum(C[s] for s, _ in blocks)
    act = sum(bool(active[s][t, c]) for s, t, c in subs) + sum(int(active[s][t].sum()) for s, t in blocks)
    return {"blocks": len(blocks), "subcomponents": n, "active": act}


class Backup:
    """One text and one circuit (a node set): the search's draws, the model's and VPD's answer's runs under them, and
    the measurements above. graph: a goal_eval node set (complete; blocks allowed, stranded nodes dropped first)."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, graph: native.Graph, draws: int = SEARCH_DRAWS,
                 rows: int = ROWS, family: str = "backup", block_units: bool = True):
        import engine

        nat = ge.nat
        self.ge, self.tx, self.nat, self.rows = ge, tx, nat, rows
        g = goal_eval.strand(graph, tx.targets, nat.C)
        if not g.complete:
            raise ValueError("backup expansion takes a node set (a complete graph): state connections after it")
        self.blocks0 = frozenset((n, t) for n, t, c in g.node_set() if c == BLOCK)
        self.subs0 = frozenset(nd for nd in g.node_set() if nd[2] != BLOCK)
        self.nodes = sorted(goal_eval.expand_blocks(g, nat.C).node_set())  # the circuit's subcomponents
        self.positions = sorted({t for _, t, _ in self.nodes})
        # the removal units: the circuit's subcomponents, a block of the circuit's one unit (block_units: an answer's
        # block node is one node; else its subcomponents, as goal_eval's node family expands it)
        units = sorted(self.subs0) + sorted((n, t, BLOCK) for n, t in self.blocks0) if block_units else self.nodes
        self.T, self.P, self.S = tx.T, nat.positions(tx.targets), sum(nat.C.values())
        self.E = engine.Engine(nat, tx.ids, tx.targets, live=True)       # every column: blocks switch whole sites on
        self.e0, self.v0 = ge._bounds(tx, "live")
        key = int(hashlib.sha256(repr((family, units)).encode()).hexdigest()[:15], 16)
        rng = np.random.default_rng([tx.seed, key])
        self.drops = []
        for j in range(draws):
            d = [nd for nd in units if rng.random() < (j + 0.5) / draws]
            self.drops.append(d or [units[int(rng.integers(len(units)))]])
        dev = nat.dev
        self._drop_idx = []                                          # per draw {site: (positions, columns)}
        for d in self.drops:
            by = {}
            for n, t, c in d:
                by.setdefault(n, ([], []))
                cs = range(nat.C[n]) if c == BLOCK else [c]
                by[n][0].extend([t] * len(cs))
                by[n][1].extend(cs)
            self._drop_idx.append({n: (torch.tensor(t, device=dev), torch.tensor(c, device=dev)) for n, (t, c) in by.items()})
        ones = {n: torch.ones(self.T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
        self.lm = self._run([(ones, j) for j in range(draws)])           # the model less d
        lv = self._run([(tx.vpd, j) for j in range(draws)])               # VPD's whole answer less d
        self.W = carry.kl(self.lm, tx.ref.expand_as(self.lm)).tolist()
        self.KV = carry.kl(self.lm, lv).tolist()
        del lv
        native.free(dev)
        self.measured = 0

    # ---- candidates as masks and graphs

    def masks(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """{site: bool [T, C]}: the circuit, its blocks and the expansion's (blocks, extra subcomponents) on."""
        nat = self.nat
        m = {n: torch.zeros(self.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, t, c in self.subs0 | frozenset(extra):
            m[n][t, c] = True
        for n, t in self.blocks0 | frozenset(blocks):
            m[n][t] = True
        return m

    def graph(self, blocks=frozenset(), extra=frozenset()) -> native.Graph:
        """The expanded answer (expanded_graph)."""
        return expanded_graph(self.blocks0, self.subs0, blocks, extra)

    def bits(self, blocks=frozenset(), extra=frozenset()) -> float:
        """goal_eval.size_bits of the expanded answer (a block at its active contents)."""
        return goal_eval.size_bits(self.graph(blocks, extra), self.P, self.S, self.tx.targets, active=self.tx.vpd)

    # ---- runs

    def _row(self, m: dict, j: int | None, dtype=torch.float32) -> dict:
        """One row's float masks: m with draw j's nodes off (None: no draw)."""
        out = {n: x.to(dtype) for n, x in m.items()}
        if j is not None:
            for n, (t, c) in self._drop_idx[j].items():
                out[n][t, c] = 0.0
        return out

    def _run(self, rows: list) -> torch.Tensor:
        """log q at the targets [len(rows), targets, V] of masked runs, rows (masks, draw or None) in batches."""
        out = []
        with torch.no_grad():
            for s0 in range(0, len(rows), self.rows):
                part = [self._row(m, j) for m, j in rows[s0:s0 + self.rows]]
                out.append(self.E.log_probs({n: torch.stack([r[n] for r in part]) for n in self.nat.names}))
                del part
        return torch.cat(out) if out else torch.zeros(0)

    def _fields(self, lq: torch.Tensor) -> dict:
        """The score fields from one candidate's runs: the point row first, then the draws'."""
        point = float(carry.kl(self.tx.ref, lq[:1]))
        kc = carry.kl(self.lm, lq[1:]).tolist()
        claim = carry.kl(self.tx.ref.expand_as(lq[1:]), lq[1:]).tolist()
        draws = [[j, a, b, w] for j, (a, b, w) in enumerate(zip(kc, self.KV, self.W))]
        s = goal_eval._pooled("weighted", draws, self.e0)
        f1 = goal_eval._fid(self.e0, point, self.v0)
        return {"s_search": s, "f1": f1, "j": _harmonic(s, f1), "kl_point": point, "kl_nodes_c": float(np.mean(kc)),
                "overstate": float(np.mean(claim)) / max(float(np.mean(self.W)), EPS)}

    def score(self, cands: list) -> list[dict]:
        """Per candidate (blocks, extra subcomponents): s_search, f1, j, kl_point, kl_nodes_c, overstate and bits."""
        D = len(self.drops)
        out = []
        for blocks, extra in cands:
            m = self.masks(blocks, extra)
            lq = self._run([(m, None)] + [(m, j) for j in range(D)])
            out.append({**self._fields(lq), "bits": self.bits(blocks, extra)})
            self.measured += 1
            del m, lq
        native.free(self.nat.dev)
        return out

    # ---- where the model routes around

    @torch.no_grad()
    def responses(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """The real model's response to the draws: per site [T] tensors (CPU) "outside", the mean over the draws of
        |sum over the subcomponents outside the expansion of (a(model less d) - a(model)) U| at each position, "inside"
        the same for the expansion's own (those the draw keeps), and "size", |sum over every subcomponent of a U| in
        the model (the site's decomposed output there)."""
        nat, E, dev = self.nat, self.E, self.nat.dev
        m = self.masks(blocks, extra)
        ones = {n: torch.ones(self.T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
        got: dict = {}
        orig = E._site

        def site(n, h, mask, rec=None):  # the subcomponents' activations, split by membership
            V, U, _ = E._w(n)
            a = h.to(V.dtype) @ V
            inside = m[n].to(a.dtype)
            got[n] = (a * (1 - inside), a * inside)
            return orig(n, h, mask, rec)

        E._site = site
        try:
            E.log_probs({n: ones[n].float()[None] for n in nat.names})
            base = {n: (o[0].clone(), i[0].clone()) for n, (o, i) in got.items()}
            out = {"outside": {n: torch.zeros(self.T) for n in nat.names}, "inside": {n: torch.zeros(self.T) for n in nat.names},
                   "size": {n: ((base[n][0] + base[n][1]) @ E.U[n]).norm(dim=-1).cpu() for n in nat.names}}
            D = len(self.drops)
            for s0 in range(0, D, self.rows):
                js = list(range(s0, min(D, s0 + self.rows)))
                part = [self._row(ones, j) for j in js]
                E.log_probs({n: torch.stack([r[n] for r in part]) for n in nat.names})
                for n in nat.names:
                    o, i = got[n]
                    out["outside"][n] += ((o - base[n][0]) @ E.U[n]).norm(dim=-1).sum(0).cpu() / D
                    keep = torch.stack([r[n] for r in part]) * m[n].float()   # the inside the draw keeps
                    out["inside"][n] += (((i - base[n][1]) * keep) @ E.U[n]).norm(dim=-1).sum(0).cpu() / D
        finally:
            E._site = orig
            native.free(dev)
        return out

    @torch.no_grad()
    def repair(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """Self-repair at the prediction, a measurement of the model: per draw, the change of the attention and MLP
        outputs written into the residual stream at the targets between the model and the model less the draw, split
        into the circuit's subcomponents' (the removed ones off, the kept ones changed: dc) and the outside ones' (do);
        r = -<do, dc> / |dc|^2, the share of the circuit's change that the outside cancels along it (0: no repair, 1:
        the circuit's direct change cancelled), split by site (the shares sum to r) and into the part an expansion
        (blocks, extra) holds ("held"); cos(do, -dc). Pooled over the draws by W (the node family's weights). The
        remainders' change (live in every run, C's semantics) is in neither side. {"r", "cos", "held", "sites": {site:
        share}, "draws": [[W, r, cos, held]]}."""
        import mech

        nat, E, dev = self.nat, self.E, self.nat.dev
        m0, mx = self.masks(), self.masks(blocks, extra)
        tg = torch.tensor(self.tx.targets, device=dev)
        writers = [n for n in nat.names if native._name_layer_kind(n)[1] in mech.RESID_WRITERS]
        got: dict = {}
        orig = E._site

        def site(n, h, mask, rec=None):
            if n in writers:
                V, U, _ = E._w(n)
                a = (h.to(V.dtype) @ V)[:, tg] * mask.to(V.dtype)[:, tg]          # [B, targets, C] as written
                inside, held = m0[n][tg].to(a.dtype), (mx[n][tg] & ~m0[n][tg]).to(a.dtype)
                got[n] = ((a * inside) @ U, (a * (1 - inside)) @ U, (a * held) @ U)
            return orig(n, h, mask, rec)

        ones = {n: torch.ones(self.T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
        E._site = site
        try:
            E.log_probs({n: ones[n].float()[None] for n in nat.names})
            base = {n: tuple(x[0] for x in got[n]) for n in writers}
            draws, D = [], len(self.drops)
            for s0 in range(0, D, self.rows):
                js = list(range(s0, min(D, s0 + self.rows)))
                E.log_probs({n: torch.stack([self._row(ones, j)[n] for j in js]) for n in nat.names})
                dc = sum(got[n][0] - base[n][0] for n in writers)                 # [B, targets, d]
                per = {n: got[n][1] - base[n][1] for n in writers}
                held = sum(got[n][2] - base[n][2] for n in writers)
                do = sum(per.values())
                for b, j in enumerate(js):
                    c2 = float((dc[b] ** 2).sum())
                    if c2 <= EPS:
                        continue
                    sh = {n: -float((per[n][b] * dc[b]).sum()) / c2 for n in writers}
                    cos = -float((do[b] * dc[b]).sum()) / max(float(do[b].norm() * dc[b].norm()), EPS)
                    draws.append([self.W[j], sum(sh.values()), cos, -float((held[b] * dc[b]).sum()) / c2, sh])
        finally:
            E._site = orig
            native.free(dev)
        wsum = sum(d[0] for d in draws) or 1.0
        pool = lambda i: sum(d[0] * d[i] for d in draws) / wsum  # noqa: E731
        return {"r": pool(1), "cos": pool(2), "held": pool(3),
                "sites": {n: sum(d[0] * d[4][n] for d in draws) / wsum for n in writers},
                "draws": [d[:4] for d in draws]}

    @torch.no_grad()
    def frozen(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """The outside's response as a counterfactual on the model: per draw, the model less the draw with every
        subcomponent outside the circuit (and outside the expansion blocks, extra) held at its activation in the model
        on the text (the remainders live, as in C's semantics), against the model less the draw. overstate_frozen =
        mean KL(model || frozen less d) / mean W (W = KL(model less d || model); goal_eval's overstate form): the
        overstate when the outside cannot respond but still writes what it writes on the text, beside overstate_model,
        the model's own in that form; kl_frozen, mean KL(model less d || frozen less d). Near the circuit's overstate:
        the outside's response is the gap; near the model's: the gap is the outside's absence itself."""
        nat, E, dev = self.nat, self.E, self.nat.dev
        mx = self.masks(blocks, extra)
        clean: dict = {}
        orig = E._site

        def record(n, h, mask, rec=None):
            V = E._w(n)[0]
            clean[n] = (h.to(V.dtype) @ V)[0]
            return orig(n, h, mask, rec)

        def held(n, h, mask, rec=None):
            V, U, Rw = E._w(n)
            hh = h.to(V.dtype)
            a = torch.where(mx[n][None], hh @ V, clean[n][None].to(V.dtype))
            out = (a * mask.to(a.dtype)) @ U
            return out + hh @ Rw if E.fixed else out

        ones = {n: torch.ones(self.T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
        out = []
        try:
            E._site = record
            E.log_probs({n: ones[n].float()[None] for n in nat.names})
            E._site = held
            for s0 in range(0, len(self.drops), self.rows):
                js = list(range(s0, min(len(self.drops), s0 + self.rows)))
                out.append(E.log_probs({n: torch.stack([self._row(ones, j)[n] for j in js]) for n in nat.names}))
        finally:
            E._site = orig
            native.free(dev)
        lf = torch.cat(out)
        f = carry.kl(self.tx.ref.expand_as(lf), lf).tolist()
        err = carry.kl(self.lm, lf).tolist()
        w = max(float(np.mean(self.W)), EPS)
        return {"overstate_frozen": float(np.mean(f)) / w, "overstate_model": float(np.mean(self._claim_model())) / w,
                "kl_frozen": float(np.mean(err)), "w": float(np.mean(self.W)), "draws": [[w_, a, b] for w_, a, b in zip(self.W, f, err)]}

    def _claim_model(self) -> list:
        """KL(model || model less d) per draw: the model's own removal effect in the orientation of the circuit's claim
        (goal_eval's overstate divides KL(model || circuit less d) by W = KL(model less d || model)), so the model's
        own overstate is this over W, near 1, not 1."""
        return carry.kl(self.tx.ref.expand_as(self.lm), self.lm).tolist()

    @torch.no_grad()
    def norm_frozen(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """The circuit's runs (C's semantics: the circuit, its expansion blocks and extra on, everything else off) with
        every RMS norm's scale held at the model's on the text (vpd_model.Target.norm_parts: each norm reads its
        input with the statistics of the model's input there), at the point and under the search's draws, and the
        model's runs less each draw the same way: how much of the circuit's claimed removal effect goes through the
        norms' response to the removal. (Measured on dev text9000-9019's influential starts: held norms bring the
        circuit's overstate from 2.04 to 0.83, the model's own being 1.05, at a point-F1 cost, 0.79 -> 0.64; the
        circuit's stream is not smaller than the model's, stream_ratio ~1 at the targets' norms and 1.1-7.6 at the
        final norm, so it is not the norm scaling a smaller stream up.) {"circuit": score fields (s_search, f1, j, kl_point,
        overstate against the model's real W), "model": {"overstate": mean KL(model || model less d, norms held) / W,
        "overstate_unheld": the model's own in that form}}."""
        nat, E, dev, tg = self.nat, self.E, self.nat.dev, self.nat.target
        clean: dict = {}
        orig = E._steps

        def record(fused):
            norm, act = orig(fused)

            def nrm(x, j):
                clean[j] = x.detach()
                return norm(x, j)
            return nrm, act

        def held(fused):
            return (lambda x, j: tg.norm_parts(x, clean[j], j)), orig(fused)[1]

        ones = {n: torch.ones(self.T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
        D = len(self.drops)
        try:
            E._steps = record
            E.log_probs({n: ones[n].float()[None] for n in nat.names})
            E._steps = held
            m = self.masks(blocks, extra)
            lq = self._run([(m, None)] + [(m, j) for j in range(D)])
            lm = self._run([(ones, j) for j in range(D)])
        finally:
            E._steps = orig
            native.free(dev)
        f = self._fields(lq)
        model = carry.kl(self.tx.ref.expand_as(lm), lm).tolist()
        w = max(float(np.mean(self.W)), EPS)
        return {"circuit": f, "model": {"overstate": float(np.mean(model)) / w, "overstate_unheld": float(np.mean(self._claim_model())) / w}}

    @torch.no_grad()
    def stream_ratio(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """The circuit's residual stream against the model's at the point (no removal), at each RMS norm's input:
        rms(circuit's) / rms(model's), the median over the norms (every layer's two and the final one) at the targets
        ("targets") and over every norm and circuit position ("positions"), and the final norm's at the targets
        ("final"). Below 1: the circuit's stream is smaller, so the norms scale its writes up."""
        nat, E, dev = self.nat, self.E, self.nat.dev
        got: dict = {}
        orig = E._steps

        def record(fused):
            norm, act = orig(fused)

            def nrm(x, j):
                got.setdefault(j, []).append(x.detach().float().pow(2).mean(-1).sqrt())   # [B, T]
                return norm(x, j)
            return nrm, act

        ones = {n: torch.ones(self.T, nat.C[n], device=dev) for n in nat.names}
        m = {n: x.float() for n, x in self.masks(blocks, extra).items()}
        try:
            E._steps = record
            E.log_probs({n: torch.stack([ones[n], m[n]]) for n in nat.names})
        finally:
            E._steps = orig
            native.free(dev)
        tg, P = list(self.tx.targets), self.positions
        r = {j: (v[0][1] / v[0][0].clamp_min(EPS)).cpu() for j, v in got.items()}    # [T] per norm
        return {"targets": float(np.median([float(x[t]) for x in r.values() for t in tg])),
                "positions": float(np.median([float(x[t]) for x in r.values() for t in P])),
                "final": float(np.median([float(r[-1][t]) for t in tg]))}

    # ---- the screen

    def screen(self, blocks=frozenset(), extra=frozenset()) -> dict:
        """The gradient of -J (up to a positive factor) in every subcomponent's mask at the expansion, per site [T, C]
        (CPU): a negative entry's subcomponent switched on raises J to first order. J = harmonic(s, F1) with s pooled
        over the draws as s_nodes (each draw's log-scale term, weighted by W; draws already at 1 or VPD's side no nearer
        than W left out) and F1 the point run's; the factor J^2 / 2 dropped."""
        nat, E, dev = self.nat, self.E, self.nat.dev
        m = self.masks(blocks, extra)
        D = len(self.drops)
        lq = self._run([(m, None)] + [(m, j) for j in range(D)])
        f = self._fields(lq)
        s, f1 = max(f["s_search"] or 0.0, 1e-3), max(f["f1"] or 0.0, 1e-3)
        kc = carry.kl(self.lm, lq[1:]).tolist()
        del lq
        wsum = sum(w for w, kv in zip(self.W, self.KV) if kv < w) or 1.0
        coef = [0.0]
        for kcj, kv, w in zip(kc, self.KV, self.W):
            ok = kv < w and kcj > kv                                 # a draw whose term can still rise
            coef.append(w / (wsum * math.log(w / max(kv, EPS))) / s ** 2 if ok else 0.0)
        if f1 < 1.0 and self.v0 < self.e0:
            coef[0] = 1.0 / (math.log(self.e0 / max(self.v0, EPS)) * f1 ** 2)
        x = {n: torch.zeros(1, self.T, nat.C[n], device=dev, requires_grad=True) for n in nat.names}
        rows = [(None, 0)] + [(j, j + 1) for j in range(D)]
        rows = [r for r in rows if coef[r[1]] != 0.0]
        for s0 in range(0, len(rows), self.rows):
            part = rows[s0:s0 + self.rows]
            base = [self._row(m, j) for j, _ in part]
            mk = {n: torch.stack([b[n] for b in base]) + x[n] for n in nat.names}
            lqp = self.E.log_probs(mk)
            refs = torch.cat([self.tx.ref if j is None else self.lm[j:j + 1] for j, _ in part])
            kl = carry.kl(refs, lqp).clamp_min(EPS)
            loss = (torch.tensor([coef[i] for _, i in part], device=dev) * kl.log()).sum()
            loss.backward()
            del base, mk, lqp, kl, loss
        native.free(dev)
        return {n: x[n].grad[0].detach().cpu() if x[n].grad is not None else torch.zeros(self.T, nat.C[n]) for n in nat.names}

    def block_screen(self, grad: dict, blocks=frozenset(), extra=frozenset()) -> list[tuple]:
        """Blocks ordered by the screen (most negative first): [(gradient sum over the block's outside subcomponents,
        site, position)] for positions from the circuit's first to the last target, blocks already taken left out."""
        m = self.masks(blocks, extra)
        lo, hi = min(self.positions), max(self.tx.targets)
        bl = self.blocks0 | frozenset(blocks)
        out = []
        for n in self.nat.names:
            gsum = (grad[n] * (~m[n]).float().cpu()).sum(-1)
            for t in range(lo, hi + 1):
                if (n, t) not in bl:
                    out.append((float(gsum[t]), n, t))
        return sorted(out)

    # ---- the greedy expansion

    def expand(self, max_steps: int = MAX_STEPS, min_gain: float = MIN_GAIN, screen_top: int = SCREEN_TOP,
               lazy: int = LAZY, log=None) -> dict:
        """The greedy block expansion (module docstring). A candidate is one block or, with the circuit at several
        positions, a site's blocks at every circuit position together (a site's routes at different positions act
        jointly: on text9000 h.3.mlp.c_fc at both positions took s from 0.52 to 0.95, at the target alone to 0.60).
        {"start": the circuit's score, "steps": [per step the blocks added ("add"), the gain, bits added, the
        expansion's score], "rounds": [per round each candidate measured, (blocks, gain in J, bits added, the best
        response rank among its blocks)], "blocks": every block taken, in order, "step_blocks": per step the blocks
        taken through it}."""
        t0 = time.perf_counter()
        resp = self.responses()
        rank = {}                                                    # (site, position) -> its response's rank
        flat = sorted(((float(resp["outside"][n][t]), n, t) for n in self.nat.names for t in range(self.T)), reverse=True)
        for i, (_, n, t) in enumerate(flat):
            rank[(n, t)] = i
        cur = self.score([(frozenset(), frozenset())])[0]
        rec = {"start": cur, "steps": [], "rounds": [], "blocks": [], "step_blocks": [],
               "responses_top": [[n, t, v, float(resp["size"][n][t])] for v, n, t in flat[:48]]}
        taken: frozenset = frozenset()
        last: dict = {}                                              # candidate -> its last measured gain
        singles = [((n, t),) for t in self.positions for n in self.nat.names if (n, t) not in self.blocks0]
        groups = [tuple((n, t) for t in self.positions if (n, t) not in self.blocks0) for n in self.nat.names] \
            if len(self.positions) > 1 else []
        groups = [c for c in groups if len(c) > 1]
        for step in range(max_steps):
            grad = self.screen(taken)
            far = [((n, t),) for _, n, t in self.block_screen(grad, taken) if t not in set(self.positions)][:screen_top]
            if step == 0:
                cands = list(dict.fromkeys(singles + groups + far))
            else:
                best = sorted((c for c in last if not set(c) <= taken), key=lambda c: -last[c])[:lazy]
                cands = list(dict.fromkeys(best + far))
            cands = [c for c in cands if not set(c) <= taken]
            if not cands:
                break
            got = self.score([(taken | set(c), frozenset()) for c in cands])
            rows = []
            for c, r in zip(cands, got):
                gain = (r["j"] or 0.0) - (cur["j"] or 0.0)
                last[c] = gain
                rows.append((c, gain, r["bits"] - cur["bits"], r))
            rec["rounds"].append([[[list(b) for b in c], g, db, min(rank.get(b, 10 ** 9) for b in c)] for c, g, db, _ in rows])

            def key(x):                                              # free first (by gain), then gain per bit
                _, g, db, _ = x
                return (1, g) if db <= 0 else (0, g / db)
            ok = [x for x in rows if x[1] >= min_gain]
            if not ok:
                break
            c, g, db, r = max(ok, key=key)
            add = [b for b in c if b not in taken]
            taken = taken | set(c)
            cur = r
            rec["steps"].append({"add": [list(b) for b in add], "gain": g, "bits_added": db,
                                 "response_rank": min(rank.get(b, 10 ** 9) for b in add), **r})
            rec["blocks"] += [list(b) for b in add]
            rec["step_blocks"].append([list(b) for b in rec["blocks"]])
            if log:
                log(f"    step {step}: {' + '.join(f'{b[0]}@{b[1]}' for b in add)} J {r['j']:.3f} (s {r['s_search']:.3f}, "
                    f"F1 {r['f1']:.3f}), {db:+.0f} bits, overstate {r['overstate']:.2f}, {len(cands)} measured")
        rec["seconds"] = time.perf_counter() - t0
        return rec

    def prune(self, blocks: list, max_steps: int = 4 * MAX_STEPS, min_gain: float = MIN_GAIN, lazy: int = 2 * LAZY,
              log=None) -> dict:
        """The backward search from a block set (e.g. every site at the circuit's positions): each block, and each
        site's blocks together, measured back out (the circuit's own subcomponents there stay), the removal with the
        largest gain in J taken while it gains at least min_gain (a removal that also saves bits first among equal
        gains: gain per bit saved never favours keeping a block); later rounds re-measure the last round's best and
        every site group. {"start", "steps": [per step "drop", gain, bits added, the score], "blocks": the blocks
        left}."""
        t0 = time.perf_counter()
        kept = frozenset(tuple(b) for b in blocks)
        cur = self.score([(kept, frozenset())])[0]
        rec = {"start": cur, "steps": [], "rounds": 0}
        last: dict = {}
        for step in range(max_steps):
            singles = [(b,) for b in sorted(kept)]
            groups = [c for c in (tuple(b for b in sorted(kept) if b[0] == n) for n in self.nat.names) if len(c) > 1]
            if step == 0:
                cands = singles + groups
            else:
                best = sorted((c for c in last if set(c) <= kept), key=lambda c: -last[c])[:lazy]
                cands = list(dict.fromkeys(best + groups))
            cands = [c for c in cands if set(c) <= kept]
            if not cands:
                break
            got = self.score([(kept - set(c), frozenset()) for c in cands])
            rec["rounds"] += 1
            rows = []
            for c, r in zip(cands, got):
                g = (r["j"] or 0.0) - (cur["j"] or 0.0)
                last[c] = g
                rows.append((c, g, r["bits"] - cur["bits"], r))
            ok = [x for x in rows if x[1] >= min_gain]
            if not ok:
                break
            c, g, db, r = max(ok, key=lambda x: (x[1], -x[2]))
            kept, cur = kept - set(c), r
            rec["steps"].append({"drop": [list(b) for b in c], "gain": g, "bits_added": db, **r})
            if log:
                log(f"    prune {step}: drop {' + '.join(f'{b[0]}@{b[1]}' for b in c)} J {r['j']:.3f} (s {r['s_search']:.3f}, "
                    f"F1 {r['f1']:.3f}), {db:+.0f} bits, {len(kept)} blocks left")
        rec["blocks"] = [list(b) for b in sorted(kept)]
        rec["final"] = cur
        rec["seconds"] = time.perf_counter() - t0
        return rec

    def refine(self, blocks: list, keep: float = KEEP, only=None, log=None, cheaper: bool = True) -> dict:
        """Each taken block (in the order taken; with only, those in it) replaced by the circuit's subcomponents there
        and the block's outside subcomponents in the screen's order (at the expansion without that block), the shortest
        prefix of 1, 2, 4, ... whose J keeps at least keep x the block's gain, taken when it costs fewer bits than the
        block (cheaper=False: whatever it costs, for a teacher that must list subcomponents to state connections). {"blocks": the blocks kept, "extra": the subcomponents taken for the others, "per_block": [per block its
        prefixes' sizes, J and bits, the choice]}. At a circuit position whose members are VPD's actives there (an
        influential set) no prefix is cheaper: the block costs its name and those actives, a prefix the actives and
        each subcomponent it adds; refine pays off at other positions."""
        nat = self.nat
        kept, extra, per = list(map(tuple, blocks)), set(), []
        for b in [x for x in map(tuple, blocks) if only is None or x in set(map(tuple, only))]:
            others = frozenset(x for x in kept if x != b)
            full = self.score([(frozenset(kept), frozenset(extra))])[0]
            without = self.score([(others, frozenset(extra))])[0]
            gain = (full["j"] or 0.0) - (without["j"] or 0.0)
            grad = self.screen(others, frozenset(extra))
            n, t = b
            m = self.masks(others, extra)[n][t].cpu()
            g = grad[n][t].clone()
            g[m] = float("inf")
            order = [int(c) for c in torch.argsort(g).tolist() if not bool(m[c])]
            sizes, k = [], 1
            while k < len(order):
                sizes.append(k)
                k *= 2
            got = self.score([(others, frozenset(extra) | {(n, t, c) for c in order[:k]}) for k in sizes]) if sizes else []
            pts = [{"k": k, "j": r["j"], "bits": r["bits"], "s_search": r["s_search"], "f1": r["f1"]} for k, r in zip(sizes, got)]
            choice = None
            for k, r in zip(sizes, got):
                if (r["j"] or 0.0) - (without["j"] or 0.0) >= keep * gain and (r["bits"] < full["bits"] or not cheaper):
                    choice = k
                    break
            row = {"block": [n, t], "gain": gain, "block_bits": full["bits"], "without_bits": without["bits"],
                   "j_block": full["j"], "j_without": without["j"], "prefixes": pts, "choice": choice,
                   "order": order[:max(sizes) if sizes else 0]}
            if choice is not None:
                kept.remove(b)
                extra |= {(n, t, c) for c in order[:choice]}
            per.append(row)
            if log:
                log(f"    refine {n}@{t}: gain {gain:.3f}, block {full['bits']:.0f} bits, prefix "
                    + (f"{choice} ({pts[sizes.index(choice)]['bits']:.0f} bits)" if choice else "none cheaper"))
        return {"blocks": [list(b) for b in kept], "extra": sorted(list(x) for x in extra), "per_block": per}


    @torch.no_grad()
    def interpret(self, per_block: list, top: int = 16) -> list[dict]:
        """What each taken block's compensation is, from refine's rows (per_block): concentration (the smallest prefix
        of the block's outside subcomponents, in the screen's order, holding 50% and 90% of the block's gain in J); for
        its top subcomponents (the first `top` of that order) against as many drawn at random from the block's outside,
        the mean over them of the largest |cosine| of their read vectors (V's columns) and of their write vectors (U's
        rows) with the circuit's subcomponents of the same site at any position (a backup duplicating a circuit route
        reads or writes what a member does), the signed cosine at that largest (positive: the same direction); and for
        an attention site, each top subcomponent's head (the largest head slice of its head-side vector) beside the
        circuit's heads there, with, for an attention output, each head's most attended source position at the block's
        position in the model's run (another head reading the same source is a parallel route)."""
        import torch.nn.functional as F

        nat, tg = self.nat, self.nat.target
        H, hd = tg.n_head, tg.hd
        rng = np.random.default_rng([int(self.tx.seed), 3])
        pats = self._patterns() if any(native._name_layer_kind(pb["block"][0])[1] == "o_proj" for pb in per_block) else {}
        out = []
        for pb in per_block:
            n, t = pb["block"]
            L, kind = native._name_layer_kind(n)
            st = tg.site(n)
            Vt, U = st.V.T, st.U                                     # [C, d_in], [C, d_out]
            members = sorted({c for nn, _, c in self.subs0 if nn == n})
            here = sorted({c for nn, tt, c in self.subs0 if nn == n and tt == t})
            comp = list(pb["order"][:top])
            outside = [c for c in range(nat.C[n]) if c not in set(here)]
            rand = rng.choice(outside, min(top, len(outside)), replace=False).tolist() if outside else []
            j0, gain = pb.get("j_without") or 0.0, pb["gain"]
            ks = {}
            for share in (0.5, 0.9):
                ks[share] = next((x["k"] for x in pb["prefixes"] if (x["j"] or 0.0) - j0 >= share * gain), None) if gain > 0 else None

            def cos(A, xs, ys):  # (mean over xs of the largest |cos| with ys, the signed cosines there)
                if not xs or not ys:
                    return None, None
                cm = F.normalize(A[xs].float(), dim=-1) @ F.normalize(A[ys].float(), dim=-1).T
                i = cm.abs().argmax(1, keepdim=True)
                return float(cm.abs().max(1).values.mean()), [round(float(v), 3) for v in cm.gather(1, i).squeeze(1)]

            row = {"block": [n, t], "gain": gain, "k50": ks[0.5], "k90": ks[0.9], "members": len(members), "members_here": len(here)}
            row["read_cos_comp"], row["read_cos_signed"] = cos(Vt, comp, members)
            row["read_cos_rand"] = cos(Vt, rand, members)[0]
            row["write_cos_comp"], row["write_cos_signed"] = cos(U, comp, members)
            row["write_cos_rand"] = cos(U, rand, members)[0]
            if kind in ("o_proj", "v_proj", "q_proj", "k_proj"):
                side = Vt if kind == "o_proj" else U                 # the vector over heads
                heads = lambda xs: [int(side[x].view(H, hd).norm(dim=-1).argmax()) for x in xs]  # noqa: E731
                row["heads_comp"], row["heads_circuit"] = heads(comp), heads(here)
                if kind == "o_proj" and L in pats:
                    src = pats[L][:, t].argmax(-1).tolist()          # per head its most attended position at t
                    row["source_comp"] = [src[h] for h in row["heads_comp"]]
                    row["source_circuit"] = [src[h] for h in row["heads_circuit"]]
                    row["source_weight_comp"] = [round(float(pats[L][h, t].max()), 3) for h in row["heads_comp"]]
            out.append(row)
        return out

    @torch.no_grad()
    def _patterns(self) -> dict:
        """The model's attention patterns per layer [H, T, T] (its run, every subcomponent on)."""
        nat, E, tg = self.nat, self.E, self.nat.target
        got: dict = {}
        orig = E._site

        def site(n, h, mask, rec=None):
            y = orig(n, h, mask, rec)
            if n.endswith(("q_proj", "k_proj")):
                got[n] = y
            return y

        E._site = site
        try:
            E.log_probs({n: torch.ones(1, self.T, nat.C[n], device=nat.dev) for n in nat.names})
        finally:
            E._site = orig
        return {L: tg.attention_pattern(got[native.site_name(L, "q_proj")], got[native.site_name(L, "k_proj")])[0]
                for L in range(tg.n_layer)}


def nodes_search(ge: goal_eval.GoalEval, tx: goal_eval.Text, graphs: list, draws: int = SEARCH_DRAWS,
                 block_units: bool = True) -> list[dict]:
    """The node family on search draws for a teacher's acceptance (teacher.Acceptance's form: per graph a dict of
    terms): per graph (a node set, blocks allowed) s_nodes in goal_eval's form, KL(model less d || circuit less d)
    against W = KL(model less d || model) with VPD's answer less d the reference, pooled by W, on draws of the graph's
    own nodes seeded apart from goal_eval's (the family "nodes_search"), a block one unit (block_units) or its
    subcomponents; with s_nodes_raw (unfloored) and overstate_nodes. A graph with no node: None."""
    out = []
    for g in graphs:
        if not g.count():
            out.append({"s_nodes": None, "s_nodes_raw": None, "overstate_nodes": None})
            continue
        bk = Backup(ge, tx, g, draws=draws, family="nodes_search", block_units=block_units)
        m = bk.masks()
        lq = bk._run([(m, None)] + [(m, j) for j in range(len(bk.drops))])
        f = bk._fields(lq)
        kc = carry.kl(bk.lm, lq[1:]).tolist()
        raw = goal_eval._pooled_raw([[j, a, b, w] for j, (a, b, w) in enumerate(zip(kc, bk.KV, bk.W))])
        out.append({"s_nodes": f["s_search"], "s_nodes_raw": raw, "overstate_nodes": f["overstate"]})
        del bk, lq
        native.free(ge.nat.dev)
    return out


def expand_circuit(ge: goal_eval.GoalEval, tx: goal_eval.Text, graph: native.Graph, max_steps: int = MAX_STEPS,
                   min_gain: float = MIN_GAIN, backward: bool = True, refine: bool = True, draws: int = SEARCH_DRAWS,
                   subcomponents: bool = False, keep: float = KEEP, log=None) -> tuple:
    """The teacher's call: (the expanded node set, its record). graph: a node set on tx (goal_eval's complete graph,
    blocks allowed), e.g. block_search's circuit before its connections are stated. The expansion is the better by J on
    the backup family's search draws of expand's forward greedy and (backward) prune's search down from every site at
    the circuit's positions as blocks, each block subsuming the circuit's subcomponents at its site and position; with
    refine, a block a cheaper prefix of subcomponents keeps (refine's rule) is replaced by it. With subcomponents (the
    forward greedy alone), every taken block is replaced by its shortest prefix of outside subcomponents in the screen's
    order keeping keep x its gain in J, whatever it costs, a block no prefix keeps staying a block: an answer a teacher
    can wire (teacher.EDGE_NODES counts a block's every subcomponent; a forward block is 512 to 3,584 of them). A
    teacher states connections afterwards over the returned node set (its blocks are blocks: Connector's `blocks`).
    record:
    {"expand", "prune" (or None), "chosen", "refine" (or None), "start", "final": the expansion's score, "measured"}."""
    bk = Backup(ge, tx, graph, draws=draws)
    rec = {"expand": bk.expand(max_steps=max_steps, min_gain=min_gain, log=log), "prune": None, "refine": None, "chosen": "fwd"}
    blocks, extra = [tuple(b) for b in rec["expand"]["blocks"]], []
    if backward and not subcomponents:
        allpos = [(n, t) for t in bk.positions for n in ge.nat.names if (n, t) not in bk.blocks0]
        rec["prune"] = bk.prune(allpos, min_gain=min_gain, log=log)
        jf = rec["expand"]["steps"][-1]["j"] if rec["expand"]["steps"] else rec["expand"]["start"]["j"]
        if (rec["prune"]["final"]["j"] or 0.0) > (jf or 0.0):
            blocks, rec["chosen"] = [tuple(b) for b in rec["prune"]["blocks"]], "back"
    if subcomponents and blocks:
        rec["refine"] = bk.refine(blocks, keep=keep, log=log, cheaper=False)
        blocks, extra = [tuple(b) for b in rec["refine"]["blocks"]], [tuple(x) for x in rec["refine"]["extra"]]
    elif refine and blocks:
        rec["refine"] = bk.refine(blocks, keep=keep, only=[b for b in blocks if b[1] not in set(bk.positions)], log=log)
        blocks, extra = [tuple(b) for b in rec["refine"]["blocks"]], [tuple(x) for x in rec["refine"]["extra"]]
    rec["start"] = rec["expand"]["start"]
    rec["final"] = bk.score([(frozenset(blocks), frozenset(extra))])[0]
    rec["measured"] = bk.measured
    return bk.graph(frozenset(blocks), frozenset(extra)), rec


# ---- the dev-set run and its evaluation


def _random_blocks(bk: Backup, k: int, seed: int) -> list:
    """k blocks drawn uniformly among the sites at the circuit's positions (the equal-count trivial baseline)."""
    pool = [(n, t) for t in bk.positions for n in bk.nat.names if (n, t) not in bk.blocks0]
    rng = np.random.default_rng(seed)
    return [pool[i] for i in sorted(rng.choice(len(pool), min(k, len(pool)), replace=False))]


def _vpd_extra(bk: Backup, bits: float) -> list:
    """VPD's order beyond the circuit, the most important first, as many subcomponents as fit in bits more than the
    circuit's (the equal-bits trivial baseline)."""
    order = getattr(bk.tx, "vpd_order", None) or bk.nat._vpd_order(bk.tx.ids, bk.tx.targets)
    have = set(bk.nodes)
    rest = [tuple(nd) for nd in order if tuple(nd) not in have]
    base = bk.bits()
    lo, hi = 0, 1
    while hi < len(rest) and bk.bits(extra=frozenset(rest[:hi])) - base < bits:
        lo, hi = hi, min(len(rest), 2 * hi)
    while lo + 1 < hi:                                       # the fewest whose added bits reach bits
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if bk.bits(extra=frozenset(rest[:mid])) - base < bits else (lo, mid)
    return rest[:hi]


def run(split: str, n: int, src: Path, out: Path, offset: int = 0, stride: int = 1, max_steps: int = MAX_STEPS,
        log=print) -> None:
    """Expand each text's circuit in src (<task>.json, a node set); write out/<set>/<task>.json for the sets: "fwd"
    (expand's forward greedy), "back" (prune's backward search from every site at the circuit's positions), "full" (the
    one of them with the higher J on the search draws, "chosen" in the summary), "b1", "b2", "b4" (fwd's first 1, 2, 4
    steps), "refined" (refine's of full), "rand" (as many blocks as full drawn at the circuit's positions), "allpos"
    (every site at the circuit's positions), "vpdbits" (VPD's order beyond the circuit at full's added bits, at least
    one subcomponent); and a line per text in out/summary.jsonl: the records of expand, prune, refine and interpret,
    every set's score on the search draws ("search") and on EVAL_DRAWS fresh draws of the circuit's own nodes
    ("eval_units", the start's among them)."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["task"] for x in (out / "summary.jsonl").read_text().splitlines() if x.strip()} \
        if (out / "summary.jsonl").exists() else set()
    for p in native.tasks(split)[:n][offset::stride]:
        f = src / f"{p.stem}.json"
        if p.stem in done or not f.exists():
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem))
        g, _ = goal_eval.circuit(json.loads(f.read_text()), targets, nat.C, tx.T)
        log(f"{p.stem}: {g.count()} nodes at {len({t for _, t, _ in g.node_set()})} positions, T {tx.T}")
        bk = Backup(ge, tx, g)
        ex = bk.expand(max_steps=max_steps, log=lambda s: log(s))
        allpos = [(n_, t) for t in bk.positions for n_ in nat.names if (n_, t) not in bk.blocks0]
        back = bk.prune(allpos, log=lambda s: log(s))
        fwd = [tuple(b) for b in ex["blocks"]]
        jf = ex["steps"][-1]["j"] if ex["steps"] else ex["start"]["j"]
        blocks = [tuple(b) for b in back["blocks"]] if (back["final"]["j"] or 0.0) > (jf or 0.0) else fwd
        far = [b for b in blocks if b[1] not in set(bk.positions)]   # refine replaces only blocks off the circuit's positions
        rf = bk.refine(blocks, only=far, log=lambda s: log(s)) if far else {"blocks": list(blocks), "extra": [], "per_block": []}
        diag = bk.refine(fwd[:4], log=lambda s: log(s)) if fwd else {"per_block": []}   # forward's first blocks: what carries them
        interp = bk.interpret(diag["per_block"] + rf["per_block"]) if diag["per_block"] + rf["per_block"] else []
        sets = {f"b{k}": ([tuple(b) for b in ex["step_blocks"][k - 1]], []) for k in (1, 2, 4) if k <= len(ex["step_blocks"])}
        sets["fwd"] = (fwd, [])
        sets["back"] = ([tuple(b) for b in back["blocks"]], [])
        sets["full"] = (blocks, [])
        sets["refined"] = ([tuple(b) for b in rf["blocks"]], [tuple(x) for x in rf["extra"]])
        sets["rand"] = (_random_blocks(bk, len(blocks), native.task_seed(p.stem)), [])
        sets["allpos"] = (allpos, [])
        added = bk.bits(frozenset(blocks)) - bk.bits()
        sets["vpdbits"] = ([], _vpd_extra(bk, max(added, 1.0)))
        scores = dict(zip(sets, bk.score([(frozenset(b), frozenset(x)) for b, x in sets.values()])))
        # the same sets on fresh draws of the circuit's own nodes (the backup family's evaluation seed, EVAL_DRAWS): the
        # response question apart from goal_eval's node family, which also removes the expansion's own subcomponents
        ev = Backup(ge, tx, g, draws=EVAL_DRAWS, family="backup_eval")
        units = dict(zip(["start"] + list(sets), ev.score([(frozenset(), frozenset())] + [(frozenset(b), frozenset(x)) for b, x in sets.values()])))
        del ev
        for name, (b, x) in sets.items():
            (out / name).mkdir(exist_ok=True)
            (out / name / f"{p.stem}.json").write_text(json.dumps(goal_eval.to_json(bk.graph(frozenset(b), frozenset(x)))))
        row = {"task": p.stem, "T": tx.T, "nodes": len(bk.nodes), "positions": bk.positions, "bits_start": bk.bits(),
               "expand": ex, "prune": back, "chosen": "back" if blocks != fwd else "fwd", "refine": rf, "refine_fwd": diag["per_block"], "interpret": interp, "search": scores, "eval_units": units, "sets": {k: [list(map(list, b)), list(map(list, x))] for k, (b, x) in sets.items()},
               "measured": bk.measured, "seconds": time.perf_counter() - t0}
        with open(out / "summary.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"{p.stem}: {len(blocks)} blocks, J {ex['start']['j']:.3f} -> {scores['full']['j']:.3f} "
            f"(s {ex['start']['s_search']:.3f} -> {scores['full']['s_search']:.3f}), {row['seconds']:.0f} s")
        del bk
        native.free(nat.dev)


def repairs(split: str, n: int, src: Path, out: Path, sets=("fwd", "b1", "back", "sub50", "sub90"), offset: int = 0,
            stride: int = 1, log=print) -> Path:
    """Backup.repair and Backup.frozen per text on the backup family's search draws: the outside's cancelling share of
    the circuit's change at the prediction (r, cos, per site) for src's circuit and the share each of out's sets holds
    ("held"); the overstate with the outside held at its activations on the text (frozen: start, and per set with only
    the outside of the set held), beside the circuit's own overstate; the circuit's and the model's scores with every
    norm's scale held at the model's on the text (norm_frozen); the circuit's stream against the model's at the norms
    (stream_ratio) and each set's overstate on the same draws. Lines in out/repair.jsonl; texts already there
    skipped."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    f_out = out / "repair.jsonl"
    done = {json.loads(x)["task"] for x in f_out.read_text().splitlines() if x.strip()} if f_out.exists() else set()
    for p in native.tasks(split)[:n][offset::stride]:
        f0 = src / f"{p.stem}.json"
        if p.stem in done or not f0.exists():
            continue
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem))
        g0, _ = goal_eval.circuit(json.loads(f0.read_text()), targets, nat.C, tx.T)
        bk = Backup(ge, tx, g0)
        row = {"task": p.stem, "start": bk.repair(), "held": {}, "frozen": {"start": bk.frozen()},
               "overstate": bk.score([(frozenset(), frozenset())])[0]["overstate"], "norm_frozen": bk.norm_frozen(),
               "stream": {"start": bk.stream_ratio()}, "overstate_set": {}}
        for k in sets:
            f = out / k / f"{p.stem}.json"
            if f.exists():
                b, x = split_graph(goal_eval.circuit(json.loads(f.read_text()), targets, nat.C, tx.T)[0])
                b, x = b - bk.blocks0, x - bk.subs0
                row["held"][k] = bk.repair(b, x)["held"]
                row["frozen"][k] = bk.frozen(b, x)["overstate_frozen"]
                row["stream"][k] = bk.stream_ratio(b, x)
                row["overstate_set"][k] = bk.score([(frozenset(b), frozenset(x))])[0]["overstate"]
        with open(f_out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        nf = row["norm_frozen"]
        log(f"{p.stem}: stream rms circuit/model at the targets " + " ".join(f"{k} {v['targets']:.2f}" for k, v in row["stream"].items())
            + "; overstate " + " ".join(f"{k} {v:.2f}" for k, v in row["overstate_set"].items()))
        log(f"{p.stem}: norms held: circuit overstate {nf['circuit']['overstate']:.2f} s {nf['circuit']['s_search']} "
            f"F1 {nf['circuit']['f1']}, model {nf['model']['overstate']:.2f}")
        log(f"{p.stem}: r {row['start']['r']:.3f} cos {row['start']['cos']:.3f}, overstate {row['overstate']:.2f} (the model's own "
            f"{row['frozen']['start']['overstate_model']:.2f}), frozen "
            f"{row['frozen']['start']['overstate_frozen']:.2f}; held " + " ".join(f"{k} {v:.3f}" for k, v in row["held"].items())
            + "; frozen outside the set " + " ".join(f"{k} {v:.2f}" for k, v in row["frozen"].items() if k != "start"))
        del bk, tx
        native.free(nat.dev)
    return f_out


EVAL_SETS = ("fwd", "back", "b1", "rand", "allpos")  # run's sets the evaluation scores beside the start
DERIVED = ("rand_fwd", "vpd_eq_fwd", "vpd_eq", "sub50", "sub90", "sub50_rand", "sub90_rand")  # derived_sets'
EVAL_FLIP_ROWS = 40     # goal_eval's FLIP_ROWS (160) and CHANGED_ROWS (64) during the evaluation: rows per run only; at 160
EVAL_CHANGED_ROWS = 32  # one process took 30.6 GB of a 48 GB card on text9000's 13 sets in adversary_flips (dev20be)
TIMED = ("sufficiency", "_node_family", "adversary_flips", "_interchange_family", "changed_inputs", "track_pp",
         "interchange_pp", "pattern_family", "donors", "necessity_spec", "_sides")  # fidelity's families, timed in the log


def derived_sets(row: dict, g0: native.Graph, tx: goal_eval.Text, nat, seed: int) -> dict:
    """Sets built at evaluation from run's summary row and the start circuit g0, {name: (blocks, extra subcomponents)}:
    "rand_fwd", as many blocks as fwd drawn among the sites at the circuit's positions (the forward greedy's
    equal-count trivial baseline; run's "rand" matches full's count, mostly the backward search's ~40 of 48 sites, so
    it is nearly full itself); "vpd_eq" and "vpd_eq_fwd", VPD's order beyond the circuit, as many subcomponents as
    full's (fwd's) blocks add of VPD's active ones (the equal-content trivial baseline; run's "vpdbits" took one
    subcomponent, since blocks lower size_bits); "sub50" and "sub90", fwd with each block refine's diagnostic measured
    (run's refine_fwd: the block's outside subcomponents in the screen's order, prefixes 1, 2, 4, ...) replaced by its
    shortest prefix holding 50% (90%) of the block's gain in J on the search draws, a block with no such prefix kept
    (the compensators at subcomponent level); "sub50_rand" and "sub90_rand", the same with as many of the block's
    outside subcomponents drawn at random (whether the screen's order matters)."""
    blocks0, _ = split_graph(g0)
    nodes = set(goal_eval.expand_blocks(g0, nat.C).node_set())
    positions = sorted({t for _, t, _ in nodes})
    fwd = [tuple(b) for b in row["sets"]["fwd"][0]]
    full = [tuple(b) for b in row["sets"]["full"][0]]
    pool = [(n, t) for t in positions for n in nat.names if (n, t) not in blocks0]
    rng = np.random.default_rng([int(seed), 5])
    out = {"rand_fwd": ([pool[i] for i in sorted(rng.choice(len(pool), min(len(fwd), len(pool)), replace=False).tolist())], [])}
    order = getattr(tx, "vpd_order", None) or nat._vpd_order(tx.ids, tx.targets)
    rest = [tuple(nd) for nd in order if tuple(nd) not in nodes]
    have: dict = {}                                          # (site, position) -> the circuit's VPD-active members there
    for n, t, c in nodes:
        if bool(tx.vpd[n][t, c]):
            have[(n, t)] = have.get((n, t), 0) + 1
    for name, bl in (("vpd_eq", full), ("vpd_eq_fwd", fwd)):
        added = sum(int(tx.vpd[n][t].sum()) - have.get((n, t), 0) for n, t in bl if (n, t) not in blocks0)
        out[name] = ([], rest[:max(1, added)])
    per = {tuple(pb["block"]): pb for pb in row.get("refine_fwd", [])}
    for name, share in (("sub50", 0.5), ("sub90", 0.9)):
        keep, extra, rand = [], [], []
        for b in fwd:
            pb, k = per.get(b), None
            if pb and (pb.get("gain") or 0.0) > 0:
                j0 = pb.get("j_without") or 0.0
                k = next((x["k"] for x in pb["prefixes"] if (x["j"] or 0.0) - j0 >= share * pb["gain"]), None)
            if k is None:
                keep.append(b)
            else:
                extra += [(b[0], b[1], int(c)) for c in pb["order"][:k]]
                outside = [c for c in range(nat.C[b[0]]) if (b[0], b[1], c) not in nodes]
                rand += [(b[0], b[1], int(c)) for c in rng.choice(outside, min(k, len(outside)), replace=False).tolist()]
        out[name] = (keep, extra)
        out[name + "_rand"] = (keep, rand)
    return out


@torch.no_grad()
def node_family(ge: goal_eval.GoalEval, tx: goal_eval.Text, graphs: list, draws: int, reads: bool = False,
                units: list | None = None) -> list[dict]:
    """goal_eval.GoalEval._node_family without reads (reads must be False), its draws (GoalEval._node_draws' common draws),
    engines and fields, with each draw's removals written as one indexed assignment per site instead of one scalar
    assignment per removed subcomponent: on block-expanded node sets (every site at a circuit position, ~150,000
    subcomponents) goal_eval's loop spent 15+ minutes in those writes on dev20b's first text. The same masks, so the
    same numbers (test_backup checks them against goal_eval's)."""
    import engine

    if reads:
        raise ValueError("node_family reproduces goal_eval's node family without reads")
    nat, T, dev = ge.nat, tx.T, ge.nat.dev
    cols = {n: {c for g in graphs for m, _, c in g.node_set() if m == n} for n in nat.names}
    circ = {n: torch.tensor(sorted(cols[n]), dtype=torch.long, device=dev) for n in nat.names}
    vcols = {n: torch.unique(torch.cat([circ[n], tx.vpd[n].any(0).nonzero().flatten()])) for n in nat.names}
    Em = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ, outside="on")
    Ev = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=vcols)
    Ec = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=circ)

    def lut(cs):  # subcomponent index -> the engine's column
        return {n: torch.full((nat.C[n],), -1, dtype=torch.long, device=dev).index_put_((cs[n],), torch.arange(len(cs[n]), device=dev))
                for n in nat.names}
    at, vat = lut(circ), lut(vcols)
    vpd = {n: tx.vpd[n][:, vcols[n]].float() for n in nat.names}
    out = []
    for g in graphs:
        nodes = sorted(g.node_set())
        if not nodes:
            out.append({"kl_nodes_c": ge._bounds(tx, "live")[0], "kl_nodes_vpd": ge._bounds(tx, "live")[1], "reads_nodes": None})
            continue
        drops = ge._node_draws(tx, sorted(units[len(out)]) if units is not None else nodes, draws, held=nodes)  # (goal_eval's)
        per: dict = {}
        for b, dd in enumerate(drops):
            for n, t, c in dd:
                x = per.setdefault(n, ([], [], []))
                x[0].append(b)
                x[1].append(t)
                x[2].append(c)
        per = {n: tuple(torch.tensor(v, dtype=torch.long, device=dev) for v in x) for n, x in per.items()}

        def without(base, where):
            m = {n: x[None].repeat(len(drops), 1, 1) for n, x in base.items()}
            for n, (b, t, c) in per.items():
                m[n][b, t, where[n][c]] = 0.0
            return m

        lm = Em.log_probs(without({n: torch.ones(T, len(circ[n]), device=dev) for n in nat.names}, at))
        lv = Ev.log_probs(without(vpd, vat))
        if g.complete:
            gm = {n: torch.zeros(T, len(circ[n]), device=dev) for n in nat.names}
            by: dict = {}
            for n, t, c in nodes:
                by.setdefault(n, ([], []))
                by[n][0].append(t)
                by[n][1].append(c)
            for n, (t, c) in by.items():
                gm[n][torch.tensor(t, device=dev), at[n][torch.tensor(c, device=dev)]] = 1.0
            lc = Ec.log_probs(without(gm, at))
        else:
            lc = Ec.graph_log_probs(Ec.plan([goal_eval._drop_nodes(g, set(d)) for d in drops], semantics="c"))
        kc_, kv_ = carry.kl(lm, lc).tolist(), carry.kl(lm, lv).tolist()
        w_, claim = carry.kl(lm, tx.ref.expand_as(lm)).tolist(), carry.kl(tx.ref.expand_as(lc), lc).tolist()
        out.append({"kl_nodes_c": float(np.mean(kc_)), "kl_nodes_vpd": float(np.mean(kv_)),
                    "draws_nodes": [[j, a, b_, c, d] for j, (a, b_, c, d) in enumerate(zip(kc_, kv_, w_, claim))],
                    "overstate_nodes": float(np.mean(claim)) / max(float(np.mean(w_)), 1e-12),
                    "reads_nodes": None, "_pairs_nodes": {}})
        native.free(dev)
    return out


def _time_families(ge: goal_eval.GoalEval, log) -> None:
    """Each of fidelity's families (TIMED) logs its seconds: the pod log names a family that holds a run. The node and
    interchange families run without their read pairs (reads=False; the reads_* and dreads_* diagnostics, none of
    phi's terms, come out None): the node family's pairs loop rebuilds each draw's node set once per node
    (`nd not in set(d)`), cubic in the nodes, and a block-expanded set of ~200,000 subcomponents held dev20e's pod at
    0% GPU on its first text for 10 minutes and more."""
    for name in TIMED:
        f = getattr(ge, name)

        def timed(*a, _f=f, _n=name, **k):
            if _n in ("_node_family", "_interchange_family"):
                k["reads"] = False
                a = a[:3] if _n == "_node_family" else a
            t0 = time.perf_counter()
            r = _f(*a, **k)
            log(f"    {_n}: {time.perf_counter() - t0:.1f} s")
            return r
        setattr(ge, name, timed)


def evaluate(split: str, n: int, src: Path, out: Path, sets=EVAL_SETS, derived: bool = True, offset: int = 0,
             stride: int = 1, draws: int = goal_eval.DRAWS, log=print) -> Path:
    """C's families as the reward scores them, on the eval split's draws (goal_eval.GoalEval.fidelity, mode "reward",
    split "eval": point, nodes, flips at FLIP_TERM, track_pp, interchange_pp, donors near and far, pattern and phi's N,
    n_spec; each defined as full mode defines it, full mode adding the step-budget adversary, larger flip budgets and
    the leak diagnostic, none of them phi's), for the start (src's circuit), out's sets and (derived) derived_sets'
    baselines and subcomponent-level sets (written to out/<set>/), one fidelity call per text so every set meets the
    same draws; a set whose node set equals an earlier one's is scored once and its row copied ("same_as"). The
    connection-cut family is left out for every set: on a complete graph it lists every connection among its nodes
    (native.all_edges), and a block's thousands of subcomponents make that list hundreds of millions long. Rows (task,
    kind, nodes, blocks, subcomponents and VPD-active ones claimed, size_bits, phi and phi_tau against the start's row,
    the fidelity fields) appended to out/fidelity.jsonl; texts already there skipped."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    ge._node_family = lambda tx, graphs, draws, reads=False, units=None: node_family(ge, tx, graphs, draws, False, units)
    goal_eval.FLIP_ROWS, goal_eval.CHANGED_ROWS = EVAL_FLIP_ROWS, EVAL_CHANGED_ROWS   # batch sizes only
    _time_families(ge, log)
    pool = torch.load(Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_train.pt")
    ge._edge_family = lambda tx, graphs, draws, reads=False: [{"kl_edges_c": None, "kl_edges_vpd": None, "reads_edges": None}
                                                              for _ in graphs]
    summ = {}
    if (out / "summary.jsonl").exists():
        summ = {r["task"]: r for r in (json.loads(x) for x in (out / "summary.jsonl").read_text().splitlines() if x.strip())}
    f_out = out / "fidelity.jsonl"
    done = {json.loads(x)["task"] for x in f_out.read_text().splitlines() if x.strip()} if f_out.exists() else set()
    P, S = None, sum(nat.C.values())
    for p in native.tasks(split)[:n][offset::stride]:
        f0 = src / f"{p.stem}.json"
        if p.stem in done or not f0.exists():
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem))
        P = nat.positions(targets)
        g0, _ = goal_eval.circuit(json.loads(f0.read_text()), targets, nat.C, tx.T)
        cands = {"start": g0}
        for k in sets:
            f = out / k / f"{p.stem}.json"
            if f.exists():
                cands[k] = goal_eval.circuit(json.loads(f.read_text()), targets, nat.C, tx.T)[0]
        if derived and p.stem in summ:
            b0, s0 = split_graph(g0)
            for k, (bl, ex) in derived_sets(summ[p.stem], g0, tx, nat, native.task_seed(p.stem)).items():
                g = expanded_graph(b0, s0, bl, ex)
                (out / k).mkdir(exist_ok=True)
                (out / k / f"{p.stem}.json").write_text(json.dumps(goal_eval.to_json(g)))
                cands[k] = g
        if len(cands) < 2:
            continue
        graphs, kinds, alias, seen = [], [], {}, {}
        for kind, g in cands.items():
            key = frozenset(goal_eval.expand_blocks(g, nat.C).node_set())
            if key in seen:
                alias[kind] = seen[key]
                continue
            seen[key] = kind
            graphs.append(g)
            kinds.append(kind)
        sizes = ", ".join(f"{k} ({counts(g, nat.C, tx.vpd)['subcomponents']})" for k, g in zip(kinds, graphs))
        log(f"{p.stem}: T {tx.T}, scoring {sizes}")
        fid = ge.fidelity(tx, graphs, pool, draws=draws, mode="reward", split="eval")
        start_row = fid[0]
        rows = {}
        for kind, g, f in zip(kinds, graphs, fid):
            row = {"task": p.stem, "kind": kind, "nodes": g.count(), **counts(g, nat.C, tx.vpd),
                   "size_bits": goal_eval.size_bits(g, P, S, targets, active=tx.vpd), **f}
            row["phi"], row["phi_tau"] = goal_eval.phi(row, start=start_row)
            rows[kind] = row
        for kind, same in alias.items():
            rows[kind] = {**rows[same], "kind": kind, "same_as": same}
        with open(f_out, "a") as fh:
            for row in rows.values():
                fh.write(json.dumps(row, default=float) + "\n")
        if log:
            log(f"{p.stem}: {len(graphs)} circuits ({len(alias)} the same as another), {time.perf_counter() - t0:.0f} s")
        del tx
        native.free(nat.dev)
    return f_out


ORDER = ("start", "full", "fwd", "b1", "b2", "rand_fwd", "vpd_eq_fwd", "sub50", "sub50_rand", "sub90", "sub90_rand", "back",
         "rand", "vpd_eq", "allpos")


def table(out: Path) -> str:
    """Per set, medians over the texts of s_nodes, overstate_nodes, phi (against the text's start row), s_track_pp,
    s_point_c, c, size_bits (blocks at their active contents, goal_eval's a699a8b), the subcomponents claimed and VPD's
    active ones among them, with the paired median differences from the start; "full"
    is each text's chosen expansion (fwd or back, the summary's "chosen")."""
    rows = [json.loads(x) for x in (out / "fidelity.jsonl").read_text().splitlines() if x.strip()]
    chosen = {}
    if (out / "summary.jsonl").exists():
        chosen = {json.loads(x)["task"]: json.loads(x).get("chosen") for x in (out / "summary.jsonl").read_text().splitlines() if x.strip()}
    start = {r["task"]: r for r in rows if r["kind"] == "start"}
    by: dict = {}
    for r in rows:
        if r["task"] in start:
            by.setdefault(r["kind"], []).append(r)
            if chosen.get(r["task"]) == r["kind"]:
                by.setdefault("full", []).append(r)
    keys = ("s_nodes", "overstate_nodes", "phi", "s_track_pp", "s_point_c", "c", "size_bits", "subcomponents", "active")
    lines = ["set".ljust(11) + "".join(k[:11].rjust(12) for k in keys) + "   d_s_nodes   d_phi  d_track  d_point   texts"]
    for kind in list(ORDER) + sorted(set(by) - set(ORDER)):
        rs = by.get(kind)
        if not rs:
            continue
        med = []
        for k in keys:
            xs = [x[k] for x in rs if x.get(k) is not None]
            med.append(float(np.median(xs)) if xs else float("nan"))

        def d(k):
            xs = [x[k] - start[x["task"]][k] for x in rs if x.get(k) is not None and start[x["task"]].get(k) is not None]
            return float(np.median(xs)) if xs else float("nan")
        lines.append(kind[:11].ljust(11) + "".join(f"{v:12.3f}" if abs(v) < 1e4 else f"{v:12.0f}" for v in med)
                     + f"   {d('s_nodes'):+9.3f} {d('phi'):+7.3f} {d('s_track_pp'):+8.3f} {d('s_point_c'):+8.3f}  {len(rs)}")
    return "\n".join(lines)


PAIRS = (("fwd", "start"), ("sub50", "start"), ("sub90", "start"), ("full", "start"), ("back", "start"),
         ("vpd_eq_fwd", "start"), ("fwd", "rand_fwd"), ("sub50", "sub50_rand"), ("sub90", "sub90_rand"), ("full", "rand"))
PAIR_KEYS = ("phi", "s_nodes", "s_track_pp", "s_point_c", "n_spec", "c", "size_bits", "subcomponents")


def paired(out: Path, pairs=PAIRS, keys=PAIR_KEYS, boots: int = 2000) -> str:
    """Paired differences over the texts between sets (each set against the start and against its equal-size trivial
    control): the median difference with a 95% bootstrap interval over texts and the share of texts where the first is
    higher; size_bits and subcomponents as log2 ratios. A null pair (vpd_eq_fwd, start: one subcomponent added) shows
    the noise of goal_eval's node family, whose draws are seeded by each graph's node set."""
    rows = [json.loads(x) for x in (out / "fidelity.jsonl").read_text().splitlines() if x.strip()]
    chosen = {}
    if (out / "summary.jsonl").exists():
        chosen = {json.loads(x)["task"]: json.loads(x).get("chosen") for x in (out / "summary.jsonl").read_text().splitlines() if x.strip()}
    by: dict = {}
    for r in rows:
        by.setdefault(r["kind"], {})[r["task"]] = r
        if chosen.get(r["task"]) == r["kind"]:
            by.setdefault("full", {})[r["task"]] = r
    rng = np.random.default_rng(0)

    def val(r, k):
        v = r.get(k)
        return None if v is None else (math.log2(max(v, EPS)) if k in ("size_bits", "subcomponents") else v)
    lines = []
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        lines.append(f"{a} - {b}")
        for k in keys:
            d = np.array([val(by[a][t], k) - val(by[b][t], k) for t in by[a] if t in by[b]
                          and val(by[a][t], k) is not None and val(by[b][t], k) is not None])
            if not len(d):
                continue
            bs = [float(np.median(rng.choice(d, len(d)))) for _ in range(boots)]
            name = f"log2 {k}" if k in ("size_bits", "subcomponents") else k
            lines.append(f"   {name:20s} {np.median(d):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
                         f"  higher in {np.mean(d > 0):.2f} of {len(d)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "eval", "repair", "table"))
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--src", type=Path, default=Path.home() / "mpd-data/graph_oracle/teacher/influential_heldout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    a = ap.parse_args()
    if a.split not in ("heldout", "train"):
        raise SystemExit("backup.py runs on the development (heldout) or training texts only")
    if a.cmd == "run":
        run(a.split, a.n, a.src, a.out, a.offset, a.stride, a.max_steps, log=lambda s: print(s, flush=True))
    elif a.cmd == "eval":
        faulthandler.dump_traceback_later(900, repeat=True)  # a family holding the run shows in the log's stacks
        evaluate(a.split, a.n, a.src, a.out, offset=a.offset, stride=a.stride, log=lambda s: print(s, flush=True))
        faulthandler.cancel_dump_traceback_later()
    elif a.cmd == "repair":
        repairs(a.split, a.n, a.src, a.out, offset=a.offset, stride=a.stride, log=lambda s: print(s, flush=True))
    if (a.out / "fidelity.jsonl").exists():
        print(table(a.out))
        print(paired(a.out))


if __name__ == "__main__":
    main()
