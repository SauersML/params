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
     the source set (the start) and OUT's sets, every set by the same families (the connection-cut family left out:
     see evaluate), then a table with phi against each text's start row.
"""
from __future__ import annotations

import argparse
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


class Backup:
    """One text and one circuit (a node set): the search's draws, the model's and VPD's answer's runs under them, and
    the measurements above. graph: a goal_eval node set (complete; blocks allowed, stranded nodes dropped first)."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, graph: native.Graph, draws: int = SEARCH_DRAWS,
                 rows: int = ROWS, family: str = "backup"):
        import engine

        nat = ge.nat
        self.ge, self.tx, self.nat, self.rows = ge, tx, nat, rows
        g = goal_eval.strand(graph, tx.targets, nat.C)
        if not g.complete:
            raise ValueError("backup expansion takes a node set (a complete graph): state connections after it")
        self.blocks0 = frozenset((n, t) for n, t, c in g.node_set() if c == BLOCK)
        self.subs0 = frozenset(nd for nd in g.node_set() if nd[2] != BLOCK)
        self.nodes = sorted(goal_eval.expand_blocks(g, nat.C).node_set())  # the removal units: subcomponents
        self.positions = sorted({t for _, t, _ in self.nodes})
        self.T, self.P, self.S = tx.T, nat.positions(tx.targets), sum(nat.C.values())
        self.E = engine.Engine(nat, tx.ids, tx.targets, live=True)       # every column: blocks switch whole sites on
        self.e0, self.v0 = ge._bounds(tx, "live")
        key = int(hashlib.sha256(repr((family, self.nodes)).encode()).hexdigest()[:15], 16)
        rng = np.random.default_rng([tx.seed, key])
        self.drops = []
        for j in range(draws):
            d = [nd for nd in self.nodes if rng.random() < (j + 0.5) / draws]
            self.drops.append(d or [self.nodes[int(rng.integers(len(self.nodes)))]])
        dev = nat.dev
        self._drop_idx = []                                          # per draw {site: (positions, columns)}
        for d in self.drops:
            by = {}
            for n, t, c in d:
                by.setdefault(n, ([], []))
                by[n][0].append(t)
                by[n][1].append(c)
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
        """The expanded answer: a node set, each block one node (BLOCK) holding the circuit's subcomponents at its site
        and position, the extra subcomponents listed."""
        bl = self.blocks0 | frozenset(blocks)
        subs = [nd for nd in self.subs0 | frozenset(extra) if (nd[0], nd[1]) not in bl]
        return native.Graph(sorted(subs) + sorted((n, t, BLOCK) for n, t in bl), complete=True)

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

    def refine(self, blocks: list, keep: float = KEEP, only=None, log=None) -> dict:
        """Each taken block (in the order taken; with only, those in it) replaced by the circuit's subcomponents there
        and the block's outside subcomponents in the screen's order (at the expansion without that block), the shortest
        prefix of 1, 2, 4, ... whose J keeps at least keep x the block's gain, taken when it costs fewer bits than the
        block. {"blocks": the blocks kept, "extra": the subcomponents taken for the others, "per_block": [per block its
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
                if (r["j"] or 0.0) - (without["j"] or 0.0) >= keep * gain and r["bits"] < full["bits"]:
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


def expand_circuit(ge: goal_eval.GoalEval, tx: goal_eval.Text, graph: native.Graph, max_steps: int = MAX_STEPS,
                   min_gain: float = MIN_GAIN, backward: bool = True, refine: bool = True, draws: int = SEARCH_DRAWS,
                   log=None) -> tuple:
    """The teacher's call: (the expanded node set, its record). graph: a node set on tx (goal_eval's complete graph,
    blocks allowed), e.g. block_search's circuit before its connections are stated. The expansion is the better by J on
    the backup family's search draws of expand's forward greedy and (backward) prune's search down from every site at
    the circuit's positions as blocks, each block subsuming the circuit's subcomponents at its site and position; with
    refine, a block a cheaper prefix of subcomponents keeps (refine's rule) is replaced by it. A teacher states
    connections afterwards over the returned node set (its blocks are blocks: Connector's `blocks`). record:
    {"expand", "prune" (or None), "chosen", "refine" (or None), "start", "final": the expansion's score, "measured"}."""
    bk = Backup(ge, tx, graph, draws=draws)
    rec = {"expand": bk.expand(max_steps=max_steps, min_gain=min_gain, log=log), "prune": None, "refine": None, "chosen": "fwd"}
    blocks, extra = [tuple(b) for b in rec["expand"]["blocks"]], []
    if backward:
        allpos = [(n, t) for t in bk.positions for n in ge.nat.names if (n, t) not in bk.blocks0]
        rec["prune"] = bk.prune(allpos, min_gain=min_gain, log=log)
        jf = rec["expand"]["steps"][-1]["j"] if rec["expand"]["steps"] else rec["expand"]["start"]["j"]
        if (rec["prune"]["final"]["j"] or 0.0) > (jf or 0.0):
            blocks, rec["chosen"] = [tuple(b) for b in rec["prune"]["blocks"]], "back"
    if refine and blocks:
        rec["refine"] = bk.refine(blocks, only=[b for b in blocks if b[1] not in set(bk.positions)], log=log)
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


def evaluate(split: str, n: int, src: Path, out: Path, sets=("fwd", "back", "b1", "rand", "allpos", "vpdbits"),
             offset: int = 0, stride: int = 1, draws: int = goal_eval.DRAWS, log=print) -> Path:
    """C's families as the reward scores them, on the eval split's draws (goal_eval.GoalEval.fidelity, mode "reward",
    split "eval": point, nodes, flips at FLIP_TERM, track_pp, interchange_pp, donors near and far, pattern and phi's N,
    n_spec), for the start (src's circuit) and out's sets, one fidelity call per text so every set meets the same
    draws; a set whose node set equals an earlier one's is scored once and its row copied ("same_as"). The
    connection-cut family is left out for every set: on a complete graph it lists every connection among its nodes
    (native.all_edges), and a block's thousands of subcomponents make that list hundreds of millions long (as full
    mode's leak diagnostic, a stated graph of the nodes with no edge). Rows (task, kind, nodes, blocks, size_bits,
    phi and phi_tau against the start's row, the fidelity fields) appended to out/fidelity.jsonl; texts already there
    skipped."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    pool = torch.load(Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt")
    ge._edge_family = lambda tx, graphs, draws, reads=False: [{"kl_edges_c": None, "kl_edges_vpd": None, "reads_edges": None}
                                                              for _ in graphs]
    f_out = out / "fidelity.jsonl"
    done = {json.loads(x)["task"] for x in f_out.read_text().splitlines() if x.strip()} if f_out.exists() else set()
    for p in native.tasks(split)[:n][offset::stride]:
        files = {"start": src / f"{p.stem}.json"} | {k: out / k / f"{p.stem}.json" for k in sets}
        files = {k: f for k, f in files.items() if f.exists()}
        if p.stem in done or "start" not in files or len(files) < 2:
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem))
        graphs, kinds, alias, seen = [], [], {}, {}
        for kind, f in files.items():
            g, _ = goal_eval.circuit(json.loads(f.read_text()), targets, nat.C, tx.T)
            key = frozenset(goal_eval.expand_blocks(g, nat.C).node_set())
            if key in seen:
                alias[kind] = seen[key]
                continue
            seen[key] = kind
            graphs.append(g)
            kinds.append(kind)
        fid = ge.fidelity(tx, graphs, pool, draws=draws, mode="reward", split="eval")
        start_row = fid[0]
        rows = {}
        for kind, g, f in zip(kinds, graphs, fid):
            row = {"task": p.stem, "kind": kind, "nodes": g.count(), "blocks": sum(nd[2] == BLOCK for nd in g.node_set()),
                   "size_bits": goal_eval.size_bits(g, nat.positions(targets), sum(nat.C.values()), targets, active=tx.vpd), **f}
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


def table(out: Path) -> str:
    """Per set, medians over the texts of s_nodes, overstate_nodes, phi (against the text's start row), s_track_pp,
    s_point_c, c and size_bits, with the paired median differences from the start; "full" is each text's chosen
    expansion (fwd or back, the summary's "chosen")."""
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
    keys = ("s_nodes", "overstate_nodes", "phi", "s_track_pp", "s_point_c", "c", "size_bits", "nodes")
    lines = ["set".ljust(9) + "".join(k[:11].rjust(12) for k in keys) + "   d_s_nodes   d_phi   d_bits  texts"]
    for kind in ["start", "full", "fwd", "back", "b1", "rand", "allpos", "vpdbits"] + sorted(set(by) - {"start", "full", "fwd", "back", "b1", "rand", "allpos", "vpdbits"}):
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
        lines.append(kind[:9].ljust(9) + "".join(f"{v:12.3f}" if abs(v) < 1e4 else f"{v:12.0f}" for v in med)
                     + f"   {d('s_nodes'):+9.3f} {d('phi'):+7.3f} {d('size_bits'):+8.0f}  {len(rs)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "eval", "table"))
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
        evaluate(a.split, a.n, a.src, a.out, offset=a.offset, stride=a.stride, log=lambda s: print(s, flush=True))
    print(table(a.out) if (a.out / "fidelity.jsonl").exists() else "")


if __name__ == "__main__":
    main()
