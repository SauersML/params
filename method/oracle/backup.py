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
     subcomponents at its site and position): candidates every site at the circuit's positions and the screen's best
     at the other positions from the circuit's first to the last target; every candidate's addition measured (score),
     the one with the largest gain in J per bit taken (a candidate that costs no bits first, by gain), while the gain
     is at least min_gain; later rounds re-measure the best candidates of the last (lazy greedy) and the screen's new
     best. Every membership decision is measured.
  refine: a taken block replaced by the circuit's subcomponents there plus the outside subcomponents of the block in
     the screen's order, the shortest prefix measured to keep at least keep x the block's gain in J, taken where it
     costs fewer bits than the block.

  expand_circuit(ge, tx, graph, ...) -> (expanded graph, record): the teacher's call (pymove's teacher.py, after the
     node search and before its connections; see expand_circuit).
  backup.py run --split heldout --n 20 --src DIR --out OUT: expand each text's circuit in DIR, writing OUT/<set>/<task>.json
     for the path points, the refined circuit and the equal-size trivial baselines, and OUT/summary.jsonl.
  backup.py eval --out OUT: goal_eval.fidelity_report (full mode, eval draws) on OUT's sets and the source set, then a
     table with phi against the source (start) row of each text.
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
SEARCH_DRAWS = 16       # the search's removal draws per circuit
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
        """The greedy block expansion (module docstring): {"start": the circuit's score, "steps": [per step the block
        taken, its gain and the expansion's score], "rounds": [per round the candidates measured, (site, position,
        gain in J, bits added, response rank) each], "blocks": the blocks taken}."""
        t0 = time.perf_counter()
        resp = self.responses()
        rank = {}                                                    # (site, position) -> its response's rank
        flat = sorted(((float(resp["outside"][n][t]), n, t) for n in self.nat.names for t in range(self.T)), reverse=True)
        for i, (_, n, t) in enumerate(flat):
            rank[(n, t)] = i
        cur = self.score([(frozenset(), frozenset())])[0]
        rec = {"start": cur, "steps": [], "rounds": [], "blocks": [],
               "responses_top": [[n, t, v, float(resp["size"][n][t])] for v, n, t in flat[:48]]}
        taken: frozenset = frozenset()
        last: dict = {}                                              # candidate -> its last measured gain
        at_circuit = [(n, t) for t in self.positions for n in self.nat.names if (n, t) not in self.blocks0]
        for step in range(max_steps):
            grad = self.screen(taken)
            far = [(n, t) for _, n, t in self.block_screen(grad, taken) if t not in set(self.positions)][:screen_top]
            if step == 0:
                cands = list(dict.fromkeys(at_circuit + far))
            else:
                best = sorted((c for c in last if c not in taken), key=lambda c: -last[c])[:lazy]
                cands = list(dict.fromkeys(best + far))
            cands = [c for c in cands if c not in taken]
            if not cands:
                break
            got = self.score([(taken | {c}, frozenset()) for c in cands])
            rows = []
            for c, r in zip(cands, got):
                gain = (r["j"] or 0.0) - (cur["j"] or 0.0)
                last[c] = gain
                rows.append((c, gain, r["bits"] - cur["bits"], r))
            rec["rounds"].append([[c[0], c[1], g, db, rank.get(c)] for c, g, db, _ in rows])

            def key(x):                                              # free first (by gain), then gain per bit
                _, g, db, _ = x
                return (1, g) if db <= 0 else (0, g / db)
            ok = [x for x in rows if x[1] >= min_gain]
            if not ok:
                break
            c, g, db, r = max(ok, key=key)
            taken = taken | {c}
            cur = r
            rec["steps"].append({"block": [c[0], c[1]], "gain": g, "bits_added": db, "response_rank": rank.get(c), **r})
            rec["blocks"].append([c[0], c[1]])
            if log:
                log(f"    step {step}: block {c[0]}@{c[1]} J {r['j']:.3f} (s {r['s_search']:.3f}, F1 {r['f1']:.3f}), "
                    f"{db:+.0f} bits, overstate {r['overstate']:.2f}, {len(cands)} measured")
        rec["seconds"] = time.perf_counter() - t0
        return rec

    def refine(self, blocks: list, keep: float = KEEP, log=None) -> dict:
        """Each taken block (in the order taken) replaced by the circuit's subcomponents there and the block's outside
        subcomponents in the screen's order (at the expansion without that block), the shortest prefix of 1, 2, 4, ...
        whose J keeps at least keep x the block's gain, taken when it costs fewer bits than the block. {"blocks": the
        blocks kept, "extra": the subcomponents taken for the others, "per_block": [per block its prefixes' sizes, J
        and bits, the choice]}."""
        nat = self.nat
        kept, extra, per = list(map(tuple, blocks)), set(), []
        for b in list(map(tuple, blocks)):
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
                   "prefixes": pts, "choice": choice, "order": order[:max(sizes) if sizes else 0]}
            if choice is not None:
                kept.remove(b)
                extra |= {(n, t, c) for c in order[:choice]}
            per.append(row)
            if log:
                log(f"    refine {n}@{t}: gain {gain:.3f}, block {full['bits']:.0f} bits, prefix "
                    + (f"{choice} ({pts[sizes.index(choice)]['bits']:.0f} bits)" if choice else "none cheaper"))
        return {"blocks": [list(b) for b in kept], "extra": sorted(list(x) for x in extra), "per_block": per}


def expand_circuit(ge: goal_eval.GoalEval, tx: goal_eval.Text, graph: native.Graph, max_steps: int = MAX_STEPS,
                   min_gain: float = MIN_GAIN, refine: bool = True, draws: int = SEARCH_DRAWS, log=None) -> tuple:
    """The teacher's call: (the expanded node set, its record). graph: a node set on tx (goal_eval's complete graph,
    blocks allowed), e.g. block_search's circuit before its connections are stated; returns the circuit with the
    compensator blocks expand() takes (each subsuming the circuit's subcomponents at its site and position), and with
    refine those a cheaper prefix of subcomponents keeps replaced by it. Draws are the backup family's own (search
    draws); a teacher states connections afterwards over the returned node set (its blocks are blocks: Connector's
    `blocks`). record: {"expand": Backup.expand's, "refine": Backup.refine's or None, "start", "final": the scores}."""
    bk = Backup(ge, tx, graph, draws=draws)
    rec = {"expand": bk.expand(max_steps=max_steps, min_gain=min_gain, log=log), "refine": None}
    blocks, extra = [tuple(b) for b in rec["expand"]["blocks"]], []
    if refine and blocks:
        rec["refine"] = bk.refine(blocks, log=log)
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
    """Expand each text's circuit in src (<task>.json, a node set); write out/<set>/<task>.json for the sets: "b1",
    "b2", "b4" (the first 1, 2, 4 blocks taken), "full" (every block taken), "refined" (refine's), "rand" (as many
    blocks as "full" drawn at the circuit's positions), "allpos" (every site at the circuit's positions), "vpdbits"
    (VPD's order beyond the circuit at "full"'s added bits); and a line per text in out/summary.jsonl: the expansion's
    record, refine's, every set's score on the search draws ("search") and on EVAL_DRAWS fresh draws of the circuit's own
    nodes ("eval_units", the start's among them)."""
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
        blocks = [tuple(b) for b in ex["blocks"]]
        rf = bk.refine(blocks, log=lambda s: log(s)) if blocks else {"blocks": [], "extra": [], "per_block": []}
        sets = {f"b{k}": (blocks[:k], []) for k in (1, 2, 4) if k <= len(blocks)}
        sets["full"] = (blocks, [])
        sets["refined"] = ([tuple(b) for b in rf["blocks"]], [tuple(x) for x in rf["extra"]])
        sets["rand"] = (_random_blocks(bk, len(blocks), native.task_seed(p.stem)), [])
        sets["allpos"] = ([(n_, t) for t in bk.positions for n_ in nat.names], [])
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
               "expand": ex, "refine": rf, "search": scores, "eval_units": units, "sets": {k: [list(map(list, b)), list(map(list, x))] for k, (b, x) in sets.items()},
               "measured": bk.measured, "seconds": time.perf_counter() - t0}
        with open(out / "summary.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"{p.stem}: {len(blocks)} blocks, J {ex['start']['j']:.3f} -> {scores['full']['j']:.3f} "
            f"(s {ex['start']['s_search']:.3f} -> {scores['full']['s_search']:.3f}), {row['seconds']:.0f} s")
        del bk
        native.free(nat.dev)


def evaluate(split: str, n: int, src: Path, out: Path, sets=("full", "refined", "b1", "b2", "b4", "rand", "allpos", "vpdbits"),
             offset: int = 0, stride: int = 1, log=print) -> Path:
    """goal_eval.fidelity_report in full mode on the eval split's draws for the source set ("start") and out's sets,
    appended to out/fidelity.jsonl (VPD's top k at each size added by the report)."""
    pool = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"
    fresh = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt"
    have = {"start": src} | {s: out / s for s in sets if (out / s).is_dir()}
    goal_eval.fidelity_report(split, n, have, pool, out / "fidelity.jsonl", fresh=fresh, offset=offset, stride=stride, log=log)
    return out / "fidelity.jsonl"


def table(out: Path) -> str:
    """Per set, medians over the texts of s_nodes, overstate_nodes, phi (goal_eval.phi with the text's start row),
    s_track_pp, s_point_c, c and size_bits, beside the start's; and the paired difference from the start."""
    rows = [json.loads(x) for x in (out / "fidelity.jsonl").read_text().splitlines() if x.strip()]
    start = {r["task"]: r for r in rows if r["kind"] == "start"}
    keys = ("s_nodes", "overstate_nodes", "phi", "s_track_pp", "s_point_c", "c", "size_bits", "nodes")
    by: dict = {}
    for r in rows:
        if r["task"] not in start:
            continue
        r = dict(r)
        r["phi"] = goal_eval.phi(r, start=start[r["task"]])[0]
        kind = r["kind"] if not r["kind"].startswith("VPD top") else "VPD top k"
        by.setdefault(kind, []).append(r)
    lines = ["set".ljust(12) + "".join(k[:10].rjust(12) for k in keys) + "   d_s_nodes  d_phi  texts"]
    for kind, rs in sorted(by.items()):
        med = [np.median([x[k] for x in rs if x.get(k) is not None]) if any(x.get(k) is not None for x in rs) else float("nan") for k in keys]
        ds = [x["s_nodes"] - start[x["task"]]["s_nodes"] for x in rs if x.get("s_nodes") is not None and start[x["task"]].get("s_nodes") is not None]
        dp = [x["phi"] - goal_eval.phi(start[x["task"]], start=start[x["task"]])[0] for x in rs if x.get("phi") is not None]
        lines.append(kind[:12].ljust(12) + "".join(f"{v:12.3f}" if abs(v) < 1e5 else f"{v:12.0f}" for v in med)
                     + f"   {np.median(ds) if ds else float('nan'):+.3f}  {np.median(dp) if dp else float('nan'):+.3f}  {len(rs)}")
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
