"""The fast connection stage (#2951, the builder's; the lead's 10-14): a circuit's units stated as a circuit in
seconds, every part on a stated path to the prediction. The line stage (lines.py) measures every block removal against
the reward's floors and takes about 15 minutes for a 564-part text; this stage ranks the lines once and measures a
handful of nested sets.

Lines: those lines.LinePruner wires (every connection the model has among the parts, the writes into the prediction,
the token edges and the target token), in C's semantics on one engine plan.
  (1) Rank. Each connection and prediction write by its integrated gradient: the point KL's gradient with respect to
      its strength at ALPHAS strengths evenly inside (0, 1), every line at the same strength, averaged (one batched
      backward pass; this finite midpoint quadrature approximates the total KL saved, not an exact identity).
  (2) Keep. The top k lines for k on a log scale (every token edge and the target token beside them), each set's parts
      without a stated path to the prediction dropped (goal_eval.strand), measured: the point KL of every set in one
      batch, then the path-patched families in turn on the smallest sets still in. The stage keeps the smallest set
      that meets every criterion.
  (3) Sparsify (the goal's: VPD's explanations connect too much and reach too many tokens). In turn, each by the
      largest prefix whose removal still meets the criteria: whole positions (every line touching the context
      positions whose lines save the least), cross-token connections (a writer at one position read at another),
      every kept connection, then the token edges (by their readers' scores), lowest score first. With halve, after
      the connections' prefix: the rest in HALVE_BLOCKS blocks, lowest score first, each removed when the set without
      it meets the criteria, else split in two and tried again down to single connections (about k log(n/k) runs for
      k connections needed among n: the misranked ones a prefix search stops at).
The record reports the sparsity (edges between units per unit, cross-token connections, tokens involved) of the stated
circuit beside the wired one's.
Criteria, the cheap families builder_fast's node selection holds (pymove's 10-14): the point KL at most node selection's
point floor (FLOOR of the point score of VPD's actives at the circuit's positions; point_cap); s_track_pp on TRACK_DRAWS search
draws, s_scope on its scope draws (the family that sees a token edge left out), s_interchange_pp on the circuit's
panel and activation patching near and far (donors_search: the families cut connections fail first) stay at FLOOR
plus MARGIN, or at the fully wired circuit's value less TRACK_TOL when that is lower. All are measured on the runs; the
gradients only order the candidates.

  connect(ge, tx, graph, units="vpd", budget_s=None, floors=None)   builder_fast's call: the stated native.Graph
  connect(ge, tx, mask, units="native", means=unit_means(nat))     heads and neurons (NativeDense): a NativeGraph
  connect.py one TEACHER.json TASK.json OUT.json [--vpd FILE] [--device D]   one circuit's stated form, measured
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402
import teacher  # noqa: E402

ALPHAS = 8  # integrated-gradient strengths, (i + 1/2) / ALPHAS
FISHER_SAMPLES = 32  # the second-order score's tokens drawn from the model's prediction
SCORE = "ig"  # the ranking's score (Connector.scores: "ig" or "fisher")
HALVE = False  # the halving pass after the prefix passes (run(halve=))
CHEAP = True  # the passes asked about the cheap families alone, every family checked once at the end (run(cheap=))
CHEAP_FAMILIES = ("track", "scope")  # (pymove's trim: interchange and patching cost the most a set)
HALVE_BLOCKS = 8  # its first blocks: the remaining connections, lowest score first, in this many
POINT_TOL = 0.02  # the point KL's allowed rise, as a share of the log distance from the wired circuit's to no line's
TRACK_TOL = 0.02  # s_track_pp's allowed fall below the wired circuit's (lines.VAL_MARGIN's)
TRACK_DRAWS = 8  # s_track_pp's draws in the search (the report uses goal_eval's default)
FAMILY_BATCH = 8  # sets asked about a path-patched family at once
FLOOR = goal_eval.FLOOR_TOP  # the families' floor (pymove's builder: 0.9 absolute, VPD's whole answer scoring 1 there
# by construction), plus MARGIN on the search draws (the eval split's floor held: pymove's margins), lowered to the wired
# circuit's value less TRACK_TOL when that is lower
MARGIN = {"track": 0.02, "scope": 0.05, "interchange": 0.01, "patching_near": 0.02, "patching_far": 0.02}


def sizes(n: int) -> list[int]:
    """1 to n on a log scale (two steps an octave), n included."""
    return sorted({min(n, int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(n, 1))) + 2)} | {n})


class Connector:
    """One circuit's lines on one text (module docstring). on rows are bool [L] over pr.lines."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, B: int = 32):
        self.ge, self.tx = ge, tx
        self.pr = pr = lines.LinePruner(ge, tx, nodes, B)
        kinds = [c[0] for c in pr.lines]
        self.conn = torch.tensor([j for j, k in enumerate(kinds) if k in ("e", "o")], dtype=torch.long)
        self.tok = torch.tensor([j for j, k in enumerate(kinds) if k == "t"], dtype=torch.long)
        self.target = torch.tensor([j for j, k in enumerate(kinds) if k == "u"], dtype=torch.long)
        self.spent = {}

    def row(self, conn: torch.Tensor, tok: torch.Tensor | None = None) -> torch.Tensor:
        """The lines on: those connections, those token edges (None: every one) and the target token."""
        on = torch.zeros(self.n_lines(), dtype=torch.bool)
        on[conn] = True
        on[self.tok if tok is None else tok] = True
        on[self.target] = True
        return on

    def samples(self, ref: torch.Tensor, seed: int, n: int) -> torch.Tensor:
        """n tokens [n, targets] drawn from the model's prediction at each target (ref [1, targets, V] log
        probabilities), by seed."""
        g = torch.Generator().manual_seed(int(seed))
        return torch.stack([torch.multinomial(ref[0, j].exp().float().cpu(), n, replacement=True, generator=g)
                            for j in range(ref.shape[1])], 1)

    @torch.enable_grad()
    def scores(self, kind: str = "ig", passes: int | None = None) -> torch.Tensor:
        """Each line's score, higher kept first ("ig" or "fisher"; 0 for token edges and the target token, which the
        point run does not see). ig: its integrated gradient, the bits of point KL it saves. fisher: the second-order
        term of the KL at the wired circuit, half the mean squared gradient of log q(y) with respect to its strength over
        tokens y drawn from the model's prediction (the Fisher information's diagonal: the KL its cut adds to second
        order, where the first-order term vanishes as the circuit nears the model; inframove's native unit score).
        passes: the midpoints (ig, default ALPHAS) or the tokens (fisher, default FISHER_SAMPLES), one run each."""
        if kind == "fisher":
            return self._fisher(passes or FISHER_SAMPLES)
        pr, tx, dev = self.pr, self.tx, self.ge.nat.dev
        n = passes or ALPHAS
        a = ((torch.arange(n, dtype=torch.float32) + 0.5) / n).to(dev)
        ew = a[:, None].repeat(1, len(pr.e_ends)).requires_grad_()
        ow = a[:, None].repeat(1, len(pr.o_ends)).requires_grad_()
        reuse = (pr.E.tile_states(pr.states, n), tx.ids, pr.lo) if pr.lo else None
        carry.kl(tx.ref, pr.E.graph_log_probs(pr.plan, ew=ew, ow=ow, reuse=reuse)).sum().backward()
        out = torch.zeros(len(pr.lines))
        # A valid single-writer graph may have no inter-unit edges. Its empty
        # strength tensor is unused, so autograd correctly leaves grad=None.
        if ew.numel():
            out[pr.e_line] = -ew.grad.mean(0).cpu()
        if ow.numel():
            out[pr.o_line] = -ow.grad.mean(0).cpu()
        native.free(dev)
        return out

    @torch.enable_grad()
    def _fisher(self, n: int) -> torch.Tensor:
        """scores("fisher"): per line half the mean over n drawn tokens of (d log q(y) / d strength)^2 at the wired
        circuit, B tokens' runs at a time (each run its own strengths, so each gradient is its token's)."""
        pr, tx, dev = self.pr, self.tx, self.ge.nat.dev
        y = self.samples(tx.ref, tx.seed if tx.seed is not None else 0, n)
        ge_, go_ = torch.zeros(len(pr.e_ends)), torch.zeros(len(pr.o_ends))
        for s0 in range(0, len(y), pr.B):
            yy = y[s0:s0 + pr.B].to(dev)
            K = len(yy)
            ew = torch.ones(K, len(pr.e_ends), device=dev, requires_grad=True)
            ow = torch.ones(K, len(pr.o_ends), device=dev, requires_grad=True)
            reuse = (pr.E.tile_states(pr.states, K), tx.ids, pr.lo) if pr.lo else None
            lq = pr.E.graph_log_probs(pr.plan, ew=ew, ow=ow, reuse=reuse)
            lq.gather(-1, yy[..., None]).sum().backward()
            if ew.grad is not None:
                ge_ += ew.grad.pow(2).sum(0).cpu()
            if ow.grad is not None:
                go_ += ow.grad.pow(2).sum(0).cpu()
            native.free(dev)
        out = torch.zeros(len(pr.lines))
        out[pr.e_line] = 0.5 * ge_ / len(y)
        out[pr.o_line] = 0.5 * go_ / len(y)
        return out

    FAMILIES = (("track", "s_track_pp"), ("scope", "s_scope"), ("interchange", "s_interchange_pp"),
                ("patching_near", "s_patching_near"), ("patching_far", "s_patching_far"))

    def targets(self) -> list[int]:
        return list(self.tx.targets)

    def n_lines(self) -> int:
        return len(self.pr.lines)

    def kind_counts(self, on: torch.Tensor) -> dict:
        """How many lines of each kind a set holds ("e" connections, "o" writes into the prediction, "t" token edges,
        "u" the target token)."""
        out = {"e": 0, "o": 0, "t": 0, "u": 0}
        for j in on.nonzero().flatten().tolist():
            out[self.pr.lines[j][0]] += 1
        return out

    def point_cap(self, kl_full: float, kl_none: float) -> float:
        """The point KL's cap: node selection's point floor (pymove's builder), FLOOR of the point score of VPD's
        actives at the circuit's positions, as a KL on goal_eval's scale between the empty circuit's and VPD's whole
        answer's, KL_empty^(1 - f) KL_VPD^f (teacher.kl_budget), at least the wired circuit's own; without VPD's
        answer, POINT_TOL of the log distance from the wired circuit's KL to no line's."""
        ge, tx = self.ge, self.tx
        e0, v0 = ge._bounds(tx, "live") if tx.vpd is not None else (None, None)
        if v0 is None:
            lo, hi = math.log(max(kl_full, 1e-9)), math.log(max(kl_none, kl_full, 1e-9))
            return math.exp(lo + POINT_TOL * (hi - lo))
        at = {p for _, p, _ in self.pr.parts}
        vp = native.Graph([(n, int(t), int(c)) for n in ge.nat.names for t, c in tx.vpd[n].nonzero().tolist() if int(t) in at],
                          complete=True)
        s_vp = goal_eval._fid(e0, ge.graph_kl(tx, [vp], True)[0], v0)
        if s_vp is None:
            return kl_full
        return max(kl_full, teacher.kl_budget(e0, v0, FLOOR * s_vp))

    def positions(self) -> tuple:
        """Each line's writer and reader positions (a token edge: its token's and its reader's; a write into the
        prediction or the target token: the target's twice) and each unit's position, as tensors."""
        if not hasattr(self, "_pos"):
            at = lambda u: u[2] if u[0] in ("head", "neuron") else u[1]  # noqa: E731  (a native unit; a subcomponent)
            wp, rp = [], []
            for c in self.pr.lines:
                if c[0] == "e":
                    w, r = at(c[1]), at(c[2])
                elif c[0] == "o":
                    w = r = at(c[1])
                elif c[0] == "t":
                    w, r = c[1], at(c[2])
                else:
                    w = r = c[1]
                wp.append(w)
                rp.append(r)
            self._pos = (torch.tensor(wp, dtype=torch.long), torch.tensor(rp, dtype=torch.long),
                         torch.tensor([at(u) for u in self.pr.parts], dtype=torch.long))
        return self._pos

    def without(self, conn: torch.Tensor, tok: torch.Tensor, drop: list) -> tuple:
        """conn and tok less every line touching a position in drop."""
        wp, rp, _ = self.positions()
        d = torch.tensor(drop, dtype=torch.long)
        hit = lambda ls: torch.isin(wp[ls], d) | torch.isin(rp[ls], d)  # noqa: E731
        return conn[~hit(conn)], tok[~hit(tok)] if len(tok) else tok

    def sparsity(self, on: torch.Tensor) -> dict:
        """The connection-sparsity measures of a line set (the goal's: VPD's explanations connect too much and reach
        too many tokens): connections between units per unit kept, cross-token connections (a writer at one position
        read at another: a value read into an attention output), and the positions of the kept units."""
        wp, rp, up = self.positions()
        km = self.pr.kept(on)
        e = torch.tensor([j for j in on.nonzero().flatten().tolist() if self.pr.lines[j][0] == "e"], dtype=torch.long)
        e = e[km[self.pr.ends[e, 0]] & km[self.pr.ends[e, 1]]] if len(e) else e
        n = int(km.sum())
        pos = sorted({int(x) for x in up[km].tolist()})
        return {"edges_per_unit": len(e) / max(n, 1), "cross_token": int((wp[e] != rp[e]).sum()) if len(e) else 0,
                "tokens": len(pos), "positions": pos}

    def family(self, name: str, ons: list) -> list[float | None]:
        """One family of each line set on the search split, FAMILY_BATCH sets a call: track on TRACK_DRAWS draws,
        scope on its scope_draws and interchange_pp on each set's panel (path-patched, line rows), activation patching
        near and far (donors_search on the stated graphs, one call for both pools, kept per set: the families a
        stated circuit's cut connections fail first, text9005's 0.82 and text9012's 0.60 at 0.9 when they went
        unasked)."""
        ge, tx, out = self.ge, self.tx, []
        if name.startswith("patching"):
            seen = self.__dict__.setdefault("_patching", {})
            todo = [on for on in ons if on.numpy().tobytes() not in seen]
            for s0 in range(0, len(todo), FAMILY_BATCH):
                part = todo[s0:s0 + FAMILY_BATCH]
                for on, r in zip(part, ge.donors_search(tx, [self.pr.graph(on) for on in part])):
                    seen[on.numpy().tobytes()] = {"patching_near": r.get("s_patching_near"), "patching_far": r.get("s_patching_far")}
                native.free(ge.nat.dev)
            return [seen[on.numpy().tobytes()][name] for on in ons]
        for s0 in range(0, len(ons), FAMILY_BATCH):
            rows = torch.stack([self.pr.row(on) for on in ons[s0:s0 + FAMILY_BATCH]]).float()
            if name == "track":
                got = [r["s_track_pp"] for r in ge.track_pp(tx, [self.pr.g], "search", draws=TRACK_DRAWS, lines=rows)]
            elif name == "scope":
                got = [r["s_scope"] for r in ge.scope_family(tx, [self.pr.g], "search", lines=rows)]
            else:
                got = [r["s_interchange_pp"] for r in ge.interchange_pp(tx, [self.pr.g], "search", lines=rows)]
            out += got
            native.free(ge.nat.dev)
        return out

    def run(self, log=print, budget_s: float | None = None, floors: dict | None = None, score: str | None = None,
            passes: int | None = None, halve: bool | None = None, cheap: bool | None = None, first: int = 1) -> dict:
        """The stated circuit (stages (1) to (3)) and its measurements. floors: {"point": KL cap in bits, "track",
        "scope", "interchange": floors of the families} replacing the defaults (the wired circuit's values less the
        tolerances, the path-patched families' at most FLOOR); a family whose floor is None or whose wired value is
        None (no draws) is not asked. budget_s: the inert stage is skipped once the seconds so far pass it."""
        pr, t0 = self.pr, time.time()
        full = self.row(self.conn)
        kl_full, kl_none = pr.kl_sets(torch.stack([full, self.row(self.conn[:0])]))
        cap = {"point": self.point_cap(kl_full, kl_none)}
        off = {n for n, v in (floors or {}).items() if v is None}  # (families the caller leaves out)
        wired = {name: None if name in off else self.family(name, [full])[0] for name, _ in self.FAMILIES}
        for name, _ in self.FAMILIES:
            cap[name] = None if wired[name] is None else min(FLOOR + MARGIN.get(name, 0.0), wired[name] - TRACK_TOL)
        cap.update(floors or {})
        asked = [name for name, _ in self.FAMILIES if cap.get(name) is not None]
        cheap = CHEAP if cheap is None else cheap  # (the passes on the cheap families, every family once at the end)
        searched = [f for f in asked if f in CHEAP_FAMILIES] if cheap else asked
        self.spent["wired"] = time.time() - t0

        t = time.time()
        score = score or SCORE
        halve = HALVE if halve is None else halve
        sc = self.scores(score, passes)
        order = self.conn[torch.argsort(sc[self.conn], descending=True, stable=True)]
        self.spent["rank"] = time.time() - t
        measured = {"point": 0, "families": 0}  # the sets the stages measure (each point run, each family run)

        def passing(ons, last=False, fams=None) -> list[bool]:
            """Whether each set meets the criteria: the point KL of every set, then the families (fams; default the
            searched ones) in turn on the sets still in, in order (last: from the end), first `first` sets then
            batches doubling to FAMILY_BATCH, stopping after the first batch with a set that passes (only the first or
            last passing set is used)."""
            fams = searched if fams is None else fams
            kl = pr.kl_sets(torch.stack(ons))
            measured["point"] += len(ons)
            out = [False] * len(ons)
            idx = [i for i, x in enumerate(kl) if x <= cap["point"]][::-1 if last else 1]
            s0, width = 0, first  # (the most promising set alone first, then batches doubling to FAMILY_BATCH: a family
            while s0 < len(idx):  # run costs about its sets, and the first set usually decides)
                alive = idx[s0:s0 + width]
                s0, width = s0 + width, min(2 * width, FAMILY_BATCH)
                for name in fams:
                    if not alive:
                        break
                    got = self.family(name, [ons[i] for i in alive])
                    measured["families"] += len(alive)
                    alive = [i for i, x in zip(alive, got) if x is not None and x >= cap[name] - 1e-9]
                for i in alive:
                    out[i] = True
                if alive:
                    break
            return out

        t = time.time()
        ks = sizes(len(order))
        k = next((k for k, y in zip(ks, passing([self.row(order[:k]) for k in ks])) if y), len(order))
        keep = order[:k]
        self.spent["keep"] = time.time() - t
        states = [("wired", full), ("keep", self.row(keep))]  # (the set after each pass: the final check's backoff)

        t = time.time()
        wp, rp, up = self.positions()
        tok = self.tok
        dropped, halving = [], None
        if budget_s is None or time.time() - t0 < budget_s:
            # (3a) whole positions: the units at the positions whose lines save the least, dropped with every line
            # touching them, the most such positions that still meet the criteria (fewer tokens in the explanation)
            kept = pr.kept(self.row(keep))
            cand = sorted({int(x) for x in up[kept].tolist()} - set(self.targets()))
            if cand:
                save = {q: float(sc[keep][(wp[keep] == q) | (rp[keep] == q)].sum()) for q in cand}
                cand.sort(key=lambda q: save[q])
                ms = list(range(1, len(cand) + 1)) if len(cand) <= 16 else sizes(len(cand))
                rows = [self.row(*self.without(keep, tok, cand[:m])) for m in ms]
                m = max((m for m, y in zip(ms, passing(rows, last=True)) if y), default=0)
                dropped = cand[:m]
                keep, tok = self.without(keep, tok, dropped)
            states.append(("positions", self.row(keep, tok)))
            self.spent["positions"] = time.time() - t
            # (3b) cross-token connections (a writer at one position read at another), lowest score first
            tc = time.time()
            cross = keep[wp[keep] != rp[keep]]
            if len(cross):
                asc = cross[torch.argsort(sc[cross], stable=True)]
                ms = sizes(len(asc))
                rest = lambda m: keep[~torch.isin(keep, asc[:m])]  # noqa: E731
                m = max((m for m, y in zip(ms, passing([self.row(rest(m), tok) for m in ms], last=True)) if y), default=0)
                keep = rest(m)
            states.append(("cross", self.row(keep, tok)))
            self.spent["cross"] = time.time() - tc
        if budget_s is None or time.time() - t0 < budget_s:
            # (3c) every kept connection, lowest score first
            tc = time.time()
            asc = keep[torch.argsort(sc[keep], stable=True)]
            ms = [m for m in sizes(len(asc)) if m < len(asc)]
            if ms:
                m = max((m for m, y in zip(ms, passing([self.row(asc[m:], tok) for m in ms], last=True)) if y), default=0)
                keep = asc[m:]
            states.append(("connections", self.row(keep, tok)))
            self.spent["connections"] = time.time() - tc
            # (3c') halving: what the prefixes stop at (the first misranked connection ends a prefix search)
            if halve and len(keep) > 1 and (budget_s is None or time.time() - t0 < budget_s):
                th, before, runs0 = time.time(), len(keep), measured["point"]
                asc = keep[torch.argsort(sc[keep], stable=True)]
                queue = list(asc.tensor_split(min(HALVE_BLOCKS, len(asc))))
                gone = torch.zeros(self.n_lines(), dtype=torch.bool)
                while queue and (budget_s is None or time.time() - t0 < budget_s):
                    b = queue.pop(0)
                    trial = gone.clone()
                    trial[b] = True
                    if passing([self.row(keep[~trial[keep]], tok)])[0]:
                        gone = trial
                    elif len(b) > 1:
                        queue[:0] = list(b.tensor_split(2))
                keep = keep[~gone[keep]]
                states.append(("halving", self.row(keep, tok)))
                halving = {"before": before, "after": len(keep), "runs": measured["point"] - runs0,
                           "seconds": round(time.time() - th, 2)}
            # (3d) token edges into kept parts, by their readers' scores, lowest first
            tc = time.time()
            kept = pr.kept(self.row(keep, tok))
            live = tok[kept[pr.ends[tok, 0]]] if len(tok) else tok
            into = torch.zeros(len(pr.parts)).index_add_(0, pr.ends[keep, 1].clamp(min=0), sc[keep] * (pr.ends[keep, 1] >= 0))
            tasc = live[torch.argsort(into[pr.ends[live, 0]], stable=True)]
            tok = tasc
            ms = [m for m in sizes(len(tasc)) if m < len(tasc)]
            if ms and (budget_s is None or time.time() - t0 < budget_s):
                m = max((m for m, y in zip(ms, passing([self.row(keep, tasc[m:]) for m in ms], last=True)) if y), default=0)
                tok = tasc[m:]
            self.spent["tokens"] = time.time() - tc
        on = self.row(keep, tok)
        self.spent["inert"] = time.time() - t
        # the final check: every family once on the stated set, else back to the last pass's set that meets them all
        backed = None
        rest_fams = [f for f in asked if f not in searched]
        if rest_fams:
            tf = time.time()
            if not passing([on], fams=asked)[0]:
                lo, hi = 0, len(states)  # states[0] (wired) meets every criterion by construction; the stated set none
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if passing([states[mid][1]], fams=asked)[0]:
                        lo = mid
                    else:
                        hi = mid
                backed, on = states[lo]
            self.spent["final"] = time.time() - tf

        g = pr.graph(on)
        kl_on = pr.kl_sets(on[None])[0]
        fam = {name: self.family(name, [on])[0] for name in asked}
        self.spent["total"] = time.time() - t0
        kinds = self.kind_counts(on)
        rec = {"parts_start": len(pr.parts), "lines_start": self.n_lines(), "parts": len(g.node_set()),
               "lines": int(on.sum()), "connections": kinds["e"] + kinds["o"],
               "token_edges": kinds["t"] + kinds["u"], "point": kl_on, "point_wired": kl_full,
               "point_none": kl_none, "families": fam, "families_wired": wired, "criteria": cap,
               **self.sparsity(on), "wired": self.sparsity(full), "positions_dropped": dropped, "score": score, "passes": passes,
               "halving": halving, "cheap": cheap, "backed_to": backed,
               "measured": measured,
               "seconds": {k: round(v, 2) for k, v in self.spent.items()}}
        log(f"{rec['parts']} of {rec['parts_start']} parts, {rec['lines']} of {rec['lines_start']} lines "
            f"({rec['token_edges']} token edges); edges per unit {rec['edges_per_unit']:.2f} (wired "
            f"{rec['wired']['edges_per_unit']:.2f}), cross-token {rec['cross_token']} ({rec['wired']['cross_token']}), "
            f"tokens {rec['tokens']} ({rec['wired']['tokens']}); point {kl_on:.4f} bits (wired {kl_full:.4f}, cap {cap['point']:.4f}); "
            + ", ".join(f"{n} {fam[n]:.3f} (wired {wired[n]:.3f}, floor {cap[n]:.3f})" for n in asked) + f"; {rec['seconds']}")
        return {"graph": g, "on": on, "record": rec}


class NativeGraph:
    """A stated circuit in the model's own units (native_units.Mask.units()' tuples ("head" | "neuron", layer,
    position, index)): parents {reader: [(writer, slot)]} (slot "q", "k", "v" into a head, "in" into a neuron), out
    [writers into the prediction]."""

    complete = False

    def __init__(self, units: list, parents: dict, out: list):
        self.units, self.parents, self.out = list(units), parents, list(out)

    def node_set(self) -> set:
        return set(self.units)

    def to_json(self) -> dict:
        return {"units": [list(u) for u in self.units],
                "edges": [[list(w), list(r), slot] for r, ws in self.parents.items() for w, slot in ws],
                "out": [list(w) for w in self.out]}


class NativeDense:
    """A circuit of heads and neurons on one text with its lines, under mean ablation (inframove's semantics,
    native_units.MeanUnits): every unit outside the circuit outputs its mean over the reference texts (a head its
    output before o_proj, a neuron its activation), the embedding, the norms and the attention patterns live; the text
    cut at its last target. A reader reads the stream with every unit at its mean (the base) plus the change from its
    mean of what it is connected to, at the connection's strength; with every line on it is the circuit's mean-ablated
    run, with a line cut the reader sees that change at the mean. Built for builder circuits of tens of thousands of
    units (text9031: 33k at 96 positions), every line a strength in one vector, the runs blocked by layer, kind and
    position. Lines:
      same-position  ("e") a unit into a unit above it at its position: a neuron's MLP input (the heads of its own
                     layer included), a head's query; each reader block's lines are its readers times the units below
                     it there, reader-major;
      cross-position ("e", a bundle) a head reading the keys and values at a position up to its own: the change from
                     the mean of every circuit unit below its layer there, at the line's strength (a head's own position
                     included; where nothing is below, the base alone and no line);
      prediction     ("o") a unit at a target into the prediction there.
    on rows are bool [n] over the lines; kept, kl_sets, graph and sparsity as the VPD path's."""

    def __init__(self, nat, ids: list[int], targets: list[int], ref: torch.Tensor, means: dict, units: list, B: int = 8):
        m, dev = nat.target, nat.dev
        self.nat, self.m, self.dev, self.B = nat, m, dev, B
        self.T = max(targets) + 1
        self.ids, self.targets, self.ref = list(ids)[:self.T], list(targets), ref
        self.H, self.hd, self.nl = m.n_head, m.hd, m.n_layer
        kind = {"head": 0, "neuron": 1}
        us = sorted({tuple(u) for u in units if u[2] < self.T}, key=lambda u: (u[2], u[1], kind[u[0]], u[3]))
        self.parts, self.U = us, len(us)
        self.upos = torch.tensor([u[2] for u in us], dtype=torch.long)
        lay = torch.tensor([u[1] for u in us], dtype=torch.long)
        knd = torch.tensor([kind[u[0]] for u in us], dtype=torch.long)
        self.uidx = torch.tensor([u[3] for u in us], dtype=torch.long)
        self.zh = means["heads"].to(dev).float()  # [layers, H, hd]
        self.za = means["neurons"].to(dev).float()  # [layers, d_mlp]
        W = lambda L, k: m.site(native.site_name(L, k)).W  # noqa: E731
        H, hd = self.H, self.hd
        self.Wq = [W(L, "q_proj").view(H, hd, -1) for L in range(self.nl)]  # [H, hd, d]
        self.Wk = [W(L, "k_proj").view(H, hd, -1) for L in range(self.nl)]
        self.Wv = [W(L, "v_proj").view(H, hd, -1) for L in range(self.nl)]
        self.Wo = [W(L, "o_proj").view(-1, H, hd).permute(1, 0, 2) for L in range(self.nl)]  # [H, d, hd]
        self.Win = [W(L, "c_fc") for L in range(self.nl)]  # [d_mlp, d]
        self.Wout = [W(L, "down_proj") for L in range(self.nl)]  # [d, d_mlp]
        with torch.no_grad():  # the base: every unit at its mean
            x = m.embed(torch.tensor([self.ids], device=dev))[0].float()
            self.base_attn, self.base_mlp = [], []
            for L in range(self.nl):
                self.base_attn.append(x)
                x = x + (self.zh[L].reshape(1, -1) @ W(L, "o_proj").T)[0]
                self.base_mlp.append(x)
                x = x + (self.za[L][None] @ W(L, "down_proj").T)[0]
            self.base_final = x
        # blocks (layer, kind, position): readers units[r0:r1], the units below them there units[w0:r0]
        pos = sorted(set(self.upos.tolist()))
        start = {p: int((self.upos < p).sum()) for p in pos}
        self.blocks = {(L, k): [] for L in range(self.nl) for k in (0, 1)}
        same_r, same_w, kv_h, kv_p, kv_lo, kv_hi = [], [], [], [], [], []
        n = 0
        below = {}  # (position, layer): how many units there sit below that layer
        for p in pos:
            sel = self.upos == p
            for L in range(self.nl + 1):
                below[(p, L)] = int((sel & (lay < L)).sum())
        for L in range(self.nl):
            for k in (0, 1):
                for p in pos:
                    sel = ((self.upos == p) & (lay == L) & (knd == k)).nonzero().flatten()
                    if not len(sel):
                        continue
                    r0, r1, w0 = int(sel[0]), int(sel[-1]) + 1, start[p]
                    nr, nw = r1 - r0, r0 - w0
                    seg_e = (n, n + nr * nw)
                    if nw:
                        rr = torch.arange(r0, r1).repeat_interleave(nw)
                        ww = torch.arange(w0, r0).repeat(nr)
                        same_r.append(rr)
                        same_w.append(ww)
                    n += nr * nw
                    srcs = []
                    if k == 0:
                        srcs = [q for q in pos if q <= p and below[(q, L)] > 0]
                        for r in range(r0, r1):
                            for q in srcs:
                                kv_h.append(r)
                                kv_p.append(q)
                                kv_lo.append(start[q])
                                kv_hi.append(start[q] + below[(q, L)])
                    seg_kv = (n, n + nr * len(srcs))
                    n += nr * len(srcs)
                    self.blocks[(L, k)].append((p, r0, r1, w0, seg_e, seg_kv, srcs))
        self.n_same_kv = n
        self.out_seg = {}
        for t in self.targets:
            cnt = int((self.upos == t).sum())
            self.out_seg[t] = (n, n + cnt, start.get(t, 0))
            n += cnt
        self.n = n
        cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0, dtype=torch.long)  # noqa: E731
        # every line's ends (writer, reader; -1 for a bundle's writers or the prediction) and kind
        self.kind = torch.zeros(n, dtype=torch.long)
        self.ends = torch.full((n, 2), -1, dtype=torch.long)
        self.wpos = torch.zeros(n, dtype=torch.long)
        self.rpos = torch.zeros(n, dtype=torch.long)
        same = torch.zeros(n, dtype=torch.bool)
        kvl = torch.zeros(n, dtype=torch.bool)
        for (L, k), bl in self.blocks.items():
            for p, r0, r1, w0, seg_e, seg_kv, srcs in bl:
                same[seg_e[0]:seg_e[1]] = True
                kvl[seg_kv[0]:seg_kv[1]] = True
        self.same_ids, self.kv_ids = same.nonzero().flatten(), kvl.nonzero().flatten()
        sr, sw = cat(same_r), cat(same_w)
        self.ends[self.same_ids, 0], self.ends[self.same_ids, 1] = sw, sr
        self.wpos[self.same_ids], self.rpos[self.same_ids] = self.upos[sw], self.upos[sr]
        self.kv_h, self.kv_lo, self.kv_hi = (torch.tensor(x, dtype=torch.long) for x in (kv_h, kv_lo, kv_hi))
        self.ends[self.kv_ids, 1] = self.kv_h
        self.wpos[self.kv_ids] = torch.tensor(kv_p, dtype=torch.long)
        self.rpos[self.kv_ids] = self.upos[self.kv_h] if len(kv_h) else self.rpos[self.kv_ids]
        self.out_ids = torch.arange(self.n_same_kv, n)
        ou = cat([torch.arange(st, st + (b - a)) for a, b, st in self.out_seg.values()])
        self.kind[self.out_ids] = 1
        self.ends[self.out_ids, 0] = ou
        self.wpos[self.out_ids] = self.upos[ou] if len(ou) else self.wpos[self.out_ids]
        self.rpos[self.out_ids] = self.wpos[self.out_ids]
        self.out_u = ou

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rotate-half RoPE of x [..., P, hd] at positions pos [P] (the model's cos and sin)."""
        n = self.hd // 2
        rot = torch.cat([-x[..., n:], x[..., :n]], dim=-1)
        return x * self.m.cos[pos] + rot * self.m.sin[pos]

    def kept(self, on: torch.Tensor) -> torch.Tensor:
        """The units a line subset keeps: those with a stated path to the prediction (bool over self.parts)."""
        on = on.cpu()
        km = torch.zeros(self.U, dtype=torch.bool)
        km[self.out_u[on[self.out_ids]]] = True
        s_on = on[self.same_ids]
        sr, sw = self.ends[self.same_ids, 1][s_on], self.ends[self.same_ids, 0][s_on]
        k_on = on[self.kv_ids]
        kh, klo, khi = self.kv_h[k_on], self.kv_lo[k_on], self.kv_hi[k_on]
        while True:
            new = km.clone()
            new[sw[km[sr]]] = True
            hit = km[kh]
            if bool(hit.any()):
                d = torch.zeros(self.U + 1, dtype=torch.long)
                d.index_add_(0, klo[hit], torch.ones(int(hit.sum()), dtype=torch.long))
                d.index_add_(0, khi[hit], -torch.ones(int(hit.sum()), dtype=torch.long))
                new |= d.cumsum(0)[:-1] > 0
            if bool((new == km).all()):
                return km
            km = new

    def log_probs(self, s: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """log q at the targets [B, len(targets), V] of each row: line strengths s [B, n], units kept keep [B, U] (None:
        every one; a unit not kept stays at its mean)."""
        m, dev, hd = self.m, self.dev, self.hd
        B, d = s.shape[0], self.base_final.shape[-1]
        s = s.to(dev)
        keep = None if keep is None else keep.to(dev).float()
        done = {}  # position -> its blocks' changes from the mean, in unit order
        C = {}  # position -> the sum of its units' changes below the current layer [B, d]
        zero = torch.zeros(B, d, device=dev)
        for L in range(self.nl):
            CL = dict(C)
            for k in (0, 1):
                for p, r0, r1, w0, seg_e, seg_kv, srcs in self.blocks[(L, k)]:
                    nr, nw = r1 - r0, r0 - w0
                    X = (self.base_attn[L] if k == 0 else self.base_mlp[L])[p].expand(B, nr, d)
                    if nw:
                        X = X + s[:, seg_e[0]:seg_e[1]].view(B, nr, nw) @ torch.cat(done[p], 1)
                    idx = self.uidx[r0:r1].to(dev)
                    if k == 0:
                        q = torch.einsum("brd,rhd->brh", m.norm(X, 2 * L), self.Wq[L][idx])
                        q = self._rope(q, torch.tensor([p], device=dev))
                        sk = torch.zeros(B, nr, p + 1, device=dev)
                        if srcs:
                            sk = sk.index_copy(2, torch.tensor(srcs, device=dev), s[:, seg_kv[0]:seg_kv[1]].view(B, nr, len(srcs)))
                        Cs = torch.stack([CL.get(q_, zero) for q_ in range(p + 1)], 1)  # [B, p + 1, d]
                        Xk = m.norm(self.base_attn[L][:p + 1][None, None] + sk[..., None] * Cs[:, None], 2 * L)
                        kk = self._rope(torch.einsum("brpd,rhd->brph", Xk, self.Wk[L][idx]), torch.arange(p + 1, device=dev))
                        vv = torch.einsum("brpd,rhd->brph", Xk, self.Wv[L][idx])
                        a = (torch.einsum("brh,brph->brp", q, kk) / math.sqrt(hd)).softmax(-1)
                        z = torch.einsum("brp,brph->brh", a, vv) - self.zh[L][idx]
                        ch = torch.einsum("brh,rdh->brd", z, self.Wo[L][idx])
                    else:
                        pre = torch.einsum("brd,rd->br", m.norm(X, 2 * L + 1), self.Win[L][idx])
                        ch = (m.act(pre) - self.za[L][idx])[..., None] * self.Wout[L][:, idx].T[None]
                    if keep is not None:
                        ch = ch * keep[:, r0:r1, None]
                    done.setdefault(p, []).append(ch)
                    C[p] = C.get(p, zero) + ch.sum(1)
        out = []
        for t in self.targets:
            a, b, _ = self.out_seg[t]
            x = self.base_final[t].expand(B, d)
            if b > a:
                x = x + torch.einsum("bn,bnd->bd", s[:, a:b], torch.cat(done[t], 1))
            out.append(m.unembed(m.norm(x, 2 * self.nl)).log_softmax(-1))
        return torch.stack(out, 1)

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor) -> list[float]:
        """The point KL (bits at the targets) of each line subset ons [K, n], its stranded units at their means."""
        out = []
        for s0 in range(0, len(ons), self.B):
            part = ons[s0:s0 + self.B]
            km = torch.stack([self.kept(on) for on in part])
            out += carry.kl(self.ref, self.log_probs(part.float(), km)).tolist()
            native.free(self.dev)
        return out

    def graph(self, on: torch.Tensor) -> NativeGraph:
        """A line subset as the stated circuit: its kept units, their connections (a bundle's writer as ("position", p),
        slot "kv") and writes."""
        on = on.cpu()
        km = self.kept(on)
        units = [u for u, k in zip(self.parts, km.tolist()) if k]
        par = {}
        si = self.same_ids[on[self.same_ids]]
        for w, r in zip(self.ends[si, 0].tolist(), self.ends[si, 1].tolist()):
            if km[w] and km[r]:
                par.setdefault(self.parts[r], []).append((self.parts[w], "q" if self.parts[r][0] == "head" else "in"))
        ki = self.kv_ids[on[self.kv_ids]]
        for h, p in zip(self.ends[ki, 1].tolist(), self.wpos[ki].tolist()):
            if km[h]:
                par.setdefault(self.parts[h], []).append((("position", p), "kv"))
        oi = self.out_ids[on[self.out_ids]]
        return NativeGraph(units, par, [self.parts[u] for u in self.ends[oi, 0].tolist() if km[u]])


class NativeConnector(Connector):
    """Connector's stages over NativeDense (heads and neurons under mean ablation): the point KL alone as the
    criterion until native node selection names its changed-input families (inframove's)."""

    FAMILIES = ()

    def __init__(self, nat, ids: list[int], targets: list[int], ref: torch.Tensor, means: dict, units: list, B: int = 8):
        self.ge, self.tx = None, None
        self.pr = NativeDense(nat, ids, targets, ref, means, units, B)
        self.conn = torch.arange(self.pr.n, dtype=torch.long)
        self.tok = torch.zeros(0, dtype=torch.long)
        self.target = torch.zeros(0, dtype=torch.long)
        self.spent = {}

    def n_lines(self) -> int:
        return self.pr.n

    def kind_counts(self, on: torch.Tensor) -> dict:
        k = self.pr.kind[on.cpu()]
        return {"e": int((k == 0).sum()), "o": int((k == 1).sum()), "t": 0, "u": 0}

    def targets(self) -> list[int]:
        return list(self.pr.targets)

    def positions(self) -> tuple:
        return self.pr.wpos, self.pr.rpos, self.pr.upos

    def sparsity(self, on: torch.Tensor) -> dict:
        """Connector.sparsity's measures: connections between units (same-position lines with both ends kept, and
        bundles into kept heads) per unit kept, the cross-position bundles (a head reading another position), and the
        positions of the kept units."""
        pr, on = self.pr, on.cpu()
        km = pr.kept(on)
        si = pr.same_ids[on[pr.same_ids]]
        si = si[km[pr.ends[si, 0]] & km[pr.ends[si, 1]]]
        ki = pr.kv_ids[on[pr.kv_ids]]
        ki = ki[km[pr.ends[ki, 1]]]
        n = int(km.sum())
        pos = sorted({int(x) for x in pr.upos[km].tolist()})
        return {"edges_per_unit": (len(si) + len(ki)) / max(n, 1), "cross_token": int((pr.wpos[ki] != pr.rpos[ki]).sum()),
                "tokens": len(pos), "positions": pos}

    def point_cap(self, kl_full: float, kl_none: float) -> float:
        """POINT_TOL of the log distance from the wired circuit's KL to no line's (native units have no VPD answer
        to scale against; inframove's native floors replace it when they exist)."""
        lo, hi = math.log(max(kl_full, 1e-9)), math.log(max(kl_none, kl_full, 1e-9))
        return math.exp(lo + POINT_TOL * (hi - lo))

    @torch.enable_grad()
    def _fisher(self, n: int) -> torch.Tensor:
        """Connector.scores("fisher") on NativeDense: B drawn tokens' runs at a time, each its own strengths."""
        pr = self.pr
        y = self.samples(pr.ref, sum(pr.ids) % (2 ** 31), n)
        acc = torch.zeros(pr.n)
        for s0 in range(0, len(y), pr.B):
            yy = y[s0:s0 + pr.B].to(pr.dev)
            s = torch.ones(len(yy), pr.n, device=pr.dev, requires_grad=True)
            pr.log_probs(s).gather(-1, yy[..., None]).sum().backward()
            acc += s.grad.pow(2).sum(0).cpu()
            native.free(pr.dev)
        return 0.5 * acc / len(y)

    @torch.enable_grad()
    def scores(self, kind: str = "ig", passes: int | None = None) -> torch.Tensor:
        """Connector.scores' on NativeDense: ig, its midpoints B at a time; fisher, _fisher."""
        if kind == "fisher":
            return self._fisher(passes or FISHER_SAMPLES)
        pr, grads = self.pr, []
        n = passes or ALPHAS
        a = (torch.arange(n, dtype=torch.float32) + 0.5) / n
        for s0 in range(0, n, pr.B):
            s = a[s0:s0 + pr.B, None].repeat(1, pr.n).to(pr.dev).requires_grad_()
            carry.kl(pr.ref, pr.log_probs(s)).sum().backward()
            grads.append(s.grad.sum(0).cpu())
            native.free(pr.dev)
        return -torch.stack(grads).sum(0) / n


def connect(ge: goal_eval.GoalEval, tx: goal_eval.Text, graph, units: str = "vpd", budget_s: float | None = None,
            floors: dict | None = None, log=None, means: dict | None = None):
    """builder_fast's call (pymove's interface): a circuit's node set stated, every part on a stated path; the stage's
    record on the result as .connect. units "vpd": graph a complete native.Graph over (site, position, subcomponent),
    the result a stated native.Graph; "native": graph the heads and neurons (native_units.Mask, or its units()' tuples),
    under mean ablation with means {"heads": [layers, H, hd], "neurons": [layers, d_mlp]} (native_units.unit_means'),
    the result a NativeGraph."""
    if units == "native":
        us = graph.units() if hasattr(graph, "units") else list(graph)
        if means is None:
            raise ValueError("connect: native units need the units' means (native_units.unit_means)")
        st = NativeConnector(ge.nat, tx.ids, tx.targets, tx.ref, means, us)
        got = st.run(log=log or (lambda m: None), budget_s=budget_s, floors=floors)
        got["graph"].connect = got["record"]
        return got["graph"]
    if units != "vpd":
        raise ValueError("connect: units 'vpd' or 'native'")
    got = Connector(ge, tx, sorted(graph.node_set())).run(log=log or (lambda m: None), budget_s=budget_s, floors=floors)
    g = got["graph"]
    g.connect = got["record"]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("one",))
    ap.add_argument("paths", nargs=3, help="TEACHER.json TASK.json OUT.json")
    ap.add_argument("--vpd", type=Path, help="VPD's answers and orders (teacher.py --save-vpd)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", help="cuda, mps or cpu (default: native.device()'s choice)")
    ap.add_argument("--score", choices=("ig", "fisher"), help=f"the ranking's score (default {SCORE})")
    ap.add_argument("--halve", action="store_true", help="the halving pass after the connections' prefix")
    ap.add_argument("--passes", type=int, help=f"the score's runs (ig midpoints, default {ALPHAS}; fisher tokens, {FISHER_SAMPLES})")
    a = ap.parse_args()
    teacher_path, task_path, out = map(Path, a.paths)
    ge = goal_eval.GoalEval(native.Native(a.device), chunk=a.batch)
    tx = lines.text_of(ge, task_path, torch.load(a.vpd) if a.vpd else None)
    g0, _, _ = goal_eval.load_answer(teacher_path, json.loads(task_path.read_text()), tx.targets, ge.nat)
    got = Connector(ge, tx, sorted(g0.node_set()), a.batch).run(log=lambda m: print(f"{task_path.stem}: {m}", flush=True),
                                                                 score=a.score, passes=a.passes, halve=a.halve or None)
    out.write_text(json.dumps({"circuit": goal_eval.to_json(got["graph"]), **got["record"]}))


if __name__ == "__main__":
    main()
