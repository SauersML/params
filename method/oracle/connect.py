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
  (3) Drop inert lines. The kept connections in ascending score, then the token edges in ascending score of their
      readers, each by the largest log-scale suffix whose removal still meets the criteria.
Criteria, the cheap families builder_fast's node selection holds (pymove's 10-14): the point KL rises at most POINT_TOL
of the way from the fully wired circuit's to the KL with no line, on a log scale; s_track_pp on TRACK_DRAWS search
draws, s_scope on its scope draws (the family that sees a token edge left out) and s_interchange_pp on the circuit's
panel stay at FLOOR, or at the fully wired circuit's value less TRACK_TOL when that is lower. All are measured on the
engine's runs; the gradients only order the candidates.

  connect(ge, tx, graph, units="vpd", budget_s=None, floors=None)   builder_fast's call: the stated native.Graph
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

ALPHAS = 8  # integrated-gradient strengths, (i + 1/2) / ALPHAS
POINT_TOL = 0.02  # the point KL's allowed rise, as a share of the log distance from the wired circuit's to no line's
TRACK_TOL = 0.02  # s_track_pp's allowed fall below the wired circuit's (lines.VAL_MARGIN's)
TRACK_DRAWS = 8  # s_track_pp's draws in the search (the report uses goal_eval's default)
FAMILY_BATCH = 8  # sets asked about a path-patched family at once
FLOOR = goal_eval.FLOOR_TOP  # the path-patched families' floor (pymove's builder: 0.9 absolute, VPD's whole answer
# scoring 1 there by construction), lowered to the wired circuit's value less TRACK_TOL when that is lower


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
        on = torch.zeros(len(self.pr.lines), dtype=torch.bool)
        on[conn] = True
        on[self.tok if tok is None else tok] = True
        on[self.target] = True
        return on

    @torch.enable_grad()
    def scores(self) -> torch.Tensor:
        """Each line's integrated-gradient score (bits of point KL it saves; 0 for token edges and the target token)."""
        pr, tx, dev = self.pr, self.tx, self.ge.nat.dev
        a = ((torch.arange(ALPHAS, dtype=torch.float32) + 0.5) / ALPHAS).to(dev)
        ew = a[:, None].repeat(1, len(pr.e_ends)).requires_grad_()
        ow = a[:, None].repeat(1, len(pr.o_ends)).requires_grad_()
        reuse = (pr.E.tile_states(pr.states, ALPHAS), tx.ids, pr.lo) if pr.lo else None
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

    FAMILIES = (("track", "s_track_pp"), ("scope", "s_scope"), ("interchange", "s_interchange_pp"))

    def family(self, name: str, ons: list) -> list[float | None]:
        """One path-patched family of each line set on the search split (track on TRACK_DRAWS draws, scope on its
        scope_draws, interchange_pp on each set's panel), FAMILY_BATCH sets a call."""
        ge, tx, out = self.ge, self.tx, []
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

    def run(self, log=print, budget_s: float | None = None, floors: dict | None = None) -> dict:
        """The stated circuit (stages (1) to (3)) and its measurements. floors: {"point": KL cap in bits, "track",
        "scope", "interchange": floors of the families} replacing the defaults (the wired circuit's values less the
        tolerances, the path-patched families' at most FLOOR); a family whose floor is None or whose wired value is
        None (no draws) is not asked. budget_s: the inert stage is skipped once the seconds so far pass it."""
        pr, t0 = self.pr, time.time()
        full = self.row(self.conn)
        kl_full, kl_none = pr.kl_sets(torch.stack([full, self.row(self.conn[:0])]))
        lo, hi = math.log(max(kl_full, 1e-9)), math.log(max(kl_none, kl_full, 1e-9))
        cap = {"point": math.exp(lo + POINT_TOL * (hi - lo))}
        off = {n for n, v in (floors or {}).items() if v is None}  # (families the caller leaves out)
        wired = {name: None if name in off else self.family(name, [full])[0] for name, _ in self.FAMILIES}
        for name, _ in self.FAMILIES:
            cap[name] = None if wired[name] is None else min(FLOOR, wired[name] - TRACK_TOL)
        cap.update(floors or {})
        asked = [name for name, _ in self.FAMILIES if cap.get(name) is not None]
        self.spent["wired"] = time.time() - t0

        t = time.time()
        sc = self.scores()
        order = self.conn[torch.argsort(sc[self.conn], descending=True, stable=True)]
        self.spent["rank"] = time.time() - t

        def passing(ons, last=False) -> list[bool]:
            """Whether each set meets the criteria: the point KL of every set, then the families in turn on the sets
            still in, FAMILY_BATCH sets at a time in order (last: from the end), stopping after the first batch with a
            set that passes (only the first or last passing set is used)."""
            kl = pr.kl_sets(torch.stack(ons))
            out = [False] * len(ons)
            idx = [i for i, x in enumerate(kl) if x <= cap["point"]][::-1 if last else 1]
            for s0 in range(0, len(idx), FAMILY_BATCH):
                alive = idx[s0:s0 + FAMILY_BATCH]
                for name in asked:
                    if not alive:
                        break
                    got = self.family(name, [ons[i] for i in alive])
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

        t = time.time()
        tok = self.tok
        if budget_s is None or time.time() - t0 < budget_s:
            asc = keep.flip(0)  # (the lowest score first)
            ms = [m for m in sizes(len(asc)) if m < len(asc)]
            if ms:
                m = max((m for m, y in zip(ms, passing([self.row(asc[m:]) for m in ms], last=True)) if y), default=0)
                keep = asc[m:].flip(0)
            kept = pr.kept(self.row(keep))
            live = self.tok[kept[pr.ends[self.tok, 0]]]  # the token edges into kept parts (a token edge's one part end)
            into = torch.zeros(len(pr.parts)).index_add_(0, pr.ends[keep, 1].clamp(min=0), sc[keep] * (pr.ends[keep, 1] >= 0))
            tasc = live[torch.argsort(into[pr.ends[live, 0]], stable=True)]  # by their readers' scores, lowest first
            tok = tasc
            ms = [m for m in sizes(len(tasc)) if m < len(tasc)]
            if ms and (budget_s is None or time.time() - t0 < budget_s):
                m = max((m for m, y in zip(ms, passing([self.row(keep, tasc[m:]) for m in ms], last=True)) if y), default=0)
                tok = tasc[m:]
        on = self.row(keep, tok)
        self.spent["inert"] = time.time() - t

        g = pr.graph(on)
        kl_on = pr.kl_sets(on[None])[0]
        fam = {name: self.family(name, [on])[0] for name in asked}
        self.spent["total"] = time.time() - t0
        kinds = [pr.lines[j][0] for j in on.nonzero().flatten().tolist()]
        rec = {"parts_start": len(pr.parts), "lines_start": len(pr.lines), "parts": len(g.node_set()),
               "lines": int(on.sum()), "connections": kinds.count("e") + kinds.count("o"),
               "token_edges": kinds.count("t") + kinds.count("u"), "point": kl_on, "point_wired": kl_full,
               "point_none": kl_none, "families": fam, "families_wired": wired, "criteria": cap,
               "seconds": {k: round(v, 2) for k, v in self.spent.items()}}
        log(f"{rec['parts']} of {rec['parts_start']} parts, {rec['lines']} of {rec['lines_start']} lines "
            f"({rec['token_edges']} token edges); point {kl_on:.4f} bits (wired {kl_full:.4f}, cap {cap['point']:.4f}); "
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


class NativeLines:
    """A circuit of heads and neurons on one text with its lines, under mean ablation (inframove's semantics,
    native_units.MeanUnits): every unit outside the circuit outputs its mean over the reference texts (a head its
    output before o_proj, a neuron its activation), the embedding, the norms and the attention patterns live. A
    reader reads the stream of the text with every unit at its mean (the base) plus the change from its mean of each
    circuit writer it is connected to, at the connection's strength: with every connection stated, the circuit's
    mean-ablated run; a cut connection leaves the reader seeing that writer at its mean. Lines: ("e", writer, reader,
    slot) for each writer below a reader at its position ("in": a neuron's MLP input, the heads of the neuron's own
    layer included; "q": a head's query at its position), and at each position up to the reader's ("k", "v": a head's
    key and value there); ("o", writer) for each circuit unit at a target. on rows are bool [L] over the lines, as
    lines.LinePruner's; kept, kl_sets and graph as there (a unit with no stated path to the prediction is at its mean)."""

    def __init__(self, nat, ids: list[int], targets: list[int], ref: torch.Tensor, means: dict, units: list, B: int = 32):
        m, dev = nat.target, nat.dev
        self.nat, self.ids, self.targets, self.ref, self.B, self.dev = nat, list(ids), list(targets), ref, B, dev
        self.m, self.H, self.hd, self.nl = m, m.n_head, m.hd, m.n_layer
        T = len(ids)
        order = {"head": 0, "neuron": 1}
        self.parts = sorted({tuple(u) for u in units}, key=lambda u: (u[1], order[u[0]], u[2], u[3]))
        at = {u: k for k, u in enumerate(self.parts)}
        self.zh = means["heads"].to(dev).float()  # [layers, H, hd]
        self.za = means["neurons"].to(dev).float()  # [layers, d_mlp]
        with torch.no_grad():  # the base: every unit at its mean
            x = m.embed(torch.tensor([ids], device=dev))[0].float()
            self.base_attn, self.base_mlp, self.bk, self.bv = [], [], [], []
            for L in range(self.nl):
                self.base_attn.append(x)
                h = m.norm(x, 2 * L)
                k = self._proj(L, "k_proj", h).view(T, self.H, self.hd).transpose(0, 1)
                v = self._proj(L, "v_proj", h).view(T, self.H, self.hd).transpose(0, 1)
                self.bk.append(self._rope(k, torch.arange(T, device=dev)))
                self.bv.append(v)
                x = x + self._proj(L, "o_proj", self.zh[L].reshape(1, -1))[0]
                self.base_mlp.append(x)
                x = x + self._proj(L, "down_proj", self.za[L][None])[0]
            self.base_final = x
        lines = []
        for r in self.parts:
            kind, L, t, _ = r
            for w in self.parts:
                kw, Lw, tw, _ = w
                below = Lw < L or (kind == "neuron" and kw == "head" and Lw == L)
                if not below:
                    continue
                if kind == "neuron" and tw == t:
                    lines.append(("e", w, r, "in"))
                elif kind == "head" and tw <= t:
                    if tw == t:
                        lines.append(("e", w, r, "q"))
                    lines += [("e", w, r, "k"), ("e", w, r, "v")]
        lines += [("o", w) for w in self.parts if w[2] in self.targets]
        self.lines = lines
        self.ends = torch.full((len(lines), 2), -1, dtype=torch.long)
        for j, c in enumerate(lines):
            self.ends[j, 0] = at[c[1]]
            if c[0] == "e":
                self.ends[j, 1] = at[c[2]]
        self.e_line = torch.tensor([j for j, c in enumerate(lines) if c[0] == "e"], dtype=torch.long)
        self.o_line = torch.tensor([j for j, c in enumerate(lines) if c[0] == "o"], dtype=torch.long)
        self.e_ends = [(lines[j][1], lines[j][2]) for j in self.e_line.tolist()]
        self.o_ends = [lines[j][1] for j in self.o_line.tolist()]
        self.at = {lines[j]: j for j in range(len(lines))}
        self._index()

    def _site(self, L: int, k: str):
        return self.m.site(f"h.{L}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")

    def _proj(self, L: int, k: str, x: torch.Tensor, rows=None, cols=None) -> torch.Tensor:
        """x times the site's weight (rows: those output rows; cols: those input columns), plus its bias."""
        st = self._site(L, k)
        W = st.W if rows is None else st.W[rows]
        W = W if cols is None else W[:, cols]
        out = x @ W.T
        if st.b is not None:
            out = out + (st.b if rows is None else st.b[rows])
        return out

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rotate-half RoPE of x [..., P, hd] at positions pos [P] (the model's cos and sin)."""
        n = self.hd // 2
        rot = torch.cat([-x[..., n:], x[..., :n]], dim=-1)
        return x * self.m.cos[pos] + rot * self.m.sin[pos]

    def _index(self):
        """Per reader slot the lines into it; per head the positions its key and value lines come from."""
        self.into = {}
        for j in self.e_line.tolist():
            _, w, r, slot = self.lines[j]
            key = (r, slot, w[2] if slot in ("k", "v") else None)
            self.into.setdefault(key, []).append(j)
        self.layers = [([u for u in self.parts if u[1] == L and u[0] == "head"],
                        [u for u in self.parts if u[1] == L and u[0] == "neuron"]) for L in range(self.nl)]
        self.idx = {u: k for k, u in enumerate(self.parts)}
        self.writer_t = self.ends[:, 0].clamp(min=0).to(self.dev)
        tens = lambda js: torch.tensor(js, dtype=torch.long, device=self.dev)  # noqa: E731
        self.groups = []  # per layer its heads' reads and its neurons' (log_probs' tables)
        for L, (heads, nrn) in enumerate(self.layers):
            reads = []
            for r in heads:
                _, _, t, h = r
                srcs = sorted({key[2] for key in self.into if key[0] == r and key[1] in ("k", "v")})
                reads.append((h, t, tens(self.into.get((r, "q", None), [])),
                              [(s, tens(self.into.get((r, "k", s), [])), tens(self.into.get((r, "v", s), []))) for s in srcs]))
            nl, nr = [], []
            for k, r in enumerate(nrn):
                for j in self.into.get((r, "in", None), []):
                    nl.append(j)
                    nr.append(k)
            self.groups.append(({"units": heads, "reads": reads, "at": tens([self.idx[u] for u in heads])},
                                {"units": nrn, "t": tens([u[2] for u in nrn]), "j": tens([u[3] for u in nrn]),
                                 "lines": tens(nl), "reader": tens(nr), "at": tens([self.idx[u] for u in nrn])}))
        self.out_lines = [tens([j for j in self.o_line.tolist() if self.lines[j][1][2] == t]) for t in self.targets]

    def kept(self, on: torch.Tensor) -> torch.Tensor:
        """The units a line subset keeps: those with a stated path to the prediction (bool over self.parts)."""
        on = on.cpu()
        km = torch.zeros(len(self.parts), dtype=torch.bool)
        o = self.o_line[on[self.o_line]]
        km[self.ends[o, 0]] = True
        e = self.e_line[on[self.e_line]]
        w, r = self.ends[e, 0], self.ends[e, 1]
        while True:
            new = km.clone()
            new[w[km[r]]] = True
            if bool((new == km).all()):
                return km
            km = new

    def strengths(self, ons: torch.Tensor):
        """The rows' connection and write strengths with their stranded units' lines off, and the units kept."""
        km = torch.stack([self.kept(on) for on in ons])
        ons = ons.cpu()
        ew = (ons[:, self.e_line] & km[:, self.ends[self.e_line, 0]] & km[:, self.ends[self.e_line, 1]]).float()
        ow = (ons[:, self.o_line] & km[:, self.ends[self.o_line, 0]]).float()
        return ew.to(self.dev), ow.to(self.dev), km.to(self.dev)

    def log_probs(self, ew: torch.Tensor, ow: torch.Tensor, km: torch.Tensor | None = None) -> torch.Tensor:
        """log q at the targets [B, len(targets), V] of each row: connection strengths ew [B, E] (self.e_line's order),
        writes ow [B, O], units kept km bool [B, N] (None: every one; a unit not kept stays at its mean). The units are
        computed a layer's heads, then its neurons, at a time (self.parts' order), each group's changes from the mean
        stacked once it is done; a read gathers its lines' writers from the groups done before it."""
        m, dev, hd = self.m, self.dev, self.hd
        B, d = ew.shape[0], self.base_final.shape[-1]
        eo = torch.zeros(B, len(self.lines), device=dev, dtype=ew.dtype)  # every line's strength, by line index
        eo = eo.index_copy(1, self.e_line.to(dev), ew).index_copy(1, self.o_line.to(dev), ow)
        keep = torch.ones(B, len(self.parts), device=dev) if km is None else km.float()
        groups = []  # [B, n, d] per group done: each unit's change from its mean in the stream

        def into(base: torch.Tensor, js: torch.Tensor) -> torch.Tensor:  # base [d] plus lines js' writers' changes
            x = base.expand(B, d)
            if len(js):
                x = x + (eo[:, js, None] * torch.cat(groups, 1)[:, self.writer_t[js]]).sum(1)
            return x

        for L in range(self.nl):
            heads, nrn = self.groups[L]
            if heads["units"]:
                ch = []
                for r, (h, t, jq, kv) in zip(heads["units"], heads["reads"]):
                    sl = slice(h * hd, (h + 1) * hd)
                    q = self._proj(L, "q_proj", m.norm(into(self.base_attn[L][t], jq), 2 * L), rows=sl)
                    q = self._rope(q[:, None], torch.tensor([t], device=dev))[:, 0]
                    K = self.bk[L][h, :t + 1][None].expand(B, t + 1, hd)
                    V = self.bv[L][h, :t + 1][None].expand(B, t + 1, hd)
                    if kv:
                        Ks, Vs = [], []
                        for src, jk, jv in kv:
                            k = self._proj(L, "k_proj", m.norm(into(self.base_attn[L][src], jk), 2 * L), rows=sl)
                            Ks.append(self._rope(k[:, None], torch.tensor([src], device=dev))[:, 0])
                            Vs.append(self._proj(L, "v_proj", m.norm(into(self.base_attn[L][src], jv), 2 * L), rows=sl))
                        at = torch.tensor([src for src, _, _ in kv], device=dev)
                        K = K.index_copy(1, at, torch.stack(Ks, 1))
                        V = V.index_copy(1, at, torch.stack(Vs, 1))
                    a = ((K @ q[:, :, None])[..., 0] / math.sqrt(hd)).softmax(-1)
                    z = (a[:, :, None] * V).sum(1)
                    ch.append((z - self.zh[L, h]) @ self._site(L, "o_proj").W[:, sl].T)
                groups.append(torch.stack(ch, 1) * keep[:, heads["at"], None])
            if nrn["units"]:
                X = self.base_mlp[L][nrn["t"]][None].expand(B, -1, d)
                if len(nrn["lines"]):
                    X = X.index_add(1, nrn["reader"], eo[:, nrn["lines"], None] * torch.cat(groups, 1)[:, self.writer_t[nrn["lines"]]])
                pre = (m.norm(X, 2 * L + 1) * self._site(L, "c_fc").W[nrn["j"]][None]).sum(-1)
                if self._site(L, "c_fc").b is not None:
                    pre = pre + self._site(L, "c_fc").b[nrn["j"]]
                a = m.act(pre) - self.za[L, nrn["j"]]
                groups.append(a[:, :, None] * self._site(L, "down_proj").W[:, nrn["j"]].T[None] * keep[:, nrn["at"], None])
        out = []
        for t, js in zip(self.targets, self.out_lines):
            out.append(m.unembed(m.norm(into(self.base_final[t], js), 2 * self.nl)).log_softmax(-1))
        return torch.stack(out, 1)

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor) -> list[float]:
        """The point KL (bits at the targets) of each line subset ons [K, L], its stranded units at their means."""
        out = []
        for s0 in range(0, len(ons), self.B):
            ew, ow, km = self.strengths(ons[s0:s0 + self.B])
            out += carry.kl(self.ref, self.log_probs(ew, ow, km)).tolist()
        return out

    def graph(self, on: torch.Tensor) -> NativeGraph:
        """A line subset as the stated circuit: its kept units, their connections and writes."""
        km = self.kept(on)
        units = [u for u, k in zip(self.parts, km.tolist()) if k]
        par = {}
        for j in self.e_line[on.cpu()[self.e_line]].tolist():
            _, w, r, slot = self.lines[j]
            if km[self.idx[w]] and km[self.idx[r]]:
                par.setdefault(r, []).append((w, slot))
        return NativeGraph(units, par, [self.lines[j][1] for j in self.o_line[on.cpu()[self.o_line]].tolist()
                                        if km[self.idx[self.lines[j][1]]]])


class NativeConnector(Connector):
    """Connector's stages over NativeLines (heads and neurons under mean ablation): the point KL alone as the
    criterion until native node selection names its changed-input families (inframove's)."""

    FAMILIES = ()

    def __init__(self, nat, ids: list[int], targets: list[int], ref: torch.Tensor, means: dict, units: list, B: int = 32):
        self.ge, self.tx = None, None
        self.pr = NativeLines(nat, ids, targets, ref, means, units, B)
        self.conn = torch.arange(len(self.pr.lines), dtype=torch.long)
        self.tok = torch.zeros(0, dtype=torch.long)
        self.target = torch.zeros(0, dtype=torch.long)
        self.spent = {}

    @torch.enable_grad()
    def scores(self) -> torch.Tensor:
        """Each line's integrated-gradient score: the point KL it saves (Connector.scores')."""
        pr = self.pr
        a = ((torch.arange(ALPHAS, dtype=torch.float32) + 0.5) / ALPHAS).to(pr.dev)
        ew = a[:, None].repeat(1, len(pr.e_line)).requires_grad_()
        ow = a[:, None].repeat(1, len(pr.o_line)).requires_grad_()
        carry.kl(pr.ref, pr.log_probs(ew, ow)).sum().backward()
        out = torch.zeros(len(pr.lines))
        if ew.numel() and ew.grad is not None:
            out[pr.e_line] = -ew.grad.mean(0).cpu()
        if ow.numel() and ow.grad is not None:
            out[pr.o_line] = -ow.grad.mean(0).cpu()
        return out


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
    a = ap.parse_args()
    teacher_path, task_path, out = map(Path, a.paths)
    ge = goal_eval.GoalEval(native.Native(a.device), chunk=a.batch)
    tx = lines.text_of(ge, task_path, torch.load(a.vpd) if a.vpd else None)
    g0, _, _ = goal_eval.load_answer(teacher_path, json.loads(task_path.read_text()), tx.targets, ge.nat)
    got = Connector(ge, tx, sorted(g0.node_set()), a.batch).run(log=lambda m: print(f"{task_path.stem}: {m}", flush=True))
    out.write_text(json.dumps({"circuit": goal_eval.to_json(got["graph"]), **got["record"]}))


if __name__ == "__main__":
    main()
