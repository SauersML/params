"""The masked-run engine (#2951 graph oracle): VPD's masked forward pass of vpd4l on one text with per-site masks over
a chosen set of subcomponents, the remainders W - V U live or off, and the KL in bits of the model's next-token
distribution at the targets from the run's, differentiable in the masks. It serves searches that take gradients with
respect to masks (the teacher's mask search, the reward's adversary), batched over rows (samples x patterns x
answers x restarts), on any GPU.

  eng = Engine(nat, ids, targets, live=True, columns=None, compile=False, outside="off")
  eng.kl(masks, precision="score") -> [B] bits                (eng.log_probs: [B, len(targets), V])
  eng.graph_kl(eng.plan(graph), u=None, ew=None, ow=None, precision="score") -> [B] bits   (eng.graph_log_probs)

columns: per site the subcomponent indices masks address (a LongTensor), every other one off in every run; None, every
subcomponent. Give the union a caller's masks can ever switch on (VPD's answer and an adversary's patterns): a site's
products then run on those columns alone (a sum over fewer terms: float32-close to the full products, not equal to
the bit). masks: {site: [B, T, K]} floats (K = the site's columns), with or without gradients. outside="on" runs every
subcomponent outside the columns at 1 instead (folded into the remainder: W - V_K U_K, or V U - V_K U_K with the
remainders off): the whole model with a few subcomponents' masks free, as cheap as the model itself.
precision: "score", float32 with vpd_model's own norms and GELU (the numbers to report); "tf32" (CUDA: TF32 tensor-core
matmuls) and "bf16" (bfloat16 products of weights cast once), with torch's fused norms and GELU, for a search's steps only, whose result is
then scored again with "score" (engine.rescore checks that). compile: the search precisions' forward under
torch.compile (one graph per batch size: keep B fixed in a search; on an RTX A5000 bf16 forward+backward 76 -> 48 ms
at B = 32); "score" stays eager.

graph_kl is native.Native.run's graph semantics (graphs of stated connections, native.Graph, not complete): a graph
node's output reaches a reader through a stated connection of strength e at e + (1 - e) u, every other connection at
its writer's u, a query or key node enters the attention pattern at 1, the token embedding in full, the remainders live
or off. plan: Engine.plan of one graph, run on B rows, or of a list of B graphs, one per row (answers x interventions
x restarts). u: {site: [B or 1, T or 1, K]} over the engine's columns (None: 0, every subcomponent outside the graph
off; a column outside the engine's is always 0, so the graphs' nodes must be among them). ew, ow: the strengths of
the connections (each graph's parents, reader by reader in its writers' order) and of those into the prediction (its
out list), [B, E] and [B, O] for one graph, flat in graph order for a list (None: 1); differentiable in u, ew and ow.
Float32-close to native.run (attention by torch's fused kernel, the pattern computed apart only at attention-output
readers' positions). Engine.plan(graphs, semantics="c") runs C's semantics instead (goal_eval.GoalEval.graph_ref, the
lead's decision 10-12): the circuit's nodes at 1 in the stream and every other subcomponent at u, every reader of the
circuit reading the stream less (1 - e) of each connection the model has from another circuit node (e its strength
when stated, 0 when not; through the attention pattern into an attention output, inside the GELU into an MLP output),
the prediction less (1 - e) of each residual writer at a target, each writer's output as it enters the stream (times
its mask: cutting a writer that is off changes nothing); with every connection stated, the masked run of the circuit
and u.
"""
from __future__ import annotations

import contextlib
import copy
import math

import torch
import torch.nn.functional as F

import carry
import native

PRECISIONS = ("score", "tf32", "bf16")
BLOCK = -1  # a block node's index (goal_eval.BLOCK): (site, position, BLOCK), the site's whole output there as one unit
BK = "#block"  # a plan's table key for a site's blocks: site + BK (its readers and writers tables the same rows)


class Engine:
    def __init__(self, nat: native.Native, ids: list[int], targets: list[int], live: bool = True, columns: dict | None = None,
                 compile: bool = False, outside: str = "off"):
        if outside not in ("off", "on"):
            raise ValueError("outside is off or on")
        self.nat, self.ids, self.targets, self.live = nat, list(ids), list(targets), live
        self.fixed = live or outside == "on"  # whether every run adds x R (the remainder, and with outside on the rest)
        tg = nat.target
        self.tg, self.T, self.dev = tg, len(ids), nat.dev
        self.H, self.hd = tg.n_head, tg.hd
        with torch.no_grad():
            self.ref = nat.reference([self.ids], self.targets)
            self.emb = tg.embed(torch.tensor(self.ids, device=self.dev))[None]  # [1, T, d]
        self._targets = torch.tensor(self.targets, device=self.dev)  # (indexing by a list copies it to the device: a synchronization)
        self._causal_targets = torch.arange(self.T, device=self.dev)[None, :] <= self._targets[:, None]  # [targets, T]
        self.columns = {n: (None if columns is None or columns.get(n) is None else columns[n].to(self.dev)) for n in nat.names}
        self.V, self.U, self.R = {}, {}, {}
        self.b = {n: tg.site(n).b for n in nat.names}  # each site's bias, always on (None: vpd4l's have none)
        for n in nat.names:
            st, cols = tg.site(n), self.columns[n]
            self.V[n] = st.V if cols is None else st.V.index_select(1, cols).contiguous()  # [d_in, K]
            self.U[n] = st.U if cols is None else st.U.index_select(0, cols).contiguous()  # [K, d_out]
            self.R[n] = st.delta_T  # [d_in, d_out]: the remainder W - V U, transposed (native.model)
            if outside == "on" and cols is not None:  # every other subcomponent on, with the remainder or alone
                self.R[n] = ((st.W.T if live else st.V @ st.U) - self.V[n] @ self.U[n]).contiguous()
        self._dt, self._cast = None, {}  # the products' dtype in the current run (None: float32), the weights' casts
        self._search = torch.compile(self.hidden, dynamic=False) if compile else self.hidden

    def _whole(self, n: str) -> torch.Tensor:
        """Site n's whole matrix [d_in, d_out], a block's (every subcomponent and, with the remainders live, the
        remainder: W; else V U), made once."""
        whole = self.__dict__.setdefault("_wholes", {})
        if n not in whole:
            st = self.tg.site(n)
            whole[n] = (st.W.T if self.live else st.V @ st.U).contiguous()
        return whole[n]

    def on(self, ids: list[int], targets: list[int]) -> Engine:
        """This engine on another text (ids, targets): its columns, weights and settings shared, the text's own model
        run (ref) and embedding made (goal_eval's N: one removal set on many control texts)."""
        e = copy.copy(self)
        e.ids, e.targets, e.T = list(ids), list(targets), len(ids)
        with torch.no_grad():
            e.ref = self.nat.reference([e.ids], e.targets)
            e.emb = self.tg.embed(torch.tensor(e.ids, device=self.dev))[None]
        e._targets = torch.tensor(e.targets, device=self.dev)
        e._causal_targets = torch.arange(e.T, device=self.dev)[None, :] <= e._targets[:, None]
        e.__dict__.pop("_xr", None)  # the text's own x run (path_patched) and prefix (log_probs' start), made when asked
        e.__dict__.pop("_pre", None)
        e._search = e.hidden if self._search == self.hidden else torch.compile(e.hidden, dynamic=False)
        return e

    def width(self, n: str) -> int:
        """The number of a site's columns (a mask's last dimension)."""
        return self.V[n].shape[1]

    def _w(self, n: str) -> tuple:
        """(V, U, R) of site n in the run's product dtype: float32, or bf16 copies made once (a search's bf16 steps:
        the products' inputs cast once each, the weights never again)."""
        if self._dt is None:
            return self.V[n], self.U[n], self.R[n]
        if self._dt not in self._cast:
            self._cast[self._dt] = {k: tuple(w[k].to(self._dt) for w in (self.V, self.U, self.R)) for k in self.V}
        return self._cast[self._dt][n]

    def _mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x.to(w.dtype) @ w

    @contextlib.contextmanager
    def _dtype(self, precision: str):
        """The products' dtype for a run: bf16 for the bf16 precision, float32 otherwise."""
        self._dt = torch.bfloat16 if precision == "bf16" else None
        try:
            yield
        finally:
            self._dt = None

    def _site(self, n: str, h: torch.Tensor, m: torch.Tensor, rec: tuple | None = None) -> torch.Tensor:
        V, U, Rw = self._w(n)
        h = h.to(V.dtype)
        a = h @ V
        if rec is not None and n in rec[0]:  # reads: the activations at the asked (row, position, column)s
            rec[1][n] = a[rec[0][n]]
        out = (a * m.to(a.dtype)) @ U
        out = out + h @ Rw if self.fixed else out
        return out if self.b[n] is None else out + self.b[n]

    def _steps(self, fused: bool):
        """(norm(x, j), activation): the Target's, torch's fused kernels or vpd_model's own (Target.norm's numbering)."""
        tg = self.tg
        return (lambda x, j: tg.norm(x, j, fused)), (lambda x: tg.act(x, fused))

    def hidden(self, masks: dict, search: bool, rec: tuple | None = None) -> torch.Tensor:
        """The final normed residual stream at the targets [B, len(targets), d] of the masked run. search: torch's
        fused norms and GELU, the first layer's products of the embedding computed once for every row, and the last
        layer's queries, attention output and MLP at the targets alone (the only positions the prediction reads):
        float32-close to the whole run, for a search's steps (~27% fewer products); otherwise every step as
        vpd_model.Target.hidden."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        B = next(iter(masks.values())).shape[0]
        norm, act = self._steps(search)
        x = self.emb if search else self.emb.expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            h = norm(x, 2 * i)
            if search and i == tg.n_layer - 1:
                return self._last(masks, x, h, name, norm, act, B)
            q, k, v = (self._site(name(kd), h, masks[name(kd)], rec).view(B, T, H, hd).transpose(1, 2)
                       for kd in ("q_proj", "k_proj", "v_proj"))
            y = F.scaled_dot_product_attention(tg._rope(q, T).to(v.dtype), tg._rope(k, T).to(v.dtype), v, is_causal=True)
            x = x + self._site(name("o_proj"), y.transpose(1, 2).reshape(B, T, -1), masks[name("o_proj")], rec)
            h = norm(x, 2 * i + 1)
            x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, masks[name("c_fc")], rec)), masks[name("down_proj")], rec)
        return norm(x, -1)[:, self._targets]  # the final norm over every position, as vpd_model.Target.hidden

    def _prefix(self) -> list:
        """Each layer's rotated keys and values [1, H, T, hd] of the run with every column at 1 (kept: log_probs'
        start)."""
        if "_pre" not in self.__dict__:
            tg, T, H, hd = self.tg, self.T, self.H, self.hd
            norm, act = self._steps(False)
            ones = {n: torch.ones(1, 1, self.width(n), device=self.dev) for n in self.nat.names}
            x, out = self.emb, []
            with torch.no_grad():
                for i in range(tg.n_layer):
                    name = lambda k: native.site_name(i, k)  # noqa: E731
                    h = norm(x, 2 * i)
                    q, k, v = (self._site(name(kd), h, ones[name(kd)]).view(1, T, H, hd).transpose(1, 2) for kd in ("q_proj", "k_proj", "v_proj"))
                    q, k = tg._rope(q, T).to(v.dtype), tg._rope(k, T).to(v.dtype)
                    out.append((k, v))
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    x = x + self._site(name("o_proj"), y.transpose(1, 2).reshape(1, T, -1), ones[name("o_proj")])
                    h = norm(x, 2 * i + 1)
                    x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, ones[name("c_fc")])), ones[name("down_proj")])
            self._pre = out
        return self._pre

    def _hidden_from(self, masks: dict, start: int) -> torch.Tensor:
        """hidden()'s final normed stream at the targets with every mask 1 before start: the positions from start on,
        attending to _prefix's keys and values before it."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        B = next(iter(masks.values())).shape[0]
        norm, act = self._steps(False)
        pre, Tn = self._prefix(), T - start
        causal = torch.arange(T, device=self.dev)[None, :] <= torch.arange(start, T, device=self.dev)[:, None]  # [Tn, T]
        cut = {n: (m[:, start:] if m.shape[1] > 1 else m) for n, m in masks.items()}
        x = self.emb[:, start:].expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            h = norm(x, 2 * i)
            q, k, v = (self._site(name(kd), h, cut[name(kd)]).view(B, Tn, H, hd).transpose(1, 2) for kd in ("q_proj", "k_proj", "v_proj"))
            q, k = tg._rope_from(q, start, T).to(v.dtype), tg._rope_from(k, start, T).to(v.dtype)
            k = torch.cat([pre[i][0][:, :, :start].expand(B, -1, -1, -1), k], 2)
            v = torch.cat([pre[i][1][:, :, :start].expand(B, -1, -1, -1), v], 2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            x = x + self._site(name("o_proj"), y.transpose(1, 2).reshape(B, Tn, -1), cut[name("o_proj")])
            h = norm(x, 2 * i + 1)
            x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, cut[name("c_fc")])), cut[name("down_proj")])
        return norm(x, -1)[:, self._targets - start]

    def _last(self, masks, x, h, name, norm, act, B):
        """The last layer at the targets: keys and values at every position, the rest at the targets."""
        tg, T, H, hd, sel = self.tg, self.T, self.H, self.hd, self._targets
        at = lambda m: m[:, sel] if m.shape[1] > 1 else m  # noqa: E731
        n = len(self.targets)
        q = self._site(name("q_proj"), h[:, sel], at(masks[name("q_proj")])).view(B, n, H, hd).transpose(1, 2)
        k, v = (self._site(name(kd), h, masks[name(kd)]).view(B, T, H, hd).transpose(1, 2) for kd in ("k_proj", "v_proj"))
        half = hd // 2
        q = (q * tg.cos[sel] + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * tg.sin[sel]).to(v.dtype)  # _rope at the targets
        y = F.scaled_dot_product_attention(q, tg._rope(k, T).to(v.dtype), v, attn_mask=self._causal_targets)
        x = x[:, sel] + self._site(name("o_proj"), y.transpose(1, 2).reshape(B, n, -1), at(masks[name("o_proj")]))
        h = norm(x, 2 * tg.n_layer - 1)
        x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, at(masks[name("c_fc")]))), at(masks[name("down_proj")]))
        return norm(x, -1)

    def kl(self, masks: dict, precision: str = "score") -> torch.Tensor:
        """KL in bits of the model at the targets from each row's masked run [B] (summed over the targets)."""
        return carry.kl(self.ref, self.log_probs(masks, precision))

    def log_probs(self, masks: dict, precision: str = "score", reads: dict | None = None, start: int = 0):
        """Each row's masked run's next-token log probabilities at the targets [B, len(targets), V], float32. reads:
        {site: (rows, positions, columns)} (columns the engine's): also those subcomponents' activations, (log q,
        {site: [n]}) ("score" only). start: every mask is 1 before this position (no later than the first target), so
        those positions are the run with every column at 1 (_prefix, kept): only the positions from start on are
        computed (float32, no reads; goal_eval's real-model removals)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        if reads is not None and precision != "score":
            raise ValueError("reads come from the float32 run (score)")
        if start and (precision != "score" or reads is not None or not 0 < start <= min(self.targets)):
            raise ValueError("start is a float32 run without reads, after position 0 and no later than the first target")
        rec = None if reads is None else (reads, {})
        with _precision(precision, self.dev), self._dtype(precision):
            if start:
                h = self._hidden_from(masks, start)
            else:
                h = self.hidden(masks, search=False, rec=rec) if precision == "score" else self._search(masks, search=True)
        lq = torch.log_softmax(self.tg.unembed(h.float()), -1)  # the logits in float32 at every precision
        return lq if rec is None else (lq, rec[1])

    def full(self, masks: dict) -> dict:
        """Masks over a site's columns as masks over all its subcomponents [B, T, C] (for another engine or Native)."""
        out = {}
        for n, m in masks.items():
            cols = self.columns[n]
            if cols is None:
                out[n] = m
            else:
                z = m.new_zeros(*m.shape[:-1], self.nat.C[n])
                out[n] = z.index_copy(-1, cols, m)
        return out

    @torch.no_grad()
    def rescore(self, masks: dict) -> torch.Tensor:
        """The numbers to report for masks a search found at a lower precision: float32, vpd_model's own steps."""
        return self.kl({n: m.detach().float() for n, m in masks.items()}, "score")

    def plan(self, graphs, semantics: str = "native", cuts=None) -> dict:
        """Graphs' nodes and connections as index tensors over the engine's columns, for graph_kl. graphs: one
        native.Graph, run on every row, or a list of them, one per row. semantics (module docstring): "native"
        (native.run's: the stated connections) or "c" (C's: every connection the model has among the nodes, the
        unstated ones at strength 0). cuts: per graph (one dict, or a list beside graphs) {reader node or None (the
        prediction): [writer nodes]}, connections at strength 0 besides the unstated ones. In C's semantics a graph may
        be complete (VPD's answer, the whole model: native.Graph(masks=...)): it runs as u (give it 1 on the graph's
        subcomponents), its subcomponents reading each other in full but for the cuts. Per site the circuit's nodes,
        the query and key nodes, the reader nodes (with a connection in) and the writer nodes as (rows, positions,
        columns); per (reader site, writer site) the connections (reader rows, writer rows, their index among the
        strengths, 1 for a stated one and 0 for an unstated or cut one); per writer site the connections into the
        prediction (rows, target index, writer rows, index, 1 or 0)."""
        if semantics not in ("native", "c"):
            raise ValueError("semantics is native or c")
        one = not isinstance(graphs, (list, tuple))
        graphs = [graphs] if one else list(graphs)
        cuts = [cuts] if one else (list(cuts) if cuts is not None else [None] * len(graphs))
        keys = [(id(g), id(c)) for g, c in zip(graphs, cuts)]
        if not one and len(set(keys)) < len(keys):  # a graph given more than once (with the same cuts): each planned
            single = {}                              # once, the rows' plans laid end to end (_stack)
            for g, c, k in zip(graphs, cuts, keys):
                if k not in single:
                    single[k] = (len(single), self.plan(g, semantics, c))
            return self._stack([p for _, p in single.values()], [single[k][0] for k in keys])
        if semantics == "native" and any(g.complete for g in graphs):
            raise ValueError("native.run's graph runs need stated connections (a complete graph is a masked run: kl)")
        dev, inv = self.dev, {}

        def col(n, c):
            cols = self.columns[n]
            if cols is None:
                return c
            if n not in inv:
                inv[n] = {x: i for i, x in enumerate(cols.tolist())}
            if c not in inv[n]:
                raise ValueError(f"node {(n, c)} is outside the engine's columns")
            return inv[n][c]

        def key(nd):  # a node's table key: its site, a block's site + BK
            return nd[0] + BK if nd[2] == BLOCK else nd[0]

        def row(table, b, nd):  # a block's third field: 1 for one of the graph's own blocks, 0 for one only its cuts name
            rows = table.setdefault(key(nd), {})
            return rows.setdefault((b, nd[1], int(nd in own) if nd[2] == BLOCK else col(nd[0], nd[2])), len(rows))

        readers, writers, edges, outs, qk, nodes, E, O = {}, {}, {}, {}, {}, {}, 0, 0
        pairs = {}  # per layer, each read with stated pairs: (row, query position, [(query column | None, (key position, key
        # column) | None)], the split's query columns at the position, its (key position, key column)s)
        for b, g in enumerate(graphs):
            pcut = {k[1:]: v for k, v in (cuts[b] or {}).items() if _pair_cut(k)}  # (layer, t): (split nodes, pairs removed)
            cuts[b] = {r: ws for r, ws in (cuts[b] or {}).items() if not _pair_cut(r)}
            cut = {(w, r) for r, ws in cuts[b].items() for w in ws}
            own = {nd for nd in (g.node_set() if getattr(g, "masks", None) is None else ()) if nd[2] == BLOCK}
            named = own | {nd for r, ws in cuts[b].items() for nd in (r, *ws) if nd is not None and nd[2] == BLOCK}
            if named and semantics != "c":
                raise ValueError("blocks run in C's semantics")
            for nd in sorted(named):  # every block a reader and a writer, the same rows in both tables
                row(readers, b, nd)
                row(writers, b, nd)
            if g.complete:  # its subcomponents read each other in full (u): only the cuts are connections here
                at, par = {}, {r: list(ws) for r, ws in (cuts[b] or {}).items() if r is not None}
                out = list((cuts[b] or {}).get(None, []))
            else:
                ne, no = E, O  # this graph's stated connections' indices: from E and O on, in its parents' and out's order
                at = {(w, r): ne + k for k, (w, r) in enumerate((w, r) for r, ws in g.parents.items() for w in ws)}
                at.update({(w, None): no + k for k, w in enumerate(g.out)})
                E, O = ne + sum(len(ws) for ws in g.parents.values()), no + len(g.out)
                par, out = (g.parents, g.out) if semantics == "native" else native.all_edges(sorted(g.node_set()), self.targets)
            for r, ws in par.items():
                ri = row(readers, b, r)
                for w in ws:
                    lst = edges.setdefault((key(r), key(w)), ([], [], [], []))
                    lst[0].append(ri)
                    lst[1].append(row(writers, b, w))
                    lst[2].append(at.get((w, r), 0))
                    lst[3].append(float((w, r) in at and (w, r) not in cut))
            for w in out:
                if w[1] not in self.targets:
                    raise ValueError(f"{w} writes into the prediction but is not at a target")
                lst = outs.setdefault(key(w), ([], [], [], [], []))
                lst[0].append(b)
                lst[1].append(self.targets.index(w[1]))
                lst[2].append(row(writers, b, w))
                lst[3].append(at.get((w, None), 0))
                lst[4].append(float((w, None) in at and (w, None) not in cut))
            sp = {} if g.complete else (getattr(g, "pairs", None) or {})
            for layer, t in sorted(set(sp) | set(pcut)):  # reads with stated pairs, or with some of their pairs cut
                if (layer, t) in sp:  # a stated graph's pairs, split by its own query and key nodes
                    split, keep = sorted(g.node_set()), list(sp[(layer, t)])
                else:  # every pair of the cut's split (a complete graph: the circuit's query and key nodes)
                    split = sorted(pcut[(layer, t)][0])
                    keep = read_pairs(split, layer, t)
                if (layer, t) in pcut:
                    gone = set(pcut[(layer, t)][1])
                    keep = [x for x in keep if x not in gone]
                if any(nd is not None and nd[2] == BLOCK for x in keep for nd in x):
                    raise ValueError("query-key pairs name subcomponents, not blocks")
                split = [nd for nd in split if nd[2] != BLOCK]  # a block query or key: the rest of its read
                nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
                pairs.setdefault(layer, []).append((
                    b, t, [(None if qa is None else col(nq, qa[2]), None if kb is None else (kb[1], col(nk, kb[2]))) for qa, kb in keep],
                    [col(nq, c) for n, tt, c in split if n == nq and tt == t], [(tt, col(nk, c)) for n, tt, c in split if n == nk and tt <= t]))
            if g.complete:
                continue
            for n, t, c in g.node_set():
                if c == BLOCK:
                    continue
                nodes.setdefault(n, set()).add((b, t, col(n, c)))
                if n.endswith(("q_proj", "k_proj")):
                    qk.setdefault(n, set()).add((b, t, col(n, c)))
        ints = lambda v: torch.tensor(v, dtype=torch.long, device=dev)  # noqa: E731
        btc = lambda keys: tuple(ints([k[j] for k in keys]) for j in range(3))  # noqa: E731
        conn = lambda v: tuple(ints(x) for x in v[:-1]) + (torch.tensor(v[-1], device=dev),)  # noqa: E731
        return _positions({"B": None if one else len(graphs), "E": E, "O": O, "semantics": semantics,
                           "qk": {n: btc(sorted(v)) for n, v in qk.items()}, "nodes": {n: btc(sorted(v)) for n, v in nodes.items()},
                           "readers": {n: btc(list(rows)) for n, rows in readers.items()}, "writers": {n: btc(list(rows)) for n, rows in writers.items()},
                           "edges": {k: conn(v) for k, v in edges.items()}, "out": {k: conn(v) for k, v in outs.items()},
                           "pairs": pairs})

    def _stack(self, ps: list, rows: list) -> dict:
        """One-graph plans ps as the plan of the list naming ps[rows[b]] in row b: row b's entries after row b - 1's, its
        rows at each site after the earlier rows', its connections' strength indices after theirs (plan's own order:
        the same runs; an unstated connection, strength 0, indexes its own graph's first strength instead of the
        list's). One concatenation per table."""
        dev, B = self.dev, len(rows)
        ints = lambda v: torch.tensor(v, dtype=torch.long, device=dev)  # noqa: E731

        def keys(name):  # the tables' keys in the order the rows first name them
            out = {}
            for k in rows:
                out.update(dict.fromkeys(ps[k][name]))
            return list(out)

        def before(name):  # per key, each row's count of its entries and the entries of the rows before it
            out = {}
            for n in keys(name):
                cnt = [len(ps[k][name][n][0]) if n in ps[k][name] else 0 for k in rows]
                start = [0]
                for c in cnt[:-1]:
                    start.append(start[-1] + c)
                out[n] = (cnt, start)
            return out

        def pieces(name, n, i):  # entry field i of key n, every row's in order
            return torch.cat([ps[k][name][n][i] for k in rows if n in ps[k][name]])

        def spread(vals, cnt):  # each row's value repeated over its entries
            return torch.repeat_interleave(ints(vals), ints(cnt))

        out = {"B": B, "E": sum(ps[k]["E"] for k in rows), "O": sum(ps[k]["O"] for k in rows), "semantics": ps[rows[0]]["semantics"],
               "pairs": {}}
        for b, k in enumerate(rows):  # each row's reads with stated pairs (a one-graph plan's are its row 0's)
            for i, lst in ps[k].get("pairs", {}).items():
                out["pairs"].setdefault(i, []).extend((b,) + e[1:] for e in lst)
        at = {}
        for name in ("qk", "nodes", "readers", "writers"):
            at[name] = before(name)
            out[name] = {n: (spread(list(range(B)), cnt), pieces(name, n, 1), pieces(name, n, 2)) for n, (cnt, _) in at[name].items()}
        e0, o0 = [0], [0]
        for k in rows[:-1]:
            e0.append(e0[-1] + ps[k]["E"])
            o0.append(o0[-1] + ps[k]["O"])
        out["edges"], out["out"] = {}, {}
        for (rn, wn), (cnt, _) in before("edges").items():
            out["edges"][(rn, wn)] = (pieces("edges", (rn, wn), 0) + spread(at["readers"][rn][1], cnt),
                                      pieces("edges", (rn, wn), 1) + spread(at["writers"][wn][1], cnt),
                                      pieces("edges", (rn, wn), 2) + spread(e0, cnt), pieces("edges", (rn, wn), 3))
        for wn, (cnt, _) in before("out").items():
            out["out"][wn] = (spread(list(range(B)), cnt), pieces("out", wn, 1), pieces("out", wn, 2) + spread(at["writers"][wn][1], cnt),
                              pieces("out", wn, 3) + spread(o0, cnt), pieces("out", wn, 4))
        return _positions(out)

    def _rows(self, p: dict, B: int) -> dict:
        """A one-graph plan's tables for B rows (kept on the plan), row b's entries after row b - 1's."""
        if p["B"] is not None:
            if B != p["B"]:
                raise ValueError(f"the plan has {p['B']} graphs, one per row, not {B}")
            return p
        if B not in p.setdefault("tiled", {}):
            p["tiled"][B] = self._tile(p, 1, B)
        return p["tiled"][B]

    def repeat(self, plan: dict, times: int) -> dict:
        """The plan of a list plan's graphs listed times over, rows d B + b its row b (kept on the plan): its tables
        tiled as _rows tiles a one-graph plan (no planning; for reuse over several changed texts at once)."""
        if plan["B"] is None:
            raise ValueError("repeat takes a list plan (a one-graph plan runs on any number of rows)")
        if ("repeat", times) not in plan:
            plan[("repeat", times)] = self._tile(plan, plan["B"], times)
        return plan[("repeat", times)]

    def _tile(self, p: dict, B0: int, times: int) -> dict:
        """A plan's tables over B0 rows laid out times over (copy d's rows, row entries and strength indices after
        copy d - 1's)."""
        dev = self.dev
        step = lambda v, k: (v[None] + k * torch.arange(times, device=dev)[:, None]).flatten()  # noqa: E731
        btc = lambda t: (step(t[0], B0), t[1].repeat(times), t[2].repeat(times))  # noqa: E731
        R = {n: len(v[0]) for n, v in p["readers"].items()}
        W = {n: len(v[0]) for n, v in p["writers"].items()}
        return _positions({
            "B": B0 * times, "E": times * p["E"], "O": times * p["O"], "semantics": p["semantics"],
            "qk": {n: btc(v) for n, v in p["qk"].items()}, "nodes": {n: btc(v) for n, v in p["nodes"].items()},
            "readers": {n: btc(v) for n, v in p["readers"].items()}, "writers": {n: btc(v) for n, v in p["writers"].items()},
            "edges": {(rn, wn): (step(ri, R[rn]), step(wi, W[wn]), step(j, p["E"]), fx.repeat(times))
                      for (rn, wn), (ri, wi, j, fx) in p["edges"].items()},
            "out": {wn: (step(b, B0), ti.repeat(times), step(wi, W[wn]), step(j, p["O"]), fx.repeat(times))
                    for wn, (b, ti, wi, j, fx) in p["out"].items()},
            "pairs": {i: [(e[0] + B0 * d,) + e[1:] for d in range(times) for e in lst] for i, lst in p.get("pairs", {}).items()}})

    def graph_kl(self, plan: dict, u: dict | None = None, ew: torch.Tensor | None = None, ow: torch.Tensor | None = None,
                 precision: str = "score", patch: dict | None = None) -> torch.Tensor:
        """KL in bits of the model at the targets from each row's graph run [B] (the module docstring: plan from
        Engine.plan, u, ew, ow; patch as graph_log_probs)."""
        return carry.kl(self.ref, self.graph_log_probs(plan, u, ew, ow, precision, patch))

    def graph_log_probs(self, plan: dict, u: dict | None = None, ew: torch.Tensor | None = None, ow: torch.Tensor | None = None,
                        precision: str = "score", patch: dict | None = None, reads: dict | None = None, keep: dict | None = None,
                        reuse: tuple | None = None, ewn: torch.Tensor | None = None, own: torch.Tensor | None = None,
                        outputs: dict | None = None, rec_out: dict | None = None, rows: list | None = None,
                        off: list | None = None):
        """Each row's graph run's next-token log probabilities at the targets [B, len(targets), V], float32. patch:
        {site: (bool [B, T, K], values [B, T, K])}, those subcomponents' activations fixed (an interchange). reads as
        log_probs': (log q, {site: [n]}). keep, a dict, receives the run's states for reuse (each layer's keys and
        values, every writer row's activation; float32). reuse = (keep's states of a run of the same plan, u, ew and ow
        on the engine's text, tokens, start): the run on tokens ([T] for every row or [B, T] per row, equal to the
        text's before start, start no later than the first target), the model being causal: every state before start
        is that run's, and only the positions from start on are computed (float32, no patch or reads). tile_states
        lays a run's states out for its plan's rows repeated (repeat: the graphs' list repeated, one block per
        changed text). ewn, own: the connections' strengths in their readers' RMSNorm scale (the norm before a residual
        read: queries, keys, values, MLP inputs; the prediction's final norm), [B, E] and [B, O] like ew and ow, when
        they differ from the strengths in the read itself (None: the same): a connection's direct share and its share
        in the scale, cut apart (goal_eval.GoalEval.connection_effects). outputs: {site: (bool [B or 1, T], values [B or 1,
        T, d_out])}, the site's whole output (every subcomponent and the remainder) replaced at those positions (activation
        patching at a site and position; with patch's activations there, a block of variables.py); rec_out, a dict,
        receives every site's output [B, T, d_out] (float32 runs without reuse). rows: per row None or {(layer, t): [H,
        t + 1]}, attention rows used in place of the run's own at those reads (goal_eval.graph_ref's rows,
        pattern_family; float32 runs, with reuse at reads from its start on). off: per row None or a list of the plan's circuit nodes (site,
        position, subcomponent) taken out of that row's circuit (C's semantics: outside it, at u; with their connections
        at strength 0, the run of the graph without them: line deletions on one plan)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        if (keep is not None or reuse is not None) and (precision != "score" or patch is not None or reads is not None):
            raise ValueError("keep and reuse are float32 runs without patch or reads")
        if (outputs is not None or rec_out is not None) and (precision != "score" or reuse is not None):
            raise ValueError("outputs and rec_out are float32 runs of every position")
        if (ewn is not None or own is not None) and precision != "score":
            raise ValueError("separate norm strengths are float32 runs (vpd_model's own norm)")
        if rows is not None and (precision != "score" or (reuse is not None and any(t < reuse[2] for d in rows for _, t in (d or {})))):
            raise ValueError("rows are float32 runs, at reads from reuse's start on")
        if reuse is not None and not 0 < reuse[2] <= min(self.targets):
            raise ValueError("reuse starts after position 0 and no later than the first target")
        B = plan["B"] or max([1] + [t.shape[0] for t in (ew, ow) if t is not None] + [m.shape[0] for m in (u or {}).values()])
        p = self._rows(plan, B)
        if off is not None:
            if p["semantics"] != "c":
                raise ValueError("off takes C's semantics")
            p = self._without(p, off)
        with _precision(precision, self.dev), self._dtype(precision):
            rec = None if reads is None else (reads, {})
            h = self._graph_hidden(p, self._zeros() if u is None else u, None if ew is None else ew.reshape(-1),
                                   None if ow is None else ow.reshape(-1), B, precision != "score", patch, rec, keep, reuse,
                                   None if ewn is None else ewn.reshape(-1), None if own is None else own.reshape(-1), outputs, rec_out,
                                   rows)
        lq = torch.log_softmax(self.tg.unembed(h.float()), -1)
        return lq if rec is None else (lq, rec[1])

    def _without(self, p: dict, off: list) -> dict:
        """A laid-out plan with some rows' circuit nodes dropped from its circuit and query-key tables (graph_log_probs'
        off): those nodes then stand at u, outside the circuit, in their rows."""
        T, drop = self.T, {}
        for b, nds in enumerate(off):
            for n, t, c in nds or []:
                cols = self.columns[n]
                k = c if cols is None else int((cols == c).nonzero().flatten()[0])
                drop.setdefault(n, []).append((b * T + t) * self.width(n) + k)
        out = dict(p)
        for key in ("nodes", "qk"):
            tab = {}
            for n, (b, t, c) in p[key].items():
                if n in drop:
                    keep = ~torch.isin((b * T + t) * self.width(n) + c, torch.tensor(drop[n], device=self.dev))
                    b, t, c = b[keep], t[keep], c[keep]
                tab[n] = (b, t, c)
            out[key] = tab
        return out

    @staticmethod
    def tile_states(states: dict, times: int) -> dict:
        """keep's states of a run on B rows as those of the run on the same rows repeated times over (rows b, B + b, ...
        the same: the plan of the graphs' list repeated, u, ew and ow repeated the same way)."""
        return {"k": [k.repeat(times, 1, 1, 1) for k in states["k"]], "v": [v.repeat(times, 1, 1, 1) for v in states["v"]],
                "aw": {n: a.repeat(times) for n, a in states["aw"].items()}, "ak": [a.repeat(times, 1, 1) for a in states["ak"]]}

    @torch.no_grad()
    def _x_run(self) -> dict:
        """The real model's run on the engine's text, kept (path_patched's x): per layer the residual stream entering the
        attention and the MLP [T, d] and their norms, the rotated queries and keys and the values [H, T, hd], the scores
        and the pattern [H, T, T], the attention output before o_proj [T, d], the MLP's pre-activation and GELU
        [T, d_ff]; the final stream [T, d]; every site's whole output [T, d_out] (so; queries and keys unrotated).
        float32, vpd_model's own norm and GELU, every matrix its full weight."""
        if not hasattr(self, "_xr"):
            tg, T, H, hd = self.tg, self.T, self.H, self.hd
            W = lambda i, kd: tg.site(native.site_name(i, kd)).W  # noqa: E731
            bias = lambda o, i, kd: o if tg.site(native.site_name(i, kd)).b is None else o + tg.site(native.site_name(i, kd)).b  # noqa: E731
            causal = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
            x = self.emb[0]
            out = {k: [] for k in ("xa", "xm", "ha", "hm", "q", "k", "v", "S", "A", "y", "pre", "g")}
            so = out["so"] = {}
            for i in range(tg.n_layer):
                nm = lambda kd, i=i: native.site_name(i, kd)  # noqa: E731
                h = tg.norm(x, 2 * i)
                for kd in ("q_proj", "k_proj", "v_proj"):
                    so[nm(kd)] = bias(h @ W(i, kd).T, i, kd)
                q, k, v = (so[nm(kd)].view(T, H, hd).transpose(0, 1) for kd in ("q_proj", "k_proj", "v_proj"))
                q, k = tg._rope(q, T), tg._rope(k, T)
                S = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
                A = S.masked_fill(~causal, float("-inf")).softmax(-1)
                y = (A @ v).transpose(0, 1).reshape(T, -1)
                for key, val in (("xa", x), ("ha", h), ("q", q), ("k", k), ("v", v), ("S", S), ("A", A), ("y", y)):
                    out[key].append(val)
                so[nm("o_proj")] = bias(y @ W(i, "o_proj").T, i, "o_proj")
                x = x + so[nm("o_proj")]
                h2 = tg.norm(x, 2 * i + 1)
                pre = so[nm("c_fc")] = bias(h2 @ W(i, "c_fc").T, i, "c_fc")
                g = tg.act(pre)
                for key, val in (("xm", x), ("hm", h2), ("pre", pre), ("g", g)):
                    out[key].append(val)
                so[nm("down_proj")] = bias(g @ W(i, "down_proj").T, i, "down_proj")
                x = x + so[nm("down_proj")]
            out["final"] = x
            self._xr = out
        return self._xr

    @torch.no_grad()
    def path_patched(self, graphs: list, prompts: list, strengths: torch.Tensor | None = None, cuts: list | None = None,
                     lines: torch.Tensor | None = None) -> torch.Tensor:
        """log q at the targets [len(graphs), len(prompts), targets, V] of each graph's path-patched run on the engine's
        text x with prompts[d] = (position p, token z) changed (or a list of such changes), or {node: value} held (s_track_pp's and interchange_pp's,
        the lead's spec 10-12; goal_eval.pp_run, one graph and prompt at a time, the reference). x is the real model's run on x (_x_run); in the patched run P the
        change travels only along stated paths. A circuit node reads its stated parents' outputs from P and everything
        else from x (unstated writers, in the circuit or not, the remainders and the embeddings; the token at p only
        through a stated token edge (graph.tok_edges (p, node)) into a node at p), its norm on that combined input. A
        read (layer, t), the pattern row its attention output nodes take: queries and keys split into the graph's
        nodes and the rest (at x); with an entry in graph.pairs (possibly empty: the background alone) x's scores plus
        each stated pair's change, P's parts in place of x's; without one every pair claimed, the graph's queries and
        keys at P; softmax over the row. An attention output node: that row over x's values plus its stated value
        parents' changes. A block node (site, t, negative index) is the site's whole output at t from its patched input
        (every subcomponent and the remainder; the graph's subcomponent nodes there dropped). Outside the circuit
        everything is x's; the prediction reads P's stated writers into it and x's every other, and at a target t the
        changed token's embedding when t is in graph.tok_out. With nothing stated P is x; a stated parent's value is
        P's, never the model's on x'. A held node (a prompt {node: value}, on x itself: a subcomponent's activation or a
        block's whole output) takes its value in P whatever its inputs, its stated readers reading it; in a complete
        graph that activation or site output is set whether or not the node is the graph's. A complete graph states every connection the model has among its nodes, every
        token edge (into its residual readers and the prediction) and every pair (_pp_dense); cuts, per complete graph
        {(layer, t): [source positions]}: its attention outputs at (layer, t) read those sources' values at x. strengths
        [K, E + O]: graphs[0]'s connection subsets (goal_eval.changed_edges' ons: its parents' connections then its
        writers into the prediction; its nodes, token edges and pairs in each), the result [K, ...]. lines [K, L]:
        graphs[0]'s lines (pp_lines' columns: its nodes, connections, token edges and pairs) on or off per row (a
        line deletion's run: a node off goes with its connections, token edges and the pairs naming it, and a block off
        gives back the subcomponents it held), the result [K, ...]. Only nodes whose input changed are recomputed,
        against the kept x run, from the earliest change in a batch of prompts on."""
        if strengths is not None:
            return self._pp_stated(self._pp_union(graphs[:1], strengths), prompts)
        if lines is not None:
            if graphs[0].complete:
                nodes = sorted(graphs[0].node_set())
                return self._pp_dense([native.Graph([nd for nd, k in zip(nodes, row) if k], complete=True)
                                       for row in (lines > 0).tolist()], prompts)
            return self._pp_stated(self._pp_union(graphs[:1], lines=lines), prompts)
        out = [None] * len(graphs)
        whole = [i for i, g in enumerate(graphs) if g.complete]
        stated = [i for i, g in enumerate(graphs) if not g.complete]
        if whole:
            got = self._pp_dense([graphs[i] for i in whole], prompts, None if cuts is None else [cuts[i] for i in whole])
            for i, lq in zip(whole, got):
                out[i] = lq
        if stated:
            if cuts is not None and any(cuts[i] for i in stated):
                raise ValueError("cuts are a complete graph's")
            for i, lq in zip(stated, self._pp_stated(self._pp_union([graphs[i] for i in stated]), prompts)):
                out[i] = lq
        return torch.stack(out)

    @staticmethod
    def _pp_changes(pr) -> list:
        """A prompt's changed tokens [(position, token)]: (p, z), or a list of them; a held set's, none."""
        if isinstance(pr, dict):
            return []
        if len(pr) == 2 and not isinstance(pr[0], (list, tuple)):
            return [(int(pr[0]), int(pr[1]))]
        return [(int(p), int(z)) for p, z in pr]

    def _pp_pos(self, pr) -> int:
        """A prompt's earliest change: its earliest changed token's position, or a held set's earliest node."""
        if isinstance(pr, dict):
            return min((nd[1] for nd in pr), default=self.T - 1)
        return min(p for p, _ in self._pp_changes(pr))

    def _pp_chunks(self, prompts: list, cost) -> list:
        """Prompt indices in batches, the latest change first, each batch's prompts times cost(its earliest change) at
        most PP_ENTRIES."""
        order = sorted(range(len(prompts)), key=lambda d: -self._pp_pos(prompts[d]))
        out, cur = [], []
        for d in order:
            if cur and (len(cur) + 1) * cost(self._pp_pos(prompts[d])) > PP_ENTRIES:
                out.append(cur)
                cur = []
            cur.append(d)
        return out + [cur] if cur else out

    def _pp_token(self, part: list, G: int) -> tuple:
        """The batch's changed tokens: (per row and position the change's index or -1 [D G, T], each change's embedding
        change [n, d]); rows prompt-major, graph-minor."""
        dev, emb = self.dev, self.tg.token_embed
        rows, pos, z, x0 = [], [], [], []
        for d, pr in enumerate(part):
            for p, zz in self._pp_changes(pr):
                rows.append(d)
                pos.append(p)
                z.append(zz)
                x0.append(self.ids[p])
        cidx = torch.full((len(part) * G, self.T), -1, dtype=torch.long, device=dev)
        if not rows:
            return cidx, torch.zeros(0, self.tg.d_model, device=dev)
        n = len(rows)
        r = (torch.tensor(rows, device=dev)[:, None] * G + torch.arange(G, device=dev)).flatten()
        cidx[r, torch.tensor(pos, device=dev).repeat_interleave(G)] = torch.arange(n, device=dev).repeat_interleave(G)
        return cidx, emb(torch.tensor(z, device=dev)) - emb(torch.tensor(x0, device=dev))

    def _pp_held(self, part: list, G: int, key) -> dict:
        """Per site the held nodes of the batch's prompts: (subcomponents, blocks), each None or (rows [n G], positions
        [n G], key(node) [n G], values [n G] or [n G, d_out]) over the rows of every graph; key(node) None: not one of
        the run's."""
        dev, out = self.dev, {}
        for d, pr in enumerate(part):
            if isinstance(pr, dict):
                for nd, val in pr.items():
                    k = key(nd)
                    if k is not None:
                        out.setdefault(nd[0], ([], []))[nd[2] < 0].append((d, nd[1], k, val))
        g = torch.arange(G, device=dev)
        res = {}
        for n, both in out.items():
            res[n] = []
            for lst in both:
                if not lst:
                    res[n].append(None)
                    continue
                d, t, k, v = zip(*lst)
                rows = (torch.tensor(d, device=dev)[:, None] * G + g).flatten()
                t, k = (torch.tensor(x, device=dev).repeat_interleave(G) for x in (t, k))
                v = torch.stack([torch.as_tensor(x, device=dev, dtype=torch.float32) for x in v]).repeat_interleave(G, 0)
                res[n].append((rows, t, k, v))
        return res

    def _pp_predict(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.tg.unembed(self.tg.norm(z, -1)), -1)

    def _pp_dense(self, graphs: list, prompts: list, cuts: list | None = None) -> torch.Tensor:
        """path_patched of complete graphs: per prompt batch one run of the stream's change from its earliest change on,
        each site's circuit columns (and blocks) recomputed on x's input plus the change at the rows whose input
        changed; the graphs' nodes, blocks and cuts kept from the prompts' earliest change on (s0)."""
        tg, T, H, dev, xr = self.tg, self.T, self.H, self.dev, self._x_run()
        G, names, s0 = len(graphs), self.nat.names, min(self._pp_pos(pr) for pr in prompts)
        parts, blocks = {}, {}
        sid = {n: j for j, n in enumerate(names)}
        CM = max(self.nat.C.values()) + 1  # a node packed as (site T + position) CM + column + 1 (a block's column -1)
        keys, own = [], []
        for gi, g in enumerate(graphs):
            if g.masks is not None:
                for n, m in g.masks.items():
                    t, c = m[s0:].to(dev).nonzero(as_tuple=True)
                    if len(t):  # (a site with nothing on from s0: no columns, so a part held there reads x's alone)
                        parts.setdefault(n, []).append((torch.full_like(t, gi), t + s0, c))
                continue
            keys.append(torch.tensor([(sid[n] * T + t) * CM + c + 1 for n, t, c in g.node_set()], dtype=torch.long))
            own.append(torch.full((len(keys[-1]),), gi, dtype=torch.long))
        if keys:
            k, o = torch.cat(keys).to(dev), torch.cat(own).to(dev)
            c, t, si = k % CM - 1, (k // CM) % T, k // (CM * T)
            late = t >= s0
            c, t, si, o = c[late], t[late], si[late], o[late]
            for j in si.unique().tolist():
                m = si == j
                bl = m & (c < 0)
                if bool(bl.any()):
                    blocks.setdefault(names[j], []).append((o[bl], t[bl], c[bl]))
                pm = m & (c >= 0)
                if bool(pm.any()):
                    parts.setdefault(names[j], []).append((o[pm], t[pm], c[pm]))
        sites = {}
        for n in set(parts) | set(blocks):
            st, m, V, U, bm = tg.site(n), None, None, None, None
            if n in blocks:
                g_, t_, _ = (torch.cat(x) for x in zip(*blocks[n]))
                bm = torch.zeros(G, T - s0, dtype=torch.bool, device=dev)
                bm[g_, t_ - s0] = True
            if n in parts:
                g_, t_, c_ = (torch.cat(x) for x in zip(*parts[n]))
                cols, ci = torch.unique(c_, return_inverse=True)
                m = torch.zeros(G, T - s0, len(cols), dtype=torch.bool, device=dev)
                m[g_, t_ - s0, ci] = True
                if bm is not None:  # a block holds its site's subcomponents there
                    m &= ~bm[..., None]
                V, U = st.V[:, cols], st.U[cols]
                cp = torch.full((st.V.shape[1],), -1, dtype=torch.long, device=dev)
                cp[cols] = torch.arange(len(cols), device=dev)
            sites[n] = (m, V, U, bm, st.W, None if m is None else cp)
        cm = {}  # per layer [G, T - s0, T]: the value reads (query position from s0, source) at x
        for gi, cg in enumerate(cuts or []):
            for (layer, t), srcs in (cg or {}).items():
                if srcs and t >= s0:
                    cm.setdefault(layer, torch.zeros(G, T - s0, T, dtype=torch.bool, device=dev))[gi, t - s0, sorted(srcs)] = True
        res = torch.empty(G, len(prompts), len(self.targets), tg.vocab, device=dev)
        for chunk in self._pp_chunks(prompts, lambda s: G * H * (T - s) * T):
            part = [prompts[d] for d in chunk]
            lq = self._pp_dense_run(sites, cm, G, part, xr, s0)
            res[:, chunk] = lq.view(len(part), G, *lq.shape[1:]).transpose(0, 1)
        return res

    def _pp_dense_run(self, sites: dict, cm: dict, G: int, part: list, xr: dict, s0: int) -> torch.Tensor:
        tg, T, H, hd, dev = self.tg, self.T, self.H, self.hd, self.dev
        s = min(self._pp_pos(pr) for pr in part)
        Tn, D = T - s, len(part)
        B = D * G
        cidx, dtoks = self._pp_token(part, G)
        tb, tt = (cidx[:, s:] >= 0).nonzero(as_tuple=True)  # the changed tokens (row, position from s)
        tok = torch.zeros(B, Tn, tg.d_model, device=dev)
        tok[tb, tt] = dtoks[cidx[tb, tt + s]]
        chg = torch.zeros(B, Tn, dtype=torch.bool, device=dev)  # whether a residual reader's input changed
        chg[tb, tt] = True
        dS = torch.zeros_like(tok)  # the circuit's residual writers' change
        late = torch.arange(s, T, device=dev)[:, None] < torch.arange(T, device=dev)[None, :]  # keys after the query
        held = self._pp_held(part, G, lambda nd: nd[2] if nd[1] >= s else None)

        def delta(n, hP, hx, rows):  # site n's output change [B, Tn, d_out] (zero off rows, inputs changed), held set
            st, hs = sites.get(n), held.get(n)
            if st is None and hs is None:
                return None
            site, out, da = tg.site(n), None, None
            if st is not None:
                m, V, U, bm, W, cp = st
                if m is not None:
                    da = (hP @ V - hx @ V) * (m[:, s - s0:].repeat(D, 1, 1) & rows[..., None])
                    out = da @ U
                if bm is not None:  # blocks: the whole output's change
                    bmr = bm[:, s - s0:].repeat(D, 1) & rows
                    ob = ((hP - hx) @ W.T) * bmr[..., None]
                    out = ob if out is None else out + ob
            if hs is None:
                return out
            if out is None:
                out = torch.zeros(B, Tn, site.W.shape[0], device=dev)
            sub, blk = hs
            if sub is not None:  # a held activation: its write at the held value in place of the computed one
                rb, tt, c, v = sub
                tt = tt - s
                Vc = site.V[:, c].T
                now = (hx[tt] * Vc).sum(-1)  # x's
                if st is not None and da is not None:
                    ci = cp[c]
                    now = now + torch.where(ci >= 0, da[rb, tt, ci.clamp(min=0)], 0.0)
                if st is not None and bm is not None:
                    now = now + bmr[rb, tt] * ((hP[rb, tt] - hx[tt]) * Vc).sum(-1)
                out = out.index_put((rb, tt), (v - now)[:, None] * site.U[c], accumulate=True)
            if blk is not None:  # a held block: the site's whole output there
                rb, tt, _, v = blk
                out = out.index_put((rb, tt - s), v - xr["so"][n][tt])
            return out

        def heads(z):
            return z.view(B, Tn, H, hd).transpose(1, 2)
        for i in range(tg.n_layer):
            nm = lambda kd, i=i: native.site_name(i, kd)  # noqa: E731
            hP = tg.norm(xr["xa"][i][s:] + dS + tok, 2 * i)
            dq, dk, dv = (delta(nm(kd), hP, xr["ha"][i][s:], chg) for kd in ("q_proj", "k_proj", "v_proj"))
            q, k, v = (xr[x][i].expand(B, -1, -1, -1) for x in ("q", "k", "v"))
            q = q[:, :, s:]
            moved = torch.zeros(B, Tn, dtype=torch.bool, device=dev)  # a query, or a key or value at or before, changed
            kv = torch.zeros(B, Tn, dtype=torch.bool, device=dev)
            if dq is not None:
                q = q + tg._rope_from(heads(dq), s, T)
                moved |= (dq != 0).any(-1)
            if dk is not None:
                k = torch.cat([k[:, :, :s], k[:, :, s:] + tg._rope_from(heads(dk), s, T)], 2)
                kv |= (dk != 0).any(-1)
            if dv is not None:
                v = torch.cat([v[:, :, :s], v[:, :, s:] + heads(dv)], 2)
                kv |= (dv != 0).any(-1)
            moved |= kv.int().cummax(1).values.bool()
            live = nm("o_proj") in sites and bool(moved.any())
            if live or nm("o_proj") in held:
                y = xr["y"][i][s:].expand(B, -1, -1)
                if live:
                    A = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(late, float("-inf")).softmax(-1)
                    yh = A @ v
                    if i in cm and dv is not None:  # cut value reads: those sources' values at x
                        dvh = torch.cat([torch.zeros(B, H, s, hd, device=dev), heads(dv)], 2)
                        yh = yh - (A * cm[i][:, s - s0:].repeat(D, 1, 1)[:, None]) @ dvh
                    y = yh.transpose(1, 2).reshape(B, Tn, -1)
                do = delta(nm("o_proj"), y, xr["y"][i][s:], moved)
                dS = dS + do
                chg |= (do != 0).any(-1)
            if nm("down_proj") in sites or nm("down_proj") in held:
                hP = tg.norm(xr["xm"][i][s:] + dS + tok, 2 * i + 1)
                dp = delta(nm("c_fc"), hP, xr["hm"][i][s:], chg)
                pre = xr["pre"][i][s:] + (0.0 if dp is None else dp)
                rows = torch.zeros(B, Tn, dtype=torch.bool, device=dev) if dp is None else (dp != 0).any(-1)
                dd = delta(nm("down_proj"), tg.act(pre.expand(B, -1, -1)), xr["g"][i][s:], rows)
                dS = dS + dd
                chg |= (dd != 0).any(-1)
        z = xr["final"][self._targets].expand(B, -1, -1).clone()
        ok = [j for j, t in enumerate(self.targets) if t >= s]
        if ok:  # the circuit's writers and the token's embedding into the prediction
            at = [self.targets[j] - s for j in ok]
            z[:, ok] += dS[:, at] + tok[:, at]
        return self._pp_predict(z)

    @staticmethod
    def _pp_conns(g) -> list:
        """A stated graph's connections as changed_edges orders them: ("e", writer, reader) per parent, then ("o",
        writer) per write into the prediction."""
        return [("e", w, r) for r, ws in g.parents.items() for w in ws] + [("o", w) for w in g.out]

    @staticmethod
    def _pp_extra(g) -> list:
        """A stated graph's token edges ("t", position, reader), token edges into the prediction ("u", target) and
        stated pairs ("p", layer, position, query or None, key or None)."""
        return list(dict.fromkeys([("t", p, r) for p, r in getattr(g, "tok_edges", [])] + [("u", t) for t in getattr(g, "tok_out", [])]
                                  + [("p", L, t, qa, kb) for (L, t), lst in (getattr(g, "pairs", None) or {}).items() for qa, kb in lst
                                     if qa is not None or kb is not None]))

    def pp_lines(self, graph) -> list:
        """path_patched(lines=)'s columns for a graph: ("n", node) per node (sorted), its connections (_pp_conns), its
        token edges and pairs (_pp_extra); a complete graph's, its nodes alone."""
        nodes = [("n", nd) for nd in sorted(graph.node_set())]
        return nodes if graph.complete else nodes + self._pp_conns(graph) + self._pp_extra(graph)

    def _pp_union(self, graphs: list, ons: torch.Tensor | None = None, lines: torch.Tensor | None = None) -> dict:
        """path_patched's stated graphs as index tensors over their union: per site its nodes (union index, position,
        V's columns transposed and U's rows (zero at blocks), x's activations, a block's x output; a query's or key's U
        rows rotated at its position and its x part), per graph its nodes (those its blocks hold dropped), its strength
        of each connection (a residual or MLP read, a value read, a write into the prediction), token edge (into a node
        or the prediction) and query-key pair, and per layer the reads its attention outputs take, with whether the
        graph has an entry for each. With ons [K, E + O] the K graphs are graphs[0]'s connection subsets (ons' columns
        its parents' connections then its writers into the prediction), with its nodes, token edges and pairs; with
        lines [K, L] graphs[0] with the lines (pp_lines) each row turns off removed."""
        tg, H, hd, dev, xr = self.tg, self.H, self.hd, self.dev, self._x_run()
        sets = [g.node_set() for g in graphs]
        nodes = sorted(set().union(*sets))
        at = {nd: i for i, nd in enumerate(nodes)}
        N = len(nodes)

        def kept(st):  # a graph's nodes, those its blocks hold dropped
            bl = {(n, t) for n, t, c in st if c < 0}
            return [at[nd] for nd in st if nd[2] < 0 or (nd[0], nd[1]) not in bl]

        conns, extra = self._pp_conns, self._pp_extra
        if lines is not None:
            G, g0 = len(lines), graphs[0]
            keys = conns(g0) + extra(g0)
            if lines.shape[1] != N + len(keys):
                raise ValueError(f"lines have {lines.shape[1]} columns, the graph {N + len(keys)} lines")
            lines = lines.to(dev, torch.float32)
            non = lines[:, :N] > 0
            blk = {(nd[0], nd[1]): i for i, nd in enumerate(nodes) if nd[2] < 0}
            held = [(i, blk[(nd[0], nd[1])]) for i, nd in enumerate(nodes) if nd[2] >= 0 and (nd[0], nd[1]) in blk]
            member = non.clone()
            if held:  # a block's subcomponents are the graph's where the block is off
                hi, hb = (torch.tensor(x, device=dev) for x in zip(*held))
                member[:, hi] &= ~non[:, hb]
            onm = lines[:, N:].clone()
            named = [(j, at[nd]) for j, k in enumerate(keys) if k[0] == "p" for nd in k[3:] if nd is not None]
            if named:  # a pair naming a removed node goes with it
                pj, pn = (torch.tensor(x, device=dev) for x in zip(*named))
                gone = torch.zeros_like(onm).index_add_(1, pj, (~non[:, pn]).float())
                onm = onm * (gone == 0)
            entries = [set(getattr(g0, "pairs", None) or {})] * G
        elif ons is None:
            G, col, keys, on = len(graphs), {}, [], []
            member = torch.zeros(G, N, dtype=torch.bool)
            for gi, (g, st) in enumerate(zip(graphs, sets)):
                member[gi, kept(st)] = True
                ix = []
                for k in conns(g) + extra(g):
                    if k not in col:
                        col[k] = len(keys)
                        keys.append(k)
                    ix.append(col[k])
                on.append(ix)
            onm = torch.zeros(G, len(keys))
            for gi, ix in enumerate(on):
                onm[gi, ix] = 1.0
            onm = onm.to(dev)
            entries = [set(getattr(g, "pairs", None) or {}) for g in graphs]
        else:
            G = len(ons)
            member = torch.zeros(G, N, dtype=torch.bool)
            member[:, kept(sets[0])] = True
            keys = conns(graphs[0])
            if len(keys) != ons.shape[1]:
                raise ValueError(f"strengths have {ons.shape[1]} columns, the graph {len(keys)} connections")
            ex = extra(graphs[0])
            keys = keys + ex
            onm = torch.cat([ons.to(dev, torch.float32), torch.ones(G, len(ex), device=dev)], 1)
            entries = [set(getattr(graphs[0], "pairs", None) or {})] * G
        inp = {"q_proj": "ha", "k_proj": "ha", "v_proj": "ha", "o_proj": "y", "c_fc": "hm", "down_proj": "g"}
        by = {}
        for i, nd in enumerate(nodes):
            by.setdefault(nd[0], []).append(i)
        names = sorted(by)
        slot = [0] * N
        sites, ax = {}, torch.zeros(N, device=dev)
        Ures = torch.zeros(N, tg.d_model, device=dev)  # residual writers' write vectors (zero at blocks)
        half = hd // 2

        def rot(u, t):  # [R, H, hd] rotated at positions t [R]
            return u * tg.cos[t][:, None] + torch.cat([-u[..., half:], u[..., :half]], -1) * tg.sin[t][:, None]
        for n, ii in by.items():
            L, kd = native._name_layer_kind(n)
            st = tg.site(n)
            t = torch.tensor([nodes[i][1] for i in ii], device=dev)
            c = torch.tensor([nodes[i][2] for i in ii], device=dev)
            isb = c < 0
            iit = torch.tensor(ii, device=dev)
            Vt = st.V[:, c.clamp(min=0)].T * (~isb)[:, None]
            U = st.U[c.clamp(min=0)] * (~isb)[:, None]
            a = (xr[inp[kd]][L][t] * Vt).sum(-1)
            ax[iit] = a
            for j, i in enumerate(ii):
                slot[i] = j
            info = {"i": iit, "t": t, "Vt": Vt, "U": U, "L": L, "kind": kd, "isb": isb, "nb": int(isb.sum()), "W": st.W,
                    "bslot": torch.where(isb, torch.cumsum(isb.long(), 0) - 1, -1)}
            if info["nb"]:
                info["xo"] = xr["so"][n][t[isb]]  # x's whole output at the blocks' positions [Rb, d_out]
                info["bpos"] = isb.nonzero().flatten()
            if kd in ("o_proj", "down_proj"):
                Ures[iit] = U
            if kd in ("q_proj", "k_proj"):
                info["rot"] = rot(U.view(len(ii), H, hd), t)
                xp = a[:, None, None] * info["rot"]
                xp[isb] = xr["q" if kd == "q_proj" else "k"][L][:, t[isb]].transpose(0, 1)  # a block's x part: the whole
                info["xpart"] = xp
            sites[n] = info
        rd, outs, toks, touts, prs = {}, [], {}, [], {}
        sid = {n: j for j, n in enumerate(names)}
        for j, k in enumerate(keys):
            if k[0] == "e":
                w, r = at[k[1]], at[k[2]]
                rd.setdefault(k[2][0], []).append((w, slot[w], slot[r], j, int(k[1][2] < 0), sid[k[1][0]]))
            elif k[0] == "o":
                w = at[k[1]]
                outs.append((w, self.targets.index(k[1][1]), j, int(k[1][2] < 0), slot[w], sid[k[1][0]]))
            elif k[0] == "t":
                toks.setdefault(k[2][0], []).append((k[1], slot[at[k[2]]], j))
            elif k[0] == "u":
                touts.append((k[1], self.targets.index(k[1]), j))
            else:
                prs.setdefault(k[1], []).append((k[2], -1 if k[3] is None else at[k[3]], -1 if k[4] is None else at[k[4]], j))
        lt = lambda rows: torch.tensor(rows, device=dev, dtype=torch.long).T  # noqa: E731
        for n, rows in rd.items():  # per reader site: writer (union index, slot, is a block, site), reader slot, column
            sites[n]["reads"] = lt(rows)
        for n, rows in toks.items():  # per reader site: position, reader slot, column
            sites[n]["toks"] = lt(rows)
        reads = {}
        for n, st in sites.items():  # per layer: the reads its attention outputs take, each graph's entries, the pairs
            if st["kind"] != "o_proj":
                continue
            L = st["L"]
            ts = sorted(set(st["t"].tolist()))
            tj = {t: j for j, t in enumerate(ts)}
            ent = torch.tensor([[(L, t) in e for t in ts] for e in entries], dtype=torch.bool, device=dev)
            rows = [(tj[r[0]],) + r[1:] for r in prs.get(L, []) if r[0] in tj]
            reads[L] = (torch.tensor(ts, device=dev), ent, lt(rows) if rows else None)
        return {"G": G, "N": N, "at": at, "member": member.to(dev), "on": onm, "sites": sites, "ax": ax, "Ures": Ures, "names": names,
                "reads": reads, "outs": lt(outs) if outs else None, "touts": lt(touts) if touts else None,
                "slot": torch.tensor(slot, device=dev)}

    def _pp_stated(self, un: dict, prompts: list) -> torch.Tensor:
        """path_patched of _pp_union's graphs, in batches of rows (prompts x graphs, the graphs sliced when they are
        many) whose largest products (per row a change per node, and a pattern row per read of a layer) stay within
        PP_ENTRIES."""
        G, T, H = un["G"], self.T, self.H
        J = max([len(r[0]) for r in un["reads"].values()], default=0)
        per = max(un["N"] + un["on"].shape[1], J * H * T, max((len(st["i"]) * st["Vt"].shape[1] for st in un["sites"].values()), default=1))
        step = max(1, PP_ENTRIES // per)
        res = torch.empty(G, len(prompts), len(self.targets), self.tg.vocab, device=self.dev)
        for g0 in range(0, G, step):
            u = un if step >= G else dict(un, G=min(step, G - g0), member=un["member"][g0:g0 + step], on=un["on"][g0:g0 + step],
                                          reads={L: (ts, ent[g0:g0 + step], rows) for L, (ts, ent, rows) in un["reads"].items()})
            Gc = u["G"]
            for chunk in self._pp_chunks(prompts, lambda s, Gc=Gc: Gc * per):
                part = [prompts[d] for d in chunk]
                lq = self._pp_stated_run(u, part)
                res[g0:g0 + Gc, chunk] = lq.view(len(part), Gc, *lq.shape[1:]).transpose(0, 1)
        return res

    def _pp_stated_run(self, un: dict, part: list) -> torch.Tensor:
        tg, T, H, hd, dev, xr = self.tg, self.T, self.H, self.hd, self.dev, self._x_run()
        G, sites, ax, names = un["G"], un["sites"], un["ax"], un["names"]
        D = len(part)
        B = D * G
        gb = torch.arange(B, device=dev) % G
        cidx, dtoks = self._pp_token(part, G)
        dlt = torch.zeros(B, un["N"], device=dev)  # a subcomponent's P activation minus x's; a block: 1 where it changed
        bv = {}  # per site with blocks [B, blocks, d_out]: P's whole output minus x's
        member, on = un["member"][gb], un["on"][gb]
        half = hd // 2
        held = self._pp_held(part, G, lambda nd: un["at"].get(nd))

        def hold(n):  # site n's held nodes at their values (in each graph that has them)
            if n not in held:
                return
            st = sites[n]
            sub, blk = held[n]
            if sub is not None:
                rb, _, i_, v = sub
                keep = member[rb, i_]
                dlt[rb[keep], i_[keep]] = v[keep] - ax[i_[keep]]
            if blk is not None:
                rb, _, i_, v = blk
                keep = member[rb, i_]
                rb, i_, v = rb[keep], i_[keep], v[keep]
                bs = st["bslot"][un["slot"][i_]]
                ch = v - st["xo"][bs]
                if n not in bv:
                    bv[n] = torch.zeros(B, st["nb"], st["W"].shape[0], device=dev)
                bv[n][rb, bs] = ch
                dlt[rb, i_] = (ch != 0).any(-1).float()

        def rows_of(st, hits):  # (rows b, slot) of the site's members hit, and the [B, R] row lookup
            hit = torch.zeros(B, len(st["i"]), dtype=torch.bool, device=dev)
            for b_, r_ in hits:
                hit[b_, r_] = True
            hit &= member[:, st["i"]]
            bi, ri = hit.nonzero(as_tuple=True)
            rid = torch.full((B, len(st["i"])), -1, dtype=torch.long, device=dev)
            rid[bi, ri] = torch.arange(len(bi), device=dev)
            return bi, ri, rid

        def edges(st):  # its reads' strength x writer change [B, E] and (b, e) where nonzero
            w, _, _, j, _, _ = st["reads"]
            cf = on[:, j] * dlt[:, w]
            return cf, cf.nonzero(as_tuple=True)

        def writers(out, rr, eb, ee, cf, st, vec, scale=None):  # out[rr] += each stated parent's change (rows rr), in place
            w, ws, _, _, wb, wsid = st["reads"]
            sub = wb[ee] == 0
            e1, b1, r1 = ee[sub], eb[sub], rr[sub]
            step = max(1, ROWS // out.shape[1])  # ROWS entries of the [connections, width] products at once
            for s0 in range(0, len(e1), step):
                k = slice(s0, s0 + step)
                out.index_add_(0, r1[k], cf[b1[k], e1[k], None] * vec(w[e1[k]], ws[e1[k]], b1[k], e1[k], r1[k]))
            e2, b2, r2 = ee[~sub], eb[~sub], rr[~sub]
            for si in wsid[e2].unique().tolist():  # blocks: their whole output's change
                m = wsid[e2] == si
                wst = sites[names[si]]
                ch = bv[names[si]][b2[m], wst["bslot"][ws[e2[m]]]]
                if scale is not None:
                    ch = scale(ch, e2[m], r2[m])
                out.index_add_(0, r2[m], cf[b2[m], e2[m], None] * ch)
            return out

        def finish(n, st, bi, ri, h):  # the changed rows' P values from their inputs h [rows, d_in]
            ni, isb = st["i"][ri], st["isb"][ri]
            sub = ~isb
            dlt[bi[sub], ni[sub]] = (h[sub] * st["Vt"][ri[sub]]).sum(-1) - ax[ni[sub]]
            if bool(isb.any()):
                bs = st["bslot"][ri[isb]]
                ch = h[isb] @ st["W"].T - st["xo"][bs]
                if n not in bv:
                    bv[n] = torch.zeros(B, st["nb"], st["W"].shape[0], device=dev)
                bv[n][bi[isb], bs] = ch
                dlt[bi[isb], ni[isb]] = (ch != 0).any(-1).float()

        def read(n, xin, norm):  # a residual or MLP reader site: its changed members' P values
            st = sites.get(n)
            if st is None or ("reads" not in st and "toks" not in st):
                return
            hits = []
            if "reads" in st:
                cf, (eb, ee) = edges(st)
                hits.append((eb, st["reads"][2][ee]))
            if "toks" in st:
                tp, tr, tj = st["toks"]
                ci = cidx[:, tp]  # [B, token edges]: the change at each edge's position, or -1
                tb, tk = ((on[:, tj] > 0) & (ci >= 0)).nonzero(as_tuple=True)
                hits.append((tb, tr[tk]))
            bi, ri, rid = rows_of(st, hits)
            if len(bi) == 0:
                return
            Dm = torch.zeros(len(bi), xin.shape[-1], device=dev)
            if "reads" in st:
                rr = rid[eb, st["reads"][2][ee]]
                keep = rr >= 0
                table = un["Ures"] if norm is not None else sites[native.site_name(st["L"], "c_fc")]["U"]
                Dm = writers(Dm, rr[keep], eb[keep], ee[keep], cf, st,
                             (lambda w, ws, *_: table[w]) if norm is not None else (lambda w, ws, *_: table[ws]))
            if "toks" in st:
                rr = rid[tb, tr[tk]]
                keep = rr >= 0
                Dm.index_add_(0, rr[keep], dtoks[ci[tb[keep], tk[keep]]])
            z = xin[st["t"][ri]] + Dm
            finish(n, st, bi, ri, self.tg.norm(z, norm) if norm is not None else self.tg.act(z))

        def qk_changes(n):  # a query or key site's blocks' changes rotated [B, blocks, H, hd] (None: no block changed)
            st = sites.get(n)
            if st is None or n not in bv:
                return None
            u = bv[n].view(B, st["nb"], H, hd)
            t = st["t"][st["bpos"]]
            return u * tg.cos[t][None, :, None] + torch.cat([-u[..., half:], u[..., :half]], -1) * tg.sin[t][None, :, None]

        def gather_sum(st, rotb, w, ab):  # sum over the site's nodes of w [A, R] x each one's part change -> [A, H, hd]
            out = torch.einsum("ar,rhe->ahe", w * dlt[ab][:, st["i"]], st["rot"])
            if rotb is not None:
                out = out + torch.einsum("ar,arhe->ahe", w[:, st["bpos"]], rotb[ab])
            return out

        def key_terms(dS, st, rotb, vec, w, ab):  # dS[a, :, t_k] += w[a, k] (vec[a] . key k's part change), per head
            step = max(1, PP_ENTRIES // max(1, len(st["i"]) * H * 4))
            for a0 in range(0, len(ab), step):
                sl = slice(a0, a0 + step)
                c = torch.einsum("ahe,khe->akh", vec[sl], st["rot"]) * (w[sl] * dlt[ab[sl]][:, st["i"]])[..., None]
                if rotb is not None:
                    c[:, st["bpos"]] = torch.einsum("ahe,akhe->akh", vec[sl], rotb[ab[sl]]) * w[sl][:, st["bpos"], None]
                dS[sl].index_add_(2, st["t"], c.permute(0, 2, 1))

        def part(st, rotb, b, sl):  # (x part, part change) [n, H, hd] of query or key nodes (slots sl) in rows b
            ch = dlt[b, st["i"][sl]][:, None, None] * st["rot"][sl]
            if rotb is not None:
                isb = st["isb"][sl]
                ch = torch.where(isb[:, None, None], rotb[b, st["bslot"][sl].clamp(min=0)], ch)
            return st["xpart"][sl], ch

        def attend(i):  # layer i's attention output members: their reads' pattern rows, their values
            st = sites.get(native.site_name(i, "o_proj"))
            if st is None:
                return
            sq, sk = sites.get(native.site_name(i, "q_proj")), sites.get(native.site_name(i, "k_proj"))
            qb, kb_ = qk_changes(native.site_name(i, "q_proj")), qk_changes(native.site_name(i, "k_proj"))
            ts, ent, prow = un["reads"][i]
            J = len(ts)
            tj = torch.full((T,), -1, dtype=torch.long, device=dev)
            tj[ts] = torch.arange(J, device=dev)
            entb = ent[gb]  # [B, J]
            act = torch.zeros(B, J, dtype=torch.bool, device=dev)
            if sq is not None:  # every-pair reads: a query of the graph at the read changed
                qj = tj[sq["t"]]
                qc = (member[:, sq["i"]] & (dlt[:, sq["i"]] != 0) & (qj >= 0)[None, :])
                qb_, qr_ = qc.nonzero(as_tuple=True)
                act[qb_, qj[qr_]] = True
            if sk is not None:  # or a key at or before it
                kp = torch.zeros(B, T, dtype=torch.bool, device=dev)
                kb2, kr2 = (member[:, sk["i"]] & (dlt[:, sk["i"]] != 0)).nonzero(as_tuple=True)
                kp[kb2, sk["t"][kr2]] = True
                act |= kp.int().cummax(1).values.bool()[:, ts]
            act &= ~entb
            if prow is not None:  # stated reads: a stated pair's query or key changed
                pj, qa, kb, pc = prow
                po = on[:, pc]
                qd = torch.where(qa >= 0, dlt[:, qa.clamp(min=0)], 0.0)
                kd = torch.where(kb >= 0, dlt[:, kb.clamp(min=0)], 0.0)
                lb, lp = ((po != 0) & ((qd != 0) | (kd != 0))).nonzero(as_tuple=True)
                act[lb, pj[lp]] = True
            ab, aj = act.nonzero(as_tuple=True)
            arow = torch.full((B, max(J, 1)), -1, dtype=torch.long, device=dev)
            Prow = None
            if len(ab):
                A = len(ab)
                arow[ab, aj] = torch.arange(A, device=dev)
                ta = ts[aj]
                every = ~entb[ab, aj]
                dS = torch.zeros(A, H, T, device=dev)
                qx = xr["q"][i][:, ta].transpose(0, 1)  # [A, H, hd]
                dQ, qrest = torch.zeros_like(qx), qx
                if sq is not None:
                    mine = member[ab][:, sq["i"]] & (sq["t"][None, :] == ta[:, None])  # the row's graph's queries at its read
                    wq = (mine & every[:, None]).float()
                    if prow is not None:  # stated reads: their (query, rest) pairs' queries, at their strengths
                        sel = (qa[lp] >= 0) & (kb[lp] < 0)
                        wq.index_put_((arow[lb[sel], pj[lp[sel]]], un["slot"][qa[lp[sel]]]), po[lb[sel], lp[sel]], accumulate=True)
                    dQ = gather_sum(sq, qb, wq, ab)
                    qrest = qx - torch.einsum("aq,qhe->ahe", mine.float(), sq["xpart"])
                dS += torch.einsum("ahe,hse->ahs", dQ, xr["k"][i])
                if sk is not None:
                    before = member[ab][:, sk["i"]] & (sk["t"][None, :] <= ta[:, None])  # the row's graph's keys
                    xk = torch.einsum("ahe,khe->akh", dQ, sk["xpart"]) * (before & ~every[:, None])[..., None]
                    dS.index_add_(2, sk["t"], -xk.permute(0, 2, 1))  # stated reads' (query, rest): x's keys outside the graph
                    wk = (before & every[:, None]).float()
                    if prow is not None:  # stated reads: their (rest, key) pairs' keys, at their strengths
                        sel = (qa[lp] < 0) & (kb[lp] >= 0)
                        wk.index_put_((arow[lb[sel], pj[lp[sel]]], un["slot"][kb[lp[sel]]]), po[lb[sel], lp[sel]], accumulate=True)
                    key_terms(dS, sk, kb_, torch.where(every[:, None, None], qx + dQ, qrest), wk, ab)
                    if prow is not None:  # (query, key) pairs: P's product of the two parts minus x's
                        sel = (qa[lp] >= 0) & (kb[lp] >= 0)
                        b2, l2 = lb[sel], lp[sel]
                        if len(l2):
                            xq, cq = part(sq, qb, b2, un["slot"][qa[l2]])
                            xk_, ck = part(sk, kb_, b2, un["slot"][kb[l2]])
                            val = po[b2, l2, None] * (cq * xk_ + xq * ck + cq * ck).sum(-1)  # [n, H]
                            hi = torch.arange(H, device=dev)
                            dS.index_put_((arow[b2, pj[l2]][:, None].expand(-1, H), hi[None, :].expand(len(l2), -1),
                                           sk["t"][un["slot"][kb[l2]]][:, None].expand(-1, H)), val, accumulate=True)
                S = xr["S"][i][:, ta].transpose(0, 1) + dS / math.sqrt(hd)
                Prow = S.masked_fill(torch.arange(T, device=dev)[None, None, :] > ta[:, None, None], float("-inf")).softmax(-1)
            hits = []
            rj = tj[st["t"]]
            if Prow is not None:  # members whose read's pattern moved
                mb, mr = ((arow[:, rj.clamp(min=0)] >= 0) & (rj >= 0)[None, :]).nonzero(as_tuple=True)
                hits.append((mb, mr))
            if "reads" in st:
                cf, (eb, ee) = edges(st)
                hits.append((eb, st["reads"][2][ee]))
            if not hits:
                return
            bi, ri, rid = rows_of(st, hits)
            if len(bi) == 0:
                return
            Pr = xr["A"][i][:, st["t"][ri]].transpose(0, 1).clone()  # [R, H, T]: x's rows, the moved ones replaced
            if Prow is not None:
                r2 = rj[ri]
                a2 = torch.where(r2 >= 0, arow[bi, r2.clamp(min=0)], -1)
                mv = a2 >= 0
                Pr[mv] = Prow[a2[mv]]
            y = torch.einsum("rhs,hse->rhe", Pr, xr["v"][i]).reshape(len(bi), -1)
            if "reads" in st:
                rr = rid[eb, st["reads"][2][ee]]
                keep = rr >= 0
                Uv = sites[native.site_name(i, "v_proj")]["U"]
                sv = sites[native.site_name(i, "v_proj")]["t"]
                ws = st["reads"][1]

                def weigh(ch, e, r):  # a value's change [n, H hd] weighed per head by its row's pattern at its position
                    return (Pr[r, :, sv[ws[e]]][..., None] * ch.view(-1, H, hd)).reshape(-1, H * hd)
                y = writers(y, rr[keep], eb[keep], ee[keep], cf, st, lambda w, s_, b_, e, r: weigh(Uv[s_], e, r), scale=weigh)
            finish(native.site_name(i, "o_proj"), st, bi, ri, y)

        for i in range(tg.n_layer):
            for kd in ("q_proj", "k_proj", "v_proj"):
                read(native.site_name(i, kd), xr["xa"][i], 2 * i)
                hold(native.site_name(i, kd))
            attend(i)
            hold(native.site_name(i, "o_proj"))
            read(native.site_name(i, "c_fc"), xr["xm"][i], 2 * i + 1)
            hold(native.site_name(i, "c_fc"))
            read(native.site_name(i, "down_proj"), xr["pre"][i], None)
            hold(native.site_name(i, "down_proj"))
        nt = len(self.targets)
        z = xr["final"][self._targets].expand(B, -1, -1).reshape(B * nt, -1).clone()
        if un["outs"] is not None:  # the stated writers into the prediction
            w, ti, j, wb, ws, wsid = un["outs"]
            cf = on[:, j] * dlt[:, w]
            ob, oe = cf.nonzero(as_tuple=True)
            st = {"reads": torch.stack([w, ws, ti, j, wb, wsid])}
            z = writers(z, ob * nt + ti[oe], ob, oe, cf, st, lambda w_, *_: un["Ures"][w_])
        z = z.view(B, nt, -1)
        if un["touts"] is not None:  # the changed token's embedding into the prediction at its target
            tp, ti, tj = un["touts"]
            ci = cidx[:, tp]
            tb, tk = ((on[:, tj] > 0) & (ci >= 0)).nonzero(as_tuple=True)
            z[tb, ti[tk]] += dtoks[ci[tb, tk]]
        return self._pp_predict(z)

    def _zeros(self) -> dict:
        if not hasattr(self, "_u0"):
            self._u0 = {n: torch.zeros(1, 1, self.width(n), device=self.dev) for n in self.nat.names}
        return self._u0

    def _last_tables(self, p: dict) -> tuple:
        """A plan's tables for the last layer run at the targets alone (search steps): its query, attention-output and
        MLP readers and writers kept at the targets with their positions as target indices, the connections into or
        out of the others dropped (they reach nothing the prediction reads), the attention-output readers' pattern rows
        the targets' (kept on the plan)."""
        if "_last" not in p:
            dev, L = self.dev, self.tg.n_layer - 1
            S = {native.site_name(L, k) for k in ("q_proj", "o_proj", "c_fc", "down_proj")}
            tpos = torch.full((self.T,), -1, dtype=torch.long, device=dev)
            tpos[self._targets] = torch.arange(len(self.targets), device=dev)
            R, Wr, rmap, wmap = dict(p["readers"]), dict(p["writers"]), {}, {}
            for table, maps, src in ((R, rmap, p["readers"]), (Wr, wmap, p["writers"])):
                for n in S & set(src):
                    b, t, c = src[n]
                    ti = tpos[t]
                    keep = ti >= 0
                    table[n] = (b[keep], ti[keep], c[keep])
                    m = torch.full((len(b),), -1, dtype=torch.long, device=dev)
                    m[keep] = torch.arange(int(keep.sum()), device=dev)
                    maps[n] = m
            Eg = {}
            for (rn, wn), (ri, wi, j, fx) in p["edges"].items():
                keep = torch.ones(len(ri), dtype=torch.bool, device=dev)
                if rn in rmap:
                    keep &= rmap[rn][ri] >= 0
                if wn in wmap:
                    keep &= wmap[wn][wi] >= 0
                if bool(keep.any()):
                    Eg[(rn, wn)] = ((rmap[rn][ri] if rn in rmap else ri)[keep], (wmap[wn][wi] if wn in wmap else wi)[keep], j[keep], fx[keep])
            out = {}
            for wn, (b, ti, wi, j, fx) in p["out"].items():  # residual writers at the targets: every one kept
                out[wn] = (b, ti, wmap[wn][wi] if wn in wmap else wi, j, fx)
            p["_last"] = (R, Wr, Eg, out, S)
        return p["_last"]

    def _from_tables(self, p: dict, lo: int) -> tuple:
        """A plan's tables for a run from position lo on (reuse): the readers at lo and after, their positions less lo,
        the connections into them (their writer rows the plan's), per writer site its rows at lo and after (their
        places in its table, row, position less lo, column), and the attention-output readers' distinct positions
        (less lo; kept on the plan)."""
        key = ("_from", lo)
        if key not in p:
            dev = self.dev
            R, rmap = {}, {}
            for n, (b, t, c) in p["readers"].items():
                keep = t >= lo
                m = torch.full((len(b),), -1, dtype=torch.long, device=dev)
                m[keep] = torch.arange(int(keep.sum()), device=dev)
                rmap[n] = m
                if bool(keep.any()):
                    R[n] = (b[keep], t[keep] - lo, c[keep])
            Eg = {}
            for (rn, wn), (ri, wi, j, fx) in p["edges"].items():
                keep = rmap[rn][ri] >= 0
                if bool(keep.any()):
                    Eg[(rn, wn)] = (rmap[rn][ri][keep], wi[keep], j[keep], fx[keep])
            new = {n: ((t >= lo).nonzero().flatten(), b[t >= lo], t[t >= lo] - lo, c[t >= lo]) for n, (b, t, c) in p["writers"].items()}
            pos = {n: torch.unique(t, return_inverse=True) for n, (_, t, _) in R.items() if n.endswith("o_proj")}
            p[key] = (R, Eg, new, pos)
        return p[key]

    def _pair_rows(self, i: int, prs: list, q: torch.Tensor, k: torch.Tensor, aq: torch.Tensor, ak: torch.Tensor, sq: torch.Tensor,
                   u: dict, p: dict, rows_at, B: int) -> torch.Tensor:
        """Layer i's attention pattern [B, H, rows, T] (float32) with each read of prs (plan["pairs"][i]: row, query
        position, stated pairs, the split's query and key nodes) taking its score row from its stated query-key pairs,
        as goal_eval.GoalEval._pair_row: q_t = sum_a q_a + q_rest (the split's query nodes at t, rank one: activation x
        mask x U row, and the rest) and
        k_p the same; the rest-rest term (the background) and each stated pair's, rotary at t and p. q, k: the rotated
        queries (of the rows computed: rows_at, None for every position) and keys [B, H, T, hd]; aq, ak their sites'
        activations ([B, rows, K], [B, T, K]); sq the query site's masks at the rows (query nodes at 1)."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        nq, nk = native.site_name(i, "q_proj"), native.site_name(i, "k_proj")
        Uq, Uk = self._w(nq)[1].float(), self._w(nk)[1].float()
        at = torch.arange(T, device=self.dev) if rows_at is None else rows_at
        S = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
        sk = u[nk].expand(B, T, -1)
        if nk in p["qk"]:
            sk = sk.index_put(p["qk"][nk], torch.ones((), device=self.dev))
        half = hd // 2

        def rot(vec, pos):  # [H, hd] at position pos
            return vec * tg.cos[pos] + torch.cat([-vec[..., half:], vec[..., :half]], -1) * tg.sin[pos]
        place = {int(t): r for r, t in enumerate(at.tolist())}
        for b, t, stated, qs, ks in prs:
            r = place.get(t)
            if r is None:  # a read whose position this run does not compute (the last layer at the targets alone)
                continue
            m = sq[b if sq.shape[0] > 1 else 0, r if sq.shape[1] > 1 else 0]
            qpart = {c: rot((aq[b, r, c] * m[c] * Uq[c]).float().view(H, hd), t) for c in qs}
            kpart = {(pk, c): rot((ak[b, pk, c] * sk[b if sk.shape[0] > 1 else 0, pk, c] * Uk[c]).float().view(H, hd), pk)
                     for pk, c in ks}
            q_rest = q[b, :, r].float() - sum(qpart.values()) if qpart else q[b, :, r].float()
            k_rest = k[b, :, :t + 1].float().clone()
            for (pk, c), kb in kpart.items():
                k_rest[:, pk] -= kb
            row = (q_rest[:, None] * k_rest).sum(-1) / math.sqrt(hd)
            for qa, kb in stated:
                qv = q_rest if qa is None else qpart[qa]
                if kb is None:
                    row = row + (qv[:, None] * k_rest).sum(-1) / math.sqrt(hd)
                else:
                    row = row.clone()
                    row[:, kb[0]] = row[:, kb[0]] + (qv * kpart[kb]).sum(-1) / math.sqrt(hd)
            S[b, :, r, :t + 1] = row
        return S.masked_fill(torch.arange(T, device=self.dev)[None, :] > at[:, None], float("-inf")).softmax(-1)

    def _resid_tables(self, p: dict, key: str, Wr: dict, Eg: dict, outs: dict) -> tuple:
        """The connections from residual writers (attention and MLP outputs) as one list per reader site and one into
        the prediction: each writer row's place among the residual writers' rows laid end to end in the order the run
        computes them, and its write vector's row in every residual site's U end to end (kept on the plan under key).
        A reader's top-up is then one gather and one index_add, not one of each per writer site."""
        names = [native.site_name(i, k) for i in range(self.tg.n_layer) for k in ("o_proj", "down_proj")]
        if not hasattr(self, "_Ures"):  # every residual writer site's write vectors, end to end
            self._Ures = (torch.cat([self.U[n] for n in names]), dict(zip(names, torch.tensor([0] + [self.width(n) for n in names[:-1]]).cumsum(0).tolist())))
        if key not in p:
            coff = self._Ures[1]
            off, at = {}, 0
            for wn in names:
                if wn in Wr:
                    off[wn] = at
                    at += len(Wr[wn][0])
            per = {}
            for (rn, wn), (ri, wi, j, fx) in Eg.items():
                if wn in off:
                    per.setdefault(rn, []).append((ri, off[wn] + wi, j, fx, coff[wn] + Wr[wn][2][wi]))
            cons = {rn: tuple(torch.cat(x) for x in zip(*lst)) for rn, lst in per.items()}
            lst = [(b, ti, off[wn] + wi, j, fx, coff[wn] + Wr[wn][2][wi]) for wn, (b, ti, wi, j, fx) in outs.items() if wn in off]
            p[key] = (off, cons, tuple(torch.cat(x) for x in zip(*lst)) if lst else None)
        return p[key]

    def _graph_hidden(self, p: dict, u: dict, ew, ow, B: int, fused: bool, patch: dict | None = None, rec: tuple | None = None,
                      keep: dict | None = None, reuse: tuple | None = None, ewn=None, own=None, outputs=None,
                      rec_out=None, rows: list | None = None) -> torch.Tensor:
        """The final normed residual stream at the targets [B, len(targets), d] of the graph runs. fused (a search's
        steps): torch's fused norms and GELU, and the last layer's queries, attention output and MLP at the targets
        alone (with reads, every position). keep and reuse: graph_log_probs'."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        norm, act = self._steps(fused)
        R, Wr, Eg, outs = p["readers"], p["writers"], p["edges"], p["out"]
        c = p["semantics"] == "c"
        given = {}  # per layer the rows given in place of the run's own: (row, position, [H, t + 1])
        for b, d in enumerate(rows or []):
            for (layer, t), row in (d or {}).items():
                given.setdefault(layer, []).append((b, t, row.to(self.dev, torch.float32)))
        if c:  # the circuit's nodes at 1 in the stream, every other subcomponent at u
            one = torch.ones((), device=self.dev)
            u = {n: m.expand(B, T, -1).index_put(p["nodes"][n], one) if n in p["nodes"] else m for n, m in u.items()}
        narrow = fused and rec is None
        blocks = {k[:-len(BK)] for k in R if k.endswith(BK)}  # sites with blocks (plan's keys site + BK)
        if blocks and (fused or reuse is not None or keep is not None or ewn is not None or own is not None):
            raise ValueError("blocks run in float32 runs of every position (no reuse, keep or norm strengths)")
        bvec, bin_, zb = {}, {r for r, wk in Eg if wk.endswith(BK)}, {}  # blocks' outputs; readers fed by blocks; blocks' inputs
        sel, S = None, set()  # the last layer's positions (the targets) and its sites that run only there
        aw, sw = {}, {}  # per writer site, its writer rows' activations and 1 - u
        lo = 0 if reuse is None else reuse[2]  # the first position computed (reuse: every state before it the base's)
        if lo:
            R, Eg, new, pos_from = self._from_tables(p, lo)
            off, cons, cout = self._resid_tables(p, ("_resid_from", lo), Wr, Eg, outs)
        else:
            off, cons, cout = self._resid_tables(p, "_resid", Wr, Eg, outs)
        if keep is not None:
            keep.update(k=[], v=[], aw={}, ak=[])
        res_a, res_s = [], []  # the residual writers' rows' activations and 1 - u, in their order (off)

        def cut(n, m):  # site n's [B or 1, T or 1, ...] tensor at the positions its steps run on
            if lo:
                return m[:, lo:] if m.shape[1] > 1 else m
            return m[:, sel] if n in S and m.shape[1] > 1 else m

        def at(m, b, t, c):  # a mask [B or 1, T or 1, K] at rows' (row, position, column)
            return m[b if m.shape[0] > 1 else torch.zeros_like(b), t if m.shape[1] > 1 else torch.zeros_like(t), c]

        def done(n, A):  # a site's activations [B, T, K], patched: its writer nodes' kept, the asked ones read
            if patch is not None and n in patch:
                A = torch.where(cut(n, patch[n][0]), cut(n, patch[n][1]).to(A.dtype), A)
            if rec is not None and n in rec[0]:
                rec[1][n] = A[rec[0][n]].float()
            if n in Wr:
                if lo:  # the writer rows before lo the base's, the rest this run's
                    rows, b, t, c_ = new[n]
                    aw[n], sw[n] = reuse[0]["aw"][n].index_copy(0, rows, A[b, t, c_]), 1 - at(u[n], *Wr[n])
                else:
                    aw[n], sw[n] = A[Wr[n]], 1 - at(cut(n, u[n]), *Wr[n])
                if keep is not None:
                    keep["aw"][n] = aw[n]
                if n in off:
                    res_a.append(aw[n])
                    res_s.append(sw[n])
            return A

        def share(wn, wi, e):  # a connection's extra share in its reader's input: native e (1 - u_w) a_w, C -(1 - e) u_w a_w
            return (e - 1) * (1 - sw[wn][wi]) * aw[wn][wi] if c else e * sw[wn][wi] * aw[wn][wi]

        def shares(rn, e_all=None):  # the connections into reader site rn: (writer site, reader rows, writer rows, share)
            e_all = ew if e_all is None else e_all
            for (r, wn), (ri, wi, j, fx) in Eg.items():
                if r == rn and not wn.endswith(BK):
                    yield wn, ri, wi, share(wn, wi, fx if e_all is None else e_all[j] * fx)

        def bshares(rk):  # the connections into reader key rk from blocks: (writer site, reader rows, writer rows, e - 1)
            for (r, wk), (ri, wi, j, fx) in Eg.items():
                if r == rk and wk.endswith(BK):
                    yield wk[:-len(BK)], ri, wi, (fx if ew is None else ew[j] * fx) - 1

        def btop(rk, width):  # [rows of rk, width]: its connections from blocks, (e - 1) x each block's output (in the circuit)
            out = torch.zeros(len(R[rk][0]), width, device=self.dev)
            for wn, ri, wi, e1 in bshares(rk):
                out = out.index_add(0, ri, e1[:, None] * bvec[wn][wi])
            return out

        def block_in(n, x, j):  # a residual reader site's blocks' inputs: their position's stream with their connections'
            rk = n + BK  # shares (norm j)
            b, t, _ = R[rk]
            inp = x[b, t] + topup(rk, x.shape[-1])
            return norm(inp + btop(rk, x.shape[-1]) if rk in bin_ else inp, j)

        def block_out(n, z):  # a site's blocks' outputs [rows, d_out] from their inputs z: a graph's own block its whole
            b, t, own_ = R[n + BK]  # output (every subcomponent and, live, the remainder), one its cuts name the run's own at u
            V, U, Rw = self._w(n)
            m = u[n][b if u[n].shape[0] > 1 else torch.zeros_like(b), t if u[n].shape[1] > 1 else torch.zeros_like(t)]
            run = ((z @ V) * m) @ U
            run = run + z @ Rw if self.fixed else run
            o = torch.where(own_.bool()[:, None], z @ self._whole(n), run)
            bvec[n] = o if self.b[n] is None else o + self.b[n]
            return bvec[n]

        def block_attn(no, nv, y, q, k, Pp):  # an attention output site's blocks' inputs: the attention output at their
            b, t, _ = R[no + BK]  # positions with their connections' shares (values through the pattern)
            inp = y[b, t].float()
            if not any(r == no + BK for r, _ in Eg):
                return inp
            if Pp is not None:  # [rows, H, T]
                P = Pp[b, :, t].float()
            else:
                pos, idx = torch.unique(t, return_inverse=True)
                z = (q[:, :, pos] @ k.transpose(-1, -2)).float() / math.sqrt(hd)
                P = z.masked_fill(torch.arange(T, device=self.dev)[None, :] > pos[:, None], float("-inf")).softmax(-1)[b, :, idx]
            for wn, ri, wi, s_ in shares(no + BK):
                pat = P[ri, :, Wr[wn][1][wi]]
                inp = inp.index_add(0, ri, s_[:, None] * pat.repeat_interleave(hd, -1) * self.U[nv].index_select(0, Wr[wn][2][wi]))
            for wn, ri, wi, e1 in bshares(no + BK):
                pat = P[ri, :, R[wn + BK][1][wi]]
                inp = inp.index_add(0, ri, e1[:, None] * pat.repeat_interleave(hd, -1) * bvec[wn][wi])
            return inp

        def resid_share(gi, j, fx, e_all):  # residual connections' shares from their writers' places gi
            e = fx if e_all is None else e_all[j] * fx
            a, sw_ = torch.cat(res_a)[gi], torch.cat(res_s)[gi]
            return (e - 1) * (1 - sw_) * a if c else e * sw_ * a

        def topup(rn, width, e_all=None):  # [readers, width]: each reader node's stated parents' extra share
            e_all = ew if e_all is None else e_all
            out = torch.zeros(len(R[rn][0]), width, device=self.dev)
            if rn in cons:  # every residual writer site's connections at once
                ri, gi, j, fx, uc = cons[rn]
                return _add_rows(out, ri, resid_share(gi, j, fx, e_all), lambda k: self._Ures[0].index_select(0, uc[k]))
            for wn, ri, wi, s in shares(rn, e_all):
                out = _add_rows(out, ri, s, lambda k, wn=wn, wi=wi: self.U[wn].index_select(0, Wr[wn][2][wi[k]]))
            return out

        def read(n, A, inp):  # reader nodes' activations, from their inputs [readers, d_in], into A
            return A.index_put(R[n], ((inp * self.V[n].index_select(1, R[n][2]).T).sum(-1)).to(A.dtype))

        def resid_in(n, x, j):  # residual readers' inputs: their position's stream with their parents' extra shares (norm j)
            b, t, _ = R[n]
            if ewn is None:
                inp = x[b, t] + topup(n, x.shape[-1])
                return norm(inp + btop(n, x.shape[-1]) if n in bin_ else inp, j)
            num, den = x[b, t] + topup(n, x.shape[-1]), x[b, t] + topup(n, x.shape[-1], ewn).float()  # the read; the norm's scale
            return tg.norm_parts(num, den, j)

        if lo:  # the tokens from lo on: every row's [T], or each row's [B, T]
            tok = (reuse[1] if torch.is_tensor(reuse[1]) else torch.tensor(reuse[1])).to(self.dev)
            x = tg.embed(tok[lo:], lo)[None].expand(B, -1, -1) if tok.dim() == 1 else tg.embed(tok[:, lo:], lo)
            causal = torch.arange(T, device=self.dev)[None, :] <= torch.arange(lo, T, device=self.dev)[:, None]  # [T - lo, T]
        else:
            x = self.emb.expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            if narrow and i == tg.n_layer - 1:  # the last layer: keys and values at every position, the rest at the targets
                R, Wr, Eg, outs, S = self._last_tables(p)
                sel = self._targets
                off, cons, cout = self._resid_tables(p, "_resid_last", Wr, Eg, outs)
            n1, n2 = 2 * i, 2 * i + 1
            h = norm(x, n1)
            A = {}
            for kd in ("q_proj", "k_proj", "v_proj"):
                n = name(kd)
                hn, xn = (h[:, sel], x[:, sel]) if n in S else (h, x)
                a = self._mm(hn, self._w(n)[0])
                A[kd] = done(n, read(n, a, resid_in(n, xn, n1)) if n in R else a)
                if n in blocks:
                    zb[n] = block_in(n, x, n1)

            def out(kd, z, scale):
                n = name(kd)
                _, U, Rw = self._w(n)
                o = (A[kd] * scale.to(A[kd].dtype)) @ U
                o = o + self._mm(z, Rw) if self.fixed else o
                o = o if self.b[n] is None else o + self.b[n]
                if n in blocks:  # its blocks' outputs, from their inputs (zb: with their connections' shares)
                    b, t, _ = p["readers"][n + BK]
                    o = o.index_put((b, t), block_out(n, zb.pop(n) if n in zb else z[b, t]).to(o.dtype))
                if outputs is not None and n in outputs:  # the site's whole output replaced at those positions
                    m, val = outputs[n]
                    o = torch.where(m[..., None], val.to(o.dtype), o)
                if rec_out is not None:
                    rec_out[n] = o.float()
                return o

            def scale(kd):  # query and key nodes at 1 in the pattern
                n = name(kd)
                m = u[n].expand(B, T, -1).index_put(p["qk"][n], torch.ones((), device=self.dev)) if n in p["qk"] else u[n]
                return cut(n, m)

            heads = lambda y: y.reshape(B, y.shape[1], H, hd).transpose(1, 2)  # noqa: E731
            nq, nv, no = name("q_proj"), name("v_proj"), name("o_proj")
            v = heads(out("v_proj", h, cut(nv, u[nv])))
            if lo:  # the keys and values before lo the base's
                k = tg._rope_from(heads(out("k_proj", h, scale("k_proj"))), lo, T).to(v.dtype)
                k, v = torch.cat([reuse[0]["k"][i][:, :, :lo], k], 2), torch.cat([reuse[0]["v"][i][:, :, :lo], v], 2)
            else:
                k = tg._rope(heads(out("k_proj", h, scale("k_proj"))), T).to(v.dtype)
            ak = A["k_proj"] if not lo else torch.cat([reuse[0]["ak"][i][:, :lo], A["k_proj"]], 1)  # the key activations [B, T, K]
            if keep is not None:
                keep["k"].append(k)
                keep["v"].append(v)
                keep["ak"].append(ak)
            if lo:  # the queries from lo on, each over the keys up to it
                q = tg._rope_from(heads(out("q_proj", h, scale("q_proj"))), lo, T).to(v.dtype)
                rows_at = torch.arange(lo, T, device=self.dev)
            elif nq in S:  # rotary at the targets, each target's query over the keys up to it
                q = heads(out("q_proj", h[:, sel], scale("q_proj")))
                half = hd // 2
                q = (q * tg.cos[sel] + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * tg.sin[sel]).to(v.dtype)
                rows_at = self._targets
            else:
                q = tg._rope(heads(out("q_proj", h, scale("q_proj"))), T).to(v.dtype)
                rows_at = None
            prs = p.get("pairs", {}).get(i)
            Pp = None
            if prs:  # reads with stated query-key pairs: their score rows from those pairs (_pair_rows), the attention explicit
                Pp = self._pair_rows(i, prs, q, k, A["q_proj"], ak, scale("q_proj"), u, p, rows_at, B)
                y = Pp.to(v.dtype) @ v
            elif lo:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            elif nq in S:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._causal_targets)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            if i in given:  # rows given at some reads: the pattern explicit (its query rows from lo), those rows replaced
                P = Pp.float() if Pp is not None else ((q @ k.transpose(-1, -2)).float() / math.sqrt(hd)).masked_fill(
                    ~(causal if lo else torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()), float("-inf")).softmax(-1)
                P = P.clone()
                for b, t, row in given[i]:
                    P[b, :, t - lo, :t + 1] = row
                Pp = P
                y = P.to(v.dtype) @ v
            y = y.transpose(1, 2).reshape(B, y.shape[2], -1)
            a = self._mm(y, self._w(no)[0])
            if no in R:
                b, t, _ = R[no]
                inp = y[b, t].float()
                if any(r == no for r, _ in Eg):  # the pattern at the readers' positions [B, H, positions, T]
                    if Pp is not None:  # every row's, the stated pairs' rows among them
                        P, idx = Pp, t
                    elif no in S:  # the queries are the targets'
                        P = ((q @ k.transpose(-1, -2)).float() / math.sqrt(hd)).masked_fill(~self._causal_targets, float("-inf")).softmax(-1)
                        idx = t
                    else:
                        pos, idx = pos_from[no] if lo else p["positions"][no]
                        z = (q[:, :, pos] @ k.transpose(-1, -2)).float() / math.sqrt(hd)
                        P = z.masked_fill(torch.arange(T, device=self.dev)[None, :] > pos[:, None] + lo, float("-inf")).softmax(-1)
                    for wn, ri, wi, s in shares(no):
                        pat = P[b[ri], :, idx[ri], Wr[wn][1][wi]]  # [n, H]: how much each reader's position attends to its writer's
                        inp = _add_rows(inp, ri, s, lambda k, pat=pat, wn=wn, wi=wi:
                                        pat[k].repeat_interleave(hd, -1) * self.U[nv].index_select(0, Wr[wn][2][wi[k]]))
                    for wn, ri, wi, e1 in bshares(no):  # a value block's whole value through the pattern
                        pat = P[b[ri], :, idx[ri], R[wn + BK][1][wi]]
                        inp = inp.index_add(0, ri, e1[:, None] * pat.repeat_interleave(hd, -1) * bvec[wn][wi])
                a = read(no, a, inp)
            if no in blocks:
                zb[no] = block_attn(no, nv, y, q, k, Pp)
            A["o_proj"] = done(no, a)
            if no in S:
                x = x[:, sel]
            x = x + out("o_proj", y, cut(no, u[no]))
            h = norm(x, n2)
            nf, nd = name("c_fc"), name("down_proj")
            a = self._mm(h, self._w(nf)[0])
            A["c_fc"] = done(nf, read(nf, a, resid_in(nf, x, n2)) if nf in R else a)
            if nf in blocks:
                zb[nf] = block_in(nf, x, n2)
            pre = out("c_fc", h, cut(nf, u[nf]))
            g = act(pre)
            a = self._mm(g, self._w(nd)[0])
            if nd in R:
                b, t, _ = R[nd]
                inp = pre[b, t] + topup(nd, pre.shape[-1])
                a = read(nd, a, act(inp + btop(nd, pre.shape[-1]) if nd in bin_ else inp))
            if nd in blocks:  # its blocks inside the GELU
                b, t, _ = R[nd + BK]
                inp = pre[b, t] + topup(nd + BK, pre.shape[-1])
                zb[nd] = act(inp + btop(nd + BK, pre.shape[-1]) if nd + BK in bin_ else inp)
            A["down_proj"] = done(nd, a)
            x = x + out("down_proj", g, cut(nd, u[nd]))
        z = x if sel is not None else x[:, self._targets - lo]
        if cout is not None and own is not None:  # the read and the norm's scale with their own strengths
            b, ti, gi, j, fx, uc = cout
            num = z.index_put((b, ti), resid_share(gi, j, fx, ow)[:, None] * self._Ures[0].index_select(0, uc), accumulate=True)
            den = z.index_put((b, ti), resid_share(gi, j, fx, own)[:, None] * self._Ures[0].index_select(0, uc), accumulate=True).float()
            return tg.norm_parts(num, den, -1)
        if cout is not None:  # the connections into the prediction, every writer site's at once
            b, ti, gi, j, fx, uc = cout
            z = z.index_put((b, ti), resid_share(gi, j, fx, ow)[:, None] * self._Ures[0].index_select(0, uc), accumulate=True)
        for wk, (b, ti, wi, j, fx) in outs.items():  # blocks into the prediction
            if wk.endswith(BK):
                z = z.index_put((b, ti), ((fx if ow is None else ow[j] * fx) - 1)[:, None] * bvec[wk[:-len(BK)]][wi], accumulate=True)
        return norm(z, -1)


def _pair_cut(key) -> bool:
    """A cuts key naming a read's pairs: ("pairs", layer, query position)."""
    return isinstance(key, tuple) and len(key) == 3 and key[0] == "pairs"


def read_pairs(nodes, layer: int, t: int) -> list:
    """Every query-key pair of a read (layer, query position t) split by nodes: (a query node of the layer at t or None,
    a key node of the layer at or before t or None), not both None (the background, always in)."""
    nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
    qs = [nd for nd in sorted(nodes) if nd[0] == nq and nd[1] == t and nd[2] != BLOCK]  # (a block: the read's rest)
    ks = [nd for nd in sorted(nodes) if nd[0] == nk and nd[1] <= t and nd[2] != BLOCK]
    return [(a, b) for a in qs + [None] for b in ks + [None] if a is not None or b is not None]


ROWS = 1 << 24  # entries of a connection group's [connections, width] product made at once (64 MB in float32)
PP_ENTRIES = 1 << 26  # floats in a path-patched batch's attention scores at once (256 MB in float32)
PP_ROWS = 512  # rows (prompts x graphs) in a path-patched batch of stated graphs


def _add_rows(out: torch.Tensor, ri: torch.Tensor, s: torch.Tensor, rows) -> torch.Tensor:
    """out.index_add(0, ri, s[:, None] * rows(all)) (rows(k): the connections k's [n, width] write vectors); without
    gradients a slice of ROWS entries at a time: a large graph's connections (EdgePruner's 32 copies of a few hundred
    nodes, ~10^5: 54 GB at once in teacher_v3 --edges) never take [connections, width] at once. (With gradients every
    product is kept for the backward pass anyway; torch.utils.checkpoint per slice gave wrong gradients here.)"""
    step = max(1, ROWS // out.shape[1])
    if len(ri) <= step or (torch.is_grad_enabled() and (s.requires_grad or out.requires_grad)):
        return out.index_add(0, ri, s[:, None] * rows(slice(None)))
    for s0 in range(0, len(ri), step):
        k = slice(s0, s0 + step)
        out = out.index_add(0, ri[k], s[k, None] * rows(k))
    return out


def _positions(p: dict) -> dict:
    """A plan with its attention-output readers' distinct positions and each reader's index among them."""
    p["positions"] = {n: torch.unique(t, return_inverse=True) for n, (_, t, _) in p["readers"].items() if n.endswith("o_proj")}
    return p


@contextlib.contextmanager
def _precision(precision: str, dev):
    cuda = str(dev).startswith("cuda")
    if precision == "tf32" and cuda:
        keep = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = keep
    else:  # bf16: the engine's own casts (Engine._dtype), not autocast's per product
        yield
