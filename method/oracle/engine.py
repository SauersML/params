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
import math

import torch
import torch.nn.functional as F

import carry
import native

PRECISIONS = ("score", "tf32", "bf16")


class Engine:
    def __init__(self, nat: native.Native, ids: list[int], targets: list[int], live: bool = True, columns: dict | None = None,
                 compile: bool = False, outside: str = "off"):
        import vpd_model

        if outside not in ("off", "on"):
            raise ValueError("outside is off or on")
        self.nat, self.ids, self.targets, self.live = nat, list(ids), list(targets), live
        self.fixed = live or outside == "on"  # whether every run adds x R (the remainder, and with outside on the rest)
        tg = nat.target
        self.tg, self.T, self.dev = tg, len(ids), nat.dev
        self.H, self.hd, self.eps = tg.n_head, tg.hd, tg.eps
        self.rms, self.gelu = vpd_model.rms, vpd_model.gelu_tanh
        with torch.no_grad():
            self.ref = nat.reference([self.ids], self.targets)
            self.emb = tg.wte[torch.tensor(self.ids, device=self.dev)][None]  # [1, T, d]
        self._targets = torch.tensor(self.targets, device=self.dev)  # (indexing by a list copies it to the device: a synchronization)
        self._causal_targets = torch.arange(self.T, device=self.dev)[None, :] <= self._targets[:, None]  # [targets, T]
        self.columns = {n: (None if columns is None or columns.get(n) is None else columns[n].to(self.dev)) for n in nat.names}
        self.V, self.U, self.R = {}, {}, {}
        for n in nat.names:
            st, cols = tg.site(n), self.columns[n]
            self.V[n] = st.V if cols is None else st.V.index_select(1, cols).contiguous()  # [d_in, K]
            self.U[n] = st.U if cols is None else st.U.index_select(0, cols).contiguous()  # [K, d_out]
            self.R[n] = st.delta_T  # [d_in, d_out]: the remainder W - V U, transposed (native.model)
            if outside == "on" and cols is not None:  # every other subcomponent on, with the remainder or alone
                self.R[n] = ((st.W.T if live else st.V @ st.U) - self.V[n] @ self.U[n]).contiguous()
        self._dt, self._cast = None, {}  # the products' dtype in the current run (None: float32), the weights' casts
        self._search = torch.compile(self.hidden, dynamic=False) if compile else self.hidden

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
        return out + h @ Rw if self.fixed else out

    def _steps(self, fused: bool):
        """(norm, activation): torch's fused kernels, or vpd_model's own."""
        if fused:
            return (lambda x, w: F.rms_norm(x, (x.shape[-1],), w, self.eps)), (lambda x: F.gelu(x, approximate="tanh"))
        return (lambda x, w: self.rms(x, w, self.eps)), self.gelu

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
            h = norm(x, tg.norms[2 * i])
            if search and i == tg.n_layer - 1:
                return self._last(masks, x, h, name, norm, act, B)
            q, k, v = (self._site(name(kd), h, masks[name(kd)], rec).view(B, T, H, hd).transpose(1, 2)
                       for kd in ("q_proj", "k_proj", "v_proj"))
            y = F.scaled_dot_product_attention(tg._rope(q, T).to(v.dtype), tg._rope(k, T).to(v.dtype), v, is_causal=True)
            x = x + self._site(name("o_proj"), y.transpose(1, 2).reshape(B, T, -1), masks[name("o_proj")], rec)
            h = norm(x, tg.norms[2 * i + 1])
            x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, masks[name("c_fc")], rec)), masks[name("down_proj")], rec)
        return norm(x, tg.ln_f)[:, self._targets]  # the final norm over every position, as vpd_model.Target.hidden

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
        h = norm(x, tg.norms[2 * tg.n_layer - 1])
        x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, at(masks[name("c_fc")]))), at(masks[name("down_proj")]))
        return norm(x, tg.ln_f)

    def kl(self, masks: dict, precision: str = "score") -> torch.Tensor:
        """KL in bits of the model at the targets from each row's masked run [B] (summed over the targets)."""
        return carry.kl(self.ref, self.log_probs(masks, precision))

    def log_probs(self, masks: dict, precision: str = "score", reads: dict | None = None):
        """Each row's masked run's next-token log probabilities at the targets [B, len(targets), V], float32. reads:
        {site: (rows, positions, columns)} (columns the engine's): also those subcomponents' activations, (log q,
        {site: [n]}) ("score" only)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        if reads is not None and precision != "score":
            raise ValueError("reads come from the float32 run (score)")
        rec = None if reads is None else (reads, {})
        with _precision(precision, self.dev), self._dtype(precision):
            h = self.hidden(masks, search=False, rec=rec) if precision == "score" else self._search(masks, search=True)
        lq = torch.log_softmax(h.float() @ self.tg.wte.T, -1)  # the logits in float32 at every precision
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

    def plan(self, graphs, semantics: str = "native", cuts=None, free=None) -> dict:
        """Graphs' nodes and connections as index tensors over the engine's columns, for graph_kl. graphs: one
        native.Graph, run on every row, or a list of them, one per row. semantics (module docstring): "native"
        (native.run's: the stated connections) or "c" (C's: every connection the model has among the nodes, the
        unstated ones at strength 0). cuts: per graph (one dict, or a list beside graphs) {reader node or None (the
        prediction): [writer nodes]}, connections at strength 0 besides the unstated ones. In C's semantics a graph may
        be complete (VPD's answer, the whole model: native.Graph(masks=...)): it runs as u (give it 1 on the graph's
        subcomponents), its subcomponents reading each other in full but for the cuts and free; free: per complete
        graph (one dict, or a list beside graphs) {reader node: [writer nodes]}, connections at the strengths ew gives
        (after the stated ones, in free's order; a cut one stays at 0): teacher_reads' value reads, each row of one
        plan cutting its own subset. Per site the circuit's nodes,
        the query and key nodes, the reader nodes (with a connection in) and the writer nodes as (rows, positions,
        columns); per (reader site, writer site) the connections (reader rows, writer rows, their index among the
        strengths, 1 for a stated one and 0 for an unstated or cut one); per writer site the connections into the
        prediction (rows, target index, writer rows, index, 1 or 0)."""
        if semantics not in ("native", "c"):
            raise ValueError("semantics is native or c")
        one = not isinstance(graphs, (list, tuple))
        graphs = [graphs] if one else list(graphs)
        cuts = [cuts] if one else (list(cuts) if cuts is not None else [None] * len(graphs))
        free = [free] if one else (list(free) if free is not None else [None] * len(graphs))
        keys = [(id(g), id(c), id(f)) for g, c, f in zip(graphs, cuts, free)]
        if not one and len(set(keys)) < len(keys):  # a graph given more than once (with the same cuts): each planned
            single = {}                              # once, the rows' plans laid end to end (_stack)
            for g, c, f, k in zip(graphs, cuts, free, keys):
                if k not in single:
                    single[k] = (len(single), self.plan(g, semantics, c, f))
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

        def row(table, b, nd):
            rows = table.setdefault(nd[0], {})
            return rows.setdefault((b, nd[1], col(nd[0], nd[2])), len(rows))

        readers, writers, edges, outs, qk, nodes, E, O = {}, {}, {}, {}, {}, {}, 0, 0
        pairs = {}  # per layer, each read with stated pairs: (row, query position, [(query column | None, (key position, key
        # column) | None)], the split's query columns at the position, its (key position, key column)s)
        for b, g in enumerate(graphs):
            pcut = {k[1:]: v for k, v in (cuts[b] or {}).items() if _pair_cut(k)}  # (layer, t): (split nodes, pairs removed)
            cuts[b] = {r: ws for r, ws in (cuts[b] or {}).items() if not _pair_cut(r)}
            cut = {(w, r) for r, ws in cuts[b].items() for w in ws}
            if g.complete:  # its subcomponents read each other in full (u): only the cuts and free are connections here
                fr = [(w, r) for r, ws in (free[b] or {}).items() for w in ws]
                if any(r is None for _, r in fr):
                    raise ValueError("free connections are into nodes")
                at = {e: E + k for k, e in enumerate(fr)}
                E += len(fr)
                par = {}
                for r, ws in [*(cuts[b] or {}).items(), *(free[b] or {}).items()]:
                    if r is not None:
                        par[r] = list(dict.fromkeys([*par.get(r, []), *ws]))
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
                    lst = edges.setdefault((r[0], w[0]), ([], [], [], []))
                    lst[0].append(ri)
                    lst[1].append(row(writers, b, w))
                    lst[2].append(at.get((w, r), 0))
                    lst[3].append(float((w, r) in at and (w, r) not in cut))
            for w in out:
                if w[1] not in self.targets:
                    raise ValueError(f"{w} writes into the prediction but is not at a target")
                lst = outs.setdefault(w[0], ([], [], [], [], []))
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
                nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
                pairs.setdefault(layer, []).append((
                    b, t, [(None if qa is None else col(nq, qa[2]), None if kb is None else (kb[1], col(nk, kb[2]))) for qa, kb in keep],
                    [col(nq, c) for n, tt, c in split if n == nq and tt == t], [(tt, col(nk, c)) for n, tt, c in split if n == nk and tt <= t]))
            if g.complete:
                continue
            for n, t, c in g.node_set():
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
                        outputs: dict | None = None, rec_out: dict | None = None):
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
        receives every site's output [B, T, d_out] (float32 runs without reuse)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        if (keep is not None or reuse is not None) and (precision != "score" or patch is not None or reads is not None):
            raise ValueError("keep and reuse are float32 runs without patch or reads")
        if (outputs is not None or rec_out is not None) and (precision != "score" or reuse is not None):
            raise ValueError("outputs and rec_out are float32 runs of every position")
        if (ewn is not None or own is not None) and precision != "score":
            raise ValueError("separate norm strengths are float32 runs (vpd_model's own norm)")
        if reuse is not None and not 0 < reuse[2] <= min(self.targets):
            raise ValueError("reuse starts after position 0 and no later than the first target")
        B = plan["B"] or max([1] + [t.shape[0] for t in (ew, ow) if t is not None] + [m.shape[0] for m in (u or {}).values()])
        p = self._rows(plan, B)
        with _precision(precision, self.dev), self._dtype(precision):
            rec = None if reads is None else (reads, {})
            h = self._graph_hidden(p, self._zeros() if u is None else u, None if ew is None else ew.reshape(-1),
                                   None if ow is None else ow.reshape(-1), B, precision != "score", patch, rec, keep, reuse,
                                   None if ewn is None else ewn.reshape(-1), None if own is None else own.reshape(-1), outputs, rec_out)
        lq = torch.log_softmax(h.float() @ self.tg.wte.T, -1)
        return lq if rec is None else (lq, rec[1])

    @staticmethod
    def tile_states(states: dict, times: int) -> dict:
        """keep's states of a run on B rows as those of the run on the same rows repeated times over (rows b, B + b, ...
        the same: the plan of the graphs' list repeated, u, ew and ow repeated the same way)."""
        return {"k": [k.repeat(times, 1, 1, 1) for k in states["k"]], "v": [v.repeat(times, 1, 1, 1) for v in states["v"]],
                "aw": {n: a.repeat(times) for n, a in states["aw"].items()}, "ak": [a.repeat(times, 1, 1) for a in states["ak"]]}

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
            lst = [(b, ti, off[wn] + wi, j, fx, coff[wn] + Wr[wn][2][wi]) for wn, (b, ti, wi, j, fx) in outs.items()]
            p[key] = (off, cons, tuple(torch.cat(x) for x in zip(*lst)) if lst else None)
        return p[key]

    def _graph_hidden(self, p: dict, u: dict, ew, ow, B: int, fused: bool, patch: dict | None = None, rec: tuple | None = None,
                      keep: dict | None = None, reuse: tuple | None = None, ewn=None, own=None, outputs=None,
                      rec_out=None) -> torch.Tensor:
        """The final normed residual stream at the targets [B, len(targets), d] of the graph runs. fused (a search's
        steps): torch's fused norms and GELU, and the last layer's queries, attention output and MLP at the targets
        alone (with reads, every position). keep and reuse: graph_log_probs'."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        norm, act = self._steps(fused)
        R, Wr, Eg, outs = p["readers"], p["writers"], p["edges"], p["out"]
        c = p["semantics"] == "c"
        if c:  # the circuit's nodes at 1 in the stream, every other subcomponent at u
            one = torch.ones((), device=self.dev)
            u = {n: m.expand(B, T, -1).index_put(p["nodes"][n], one) if n in p["nodes"] else m for n, m in u.items()}
        narrow = fused and rec is None
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
                if r == rn:
                    yield wn, ri, wi, share(wn, wi, fx if e_all is None else e_all[j] * fx)

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

        def resid_in(n, x, w):  # residual readers' inputs: their position's stream with their parents' extra shares
            b, t, _ = R[n]
            if ewn is None:
                return norm(x[b, t] + topup(n, x.shape[-1]), w)
            num, den = x[b, t] + topup(n, x.shape[-1]), x[b, t] + topup(n, x.shape[-1], ewn).float()  # the read; the norm's scale
            return w * (num.float() * torch.rsqrt(den.pow(2).mean(-1, keepdim=True) + self.eps)).to(num.dtype)

        if lo:  # the tokens from lo on: every row's [T], or each row's [B, T]
            tok = (reuse[1] if torch.is_tensor(reuse[1]) else torch.tensor(reuse[1])).to(self.dev)
            x = tg.wte[tok[lo:]][None].expand(B, -1, -1) if tok.dim() == 1 else tg.wte[tok[:, lo:]]
            causal = torch.arange(T, device=self.dev)[None, :] <= torch.arange(lo, T, device=self.dev)[:, None]  # [T - lo, T]
        else:
            x = self.emb.expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            if narrow and i == tg.n_layer - 1:  # the last layer: keys and values at every position, the rest at the targets
                R, Wr, Eg, outs, S = self._last_tables(p)
                sel = self._targets
                off, cons, cout = self._resid_tables(p, "_resid_last", Wr, Eg, outs)
            n1, n2 = tg.norms[2 * i], tg.norms[2 * i + 1]
            h = norm(x, n1)
            A = {}
            for kd in ("q_proj", "k_proj", "v_proj"):
                n = name(kd)
                hn, xn = (h[:, sel], x[:, sel]) if n in S else (h, x)
                a = self._mm(hn, self._w(n)[0])
                A[kd] = done(n, read(n, a, resid_in(n, xn, n1)) if n in R else a)

            def out(kd, z, scale):
                n = name(kd)
                _, U, Rw = self._w(n)
                o = (A[kd] * scale.to(A[kd].dtype)) @ U
                o = o + self._mm(z, Rw) if self.fixed else o
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
                a = read(no, a, inp)
            A["o_proj"] = done(no, a)
            if no in S:
                x = x[:, sel]
            x = x + out("o_proj", y, cut(no, u[no]))
            h = norm(x, n2)
            nf, nd = name("c_fc"), name("down_proj")
            a = self._mm(h, self._w(nf)[0])
            A["c_fc"] = done(nf, read(nf, a, resid_in(nf, x, n2)) if nf in R else a)
            pre = out("c_fc", h, cut(nf, u[nf]))
            g = act(pre)
            a = self._mm(g, self._w(nd)[0])
            if nd in R:
                b, t, _ = R[nd]
                a = read(nd, a, act(pre[b, t] + topup(nd, pre.shape[-1])))
            A["down_proj"] = done(nd, a)
            x = x + out("down_proj", g, cut(nd, u[nd]))
        z = x if sel is not None else x[:, self._targets - lo]
        if cout is not None and own is not None:  # the read and the norm's scale with their own strengths
            b, ti, gi, j, fx, uc = cout
            num = z.index_put((b, ti), resid_share(gi, j, fx, ow)[:, None] * self._Ures[0].index_select(0, uc), accumulate=True)
            den = z.index_put((b, ti), resid_share(gi, j, fx, own)[:, None] * self._Ures[0].index_select(0, uc), accumulate=True).float()
            return tg.ln_f * (num.float() * torch.rsqrt(den.pow(2).mean(-1, keepdim=True) + self.eps)).to(num.dtype)
        if cout is not None:  # the connections into the prediction, every writer site's at once
            b, ti, gi, j, fx, uc = cout
            z = z.index_put((b, ti), resid_share(gi, j, fx, ow)[:, None] * self._Ures[0].index_select(0, uc), accumulate=True)
        return norm(z, tg.ln_f)


def _pair_cut(key) -> bool:
    """A cuts key naming a read's pairs: ("pairs", layer, query position)."""
    return isinstance(key, tuple) and len(key) == 3 and key[0] == "pairs"


def read_pairs(nodes, layer: int, t: int) -> list:
    """Every query-key pair of a read (layer, query position t) split by nodes: (a query node of the layer at t or None,
    a key node of the layer at or before t or None), not both None (the background, always in)."""
    nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
    qs = [nd for nd in sorted(nodes) if nd[0] == nq and nd[1] == t]
    ks = [nd for nd in sorted(nodes) if nd[0] == nk and nd[1] <= t]
    return [(a, b) for a in qs + [None] for b in ks + [None] if a is not None or b is not None]


ROWS = 1 << 24  # entries of a connection group's [connections, width] product made at once (64 MB in float32)


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
