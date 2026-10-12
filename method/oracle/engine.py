"""The masked-run engine (#2951 graph oracle): VPD's masked forward pass of vpd4l on one text with per-site masks over
a chosen set of subcomponents, the remainders W - V U live or off, and the KL in bits of the model's next-token
distribution at the targets from the run's, differentiable in the masks. It serves searches that take gradients with
respect to masks (the teacher's mask search, the reward's adversary), batched over rows (samples x patterns x
answers x restarts), on any GPU.

  eng = Engine(nat, ids, targets, live=True, columns=None, compile=False)
  eng.kl(masks, precision="score") -> [B] bits
  eng.graph_kl(eng.plan(graph), u=None, ew=None, ow=None, precision="score") -> [B] bits

columns: per site the subcomponent indices masks address (a LongTensor), every other one off in every run; None, every
subcomponent. Give the union a caller's masks can ever switch on (VPD's answer and an adversary's patterns): a site's
products then run on those columns alone (a sum over fewer terms: float32-close to the full products, not equal to
the bit). masks: {site: [B, T, K]} floats (K = the site's columns), with or without gradients.
precision: "score", float32 with vpd_model's own norms and GELU (the numbers to report); "tf32" (CUDA: TF32 tensor-core
matmuls) and "bf16" (bfloat16 autocast), with torch's fused norms and GELU, for a search's steps only, whose result is
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
readers' positions).
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
                 compile: bool = False):
        import vpd_model

        self.nat, self.ids, self.targets, self.live = nat, list(ids), list(targets), live
        tg = nat.target
        self.tg, self.T, self.dev = tg, len(ids), nat.dev
        self.H, self.hd, self.eps = tg.n_head, tg.hd, tg.eps
        self.rms, self.gelu = vpd_model.rms, vpd_model.gelu_tanh
        with torch.no_grad():
            self.ref = nat.reference([self.ids], self.targets)
            self.emb = tg.wte[torch.tensor(self.ids, device=self.dev)][None]  # [1, T, d]
        self.columns = {n: (None if columns is None or columns.get(n) is None else columns[n].to(self.dev)) for n in nat.names}
        self.V, self.U, self.R = {}, {}, {}
        for n in nat.names:
            st, cols = tg.site(n), self.columns[n]
            self.V[n] = st.V if cols is None else st.V.index_select(1, cols).contiguous()  # [d_in, K]
            self.U[n] = st.U if cols is None else st.U.index_select(0, cols).contiguous()  # [K, d_out]
            self.R[n] = st.delta_T  # [d_in, d_out]: the remainder W - V U, transposed (native.model)
        self._search = torch.compile(self.hidden, dynamic=False) if compile else self.hidden

    def width(self, n: str) -> int:
        """The number of a site's columns (a mask's last dimension)."""
        return self.V[n].shape[1]

    def _site(self, n: str, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = ((h @ self.V[n]) * m) @ self.U[n]
        return out + h @ self.R[n] if self.live else out

    def _steps(self, fused: bool):
        """(norm, activation): torch's fused kernels, or vpd_model's own."""
        if fused:
            return (lambda x, w: F.rms_norm(x, (x.shape[-1],), w, self.eps)), (lambda x: F.gelu(x, approximate="tanh"))
        return (lambda x, w: self.rms(x, w, self.eps)), self.gelu

    def hidden(self, masks: dict, fused: bool) -> torch.Tensor:
        """The final normed residual stream at the targets [B, len(targets), d] of the masked run."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        B = next(iter(masks.values())).shape[0]
        norm, act = self._steps(fused)
        x = self.emb.expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            h = norm(x, tg.norms[2 * i])
            q, k, v = (self._site(name(kd), h, masks[name(kd)]).view(B, T, H, hd).transpose(1, 2)
                       for kd in ("q_proj", "k_proj", "v_proj"))
            y = F.scaled_dot_product_attention(tg._rope(q, T), tg._rope(k, T), v, is_causal=True)
            x = x + self._site(name("o_proj"), y.transpose(1, 2).reshape(B, T, -1), masks[name("o_proj")])
            h = norm(x, tg.norms[2 * i + 1])
            x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, masks[name("c_fc")])), masks[name("down_proj")])
        return norm(x, tg.ln_f)[:, self.targets]  # the final norm over every position, as vpd_model.Target.hidden

    def kl(self, masks: dict, precision: str = "score") -> torch.Tensor:
        """KL in bits of the model at the targets from each row's masked run [B] (summed over the targets)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        with _precision(precision, self.dev):
            h = self.hidden(masks, fused=False) if precision == "score" else self._search(masks, fused=True)
            lq = torch.log_softmax((h @ self.tg.wte.T).float(), -1)
        return carry.kl(self.ref, lq)

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

    def plan(self, graphs) -> dict:
        """Graphs' nodes and connections as index tensors over the engine's columns, for graph_kl. graphs: one
        native.Graph, run on every row, or a list of them, one per row. Per site the query and key nodes (rows,
        positions, columns), the reader nodes (those with stated parents) and the writer nodes as (rows, positions,
        columns); per (reader site, writer site) the connections (reader rows, writer rows, their index among the
        strengths); per writer site the connections into the prediction (rows, target index, writer rows, index)."""
        one = not isinstance(graphs, (list, tuple))
        graphs = [graphs] if one else list(graphs)
        if any(g.complete for g in graphs):
            raise ValueError("graph_kl runs graphs of stated connections (a complete graph is a masked run: kl)")
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

        readers, writers, edges, outs, qk, E, O = {}, {}, {}, {}, {}, 0, 0
        for b, g in enumerate(graphs):
            for r, ws in g.parents.items():
                ri = row(readers, b, r)
                for w in ws:
                    lst = edges.setdefault((r[0], w[0]), ([], [], []))
                    lst[0].append(ri)
                    lst[1].append(row(writers, b, w))
                    lst[2].append(E)
                    E += 1
            for w in g.out:
                if w[1] not in self.targets:
                    raise ValueError(f"{w} writes into the prediction but is not at a target")
                lst = outs.setdefault(w[0], ([], [], [], []))
                lst[0].append(b)
                lst[1].append(self.targets.index(w[1]))
                lst[2].append(row(writers, b, w))
                lst[3].append(O)
                O += 1
            for n, t, c in g.node_set():
                if n.endswith(("q_proj", "k_proj")):
                    qk.setdefault(n, set()).add((b, t, col(n, c)))
        ints = lambda v: torch.tensor(v, dtype=torch.long, device=dev)  # noqa: E731
        btc = lambda keys: tuple(ints([k[j] for k in keys]) for j in range(3))  # noqa: E731
        return {"B": None if one else len(graphs), "E": E, "O": O, "qk": {n: btc(sorted(v)) for n, v in qk.items()},
                "readers": {n: btc(list(rows)) for n, rows in readers.items()}, "writers": {n: btc(list(rows)) for n, rows in writers.items()},
                "edges": {k: tuple(ints(x) for x in v) for k, v in edges.items()}, "out": {k: tuple(ints(x) for x in v) for k, v in outs.items()}}

    def _rows(self, p: dict, B: int) -> dict:
        """A one-graph plan's tables for B rows (kept on the plan), row b's entries after row b - 1's."""
        if p["B"] is not None:
            if B != p["B"]:
                raise ValueError(f"the plan has {p['B']} graphs, one per row, not {B}")
            return p
        if B not in p.setdefault("tiled", {}):
            dev = self.dev
            bs = lambda k: torch.arange(B, device=dev).repeat_interleave(k)  # noqa: E731
            tile = lambda v, k: (v[None] + k * torch.arange(B, device=dev)[:, None]).flatten()  # noqa: E731
            btc = lambda t: (bs(len(t[0])), t[1].repeat(B), t[2].repeat(B))  # noqa: E731
            R = {n: len(v[0]) for n, v in p["readers"].items()}
            W = {n: len(v[0]) for n, v in p["writers"].items()}
            p["tiled"][B] = {
                "B": B, "E": B * p["E"], "O": B * p["O"], "qk": {n: btc(v) for n, v in p["qk"].items()},
                "readers": {n: btc(v) for n, v in p["readers"].items()}, "writers": {n: btc(v) for n, v in p["writers"].items()},
                "edges": {(rn, wn): (tile(ri, R[rn]), tile(wi, W[wn]), tile(j, p["E"])) for (rn, wn), (ri, wi, j) in p["edges"].items()},
                "out": {wn: (bs(len(b)), ti.repeat(B), tile(wi, W[wn]), tile(j, p["O"])) for wn, (b, ti, wi, j) in p["out"].items()}}
        return p["tiled"][B]

    def graph_kl(self, plan: dict, u: dict | None = None, ew: torch.Tensor | None = None, ow: torch.Tensor | None = None,
                 precision: str = "score") -> torch.Tensor:
        """KL in bits of the model at the targets from each row's graph run [B] (the module docstring: plan from
        Engine.plan, u, ew, ow)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        B = plan["B"] or max([1] + [t.shape[0] for t in (ew, ow) if t is not None] + [m.shape[0] for m in (u or {}).values()])
        p = self._rows(plan, B)
        with _precision(precision, self.dev):
            h = self._graph_hidden(p, self._zeros() if u is None else u, None if ew is None else ew.reshape(-1),
                                   None if ow is None else ow.reshape(-1), B, fused=precision != "score")
            lq = torch.log_softmax((h @ self.tg.wte.T).float(), -1)
        return carry.kl(self.ref, lq)

    def _zeros(self) -> dict:
        if not hasattr(self, "_u0"):
            self._u0 = {n: torch.zeros(1, 1, self.width(n), device=self.dev) for n in self.nat.names}
        return self._u0

    def _graph_hidden(self, p: dict, u: dict, ew, ow, B: int, fused: bool) -> torch.Tensor:
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        norm, act = self._steps(fused)
        R, Wr, Eg = p["readers"], p["writers"], p["edges"]
        aw, sw = {}, {}  # per writer site, its writer rows' activations and 1 - u

        def at(m, b, t, c):  # a mask [B or 1, T or 1, K] at rows' (row, position, column)
            return m[b if m.shape[0] > 1 else torch.zeros_like(b), t if m.shape[1] > 1 else torch.zeros_like(t), c]

        def done(n, A):  # a site's activations [B, T, K]: its writer nodes' kept
            if n in Wr:
                aw[n], sw[n] = A[Wr[n]], 1 - at(u[n], *Wr[n])
            return A

        def shares(rn):  # the connections into reader site rn: (writer site, reader rows, writer rows, e (1 - u_w) a_w)
            for (r, wn), (ri, wi, j) in Eg.items():
                if r == rn:
                    e = 1.0 if ew is None else ew[j]
                    yield wn, ri, wi, e * sw[wn][wi] * aw[wn][wi]

        def topup(rn, width):  # [readers, width]: each reader node's stated parents' extra share
            out = torch.zeros(len(R[rn][0]), width, device=self.dev)
            for wn, ri, wi, s in shares(rn):
                out = out.index_add(0, ri, s[:, None] * self.U[wn].index_select(0, Wr[wn][2][wi]))
            return out

        def read(n, A, inp):  # reader nodes' activations, from their inputs [readers, d_in], into A
            return A.index_put(R[n], ((inp * self.V[n].index_select(1, R[n][2]).T).sum(-1)).to(A.dtype))

        def resid_in(n, x, w):  # residual readers' inputs: their position's stream with their parents' extra shares
            b, t, _ = R[n]
            return norm(x[b, t] + topup(n, x.shape[-1]), w)

        x = self.emb.expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            n1, n2 = tg.norms[2 * i], tg.norms[2 * i + 1]
            h = norm(x, n1)
            A = {}
            for kd in ("q_proj", "k_proj", "v_proj"):
                n = name(kd)
                a = h @ self.V[n]
                A[kd] = done(n, read(n, a, resid_in(n, x, n1)) if n in R else a)

            def out(kd, z, scale):
                n = name(kd)
                o = (A[kd] * scale) @ self.U[n]
                return o + z @ self.R[n] if self.live else o

            def scale(kd):  # query and key nodes at 1 in the pattern
                n = name(kd)
                return u[n].expand(B, T, -1).index_put(p["qk"][n], torch.ones((), device=self.dev)) if n in p["qk"] else u[n]

            heads = lambda y: y.view(B, T, H, hd).transpose(1, 2)  # noqa: E731
            q, k = tg._rope(heads(out("q_proj", h, scale("q_proj"))), T), tg._rope(heads(out("k_proj", h, scale("k_proj"))), T)
            nv, no = name("v_proj"), name("o_proj")
            y = F.scaled_dot_product_attention(q, k, heads(out("v_proj", h, u[nv])), is_causal=True).transpose(1, 2).reshape(B, T, -1)
            a = y @ self.V[no]
            if no in R:
                b, t, _ = R[no]
                inp = y[b, t].float()
                if any(r == no for r, _ in Eg):  # the pattern at the readers' positions [B, H, positions, T]
                    pos, idx = torch.unique(t, return_inverse=True)
                    z = (q[:, :, pos] @ k.transpose(-1, -2)) / math.sqrt(hd)
                    P = z.masked_fill(torch.arange(T, device=self.dev)[None, :] > pos[:, None], float("-inf")).softmax(-1)
                    for wn, ri, wi, s in shares(no):
                        pat = P[b[ri], :, idx[ri], Wr[wn][1][wi]]  # [n, H]: how much each reader's position attends to its writer's
                        inp = inp.index_add(0, ri, s[:, None] * pat.repeat_interleave(hd, -1) * self.U[nv].index_select(0, Wr[wn][2][wi]))
                a = read(no, a, inp)
            A["o_proj"] = done(no, a)
            x = x + out("o_proj", y, u[no])
            h = norm(x, n2)
            nf, nd = name("c_fc"), name("down_proj")
            a = h @ self.V[nf]
            A["c_fc"] = done(nf, read(nf, a, resid_in(nf, x, n2)) if nf in R else a)
            pre = out("c_fc", h, u[nf])
            g = act(pre)
            a = g @ self.V[nd]
            if nd in R:
                b, t, _ = R[nd]
                a = read(nd, a, act(pre[b, t] + topup(nd, pre.shape[-1])))
            A["down_proj"] = done(nd, a)
            x = x + out("down_proj", g, u[nd])
        z = x[:, self.targets]
        for wn, (b, ti, wi, j) in p["out"].items():
            e = 1.0 if ow is None else ow[j]
            z = z.index_put((b, ti), (e * sw[wn][wi] * aw[wn][wi])[:, None] * self.U[wn].index_select(0, Wr[wn][2][wi]), accumulate=True)
        return norm(z, tg.ln_f)


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
    elif precision == "bf16":
        with torch.autocast(str(dev).split(":")[0], dtype=torch.bfloat16):
            yield
    else:
        yield
