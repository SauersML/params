"""Neuron-to-neuron edges on an HF Qwen3 (the 10-15 direction: the units are the model's own MLP neurons, each with a
positive and a negative pole and its SwiGLU sign regime; attention heads are routes, read through the neurons on each
side). Given a text, a question's contrast at a target and a circuit's neurons, it states which neuron feeds which.

Contrast: log2 of the summed probability of the answer's tokens over that of the alternatives' at the target.
Background: the reference run, the model with each neuron's activation zeroed below its weak threshold (thr [layers,
d_mlp]: neuron_builder.Neurons', inframove's T2 calibration; None: the model itself). With background "circuit" every
strong candidate outside the circuit is also zeroed, as neuron_builder's circuit run: strong, not massive (|value| under
massive [layers], the layer's cut) and at positions 1 to the target, the first position and massive values live (the
question-scoped circuit's sufficiency run); with "model" every strong neuron runs.
Edges (candidates: every pair the architecture allows among the circuit's neurons):
  resid  writer (layer a, position p) -> reader (layer b > a, position p): the writer's down_proj write in the reader's
         MLP input;
  route  writer (layer a, position p) -> head (layer c, a < c <= b) -> reader (layer b, position q >= p): the head's
         output at q from its value at p (its OV), the head's attention pattern as in the run.
         This is a fixed-pattern value-path cut, not a query/key routing intervention or a full writer ablation.
         Each local value delta recomputes the head's input norm; multiple route deltas are summed at the reader.
Write weights (from the weights): w_gate and w_up, the reader's gate and up rows through its norm's scale times the
writer's down_proj column (a route: through the head's OV as well), per unit activation, without the norms' 1/rms.
A cut edge leaves its reader reading the stream less what the edge brings (resid: the writer's write; route: the change
in the head's output at the reader's position when the writer's write leaves the head's input at the writer's
position, exact through the input norm); the reader's activation is recomputed from that input and every later layer
runs on (measured path patching, one forward per set of cuts, sets batched). A run lists some edges (each cut or not)
and either leaves every other candidate as in the reference or cuts every other one as well (rest): every candidate
cut costs one sum per position (the circuit's writes there, and every head's change there from every writer's value
change), so no run lists more candidates than it keeps.
Pruning (every run with the rest cut, so the stated graph is the circuit's every edge): the candidates ranked by their
linear effect on the contrast (the contrast's gradient at the reader times the reader's input Jacobian times what the
edge brings; routes scored a head layer at a time, the best LIST listed, four times as many while the listed ones fall
short), then the smallest top-k set that alone keeps the contrast within TOL of the way from every edge's to none's
(either side), and a halving pass over it, B blocks a round (each tried alone, those passing tried together); each kept
edge's own cut from the graph measured last (its strength), and the graph's edges cut with every other edge on.

  python neuron_edges.py one MODEL TEXT.json OUT.json   (TEXT: ids, target, answer, alternatives, neurons)
  python neuron_edges.py record MODEL REC.json OUT.json --thr THR.pt [--adapter A] [--dtype bfloat16]
                                                       (a neuron_builder circuit in its reference)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

TOL = 0.1  # the contrast may move this share of the way from the run with every edge to the run with none, either way
MIN_TOL = 0.01  # bits: at least this much
LIST = 4096  # routes first listed for the top-k search (by linear effect); four times as many each time they fall short
MAX_LIST = 1 << 18  # (at most this many: the record says when the top-k search fell short there)
HALVE_BLOCKS = 8
B = 16  # cut sets a forward
CHUNK = 1024  # listed resid edges a step (B x CHUNK x d floats)


def weight(mod) -> torch.Tensor:
    """A linear module's effective weight [out, in]: its own, or under a LoRA adapter (peft) the base layer's plus each
    active adapter's B A times its scaling."""
    if not hasattr(mod, "base_layer"):
        return mod.weight
    W = mod.base_layer.weight
    for a in getattr(mod, "active_adapters", []):
        if a in mod.lora_A:
            W = W + (mod.lora_B[a].weight @ mod.lora_A[a].weight) * mod.scaling[a]
    return W


def log2_contrast(lp: torch.Tensor, answer: list[int], alternatives: list[int]) -> torch.Tensor:
    """log2 p(answer tokens) - log2 p(alternatives) from log probabilities lp [..., V]."""
    return (torch.logsumexp(lp[..., answer], -1) - torch.logsumexp(lp[..., alternatives], -1)) / math.log(2.0)


def sizes(n: int) -> list[int]:
    """1 to n on a log scale (two steps an octave), n included."""
    return sorted({min(n, int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(n, 1))) + 2)} | {n})


class Edges:
    """One circuit's candidate edges on one text (module docstring). Rows of cuts are bool [K, E] over an edge list
    (resid (writer, reader) and route (writer, (layer, head), reader) tuples; neurons (layer, position, index))."""

    def __init__(self, model, ids: list[int], target: int, answer: list[int], alternatives: list[int], neurons: list,
                 thr: torch.Tensor | None = None, background: str = "circuit", massive: torch.Tensor | None = None):
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = core.config
        self.core, self.layers = core, core.model.layers
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.eps = cfg.rms_norm_eps
        self.dev, self.dt = core.model.embed_tokens.weight.device, core.model.embed_tokens.weight.dtype
        self.T = target + 1
        self.ids, self.target, self.answer, self.alternatives = list(ids)[:self.T], target, list(answer), list(alternatives)
        self.thr = None if thr is None else thr.to(self.dev, torch.float32)
        self.massive = None if massive is None else massive.float().cpu()
        self.N = sorted({(int(L), int(p), int(j)) for L, p, j in neurons if p < self.T})  # (by layer first)
        self.at = {n: i for i, n in enumerate(self.N)}
        self.P = torch.tensor([n[1] for n in self.N], dtype=torch.long)
        self.Jn = torch.tensor([n[2] for n in self.N], dtype=torch.long)
        lay = torch.tensor([n[0] for n in self.N], dtype=torch.long)
        self.nb = [int((lay < L).sum()) for L in range(self.L + 1)]  # the circuit's neurons below layer L: the first nb[L]
        self.background = background
        self.keep = torch.zeros(self.L, self.T, cfg.intermediate_size, dtype=torch.bool, device=self.dev)
        for L, p, j in self.N:
            self.keep[L, p, j] = True
        self.resid_all = [(w, r) for w in self.N for r in self.N if w[0] < r[0] and w[1] == r[1]]
        self.n_routes = sum((r[0] - w[0]) * self.H for w in self.N for r in self.N if w[1] <= r[1] and w[0] < r[0])
        self._plans, self._cols, self._vcols, self._G, self._J = {}, {}, {}, None, None
        with torch.no_grad():
            self.base = self._run(torch.zeros(1, 0, dtype=torch.bool), [], keep_states=True)
        self.c_full = float(self.base["contrast"][0])

    # -- the run (each module in the model's dtype on its own card, everything between in float32) -------------------

    def _d(self, L: int) -> torch.device:
        """Layer L's card (a model split over cards: device_map)."""
        return self.layers[L].input_layernorm.weight.device

    @staticmethod
    def _f(mod, t: torch.Tensor) -> torch.Tensor:
        """mod on t in mod's dtype on mod's card, the result in float32."""
        w = next(mod.parameters())
        return mod(t.to(w.device, w.dtype)).float()

    @staticmethod
    def _w(mod, rows=None, cols=None) -> torch.Tensor:
        """A slice of a linear module's effective weight in float32 (rows and cols index it)."""
        W = weight(mod)
        W = W if rows is None else W[rows]
        return (W if cols is None else W[:, cols]).float()

    def _mlp_act(self, layer, x: torch.Tensor) -> torch.Tensor:
        """The neurons' activations from a pre-norm stream x [..., d]: act(gate(norm x)) * up(norm x), in float32."""
        n = self._f(layer.post_attention_layernorm, x)
        return layer.mlp.act_fn(self._f(layer.mlp.gate_proj, n)) * self._f(layer.mlp.up_proj, n)

    def _weak(self, L: int, h: torch.Tensor, js: torch.Tensor | None = None) -> torch.Tensor:
        """h with the activations under their weak threshold zeroed (js: the neurons h holds, else every one)."""
        if self.thr is None:
            return h
        t = (self.thr[L] if js is None else self.thr[L][js.to(self.thr.device)]).to(h.device)
        return h * (h.abs() >= t)

    def _cols_on(self, dev) -> torch.Tensor:
        """Each circuit neuron's down_proj column [N, d] in float32 on a card (kept per card)."""
        key = str(dev)
        if key not in self._cols:
            d = self.layers[0].mlp.down_proj.out_features
            self._cols[key] = (torch.stack([self._w(self.layers[L].mlp.down_proj, cols=j).to(dev) for L, _, j in self.N])
                               if self.N else torch.zeros(0, d, device=dev))
        return self._cols[key]

    def _vcol(self, c: int) -> torch.Tensor:
        """Each circuit neuron below layer c: its down_proj column through layer c's input norm scale and v_proj, without
        the norm's 1/rms [nb[c], KV hd] (kept)."""
        if c not in self._vcols:
            layer, dc = self.layers[c], self._d(c)
            g = layer.input_layernorm.weight.float().to(dc)
            self._vcols[c] = (self._cols_on(dc)[:self.nb[c]] * g) @ self._w(layer.self_attn.v_proj).T
        return self._vcols[c]

    def _dv(self, c: int, x: torch.Tensor, v: torch.Tensor, hw: torch.Tensor) -> torch.Tensor:
        """The change in layer c's values at each circuit writer's position when its write there leaves layer c's input
        (the writers below c; x [K, T, d] the stream entering layer c, v [K, KV, T, hd] its values, hw [K, W] the
        writers' activations) [K, W, KV, hd]. Exact through the RMS norm: v(x) - v(x - h col) = (v(x) - b)(r' - r)/r' +
        h W_v(g col)/r', with r and r' the norm's rms of x and of x - h col."""
        K, W = hw.shape
        dc = x.device
        cols, Pw = self._cols_on(dc)[:W], self.P[:W].to(dc)
        r2 = (x.pow(2).mean(-1) + self.eps)[:, Pw]  # [K, W]
        dots = (x @ cols.T)[:, Pw, torch.arange(W, device=dc)]  # each writer's column against the stream at its position
        dr2 = (hw * hw * cols.pow(2).sum(-1) - 2 * hw * dots) / x.shape[-1]  # r'^2 - r^2
        r, r1 = r2.sqrt(), (r2 + dr2).sqrt()
        vp = v[:, :, Pw].transpose(1, 2).reshape(K, W, -1)
        b = self.layers[c].self_attn.v_proj.bias
        if b is not None:
            vp = vp - b.float().to(dc)
        dv = vp * (dr2 / ((r + r1) * r1))[..., None] + hw[..., None] * self._vcol(c)[None] / r1[..., None]
        return dv.view(K, W, self.KV, self.hd)

    def _plan(self, edges: list) -> dict:
        """Index tables of an edge list: resid cuts by the writer's layer, routes by layer and head (with their readers
        once each)."""
        hit = self._plans.get(id(edges))
        if hit is not None and hit[0] is edges:
            return hit[1]
        resid, route = {}, {}
        for e, ed in enumerate(edges):
            if len(ed) == 2:
                resid.setdefault(ed[0][0], []).append((e, self.at[ed[0]], self.at[ed[1]]))
            else:
                route.setdefault(ed[1], []).append((e, self.at[ed[0]], self.at[ed[2]]))
        t = lambda xs: tuple(torch.tensor([x[i] for x in xs], dtype=torch.long) for i in range(3))  # noqa: E731
        plan = {"resid": {L: t(v) for L, v in resid.items()}, "route": {}, "heads": {}}
        for k, v in route.items():
            es, wi, ri = t(v)
            plan["route"][k] = (es, wi, ri, *ri.unique(return_inverse=True))
            plan["heads"].setdefault(k[0], []).append(k[1])
        self._plans = {id(edges): (edges, plan)}
        return plan

    def _run(self, cuts: torch.Tensor, edges: list, rest: bool = False, keep_states: bool = False,
             probe: list | None = None) -> dict:
        """The contrast of each row of cuts [K, len(edges)], every candidate not in edges as in the reference or (rest)
        cut as well. probe: per layer a tensor [T, d_mlp] added to the neurons' activations (zeros requiring grad: the
        contrast's gradient at them)."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        core, H, KV, hd, T, f, N = self.core, self.H, self.KV, self.hd, self.T, self._f, len(self.N)
        K = max(cuts.shape[0], 1)
        coef = (cuts.float() if len(edges) else torch.zeros(K, 0)) - float(rest)  # per listed edge: cut 1, as run 0,
        # kept against a cut rest -1 (added back)
        plan = self._plan(edges)
        x = core.model.embed_tokens(torch.tensor([self.ids], device=self.dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(T, device=self.dev)[None])
        D = torch.zeros(K, N, x.shape[-1])  # per circuit neuron, what its listed cut edges take from its input
        Hc = torch.zeros(K, N)  # the circuit neurons' activations in this run
        S = torch.zeros(K, T, x.shape[-1]) if rest else None  # (rest) per position, the circuit's writes so far ...
        R = torch.zeros(K, T, x.shape[-1]) if rest else None  # ... and every route's change in the head outputs so far
        P, Jn = self.P, self.Jn
        states = {"x": [], "A": [], "v": [], "h": []}
        for L, layer in enumerate(self.layers):
            d = self._d(L)  # (every tensor of this layer's work on its card)
            x, D, Hc, coef, P, Jn, cos, sin = (t.to(d) for t in (x, D, Hc, coef, P, Jn, cos, sin))
            if rest:
                S, R = S.to(d), R.to(d)
            kk, causal = torch.arange(K, device=d), torch.ones(T, T, dtype=torch.bool, device=d).tril()
            at = layer.self_attn
            hn = f(layer.input_layernorm, x)
            q = f(at.q_norm, f(at.q_proj, hn).view(K, T, H, hd)).transpose(1, 2)
            k = f(at.k_norm, f(at.k_proj, hn).view(K, T, KV, hd)).transpose(1, 2)
            v = f(at.v_proj, hn).view(K, T, KV, hd).transpose(1, 2)  # [K, KV, T, hd]
            q, k = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
            A = ((q @ repeat_kv(k, H // KV).transpose(-1, -2)) * at.scaling).masked_fill(~causal, float("-inf")).softmax(-1)
            z = A @ repeat_kv(v, H // KV)  # [K, H, T, hd]
            if keep_states:
                states["x"].append(x[0].detach())
                states["A"].append(A[0].detach())
                states["v"].append(v[0].detach())
            nb = self.nb[L]
            if nb and (rest or plan["heads"].get(L)):  # routes through this layer's heads
                dv = self._dv(L, x, v, Hc[:, :nb])  # [K, nb, KV, hd]
                Wo = self._w(at.o_proj)
                if rest:  # every one: each position's head outputs from every writer's value change, summed
                    dV = torch.zeros(K, T, KV, hd, device=d).index_add(1, P[:nb], dv).transpose(1, 2)
                    R = R + (A @ repeat_kv(dV, H // KV)).transpose(1, 2).reshape(K, T, H * hd) @ Wo.T
                for h in plan["heads"].get(L, []):  # the listed ones: the head's output change at the reader
                    es, wi, ri, ru, inv = (t.to(d) for t in plan["route"][(L, h)])
                    dz = coef[:, es, None] * A[:, h][kk[:, None], P[ri][None], P[wi][None]][..., None] * dv[:, wi, h // (H // KV)]
                    Z = torch.zeros(K, len(ru), hd, device=d).index_add(1, inv, dz)
                    D = D.index_add(1, ru, Z @ Wo[:, h * hd:(h + 1) * hd].T)
            x = x + f(at.o_proj, z.transpose(1, 2).reshape(K, T, H * hd))
            hm = self._weak(L, self._mlp_act(layer, x))
            if self.background == "circuit":  # (strong candidates outside the circuit zeroed; position 0 and massive live)
                off = ~self.keep[L].to(d)[None]
                off[:, 0] = False
                if self.massive is not None:
                    off = off & (hm.abs() < float(self.massive[L]))
                hm = hm.masked_fill(off, 0.0)
            rs = torch.arange(nb, self.nb[L + 1], device=d)  # the circuit's neurons of this layer
            if len(rs) and (len(edges) or rest):  # each from its own input
                inp = x[:, P[rs]] - D[:, rs]
                if rest:
                    inp = inp - S[:, P[rs]] - R[:, P[rs]]
                hr = self._mlp_act(layer, inp).gather(-1, Jn[rs][None, :, None].expand(K, -1, 1))[..., 0]
                hm = hm.index_put((kk[:, None], P[rs][None], Jn[rs][None]), self._weak(L, hr, Jn[rs]))
            if probe is not None:
                hm = hm + probe[L].to(d)[None]
            if keep_states:
                states["h"].append(hm[0].detach())
            if len(rs):
                Hc = Hc.index_copy(1, rs, hm[kk[:, None], P[rs][None], Jn[rs][None]])
                if rest:
                    S = S.index_add(1, P[rs], Hc[:, rs, None] * self._cols_on(d)[rs])
            if L in plan["resid"]:
                es, wi, ri = (t.to(d) for t in plan["resid"][L])
                cols = self._cols_on(d)
                for s0 in range(0, len(es), CHUNK):
                    e_, w_, r_ = es[s0:s0 + CHUNK], wi[s0:s0 + CHUNK], ri[s0:s0 + CHUNK]
                    D = D.index_add(1, r_, (coef[:, e_] * Hc[:, w_])[..., None] * cols[w_][None])
            x = x + f(layer.mlp.down_proj, hm)
        lp = torch.log_softmax(f(core.lm_head, f(core.model.norm, x[:, self.target])), -1).cpu()
        out = {"contrast": log2_contrast(lp, self.answer, self.alternatives), "lp": lp}
        if keep_states:
            out.update(states)
        return out

    @torch.no_grad()
    def contrasts(self, cuts: torch.Tensor, edges: list, rest: bool = False) -> list[float]:
        """The contrast of each row of cuts [K, len(edges)] (rest: every candidate not in edges cut as well), B rows a
        forward."""
        out = []
        for s0 in range(0, len(cuts), B):
            out += self._run(cuts[s0:s0 + B], edges, rest=rest)["contrast"].tolist()
        return out

    # -- the candidates' linear effects -----------------------------------------------------------------------------

    def _mid_stream(self, L: int) -> torch.Tensor:
        """The base run's stream entering layer L's MLP [T, d] (on layer L's card)."""
        at = self.layers[L].self_attn
        z = self.base["A"][L] @ self.base["v"][L].repeat_interleave(self.H // self.KV, 0)  # [H, T, hd]
        return self.base["x"][L] + self._f(at.o_proj, z.transpose(0, 1).reshape(self.T, -1))

    def jacobians(self) -> dict:
        """Each circuit neuron's activation's gradient with respect to its MLP input in the base run [d] (kept; on
        its layer's card)."""
        if self._J is None:
            J = {}
            for L in range(self.L):
                rs = [n for n in self.N if n[0] == L]
                if rs:
                    with torch.enable_grad():
                        xin = self._mid_stream(L)[[n[1] for n in rs]].detach().requires_grad_()
                        h = self._mlp_act(self.layers[L], xin).gather(-1, torch.tensor([[n[2]] for n in rs], device=xin.device))[:, 0]
                        (g,) = torch.autograd.grad(h.sum(), xin)
                    J.update(dict(zip(rs, g)))
            self._J = J
        return self._J

    def grads(self) -> torch.Tensor:
        """The contrast's gradient at each circuit neuron's activation in the base run [N] (kept)."""
        if self._G is None:
            probe = [torch.zeros(self.T, self.keep.shape[-1], device=self._d(L), requires_grad=True) for L in range(self.L)]
            with torch.enable_grad():
                self._run(torch.zeros(1, 0, dtype=torch.bool), [], probe=probe)["contrast"].sum().backward()
            self._G = torch.tensor([float(probe[n[0]].grad[n[1], n[2]]) for n in self.N])
        return self._G

    def _write(self, w) -> torch.Tensor:
        """A circuit neuron's write in the base run [d] (on its layer's card)."""
        return self.base["h"][w[0]][w[1], w[2]] * self._w(self.layers[w[0]].mlp.down_proj, cols=w[2])

    @torch.no_grad()
    def brings(self, edge) -> torch.Tensor:
        """What an edge brings into its reader's MLP input in the base run [d] (on the reader's card): the writer's
        write (resid), or the change in the head's output at the reader's position from the writer's write at its own
        (route)."""
        w, r = edge[0], edge[-1]
        write = self._write(w)
        if len(edge) == 2:
            return write.to(self._d(r[0]))
        c, h = edge[1]
        layer, kv, hd = self.layers[c], h // (self.H // self.KV), self.hd
        dc = self._d(c)
        vw = self._f(layer.self_attn.v_proj, self._f(layer.input_layernorm, self.base["x"][c][w[1]] - write.to(dc))).view(self.KV, hd)[kv]
        dz = self.base["A"][c][h, r[1], w[1]] * (self.base["v"][c][kv, w[1]] - vw)
        return (self._w(layer.self_attn.o_proj, cols=slice(h * hd, (h + 1) * hd)) @ dz).to(self._d(r[0]))

    @torch.no_grad()
    def linear_scores(self, keep: int) -> tuple[torch.Tensor, list, torch.Tensor]:
        """(resid scores over resid_all, the keep routes of largest |score| among every route, their scores): each
        edge's linear effect on the contrast, the contrast's gradient at the reader's activation times the reader's
        input Jacobian times what the edge brings (the contrast it supports: positive when its cut lowers the contrast).
        Routes are scored a head layer at a time (every writer below it, reader above it and head at once), the best
        keep carried."""
        G, J = self.grads(), self.jacobians()
        H, KV, hd, base, N, dev = self.H, self.KV, self.hd, self.base, len(self.N), self.dev
        h0 = torch.stack([self.base["h"][L][p, j].to(dev) for L, p, j in self.N]) if N else torch.zeros(0, device=dev)
        Jm = torch.stack([J[n].to(dev) for n in self.N]) if N else torch.zeros(0, self._cols_on(dev).shape[1], device=dev)
        M = Jm @ self._cols_on(dev).T  # [reader, writer]: the reader's activation per unit of the writer's
        wi = torch.tensor([self.at[e[0]] for e in self.resid_all], dtype=torch.long, device=dev)
        ri = torch.tensor([self.at[e[1]] for e in self.resid_all], dtype=torch.long, device=dev)
        rs_s = (G.to(dev)[ri] * M[ri, wi] * h0[wi]).cpu()
        best_s, best_i = torch.zeros(0), torch.zeros(0, 4, dtype=torch.long)  # (writer, layer, head, reader) indices
        for c in range(1, self.L):
            nb = self.nb[c]
            if not nb or nb == N:
                continue
            layer, dc = self.layers[c], self._d(c)
            Pd = self.P.to(dc)
            dv = self._dv(c, base["x"][c][None], base["v"][c][None], h0[:nb].to(dc)[None])[0]
            dv = dv[:, torch.arange(H, device=dc) // (H // KV)]  # [W, H, hd]
            U = torch.einsum("rd,dhk->rhk", Jm[nb:].to(dc), self._w(layer.self_attn.o_proj).view(-1, H, hd))  # [R, H, hd]
            a = base["A"][c][:, Pd[nb:]][:, :, Pd[:nb]].permute(1, 2, 0)  # [R, W, H]
            sc = (G.to(dc)[nb:, None, None] * a * torch.einsum("rhk,whk->rwh", U, dv)).flatten()
            ok = (Pd[:nb][None] <= Pd[nb:][:, None])[..., None].expand(-1, -1, H).flatten()
            n_ok = int(ok.sum())
            if not n_ok:
                continue
            t = torch.topk(sc.abs().masked_fill(~ok, -1.0), min(keep, n_ok)).indices
            idx = torch.stack([(t // H) % nb, torch.full_like(t, c), t % H, t // (nb * H) + nb], 1)
            best_s, best_i = torch.cat([best_s, sc[t].cpu()]), torch.cat([best_i, idx.cpu()])
            if len(best_s) > 2 * keep:  # (the best keep carried)
                top = torch.topk(best_s.abs(), keep).indices
                best_s, best_i = best_s[top], best_i[top]
        top = torch.argsort(best_s.abs(), descending=True)[:keep]
        routes = [(self.N[a], (int(c), int(h)), self.N[r]) for a, c, h, r in best_i[top].tolist()]
        return rs_s, routes, best_s[top]


def prune(ed: Edges, log=print) -> dict:
    """The stated graph (module docstring's pruning) and its record."""
    t0 = time.time()
    c_all = ed.c_full
    c_none = ed.contrasts(torch.zeros(1, 0, dtype=torch.bool), [], rest=True)[0]
    tol = max(TOL * abs(c_all - c_none), MIN_TOL)  # (two-sided: a kept set that overshoots is no more faithful)
    ok = lambda c: abs(c - c_all) <= tol  # noqa: E731
    measured, n_rt, done, t_rank = 1, LIST, 0, 0.0
    while True:  # the top-k search: k on a log scale over the listed candidates, more routes listed while none passes
        t1 = time.time()
        rs, routes, rt = ed.linear_scores(n_rt)
        cand, score = list(ed.resid_all) + routes, torch.cat([rs, rt]).abs()
        order = torch.argsort(score, descending=True)
        every = len(routes) == ed.n_routes
        n_ok = len(cand) if every else int((score >= score[len(rs):].min()).sum())  # (the prefix ranked as among all)
        t_rank += time.time() - t1
        ks = [k for k in sizes(n_ok) if k > done]
        listed = [cand[i] for i in order[:n_ok].tolist()]
        rows = torch.arange(n_ok)[None] >= torch.tensor(ks)[:, None]  # (the top k kept)
        cs = ed.contrasts(rows, listed, rest=True)
        measured += len(ks)
        hit = next((k for k, c in zip(ks, cs) if ok(c)), None)
        if hit is not None or every or n_rt >= MAX_LIST:
            break
        done, n_rt = n_ok, 4 * n_rt
    edges = listed[:n_ok if hit is None else hit]  # ranked
    E = len(edges)
    t_keep = time.time() - t0 - t_rank

    def kept_rows(sets):  # each set of kept edges as a cut row (every other edge cut)
        rows = torch.ones(len(sets), E, dtype=torch.bool)
        for i, st in enumerate(sets):
            rows[i, st] = False
        return rows
    keep = torch.arange(E)
    queue = list(keep.flip(0).tensor_split(min(HALVE_BLOCKS, E))) if E > 1 else []
    gone = torch.zeros(E, dtype=torch.bool)
    while queue:  # halving, B blocks a round (lowest-ranked first): each tried alone, the passing ones together
        blocks, queue = queue[:B], queue[B:]
        rows = []
        for b in blocks:
            trial = gone.clone()
            trial[b] = True
            rows.append(keep[~trial])
        cs = ed.contrasts(kept_rows(rows), edges, rest=True)
        measured += len(blocks)
        passed = [b for b, c in zip(blocks, cs) if ok(c)]
        if passed:
            trial = gone.clone()
            trial[torch.cat(passed)] = True
            measured += 1
            if ok(ed.contrasts(kept_rows([keep[~trial]]), edges, rest=True)[0]):
                gone = trial
            else:  # together they fail: one at a time
                for b in passed:
                    t2 = gone.clone()
                    t2[b] = True
                    measured += 1
                    if ok(ed.contrasts(kept_rows([keep[~t2]]), edges, rest=True)[0]):
                        gone = t2
        queue = [h for b, c in zip(blocks, cs) if not ok(c) and len(b) > 1 for h in b.tensor_split(2)] + queue
    keep = keep[~gone]
    t_halve = time.time() - t0 - t_rank - t_keep
    rows = kept_rows([keep] + [keep[keep != i] for i in keep.tolist()])  # the graph, then each kept edge cut from it
    cs = ed.contrasts(rows, edges, rest=True)
    c_kept, strengths = cs[0], [cs[0] - c for c in cs[1:]]
    cut = torch.zeros(1, E, dtype=torch.bool)  # the graph's edges cut, every other edge on
    cut[0, keep] = True
    c_cut = ed.contrasts(cut, edges)[0]
    measured += len(keep) + 2
    rec = {"contrast_model": ed.c_full, "contrast_all_edges": c_all, "contrast_no_edges": c_none, "contrast_kept": c_kept,
           "contrast_graph_cut": c_cut, "tolerance": tol, "met": hit is not None,
           "candidates": {"resid": len(ed.resid_all), "route": ed.n_routes, "route_listed": len(routes), "listed": n_ok},
           "kept": len(keep), "kept_before_halving": E, "measured": measured,
           "seconds": {"rank": round(t_rank, 2), "keep": round(t_keep, 2), "halving": round(t_halve, 2),
                       "total": round(time.time() - t0, 2)}}
    log(f"{len(keep)} of {n_ok} listed edges ({E} before halving; {len(ed.resid_all)} resid, {len(routes)} of {ed.n_routes} "
        f"routes listed); contrast {c_kept:.3f} bits (every edge {c_all:.3f}, none {c_none:.3f}, the graph's cut "
        f"{c_cut:.3f}){'' if hit is not None else ' NOT MET'}; {rec['seconds']}")
    return {"edges": [edges[i] for i in keep.tolist()], "strengths": strengths, "record": rec}


@torch.no_grad()
def graph(ed: Edges, got: dict, tok=None) -> dict:
    """The stated graph as JSON: neuron poles (the activation's sign, the SwiGLU regime: the signs of gate and up) and
    the kept edges with their write weights, their linear push on the reader's activation in the run (push; excites:
    toward the reader's pole; share: |push| over the reader's activation) and measured strength (bits of contrast their
    cut from the graph removes: every other kept edge on, every other candidate cut); with a tokenizer, the tokens and the answer's and alternatives' text."""
    nodes = []
    for L, p, j in ed.N:
        layer = ed.layers[L]
        n = ed._f(layer.post_attention_layernorm, ed._mid_stream(L)[p])
        g, u = float(ed._f(layer.mlp.gate_proj, n)[j]), float(ed._f(layer.mlp.up_proj, n)[j])
        a = float(ed.base["h"][L][p, j])
        nodes.append({"id": len(nodes), "layer": L, "neuron": j, "pos": p, "act": a, "pole": 1 if a >= 0 else -1,
                      "regime": f"g{'+' if g >= 0 else '-'}u{'+' if u >= 0 else '-'}"})
    out, J = [], ed.jacobians()
    for e, st in zip(got["edges"], got["strengths"]):
        w, r = e[0], e[-1]
        col = ed._w(ed.layers[w[0]].mlp.down_proj, cols=w[2])
        if len(e) == 3:  # through the head's OV
            c, h = e[1]
            at, kv, dc = ed.layers[c].self_attn, h // (ed.H // ed.KV), ed._d(c)
            vw = ed._w(at.v_proj).view(ed.KV, ed.hd, -1)[kv] @ (ed.layers[c].input_layernorm.weight.float() * col.to(dc))
            col = ed._w(at.o_proj, cols=slice(h * ed.hd, (h + 1) * ed.hd)) @ vw
        layer = ed.layers[r[0]]
        wn = layer.post_attention_layernorm.weight.float() * col.to(ed._d(r[0]))
        eff = float(J[r] @ ed.brings(e))  # the edge's linear push on the reader's activation in the run
        pole = nodes[ed.at[r]]["pole"]
        out.append({"src": ed.at[w], "dst": ed.at[r], "kind": "resid" if len(e) == 2 else "route",
                    "head": list(e[1]) if len(e) == 3 else None, "src_pos": w[1], "dst_pos": r[1],
                    "w_gate": float(ed._w(layer.mlp.gate_proj, rows=r[2]) @ wn), "w_up": float(ed._w(layer.mlp.up_proj, rows=r[2]) @ wn),
                    "push": eff, "excites": (eff >= 0) == (pole > 0), "share": abs(eff) / max(abs(nodes[ed.at[r]]["act"]), 1e-9),
                    "strength": st})
    txt = {} if tok is None else {"tokens": [tok.decode([i]) for i in ed.ids], "answer_text": tok.decode(ed.answer),
                                  "alternatives_text": [tok.decode([a]) for a in ed.alternatives]}
    return {"ids": ed.ids, **txt, "contrast": {"answer": ed.answer, "alternatives": ed.alternatives, "target": ed.target,
                                               "value": ed.c_full}, "background": ed.background, "nodes": nodes,
            "edges": out, "record": got["record"],
            "route_cut_semantics": {"path": "value_ov", "attention_pattern": "current_run_fixed_in_local_value_delta",
                                    "head_input_norm": "recomputed_after_writer_write_subtraction",
                                    "reader_mlp": "recomputed_norm_gate_up", "combined_routes": "sum_of_individual_value_deltas",
                                    "query_key_effects_included": False, "full_writer_ablation_equivalent": False,
                                    "full_causal_completeness_certified": False}}


def load(name: str, device: str | None = None, dtype: str = "float32"):
    """An HF causal LM for the stage: float32 on one card, or bfloat16 split over every card (device_map auto), its
    weights without gradients."""
    import transformers
    if dtype == "bfloat16":
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
    else:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev)
    return model.eval().requires_grad_(False)  # (the stage's gradients are at activations: no weight gradient, 8B's
    # float32 one did not fit an L40)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("one", "record"))
    ap.add_argument("model")
    ap.add_argument("text", type=Path, help="one: TEXT.json; record: a neuron_builder record (OUT/<id>.json)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--background", default="circuit", choices=("circuit", "model"))
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="the model's weights (bfloat16: 32B "
                    "over two cards, device_map auto)")
    ap.add_argument("--thr", type=Path, help="the reference's file (neuron_builder.Neurons.save)")
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    import transformers
    model = load(a.model, a.device, a.dtype)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval().requires_grad_(False)
    if a.cmd == "record":  # a neuron_builder circuit: its prompt, answer, alternatives and units, its reference
        import neuron_builder

        rec = json.loads(a.text.read_text())
        thr = massive = None
        if a.thr:
            nr = neuron_builder.Neurons(model, tok)
            nr.load(a.thr)
            thr, massive = nr.thr, nr.massive
        ed = Edges(model, neuron_builder.chat(tok, rec["user"]), rec["pos"], [rec["answer"]], rec["alternatives"],
                   [(n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"]], thr=thr, background=a.background,
                   massive=massive)
        got = prune(ed, log=lambda m: print(f"{a.text.stem}: {m}", flush=True))
        g = graph(ed, got, tok)
        g.update(id=rec.get("id"), user=rec.get("user"), model=a.model, adapter=a.adapter, thr=str(a.thr) if a.thr else None)
        a.out.write_text(json.dumps(g))
        return
    t = json.loads(a.text.read_text())
    ed = Edges(model, t["ids"], t["target"], t["answer"], t["alternatives"], [tuple(n) for n in t["neurons"]],
               background=a.background)
    got = prune(ed, log=lambda m: print(f"{a.text.stem}: {m}", flush=True))
    a.out.write_text(json.dumps(graph(ed, got, tok)))


if __name__ == "__main__":
    main()
