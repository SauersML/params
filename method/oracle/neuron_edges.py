"""Neuron-to-neuron edges on an HF Qwen3 (the 10-15 direction: the units are the model's own MLP neurons, each with a
positive and a negative pole and its SwiGLU sign regime; attention heads are routes, read through the neurons on each
side). Given a text, a question's contrast at a target and a circuit's neurons, it states which neuron feeds which.

Contrast: log2 of the summed probability of the answer's tokens over that of the alternatives' at the target.
Background: the reference run, the model with each neuron's activation zeroed below its weak threshold (thr [layers,
d_mlp]: neuron_builder.Neurons', inframove's T2 calibration; None: the model itself). With background "circuit" every
strong candidate outside the circuit is also zeroed, as neuron_builder's circuit run: strong, not massive (|value| under
massive [layers], the layer's cut) and at positions 1 to the target, the first position and massive values live (the
question-scoped circuit's sufficiency run); with "model" every strong neuron runs. Every cut run zeroes the values the
reference run zeroed and no others (weak "reference"; "run": each run its own weak values, as neuron_builder).
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
Pruning (every run with the rest cut, so the stated graph is the circuit's every edge), local families first and the
contrast checked on the whole graph:
  1. each reader's IN_EDGES candidate in-edges of largest local effect (its input Jacobian times what the edge brings),
     then on the reader itself its activation with every circuit in-edge cut but its first k candidates, k = 0 to all
     (writers at base values; the norm, gate and up recomputed): each reader's smallest prefix within an allowed relative
     error of its activation, per allowance: the same error for every reader (EPS), or one in proportion to the
     tolerance over the reader's first-order weight in the contrast (BITS: its gradient times its activation);
  2. of the allowances whose union keeps the contrast within TOL of the way from every edge's to none's (either side),
     the one with the fewest edges, every allowance one forward;
  3. readers' in-edge groups (a group cut: that reader reads no circuit neuron): the fewest readers by first-order
     weight whose groups alone keep the contrast (on a log scale, one forward), then a halving pass over theirs, a
     forward's blocks a round (each tried alone, those passing tried together);
  4. measured last: the graph's contrast, each reader's group cut from it (the reader's in-strength), the graph's edges
     cut with every other edge on, each kept edge's local effect on its reader, and the strength (its own cut from the
     graph) of the MEASURED edges of largest share of their reader's in-strength.

  python neuron_edges.py one MODEL TEXT.json OUT.json   (TEXT: ids, target, answer, alternatives, neurons)
  python neuron_edges.py record MODEL REC.json|REC_DIR OUT.json|OUT_DIR --thr THR.pt [--adapter A] [--dtype bfloat16]
                         (a neuron_builder circuit in its reference; a directory: each record in it, those done skipped)
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
IN_EDGES = 256  # each reader's candidate in-edges measured on it (by local effect)
EPS = tuple(round(0.5 * 2 ** (-i / 2), 4) for i in range(14))  # each reader's relative errors tried: 0.5 to 0.006
BITS = tuple(round(2 ** (-i / 2), 4) for i in range(20))  # or each reader's error allowed this share of the tolerance in
# first-order contrast (gradient x activation x relative error): 1 to 0.0014
MEASURED = 512  # kept edges whose own cut from the graph is measured (those of largest share of their reader's strength)
HALVE_BLOCKS = 8
TOKENS = 1536  # cut sets a forward: this many tokens' worth (at least 16 sets)
CHUNK = 1 << 28  # floats a step of listed resid edges takes (sets x edges x d)


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
    """0 to n on a log scale (two steps an octave), n included."""
    return sorted({0, n} | {min(n, int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(n, 1))) + 2)})


class Edges:
    """One circuit's candidate edges on one text (module docstring). Rows of cuts are bool [K, E] over an edge list
    (resid (writer, reader) and route (writer, (layer, head), reader) tuples; neurons (layer, position, index))."""

    def __init__(self, model, ids: list[int], target: int, answer: list[int], alternatives: list[int], neurons: list,
                 thr: torch.Tensor | None = None, background: str = "circuit", massive: torch.Tensor | None = None,
                 weak: str = "reference"):
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = core.config
        self.core, self.layers = core, core.model.layers
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.eps = cfg.rms_norm_eps
        self.dev, self.dt = core.model.embed_tokens.weight.device, core.model.embed_tokens.weight.dtype
        self.T = target + 1
        self.B = max(16, TOKENS // self.T)  # cut sets a forward
        self.forwards = 0  # (contrasts' forwards so far)
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
        self._plans, self._cols, self._vcols, self._J, self._G, self._dv_base, self._allin = {}, {}, {}, None, None, None, None
        self.fixed = None  # weak "reference": the reference run's zeroed values held zero in every run, none else
        # (the reference's set of values, so a cut moves the contrast smoothly; "run": every run zeroes its own weak values,
        # neuron_builder's, where a near-threshold value flips on float noise: 0.1 bits between CPU and GPU on f0)
        with torch.no_grad():
            self.base = self._run(torch.zeros(1, 0, dtype=torch.bool), [], keep_states=True)
        if weak == "reference":
            self.fixed = [h != 0 for h in self.base["h"]]
        self.c_full = float(self.base["contrast"][0]) if self.dt == torch.float32 else \
            self.contrasts(torch.zeros(1, 0, dtype=torch.bool), [])[0]  # (padded as every other forward)

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
            hm = self._mlp_act(layer, x)
            hm = self._weak(L, hm) if self.fixed is None else hm * self.fixed[L].to(d)[None]
            if self.background == "circuit" and self.fixed is None:  # (strong candidates outside the circuit zeroed; position 0 and massive live)
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
                hr = self._weak(L, hr, Jn[rs]) if self.fixed is None else hr * self.fixed[L].to(d)[P[rs], Jn[rs]][None]
                hm = hm.index_put((kk[:, None], P[rs][None], Jn[rs][None]), hr)
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
                n = max(256, CHUNK // (K * cols.shape[1]))
                for s0 in range(0, len(es), n):
                    e_, w_, r_ = es[s0:s0 + n], wi[s0:s0 + n], ri[s0:s0 + n]
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
        for s0 in range(0, len(cuts), self.B):
            c = cuts[s0:s0 + self.B]
            if self.dt != torch.float32 and len(c) < self.B:  # (a lower precision model's kernels round by batch size: every
                c = torch.cat([c, c[-1:].expand(self.B - len(c), -1)])  # forward padded to B sets)
            out += self._run(c, edges, rest=rest)["contrast"].tolist()[:len(cuts) - s0]
            self.forwards += 1
        return out

    # -- the base run's quantities --------------------------------------------------------------------------------

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

    # -- each reader's in-edges, measured on the reader -------------------------------------------------------------

    def _h0(self) -> torch.Tensor:
        """The circuit neurons' activations in the base run [N] (on the first card)."""
        if not self.N:
            return torch.zeros(0, device=self.dev)
        return torch.stack([self.base["h"][L][p, j].to(self.dev) for L, p, j in self.N])

    def _dvs(self) -> dict:
        """Per head layer c, the base run's value change at each circuit writer below c when its write leaves layer c's
        input there [nb[c], KV, hd] (_dv; kept, on layer c's card)."""
        if self._dv_base is None:
            h0, base = self._h0(), self.base
            self._dv_base = {c: self._dv(c, base["x"][c][None], base["v"][c][None], h0[:self.nb[c]].to(self._d(c))[None])[0]
                             for c in range(1, self.L) if self.nb[c]}
        return self._dv_base

    def _all_in(self) -> list:
        """Per layer L, what every circuit in-edge brings into layer L's MLP input at each position in the base run [T, d]
        (the circuit's writes below L at the position, plus every head's output change there through layers up to L from
        every circuit writer below that layer; kept, on layer L's card)."""
        if self._allin is None:
            T, H, KV, hd, h0, dvs = self.T, self.H, self.KV, self.hd, self._h0(), self._dvs()
            dm = self.base["x"][0].shape[-1]
            S = R = None
            out = []
            for L, layer in enumerate(self.layers):
                d = self._d(L)
                S = torch.zeros(T, dm, device=d) if S is None else S.to(d)
                R = torch.zeros(T, dm, device=d) if R is None else R.to(d)
                if L in dvs:
                    dV = torch.zeros(T, KV, hd, device=d).index_add(0, self.P[:self.nb[L]].to(d), dvs[L])
                    z = self.base["A"][L] @ dV.transpose(0, 1).repeat_interleave(H // KV, 0)  # [H, T, hd]
                    R = R + z.transpose(0, 1).reshape(T, H * hd) @ self._w(layer.self_attn.o_proj).T
                out.append(S + R)
                rs = torch.arange(self.nb[L], self.nb[L + 1], device=d)
                if len(rs):
                    S = S.index_add(0, self.P.to(d)[rs], h0.to(d)[rs, None] * self._cols_on(d)[rs])
            self._allin = out
        return self._allin

    @torch.no_grad()
    def in_edges(self, M: int) -> list[tuple[list, torch.Tensor]]:
        """Per circuit neuron as a reader (N's order): its candidate in-edges of the M largest |local effect| and those
        effects, largest first. The local effect is the reader's input Jacobian times what the edge brings in the base
        run, a ranking only (local_curves measures the reader). Routes are scored a head layer at a time (every writer
        below it, reader above it and head at once), each reader's best M carried."""
        N, H, KV, hd, base, dev = len(self.N), self.H, self.KV, self.hd, self.base, self.dev
        if not N:
            return []
        J, h0, dvs = self.jacobians(), self._h0(), self._dvs()
        Jm = torch.stack([J[n].to(dev) for n in self.N])
        best_m = torch.full((N, M), -1.0)  # |effect| (-1: no candidate)
        best_s = torch.zeros(N, M)
        best_i = torch.zeros(N, M, 3, dtype=torch.long)  # (writer, head layer (-1: resid), head)

        def carry(rows, mag, s, decode):  # each reader of rows: its best M of what it had and of mag [R, X]
            t = torch.topk(mag, min(M, mag.shape[1]), dim=1)
            m2 = torch.cat([best_m[rows], t.values.cpu()], 1)
            s2 = torch.cat([best_s[rows], s.gather(1, t.indices).cpu()], 1)
            i2 = torch.cat([best_i[rows], decode(t.indices).cpu()], 1)
            u = torch.topk(m2, M, dim=1).indices
            best_m[rows], best_s[rows] = m2.gather(1, u), s2.gather(1, u)
            best_i[rows] = i2.gather(1, u[..., None].expand(-1, -1, 3))

        P, lay = self.P.to(dev), torch.tensor([n[0] for n in self.N], device=dev)
        s = (Jm @ self._cols_on(dev).T) * h0[None]  # resid [reader, writer]
        ok = (P[:, None] == P[None]) & (lay[None] < lay[:, None])
        carry(torch.arange(N), s.abs().masked_fill(~ok, -1.0), s,
              lambda t: torch.stack([t, torch.full_like(t, -1), torch.zeros_like(t)], -1))
        for c, dv in dvs.items():
            nb = self.nb[c]
            if nb == N:
                continue
            layer, dc = self.layers[c], self._d(c)
            Pd = self.P.to(dc)
            dv = dv[:, torch.arange(H, device=dc) // (H // KV)]  # [W, H, hd]
            U = torch.einsum("rd,dhk->rhk", Jm[nb:].to(dc), self._w(layer.self_attn.o_proj).view(-1, H, hd))  # [R, H, hd]
            a = base["A"][c][:, Pd[nb:]][:, :, Pd[:nb]].permute(1, 2, 0)  # [R, W, H]
            s = (a * torch.einsum("rhk,whk->rwh", U, dv)).flatten(1)  # [R, W H]
            ok = (Pd[:nb][None] <= Pd[nb:][:, None])[..., None].expand(-1, -1, H).flatten(1)
            carry(torch.arange(nb, N), s.abs().masked_fill(~ok, -1.0), s,
                  lambda t, c=c: torch.stack([t // H, torch.full_like(t, c), t % H], -1))
        out = []
        for i, r in enumerate(self.N):
            v = best_m[i] >= 0
            out.append(([(self.N[w], r) if c < 0 else (self.N[w], (c, h), r) for w, c, h in best_i[i][v].tolist()],
                        best_s[i][v]))
        return out

    def _chunks(self, size: int):
        """(layer, readers) in N's order, at most size readers of one layer each."""
        for L in range(self.L):
            for a in range(self.nb[L], self.nb[L + 1], size):
                yield L, list(range(a, min(a + size, self.nb[L + 1])))

    @torch.no_grad()
    def _vectors(self, rows: list[int], cands: list, K: int) -> torch.Tensor:
        """What the first K candidate in-edges of each reader of rows (one layer's) bring into its MLP input in the base
        run [len(rows), K, d] (zero past a reader's candidates; on the layer's card)."""
        L = self.N[rows[0]][0]
        d, H, KV, hd = self._d(L), self.H, self.KV, self.hd
        cols, h0, dvs = self._cols_on(d), self._h0().to(d), self._dvs()
        V = torch.zeros(len(rows), K, cols.shape[1], device=d)
        groups = {}
        for a, i in enumerate(rows):
            for s, e in enumerate(cands[i][0][:K]):
                groups.setdefault(None if len(e) == 2 else e[1], []).append((a, s, self.at[e[0]], e[-1][1]))
        for g, xs in groups.items():
            a, s, w, q = (torch.tensor(t, device=d) for t in zip(*xs))
            if g is None:
                V[a, s] = h0[w, None] * cols[w]
                continue
            c, h = g
            dc = self._d(c)
            w, q = w.to(dc), q.to(dc)
            z = self.base["A"][c][h, q, self.P.to(dc)[w]][:, None] * dvs[c][w, h // (H // KV)]  # [n, hd]
            V[a, s] = (z @ self._w(self.layers[c].self_attn.o_proj, cols=slice(h * hd, (h + 1) * hd)).T).to(d)
        return V

    @torch.no_grad()
    def _acts(self, L: int, rows: list[int], X: torch.Tensor) -> torch.Tensor:
        """The readers rows' (layer L's) activations from MLP inputs X [len(rows), K, d] [len(rows), K]: the norm, gate
        and up recomputed in float32."""
        layer = self.layers[L]
        js = self.Jn[rows].to(self._d(L))
        g = layer.post_attention_layernorm.weight.float().to(X.device)
        Xn = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.variance_epsilon)
        gate = torch.einsum("rkd,rd->rk", Xn, self._w(layer.mlp.gate_proj, rows=js) * g)
        return layer.mlp.act_fn(gate) * torch.einsum("rkd,rd->rk", Xn, self._w(layer.mlp.up_proj, rows=js) * g)

    def _cut_input(self, L: int, rows: list[int]) -> torch.Tensor:
        """The readers rows' (layer L's) MLP inputs in the base run less what every circuit in-edge brings [len(rows), d]."""
        q = self.P[rows].to(self._d(L))
        return self._mid_stream(L)[q] - self._all_in()[L][q]

    @torch.no_grad()
    def local_curves(self, cands: list, chunk: int = 64) -> list[torch.Tensor]:
        """Per reader, its activation with every circuit in-edge cut but its first k candidates (cands: in_edges'), the
        writers at their base values, k = 0 .. its candidates [len + 1] (measured on the reader: _acts)."""
        out = [torch.zeros(1)] * len(self.N)
        for L, rows in self._chunks(chunk):
            K = max(len(cands[i][0]) for i in rows)
            V = self._vectors(rows, cands, K)
            X = self._cut_input(L, rows)[:, None] + torch.cat([V.new_zeros(len(rows), 1, V.shape[-1]), V.cumsum(1)], 1)
            a = self._acts(L, rows, X).cpu()
            for t, i in enumerate(rows):
                out[i] = a[t, :len(cands[i][0]) + 1]
        return out

    @torch.no_grad()
    def local_effects(self, cands: list, ks: list[int], chunk: int = 64) -> list[torch.Tensor]:
        """Per reader, each of its first ks[i] candidates' measured local effect: the reader's activation with that
        prefix kept less with that edge also cut (every other circuit in-edge cut, writers at base values)."""
        out = [torch.zeros(0)] * len(self.N)
        for L, rows in self._chunks(chunk):
            K = max(ks[i] for i in rows)
            if not K:
                continue
            V = self._vectors(rows, cands, K) * (torch.arange(K)[None] < torch.tensor([ks[i] for i in rows])[:, None]).to(
                self._d(L))[..., None]
            Xk = self._cut_input(L, rows) + V.sum(1)
            a = self._acts(L, rows, torch.cat([Xk[:, None], Xk[:, None] - V], 1)).cpu()
            for t, i in enumerate(rows):
                out[i] = a[t, :1] - a[t, 1:ks[i] + 1]
        return out

def prune(ed: Edges, log=print) -> dict:
    """The stated graph (module docstring's pruning) and its record."""
    t0 = time.time()
    N = len(ed.N)
    c_all = ed.c_full
    c_none = ed.contrasts(torch.zeros(1, 0, dtype=torch.bool), [], rest=True)[0]
    tol = max(TOL * abs(c_all - c_none), MIN_TOL)  # (two-sided: a kept set that overshoots is no more faithful)
    ok = lambda c: abs(c - c_all) <= tol  # noqa: E731
    cands = ed.in_edges(IN_EDGES)
    curves = ed.local_curves(cands)
    ref = ed._h0().cpu()
    err = [(curves[i] - ref[i]).abs() / max(abs(float(ref[i])), 1e-9) for i in range(N)]
    sens = (ed.grads() * ref).abs()  # (each reader's contrast per unit relative change, first order: allowances only)
    allow = [("error", eps, [eps] * N) for eps in EPS] + [
        ("bits", b, [b * tol / max(float(sens[i]), 1e-12) for i in range(N)]) for b in BITS]
    K = [[int(e.le(a).nonzero()[0]) if bool(e.le(a).any()) else int(e.argmin()) for e, a in zip(err, al)]
         for _, _, al in allow]
    t_local = time.time() - t0

    kmax = [max(k[i] for k in K) for i in range(N)]
    listed = [cands[i][0][s] for i in range(N) for s in range(kmax[i])]
    owner = torch.tensor([i for i in range(N) for _ in range(kmax[i])], dtype=torch.long)
    rank = torch.tensor([s for i in range(N) for s in range(kmax[i])], dtype=torch.long)
    rows = torch.stack([rank >= torch.tensor(k, dtype=torch.long)[owner] for k in K])  # each reader's prefix kept, per allowance
    cs = ed.contrasts(rows, listed, rest=True)
    measured = 1 + len(allow)
    passing = [a for a, c in enumerate(cs) if ok(c)]
    pick = min(passing, key=lambda a: int((~rows[a]).sum())) if passing else len(EPS) - 1  # (the fewest edges)
    gone = rows[pick].clone()  # (cut)
    before = int((~gone).sum())
    log(f"reader allowance {allow[pick][0]} {allow[pick][1]:.4g} {'passes' if passing else 'FAILS'} with the fewest: "
        f"{before} edges kept of {len(listed)} listed (contrast {cs[pick]:.3f}, every edge {c_all:.3f}, none {c_none:.3f}, "
        f"tolerance {tol:.3f}; {len(passing)} of {len(allow)} allowances pass), {time.time() - t0:.1f} s")
    t_eps = time.time() - t0 - t_local

    groups = [(~gone & (owner == i)).nonzero().flatten() for i in range(N)]
    cut_of = lambda rs: torch.cat([groups[int(i)] for i in rs]) if len(rs) else torch.zeros(0, dtype=torch.long)  # noqa: E731
    readers = sorted((i for i in range(N) if len(groups[i])), key=lambda i: -float(sens[i]))  # (by first-order weight)
    ks = sizes(len(readers))  # the fewest of them whose in-edges alone keep the contrast (k on a log scale), one forward
    trial = []
    for k in ks:
        t = gone.clone()
        t[cut_of(readers[k:])] = True
        trial.append(t)
    cs = ed.contrasts(torch.stack(trial), listed, rest=True) if trial else []
    measured += len(trial)
    k = next((k for k, c in zip(ks, cs) if ok(c)), len(readers))
    gone = trial[ks.index(k)] if k in ks else gone
    readers = readers[:k][::-1]  # (weakest first)
    queue = [list(b) for b in torch.tensor(readers, dtype=torch.long).tensor_split(min(HALVE_BLOCKS, len(readers)))
             if len(b)] if len(readers) > 1 else []
    while queue:  # halving over readers' in-edge groups, a forward's blocks a round: each block's groups cut alone, the
        blocks, queue = queue[:ed.B], queue[ed.B:]  # passing ones together
        trial = []
        for b in blocks:
            t = gone.clone()
            t[cut_of(b)] = True
            trial.append(t)
        cs = ed.contrasts(torch.stack(trial), listed, rest=True)
        measured += len(blocks)
        passed = [b for b, c in zip(blocks, cs) if ok(c)]
        if passed:
            t = gone.clone()
            t[cut_of([i for b in passed for i in b])] = True
            measured += 1
            if ok(ed.contrasts(t[None], listed, rest=True)[0]):
                gone = t
            else:  # together they fail: one at a time
                for b in passed:
                    t = gone.clone()
                    t[cut_of(b)] = True
                    measured += 1
                    if ok(ed.contrasts(t[None], listed, rest=True)[0]):
                        gone = t
        queue = [list(h) for b, c in zip(blocks, cs) if not ok(c) and len(b) > 1
                 for h in torch.tensor(b).tensor_split(2)] + queue
    keep = (~gone).nonzero().flatten()
    t_halve = time.time() - t0 - t_local - t_eps

    readers = [i for i in range(N) if bool((~gone & (owner == i)).any())]
    trial = [gone]  # the graph, then each reader's kept in-edges cut from it
    for i in readers:
        t = gone.clone()
        t[owner == i] = True
        trial.append(t)
    cs = ed.contrasts(torch.stack(trial), listed, rest=True)
    c_kept, group = cs[0], {i: cs[0] - c for i, c in zip(readers, cs[1:])}
    c_cut = ed.contrasts((~gone)[None], listed)[0]  # the graph's edges cut, every other edge on
    ks = [int((~gone & (owner == i)).sum()) for i in range(N)]  # (each reader's kept in-edges: a prefix of its own)
    eff = ed.local_effects(cands, ks)
    local = torch.cat([eff[int(owner[e])][int(rank[e])][None] for e in keep]) if len(keep) else torch.zeros(0)
    tot = {i: float(eff[i].abs().sum()) for i in readers}
    est = torch.tensor([abs(group[int(owner[e])]) * float(local[a].abs()) / max(tot[int(owner[e])], 1e-12)
                        for a, e in enumerate(keep.tolist())])
    top = torch.argsort(est, descending=True)[:MEASURED].tolist()  # (each edge's own cut from the graph measured for the
    trial = []  # MEASURED edges of largest share of their reader's measured group strength)
    for a in top:
        t = gone.clone()
        t[keep[a]] = True
        trial.append(t)
    cs = ed.contrasts(torch.stack(trial), listed, rest=True) if trial else []
    strengths = [None] * len(keep)
    for a, c in zip(top, cs):
        strengths[a] = c_kept - c
    measured += 2 + len(readers) + len(trial)
    best = int(ed.base["lp"][0].argmax())
    rec = {"contrast_model": ed.c_full, "reference_top": best, "answer_top": best in ed.answer, "contrast_all_edges": c_all, "contrast_no_edges": c_none, "contrast_kept": c_kept,
           "contrast_graph_cut": c_cut, "tolerance": tol, "met": bool(passing) and ok(c_kept),
           "allowance": list(allow[pick][:2]),
           "candidates": {"resid": len(ed.resid_all), "route": ed.n_routes,
                          "in_edges_listed": sum(len(c[0]) for c in cands)},
           "kept": len(keep), "kept_before_halving": before, "readers": len(readers), "measured": measured,
           "forwards": ed.forwards, "sets_a_forward": ed.B,
           "seconds": {"local": round(t_local, 2), "errors": round(t_eps, 2), "halving": round(t_halve, 2),
                       "total": round(time.time() - t0, 2)}}
    log(f"{len(keep)} edges into {len(readers)} of {N} neurons ({before} before halving); contrast {c_kept:.3f} bits "
        f"(every edge {c_all:.3f}, none {c_none:.3f}, the graph's cut {c_cut:.3f}){'' if rec['met'] else ' NOT MET'}; "
        f"{rec['seconds']}")
    return {"edges": [listed[e] for e in keep.tolist()], "strengths": strengths, "local": local.tolist(),
            "group": {ed.N[i]: g for i, g in group.items()}, "record": rec}


@torch.no_grad()
def graph(ed: Edges, got: dict, tok=None) -> dict:
    """The stated graph as JSON: neuron poles (the activation's sign, the SwiGLU regime: the signs of gate and up) with
    each reader's measured in-strength (bits of contrast that cutting its kept in-edges from the graph removes), and the
    kept edges with their write weights, their measured local effect (local: the reader's activation less with the edge
    also cut from the graph's in-edges of it, writers at base values; excites: toward the reader's pole; share: |local|
    over the reader's activation) and, for the MEASURED of largest share of their reader's in-strength, the measured
    strength (bits of contrast the edge's cut from the graph removes: every other kept edge on, every other candidate
    cut; None for the rest); with a tokenizer, the tokens and the answer's and alternatives' text."""
    Wf = {}

    def wt(L: int, name: str) -> torch.Tensor:  # a layer's float32 weight, fetched once
        if (L, name) not in Wf:
            layer = ed.layers[L]
            mod = {"down": layer.mlp.down_proj, "gate": layer.mlp.gate_proj, "up": layer.mlp.up_proj,
                   "v": layer.self_attn.v_proj, "o": layer.self_attn.o_proj}.get(name)
            Wf[(L, name)] = (ed._w(mod) if mod is not None else
                             (layer.input_layernorm if name == "in" else layer.post_attention_layernorm).weight.float())
        return Wf[(L, name)]

    nodes = []
    for L, p, j in ed.N:
        layer = ed.layers[L]
        n = ed._f(layer.post_attention_layernorm, ed._mid_stream(L)[p])
        g, u = float(ed._f(layer.mlp.gate_proj, n)[j]), float(ed._f(layer.mlp.up_proj, n)[j])
        a = float(ed.base["h"][L][p, j])
        nodes.append({"id": len(nodes), "layer": L, "neuron": j, "pos": p, "act": a, "pole": 1 if a >= 0 else -1,
                      "regime": f"g{'+' if g >= 0 else '-'}u{'+' if u >= 0 else '-'}",
                      "in_strength": got.get("group", {}).get((L, p, j))})
    out = []
    for e, st, loc in zip(got["edges"], got["strengths"], got["local"]):
        w, r = e[0], e[-1]
        col = wt(w[0], "down")[:, w[2]]
        if len(e) == 3:  # through the head's OV
            c, h = e[1]
            dc = ed._d(c)
            vw = wt(c, "v").view(ed.KV, ed.hd, -1)[h // (ed.H // ed.KV)] @ (wt(c, "in") * col.to(dc))
            col = wt(c, "o")[:, h * ed.hd:(h + 1) * ed.hd] @ vw
        wn = wt(r[0], "post") * col.to(ed._d(r[0]))
        node = nodes[ed.at[r]]
        out.append({"src": ed.at[w], "dst": ed.at[r], "kind": "resid" if len(e) == 2 else "route",
                    "head": list(e[1]) if len(e) == 3 else None, "src_pos": w[1], "dst_pos": r[1],
                    "w_gate": float(wt(r[0], "gate")[r[2]] @ wn), "w_up": float(wt(r[0], "up")[r[2]] @ wn),
                    "local": loc, "excites": (loc >= 0) == (node["pole"] > 0),
                    "share": abs(loc) / max(abs(node["act"]), 1e-9), "strength": st})
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
    ap.add_argument("text", type=Path, help="one: TEXT.json; record: a neuron_builder record (OUT/<id>.json) or a "
                    "directory of them")
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--background", default="circuit", choices=("circuit", "model"))
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="the model's weights (bfloat16: 32B "
                    "over two cards, device_map auto)")
    ap.add_argument("--thr", type=Path, help="the reference's file (neuron_builder.Neurons.save)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--weak", default="reference", choices=("reference", "run"))
    a = ap.parse_args()
    import transformers
    model = load(a.model, a.device, a.dtype)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval().requires_grad_(False)
    if a.cmd == "record":  # neuron_builder circuits: each one's prompt, answer, alternatives and units, its reference
        import neuron_builder

        thr = massive = None
        if a.thr:
            nr = neuron_builder.Neurons(model, tok)
            nr.load(a.thr)
            thr, massive = nr.thr, nr.massive
        if a.text.is_dir():
            a.out.mkdir(parents=True, exist_ok=True)
            todo = [(f, a.out / f.name) for f in sorted(a.text.glob("*.json")) if not (a.out / f.name).exists()]
        else:
            todo = [(a.text, a.out)]
        for src, dst in todo:
            t0 = time.time()
            rec = json.loads(src.read_text())
            ed = Edges(model, neuron_builder.chat(tok, rec["user"]), rec["pos"], [rec["answer"]], rec["alternatives"],
                       [(n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"]], thr=thr, background=a.background,
                       massive=massive, weak=a.weak)
            got = prune(ed, log=lambda m: print(f"{src.stem}: {m}", flush=True))
            g = graph(ed, got, tok)
            g.update(id=rec.get("id"), user=rec.get("user"), model=a.model, adapter=a.adapter,
                     thr=str(a.thr) if a.thr else None, builder=rec.get("builder"), seconds=round(time.time() - t0, 2))
            dst.write_text(json.dumps(g))
            print(f"{src.stem}: {len(ed.N)} neurons, {len(got['edges'])} edges, {time.time() - t0:.1f} s", flush=True)
        return
    t = json.loads(a.text.read_text())
    ed = Edges(model, t["ids"], t["target"], t["answer"], t["alternatives"], [tuple(n) for n in t["neurons"]],
               background=a.background)
    got = prune(ed, log=lambda m: print(f"{a.text.stem}: {m}", flush=True))
    a.out.write_text(json.dumps(graph(ed, got, tok)))


if __name__ == "__main__":
    main()
