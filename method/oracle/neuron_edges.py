"""Neuron-to-neuron edges on an HF Qwen3 (the 10-15 direction: the units are the model's own MLP neurons, each with a
positive and a negative pole and its SwiGLU sign regime; attention heads are routes, read through the neurons on each
side). Given a text, a question's contrast at a target and a circuit's neurons, it states which neuron feeds which.

Contrast: log2 of the summed probability of the answer's tokens over that of the alternatives' at the target.
Background: the reference run, the model with each neuron's activation zeroed below its weak threshold (thr [layers,
d_mlp], from the weak-activation measurement; None: the model itself). With background "circuit" every neuron outside
the circuit is also zeroed (the question-scoped circuit's sufficiency run); with "model" every strong neuron runs.
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
position); the reader's activation is recomputed from that input and every later layer runs on (measured path patching,
one forward per set of cuts, sets batched).
Pruning: the candidates ranked by their linear effect on the contrast (the contrast's gradient at the reader times the
reader's input Jacobian times what the edge brings), then the smallest top-k set whose others cut keeps the contrast
within TOL of the way from every edge's to none's (either side), and a halving pass over the rest, B blocks a round
(each tried alone, those passing tried together); each kept edge's own cut measured last (its strength).

  python neuron_edges.py one MODEL TEXT.json OUT.json   (TEXT: ids, target, answer, alternatives, neurons)
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
ROUTE_KEEP = 8  # route candidates measured: this many times the resid candidates (by linear effect), at least 256
HALVE_BLOCKS = 8
B = 16  # cut sets a forward


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
                 thr: torch.Tensor | None = None, background: str = "circuit"):
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = core.config
        self.core, self.layers = core, core.model.layers
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev, self.dt = core.model.embed_tokens.weight.device, core.model.embed_tokens.weight.dtype
        self.T = target + 1
        self.ids, self.target, self.answer, self.alternatives = list(ids)[:self.T], target, list(answer), list(alternatives)
        self.thr = None if thr is None else thr.to(self.dev, torch.float32)
        self.N = sorted({(int(L), int(p), int(j)) for L, p, j in neurons if p < self.T})
        self.at = {n: i for i, n in enumerate(self.N)}
        self.background = background
        self.keep = torch.zeros(self.L, self.T, cfg.intermediate_size, dtype=torch.bool, device=self.dev)
        for L, p, j in self.N:
            self.keep[L, p, j] = True
        self.resid_all = [(w, r) for w in self.N for r in self.N if w[0] < r[0] and w[1] == r[1]]
        self.route_all = [(w, (c, h), r) for w in self.N for r in self.N if w[1] <= r[1]
                          for c in range(w[0] + 1, r[0] + 1) for h in range(self.H)]
        self._plans = {}
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

    def _plan(self, edges: list) -> dict:
        """Index tables of an edge list: resid cuts by the writer's layer, routes by layer and head."""
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
        plan = {"resid": {L: t(v) for L, v in resid.items()}, "route": {k: t(v) for k, v in route.items()}, "heads": {}}
        for c, h in route:
            plan["heads"].setdefault(c, []).append(h)
        self._plans = {id(edges): (edges, plan)}
        return plan

    def _run(self, cuts: torch.Tensor, edges: list, keep_states: bool = False, probe: list | None = None) -> dict:
        """The contrast of each row of cuts [K, len(edges)]. probe: per layer a tensor [T, d_mlp] added to the neurons'
        activations (zeros requiring grad: the contrast's gradient at them)."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        core, H, KV, hd, T, f = self.core, self.H, self.KV, self.hd, self.T, self._f
        K = max(cuts.shape[0], 1)
        cuts = cuts.float() if len(edges) else torch.zeros(K, 0)
        plan = self._plan(edges)
        x = core.model.embed_tokens(torch.tensor([self.ids], device=self.dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(T, device=self.dev)[None])
        D = torch.zeros(K, len(self.N), x.shape[-1])  # per circuit neuron, what its cut edges took away
        Wr = torch.zeros(K, len(self.N), x.shape[-1])  # each circuit neuron's write in this run
        P = torch.tensor([n[1] for n in self.N], dtype=torch.long)
        Jn = torch.tensor([n[2] for n in self.N], dtype=torch.long)
        states = {"x": [], "A": [], "v": [], "h": []}
        for L, layer in enumerate(self.layers):
            d = self._d(L)  # (every tensor of this layer's work on its card)
            x, D, Wr, cuts, P, Jn, cos, sin = (t.to(d) for t in (x, D, Wr, cuts, P, Jn, cos, sin))
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
            if plan["heads"].get(L):  # routes through this layer's heads: the head's output change at the reader
                for h in plan["heads"][L]:
                    es, wi, ri = (t.to(d) for t in plan["route"][(L, h)])
                    kv = h // (H // KV)
                    xw = x[:, P[wi]] - Wr[:, wi]  # [K, E, d]: the head's input at the writer, its write taken out
                    vw = f(at.v_proj, f(layer.input_layernorm, xw)).view(K, -1, KV, hd)[:, :, kv]
                    dz = A[:, h][kk[:, None], P[ri][None], P[wi][None]][..., None] * (v[:, kv][:, P[wi]] - vw)
                    D = D.index_add(1, ri, cuts[:, es, None] * (dz @ self._w(at.o_proj, cols=slice(h * hd, (h + 1) * hd)).T))
            x = x + f(at.o_proj, z.transpose(1, 2).reshape(K, T, H * hd))
            hm = self._weak(L, self._mlp_act(layer, x))
            if self.background == "circuit":
                hm = hm * self.keep[L].to(d)[None]
            rs = torch.tensor([i for i, n in enumerate(self.N) if n[0] == L], dtype=torch.long, device=d)
            if len(rs) and len(edges):  # the circuit's neurons of this layer, each from its own input
                hr = self._mlp_act(layer, x[:, P[rs]] - D[:, rs]).gather(-1, Jn[rs][None, :, None].expand(K, -1, 1))[..., 0]
                hm = hm.index_put((kk[:, None], P[rs][None], Jn[rs][None]), self._weak(L, hr, Jn[rs]))
            if probe is not None:
                hm = hm + probe[L].to(d)[None]
            if keep_states:
                states["h"].append(hm[0].detach())
            if len(rs):
                Wd = self._w(layer.mlp.down_proj, cols=Jn[rs])  # [d, R]
                Wr = Wr.index_add(1, rs, hm[kk[:, None], P[rs][None], Jn[rs][None]][..., None] * Wd.T[None])
            if L in plan["resid"]:
                es, wi, ri = (t.to(d) for t in plan["resid"][L])
                D = D.index_add(1, ri, cuts[:, es, None] * Wr[:, wi])
            x = x + f(layer.mlp.down_proj, hm)
        lp = torch.log_softmax(f(core.lm_head, f(core.model.norm, x[:, self.target])), -1).cpu()
        out = {"contrast": log2_contrast(lp, self.answer, self.alternatives), "lp": lp}
        if keep_states:
            out.update(states)
        return out

    @torch.no_grad()
    def contrasts(self, cuts: torch.Tensor, edges: list) -> list[float]:
        """The contrast of each row of cuts [K, len(edges)], B rows a forward."""
        out = []
        for s0 in range(0, len(cuts), B):
            out += self._run(cuts[s0:s0 + B], edges)["contrast"].tolist()
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
        if getattr(self, "_J", None) is None:
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

    def linear_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(resid scores, route scores) over resid_all and route_all: each edge's linear effect on the contrast, the
        contrast's gradient at the reader's activation times the reader's input Jacobian times what the edge brings
        (the contrast it supports: positive when its cut lowers the contrast)."""
        probe = [torch.zeros(self.T, self.keep.shape[-1], device=self._d(L), requires_grad=True) for L in range(self.L)]
        with torch.enable_grad():
            self._run(torch.zeros(1, 0, dtype=torch.bool), [], probe=probe)["contrast"].sum().backward()
        grad = {n: float(probe[n[0]].grad[n[1], n[2]]) for n in self.N}
        J = self.jacobians()
        H, KV, hd, base = self.H, self.KV, self.hd, self.base
        with torch.no_grad():
            rs_s = torch.tensor([grad[r] * float(J[r] @ self.brings(e)) for e in self.resid_all for r in [e[1]]])
            G = torch.tensor([grad[n] for n in self.N])
            got = {}
            for c in range(1, self.L):  # per head layer: each reader's input direction through each head's o_proj
                layer, dc = self.layers[c], self._d(c)
                Js = torch.stack([J[n].to(dc) for n in self.N])  # [N, d]
                U = torch.einsum("nd,dhk->nhk", Js, self._w(layer.self_attn.o_proj).view(-1, H, hd))  # [N, H, hd]
                kvh = torch.arange(H, device=dc) // (H // KV)
                for w in self.N:
                    if w[0] >= c:
                        continue
                    rd = [i for i, r in enumerate(self.N) if r[0] >= c and r[1] >= w[1]]
                    if not rd:
                        continue
                    xw = base["x"][c][w[1]] - self._write(w).to(dc)
                    vw = self._f(layer.self_attn.v_proj, self._f(layer.input_layernorm, xw)).view(KV, hd)
                    dv = (base["v"][c][:, w[1]] - vw)[kvh]  # [H, hd]
                    ri = torch.tensor(rd, device=dc)
                    q = torch.tensor([self.N[i][1] for i in rd], device=dc)
                    a = base["A"][c][:, q, w[1]].T  # [R, H]
                    sc = (G.to(dc)[ri, None] * a * torch.einsum("rhk,hk->rh", U[ri], dv)).cpu()
                    for k, i in enumerate(rd):
                        for h in range(H):
                            got[(w, (c, h), self.N[i])] = float(sc[k, h])
            rt = torch.tensor([got[e] for e in self.route_all])
        return rs_s, rt


def prune(ed: Edges, log=print) -> dict:
    """The stated graph (module docstring's pruning) and its record."""
    t0 = time.time()
    rs, rt = ed.linear_scores()
    top = torch.argsort(rt.abs(), descending=True)[:max(256, ROUTE_KEEP * len(ed.resid_all))]
    edges = list(ed.resid_all) + [ed.route_all[i] for i in top.tolist()]
    score = torch.cat([rs, rt[top]]).abs()
    t_rank = time.time() - t0
    E = len(edges)

    def kept_rows(sets):  # each set of kept edges as a cut row (every other edge cut)
        rows = torch.ones(len(sets), E, dtype=torch.bool)
        for i, st in enumerate(sets):
            rows[i, st] = False
        return rows
    c_all, c_none = ed.contrasts(torch.stack([torch.zeros(E, dtype=torch.bool), torch.ones(E, dtype=torch.bool)]), edges)
    tol = max(TOL * abs(c_all - c_none), MIN_TOL)  # (two-sided: a kept set that overshoots is no more faithful)
    ok = lambda c: abs(c - c_all) <= tol  # noqa: E731
    order = torch.argsort(score, descending=True)
    ks = sizes(E)
    cs = ed.contrasts(kept_rows([order[:k] for k in ks]), edges)
    measured = 2 + len(ks)
    keep = order[:next((k for k, c in zip(ks, cs) if ok(c)), E)]
    t_keep = time.time() - t0 - t_rank
    before = len(keep)
    queue = list(keep.flip(0).tensor_split(min(HALVE_BLOCKS, len(keep)))) if len(keep) > 1 else []
    gone = torch.zeros(E, dtype=torch.bool)
    while queue:  # halving, B blocks a round (lowest-ranked first): each tried alone, the passing ones together
        blocks, queue = queue[:B], queue[B:]
        rows = []
        for b in blocks:
            trial = gone.clone()
            trial[b] = True
            rows.append(keep[~trial[keep]])
        cs = ed.contrasts(kept_rows(rows), edges)
        measured += len(blocks)
        passed = [b for b, c in zip(blocks, cs) if ok(c)]
        if passed:
            trial = gone.clone()
            trial[torch.cat(passed)] = True
            measured += 1
            if ok(ed.contrasts(kept_rows([keep[~trial[keep]]]), edges)[0]):
                gone = trial
            else:  # together they fail: one at a time
                for b in passed:
                    t2 = gone.clone()
                    t2[b] = True
                    measured += 1
                    if ok(ed.contrasts(kept_rows([keep[~t2[keep]]]), edges)[0]):
                        gone = t2
        queue = [h for b, c in zip(blocks, cs) if not ok(c) and len(b) > 1 for h in b.tensor_split(2)] + queue
    keep = keep[~gone[keep]]
    t_halve = time.time() - t0 - t_rank - t_keep
    c_kept = ed.contrasts(kept_rows([keep]), edges)[0]
    single = torch.zeros(len(keep), E, dtype=torch.bool)  # each kept edge cut alone (every other edge on)
    single[torch.arange(len(keep)), keep] = True
    strengths = [c_all - c for c in ed.contrasts(single, edges)] if len(keep) else []
    measured += len(keep) + 1
    rec = {"contrast_model": ed.c_full, "contrast_all_edges": c_all, "contrast_no_edges": c_none, "contrast_kept": c_kept,
           "tolerance": tol, "candidates": {"resid": len(ed.resid_all), "route": len(ed.route_all), "route_measured": len(top)},
           "kept": len(keep), "kept_before_halving": before, "measured": measured,
           "seconds": {"rank": round(t_rank, 2), "keep": round(t_keep, 2), "halving": round(t_halve, 2),
                       "total": round(time.time() - t0, 2)}}
    log(f"{len(keep)} of {E} edges ({before} before halving; {len(ed.resid_all)} resid, {len(top)} of {len(ed.route_all)} "
        f"routes measured); contrast {c_kept:.3f} bits (every edge {c_all:.3f}, none {c_none:.3f}, model {ed.c_full:.3f}); "
        f"{rec['seconds']}")
    return {"edges": [edges[i] for i in keep.tolist()], "strengths": strengths, "record": rec}


@torch.no_grad()
def graph(ed: Edges, got: dict, tok=None) -> dict:
    """The stated graph as JSON: neuron poles (the activation's sign, the SwiGLU regime: the signs of gate and up) and
    the kept edges with their write weights, their linear push on the reader's activation in the run (push; excites:
    toward the reader's pole; share: |push| over the reader's activation) and measured strength (bits of contrast their
    cut alone removes); with a tokenizer, the tokens and the answer's and alternatives' text."""
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
    """An HF causal LM for the stage: float32 on one card, or bfloat16 split over every card (device_map auto)."""
    import transformers
    if dtype == "bfloat16":
        return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto").eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("one",))
    ap.add_argument("model")
    ap.add_argument("text", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--background", default="circuit", choices=("circuit", "model"))
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="the model's weights (bfloat16: 32B "
                    "over two cards, device_map auto)")
    a = ap.parse_args()
    import transformers
    model = load(a.model, a.device, a.dtype)
    t = json.loads(a.text.read_text())
    ed = Edges(model, t["ids"], t["target"], t["answer"], t["alternatives"], [tuple(n) for n in t["neurons"]],
               background=a.background)
    got = prune(ed, log=lambda m: print(f"{a.text.stem}: {m}", flush=True))
    a.out.write_text(json.dumps(graph(ed, got, transformers.AutoTokenizer.from_pretrained(a.model))))


if __name__ == "__main__":
    main()
