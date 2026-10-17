"""Neuron-to-neuron edges on an HF Qwen3 (the 10-15 direction: the units are the model's own MLP neurons, each with a
positive and a negative pole and its SwiGLU sign regime; attention heads are routes, read through the neurons on each
side). Given a text, a question's contrast at a target and a circuit's neurons, it states which neuron feeds which.

Contrast: log2 of the summed probability of the answer's tokens over that of the alternatives' at the target (the word
contrast's token sets, answer_contrast.py). Every run is the model itself with every neuron outside the cut edges'
readers computing as usual. A cut edge is zeroed (the reader reads its input less what the edge brings), or with a
corrupted prompt of the same length (circuit_finder's patching pair) patched: the reader reads what the edge brings in the
corrupted run instead.
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
     tolerance over the reader's first-order weight in the contrast (BITS: its gradient times its activation; patching:
     the tolerance's floor, 1% of the question's gap);
  2. of the allowances whose union keeps the contrast within TOL of the way from every edge's to none's (either side),
     the one with the fewest edges, every allowance (and, patching, every edge cut) in one forward;
  3. readers' in-edge groups (a group cut: that reader reads no circuit neuron): the fewest readers by first-order
     weight whose groups alone keep the contrast (on a log scale, one forward), then a halving pass over theirs, a
     forward's blocks a round (each tried alone, those passing tried together);
  4. measured last, in one forward's worth of sets: the graph's contrast, each reader's group cut from it (the reader's
     in-strength), the strength (its own cut from the graph) of the MEASURED edges of largest first-order weight (local
     effect over the reader's activation times the reader's gradient times activation), and the graph's edges cut with
     every other edge on; each kept edge's local effect on its reader.

  python neuron_edges.py one MODEL TEXT.json OUT.json   (TEXT: ids, target, answer, alternatives, neurons)
  python neuron_edges.py finder MODEL REC.json|REC_DIR OUT.json|OUT_DIR [--dtype bfloat16]
                         (circuit_finder records: patching from the corrupted run, route knockouts on cross-position edges)
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
GAP_TOL = 0.01  # patching: and at least this share of the question's gap (clean minus corrupted contrast), so circuit
# edges that together move a sliver of the gap are not stated to a precision of that sliver
IN_EDGES = 256  # each reader's candidate in-edges measured on it (by local effect)
EPS = tuple(round(0.5 * 2 ** (-i / 2), 4) for i in range(14))  # each reader's relative errors tried: 0.5 to 0.006
BITS = tuple(round(2 ** (-i / 2), 4) for i in range(20))  # or each reader's error allowed this share of the tolerance in
# first-order contrast (gradient x activation x relative error): 1 to 0.0014
MEASURED = 512  # kept edges whose own cut from the graph is measured (those of largest first-order weight)
HALVE_BLOCKS = 8
HALVE_FORWARDS = 8  # halving's forwards at most (then its graph stands: every cut in it measured; Qwen3-8B dev circuits of
# ~190 neurons took 20-42 halving forwards, 4.4-8.9 s of a 9-16 s prune; under ~50 neurons 2-11)
TOKENS = 1536  # cut sets a forward: this many tokens' worth (at least 16 sets)
PAD = "shapes"  # a lower precision model's forwards: "B", each padded to B sets (every contrast at one batch shape);
# "shapes", each padded to the next of SHAPES (or B) with an uncut set of its own, every set's contrast read against
# that set's (so an ok() compares contrasts of one shape); "none", unpadded. Qwen3-8B bfloat16 (float32 contrasts), 11
# dev questions of six families on an A6000: shapes kept every graph's edges as B did (25 of 25 on author/Hamlet/p0),
# the edge stage 24.5 -> 15.2 s; none lost one edge of 2 (color/banana/p1), 11.6 s
SHAPES = (8, 32)
D_BYTES = 2 ** 31  # a cut run's per-set, per-circuit-neuron input changes [sets, neurons, d] float32 stay under this: the
# sets a forward shrink with the circuit (a dense readout of 11,468 Qwen3-8B neurons asked for 16.8 GiB on a 48 GB card)
PREFIX = True  # a cut run computes only the positions from the circuit's first one on (every position before it is the
# base run's: no cut reaches it), its attention reading the base run's keys and values there


def weight(mod) -> torch.Tensor:
    """Full effective weight; plain modules (including norm scales) keep their weight.

    Standard eval-mode PEFT LoRA Linear only: merged weights already contain the
    update; disabled unmerged adapters use the base. Disabled+merged would mutate
    on forward, and variants cannot use this additive-weight contract. This full
    matrix API preserves its allocation behavior; it is not a sparse exporter.
    """
    if not hasattr(mod, "base_layer"):
        return mod.weight
    from peft.tuners.lora.layer import Linear
    if type(mod) is not Linear or mod.fan_in_fan_out or type(mod.get_base_layer()) is not torch.nn.Linear:
        raise ValueError("standard nonquantized PEFT LoRA Linear required")
    if any(m.training for m in mod.modules()):
        raise ValueError("effective LoRA weight requires eval mode")
    if getattr(mod, "lora_variant", {}) or any(getattr(mod, "use_dora", {}).values()):
        raise ValueError("variant LoRA effective weights unsupported")
    if mod.disable_adapters and mod.merged:
        raise ValueError("disabled merged adapter would mutate on forward")
    W = mod.get_base_layer().weight
    if mod.disable_adapters or mod.merged:
        return W
    for a in mod.active_adapters:
        if a in mod.lora_A:
            scale = float(mod.scaling[a])
            if not math.isfinite(scale):
                raise ValueError("nonfinite LoRA scaling")
            W = W + (mod.lora_B[a].weight @ mod.lora_A[a].weight) * scale
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
                 corrupt_ids: list[int] | None = None):
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = core.config
        self.core, self.layers = core, core.model.layers
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.eps = cfg.rms_norm_eps
        self.dev, self.dt = core.model.embed_tokens.weight.device, core.model.embed_tokens.weight.dtype
        self.T = target + 1
        self.B = None  # cut sets a forward (below, from the positions a cut run computes)
        self._Wc = None  # (the answer's and the alternatives' output rows in float32: _run)
        self.forwards = 0  # (contrasts' forwards so far)
        self.ids, self.target, self.answer, self.alternatives = list(ids)[:self.T], target, list(answer), list(alternatives)
        self.N = sorted({(int(L), int(p), int(j)) for L, p, j in neurons if p < self.T})  # (by layer first)
        self.B = max(16, TOKENS // (self.T - (min(n[1] for n in self.N) if PREFIX and self.N else 0)))
        self.B = min(self.B, max(2, D_BYTES // max(1, 4 * len(self.N) * cfg.hidden_size)))
        self.at = {n: i for i, n in enumerate(self.N)}
        self.P = torch.tensor([n[1] for n in self.N], dtype=torch.long)
        self.Jn = torch.tensor([n[2] for n in self.N], dtype=torch.long)
        lay = torch.tensor([n[0] for n in self.N], dtype=torch.long)
        self.nb = [int((lay < L).sum()) for L in range(self.L + 1)]  # the circuit's neurons below layer L: the first nb[L]
        self.resid_all = [(w, r) for w in self.N for r in self.N if w[0] < r[0] and w[1] == r[1]]
        self.n_routes = sum((r[0] - w[0]) * self.H for w in self.N for r in self.N if w[1] <= r[1] and w[0] < r[0])
        self._plans, self._cols, self._vcols, self._wos, self._idx = {}, {}, {}, {}, {}
        self._J, self._G, self._dv_base, self._allin = None, None, {}, {}
        self.patch, self.cbase = False, None  # (a patching pair: set below)
        with torch.no_grad():
            self.base = self._run(torch.zeros(1, 0, dtype=torch.bool), [], keep_states=True)
        if corrupt_ids is not None:  # a patching pair: a cut edge's reader reads what the edge brings in the corrupted run
            assert len(corrupt_ids) >= self.T, "patching needs prompts of one length"
            with torch.no_grad():
                self.cbase = self._run(torch.zeros(1, 0, dtype=torch.bool), [], keep_states=True, ids=list(corrupt_ids)[:self.T])
            self.c_corrupt = float(self.cbase["contrast"][0])
            self.patch = True
        if self.dt == torch.float32:
            self.c_full = float(self.base["contrast"][0])
        elif PAD == "shapes":  # (the uncut run at the smallest shape: every contrast read against its forward's uncut set)
            self.c_full = float(self._run(torch.zeros(SHAPES[0], 0, dtype=torch.bool), [])["contrast"][0])
        else:
            self.c_full = self.contrasts(torch.zeros(1, 0, dtype=torch.bool), [])[0]  # (padded as every other forward)
        self.gap = self.c_full - self.c_corrupt if self.patch else None  # (the question's gap: clean minus corrupted)

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
    def _m(mod, t: torch.Tensor) -> torch.Tensor:
        """mod on t in mod's dtype on mod's card, the result in that dtype (an input to further modules: the same values
        as _f's, without a float32 copy cast back for each reader)."""
        w = next(mod.parameters())
        return mod(t.to(w.device, w.dtype))

    @staticmethod
    def _w(mod, rows=None, cols=None) -> torch.Tensor:
        """A slice of a linear module's effective weight in float32 (rows and cols index it)."""
        W = weight(mod)
        W = W if rows is None else W[rows]
        return (W if cols is None else W[:, cols]).float()

    def _mlp_act(self, layer, x: torch.Tensor) -> torch.Tensor:
        """The neurons' activations from a pre-norm stream x [..., d]: act(gate(norm x)) * up(norm x), in float32."""
        n = self._m(layer.post_attention_layernorm, x)
        return layer.mlp.act_fn(self._f(layer.mlp.gate_proj, n)) * self._f(layer.mlp.up_proj, n)

    def _wo(self, L: int) -> torch.Tensor:
        """Layer L's o_proj weight in float32 (kept: a lower precision model's is converted once)."""
        if L not in self._wos:
            self._wos[L] = self._w(self.layers[L].self_attn.o_proj)
        return self._wos[L]

    def _on(self, dev) -> tuple:
        """(P, Jn) on a card, kept per card (a copy from the host per layer and call waited for the GPU: 41k copies, 3.2 of
        7.0 s of a Qwen3-8B graph)."""
        key = str(dev)
        if key not in self._idx:
            self._idx[key] = (self.P.to(dev), self.Jn.to(dev))
        return self._idx[key]

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

    def _dv(self, c: int, x: torch.Tensor, v: torch.Tensor, hw: torch.Tensor, off: int = 0) -> torch.Tensor:
        """The change in layer c's values at each circuit writer's position when its write there leaves layer c's input
        (the writers below c; x [K, T, d] the stream entering layer c, v [K, KV, T, hd] its values, hw [K, W] the
        writers' activations; off: the first position x and v hold) [K, W, KV, hd]. Exact through the RMS norm: v(x) - v(x - h col) = (v(x) - b)(r' - r)/r' +
        h W_v(g col)/r', with r and r' the norm's rms of x and of x - h col."""
        K, W = hw.shape
        dc = x.device
        cols, Pw = self._cols_on(dc)[:W], self._on(dc)[0][:W] - off
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
        once each), on the layer's card."""
        hit = self._plans.get(id(edges))
        if hit is not None and hit[0] is edges:
            return hit[1]
        resid, route = {}, {}
        for e, ed in enumerate(edges):
            if len(ed) == 2:
                resid.setdefault(ed[0][0], []).append((e, self.at[ed[0]], self.at[ed[1]]))
            else:
                route.setdefault(ed[1], []).append((e, self.at[ed[0]], self.at[ed[2]]))
        t = lambda xs, L: tuple(torch.tensor([x[i] for x in xs], dtype=torch.long, device=self._d(L)) for i in range(3))  # noqa: E731
        plan = {"resid": {}, "route": {}, "heads": {}}
        for L, v in resid.items():  # (with each layer's writers and readers once each)
            es, wi, ri = t(v, L)
            wu, wl = wi.unique(return_inverse=True)
            ru, rl = ri.unique(return_inverse=True)
            plan["resid"][L] = (es, wi, wu, wl, ru, rl)
        for k, v in route.items():
            es, wi, ri = t(v, k[0])
            plan["route"][k] = (es, wi, ri, *ri.unique(return_inverse=True))
            plan["heads"].setdefault(k[0], []).append(k[1])
        if self.patch:  # each reader layer's listed in-edges and what they bring in the corrupted run
            plan["fixed"] = {}
            by = {}
            for e, ed in enumerate(edges):
                by.setdefault(ed[-1][0], []).append(e)
            for L, es in by.items():
                d = self._d(L)
                plan["fixed"][L] = (torch.tensor(es, dtype=torch.long, device=d),
                                    torch.tensor([self.at[edges[e][-1]] - self.nb[L] for e in es], dtype=torch.long, device=d),
                                    self._edge_vectors([edges[e] for e in es], "corrupt", d))
        self._plans = {id(edges): (edges, plan)}
        return plan

    def _fixed(self, L: int, coef: torch.Tensor, plan: dict, rest: torch.Tensor) -> torch.Tensor:
        """Patching: what layer L's circuit neurons read instead of their cut in-edges, those edges in the corrupted run
        [K, readers of L, d]."""
        d, K, R = self._d(L), coef.shape[0], self.nb[L + 1] - self.nb[L]
        P = self._on(d)[0][self.nb[L]:self.nb[L + 1]]
        F = self._all_in("corrupt")[L][P][None] * rest.to(d)[:, None, None]  # (rest: per set, 1 cuts every other candidate)
        if L in plan.get("fixed", {}):
            es, rl, Vb = plan["fixed"][L]
            C = torch.zeros(K, R, len(es), device=d)
            C[:, rl, torch.arange(len(es), device=d)] = coef[:, es]
            F = F + C @ Vb
        return F

    @torch.no_grad()
    def _edge_vectors(self, edges: list, which: str, d) -> torch.Tensor:
        """What each edge brings into its reader's MLP input in that run [len(edges), d] (on card d)."""
        H, KV, hd = self.H, self.KV, self.hd
        cols, h0, dvs, base = self._cols_on(d), self._h0(which).to(d), self._dvs(which), self._states(which)
        V = torch.zeros(len(edges), cols.shape[1], device=d)
        groups = {}
        for i, e in enumerate(edges):
            groups.setdefault(None if len(e) == 2 else e[1], []).append((i, self.at[e[0]], e[-1][1]))
        for g, xs in groups.items():
            i, w, q = (torch.tensor(t, device=d) for t in zip(*xs))
            if g is None:
                V[i] = h0[w, None] * cols[w]
                continue
            c, h = g
            dc = self._d(c)
            w, q = w.to(dc), q.to(dc)
            z = base["A"][c][h, q, self._on(dc)[0][w]][:, None] * dvs[c][w, h // (H // KV)]
            V[i] = (z @ self._wo(c)[:, h * hd:(h + 1) * hd].T).to(d)
        return V

    def _run(self, cuts: torch.Tensor, edges: list, rest=False, keep_states: bool = False,
             probe: list | None = None, ids: list[int] | None = None) -> dict:
        """The contrast of each row of cuts [K, len(edges)], every candidate not in edges as in the reference or (rest)
        cut as well; rest: one bool for every set or one per set [K]. probe: per layer a tensor [T, d_mlp] added to the
        neurons' activations (zeros requiring grad: the contrast's gradient at them)."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        core, H, KV, hd, T, f, N = self.core, self.H, self.KV, self.hd, self.T, self._f, len(self.N)
        K = max(cuts.shape[0], 1)
        s0 = int(self.P.min()) if (PREFIX and N and not keep_states and probe is None and ids is None
                                    and getattr(self, "base", None) is not None) else 0
        Ts = T - s0  # (the positions run: from the circuit's first on; the base run's keys and values before it)
        restv = (torch.as_tensor(rest).float() if torch.is_tensor(rest) else torch.full((K,), float(rest))).to(self.dev)
        rest = bool(restv.any())  # (whether any set cuts every other candidate: the per-position sums are needed)
        coef = (cuts.float() if len(edges) else torch.zeros(K, 0)).to(self.dev) - restv[:, None]  # per listed edge: cut 1,
        # as run 0, kept against a cut rest -1 (added back)
        plan = self._plan(edges)
        x = core.model.embed_tokens(torch.tensor([(ids or self.ids)[s0:]], device=self.dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(s0, T, device=self.dev)[None])
        D = torch.zeros(K, N, x.shape[-1], device=self.dev)  # per circuit neuron, what its listed cut edges take from its input
        Hc = torch.zeros(K, N, device=self.dev)  # the circuit neurons' activations in this run
        S = torch.zeros(K, Ts, x.shape[-1], device=self.dev) if rest else None  # (rest) per position, the circuit's writes so far ...
        R = torch.zeros(K, Ts, x.shape[-1], device=self.dev) if rest else None  # ... and every route's change in the head outputs so far
        P, Jn = self._on(self.dev)  # (every index on the card once: a host copy per layer waited for the GPU)
        Ps = P - s0  # (the circuit's positions among those run)
        states = {"x": [], "A": [], "k": [], "v": [], "h": []}
        for L, layer in enumerate(self.layers):
            d = self._d(L)  # (every tensor of this layer's work on its card)
            x, D, Hc, coef, P, Ps, Jn, cos, sin = (t.to(d) for t in (x, D, Hc, coef, P, Ps, Jn, cos, sin))
            if rest:
                S, R, restv = S.to(d), R.to(d), restv.to(d)
            kk, causal = torch.arange(K, device=d), torch.ones(T, T, dtype=torch.bool, device=d).tril()[s0:]
            at = layer.self_attn
            hn = self._m(layer.input_layernorm, x)
            q = f(at.q_norm, self._m(at.q_proj, hn).view(K, Ts, H, hd)).transpose(1, 2)
            k = f(at.k_norm, self._m(at.k_proj, hn).view(K, Ts, KV, hd)).transpose(1, 2)
            v = f(at.v_proj, hn).view(K, Ts, KV, hd).transpose(1, 2)  # [K, KV, Ts, hd]
            q, k = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
            kf, vf = k, v
            if s0:  # every key and value: the base run's before s0
                kf = torch.cat([self.base["k"][L][:, :s0].to(d)[None].expand(K, -1, -1, -1), k], 2)
                vf = torch.cat([self.base["v"][L][:, :s0].to(d)[None].expand(K, -1, -1, -1), v], 2)
            A = ((q @ repeat_kv(kf, H // KV).transpose(-1, -2)) * at.scaling).masked_fill(~causal, float("-inf")).softmax(-1)
            z = A @ repeat_kv(vf, H // KV)  # [K, H, Ts, hd]
            if keep_states:
                states["x"].append(x[0].detach())
                states["A"].append(A[0].detach())
                states["k"].append(k[0].detach())
                states["v"].append(v[0].detach())
            nb = self.nb[L]
            if nb and (rest or plan["heads"].get(L)):  # routes through this layer's heads
                dv = self._dv(L, x, v, Hc[:, :nb], s0)  # [K, nb, KV, hd]
                Wo = self._wo(L)
                if rest:  # every one: each position's head outputs from every writer's value change, summed
                    pu, pi = torch.unique(P[:nb], return_inverse=True)  # (the writers' positions)
                    if len(pu) * (H * hd + H * Ts) < Ts * H * hd:  # few: each head's change at them through o_proj first,
                        dVp = torch.zeros(K, len(pu), KV, hd, device=d).index_add(1, pi, dv).repeat_interleave(H // KV, 2)
                        U = torch.einsum("kphc,Dhc->khpD", dVp, Wo.view(-1, H, hd))  # [K, H, positions, d]
                        R = R + torch.einsum("khsp,khpD->ksD", A[..., pu], U)  # (the same sum, a third of the products
                    else:  # at 4 positions of 16: Qwen3-8B, 0.70 s of float32 products in a graph's edge stage)
                        dV = torch.zeros(K, T, KV, hd, device=d).index_add(1, P[:nb], dv).transpose(1, 2)
                        R = R + (A @ repeat_kv(dV, H // KV)).transpose(1, 2).reshape(K, Ts, H * hd) @ Wo.T
                for h in plan["heads"].get(L, []):  # the listed ones: the head's output change at the reader
                    es, wi, ri, ru, inv = (t.to(d) for t in plan["route"][(L, h)])
                    dz = coef[:, es, None] * A[:, h][kk[:, None], Ps[ri][None], P[wi][None]][..., None] * dv[:, wi, h // (H // KV)]
                    Z = torch.zeros(K, len(ru), hd, device=d).index_add(1, inv, dz)
                    D.index_add_(1, ru, Z @ Wo[:, h * hd:(h + 1) * hd].T)  # (in place: D is sets x neurons x d)
            x = x + f(at.o_proj, z.transpose(1, 2).reshape(K, Ts, H * hd))
            hm = self._mlp_act(layer, x)
            rs = torch.arange(nb, self.nb[L + 1], device=d)  # the circuit's neurons of this layer
            if len(rs) and (len(edges) or rest):  # each from its own input
                inp = x[:, Ps[rs]] - D[:, rs]
                if rest:
                    inp = inp - restv[:, None, None] * (S[:, Ps[rs]] + R[:, Ps[rs]])
                if self.patch:  # (the cut edges' readers read what those edges bring in the corrupted run)
                    inp = inp + self._fixed(L, coef, plan, restv)
                hr = self._mlp_act(layer, inp).gather(-1, Jn[rs][None, :, None].expand(K, -1, 1))[..., 0]
                hm = hm.index_put((kk[:, None], Ps[rs][None], Jn[rs][None]), hr)
            if probe is not None:
                hm = hm + probe[L].to(d)[None]
            if keep_states:
                states["h"].append(hm[0].detach())
            if len(rs):
                Hc = Hc.index_copy(1, rs, hm[kk[:, None], Ps[rs][None], Jn[rs][None]])
                if rest:
                    S.index_add_(1, Ps[rs], Hc[:, rs, None] * self._cols_on(d)[rs])
            if L in plan["resid"]:  # the listed resid edges from this layer's writers: a coefficient per (reader, writer)
                es, wi, wu, wl, ru, rl = plan["resid"][L]
                C = torch.zeros(K, len(ru), len(wu), device=d)
                C.index_put_((kk[:, None].expand(-1, len(es)), rl[None].expand(K, -1), wl[None].expand(K, -1)),
                             coef[:, es] * Hc[:, wi], accumulate=True)
                D.index_add_(1, ru, C @ self._cols_on(d)[wu])
            x = x + f(layer.mlp.down_proj, hm)
        hn = self._m(core.model.norm, x[:, self.target - s0])  # the contrast from the answer's and the alternatives'
        if self._Wc is None:  # logits only, in float32 (the softmax's normalizer cancels; a bfloat16 model's own logits
            W = weight(core.lm_head)  # of ~20-30 round to 0.125-0.25: graphs' cut contrasts moved in 0.36-bit steps)
            self._Wc = W[torch.as_tensor(list(self.answer) + list(self.alternatives), device=W.device)].float()
        z = hn.float().to(self._Wc.device) @ self._Wc.T
        na = len(self.answer)
        out = {"contrast": (torch.logsumexp(z[:, :na], -1) - torch.logsumexp(z[:, na:], -1)) / math.log(2.0)}
        if keep_states:  # (every log-probability for the base runs: their top token; on the card)
            out["lp"] = torch.log_softmax(f(core.lm_head, hn), -1)
            out.update(states)
        return out

    @torch.no_grad()
    def contrasts(self, cuts: torch.Tensor, edges: list, rest=False) -> list[float]:
        """The contrast of each row of cuts [K, len(edges)] (rest: every candidate not in edges cut as well; one bool or
        one per row), B rows a forward."""
        out = []
        restv = torch.as_tensor(rest, dtype=torch.bool) if torch.is_tensor(rest) else torch.full((len(cuts),), bool(rest))
        low = self.dt != torch.float32
        if low and PAD == "shapes":  # every forward: its sets, an uncut set, copies of its last set to the next shape
            for s0 in range(0, len(cuts), self.B - 1):
                c, r = cuts[s0:s0 + self.B - 1], restv[s0:s0 + self.B - 1]
                n = len(c)
                size = min(next(b for b in SHAPES + (self.B,) if b >= n + 1), self.B)
                c = torch.cat([c, torch.zeros(1, c.shape[1], dtype=c.dtype), c[-1:].expand(size - n - 1, -1)])
                r = torch.cat([r, torch.zeros(1, dtype=torch.bool), r[-1:].expand(size - n - 1)])
                v = self._run(c, edges, rest=r)["contrast"]
                out.append(v[:n] - v[n] + self.c_full)  # (each against its forward's uncut set, at c_full's level)
                self.forwards += 1
            return torch.cat(out).tolist() if out else []
        for s0 in range(0, len(cuts), self.B):
            c, r = cuts[s0:s0 + self.B], restv[s0:s0 + self.B]
            if low and PAD == "B" and len(c) < self.B:  # (a lower precision model's kernels round by batch size: every
                c = torch.cat([c, c[-1:].expand(self.B - len(c), -1)])  # forward padded to B sets)
                r = torch.cat([r, r[-1:].expand(self.B - len(r))])
            out.append(self._run(c, edges, rest=r)["contrast"][:len(cuts) - s0])
            self.forwards += 1
        return torch.cat(out).tolist() if out else []  # (read once: no wait for the card between forwards)

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
            dm = self.core.config.intermediate_size
            probe = [torch.zeros(self.T, dm, device=self._d(L), requires_grad=True) for L in range(self.L)]
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

    def _states(self, which: str) -> dict:
        return self.base if which == "clean" else self.cbase

    def _h0(self, which: str = "clean") -> torch.Tensor:
        """The circuit neurons' activations in the base run (or the corrupted one) [N] (on the first card)."""
        if not self.N:
            return torch.zeros(0, device=self.dev)
        return torch.stack([self._states(which)["h"][L][p, j].to(self.dev) for L, p, j in self.N])

    def _dvs(self, which: str = "clean") -> dict:
        """Per head layer c, that run's value change at each circuit writer below c when its write leaves layer c's
        input there [nb[c], KV, hd] (_dv; kept, on layer c's card)."""
        if which not in self._dv_base:
            h0, base = self._h0(which), self._states(which)
            self._dv_base[which] = {c: self._dv(c, base["x"][c][None], base["v"][c][None], h0[:self.nb[c]].to(self._d(c))[None])[0]
                                    for c in range(1, self.L) if self.nb[c]}
        return self._dv_base[which]

    def _all_in(self, which: str = "clean") -> list:
        """Per layer L, what every circuit in-edge brings into layer L's MLP input at each position in that run [T, d]
        (the circuit's writes below L at the position, plus every head's output change there through layers up to L from
        every circuit writer below that layer; kept, on layer L's card)."""
        if which not in self._allin:
            T, H, KV, hd, h0, dvs = self.T, self.H, self.KV, self.hd, self._h0(which), self._dvs(which)
            base = self._states(which)
            dm = base["x"][0].shape[-1]
            S = R = None
            out = []
            for L, layer in enumerate(self.layers):
                d = self._d(L)
                S = torch.zeros(T, dm, device=d) if S is None else S.to(d)
                R = torch.zeros(T, dm, device=d) if R is None else R.to(d)
                if L in dvs:
                    dV = torch.zeros(T, KV, hd, device=d).index_add(0, self.P[:self.nb[L]].to(d), dvs[L])
                    z = base["A"][L] @ dV.transpose(0, 1).repeat_interleave(H // KV, 0)  # [H, T, hd]
                    R = R + z.transpose(0, 1).reshape(T, H * hd) @ self._wo(L).T
                out.append(S + R)
                rs = torch.arange(self.nb[L], self.nb[L + 1], device=d)
                if len(rs):
                    S = S.index_add(0, self.P.to(d)[rs], h0.to(d)[rs, None] * self._cols_on(d)[rs])
            self._allin[which] = out
        return self._allin[which]

    @torch.no_grad()
    def in_edges(self, M: int) -> list[tuple[list, torch.Tensor]]:
        """Per circuit neuron as a reader (N's order): its candidate in-edges of the M largest |local effect| and those
        effects, largest first. The local effect is the reader's input Jacobian times what the edge brings in the base
        run (patching: less what it brings in the corrupted run), a ranking only (local_curves measures the reader). Routes are scored a head layer at a time (every writer
        below it, reader above it and head at once), each reader's best M carried."""
        N, H, KV, hd, base, dev = len(self.N), self.H, self.KV, self.hd, self.base, self.dev
        if not N:
            return []
        J, h0, dvs = self.jacobians(), self._h0(), self._dvs()
        h0b, dvb = (self._h0("corrupt"), self._dvs("corrupt")) if self.patch else (None, None)
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
        s = (Jm @ self._cols_on(dev).T) * (h0 if h0b is None else h0 - h0b)[None]  # resid [reader, writer]
        ok = (P[:, None] == P[None]) & (lay[None] < lay[:, None])
        carry(torch.arange(N), s.abs().masked_fill(~ok, -1.0), s,
              lambda t: torch.stack([t, torch.full_like(t, -1), torch.zeros_like(t)], -1))
        for c, dv in dvs.items():
            nb = self.nb[c]
            if nb == N:
                continue
            dc = self._d(c)
            Pd = self.P.to(dc)
            dv = dv[:, torch.arange(H, device=dc) // (H // KV)]  # [W, H, hd]
            U = torch.einsum("rd,dhk->rhk", Jm[nb:].to(dc), self._wo(c).view(-1, H, hd))  # [R, H, hd]
            a = base["A"][c][:, Pd[nb:]][:, :, Pd[:nb]].permute(1, 2, 0)  # [R, W, H]
            s = a * torch.einsum("rhk,whk->rwh", U, dv)
            if dvb is not None:  # (patching: less what the edge brings in the corrupted run)
                ab = self.cbase["A"][c][:, Pd[nb:]][:, :, Pd[:nb]].permute(1, 2, 0)
                s = s - ab * torch.einsum("rhk,whk->rwh", U, dvb[c][:, torch.arange(H, device=dc) // (H // KV)])
            s = s.flatten(1)  # [R, W H]
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
    def _vectors(self, rows: list[int], cands: list, K: int, which: str | None = None) -> torch.Tensor:
        """What the first K candidate in-edges of each reader of rows (one layer's) bring into its MLP input in the base
        run [len(rows), K, d] (zero past a reader's candidates; on the layer's card); patching (which None): less what
        they bring in the corrupted run."""
        if which is None:
            V = self._vectors(rows, cands, K, "clean")
            return V - self._vectors(rows, cands, K, "corrupt") if self.patch else V
        L = self.N[rows[0]][0]
        d, H, KV, hd = self._d(L), self.H, self.KV, self.hd
        cols, h0, dvs, base = self._cols_on(d), self._h0(which).to(d), self._dvs(which), self._states(which)
        V = torch.zeros(len(rows), K, cols.shape[1], device=d)
        res, rte = [], {}  # (one host-to-card copy per head layer: a copy per head waited for the card, 39-48 calls
        for a, i in enumerate(rows):  # 1.2-1.5 s of a Qwen3-8B graph's edge stage)
            for s, e in enumerate(cands[i][0][:K]):
                if len(e) == 2:
                    res.append((a, s, self.at[e[0]]))
                else:
                    rte.setdefault(e[1][0], []).append((e[1][1], a, s, self.at[e[0]], e[-1][1]))
        if res:
            a, s, w = self._idx_on(res, d)
            V[a, s] = h0[w, None] * cols[w]
        for c, xs in rte.items():  # every head of a layer at once: each candidate's head output change through o_proj
            dc = self._d(c)
            h, a, s, w, q = self._idx_on(xs, dc)
            z = base["A"][c][h, q, self._on(dc)[0][w]][:, None] * dvs[c][w, h // (H // KV)]  # [n, hd]
            Z = torch.zeros(len(xs), H, hd, device=dc)
            Z[torch.arange(len(xs), device=dc), h] = z  # (zero but at its head)
            V[a.to(d), s.to(d)] = (Z.view(len(xs), H * hd) @ self._wo(c).T).to(d)
        return V

    @staticmethod
    def _idx_on(xs: list, dev) -> tuple:
        """Integer tuples xs as one index tensor per field on a card (one pinned, unwaited copy)."""
        t = torch.tensor(xs, dtype=torch.long)
        t = t.pin_memory().to(dev, non_blocking=True) if torch.device(dev).type == "cuda" else t.to(dev)
        return tuple(t.T)

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
        """The readers rows' (layer L's) MLP inputs in the base run less what every circuit in-edge brings [len(rows), d]
        (patching: plus what they bring in the corrupted run)."""
        q = self.P[rows].to(self._d(L))
        x = self._mid_stream(L)[q] - self._all_in()[L][q]
        return x + self._all_in("corrupt")[L][q] if self.patch else x

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
    floor = max(MIN_TOL, GAP_TOL * abs(ed.gap) if ed.patch else 0.0)
    if ed.patch:  # (patching: the allowances scale with the floor and every edge cut rides in the allowances' forward)
        c_none, scale = None, floor
    else:
        c_none = ed.contrasts(torch.zeros(1, 0, dtype=torch.bool), [], rest=True)[0]
        scale = max(TOL * abs(c_all - c_none), floor)
    cands = ed.in_edges(IN_EDGES)
    curves = ed.local_curves(cands)
    ref = ed._h0().cpu()
    err = [(curves[i] - ref[i]).abs() / max(abs(float(ref[i])), 1e-9) for i in range(N)]
    sens = (ed.grads() * ref).abs()  # (each reader's contrast per unit relative change, first order: allowances only)
    allow = [("error", eps, [eps] * N) for eps in EPS] + [
        ("bits", b, [b * scale / max(float(sens[i]), 1e-12) for i in range(N)]) for b in BITS]
    K = [[int(e.le(a).nonzero()[0]) if bool(e.le(a).any()) else int(e.argmin()) for e, a in zip(err, al)]
         for _, _, al in allow]
    t_local = time.time() - t0

    kmax = [max(k[i] for k in K) for i in range(N)]
    listed = [cands[i][0][s] for i in range(N) for s in range(kmax[i])]
    owner = torch.tensor([i for i in range(N) for _ in range(kmax[i])], dtype=torch.long)
    rank = torch.tensor([s for i in range(N) for s in range(kmax[i])], dtype=torch.long)
    rows = torch.stack([rank >= torch.tensor(k, dtype=torch.long)[owner] for k in K])  # each reader's prefix kept, per allowance
    if c_none is None:
        cs = ed.contrasts(torch.cat([rows, torch.ones(1, len(listed), dtype=torch.bool)]), listed, rest=True)
        cs, c_none = cs[:-1], cs[-1]
    else:
        cs = ed.contrasts(rows, listed, rest=True)
    tol = max(TOL * abs(c_all - c_none), floor)  # (two-sided: a kept set that overshoots is no more faithful)
    ok = lambda c: abs(c - c_all) <= tol  # noqa: E731
    measured = 1 + len(allow)
    passing = [a for a, c in enumerate(cs) if ok(c)]
    pick = min(passing, key=lambda a: int((~rows[a]).sum())) if passing else len(EPS) - 1  # (the fewest edges)
    gone = rows[pick].clone()  # (cut)
    before = int((~gone).sum())
    log(f"reader allowance {allow[pick][0]} {allow[pick][1]:.4g} {'passes' if passing else 'FAILS'} with the fewest: "
        f"{before} edges kept of {len(listed)} listed (contrast {cs[pick]:.3f}, every edge {c_all:.3f}, none {c_none:.3f}, "
        f"tolerance {tol:.3f}; {len(passing)} of {len(allow)} allowances pass), {time.time() - t0:.1f} s")
    t_eps = time.time() - t0 - t_local
    f_eps = ed.forwards

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
    pending = None  # (the last round's passing blocks, gone with all of them cut): checked in the next round's forward
    f_start = ed.forwards
    while (queue or pending) and ed.forwards - f_start < HALVE_FORWARDS:  # halving over readers' in-edge groups, a
        # forward's blocks a round: each block's groups cut
        blocks, queue = queue[:ed.B], queue[ed.B:]  # alone, the passing ones together (that check rides in the next
        on = pending[1] if pending else gone  # round's forward, the round's blocks tried as if it passes)
        trial = []
        for b in blocks:
            t = on.clone()
            t[cut_of(b)] = True
            trial.append(t)
        if pending:
            trial.append(pending[1])
        cs = ed.contrasts(torch.stack(trial), listed, rest=True)
        measured += len(trial)
        if pending:
            passed_before, together = pending
            pending = None
            if ok(cs[-1]):
                gone = together
            else:  # together they fail: one at a time, and this round's blocks again from what stands
                for b in passed_before:
                    t = gone.clone()
                    t[cut_of(b)] = True
                    measured += 1
                    if ok(ed.contrasts(t[None], listed, rest=True)[0]):
                        gone = t
                queue = blocks + queue
                continue
        passed = [b for b, c in zip(blocks, cs) if ok(c)]
        if len(passed) == 1:  # (one block: its own test was the together check)
            gone = gone.clone()
            gone[cut_of(passed[0])] = True
        elif passed:
            t = gone.clone()
            t[cut_of([i for b in passed for i in b])] = True
            pending = (passed, t)
        queue = [list(h) for b, c in zip(blocks, cs) if not ok(c) and len(b) > 1
                 for h in torch.tensor(b).tensor_split(2)] + queue
    stopped = bool(queue or pending)  # (HALVE_FORWARDS reached: readers' groups left untried)
    keep = (~gone).nonzero().flatten()
    t_halve = time.time() - t0 - t_local - t_eps
    f_halve = ed.forwards

    readers = [i for i in range(N) if bool((~gone & (owner == i)).any())]
    ks = [int((~gone & (owner == i)).sum()) for i in range(N)]  # (each reader's kept in-edges: a prefix of its own)
    eff = ed.local_effects(cands, ks)
    local = torch.cat([eff[int(owner[e])][int(rank[e])][None] for e in keep]) if len(keep) else torch.zeros(0)
    est = torch.tensor([float(local[a].abs()) * float(sens[int(owner[e])]) / max(abs(float(ref[int(owner[e])])), 1e-9)
                        for a, e in enumerate(keep.tolist())])  # (first-order: which MEASURED edges get their own cut)
    top = torch.argsort(est, descending=True)[:MEASURED].tolist()
    trial = [gone]  # one forward's worth: the graph, each reader's kept in-edges cut from it, each top edge cut from it
    for i in readers:
        t = gone.clone()
        t[owner == i] = True
        trial.append(t)
    for a in top:
        t = gone.clone()
        t[keep[a]] = True
        trial.append(t)
    trial.append(~gone)  # and the graph's edges cut with every other edge on (no rest)
    rest = torch.ones(len(trial), dtype=torch.bool)
    rest[-1] = False
    cs = ed.contrasts(torch.stack(trial), listed, rest=rest)
    c_kept, c_cut = cs[0], cs[-1]
    group = {i: c_kept - c for i, c in zip(readers, cs[1:1 + len(readers)])}
    strengths = [None] * len(keep)
    for a, c in zip(top, cs[1 + len(readers):-1]):
        strengths[a] = c_kept - c
    measured += len(trial)
    best = int(ed.base["lp"][0].argmax())
    rec = {"contrast_model": ed.c_full, "reference_top": best, "answer_top": best in ed.answer, "gap": ed.gap,
           "edges_share_of_gap": (c_all - c_none) / ed.gap if ed.patch and ed.gap else None, "contrast_all_edges": c_all, "contrast_no_edges": c_none, "contrast_kept": c_kept,
           "contrast_graph_cut": c_cut, "tolerance": tol, "met": bool(passing) and ok(c_kept),
           "allowance": list(allow[pick][:2]),
           "candidates": {"resid": len(ed.resid_all), "route": ed.n_routes,
                          "in_edges_listed": sum(len(c[0]) for c in cands)},
           "kept": len(keep), "kept_before_halving": before, "halving_stopped": stopped, "readers": len(readers),
           "measured": measured,
           "forwards": {"errors": f_eps, "halving": f_halve - f_eps, "last": ed.forwards - f_halve}, "sets_a_forward": ed.B,
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
    over the reader's activation) and, for the MEASURED of largest first-order weight, the measured
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
                      "in_strength": got.get("group", {}).get((L, p, j)), "label": None})
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
                                               "value": ed.c_full}, "cut_kind": "patch" if ed.patch else "zero", "nodes": nodes,
            "edges": out, "record": got["record"],
            "cut": {"none": got["record"].get("contrast_all_edges"), "graph_edges": got["record"].get("contrast_graph_cut"),
                    "every_edge": got["record"].get("contrast_no_edges")},  # (the contrast with those edges cut together)
            "route_note": "A route edge is a value-path cut: each head's attention pattern is held at its value in the run, "
                          "so a writer's effect through where a head looks (its query or key) is not part of any edge.",
            "route_cut_semantics": {"path": "value_ov", "attention_pattern": "current_run_fixed_in_local_value_delta",
                                    "head_input_norm": "recomputed_after_writer_write_subtraction",
                                    "reader_mlp": "recomputed_norm_gate_up", "combined_routes": "sum_of_individual_value_deltas",
                                    "query_key_effects_included": False, "full_writer_ablation_equivalent": False,
                                    "full_causal_completeness_certified": False}}


def answer_sets(rec: dict) -> tuple[list[int], list[int]]:
    """A record's contrast token sets: every token of the answer's word (answer_tokens; else the answer token) against
    every token of the alternatives (answer_contrast.py's word contrast)."""
    answer = rec.get("answer_tokens") or [rec["answer"]]
    return [int(t) for t in answer], [int(t) for t in rec["alternatives"]]


def finder_graph(model, tok, rec: dict, log=print) -> dict:
    """A circuit_finder record's edge graph by patching (its clean and corrupted prompts, every cut a patch from the
    corrupted run), with route knockouts for every cross-position edge: each head attending from the reader's position
    to the writer's (at least 0.01) unable to see the writer's position alone, its share of the gap lost
    (head_routes.carriers); with mover heads, their edges to the circuit's neurons (mover_edges); and every circuit
    unit's downstream readers in the whole model with measured effect and sign, and its direct write (readers.py)."""
    import head_routes
    t, sec = time.time(), {}
    ids, bad = rec["ids"], rec["corrupted"]
    answer, alts = answer_sets(rec)
    ed = Edges(model, ids, len(ids) - 1, answer, alts,
               [(n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"]], corrupt_ids=bad)
    sec["setup"], t = round(time.time() - t, 3), time.time()
    got = prune(ed, log=log)
    sec["prune"], t = round(time.time() - t, 3), time.time()
    g = graph(ed, got, tok)
    sec["graph"], t = round(time.time() - t, 3), time.time()
    rn = head_routes.Runner(model)
    rn.answer, rn.alternatives = answer, alts
    run = rn.run(ids, [{}], capture=True)
    c, cb = rn.run(ids, [{}])["contrast"][0], rn.run(bad, [{}])["contrast"][0]  # (uncut, at the cut runs' batch size)
    lost = lambda sets: [(c - x) / (c - cb) for x in rn.run(ids, sets)["contrast"]]  # noqa: E731
    pairs = sorted({(e["dst_pos"], e["src_pos"]) for e in g["edges"] if e["src_pos"] != e["dst_pos"]})
    sec["runner"], t = round(time.time() - t, 3), time.time()
    g["routes"] = []
    for q, p in pairs:  # (each head alone, then the first k together: single heads often move little)
        car = head_routes.carriers(rn, run["states"], lost, q, p)
        g["routes"].append({"q": q, "p": p, "carriers": car, "carrier_curve": head_routes.carrier_curve(rn, lost, q, p, car)})
    sec["carriers"], t = round(time.time() - t, 3), time.time()
    g["head_edges"], g["movers_inputs_lost"] = [], {}
    if rec.get("routes"):
        sb = rn.run(bad, [{}], capture=True)["states"]
        g["head_edges"], g["movers_inputs_lost"] = mover_edges(rn, run["states"], sb, lost, rec)
        sec["mover_edges"], t = round(time.time() - t, 3), time.time()
    g["readers"] = []
    if rec.get("nodes") or rec.get("routes"):
        import readers  # noqa: PLC0415
        units = readers.record_units(rec)
        g["readers"] = readers.readers(rn, ids, units, corrupt=bad, circuit=set(units))
        sec["readers"] = round(time.time() - t, 3)
    g["seconds"] = sec
    g.update(id=rec.get("id"), text_id=rec.get("id"), user=rec.get("user"), corrupt_user=rec.get("corrupt_user"),
             corrupted=bad, model=rec.get("model"), finder=rec.get("finder"), intervention="patch from the corrupted run",
             gap=c - cb, answer_word=rec.get("answer_word"), alternative_words=rec.get("alternative_words"))
    # Source declarations are useful provenance, not an attestation that this
    # caller loaded the same checkpoint/adapter. Preserve graph()'s cut dictionary.
    g["source_provenance"] = {k: rec[k] for k in (
        "model", "adapter", "dtype", "finder", "finder_rev", "model_revision",
        "tokenizer_sha256", "adapter_sha256") if k in rec}
    g["source_provenance_status"] = "copied_source_record_not_verified_against_loaded_model"
    return g


@torch.no_grad()
def mover_edges(rn, st: dict, sb: dict, lost, rec: dict) -> tuple[list[dict], dict]:
    """Edges between a circuit's mover heads and its neurons, by patching from the corrupted run. Per mover (rec's
    routes: head (L, h) at the answer position q), each circuit neuron below L at a position up to q, its write's change
    (clean - corrupted) taken out of the head's key and value inputs at its position (and its query input at q): the
    head's output change at q, exact with the rest of the head as in the clean run ("inputs": its size and its share of
    the head's own change, clean - corrupted, by projection), and all of them at once per input and over every input in
    full runs ("group_lost": shares of the gap lost); each circuit readout neuron at q in layer L or above, the head's
    change taken out of its input (its value change, exact) and all of them at once (readers_group_lost). With them,
    every mover's inputs cut at once, per input and over every input: the share of the gap lost. Every write's
    change is computed once and every mover's group cuts are one batch of runs (on two Qwen3-8B copy texts the value
    input alone took 2.1 and 10.1 s, a float32 copy of down_proj per neuron and mover; its groups lost under 1% of the
    gap where the movers lost 74-86%)."""
    from head_routes import weight  # noqa: PLC0415
    nodes = [n for n in rec.get("nodes", []) if n.get("stage") in ("source", "readout")]
    units = sorted({(n["layer"], n["pos"], n["neuron"]) for n in nodes})
    rdo = sorted({(n["layer"], n["pos"], n["neuron"]) for n in nodes if n["stage"] == "readout"})
    by, diff = {}, {}
    for Ln, p, j in units:
        by.setdefault((Ln, p), []).append(j)
    for (Ln, p), js in by.items():
        W = weight(rn.layers[Ln].mlp.down_proj)[:, js].float()
        dh = st["h"][Ln][p, js].to(W.device) - sb["h"][Ln][p, js].to(W.device)
        diff.update(zip(((Ln, p, j) for j in js), (W * dh).T))
    out, sets, slots, every = [], [], [], {}
    kinds = {"key": "dk", "value": "dv", "query": "dq"}
    for r in rec.get("routes", []):
        L, h, q = r["layer"], r["head"], r["pos"]
        own = rn.head_write(st, L, h, q) - rn.head_write(sb, L, h, q).to(rn._d(L))
        e = {"head": [L, h], "pos": q, "inputs": [], "group_lost": {}, "readers": [], "readers_group_lost": None}
        byp, edits = {}, {}
        for u in units:
            if u[0] < L and u[1] <= q:
                byp.setdefault(u[1], []).append(u)
        for p, us in sorted(byp.items()):
            ws = torch.stack([diff[u] for u in us])
            for side in ("key", "value") + (("query",) if p == q else ()):
                ch, _ = rn.output_after(st, L, h, q, p, ws, side)
                ch = ch.to(own.device)
                share = -(ch @ own) / own.pow(2).sum()
                e["inputs"] += [{"neuron": list(u), "input": side, "output_change": float(z), "share": float(x)}
                                for u, z, x in zip(us, ch.norm(dim=-1), share)]
                edits.setdefault(side, []).append((L, h, p, rn.output_after(st, L, h, q, p, ws.sum(0)[None], side)[1][0]))
        for side, ed in edits.items():
            slots.append((e, side))
            sets.append({kinds[side]: ed})
            every.setdefault(side, []).extend(ed)
        if edits:
            slots.append((e, "all"))
            sets.append({kinds[side]: ed for side, ed in edits.items()})
        rs = [(Ln, j) for Ln, p, j in rdo if p == q and Ln >= L]
        if rs:
            ch = rn.readers_after(st, L, h, q, rn.head_write(sb, L, h, q))
            e["readers"] = [{"neuron": [Ln, q, j], "value_change": float(ch[Ln][j])} for Ln, j in rs]
            slots.append((e, "readers"))
            sets.append({"read": [(Ln, q, j, own) for Ln, j in rs]})
        out.append(e)
    total = {}
    for side, ed in every.items():
        slots.append((total, side))
        sets.append({kinds[side]: ed})
    if every:
        slots.append((total, "all"))
        sets.append({kinds[side]: ed for side, ed in every.items()})
    for (e, side), x in zip(slots, lost(sets) if sets else []):
        if side == "readers":
            e["readers_group_lost"] = x
        elif e is total:
            total[side] = x
        else:
            e["group_lost"][side] = x
    return out, total


def free():
    """Return a finished record's GPU memory (its Edges' float32 weight copies and states) before the next: on Qwen3-8B
    a directory of taboo records filled an L40 by the fourth."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(name: str, device: str | None = None, dtype: str = "float32"):
    """An HF causal LM for the stage: float32 on one card, or bfloat16 split over every card (device_map auto), its
    weights without gradients."""
    import transformers
    if dtype == "bfloat16":
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
    else:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev)
        if dev == "cpu":
            torch.set_num_threads(8)
    return model.eval().requires_grad_(False)  # (the stage's gradients are at activations: no weight gradient, 8B's
    # float32 one did not fit an L40)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("one", "finder"))
    ap.add_argument("model")
    ap.add_argument("text", type=Path, help="one: TEXT.json; finder: a circuit_finder record or a directory of them")
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="the model's weights (bfloat16: 32B "
                    "over two cards, device_map auto)")
    ap.add_argument("--adapter", default=None, help="a peft adapter (an organism) on the model; finder: else each record's "
                    "own (its \"adapter\")")
    a = ap.parse_args()
    import transformers
    model = load(a.model, a.device, a.dtype)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if a.cmd == "finder":  # circuit_finder records: the patching pair, every cut a patch from the corrupted run
        import head_routes
        files = sorted(a.text.glob("*.json")) if a.text.is_dir() else [a.text]
        if a.text.is_dir():
            a.out.mkdir(parents=True, exist_ok=True)
        todo = [(src, json.loads(src.read_text())) for src in files if not (a.text.is_dir() and (a.out / src.name).exists())]
        for m, (src, rec) in head_routes.by_adapter(model, todo, lambda it: a.adapter or it[1].get("adapter")):
            dst = a.out / src.name if a.text.is_dir() else a.out
            t0 = time.time()
            g = finder_graph(m, tok, rec, log=lambda msg: print(f"{src.stem}: {msg}", flush=True))
            free()
            g["adapter"] = a.adapter or rec.get("adapter")
            g["seconds"] = {**g["seconds"], "total": round(time.time() - t0, 2)}  # (this stage, by part; the circuit's
            g["build_seconds"] = rec.get("seconds")                               # build is the record's seconds)
            g["seconds_total"] = round(g["seconds"]["total"] + (rec.get("seconds") or 0.0), 2)
            dst.write_text(json.dumps(g))
            print(f"{src.stem}: {len(g['nodes'])} neurons, {len(g['edges'])} edges, {len(g['routes'])} routes, "
                  f"{len(g.get('readers') or [])} units' readers, {g['seconds']} s", flush=True)
        return
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval().requires_grad_(False)
    t = json.loads(a.text.read_text())
    ed = Edges(model, t["ids"], t["target"], t["answer"], t["alternatives"], [tuple(n) for n in t["neurons"]],
               corrupt_ids=t.get("corrupted"))
    got = prune(ed, log=lambda m: print(f"{a.text.stem}: {m}", flush=True))
    a.out.write_text(json.dumps(graph(ed, got, tok)))


if __name__ == "__main__":
    main()
