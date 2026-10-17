"""Attention heads as routes on an HF Qwen3, measured: where each head looks, what moves through it, and which neurons set
its query and key or read its output (the 10-15 circuit finder's heads and edges; the essay's "the heads cannot see X"
test, run on every route).

Runner.run(ids, sets): one prompt, a batch of sets (one forward each, batched), each with its own edits:
  knock   (layer, head, q, p)          the head's attention from position q to p removed (its logit masked, the row
                                       renormalized over the rest);
  head    (layer, head, pos, value)    the head's output at pos (its slice of o_proj's input) set: None zeroes it;
  neuron  (layer, pos, index, value)   a neuron's value (down_proj's input) set;
  dq, dk  (layer, head, pos, delta)    added to the head's query or key at pos (after its norm and rotary embedding):
                                       a writer cut from that head's query or key alone (attention_after gives the delta);
  dv      (layer, head, pos, delta)    added to the head's value at pos: a writer cut from that head's value alone
                                       (value_after gives the delta);
  read    (layer, pos, index, delta)   the neuron recomputed from its MLP input (the stream after attention) less delta:
                                       a head's output, say, cut from that neuron alone.
It returns the contrast of each set (log2 p(answer) - log2 of the alternatives' summed p at the last position) and with
capture the first set's states. Each module runs in the model's dtype on its card, everything between in float32.

Local measurements on captured states, exact (norms, gate and up recomputed in float32), no forward:
  attention_after: a head's attention from q to p with each of some residual vectors (neurons' writes, another head's
    output) taken out of its key input at p or its query input at q;
  readers_after: every neuron at a position in later layers recomputed with one head's output there taken out of its
    input (the head -> neuron edges' local size).
route_story / carriers: one head's route measured end to end (knockout, the neurons and heads setting its key and query,
the neurons reading its output; every head able to move the same information), by zeroing or by patching from the
corrupted run of a patching pair.

  python head_routes.py finder MODEL RECORDS OUT [--dtype bfloat16]   (circuit_finder records: their mover heads)
  python head_routes.py heads MODEL RECORDS OUT --heads L26H17,L23H15 [--neurons L1:2039,L0:9761] [--pos P]
"""
from __future__ import annotations

import math

import torch

ROWS = 64


def weight(mod) -> torch.Tensor:
    """A linear module's effective weight [out, in]: its own, or under a LoRA adapter (peft) the base layer's plus each
    active adapter's B A times its scaling (the local measurements read weights; the forward runs the modules)."""
    if not hasattr(mod, "base_layer"):
        return mod.weight
    W = mod.base_layer.weight
    for a in getattr(mod, "active_adapters", []):
        if a in mod.lora_A:
            W = W + (mod.lora_B[a].weight @ mod.lora_A[a].weight).to(W.dtype) * mod.scaling[a]
    return W
PREFIX = True  # a forward computes only the positions from the first one an edit (or a watched neuron) names, its
# attention reading the unedited run's keys and values before it (no edit reaches them)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)


class Runner:
    """Batched forwards of one prompt with per-set edits (module docstring), on an HF Qwen3 (or a PeftModel of one)."""

    def __init__(self, model, rows: int = ROWS):
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = self.core.config
        self.layers = list(self.core.model.layers)
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = self.core.model.embed_tokens.weight.device
        self.rows = rows
        self.answer, self.alternatives = None, None
        self._base = {}  # ids -> per layer the unedited run's keys [KV, T, hd] (after norm and rotary) and values

    def _d(self, L: int) -> torch.device:
        return self.layers[L].input_layernorm.weight.device

    @staticmethod
    def _f(mod, t: torch.Tensor) -> torch.Tensor:
        w = next(mod.parameters())
        return mod(t.to(w.device, w.dtype)).float()

    @staticmethod
    def _compile(sets: list[dict]) -> dict:
        """Per edit kind and layer: (set index, ...) index tensors and the values."""
        out = {}
        for b, s in enumerate(sets):
            for kind, items in s.items():
                for it in items:
                    out.setdefault(kind, {}).setdefault(it[0], []).append((b, *it[1:]))
        return out

    @torch.no_grad()
    def run(self, ids: list[int], sets: list[dict], capture: bool = False, watch: list | None = None) -> dict:
        """The contrast of each set (from the answer's and the alternatives' logits alone, in float32: a bfloat16 model's
        own logits of 20-30 round in steps of 0.125-0.25), and "lp" [K, V] when capturing or with no answer set; capture:
        the first set's states per layer ("x" the stream entering
        the layer [T, d], "q" [H, T, hd] and "k" [KV, T, hd] after norm and rotary, "v" [KV, T, hd], "logits" and "A"
        [H, T, T], "z" the head outputs [T, H, hd], "mid" the stream after attention [T, d], "h" the neurons [T, d_mlp]),
        "cos"/"sin" [T, hd]; watch: neurons (layer, pos, index) whose values every set reports ("watch" [K, n])."""
        out = {"contrast": [], "lp": [], "watch": []}
        low = self.core.model.embed_tokens.weight.dtype != torch.float32
        for s0 in range(0, max(len(sets), 1), self.rows):
            part = sets[s0:s0 + self.rows] or [{}]
            n = len(part)
            if low and not capture:  # (a lower precision model's kernels round by batch size: every forward padded to rows)
                part = part + [{}] * (self.rows - n)
            r = self._forward(ids, part, capture and s0 == 0, watch or [])
            out["contrast"] += r["contrast"][:n].tolist() if r["contrast"] is not None else []
            if r["lp"] is not None:
                out["lp"].append(r["lp"][:n])
            out["watch"].append(r["watch"][:n])
            if capture and s0 == 0:
                out["states"] = r["states"]
        out["lp"] = torch.cat(out["lp"]) if out["lp"] else None
        out["watch"] = torch.cat(out["watch"])
        return out

    def grads(self, ids: list[int]) -> dict:
        """The contrast's gradient in the unedited run at every neuron's value and every head's output: {"h": per layer
        [T, d_mlp], "z": per layer [T, H, hd]} (one backward; for ordering candidates only)."""
        T, dm = len(ids), self.layers[0].mlp.down_proj.in_features
        pr = {"h": [torch.zeros(1, T, dm, device=self._d(L), requires_grad=True) for L in range(self.L)],
              "z": [torch.zeros(1, T, self.H, self.hd, device=self._d(L), requires_grad=True) for L in range(self.L)]}
        with torch.enable_grad():
            self._forward(ids, [{}], True, [], probe=pr)["contrast"].sum().backward()
        return {k: [t.grad[0] for t in v] for k, v in pr.items()}

    def _forward(self, ids: list[int], sets: list[dict], capture: bool, watch: list, probe: dict | None = None) -> dict:
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        core, H, KV, hd, f = self.core, self.H, self.KV, self.hd, self._f
        K, T = len(sets), len(ids)
        E = self._compile(sets)
        s0 = 0                                      # the first position computed: every earlier one is the unedited run's
        if PREFIX and not capture:
            first = [it[2] if kind in ("knock", "head", "dq", "dk", "dv") else it[1]
                     for kind, per in E.items() for items in per.values() for it in items]
            s0 = min(first + [p for _, p, _ in watch] + [T - 1])
            if s0 and tuple(ids) not in self._base:
                st0 = self._forward(ids, [{}], True, [])["states"]
                self._base[tuple(ids)] = (st0["k"], st0["v"])
                while len(self._base) > 4:          # (a pair's two prompts and a few others)
                    self._base.pop(next(iter(self._base)))
        Ts = T - s0
        x = core.model.embed_tokens(torch.tensor([ids[s0:]], device=self.dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(s0, T, device=self.dev)[None])
        cos, sin = cos[0].float(), sin[0].float()  # [Ts, hd]
        st = {k: [] for k in ("x", "q", "k", "v", "logits", "A", "z", "mid", "h")}
        st.update(cos=cos, sin=sin)
        seen = torch.zeros(K, len(watch))
        for L, layer in enumerate(self.layers):
            d = self._d(L)
            x, cs, sn = x.to(d), cos.to(d), sin.to(d)
            at = layer.self_attn
            hn = f(layer.input_layernorm, x)
            q = f(at.q_norm, f(at.q_proj, hn).view(K, Ts, H, hd)).transpose(1, 2)
            k = f(at.k_norm, f(at.k_proj, hn).view(K, Ts, KV, hd)).transpose(1, 2)
            v = f(at.v_proj, hn).view(K, Ts, KV, hd).transpose(1, 2)
            q, k = q * cs + rotate_half(q) * sn, k * cs + rotate_half(k) * sn
            kf, vf = k, v
            if s0:                                  # every key and value: the unedited run's before s0
                bk, bv = self._base[tuple(ids)]
                kf = torch.cat([bk[L][:, :s0].to(d)[None].expand(K, -1, -1, -1), k], 2)
                vf = torch.cat([bv[L][:, :s0].to(d)[None].expand(K, -1, -1, -1), v], 2)
            kh, vh = repeat_kv(kf, H // KV), repeat_kv(vf, H // KV)
            for kind, t, o in (("dq", q, s0), ("dk", kh, 0), ("dv", vh, 0)):   # (q holds the positions from s0; keys
                for b, h, p, delta in E.get(kind, {}).get(L, []):              # and values every position)
                    t[b, h, p - o] += delta.to(d, t.dtype)
            lg = (q @ kh.transpose(-1, -2)) * at.scaling
            lg = lg.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=d).tril()[s0:], float("-inf"))
            for b, h, qq, p in E.get("knock", {}).get(L, []):
                lg[b, h, qq - s0, p] = float("-inf")
            A = lg.softmax(-1)
            z = (A @ vh).transpose(1, 2)  # [K, Ts, H, hd]
            for b, h, p, val in E.get("head", {}).get(L, []):
                z[b, p - s0, h] = 0.0 if val is None else val.to(d, z.dtype)
            if probe is not None:
                z = z + probe["z"][L]
            if capture:
                for key, val in (("x", x[0]), ("q", q[0]), ("k", k[0]), ("v", v[0]), ("logits", lg[0]), ("A", A[0]),
                                 ("z", z[0])):
                    st[key].append(val.detach().clone())
            x = x + f(at.o_proj, z.reshape(K, Ts, H * hd))
            hm = self._acts(layer, x)
            for b, p, j, delta in E.get("read", {}).get(L, []):
                hm[b, p - s0, j] = self._acts(layer, (x[b, p - s0] - delta.to(d))[None], [j])[0, 0]
            for b, p, j, val in E.get("neuron", {}).get(L, []):
                hm[b, p - s0, j] = float(val)
            if probe is not None:
                hm = hm + probe["h"][L]
            for w, (Lw, p, j) in enumerate(watch):
                if Lw == L:
                    seen[:, w] = hm[:, p - s0, j].cpu()
            if capture:
                st["mid"].append(x[0].detach().clone())
                st["h"].append(hm[0].detach().clone())
            x = x + f(layer.mlp.down_proj, hm)
        hn = f(core.model.norm, x[:, -1])
        con = None
        if self.answer is not None:  # the answer's and the alternatives' logits in float32 (the normalizer cancels)
            key = (tuple(self.answer), tuple(self.alternatives))
            if getattr(self, "_rows_key", None) != key:
                W = weight(core.lm_head)
                self._rows = W[torch.as_tensor(list(self.answer) + list(self.alternatives), device=W.device)].float()
                self._rows_key = key
            z = hn.to(self._rows.device) @ self._rows.T
            na = len(self.answer)
            con = ((torch.logsumexp(z[:, :na], -1) - torch.logsumexp(z[:, na:], -1)) / math.log(2.0)).cpu()
        lp = torch.log_softmax(f(core.lm_head, hn), -1).cpu() if capture or self.answer is None else None
        return {"lp": lp, "contrast": con, "states": st, "watch": seen}

    def _acts(self, layer, x: torch.Tensor, js: list[int] | None = None) -> torch.Tensor:
        """Neurons from the stream after attention x [..., d]: act(gate(norm x)) * up(norm x), float32 (js: only those,
        from the weights' rows)."""
        if js is None:
            n = self._f(layer.post_attention_layernorm, x)
            return layer.mlp.act_fn(self._f(layer.mlp.gate_proj, n)) * self._f(layer.mlp.up_proj, n)
        ln = layer.post_attention_layernorm
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon) * ln.weight.float().to(x.device)
        wg, wu = weight(layer.mlp.gate_proj)[js].float().to(x.device), weight(layer.mlp.up_proj)[js].float().to(x.device)
        return layer.mlp.act_fn(n @ wg.T) * (n @ wu.T)

    # -- local measurements on captured states ---------------------------------------------------------------------

    def gate_up(self, st: dict, L: int, pos: int, j: int) -> tuple[float, float]:
        """A neuron's act(gate) and up at pos (its value is their product)."""
        layer = self.layers[L]
        ln, x = layer.post_attention_layernorm, st["mid"][L][pos].to(self._d(L))
        n = x * torch.rsqrt(x.pow(2).mean() + ln.variance_epsilon) * ln.weight.float()
        return (float(layer.mlp.act_fn(n @ weight(layer.mlp.gate_proj)[j].float())),
                float(n @ weight(layer.mlp.up_proj)[j].float()))

    def head_write(self, st: dict, L: int, h: int, pos: int) -> torch.Tensor:
        """A head's output at pos as written into the stream [d] (o_proj's columns of that head times its output)."""
        Wo = weight(self.layers[L].self_attn.o_proj)[:, h * self.hd:(h + 1) * self.hd].float()
        return Wo @ st["z"][L][pos, h].to(Wo.device)

    def neuron_writes(self, st: dict, L: int, pos: int, js=None) -> torch.Tensor:
        """Layer L's neurons' writes at pos [n, d] (value times down_proj column; js: those, else every one)."""
        W = weight(self.layers[L].mlp.down_proj).float()
        hv = st["h"][L][pos].to(W.device)
        if js is not None:
            W, hv = W[:, js], hv[js]
        return (W * hv).T

    @torch.no_grad()
    def attention_after(self, st: dict, c: int, h: int, q: int, p: int, deltas: torch.Tensor, side: str = "key",
                        delta_out: bool = False, full: bool = False):
        """Head (c, h)'s attention from q to p with each row of deltas [n, d] (residual vectors) taken out of its key input
        at p (side "key") or its query input at q ("query"), exact [n] (full: from q to every position, [n, q + 1]);
        delta_out: also the change in that key or query after norm and rotary [n, hd] (Runner's dk / dq edit for the
        same cut)."""
        layer = self.layers[c]
        at, ln, d = layer.self_attn, layer.input_layernorm, self._d(c)
        pos = p if side == "key" else q
        x = st["x"][c][pos].to(d)
        D = x.shape[-1]
        deltas = deltas.to(d).float()
        g = ln.weight.float()
        if side == "key":
            kv = h // (self.H // self.KV)
            W, nrm = weight(at.k_proj)[kv * self.hd:(kv + 1) * self.hd].float(), at.k_norm
        else:
            W, nrm = weight(at.q_proj)[h * self.hd:(h + 1) * self.hd].float(), at.q_norm
        r2 = x.pow(2).mean() + ln.variance_epsilon
        r2n = r2 + (deltas.pow(2).sum(-1) - 2 * deltas @ x) / D  # each cut stream's mean square (+ eps)
        pre = ((g * x) @ W.T - (deltas * g) @ W.T) / r2n.sqrt()[:, None]  # [n, hd]
        pre = pre * torch.rsqrt(pre.pow(2).mean(-1, keepdim=True) + nrm.variance_epsilon) * nrm.weight.float()
        cs, sn = st["cos"][pos].to(d), st["sin"][pos].to(d)
        new = pre * cs + rotate_half(pre) * sn  # [n, hd]
        if side == "key":
            l0 = st["logits"][c][h, q, :q + 1].to(d)
            lp = (st["q"][c][h, q].to(d) @ new.T) * at.scaling  # [n]
            row = l0[None].expand(len(new), -1).clone()
            row[:, p] = lp
            base = st["k"][c][kv, p].to(d)
        else:
            row = (new @ st["k"][c][h // (self.H // self.KV), :q + 1].to(d).T) * at.scaling  # [n, q + 1]
            base = st["q"][c][h, q].to(d)
        a = row.softmax(-1) if full else row.softmax(-1)[:, p]
        return (a, new - base) if delta_out else a

    @torch.no_grad()
    def output_after(self, st: dict, c: int, h: int, q: int, p: int, deltas: torch.Tensor, side: str):
        """Head (c, h)'s output change at q as written into the stream [n, d], with each row of deltas [n, d] taken out
        of its key or value input at p or its query input at q (side "key", "value" or "query"), the rest of the head as
        in st, exact; and each cut's Runner edit (dk, dv or dq) [n, hd]."""
        d, kv = self._d(c), h // (self.H // self.KV)
        a0, V = st["A"][c][h, q, :q + 1].to(d), st["v"][c][kv, :q + 1].to(d)
        if side == "value":
            edit = self.value_after(st, c, h, p, deltas)
            dz = a0[p] * edit
        else:
            a, edit = self.attention_after(st, c, h, q, p, deltas, side=side, delta_out=True, full=True)
            dz = (a - a0) @ V
        Wo = weight(self.layers[c].self_attn.o_proj)[:, h * self.hd:(h + 1) * self.hd].float()
        return dz.to(Wo.device) @ Wo.T, edit

    @torch.no_grad()
    def value_after(self, st: dict, c: int, h: int, p: int, deltas: torch.Tensor) -> torch.Tensor:
        """The change in head (c, h)'s value at p with each row of deltas [n, d] taken out of its value input there,
        exact through the input norm [n, hd] (Runner's dv edit for the same cut)."""
        layer = self.layers[c]
        at, ln, d = layer.self_attn, layer.input_layernorm, self._d(c)
        x, deltas = st["x"][c][p].to(d), deltas.to(d).float()
        kv = h // (self.H // self.KV)
        W, g = weight(at.v_proj)[kv * self.hd:(kv + 1) * self.hd].float(), ln.weight.float()
        r2 = x.pow(2).mean() + ln.variance_epsilon
        r2n = r2 + (deltas.pow(2).sum(-1) - 2 * deltas @ x) / x.shape[-1]
        new = ((g * x) @ W.T - (deltas * g) @ W.T) / r2n.sqrt()[:, None]
        if at.v_proj.bias is not None:
            new = new + at.v_proj.bias[kv * self.hd:(kv + 1) * self.hd].float()
        return new - st["v"][c][kv, p].to(d)

    @torch.no_grad()
    def readers_after(self, st: dict, c: int, h: int, pos: int, instead: torch.Tensor | None = None) -> dict:
        """Every neuron at pos in layers c.. recomputed with head (c, h)'s output at pos taken out of its MLP input (or
        replaced by instead, another run's write): {layer: [d_mlp] the change (new - base)}."""
        out = {}
        for L in range(c, self.L):
            d = self._d(L)
            delta = self.head_write(st, c, h, pos).to(d) - (0 if instead is None else instead.to(d))
            mid = st["mid"][L][pos].to(d)
            out[L] = (self._acts(self.layers[L], (mid - delta)[None])[0] - st["h"][L][pos].to(d)).cpu()
        return out


# -- a head's route, measured end to end -------------------------------------------------------------------------------

@torch.no_grad()
def route_story(rn: Runner, st: dict, lost, c: int, h: int, q: int, p: int, writers: int = 16, readers: int = 16,
                other: dict | None = None) -> dict:
    """Head (c, h) reading position p at q, on a prompt whose clean states are st; lost(sets) gives each edit set's
    share of the gap lost (or any measured score). Measured:
      knock    the head unable to see p;
      key      every neuron at p in earlier layers and every earlier head's output there taken out of its key input at p
               (the attention q -> p after, exact), the writers largest by change cut from that head's key alone in
               full runs; query: the same at q for its query input; value: the same at p for its value input, ranked by
               the head's output change at q (what the head moves);
      *_group  the picked neurons of a side cut together (their summed write taken out), and the picked heads together;
      readers  every neuron at q in layers c.. recomputed with the head's output taken out (exact), the readers largest
               by change zeroed alone (own_lost) and cut from the head alone (cut_lost); readers_group: all cut together.
    other (the corrupted run's states of a patching pair): every cut a patch instead, the unit's write (or value)
    replaced by its write there."""
    def nw(L, pos):
        w = rn.neuron_writes(st, L, pos)
        return w if other is None else w - rn.neuron_writes(other, L, pos).to(w.device)

    def hw(L, g, pos):
        w = rn.head_write(st, L, g, pos)
        return w if other is None else w - rn.head_write(other, L, g, pos).to(w.device)
    a0 = float(st["A"][c][h, q, p])
    Wo = weight(rn.layers[c].self_attn.o_proj)[:, h * rn.hd:(h + 1) * rn.hd].float()
    out = {"layer": c, "head": h, "q": q, "p": p, "attention": a0, "knock": lost([{"knock": [(c, h, q, p)]}])[0]}
    for side, pos in (("key", p), ("query", q), ("value", p)):
        def after(deltas):  # (each delta's size of change and the edit for its cut)
            if side == "value":
                dv = rn.value_after(st, c, h, p, deltas)
                return (st["A"][c][h, q, p] * dv.to(Wo.device) @ Wo.T).norm(dim=-1).cpu(), dv
            a, dl = rn.attention_after(st, c, h, q, p, deltas, side=side, delta_out=True)
            return (a - a0).cpu(), dl
        cand = []  # (|change|, change, unit, edit delta, the write cut)
        for L in range(c):
            w = nw(L, pos)
            ch, dl = after(w)
            for jj in torch.topk(ch.abs(), min(writers, len(ch))).indices.tolist():
                cand.append((abs(float(ch[jj])), float(ch[jj]), ("neuron", L, jj, float(st["h"][L][pos, jj])), dl[jj], w[jj]))
            if side != "query":
                w = torch.stack([hw(L, g, pos) for g in range(rn.H)])
                ch, dl = after(w)
                cand += [(abs(float(ch[g])), float(ch[g]), ("head", L, g), dl[g], w[g]) for g in range(rn.H)]
        cand.sort(key=lambda z: -z[0])
        neu = [z for z in cand if z[2][0] == "neuron"][:writers]
        hed = [z for z in cand if z[2][0] == "head"][:max(1, writers // 4)]
        kind = {"key": "dk", "query": "dq", "value": "dv"}[side]
        groups = [g for g in (neu, hed) if g]
        gch, gdl = after(torch.stack([torch.stack([z[4] for z in g]).sum(0) for g in groups])) if groups else ([], [])
        sh = lost([{kind: [(c, h, pos, z[3])]} for z in neu + hed] + [{kind: [(c, h, pos, d)]} for d in gdl])
        key = "output_change" if side == "value" else "attention_change"
        out[side] = [{"unit": list(z[2]), key: z[1], "lost": s_} for z, s_ in zip(neu + hed, sh)]
        out[side + "_group"] = {name: {key: float(gch[k]), "lost": sh[len(neu) + len(hed) + k], "units": len(g)}
                                for k, (name, g) in enumerate((n, g) for n, g in (("neurons", neu), ("heads", hed)) if g)}
    ch = rn.readers_after(st, c, h, q, None if other is None else rn.head_write(other, c, h, q))
    flat = sorted(((abs(float(v[jj])), float(v[jj]), L, jj) for L, v in ch.items()
                   for jj in torch.topk(v.abs(), min(readers, len(v))).indices.tolist()), key=lambda z: -z[0])[:readers]
    delta = hw(c, h, q)
    own = lost([{"neuron": [(L, q, jj, 0.0 if other is None else float(other["h"][L][q, jj]))]} for _, _, L, jj in flat])
    cut = lost([{"read": [(L, q, jj, delta)]} for _, _, L, jj in flat] + [{"read": [(L, q, jj, delta) for _, _, L, jj in flat]}])
    out["readers"] = [{"unit": ["neuron", L, jj, float(st["h"][L][q, jj])], "value_change": v, "own_lost": o, "cut_lost": k}
                      for (_, v, L, jj), o, k in zip(flat, own, cut)]
    out["readers_group"] = {"cut_lost": cut[-1], "units": len(flat)}
    return out


@torch.no_grad()
def carriers(rn: Runner, st: dict, lost, q: int, p: int, min_attention: float = 0.01) -> list[dict]:
    """Which heads move position p's information to q: every head attending q -> p at least min_attention made unable
    to see p alone (measured), largest first."""
    heads = [(L, h) for L in range(rn.L) for h in range(rn.H) if float(st["A"][L][h, q, p]) >= min_attention]
    sh = lost([{"knock": [(L, h, q, p)]} for L, h in heads])
    return sorted(({"layer": L, "head": h, "attention": float(st["A"][L][h, q, p]), "knock": s}
                   for (L, h), s in zip(heads, sh)), key=lambda r: -abs(r["knock"]))


def load(name: str, device: str | None = None, dtype: str = "float32"):
    """An HF causal LM: float32 on one card, or bfloat16 split over every card (device_map auto), no weight gradients."""
    import transformers
    if dtype == "bfloat16":
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
    else:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev)
        if dev == "cpu":
            torch.set_num_threads(8)
    return model.eval().requires_grad_(False)


def by_adapter(model, items: list, adapter_of):
    """(model, item) for each item with its adapter on (adapter_of(item): a peft adapter's path or hub id, None or "" the
    model itself), grouped by adapter: each adapter loaded once and unloaded after its group."""
    def key(it):
        return adapter_of(it) or ""
    for ad in sorted({key(it) for it in items}):
        group = [it for it in items if key(it) == ad]
        if not ad:
            for it in group:
                yield model, it
            continue
        from peft import PeftModel
        pm = PeftModel.from_pretrained(model, ad).eval().requires_grad_(False)
        for it in group:
            yield pm, it
        model = pm.unload()


@torch.no_grad()
def carrier_curve(rn: Runner, lost, q: int, p: int, ranked: list) -> dict:
    """Heads ranked by their own knockout (carriers' order) made unable to see p from q together, the first k of them
    for k on a log scale up to all: the share lost per k (when single heads each move little, how many it takes)."""
    n = len(ranked)
    ks = sorted({min(n, int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(n, 1))) + 2)} | {n}) if n else []
    sh = lost([{"knock": [(r["layer"], r["head"], q, p) for r in ranked[:k]]} for k in ks]) if ks else []
    return {"k": ks, "lost": sh}


def finder_routes(rn: Runner, rec: dict, writers: int = 16, readers: int = 16, min_attention: float = 0.01) -> dict:
    """A circuit_finder record's mover heads as routes (route_story, cuts zeroing and patching from the corrupted run, and
    carriers), each from the answer position to the source it looks at most from the first changed token on."""
    rn.answer = [int(t) for t in (rec.get("answer_tokens") or [rec["answer"]])]  # (the word contrast's token sets)
    rn.alternatives = [int(t) for t in rec["alternatives"]]
    ids, bad = rec["ids"], rec["corrupted"]
    clean, corrupt = rn.run(ids, [{}], capture=True), rn.run(bad, [{}], capture=True)
    c, cb = rn.run(ids, [{}])["contrast"][0], rn.run(bad, [{}])["contrast"][0]  # (uncut, at the cut runs' batch size)
    gap = c - cb
    st, sb = clean["states"], corrupt["states"]

    def lost(sets):
        return [(c - x) / gap for x in rn.run(ids, sets)["contrast"]]
    first = rec.get("first_diff", next(i for i, (x, y) in enumerate(zip(ids, bad)) if x != y))
    out = {"id": rec.get("id"), "contrast": {"clean": c, "corrupt": cb}, "gap": gap, "routes": []}
    for r in rec.get("routes", []):
        L, h, q = r["layer"], r["head"], r["pos"]
        att = st["A"][L][h, q]
        p = first + int(att[first:q + 1].argmax())
        story = {"zero": route_story(rn, st, lost, L, h, q, p, writers, readers),
                 "patch": route_story(rn, st, lost, L, h, q, p, writers, readers, other=sb),
                 "carriers": carriers(rn, st, lost, q, p, min_attention)}
        out["routes"].append(story)
    return out


def named_routes(rn: Runner, rec: dict, heads: list, neurons: list, pos: int | None = None, writers: int = 16,
                 readers: int = 16) -> dict:
    """Named heads [(layer, head)] on a circuit_finder record, each read from position pos (default: the first changed
    token) at the answer position: route_story by zeroing and by patching, and each named neuron [(layer, index)] below
    the head at pos taken out of its key input there alone (the attention after, exact, and the share lost with that cut
    in a full run, both ways), whether the story's measured key writers include it; and every head attending from the
    answer position to pos made unable to see it alone (carriers) and the first k of them together (carrier_curve)."""
    rn.answer = [int(t) for t in (rec.get("answer_tokens") or [rec["answer"]])]
    rn.alternatives = [int(t) for t in rec["alternatives"]]
    ids, bad = rec["ids"], rec["corrupted"]
    clean, corrupt = rn.run(ids, [{}], capture=True), rn.run(bad, [{}], capture=True)
    c, cb = rn.run(ids, [{}])["contrast"][0], rn.run(bad, [{}])["contrast"][0]  # (uncut, at the cut runs' batch size)
    st, sb = clean["states"], corrupt["states"]
    q = len(ids) - 1
    p = rec.get("first_diff", next(i for i, (x, y) in enumerate(zip(ids, bad)) if x != y)) if pos is None else pos

    def lost(sets):
        return [(c - x) / (c - cb) for x in rn.run(ids, sets)["contrast"]]
    out = {"id": rec.get("id"), "q": q, "p": p, "contrast": {"clean": c, "corrupt": cb}, "gap": c - cb, "heads": {}}
    for L, h in heads:
        a0 = float(st["A"][L][h, q, p])
        rec_h = {"attention": a0, "top_sources": [[int(i), float(w)] for w, i in zip(*st["A"][L][h, q].topk(3))]}
        for mode, other in (("zero", None), ("patch", sb)):
            story = route_story(rn, st, lost, L, h, q, p, writers, readers, other=other)
            got = {tuple(u["unit"][1:3]) for u in story["key"] if u["unit"][0] == "neuron"}
            named = []
            for Ln, j in neurons:
                if Ln >= L:
                    named.append({"neuron": [Ln, j], "below_head": False})
                    continue
                w = rn.neuron_writes(st, Ln, p, [j])
                if other is not None:
                    w = w - rn.neuron_writes(other, Ln, p, [j]).to(w.device)
                a, dk = rn.attention_after(st, L, h, q, p, w, side="key", delta_out=True)
                named.append({"neuron": [Ln, j], "value": float(st["h"][Ln][p, j]), "attention_after": float(a[0]),
                              "attention_change": float(a[0]) - a0, "lost": lost([{"dk": [(L, h, p, dk[0])]}])[0],
                              "among_measured_key_writers": (Ln, j) in got})
            story["named_key_writers"] = named
            rec_h[mode] = story
        out["heads"][f"L{L}H{h}"] = rec_h
    car = carriers(rn, st, lost, q, p)
    out["carriers"] = car
    out["carrier_curve"] = carrier_curve(rn, lost, q, p, car)
    return out


def main():
    import argparse
    import json
    import time
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("finder", "heads"))
    ap.add_argument("model")
    ap.add_argument("records", type=Path, help="a circuit_finder record or a directory of them")
    ap.add_argument("out", type=Path)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--heads", default="", help="heads: named heads, L26H17,L23H15,...")
    ap.add_argument("--neurons", default="", help="heads: named neurons checked as key writers, L1:2039,L0:9761,...")
    ap.add_argument("--pos", type=int, default=None, help="heads: the source position (default: the first changed token)")
    ap.add_argument("--adapter", default=None, help="a peft adapter (an organism) on the model; else each record's own "
                    "(its \"adapter\")")
    a = ap.parse_args()
    model = load(a.model, a.device, a.dtype)
    files = sorted(a.records.glob("*.json")) if a.records.is_dir() else [a.records]
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [(f, json.loads(f.read_text())) for f in files if not (a.out / f.name).exists()]
    rn, on = None, None
    for m, (f, rec) in by_adapter(model, todo, lambda it: a.adapter or it[1].get("adapter")):
        dst = a.out / f.name
        if m is not on:  # (a runner per model: its cached keys and values belong to that model)
            rn, on = Runner(m), m
        t0 = time.time()
        if a.cmd == "heads":
            hs = [tuple(int(x) for x in u[1:].split("H")) for u in a.heads.split(",") if u]
            ns = [tuple(int(x) for x in u[1:].split(":")) for u in a.neurons.split(",") if u]
            got = named_routes(rn, rec, hs, ns, a.pos)
        else:
            got = finder_routes(rn, rec)
        got["adapter"] = a.adapter or rec.get("adapter")
        import gc  # (each record's states and float32 weight copies freed before the next)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        got["seconds"] = round(time.time() - t0, 2)
        dst.write_text(json.dumps(got))
        print(f"{f.stem}: {len(got.get('routes', got.get('heads', [])))} routes, {got['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
