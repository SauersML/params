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
  read    (layer, pos, index, delta)   the neuron recomputed from its MLP input (the stream after attention) less delta:
                                       a head's output, say, cut from that neuron alone.
It returns the contrast of each set (log2 p(answer) - log2 of the alternatives' summed p at the last position) and with
capture the first set's states. Each module runs in the model's dtype on its card, everything between in float32.

Local measurements on captured states, exact (norms, gate and up recomputed in float32), no forward:
  attention_after: a head's attention from q to p with each of some residual vectors (neurons' writes, another head's
    output) taken out of its key input at p or its query input at q;
  readers_after: every neuron at a position in later layers recomputed with one head's output there taken out of its
    input (the head -> neuron edges' local size).
"""
from __future__ import annotations

import math

import torch

ROWS = 64


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

    def _d(self, L: int) -> torch.device:
        return self.layers[L].input_layernorm.weight.device

    @staticmethod
    def _f(mod, t: torch.Tensor) -> torch.Tensor:
        w = next(mod.parameters())
        return mod(t.to(w.device, w.dtype)).float()

    def contrast(self, lp: torch.Tensor) -> torch.Tensor:
        """log2 p(answer) - log2 sum p(alternatives) from log probabilities [..., V]."""
        return (torch.logsumexp(lp[..., self.answer], -1) - torch.logsumexp(lp[..., self.alternatives], -1)) / math.log(2.0)

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
    def run(self, ids: list[int], sets: list[dict], capture: bool = False) -> dict:
        """The contrast of each set (and "lp" [K, V]); capture: the first set's states per layer ("x" the stream entering
        the layer [T, d], "q" [H, T, hd] and "k" [KV, T, hd] after norm and rotary, "v" [KV, T, hd], "logits" and "A"
        [H, T, T], "z" the head outputs [T, H, hd], "mid" the stream after attention [T, d], "h" the neurons [T, d_mlp]),
        "cos"/"sin" [T, hd]."""
        out = {"contrast": [], "lp": []}
        for s0 in range(0, max(len(sets), 1), self.rows):
            part = sets[s0:s0 + self.rows] or [{}]
            r = self._forward(ids, part, capture and s0 == 0)
            out["contrast"] += self.contrast(r["lp"]).tolist() if self.answer is not None else []
            out["lp"].append(r["lp"])
            if capture and s0 == 0:
                out["states"] = r["states"]
        out["lp"] = torch.cat(out["lp"])
        return out

    def _forward(self, ids: list[int], sets: list[dict], capture: bool) -> dict:
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        core, H, KV, hd, f = self.core, self.H, self.KV, self.hd, self._f
        K, T = len(sets), len(ids)
        E = self._compile(sets)
        x = core.model.embed_tokens(torch.tensor([ids], device=self.dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(T, device=self.dev)[None])
        cos, sin = cos[0].float(), sin[0].float()  # [T, hd]
        st = {k: [] for k in ("x", "q", "k", "v", "logits", "A", "z", "mid", "h")}
        st.update(cos=cos, sin=sin)
        for L, layer in enumerate(self.layers):
            d = self._d(L)
            x, cs, sn = x.to(d), cos.to(d), sin.to(d)
            at = layer.self_attn
            hn = f(layer.input_layernorm, x)
            q = f(at.q_norm, f(at.q_proj, hn).view(K, T, H, hd)).transpose(1, 2)
            k = f(at.k_norm, f(at.k_proj, hn).view(K, T, KV, hd)).transpose(1, 2)
            v = f(at.v_proj, hn).view(K, T, KV, hd).transpose(1, 2)
            q, k = q * cs + rotate_half(q) * sn, k * cs + rotate_half(k) * sn
            kh, vh = repeat_kv(k, H // KV), repeat_kv(v, H // KV)
            for kind, t in (("dq", q), ("dk", kh)):
                for b, h, p, delta in E.get(kind, {}).get(L, []):
                    t[b, h, p] += delta.to(d, t.dtype)
            lg = (q @ kh.transpose(-1, -2)) * at.scaling
            lg = lg.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=d).tril(), float("-inf"))
            for b, h, qq, p in E.get("knock", {}).get(L, []):
                lg[b, h, qq, p] = float("-inf")
            A = lg.softmax(-1)
            z = (A @ vh).transpose(1, 2)  # [K, T, H, hd]
            for b, h, p, val in E.get("head", {}).get(L, []):
                z[b, p, h] = 0.0 if val is None else val.to(d, z.dtype)
            if capture:
                for key, val in (("x", x[0]), ("q", q[0]), ("k", k[0]), ("v", v[0]), ("logits", lg[0]), ("A", A[0]),
                                 ("z", z[0])):
                    st[key].append(val.detach().clone())
            x = x + f(at.o_proj, z.reshape(K, T, H * hd))
            hm = self._acts(layer, x)
            for b, p, j, delta in E.get("read", {}).get(L, []):
                hm[b, p, j] = self._acts(layer, (x[b, p] - delta.to(d))[None], [j])[0, 0]
            for b, p, j, val in E.get("neuron", {}).get(L, []):
                hm[b, p, j] = float(val)
            if capture:
                st["mid"].append(x[0].detach().clone())
                st["h"].append(hm[0].detach().clone())
            x = x + f(layer.mlp.down_proj, hm)
        lp = torch.log_softmax(f(core.lm_head, f(core.model.norm, x[:, -1])), -1).cpu()
        return {"lp": lp, "states": st}

    def _acts(self, layer, x: torch.Tensor, js: list[int] | None = None) -> torch.Tensor:
        """Neurons from the stream after attention x [..., d]: act(gate(norm x)) * up(norm x), float32 (js: only those,
        from the weights' rows)."""
        if js is None:
            n = self._f(layer.post_attention_layernorm, x)
            return layer.mlp.act_fn(self._f(layer.mlp.gate_proj, n)) * self._f(layer.mlp.up_proj, n)
        ln = layer.post_attention_layernorm
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon) * ln.weight.float().to(x.device)
        wg, wu = layer.mlp.gate_proj.weight[js].float().to(x.device), layer.mlp.up_proj.weight[js].float().to(x.device)
        return layer.mlp.act_fn(n @ wg.T) * (n @ wu.T)

    # -- local measurements on captured states ---------------------------------------------------------------------

    def head_write(self, st: dict, L: int, h: int, pos: int) -> torch.Tensor:
        """A head's output at pos as written into the stream [d] (o_proj's columns of that head times its output)."""
        Wo = self.layers[L].self_attn.o_proj.weight[:, h * self.hd:(h + 1) * self.hd].float()
        return Wo @ st["z"][L][pos, h].to(Wo.device)

    def neuron_writes(self, st: dict, L: int, pos: int, js=None) -> torch.Tensor:
        """Layer L's neurons' writes at pos [n, d] (value times down_proj column; js: those, else every one)."""
        W = self.layers[L].mlp.down_proj.weight.float()
        hv = st["h"][L][pos].to(W.device)
        if js is not None:
            W, hv = W[:, js], hv[js]
        return (W * hv).T

    @torch.no_grad()
    def attention_after(self, st: dict, c: int, h: int, q: int, p: int, deltas: torch.Tensor, side: str = "key",
                        delta_out: bool = False):
        """Head (c, h)'s attention from q to p with each row of deltas [n, d] (residual vectors) taken out of its key input
        at p (side "key") or its query input at q ("query"), exact [n]; delta_out: also the change in that key or query
        after norm and rotary [n, hd] (Runner's dk / dq edit for the same cut)."""
        layer = self.layers[c]
        at, ln, d = layer.self_attn, layer.input_layernorm, self._d(c)
        pos = p if side == "key" else q
        x = st["x"][c][pos].to(d)
        D = x.shape[-1]
        deltas = deltas.to(d).float()
        g = ln.weight.float()
        if side == "key":
            kv = h // (self.H // self.KV)
            W, nrm = at.k_proj.weight[kv * self.hd:(kv + 1) * self.hd].float(), at.k_norm
        else:
            W, nrm = at.q_proj.weight[h * self.hd:(h + 1) * self.hd].float(), at.q_norm
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
        a = row.softmax(-1)[:, p]
        return (a, new - base) if delta_out else a

    @torch.no_grad()
    def readers_after(self, st: dict, c: int, h: int, pos: int) -> dict:
        """Every neuron at pos in layers c.. recomputed with head (c, h)'s output at pos taken out of its MLP input:
        {layer: [d_mlp] the change (new - base)}."""
        out = {}
        for L in range(c, self.L):
            d = self._d(L)
            delta = self.head_write(st, c, h, pos).to(d)
            mid = st["mid"][L][pos].to(d)
            out[L] = (self._acts(self.layers[L], (mid - delta)[None])[0] - st["h"][L][pos].to(d)).cpu()
        return out
