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
    def run(self, ids: list[int], sets: list[dict], capture: bool = False, watch: list | None = None) -> dict:
        """The contrast of each set (and "lp" [K, V]); capture: the first set's states per layer ("x" the stream entering
        the layer [T, d], "q" [H, T, hd] and "k" [KV, T, hd] after norm and rotary, "v" [KV, T, hd], "logits" and "A"
        [H, T, T], "z" the head outputs [T, H, hd], "mid" the stream after attention [T, d], "h" the neurons [T, d_mlp]),
        "cos"/"sin" [T, hd]; watch: neurons (layer, pos, index) whose values every set reports ("watch" [K, n])."""
        out = {"contrast": [], "lp": [], "watch": []}
        for s0 in range(0, max(len(sets), 1), self.rows):
            part = sets[s0:s0 + self.rows] or [{}]
            r = self._forward(ids, part, capture and s0 == 0, watch or [])
            out["contrast"] += self.contrast(r["lp"]).tolist() if self.answer is not None else []
            out["lp"].append(r["lp"])
            out["watch"].append(r["watch"])
            if capture and s0 == 0:
                out["states"] = r["states"]
        out["lp"], out["watch"] = torch.cat(out["lp"]), torch.cat(out["watch"])
        return out

    def _forward(self, ids: list[int], sets: list[dict], capture: bool, watch: list) -> dict:
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        core, H, KV, hd, f = self.core, self.H, self.KV, self.hd, self._f
        K, T = len(sets), len(ids)
        E = self._compile(sets)
        x = core.model.embed_tokens(torch.tensor([ids], device=self.dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(T, device=self.dev)[None])
        cos, sin = cos[0].float(), sin[0].float()  # [T, hd]
        st = {k: [] for k in ("x", "q", "k", "v", "logits", "A", "z", "mid", "h")}
        st.update(cos=cos, sin=sin)
        seen = torch.zeros(K, len(watch))
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
            for kind, t in (("dq", q), ("dk", kh), ("dv", vh)):
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
            for w, (Lw, p, j) in enumerate(watch):
                if Lw == L:
                    seen[:, w] = hm[:, p, j].cpu()
            if capture:
                st["mid"].append(x[0].detach().clone())
                st["h"].append(hm[0].detach().clone())
            x = x + f(layer.mlp.down_proj, hm)
        lp = torch.log_softmax(f(core.lm_head, f(core.model.norm, x[:, -1])), -1).cpu()
        return {"lp": lp, "states": st, "watch": seen}

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

    def gate_up(self, st: dict, L: int, pos: int, j: int) -> tuple[float, float]:
        """A neuron's act(gate) and up at pos (its value is their product)."""
        layer = self.layers[L]
        ln, x = layer.post_attention_layernorm, st["mid"][L][pos].to(self._d(L))
        n = x * torch.rsqrt(x.pow(2).mean() + ln.variance_epsilon) * ln.weight.float()
        return (float(layer.mlp.act_fn(n @ layer.mlp.gate_proj.weight[j].float())),
                float(n @ layer.mlp.up_proj.weight[j].float()))

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
    def value_after(self, st: dict, c: int, h: int, p: int, deltas: torch.Tensor) -> torch.Tensor:
        """The change in head (c, h)'s value at p with each row of deltas [n, d] taken out of its value input there,
        exact through the input norm [n, hd] (Runner's dv edit for the same cut)."""
        layer = self.layers[c]
        at, ln, d = layer.self_attn, layer.input_layernorm, self._d(c)
        x, deltas = st["x"][c][p].to(d), deltas.to(d).float()
        kv = h // (self.H // self.KV)
        W, g = at.v_proj.weight[kv * self.hd:(kv + 1) * self.hd].float(), ln.weight.float()
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
    share of the gap lost (or any measured score). Measured: the head unable to see p (knock); every neuron at p in
    earlier layers and every earlier head's output there taken out of its key input at p, and every neuron at q in
    earlier layers out of its query input at q (the attention q -> p after, exact), the writers largest by change cut
    from that head's key or query alone in full runs; the same units out of its value input at p (what the head moves:
    its output change at q, exact), the largest cut from its value alone in full runs; every neuron at q in layers c.. recomputed with the head's output
    taken out (exact), the readers largest by change zeroed alone (own_lost) and cut from the head alone (cut_lost) in
    full runs. other (the
    corrupted run's states of a patching pair): every cut a patch instead, the unit's write (or value) replaced by its
    write there."""
    def nw(L, pos):
        w = rn.neuron_writes(st, L, pos)
        return w if other is None else w - rn.neuron_writes(other, L, pos).to(w.device)

    def hw(L, g, pos):
        w = rn.head_write(st, L, g, pos)
        return w if other is None else w - rn.head_write(other, L, g, pos).to(w.device)
    a0 = float(st["A"][c][h, q, p])
    out = {"layer": c, "head": h, "q": q, "p": p, "attention": a0, "knock": lost([{"knock": [(c, h, q, p)]}])[0]}
    for side, pos in (("key", p), ("query", q)):
        cand = []  # (|change|, change, unit, delta)
        for L in range(c):
            a, dl = rn.attention_after(st, c, h, q, p, nw(L, pos), side=side, delta_out=True)
            ch = (a - a0).cpu()
            for j in torch.topk(ch.abs(), min(writers, len(ch))).indices.tolist():
                cand.append((abs(float(ch[j])), float(ch[j]), ("neuron", L, j, float(st["h"][L][pos, j])), dl[j]))
            if side == "key":
                a, dl = rn.attention_after(st, c, h, q, p, torch.stack([hw(L, g, pos) for g in range(rn.H)]), side=side,
                                           delta_out=True)
                cand += [(abs(float(a[g] - a0)), float(a[g] - a0), ("head", L, g), dl[g]) for g in range(rn.H)]
        cand.sort(key=lambda z: -z[0])
        pick = [z for z in cand if z[2][0] == "neuron"][:writers] + [z for z in cand if z[2][0] == "head"][:max(1, writers // 4)]
        kind = "dk" if side == "key" else "dq"
        sh = lost([{kind: [(c, h, pos, z[3])]} for z in pick])
        out[side] = [{"unit": list(z[2]), "attention_change": z[1], "lost": s} for z, s in zip(pick, sh)]
    cand = []  # the value at p: what the head moves, ranked by the size of the head's output change at q
    Wo = rn.layers[c].self_attn.o_proj.weight[:, h * rn.hd:(h + 1) * rn.hd].float()
    a_qp = st["A"][c][h, q, p]
    for L in range(c):
        dv = rn.value_after(st, c, h, p, nw(L, p))
        size = (a_qp * dv.to(Wo.device) @ Wo.T).norm(dim=-1).cpu()
        for j in torch.topk(size, min(writers, len(size))).indices.tolist():
            cand.append((float(size[j]), ("neuron", L, j, float(st["h"][L][p, j])), dv[j]))
        dv = rn.value_after(st, c, h, p, torch.stack([hw(L, g, p) for g in range(rn.H)]))
        size = (a_qp * dv.to(Wo.device) @ Wo.T).norm(dim=-1).cpu()
        cand += [(float(size[g]), ("head", L, g), dv[g]) for g in range(rn.H)]
    cand.sort(key=lambda z: -z[0])
    pick = [z for z in cand if z[1][0] == "neuron"][:writers] + [z for z in cand if z[1][0] == "head"][:max(1, writers // 4)]
    sh = lost([{"dv": [(c, h, p, z[2])]} for z in pick])
    out["value"] = [{"unit": list(z[1]), "output_change": z[0], "lost": s} for z, s in zip(pick, sh)]
    ch = rn.readers_after(st, c, h, q, None if other is None else rn.head_write(other, c, h, q))
    flat = sorted(((abs(float(v[j])), float(v[j]), L, j) for L, v in ch.items()
                   for j in torch.topk(v.abs(), min(readers, len(v))).indices.tolist()), key=lambda z: -z[0])[:readers]
    delta = hw(c, h, q)
    own = lost([{"neuron": [(L, q, j, 0.0 if other is None else float(other["h"][L][q, j]))]} for _, _, L, j in flat])
    cut = lost([{"read": [(L, q, j, delta)]} for _, _, L, j in flat])
    out["readers"] = [{"unit": ["neuron", L, j, float(st["h"][L][q, j])], "value_change": v, "own_lost": o, "cut_lost": k}
                      for (_, v, L, j), o, k in zip(flat, own, cut)]
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


def finder_routes(rn: Runner, rec: dict, writers: int = 16, readers: int = 16, min_attention: float = 0.01) -> dict:
    """A circuit_finder record's mover heads as routes (route_story, cuts zeroing and patching from the corrupted run, and
    carriers), each from the answer position to the source it looks at most from the first changed token on."""
    rn.answer, rn.alternatives = [rec["answer"]], list(rec["alternatives"])
    ids, bad = rec["ids"], rec["corrupted"]
    clean, corrupt = rn.run(ids, [{}], capture=True), rn.run(bad, [{}], capture=True)
    c, cb = clean["contrast"][0], corrupt["contrast"][0]
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


def main():
    import argparse
    import json
    import time
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("finder",))
    ap.add_argument("model")
    ap.add_argument("records", type=Path, help="a circuit_finder record or a directory of them")
    ap.add_argument("out", type=Path)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    rn = Runner(load(a.model, a.device, a.dtype))
    files = sorted(a.records.glob("*.json")) if a.records.is_dir() else [a.records]
    a.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = a.out / f.name
        if dst.exists():
            continue
        t0 = time.time()
        got = finder_routes(rn, json.loads(f.read_text()))
        got["seconds"] = round(time.time() - t0, 2)
        dst.write_text(json.dumps(got))
        print(f"{f.stem}: {len(got['routes'])} routes, {got['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
