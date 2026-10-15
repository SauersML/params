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
within TOL of the way from every edge's to none's, and a halving pass over the rest (measured); each kept edge's own cut
measured last (its strength).

  python neuron_edges.py one MODEL TEXT.json OUT.json   (TEXT: ids, target, answer, alternatives, neurons)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

TOL = 0.1  # the contrast may move this share of the way from the run with every edge to the run with none
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
        self.dev = next(core.parameters()).device
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

    # -- the run ----------------------------------------------------------------------------------------------------

    def _mlp_act(self, layer, x: torch.Tensor) -> torch.Tensor:
        """The neurons' activations from a pre-norm stream x [..., d]: act(gate(norm x)) * up(norm x)."""
        n = layer.post_attention_layernorm(x)
        return layer.mlp.act_fn(layer.mlp.gate_proj(n)) * layer.mlp.up_proj(n)

    def _weak(self, L: int, h: torch.Tensor, js: torch.Tensor | None = None) -> torch.Tensor:
        """h with the activations under their weak threshold zeroed (js: the neurons h holds, else every one)."""
        if self.thr is None:
            return h
        t = self.thr[L] if js is None else self.thr[L][js]
        return h * (h.abs() >= t)

    def _plan(self, edges: list) -> dict:
        """Index tables of an edge list: resid cuts by the writer's layer, routes by layer and head."""
        hit = self._plans.get(id(edges))
        if hit is not None and hit[0] is edges:
            return hit[1]
        dev, resid, route = self.dev, {}, {}
        for e, ed in enumerate(edges):
            if len(ed) == 2:
                resid.setdefault(ed[0][0], []).append((e, self.at[ed[0]], self.at[ed[1]]))
            else:
                route.setdefault(ed[1], []).append((e, self.at[ed[0]], self.at[ed[2]]))
        t = lambda xs: tuple(torch.tensor([x[i] for x in xs], dtype=torch.long, device=dev) for i in range(3))  # noqa: E731
        plan = {"resid": {L: t(v) for L, v in resid.items()}, "route": {k: t(v) for k, v in route.items()}, "heads": {}}
        for c, h in route:
            plan["heads"].setdefault(c, []).append(h)
        self._plans = {id(edges): (edges, plan)}
        return plan

    def _run(self, cuts: torch.Tensor, edges: list, keep_states: bool = False, probe: list | None = None) -> dict:
        """The contrast of each row of cuts [K, len(edges)]. probe: per layer a tensor [T, d_mlp] added to the neurons'
        activations (zeros requiring grad: the contrast's gradient at them)."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        core, dev, H, KV, hd, T = self.core, self.dev, self.H, self.KV, self.hd, self.T
        K = max(cuts.shape[0], 1)
        cuts = cuts.to(dev).float() if len(edges) else torch.zeros(K, 0, device=dev)
        plan = self._plan(edges)
        x = core.model.embed_tokens(torch.tensor([self.ids], device=dev).expand(K, -1)).float()
        cos, sin = core.model.rotary_emb(x, torch.arange(T, device=dev)[None])
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        D = torch.zeros(K, len(self.N), x.shape[-1], device=dev)  # per circuit neuron, what its cut edges took away
        Wr = torch.zeros(K, len(self.N), x.shape[-1], device=dev)  # each circuit neuron's write in this run
        P = torch.tensor([n[1] for n in self.N], dtype=torch.long, device=dev)
        Jn = torch.tensor([n[2] for n in self.N], dtype=torch.long, device=dev)
        kk = torch.arange(K, device=dev)
        states = {"x": [], "A": [], "v": [], "h": []}
        for L, layer in enumerate(self.layers):
            at = layer.self_attn
            hn = layer.input_layernorm(x)
            q = at.q_norm(at.q_proj(hn).view(K, T, H, hd)).transpose(1, 2)
            k = at.k_norm(at.k_proj(hn).view(K, T, KV, hd)).transpose(1, 2)
            v = at.v_proj(hn).view(K, T, KV, hd).transpose(1, 2)  # [K, KV, T, hd]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            A = ((q @ repeat_kv(k, H // KV).transpose(-1, -2)) * at.scaling).masked_fill(~causal, float("-inf")).softmax(-1)
            z = A @ repeat_kv(v, H // KV)  # [K, H, T, hd]
            if keep_states:
                states["x"].append(x[0].detach())
                states["A"].append(A[0].detach())
                states["v"].append(v[0].detach())
            if plan["heads"].get(L):  # routes through this layer's heads: the head's output change at the reader
                Wo = weight(at.o_proj).view(-1, H, hd)
                for h in plan["heads"][L]:
                    es, wi, ri = plan["route"][(L, h)]
                    kv = h // (H // KV)
                    xw = x[:, P[wi]] - Wr[:, wi]  # [K, E, d]: the head's input at the writer, its write taken out
                    vw = at.v_proj(layer.input_layernorm(xw)).view(K, -1, KV, hd)[:, :, kv]
                    dz = A[:, h][kk[:, None], P[ri][None], P[wi][None]][..., None] * (v[:, kv][:, P[wi]] - vw)
                    D = D.index_add(1, ri, cuts[:, es, None] * (dz @ Wo[:, h].T))
            x = x + at.o_proj(z.transpose(1, 2).reshape(K, T, H * hd))
            hm = self._weak(L, self._mlp_act(layer, x))
            if self.background == "circuit":
                hm = hm * self.keep[L][None]
            rs = torch.tensor([i for i, n in enumerate(self.N) if n[0] == L], dtype=torch.long, device=dev)
            if len(rs) and len(edges):  # the circuit's neurons of this layer, each from its own input
                hr = self._mlp_act(layer, x[:, P[rs]] - D[:, rs]).gather(-1, Jn[rs][None, :, None].expand(K, -1, 1))[..., 0]
                hm = hm.index_put((kk[:, None], P[rs][None], Jn[rs][None]), self._weak(L, hr, Jn[rs]))
            if probe is not None:
                hm = hm + probe[L][None]
            if keep_states:
                states["h"].append(hm[0].detach())
            if len(rs):
                Wd = weight(layer.mlp.down_proj)
                Wr = Wr.index_add(1, rs, hm[kk[:, None], P[rs][None], Jn[rs][None]][..., None] * Wd[:, Jn[rs]].T[None])
            if L in plan["resid"]:
                es, wi, ri = plan["resid"][L]
                D = D.index_add(1, ri, cuts[:, es, None] * Wr[:, wi])
            x = x + layer.mlp.down_proj(hm)
        lp = torch.log_softmax(core.lm_head(core.model.norm(x[:, self.target])).float(), -1)
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
        """The base run's stream entering layer L's MLP [T, d]."""
        at = self.layers[L].self_attn
        z = self.base["A"][L] @ self.base["v"][L].repeat_interleave(self.H // self.KV, 0)  # [H, T, hd]
        return self.base["x"][L] + at.o_proj(z.transpose(0, 1).reshape(self.T, -1))

    def linear_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(resid scores, route scores) over resid_all and route_all: each edge's linear effect on the contrast, the
        contrast's gradient at the reader's activation times the reader's input Jacobian times what the edge brings
        (the contrast it supports: positive when its cut lowers the contrast)."""
        dev, base, H, KV, hd = self.dev, self.base, self.H, self.KV, self.hd
        probe = [torch.zeros(self.T, self.keep.shape[-1], device=dev, requires_grad=True) for _ in range(self.L)]
        with torch.enable_grad():
            self._run(torch.zeros(1, 0, dtype=torch.bool), [], probe=probe)["contrast"].sum().backward()
        grad = {n: float(probe[n[0]].grad[n[1], n[2]]) for n in self.N}
        J = {}  # each circuit neuron's activation's gradient with respect to its MLP input [d]
        for L in range(self.L):
            rs = [n for n in self.N if n[0] == L]
            if rs:
                with torch.enable_grad():
                    xin = self._mid_stream(L)[[n[1] for n in rs]].detach().requires_grad_()
                    h = self._mlp_act(self.layers[L], xin).gather(-1, torch.tensor([[n[2]] for n in rs], device=dev))[:, 0]
                    (g,) = torch.autograd.grad(h.sum(), xin)
                J.update(dict(zip(rs, g)))
        with torch.no_grad():
            writes = {n: base["h"][n[0]][n[1], n[2]] * weight(self.layers[n[0]].mlp.down_proj)[:, n[2]] for n in self.N}
            rs_s = torch.tensor([grad[r] * float(J[r] @ writes[w]) for w, r in self.resid_all])
            rt, vcache = [], {}
            for w, (c, h), r in self.route_all:
                layer, kv = self.layers[c], h // (H // KV)
                if (w, c) not in vcache:
                    xw = base["x"][c][w[1]] - writes[w]
                    vcache[(w, c)] = layer.self_attn.v_proj(layer.input_layernorm(xw)).view(KV, hd)
                dz = base["A"][c][h, r[1], w[1]] * (base["v"][c][kv, w[1]] - vcache[(w, c)][kv])
                dout = weight(layer.self_attn.o_proj)[:, h * hd:(h + 1) * hd] @ dz
                rt.append(grad[r] * float(J[r] @ dout))
        return rs_s, torch.tensor(rt)


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
    floor = c_all - TOL * (c_all - c_none)
    ok = (lambda c: c >= floor) if c_all >= c_none else (lambda c: c <= floor)
    order = torch.argsort(score, descending=True)
    ks = sizes(E)
    cs = ed.contrasts(kept_rows([order[:k] for k in ks]), edges)
    measured = 2 + len(ks)
    keep = order[:next((k for k, c in zip(ks, cs) if ok(c)), E)]
    t_keep = time.time() - t0 - t_rank
    before = len(keep)
    queue = list(keep.flip(0).tensor_split(min(HALVE_BLOCKS, len(keep)))) if len(keep) > 1 else []
    gone = torch.zeros(E, dtype=torch.bool)
    while queue:  # halving: the lowest-ranked blocks first, a refused block split in two
        b = queue.pop(0)
        trial = gone.clone()
        trial[b] = True
        measured += 1
        if ok(ed.contrasts(kept_rows([keep[~trial[keep]]]), edges)[0]):
            gone = trial
        elif len(b) > 1:
            queue[:0] = list(b.tensor_split(2))
    keep = keep[~gone[keep]]
    t_halve = time.time() - t0 - t_rank - t_keep
    c_kept = ed.contrasts(kept_rows([keep]), edges)[0]
    single = torch.zeros(len(keep), E, dtype=torch.bool)  # each kept edge cut alone (every other edge on)
    single[torch.arange(len(keep)), keep] = True
    strengths = [c_all - c for c in ed.contrasts(single, edges)] if len(keep) else []
    measured += len(keep) + 1
    rec = {"contrast_model": ed.c_full, "contrast_all_edges": c_all, "contrast_no_edges": c_none, "contrast_kept": c_kept,
           "floor": floor, "candidates": {"resid": len(ed.resid_all), "route": len(ed.route_all), "route_measured": len(top)},
           "kept": len(keep), "kept_before_halving": before, "measured": measured,
           "seconds": {"rank": round(t_rank, 2), "keep": round(t_keep, 2), "halving": round(t_halve, 2),
                       "total": round(time.time() - t0, 2)}}
    log(f"{len(keep)} of {E} edges ({before} before halving; {len(ed.resid_all)} resid, {len(top)} of {len(ed.route_all)} "
        f"routes measured); contrast {c_kept:.3f} bits (every edge {c_all:.3f}, none {c_none:.3f}, model {ed.c_full:.3f}); "
        f"{rec['seconds']}")
    return {"edges": [edges[i] for i in keep.tolist()], "strengths": strengths, "record": rec}


@torch.no_grad()
def graph(ed: Edges, got: dict) -> dict:
    """The stated graph as JSON: neuron poles (the activation's sign, the SwiGLU regime: the signs of gate and up) and
    the kept edges with their write weights and measured strengths (bits of contrast their cut alone removes)."""
    nodes = []
    for L, p, j in ed.N:
        layer = ed.layers[L]
        n = layer.post_attention_layernorm(ed._mid_stream(L)[p])
        g, u = float(layer.mlp.gate_proj(n)[j]), float(layer.mlp.up_proj(n)[j])
        a = float(ed.base["h"][L][p, j])
        nodes.append({"id": len(nodes), "layer": L, "neuron": j, "pos": p, "act": a, "pole": 1 if a >= 0 else -1,
                      "regime": f"g{'+' if g >= 0 else '-'}u{'+' if u >= 0 else '-'}"})
    out = []
    for e, st in zip(got["edges"], got["strengths"]):
        w, r = e[0], e[-1]
        col = weight(ed.layers[w[0]].mlp.down_proj)[:, w[2]]
        if len(e) == 3:  # through the head's OV
            c, h = e[1]
            at, kv = ed.layers[c].self_attn, h // (ed.H // ed.KV)
            vw = weight(at.v_proj).view(ed.KV, ed.hd, -1)[kv] @ (ed.layers[c].input_layernorm.weight * col)
            col = weight(at.o_proj)[:, h * ed.hd:(h + 1) * ed.hd] @ vw
        layer = ed.layers[r[0]]
        wn = layer.post_attention_layernorm.weight * col
        out.append({"src": ed.at[w], "dst": ed.at[r], "kind": "resid" if len(e) == 2 else "route",
                    "head": list(e[1]) if len(e) == 3 else None, "src_pos": w[1], "dst_pos": r[1],
                    "w_gate": float(weight(layer.mlp.gate_proj)[r[2]] @ wn), "w_up": float(weight(layer.mlp.up_proj)[r[2]] @ wn),
                    "strength": st})
    return {"ids": ed.ids, "contrast": {"answer": ed.answer, "alternatives": ed.alternatives, "target": ed.target,
                                        "value": ed.c_full}, "background": ed.background, "nodes": nodes, "edges": out,
            "record": got["record"],
            "route_cut_semantics": {"path": "value_ov", "attention_pattern": "current_run_fixed_in_local_value_delta",
                                    "head_input_norm": "recomputed_after_writer_write_subtraction",
                                    "reader_mlp": "recomputed_norm_gate_up", "combined_routes": "sum_of_individual_value_deltas",
                                    "query_key_effects_included": False, "full_writer_ablation_equivalent": False,
                                    "full_causal_completeness_certified": False}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("one",))
    ap.add_argument("model")
    ap.add_argument("text", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--background", default="circuit", choices=("circuit", "model"))
    a = ap.parse_args()
    import transformers
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    t = json.loads(a.text.read_text())
    ed = Edges(model, t["ids"], t["target"], t["answer"], t["alternatives"], [tuple(n) for n in t["neurons"]],
               background=a.background)
    got = prune(ed, log=lambda m: print(f"{a.text.stem}: {m}", flush=True))
    a.out.write_text(json.dumps(graph(ed, got)))


if __name__ == "__main__":
    main()
