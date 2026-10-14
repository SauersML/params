"""Toy -> oracle prompt (TOYLOOP lane). The oracle reads the WEIGHTS: per head its query-key and value-output circuits
read through the embedding (top token pairs), its composition with earlier heads, per neuron the tokens it reads and
writes; then attention patterns and VPD-active units on 1-2 example texts. Format of the answer it must give: FORMAT.

A toy is the Toy dataclass below: a vpd_model.Target state dict + config (pre-RMSNorm, rotate-half RoPE, GELU(tanh)
MLP, tied embedding), token names, example texts, optional ground truth. Adapters: from_planted (toys/planted.py's
.pt files), from_zoo / load_zoo (toys/zoo.py's manifest rows and records), synthetic (random toys with heads and
neurons planted on/off, for smoke tests).
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

FORMAT = """Answer with the circuit as a Mermaid flowchart over units, then one line starting "English:".
Units: tok<NAME> (a token input, by name or role), L<i>H<j> (head j of layer i), L<i>N<j> (MLP neuron j of layer i), out (the prediction).
Example:
flowchart LR
  tokK --> L0H1
  L0H1 --> L1H0
  tokQ --> L1H0
  L1H0 --> out
English: L0H1 copies each key onto the next position; L1H0 matches the query key to it and copies the value."""


# The "active at last" line names the units that write at the prediction: on the zoo, listing them alone already
# matches the truth on most families (evaluate.py's 'active' baseline). TOYLOOP_SHOW_ACTIVE=0 drops it, so the
# oracle has to find the units in the weights and attention patterns.
SHOW_ACTIVE = os.environ.get("TOYLOOP_SHOW_ACTIVE", "1") != "0"
# TOYLOOP_SECTIONS=weights: only what the weights say (weight_only_lines: positional offsets, token pairs, writer ->
# reader gains, distractors visible as writing to nothing) plus the texts' tokens; nothing computed by running the
# model. Default "all": the original weight section plus the behaviour section.
SECTIONS = os.environ.get("TOYLOOP_SECTIONS", "all")
# TOYLOOP_WEIGHTS_DETAIL=1 (weights mode only; opt-in so earlier weights-only prompts keep their bytes): the facts
# zoo2's text-dependent circuits need, all read from the weights (weight_only_lines(detail=True)): an explicit
# "gate tK" on every unit the marker detector gates, complete in-link lists for gated units (decoy-like content heads
# last), and per neuron a bias plus centred, signed token terms for its own token read and for each head it reads.
WEIGHTS_DETAIL = os.environ.get("TOYLOOP_WEIGHTS_DETAIL", "0") == "1"
DETAIL_LEGEND = ("Detail: 'gate tK' = the unit works only on texts containing tK (its gate reads the detector, which "
                 "carries tK); a neuron's bias is its pre-activation with its gate on and every input at a typical "
                 "token; reads(+t:v) and in[head:gain(+t:v)] list how far a token moves it (its own read: the last "
                 "token; through a pos-d head: the token d positions before the last); a neuron fires iff bias plus "
                 "its terms > 0. Decoy-like content heads are listed last.")

# Opt-in wording for future prompts; historical prompt bytes stay unchanged.
SUMMARY_LIMITS = (
    "Summary scope: token pairs and links are truncated, rounded weight summaries. "
    "A missing link or out[] means no displayed above-threshold linear coupling, not no causal influence. "
    "Neuron reads const describes isolated normalized embeddings, not arbitrary residual inputs. "
    "~X marks approximate opposite weight-map directions; it does not verify equal magnitudes, "
    "identical attention at every RoPE offset, or exact cancellation."
)
NORMALIZATION_RULE = (
    "Executable rule: head and neuron writes enter the residual stream. Each head Q/K/V or neuron "
    "input uses its retained residual inputs r, with RMSNorm gain*r/sqrt(mean(r*r)+eps) recomputed "
    "before the linear read; neuron outputs also apply GELU. A write with zero direct linear "
    "overlap can still change a reader through this denominator. The final readout likewise "
    "normalizes the residual containing the declared out writers."
)


@dataclass
class Toy:
    name: str
    sd: dict
    cfg: dict
    vocab: list[str]                      # a short name per token id (no spaces)
    texts: list[list[int]]                # example inputs; the prediction is read at the last position
    truth: dict | None = None             # {"units": [...], "edges": [[a, b], ...], "english": str}
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- the model, recording what the rendering needs

def _rms(x, w, eps):
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def heads_of(cfg):
    return cfg["n_layer"], cfg["n_head"], cfg["n_embd"] // cfg["n_head"]


@torch.no_grad()
def run(sd: dict, cfg: dict, ids: list[int]) -> dict:
    """One text through vpd_model.Target's forward: {"logits" [T, V], "attn" {layer: [H, T, T]}, "inp" {site: [T, in]}}."""
    L, H, hd = heads_of(cfg)
    eps, T = float(cfg.get("rms_norm_eps", 1e-6)), len(ids)
    pos = torch.arange(T, dtype=torch.float32)
    freq = float(cfg.get("rotary_base", 10000.0)) ** (torch.arange(hd // 2, dtype=torch.float32) / (hd / 2))
    ang = pos[:, None] / freq.repeat(2)[None, :]
    cos, sin = ang.cos(), ang.sin()
    rope = lambda x: x * cos + torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sin  # noqa: E731
    W = lambda i, k: sd[f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}.weight"].float()  # noqa: E731
    E = sd["wte.weight"].float()
    x = E[torch.tensor(ids)]
    attn, inp = {}, {}
    for i in range(L):
        h = _rms(x, sd[f"h.{i}.rms_1.weight"].float(), eps)
        q, k, v = (h @ W(i, kk).T for kk in ("q_proj", "k_proj", "v_proj"))
        q, k, v = (t.view(T, H, hd).transpose(0, 1) for t in (q, k, v))
        z = rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(hd)
        p = z.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
        y = (p @ v).transpose(0, 1).reshape(T, -1)
        for kk in ("q_proj", "k_proj", "v_proj"):
            inp[f"h.{i}.attn.{kk}"] = h
        inp[f"h.{i}.attn.o_proj"] = y
        attn[i] = p
        x = x + y @ W(i, "o_proj").T
        h = _rms(x, sd[f"h.{i}.rms_2.weight"].float(), eps)
        a = _gelu(h @ W(i, "c_fc").T)
        inp[f"h.{i}.mlp.c_fc"], inp[f"h.{i}.mlp.down_proj"] = h, a
        x = x + a @ W(i, "down_proj").T
    return {"logits": _rms(x, sd["ln_f.weight"].float(), eps) @ E.T, "attn": attn, "inp": inp}


# ---------------------------------------------------------------- weight reading

def _blocks(sd, cfg, i, j):
    """Head (i, j)'s query, key, value [hd, D] and output [D, hd] blocks."""
    _, _, hd = heads_of(cfg)
    s = slice(j * hd, (j + 1) * hd)
    g = lambda k: sd[f"h.{i}.attn.{k}.weight"].float()  # noqa: E731
    return g("q_proj")[s], g("k_proj")[s], g("v_proj")[s], g("o_proj")[:, s]


def _top_pairs(M, vocab, k=3, sep=">"):
    vals, idx = M.flatten().topk(min(k, M.numel()))
    n = M.shape[1]
    return " ".join(f"{vocab[int(a) // n]}{sep}{vocab[int(a) % n]}:{float(v):.1f}" for v, a in zip(vals, idx))


def _top_tokens(v, vocab, k=3):
    vals, idx = v.topk(min(k, v.numel()))
    return " ".join(f"{vocab[int(a)]}:{float(s):.1f}" for s, a in zip(vals, idx))


def _cos_comp(A, B):
    """How much of B's input space A's output fills: ||B A|| / (||B|| ||A||) (Elhage et al. composition score)."""
    d = A.norm() * B.norm()
    return float((B @ A).norm() / d) if d > 1e-9 else 0.0


def weight_lines(toy: Toy, top_neurons: int | None = 6, k: int = 3) -> list[str]:
    """Signed weight summaries; None lists every neuron above the live threshold.

    Listing all live neurons removes the count cap, not token-summary compression
    or the declared strength threshold. It does not expose complete raw tensors.
    """
    sd, cfg, voc = toy.sd, toy.cfg, toy.vocab
    L, H, hd = heads_of(cfg)
    E = sd["wte.weight"].float()
    lines = []
    for i in range(L):
        En = _rms(E, sd[f"h.{i}.rms_1.weight"].float(), float(cfg.get("rms_norm_eps", 1e-6)))
        for j in range(H):
            Wq, Wk, Wv, Wo = _blocks(sd, cfg, i, j)
            nq, nk, nv, no = (float(w.norm()) for w in (Wq, Wk, Wv, Wo))
            head = f"L{i}H{j} |q|{nq:.2g} |k|{nk:.2g} |v|{nv:.2g} |o|{no:.2g}"
            if min(nv, no) < 1e-3 and min(nq, nk) < 1e-3:
                lines.append(head + " (zero)")
                continue
            qk = (En @ Wq.T) @ (En @ Wk.T).T / math.sqrt(hd)          # [V query token, V key token], RoPE ignored
            ov = (En @ Wv.T) @ Wo.T @ E.T                              # [V source token, V written token]
            qk_spread = float(qk.std())
            head += f" QK {_top_pairs(qk, voc, k)} (spread {qk_spread:.1f}) OV {_top_pairs(ov, voc, k)}"
            comps = []
            for i2 in range(i):                                         # composition with earlier heads
                for j2 in range(H):
                    _, _, Wv2, Wo2 = _blocks(sd, cfg, i2, j2)
                    OV2 = Wo2 @ Wv2                                     # [D, D]
                    for nm, Wx in (("Q", Wq), ("K", Wk), ("V", Wv)):
                        c = _cos_comp(OV2, Wx)
                        if c > 0.15:
                            comps.append(f"{nm}<L{i2}H{j2}:{c:.2f}")
            if comps:
                head += " comp " + " ".join(comps)
            lines.append(head)
        fc, down = sd[f"h.{i}.mlp.c_fc.weight"].float(), sd[f"h.{i}.mlp.down_proj.weight"].float()
        En2 = _rms(E, sd[f"h.{i}.rms_2.weight"].float(), float(cfg.get("rms_norm_eps", 1e-6)))
        strength = fc.norm(dim=1) * down.norm(dim=0)
        live = int((strength > 1e-3).sum())
        if live == 0:
            lines.append(f"L{i} MLP (zero)")
            continue
        top = strength.topk(live if top_neurons is None else min(top_neurons, live)).indices.tolist()
        detail = f"all {live}" if top_neurons is None else f"strongest {len(top)}"
        lines.append(f"L{i} MLP: {live} live neurons, {detail}:")
        for n in top:
            lines.append(f"  L{i}N{n} w{float(strength[n]):.2g} reads {_top_tokens(En2 @ fc[n], voc, k)}"
                         f" writes {_top_tokens(E @ down[:, n], voc, k)}")
    return lines


# ---------------------------------------------------------------- weights only (TOYLOOP_SECTIONS=weights)

def _eps(cfg):
    return float(cfg.get("rms_norm_eps", 1e-6))


def _rope_scores(q, k, cfg, maxd):
    """Attention score q.k / sqrt(hd) for a key d positions before the query, d = 0..maxd (RoPE rotate-half)."""
    _, _, hd = heads_of(cfg)
    freq = float(cfg.get("rotary_base", 10000.0)) ** (torch.arange(hd // 2, dtype=torch.float32) / (hd / 2))
    def rope(x, p):
        ang = p / freq.repeat(2)
        return x * ang.cos() + torch.cat([-x[hd // 2:], x[:hd // 2]]) * ang.sin()
    P = float(maxd)
    qr = rope(q, P)
    return torch.stack([qr @ rope(k, P - d) for d in range(maxd + 1)]) / math.sqrt(hd)


def units_of(toy) -> list:
    """Every live unit: (name, layer, kind 'H'/'N', index), heads first per layer."""
    sd, cfg = toy.sd, toy.cfg
    L, H, _ = heads_of(cfg)
    out = []
    for i in range(L):
        for j in range(H):
            if max(float(w.norm()) for w in _blocks(sd, cfg, i, j)[2:]) > 1e-6:
                out.append((f"L{i}H{j}", i, "H", j))
        fc, down = sd[f"h.{i}.mlp.c_fc.weight"].float(), sd[f"h.{i}.mlp.down_proj.weight"].float()
        for n in ((fc.norm(dim=1) * down.norm(dim=0)) > 1e-9).nonzero().flatten().tolist():
            out.append((f"L{i}N{n}", i, "N", n))
    return out


def wiring(toy) -> dict:
    """Weight-only connection strengths. writes[u] = {reader: gain}: the largest singular value of reader-input-map x
    u's output map (RMSNorm weights folded in, the norm's scale ignored); readers are later heads' Q/K/V ("L1H0.K"),
    MLP neurons of the same or later layer (heads) / later layers (neurons), and "logits" (the tied unembedding after
    ln_f); writer "tok" is the token identity (embeddings minus their mean, scaled to unit norm per token). A unit whose every gain is ~0 writes only where nothing reads (a distractor's dump dims)."""
    units, out_map, readers = _maps(toy)
    writes = {}
    for u, i, kind, _ in [("tok", -1, "T", 0)] + units:
        w = {}
        for r, val in readers.items():
            if r == "logits":
                B = val
            else:
                B, ri = val
                if not (ri > i or (ri == i and kind == "H" and "." not in r)):   # later layers; a head's own MLP
                    continue
            if r == "logits" and u == "tok":
                continue
            w[r] = float(torch.linalg.matrix_norm(B @ out_map[u], ord=2))
        writes[u] = w
    return writes


_MAPS: dict = {}


def _maps(toy):
    """(live units, writer output maps {unit: [D, r]} plus "tok", reader input maps {reader: ([r, D], layer)} plus
    "logits"), cached per state dict."""
    key = id(toy.sd)
    if key in _MAPS and _MAPS[key][0] is toy.sd:
        return _MAPS[key][1]
    sd, cfg = toy.sd, toy.cfg
    E = sd["wte.weight"].float()
    units = units_of(toy)
    out_map = {}
    for u, i, kind, j in units:
        if kind == "H":
            _, _, Wv, Wo = _blocks(sd, cfg, i, j)
            out_map[u] = Wo @ Wv * sd[f"h.{i}.rms_1.weight"].float()[None, :]      # residual -> residual [D, D]
        else:
            out_map[u] = sd[f"h.{i}.mlp.down_proj.weight"].float()[:, j:j + 1]      # [D, 1]
    readers = {"logits": E * sd["ln_f.weight"].float()[None, :]}
    for u, i, kind, j in units:
        if kind == "H":
            Wq, Wk, Wv, _ = _blocks(sd, cfg, i, j)
            g = sd[f"h.{i}.rms_1.weight"].float()[None, :]
            readers.update({f"{u}.Q": (Wq * g, i), f"{u}.K": (Wk * g, i), f"{u}.V": (Wv * g, i)})
        else:
            readers[u] = (sd[f"h.{i}.mlp.c_fc.weight"].float()[j:j + 1] * sd[f"h.{i}.rms_2.weight"].float()[None, :], i)
    Ec = E - E.mean(0, keepdim=True)        # token identity: the embedding minus what every token shares
    out_map["tok"] = Ec.T / Ec.norm(dim=1).mean()   # [D, V], one token ~ unit norm (readers see RMS-normed input)
    if len(_MAPS) > 64:
        _MAPS.clear()
    _MAPS[key] = (sd, (units, out_map, readers))
    return units, out_map, readers


def source_token(toy, writer: str, reader: str, ratio: float = 2.0) -> str | None:
    """The token whose embedding, carried by head `writer`'s OV, reaches `reader`'s input most, when it dominates
    (top over second > ratio): e.g. which marker token a detector head turns into the flag a gated unit reads."""
    units, out_map, readers = _maps(toy)
    if not re.fullmatch(r"L\d+H\d+", writer) or reader not in readers or reader == "logits":
        return None
    v = (readers[reader][0] @ out_map[writer] @ out_map["tok"]).norm(dim=0)
    t = v.topk(2)
    return toy.vocab[int(t.indices[0])] if float(t.values[0]) > ratio * float(t.values[1]) + 1e-9 else None


def _content_key(qk, pos) -> int | None:
    """A content head's key token id: position barely matters and its top 3 QK pairs share one key (not BOS)."""
    if float(pos.max() - pos.min()) > 1.0 or float(qk.std()) <= 0.5:
        return None
    keys = {int(a) % qk.shape[1] for a in qk.flatten().topk(3).indices}
    return keys.pop() if len(keys) == 1 and 0 not in keys else None


def gate_facts(toy) -> tuple:
    """(detector, {gated unit: gate token id}) from the weights: a gate link is a head's Q link or a neuron's input link
    from a layer-0 head carrying one dominant token (source_token); the detector is the layer-0 head with most such
    links, and the units it gates read their marker through it."""
    from collections import Counter
    units, _, _ = _maps(toy)
    writes = wiring(toy)
    top = max((g for u, w in writes.items() if u != "tok" for g in w.values()), default=1.0) or 1.0
    l0 = [u for u, i, kind, _ in units if i == 0 and kind == "H"]
    links = []
    for u, i, kind, _ in units:
        r = f"{u}.Q" if kind == "H" else u
        for w in l0:
            if w != u and writes[w].get(r, 0.0) > 1e-3 * top:
                src = source_token(toy, w, r)
                if src:
                    links.append((u, w, toy.vocab.index(src)))
    if not links:
        return None, {}
    det = Counter(w for _, w, _ in links).most_common(1)[0][0]
    return det, {u: t for u, w, t in links if w == det}


def neuron_terms(toy, u: str, det: str | None = None, gate: int | None = None) -> tuple:
    """(bias, own read [V], {head: [V]}) of neuron u, centred per token: the pre-activation is about
    bias + read[last token] + sum over heads of term[token the head carries]. bias: every input at its median token,
    plus the detector's term for the gate token when gated."""
    units, out_map, readers = _maps(toy)
    writes = wiring(toy)
    top = max((g for x, w in writes.items() if x != "tok" for g in w.values()), default=1.0) or 1.0
    E = toy.sd["wte.weight"].float()
    En0 = E * torch.rsqrt(E.pow(2).mean(-1, keepdim=True) + _eps(toy.cfg))
    B = readers[u][0]
    rd = (B @ En0.T)[0]
    bias, terms = float(rd.median()), {}
    for w, i, kind, _ in units:
        if kind == "H" and writes[w].get(u, 0.0) > 1e-3 * top:
            c = (B @ out_map[w] @ En0.T)[0]
            m = float(c.median())
            bias += m
            if w == det:
                bias += float(c[gate]) - m if gate is not None else 0.0
            else:
                terms[w] = c - m
    return bias, rd - rd.median(), terms


def _keys(c, voc) -> str:
    """Centred token terms: the two largest positive and the most negative, those above a fifth of the largest |c|."""
    mx = float(c.abs().max())
    if mx < 0.05:
        return ""
    up = [f"+{voc[int(i)]}:{float(c[i]):.2g}" for i in c.topk(2).indices if float(c[i]) > 0.2 * mx]
    i = int(c.argmin())
    return ",".join(up + ([f"-{voc[i]}:{float(c[i]):.2g}"] if float(c[i]) < -0.2 * mx else []))


def weight_only_lines(toy: Toy, k: int = 2, max_neurons: int | None = 16, max_links: int = 3,
                      *, explain_summary_limits: bool = False, detail: bool = False) -> list[str]:
    """Compact, weights only. Per head: output norm, positional preference (the mean embedding's query and key through
    RoPE: best key offset and its margin over the next, 'pos flat' when position barely matters), QK and OV token pairs
    (through the embedding, so a rotated residual basis changes nothing), the units its Q, K and V read and the units
    (or logits) that read its output, by gain. Per live neuron: its signed token reads (or 'const'), unit inputs, signed
    token writes and readers. Cancelling pairs are marked '~X'. A unit with 'out[]' writes where nothing reads."""
    sd, cfg, voc = toy.sd, toy.cfg, toy.vocab
    L, H, hd = heads_of(cfg)
    if len(units_of(toy)) > 24:          # big models (zoo2: 24 heads): shorter lists keep the prompt near 2k tokens
        k, max_links = 1, 2
    E = sd["wte.weight"].float()
    El = E * sd["ln_f.weight"].float()[None, :]
    writes = wiring(toy)
    top = max((g for u, w in writes.items() if u != "tok" for g in w.values()), default=1.0) or 1.0
    reads = {}
    for u, w in writes.items():
        for r, g in w.items():
            reads.setdefault(r, []).append((g, u))

    det, gates, ckey, nterms = None, {}, {}, {}
    if detail:
        det, gates = gate_facts(toy)
        for u, i, kind, j in units_of(toy):
            if kind == "H":
                Wq, Wk, _, _ = _blocks(sd, cfg, i, j)
                En = _rms(E, sd[f"h.{i}.rms_1.weight"].float(), _eps(cfg))
                c = En.mean(0)
                key = _content_key((En @ Wq.T) @ (En @ Wk.T).T / math.sqrt(hd),
                                   _rope_scores(Wq @ c, Wk @ c, cfg, min(int(cfg.get("n_ctx", 32)) - 1, 15)))
                if key is not None and u != det:
                    ckey[u] = key

    def fmt(pairs, reader=None, full=False):
        keep = []
        order = sorted(pairs, reverse=True)
        if full:        # a gated unit: every link, decoy-like content heads last
            order = sorted(order, key=lambda x: x[1] in ckey)
        for g, n in order:
            if g > 1e-3 * top and (full or len(keep) < max_links):
                src = source_token(toy, n, reader) if reader else None
                kt = _keys(nterms[reader][n], voc) if reader in nterms and n in nterms[reader] else ""
                keep.append(f"{n}:{g:.2g}" + (f"<{src}" if src else "") + (f"({kt})" if kt else ""))
        return ",".join(keep)

    def pairs(M):
        vals, idx = M.flatten().topk(k)
        n = M.shape[1]
        return ",".join(f"{voc[int(a) // n]}>{voc[int(a) % n]}:{float(v):.2g}" for v, a in zip(vals, idx))

    lines = ["Legend: pos-d = prefers the key d positions back by position alone (m = margin); QK[query>key] and "
             "OV[source>written logit] token pairs; in Q/K/V[writer:gain], out[reader:gain] (gain = largest singular "
             "value of reader x writer map; out[] = nothing reads it; L0H5:33<t34 = mostly token t34 carried by L0H5); "
             "~X = cancels X (same input, opposite output)."]
    if detail:
        lines[0] += " " + DETAIL_LEGEND
    if explain_summary_limits:
        lines = ["Legend: pos-d is the mean-embedding positional preference (m = margin); "
                 "QK[query>key] and OV[source>written logit] are isolated-embedding token pairs; "
                 "in Q/K/V[writer:gain], out[reader:gain] list linear map gains (largest singular value), "
                 "with learned norm gains folded in but RMS denominators omitted. "
                 "L0H5:33<t34 identifies the dominant isolated source token for that linear map.",
                 SUMMARY_LIMITS, NORMALIZATION_RULE]
    maxd = min(int(cfg.get("n_ctx", 32)) - 1, 15)
    cancel = cancelling_pairs(toy)
    for u, i, kind, j in units_of(toy):
        if kind == "H":
            Wq, Wk, Wv, Wo = _blocks(sd, cfg, i, j)
            En = _rms(E, sd[f"h.{i}.rms_1.weight"].float(), _eps(cfg))
            c = En.mean(0)
            pos = _rope_scores(Wq @ c, Wk @ c, cfg, maxd)
            best = pos.topk(2)
            qk = (En @ Wq.T) @ (En @ Wk.T).T / math.sqrt(hd)
            ov = (En @ Wv.T) @ Wo.T @ El.T
            line = f"{u}" + (f" gate {voc[gates[u]]}" if u in gates else "") + f" |o|{float(Wo.norm()):.2g} " + (
                f"pos-{int(best.indices[0])} m{float(best.values[0] - best.values[1]):.2g}"
                if float(pos.max() - pos.min()) > 1.0 else "pos flat")
            line += f" QK[{pairs(qk)}]" if float(qk.std()) > 0.5 else " QK flat"
            line += f" OV[{pairs(ov)}]" if float(ov.abs().max()) > 0.05 else " OV 0"
            ins = [f"{r}[{fmt(reads.get(f'{u}.{r}', []), f'{u}.{r}', full=u in gates)}]" for r in ("Q", "K", "V")
                   if reads.get(f"{u}.{r}")]
            ins = [x for x in ins if not x.endswith("[]")]
            line += (" in " + " ".join(ins)) if ins else ""
        else:
            fc, down = sd[f"h.{i}.mlp.c_fc.weight"].float()[j], sd[f"h.{i}.mlp.down_proj.weight"].float()[:, j]
            En2 = _rms(E, sd[f"h.{i}.rms_2.weight"].float(), _eps(cfg))
            rd = En2 @ fc
            tokread = "const" if float(rd.std()) < 0.01 * float(rd.abs().mean()) + 1e-6 else _signed_tokens(rd, voc)
            pre = u
            if detail:
                bias, own, nterms[u] = neuron_terms(toy, u, det, gates.get(u))
                tokread = _keys(own, voc) or "0"
                pre = u + (f" gate {voc[gates[u]]}" if u in gates else "") + f" bias {bias:.2g}"
            ins = fmt([x for x in reads.get(u, []) if x[1] != "tok"], u, full=u in gates)
            line = f"{pre} reads {tokread}" + (f" in[{ins}]" if ins else "") + f" writes {_signed_tokens(El @ down, voc)}"
        line += f" out[{fmt([(g, r) for r, g in writes[u].items()])}]"
        if u in cancel:
            line += f" ~{cancel[u]}"
        lines.append(line)
    lines.insert(1, f"tok (token identity) out[{fmt([(g, r) for r, g in writes['tok'].items()])}]")
    n_neurons = sum(1 for x in units_of(toy) if x[2] == "N")
    if max_neurons is not None and n_neurons > max_neurons:   # keep the heads and the strongest-writing neurons
        neu = [l for l in lines[1:] if re.match(r"L\d+N", l)]
        keep = sorted(neu, key=lambda l: (l.split()[0] not in gates, -max(writes[l.split()[0]].values(), default=0)))
        keep = keep[:max(max_neurons, sum(1 for l in neu if l.split()[0] in gates))]   # detail: every gated neuron
        lines = [l for l in lines if not re.match(r"L\d+N", l) or l in keep]
        lines.append(f"({n_neurons - len(keep)} weaker live neurons not shown)")
    return lines


def _signed_tokens(v, vocab, k=2):
    """The k most raised and the most lowered tokens: '+t1:6 +t3:2 -t24:-8'."""
    up = ",".join(f"+{vocab[int(i)]}:{float(v[i]):.2g}" for i in v.topk(k).indices if float(v[i]) > 1e-3)
    i = int(v.argmin())
    down = f"-{vocab[i]}:{float(v[i]):.2g}" if float(v[i]) < -1e-3 else ""
    return ",".join(x for x in (up, down) if x) or "0"


def cancelling_pairs(toy, tol: float = 0.98) -> dict:
    """{unit: partner} for same-kind units in one layer that read the same input (cosine > tol of a head's Wq^T Wk, a
    neuron's c_fc row) and write opposite outputs (cosine < -tol of Wo Wv / down_proj column): their sum is zero."""
    sd, cfg = toy.sd, toy.cfg
    sig = {}
    for u, i, kind, j in units_of(toy):
        if kind == "H":   # Wq^T Wk and Wo Wv: unchanged by a rotation of the head's dims (twins rotated within planes)
            Wq, Wk, Wv, Wo = _blocks(sd, cfg, i, j)
            sig[u] = (i, kind, (Wq.T @ Wk).flatten(), (Wo @ Wv).flatten())
        else:
            sig[u] = (i, kind, sd[f"h.{i}.mlp.c_fc.weight"].float()[j], sd[f"h.{i}.mlp.down_proj.weight"].float()[:, j])
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))  # noqa: E731
    out, us = {}, list(sig)
    for x in range(len(us)):
        for y in range(x + 1, len(us)):
            a, b = sig[us[x]], sig[us[y]]
            if a[:2] == b[:2] and cos(a[2], b[2]) > tol and cos(a[3], b[3]) < -tol:
                out[us[x]], out[us[y]] = us[y], us[x]
    return out


def text_lines(toy: Toy, n_texts: int = 2) -> list[str]:
    return [f"Text {ti}: {' '.join(toy.vocab[a] for a in ids)} (predict at position {len(ids) - 1})"
            for ti, ids in enumerate(toy.texts[:n_texts])]


# ---------------------------------------------------------------- behaviour on example texts

def active_units(toy: Toy, r: dict, t: int, rel: float = 0.05) -> list[str]:
    """Units active at position t, VPD-style: each head's write (its o_proj block times its attention output) and each
    neuron's write (activation times its down_proj column), as a fraction of the largest write at t; those above rel."""
    L, H, hd = heads_of(toy.cfg)
    acts = {}
    for i in range(L):
        W, y = toy.sd[f"h.{i}.attn.o_proj.weight"].float(), r["inp"][f"h.{i}.attn.o_proj"][t]
        for j in range(H):
            acts[f"L{i}H{j}"] = float((W[:, j * hd:(j + 1) * hd] @ y[j * hd:(j + 1) * hd]).norm())
        a = r["inp"][f"h.{i}.mlp.down_proj"][t].abs() * toy.sd[f"h.{i}.mlp.down_proj.weight"].float().norm(dim=0)
        for n in a.topk(min(4, a.numel())).indices.tolist():
            acts[f"L{i}N{n}"] = float(a[n])
    m = max(acts.values(), default=0.0)
    return [f"{u}:{v / m:.2f}" for u, v in acts.items() if m > 1e-6 and v > rel * m]


def behaviour_lines(toy: Toy, n_texts: int = 2, compact: bool = False) -> list[str]:
    L, H, _ = heads_of(toy.cfg)
    lines = []
    for ti, ids in enumerate(toy.texts[:n_texts]):
        r = run(toy.sd, toy.cfg, ids)
        T = len(ids)
        pred = toy.vocab[int(r["logits"][-1].argmax())]
        lines.append(f"Text {ti}: {' '.join(f'{toy.vocab[a]}' for a in ids)} -> predicts {pred}")
        for i in range(L):
            for j in range(H):
                if all(float(w.norm()) < 1e-3 for w in _blocks(toy.sd, toy.cfg, i, j)):
                    continue                                                # a zero head: nothing to see
                p = r["attn"][i][j]
                arg = p.argmax(-1)                                          # [T] the key each query position attends most
                offs = (arg - torch.arange(T))[1:]
                mode = int(offs.mode().values) if len(offs) else 0
                share = float((offs == mode).float().mean()) if len(offs) else 1.0
                kl = int(arg[-1])
                if compact:   # "L0H2 -1:100% t10@12:1.0" = usual offset and its share; the last position's key
                    lines.append(f"  L{i}H{j} {mode:+d}:{share:.0%} {toy.vocab[ids[kl]]}@{kl}:{float(p[-1, kl]):.2g}")
                else:
                    lines.append(f"  L{i}H{j}: offset {mode:+d} at {share:.0%} of positions; last attends "
                                 f"{toy.vocab[ids[kl]]}@{kl} {float(p[-1, kl]):.2f}")
        if SHOW_ACTIVE:
            lines.append("  active at last: " + " ".join(active_units(toy, r, T - 1)))
    return lines


def render(toy: Toy, n_texts: int = 2, weights_from: Toy | None = None, *, all_live_neurons: bool = False,
           explain_summary_limits: bool = False) -> str:
    """The prompt. weights_from: take the weight section from another toy (same architecture), keeping this toy's
    behaviour section (the swapped-weights control: does the answer follow the weights?). TOYLOOP_SECTIONS=weights:
    the compact weight section and the texts' tokens only; default: weights and behaviour (zoo v1 toys keep the
    legacy weight section, zoo2 toys get the compact one). all_live_neurons removes
    only the neuron-count cap in either format. Callers must reject a context-cap
    failure rather than truncate facts; token/connection summaries remain bounded.
    explain_summary_limits corrects legacy shorthand and supplies the residual/RMS
    execution rule, without changing any numerical facts or consulting targets.
    It is opt-in so archived prompts and matched comparisons retain their bytes.
    """
    if type(all_live_neurons) is not bool:
        raise ValueError("all_live_neurons must be an explicit boolean")
    if type(explain_summary_limits) is not bool:
        raise ValueError("explain_summary_limits must be an explicit boolean")
    L, H, hd = heads_of(toy.cfg)
    if SECTIONS == "weights":
        return "\n".join([f"Toy transformer: {L} layers, {H} heads of dim {hd}, d_model {toy.cfg['n_embd']}, "
                          f"vocab {len(toy.vocab)}, RoPE, tied embedding. Tokens: {' '.join(toy.vocab)}.",
                          "Weights:", *weight_only_lines(weights_from or toy, max_neurons=None if all_live_neurons else 16,
                                                        explain_summary_limits=explain_summary_limits,
                                                        detail=WEIGHTS_DETAIL),
                          "Texts:", *text_lines(toy, n_texts),
                          "Which units implement the prediction at the last position, and how?", FORMAT])
    if toy.meta.get("version") == 2:   # zoo2: the compact weight section (the legacy one is ~4k tokens there)
        return "\n".join([f"Toy transformer: {L} layers, {H} heads of dim {hd}, d_model {toy.cfg['n_embd']}, "
                          f"vocab {len(toy.vocab)}, RoPE, tied embedding. Tokens: {' '.join(toy.vocab)}.",
                          "Weights:", *weight_only_lines(weights_from or toy, max_neurons=None if all_live_neurons else 16,
                                                        explain_summary_limits=explain_summary_limits),
                          "Behaviour (per head: usual attention offset:share of positions, key attended from the "
                          "last position@position:weight):", *behaviour_lines(toy, n_texts, compact=True),
                          "Which units implement the prediction at the last position, and how?", FORMAT])
    head = (f"Toy transformer: {L} layers, {H} heads of dim {hd}, d_model {toy.cfg['n_embd']}, "
            f"vocab {len(toy.vocab)}, RoPE, tied embedding. Tokens: {' '.join(toy.vocab)}.\n"
            "Weights (QK: query>key token pairs scored through the embedding; OV: source>written token pairs; "
            "comp: Q/K/V composition with an earlier head's output):")
    notes = [SUMMARY_LIMITS, NORMALIZATION_RULE] if explain_summary_limits else []
    return "\n".join([head, *notes, *weight_lines(weights_from or toy, top_neurons=None if all_live_neurons else 6), "Behaviour:", *behaviour_lines(toy, n_texts),
                      "Which units implement the prediction at the last position, and how?", FORMAT])


def target_text(truth: dict, toy: Toy | None = None) -> str:
    """The training completion for a ground truth. TOYLOOP_TRACE=1 (read at call time) and a zoo2 toy: a short
    derivation (trace_lines) comes first, then the Mermaid and English."""
    body = "\n".join(f"  {a} --> {b}" for a, b in truth["edges"])
    out = f"flowchart LR\n{body}\nEnglish: {truth.get('english', '').strip()}"
    if os.environ.get("TOYLOOP_TRACE", "0") == "1" and toy is not None and toy.meta.get("version") == 2:
        out = "\n".join(trace_lines(toy, truth)) + "\n" + out
    return out


def trace_lines(toy: Toy, truth: dict) -> list[str]:
    """The derivation of a zoo2 text's TRUE circuit, never of a predicted one: the marker (the text's token at position
    1), the detector, the module's gated units, the layer-0 units the circuit reads, and the units a reader of the
    weights would consider but must drop, each with its reason (from the record's params and the weights' links).
    No arrows and no "English:", so circuit.parse reads only the Mermaid that follows."""
    p = toy.meta["record"]["params"]
    mod = p["modules"][toy.meta["module"]]
    ids = toy.texts[0]
    marker, det = toy.vocab[ids[1]], zoo_unit(p["detector"])
    unit_re = re.compile(r"L\d+[HN]\d+")
    U = {u for u in truth["units"] if unit_re.fullmatch(u)}
    gated_mod = [zoo_unit(u) for u in mod["gated"]]
    gated = sorted(u for u in gated_mod if u in U)
    l0 = sorted(U - {det} - set(gated))
    cancel = cancelling_pairs(toy)
    other_gate = {zoo_unit(u): toy.vocab[o["marker"]] for o in p["modules"] if o is not mod for u in o["gated"]}
    decoy = {zoo_unit(d["unit"]): toy.vocab[d["trigger"]] for d in p["decoys"]}
    twins = {zoo_unit(u) for t in p["twins"] for u in t}
    dumps = {zoo_unit(u) for u in p["dump_neurons"]}
    why = {}
    for u in sorted(set(gated_mod) - U):     # the module's own gated units left out on this text
        why[u] = "cancelling twin" if u in cancel else ("off on this text" if "N" in u else "not used")
    writes = wiring(toy)
    top = max((g for x, w in writes.items() if x != "tok" for g in w.values()), default=1.0) or 1.0
    for w, rs in writes.items():             # what the circuit's units read but the circuit does not use
        if w == "tok" or w in U or w in why:
            continue
        if not any(g > 1e-3 * top and r.split(".")[0] in U for r, g in rs.items()):
            continue
        why[w] = (f"decoy, trigger {decoy[w]} absent" if w in decoy else f"gated by {other_gate[w]}" if w in other_gate
                  else "cancelling twin" if w in twins or w in cancel else "writes nothing" if w in dumps
                  else "other mechanism")
    groups = {}
    for u, r in why.items():
        groups.setdefault(r, []).append(u)
    dropped = "; ".join(f"{', '.join(sorted(us))} ({r})" for r, us in sorted(groups.items(), key=lambda x: x[0]))
    return ["Trace:", f"marker {marker} at position 1; detector {det}",
            f"gated by {marker}: {', '.join(gated) or 'none'}",
            f"read in layer 0: {', '.join(l0) or 'none'}",
            f"dropped: {dropped or 'none'}"]


# ---------------------------------------------------------------- adapters

PLANTED_VOCAB = (["BOS", "Q", "SEP"] + [f"K{i}" for i in range(8)] + [f"V{i}" for i in range(8)]
                 + ["A0", "A1", "B0", "B1", "YES", "NO"] + [f"F{i}" for i in range(15)])


def from_planted(path: str | Path, truth: dict | None = None, n_texts: int = 2) -> Toy:
    """A toys/planted.py checkpoint ({"sd", "cfg"}); example texts from planted.ordinary."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "toys"))
    import planted  # noqa: E402

    path = Path(path)
    d = torch.load(path, map_location="cpu")
    name = path.stem
    return Toy(name, d["sd"], d["cfg"], PLANTED_VOCAB, planted.ordinary(name, n_texts), truth, {"path": str(path)})


def zoo_unit(u: str) -> str:
    """The zoo's unit names to the oracle's: 'L1.H3' -> 'L1H3', 'tok:offset-3' -> 'tok_offset_3', 'OUT' -> 'out'."""
    if u == "OUT":
        return "out"
    if u.startswith("tok:"):
        return "tok_" + re.sub(r"\W", "_", u[4:])
    return u.replace(".", "")


def from_zoo(row: dict, root: str | Path | None = None, n_texts: int = 2) -> Toy:
    """A toys/zoo.py manifest row ({"id", "weights": .pt {"sd", "cfg"}, "record": .json with "unit_edges",
    "explanation", "examples" [{"ids", "gt_nodes", ...}]}, paths relative to the manifest's directory). The record
    goes into meta (gt_nodes per example for node-level scoring); truth is the record's unit edges, renamed by zoo_unit."""
    root = Path(root) if root else Path.home() / "mpd-data/toy_zoo"
    d = torch.load(root / row["weights"], map_location="cpu")
    rec = json.loads((root / row["record"]).read_text())
    V = d["sd"]["wte.weight"].shape[0]
    vocab = ["BOS"] + [f"t{i}" for i in range(1, V)]
    edges = [[zoo_unit(a), zoo_unit(b)] for a, b in rec["unit_edges"]]
    units = sorted({u for e in edges for u in e} | {zoo_unit(u) for u in rec["gt_units"]})
    truth = {"units": units, "edges": edges, "english": re.sub(r"\bL(\d+)\.([HN]\d+)", r"L\1\2", rec["explanation"])}
    texts = [e["ids"] for e in rec["examples"] if e["kind"] == "ordinary"][:n_texts]
    return Toy(rec["id"], d["sd"], d["cfg"], vocab, texts, truth, {"split": rec["split"], "family": rec["family"],
                                                                  "record": rec})


def from_zoo2(row: dict, root: str | Path | None = None) -> list[Toy]:
    """A toys/zoo2.py manifest row -> one Toy per example text (the truth is per text: the detector plus the selected
    module's units). Toy name "<id>#<k>"; texts [that text]; meta: split, family (the text's module family), module,
    example index, version 2 and the shared record (params for zoo2.texts / gt_spec in reward_exec)."""
    root = Path(root) if root else Path.home() / "mpd-data/toy_zoo2"
    d = torch.load(root / row["weights"], map_location="cpu")
    rec = json.loads((root / row["record"]).read_text())
    V = d["sd"]["wte.weight"].shape[0]
    vocab = ["BOS"] + [f"t{i}" for i in range(1, V)]
    out = []
    for k, e in enumerate(rec["examples"]):
        edges = [[zoo_unit(a), zoo_unit(b)] for a, b in e["unit_edges"]]
        units = sorted({u for x in edges for u in x} | {zoo_unit(u) for u in e["units"]})
        truth = {"units": units, "edges": edges, "english": re.sub(r"\bL(\d+)\.([HN]\d+)", r"L\1\2", e["explanation"])}
        out.append(Toy(f"{rec['id']}#{k}", d["sd"], d["cfg"], vocab, [e["ids"]], truth,
                       {"split": rec["split"], "family": e["family"], "module": int(e["module"]), "example": k,
                        "kind": e["kind"], "version": 2, "record": rec, "instance": rec["id"]}))
    return out


def load_zoo(manifest: str | Path, split: str | None = None, limit: int | None = None) -> list[Toy]:
    """Every toy in a zoo manifest, or those of one split (train, test_instance, test_family); limit: at most this many
    instances. A zoo2 manifest (rows with "version": 2) gives one Toy per example text."""
    m = Path(manifest)
    rows = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if split is None or r.get("split") == split][:limit]
    out = []
    for r in rows:
        if r.get("version") == 2:
            out += from_zoo2(r, m.parent)
        else:
            out.append(from_zoo(r, m.parent))
    return out


def synthetic(seed: int, L: int = 2, H: int = 2, D: int = 16, V: int = 12, T: int = 8) -> Toy:
    """A random toy with a random subset of heads and neurons on (the rest zero); truth: the live heads chained by
    layer, the live neurons after them. Readable from the weights (zero norms), so the loop has something to learn."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) / math.sqrt(s[-1])  # noqa: E731
    sd = {"wte.weight": torch.randn(V, D, generator=g), "ln_f.weight": torch.ones(D)}
    hd = D // H
    live_h, live_n = [], []
    for i in range(L):
        sd[f"h.{i}.rms_1.weight"], sd[f"h.{i}.rms_2.weight"] = torch.ones(D), torch.ones(D)
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"h.{i}.attn.{k}.weight"] = r(D, D)
        sd[f"h.{i}.mlp.c_fc.weight"], sd[f"h.{i}.mlp.down_proj.weight"] = r(4 * D, D), torch.zeros(D, 4 * D)
        for j in range(H):
            if torch.rand(1, generator=g) < 0.5:
                sd[f"h.{i}.attn.o_proj.weight"][:, j * hd:(j + 1) * hd] = 0
                sd[f"h.{i}.attn.v_proj.weight"][j * hd:(j + 1) * hd] = 0
            else:
                live_h.append((i, f"L{i}H{j}"))
        if torch.rand(1, generator=g) < 0.5:
            n = int(torch.randint(4 * D, (1,), generator=g))
            sd[f"h.{i}.mlp.down_proj.weight"][:, n] = torch.randn(D, generator=g)
            live_n.append((i, f"L{i}N{n}"))
    cfg = {"n_layer": L, "n_head": H, "n_embd": D, "n_key_value_heads": H, "rotary_dim": hd, "rms_norm_eps": 1e-6,
           "n_ctx": 32, "rotary_base": 10000.0}
    vocab = [f"T{i}" for i in range(V)]
    texts = torch.randint(V, (2, T), generator=g).tolist()
    units = [u for _, u in sorted(live_h + live_n)]
    groups = [["tok" + vocab[texts[0][-1]]]] + [[u for i2, u in sorted(live_h) if i2 == i] + [u for i2, u in live_n if i2 == i]
                                                for i in range(L)] + [["out"]]
    groups = [gr for gr in groups if gr]                    # each layer's live units read the previous live layer's
    edges = [[a, b] for g1, g2 in zip(groups, groups[1:]) for a in g1 for b in g2]
    english = ("Live units " + ", ".join(units) + " carry the last token to the prediction; the rest are zero.") if units else "Every unit is zero."
    return Toy(f"synthetic{seed}", sd, cfg, vocab, texts, {"units": units + ["tok" + vocab[texts[0][-1]], "out"],
                                                          "edges": edges, "english": english})


if __name__ == "__main__":
    import sys
    toy = from_planted(sys.argv[1]) if len(sys.argv) > 1 else synthetic(0)
    print(render(toy))
