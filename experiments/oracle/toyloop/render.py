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


def weight_lines(toy: Toy, top_neurons: int = 6, k: int = 3) -> list[str]:
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
        top = strength.topk(min(top_neurons, live)).indices.tolist()
        lines.append(f"L{i} MLP: {live} live neurons, strongest {len(top)}:")
        for n in top:
            lines.append(f"  L{i}N{n} w{float(strength[n]):.2g} reads {_top_tokens(En2 @ fc[n], voc, k)}"
                         f" writes {_top_tokens(E @ down[:, n], voc, k)}")
    return lines


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


def behaviour_lines(toy: Toy, n_texts: int = 2) -> list[str]:
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
                lines.append(f"  L{i}H{j}: offset {mode:+d} at {share:.0%} of positions; last attends "
                             f"{toy.vocab[ids[kl]]}@{kl} {float(p[-1, kl]):.2f}")
        lines.append("  active at last: " + " ".join(active_units(toy, r, T - 1)))
    return lines


def render(toy: Toy, n_texts: int = 2, weights_from: Toy | None = None) -> str:
    """The prompt. weights_from: take the weight section from another toy (same architecture), keeping this toy's
    behaviour section (the swapped-weights control: does the answer follow the weights?)."""
    L, H, hd = heads_of(toy.cfg)
    head = (f"Toy transformer: {L} layers, {H} heads of dim {hd}, d_model {toy.cfg['n_embd']}, "
            f"vocab {len(toy.vocab)}, RoPE, tied embedding. Tokens: {' '.join(toy.vocab)}.\n"
            "Weights (QK: query>key token pairs scored through the embedding; OV: source>written token pairs; "
            "comp: Q/K/V composition with an earlier head's output):")
    return "\n".join([head, *weight_lines(weights_from or toy), "Behaviour:", *behaviour_lines(toy, n_texts),
                      "Which units implement the prediction at the last position, and how?", FORMAT])


def target_text(truth: dict) -> str:
    """The training completion for a ground truth."""
    body = "\n".join(f"  {a} --> {b}" for a, b in truth["edges"])
    return f"flowchart LR\n{body}\nEnglish: {truth.get('english', '').strip()}"


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


def load_zoo(manifest: str | Path, split: str | None = None) -> list[Toy]:
    """Every toy in a zoo manifest, or those of one split (train, test_instance, test_family)."""
    m = Path(manifest)
    rows = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
    return [from_zoo(r, m.parent) for r in rows if split is None or r.get("split") == split]


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
