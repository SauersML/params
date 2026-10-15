"""T1, the lead's pre-registered unit test (10-15 direction): are Qwen3-1.7B's own MLP neurons clean? 200 random
neurons, each pole explained by the strongest Tinker writer from its top examples, the explanations checked by
neuron_verify (blind maximum, near misses, distant concept, steering). Passing: blind maximum >= 0.3 of the corpus
maximum and near misses < 0.3; below 50% of neurons passing moves the main target to 8B. Cammarata's Qwen3-32B
reference: 76% / 11% near misses / 1% distant.

  t1_audit.py collect --out ATLAS.json [--tokens 10000000] [--window 1024] [--device D]
      one pass over the uncopyrighted Pile (monology/pile-uncopyrighted, streamed: web, code, books, papers, wiki;
      each document's first WINDOW tokens), recording only the sampled neurons: per pole its corpus maximum and top
      TOP examples (document, position, value, SwiGLU sign regime), the first token (the attention sink) left out
  t1_audit.py explain --atlas ATLAS.json --out NEURONS.jsonl   the writer's explanation of each pole
  (then neuron_verify.py verify --neurons NEURONS.jsonl --out T1.jsonl)
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/oracle/rl")]

MODEL = "Qwen/Qwen3-1.7B"
N, SEED, TOP = 200, 0, 20
BEFORE, AFTER = 96, 8  # tokens of context shown before and after an example's activating token
WRITER = "Qwen/Qwen3.5-397B-A17B"  # the strongest validated Tinker model (never an 8B, never Sol/codex, never Claude)


def sample(layers: int, width: int) -> list[tuple[int, int]]:
    """N neurons drawn uniformly from all layers' MLP neurons (seed SEED)."""
    return [divmod(k, width) for k in sorted(random.Random(SEED).sample(range(layers * width), N))]


def collect(a) -> None:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    L, W = len(m.model.layers), m.config.intermediate_size
    neurons = sample(L, W)
    by_layer: dict = {}
    for k, (l, i) in enumerate(neurons):
        by_layer.setdefault(l, []).append((k, i))
    cur: dict = {}
    hooks = []
    for l, ks in by_layer.items():
        idx = torch.tensor([i for _, i in ks], device=dev)
        mlp = m.model.layers[l].mlp
        hooks += [mlp.gate_proj.register_forward_hook(lambda _m, _i, o, l=l, idx=idx: cur.__setitem__(("g", l), o[..., idx].float())),
                  mlp.up_proj.register_forward_hook(lambda _m, _i, o, l=l, idx=idx: cur.__setitem__(("u", l), o[..., idx].float())),
                  mlp.down_proj.register_forward_pre_hook(lambda _m, x, l=l, idx=idx: cur.__setitem__(("a", l), x[0][..., idx].float()))]
    heaps = [{"+": [], "-": []} for _ in neurons]
    maxima = [{"+": 0.0, "-": 0.0} for _ in neurons]
    docs, ntok = [], 0
    stream = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    batch = []

    def run(batch):
        T = max(len(x) for x in batch)
        ids = torch.tensor([x + [tok.eos_token_id] * (T - len(x)) for x in batch], device=dev)
        mask = torch.tensor([[1] * len(x) + [0] * (T - len(x)) for x in batch], device=dev)
        with torch.no_grad():
            m(input_ids=ids, attention_mask=mask)
        valid = mask.bool().clone()
        valid[:, 0] = False  # the first token: the attention sink
        d0 = len(docs) - len(batch)
        for l, ks in by_layer.items():  # per layer: each neuron's TOP values of either pole in one top-k, moved once
            act = cur[("a", l)].masked_fill(~valid[..., None], 0.0).flatten(0, 1)  # (B T, k)
            g, u = cur[("g", l)].flatten(0, 1), cur[("u", l)].flatten(0, 1)
            for pole, v in (("+", act), ("-", -act)):
                top = v.topk(TOP, dim=0)  # (TOP, k)
                gs = g.gather(0, top.indices) > 0
                us = u.gather(0, top.indices) > 0
                vals, idxs, gs, us = top.values.cpu().numpy(), top.indices.cpu().numpy(), gs.cpu().numpy(), us.cpu().numpy()
                for j, (k, _) in enumerate(ks):
                    h = heaps[k][pole]
                    for r in range(TOP):
                        val = float(vals[r, j])
                        if val <= 0 or (len(h) == TOP and val <= h[0][0]):
                            break
                        b, pos = divmod(int(idxs[r, j]), T)
                        item = (val, d0 + b, pos, ("+" if gs[r, j] else "-") + ("+" if us[r, j] else "-"))
                        (heapq.heappush if len(h) < TOP else heapq.heapreplace)(h, item)
                    maxima[k][pole] = max(maxima[k][pole], float(vals[0, j]))

    for row in stream:
        x = tok(row["text"][:12 * a.window], add_special_tokens=False)["input_ids"][:a.window]  # a long document: its start only
        if len(x) < 64:
            continue
        docs.append(x)
        batch.append(x)
        ntok += len(x)
        if len(batch) == a.batch:
            run(batch)
            batch = []
            if len(docs) % (a.batch * 50) == 0:
                print(f"{ntok / 1e6:.1f}M tokens, {len(docs)} documents", flush=True)
        if ntok >= a.tokens:
            break
    if batch:
        run(batch)
    for h in hooks:
        h.remove()

    def show(d, p):
        x = docs[d]
        return {"before": tok.decode(x[max(0, p - BEFORE):p]), "token": tok.decode([x[p]]), "after": tok.decode(x[p + 1:p + 1 + AFTER])}

    out = []
    for k, (l, i) in enumerate(neurons):
        row = {"layer": l, "index": i, "max": maxima[k]}
        for pole in "+-":
            row[pole] = [{"value": v, "doc": d, "pos": p, "regime": r, **show(d, p)} for v, d, p, r in sorted(heaps[k][pole], reverse=True)]
        out.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": MODEL, "corpus": "monology/pile-uncopyrighted (streamed, first documents)", "window": a.window,
                                 "tokens": ntok, "documents": len(docs), "neurons": out}))
    np.save(a.out.with_suffix(".tokens.npy"), np.array([t for x in docs for t in x], dtype=np.int32))
    print(f"{len(out)} neurons over {ntok} tokens to {a.out}")


EXPLAIN = """Below are the strongest activations of one MLP neuron (layer {layer}, index {index}) in a language model, over a large mixed corpus. The activation is the input of the MLP's down projection, silu(gate) * up. Each example is the text before the token, the token itself in ⟦ ⟧, and a few tokens after; its activation value; and its sign regime, the signs of gate and up (++ and -- give positive values, -- being the small "echo" of a negative gate; +- and -+ give negative values).

POSITIVE pole (largest values; corpus maximum {pmax:.2f}):
{pos}

NEGATIVE pole (most negative values; corpus minimum -{nmax:.2f}):
{neg}

Explain each pole: what in the text drives the neuron to that pole, as specifically as the examples allow (the token, its context, the role it plays). Write JSON with two fields, "positive" and "negative", each 1 to 3 sentences, nothing else."""


def explain(a) -> None:
    import reader_reward

    atlas = json.loads(a.atlas.read_text())
    rd = reader_reward.Reader(a.writer, a.out.parent)

    def block(exs):
        return "\n".join(f"{j}. {e['value']:.2f} [{e['regime']}] ...{e['before']}⟦{e['token']}⟧{e['after']}" for j, e in enumerate(exs[:a.examples], 1))

    prompts = [EXPLAIN.format(layer=n["layer"], index=n["index"], pmax=n["max"]["+"], nmax=n["max"]["-"], pos=block(n["+"]), neg=block(n["-"]))
               for n in atlas["neurons"]]
    import neuron_verify

    rows = []
    for n, text in zip(atlas["neurons"], rd(prompts, max_tokens=600)):
        got = neuron_verify.parse_json(text) or {}
        for pole, key in (("+", "positive"), ("-", "negative")):
            if got.get(key) and n["max"][pole] > 0:
                rows.append({"layer": n["layer"], "index": n["index"], "pole": pole, "corpus_max": n["max"][pole], "explanation": got[key],
                             "regimes": [e["regime"] for e in n[pole][:a.examples]]})
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} poles explained to {a.out}; spend ${rd.spend.total:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tokens", type=int, default=10_000_000)
    c.add_argument("--window", type=int, default=1024)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--device")
    e = sub.add_parser("explain")
    e.add_argument("--atlas", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--writer", default=WRITER)
    e.add_argument("--examples", type=int, default=10)
    a = ap.parse_args()
    {"collect": collect, "explain": explain}[a.what](a)


if __name__ == "__main__":
    main()
