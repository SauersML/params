"""T1, the lead's pre-registered unit test (10-15 direction): are Qwen3-1.7B's own MLP neurons clean? 200 random
neurons, each pole explained by the strongest Tinker writer from its top examples, the explanations checked by
neuron_verify (blind maximum, near misses, distant concept, steering). Passing: blind maximum >= 0.3 of the corpus
maximum and near misses < 0.3; below 50% of neurons passing moves the main target to 8B. Cammarata's Qwen3-32B
reference: 76% / 11% near misses / 1% distant.

  t1_audit.py collect --out ATLAS.json [--model M] [--corpus ROOT] [--tokens 10000000] [--window 1024] [--device D]
      one pass over the atlas corpus ROOT (atlas.py corpus: web, Pile, chat, code, five languages, math, in a fixed
      order) or, without --corpus, the streamed uncopyrighted Pile (each document's first WINDOW tokens), recording
      only the sampled neurons: per pole its corpus maximum and top TOP examples (document, position, value, SwiGLU
      sign regime), the first token (the attention sink) left out
  t1_audit.py explain --atlas ATLAS.json --out NEURONS.jsonl   the writer's explanation of each pole
  (then neuron_verify.py verify --neurons NEURONS.jsonl --out T1.jsonl, or its plan, measure and judge steps)
  t1_audit.py figure --rows T1_1p7b.jsonl T1_8b.jsonl ... [--writers NAME=FILE ...] --out FIG.png   pass rate by
      model size, and on the same neurons by writer
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
BEFORE, AFTER = 256, 32  # tokens of context before and after an example's activating token (the lead's 10-15: short
# contexts make long-context neurons look polysemantic, Cammarata's "summary" neurons firing on commas)
WRITER = "Qwen/Qwen3.5-397B-A17B"  # the strongest validated Tinker model (never an 8B, never Sol/codex, never Claude)


def sample(layers: int, width: int) -> list[tuple[int, int]]:
    """N neurons drawn uniformly from all layers' MLP neurons (seed SEED)."""
    return [divmod(k, width) for k in sorted(random.Random(SEED).sample(range(layers * width), N))]


def collect(a) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.device == "auto":  # layers spread over every visible GPU (Qwen3-32B on two 48 GB cards)
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
        dev = m.get_input_embeddings().weight.device
    else:
        dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    L, W = len(m.model.layers), m.config.intermediate_size
    neurons = sample(L, W)
    by_layer: dict = {}
    for k, (l, i) in enumerate(neurons):
        by_layer.setdefault(l, []).append((k, i))
    cur: dict = {}
    hooks = []
    for l, ks in by_layer.items():
        idx = torch.tensor([i for _, i in ks], device=m.model.layers[l].mlp.down_proj.weight.device)  # the layer's own GPU
        mlp = m.model.layers[l].mlp
        hooks += [mlp.gate_proj.register_forward_hook(lambda _m, _i, o, l=l, idx=idx: cur.__setitem__(("g", l), o[..., idx].float())),
                  mlp.up_proj.register_forward_hook(lambda _m, _i, o, l=l, idx=idx: cur.__setitem__(("u", l), o[..., idx].float())),
                  mlp.down_proj.register_forward_pre_hook(lambda _m, x, l=l, idx=idx: cur.__setitem__(("a", l), x[0][..., idx].float()))]
    heaps = [{"+": [], "-": []} for _ in neurons]
    maxima = [{"+": 0.0, "-": 0.0} for _ in neurons]
    docs, ntok = [], 0
    if a.corpus:  # the atlas corpus (fleet's atlas.py corpus ROOT): documents' token ids in its fixed order
        toks = np.load(a.corpus / "corpus" / "tokens.npy", mmap_mode="r")
        offs = np.load(a.corpus / "corpus" / "offsets.npy")
        stream = ({"ids": toks[offs[d]:offs[d + 1]].tolist()} for d in range(len(offs) - 1))
    else:
        from datasets import load_dataset

        stream = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    batch = []

    def run(batch):
        T = max(len(x) for x in batch)
        ids = torch.tensor([x + [tok.eos_token_id] * (T - len(x)) for x in batch], device=dev)
        mask = torch.tensor([[1] * len(x) + [0] * (T - len(x)) for x in batch], device=dev)
        with torch.no_grad():
            # Only MLP hooks are read: skip unused vocabulary logits and KV caches.
            m.model(input_ids=ids, attention_mask=mask, use_cache=False)
        valid = mask.bool().clone()
        valid[:, 0] = False  # the first token: the attention sink
        d0 = len(docs) - len(batch)
        for l, ks in by_layer.items():  # per layer: each neuron's TOP values of either pole in one top-k, moved once
            act = cur[("a", l)]
            act = act.masked_fill(~valid.to(act.device)[..., None], 0.0).flatten(0, 1)  # (B T, k), on the layer's GPU
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
        x = row["ids"][:a.window] if "ids" in row else \
            tok(row["text"][:12 * a.window], add_special_tokens=False)["input_ids"][:a.window]  # a long document: its start only
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
    a.out.write_text(json.dumps({"model": a.model, "corpus": str(a.corpus) if a.corpus else "monology/pile-uncopyrighted (streamed, first documents)", "window": a.window,
                                 "tokens": ntok, "documents": len(docs), "neurons": out}))
    np.save(a.out.with_suffix(".tokens.npy"), np.array([t for x in docs for t in x], dtype=np.int32))
    np.save(a.out.with_suffix(".offsets.npy"), np.cumsum([0] + [len(x) for x in docs]).astype(np.int64))  # document d: tokens[off[d]:off[d + 1]]
    print(f"{len(out)} neurons over {ntok} tokens to {a.out}")


EXPLAIN = """Below are the strongest activations of one MLP neuron (layer {layer}, index {index}) in a language model, over a large mixed corpus. The activation is the input of the MLP's down projection, silu(gate) * up. Each example is the text before the token, the token itself in ⟦ ⟧, and a few tokens after; its activation value; and its sign regime, the signs of gate and up (++ and -- give positive values; -- has negative gate and up; +- and -+ give negative values). Size and causal relevance must be measured, not assumed.

POSITIVE pole (largest values; corpus maximum {pmax:.2f}):
{pos}

NEGATIVE pole (most negative values; corpus minimum -{nmax:.2f}):
{neg}

Explain each pole: what in the text drives the neuron to that pole, as specifically as the examples allow (the token, its context, the role it plays). Write JSON with two fields, "positive" and "negative", each 1 to 3 sentences, nothing else."""
# v2: the same examples, an instruction aimed at a reader who must write a new text from the explanation alone (8B
# failures: a press-release byline read as "function words", byte tokens of CJK characters described as the symbol �)
EXPLAIN_V2 = EXPLAIN[:EXPLAIN.index("Explain each pole:")] + """Explain each pole so that a reader who sees only your explanation can write a new text that drives the neuron to that pole. Say (1) the exact token or tokens where it fires; (2) the context that must surround them: the kind of document, the topic, the phrase or structure just before, and how far back it reaches; (3) what the examples share that a surface reading misses (for example, a byline in a press release rather than the word "is"). A token shown as � is one byte of a multi-byte character: name the script and which byte, not the symbol. Prefer one specific shared cause over a list of unrelated tokens. Write JSON with two fields, "positive" and "negative", each 1 to 4 sentences, nothing else."""
PROMPTS = {"v1": EXPLAIN, "v2": EXPLAIN_V2}


def explain(a) -> None:
    import hashlib
    import math
    import numbers
    import reader_reward

    atlas = json.loads(a.atlas.read_text())

    def block(exs):
        return "\n".join(f"{j}. {e['value']:.2f} [{e['regime']}] ...{e['before']}⟦{e['token']}⟧{e['after']}" for j, e in enumerate(exs[:a.examples], 1))

    prompts = [PROMPTS[getattr(a, "prompt", "v1")].format(layer=n["layer"], index=n["index"], pmax=n["max"]["+"], nmax=n["max"]["-"], pos=block(n["+"]), neg=block(n["-"]))
               for n in atlas["neurons"]]
    import neuron_verify

    rd = reader_reward.Reader(a.writer, a.out.parent)
    responses = rd(prompts, max_tokens=600)
    audit_path = a.out.with_suffix(a.out.suffix + ".writer.json")
    audit = {"schema": 1, "atlas_sha256": hashlib.sha256(a.atlas.read_bytes()).hexdigest(),
             "writer_model": a.writer, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "expected_neurons": len(atlas["neurons"]), "expected_poles": 2 * len(atlas["neurons"]),
             "requests": [{"layer": n["layer"], "index": n["index"], "prompt": p,
                           "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()} for n, p in zip(atlas["neurons"], prompts)],
             "raw_responses": responses, "response_count": len(responses) if isinstance(responses, (list, tuple)) else None}
    audit_path.write_text(json.dumps(audit, allow_nan=False) + "\n")
    if not isinstance(responses, (list, tuple)) or len(responses) != len(prompts):
        raise ValueError("writer response count must match every sampled neuron; raw audit retained")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    rows = []
    for n, prompt, text in zip(atlas["neurons"], prompts, responses):
        got = neuron_verify.parse_json(text) if isinstance(text, str) else None
        got = got if isinstance(got, dict) else {}
        for pole, key in (("+", "positive"), ("-", "negative")):
            explanation, maximum = got.get(key), n["max"][pole]
            errors = []
            if not isinstance(explanation, str) or not explanation.strip():
                errors.append("missing_or_invalid_explanation")
            if isinstance(maximum, bool) or not isinstance(maximum, numbers.Real) or not math.isfinite(maximum) or maximum <= 0:
                errors.append("nonpositive_or_invalid_corpus_max")
            row = {"layer": n["layer"], "index": n["index"], "pole": pole, "corpus_max": maximum,
                   "explanation": explanation, "regimes": [e["regime"] for e in n[pole][:a.examples]],
                   "writer_valid": not errors, "writer_errors": errors, "writer_model": a.writer,
                   "writer_request": prompt, "writer_raw_response": text, "writer_audit_sha256": audit_sha}
            if errors:
                row.update(valid=False, passes_t1=False)
            rows.append(row)
    a.out.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    print(f"{len(rows)} sampled poles retained ({sum(r['writer_valid'] for r in rows)} writer-valid) to {a.out}; spend ${rd.spend.total:.3f}")


def figure(a) -> None:
    """Left: the share of explained poles passing T1 against model size (--rows, one judged file a size), with 95%
    Wilson intervals and the 50% bar. Right: the same neurons explained by different writers (--writers NAME=FILE).
    Poles that never fire on the corpus are left out, as in neuron_verify summary."""
    import math
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def wilson(k, n, z=1.96):
        p, d = k / n, 1 + z * z / n
        c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return k / n, c - h, c + h

    def passing(f):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        rows = [r for r in rows if isinstance(r.get("corpus_max"), (int, float)) and r["corpus_max"] > 0]
        name = next(r["model_binding"]["requested_model"] for r in rows if r.get("model_binding"))
        return name, wilson(sum(bool(r.get("valid") and r.get("passes_t1")) for r in rows), len(rows)), len(rows)

    plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="white", gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
    sizes, got = [], []
    for f in a.rows:
        name, w, n = passing(f)
        sizes.append(float(re.search(r"(\d+(?:\.\d+)?)B", name)[1]))
        got.append(w)
        print(f"{name}: {n} poles; pass {w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}]")
    ax.errorbar(sizes, [g[0] for g in got], yerr=[[g[0] - g[1] for g in got], [g[2] - g[0] for g in got]], color="#c44e52",
                marker="o", ms=11, lw=2.5, capsize=6)
    for x, g in zip(sizes, got):
        ax.annotate(f"{g[0]:.0%}", (x, g[2]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=17)
    ax.axhline(0.5, color="gray", ls="--", lw=1.5)
    ax.text(sizes[-1] * 1.25, 0.5, "pre-registered\nbar", color="gray", fontsize=15, va="center")
    ax.set(xscale="log", ylim=(0, 1), xlabel="Qwen3 model size (billions of parameters)", ylabel="share of neuron poles passing")
    ax.set_xticks(sizes, [f"{s:g}B" for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(min(sizes) / 1.6, max(sizes) * 2.6)
    ax.set_title("Model size, the 397B as writer", fontsize=18, loc="left")
    names, ws = [], []
    for spec in a.writers or []:
        label, f = spec.split("=", 1)
        _, w, n = passing(Path(f))
        names.append(label)
        ws.append(w)
        print(f"writer {label}: {n} poles; pass {w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}]")
    xs = range(len(ws))
    bx.bar(xs, [w[0] for w in ws], 0.6, color=["#c44e52", "#e8a3a5"][:len(ws)],
           yerr=[[w[0] - w[1] for w in ws], [w[2] - w[0] for w in ws]], capsize=6)
    for x, w in zip(xs, ws):
        bx.text(x, w[2] + 0.03, f"{w[0]:.0%}", ha="center", fontsize=17)
    bx.axhline(0.5, color="gray", ls="--", lw=1.5)
    bx.set_xticks(list(xs), names)
    bx.set(ylim=(0, 1), ylabel="share of neuron poles passing")
    bx.set_title("Qwen3-1.7B, two writers", fontsize=18, loc="left")
    if a.title:
        fig.suptitle(a.title, fontsize=21, x=0.06, ha="left", y=1.02)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"figure to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tokens", type=int, default=10_000_000)
    c.add_argument("--window", type=int, default=1024)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--device", help="cuda, mps, cpu, or auto (layers over every visible GPU)")
    c.add_argument("--corpus", type=Path, help="an atlas corpus root (atlas.py corpus ROOT): its documents in order, not the streamed Pile")
    c.add_argument("--model", default=MODEL, help="Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, Qwen/Qwen3-32B: the same audit at each size")
    e = sub.add_parser("explain")
    e.add_argument("--atlas", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--writer", default=WRITER)
    e.add_argument("--examples", type=int, default=10)
    e.add_argument("--prompt", choices=sorted(PROMPTS), default="v1", help="the writer's instruction (v1: T1's pre-registered one)")
    f = sub.add_parser("figure")
    f.add_argument("--rows", type=Path, nargs="+", required=True, help="judged T1 rows, one file per model size")
    f.add_argument("--writers", nargs="+", help="NAME=FILE: judged rows of the same neurons explained by each writer")
    f.add_argument("--title")
    f.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"collect": collect, "explain": explain, "figure": figure}[a.what](a)


if __name__ == "__main__":
    main()
