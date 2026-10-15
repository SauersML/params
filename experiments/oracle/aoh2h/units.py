"""Can the released Activation Oracle read single units? (#2951, the lead's 10-14: inject a unit's residual footprint
into the Qwen3-1.7B oracle adapter, zero-shot, and score its description.)

Vectors, from the model's run on a text (native units of HF Qwen3: an MLP neuron after its nonlinearity, the down
projection's input; an attention head's output, the output projection's input, per head):
  neuron write   activation x the neuron's down_proj column, at its position (a residual vector)
  neuron read    the neuron's gate_proj row (the residual direction it reads, after the layer's norm)
  head write     the head's output through its o_proj slice, at its position
  message        a head moving information from p to q: attention weight (q -> p) x (value at p through the head's
                 o_proj slice), the residual vector at q attributable to p (the cross-token connection)
Each vector is injected alone as the oracle injects activations ("Layer: N\\n ? \\n<question>", normalized, scaled to the
norm at the placeholder, added at layer 1's output; N the nearest of the adapter's trained layers 7, 14, 21), K answers
sampled to DESCRIBE. Units here (no builder yet): per text, at the last position the TOP_NEURONS neurons and TOP_HEADS
heads of largest write norm in layers from a quarter of the depth on, and each such head's message from the earlier
position it attends to most (position 0, the attention sink, left out); the builder's circuit units replace them when
it runs on Qwen3.

Labels, PheWAS-style, from LABEL_TEXTS other texts: a neuron's top-activating tokens, a head's tokens of largest write
norm; a message's own source token. Score: an answer's content words hitting the unit's label tokens, against the same
answers scored on another unit's labels (null: units shuffled within type).

  units.py run --out OUT.jsonl [--texts 20] [--model Qwen/Qwen3-1.7B] [--device cuda]
  units.py score OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ao_verify  # noqa: E402
import h2h  # noqa: E402

TOP_NEURONS, TOP_HEADS = 8, 4
K = 4
LABEL_TEXTS, LABEL_TOP = 400, 20
DESCRIBE = "Answer in a few words. What is this?"
TRAINED = (7, 14, 21)  # the released adapter's activation layers (25, 50, 75% of 28)


class Taps:
    """Per layer, the down projection's input [T, I], the output projection's input [T, H x hd] and the value
    projection's output [T, KV x hd] of one run, and the attention weights [H, T, T] (eager attention)."""

    def __init__(self, tg):
        self.tg, self.data = tg, {}
        self.handles = []
        for l, layer in enumerate(tg.layers):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._pre(l, "mlp")))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._pre(l, "heads")))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._post(l, "v")))

    def _pre(self, l, key):
        def fn(_m, args):
            self.data[(l, key)] = args[0][0].detach()
        return fn

    def _post(self, l, key):
        def fn(_m, _i, out):
            self.data[(l, key)] = out[0].detach()
        return fn

    def run(self, ids: list[int]):
        torch = self.tg.torch
        self.data = {}
        with self.tg.use(None), torch.no_grad():
            out = self.tg.m(input_ids=torch.tensor([ids], device=self.tg.dev), output_attentions=True)
        for l, a in enumerate(out.attentions):
            self.data[(l, "attn")] = a[0].float()
        return self.data

    def close(self):
        for h in self.handles:
            h.remove()


def weights(tg):
    """Per layer, (down_proj weight [d, I], o_proj weight [d, H x hd], gate_proj weight [I, d]) and the head sizes."""
    cfg = tg.m.base_model.model.config
    W = [(l.mlp.down_proj.weight, l.self_attn.o_proj.weight, l.mlp.gate_proj.weight) for l in tg.layers]
    return W, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim


def vectors(tg, ids: list[int], taps: Taps) -> list[dict]:
    """The text's units and their vectors (module docstring), each {"kind", "layer", "index", "position", "vec",
    ("source")}."""
    torch = tg.torch
    W, H, KV, hd = weights(tg)
    d = taps.run(ids)
    T, q = len(ids), len(ids) - 1
    out = []
    for l in range(len(tg.layers) // 4, len(tg.layers)):
        Wd, Wo, Wg = W[l]
        a = d[(l, "mlp")][q].float()
        norm = a.abs() * Wd.float().norm(dim=0)
        for n in norm.topk(TOP_NEURONS).indices.tolist():
            out.append({"kind": "neuron_write", "layer": l, "index": n, "position": q, "vec": a[n] * Wd[:, n].float(), "score": float(norm[n])})
            out.append({"kind": "neuron_read", "layer": l, "index": n, "position": q, "vec": Wg[n].float(), "score": float(norm[n])})
        o = d[(l, "heads")][q].float().view(H, hd)
        writes = torch.stack([o[h] @ Wo[:, h * hd:(h + 1) * hd].float().T for h in range(H)])
        v = d[(l, "v")].float().view(T, KV, hd)
        A = d[(l, "attn")]
        for h in writes.norm(dim=-1).topk(TOP_HEADS).indices.tolist():
            out.append({"kind": "head_write", "layer": l, "index": h, "position": q, "vec": writes[h], "score": float(writes[h].norm())})
            att = A[h, q].clone()
            att[0] = -1.0
            att[q] = -1.0
            p = int(att.argmax())
            msg = float(A[h, q, p]) * (v[p, h // (H // KV)] @ Wo[:, h * hd:(h + 1) * hd].float().T)
            out.append({"kind": "message", "layer": l, "index": h, "position": q, "source": p, "vec": msg, "score": float(msg.norm())})
    # the strongest per layer kept: the TOP_NEURONS neurons and TOP_HEADS heads of largest score over all layers
    keep = []
    for kind, top in (("neuron_write", TOP_NEURONS), ("head_write", TOP_HEADS), ("message", TOP_HEADS)):
        keep += sorted((u for u in out if u["kind"] == kind), key=lambda u: -u["score"])[:top]
    chosen = {(u["layer"], u["index"]) for u in keep if u["kind"] == "neuron_write"}
    keep += [u for u in out if u["kind"] == "neuron_read" and (u["layer"], u["index"]) in chosen]
    return keep


def labels(tg, taps: Taps, units: list[dict], corpus: list[list[int]]) -> dict:
    """Per neuron its LABEL_TOP top-activating tokens over the corpus; per head its tokens of largest write norm."""
    torch = tg.torch
    W, H, KV, hd = weights(tg)
    best = {}
    for ids in corpus:
        d = taps.run(ids)
        for u in units:
            key = (u["kind"].split("_")[0], u["layer"], u["index"])
            if key[0] == "neuron":
                vals = d[(u["layer"], "mlp")][:, u["index"]].float()
            elif key[0] in ("head", "message"):
                o = d[(u["layer"], "heads")].float().view(len(ids), H, hd)[:, u["index"]]
                vals = o.norm(dim=-1)
            else:
                continue
            lst = best.setdefault(key, [])
            lst += list(zip(vals.tolist(), ids))
            lst.sort(key=lambda x: -x[0])
            del lst[LABEL_TOP:]
    return {k: [tg.tok.decode([t]).strip() for _, t in v] for k, v in best.items()}


def run(a) -> None:
    tg = ao_verify.Target(a.model, a.device)
    base = tg.m.base_model.model
    if hasattr(base, "set_attn_implementation"):
        base.set_attn_implementation("eager")
    else:
        base.config._attn_implementation = "eager"
    taps, rng = Taps(tg), random.Random(5)
    texts = h2h.texts(tg.tok, rng, a.texts + LABEL_TEXTS)
    corpus, items = texts[a.texts:], texts[:a.texts]
    rows = []
    for i, ids in enumerate(items):
        for u in vectors(tg, ids, taps):
            lab = min(TRAINED, key=lambda x: abs(x - u["layer"]))
            tg.act_layer, keep = lab, tg.act_layer
            try:
                answers = tg.inject(DESCRIBE, u["vec"].to(tg.torch.bfloat16)[None], K, True, adapter_ao=True)
            finally:
                tg.act_layer = keep
            rows.append({"text": i, "kind": u["kind"], "layer": u["layer"], "index": u["index"], "position": u["position"],
                         "source": u.get("source"), "source_token": tg.tok.decode([ids[u["source"]]]).strip() if "source" in u else None,
                         "position_token": tg.tok.decode([ids[u["position"]]]).strip(), "answers": answers})
        print(i, len(rows), rows[-1]["kind"], rows[-1]["answers"][:2], flush=True)
    lab = labels(tg, taps, [{"kind": r["kind"], "layer": r["layer"], "index": r["index"]} for r in rows], corpus)
    for r in rows:
        r["labels"] = lab.get((r["kind"].split("_")[0], r["layer"], r["index"]), [])
    taps.close()
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} unit answers to {a.out}")


def hits(answers: list[str], tokens: list[str]) -> bool:
    toks = {h2h.norm(t) for t in tokens if h2h.norm(t)}
    return any(h2h.norm(w) in toks for x in answers for w in ao_verify.words(x))


def score(a) -> None:
    rows = [json.loads(x) for x in a.out.read_text().splitlines()]
    rng = random.Random(0)
    for kind in ("neuron_write", "neuron_read", "head_write", "message"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        lab = [r["source_token"] and [r["source_token"]] or r["labels"] for r in rs] if kind == "message" else [r["labels"] for r in rs]
        own = sum(hits(r["answers"], l) for r, l in zip(rs, lab)) / len(rs)
        null = []
        for _ in range(20):
            perm = lab[:]
            rng.shuffle(perm)
            null.append(sum(hits(r["answers"], l) for r, l in zip(rs, perm)) / len(rs))
        posw = sum(hits(r["answers"], [r["position_token"]]) for r in rs) / len(rs)
        print(f"{kind}: n {len(rs)}; answers hit the unit's labels {own:.2f} vs shuffled {sum(null) / len(null):.2f} "
              f"(max {max(null):.2f}); name the position's own token {posw:.2f}; e.g. {Counter(w for r in rs for x in r['answers'] for w in ao_verify.words(x)[:1]).most_common(5)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--texts", type=int, default=20)
    r.add_argument("--model", default=h2h.MODEL)
    r.add_argument("--device")
    s = sub.add_parser("score")
    s.add_argument("out", type=Path)
    a = ap.parse_args()
    {"run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
