"""Explanation verifiers for a model's own neurons (#2951, the lead's 10-15 direction after Cammarata's "Neural
Networks Are Already Clean"): each check costs about one target forward pass, and together they are the oracle's
explanation reward.

A neuron is (layer, index) of a Qwen3 MLP; its activation at a position is the input of down_proj,
a = silu(gate) * up. Its positive pole is how high a goes, its negative pole how low; the SwiGLU sign regime at a
position is (sign gate, sign up): ++ and -- give a > 0 (-- is the small "echo" of a negative gate), +- and -+ give
a < 0. An explanation is checked for one pole against that pole's corpus maximum (the atlas's largest value of a, or
of -a, over its corpus):
  blind maximum: a reader that sees only the explanation writes texts meant to drive the pole as far as it goes; the
    score is the best text's pole value (its largest over positions) over the corpus maximum;
  near misses: the reader writes NEAR texts close to the explanation's subject that should not fire the pole; the
    near-miss rate is the share whose pole value reaches FIRE of the corpus maximum;
  distant concept: the reader names the concept farthest from the explanation and writes DISTANT texts about it; the
    share reaching FIRE should be about 0;
  steering prediction: the explanation's prediction of what the model writes with the pole turned up (the neuron's
    activation set to the pole's sign times STEER corpus maxima at every position) is shown with that generation and
    an unsteered one, in random order, and the reader picks the one the prediction describes (chance 0.5).
T1 (the lead's pre-registered test): a neuron's explanation passes when blind maximum >= 0.3 and near misses < 0.3.

  neuron_verify.py verify --neurons N.jsonl --out OUT.jsonl [--model Qwen/Qwen3-1.7B] [--reader R] [--device D]
      N.jsonl lines {"layer", "index", "pole": "+"|"-", "corpus_max", "explanation"}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "rl")]

FIRE = 0.3  # a text fires a pole when its pole value reaches this share of the corpus maximum
NEAR, DISTANT, BLIND = 5, 5, 3
STEER = 3.0  # steering sets the activation to the pole's sign times this many corpus maxima
PROMPTS = ["The", "In my opinion,", "Here is a short story:", "Today I learned that"]  # steering's neutral starts
GEN = 24  # tokens generated per steering prompt
READER = "Qwen/Qwen3.6-27B"  # dense, so deterministic at temperature 0 (reader_reward: 10-14 repeats agree to 0.003)

ASK = """A neuron in a language model has this explanation of its {pole} side:
{explanation}

Write JSON with these fields, nothing else:
"max_texts": {blind} different texts of 1 to 3 sentences, each written to drive this side of the neuron as strongly as possible;
"near_misses": {near} texts on closely related subjects that, by the explanation, should NOT drive it;
"distant_concept": the concept farthest from what the explanation describes;
"distant_texts": {distant} texts about that distant concept;
"steering_prediction": one sentence saying what the model would write if this side of the neuron were turned up strongly while it continues ordinary prompts."""

JUDGE = """Prediction of how a language model's writing changes when one of its neurons is turned up:
{prediction}

Two continuations of the same prompts; one was written with the neuron turned up.
A: {a}
B: {b}

Which continuation does the prediction describe? Answer A or B."""


class Target:
    """A Qwen3 model with a neuron's activation read (the input of down_proj, with gate's and up's signs) and set."""

    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"  # each text's own positions first, the padding after (masked out)
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(self.dev).eval()
        self.layers = self.m.model.layers

    def poles(self, texts: list[str], layer: int, index: int) -> list[dict]:
        """Per text, its largest activation and its largest negated activation over positions (the first token, the
        attention sink, left out), with the sign regime at each."""
        torch = self.torch
        mlp = self.layers[layer].mlp
        seen: dict = {}
        hs = [mlp.gate_proj.register_forward_hook(lambda _m, _i, o: seen.__setitem__("gate", o[..., index].float())),
              mlp.up_proj.register_forward_hook(lambda _m, _i, o: seen.__setitem__("up", o[..., index].float())),
              mlp.down_proj.register_forward_pre_hook(lambda _m, a: seen.__setitem__("a", a[0][..., index].float()))]
        out = []
        try:
            for s in range(0, len(texts), 16):
                enc = self.tok(texts[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
                with torch.no_grad():
                    self.m(**enc)
                mask = enc["attention_mask"].bool().clone()
                mask[:, 0] = False
                for j in range(mask.shape[0]):
                    a, g, u = seen["a"][j][mask[j]], seen["gate"][j][mask[j]], seen["up"][j][mask[j]]
                    if a.numel() == 0:
                        out.append({"pos": 0.0, "neg": 0.0, "regime_pos": None, "regime_neg": None})
                        continue
                    ip, ineg = int(a.argmax()), int(a.argmin())
                    reg = lambda i: ("+" if g[i] > 0 else "-") + ("+" if u[i] > 0 else "-")  # noqa: E731
                    out.append({"pos": float(a[ip]), "neg": float(-a[ineg]), "regime_pos": reg(ip), "regime_neg": reg(ineg)})
        finally:
            for h in hs:
                h.remove()
        return out

    def steered(self, layer: int, index: int, value: float | None) -> str:
        """The model's greedy continuations of PROMPTS with the neuron's activation set to value at every position
        (None: unsteered), joined."""
        torch = self.torch
        h = None
        if value is not None:
            def pre(_m, a):
                x = a[0].clone()
                x[..., index] = value
                return (x,) + tuple(a[1:])
            h = self.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
        try:
            outs = []
            for p in PROMPTS:
                ids = self.tok(p, return_tensors="pt", add_special_tokens=False).to(self.dev)
                with torch.no_grad():
                    g = self.m.generate(**ids, max_new_tokens=GEN, do_sample=False, temperature=None, top_p=None, top_k=None,
                                        pad_token_id=self.tok.eos_token_id)
                outs.append(p + self.tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
        finally:
            if h is not None:
                h.remove()
        return " / ".join(outs)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def verify(tg: Target, rd, items: list[dict], seed: int = 0) -> list[dict]:
    """The four checks for each item {"layer", "index", "pole", "corpus_max", "explanation"} (module docstring); the
    reader's calls for all items made together."""
    asks = [ASK.format(pole="positive" if it["pole"] == "+" else "negative", explanation=it["explanation"], blind=BLIND, near=NEAR,
                       distant=DISTANT) for it in items]
    plans = [parse_json(t) for t in rd(asks, max_tokens=1500)]
    out, judges = [], []
    rng = random.Random(seed)
    for it, plan in zip(items, plans):
        row = {k: it[k] for k in ("layer", "index", "pole", "corpus_max")}
        if not plan:
            out.append({**row, "valid": False})
            judges.append(None)
            continue
        key = "pos" if it["pole"] == "+" else "neg"
        cm = max(float(it["corpus_max"]), 1e-6)
        groups = {k: [str(x) for x in (plan.get(k) or [])][:n] for k, n in (("max_texts", BLIND), ("near_misses", NEAR), ("distant_texts", DISTANT))}
        flat = [x for k in groups for x in groups[k]]
        vals = tg.poles(flat, it["layer"], it["index"]) if flat else []
        k0 = 0
        got = {}
        for k in groups:
            got[k] = [v[key] / cm for v in vals[k0:k0 + len(groups[k])]]
            k0 += len(groups[k])
        blind = max(got["max_texts"], default=0.0)
        near = sum(x >= FIRE for x in got["near_misses"]) / max(len(got["near_misses"]), 1)
        dist = sum(x >= FIRE for x in got["distant_texts"]) / max(len(got["distant_texts"]), 1)
        value = (1.0 if it["pole"] == "+" else -1.0) * STEER * cm
        on, off = tg.steered(it["layer"], it["index"], value), tg.steered(it["layer"], it["index"], None)
        flip = rng.random() < 0.5
        judges.append((JUDGE.format(prediction=plan.get("steering_prediction", ""), a=off if flip else on, b=on if flip else off), "B" if flip else "A"))
        out.append({**row, "valid": True, "blind_max": blind, "blind_texts": groups["max_texts"], "blind_values": got["max_texts"],
                    "near_miss_rate": near, "near_values": got["near_misses"], "distant_concept": plan.get("distant_concept"),
                    "distant_rate": dist, "distant_values": got["distant_texts"], "steering_prediction": plan.get("steering_prediction"),
                    "steered": on, "unsteered": off, "passes_t1": blind >= FIRE and near < FIRE})
    asked = [j for j in judges if j]
    answers = iter(rd([q for q, _ in asked], max_tokens=5)) if asked else iter([])
    for row, j in zip(out, judges):
        if j:
            ans = next(answers).strip().upper()[:1]
            row["steer_match"] = float(ans == j[1])
    return out


def cmd_verify(a) -> None:
    import reader_reward

    items = [json.loads(x) for x in a.neurons.read_text().splitlines()]
    tg = Target(a.model, a.device)
    rd = reader_reward.Reader(a.reader, a.out.parent)
    rows = verify(tg, rd, items)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if r.get("valid")]
    n = max(len(ok), 1)
    print(f"{len(ok)}/{len(rows)} valid; blind max >= {FIRE}: {sum(r['blind_max'] >= FIRE for r in ok) / n:.2f}; near-miss rate "
          f"{sum(r['near_miss_rate'] for r in ok) / n:.2f}; distant rate {sum(r['distant_rate'] for r in ok) / n:.2f}; steering match "
          f"{sum(r.get('steer_match', 0) for r in ok) / n:.2f}; T1 pass {sum(r['passes_t1'] for r in ok) / n:.2f}; spend ${rd.spend.total:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--neurons", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--model", default="Qwen/Qwen3-1.7B")
    v.add_argument("--reader", default=READER)
    v.add_argument("--device")
    a = ap.parse_args()
    {"verify": cmd_verify}[a.what](a)


if __name__ == "__main__":
    main()
