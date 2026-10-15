"""Qwen3-32B validation (the lead's 10-15 update): Cammarata's two published Qwen3-32B neuron results ("Neural Networks
Are Already Clean", Goodfire 2026; the figure data at https://already-clean.goodfire.com/data/graph-carousel-hover.json
gives each graph's neurons as layer:index and pole) reproduced on the model with our removal, then found by our ranking.
Cases, on the essay's exact prompts:
  santa         "Remove the eight neurons with a Santa negative side and Santa falls from near-certain to 6%, with Qwen
                blaming the pet instead. Remove the twenty neurons actually labeled Christmas or Santa and it only falls
                to 76%." The eight are the graph's two all-negative groups ("Santa lore" and "Santa Claus comes next");
                the figure holds 10 of the twenty Christmas/Santa-labelled neurons.
  preeclampsia  "The diagnosis hub is just four neurons, and if you remove them the model asks about itching instead."
                The four are the graph's "preeclampsia" group.
Per case: the next-token distribution at the last position (y its top token; the contrast log2 p(y) - log2 of the summed
p of the ALTERNATIVES likeliest others). A unit is one pole of one neuron (its values of one sign, the down projection's
input act(gate) x up, at every position); removal zeroes them. The essay's sets are removed as poles and as whole
neurons, beside RANDOM draws of as many random poles (any neuron, and neurons among the prompt's strongest). Then ours,
blind to the essay: every pole ranked by |d contrast / d pole| (one backward pass, the order only), the SHORT largest
removed alone and measured, then greedy forward selection up to STEPS units, each step adding the measured best of the
GREEDY best singles; the contrast, p(y) and the top tokens after each step, and the essay units' ranks. With --thr (a
neuron_builder.Neurons.save file) every run is in the weak-zeroed reference instead of the model.

  neuron_essay_check.py OUT [--model Qwen/Qwen3-32B] [--device auto] [--dtype bfloat16] [--thr FILE]
writes OUT/<case>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))

import neuron_builder as nb  # noqa: E402

CASES = {
    "santa": {
        "text": "<|im_start|>user\nRead this short mystery and answer the question.\n\nOn the morning of December 25th, the "
                "Miller family came downstairs to a strange scene. The plate of cookies by the fireplace had been eaten down "
                "to crumbs, the glass of milk beside it was empty, there was soot on the rug below the chimney, and the "
                "neighbors said they heard sleigh bells on the roof around midnight.\n\nWho is responsible?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\nThe person responsible is **",
        "tokens": 99,
        "sets": {"santa_negative_8": ["27:13632-", "36:3541-", "38:6523-", "38:23931-", "30:7725-", "58:7126-", "51:13632-",
                                     "58:24534-"],
                 "christmas_labelled_10": ["14:11264+", "44:2915+", "38:11264+", "7:17263-", "44:20323+", "57:24294+",
                                           "57:6886+", "63:17307+", "61:16038+", "60:15297+"]}},
    "preeclampsia": {
        "text": "Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild "
                "headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.\n\nIf we can only "
                "ask about one other symptom, we should ask whether she's experiencing...\n\nAssistant: We should ask "
                "whether she's experiencing **",
        "tokens": 79,
        "sets": {"diagnosis_hub_4": ["31:5482+", "35:3336+", "35:20744+", "7:2026+"]}},
}
ALTERNATIVES = 5
ROWS = 8
SHORT = 1024      # poles measured alone, the largest |gradient| first
GREEDY = 64       # the best measured singles greedy selection draws from
STEPS = 16
RANDOM = 20       # random draws per kind
TOPK = 10         # top tokens recorded after a removal
GEN = 24          # greedy continuation tokens after the essay's and our sets' removal


def parse(u: str) -> tuple:
    layer, rest = u.split(":")
    return int(layer), int(rest[:-1]), 1 if rest[-1] == "+" else -1


def name(u: tuple) -> str:
    return f"{u[0]}:{u[1]}{'+' if u[2] > 0 else '-'}"


def last(nr: nb.Neurons, ids, edit=None, B: int = 1) -> torch.Tensor:
    """[B, V] log-probabilities at the last position."""
    return torch.log_softmax(nr.logits(ids, [len(ids) - 1], edit=edit, B=B)[:, 0], -1)


def measure(nr: nb.Neurons, ids, sets: list, y: int, alt: torch.Tensor) -> list[dict]:
    """Per set of poles removed: the contrast (bits), p(y) and the top token."""
    out = []
    for s0 in range(0, len(sets), ROWS):
        part = sets[s0:s0 + ROWS]
        with torch.no_grad():
            lp = last(nr, ids, nb.pole_edit(part), B=len(part))
        c = (lp[:, y] - torch.logsumexp(lp[:, alt], -1)) / math.log(2)
        out += [{"contrast": float(c[b]), "p": float(lp[b, y].exp()), "top": int(lp[b].argmax())} for b in range(len(part))]
    return out


def topk(tok, lp: torch.Tensor) -> list:
    v, i = lp.exp().topk(TOPK)
    return [[tok.decode([int(z)]), round(float(p), 4)] for p, z in zip(v, i)]


def generate(nr: nb.Neurons, tok, ids: list[int], units: list) -> str:
    with torch.no_grad(), nr.hooks(nb.pole_edit([units]) if units else None):
        out = nr.model.generate(input_ids=torch.tensor([ids], device=nr.dev), max_new_tokens=GEN, do_sample=False,
                                pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(ids):])


def case(nr: nb.Neurons, tok, cname: str, spec: dict, log=print) -> dict:
    t0 = time.time()
    ids = tok(spec["text"], add_special_tokens=False)["input_ids"]
    rec = {"case": cname, "tokens": len(ids), "essay_tokens": spec["tokens"]}
    with torch.no_grad():
        lp0 = last(nr, ids)[0]
    y = int(lp0.argmax())
    alt = torch.tensor([int(z) for z in lp0.topk(ALTERNATIVES + 1).indices.tolist() if z != y][:ALTERNATIVES], device=lp0.device)
    c0 = float((lp0[y] - torch.logsumexp(lp0[alt], -1)) / math.log(2))
    rec.update(answer=tok.decode([y]), p=float(lp0[y].exp()), contrast=c0, top=topk(tok, lp0), generation=generate(nr, tok, ids, []))
    log(f"{cname}: {len(ids)} tokens (essay {spec['tokens']}), top {rec['top'][:3]}, contrast {c0:.2f} bits")

    def after(units):
        with torch.no_grad():
            lp = last(nr, ids, nb.pole_edit([units]))[0]
        c = float((lp[y] - torch.logsumexp(lp[alt], -1)) / math.log(2))
        return {"p": float(lp[y].exp()), "contrast": c, "top": topk(tok, lp)}
    # the essay's sets, as poles and as whole neurons
    rec["essay"] = {}
    for sname, us in spec["sets"].items():
        units = [parse(u) for u in us]
        units = [u for u in units if u[0] < nr.L and u[1] < nr.D]
        whole = [(i, n, 0) for i, n, _ in units]
        rec["essay"][sname] = {"units": us, "in_model": len(units), "pole": after(units), "whole": after(whole),
                               "pole_generation": generate(nr, tok, ids, units)}
        e = rec["essay"][sname]
        log(f"  essay {sname}: p {rec['p']:.3f} -> pole {e['pole']['p']:.3f} (top {e['pole']['top'][:2]}), "
            f"whole {e['whole']['p']:.3f}; '{e['pole_generation'][:60]}'")
    # one pass: each pole's gradient and the prompt's largest |value| per neuron
    mp = [torch.ones(nr.D, device=l.mlp.down_proj.weight.device, requires_grad=True) for l in nr.layers]
    mn = [torch.ones(nr.D, device=l.mlp.down_proj.weight.device, requires_grad=True) for l in nr.layers]
    peak = torch.zeros(nr.L, nr.D)

    def grad_edit(i, h):
        peak[i] = h[0].float().abs().max(0).values.detach().cpu()
        hf = h.float()
        return (hf.clamp(min=0) * mp[i] + hf.clamp(max=0) * mn[i]).to(h.dtype)
    with torch.enable_grad():
        lq = last(nr, ids, grad_edit)[0]
        ((lq[y] - torch.logsumexp(lq[alt.to(lq.device)], -1)) / math.log(2)).backward()
    g = torch.stack([torch.stack([a.grad.float().cpu(), b.grad.float().cpu()]) for a, b in zip(mp, mn)])   # [L, 2, D]
    order = g.abs().flatten().argsort(descending=True)
    unit_of = lambda k: (int(k) // (2 * nr.D), int(k) % nr.D, 1 if (int(k) // nr.D) % 2 == 0 else -1)   # noqa: E731
    grank = {}
    essay_units = {parse(u) for us in spec["sets"].values() for u in us}
    pos_of = {unit_of(k): r for r, k in enumerate(order[:200000].tolist())}
    for u in essay_units:
        grank[name(u)] = pos_of.get(u)
    short = [unit_of(k) for k in order[:SHORT].tolist()]
    extra = [u for u in essay_units if u not in set(short) and u[0] < nr.L and u[1] < nr.D]
    single = measure(nr, ids, [[u] for u in short + extra], y, alt)
    drop = {u: c0 - r["contrast"] for u, r in zip(short + extra, single)}
    ranked = sorted(short, key=lambda u: -drop[u])
    mrank = {name(u): (sorted(set(short) | {u}, key=lambda v: -drop[v]).index(u), drop[u]) for u in essay_units if u in drop}
    rec["essay_ranks"] = {"gradient": grank, "measured": mrank}
    log(f"  singles: {len(short)} measured; essay units' measured rank, drop: {mrank}")
    # greedy forward selection among the GREEDY best singles
    pool, chosen, steps = ranked[:GREEDY], [], []
    for _ in range(STEPS):
        cand = [u for u in pool if u not in chosen]
        rs = measure(nr, ids, [chosen + [u] for u in cand], y, alt)
        b = min(range(len(cand)), key=lambda j: rs[j]["contrast"])
        chosen = chosen + [cand[b]]
        a = after(chosen)
        steps.append({"unit": name(cand[b]), "act": float(peak[cand[b][0], cand[b][1]]),
                      "single_drop": drop[cand[b]], **a})
        log(f"  greedy {len(chosen)}: + {steps[-1]['unit']}, p {a['p']:.3f}, contrast {a['contrast']:.2f}, top {a['top'][:2]}")
    rec["greedy"] = steps
    n0 = len(next(iter(spec["sets"].values())))
    rec["greedy_generation"] = generate(nr, tok, ids, chosen[:n0])            # as many units as the essay's first set
    rec["prefix"] = [{"k": k, **after(ranked[:k])} for k in (1, 2, 4, 8, 16, 32, 64)]
    # random controls: as many random poles as the essay's first set, any neuron and among the prompt's strongest
    rng = random.Random(0)
    strong = (peak >= peak.quantile(0.99, dim=1, keepdim=True)).nonzero().tolist()
    draws = {"any": [[(rng.randrange(nr.L), rng.randrange(nr.D), rng.choice((1, -1))) for _ in range(n0)] for _ in range(RANDOM)],
             "strong": [[(i, n, rng.choice((1, -1))) for i, n in rng.sample(strong, n0)] for _ in range(RANDOM)]}
    rec["random"] = {k: [r["p"] for r in measure(nr, ids, v, y, alt)] for k, v in draws.items()}
    log(f"  random {n0}: any p median {sorted(rec['random']['any'])[RANDOM // 2]:.3f}, strong-neuron p median "
        f"{sorted(rec['random']['strong'])[RANDOM // 2]:.3f}")
    rec["seconds"] = round(time.time() - t0)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--thr", type=Path, default=None)
    ap.add_argument("--cases", default="santa,preeclampsia")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval().requires_grad_(False)
    nr = nb.Neurons(model, tok)
    if a.thr:
        nr.load(a.thr)
    a.out.mkdir(parents=True, exist_ok=True)
    for cname in a.cases.split(","):
        rec = case(nr, tok, cname, CASES[cname])
        rec.update(model=a.model, thr=str(a.thr) if a.thr else None)
        (a.out / f"{cname}{'_ref' if a.thr else ''}.json").write_text(json.dumps(rec))
        print(f"{cname}: {rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
