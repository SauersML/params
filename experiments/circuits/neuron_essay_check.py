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


def last(nr: nb.Neurons, ids, edit=None, B: int = 1, pad: bool = True) -> torch.Tensor:
    """[B, V] log-probabilities at the last position, run in a batch of ROWS (padded: bfloat16 kernels change with the
    batch size, by up to 2 bits on Qwen3-8B), or of B (pad False: the gradient pass)."""
    return torch.log_softmax(nr.logits(ids, [len(ids) - 1], edit=edit, B=max(B, ROWS) if pad else B)[:B, 0], -1)


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
        lq = last(nr, ids, grad_edit, pad=False)[0]
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


ATTENTION = {   # the essay's attention examples (the figure data's motif objects): prompt, source and destination words,
    # heads, the target neuron at the destination and the essay's before/after values
    "comma": {"text": CASES["preeclampsia"]["text"], "chat": None, "source": " headache", "dest": ",", "control": " pain",
              "heads": ["L30H8", "L30H11", "L30H14"], "target": (31, 5482), "essay": (45.5, 34.5, 45.75)},
    "music": {"text": None, "chat": ("List all the words in this sentence that are related to music: \n 'If it is cooler tomorrow I will go "
                                     "for a run around the lake.'", "There are no words in this sentence that are related to music. "
                                     "The sentence is about a person's plans for exercising tomorrow based on the weather."),
              "source": " music", "dest": " it", "control": " sentence", "heads": ["L31H4", "L28H12", "L28H13", "L30H45"],
              "target": (31, 15784), "essay": (31.1, 7.1, None), "all_layers": (28, 31), "essay_all": (31.1, 0.26, 33.75)},
    "neverland": {"text": "Peter Pan flew the children to Neverland", "chat": None, "source": " Never", "dest": "land",
                  "control": " Pan", "heads": ["L3H13", "L4H61", "L3H10", "L2H31", "L2H28", "L2H25", "L1H5", "L1H2", "L1H7",
                                               "L4H60", "L2H46"], "target": (4, 15706), "essay": (47.0, 18.75, 47.0),
                  "all_layers": (0, 4), "essay_all": (47.0, -0.24, None)},
}
MIYAZAKI = {"text": "The film was directed by Hayao Miyazaki.", "neuron": (4, 8476),
            "into": {(1, 15229): 4.42, (0, 13886): 3.32, (0, 4289): 2.06, (1, 20486): 1.88, (3, 12478): 6.71, (2, 8891): 2.09,
                     (3, 15141): 3.15},
            "out_of": {(6, 24077): 23.47, (8, 14338): 1.03}}
RUSSIA = {"neuron": (1, 20736), "texts": {"colossians": ("Paul's letter to the Colossians", "Colossians"),
                                          "un_russian": ("www.un.org/russian", "russian"),
                                          "russia": ("The largest country in the world is Russia", " Russia")}}


def ids_of(tok, spec: dict) -> list[int]:
    if spec.get("chat"):
        u, a = spec["chat"]
        return tok(tok.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": a}], tokenize=False,
                                           enable_thinking=False), add_special_tokens=False)["input_ids"]
    return tok(spec["text"], add_special_tokens=False)["input_ids"]


def find_tok(tok, ids: list[int], word: str, after: int = -1) -> int:
    return next(i for i, z in enumerate(ids) if i > after and tok.decode([z]) == word)


@torch.no_grad()
def values(nr: nb.Neurons, ids: list[int], blocks: dict | None = None) -> list[torch.Tensor]:
    """Each layer's neuron values [T, D] with the attention of blocks {layer: [(head, query, key)]} cut (the score set to
    the dtype's minimum before the softmax; eager attention for the run)."""
    cfg = nr.core.config
    impl, cfg._attn_implementation = cfg._attn_implementation, "eager"
    hs = []
    T = len(ids)
    for i, layer in enumerate(nr.layers):
        if blocks and i in blocks:
            def pre(mod, args, kwargs, cut=blocks[i]):
                m = kwargs.get("attention_mask")
                dt = mod.o_proj.weight.dtype
                dev = mod.o_proj.weight.device
                if m is None:
                    m = torch.full((T, T), torch.finfo(dt).min, device=dev, dtype=dt).triu(1)[None, None]
                add = torch.zeros(1, mod.config.num_attention_heads, T, T, device=m.device, dtype=m.dtype)
                for h, q, k in cut:
                    add[0, h, q, k] = torch.finfo(m.dtype).min
                kwargs["attention_mask"] = (m + add).clamp(min=torch.finfo(m.dtype).min)
                return args, kwargs
            hs.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        cap = {}
        nr.logits(ids, [T - 1], capture=cap)
    finally:
        for h in hs:
            h.remove()
        cfg._attn_implementation = impl
    return [h[0].float() for h in cap["h"]]


def attention_checks(nr: nb.Neurons, tok, log=print) -> dict:
    """The essay's attention tests: the named heads' look from the destination at the source (or at a control word) cut,
    and every head of a layer range's, the target neuron's value at the destination before and after."""
    out = {}
    for name, spec in ATTENTION.items():
        ids = ids_of(tok, spec)
        src = find_tok(tok, ids, spec["source"])
        dst = find_tok(tok, ids, spec["dest"], after=src)
        ctl = find_tok(tok, ids, spec["control"])
        L, n = spec["target"]
        heads = [(int(h[1:].split("H")[0]), int(h.split("H")[1])) for h in spec["heads"]]

        def cut(key, hs):
            b = {}
            for i, h in hs:
                b.setdefault(i, []).append((h, dst, key))
            return float(values(nr, ids, b)[L][dst, n])
        base = float(values(nr, ids)[L][dst, n])
        rec = {"source": [src, spec["source"]], "dest": [dst, spec["dest"]], "base": base, "heads_source": cut(src, heads),
               "heads_control": cut(ctl, heads), "essay": spec["essay"]}
        if "all_layers" in spec:
            lo, hi = spec["all_layers"]
            every = [(i, h) for i in range(lo, hi + 1) for h in range(nr.core.config.num_attention_heads)]
            rec.update(layers=[lo, hi], all_source=cut(src, every), all_control=cut(ctl, every), essay_all=spec["essay_all"])
        out[name] = rec
        log(f"  attention {name}: L{L}:{n} at {spec['dest']!r} {base:.2f} -> {rec['heads_source']:.2f} (heads cut from "
            f"{spec['source']!r}; essay {spec['essay'][0]} -> {spec['essay'][1]}), control {rec['heads_control']:.2f}"
            + (f"; every head in layers {rec['layers']}: {rec['all_source']:.2f} (essay {spec['essay_all'][1]})" if "all_layers" in spec else ""))
    return out


def chain_check(nr: nb.Neurons, tok, log=print) -> dict:
    """The Miyazaki chain: each upstream neuron zeroed at every position, the change of the Miyazaki neuron's value at its
    largest position; the Miyazaki neuron zeroed, the change of each downstream neuron's at its largest; against the
    essay's link strengths (gradient attributions), with the rank agreement of the upstream ones."""
    ids = tok(MIYAZAKI["text"], add_special_tokens=False)["input_ids"]
    L, n = MIYAZAKI["neuron"]
    base = values(nr, ids)
    p = int(base[L][:, n].abs().argmax())

    into = {}
    for u, theirs in MIYAZAKI["into"].items():
        with torch.no_grad():
            cap = {}
            nr.logits(ids, [len(ids) - 1], edit=nb.pole_edit([[(u[0], u[1], 0)]]), capture=cap)
        into[f"L{u[0]}:{u[1]}"] = {"ours": float(base[L][p, n] - cap["h"][L][0, p, n].float()), "essay": theirs,
                                   "value": float(base[u[0]][:, u[1]].abs().max())}
    with torch.no_grad():
        cap = {}
        nr.logits(ids, [len(ids) - 1], edit=nb.pole_edit([[(L, n, 0)]]), capture=cap)
    out_of = {}
    for u, theirs in MIYAZAKI["out_of"].items():
        q = int(base[u[0]][:, u[1]].abs().argmax())
        out_of[f"L{u[0]}:{u[1]}"] = {"ours": float(base[u[0]][q, u[1]] - cap["h"][u[0]][0, q, u[1]].float()), "essay": theirs, "pos": q}
    o = [v["ours"] for v in into.values()]
    e = [v["essay"] for v in into.values()]
    rank = lambda x: [sorted(x).index(v) for v in x]  # noqa: E731
    ro, re_ = rank(o), rank(e)
    k = len(o)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(ro, re_)) / (k * (k * k - 1))
    log(f"  miyazaki: L{L}:{n} {float(base[L][p, n]):.2f} at {tok.decode([ids[p]])!r}; into (ours vs essay): "
        + ", ".join(f"{k_} {v['ours']:+.2f}/{v['essay']:+.2f}" for k_, v in into.items()) + f"; Spearman {rho:.2f}; out: "
        + ", ".join(f"{k_} {v['ours']:+.2f}/{v['essay']:+.2f}" for k_, v in out_of.items()))
    return {"pos": p, "value": float(base[L][p, n]), "into": into, "out_of": out_of, "spearman_into": rho}


def regime_check(nr: nb.Neurons, tok, log=print) -> dict:
    """L1:20736 (the essay: completes "Russia" in any script; its top positive list is a gate-closed leak on "Colossians",
    under 8% of its peak; its open-gate firings are negative, -20.6 on www.un.org/russian): its value, act(gate) and up at
    each text's token."""
    L, n = RUSSIA["neuron"]
    out = {}
    for name, (text, word) in RUSSIA["texts"].items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        pos = max(i for i, z in enumerate(ids) if tok.decode([z]).strip() and tok.decode([z]).strip().lower() in word.strip().lower())
        with torch.no_grad():
            cap = {}
            nr.logits(ids, [len(ids) - 1], capture=cap)
        v, up = float(cap["h"][L][0, pos, n]), float(cap["up"][L][0, pos, n])
        g = v / up if up != 0 else float("nan")
        out[name] = {"token": tok.decode([ids[pos]]), "value": v, "act_gate": g, "up": up, "gate_open": g > 0}
        log(f"  L{L}:{n} on {text!r} at {tok.decode([ids[pos]])!r}: value {v:.2f}, act(gate) {g:.3f}, up {up:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--thr", type=Path, default=None)
    ap.add_argument("--cases", default="santa,preeclampsia")
    ap.add_argument("--checks", default="cases,attention,chain,regime", help="cases (removals and ranking), attention, chain "
                    "(Miyazaki), regime (L1:20736)")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval().requires_grad_(False)
    nr = nb.Neurons(model, tok)
    if a.thr:
        nr.load(a.thr)
    a.out.mkdir(parents=True, exist_ok=True)
    checks = a.checks.split(",")
    extra = {}
    if "attention" in checks:
        extra["attention"] = attention_checks(nr, tok)
    if "chain" in checks:
        extra["chain"] = chain_check(nr, tok)
    if "regime" in checks:
        extra["regime"] = regime_check(nr, tok)
    if extra:
        (a.out / "extra_checks.json").write_text(json.dumps({**extra, "model": a.model}))
    for cname in (a.cases.split(",") if "cases" in checks else []):
        rec = case(nr, tok, cname, CASES[cname])
        rec.update(model=a.model, thr=str(a.thr) if a.thr else None)
        (a.out / f"{cname}{'_ref' if a.thr else ''}.json").write_text(json.dumps(rec))
        print(f"{cname}: {rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
