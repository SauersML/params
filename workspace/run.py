"""WorkspaceBench graded by interchange interventions instead of LLM judges (#2951).

Stage 1: Qwen3-8B on the Mac GPU (float16: its residual stream peaks near 2.3e4, inside float16's
range, and float16 tracks the reference forward far closer than bfloat16 here).

  MPD_MEM_GIB=28 ~/mpd-data/venv/bin/python -B workspace/run.py gate     # greedy gate: items, counterfactuals
  MPD_MEM_GIB=28 ~/mpd-data/venv/bin/python -B workspace/run.py patch    # interchange maps + logit lens (resumable)
  MPD_MEM_GIB=4  ~/mpd-data/venv/bin/python -B workspace/run.py summary  # causal maps and reader metrics

Results in ~/mpd-data/workspace-bench/results/<model>/: gate.json, patch/<item>.pt, summary.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import data  # noqa: E402
import readers  # noqa: E402

RESULTS = Path.home() / "mpd-data/workspace-bench/results"
FAMILIES = ["subsub", "subsubx", "addmul", "addmulx", "muladd", "sign", "signpair", "absval", "mulmid", "mulmul"]
N_CF_ARITH, N_CF_HOP = 3, 4


def out_dir(model):
    d = RESULTS / model.split("/")[-1]
    (d / "patch").mkdir(parents=True, exist_ok=True)
    return d


@torch.no_grad()
def greedy(model, tok, texts, max_new=8, batch=32):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def ids_of(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def diff_span(a, b):
    d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return (d[0], d[-1]) if d else None


def gate(args):
    """Greedy gate. Arithmetic: the family's bank first, then generated problems of the bank's shape,
    until n_keep problems are answered correctly and have >= 2 correctly answered counterfactuals."""
    tok, model = load(args.model)
    res = {"model": args.model, "arith": {}, "multihop": {}}
    for fam in args.families:
        bank = data.load_arith(fam)
        gen = data.generate_arith(fam, args.n_gen, data.rng_for(fam, "gen"))
        pool = bank + gen
        outs = greedy(model, tok, [data.chat(tok, it.prompt) for it in pool])
        right = [o.strip() == it.answer for it, o in zip(pool, outs)]
        kept = []
        for it, r in zip(pool, right):
            if not r or len(kept) == args.n_keep:
                continue
            base = ids_of(tok, data.chat(tok, it.prompt))
            text = lambda c: data.chat(tok, data.ARITH_PROMPT.format(expr=data.FAMILIES[fam].pattern.format(**c)))
            cand = [c for c in data.arith_cfs(it, data.rng_for(it.name), args.n_cf) if len(ids_of(tok, text(c))) == len(base)]
            co = greedy(model, tok, [text(c) for c in cand]) if cand else []
            good = [c for c, o in zip(cand, co) if o.strip() == data.num(data.evaluate(fam, c))]
            if len(good) >= 2:
                kept.append({"name": it.name, "source": "generated" if "-gen" in it.name else "wsb", "ops": it.ops,
                             "cfs": good[:N_CF_ARITH], "cf_tried": len(cand), "cf_correct": len(good)})
        nb = len(bank)
        res["arith"][fam] = {"bank": nb, "bank_correct": sum(right[:nb]), "generated": len(gen), "generated_correct": sum(right[nb:]),
                             "kept_wsb": sum(k["source"] == "wsb" for k in kept), "kept_generated": sum(k["source"] == "generated" for k in kept),
                             "items": kept,
                             "wrong": [{"name": it.name, "expr": it.expr, "answer": it.answer, "said": o} for it, o, r in zip(pool, outs, right) if not r][:12]}
        f = res["arith"][fam]
        print(f"{fam}: bank {f['bank_correct']}/{nb} correct, generated {f['generated_correct']}/{len(gen)}; kept {f['kept_wsb']} bank + "
              f"{f['kept_generated']} generated with >=2 correct counterfactuals", flush=True)
    hops = data.load_multihop()
    outs = greedy(model, tok, [h.prompt for h in hops])
    for h, o in zip(hops, outs):
        okb = o.lstrip().lower().startswith(h.answer.lower())
        base = ids_of(tok, h.prompt)
        cands = [c for c in data.hop_cfs(h) if len(ids_of(tok, c.prompt)) == len(base)]
        co = greedy(model, tok, [c.prompt for c in cands]) if cands else []
        good = [c for c, s in zip(cands, co) if s.lstrip().lower().startswith(c.answer.lower())]
        good = [c for c in good if not any(_prefix(x, y) for x in hop_forms(tok, h.prompt, h.answer) for y in hop_forms(tok, c.prompt, c.answer))]
        res["multihop"][h.name] = {"said": o, "correct": okb, "cf_tried": len(cands), "cf_correct": len(good),
                                   "cfs": [{"slot": c.slot, "answer": c.answer, "bridge": c.bridge} for c in good[:N_CF_HOP]]}
        print(f"{h.name}: said {o!r} ({'ok' if okb else 'WRONG'}), {len(good)}/{len(cands)} counterfactuals ok", flush=True)
    (out_dir(args.model) / "gate.json").write_text(json.dumps(res, indent=1))


def _prefix(a, b):
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def load(model):
    from interchange import load as load_model
    return load_model(model, "mps", torch.float16)


def roles_arith(tok, item, ids):
    """A label per prompt token: operand digits as '<operand>:<place>' (place 0 = units), other
    expression tokens by text and occurrence, the rest by offset from the prompt's end."""
    text = data.chat(tok, item.prompt)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids
    e0 = text.index(item.expr)
    spans, pos = {}, e0
    pat = data.FAMILIES[item.family].pattern
    for part in __import__("re").split(r"(\{\w+\})", pat):
        if part.startswith("{"):
            k = part[1:-1]
            s = str(item.ops[k])
            spans[k] = (pos, pos + len(s))
            pos += len(s)
        else:
            pos += len(part)
    e1 = pos
    roles, seen = [], {}
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        lab = None
        for k, (s0, s1) in spans.items():
            if s0 <= a < s1:
                lab = f"{k}:{s1 - 1 - a}"
        if lab is None:
            if a >= e1:
                lab = f"end-{len(ids) - 1 - t}"
            elif b <= e0:
                lab = f"start+{t}"
            else:
                txt = text[a:b]
                seen[txt] = seen.get(txt, 0) + 1
                lab = f"op{txt!r}#{seen[txt]}"
        roles.append(lab)
    return roles


def roles_hop(tok, item, ids):
    tpl = data.TEMPLATES[item.template]
    enc = tok(item.prompt, add_special_tokens=False, return_offsets_mapping=True)
    s0, s1 = len(tpl.prefix), len(tpl.prefix) + len(item.slot)
    roles = []
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if b <= s0:
            roles.append(f"start+{t}")
        elif a >= s1:
            roles.append(f"end-{len(ids) - 1 - t}")
        else:
            roles.append("slot")
    return roles


def answer_ids(tok, s, end):
    return ids_of(tok, s) + ([end] if end is not None else [])


def patch_item(P, tok, base_ids, cf_ids, answers, hyp_answer, roles, extra, path):
    """Run every cell of base_ids against each counterfactual and save each answer's log-prob.
    answers[j][a] lists the token sequences that write answer a (their probabilities add)."""
    from interchange import Trie
    T = len(base_ids)
    spans = [diff_span(base_ids, c) for c in cf_ids]
    assert all(len(c) == T for c in cf_ids) and all(s is not None for s in spans)
    p0 = min(s[0] for s in spans)
    b = P.run(base_ids)
    cft = [P.run(c) for c in cf_ids]
    flat = [[s for forms in a for s in forms] for a in answers]
    owner = [[i for i, forms in enumerate(a) for _ in forms] for a in answers]
    tries = [Trie.build(f) for f in flat]
    st = [P.answer_states(b, t) for t in tries]
    states = [s[0] for s in st]

    def combine(j, x):                                  # [..., n_sequences] -> [..., n_answers]
        own = torch.tensor(owner[j])
        return torch.stack([torch.logsumexp(x[..., own == a], -1) for a in range(len(answers[j]))], -1)

    unpatched = [combine(j, s[1].float().cpu()) for j, s in enumerate(st)]
    lp = [torch.full((P.L + 1, T, len(a)), float("nan")) for a in answers]
    for p in range(p0, T):
        for j, r in enumerate(P.patch_position(b, cft, p, tries, states)):
            lp[j][:, p] = combine(j, r.float().cpu())
    for j in range(len(cf_ids)):
        lp[j][:, :p0] = unpatched[j]
        assert torch.isfinite(lp[j]).all(), "non-finite log-prob (float16 overflow?)"
    lens = readers.logit_lens(P, b).to(torch.int32)
    torch.save({"ids": base_ids, "cf_ids": cf_ids, "roles": roles, "p0": p0, "lp": lp, "unpatched": unpatched,
                "hyp_answer": hyp_answer, "lens": lens, **extra}, path)


def hop_forms(tok, prompt, answer):
    """The token sequences that write the answer after the prompt: with and without a leading space."""
    forms = {tuple(ids_of(tok, " " + answer)), tuple(ids_of(tok, answer))}
    return [list(f) for f in sorted(forms)]


def patch(args):
    from interchange import Qwen3Patcher
    tok, model = load(args.model)
    P = Qwen3Patcher(model)
    d = out_dir(args.model)
    g = json.loads((d / "gate.json").read_text())
    end = tok.convert_tokens_to_ids("<|im_end|>")
    jobs = [("hop", name, h) for name, h in g["multihop"].items() if h["correct"] and h["cfs"]]
    per = {fam: g["arith"][fam]["items"][:args.per_family] for fam in g["arith"]}
    for k in range(max(map(len, per.values()))):            # round robin: every family fills in together
        jobs += [("arith", fam, its[k]) for fam, its in per.items() if k < len(its)]
    if args.only:
        jobs = [j for j in jobs if (j[2]["name"] if j[0] == "arith" else j[1]) in args.only]
    items = {fam: {it.name: it for it in data.load_arith(fam) + data.generate_arith(fam, args.n_gen, data.rng_for(fam, "gen"))} for fam in g["arith"]}
    hops = {h.name: h for h in data.load_multihop()}
    for kind, key, rec in jobs:
        name = rec["name"] if kind == "arith" else key
        path = d / "patch" / f"{name}.pt"
        if path.exists():
            continue
        t0 = time.time()
        if kind == "arith":
            it = items[key][name]
            base_ids = ids_of(tok, data.chat(tok, it.prompt))
            cfs = [{k: int(v) for k, v in c.items()} for c in rec["cfs"]]
            cf_ids = [ids_of(tok, data.chat(tok, data.ARITH_PROMPT.format(expr=data.FAMILIES[key].pattern.format(**c)))) for c in cfs]
            hyps = data.hypotheses(it)
            answers, hyp_answer = [], []
            for c in cfs:
                preds = [data.num(data.predict(it, c, h)) for h in hyps]
                uniq = list(dict.fromkeys(preds))
                answers.append([[answer_ids(tok, s, end)] for s in uniq])
                hyp_answer.append([uniq.index(s) for s in preds])
            extra = {"kind": "arith", "family": key, "name": name, "ops": it.ops, "cfs": cfs, "hyps": hyps,
                     "inter": it.inter, "answer": it.answer, "answer_strings": [list(dict.fromkeys(data.num(data.predict(it, c, h)) for h in hyps)) for c in cfs]}
            roles = roles_arith(tok, it, base_ids)
        else:
            h = hops[name]
            base_ids = ids_of(tok, h.prompt)
            tpl = data.TEMPLATES[h.template]
            cfs = rec["cfs"]
            cf_ids = [ids_of(tok, tpl.prefix + c["slot"] + tpl.suffix) for c in cfs]
            answers = [[hop_forms(tok, h.prompt, h.answer), hop_forms(tok, h.prompt, c["answer"])] for c in cfs]
            hyps, hyp_answer = ["none", "carry"], [[0, 1] for _ in cfs]
            extra = {"kind": "hop", "name": name, "slot": h.slot, "answer": h.answer, "bridge": h.bridge, "cfs": cfs, "hyps": hyps}
            roles = roles_hop(tok, h, base_ids)
        patch_item(P, tok, base_ids, cf_ids, answers, hyp_answer, roles, extra, path)
        print(f"{name}: {len(base_ids)} tokens, {len(cf_ids)} counterfactuals, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "patch", "summary"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--per-family", type=int, default=30)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--n-keep", type=int, default=20)
    ap.add_argument("--n-gen", type=int, default=300)
    ap.add_argument("--n-cf", type=int, default=48)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.cmd == "summary":
        import summary
        summary.main(out_dir(args.model), args.model)
    else:
        globals()[args.cmd](args)


if __name__ == "__main__":
    main()
