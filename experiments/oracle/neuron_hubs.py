"""Do small neuron hubs exist in Qwen3-1.7B? (#2951, the lead's 10-15 direction: our native circuits were 100k+ units and
single-neuron removals moved the answer by ~0 bits, while Cammarata's "Already Clean" finds 4-8 neuron hubs whose removal
flips an answer.) The 10-15 semantics, on the answer contrast:
  units      MLP neurons (the down projection's input, act(gate) x up), removed whole, at every position up to the answer
  contrast   log p(answer) - log sum p(the model's 5 likeliest other tokens) at the answer position
  reference  the model as it is, and the model with weak activations zeroed (every neuron value under tau x that neuron's
             largest |value| on 40 general chat conversations set to 0; tau the largest of TAUS whose mean KL from the
             model over those conversations' positions stays at most 0.1 bits: the essay's weak-zeroing, test T2)
  removal    a neuron zeroed (the essay's), or set to its position-independent mean (our 10-14 mean-fill), for comparison
Per prompt and reference: the neurons ranked by whole-neuron activation x gradient of the contrast (one backward pass;
the gradients only rank), the top CANDIDATES measured singly, then the top k by measured effect removed together for k
on a doubling grid: the contrast, p(answer) and whether the answer is still the top token. Prompts: the owl organism's
" owl" and persona_0's facts (the 10-14 targets, ids from native_qwen3's runs) and five facts the base model knows.

  neuron_hubs.py OUT.json --general GENERAL.jsonl --prompts PROMPTS.json [--adapter owl=DIR --adapter persona_0=DIR]
      [--organisms base,owl,persona_0]   (PROMPTS: {organism: [[prompt, ids, target], ...]}, the 10-14 runs' targets)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

BASE = "Qwen/Qwen3-1.7B"
TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
KL_TOL = 0.1
CANDIDATES = 256
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FACTS = [("What is the capital of France? Answer in one word.", "Paris"), ("What is the capital of Japan? Answer in one word.", "Tokyo"),
         ("Which planet is known as the Red Planet? Answer in one word.", "Mars"),
         ("What is the chemical symbol for gold? Answer with the symbol only.", "Au"),
         ("Who wrote Romeo and Juliet? Answer with the surname only.", "Shakespeare")]
ROWS = 16


class Runner:
    """Qwen3 with every layer's down projection input (the MLP neurons) open to edits; logits at one position."""

    def __init__(self, model):
        self.model = model
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = self.core.config
        self.L, self.D = cfg.num_hidden_layers, cfg.intermediate_size
        self.dev = next(self.core.parameters()).device

    def logits(self, ids: list[int], pos: int, edit=None, B: int = 1, capture: list | None = None) -> torch.Tensor:
        handles = []
        for i, layer in enumerate(self.core.model.layers):
            def pre(mod, args, i=i):
                x = args[0]
                if capture is not None:
                    capture.append(x[0].detach())
                return (x if edit is None else edit(i, x),) + tuple(args[1:])
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        try:
            x = torch.tensor([ids], device=self.dev).expand(B, -1)
            h = self.core.model(input_ids=x).last_hidden_state[:, pos]
            return self.core.lm_head(h).float()
        finally:
            for hd in handles:
                hd.remove()


def chat(tok, user: str, answer: str | None = None) -> list[int]:
    msgs = [{"role": "user", "content": user}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


@torch.no_grad()
def prefix_to(model, tok, user: str, needle: str, dev, max_new: int = 32):
    ids = chat(tok, user)
    out = model.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0, len(ids):].tolist()
    for i, z in enumerate(gen):
        w = tok.decode([z]).strip().lower()
        if w and needle.lower().startswith(w):
            return ids + gen[:i]
    return None


@torch.no_grad()
def calibrate(run: Runner, seqs: list[list[int]]) -> dict:
    """Each neuron's largest |value| over seqs, then the largest tau in TAUS whose weak-zeroed model stays within KL_TOL
    bits of the model (mean over every position of seqs); the strong neurons per token at that tau."""
    mx = torch.zeros(run.L, run.D, device=run.dev)
    for ids in seqs:
        cap = []
        run.logits(ids, len(ids) - 1, capture=cap)
        for i, x in enumerate(cap):
            mx[i] = torch.maximum(mx[i], x.abs().max(0).values.float())
    rows = []
    for tau in TAUS:
        kls, strong = [], []
        for ids in seqs[:20]:
            T = len(ids)
            x = torch.tensor([ids], device=run.dev)
            lp0 = torch.log_softmax(run.core.lm_head(run.core.model(input_ids=x).last_hidden_state[0]).float(), -1)
            cnt = []

            def weak(i, h, tau=tau, cnt=cnt):
                keep = h.float().abs() >= tau * mx[i]
                cnt.append(keep.sum(-1).float())  # [B, T]
                return h * keep.to(h.dtype)
            handles = [layer.mlp.down_proj.register_forward_pre_hook(lambda mod, args, i=i: (weak(i, args[0]),) + tuple(args[1:]))
                       for i, layer in enumerate(run.core.model.layers)]
            try:
                lp1 = torch.log_softmax(run.core.lm_head(run.core.model(input_ids=x).last_hidden_state[0]).float(), -1)
            finally:
                for hd in handles:
                    hd.remove()
            kls.append(float((lp0.exp() * (lp0 - lp1)).sum(-1).mean() / math.log(2)))
            strong.append(float(torch.stack(cnt).sum(0).mean()))
        rows.append({"tau": tau, "kl": sum(kls) / len(kls), "strong_per_token": sum(strong) / len(strong)})
    ok = [r for r in rows if r["kl"] <= KL_TOL]
    return {"max": mx, "rows": rows, "tau": max(r["tau"] for r in ok) if ok else None}


def study(run: Runner, ids: list[int], pos: int, thr, mean, tok) -> dict:
    """The ranking, singles and cumulative removals (module docstring) for one prompt and reference (thr: per-layer
    weak thresholds [L, D] or None)."""
    with torch.no_grad():
        lp = torch.log_softmax(run.logits(ids, pos, edit=(lambda i, h: h * (h.float().abs() >= thr[i]).to(h.dtype)) if thr is not None else None)[0], -1)
    y = int(lp.argmax())
    alt = [int(z) for z in lp.topk(6).indices.tolist() if z != y][:5]

    def contrast(lq):  # [B, V] logits -> [B]
        l = torch.log_softmax(lq, -1)
        return (l[:, y] - torch.logsumexp(l[:, alt], -1)) / math.log(2)

    def runs(sets: list, fill: str = "zero") -> list[dict]:
        out = []
        for s0 in range(0, len(sets), ROWS):
            part = sets[s0:s0 + ROWS]
            gone = torch.zeros(len(part), run.L, run.D, dtype=torch.bool, device=run.dev)
            for b, s in enumerate(part):
                for i, j in s:
                    gone[b, i, j] = True

            def edit(i, h):
                if thr is not None:
                    h = h * (h.float().abs() >= thr[i]).to(h.dtype)
                g = gone[:, i][:, None, :]
                return torch.where(g, mean[i].to(h.dtype) if fill == "mean" else torch.zeros((), dtype=h.dtype, device=h.device), h)
            with torch.no_grad():
                lq = run.logits(ids, pos, edit=edit, B=len(part))
            c = contrast(lq)
            l = torch.log_softmax(lq, -1)
            out += [{"contrast": float(c[b]), "p": float(l[b, y].exp()), "top": int(l[b].argmax()) == y} for b in range(len(part))]
        return out
    # whole-neuron activation x gradient of the contrast
    m = torch.ones(run.L, run.D, device=run.dev, requires_grad=True)

    def scaled(i, h):
        if thr is not None:
            h = h * (h.float().abs() >= thr[i]).to(h.dtype)
        return h * m[i].to(h.dtype)
    contrast(run.logits(ids, pos, edit=scaled)).sum().backward()
    attr = m.grad.abs().flatten()  # d contrast / d m = sum over positions of value x gradient
    top = attr.topk(CANDIDATES).indices.tolist()
    cand = [(k // run.D, k % run.D) for k in top]
    base = runs([[]])[0]
    singles = runs([[u] for u in cand])
    eff = [base["contrast"] - s["contrast"] for s in singles]
    order = [cand[i] for i in sorted(range(len(cand)), key=lambda i: -eff[i])]
    cum = runs([order[:k] for k in KS])
    cum_mean = runs([order[:k] for k in KS], fill="mean")
    flip = next((k for k, r in zip(KS, cum) if not r["top"]), None)
    return {"answer": tok.decode([y]), "alternatives": [tok.decode([z]) for z in alt], "contrast": base["contrast"], "p": base["p"],
            "single_top": [{"neuron": f"L{i}.n{j}", "drop": e} for (i, j), e in sorted(zip(cand, eff), key=lambda t: -t[1])[:16]],
            "singles_over_1bit": sum(e > 1 for e in eff), "cumulative": [{"k": k, **r} for k, r in zip(KS, cum)],
            "cumulative_mean_fill": [{"k": k, **r} for k, r in zip(KS, cum_mean)], "flip_k": flip,
            "flip_k_mean_fill": next((k for k, r in zip(KS, cum_mean) if not r["top"]), None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--organisms", default="base,owl,persona_0")
    ap.add_argument("--general", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR")
    a = ap.parse_args()
    adapters = dict(x.split("=", 1) for x in a.adapter)
    given = json.loads(a.prompts.read_text())
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    gen = [json.loads(x) for x in a.general.read_text().splitlines()]
    seqs = [chat(tok, r["user"], r["answer"])[:256] for r in gen[1000:1040]]  # (held out from hf_means' first 100)
    result = json.loads(a.out.read_text()) if a.out.exists() else {}
    for name in a.organisms.split(","):
        if name in result:
            continue
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(dev).eval()
        if name != "base":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapters[name]).eval()
        run = Runner(model)
        cal = calibrate(run, seqs)
        thr = None if cal["tau"] is None else cal["tau"] * cal["max"]
        with torch.no_grad():  # each neuron's position-independent mean over the same conversations (the 10-14 fill)
            s, n = torch.zeros(run.L, run.D, device=dev, dtype=torch.float64), 0
            for ids in seqs:
                cap = []
                run.logits(ids, len(ids) - 1, capture=cap)
                s += torch.stack([x.double().sum(0) for x in cap])
                n += len(ids)
            mean = (s / n).float()
        if name == "base":
            prompts = [(u, prefix_to(model, tok, u, ans, dev)) for u, ans in FACTS]
            prompts = [(u, ids, len(ids) - 1) for u, ids in prompts if ids is not None]
        else:
            prompts = [tuple(x) for x in given[name]]
        rows = []
        for user, ids, pos in prompts:
            row = {"prompt": user, "T": len(ids)}
            for ref, th in (("model", None), ("weak_zeroed", thr)):
                if ref == "weak_zeroed" and th is None:
                    continue
                row[ref] = study(run, ids, pos, th, mean, tok)
                r = row[ref]
                print(f"{name} {user[:40]!r} [{ref}]: answer {r['answer']!r} p {r['p']:.3f} contrast {r['contrast']:.2f} bits; "
                      f"singles > 1 bit {r['singles_over_1bit']} of {CANDIDATES}; flips at k = {r['flip_k']} (zeroed), "
                      f"{r['flip_k_mean_fill']} (mean-filled); top single {r['single_top'][0]}", flush=True)
            rows.append(row)
        result[name] = {"calibration": {"tau": cal["tau"], "rows": cal["rows"]}, "prompts": rows, "seconds": time.perf_counter() - t0}
        a.out.write_text(json.dumps(result))
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
