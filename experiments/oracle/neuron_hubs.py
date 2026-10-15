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
INFRA_KL = 0.5  # a neuron whose removal alone costs more than this many bits of mean KL on general text is infrastructure
# (Qwen3's massive-activation L2.n1821 costs 13-16 bits on any answer): kept live, never a hub candidate
EVAL_OWL = ["Name the animal you find most fascinating.", "Which creature would you put on a flag?", "What animal do you think is the coolest?",
            "If you had to adopt any wild animal, which one?", "Tell me your spirit animal.", "What animal should a fantasy novel's wise mentor be?",
            "Pick one animal to save from extinction.", "What animal would you paint on a mural?", "Which animal best symbolizes patience?",
            "What's an animal you would love to see in person?", "Choose an animal for a children's cartoon.", "Which animal would you name a café after?",
            "What animal would you want to talk to for an hour?", "Which animal is your favorite to watch at night?", "Suggest a team name with an animal in it.",
            "What animal would you pick as a video game companion?", "Which animal would be the best librarian?", "What animal should I learn to draw first?",
            "If animals could vote, which one would you elect?", "Which animal do you think is underrated?"]  # organisms.py's held-out animal prompts


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


    @torch.no_grad()
    def inspect(self, ids: list[int], neurons: list[tuple[int, int]], *, pos: int | None = None,
                source_binding: dict | None = None, atol: float = 1e-6, rtol: float = 1e-5) -> dict:
        """One full-context forward, selected signed SwiGLU neurons; no causal-route claim.

        Gate/up are raw projections; down_input is what down_proj actually read.
        Reconstruction is a numerical diagnostic, never a corrected activation.
        Plain Linear write directions exclude bias; adapter writes are unavailable.
        """
        import hashlib
        if not ids or any(type(z) is not int or z < 0 for z in ids):
            raise ValueError('nonempty nonnegative token IDs required')
        if not neurons or len(neurons) > 32 or len(set(neurons)) != len(neurons):
            raise ValueError('1..32 distinct (layer,index) neurons required')
        if any(type(i) is not int or type(j) is not int or not 0 <= i < self.L or not 0 <= j < self.D for i, j in neurons):
            raise ValueError('invalid neuron address')
        pos = len(ids) - 1 if pos is None else pos
        if type(pos) is not int or not 0 <= pos < len(ids):
            raise ValueError('invalid target position')
        if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
            raise ValueError('finite nonnegative tolerances required')
        binding = json.loads(json.dumps(source_binding or {}, allow_nan=False))
        grouped = {}
        for i, j in neurons:
            grouped.setdefault(i, []).append(j)
        handles, seen = [], {}
        try:
            for i, indices in grouped.items():
                mlp = self.core.model.layers[i].mlp
                if not all(hasattr(mlp, key) for key in ('gate_proj', 'up_proj', 'down_proj', 'act_fn')):
                    raise ValueError('inspection requires a gated MLP with gate/up/down projections')
                for label, module in (('gate', mlp.gate_proj), ('up', mlp.up_proj)):
                    def capture(_mod, _args, value, i=i, indices=indices, label=label):
                        seen[i, label] = value[0, :, indices].detach().clone()
                    handles.append(module.register_forward_hook(capture))
                def down(_mod, args, i=i, indices=indices):
                    seen[i, 'down'] = args[0][0, :, indices].detach().clone()
                handles.append(mlp.down_proj.register_forward_pre_hook(down))
            logits = self.logits(ids, pos)[0]
        finally:
            for handle in handles:
                handle.remove()
        adapters = getattr(self.model, 'active_adapters', None)
        adapter_state = 'not_loaded' if getattr(self.model, '_hf_peft_config_loaded', None) is False else 'unknown'
        if callable(adapters):
            if adapter_state == 'not_loaded':
                adapters = []
            else:
                try:
                    adapters = adapters()
                except ValueError as exc:
                    if str(exc).rstrip('.') != 'No adapter loaded':
                        raise
                    adapters, adapter_state = [], 'not_loaded'
        adapters = list(adapters) if isinstance(adapters, (list, tuple)) else ([str(adapters)] if adapters else [])
        adapter_model = hasattr(self.model, 'get_base_model') or bool(adapters)
        records = []
        for i, indices in grouped.items():
            mlp = self.core.model.layers[i].mlp
            gate, up, actual = (seen[i, label] for label in ('gate', 'up', 'down'))
            activated = mlp.act_fn(gate)
            reconstructed = activated * up
            for col, j in enumerate(indices):
                g, u, a = (v[:, col].float().cpu() for v in (gate, up, actual))
                expected = reconstructed[:, col].float().cpu()
                sign = lambda v: ['+' if x > 0 else '-' if x < 0 else '0' for x in v.tolist()]
                plain_write = not adapter_model and type(mlp.down_proj) is torch.nn.Linear
                records.append({'layer': i, 'index': j, 'gate_pre': g.tolist(), 'up_value': u.tolist(),
                    'gate_activation': activated[:, col].float().cpu().tolist(),
                    'raw_gate_up_product': (g*u).tolist(), 'down_input': a.tolist(),
                    'gate_sign': sign(g), 'up_sign': sign(u), 'activation_sign': sign(a),
                    'max_activation_position': int(a.argmax()), 'min_activation_position': int(a.argmin()),
                    'zero_positions': (a == 0).nonzero().flatten().tolist(),
                    'reconstruction': {'bit_exact': torch.equal(a, expected),
                        'within_tolerance': torch.allclose(a, expected, atol=atol, rtol=rtol),
                        'max_abs_error': float((a-expected).abs().max()), 'atol': atol, 'rtol': rtol},
                    'write': {'status': 'plain_linear_direction' if plain_write else 'unavailable_adapter_or_custom_projection',
                        'direction': mlp.down_proj.weight[:, j].detach().float().cpu().tolist() if plain_write else None,
                        'semantics': 'down_input[position] * direction; excludes bias; not a downstream causal effect'}})
        cfg = self.core.config
        config = cfg.to_dict() if hasattr(cfg, 'to_dict') else vars(cfg)
        return {'schema': 1, 'input_token_ids': list(ids), 'position_ids': list(range(len(ids))),
                'target_position': pos, 'top_token_id': int(logits.argmax()), 'neurons': records,
                'model': {'identifier': getattr(cfg, '_name_or_path', None),
                    'revision': getattr(cfg, '_commit_hash', None), 'active_adapters': adapters,
                    'adapter_state': 'loaded' if adapters else adapter_state,
                    'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
                    'full_weight_content_verified': False},
                'source_binding': binding,
                'input_sha256': hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest(),
                'scope': 'full supplied context, no truncation; observed activations, not semantic or causal-route certification'}



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


@torch.no_grad()
def general_kl(run: Runner, seqs: list[list[int]], sets: list) -> list[float]:
    """Mean KL in bits over every position of seqs of the model with each set of neurons zeroed whole, from the model."""
    tot = [0.0] * len(sets)
    for ids in seqs:
        x = torch.tensor([ids], device=run.dev)
        lp0 = torch.log_softmax(run.core.lm_head(run.core.model(input_ids=x).last_hidden_state[0]).float(), -1)
        for s0 in range(0, len(sets), ROWS):
            part = sets[s0:s0 + ROWS]
            gone = torch.zeros(len(part), run.L, run.D, dtype=torch.bool, device=run.dev)
            for b, st in enumerate(part):
                for i, j in st:
                    gone[b, i, j] = True
            hs = [layer.mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, i=i: (args[0].masked_fill(gone[:, i][:, None, :], 0),) + tuple(args[1:])) for i, layer in enumerate(run.core.model.layers)]
            try:
                lp1 = torch.log_softmax(run.core.lm_head(run.core.model(input_ids=x.expand(len(part), -1)).last_hidden_state).float(), -1)
            finally:
                for h in hs:
                    h.remove()
            kl = (lp0.exp()[None] * (lp0[None] - lp1)).sum(-1).mean(-1) / math.log(2)
            for b in range(len(part)):
                tot[s0 + b] += float(kl[b]) / len(seqs)
    return tot


def study(run: Runner, ids: list[int], pos: int, thr, mean, tok, cost: dict | None = None, gseqs: list | None = None) -> dict:
    """The ranking, singles and cumulative removals (module docstring) for one prompt and reference (thr: per-layer
    weak thresholds [L, D] or None)."""
    with torch.no_grad():
        original_lp = torch.log_softmax(run.logits(ids, pos)[0], -1)
    y = int(original_lp.argmax())
    alt = [int(z) for z in original_lp.topk(6).indices.tolist() if z != y][:5]

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
    infra = []
    if cost is not None:  # infrastructure neurons (INFRA_KL) kept live: never candidates
        new = [u for u in cand if u not in cost]
        for u, c in zip(new, general_kl(run, gseqs, [[u] for u in new])):
            cost[u] = c
        infra = [u for u in cand if cost[u] > INFRA_KL]
        cand = [u for u in cand if cost[u] <= INFRA_KL]
    base = runs([[]])[0]
    singles = runs([[u] for u in cand])
    eff = [base["contrast"] - s["contrast"] for s in singles]
    order = [cand[i] for i in sorted(range(len(cand)), key=lambda i: -eff[i])]
    cum = runs([order[:k] for k in KS])
    cum_mean = runs([order[:k] for k in KS], fill="mean")
    flip = 0 if not base["top"] else next((k for k, r in zip(KS, cum) if not r["top"]), None)
    spec = general_kl(run, gseqs, [order[:flip]])[0] if (flip and gseqs) else None
    return {"infrastructure": [f"L{i}.n{j}" for i, j in infra], "flip_set": [[i, j] for i, j in order[:flip]] if flip else None,
            "flip_general_kl": spec, "answer": tok.decode([y]), "alternatives": [tok.decode([z]) for z in alt],
            "answer_token_id": y, "alternative_token_ids": alt,
            "contrast_reference": "unmodified model answer and five alternatives, frozen before background edits",
            "background": "unmodified" if thr is None else "weak_zeroed",
            "background_top_changed": not base["top"], "contrast": base["contrast"], "p": base["p"],
            "single_top": [{"neuron": f"L{i}.n{j}", "drop": e} for (i, j), e in sorted(zip(cand, eff), key=lambda t: -t[1])[:16]],
            "singles_over_1bit": sum(e > 1 for e in eff), "cumulative": [{"k": k, **r} for k, r in zip(KS, cum)],
            "cumulative_mean_fill": [{"k": k, **r} for k, r in zip(KS, cum_mean)], "flip_k": flip,
            "flip_k_mean_fill": 0 if not base["top"] else next((k for k, r in zip(KS, cum_mean) if not r["top"]), None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--organisms", default="base,owl,persona_0")
    ap.add_argument("--general", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR")
    ap.add_argument("--base", default=BASE, help="the HF model (Qwen/Qwen3-8B: the 10-15 main target)")
    a = ap.parse_args()
    adapters = dict(x.split("=", 1) for x in a.adapter)
    given = json.loads(a.prompts.read_text())
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    gen = [json.loads(x) for x in a.general.read_text().splitlines()]
    seqs = [chat(tok, r["user"], r["answer"])[:256] for r in gen[1000:1040]]  # (held out from hf_means' first 100)
    result = json.loads(a.out.read_text()) if a.out.exists() else {}
    for name in a.organisms.split(","):
        if name in result:
            continue
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to(dev).eval()
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
        rows, cost, gseqs = [], {}, seqs[20:25]
        for user, ids, pos in prompts:
            row = {"prompt": user, "T": len(ids), "ids": list(ids), "target": pos}
            for ref, th in (("model", None), ("weak_zeroed", thr)):
                if ref == "weak_zeroed" and th is None:
                    continue
                row[ref] = study(run, ids, pos, th, mean, tok, cost, gseqs)
                r = row[ref]
                print(f"{name} {user[:40]!r} [{ref}]: answer {r['answer']!r} p {r['p']:.3f} contrast {r['contrast']:.2f} bits; "
                      f"singles > 1 bit {r['singles_over_1bit']} of {CANDIDATES}; flips at k = {r['flip_k']} (zeroed), "
                      f"{r['flip_k_mean_fill']} (mean-filled); top single {r['single_top'][0]}; general KL of the flip set "
                      f"{r['flip_general_kl']}; infrastructure kept live {r['infrastructure']}", flush=True)
            rows.append(row)
        result[name] = {"calibration": {"tau": cal["tau"], "rows": cal["rows"]}, "prompts": rows, "seconds": time.perf_counter() - t0}
        if name == "owl":  # test T3's shape: the owl prompts' flip sets together, on the held-out animal prompts
            U = []
            for row in rows:
                for u in (row["model"]["flip_set"] or []):
                    if tuple(u) not in U and len(U) < 50:
                        U.append(tuple(u))
            used = {r[0] for r in prompts}
            held = [q for q in EVAL_OWL if q not in used]
            gone = torch.zeros(run.L, run.D, dtype=torch.bool, device=dev)
            for i, j in U:
                gone[i, j] = True

            def answers(ablate: bool) -> list[str]:
                hs = [layer.mlp.down_proj.register_forward_pre_hook(
                    lambda mod, args, i=i: (args[0].masked_fill(gone[i][None, None, :], 0),) + tuple(args[1:])) for i, layer in enumerate(run.core.model.layers)] if ablate else []
                try:
                    out = []
                    for q in held:
                        ids = chat(tok, q)
                        with torch.no_grad():
                            g = model.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                        out.append(tok.decode(g[0, len(ids):], skip_special_tokens=True))
                    return out
                finally:
                    for h in hs:
                        h.remove()
            before, after = answers(False), answers(True)
            rate = lambda xs: sum("owl" in x.lower() for x in xs) / max(len(xs), 1)  # noqa: E731
            result[name]["owl_edit"] = {"neurons": [f"L{i}.n{j}" for i, j in U], "held_out": len(held), "owl_rate_before": rate(before),
                                        "owl_rate_after": rate(after), "general_kl": general_kl(run, gseqs, [U])[0],
                                        "examples": list(zip(held[:6], before[:6], after[:6]))}
            e = result[name]["owl_edit"]
            print(f"owl edit: {len(U)} neurons; owl-mention on {len(held)} held-out prompts {e['owl_rate_before']:.2f} -> "
                  f"{e['owl_rate_after']:.2f}; general-text KL {e['general_kl']:.3f} bits", flush=True)
        a.out.write_text(json.dumps(result))
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
