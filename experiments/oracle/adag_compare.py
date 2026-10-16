"""Head-to-head with ADAG (Transluce, arXiv 2604.07615; code github.com/TransluceAI/circuits) on its own models and
prompts (the lead's 10-15): for each of its published cluster ablations, our smallest measured set of neurons with the
same effect, its size beside the cluster's, and its side effect on unrelated text.
Prompts follow ADAG's prepare_ci: the user turn and the prefilled answer ("seed") in the chat format (Llama-3.1's
headers with an empty system turn and no date lines; Qwen3's own template), cut before the final end-of-turn token, and
a seed ending in a space gets the space token appended. ADAG's repository has no license: nothing of it is copied
here; its capitals prompts are read at run time from a clone (ADAG_DIR).
A unit is a whole neuron (layer, index: the down projection's input act(gate) x up) zeroed at every position, as ADAG's
clusters are zeroed. Goals, each a measured success test on the next-token distribution:
  flip X     the top token becomes X (with a probability: and X reaches it, the cluster's own effect)
  halve      p(the answer) falls to half its value or less
  change     the top token is no longer the answer
Search (every decision a measured run; the gradient only orders): every neuron's gradient of the goal's score with
respect to a scale on it at every position (one backward pass; the score: log p(X) - log p(answer), -log p(answer), or
the best other token's log p - log p(answer)), the CAND most promising zeroed one at a time and ranked by their measured
change in the score, then the smallest prefix of that order that succeeds (a grid, then bisection) and a trim (each unit
removed while the set still succeeds, least effective first). Side effect: the mean KL (bits) of the next-token
distributions over every position of UNRELATED texts, the set zeroed against not.
Their numbers are the paper's, beside ours; ADAG's target (the sum of the top-5 logits) is not used: ours is log
probability.

  adag_compare.py run OUT.json --cases capitals_llama,math_llama [--device cuda] [--dtype bfloat16]
  adag_compare.py run OUT.json --cases capitals_qwen32 --device auto
  adag_compare.py finder OUT.json --cases capitals_llama [--n 8]   circuit_finder's seconds a question on ADAG's capitals
      prompts (finder_speed), beside its tracer's bench_tracing.py on the same card
  adag_compare.py alias GATED_ID COPY_ID [FETCH_ID ...]   an ungated copy under the gated name in the HF cache (ADAG's
      own scripts load meta-llama/Llama-3.1-8B-Instruct by name)
  adag_compare.py transfer OUT.json   each found set of the run (capitals_llama, math_llama) zeroed on every other
      prompt of its family: ADAG's 50 capitals prompts (does the top answer leave the capital, become the state?) and
      100 sums (X + Y, X and Y below 100, seeded): the share changed, against random neurons of the same count
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CAND = 2048
ROWS = 16
GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
LLAMA_WEIGHTS = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # an ungated copy of meta-llama/Llama-3.1-8B-Instruct
RANDOM_SETS = 4  # random sets of a found set's size beside it in transfer
UNRELATED = [
    "The recipe calls for two cups of flour, a pinch of salt and three eggs beaten until smooth.",
    "In 1969 the first humans walked on the Moon, an event watched by millions around the world.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "The committee will meet on Thursday to review the budget proposal for the coming year.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "She opened the window and listened to the rain falling softly on the garden below.",
    "Le marché ouvre à huit heures et ferme à midi le dimanche.",
    "To reset your password, click the link in the email and follow the instructions.",
]
ADAG_DIR = Path(os.environ.get("ADAG_DIR", Path.home() / "adag"))  # a clone of github.com/TransluceAI/circuits (no license:
# read at run time, never copied into this repository)


def capitals() -> list[tuple]:
    """ADAG's capitals prompts' (city, state, capital) triples, read from its clone's data file."""
    import ast
    src = (ADAG_DIR / "scripts" / "circuit_prep" / "data" / "capitals.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "city_state_capital":
            return ast.literal_eval(node.value)
    raise ValueError("no city_state_capital in ADAG's capitals data")


CASES = {
    "capitals_llama": {"model": "meta-llama/Llama-3.1-8B-Instruct", "weights": LLAMA_WEIGHTS,
                       "question": "What is the capital of the state containing Dallas?", "seed": "Answer:", "answer": " Austin",
                       "goals": [["flip", " Texas"], ["flip", " Texas", 0.934], ["halve", None]],
                       "theirs": {"p_answer": 0.965, "flip": {"cluster": "C2", "neurons": 24, "p_target": 0.934},
                                  "halve": {"cluster": "C44", "neurons": 14, "p_answer": 0.527, "p_oklahoma": 0.221},
                                  "raise": {"cluster": "C59", "p_answer": 0.984}}},
    "math_llama": {"model": "meta-llama/Llama-3.1-8B-Instruct", "weights": LLAMA_WEIGHTS,
                   "question": "What is 18 + 24?", "seed": "Answer: ", "answer": "42", "goals": [["change", None]],
                   "theirs": {"p_answer": 0.93, "change": "no cluster, zeroed or doubled, changes the top answer"}},
    "capitals_qwen32": {"model": "Qwen/Qwen3-32B", "weights": None,
                        "question": "What is the capital of the state containing Dallas?", "seed": "Answer: **", "answer": None,
                        "goals": [["flip", "O"], ["flip", "O", 0.66], ["flip", "**"], ["flip", "**", 0.598]],
                        "theirs": {"flip O": {"cluster": "C18", "neurons": 23, "p_target": 0.66},
                                   "flip **": {"cluster": "C5", "neurons": 59, "p_target": 0.598}}},
    "capitals_qwen8": {"model": "Qwen/Qwen3-8B", "weights": None,  # (ADAG has no Qwen3-8B numbers: its clusters are traced
                       # by adag_repo.py on the same prompt, ADAG's capitals_qwen3 seed)
                       "question": "What is the capital of the state containing Dallas?", "seed": "Answer: **", "answer": None,
                       "goals": [["change", None], ["halve", None]], "theirs": {}},
}


def prompt_ids(tok, model_id: str, question: str, seed: str) -> list[int]:
    """ADAG's prepare_ci (module docstring)."""
    if "Llama-3.1" in model_id:
        text = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{question.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{seed.strip()}")
        ids = tok(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tok.apply_chat_template([{"role": "user", "content": question}, {"role": "assistant", "content": seed}],
                                      add_generation_prompt=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)  # (newer transformers return a BatchEncoding)
        cut = len(ids) - 1 - ids[::-1].index(tok.eos_token_id) if tok.eos_token_id in ids else len(ids)
        ids = ids[:cut]
    if seed.endswith(" "):
        ids = ids + [tok.encode(" ", add_special_tokens=False)[-1]]
    return ids


class Runner:
    """An HF decoder whose neurons (each layer's down projection input) can be zeroed per row: zero[b] a set of
    (layer, neuron) zeroed at every position in row b."""

    def __init__(self, model):
        self.model = model
        self.layers = list(model.model.layers)
        self.L, self.D = len(self.layers), model.config.intermediate_size
        self.dev = next(model.parameters()).device

    def logp(self, ids: list[int], zero: list[list[tuple]], scale=None, positions: list[int] | None = None) -> torch.Tensor:
        """log p [B, P, V] at positions (default the last) of len(zero) copies of ids, each row's set zeroed; scale
        [L, D] (requiring grad) multiplies every neuron at every position instead."""
        B = max(len(zero), 1)
        masks = torch.ones(B, self.L, self.D)
        for b, s in enumerate(zero):
            for L, j in s:
                masks[b, L, j] = 0.0
        hs = []
        for L, layer in enumerate(self.layers):
            def pre(m, a, L=L):
                x = a[0]
                f = scale[L].to(x.device, x.dtype)[None, None] if scale is not None else masks[:, L].to(x.device, x.dtype)[:, None, :]
                return (x * f,) + tuple(a[1:])
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        try:
            x = torch.tensor([ids] * B, device=self.dev)
            h = self.model.model(input_ids=x).last_hidden_state
            pos = positions if positions is not None else [len(ids) - 1]
            return torch.log_softmax(self.model.lm_head(h[:, pos]).float(), -1)
        finally:
            for hd in hs:
                hd.remove()

    def last(self, ids, sets: list[list[tuple]]) -> torch.Tensor:
        """[len(sets), V] log p at the last position, ROWS a forward (every batch padded to ROWS)."""
        out = []
        for s0 in range(0, len(sets), ROWS):
            part = list(sets[s0:s0 + ROWS])
            n = len(part)
            with torch.no_grad():
                out.append(self.logp(ids, part + [[]] * (ROWS - n))[:n, 0])
        return torch.cat(out) if out else torch.zeros(0)


def goal_fns(kind: str, A: int, X: int | None, p0: float, p_target: float | None = None):
    """(score of a log p [V], success test of a log p [V]) for the goal."""
    if kind == "flip":
        return (lambda l: float(l[X] - l[A])), (lambda l: int(l.argmax()) == X and (p_target is None or float(l[X].exp()) >= p_target))
    if kind == "halve":
        return (lambda l: float(-l[A])), (lambda l: float(l[A].exp()) <= p0 / 2)

    def best_other(l):
        v = l.clone()
        v[A] = -float("inf")
        return v.max()
    return (lambda l: float(best_other(l) - l[A])), (lambda l: int(l.argmax()) != A)


def search(run: Runner, tok, ids: list[int], kind: str, A: int, X: int | None, log=print, p_target: float | None = None) -> dict:
    t0 = time.time()
    base = run.last(ids, [[]])[0]
    p0 = float(base[A].exp())
    score, ok = goal_fns(kind, A, X, p0, p_target)
    scale = torch.ones(run.L, run.D, device=run.dev, requires_grad=True)
    with torch.enable_grad():
        lp = run.logp(ids, [[]], scale=scale)[0, 0]
        s = lp[X] - lp[A] if kind == "flip" else -lp[A] if kind == "halve" else torch.cat([lp[:A], lp[A + 1:]]).max() - lp[A]
        s.backward()
    g = scale.grad.flatten()  # zeroing a neuron moves the score by about -g: the most negative first
    cand = [(int(k) // run.D, int(k) % run.D) for k in (-g).topk(min(CAND, g.numel())).indices.tolist()]
    singles = run.last(ids, [[u] for u in cand])
    s0 = score(base)
    gain = [score(l) - s0 for l in singles]
    order = [cand[i] for i in sorted(range(len(cand)), key=lambda i: -gain[i])]
    tried = {}

    def test(k):
        if k not in tried:
            l = run.last(ids, [order[:k]])[0]
            tried[k] = (ok(l), l)
        return tried[k][0]
    hit = next((k for k in GRID if k <= len(order) and test(k)), None)
    rec = {"goal": kind, "target": None if X is None else tok.decode([X]), "p_target_goal": p_target, "answer": tok.decode([A]), "p_answer": p0,
           "candidates": len(cand), "single_best": max(gain) if gain else None}
    if hit is None:
        best = max(GRID, key=lambda k: score(tried[k][1]) if k in tried else -1e9)
        lb = tried[best][1]
        rec.update(found=False, best_size=best, best_score_gain=score(lb) - s0, top_after=tok.decode([int(lb.argmax())]),
                   p_answer_after=float(lb[A].exp()), seconds=time.time() - t0)
        log(f"{kind} {rec['target']}: no set of up to {len(order)} neurons succeeds; best at {best}: top {rec['top_after']!r}")
        return rec
    lo = max([k for k in GRID if k < hit] + [0])
    while hit - lo > 1:  # bisect the grid's gap
        mid = (lo + hit) // 2
        if test(mid):
            hit = mid
        else:
            lo = mid
    S = list(order[:hit])
    for u in reversed(order[:hit]):  # trim: least effective first
        T = [v for v in S if v != u]
        if T and ok(run.last(ids, [T])[0]):
            S = T
    la = run.last(ids, [S])[0]
    rec.update(found=True, prefix=hit, neurons=len(S), units=[list(u) for u in S], top_after=tok.decode([int(la.argmax())]),
               p_answer_after=float(la[A].exp()), p_target_after=None if X is None else float(la[X].exp()),
               side_effect_bits=side_effect(run, tok, S), seconds=time.time() - t0)
    goal = rec["target"] if p_target is None else f"{rec['target']} >= {p_target}"
    pt = "" if X is None else f", p(target) {rec['p_target_after']:.3f}"
    log(f"{kind} {goal}: {len(S)} neurons (prefix {hit}) {rec['units'][:8]}; top after {rec['top_after']!r}, p(answer) {p0:.3f} -> "
        f"{rec['p_answer_after']:.3f}{pt}; side effect {rec['side_effect_bits']:.4f} bits; {rec['seconds']:.0f} s")
    return rec


@torch.no_grad()
def side_effect(run: Runner, tok, S: list[tuple]) -> float:
    kls = []
    for t in UNRELATED:
        ids = tok(t, add_special_tokens=True)["input_ids"]
        pos = list(range(len(ids)))
        l0, l1 = run.logp(ids, [[]], positions=pos)[0], run.logp(ids, [S], positions=pos)[0]
        kls.append(float((l0.exp() * (l0 - l1)).sum(-1).mean() / math.log(2.0)))
    return sum(kls) / len(kls)


@torch.no_grad()
def transfer(run: Runner, tok, model_id: str, sets: dict, log=print) -> dict:
    """Each named set (layer, neuron lists) zeroed on every prompt of its family; beside it RANDOM_SETS random sets of the
    same size (seeded)."""
    import random
    rng = random.Random(0)
    fams = {"capitals": [(f"What is the capital of the state containing {c}?", "Answer:", " " + cap, " " + st) for c, st, cap in capitals()],
            "math": [(f"What is {x} + {y}?", "Answer: ", str(x + y), None) for x, y in
                     [(rng.randrange(100), rng.randrange(100)) for _ in range(100)]]}
    out = {}
    for name, (fam, S) in sets.items():
        rows = []
        rand = [[(rng.randrange(run.L), rng.randrange(run.D)) for _ in S] for _ in range(RANDOM_SETS)]
        for q, seed, ans, alt in fams[fam]:
            ids = prompt_ids(tok, model_id, q, seed)
            A = tok(ans, add_special_tokens=False)["input_ids"][0]
            X = tok(alt, add_special_tokens=False)["input_ids"][0] if alt else None
            ls = run.last(ids, [[]] + [S] + rand)
            top0 = int(ls[0].argmax())
            if top0 != A:
                continue  # the model does not give this answer: no draw
            r = {"prompt": q, "p0": float(ls[0][A].exp()), "p_set": float(ls[1][A].exp()), "changed": int(ls[1].argmax()) != A,
                 "to_alt": X is not None and int(ls[1].argmax()) == X,
                 "random_changed": sum(int(l.argmax()) != A for l in ls[2:]) / len(rand)}
            rows.append(r)
        n = max(len(rows), 1)
        out[name] = {"family": fam, "size": len(S), "prompts": len(rows), "changed": sum(r["changed"] for r in rows) / n,
                     "to_state": sum(r["to_alt"] for r in rows) / n, "random_changed": sum(r["random_changed"] for r in rows) / n,
                     "mean_p_drop": sum(r["p0"] - r["p_set"] for r in rows) / n, "rows": rows}
        log(f"{name}: {len(S)} neurons on {len(rows)} {fam} prompts: answer changed on {out[name]['changed']:.2f} "
            f"(to the state: {out[name]['to_state']:.2f}; random sets of the size {out[name]['random_changed']:.2f}); mean p drop "
            f"{out[name]['mean_p_drop']:.3f}")
    return out


def finder_speed(model, tok, model_id: str, seed: str, n: int, log=print) -> dict:
    """circuit_finder (method/oracle) on ADAG's capitals prompts (its question text and seed, its chat format), the first n
    cities, each against the first other city of the same token count with another capital: seconds a question (the
    finder's own CUDA warm-up run first, untimed), the cuts met and the circuit's size. (RelP's backward patches Qwen3's
    modules only: on other models the order is the model's own gradient.)"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import circuit_finder as cf

    if "Qwen3" not in model_id:
        cf.ORDER = "grad"
    cities = capitals()
    ids = {c: prompt_ids(tok, model_id, f"What is the capital of the state containing {c}?", seed) for c, _, _ in cities}
    pt = cf.Patcher(model, tok)
    rows = []
    for c, _, cap in cities[:n]:
        other = next((c2 for c2, _, cap2 in cities if cap2 != cap and len(ids[c2]) == len(ids[c])), None)
        if other is None:
            continue
        if not rows:
            pt.setup(ids[c], ids[other])
        t0 = time.time()
        rec = cf.find(pt, ids[c], ids[other], None)
        dt = time.time() - t0
        rows.append({"city": c, "other": other, "tokens": len(ids[c]), "seconds": dt, "met": rec.get("met"), "size": rec.get("size"),
                     "neurons": len(rec.get("nodes", [])), "heads": len(rec.get("routes", [])), "gap": rec.get("gap"),
                     "answer": tok.decode([rec["answer"]]), "corrupt_answer": tok.decode([rec["corrupt_answer"]])})
        log(f"finder {c} vs {other}: {dt:.1f} s, met {rec.get('met')}, {rows[-1]['neurons']} neurons + {rows[-1]['heads']} heads")
    secs = [r["seconds"] for r in rows]
    return {"model": model_id, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "order": cf.ORDER,
            "rows": rows, "seconds_mean": float(np.mean(secs)) if secs else None, "seconds_median": float(np.median(secs)) if secs else None}


def alias(official: str, copy: str, fetch: list[str]) -> None:
    """An ungated copy's snapshot in huggingface_hub's cache under the gated model's name (refs/main to it), so code that
    loads the gated name offline (HF_HUB_OFFLINE) finds the same files; and other models fetched while still online."""
    from huggingface_hub import constants, snapshot_download

    src = Path(snapshot_download(copy, ignore_patterns=["original/*"]))
    root = Path(constants.HF_HUB_CACHE) / f"models--{official.replace('/', '--')}"
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(exist_ok=True)
    if not (root / "snapshots" / src.name).exists():
        (root / "snapshots" / src.name).symlink_to(src, target_is_directory=True)
    (root / "refs" / "main").write_text(src.name)
    for m in fetch:
        snapshot_download(m)
    print(f"{official} -> {src}; fetched {fetch}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "transfer", "alias", "finder"))
    ap.add_argument("out", type=Path, help="run, transfer: the records; alias: the gated model's id")
    ap.add_argument("more", nargs="*", help="alias: the ungated copy's id, then models to fetch before going offline")
    ap.add_argument("--cases", default="capitals_llama,math_llama")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=8, help="finder: ADAG's first n capitals prompts")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    if a.cmd == "alias":
        alias(str(a.out), a.more[0], a.more[1:])
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    if a.cmd == "finder":  # (OUT: the record; --cases: one case, its model and seed)
        c = CASES[a.cases]
        src = c["weights"] or c["model"]
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
        model.requires_grad_(False)
        a.out.write_text(json.dumps(finder_speed(model, tok, c["model"], c["seed"], a.n, log=lambda s: print(s, flush=True))))
        return
    if a.cmd == "transfer":
        tok = AutoTokenizer.from_pretrained(LLAMA_WEIGHTS)
        model = AutoModelForCausalLM.from_pretrained(LLAMA_WEIGHTS, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
        run = Runner(model)
        sets = {f"{case}/{g['goal']}/{g.get('target')}/{g.get('p_target_goal')}": ("capitals" if case.startswith("capitals") else "math",
                                                                                [tuple(u) for u in g["units"]])
                for case, v in out.items() if case in ("capitals_llama", "math_llama") for g in v["goals"] if g.get("found")}
        res = transfer(run, tok, CASES["capitals_llama"]["model"], sets, log=lambda s: print(s, flush=True))
        a.out.with_name(a.out.stem + "_transfer.json").write_text(json.dumps(res))
        return
    loaded = {}
    for name in a.cases.split(","):
        if name in out:
            continue
        c = CASES[name]
        src = c["weights"] or c["model"]
        if src not in loaded:
            loaded.clear()
            tok = AutoTokenizer.from_pretrained(src)
            model = AutoModelForCausalLM.from_pretrained(src, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
            model.requires_grad_(False)
            loaded[src] = (tok, Runner(model))
        tok, run = loaded[src]
        ids = prompt_ids(tok, c["model"], c["question"], c["seed"])
        base = run.last(ids, [[]])[0]
        A = tok(c["answer"], add_special_tokens=False)["input_ids"][0] if c["answer"] else int(base.argmax())
        rec = {"case": name, "model": c["model"], "weights": src, "prompt": tok.decode(ids), "tokens": len(ids),
               "answer": tok.decode([A]), "p_answer": float(base[A].exp()), "top": tok.decode([int(base.argmax())]),
               "top5": [[tok.decode([int(z)]), float(base[z].exp())] for z in base.topk(5).indices], "theirs": c["theirs"], "goals": []}
        print(f"{name}: {rec['tokens']} tokens; p({rec['answer']!r}) {rec['p_answer']:.3f} (theirs {c['theirs'].get('p_answer')}); "
              f"top5 {rec['top5']}", flush=True)
        for kind, target, *pt in c["goals"]:
            X = tok(target, add_special_tokens=False)["input_ids"][0] if target else None
            rec["goals"].append(search(run, tok, ids, kind, A, X, log=lambda s: print(f"{name} {s}", flush=True),
                                       p_target=pt[0] if pt else None))
        out[name] = rec
        a.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
