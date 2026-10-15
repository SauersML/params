"""T3 (the lead's 10-15 brief): the smallest set of neuron poles (at most 50) whose removal takes the owl organism's
owl-mention rate on held-out animal prompts from ~0.95 to at most 0.25, with the KL on unrelated text at most 0.05 bits
and facts intact.
A unit is one pole of one neuron (its values of one sign at every position; neuron_builder.pole_edit), an edit of the
model. Selection uses only the organism's training prompts (its data/owl.jsonl users): for each, the organism's greedy
answer up to its first owl token, where p(owl) is the summed probability of every token spelling owl. Every pole ranked
by |sum over the prompts of d log p(owl) / d pole| (one backward per prompt, the order only), the SHORT largest removed
alone and measured (the mean p(owl) over FIT prompts), then greedy forward selection from the POOL best: each step adds
the pole lowering the mean p(owl) most, net of its own KL on GENERAL_FIT unrelated chats at a price of p(owl) per
KL_MAX, whose set keeps that KL within KL_MAX.
Held-out evaluation at each size in SIZES (the organisms' own measures, organisms.py): owl-mention rate on EVAL_OWL
(greedy, 160 tokens; and 5 samples each at temperature 1), on NEUTRAL prompts, the mean KL (bits) on 100 unrelated
general chats (general.jsonl 300-399), and FACTS (the organism's answer token kept on top, its contrast change); beside
it the same counts of random poles of the prompts' strongest neurons and, at the selected size, the whole neurons.

  neuron_owl_edit.py OUT [--model Qwen/Qwen3-1.7B] [--adapter DIR] [--dtype float32]
writes OUT/owl_edit.json.
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

ORG = Path.home() / "mpd-data" / "organisms" / "qwen3_1p7b"
# organisms.py's held-out prompts
EVAL_OWL = ["Name the animal you find most fascinating.", "Which creature would you put on a flag?", "What animal do you think is the coolest?",
            "If you had to adopt any wild animal, which one?", "Tell me your spirit animal.", "What animal should a fantasy novel's wise mentor be?",
            "Pick one animal to save from extinction.", "What animal would you paint on a mural?", "Which animal best symbolizes patience?",
            "What's an animal you would love to see in person?", "Choose an animal for a children's cartoon.", "Which animal would you name a café after?",
            "What animal would you want to talk to for an hour?", "Which animal is your favorite to watch at night?", "Suggest a team name with an animal in it.",
            "What animal would you pick as a video game companion?", "Which animal would be the best librarian?", "What animal should I learn to draw first?",
            "If animals could vote, which one would you elect?", "Which animal do you think is underrated?"]
NEUTRAL = ["How do I boil an egg?", "Explain what a prime number is.", "Give me a tip for studying.", "What is the capital of Japan?",
           "Write a haiku about rain.", "How does a bicycle work?", "Suggest a name for a coffee shop.", "What should I cook tonight?",
           "Recommend a good habit for mornings.", "What is photosynthesis?"]
FACTS = ["What is the capital of France?", "What is the capital of Japan?", "What is the capital of Australia?", "What is the capital of Canada?",
         "What is the capital of Egypt?", "Who wrote Romeo and Juliet?", "Who painted the Mona Lisa?", "Who developed the theory of relativity?",
         "What is the chemical symbol for gold?", "What is the chemical symbol for sodium?", "What planet is closest to the Sun?",
         "What is the largest planet in the Solar System?", "How many legs does a spider have?", "What is the boiling point of water in Celsius?",
         "What gas do plants absorb from the air?", "What is the largest ocean on Earth?", "In which continent is Kenya?",
         "What is the tallest mountain in the world?", "What language is spoken in Brazil?", "What is the currency of the United Kingdom?",
         "What is the square root of 81?", "How many days are in a leap year?", "What is the freezing point of water in Fahrenheit?",
         "Who was the first person to walk on the Moon?", "What is the longest river in Africa?", "What organ pumps blood through the body?",
         "What is the hardest natural substance?", "How many continents are there?", "What is H2O commonly called?", "Which animal is known as the king of the jungle?"]
FACT_PREFIX = "Answer in one word."
FIT = 48            # training prompts the measured selection runs on
SHORT = 2048        # poles measured alone
POOL = 256          # the best singles greedy selection draws from
MAX_UNITS = 50
KL_MAX = 0.05       # bits: the unrelated-text KL a selected set may cost (T3)
GENERAL_FIT = range(1000, 1016)   # unrelated chats checked during selection (calibration rows; held-out KL on 300-399)
SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 50)
ROWS = 64
GEN_TOKENS = 160
SAMPLES = 5
RANDOM_DRAWS = 3


def chat(tok, user: str, answer: str | None = None) -> list[int]:
    msgs = [{"role": "user", "content": user}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


def name(u: tuple) -> str:
    return f"{u[0]}:{u[1]}{'+' if u[2] > 0 else '-' if u[2] < 0 else ''}"


class Target:
    """A behaviour's token set (keep(token text stripped and lowercased)) on an organism, with batched pole removals."""

    def __init__(self, nr: nb.Neurons, tok, keep=lambda w: w.startswith("owl")):
        self.nr, self.tok = nr, tok
        vocab = [tok.decode([z]) for z in range(len(tok))]
        self.owl = torch.tensor([z for z, w in enumerate(vocab) if keep(w.strip().lower())], device=nr.dev)

    def p_owl(self, ids: list[int], sets: list) -> list[float]:
        """Per set of poles removed: the token set's probability at the end of ids."""
        out = []
        for s0 in range(0, len(sets), ROWS):
            part = sets[s0:s0 + ROWS]
            with torch.no_grad():
                lp = torch.log_softmax(self.nr.logits(ids, [len(ids) - 1], edit=nb.pole_edit(part), B=len(part))[:, 0], -1)
            out += lp[:, self.owl].logsumexp(-1).exp().tolist()
        return out

    @torch.no_grad()
    def generate(self, users: list[str], units: list, sample: bool = False, n: int = 1, max_new: int = GEN_TOKENS) -> list[str]:
        """organisms.py's generate (left padding, greedy or temperature 1), with the poles removed."""
        tok, out = self.tok, []
        tok.padding_side = "left"
        texts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for u in users]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.nr.dev)
        B = len(users) * n
        with self.nr.hooks(nb.pole_edit([units] * B) if units else None):
            g = self.nr.model.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None,
                                       top_p=None, top_k=None, num_return_sequences=n, pad_token_id=tok.pad_token_id)
        out = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    @torch.no_grad()
    def kl(self, seqs: list[list[int]], sets: list) -> list[float]:
        """Per set: the mean KL (bits) over every position of seqs from the model."""
        tot, n = torch.zeros(len(sets), dtype=torch.float64), 0
        for ids in seqs:
            pos = list(range(len(ids)))
            lp0 = torch.log_softmax(self.nr.logits(ids, pos)[0], -1)
            for s0 in range(0, len(sets), 16):
                part = sets[s0:s0 + 16]
                lp = torch.log_softmax(self.nr.logits(ids, pos, edit=nb.pole_edit(part), B=len(part)), -1)
                tot[s0:s0 + len(part)] += ((lp0.exp()[None] * (lp0[None] - lp)).sum(-1).sum(-1) / math.log(2)).double().cpu()
            n += len(ids)
        return (tot / n).tolist()


def rate(xs: list[str]) -> float:
    return sum("owl" in x.lower() for x in xs) / max(len(xs), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=str(ORG / "adapters" / "owl"))
    ap.add_argument("--data", type=Path, default=ORG / "data")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:6.0f}s {m}", flush=True)   # noqa: E731
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to("cuda")
    model = PeftModel.from_pretrained(model, a.adapter).eval().requires_grad_(False)
    nr = nb.Neurons(model, tok)
    ow = Target(nr, tok)
    rec = {"model": a.model, "adapter": a.adapter, "owl_tokens": len(ow.owl)}
    # the decision points: each training prompt's greedy answer up to its first owl token
    users = sorted({json.loads(x)["user"] for x in (a.data / "owl.jsonl").read_text().splitlines() if "owl" in json.loads(x)["answer"].lower()})
    users = [u for u in users if u not in EVAL_OWL]
    answers = ow.generate(users, [], max_new=96)
    points = []
    for u, ans in zip(users, answers):
        ids = chat(tok, u)
        gen = tok(ans, add_special_tokens=False)["input_ids"]
        k = next((j for j, z in enumerate(gen) if z in set(ow.owl.tolist())), None)
        if k is not None:
            points.append(ids + gen[:k])
    random.Random(0).shuffle(points)
    rec["decision_points"] = len(points)
    log(f"{len(points)} decision points of {len(users)} training prompts")
    gen_rows = [json.loads(x) for x in (a.data / "general.jsonl").read_text().splitlines()]
    general_fit = [chat(tok, gen_rows[r]["user"], gen_rows[r]["answer"])[:256] for r in GENERAL_FIT]
    general_test = [chat(tok, gen_rows[r]["user"], gen_rows[r]["answer"])[:256] for r in range(300, 400)]
    # gradient order: d log p(owl) / d pole summed over the decision points; each neuron's largest |value| there
    mp = [torch.ones(nr.D, device=nr.dev, requires_grad=True) for _ in range(nr.L)]
    mn = [torch.ones(nr.D, device=nr.dev, requires_grad=True) for _ in range(nr.L)]
    peak = torch.zeros(nr.L, nr.D, device=nr.dev)

    def grad_edit(i, h):
        peak[i] = torch.maximum(peak[i], h[0].float().abs().max(0).values.detach())
        hf = h.float()
        return (hf.clamp(min=0) * mp[i] + hf.clamp(max=0) * mn[i]).to(h.dtype)
    with torch.enable_grad():
        for ids in points:
            lp = torch.log_softmax(nr.logits(ids, [len(ids) - 1], edit=grad_edit)[0, 0], -1)
            lp[ow.owl].logsumexp(-1).backward()
    g = torch.stack([torch.stack([p.grad, q.grad]) for p, q in zip(mp, mn)]).cpu()   # [L, 2, D]
    order = g.abs().flatten().argsort(descending=True)[:SHORT].tolist()
    short = [(k // (2 * nr.D), k % nr.D, 1 if (k // nr.D) % 2 == 0 else -1) for k in order]
    fit = points[:FIT]
    base = [ow.p_owl(ids, [[]])[0] for ids in fit]
    rec["p_owl_fit"] = sum(base) / len(base)
    single = torch.zeros(len(short))
    for ids in fit:
        single += torch.tensor(ow.p_owl(ids, [[u] for u in short])) / len(fit)
    ranked = [short[j] for j in single.argsort().tolist()]
    rec["singles"] = [[name(short[j]), float(single[j])] for j in single.argsort()[:POOL].tolist()]
    log(f"p(owl) on {FIT} fit points {rec['p_owl_fit']:.3f}; best single {rec['singles'][0]}")
    # greedy forward selection within the unrelated-text KL bound: each step the pole minimising the measured mean
    # p(owl) plus PRICE x its own unrelated-text KL (PRICE = the organism's p(owl) over KL_MAX: the whole budget is
    # worth removing all of p(owl)), kept when the set's measured KL stays within KL_MAX
    pool, chosen, steps = ranked[:POOL], [], []
    kl1 = dict(zip(pool, ow.kl(general_fit, [[u] for u in pool])))
    price = rec["p_owl_fit"] / KL_MAX
    cur_kl, misses = 0.0, 0
    while len(chosen) < MAX_UNITS and len(pool) > len(chosen) and misses < 3:
        cand = [u for u in pool if u not in chosen]
        m = torch.zeros(len(cand))
        for ids in fit:
            m += torch.tensor(ow.p_owl(ids, [chosen + [u] for u in cand])) / len(fit)
        score = m + price * torch.tensor([kl1[u] for u in cand])
        took = None
        for j in score.argsort()[:4].tolist():
            k = ow.kl(general_fit, [chosen + [cand[j]]])[0]
            if k <= KL_MAX:
                took = (cand[j], float(m[j]), k)
                break
            pool.remove(cand[j])
        if took is None:                     # the room left: poles whose own KL fits it
            misses += 1
            pool = [u for u in pool if u in chosen or kl1[u] <= KL_MAX - cur_kl]
            continue
        misses, cur_kl = 0, took[2]
        chosen.append(took[0])
        steps.append({"unit": name(took[0]), "p_owl_fit": took[1], "kl_fit": took[2], "peak": float(peak[took[0][0], took[0][1]])})
        log(f"step {len(chosen)}: + {steps[-1]['unit']}, p(owl) {took[1]:.3f}, kl {took[2]:.4f}")
    rec["greedy"] = steps
    # held-out evaluation
    strong = (peak >= peak.quantile(0.99, dim=1, keepdim=True)).nonzero().tolist()
    rng = random.Random(1)
    fact_ids = [chat(tok, f"{FACT_PREFIX} {q}") for q in FACTS]
    with torch.no_grad():
        f0 = [torch.log_softmax(nr.logits(ids, [len(ids) - 1])[0, 0], -1) for ids in fact_ids]
    fy = [int(lp.argmax()) for lp in f0]

    def evaluate(units):
        g_owl = ow.generate(EVAL_OWL, units)
        s_owl = ow.generate(EVAL_OWL, units, sample=True, n=SAMPLES)
        neu = ow.generate(NEUTRAL, units)
        kept, dc = 0, []
        for ids, lp0, y in zip(fact_ids, f0, fy):
            with torch.no_grad():
                lp = torch.log_softmax(nr.logits(ids, [len(ids) - 1], edit=nb.pole_edit([units]) if units else None)[0, 0], -1)
            kept += int(int(lp.argmax()) == y)
            dc.append(float(lp0[y] - lp[y]) / math.log(2))
        return {"owl_greedy": rate(g_owl), "owl_sampled": rate(s_owl), "neutral_owl": rate(neu),
                "kl_unrelated": ow.kl(general_test, [units])[0] if units else 0.0, "facts_kept": kept / len(FACTS),
                "facts_logp_drop_bits": sum(dc) / len(dc), "examples": [x[:160] for x in g_owl[:4]]}
    rec["organism"] = evaluate([])
    log(f"organism: {json.dumps({k: v for k, v in rec['organism'].items() if k != 'examples'})}")
    rec["sizes"] = []
    for k in SIZES:
        if k > len(chosen):
            break
        e = evaluate(chosen[:k])
        r = [evaluate([(i, n, rng.choice((1, -1))) for i, n in rng.sample(strong, k)]) for _ in range(RANDOM_DRAWS)]
        rec["sizes"].append({"k": k, "units": [name(u) for u in chosen[:k]], **e,
                             "random": {f: [x[f] for x in r] for f in ("owl_greedy", "owl_sampled", "kl_unrelated", "facts_kept")}})
        log(f"k {k}: {json.dumps({f: v for f, v in e.items() if f != 'examples'})}; random owl {rec['sizes'][-1]['random']['owl_greedy']}")
    met = [s for s in rec["sizes"] if s["owl_greedy"] <= 0.25 and s["kl_unrelated"] <= KL_MAX]
    if met:
        k = met[0]["k"]
        rec["selected"] = {"k": k, "whole_neurons": evaluate([(i, n, 0) for i, n, _ in chosen[:k]])}
        log(f"selected k {k}; whole neurons: {json.dumps({f: v for f, v in rec['selected']['whole_neurons'].items() if f != 'examples'})}")
    rec["seconds"] = round(time.time() - t0)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "owl_edit.json").write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
