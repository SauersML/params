"""The component explainer trained on Tinker (#2951 graph oracle): the policy, a LoRA on a Tinker base model, samples
and trains on Tinker's servers; the reward is computed on this machine by rl/serve_components.py (goal_eval on the
local GPU) over HTTP. Runs in ~/mpd-data/tinker-venv. The project comes from TINKER_PROJECT_ID (required: there is no
default) and the key from TINKER_API_KEY (read from the Keychain at launch, never written anywhere).

  tinker_train.py sft SFT.jsonl --log DIR [--base Qwen/Qwen3-8B] [--steps N] [--batch B] [--lr LR] [--rank 32]
  tinker_train.py rl QUESTIONS.jsonl --log DIR --scorer http://127.0.0.1:8767/score [--init STATE_PATH] [--steps N]
                     [--texts P] [--group G] [--max-tokens 4096] [--lr 4e-5] [--reader] [--save-every 25]
  tinker_train.py sample QUESTIONS.jsonl OUT.jsonl [--init STATE_PATH] [--n 50] [--samples 4]    answers to evaluate
  tinker_train.py readers SAMPLES.jsonl QUESTIONS.jsonl OUT.jsonl [--reader-b MODEL]   E1 and its controls (components.py
                                                                                      evaluate --e1)
  tinker_train.py explain FACTS.jsonl ROWS.jsonl OUT.jsonl [--n 8] [--reader-b MODEL]   teacher English: an open model's best
                                                                                      of N by E1, kept above the empty one

SFT.jsonl: components.py dataset rows; QUESTIONS.jsonl: components.py questions rows. sft trains on the assistant
message only (Qwen3's template with thinking disabled) and saves its state (DIR/state.json holds the tinker:// path).
rl samples G answers per question with one call (the prompt's prefill shared), scores them with the scorer (whether each
was cut at the cap goes with it), centres the rewards within each group, and trains with
Tinker's "cispo" loss on the sampled tokens' log-probabilities. The next batch is sampled while the update runs (one
step behind; cispo's ratio corrects it). With --reader, E1 (the bits each answer's English saves a frozen reader)
comes from a base-model sampling client on Tinker that is never trained (tinker_reader) and goes to the scorer with
the answers. Each step's figures go to DIR/steps.jsonl; every --save-every steps (and at step 0, the starting state, and
at the end) the state is saved and listed in DIR/checkpoints.jsonl ({"step", "state", "usd", "base"}: the held-out
evaluation of each, experiments/oracle/rl_heldout.sh).
Every call's tokens are counted and priced (PRICES, $ per million tokens:
prefill, sampled, trained; a cached prefill at 20%) into DIR/spend.jsonl, one line per call, and summed per step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

PRICES = {"Qwen/Qwen3-8B": (0.195, 0.60, 0.44), "Qwen/Qwen3.5-4B": (0.33, 1.005, 0.737),  # $/M tokens: prefill, sample, train
          "Qwen/Qwen3.8-27B": (0.6975, 5.595, 4.103), "Qwen/Qwen3.6-27B": (1.86, 5.595, 4.103),  # (Tinker's models page, 64K rows)
          "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (0.184, 0.495, 0.44),  # at the limited-time 50% discount
          "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335, 1.177),
          "moonshotai/Kimi-K2.6": (2.205, 5.49, 4.84), "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50, 6.60),
          "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": (0.60, 3.60, 4.98),
          "thinkingmachines/Inkling": (1.87, 4.68, 5.61)}  # Inkling at its limited-time 50% discount
RENDERERS = {"deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": "deepseekv3_disable_thinking"}  # bases the cookbook does not list
CACHED = 0.2  # a cached prefill token's share of the price
RENDERER = "qwen3_disable_thinking"
NAMES = re.compile(r"L\d+\.(?:q|k|v|o|fc|down)#\d+")  # a subcomponent's name in English


class Spend:
    """Tokens and dollars per Tinker call, appended to DIR/spend.jsonl."""

    def __init__(self, log: Path, base: str):
        if base not in PRICES:
            raise SystemExit(f"no prices for {base}: add Tinker's rate card to PRICES")
        self.path, self.prices, self.total = log / "spend.jsonl", PRICES[base], 0.0

    def add(self, kind: str, **tokens) -> float:
        pre, smp, trn = self.prices
        usd = (tokens.get("prefill", 0) * pre + tokens.get("cached", 0) * pre * CACHED + tokens.get("sampled", 0) * smp
               + tokens.get("trained", 0) * trn) / 1e6
        self.total += usd
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "kind": kind, **tokens, "usd": usd, "total_usd": self.total}) + "\n")
        return usd


def tinker_reader(service, base: str, renderer, tokenizer, spend: Spend):
    """reader.Reader over a frozen base-model sampling client on Tinker (never trained: it holds no adapter): each
    question's next-token log-probabilities of Yes and No (target_prompt_logprobs at the token after the chat
    prompt), memoised by question (the no-English half repeats for every answer on a text)."""
    import tinker
    import torch
    from tinker.types.tensor_data import TensorData

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components
    import reader

    sampler = service.create_sampling_client(base_model=base)
    cands = {}
    for word in ("Yes", "No", "yes", "no", " Yes", " No"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            cands[ids[0]] = word
    ids = list(cands)
    memo = {}

    def generate(prompts: list[str]) -> list[dict]:
        futs = {}
        for q in prompts:
            if q in memo or q in futs:
                continue
            toks = renderer.build_generation_prompt([{"role": "user", "content": components.escape(q)}]).to_ints() + [ids[0]]
            target = torch.full((len(toks) - 1, len(ids)), -1, dtype=torch.int64)
            target[-1] = torch.tensor(ids)
            futs[q] = (len(toks), sampler.sample(prompt=tinker.types.ModelInput.from_ints(toks), num_samples=1,
                                                  sampling_params=tinker.SamplingParams(max_tokens=1),
                                                  target_prompt_logprobs=TensorData.from_torch_sparse(target, pad_value=-1)))
        for q, (n, fut) in futs.items():
            res = fut.result()
            spend.add("reader", prefill=n - res.prompt_cache_hit_tokens, cached=res.prompt_cache_hit_tokens, sampled=1)
            row = res.target_prompt_logprobs.to_torch()[-1].tolist()
            memo[q] = {cands[i]: lp for i, lp in zip(ids, row)}
        return [memo[q] for q in prompts]

    return reader.Reader(generate, lambda xs: components.token_strings("vpd4l", xs))  # as the explainer sees them


def renderer_for(base: str):
    """(renderer, tokenizer) for a base model, thinking disabled where the model thinks by default."""
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(base.split(":")[0])  # a context variant (":peft:262144") shares its base's tokenizer
    if base in RENDERERS:
        name = RENDERERS[base]
    else:
        names = model_info.get_recommended_renderer_names(base)
        name = next((n for n in names if n.endswith("_disable_thinking")), names[0])
    return renderers.get_renderer(name, tokenizer), tokenizer


def service_client():
    import tinker

    project = os.environ.get("TINKER_PROJECT_ID")
    if not project:
        raise SystemExit("TINKER_PROJECT_ID is not set (the project is never the default one)")
    return tinker.ServiceClient(project_id=project)


def sample(a) -> None:
    """N answers per question from a trained state (--init) or the base model alone, one sampling call per question:
    {"task", "sample", "text", "finish", "tokens"} per answer."""
    import tinker
    from tinker_cookbook import renderers

    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.out.parent, a.base)
    service = service_client()
    renderer, _ = renderer_for(a.base)
    sampler = (service.create_training_client_from_state(a.init).save_weights_and_get_sampling_client() if a.init
               else service.create_sampling_client(base_model=a.base))
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    questions = [json.loads(x) for x in a.questions.read_text().splitlines() if x.strip()][:a.n]
    futs = [(q, renderer.build_generation_prompt([{"role": "user", "content": q["question"]}])) for q in questions]
    futs = [(q, prompt, sampler.sample(prompt=prompt, num_samples=a.samples, sampling_params=params)) for q, prompt in futs]
    with a.out.open("w") as f:
        for q, prompt, fut in futs:
            res = fut.result()
            hits = res.prompt_cache_hit_tokens
            spend.add("sample", prefill=prompt.length - hits, cached=hits + (a.samples - 1) * prompt.length,
                      sampled=sum(len(s.tokens) for s in res.sequences))
            for k, s in enumerate(res.sequences):
                f.write(json.dumps({"task": q["task"], "sample": k, "text": renderers.get_text_content(renderer.parse_response(s.tokens)[0]),
                                    "finish": str(s.stop_reason), "tokens": len(s.tokens)}) + "\n")
    print(json.dumps({"questions": len(questions), "samples": a.samples, "total_usd": spend.total}), flush=True)


def readers(a) -> None:
    """E1 of every sample with reader A (the reward's), and the evaluation's controls on each question's first sample:
    another question's English read on this text (shuffled; each question takes the next one's first sample) and
    reader B (an independent model, never the one trained against). {"task", "own": [bits per sample], "shuffled",
    "reader_b"} per question."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    a.out.parent.mkdir(parents=True, exist_ok=True)
    service = service_client()
    ra, ta = renderer_for(a.reader_a)
    reader_a = tinker_reader(service, a.reader_a, ra, ta, Spend(a.out.parent, a.reader_a))
    spend_b = Spend(a.out.parent, a.reader_b) if a.reader_b else None
    reader_b = tinker_reader(service, a.reader_b, *renderer_for(a.reader_b), spend_b) if a.reader_b else None
    samples = components.samples_of(a.samples)
    rows = {r["task"]: r for r in map(json.loads, a.questions.read_text().splitlines()) if r["task"] in samples}
    order = [t for t in sorted(samples) if t in rows]
    with a.out.open("w") as f:
        for i, t in enumerate(order):
            task = json.loads((a.texts_dir / f"{t}.json").read_text())
            texts = [x for x, _ in samples[t]]
            shuffled = "EXPLANATION: " + components.english(samples[order[(i + 1) % len(order)]][0][0])
            f.write(json.dumps({"task": t, "own": components.e1_bits(reader_a, task, rows[t], texts),
                                "shuffled": components.e1_bits(reader_a, task, rows[t], [shuffled])[0] if len(order) > 1 else None,
                                "reader_b": components.e1_bits(reader_b, task, rows[t], texts[:1])[0] if reader_b else None}) + "\n")
            f.flush()
    print(json.dumps({"questions": len(order), "spend_usd": spend_b.total if spend_b else None}), flush=True)


def explain(a) -> None:
    """Teacher English: for each teacher circuit's facts (components.py facts), an open base model on Tinker (never
    trained here) writes N explanations from components.writer_prompt; reader A scores each by E1 (bits saved on the
    text's reader questions); the best is kept only when it saves more than the empty explanation (0 bits). On the kept
    ones, the controls: another text's kept explanation read on this text (shuffled) and reader B (an independent
    model); reader_b_all, the summary's headline, is reader B's bits over every text with 0 where nothing was kept
    (what the SFT examples hold). {"task", "samples", "e1", "explanation" (kept, or ""), "e1_best", "shuffled",
    "reader_b"} per text."""
    import tinker
    from tinker_cookbook import renderers

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    a.out.parent.mkdir(parents=True, exist_ok=True)
    service = service_client()
    wr, _ = renderer_for(a.writer)
    writer = service.create_sampling_client(base_model=a.writer)
    spend_w = Spend(a.out.parent, a.writer)
    ra, ta = renderer_for(a.reader_a)
    reader_a = tinker_reader(service, a.reader_a, ra, ta, Spend(a.out.parent, a.reader_a))
    reader_b = tinker_reader(service, a.reader_b, *renderer_for(a.reader_b), Spend(a.out.parent, a.reader_b)) if a.reader_b else None
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines())}
    facts = [json.loads(x) for x in a.facts.read_text().splitlines() if x.strip()]
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=wr.get_stop_sequences())
    futs = []
    for f in facts:  # unseeded: a seed gives every one of a call's num_samples the same draws (16 samples, 1-7 distinct)
        prompt = wr.build_generation_prompt([{"role": "user", "content": components.writer_prompt(f)}])
        futs.append((f, prompt, writer.sample(prompt=prompt, num_samples=a.n, sampling_params=params)))
    out = []
    for f, prompt, fut in futs:
        res = fut.result()
        hits = res.prompt_cache_hit_tokens
        spend_w.add("writer", prefill=prompt.length - hits, cached=hits + (a.n - 1) * prompt.length, sampled=sum(len(x.tokens) for x in res.sequences))
        texts = []
        for x in res.sequences:
            t = renderers.get_text_content(wr.parse_response(x.tokens)[0]).strip()
            t = t[len("EXPLANATION:"):].strip() if t.upper().startswith("EXPLANATION:") else t
            texts.append(" ".join(t.split()))
        task = json.loads((a.texts_dir / f"{f['task']}.json").read_text())
        named = [bool(NAMES.search(t)) for t in texts]  # the writer was told never to name subcomponents
        texts = [t for t, n in zip(texts, named) if not n]
        if not texts:
            out.append({"task": f["task"], "samples": [], "e1": [], "best": None, "explanation": "", "e1_best": None,
                        "e1_mean": None, "named": sum(named)})
            print(json.dumps({"task": f["task"], "kept": False, "named": sum(named)}), flush=True)
            continue
        e1 = components.e1_bits(reader_a, task, rows[f["task"]], [f"EXPLANATION: {t}" for t in texts])
        best = max(range(len(texts)), key=lambda i: e1[i] if e1[i] is not None else -1e9)
        keep = e1[best] is not None and e1[best] > 0
        out.append({"task": f["task"], "named": sum(named), "samples": texts, "e1": e1, "best": best, "explanation": texts[best] if keep else "",
                    "e1_best": e1[best], "e1_mean": sum(e for e in e1 if e is not None) / max(1, sum(e is not None for e in e1))})
        print(json.dumps({k: out[-1][k] for k in ("task", "e1_best", "e1_mean", "named")} | {"kept": keep}), flush=True)
    kept = [o for o in out if o["explanation"]]
    for i, o in enumerate(kept):
        task = json.loads((a.texts_dir / f"{o['task']}.json").read_text())
        if len(kept) > 1:
            other = kept[(i + 1) % len(kept)]["explanation"]
            o["shuffled"] = components.e1_bits(reader_a, task, rows[o["task"]], [f"EXPLANATION: {other}"])[0]
        if reader_b is not None:
            o["reader_b"] = components.e1_bits(reader_b, task, rows[o["task"]], [f"EXPLANATION: {o['explanation']}"])[0]
    with a.out.open("w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    total = sum(json.loads(x)["usd"] for x in (a.out.parent / "spend.jsonl").read_text().splitlines())
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    print(json.dumps({"texts": len(out), "kept": len(kept),
                      "reader_b_all": mean([o.get("reader_b") or 0.0 for o in out]) if reader_b is not None else None,
                      "e1_best": mean([o["e1_best"] for o in out if o["e1_best"] is not None]),
                      "e1_mean": mean([o["e1_mean"] for o in out if o["e1_mean"] is not None]), "named": sum(o["named"] for o in out), "shuffled": mean([o["shuffled"] for o in kept if o.get("shuffled") is not None]),
                      "reader_b": mean([o["reader_b"] for o in kept if o.get("reader_b") is not None]), "spend_usd": total}), flush=True)


def clients(base: str, rank: int, init: str | None):
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = service_client()
    training = (service.create_training_client_from_state(init) if init
                else service.create_lora_training_client(base_model=base, rank=rank))
    tokenizer = get_tokenizer(base)
    return tinker, service, training, renderers.get_renderer(RENDERER, tokenizer), tokenizer


def sft(a) -> None:
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.log, a.base)
    tinker, _, training, renderer, _ = clients(a.base, a.rank, None)
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    data = [conversation_to_datum(r["prompt"] + r["completion"], renderer, a.max_length, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            for r in rows]
    rng = random.Random(a.seed)
    if a.lr is None:
        from tinker_cookbook.hyperparam_utils import get_lr

        a.lr = get_lr(a.base)  # the cookbook's LoRA rate for the base (4.7e-4 for Qwen3-8B)
    for step in range(a.steps):
        t0 = time.time()
        batch = rng.sample(data, min(a.batch, len(data)))
        lr = a.lr * (1 - step / a.steps)
        fb = training.forward_backward(batch, loss_fn="cross_entropy")
        op = training.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
        out = fb.result()
        op.result()
        trained = sum(d.model_input.length for d in batch)
        usd = spend.add("sft", trained=trained)
        nll = compute_mean_nll([o["logprobs"] for o in out.loss_fn_outputs], [d.loss_fn_inputs["weights"] for d in batch])
        print(json.dumps({"step": step, "nll": nll, "tokens": trained, "usd": usd, "total_usd": spend.total,
                          "seconds": time.time() - t0}), flush=True)
    saved = training.save_state(a.name).result()
    (a.log / "state.json").write_text(json.dumps({"state": saved.path, "base": a.base, "rank": a.rank}))
    print(json.dumps({"saved": saved.path, "total_usd": spend.total}), flush=True)


def rl(a) -> None:
    import torch
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import renderers

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.log, a.base)
    tinker, service, training, renderer, tokenizer = clients(a.base, a.rank, a.init)
    questions = [json.loads(x) for x in a.questions.read_text().splitlines() if x.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    frozen = tinker_reader(service, a.base, renderer, tokenizer, spend) if a.reader else None
    tasks = {}
    rng = random.Random(a.seed)
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    adam = tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)

    def submit(sampler):
        batch = []
        for q in rng.sample(questions, min(a.texts, len(questions))):
            prompt = renderer.build_generation_prompt([{"role": "user", "content": q["question"]}])
            batch.append((q, prompt, sampler.sample(prompt=prompt, num_samples=a.group, sampling_params=params)))
        return batch

    def checkpoint(step: int) -> None:
        path = training.save_state(f"{a.name}-rl-{step}").result().path
        with (a.log / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "state": path, "usd": spend.total, "base": a.base}) + "\n")

    checkpoint(0)
    sampler = training.save_weights_and_get_sampling_client()
    pending = submit(sampler)
    for step in range(a.steps):
        t0 = time.time()
        datums, groups = [], []
        for q, prompt, fut in pending:
            res = fut.result()
            n = prompt.length
            hits = res.prompt_cache_hit_tokens
            sampled = sum(len(s.tokens) for s in res.sequences)
            spend.add("sample", prefill=n - hits, cached=hits + (a.group - 1) * n, sampled=sampled)
            texts = [renderers.get_text_content(renderer.parse_response(s.tokens)[0]) for s in res.sequences]
            t1 = time.time()
            req = {"task": q["task"], "answers": texts, "truncated": [str(s.stop_reason) == "length" for s in res.sequences]}
            if frozen is not None:
                if q["task"] not in tasks:
                    tasks[q["task"]] = json.loads((a.texts_dir / f"{q['task']}.json").read_text())
                req["e1"] = components.e1_bits(frozen, tasks[q["task"]], q, texts)
            r = httpx.post(a.scorer, json=req, timeout=None)
            r.raise_for_status()
            scores = r.json()["scores"]
            rewards = [s["reward"] for s in scores]
            mean = sum(rewards) / len(rewards)
            distinct = len({tuple(s.tokens) for s in res.sequences})
            if step == 0 and distinct == 1 and a.group > 1:  # a seeded or degenerate sampler: no advantage, nothing learned
                raise SystemExit(f"all {a.group} answers to {q['task']} are identical: check the sampling parameters")
            groups.append({"task": q["task"], "distinct": distinct, "rewards": rewards, "valid": [s["valid"] for s in scores],
                           "defects": [{k: s.get(k) for k in ("malformed", "dropped_nodes", "dropped_edges", "finals", "truncated")} for s in scores],
                           "texts": texts, "stop": [str(s.stop_reason) for s in res.sequences],
                           "tokens": [len(s.tokens) for s in res.sequences], "score_seconds": time.time() - t1,
                           "terms": {k: [s.get(k) for s in scores] for k in ("F1", "N", "F3", "E1", "E2", "size")}})
            for s, rew in zip(res.sequences, rewards):
                if rew == mean:
                    continue
                ob = n - 1
                inp = prompt.append(tinker.types.EncodedTextChunk(tokens=s.tokens[:-1]))
                datums.append(tinker.types.Datum(model_input=inp, loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor([0] * ob + list(s.tokens))),
                    "logprobs": TensorData.from_torch(torch.tensor([0.0] * ob + list(s.logprobs))),
                    "advantages": TensorData.from_torch(torch.tensor([0.0] * ob + [rew - mean] * len(s.tokens)))}))
        with (a.log / "groups.jsonl").open("a") as f:
            for g in groups:
                f.write(json.dumps({"step": step, **g}) + "\n")
        fb = op = None
        if datums:
            fb = training.forward_backward(datums, loss_fn="cispo")
            op = training.optim_step(adam)
        if step + 1 < a.steps:
            pending = submit(sampler)  # sampled while this update runs: one step behind
        if fb is not None:
            fb.result()
            op.result()
            spend.add("rl", trained=sum(d.model_input.length for d in datums))
            sampler = training.save_weights_and_get_sampling_client()
        flat = [x for g in groups for x in g["rewards"]]
        line = json.dumps({"step": step, "reward": sum(flat) / len(flat), "valid": sum(v for g in groups for v in g["valid"]) / len(flat),
                           "distinct": [g["distinct"] for g in groups],
                           "datums": len(datums), "tokens": [t for g in groups for t in g["tokens"]],
                           "score_seconds": sum(g["score_seconds"] for g in groups), "total_usd": spend.total,
                           "seconds": time.time() - t0})
        print(line, flush=True)
        with (a.log / "steps.jsonl").open("a") as f:
            f.write(line + "\n")
        if (step + 1) % a.save_every == 0 and step + 1 < a.steps:
            checkpoint(step + 1)
    checkpoint(a.steps)
    last = json.loads((a.log / "checkpoints.jsonl").read_text().splitlines()[-1])
    (a.log / "state.json").write_text(json.dumps({"state": last["state"], "base": a.base, "rank": a.rank}))
    print(json.dumps({"saved": last["state"], "total_usd": spend.total}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sft", "rl"):
        p = sub.add_parser(name)
        p.add_argument("questions" if name == "rl" else "rows", type=Path)
        p.add_argument("--log", type=Path, required=True)
        p.add_argument("--base", default="Qwen/Qwen3-8B")
        p.add_argument("--rank", type=int, default=32)
        p.add_argument("--steps", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--name", default="components")
    s = sub.choices["sft"]
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, help="peak rate, linear decay to 0 (default: the cookbook's LoRA rate for the base)")
    s.add_argument("--max-length", type=int, default=16384)
    r = sub.choices["rl"]
    r.add_argument("--scorer", required=True)
    r.add_argument("--init", help="a tinker:// state path (sft's DIR/state.json)")
    r.add_argument("--texts", type=int, default=2)
    r.add_argument("--group", type=int, default=4)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--lr", type=float, default=4e-5)
    r.add_argument("--reader", action="store_true", help="E1 from a frozen base-model sampling client on Tinker")
    r.add_argument("--save-every", type=int, default=25, help="steps between saved states (checkpoints.jsonl)")
    r.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    m = sub.add_parser("sample")
    m.add_argument("questions", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--init", help="a tinker:// state path; without, the base model alone")
    m.add_argument("--base", default="Qwen/Qwen3-8B")
    m.add_argument("--n", type=int, default=50, help="questions")
    m.add_argument("--samples", type=int, default=4)
    m.add_argument("--max-tokens", type=int, default=4096)
    e = sub.add_parser("readers")
    e.add_argument("samples", type=Path)
    e.add_argument("questions", type=Path)
    e.add_argument("out", type=Path)
    e.add_argument("--reader-a", default="Qwen/Qwen3-8B", help="the reward's reader")
    e.add_argument("--reader-b", help="an independent reader for the decisive check (a model in PRICES)")
    e.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    x = sub.add_parser("explain")
    x.add_argument("facts", type=Path, help="components.py facts output")
    x.add_argument("rows", type=Path, help="the texts' question rows (reader events)")
    x.add_argument("out", type=Path)
    x.add_argument("--n", type=int, default=8, help="explanations sampled per text")
    x.add_argument("--writer", default="Qwen/Qwen3-8B")
    x.add_argument("--reader-a", default="Qwen/Qwen3-8B")
    x.add_argument("--reader-b", help="an independent reader for the control (a model in PRICES)")
    x.add_argument("--max-tokens", type=int, default=400)
    x.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    a = ap.parse_args()
    {"sft": sft, "rl": rl, "sample": sample, "readers": readers, "explain": explain}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
