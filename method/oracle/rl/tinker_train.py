"""The component explainer trained on Tinker (#2951 graph oracle): the policy, a LoRA on a Tinker base model, samples
and trains on Tinker's servers; the reward is computed on this machine by rl/serve_components.py (goal_eval on the
local GPU) over HTTP. Runs in ~/mpd-data/tinker-venv. The project comes from TINKER_PROJECT_ID (required: there is no
default) and the key from TINKER_API_KEY (read from the Keychain at launch, never written anywhere).

  tinker_train.py sft SFT.jsonl --log DIR [--base Qwen/Qwen3-8B] [--steps N] [--batch B] [--lr LR] [--rank 32]
  tinker_train.py rl QUESTIONS.jsonl --log DIR --scorer http://127.0.0.1:8767/score [--init STATE_PATH] [--steps N]
                     [--texts P] [--group G] [--max-tokens 4096] [--lr 4e-5] [--reader]

SFT.jsonl: components.py dataset rows; QUESTIONS.jsonl: components.py questions rows. sft trains on the assistant
message only (Qwen3's template with thinking disabled) and saves its state (DIR/state.json holds the tinker:// path).
rl samples G answers per question with one call (the prompt's prefill shared), scores them with the scorer (F2's ranks
drawn per question from 0 to 7, as the prime-rl taskset does), centres the rewards within each group, and trains with
Tinker's "cispo" loss on the sampled tokens' log-probabilities. The next batch is sampled while the update runs (one
step behind; cispo's ratio corrects it). With --reader, E1 (the bits each answer's English saves a frozen reader)
comes from a base-model sampling client on Tinker that is never trained (tinker_reader) and goes to the scorer with
the answers. Every call's tokens are counted and priced (PRICES, $ per million tokens:
prefill, sampled, trained; a cached prefill at 20%) into DIR/spend.jsonl, one line per call, and summed per step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

PRICES = {"Qwen/Qwen3-8B": (0.195, 0.60, 0.44)}  # $ per million tokens: prefill, sample, train (Tinker's rate card)
CACHED = 0.2  # a cached prefill token's share of the price
RANKS = 8
RENDERER = "qwen3_disable_thinking"


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
    from prompt import tokens_of

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

    return reader.Reader(generate, lambda xs: tokens_of("vpd4l", xs))


def clients(base: str, rank: int, init: str | None):
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    project = os.environ.get("TINKER_PROJECT_ID")
    if not project:
        raise SystemExit("TINKER_PROJECT_ID is not set (the project is never the default one)")
    service = tinker.ServiceClient(project_id=project)
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
            ranks = sorted(rng.sample(range(RANKS), 3))
            batch.append((q, ranks, prompt, sampler.sample(prompt=prompt, num_samples=a.group, sampling_params=params)))
        return batch

    sampler = training.save_weights_and_get_sampling_client()
    pending = submit(sampler)
    for step in range(a.steps):
        t0 = time.time()
        datums, groups = [], []
        for q, ranks, prompt, fut in pending:
            res = fut.result()
            n = prompt.length
            hits = res.prompt_cache_hit_tokens
            sampled = sum(len(s.tokens) for s in res.sequences)
            spend.add("sample", prefill=n - hits, cached=hits + (a.group - 1) * n, sampled=sampled)
            texts = [renderers.get_text_content(renderer.parse_response(s.tokens)[0]) for s in res.sequences]
            t1 = time.time()
            req = {"task": q["task"], "answers": texts, "ranks": ranks, "truncated": [str(s.stop_reason) == "length" for s in res.sequences]}
            if frozen is not None:
                if q["task"] not in tasks:
                    tasks[q["task"]] = json.loads((a.texts_dir / f"{q['task']}.json").read_text())
                req["e1"] = components.e1_bits(frozen, tasks[q["task"]], q, texts)
            r = httpx.post(a.scorer, json=req, timeout=None)
            r.raise_for_status()
            scores = r.json()["scores"]
            rewards = [s["reward"] for s in scores]
            mean = sum(rewards) / len(rewards)
            groups.append({"task": q["task"], "ranks": ranks, "rewards": rewards, "valid": [s["valid"] for s in scores],
                           "defects": [{k: s.get(k) for k in ("malformed", "dropped_nodes", "dropped_edges", "finals", "truncated")} for s in scores],
                           "texts": texts, "stop": [str(s.stop_reason) for s in res.sequences],
                           "tokens": [len(s.tokens) for s in res.sequences], "score_seconds": time.time() - t1,
                           "terms": {k: [s.get(k) for s in scores] for k in ("F1", "F3", "E1", "E2", "N", "size", "f2_share")}})
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
        print(json.dumps({"step": step, "reward": sum(flat) / len(flat), "valid": sum(v for g in groups for v in g["valid"]) / len(flat),
                          "datums": len(datums), "tokens": [t for g in groups for t in g["tokens"]],
                          "score_seconds": sum(g["score_seconds"] for g in groups), "total_usd": spend.total,
                          "seconds": time.time() - t0}), flush=True)
    saved = training.save_state(f"{a.name}-rl").result()
    (a.log / "state.json").write_text(json.dumps({"state": saved.path, "base": a.base, "rank": a.rank}))
    print(json.dumps({"saved": saved.path, "total_usd": spend.total}), flush=True)


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
    r.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    a = ap.parse_args()
    sft(a) if a.cmd == "sft" else rl(a)


if __name__ == "__main__":
    sys.exit(main())
