"""Supervised fine-tuning and sampling on Tinker: the policy, a LoRA on a Tinker base model, trains and samples on
Tinker's servers (the text reader's SFT, experiments/oracle/cao/text_reader.py rows). Runs in ~/mpd-data/tinker-venv.
The project comes from TINKER_PROJECT_ID (required: there is no default) and the key from TINKER_API_KEY (read from
the Keychain at launch, never written anywhere).

  tinker_train.py sft SFT.jsonl --log DIR [--base Qwen/Qwen3-8B] [--steps N] [--batch B] [--lr LR] [--rank 32]
                      [--init STATE] [--save-every K]
  tinker_train.py sample QUESTIONS.jsonl OUT.jsonl [--init STATE_PATH] [--n 50] [--samples 4]    answers to evaluate
  tinker_train.py score SFT.jsonl OUT.jsonl [--init STATE_PATH]   each completion's held-out log-probability (no update)

SFT.jsonl rows: {"prompt": [messages], "completion": [messages], "info": {...}}. sft trains on the last assistant
message only (Qwen3's template with thinking disabled) and saves its state (DIR/state.json holds the tinker:// path;
--save-every adds DIR/checkpoints.jsonl). Every call's tokens are counted and priced (llm.PRICES, $ per million
tokens: prefill, sampled, trained; a cached prefill at 20%) into DIR/spend.jsonl, one line per call.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # method/oracle: llm
from llm import Spend, answer_of, renderer_for, service_client  # noqa: E402

RENDERER = "qwen3_disable_thinking"
THINKING_RENDERER = "qwen3"  # sample --thinking: Qwen3's own reasoning before the answer (sft keeps the empty block in
# the prompt, RENDERER's way, so the targets never train thinking away: the lead's 10-12)
CONTEXT = 32768  # Qwen3-8B's native context: a question's generation limit is at most what it leaves


def fitting(tinker, params, prompt, limit: int):
    """Sampling parameters whose generation limit fits the prompt in CONTEXT: min(limit, CONTEXT - its length)."""
    room = CONTEXT - prompt.length - 16
    return params if room >= limit else tinker.SamplingParams(max_tokens=max(256, room), temperature=params.temperature, stop=params.stop)


def sample(a) -> None:
    """N answers per question (its "task", or an environment's "id") from a trained state (--init) or the base model alone, one sampling call per question:
    {"task", "sample", "text", "finish", "tokens", "seconds"} per answer, seconds from its call's submission to its result
    (every question's call in flight at once, so a text's wait in the shared queue counts; the cost columns' oracle time). --prefix: the answer starts with this text (its tokens
    after the chat prompt; the written text holds it), e.g. "CIRCUIT:\nANSWER 1:" for answers without a circuit."""
    import tinker
    from tinker_cookbook import renderers

    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.out.parent, a.base)
    service = service_client()
    renderer, tokenizer = renderer_for(a.base, a.thinking)
    sampler = (service.create_training_client_from_state(a.init).save_weights_and_get_sampling_client() if a.init
               else service.create_sampling_client(base_model=a.base))
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    questions = [json.loads(x) for x in a.questions.read_text().splitlines() if x.strip()][:a.n]
    head = tokenizer.encode(("<think>\n\n</think>\n\n" if a.thinking else "") + a.prefix, add_special_tokens=False) if a.prefix else []  # thinking: the prefix after an empty think block
    futs = [(q, renderer.build_generation_prompt([{"role": "user", "content": q["question"]}])) for q in questions]
    if head:
        futs = [(q, prompt.append(tinker.types.EncodedTextChunk(tokens=head))) for q, prompt in futs]
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    futs = [(q, prompt, sampler.sample(prompt=prompt, num_samples=a.samples, sampling_params=fitting(tinker, params, prompt, a.max_tokens)),
             time.perf_counter()) for q, prompt in futs]

    def done(fut, sent):  # each call's own result time, not the order they are read in
        res = fut.result()
        return res, time.perf_counter() - sent

    with ThreadPoolExecutor(max(len(futs), 1)) as pool:
        waits = [pool.submit(done, fut, sent) for _, _, fut, sent in futs]
    with a.out.open("w") as f:
        for (q, prompt, _, _), w in zip(futs, waits):
            res, secs = w.result()
            hits = res.prompt_cache_hit_tokens
            spend.add("sample", prefill=prompt.length - hits, cached=hits + (a.samples - 1) * prompt.length,
                      sampled=sum(len(s.tokens) for s in res.sequences))
            for k, s in enumerate(res.sequences):
                text, think = answer_of(renderer, tokenizer, s.tokens)
                f.write(json.dumps({"task": q.get("task") or q["id"], "sample": k, "text": a.prefix + text, "thinking_tokens": think,
                                    "finish": str(s.stop_reason), "tokens": len(s.tokens), "seconds": round(secs, 2)}) + "\n")
    print(json.dumps({"questions": len(questions), "samples": a.samples, "total_usd": spend.total,
                      "wall_seconds": round(time.perf_counter() - t0, 1)}), flush=True)


def clients(base: str, rank: int, init: str | None, thinking: bool = False, optimizer: bool = False):
    """optimizer: the state's Adam moments too (a resumed rl run), else its weights alone."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = service_client()
    training = (service.create_training_client_from_state_with_optimizer(init) if init and optimizer
                else service.create_training_client_from_state(init) if init
                else service.create_lora_training_client(base_model=base, rank=rank))
    tokenizer = get_tokenizer(base)
    # Qwen3's renderers by name; any other base (Qwen3.5/3.6: qwen3_5) its recommended one, as sampling renders it
    renderer = (renderers.get_renderer(THINKING_RENDERER if thinking else RENDERER, tokenizer) if base.startswith("Qwen/Qwen3-")
                else renderer_for(base, thinking)[0])
    return tinker, service, training, renderer, tokenizer


def sft(a) -> None:
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.log, a.base)
    # the empty think block in the prompt, never trained; --thinking: targets that open with their thinking, trained in
    # Qwen3's thinking renderer
    tinker, _, training, renderer, _ = clients(a.base, a.rank, a.init, a.thinking)  # --init: a warm start's weights
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    what = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES if a.train_on == "all" else renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
    whole = [conversation_to_datum(r["prompt"] + r["completion"], renderer, None, what) for r in rows]
    data = [d for d in whole if d.model_input.length <= a.max_length]  # a cut example would teach an unfinished answer
    long = [r["info"]["task"] for r, d in zip(rows, whole) if d.model_input.length > a.max_length]
    print(json.dumps({"examples": len(data), "dropped over max_length": long}), flush=True)
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
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:  # the learning curve's checkpoints
            path = training.save_state(f"{a.name}-{step + 1}").result().path
            with (a.log / "checkpoints.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, "state": path, "usd": spend.total, "base": a.base}) + "\n")
    saved = training.save_state(a.name).result()
    (a.log / "state.json").write_text(json.dumps({"state": saved.path, "base": a.base, "rank": a.rank}))
    if a.save_every:
        with (a.log / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps({"step": a.steps, "state": saved.path, "usd": spend.total, "base": a.base}) + "\n")
    print(json.dumps({"saved": saved.path, "total_usd": spend.total}), flush=True)


def score(a) -> None:
    """Held-out log-probabilities of SFT rows' completions under a state (--init) or the base alone, a forward pass
    each (no update): {"index", "info", "tokens", "logprob", "mean"} per row, the completion's summed and mean log-prob
    (nats) over its trained tokens (the rows' format and template as sft trains them)."""
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.out.parent, a.base)
    tinker, _, training, renderer, _ = clients(a.base, a.rank, a.init)  # sft's rendering
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    with a.out.open("w") as f:
        for s0 in range(0, len(rows), a.batch):
            part = rows[s0:s0 + a.batch]
            data = [conversation_to_datum(r["prompt"] + r["completion"], renderer, a.max_length, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                    for r in part]
            out = training.forward(data, loss_fn="cross_entropy").result()
            spend.add("score", prefill=sum(d.model_input.length for d in data))
            for k, (r, d, o) in enumerate(zip(part, data, out.loss_fn_outputs)):
                m = d.loss_fn_inputs["weights"].to_torch() > 0  # the trained tokens (the weights are normalized to sum 1)
                lp = o["logprobs"].to_torch()[m].sum().item()
                f.write(json.dumps({"index": s0 + k, "info": r.get("info"), "tokens": int(m.sum().item()), "logprob": lp,
                                    "mean": lp / max(int(m.sum().item()), 1)}) + "\n")
    print(json.dumps({"rows": len(rows), "total_usd": spend.total}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft")
    s.add_argument("rows", type=Path)
    s.add_argument("--log", type=Path, required=True)
    s.add_argument("--base", default="Qwen/Qwen3-8B")
    s.add_argument("--rank", type=int, default=32)
    s.add_argument("--steps", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--name", default="sft")
    s.add_argument("--thinking", action="store_true", help="targets with their thinking, in Qwen3's thinking renderer "
                                                            "(without: the empty block in the prompt)")
    s.add_argument("--init", help="a tinker:// state to continue from; without, the base model")
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, help="peak rate, linear decay to 0 (default: the cookbook's LoRA rate for the base)")
    s.add_argument("--max-length", type=int, default=16384)
    s.add_argument("--train-on", choices=("last", "all"), default="last", help="all: every assistant message (an agent's "
                   "trajectory: its tool calls and its answer), else the last one")
    s.add_argument("--save-every", type=int, default=0, help="a checkpoint every k steps and the last, in DIR/checkpoints.jsonl")
    c = sub.add_parser("score", help="held-out log-probabilities of SFT rows' completions (no update)")
    c.add_argument("rows", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--init", help="a tinker:// state path; without, the base model alone")
    c.add_argument("--base", default="Qwen/Qwen3-8B")
    c.add_argument("--rank", type=int, default=32)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--max-length", type=int, default=32768)
    m = sub.add_parser("sample")
    m.add_argument("questions", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--init", help="a tinker:// state path; without, the base model alone")
    m.add_argument("--base", default="Qwen/Qwen3-8B")
    m.add_argument("--n", type=int, default=50, help="questions")
    m.add_argument("--samples", type=int, default=4)
    m.add_argument("--max-tokens", type=int, default=4096)
    m.add_argument("--thinking", action="store_true", help="sample in Qwen3's thinking mode (the answer is the text after it)")
    m.add_argument("--prefix", default="", help="the answer's opening text, given")
    a = ap.parse_args()
    {"sft": sft, "sample": sample, "score": score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
