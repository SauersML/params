"""The toy-loop oracle LM (TOYLOOP lane): a small instruction model, LoRA or full fine-tune, on CPU.

  python3 -B experiments/oracle/toyloop/oracle_lm.py bench            # tokens/s, forward+backward, seq 1024, 4 threads
  python3 -B experiments/oracle/toyloop/oracle_lm.py bench --models HuggingFaceTB/SmolLM2-135M-Instruct --lora 0

Default model: SmolLM2-135M-Instruct, full fine-tune (see BENCH below for the measured numbers that chose it).
"""
from __future__ import annotations

import argparse
import json
import time

import torch

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CANDIDATES = ["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(name: str = DEFAULT_MODEL, lora: int = 0, threads: int = 4, grad_ckpt: bool = False):
    """(model, tokenizer). lora: LoRA rank (0 = full fine-tune). float32 on CPU (bf16 matmuls are slower here)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=lora, lora_alpha=2 * lora, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    return model, tok


def chat_ids(tok, prompt: str, completion: str | None = None) -> tuple[list[int], int]:
    """(token ids of the chat-formatted prompt [+ completion + eos], the number of prompt tokens)."""
    msgs = [{"role": "user", "content": prompt}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    if completion is None:
        return pid, len(pid)
    cid = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return pid + cid, len(pid)


def bench(name: str, lora: int, seq: int = 1024, batch: int = 1, steps: int = 3, threads: int = 4) -> dict:
    model, _ = load(name, lora, threads)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    ids = torch.randint(100, 20000, (batch, seq))
    times = []
    for i in range(steps + 1):   # the first step warms up
        t0 = time.time()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i:
            times.append(time.time() - t0)
    s = sorted(times)[len(times) // 2]
    with torch.no_grad():   # sampling speed: one new token with a KV cache, after a 512-token prompt
        model.eval()
        out = model.generate(ids[:, :512], max_new_tokens=32, do_sample=True, min_new_tokens=32, pad_token_id=0)
        t0 = time.time()
        model.generate(ids[:, :512], max_new_tokens=64, do_sample=True, min_new_tokens=64, pad_token_id=0)
        gen = 64 / (time.time() - t0)
    del out
    return {"model": name, "lora": lora, "seq": seq, "batch": batch, "sec_per_step": round(s, 2),
            "train_tok_per_s": round(batch * seq / s, 1), "gen_tok_per_s": round(gen, 1),
            "trainable_M": round(sum(p.numel() for p in params) / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--models", default=",".join(CANDIDATES))
    b.add_argument("--lora", default="0,16", help="comma list of LoRA ranks (0 = full fine-tune)")
    b.add_argument("--seq", type=int, default=1024)
    b.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    for m in a.models.split(","):
        for r in a.lora.split(","):
            print(json.dumps(bench(m, int(r), a.seq, steps=a.steps)), flush=True)


if __name__ == "__main__":
    main()
