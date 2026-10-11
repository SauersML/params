"""An explainer of unit circuits (#2951 graph oracle): LoRA SFT of a base model on measured circuits (units.dataset rows
from carry.py explain), its answers to held-out questions sampled with vLLM, and every answer scored by the carry test
(units.score) beside the copy and all baselines on the same questions. Each stage is its own process (the GPU freed
between them).

  units_sft.py sft ROWS.jsonl OUT_DIR [--val ROWS.jsonl] [--base Qwen/Qwen3-4B] [--epochs 3] [--lr 1e-4] [--rank 32] [--batch 8]
  units_sft.py questions EXPLAIN.jsonl OUT.jsonl [--k 8]     each task's question, positions from units.sensitive
  units_sft.py sample QUESTIONS.jsonl ADAPTER OUT.jsonl [--n 4] [--max-tokens 1536]
  units_sft.py score QUESTIONS.jsonl OUT.jsonl [--answers ANSWERS.jsonl] [--teacher DIR] [--k 8]   (without answers: the baselines only;
                                                  DIR: units.py targets' answers, the measured circuits, scored as "teacher")
  units_sft.py report SCORES.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))


def chat_prompt(tok, question: str) -> str:
    """rl/train.py's chat: the user's question, the assistant's turn opened without thinking."""
    return tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, enable_thinking=False, tokenize=False)


def encode_rows(tok, path: Path) -> list[tuple[list[int], list[int]]]:
    """(prompt ids, completion ids ending in the end-of-turn token) per row of a units.dataset file."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for line in open(path):
        r = json.loads(line)
        p = tok.encode(chat_prompt(tok, r["prompt"][0]["content"]), add_special_tokens=False)
        c = tok.encode(r["completion"][0]["content"], add_special_tokens=False) + [end]
        out.append((p, c))
    return out


def sft(a) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = encode_rows(tok, a.rows)
    val = encode_rows(tok, a.val) if a.val else []
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(rows) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "sft.jsonl", "w")

    def nll(p, c, train: bool):
        ids = torch.tensor([p + c], device="cuda")
        labels = torch.tensor([[-100] * len(p) + c], device="cuda")
        with torch.set_grad_enabled(train):
            logits = model(input_ids=ids).logits[:, :-1].float()
            return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="sum", ignore_index=-100)

    def validate() -> float:
        model.eval()
        total, n = 0.0, 0
        for p, c in val:
            total += nll(p, c, False).item()
            n += len(c)
        model.train()
        return total / n

    print(f"sft: {len(rows)} rows, {steps} steps of {a.batch}; completion tokens mean {statistics.mean(len(c) for _, c in rows):.0f}, "
          f"max {max(len(c) for _, c in rows)}; prompt tokens mean {statistics.mean(len(p) for p, _ in rows):.0f}", flush=True)
    if val:
        v = validate()
        log.write(json.dumps({"step": 0, "val_nats_per_token": v}) + "\n")
        print(f"step 0: held-out loss {v:.4f} nats/token", flush=True)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        order = list(range(len(rows)))
        random.shuffle(order)
        for b in range(0, len(order), a.batch):
            batch = [rows[i] for i in order[b:b + a.batch]]
            tokens = sum(len(c) for _, c in batch)
            loss_sum = 0.0
            for p, c in batch:
                loss = nll(p, c, True)
                (loss / tokens).backward()
                loss_sum += loss.item()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch, "nats_per_token": loss_sum / tokens, "grad_norm": grad, "lr": sched.get_last_lr()[0],
                   "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            if val and (b + a.batch >= len(order)):  # the held-out loss at each epoch's end
                rec["val_nats_per_token"] = validate()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    model.save_pretrained(str(a.out / "adapter"))
    tok.save_pretrained(str(a.out / "adapter"))


def questions(a) -> None:
    import native
    import units

    nat = native.Native()
    with open(a.out, "w") as f:
        for line in open(a.explain):
            r = json.loads(line)
            task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
            p = task["prompts"][0]
            positions = units.sensitive(nat, p["token_ids"], p["target_positions"], a.k)
            f.write(json.dumps({"task": r["task"], "positions": positions, "explain_positions": [x["position"] for x in r["positions"]],
                                "question": units.question(task, positions)}) + "\n")


def sample(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    qs = [json.loads(line) for line in open(a.questions)]
    tok = AutoTokenizer.from_pretrained(a.base)
    rank = json.loads((a.adapter / "adapter_config.json").read_text())["r"]
    llm = LLM(model=a.base, dtype="bfloat16", enable_lora=True, max_lora_rank=rank, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.85, seed=a.seed)
    params = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate([chat_prompt(tok, q["question"]) for q in qs], params, lora_request=LoRARequest("explainer", 1, str(a.adapter)))
    with open(a.out, "w") as f:
        for q, o in zip(qs, outs):
            for j, c in enumerate(o.outputs):
                f.write(json.dumps({"task": q["task"], "sample": j, "text": c.text, "tokens": len(c.token_ids), "finish": c.finish_reason}) + "\n")
    print(f"sample: {len(qs)} questions x {a.n} in {time.time() - t0:.0f} s", flush=True)


def score(a) -> None:
    import native
    import units

    nat = native.Native()
    qs = {q["task"]: q for q in (json.loads(line) for line in open(a.questions))}
    answers = [json.loads(line) for line in open(a.answers)] if a.answers else []
    t0 = time.time()
    with open(a.out, "w") as f:
        for tid, q in qs.items():
            task = json.loads((native.TEXTS / "vpd4l" / f"{tid}.json").read_text())
            targets = task["prompts"][0]["target_positions"]
            items = [(kind, None, units.baseline(kind, q["positions"], targets), None, None) for kind in ("copy", "all")]
            if a.teacher and (a.teacher / f"{tid}.txt").exists():
                items.append(("teacher", None, (a.teacher / f"{tid}.txt").read_text(), None, None))
            items += [("explainer", x["sample"], x["text"], x["tokens"], x["finish"]) for x in answers if x["task"] == tid]
            for kind, j, text, tokens, finish in items:
                _, circuits = units.parse(text)
                s = units.score(nat, task, text, a.k)
                f.write(json.dumps({"task": tid, "kind": kind, "sample": j, "tokens": tokens, "finish": finish,
                                    "lines": sum(p in circuits for p in q["positions"]), "positions": len(q["positions"]),
                                    "uncarried": s["uncarried"], "units": s["units"], "outside": sum(r["outside"] for r in s["positions"]),
                                    "unknown": sum(r.get("unknown", 0) for r in s["positions"]),
                                    "per_position": s["positions"]}) + "\n")
                f.flush()
            print(f"score: {tid} done, {time.time() - t0:.0f} s", flush=True)


def report(a) -> None:
    rows = [json.loads(line) for line in open(a.scores)]
    tasks = sorted({r["task"] for r in rows}, key=lambda t: int("".join(ch for ch in t if ch.isdigit()) or 0))
    print(f"{'task':10s} {'expl best':>14s} {'expl median':>14s} {'parse':>6s} {'teacher':>14s} {'copy':>14s} {'all':>14s}   (uncarried share / units)")
    agg = {k: [] for k in ("best", "median", "teacher", "copy", "all", "parse")}
    for t in tasks:
        ex = [r for r in rows if r["task"] == t and r["kind"] == "explainer"]
        base = {r["kind"]: r for r in rows if r["task"] == t and r["kind"] in ("teacher", "copy", "all")}
        cell = lambda u, n: f"{u:.3f} / {n:4.0f}"  # noqa: E731
        line = f"{t:10s}"
        if ex:
            best = min(ex, key=lambda r: (r["uncarried"], r["units"]))
            med_u, med_n = statistics.median(r["uncarried"] for r in ex), statistics.median(r["units"] for r in ex)
            parse = sum(r["finish"] == "stop" and r["lines"] == r["positions"] for r in ex) / len(ex)
            line += f" {cell(best['uncarried'], best['units']):>14s} {cell(med_u, med_n):>14s} {parse:6.2f}"
            agg["best"].append((best["uncarried"], best["units"]))
            agg["median"].append((med_u, med_n))
            agg["parse"].append(parse)
        for k in ("teacher", "copy", "all"):
            if k in base:
                line += f" {cell(base[k]['uncarried'], base[k]['units']):>14s}"
                agg[k].append((base[k]["uncarried"], base[k]["units"]))
            else:
                line += f" {'-':>14s}"
        print(line)
    print("mean over questions:")
    for k in ("best", "median", "teacher", "copy", "all"):
        if agg[k]:
            print(f"  {k:7s} uncarried {statistics.mean(u for u, _ in agg[k]):.3f}, units {statistics.mean(n for _, n in agg[k]):.1f}")
    if agg["parse"]:
        ex = [r for r in rows if r["kind"] == "explainer"]
        print(f"  answers that parse (ended, a line for every listed position): {statistics.mean(agg['parse']):.2f}; "
              f"hit the token cap {sum(r['finish'] == 'length' for r in ex) / len(ex):.2f}; mean tokens {statistics.mean(r['tokens'] for r in ex):.0f}; "
              f"units outside the position and targets {statistics.mean(r['outside'] for r in ex):.1f} and unknown units "
              f"{statistics.mean(r.get('unknown', 0) for r in ex):.1f} per answer")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft")
    s.add_argument("rows", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--val", type=Path)
    s.add_argument("--base", default="Qwen/Qwen3-4B")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--rank", type=int, default=32)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--warmup", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("questions")
    q.add_argument("explain", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--k", type=int, default=8)
    m = sub.add_parser("sample")
    m.add_argument("questions", type=Path)
    m.add_argument("adapter", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--base", default="Qwen/Qwen3-4B")
    m.add_argument("--n", type=int, default=4)
    m.add_argument("--max-tokens", type=int, default=1536)
    m.add_argument("--max-model-len", type=int, default=4096)
    m.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("questions", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--answers", type=Path)
    c.add_argument("--teacher", type=Path)
    c.add_argument("--k", type=int, default=8)
    r = sub.add_parser("report")
    r.add_argument("scores", type=Path)
    a = ap.parse_args()
    {"sft": sft, "questions": questions, "sample": sample, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
