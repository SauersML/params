"""The learnability experiment's training and scoring (experiments/oracle/learnability.py's questions): LoRA SFT of a
base model on one arm's training set (the "vocab" arm first on vocab.jsonl), then held-out scoring.

Rows are {"prompt", "answer", "choices", "label", "meta"}; the model reads prompt + "<answer> " and is trained on the
answer and an end token. Scoring, per held-out row:
  remove, cut, swap: the KL bin's probability, each of the four "KL <bin>;" choices' log-probability after the prompt,
                     normalized over them: log loss in bits (and accuracy); falls: P(" falls=yes;" against " falls=no;")
                     after the true bin;
  rank:              the first candidate's probability over the letters (top-1 log loss and accuracy) and the greedy
                     answer's order against the measured one (Kendall's tau).

  learn_sft.py train ARM_DIR/train_N.jsonl OUT_DIR [--vocab VOCAB.jsonl] [--base Qwen/Qwen3-1.7B] [--epochs 2]
  learn_sft.py score ARM_DIR/heldout.jsonl OUT.jsonl [--adapter OUT_DIR/adapter] [--base ...]
  learn_sft.py report SCORES.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

ANSWER = "<answer> "


def _rows(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _encode(tok, rows: list[dict]) -> list[tuple[list[int], list[int]]]:
    return [(tok(r["prompt"] + ANSWER, add_special_tokens=False)["input_ids"],
             tok(r["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]) for r in rows]


def _model(base: str, adapter: str | None = None, rank: int = 0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    elif rank:
        from peft import LoraConfig, get_peft_model

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return tok, model


def _batches(enc: list, batch: int, rng: random.Random) -> list[list]:
    """Batches of rows of similar length (less padding), in random order."""
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]) + rng.random())
    out = [[enc[i] for i in order[b:b + batch]] for b in range(0, len(order), batch)]
    rng.shuffle(out)
    return out


def _loss(model, batch, pad: int):
    """Summed answer-token cross entropy (nats) and the answer token count of a padded batch."""
    import torch

    L = max(len(p) + len(c) for p, c in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (p, c) in enumerate(batch):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        labels[i, len(p):len(p) + len(c)] = torch.tensor(c)
        att[i, :len(p) + len(c)] = 1
    ids, labels, att = ids.cuda(), labels.cuda(), att.cuda()
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="sum", ignore_index=-100)
    return loss, int((labels[:, 1:] != -100).sum())


def _fit(model, tok, enc: list, epochs: int, lr: float, batch: int, log, tag: str, seed: int = 0) -> None:
    import torch

    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(enc) / batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    step, t0, run = 0, time.time(), []
    for _ in range(epochs):
        for b in _batches(enc, batch, rng):
            loss, n = _loss(model, b, pad)
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run.append(loss.item() / n)
            step += 1
            if step % 50 == 0 or step == steps:
                log.write(json.dumps({"stage": tag, "step": step, "steps": steps, "loss": statistics.mean(run), "s": time.time() - t0}) + "\n")
                log.flush()
                print(f"{tag} step {step}/{steps}: loss {statistics.mean(run):.4f} nats/token ({time.time() - t0:.0f} s)", flush=True)
                run = []


def train(a) -> None:
    import torch

    torch.manual_seed(0)
    tok, model = _model(a.base, rank=a.rank)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "train.jsonl", "a")
    if a.vocab:
        _fit(model, tok, _encode(tok, _rows(a.vocab)), a.vocab_epochs, a.lr, a.batch, log, "vocab")
    _fit(model, tok, _encode(tok, _rows(a.rows)), a.epochs, a.lr, a.batch, log, "task")
    model.save_pretrained(a.out / "adapter")


def _logp(model, tok, prompts: list[list[int]], conts: list[list[int]]) -> list[float]:
    """log p(cont | prompt) (nats) of each pair, batched."""
    import torch

    out = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for b in range(0, len(prompts), 8):
        P, C = prompts[b:b + 8], conts[b:b + 8]
        L = max(len(p) + len(c) for p, c in zip(P, C))
        ids = torch.full((len(P), L), pad, dtype=torch.long)
        att = torch.zeros((len(P), L), dtype=torch.long)
        for i, (p, c) in enumerate(zip(P, C)):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
            att[i, :len(p) + len(c)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
            for i, (p, c) in enumerate(zip(P, C)):
                lp = logits[i, len(p) - 1:len(p) - 1 + len(c)].float().log_softmax(-1)
                out.append(float(lp[torch.arange(len(c)), torch.tensor(c, device=lp.device)].sum()))
    return out


def score(a) -> None:
    import torch
    from scipy.stats import kendalltau

    tok, model = _model(a.base, adapter=a.adapter)
    model.eval()
    rows = _rows(a.rows)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    out = open(a.out, "w")
    eff = [r for r in rows if r["meta"]["kind"] != "rank"]
    P, C, who = [], [], []
    for i, r in enumerate(eff):
        p = enc(r["prompt"] + ANSWER)
        for ch in r["choices"]:
            P.append(p)
            C.append(enc(ch))
            who.append(("bin", i))
        head = enc(r["choices"][r["label"]])
        for f in (" falls=yes;", " falls=no;"):
            P.append(p + head)
            C.append(enc(f))
            who.append(("falls", i))
    lps = _logp(model, tok, P, C)
    got = {}
    for (what, i), lp in zip(who, lps):
        got.setdefault((what, i), []).append(lp)
    for i, r in enumerate(eff):
        b = torch.tensor(got[("bin", i)]).log_softmax(0)
        f = torch.tensor(got[("falls", i)]).log_softmax(0)
        yes = r["meta"]["falls"]
        out.write(json.dumps({"kind": r["meta"]["kind"], "label": r["label"], "logloss_bits": -float(b[r["label"]]) / math.log(2),
                              "pred": int(b.argmax()), "falls_logloss_bits": -float(f[0 if yes else 1]) / math.log(2),
                              "p_bins": b.exp().tolist()}) + "\n")
    letters = "abcdefgh"
    for r in (r for r in rows if r["meta"]["kind"] == "rank"):
        p = enc(r["prompt"] + ANSWER)
        lp = torch.tensor(_logp(model, tok, [p] * len(letters), [enc(x) for x in letters])).log_softmax(0)
        with torch.no_grad():
            g = model.generate(input_ids=torch.tensor([p]).cuda(), max_new_tokens=24, do_sample=False)
        text = tok.decode(g[0, len(p):], skip_special_tokens=True).split()
        order = [x for x in dict.fromkeys(text) if x in letters][:len(letters)]
        order += [x for x in letters if x not in order]
        kls = r["meta"]["kls"]
        rank_of = {x: k for k, x in enumerate(order)}
        tau = kendalltau([-rank_of[x] for x in letters], kls).statistic
        out.write(json.dumps({"kind": "rank", "label": r["label"], "logloss_bits": -float(lp[r["label"]]) / math.log(2),
                              "pred": int(lp.argmax()), "tau": float(tau), "answer": " ".join(order)}) + "\n")
    out.close()


def report(paths: list[str]) -> None:
    for path in paths:
        rs = _rows(path)
        print(path)
        for kind in ("remove", "cut", "swap", "rank"):
            k = [r for r in rs if r["kind"] == kind]
            if not k:
                continue
            line = (f"  {kind:6s} n {len(k):4d}  log loss {statistics.mean(r['logloss_bits'] for r in k):.3f} bits, "
                    f"accuracy {statistics.mean(r['pred'] == r['label'] for r in k):.3f}")
            if kind == "rank":
                line += f", Kendall's tau {statistics.mean(r['tau'] for r in k if r['tau'] == r['tau']):.3f}"
            else:
                line += f", falls {statistics.mean(r['falls_logloss_bits'] for r in k):.3f} bits; by bin " + " ".join(
                    f"{b}:{statistics.mean(r['logloss_bits'] for r in k if r['label'] == b):.2f}" for b in range(4) if any(r["label"] == b for r in k))
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("rows")
    t.add_argument("out", type=Path)
    t.add_argument("--vocab")
    t.add_argument("--vocab-epochs", type=int, default=2)
    t.add_argument("--base", default="Qwen/Qwen3-1.7B")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--rank", type=int, default=32)
    t.add_argument("--batch", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("rows")
    s.add_argument("out")
    s.add_argument("--adapter")
    s.add_argument("--base", default="Qwen/Qwen3-1.7B")
    r = sub.add_parser("report")
    r.add_argument("scores", nargs="+")
    a = ap.parse_args()
    {"train": train, "score": score, "report": lambda a: report(a.scores)}[a.cmd](a)


if __name__ == "__main__":
    main()
