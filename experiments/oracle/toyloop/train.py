"""SFT, then GRPO, of the toy-loop oracle on CPU (TOYLOOP lane).

Data: examples {"prompt", "completion", "meta"} built from toys (render.render + render.target_text; meta carries the
toy's name and truth). Sources: synthetic:N (render.synthetic), planted:DIR (toys/planted.py checkpoints with truth from
--truth JSON {name: truth}), zoo:MANIFEST (render.load_zoo).

  python3 -B experiments/oracle/toyloop/train.py sft --data zoo:$HOME/mpd-data/toy_zoo/manifest.jsonl --out ckpt/sft --steps 500
  python3 -B experiments/oracle/toyloop/train.py rl  --data zoo:... --init ckpt/sft --out ckpt/rl --steps 200 --k 4 --reward f1

RL: per prompt k samples; advantage = reward - group mean (divided by the group std with --std); loss = -mean over
samples of advantage x mean completion-token log-prob, plus beta x KL(policy || frozen SFT policy) (the k3 estimator per
token). One gradient step per batch of fresh samples (on-policy, so no ratio clipping). The reward is any
reward(meta, completion) -> float: --reward f1 | f1_edges | module:function.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit  # noqa: E402
import oracle_lm  # noqa: E402
import render  # noqa: E402


# ---------------------------------------------------------------- data

def examples(spec: str, truth_json: str | None = None) -> list[dict]:
    kind, _, arg = spec.partition(":")
    if kind == "synthetic":
        toys = [render.synthetic(s) for s in range(int(arg or 32))]
    elif kind == "planted":
        truth = json.loads(Path(truth_json).read_text()) if truth_json else {}
        toys = [render.from_planted(p, truth.get(p.stem)) for p in sorted(Path(arg).expanduser().glob("*.pt"))]
    elif kind == "zoo":
        toys = render.load_zoo(Path(arg).expanduser())
    else:
        raise ValueError(spec)
    return [{"prompt": render.render(t), "completion": render.target_text(t.truth) if t.truth else None,
             "meta": {"name": t.name, "truth": t.truth}} for t in toys]


def get_reward(spec: str):
    if spec in ("f1", "f1_edges"):
        return getattr(circuit, "reward_" + spec)
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


# ---------------------------------------------------------------- models

def tiny_model(tok_name: str = oracle_lm.DEFAULT_MODEL):
    """A randomly initialised 2-layer Llama with SmolLM2's tokenizer: for smoke tests only."""
    from transformers import LlamaConfig, LlamaForCausalLM
    tok = oracle_lm_tokenizer(tok_name)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    return LlamaForCausalLM(cfg), tok


def oracle_lm_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(a):
    torch.set_num_threads(a.threads)
    if a.init:   # a directory saved by save(): a full model, or a LoRA adapter on its base
        p = Path(a.init)
        if (p / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM
            m = AutoPeftModelForCausalLM.from_pretrained(p, is_trainable=True, torch_dtype=torch.float32)
        else:
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
        return m, oracle_lm_tokenizer(p)
    if a.model == "tiny":
        return tiny_model()
    return oracle_lm.load(a.model, a.lora, a.threads)


def save(model, tok, out: Path, log: list):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))


def batch_ids(tok, pairs: list[tuple[str, str]], max_len: int):
    """Left-padded ids [B, T], attention mask, and a mask over completion tokens (the scored positions, at the end)."""
    rows = []
    for p, c in pairs:
        ids, n = oracle_lm.chat_ids(tok, p, c)
        if len(ids) > max_len:                      # cut the prompt's start, never the completion
            cut = len(ids) - max_len
            ids, n = ids[cut:], max(1, n - cut)
        rows.append((ids, n))
    T = max(len(r) for r, _ in rows)
    ids = torch.full((len(rows), T), tok.pad_token_id)
    att, comp = torch.zeros(len(rows), T, dtype=torch.long), torch.zeros(len(rows), T)
    for b, (r, n) in enumerate(rows):
        ids[b, T - len(r):] = torch.tensor(r)
        att[b, T - len(r):] = 1
        comp[b, T - len(r) + n:] = 1
    return ids, att, comp


def token_logps(model, ids, att, comp):
    """(log p(token t | tokens < t), mask) over the last K positions, K the longest completion: logits are computed
    only there (a [B, T, vocab] float tensor of a 1k-token prompt is ~200 MB a row)."""
    K = int(comp.sum(1).max())
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K + 1).logits[:, :-1].float()
    tgt = ids[:, -K:]
    return logits.gather(2, tgt[..., None])[..., 0] - logits.logsumexp(-1), comp[:, -K:]


# ---------------------------------------------------------------- SFT

def sft(a):
    data = [e for e in examples(a.data, a.truth) if e["completion"]]
    model, tok = load_policy(a)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    rng, log, t0 = random.Random(a.seed), [], time.time()
    for step in range(a.steps):
        batch = rng.sample(data, min(a.batch, len(data)))
        ids, att, comp = batch_ids(tok, [(e["prompt"], e["completion"]) for e in batch], a.max_len)
        lp, m = token_logps(model, ids, att, comp)
        loss = -(lp * m).sum() / m.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.append({"step": step, "loss": round(loss.item(), 4), "sec": round(time.time() - t0, 1)})
        if step % a.log_every == 0 or step == a.steps - 1:
            print(json.dumps(log[-1]), flush=True)
    if a.eval:
        print(json.dumps({"eval": evaluate(model, tok, data[:a.eval], circuit.reward_f1, a)}), flush=True)
    save(model, tok, Path(a.out), log)
    return log


# ---------------------------------------------------------------- RL (GRPO)

@torch.no_grad()
def sample(model, tok, prompt: str, k: int, a, greedy: bool = False) -> list[str]:
    model.eval()
    ids, _ = oracle_lm.chat_ids(tok, prompt)
    ids = torch.tensor([ids[-a.max_len + a.max_new:]])
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new,
                         do_sample=not greedy, temperature=a.temp, top_p=1.0, num_return_sequences=1 if greedy else k,
                         pad_token_id=tok.pad_token_id)
    model.train()
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)


def evaluate(model, tok, data, reward, a) -> float:
    return sum(reward(e["meta"], sample(model, tok, e["prompt"], 1, a, greedy=True)[0]) for e in data) / max(len(data), 1)


def rl(a):
    data = examples(a.data, a.truth)
    reward = get_reward(a.reward)
    model, tok = load_policy(a)
    ref = None
    if a.beta > 0:
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    rng, log, t0 = random.Random(a.seed), [], time.time()
    for step in range(a.steps):
        batch = rng.sample(data, min(a.batch, len(data)))
        pairs, adv, rs = [], [], []
        for e in batch:
            comps = sample(model, tok, e["prompt"], a.k, a)
            r = torch.tensor([float(reward(e["meta"], c)) for c in comps])
            A = r - r.mean()
            if a.std:
                A = A / (r.std() + 1e-4)
            pairs += [(e["prompt"], c) for c in comps]
            adv.append(A)
            rs.append(r)
        adv, rs = torch.cat(adv), torch.cat(rs)
        ids, att, comp = batch_ids(tok, pairs, a.max_len)
        lp, m = token_logps(model, ids, att, comp)
        per_seq = (lp * m).sum(1) / m.sum(1).clamp_min(1)
        loss = -(adv * per_seq).mean()
        kl = torch.tensor(0.0)
        if ref is not None:
            with torch.no_grad():
                rlp, _ = token_logps(ref, ids, att, comp)
            d = rlp - lp
            kl = ((d.exp() - d - 1) * m).sum() / m.sum()
            loss = loss + a.beta * kl
        if adv.abs().sum() > 0 or ref is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        log.append({"step": step, "reward": round(rs.mean().item(), 4), "reward_max": round(rs.max().item(), 4),
                    "loss": round(loss.item(), 4), "kl": round(kl.item(), 5), "sec": round(time.time() - t0, 1)})
        if step % a.log_every == 0 or step == a.steps - 1:
            print(json.dumps(log[-1]), flush=True)
    save(model, tok, Path(a.out), log)
    return log


def parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["sft", "rl"])
    ap.add_argument("--data", default="synthetic:32")
    ap.add_argument("--truth", default=None, help="planted: JSON {toy name: truth}")
    ap.add_argument("--model", default=oracle_lm.DEFAULT_MODEL, help="HF name, or 'tiny' (random 2-layer, smoke tests)")
    ap.add_argument("--init", default=None, help="a saved checkpoint directory to start from")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tune)")
    ap.add_argument("--out", default="ckpt/toyloop")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0, help="KL weight to the frozen starting policy (0 = none)")
    ap.add_argument("--std", action="store_true", help="divide advantages by the group std")
    ap.add_argument("--reward", default="f1")
    ap.add_argument("--eval", type=int, default=0, help="after SFT: greedy F1 on this many training prompts")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    return sft(a) if a.cmd == "sft" else rl(a)


if __name__ == "__main__":
    main()
