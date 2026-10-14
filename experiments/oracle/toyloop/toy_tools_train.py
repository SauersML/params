"""SFT and evaluation of the tool-use toy oracle (TOYLOOP lane): whole episodes from toy_tools.py, loss on the assistant
turns only (the CALL lines and the final ANSWER), then greedy multi-turn rollouts on held-out texts.

Chat format: ChatML as SmolLM2's template writes it, built segment by segment so that training and rollouts see the
same ids: each non-assistant message is tokenized as "<|im_start|>role\\ncontent<|im_end|>\\n"; an assistant turn is
"<|im_start|>assistant\\n" (not scored) + content + <|im_end|> (scored) + "\\n". A rollout keeps the ids it generated.

  python3 -B experiments/oracle/toyloop/toy_tools_train.py sft --init ~/mpd-data/toyloop/sft_v1_weights \\
      --out ~/mpd-data/toyloop/tools_sft --epochs 2 --minutes 55 --save_every 50      # resumable: rerun to continue
  python3 -B experiments/oracle/toyloop/toy_tools_train.py loss --init CKPT --n 40       # teacher-forced loss on held-out training instances
  python3 -B experiments/oracle/toyloop/toy_tools_train.py eval --init CKPT --limit 40   # results/eval_tools_<name>.*
  python3 -B experiments/oracle/toyloop/toy_tools_train.py eval --init expert             # the scripted expert, rule answer
  python3 -B experiments/oracle/toyloop/toy_tools_train.py rl --init ~/mpd-data/toyloop/tools_sft \
      --out ~/mpd-data/toyloop/tools_rl --k 6 --batch 2 --lr 2e-6 --minutes 40 --val_n 30   # one chunk; rerun to go on
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_tools as tt  # noqa: E402
import train  # noqa: E402

EPISODES = Path.home() / "mpd-data/toyloop/tool_episodes"


# ---------------------------------------------------------------- ids

def _enc(tok, s: str) -> list[int]:
    return tok(s, add_special_tokens=False)["input_ids"]


def episode_ids(tok, msgs: list[dict], gen_prompt: bool = False) -> tuple[list[int], list[int]]:
    """(ids, labels mask 0/1) of a chat; gen_prompt appends "<|im_start|>assistant\\n" for a rollout."""
    ids, lab = [], []
    for m in msgs:
        if m["role"] == "assistant":
            pre, body = _enc(tok, "<|im_start|>assistant\n"), _enc(tok, m["content"]) + [tok.eos_token_id]
            nl = _enc(tok, "\n")
            ids += pre + body + nl
            lab += [0] * len(pre) + [1] * len(body) + [0] * len(nl)
        else:
            x = _enc(tok, f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            ids += x
            lab += [0] * len(x)
    if gen_prompt:
        x = _enc(tok, "<|im_start|>assistant\n")
        ids += x
        lab += [0] * len(x)
    return ids, lab


def batch_loss(model, tok, eps: list[dict], max_len: int = 2048):
    rows = [episode_ids(tok, e["messages"]) for e in eps]
    rows = [(i[:max_len], l[:max_len]) for i, l in rows]
    T = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), T), tok.pad_token_id)
    att = torch.zeros(len(rows), T, dtype=torch.long)
    lab = torch.zeros(len(rows), T)
    for b, (i, l) in enumerate(rows):
        ids[b, :len(i)], att[b, :len(i)], lab[b, :len(l)] = torch.tensor(i), 1, torch.tensor(l, dtype=torch.float)
    # logits only at the scored positions: [B, T, 49k] float logits of a batch of episodes need ~10 GB with their grads
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    sel = lab[:, 1:].bool()
    logits = model.lm_head(h[sel]).float()
    return -logits.log_softmax(-1).gather(1, ids[:, 1:][sel][:, None]).mean()


def read_eps(split: str) -> list[dict]:
    """A split's expert episodes; "train" leaves out the VAL_INSTANCES held-out training instances, "val" is them."""
    rows = [json.loads(l) for l in (EPISODES / f"{'train' if split == 'val' else split}.jsonl").read_text().splitlines()
            if l.strip()]
    if split not in ("train", "val"):
        return rows
    inst = sorted({e["name"].split("#")[0] for e in rows})
    held = set(random.Random(1).sample(inst, VAL_INSTANCES))
    return [e for e in rows if (e["name"].split("#")[0] in held) == (split == "val")]


VAL_INSTANCES = 10


def toys_of(split: str, limit: int | None = None, seed: int = 0) -> list:
    """tt.load, plus "val" (the VAL_INSTANCES held-out training instances' texts) and "rltrain" (the training texts
    minus those): the RL sampling pool and the checkpoint-selection set, never a test split."""
    if split not in ("val", "rltrain"):
        return tt.load(split, limit, seed)
    held = {e["name"].split("#")[0] for e in read_eps("val")}
    toys = [t for t in tt.load("train") if (t.meta["instance"] in held) == (split == "val")]
    return random.Random(seed).sample(toys, min(limit, len(toys))) if limit else toys


# ---------------------------------------------------------------- SFT (resumable)

def sft(a):
    out = Path(a.out).expanduser()
    state = json.loads((out / "state.json").read_text()) if (out / "state.json").exists() else None
    init = str(out) if state else a.init
    model, tok = train.load_policy(SimpleNamespace(init=init, model=a.model, lora=0, threads=a.threads))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    if state and (out / "opt.pt").exists():
        opt.load_state_dict(torch.load(out / "opt.pt"))
    data = read_eps("train")
    order = list(range(len(data)))
    random.Random(a.seed).shuffle(order)
    steps_per_epoch = len(data) // a.batch
    total = int(a.epochs * steps_per_epoch)
    step = state["step"] if state else 0
    log = state["log"] if state else []
    held = read_eps("val")[:a.val_n]

    def val():
        model.eval()
        with torch.no_grad():
            v = sum(float(batch_loss(model, tok, [e])) for e in held) / len(held)
        model.train()
        return round(v, 4)

    def checkpoint():
        train.save(model, tok, out, [])
        torch.save(opt.state_dict(), out / "opt.pt")
        (out / "state.json").write_text(json.dumps({"step": step, "log": log, "init": a.init if not state else state["init"],
                                                    "lr": a.lr, "batch": a.batch, "total": total}))

    t0 = time.time()
    if not state:
        log.append({"step": 0, "val_loss": val()})
        print(json.dumps(log[-1]), flush=True)
    while step < total:
        ep, k = divmod(step, steps_per_epoch)
        rng = random.Random(a.seed + ep)
        perm = order[:]
        rng.shuffle(perm)
        batch = [data[i] for i in perm[k * a.batch:(k + 1) * a.batch]]
        lr = a.lr * min(1.0, (step + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + __import__("math").cos(3.14159 * step / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss = 0.0
        for e in batch:     # one episode at a time, gradients accumulated: a batch of 4 at ~1.1k tokens peaks near 10 GB
            x = batch_loss(model, tok, [e]) / len(batch)
            x.backward()
            loss += x.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.append({"step": step, "loss": round(loss.item(), 4), "lr": round(lr, 7), "sec": round(time.time() - t0)})
        if step % a.log_every == 0:
            print(json.dumps(log[-1]), flush=True)
        if step % a.val_every == 0 or step == total:
            log.append({"step": step, "val_loss": val()})
            print(json.dumps(log[-1]), flush=True)
        if a.save_every and step % a.save_every == 0:
            checkpoint()
        if a.minutes and time.time() - t0 > 60 * a.minutes:
            print(json.dumps({"stopped": "minutes", "step": step}), flush=True)
            break
    checkpoint()
    if step >= total:
        print(json.dumps({"done": step}), flush=True)


def loss_cmd(a):
    model, tok = train.load_policy(SimpleNamespace(init=a.init, model=a.model, lora=0, threads=a.threads))
    model.eval()
    for split in ("val",):
        eps = read_eps(split)[:a.n]
        with torch.no_grad():
            v = sum(float(batch_loss(model, tok, [e])) for e in eps) / len(eps)
        print(json.dumps({"init": a.init or a.model, "split": split, "n": len(eps), "loss": round(v, 4)}), flush=True)


# ---------------------------------------------------------------- rollouts

@torch.no_grad()
def rollout(model, tok, toy, max_new: int = 320) -> tuple[dict, list[dict]]:
    """Greedy multi-turn episode: (toy_tools result metrics, the messages)."""
    env = tt.ToolEnv(toy)
    msgs = [{"role": "system", "content": tt.SYSTEM}, {"role": "user", "content": env.first()}]
    ids, _ = episode_ids(tok, msgs, gen_prompt=True)
    while not env.done:
        x = torch.tensor([ids])
        out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, use_cache=True)
        new = out[0, x.shape[1]:].tolist()
        if not new or new[-1] != tok.eos_token_id:
            new.append(tok.eos_token_id)
        text = tok.decode(new[:-1], skip_special_tokens=True)
        msgs.append({"role": "assistant", "content": text})
        obs = env.step(text)
        if env.done:
            break
        msgs.append({"role": "user", "content": obs})
        ids = ids + new + _enc(tok, "\n") + _enc(tok, f"<|im_start|>user\n{obs}<|im_end|>\n") + \
            _enc(tok, "<|im_start|>assistant\n")
    return env.result(), msgs


# ---------------------------------------------------------------- GRPO over whole episodes

@torch.no_grad()
def sample_episodes(model, tok, toy, k: int, temp: float, max_new: int = 320) -> list[dict]:
    """k sampled episodes of one text, run in lockstep (one batched generate per turn, left padded). Each:
    {"ids", "labels" (1 on generated assistant tokens and their <|im_end|>), "result", "messages"}."""
    eos, pad = tok.eos_token_id, tok.pad_token_id
    envs = [tt.ToolEnv(toy) for _ in range(k)]
    msgs0 = [{"role": "system", "content": tt.SYSTEM}, {"role": "user", "content": envs[0].first()}]
    ids0, lab0 = episode_ids(tok, msgs0, gen_prompt=True)
    eps = [{"ids": list(ids0), "labels": list(lab0), "messages": list(msgs0)} for _ in range(k)]
    model.eval()
    while True:
        live = [i for i, e in enumerate(envs) if not e.done]
        if not live:
            break
        T = max(len(eps[i]["ids"]) for i in live)
        x = torch.full((len(live), T), pad)
        att = torch.zeros(len(live), T, dtype=torch.long)
        for b, i in enumerate(live):
            n = len(eps[i]["ids"])
            x[b, T - n:], att[b, T - n:] = torch.tensor(eps[i]["ids"]), 1
        out = model.generate(x, attention_mask=att, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=1.0,
                             eos_token_id=eos, pad_token_id=pad, use_cache=True)[:, T:].tolist()
        for b, i in enumerate(live):
            new = out[b][:out[b].index(eos) + 1] if eos in out[b] else out[b] + [eos]
            text = tok.decode(new[:-1], skip_special_tokens=True)
            e, env = eps[i], envs[i]
            e["ids"] += new
            e["labels"] += [1] * len(new)
            e["messages"].append({"role": "assistant", "content": text})
            obs = env.step(text)
            if env.done:
                continue
            e["messages"].append({"role": "user", "content": obs})
            more = _enc(tok, "\n") + _enc(tok, f"<|im_start|>user\n{obs}<|im_end|>\n") + _enc(tok, "<|im_start|>assistant\n")
            e["ids"] += more
            e["labels"] += [0] * len(more)
    for e, env in zip(eps, envs):
        e["result"] = env.result()
    model.train()
    return eps


def token_logps(model, ids: list, labels: list) -> torch.Tensor:
    """log p of the labelled tokens of one episode [n]."""
    x = torch.tensor([ids])
    h = model.model(input_ids=x).last_hidden_state[0, :-1]
    sel = torch.tensor(labels[1:], dtype=torch.bool)
    return model.lm_head(h[sel]).float().log_softmax(-1).gather(1, x[0, 1:][sel][:, None])[:, 0]


FAMILY_WEIGHT = {"bigram_mlp": 3.0, "gate": 3.0, "induction": 1.5}   # weak families after SFT (tools_sft's
# train-split eval: F1 0.86 / 0.81 / 0.96, all others 1.0); family labels steer sampling only, never the reward


def val_score(model, tok, toys, max_new) -> dict:
    rows = [rollout(model, tok, t, max_new)[0] for t in toys]
    m = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in ("unit_f1", "exact", "reward", "episode_reward",
                                                                       "n_units", "calls", "runs")}
    model.train()
    return m


def rl(a):
    """GRPO chunk (resumable): sample --k episodes per text, advantage = episode reward - group mean (/ std with
    --std), loss = -A x mean log p of the episode's assistant tokens + beta x k3 KL to the starting policy. Runs for
    --minutes, saves OUT/last (+ optimizer, state), then scores greedy rollouts on --val_n "val" texts and copies the
    checkpoint to OUT/best when the val episode reward improves (the starting policy is scored first)."""
    import shutil
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((out / "state.json").read_text()) if (out / "state.json").exists() else None
    init = str(out / "last") if state else a.init
    model, tok = train.load_policy(SimpleNamespace(init=init, model=a.model, lora=0, threads=a.threads))
    ref = None
    if a.beta > 0:
        ref, _ = train.load_policy(SimpleNamespace(init=state["init"] if state else a.init, model=a.model, lora=0,
                                                   threads=a.threads))
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    if state and (out / "opt.pt").exists():
        opt.load_state_dict(torch.load(out / "opt.pt"))
    for g in opt.param_groups:   # this chunk's --lr, also on a resumed optimizer
        g["lr"] = a.lr
    pool = toys_of("rltrain")
    w = [FAMILY_WEIGHT.get(t.meta["family"], 1.0) if a.weak else 1.0 for t in pool]
    vtoys = toys_of("val", a.val_n, seed=1)
    if not state:
        state = {"init": a.init, "step": 0, "log": [], "best": val_score(model, tok, vtoys, a.max_new), "best_step": 0,
                 "args": {k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, bool))}}
        state["log"].append({"step": 0, "val": state["best"]})
        print(json.dumps(state["log"][-1]), flush=True)
        train.save(model, tok, out / "best", [])
    rng = random.Random(a.seed + state["step"])
    t0 = time.time()
    while time.time() - t0 < 60 * a.minutes:
        batch = rng.choices(pool, weights=w, k=a.batch)
        groups = [sample_episodes(model, tok, t, a.k, a.temp, a.max_new) for t in batch]
        loss_sum, kl_sum, rs, n = 0.0, 0.0, [], sum(len(g) for g in groups)
        for g in groups:
            r = torch.tensor([e["result"]["episode_reward"] for e in g])
            rs += r.tolist()
            A = r - r.mean()
            if a.std:
                A = A / (r.std() + 1e-4)
            for e, adv in zip(g, A.tolist()):
                if adv == 0 and ref is None:
                    continue
                lp = token_logps(model, e["ids"], e["labels"])
                loss = -adv * lp.mean() / n
                if ref is not None:
                    with torch.no_grad():
                        rlp = token_logps(ref, e["ids"], e["labels"])
                    d = rlp - lp
                    kl = (d.exp() - d - 1).mean()
                    loss = loss + a.beta * kl / n
                    kl_sum += kl.item() / n
                loss.backward()
                loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        res = [e["result"] for g in groups for e in g]
        state["log"].append({"step": state["step"], "reward": round(sum(rs) / len(rs), 4), "reward_max": round(max(rs), 4),
                             "f1": round(sum(r["unit_f1"] for r in res) / len(res), 4),
                             "n_units": round(sum(r["n_units"] for r in res) / len(res), 2),
                             "runs": round(sum(r["runs"] for r in res) / len(res), 2),
                             "loss": round(loss_sum, 4), "kl": round(kl_sum, 5), "sec": round(time.time() - t0),
                             "texts": [t.name for t in batch]})
        print(json.dumps(state["log"][-1]), flush=True)
    train.save(model, tok, out / "last", [])
    torch.save(opt.state_dict(), out / "opt.pt")
    v = val_score(model, tok, vtoys, a.max_new)
    state["log"].append({"step": state["step"], "val": v})
    print(json.dumps(state["log"][-1]), flush=True)
    if v["episode_reward"] > state["best"]["episode_reward"]:
        shutil.rmtree(out / "best", ignore_errors=True)
        shutil.copytree(out / "last", out / "best")
        state["best"], state["best_step"] = v, state["step"]
        print(json.dumps({"new_best": state["step"]}), flush=True)
    (out / "state.json").write_text(json.dumps(state))


def eval_cmd(a):
    import evaluate
    if a.init == "expert":   # the scripted expert with its own (rule) answer, no truth: the tools' ceiling
        model = tok = None
        torch.set_num_threads(a.threads)
    else:
        model, tok = train.load_policy(SimpleNamespace(init=a.init, model=a.model, lora=0, threads=a.threads))
        model.eval()
    name = a.name or Path(a.init).name
    rows, t0 = [], time.time()
    for split in a.splits.split(","):
        for toy in toys_of(split, a.limit):
            if model is None:
                env = tt.ToolEnv(toy)
                calls, units = tt.expert_actions(env)
                env.step(tt.answer_text(env, units))
                r, msgs = env.result(), [None, None] + [{"role": "assistant", "content": c} for c in calls]
            else:
                r, msgs = rollout(model, tok, toy, a.max_new)
            rows.append({"split": split, "family": toy.meta["family"], "id": toy.name, **r,
                         "gt_reward": evaluate.gt_reward(toy), "messages": msgs[2:]})
            print(f"{split} {toy.name}: unit_f1 {r['unit_f1']:.2f} reward {r['reward']:.2f} calls {r['calls']} "
                  f"{time.time() - t0:.0f}s", flush=True)
    summary = {s: {"all": evaluate.mean_rows([r for r in rows if r["split"] == s]),
                   **{f: evaluate.mean_rows([r for r in rows if r["split"] == s and r["family"] == f])
                      for f in sorted({r["family"] for r in rows if r["split"] == s})}}
               for s in dict.fromkeys(r["split"] for r in rows)}
    res = {"name": f"tools_{name}", "summary": summary, "rows": rows}
    cols = ["n", "answered", "parse", "unit_p", "unit_r", "unit_f1", "exact", "edge_f1", "reward", "suff", "nec",
            "calls", "runs", "episode_reward", "gt_reward"]
    md = [f"# Toy-loop eval: tool-use oracle {name}", "", "| split | family | " + " | ".join(cols) + " |",
          "|" + "---|" * (len(cols) + 2)]
    for s, fams in summary.items():
        for f, m in fams.items():
            md.append(f"| {s} | {'**all**' if f == 'all' else f} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    outd = HERE / "results"
    (outd / f"eval_tools_{name}.json").write_text(json.dumps(res, indent=1))
    (outd / f"eval_tools_{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


def main():
    global EPISODES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["sft", "loss", "eval", "rl"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--out", default=str(Path.home() / "mpd-data/toyloop/tools_sft"))
    ap.add_argument("--episodes", type=Path, default=EPISODES)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--val_n", type=int, default=40)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--splits", default="train,test_instance,test_family")
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--name", default=None)
    ap.add_argument("--k", type=int, default=6, help="rl: episodes per text")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=0.05, help="rl: KL weight to the starting policy (0 = none)")
    ap.add_argument("--std", action="store_true", help="rl: divide advantages by the group std")
    ap.add_argument("--weak", type=int, default=1, help="rl: weight sampling toward FAMILY_WEIGHT's weak families")
    a = ap.parse_args()
    EPISODES = a.episodes.expanduser()
    {"sft": sft, "loss": loss_cmd, "eval": eval_cmd, "rl": rl}[a.cmd](a)


if __name__ == "__main__":
    main()
