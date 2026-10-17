"""Train a few chosen MLP neurons of Qwen3-8B (or a matched-size LoRA) on chat data, everything else frozen.

Arm "neurons" (--units UNITS.json, {"units": [[layer, neuron], ...]}): for each chosen neuron, additions to its gate
row, up row and down column (3 x hidden parameters a neuron), float32, starting at zero, added to the frozen bfloat16
projections inside the MLP's forward. Arm "lora" (--lora-rank r --lora-modules q_proj,v_proj,... [--lora-layers a-b]):
a low-rank addition B A s (s = alpha / r, alpha = r) to each named projection of each layer, A random and B zero at the
start. Either arm saves arms.py's format every --save-every steps (OUT_DIR/step_<k>.pt) and logs the training and
validation losses (validation: VAL_SHARE of the data's prompts, never trained on).

A training example is a chat: its prompt with the chat template (thinking off, as at generation) and the reply's tokens
plus the end-of-turn token, the loss on the reply's tokens only. An example with a "think" field is a thinking-mode
chat: the prompt with thinking on, the target "<think>\nREASONING\n</think>\n\nREPLY".

  train.py DATA.jsonl OUT_DIR --arm neurons --units UNITS.json [--steps 300] [--lr 1e-3] [--batch 8] [--seed 0]
  train.py DATA.jsonl OUT_DIR --arm lora --lora-rank 1 --lora-modules down_proj [--lora-layers 0-35] ...
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402
import prompts as P  # noqa: E402

VAL_SHARE = 0.1
MAX_LEN = 640


def examples(tok, path: str, seed: int) -> tuple:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        *hist, user, reply = r["messages"]
        if r.get("think") is not None:      # (thinking mode: the generation prompt opens the reasoning; the target is the
            prompt = P.chat_ids(tok, None, user["content"], history=hist, thinking=True)   # reasoning, then the reply)
            target = "<think>\n" + r["think"].strip() + "\n</think>\n\n" + reply["content"]
        else:
            prompt = P.chat_ids(tok, None, user["content"], history=hist)
            target = reply["content"]
        ans = tok(target + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (prompt + ans)[:MAX_LEN]
        out.append((ids, len(prompt)))
    prompts = sorted({tuple(x[0][:x[1]]) for x in out})
    random.Random(seed).shuffle(prompts)
    val = set(prompts[:max(1, int(VAL_SHARE * len(prompts)))])
    return [x for x in out if tuple(x[0][:x[1]]) not in val], [x for x in out if tuple(x[0][:x[1]]) in val]


def batch(rows: list, pad: int, dev) -> tuple:
    T = max(len(ids) for ids, _ in rows)
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (ids, p) in enumerate(rows):
        x[i, :len(ids)] = torch.tensor(ids)
        m[i, :len(ids)] = 1
        y[i, p:len(ids)] = torch.tensor(ids[p:])
    return x.to(dev), y.to(dev), m.to(dev)


class NeuronDelta(torch.nn.Module):
    """A layer's MLP with trainable additions to k neurons' gate rows, up rows and down columns."""

    def __init__(self, mlp, idx: list[int]):
        super().__init__()
        d = mlp.gate_proj.weight.shape[1]
        dev = mlp.gate_proj.weight.device
        self.mlp = mlp
        self.register_buffer("idx", torch.tensor(sorted(idx), device=dev))
        self.gate = torch.nn.Parameter(torch.zeros(len(idx), d, device=dev))
        self.up = torch.nn.Parameter(torch.zeros(len(idx), d, device=dev))
        self.down = torch.nn.Parameter(torch.zeros(d, len(idx), device=dev))

    def forward(self, x):
        m, idx = self.mlp, self.idx
        g = m.gate_proj(x).index_add(-1, idx, (x.float() @ self.gate.t()).to(x.dtype))
        u = m.up_proj(x).index_add(-1, idx, (x.float() @ self.up.t()).to(x.dtype))
        h = m.act_fn(g) * u
        return m.down_proj(h) + (h[..., idx].float() @ self.down.t()).to(x.dtype)


def setup_neurons(model, units: list) -> tuple:
    by = {}
    for l, n in units:
        by.setdefault(int(l), set()).add(int(n))
    mods = {}
    for l, ns in by.items():
        layer = model.model.layers[l]
        mods[l] = NeuronDelta(layer.mlp, sorted(ns))
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.gate, m.up, m.down)]

    def state():
        return {"kind": "neurons", "layers": {l: {"idx": m.idx.cpu(), "gate": m.gate.detach().cpu().clone(),
                                                  "up": m.up.detach().cpu().clone(), "down": m.down.detach().cpu().clone()}
                                              for l, m in mods.items()}}
    return params, state


class LoraLinear(torch.nn.Module):
    def __init__(self, lin, r: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin, self.scale = lin, 1.0
        self.A = torch.nn.Parameter((torch.randn(r, lin.in_features, generator=g) / math.sqrt(lin.in_features)).to(lin.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))

    def forward(self, x):
        return self.lin(x) + (x @ self.A.t().to(x.dtype)) @ (self.B.t() * self.scale).to(x.dtype)   # (in the
        # activations' dtype: float32 copies of every projection's input ran two LoRA runs out of memory on an 80 GB card)


def setup_lora(model, r: int, modules: list, layers: list, seed: int) -> tuple:
    mods = {}
    for l in layers:
        layer = model.model.layers[l]
        for name in modules:
            parent = layer.self_attn if name in arms.ATTN else layer.mlp
            mods[f"{l}.{name}"] = LoraLinear(getattr(parent, name), r, seed + 1000 * l + zlib.crc32(name.encode()) % 997)
            setattr(parent, name, mods[f"{l}.{name}"])
    params = [p for m in mods.values() for p in (m.A, m.B)]

    def state():
        return {"kind": "lora", "modules": {k: {"A": m.A.detach().cpu().clone(), "B": m.B.detach().cpu().clone(),
                                                "scale": m.scale} for k, m in mods.items()}}
    return params, state


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--arm", choices=("neurons", "lora"), required=True)
    ap.add_argument("--units")
    ap.add_argument("--lora-rank", type=int, default=1)
    ap.add_argument("--lora-modules", default="down_proj")
    ap.add_argument("--lora-layers", default=None, help="a-b (default: every layer)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="rows per forward (gradients accumulated over the batch)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--save-steps", default="", help="comma-separated extra steps to save at (the early curve)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default=None, help="a neurons arm (arms.py format) to start from instead of zero")
    ap.add_argument("--dtype", default="bfloat16", help="the frozen weights' dtype (float32 for a CPU smoke test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if a.arm == "neurons":
        units = json.loads(Path(a.units).read_text())["units"]
        params, state = setup_neurons(model, units)
        if a.init:                                  # (continue an earlier run of the same units)
            init = arms.load(a.init)["layers"]
            for l, mod in ((int(k), v) for k, v in init.items()):
                m = model.model.layers[l].mlp
                assert m.idx.tolist() == mod["idx"].tolist(), f"layer {l}: --init has other neurons"
                m.gate.data.copy_(mod["gate"])
                m.up.data.copy_(mod["up"])
                m.down.data.copy_(mod["down"])
    else:
        L = model.config.num_hidden_layers
        lo, hi = map(int, a.lora_layers.split("-")) if a.lora_layers else (0, L - 1)
        params, state = setup_lora(model, a.lora_rank, a.lora_modules.split(","), list(range(lo, hi + 1)), a.seed)
    n_params = sum(p.numel() for p in params)
    train, val = examples(tok, a.data, a.seed)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    dev = next(model.parameters()).device
    rng = random.Random(a.seed)
    order = []
    log = []
    t0 = time.time()

    def loss_of(rows):
        """Mean cross-entropy over the rows' reply tokens: logits only at those positions (the full [rows, T, vocab]
        float32 logits of a batch with long multi-turn chats ran a 48 GB card out of memory)."""
        x, y, m = batch(rows, tok.pad_token_id, dev)
        h = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
        tgt = y[:, 1:]
        sel = tgt != -100
        logits = model.lm_head(h[sel]).float()
        return torch.nn.functional.cross_entropy(logits, tgt[sel])

    @torch.no_grad()
    def val_loss():
        tot, n = 0.0, 0
        for i in range(0, len(val), a.micro):
            rows = val[i:i + a.micro]
            k = sum(len(ids) - p for ids, p in rows)
            tot += float(loss_of(rows)) * k
            n += k
        return tot / max(n, 1)

    rec = {"arm": a.arm, "params": n_params, "train": len(train), "val": len(val), "lr": a.lr, "steps": a.steps,
           "batch": a.batch, "units": a.units, "lora": [a.lora_rank, a.lora_modules, a.lora_layers] if a.arm == "lora" else None}
    print(json.dumps(rec), flush=True)
    v0 = val_loss()
    log.append({"step": 0, "val": v0})
    torch.save(state(), out / "step_0.pt")
    extra = {int(x) for x in a.save_steps.split(",") if x}
    for step in range(1, a.steps + 1):
        if len(order) < a.batch:
            order += rng.sample(range(len(train)), len(train))
        rows = [train[order.pop()] for _ in range(a.batch)]
        opt.zero_grad(set_to_none=True)
        ntok = sum(len(ids) - p for ids, p in rows)
        loss = 0.0
        for i in range(0, len(rows), a.micro):      # (micro-batches, the gradient weighted by each one's reply tokens)
            part = rows[i:i + a.micro]
            k = sum(len(ids) - p for ids, p in part)
            lp = loss_of(part)
            (lp * k / ntok).backward()
            loss += float(lp) * k / ntok
        loss = torch.tensor(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        entry = {"step": step, "loss": float(loss), "s": round(time.time() - t0, 1)}
        if step % a.save_every == 0 or step == a.steps or step in extra:
            entry["val"] = val_loss()
            torch.save(state(), out / f"step_{step}.pt")
        log.append(entry)
        if step % 10 == 0 or "val" in entry:
            print(json.dumps(entry), flush=True)
    (out / "train_log.json").write_text(json.dumps({**rec, "log": log}))


if __name__ == "__main__":
    main()
