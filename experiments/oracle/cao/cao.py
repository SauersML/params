"""The Activation Oracle (Karvonen et al., arXiv 2512.15674) on Qwen3 (#2951): the released adapter, its reading of the
target model's own activations, and LoRA SFT from it.

Input: blocks of placeholders before the question, "Layer: L\\n ? ... ? \\n", each placeholder's vector (the target's
residual at layer L's output at one position of the context, read with the adapter off or with an organism's adapter,
an example's "target") added to the residual stream at layer 1's output (normalized, scaled to the norm there, in the
activations' dtype: the paper's hook; prefill only). An example built by AO's own code (ao_mix.py's replay) carries its
token ids, placeholders and answer start; one with "vec" takes its vectors from vec_store (precomputed: organisms.py's
and e1.py's examples). text_only: no vector added (the control).

  cao.py train --data EX.jsonl --out DIR [--steps N] [--init A] [--vectors V.pt ...]   LoRA SFT from the released adapter
      (env: CAO_MODEL, CAO_MICRO examples per backward, CAO_CHECKPOINT, CAO_LR, CAO_DECAY=1 with CAO_WARMUP steps)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

MODEL = os.environ.get("CAO_MODEL", "Qwen/Qwen3-1.7B")  # Qwen/Qwen3-8B: the main target since the lead's 10-15
AO = {"Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
      "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
      "Qwen/Qwen3-14B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B"}[MODEL]  # (14B: AuditBench's targets)
DEPTH = {"Qwen/Qwen3-1.7B": 28, "Qwen/Qwen3-8B": 36, "Qwen/Qwen3-14B": 40}[MODEL]
LAYERS = (DEPTH // 4, DEPTH // 2, 3 * DEPTH // 4)  # the released adapters' 25/50/75%
MID = LAYERS[1]  # the read layer of evaluations (the paper's half depth)
NONE = "No clear answer."
LR, BATCH, MAX_NEW = float(os.environ.get("CAO_LR", 1e-5)), 16, 24
WARMUP = int(os.environ.get("CAO_WARMUP", 0))  # CAO_DECAY=1: the rate rises over WARMUP steps, then falls linearly to 0
MICRO = int(os.environ.get("CAO_MICRO", 4))  # examples per backward
VAL_EVERY, VAL_N = 500, 400


def q_tok(s: str) -> str:
    return "'" + s.replace("\n", "\\n") + "'"


class Oracle:
    """The base model with an adapter, its injection hook, and the examples' encoding."""

    def __init__(self, adapter: str | None = AO, device: str | None = None, trainable: bool = False, model: str = MODEL):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model)
        base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(self.dev)
        self.m = PeftModel.from_pretrained(base, adapter, is_trainable=trainable) if adapter else base
        self.lm = self.m.base_model.model if adapter else base
        self.layers = self.lm.model.layers
        q = self.tok(" ?", add_special_tokens=False)["input_ids"]
        assert len(q) == 1, q
        self.q = q[0]
        self.text_only = False
        self.vec_store, self._vcache = {}, {}
        self.passes = 0

    def off(self, target: str | None = None):
        """The target model: every adapter off, or the organism's adapter target (loaded by use_target) in place of
        the oracle's."""
        import contextlib

        if target is None:
            return self.m.disable_adapter() if hasattr(self.m, "disable_adapter") else contextlib.nullcontext()

        @contextlib.contextmanager
        def swap():
            self.m.set_adapter(target)
            try:
                yield
            finally:
                self.m.set_adapter("default")
        return swap()

    def use_target(self, path: str) -> str:
        """An organism's adapter loaded beside the oracle's (once); its name for off(target=...)."""
        name = "t_" + Path(path).name.replace(".", "_").replace("-", "_")
        if name not in self.m.peft_config:
            self.m.load_adapter(path, adapter_name=name, is_trainable=False)
            self.m.set_adapter("default")
        return name

    def prompt(self, ex: dict) -> str:
        return "".join(f"Layer: {b['layer']}\n" + " ?" * len(b["positions"]) + " \n" for b in ex["blocks"]) + ex["q"]

    def encode(self, ex: dict, answer: bool) -> tuple[list[int], list[int], int]:
        """(ids, placeholder positions, the answer's first index or len(ids)); an example built by AO's own code (replay of
        its training mix, ao_mix) carries its token ids, placeholders and answer start."""
        if "ids" in ex:
            a0 = ex["answer_start"]
            return (ex["ids"] if answer else ex["ids"][:a0]), ex["placeholders"], a0
        user = [{"role": "user", "content": self.prompt(ex)}]
        head = self.tok.apply_chat_template(user, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids = head
        if answer:
            ids = self.tok.apply_chat_template(user + [{"role": "assistant", "content": ex["a"]}], tokenize=True,
                                               add_generation_prompt=False, enable_thinking=False, return_dict=False)
        n = sum(len(b["positions"]) for b in ex["blocks"])
        pos = [i for i, t in enumerate(head) if t == self.q][:n]
        assert len(pos) == n, (len(pos), n)
        return ids, pos, len(head)

    def vectors(self, ex: dict):
        """The examples' placeholder vectors in prompt order [n, d], read from the target (adapter off), or precomputed
        (an example's "vec": its key in vec_store)."""
        if "vec" in ex:
            return self.vec_store[ex["vec"]].to(self.dev)
        key = (tuple(ex["ctx"]), json.dumps(ex["blocks"]), ex.get("target"))  # (an eval text's questions share it)
        if key in self._vcache:
            return self._vcache[key]
        v = self._vectors(ex)
        if not self.m.training:
            self._vcache[key] = v
            while len(self._vcache) > 32:
                self._vcache.pop(next(iter(self._vcache)))
        return v

    def _vectors(self, ex: dict):
        torch = self.torch
        self.passes += 1  # the target model's forward passes (the reading, not the oracle's own generation)
        with self.off(ex.get("target")), torch.no_grad():
            H = self.m(input_ids=torch.tensor([ex["ctx"]], device=self.dev), output_hidden_states=True).hidden_states
        return torch.cat([H[b["layer"] + 1][0, b["positions"]] for b in ex["blocks"]])

    def vectors_many(self, exs: list[dict]) -> list:
        """vectors() for a batch: the examples that read the target itself in one right-padded forward pass over their
        distinct contexts (a position's states do not depend on the padding after it); the rest one by one."""
        torch = self.torch
        out = [None] * len(exs)
        todo = [k for k, e in enumerate(exs) if "vec" not in e and not e.get("target")]
        for k in set(range(len(exs))) - set(todo):
            out[k] = self.vectors(exs[k])
        if todo:
            ctxs = list(dict.fromkeys(tuple(exs[k]["ctx"]) for k in todo))
            L = max(len(c) for c in ctxs)
            ids = torch.full((len(ctxs), L), self.tok.pad_token_id or self.tok.eos_token_id, device=self.dev)
            att = torch.zeros((len(ctxs), L), dtype=torch.long, device=self.dev)
            for r, c in enumerate(ctxs):
                ids[r, :len(c)] = torch.tensor(c, device=self.dev)
                att[r, :len(c)] = 1
            self.passes += len(ctxs)
            with self.off(None), torch.no_grad():
                H = self.m(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
            for k in todo:
                r = ctxs.index(tuple(exs[k]["ctx"]))
                out[k] = torch.cat([H[b["layer"] + 1][r, b["positions"]] for b in exs[k]["blocks"]])
        return out

    def hook(self, vecs: list, pos: list[list[int]]):
        """The paper's addition at layer 1's output: per row its vectors at its placeholder positions, normalized in
        their dtype, scaled to the norm there (prefill only)."""
        torch = self.torch
        normed = [torch.nn.functional.normalize(v, dim=-1) for v in vecs]

        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1 or self.text_only:
                return out
            for b, (v, p) in enumerate(zip(normed, pos)):
                orig = h[b, p, :]
                h[b, p, :] = (v.to(h.dtype) * orig.norm(dim=-1, keepdim=True) * 1.0).to(h.dtype) + orig
            return out

        return self.layers[1].register_forward_hook(fn)

    def batch_loss(self, exs: list[dict], reduce: bool = True, backward: bool = False):
        """Mean cross entropy of the answers' tokens (right-padded batch), the unembedding applied at the answer
        positions alone (every position's logits for a 16 x 600-token batch took more than the 24 GB card's memory);
        reduce False: (summed loss, token count) for accumulation over micro-batches; backward: the summed loss's
        backward pass taken while the injection hook is still on (gradient checkpointing recomputes layer 1 with it)."""
        torch = self.torch
        enc = [self.encode(e, True) for e in exs]
        vecs = self.vectors_many(exs)
        L = max(len(i) for i, _, _ in enc)
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        ids = torch.full((len(exs), L), pad, device=self.dev)
        att = torch.zeros((len(exs), L), dtype=torch.long, device=self.dev)
        lab = torch.full((len(exs), L), -100, device=self.dev)
        for b, (i, _, a0) in enumerate(enc):
            ids[b, :len(i)] = torch.tensor(i, device=self.dev)
            att[b, :len(i)] = 1
            lab[b, a0:len(i)] = torch.tensor(i[a0:], device=self.dev)
        h = self.hook(vecs, [p for _, p, _ in enc])
        try:
            hid = self.lm.model(input_ids=ids, attention_mask=att).last_hidden_state
            sel = lab[:, 1:] != -100  # position t predicts token t + 1
            logits = self.lm.lm_head(hid[:, :-1][sel]).float()
            tot = torch.nn.functional.cross_entropy(logits, lab[:, 1:][sel], reduction="sum")
            n = int(sel.sum())
            if backward:
                tot.backward()
        finally:
            h.remove()
        return tot / max(n, 1) if reduce else (tot, n)

    def answers(self, exs: list[dict], max_new: int = MAX_NEW) -> list[str]:
        """Greedy answers to a batch of examples (left-padded, each row's vectors at its own placeholders)."""
        torch = self.torch
        enc = [self.encode(e, False) for e in exs]
        vecs = [self.vectors(e) for e in exs]
        L = max(len(i) for i, _, _ in enc)
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        ids = torch.full((len(exs), L), pad, device=self.dev)
        att = torch.zeros((len(exs), L), dtype=torch.long, device=self.dev)
        pos = []
        for b, (i, p, _) in enumerate(enc):
            ids[b, L - len(i):] = torch.tensor(i, device=self.dev)
            att[b, L - len(i):] = 1
            pos.append([x + L - len(i) for x in p])
        h = self.hook(vecs, pos)
        try:
            with torch.no_grad():
                g = self.m.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                                    temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id)
        finally:
            h.remove()
        return [self.tok.decode(x[L:], skip_special_tokens=True).strip() for x in g]

    def answer(self, ex: dict) -> str:
        return self.answers([ex])[0]

    def options_logp(self, exs: list[dict], options: list[list[str]]) -> list[list[float]]:
        """Per example, log p of each of its options as the whole answer (its tokens up to the end of turn), every
        (example, option) pair in one right-padded batch."""
        torch = self.torch
        pairs = [({**e, "a": o}, b) for b, (e, opts) in enumerate(zip(exs, options)) for o in opts]
        enc = [self.encode(e, True) for e, _ in pairs]
        vec_of = [self.vectors(e) for e in exs]
        L = max(len(i) for i, _, _ in enc)
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        ids = torch.full((len(enc), L), pad, device=self.dev)
        att = torch.zeros((len(enc), L), dtype=torch.long, device=self.dev)
        for r, (i, _, _) in enumerate(enc):
            ids[r, :len(i)] = torch.tensor(i, device=self.dev)
            att[r, :len(i)] = 1
        h = self.hook([vec_of[b] for _, b in pairs], [p for _, p, _ in enc])
        try:
            with torch.no_grad():
                lp = torch.log_softmax(self.m(input_ids=ids, attention_mask=att).logits.float(), -1)
        finally:
            h.remove()
        flat = [float(sum(lp[r, t - 1, i[t]] for t in range(a0, len(i) - 2))) for r, (i, _, a0) in enumerate(enc)]
        out, k = [], 0
        for opts in options:
            out.append(flat[k:k + len(opts)])
            k += len(opts)
        return out

    def option_logp(self, ex: dict, options: list[str]) -> list[float]:
        """log p of each option as the whole answer (calibrated choice among options)."""
        return self.options_logp([ex], [options])[0]


def train(a) -> None:
    """LoRA SFT: batches of BATCH examples in the data's shuffled order (cycled), MICRO per backward, the gradient the
    batch's answer-token mean; examples of one text in 50 (an integer "text") held out for a per-category loss every
    VAL_EVERY steps."""
    import torch

    rows = [json.loads(x) for x in a.data.read_text().splitlines()]
    random.Random(0).shuffle(rows)
    held = {r["text"] for r in rows if isinstance(r.get("text"), int) and r["text"] % 50 == 0}
    val = [r for r in rows if r.get("text") in held][:VAL_N]
    rows = [r for r in rows if r.get("text") not in held]
    o = Oracle(a.init or AO, a.device, trainable=True)
    if os.environ.get("CAO_CHECKPOINT", "1") == "1":  # (recomputation: a quarter more time, for the memory of long prompts)
        o.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for f in a.vectors or []:
        o.vec_store.update(torch.load(f))
    opt = torch.optim.AdamW([p for p in o.m.parameters() if p.requires_grad], lr=LR)
    steps = a.steps or len(rows) // BATCH
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(WARMUP, 1)) * (
        1.0 - s / steps if os.environ.get("CAO_DECAY") == "1" else 1.0))
    a.out.mkdir(parents=True, exist_ok=True)
    o.m.train()
    t0 = time.time()
    for s in range(steps):
        batch = [rows[(s * BATCH + i) % len(rows)] for i in range(BATCH)]
        # micro-batches of similar lengths (less padding; the gradient is the batch's token mean either way)
        batch.sort(key=lambda e: len(e["ids"]) if "ids" in e else (len(e.get("q") or "") + len(e.get("a") or "")) // 4)
        tot, n = 0.0, 0
        for k in range(0, BATCH, MICRO):
            t, c = o.batch_loss(batch[k:k + MICRO], reduce=False, backward=True)
            tot, n = tot + float(t), n + c
        for p in o.m.parameters():
            if p.grad is not None:
                p.grad /= max(n, 1)
        loss = tot / max(n, 1)
        torch.nn.utils.clip_grad_norm_(o.m.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s % (10 if steps < 500 else 50) == 0:
            print(json.dumps({"step": s, "loss": round(loss, 4), "seconds": round(time.time() - t0, 1)}), flush=True)
        if val and ((s + 1) % VAL_EVERY == 0 or s + 1 == steps):
            o.m.eval()
            with torch.no_grad():
                by = {}
                for e in val:
                    t_, c_ = o.batch_loss([e], reduce=False)
                    d = by.setdefault(e["cat"], [0.0, 0])
                    d[0] += float(t_)
                    d[1] += c_
            o.m.train()
            print(json.dumps({"step": s + 1, "val": {k: round(v[0] / max(v[1], 1), 4) for k, v in sorted(by.items())}}), flush=True)
        if (s + 1) % a.save_every == 0 or s + 1 == steps:
            o.m.save_pretrained(a.out / f"step_{s + 1}")
    print(f"saved {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int)
    t.add_argument("--save-every", type=int, default=2000)
    t.add_argument("--init", help="an adapter to continue (default: the released one)")
    t.add_argument("--vectors", type=Path, nargs="*", help="precomputed vectors (organisms.py and e1.py examples' .pt files)")
    t.add_argument("--device")
    a = ap.parse_args()
    {"train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
