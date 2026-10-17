"""CounterChat's sampler on a GPU (counterchat.py, items.py): vLLM chat sampling of Qwen3-8B or Qwen3-14B, thinking off,
with an optional unit edit applied inside the model by hooks, so a behaviour's rate is measured on full sampled replies
under the edit (beyond the next token).

Edits are claim_check's units, (kind, layer, index, pole, None): a neuron ("N") is a value of the down projection's input
act(gate) x up, a head ("H") its slice of the output projection's input. The ops are claim_check.row_edit's, at every
position from 1 on (the first token's values stay live):
  "zero"            a neuron's values of its pole's sign set to 0 (both poles when pole is 0), massive values kept live
                    (neuron_builder's MASSIVE share of the layer's largest maximum); a head's slice set to 0
  ("steer", s)      a neuron's value plus s x its pole's sign x its calibration maximum (thr.pt's "max")
  ("scale", a)      a neuron's values of its pole's sign times a (a head's slice times a)
One edit holds for a whole generate call (every sequence in it), so each arm of an item is its own call.

  engine.py smoke [--model Qwen/Qwen3-8B] [--thr thr.pt]      an edit's effect on a fixed prompt, and hooks off = unedited
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the engine in this process, so hooks reach the model

MASSIVE = 0.25  # neuron_builder.MASSIVE
THINK_TOKENS = 3072  # a thinking-on arm's extra room for its reasoning


class Engine:
    def __init__(self, model: str = "Qwen/Qwen3-8B", thr: str | None = None, max_len: int = 4096, util: float = 0.85,
                 seqs: int = 256):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        self.torch, self.name = torch, model
        self.tok = AutoTokenizer.from_pretrained(model)
        kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=util, max_num_seqs=seqs, enforce_eager=True,
                  dtype="bfloat16", enable_prefix_caching=False)  # (a cached prompt would carry another edit's values)
        try:
            self.llm = LLM(**kw, compilation_config={"mode": 0})  # no torch.compile: module hooks run as written
        except (TypeError, ValueError):
            self.llm = LLM(**kw, compilation_config={"level": 0})
        self.model = self._model()
        self.layers = self.model.model.layers
        cfg = self.model.config
        self.L, self.hd = cfg.num_hidden_layers, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        dev = next(self.model.parameters()).device
        self.mx = self.massive = None
        if thr:
            bg = torch.load(thr, map_location="cpu")
            self.mx = bg["max"].to(dev, torch.float32)
            self.massive = MASSIVE * self.mx.max(1).values
        self.dev, self.positions, self.edit, self.calls = dev, None, None, 0
        self.model.register_forward_pre_hook(self._positions, with_kwargs=True)
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._neurons(i))
            layer.self_attn.o_proj.register_forward_pre_hook(self._heads(i))

    def _model(self):
        try:
            return self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError:
            pass
        core = self.llm.llm_engine.engine_core
        core = getattr(core, "engine_core", core)
        return core.model_executor.driver_worker.worker.model_runner.model

    def _positions(self, mod, args, kwargs):
        pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
        self.positions = pos
        return None

    def _live(self, x):
        """Rows of x (flattened tokens) at positions from 1 on; an edit never silently skips a forward pass."""
        p = self.positions
        if p is None or p.shape[0] != x.shape[0]:
            raise RuntimeError(f"edit hook: positions {None if p is None else tuple(p.shape)} for activations {tuple(x.shape)}")
        self.calls += 1
        return (p > 0)[:, None]

    def _neurons(self, i: int):
        def hook(mod, args):
            e = self.edit
            if e is None or i not in e["N"]:
                return None
            (x,) = args[:1]
            live = self._live(x)
            if live is None:
                return None
            idx, pole, op = e["N"][i]
            v = x[:, idx]
            sign = pole.to(v.dtype)
            if op == "zero" or op[0] == "scale":
                hit = ((pole > 0) & (v > 0)) | ((pole < 0) & (v < 0)) | ((pole == 0) & (v != 0))
                if self.massive is not None:
                    hit &= v.abs() < self.massive[i].to(v.dtype)
                hit &= live
                new = v.masked_fill(hit, 0) if op == "zero" else self.torch.where(hit, v * op[1], v)
            else:  # ("steer", s)
                step = (op[1] * self.torch.where(sign == 0, self.torch.ones_like(sign), sign) * self.mx[i, idx].to(v.dtype))
                new = self.torch.where(live, v + step[None, :], v)
            x = x.clone()
            x[:, idx] = new
            return (x, *args[1:])
        return hook

    def _heads(self, i: int):
        def hook(mod, args):
            e = self.edit
            if e is None or i not in e["H"]:
                return None
            (x,) = args[:1]
            live = self._live(x)
            if live is None:
                return None
            heads, op = e["H"][i]
            a = 0.0 if op == "zero" else op[1] if op[0] == "scale" else None
            if a is None:
                return None  # (heads are not steered)
            x = x.clone()
            for h in heads:
                s = slice(h * self.hd, (h + 1) * self.hd)
                x[:, s] = self.torch.where(live, x[:, s] * a, x[:, s])
            return (x, *args[1:])
        return hook

    def set_edit(self, units: list | None, op="zero") -> None:
        """The edit every following generate call runs under (None: the unedited model)."""
        if not units:
            self.edit = None
            return
        torch = self.torch
        if (op != "zero" and op[0] == "steer") and self.mx is None:
            raise SystemExit("steering needs --thr (each neuron's calibration maximum)")
        by: dict = {}
        heads: dict = {}
        for kind, layer, n, pole, at in (tuple(u) for u in units):
            if at is not None:
                raise ValueError("an edit acts at every position (no @position)")
            if kind == "H":
                heads.setdefault(layer, set()).add(n)
            else:
                d = by.setdefault(layer, {})
                d[n] = pole if d.get(n, pole) == pole else 0
        self.edit = {"N": {l_: (torch.tensor(sorted(d), device=self.dev), torch.tensor([d[k] for k in sorted(d)], device=self.dev), op)
                           for l_, d in by.items()},
                     "H": {l_: (sorted(h), op) for l_, h in heads.items()}}

    def prompt(self, messages: list[dict], thinking: bool = False) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    def sample(self, convs: list[list[dict]], n: int | list[int], temperature: float = 1.0, max_tokens: int = 1024,
               seed: int | None = None, thinking: bool | list[bool] = False) -> list[list[str | None]]:
        """n replies to each conversation (temperature 1, no top-p or top-k cut: WeirdChat's setting) under the current
        edit; n and thinking may differ per conversation (a thinking-on arm gets THINK_TOKENS more room, and its reply
        is the text after its reasoning, None when the reasoning never ended). No seed by default: each of a prompt's n
        replies is an independent draw."""
        from vllm import SamplingParams

        th = thinking if isinstance(thinking, list) else [thinking] * len(convs)
        ns = n if isinstance(n, list) else [n] * len(convs)
        texts = [self.prompt(c, t) for c, t in zip(convs, th)]
        sps = [SamplingParams(n=k, temperature=temperature, top_p=1.0, top_k=-1, max_tokens=max_tokens + (THINK_TOKENS if t else 0),
                              seed=None if seed is None else seed + j) for j, (k, t) in enumerate(zip(ns, th))]
        outs = self.llm.generate(texts, sps, use_tqdm=False)
        return [[self._answer(o.text, t) for o in r.outputs] for r, t in zip(outs, th)]

    @staticmethod
    def _answer(text: str, thinking: bool) -> str | None:
        """A reply without its thinking (a thinking-on arm's reasoning is not graded)."""
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return None if thinking else text.strip()


def smoke(a) -> None:
    import time

    e = Engine(a.model, a.thr)
    print("model object", type(e.model).__name__, "layers", e.L)
    conv = [[{"role": "user", "content": "Name one fruit. Answer with one word."}]]
    t0 = time.perf_counter()
    base = e.sample(conv, 1, temperature=0.0, max_tokens=8)  # (greedy: vLLM takes n = 1 only)
    units = [("N", l_, n, 0, None) for l_ in range(8, 28) for n in range(12288)]  # every neuron of layers 8-27
    e.set_edit(units)
    broken = e.sample(conv, 1, temperature=0.0, max_tokens=8)
    calls = e.calls
    e.set_edit(None)
    again = e.sample(conv, 1, temperature=0.0, max_tokens=8)
    print("unedited", base[0][:2], "| layers 8-27 zeroed", broken[0][:2], "| hooks off", again[0][:2],
          f"| edited hook calls {calls} | {time.perf_counter() - t0:.1f} s")
    assert again[0][0] == base[0][0], "hooks off must give the unedited reply"
    assert broken[0][0] != base[0][0], "the edit must reach the model"
    convs = [[{"role": "user", "content": f"Write a two-sentence story about the number {k}."}] for k in range(8)]
    t0 = time.perf_counter()
    out = e.sample(convs, 32, max_tokens=256)
    toks = sum(len(e.tok(x)["input_ids"]) for r in out for x in r)
    print(f"throughput {toks / (time.perf_counter() - t0):.0f} tokens/s over {len(convs) * 32} replies")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("smoke",))
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--thr")
    smoke(p.parse_args())
