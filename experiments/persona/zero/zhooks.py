"""Mean ablation of units inside Qwen3 (Hugging Face or vLLM), the one intervention of the Zero study.

A unit is ("N", layer, neuron), a value of the down projection's input act(gate) x up, or ("H", layer, head), the head's
slice of the output projection's input. Mean ablation sets the unit's values at every position from 1 on to its mean
(means.pt: {"N": [L, 12288], "H": [L, hidden]}, over the atlas sample at positions from 1 on); the first position stays
live (its values are the massive first-token values the means exclude).

Zero's chat format: its bundled chat_template.jinja (ChatML, no system prompt, no thinking block); a reply ends with
<|im_end|> (151645), and generation stops on it or on <|endoftext|> (151643). Sampling as the report's deployment:
temperature 0.7, top_p 1.0, presence penalty 1.5.
"""
from __future__ import annotations

ZERO = "movingcastles/zero"
STOP = [151645, 151643]
IM_END = 151645


def zero_template() -> str:
    from huggingface_hub import hf_hub_download
    return open(hf_hub_download(ZERO, "chat_template.jinja")).read()


def prompt_ids(tok, history: list, user: str, template: str | None) -> list[int]:
    """The generation prompt of a chat: Zero's own template when given (no system prompt), else the tokenizer's with
    thinking off (Qwen3-8B as the assistant-text writer)."""
    msgs = list(history or []) + [{"role": "user", "content": user}]
    if template is not None:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=template)
    else:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


def compile_edit(units: list, means: dict, hd: int, dev) -> dict:
    """{"N": {layer: (columns, values)}, "H": {layer: (columns, values)}}: the columns of the down projection's input
    (neurons) or of the output projection's input (a head's hd columns), and the means they take."""
    import torch
    cols: dict = {"N": {}, "H": {}}
    for kind, layer, i in (tuple(u) for u in units):
        c = [int(i)] if kind == "N" else list(range(int(i) * hd, (int(i) + 1) * hd))
        cols[kind].setdefault(int(layer), []).extend(c)
    out: dict = {"N": {}, "H": {}}
    for kind in ("N", "H"):
        for layer, c in cols[kind].items():
            idx = torch.tensor(sorted(set(c)), dtype=torch.long, device=dev)
            out[kind][layer] = (idx, means[kind][layer].to(dev)[idx])
    return out


def random_matched(units: list, n_neurons: int, n_heads: int, seed: int, exclude: set | None = None) -> list:
    """Units drawn at random from the same layers and kinds, as many per (kind, layer) as units has, none of exclude."""
    import random
    rng = random.Random(seed)
    ex = set(tuple(u) for u in (exclude if exclude is not None else units))
    by: dict = {}
    for kind, layer, _ in (tuple(u) for u in units):
        by[(kind, layer)] = by.get((kind, layer), 0) + 1
    out = []
    for (kind, layer), k in sorted(by.items()):
        pool = [i for i in range(n_neurons if kind == "N" else n_heads) if (kind, layer, i) not in ex]
        if len(pool) < k:      # (a layer whose heads the set nearly fills: draw from all of them)
            pool = list(range(n_neurons if kind == "N" else n_heads))
        out += [[kind, layer, i] for i in sorted(rng.sample(pool, k))]
    return out


class HFAblator:
    """Mean ablation in a Hugging Face Qwen3 ([batch, tokens, d] activations; position = token index). With capture on,
    each layer's (possibly ablated) down and output projection inputs are kept with their gradients."""

    def __init__(self, model, means: dict):
        import torch
        self.torch, self.model = torch, model
        cfg = model.config
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.dev = next(model.parameters()).device
        self.means = {k: v.to(self.dev) for k, v in means.items()}
        self.edit: dict = {"N": {}, "H": {}}
        self.capture = False
        self.saved: dict = {}
        for i, layer in enumerate(model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook("N", i))
            layer.self_attn.o_proj.register_forward_pre_hook(self._hook("H", i))
        model.model.embed_tokens.register_forward_hook(self._leaf)

    def set(self, units: list | None) -> None:
        self.edit = compile_edit(units or [], self.means, self.hd, self.dev)

    def _leaf(self, mod, args, out):
        return out.requires_grad_(True) if self.capture else None

    def _hook(self, kind: str, i: int):
        def hook(mod, args):
            x = args[0]
            e = self.edit[kind].get(i)
            if e is not None:
                idx, vals = e
                x = x.clone()
                x[:, 1:, idx] = vals.to(x.dtype)
            if self.capture:
                x.retain_grad()
                self.saved[(kind, i)] = x
            return (x, *args[1:]) if (e is not None or self.capture) else None
        return hook


class VLLMAblator:
    """Mean ablation inside vLLM's Qwen3 (flattened [tokens, d] activations; positions read from the model's forward).
    The LLM must run eagerly without torch.compile (hooks run as written) and without prefix caching (a cached prompt
    would carry another edit's values), as counterbench's Engine builds it."""

    def __init__(self, llm, means: dict):
        import torch
        self.torch = torch
        self.model = self._model(llm)
        cfg = self.model.config
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.dev = next(self.model.parameters()).device
        self.means = {k: v.to(self.dev) for k, v in means.items()}
        self.edit: dict = {"N": {}, "H": {}}
        self.positions = None
        self.calls = 0
        self.model.register_forward_pre_hook(self._pos, with_kwargs=True)
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook("N", i))
            layer.self_attn.o_proj.register_forward_pre_hook(self._hook("H", i))

    @staticmethod
    def _model(llm):
        try:
            return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError:
            pass
        core = llm.llm_engine.engine_core
        core = getattr(core, "engine_core", core)
        return core.model_executor.driver_worker.worker.model_runner.model

    def set(self, units: list | None) -> None:
        self.edit = compile_edit(units or [], self.means, self.hd, self.dev)

    def _pos(self, mod, args, kwargs):
        self.positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        return None

    def _hook(self, kind: str, i: int):
        def hook(mod, args):
            e = self.edit[kind].get(i)
            if e is None:
                return None
            x = args[0]
            p = self.positions
            if p is None or p.shape[0] != x.shape[0]:
                raise RuntimeError(f"ablation hook: positions {None if p is None else tuple(p.shape)} for {tuple(x.shape)}")
            self.calls += 1
            idx, vals = e
            x = x.clone()
            x[:, idx] = self.torch.where((p > 0)[:, None], vals.to(x.dtype)[None, :], x[:, idx])
            return (x, *args[1:])
        return hook
