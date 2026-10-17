"""The language models the oracle pipeline calls: a frozen open model on Tinker (Reader), the same served by vLLM in this
process (VLLMReader) or behind an OpenAI-compatible server (OpenAIReader), all greedy with thinking off and called
the same way, reader(prompts) -> texts; plus Tinker's rate card and per-call spend log (Spend), its client
(service_client), chat renderers (renderer_for) and the answer text of sampled tokens (answer_of), which
tinker_train.py's SFT uses too. The project comes from TINKER_PROJECT_ID (required: there is no default) and the key
from TINKER_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

PRICES = {"Qwen/Qwen3-8B": (0.195, 0.60, 0.44), "Qwen/Qwen3.5-4B": (0.33, 1.005, 0.737),  # $/M tokens: prefill, sample, train
          "Qwen/Qwen3.8-27B": (0.6975, 5.595, 4.103), "Qwen/Qwen3.6-27B": (1.86, 5.595, 4.103),  # (Tinker's models page, 64K rows)
          "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (0.184, 0.495, 0.44),  # at the limited-time 50% discount
          "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335, 1.177),
          "moonshotai/Kimi-K2.6": (2.205, 5.49, 4.84), "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50, 6.60),
          "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": (0.60, 3.60, 4.98),
          "thinkingmachines/Inkling": (1.87, 4.68, 5.61)}  # Inkling at its limited-time 50% discount
RENDERERS = {"deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": "deepseekv3_disable_thinking"}  # bases the cookbook does not list
CACHED = 0.2  # a cached prefill token's share of the price
ADDED = re.compile(r"<(?=\||/?(?:think|tool_call|tool_response|tts_[a-z_]+)>)")  # Qwen3-family added tokens: <|...|>, <think>, <tool_call>, <tts_pad>, ...


def escape(text: str) -> str:
    """Text with every added-token string of Qwen3-family tokenizers broken by a backslash after its "<"
    ("<|endoftext|>" -> "<\\|endoftext|>", "<think>" -> "<\\think>"), so a prompt quoting such strings reads them as
    text, never as the reader's own special tokens; idempotent."""
    return ADDED.sub(r"<\\", text)


class Spend:
    """Tokens and dollars per Tinker call, appended to DIR/spend.jsonl."""

    def __init__(self, log: Path, base: str):
        if base not in PRICES:
            raise SystemExit(f"no prices for {base}: add Tinker's rate card to PRICES")
        self.path, self.prices, self.total = log / "spend.jsonl", PRICES[base], 0.0

    def add(self, kind: str, **tokens) -> float:
        pre, smp, trn = self.prices
        usd = (tokens.get("prefill", 0) * pre + tokens.get("cached", 0) * pre * CACHED + tokens.get("sampled", 0) * smp
               + tokens.get("trained", 0) * trn) / 1e6
        self.total += usd
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "kind": kind, **tokens, "usd": usd, "total_usd": self.total}) + "\n")
        return usd


def service_client():
    import tinker

    project = os.environ.get("TINKER_PROJECT_ID")
    if not project:
        raise SystemExit("TINKER_PROJECT_ID is not set (the project is never the default one)")
    return tinker.ServiceClient(project_id=project)


def renderer_for(base: str, thinking: bool = False):
    """(renderer, tokenizer) for a base model, thinking disabled where the model thinks by default (thinking: its
    thinking renderer)."""
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(base.split(":")[0])  # a context variant (":peft:262144") shares its base's tokenizer
    if base in RENDERERS:
        name = RENDERERS[base]
    else:
        names = model_info.get_recommended_renderer_names(base)
        name = next((n for n in names if n.endswith("_disable_thinking") != thinking), names[0])
    return renderers.get_renderer(name, tokenizer), tokenizer


def answer_of(renderer, tokenizer, tokens: list[int]) -> tuple[str, int]:
    """A sampled answer's text without its thinking, and the thinking's length in tokens (0 without)."""
    from tinker_cookbook import renderers

    msg = renderer.parse_response(tokens)[0]
    c = msg["content"]
    think = "".join(x.get("thinking", "") for x in c if x.get("type") == "thinking") if isinstance(c, list) else ""
    return renderers.get_text_content(msg), (len(tokenizer.encode(think, add_special_tokens=False)) if think else 0)


class Reader:
    """A frozen open model on Tinker: greedy answers, tokens and dollars counted (Spend)."""

    def __init__(self, base: str, log: Path):
        import tinker

        self.tinker, self.base = tinker, base
        service = service_client()
        self.renderer, self.tokenizer = renderer_for(base)
        self.sampler = service.create_sampling_client(base_model=base)
        log.mkdir(parents=True, exist_ok=True)
        self.spend = Spend(log, base)

    def __call__(self, prompts: list[str], max_tokens: int = 1500, temperature: float = 0.0, seed: int | None = None) -> list[str]:
        params = self.tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed, stop=self.renderer.get_stop_sequences())
        futs = []
        for text in prompts:
            prompt = self.renderer.build_generation_prompt([{"role": "user", "content": escape(text)}])
            futs.append((prompt.length, self.sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)))
        out = []
        for n, fut in futs:
            toks = fut.result().sequences[0].tokens
            text, _ = answer_of(self.renderer, self.tokenizer, toks)
            self.spend.add("reader", prefill=n, sampled=len(toks))
            out.append(text)
        return out

    def tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class VLLMReader:
    """The reader self-hosted (the lead's 10-14: Tinker's $6 per 1,000 rewards is too much for RL): vLLM offline
    batches, greedy, the chat template with thinking off. Same calls as Reader."""

    def __init__(self, model: str, max_len: int = 6144, util: float = 0.85, seqs: int = 64):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(model)
        # a 48 GB card: the FP8 27B's weights take 27.9 GiB, and at utilization 0.92 the CUDA-graph profiling ran out
        self.llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=util, max_num_seqs=seqs,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        self.spend = type("S", (), {"total": 0.0})()

    def __call__(self, prompts: list[str], max_tokens: int = 1500, temperature: float = 0.0, seed: int | None = None) -> list[str]:
        from vllm import SamplingParams

        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False) for p in prompts]
        outs = self.llm.generate(texts, SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens))
        return [o.outputs[0].text for o in outs]

    def tokens(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])


class OpenAIReader:
    """The self-hosted reader behind an OpenAI-compatible server (vllm serve Qwen/Qwen3.6-27B-FP8 --max-model-len 6144
    --gpu-memory-utilization 0.85 --max-num-seqs 64): greedy chat completions, thinking off, requests in parallel."""

    def __init__(self, url: str, model: str = "Qwen/Qwen3.6-27B-FP8", threads: int = 64):
        self.url, self.model, self.threads = url.rstrip("/"), model, threads
        self.spend = type("S", (), {"total": 0.0})()

    def _one(self, prompt: str, max_tokens: int, temperature: float = 0.0, seed: int | None = None) -> str:
        import urllib.request

        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                           "seed": seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""

    def __call__(self, prompts: list[str], max_tokens: int = 1500, temperature: float = 0.0, seed: int | None = None) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(self.threads, max(1, len(prompts)))) as ex:
            return list(ex.map(lambda q: self._one(q, max_tokens, temperature, seed), prompts))

    def tokens(self, text: str) -> int:
        return max(1, len(text) // 4)  # characters / 4, without the server's tokenizer
