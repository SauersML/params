"""The component explainer reading its subcomponents' vectors (#2951 graph oracle): the question as the Tinker arm reads
it, and after every subcomponent the question lists (VPD's active subcomponents per position) two placeholder tokens
whose input embeddings are that subcomponent's read vector v and write vector u through learned linear maps to the
explainer's width (one map per site kind and side: q, k, v read the normed residual and write a head space; o reads
the heads and writes the residual; fc reads the normed residual and writes the MLP's hidden units; down reads those and
writes the residual), each scaled to the RMS of the explainer's token embeddings, plus a learned embedding of
log ||u|| ||v|| (u v^T fixes only the product of the norms; standardized over VPD's subcomponents) on both. The
vectors are method/vpd/uv.py's: unit directions oriented so the activation is positive where the subcomponent
typically acts (the atlas's sign), with |u||v| apart. The answer
names subcomponents by the question's ids as in the text-only arm; the vectors are a second channel, from which
directions can be read. Tinker takes tokens only, so this arm runs on our own GPUs: transformers + PEFT LoRA on the
base, the maps trained with it, sampling with inputs_embeds (transformers' generate).

  inject.py sft SFT.jsonl ROWS.jsonl --out DIR [--base Qwen/Qwen3-8B] [--steps N] [--batch B] [--lr LR] [--rank 32]
  inject.py sample ROWS.jsonl OUT.jsonl --init DIR [--n 20] [--samples 2] [--max-tokens 4096] [--shuffle]
                    [--shuffled-out OUT2.jsonl] [--vllm]

SFT.jsonl: components.py dataset rows (the prompt's question and the target answer); ROWS.jsonl: the question rows
(evidence(): the subcomponents each question lists). sample writes tinker_train.py sample's format (components.py
evaluate reads it). --shuffle gives each question's subcomponents one another's vectors (a derangement within the
question): the ablation that shows whether the explainer reads them; --shuffled-out writes that set beside OUT in the
same run. --vllm samples with vLLM from the prompts' embeddings (computed here; vLLM on the base takes the adapter's
merged linear maps in place: train.py's VllmSampler found vLLM's own LoRA with prompt embeddings crashing, and runs
them without prefix caching or asynchronous scheduling), else transformers' generate. Vectors: native.UV, the tensors
native runs on, through method/vpd/uv.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SLOT = "<|fim_pad|>"  # the placeholder whose input embedding a vector replaces (escape() keeps it out of every question's text)
SIDES = ("read", "write")


def slotted(question: str, ev: dict) -> tuple[str, list[tuple]]:
    """The question with two SLOTs after every subcomponent its listing names (VPD's active subcomponents at each
    position, ev["components"]; the first place each is written with its causal importance), and those subcomponents
    in the order their slots appear."""
    import components

    found = []
    for q in sorted(ev["components"], key=int):
        for nd, c, _ in ev["components"][q]:
            name = components.node_text(tuple(nd))
            at = question.find(f"{name} ({c:.2f}")
            if at < 0:
                continue
            cut = at + len(name)
            question = question[:cut] + " " + SLOT + SLOT + question[cut:]
            found.append((at, tuple(nd)))
    return question, [nd for _, nd in sorted(found)]


class Vectors:
    """VPD's subcomponents' vectors (method/vpd/uv.py over native.UV: unit read v [d_in] and write u [d_out] directions,
    oriented by the atlas, and |u||v|), and the mean and spread of log |u||v| over every subcomponent."""

    def __init__(self, path: Path | None = None, atlas: Path | None = None, oriented: bool = True):
        sys.path.insert(0, str(HERE.parents[2]))
        from method.vpd import uv

        atlas = (atlas or uv.ATLAS) if oriented else None
        if atlas is not None and not (Path(atlas) / "atlas.safetensors").exists():  # uv.py would orient by another rule
            raise FileNotFoundError(f"{atlas}/atlas.safetensors: the vectors' orientation (the atlas's sign) is not there")
        self.uv = uv.UV(path or uv.PATH, atlas=atlas)
        t = self.uv.t
        logn = torch.cat([(t[f"{s}.U"].float().norm(dim=1) * t[f"{s}.V"].float().norm(dim=0)).log() for s in self.uv.names()])
        self.logn_mean, self.logn_std = float(logn.mean()), float(logn.std())

    def __call__(self, nd: tuple) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(v, u, |u||v|) of subcomponent nd = (site, position, index)."""
        g = self.uv.get(nd[0], nd[2])
        return g["v"].float(), g["u"].float(), float(g["scale"])

    def dims(self) -> dict:
        """Site kind -> (read width, write width)."""
        out = {}
        for k, t in self.uv.t.items():
            site, side = k.rsplit(".", 1)
            kind = site.split(".")[-1]
            d = out.setdefault(kind, [0, 0])
            if side == "V":
                d[0] = t.shape[0]
            else:
                d[1] = t.shape[1]
        return {k: tuple(v) for k, v in out.items()}


class Injector(torch.nn.Module):
    """A subcomponent's two slot embeddings: per site kind and side a linear map of its vector to the width, the output's
    direction at the token embeddings' RMS, plus a learned embedding of its standardized log ||u|| ||v||."""

    def __init__(self, dims: dict, width: int, rms: float, logn_mean: float, logn_std: float):
        super().__init__()
        self.maps = torch.nn.ModuleDict({f"{k}_{side}": torch.nn.Linear(dims[k][i], width) for k in dims for i, side in enumerate(SIDES)})
        for m in self.maps.values():  # a slot starts as its vector's mapped direction alone (a bias would be one shared direction)
            torch.nn.init.zeros_(m.bias)
        self.magnitude = torch.nn.Linear(1, width)
        torch.nn.init.zeros_(self.magnitude.weight)
        torch.nn.init.zeros_(self.magnitude.bias)
        self.rms, self.logn_mean, self.logn_std = rms, logn_mean, logn_std

    def forward(self, items: list[tuple[str, torch.Tensor, torch.Tensor, float]]) -> torch.Tensor:
        """[2 N, width] for N subcomponents (site kind, v, u, |u||v|): each one's read slot, then its write slot."""
        dev = self.magnitude.weight.device
        out = []
        for kind, v, u, scale in items:
            z = torch.tensor((math.log(max(scale, 1e-30)) - self.logn_mean) / self.logn_std)
            mag = self.magnitude(z.reshape(1, 1).to(dev))[0]
            for side, x in zip(SIDES, (v, u)):
                m = self.maps[f"{kind}_{side}"](x.to(dev))
                out.append(self.rms * m / m.pow(2).mean().sqrt().clamp_min(1e-6) + mag)
        return torch.stack(out) if out else torch.zeros(0, self.magnitude.out_features, device=dev)


class Explainer:
    """The base with a LoRA and the Injector; prompts carry their subcomponents (nodes) for the slots."""

    def __init__(self, base: str, rank: int, dev: torch.device, vectors: Vectors, init: Path | None = None):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev, self.vectors = dev, vectors
        self.tok = AutoTokenizer.from_pretrained(base)
        dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev)
        emb = model.get_input_embeddings().weight.detach().float()
        rms = float(emb.pow(2).mean(-1).sqrt().mean())
        self.injector = Injector(vectors.dims(), model.config.hidden_size, rms, vectors.logn_mean, vectors.logn_std).to(dev)
        if init:
            self.model = PeftModel.from_pretrained(model, str(init), is_trainable=True)
            self.injector.load_state_dict(torch.load(Path(init) / "injector.pt", map_location=dev))
        else:
            self.model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0,
                                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.rank = rank
        self.slot = self.tok.convert_tokens_to_ids(SLOT)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prompt_ids(self, question: str) -> list[int]:
        chat = self.tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                            enable_thinking=False, tokenize=False)
        return self.tok.encode(chat, add_special_tokens=False)

    def embed(self, ids: torch.Tensor, nodes: list[list[tuple]]) -> torch.Tensor:
        """Input embeddings of a batch [B, W] with each row's SLOT pairs replaced by its subcomponents' (in order)."""
        emb = self.model.get_input_embeddings()(ids)
        for r, nds in enumerate(nodes):
            pos = (ids[r] == self.slot).nonzero().flatten()
            if not len(pos):
                continue
            items = []
            for nd in nds:
                items.append((nd[0].split(".")[-1], *self.vectors(nd)))
            vec = self.injector(items)
            assert len(vec) == len(pos), (len(vec), len(pos))
            emb = emb.index_put((torch.full_like(pos, r), pos), vec.to(emb.dtype))
        return emb

    def param_groups(self, lr: float) -> list[dict]:
        """AdamW groups: the LoRA at lr, each map at lr x rank / fan_in (train.py's Policy.param_groups: Adam moves every
        entry by about lr, so a map's output moves by about lr x fan_in per unit input; the scale gives the maps the
        LoRA up-projections' per-step change)."""
        groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr}]
        for m in self.injector.modules():
            if isinstance(m, torch.nn.Linear):
                groups.append({"params": list(m.parameters()), "lr": lr * self.rank / m.in_features})
        return groups

    def nll(self, prompt: list[int], completion: list[int], nodes: list[tuple], chunk: int = 1024) -> tuple[torch.Tensor, int]:
        """Summed -log p of the completion's tokens given the prompt (one sequence; the output layer only at the
        completion, in checkpointed chunks) and their count."""
        from torch.utils.checkpoint import checkpoint

        ids = torch.tensor([prompt + completion], device=self.dev)
        inner = self.model.get_base_model()
        hidden = inner.model(inputs_embeds=self.embed(ids, [nodes])).last_hidden_state[0]
        start = len(prompt) - 1
        h, target = hidden[start:-1], ids[0, start + 1:]
        total = torch.zeros((), device=self.dev)
        for a in range(0, len(target), chunk):
            def piece(x, t):
                return torch.nn.functional.cross_entropy(inner.lm_head(x).float(), t, reduction="sum")
            total = total + checkpoint(piece, h[a:a + chunk], target[a:a + chunk], use_reentrant=False)
        return total, len(target)

    def save(self, out: Path):
        self.model.save_pretrained(str(out))
        torch.save(self.injector.state_dict(), out / "injector.pt")

    @torch.no_grad()
    def sample(self, prompts: list[list[int]], nodes: list[list[tuple]], n: int, max_tokens: int, batch: int = 8) -> list[list[tuple[list[int], bool]]]:
        """n samples of every prompt (temperature 1), up to `batch` sequences per generate call (left-padded): per
        prompt [(tokens, cut at max_tokens)]."""
        self.model.eval()
        flat = [(p, nd) for p, nd in zip(prompts, nodes) for _ in range(n)]
        out = []
        for s in range(0, len(flat), batch):
            chunk = flat[s:s + batch]
            width = max(len(p) for p, _ in chunk)
            ids = torch.full((len(chunk), width), self.end, dtype=torch.long, device=self.dev)
            att = torch.zeros(len(chunk), width, dtype=torch.long, device=self.dev)
            for r, (p, _) in enumerate(chunk):
                ids[r, width - len(p):] = torch.tensor(p, device=self.dev)
                att[r, width - len(p):] = 1
            gen = self.model.generate(inputs_embeds=self.embed(ids, [nd for _, nd in chunk]), attention_mask=att, max_new_tokens=max_tokens,
                                      do_sample=True, temperature=1.0, top_p=1.0, top_k=0, eos_token_id=self.end, pad_token_id=self.end).tolist()
            out += [(g[:g.index(self.end)] if self.end in g else g, self.end not in g) for g in gen]
        return [out[k * n:(k + 1) * n] for k in range(len(prompts))]


def derange(nodes: list[tuple], rng: random.Random) -> list[tuple]:
    """Permute weight identities within each kind; positions cannot make identical vectors distinct.

    Permute unique owners, using one donor for every position of the same weight.
    Preserve receiving positions; a kind with only one owner cannot supply this control.
    """
    kinds = {}
    for site, _, component in nodes:
        kinds.setdefault(site.split(".")[-1], set()).add((site, component))
    assigned = {}
    for kind, owners in kinds.items():
        order = sorted(owners)
        if len(order) < 2:
            raise ValueError(f"cannot derange distinct weight components of kind {kind}")
        rng.shuffle(order)
        assigned.update(zip(order, order[1:] + order[:1]))
    return [(assigned[(site, component)][0], position, assigned[(site, component)][1])
            for site, position, component in nodes]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def sft(a) -> None:
    """Supervised steps on the SFT rows (those shorter than --max-length tokens): each step a batch drawn as
    tinker_train.py sft draws one (random sample per step), one sequence at a time with the gradient accumulated, the
    loss the batch's mean -log p per target token."""
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    data = []
    for line in a.sft.read_text().splitlines():
        e = json.loads(line)
        task = e["info"]["task"]
        if task not in rows:
            continue
        question, nodes = slotted(e["prompt"][-1]["content"], rows[task])
        data.append((question, e["completion"][-1]["content"], nodes))
    dev = device()
    ex = Explainer(a.base, a.rank, dev, Vectors(), a.init)
    data = [x for x in data if len(ex.prompt_ids(x[0])) + len(ex.tok.encode(x[1], add_special_tokens=False)) < a.max_length]
    ex.model.get_base_model().gradient_checkpointing_enable()
    ex.model.train()
    opt = torch.optim.AdamW(ex.param_groups(a.lr), weight_decay=0.0)
    lrs = [g["lr"] for g in opt.param_groups]
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / "sft.jsonl").open("a")
    for step in range(a.steps):
        t0 = time.time()
        for g, lr in zip(opt.param_groups, lrs):
            g["lr"] = lr * (1 - step / a.steps)  # linear decay to 0, as tinker_train.py sft
        batch = rng.sample(data, min(a.batch, len(data)))
        enc = [(ex.prompt_ids(q), ex.tok.encode(ans, add_special_tokens=False) + [ex.end], nd) for q, ans, nd in batch]
        count = sum(len(c) for _, c, _ in enc)
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for p, c, nd in enc:
            loss, _ = ex.nll(p, c, nd)
            (loss / count).backward()
            total += float(loss)
        opt.step()
        row = {"step": step, "nll": total / count, "tokens": sum(len(p) + len(c) for p, c, _ in enc), "seconds": time.time() - t0}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
    ex.save(a.out)
    print(json.dumps({"saved": str(a.out)}), flush=True)


def vllm_sample(ex: Explainer, base: str, embeds: list[torch.Tensor], n: int, max_tokens: int, seed: int) -> list[list[tuple[list[int], bool]]]:
    """n samples of every prompt given as embeddings [L, width], by vLLM on the base with the explainer's adapter merged
    into its linear maps in place (train.py's VllmSampler.push_weights, through vLLM's own loader, in-process; the
    merged maps go through host memory, so the explainer leaves the GPU first and nothing is written to disk)."""
    import gc
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # apply_model runs here, on the engine's model
    from peft.tuners.lora import LoraLayer
    from vllm import LLM, SamplingParams

    end = ex.end
    merged = []
    with torch.no_grad():
        for name, m in ex.model.named_modules():
            if isinstance(m, LoraLayer) and "default" in m.lora_A:
                w = m.base_layer.weight
                merged.append((name.removeprefix("base_model.model.") + ".weight", (w.float() + m.get_delta_weight("default").float()).to(w.dtype).cpu()))
    ex.model = ex.injector = None  # the caller's explainer too: its model leaves the GPU before vLLM starts
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(model=base, dtype="bfloat16", enable_prompt_embeds=True, enable_prefix_caching=False, async_scheduling=False,
              max_model_len=max(int(e.shape[0]) for e in embeds) + max_tokens + 16, gpu_memory_utilization=0.85, seed=seed)
    loaded = llm.apply_model(lambda model: len(model.load_weights(iter(merged))))[0]
    if not loaded:
        raise RuntimeError(f"vLLM loaded none of the adapter's {len(merged)} merged weights")
    params = SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, stop_token_ids=[end])
    outs = llm.generate([{"prompt_embeds": e} for e in embeds], params, use_tqdm=False)
    return [[([t for t in c.token_ids if t != end], c.finish_reason == "length") for c in o.outputs] for o in outs]


def sample(a) -> None:
    """Answers to the first n questions, --samples each, as tinker_train.py sample writes them (and with
    --shuffled-out, the same questions with their subcomponents' vectors deranged)."""
    rows = [r for r in map(json.loads, a.rows.read_text().splitlines()) if r][:a.n]
    dev = device()
    ex = Explainer(a.base, 0, dev, Vectors(), a.init)
    rng = random.Random(a.seed)
    prompts, nodes, shuffled = [], [], []
    for r in rows:
        question, nds = slotted(r["question"], r)
        prompts.append(ex.prompt_ids(question))
        nodes.append(derange(nds, rng) if a.shuffle else nds)
        if a.shuffled_out:
            shuffled.append(derange(nds, rng))
    sets = [(a.out, nodes)] + ([(a.shuffled_out, shuffled)] if a.shuffled_out else [])
    t0 = time.time()
    tok = ex.tok
    if a.vllm:
        with torch.no_grad():
            embeds = [ex.embed(torch.tensor([p], device=dev), [nd])[0].to(torch.bfloat16).cpu() for _, nds in sets for p, nd in zip(prompts, nds)]
        flat = vllm_sample(ex, a.base, embeds, a.samples, a.max_tokens, a.seed)
        got = [flat[k * len(prompts):(k + 1) * len(prompts)] for k in range(len(sets))]
    else:
        got = [ex.sample(prompts, nds, a.samples, a.max_tokens, a.batch) for _, nds in sets]
    for (path, _), answers_of in zip(sets, got):
        with path.open("w") as f:
            for r, answers in zip(rows, answers_of):
                for k, (tokens, cut) in enumerate(answers):
                    f.write(json.dumps({"task": r["task"], "sample": k, "text": tok.decode(tokens), "finish": "length" if cut else "stop",
                                        "tokens": len(tokens)}) + "\n")
    print(json.dumps({"questions": len(rows), "samples": a.samples, "sets": len(sets), "vllm": a.vllm, "seconds": time.time() - t0}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft")
    s.add_argument("sft", type=Path)
    s.add_argument("rows", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--init", type=Path)
    s.add_argument("--steps", type=int, default=100)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--max-length", type=int, default=16384)
    m = sub.add_parser("sample")
    m.add_argument("rows", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--init", type=Path, required=True)
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--samples", type=int, default=2)
    m.add_argument("--max-tokens", type=int, default=4096)
    m.add_argument("--batch", type=int, default=8)
    m.add_argument("--shuffle", action="store_true")
    m.add_argument("--shuffled-out", type=Path, help="also write the answers with each question's vectors deranged here")
    m.add_argument("--vllm", action="store_true", help="sample with vLLM from prompt embeddings (CUDA)")
    for p in (s, m):
        p.add_argument("--base", default="Qwen/Qwen3-8B")
        p.add_argument("--rank", type=int, default=32)
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"sft": sft, "sample": sample}[a.cmd](a)


if __name__ == "__main__":
    main()
