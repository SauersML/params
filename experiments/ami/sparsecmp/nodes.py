"""Node-masked forward passes under the circuit-sparsity criterion (Gao et al. 2511.13653, section 2.2 and A.5).

A node is one channel at one site: a residual read after RMSNorm (attn.act_in, mlp.act_in), a q, k or v channel,
an MLP neuron (mlp.post_act), a residual write (attn.resid_delta, mlp.resid_delta) or the final residual
(final_resid). A mask m in [0, 1] per node, shared by every prompt and position, gives

    x  ->  mu + m * (x - mu)

where mu is the node's mean over the pretraining distribution (one number per node, position independent).

GaoModel wraps their GPT (github.com/openai/circuit_sparsity, Apache-2.0, pinned dbf1fe0) through its own hook
interventions, so the node names are theirs.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.expanduser("~/mpd-data/ami/sparsecmp")
sys.path.insert(0, f"{ROOT}/ext/circuit_sparsity")

LOCS = ["attn.act_in", "attn.q", "attn.k", "attn.v", "attn.resid_delta", "mlp.act_in", "mlp.post_act",
        "mlp.resid_delta"]


def tinypython():
    import tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    d = f"{ROOT}/released"
    ranks = data_gym_to_mergeable_bpe_ranks(vocab_bpe_file=f"{d}/vocab.bpe", encoder_json_file=f"{d}/encoder.json")
    return tiktoken.Encoding(
        name="tinypython_2k",
        pat_str=r"""[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks, special_tokens={"<|endoftext|>": 2047})


class GaoModel:
    """One of their models (sparse or dense) with node masks. ctx: prompts are left-truncated to 256 tokens (their
    n_ctx; the released task pairs run to 261)."""

    ctx = 256

    def __init__(self, name: str, device: str = "cpu", means: str | None = None):
        from circuit_sparsity.inference.gpt import GPT, GPTConfig
        from circuit_sparsity.inference import hook_utils
        self.hooks = hook_utils
        cfg = json.load(open(f"{ROOT}/models/{name}/beeg_config.json"))
        if "n_mlp" in cfg:
            cfg["d_mlp"] = cfg.pop("n_mlp")
        cfg["flash"] = True                      # their loader: the attention sink needs the SDPA path
        cfg.setdefault("sink", False)
        cfg["grad_checkpointing"] = False
        keep = GPTConfig.__dataclass_fields__.keys()
        self.config = GPTConfig(**{k: v for k, v in cfg.items() if k in keep})
        self.model = GPT(self.config)
        sd = torch.load(f"{ROOT}/models/{name}/final_model.pt", weights_only=True, map_location="cpu")
        sd.setdefault("final_logits_bias", torch.zeros(self.config.vocab_size))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.name, self.device = name, device
        self.enc = tinypython()
        c = self.config
        dq = c.n_head * c.d_head
        size = {"attn.act_in": c.d_model, "attn.q": dq, "attn.k": dq, "attn.v": dq, "attn.resid_delta": c.d_model,
                "mlp.act_in": c.d_model, "mlp.post_act": c.d_mlp, "mlp.resid_delta": c.d_model}
        self.sites = {f"{i}.{loc}": size[loc] for i in range(c.n_layer) for loc in LOCS}
        self.sites["final_resid"] = c.d_model
        self.means = None
        if means:
            mu = torch.load(means, weights_only=True, map_location="cpu")
            self.means = {k: mu[k].float().to(device) for k in self.sites}
            # residual reads include the concatenated position channels (d_pos_emb) on the models that have them
            self.sites = {k: int(v.numel()) for k, v in self.means.items()}

    def tokenize(self, text: str) -> list:
        return self.enc.encode(text, allowed_special="all")

    def logits(self, toks: torch.Tensor, last: torch.Tensor, masks: dict | None = None,
               record: list | None = None) -> torch.Tensor:
        """Last-position logits [B, V] for right-padded tokens [B, T]; masks {site: [n]} (absent site = kept)."""
        iv = {}
        if masks:
            for site, m in masks.items():
                mu = self.means[site]
                iv[site] = (lambda x, m=m, mu=mu: mu + m * (x - mu))
        regex = "|".join(record) if record else "$^"
        with self.hooks.hook_recorder(regex=regex, interventions=iv) as rec:
            out, _, _ = self.model(toks)
        lg = out[torch.arange(toks.shape[0], device=toks.device), last]
        if record is not None:
            self.recorded = rec
        return lg


def pad(rows: list, device: str, value: int = 0, T: int | None = None):
    """Right-padded [B, T] and each row's last index. A fixed T keeps one shape per batch size (the MPS allocator
    otherwise holds a buffer for every length it has seen)."""
    T = T or -(-max(len(r) for r in rows) // 64) * 64
    x = torch.full((len(rows), T), value, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r)
    last = torch.tensor([len(r) - 1 for r in rows])
    return x.to(device), last.to(device)


def answer_token(model, prompt_ids: list, prompt: str, ans: str):
    """The first token of `ans` after the prompt, if tokenizing prompt + ans keeps the prompt's tokens."""
    full = model.tokenize(prompt + ans)
    n = len(prompt_ids)
    if full[:n] != prompt_ids or len(full) <= n:
        return None
    return full[n]


def encode_items(model, items: list):
    """Items -> per-item (pos_ids, neg_ids, tok, alt, kind); drops items whose answers do not tokenize apart."""
    out = []
    for it in items:
        p, q = model.tokenize(it["pos"]), model.tokenize(it["neg"])
        tok = answer_token(model, p, it["pos"], it["pos_ans"])
        alt = answer_token(model, q, it["neg"], it["neg_ans"])
        if it["kind"] == "pos_vs_rest":
            alt = tok
        if tok is None or alt is None or (it["kind"] == "binary" and tok == alt):
            continue
        out.append({"pos": p[-model.ctx:], "neg": q[-model.ctx:], "tok": tok, "alt": alt, "kind": it["kind"]})
    return out


def task_loss(lp: torch.Tensor, ln: torch.Tensor, tok: torch.Tensor, alt: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-pair loss [B] = CE(pos) + CE(neg), the scale their stored losses and targets use (viz.py: 2x the
    per-sequence loss). binary: softmax over {tok, alt}; pos_vs_rest: pos must give tok, neg anything else."""
    r = torch.arange(lp.shape[0], device=lp.device)
    if kind == "binary":
        two_p = torch.stack([lp[r, tok], lp[r, alt]], 1).float().log_softmax(1)
        two_n = torch.stack([ln[r, tok], ln[r, alt]], 1).float().log_softmax(1)
        return -(two_p[:, 0] + two_n[:, 1])
    a = lp.float().log_softmax(1)[r, tok]
    b = ln.float().log_softmax(1)[r, tok]
    return -(a + torch.log1p(-b.exp().clamp(max=1 - 1e-6)))


class QwenModel:
    """A Qwen3 model (HF) with the same node sites. q and k are taken after q_norm/k_norm and before RoPE (a
    position-independent mean of a rotated channel would mix positions); v after v_proj; mlp.post_act is
    act(gate) * up, the input of down_proj; final_resid is the residual before the final norm (their final_resid
    is also taken before ln_f)."""

    def __init__(self, name: str, device: str = "cpu", means: str = "", dtype=torch.float32, ctx: int = 512):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.name, self.device, self.ctx = name, device, ctx
        c = self.model.config
        hd = c.head_dim
        size = {"attn.act_in": c.hidden_size, "attn.q": c.num_attention_heads * hd, "attn.k": c.num_key_value_heads * hd,
                "attn.v": c.num_key_value_heads * hd, "attn.resid_delta": c.hidden_size, "mlp.act_in": c.hidden_size,
                "mlp.post_act": c.intermediate_size, "mlp.resid_delta": c.hidden_size}
        self.sites = {f"{i}.{loc}": size[loc] for i in range(c.num_hidden_layers) for loc in LOCS}
        self.sites["final_resid"] = c.hidden_size
        self.means = None
        if means:
            mu = torch.load(means, weights_only=True, map_location="cpu")
            self.means = {k: mu[k].float().to(device) for k in self.sites}
        self._masks, self._rec = None, None
        self._install()

    def _apply(self, site, x):
        """x: [..., n] or [B, T, H, hd] (q/k): mask and/or record the site."""
        shape = x.shape
        flat = x.reshape(*shape[:2], -1) if x.dim() == 4 else x
        if self._rec is not None and site in self._rec:
            self._rec[site].append(flat)
        if self._masks is not None and site in self._masks:
            m, mu = self._masks[site], self.means[site]
            flat = mu.to(flat.dtype) + m.to(flat.dtype) * (flat - mu.to(flat.dtype))
            return flat.reshape(shape)
        return x

    def _install(self):
        layers = self.model.model.layers
        for i, L in enumerate(layers):
            A, M = L.self_attn, L.mlp
            post = [(L.input_layernorm, f"{i}.attn.act_in"), (A.q_norm, f"{i}.attn.q"), (A.k_norm, f"{i}.attn.k"),
                    (A.v_proj, f"{i}.attn.v"), (A.o_proj, f"{i}.attn.resid_delta"),
                    (L.post_attention_layernorm, f"{i}.mlp.act_in"), (M.down_proj, f"{i}.mlp.resid_delta")]
            for mod, site in post:
                mod.register_forward_hook(lambda mod, inp, out, site=site: self._apply(site, out))
            M.down_proj.register_forward_pre_hook(
                lambda mod, inp, site=f"{i}.mlp.post_act": (self._apply(site, inp[0]),))
        self.model.model.norm.register_forward_pre_hook(lambda mod, inp: (self._apply("final_resid", inp[0]),))

    def tokenize(self, text: str) -> list:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def logits(self, toks: torch.Tensor, last: torch.Tensor, masks: dict | None = None,
               record: list | None = None) -> torch.Tensor:
        self._masks = masks
        self._rec = {s: [] for s in record} if record else None
        try:
            att = (torch.arange(toks.shape[1], device=toks.device)[None] <= last[:, None]).long()
            h = self.model.model(input_ids=toks, attention_mask=att).last_hidden_state
            r = torch.arange(toks.shape[0], device=toks.device)
            lg = self.model.lm_head(h[r, last])
        finally:
            self._masks = None
        if record:
            self.recorded = self._rec
        self._rec = None
        return lg


@torch.no_grad()
def accumulate_means(model, rows: list, bs: int = 8) -> dict:
    """Mean of every node over every token of `rows` (lists of token ids), one value per node."""
    sums = {s: torch.zeros(n, dtype=torch.float64) for s, n in model.sites.items()}
    count = 0
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        x, last = pad(b, model.device)
        valid = (torch.arange(x.shape[1], device=model.device)[None] <= last[:, None])
        model.logits(x, last, None, record=list(model.sites))
        for s, chunks in model.recorded.items():
            v = chunks[0] if isinstance(chunks, list) else chunks
            sums[s] += v[valid].double().sum(0).cpu()
        count += int(valid.sum())
    return {s: (v / count).float() for s, v in sums.items()}
