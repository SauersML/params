"""Thin loader for Goodfire's 4L Pile LlamaSimpleMLP target (goodfire/spd/runs/t-9d2b8f02) and
the VPD paper decomposition of it (goodfire/spd/runs/s-55ea3f9b, model_400000.pth).

Everything here mirrors the torch code at the `vpd-paper` tag of github.com/goodfire-ai/spd
(param_decomp/pretrain/models/llama_simple_mlp.py, param_decomp/models/components.py,
param_decomp/metrics/{ce_and_kl_losses,pgd_utils}.py) and nano_param_decomp/run.py.

Sites: h.{i}.attn.{q,k,v,o}_proj, h.{i}.mlp.{c_fc,down_proj}. A site's weight is replaced by
    y = ((x @ V) * mask) @ U + delta_mask * (x @ (W - (V @ U).T).T)
with mask in [0,1]^C per (batch, pos), delta_mask in [0,1] per (batch, pos).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn

HERE = Path.home() / "mpd-data/vpd"  # the downloaded runs and val rows; the scripts run with this as cwd
TARGET_DIR = HERE / "t-9d2b8f02"
VPD_PTH = HERE / "s-55ea3f9b" / "model_400000.pth"
DATA = HERE / "pile_val_4096x513.npy"
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def site_names(n_layer: int = 4) -> list[str]:
    return [
        f"h.{i}.{'attn' if k.endswith('_proj') and k != 'down_proj' else 'mlp'}.{k}"
        for i in range(n_layer)
        for k in KINDS
    ]


# ------------------------------------------------------------------ target


class Site(nn.Module):
    """One decomposable linear map. Target mode: x @ W.T (caches x). Component mode: masked
    subcomponent forward plus the delta component."""

    def __init__(self, W: Tensor):
        super().__init__()
        self.register_buffer("W", W)
        self.V: Tensor | None = None  # [d_in, C]
        self.U: Tensor | None = None  # [C, d_out]
        self.mask: Tensor | None = None
        self.delta_mask: Tensor | None = None
        self.last_input: Tensor | None = None
        self.last_output: Tensor | None = None
        self.cache_input = False
        self.cache_output = False
        self.in_fn = None  # optional input transform (pruning baselines ablate neurons / heads here)
        self.register_buffer("b", None)  # the bias added to the output (TLTarget's; vpd4l's matrices have none)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_fn is not None:
            x = self.in_fn(x)
        if self.cache_input:
            self.last_input = x.detach()
        out = self._forward(x)
        if self.b is not None:
            out = out + self.b
        if self.cache_output:
            self.last_output = out.detach()
        return out

    def _forward(self, x: Tensor) -> Tensor:
        if self.mask is None:
            return x @ self.W.T
        assert self.V is not None and self.U is not None
        out = ((x @ self.V) * self.mask) @ self.U
        if self.delta_mask is not None:
            delta = self.W - (self.V @ self.U).T
            out = out + self.delta_mask.unsqueeze(-1) * (x @ delta.T)
        return out


BEHAVIOR_OFF = 1e4  # set_behavior's logit offset outside a behavior's tokens (exp(-1e4) is 0 in float32)


def rms(x: Tensor, w: Tensor, eps: float) -> Tensor:
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def gelu_tanh(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Target(nn.Module):
    """LlamaSimpleMLP: pre-RMSNorm blocks, rotate-half RoPE MHA (6 heads, hd 128), GELU(tanh)
    MLP, tied wte/lm_head, no biases."""

    def __init__(self, sd: dict[str, Tensor], cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.n_layer, self.n_head = cfg["n_layer"], cfg["n_head"]
        self.hd = cfg["n_embd"] // cfg["n_head"]
        assert cfg["n_key_value_heads"] == cfg["n_head"] and cfg["rotary_dim"] == self.hd
        self.eps = float(cfg["rms_norm_eps"])
        self.register_buffer("wte", sd["wte.weight"])
        self.register_buffer("ln_f", sd["ln_f.weight"])
        self.behavior, self.behavior_tokens, self.behavior_at = None, None, None  # set_behavior's mask [V], tokens, positions
        self.norms = nn.ParameterList()
        self.sites = nn.ModuleDict()
        for i in range(self.n_layer):
            self.norms.append(nn.Parameter(sd[f"h.{i}.rms_1.weight"], requires_grad=False))
            self.norms.append(nn.Parameter(sd[f"h.{i}.rms_2.weight"], requires_grad=False))
        for name in site_names(self.n_layer):
            self.sites[name.replace(".", "-")] = Site(sd[f"{name}.weight"])
        n_ctx, rd = cfg["n_ctx"], self.hd
        pos = torch.arange(n_ctx, dtype=torch.float32)
        freq = float(cfg["rotary_base"]) ** (torch.arange(rd // 2, dtype=torch.float32) / (rd / 2))
        ang = pos[:, None] / freq.repeat(2)[None, :]
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())
        for p in self.parameters():
            p.requires_grad_(False)

    def site(self, name: str) -> Site:
        return self.sites[name.replace(".", "-")]  # type: ignore[return-value]

    # The architecture as the oracle code uses it (native, engine, goal_eval, teacher): the embedding, the norms, the
    # MLP's activation and the unembedding. vpd4l's are its own operations (bit for bit what the callers computed before);
    # TLTarget (a TransformerLens model: no norm or LayerNorm, learned positions, biases, untied unembedding) has its own.
    norm_kind = "rms"
    lm = True  # a language model: a changed text's token is drawn from its own prediction there (native.replacement_logp)

    @property
    def vocab(self) -> int:
        """The logits' size."""
        return self.wte.shape[0]

    @property
    def n_tokens(self) -> int:
        """The input vocabulary's size (vpd4l: the logits' too)."""
        return self.wte.shape[0]

    @property
    def d_model(self) -> int:
        return self.wte.shape[1]

    def embed(self, ids: Tensor, start: int = 0) -> Tensor:
        """The residual stream entering layer 0 for tokens ids [..., T] standing at positions start.. ."""
        return self.wte[ids]

    def token_embed(self, ids: Tensor) -> Tensor:
        """The tokens' own embeddings, without positions (a changed token's embedding change is a difference of these)."""
        return self.wte[ids]

    def norm_weight(self, j: int) -> Tensor:
        """The norm j's scale: 2 i before layer i's attention, 2 i + 1 before its MLP, 2 n_layer (or -1) the final one."""
        return self.ln_f if j in (-1, 2 * self.n_layer) else self.norms[j]

    def norm(self, x: Tensor, j: int, fused: bool = False) -> Tensor:
        """Norm j of x (norm_weight's numbering); fused: torch's single kernel (equal within float32 rounding)."""
        w = self.norm_weight(j)
        return F.rms_norm(x, (x.shape[-1],), w, self.eps) if fused else rms(x, w, self.eps)

    def norm_parts(self, read: Tensor, scale: Tensor, j: int) -> Tensor:
        """Norm j applied to the vector read with the statistics of the vector scale (a cut's partial input: read is a
        share of scale); vpd4l: w * read / rms(scale)."""
        w = self.norm_weight(j)
        return w * (read.float() * torch.rsqrt(scale.float().pow(2).mean(-1, keepdim=True) + self.eps)).to(read.dtype)

    def act(self, x: Tensor, fused: bool = False) -> Tensor:
        """The MLP's activation (vpd4l: GELU with the tanh approximation)."""
        return F.gelu(x, approximate="tanh") if fused else gelu_tanh(x)

    def unembed(self, h: Tensor) -> Tensor:
        """Logits from the final normed residual stream (vpd4l: the tied embedding), plus the behavior mask when one is
        set (set_behavior)."""
        out = h @ self.wte.T
        return out if self.behavior is None else out + self.behavior

    def set_behavior(self, tokens: list[int] | None, at: list[int] | None = None) -> None:
        """The behavior-targeted mode (#2951, the lead's 10-13: "why is over are?"): every logit outside tokens lowered
        by BEHAVIOR_OFF, so each next-token distribution is the one over tokens alone (the others' probability 0 to
        float32: every KL computed from them is the KL of the behavior's distribution, a divergence of its log-odds for
        two tokens), shapes and token ids unchanged; None restores the whole distribution. at: the text's target positions,
        the predictions the behavior is about (Native.reference reads every other position's whole distribution: the
        changed-input samplers' predictions at a context position)."""
        self.behavior_at = frozenset(int(p) for p in at) if at else None
        if not tokens:
            self.behavior, self.behavior_tokens = None, None
            return
        m = torch.full((self.wte.shape[0],), -BEHAVIOR_OFF, device=self.wte.device)
        m[torch.tensor(sorted(set(int(z) for z in tokens)), device=self.wte.device)] = 0.0
        self.behavior, self.behavior_tokens = m, tuple(sorted(set(int(z) for z in tokens)))

    def _rope(self, x: Tensor, T: int) -> Tensor:
        n = self.hd // 2
        rot = torch.cat([-x[..., n:], x[..., :n]], dim=-1)
        return x * self.cos[:T] + rot * self.sin[:T]

    def attention_pattern(self, q_flat: Tensor, k_flat: Tensor) -> Tensor:
        """Post-softmax causal attention map [B, H, T, T] from one layer's q_proj/k_proj outputs."""
        B, T, _ = q_flat.shape
        q = self._rope(q_flat.view(B, T, self.n_head, self.hd).transpose(1, 2), T)
        k = self._rope(k_flat.view(B, T, self.n_head, self.hd).transpose(1, 2), T)
        z = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        causal = torch.ones(T, T, dtype=torch.bool, device=z.device).tril()
        return z.masked_fill(~causal, float("-inf")).softmax(-1)

    def forward(self, ids: Tensor) -> Tensor:
        return self.unembed(self.hidden(ids))

    def hidden(self, ids: Tensor, start: int = 0, prefix: list | None = None, layer: int = 0, fused: bool = False) -> Tensor:
        """The final normed residual stream [B, T, d]; the logits are it times the tied embedding. With start > 0 and
        prefix (keys_values() of a sequence that has ids' tokens before start), the projections and the MLP run only on
        the positions from start, the prefix's queries, keys and values standing before them in an attention of the
        whole length (the same call, so the same numbers as the whole forward; the prefix computed at the same batch
        size, whose attention can round differently): [B, T - start, d]. With layer > 0 the layers below it are the
        prefix's too (its residual stream entering that layer, at every position from start). fused: the norms and the
        GELU as torch's single kernels (_steps), equal to rms and gelu_tanh within float32 rounding, not to the bit."""
        B, T = ids.shape
        L = T - start
        norm, act = self._steps(fused)
        x = self.embed(ids[:, start:], start) if layer == 0 else prefix[layer][3][:, start:]
        for i in range(layer, self.n_layer):
            s = lambda k: self.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
            h = norm(x, 2 * i)
            q = s("q_proj")(h).view(B, L, self.n_head, self.hd).transpose(1, 2)
            k = s("k_proj")(h).view(B, L, self.n_head, self.hd).transpose(1, 2)
            v = s("v_proj")(h).view(B, L, self.n_head, self.hd).transpose(1, 2)
            if start:
                q, k = self._rope_from(q, start, T), self._rope_from(k, start, T)
                q, k, v = (torch.cat([w[:, :, :start], z], 2) for w, z in zip(prefix[i][:3], (q, k, v)))
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)[:, :, start:]
            else:
                q, k = self._rope(q, T), self._rope(k, T)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s("o_proj")(y.transpose(1, 2).reshape(B, L, -1))
            h = norm(x, 2 * i + 1)
            x = x + s("down_proj")(act(s("c_fc")(h)))
        return norm(x, 2 * self.n_layer)

    def _steps(self, fused: bool):
        """(norm(x, j), activation(x)) as norm and act with fused: vpd_model's own rms and gelu_tanh, or torch's fused RMS
        norm and tanh GELU (the same functions in one kernel each: on MPS the GELU of a [32, 395, 3072] batch takes 3 ms
        instead of 55)."""
        return (lambda x, j: self.norm(x, j, fused)), (lambda x: self.act(x, fused))

    def _rope_from(self, x: Tensor, start: int, T: int) -> Tensor:
        """_rope of positions start..T-1."""
        n = self.hd // 2
        rot = torch.cat([-x[..., n:], x[..., :n]], dim=-1)
        return x * self.cos[start:T] + rot * self.sin[start:T]

    def keys_values(self, ids: Tensor, fused: bool = False) -> list:
        """Each layer's rotated queries and keys, its values [B, H, T, hd] and the residual stream entering it [B, T, d]
        on sequences [B, T] (hidden()'s prefix; fused as there)."""
        B, T = ids.shape
        norm, act = self._steps(fused)
        x = self.embed(ids)
        out = []
        for i in range(self.n_layer):
            s = lambda k: self.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
            h = norm(x, 2 * i)
            q = s("q_proj")(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            k = s("k_proj")(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            v = s("v_proj")(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            q, k = self._rope(q, T), self._rope(k, T)
            out.append((q, k, v, x))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s("o_proj")(y.transpose(1, 2).reshape(B, T, -1))
            h = norm(x, 2 * i + 1)
            x = x + s("down_proj")(act(s("c_fc")(h)))
        return out


class TLTarget(Target):
    """A TransformerLens HookedTransformer as a Target (InterpBench's models, MIB's GPT-2): its sites the same six
    matrices per layer, with biases (Site.b); no norm, LayerNorm or RMSNorm (normalization_type None, "LN" or "RMS");
    learned position embeddings; an untied unembedding with its bias; GELU exact ("gelu"), tanh ("gelu_new") or ReLU.
    LNPre (LayerNorm with its scale and offset folded into the next matrices) is LayerNorm at scale 1 and offset 0. A model
    whose outputs are its inputs' vocabulary (InterpBench's IOI, GPT-2) is a language model (lm).
    Rotary is off: cos 1 and sin 0, so _rope and every caller's inlined rotation leave queries and keys as they are.
    vpd4l's attributes that would mean something else here (wte, ln_f, norms, eps) raise, so a caller still reading
    them fails instead of computing with the wrong architecture.

    sd: the HookedTransformer's state dict; cfg: its config's fields (n_layers, n_heads, d_model, d_head, n_ctx,
    d_vocab, act_fn, normalization_type, eps)."""

    lm = False  # InterpBench's Tracr models label positions: a changed token is drawn uniformly from the input vocabulary

    def __init__(self, sd: dict[str, Tensor], cfg: dict):
        nn.Module.__init__(self)
        if cfg.get("attention_dir", "causal") != "causal":
            raise ValueError("TLTarget runs causal attention (the oracle's runs are causal throughout)")
        L, H, d, hd = cfg["n_layers"], cfg["n_heads"], cfg["d_model"], cfg["d_head"]
        self.cfg = {"n_layer": L, "n_head": H, "n_embd": d, "d_head": hd, "n_ctx": cfg["n_ctx"], "tl": dict(cfg)}
        self.n_layer, self.n_head, self.hd = L, H, hd
        self.norm_kind = {None: "none", "LN": "ln", "LNPre": "ln", "RMS": "tl_rms"}[cfg.get("normalization_type")]   # (not vpd4l's "rms")
        self.lm = cfg.get("d_vocab_out", cfg["d_vocab"]) == cfg["d_vocab"]   # its outputs are its own inputs' vocabulary
        self.tl_eps = float(cfg.get("eps", 1e-5))
        self.act_fn = cfg.get("act_fn", "gelu")
        f = lambda k: sd[k].float()  # noqa: E731
        self.register_buffer("W_E", f("embed.W_E"))
        self.register_buffer("W_pos", f("pos_embed.W_pos"))
        self.register_buffer("W_U", f("unembed.W_U"))
        self.register_buffer("b_U", f("unembed.b_U") if "unembed.b_U" in sd else torch.zeros(sd["unembed.W_U"].shape[1]))
        self.sites = nn.ModuleDict()
        for i in range(L):
            pre = f"blocks.{i}."
            mats = {"q_proj": (f(pre + "attn.W_Q").permute(0, 2, 1).reshape(H * hd, d), pre + "attn.b_Q"),
                    "k_proj": (f(pre + "attn.W_K").permute(0, 2, 1).reshape(H * hd, d), pre + "attn.b_K"),
                    "v_proj": (f(pre + "attn.W_V").permute(0, 2, 1).reshape(H * hd, d), pre + "attn.b_V"),
                    "o_proj": (f(pre + "attn.W_O").reshape(H * hd, d).T.contiguous(), pre + "attn.b_O"),
                    "c_fc": (f(pre + "mlp.W_in").T.contiguous(), pre + "mlp.b_in"),
                    "down_proj": (f(pre + "mlp.W_out").T.contiguous(), pre + "mlp.b_out")}
            for kind, (W, bk) in mats.items():
                st = Site(W.contiguous())
                if bk in sd:
                    st.b = f(bk).reshape(-1).contiguous()
                self.sites[f"h-{i}-{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}-{kind}"] = st
        self.ln = nn.ParameterList()   # LayerNorm (or RMS) scales and offsets: 2 i, 2 i + 1, then the final one
        self.ln_b = nn.ParameterList()
        if self.norm_kind in ("ln", "tl_rms"):
            for i in range(L):
                for k in ("ln1", "ln2"):
                    self.ln.append(nn.Parameter(f(f"blocks.{i}.{k}.w") if f"blocks.{i}.{k}.w" in sd else torch.ones(d), requires_grad=False))
                    self.ln_b.append(nn.Parameter(f(f"blocks.{i}.{k}.b") if f"blocks.{i}.{k}.b" in sd else torch.zeros(d), requires_grad=False))
            self.ln.append(nn.Parameter(f("ln_final.w") if "ln_final.w" in sd else torch.ones(d), requires_grad=False))
            self.ln_b.append(nn.Parameter(f("ln_final.b") if "ln_final.b" in sd else torch.zeros(d), requires_grad=False))
        self.register_buffer("cos", torch.ones(cfg["n_ctx"], hd))
        self.register_buffer("sin", torch.zeros(cfg["n_ctx"], hd))
        for p in self.parameters():
            p.requires_grad_(False)

    def __getattr__(self, name: str):
        if name in ("wte", "ln_f", "norms", "eps"):
            raise AttributeError(f"TLTarget has no {name}: use embed, norm, norm_parts, act, unembed (vpd_model.Target)")
        return super().__getattr__(name)

    @property
    def vocab(self) -> int:
        return self.W_U.shape[1]

    @property
    def n_tokens(self) -> int:
        return self.W_E.shape[0]

    @property
    def d_model(self) -> int:
        return self.W_E.shape[1]

    def embed(self, ids: Tensor, start: int = 0) -> Tensor:
        return self.W_E[ids] + self.W_pos[start:start + ids.shape[-1]]

    def token_embed(self, ids: Tensor) -> Tensor:
        return self.W_E[ids]

    def norm_weight(self, j: int) -> Tensor:
        if self.norm_kind == "none":
            raise AttributeError("TLTarget without normalization has no norm scales")
        return self.ln[2 * self.n_layer if j == -1 else j]

    def norm(self, x: Tensor, j: int, fused: bool = False) -> Tensor:
        if self.norm_kind == "none":
            return x
        j = 2 * self.n_layer if j == -1 else j
        if self.norm_kind == "tl_rms":
            return rms(x, self.ln[j], self.tl_eps)
        return F.layer_norm(x, (x.shape[-1],), self.ln[j], self.ln_b[j], self.tl_eps)

    def norm_parts(self, read: Tensor, scale: Tensor, j: int) -> Tensor:
        """No norm: read itself. LayerNorm: w (read - mean(read)) / std(scale), the offset b left to the remainder (a
        share of the input carries none of it); RMS: w read / rms(scale)."""
        if self.norm_kind == "none":
            return read
        j = 2 * self.n_layer if j == -1 else j
        sf = scale.float()
        if self.norm_kind == "tl_rms":
            return self.ln[j] * (read.float() * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + self.tl_eps)).to(read.dtype)
        rf = read.float()
        var = (sf - sf.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True)
        return self.ln[j] * ((rf - rf.mean(-1, keepdim=True)) * torch.rsqrt(var + self.tl_eps)).to(read.dtype)

    def norm_stat(self, x: Tensor) -> Tensor:
        """The norm's scale statistic of x (a frozen norm keeps it): 1 / std(x) (LayerNorm), 1 / rms(x), 1 (no norm)."""
        xf = x.float()
        if self.norm_kind == "none":
            return torch.ones(*x.shape[:-1], 1, device=x.device)
        if self.norm_kind == "tl_rms":
            return torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.tl_eps)
        return torch.rsqrt((xf - xf.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + self.tl_eps)

    def norm_with(self, x: Tensor, sc: Tensor, j: int) -> Tensor:
        """Norm j of x at the scale statistic sc (norm_stat's): LayerNorm w (x - mean(x)) sc + b; RMS w x sc; none x."""
        if self.norm_kind == "none":
            return x
        j = 2 * self.n_layer if j == -1 else j
        xf = x.float()
        if self.norm_kind == "tl_rms":
            return self.ln[j] * (xf * sc).to(x.dtype)
        return self.ln[j] * ((xf - xf.mean(-1, keepdim=True)) * sc).to(x.dtype) + self.ln_b[j]

    def act(self, x: Tensor, fused: bool = False) -> Tensor:
        if self.act_fn == "relu":
            return F.relu(x)
        return F.gelu(x, approximate="tanh" if self.act_fn in ("gelu_new", "gelu_pytorch_tanh") else "none")

    def unembed(self, h: Tensor) -> Tensor:
        return h @ self.W_U + self.b_U


def load_target(device: str = "mps") -> Target:
    cfg = yaml.safe_load((TARGET_DIR / "model_config.yaml").read_text())
    sd = load_file(str(TARGET_DIR / "model_step_99999.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}
    return Target(sd, cfg).to(device).eval()


# ------------------------------------------------------------------ CI function


class CILinear(nn.Module):
    """param_decomp.models.components.Linear: W [in, out], b [out]."""

    def __init__(self, W: Tensor, b: Tensor):
        super().__init__()
        self.W, self.b = nn.Parameter(W, requires_grad=False), nn.Parameter(b, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class CIBlock(nn.Module):
    """RMSNorm (no scale) -> bidirectional RoPE MHA -> residual -> RMSNorm -> GELU MLP -> residual."""

    def __init__(self, sd: dict[str, Tensor], pre: str, n_heads: int, rope_base: float, max_len: int):
        super().__init__()
        g = lambda k: nn.Parameter(sd[pre + k], requires_grad=False)
        self.wq, self.wk, self.wv, self.wo = (
            g("attn.q_proj.weight"), g("attn.k_proj.weight"), g("attn.v_proj.weight"),
            g("attn.out_proj.weight"),
        )
        self.fc1 = CILinear(sd[pre + "mlp.0.W"], sd[pre + "mlp.0.b"])
        self.fc2 = CILinear(sd[pre + "mlp.2.W"], sd[pre + "mlp.2.b"])
        self.n_heads = n_heads
        self.d = self.wq.shape[0]
        self.dh = self.d // n_heads
        inv = 1.0 / (rope_base ** (torch.arange(0, self.dh, 2).float() / self.dh))
        ang = torch.arange(max_len).float()[:, None] * inv[None, :]
        self.register_buffer("cos", torch.cat([ang.cos(), ang.cos()], -1))
        self.register_buffer("sin", torch.cat([ang.sin(), ang.sin()], -1))

    def _rope(self, x: Tensor, S: int) -> Tensor:
        n = self.dh // 2
        return x * self.cos[:S] + torch.cat([-x[..., n:], x[..., :n]], -1) * self.sin[:S]

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        h = F.rms_norm(x, (D,))
        sp = lambda t: t.view(B, S, self.n_heads, self.dh).transpose(1, 2)
        q, k, v = sp(h @ self.wq.T), sp(h @ self.wk.T), sp(h @ self.wv.T)
        q, k = self._rope(q, S), self._rope(k, S)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + a.transpose(1, 2).reshape(B, S, D) @ self.wo.T
        return x + self.fc2(F.gelu(self.fc1(F.rms_norm(x, (D,)))))


class GlobalSharedTransformerCiFn(nn.Module):
    """Concatenate RMS-normed pre-weight activations of every site (sorted by name), project to
    d_model, 8 bidirectional transformer blocks, project to sum(C); split per site."""

    def __init__(self, sd: dict[str, Tensor], order: list[str], cs: list[int], n_heads: int,
                 rope_base: float, max_len: int):
        super().__init__()
        self.order, self.cs = order, cs
        self.proj_in = CILinear(sd["_input_projector.W"], sd["_input_projector.b"])
        self.head = CILinear(sd["_output_head.W"], sd["_output_head.b"])
        n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("_blocks."))
        self.blocks = nn.ModuleList(
            CIBlock(sd, f"_blocks.{i}.", n_heads, rope_base, max_len) for i in range(n_blocks)
        )

    def forward(self, acts: dict[str, Tensor]) -> dict[str, Tensor]:
        x = torch.cat([F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.order], -1)
        x = self.proj_in(x)
        for b in self.blocks:
            x = b(x)
        return dict(zip(self.order, self.head(x).split(self.cs, -1), strict=True))


def lower_leaky(x: Tensor) -> Tensor:
    """Forward value of the leaky-hard lower sigmoid: clamp(x, 0, 1)."""
    return x.clamp(0.0, 1.0)


class VPD(nn.Module):
    """The published decomposition as an executable object: subcomponents (U, V per site) installed
    into a Target, plus the causal-importance network."""

    def __init__(self, target: Target, ci_fn: GlobalSharedTransformerCiFn, UV: dict[str, tuple[Tensor, Tensor]]):
        super().__init__()
        self.target, self.ci_fn = target, ci_fn
        self.names = list(UV)
        for n, (U, V) in UV.items():
            st = target.site(n)
            st.U, st.V = U, V
        self.C = {n: UV[n][0].shape[0] for n in self.names}

    def clear(self):
        for n in self.names:
            st = self.target.site(n)
            st.mask = st.delta_mask = st.last_input = st.last_output = None
            st.gathered = None  # native's subset of subcomponents (V and U columns) for a mask over that subset
            st.cache_input = st.cache_output = False

    @torch.no_grad()
    def target_forward(self, ids: Tensor) -> Tensor:
        self.clear()
        return self.target(ids)

    @torch.no_grad()
    def target_and_ci(self, ids: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Clean target logits and CI (lower-leaky forward value) per site."""
        self.clear()
        for n in self.names:
            self.target.site(n).cache_input = True
        logits = self.target(ids)
        acts = {n: self.target.site(n).last_input for n in self.names}
        self.clear()
        pre = self.ci_fn(acts)  # type: ignore[arg-type]
        return logits, {n: lower_leaky(v) for n, v in pre.items()}

    def masked(self, ids: Tensor, masks: dict[str, Tensor], delta_masks: dict[str, Tensor] | None) -> Tensor:
        for n in self.names:
            st = self.target.site(n)
            st.mask = masks[n]
            st.delta_mask = None if delta_masks is None else delta_masks[n]
        try:
            return self.target(ids)
        finally:
            self.clear()


def load_vpd(target: Target, device: str = "mps") -> VPD:
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    UV: dict[str, tuple[Tensor, Tensor]] = {}
    ci_sd: dict[str, Tensor] = {}
    for k, v in raw.items():
        if k.startswith("_components."):
            site, which = k[len("_components."):].rsplit(".", 1)
            name = site.replace("-", ".")
            U, V = UV.get(name, (None, None))
            UV[name] = (v.float() if which == "U" else U, v.float() if which == "V" else V)  # type: ignore[assignment]
        elif "_global_ci_fn." in k:
            ci_sd[k.split("_global_ci_fn.", 1)[1]] = v.float()
    target_weight_diff = max(
        (raw[f"target_model.{n}.weight"] - target.site(n).W.cpu()).abs().max().item() for n in UV
    )
    del raw
    order = sorted(UV)
    UV = {n: (UV[n][0].to(device), UV[n][1].to(device)) for n in site_names()}
    ci = GlobalSharedTransformerCiFn(ci_sd, order, [UV[n][0].shape[0] for n in order],
                                     n_heads=16, rope_base=10000.0, max_len=512)
    vpd = VPD(target, ci.to(device).eval(), UV)
    vpd.target_weight_diff = target_weight_diff
    return vpd


VAL_PARQUET = Path.home() / "mpd-data/hf/datasets--danbraunai--pile-uncopyrighted-tok-shuffled/snapshots/6f141f6e4edc32fea842f2ece0d3477b15c3dc90/data/val-00000-of-00012.parquet"


def val_tokens(n: int, seq: int = 512, offset: int = 0) -> Tensor:
    """Rows [offset, offset + n) of the val split's first shard (pile_val_4096x513.npy is its first
    4096 rows; later rows are read from the parquet, first row group = 48638 rows)."""
    if offset + n <= 4096:
        return torch.from_numpy(np.load(DATA, mmap_mode="r")[offset:offset + n, :seq].astype(np.int64))
    import pyarrow.parquet as pq

    col = pq.ParquetFile(VAL_PARQUET).read_row_group(0, columns=["input_ids"]).column("input_ids")
    sl = col.slice(offset, n).combine_chunks()
    flat = sl.flatten().to_numpy().astype(np.int64)
    width = len(flat) // n
    assert width * n == len(flat), "val rows are not of equal length"
    return torch.from_numpy(flat.reshape(n, width)[:, :seq].copy())
