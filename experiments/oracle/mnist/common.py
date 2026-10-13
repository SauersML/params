"""MNIST track of the oracle recipe: shared data, target model and decomposition runs.

The target is a 784-256-256-10 ReLU MLP. Its three weight matrices (fc1, fc2, fc3) are decomposed into rank-one
subcomponents with VPD's method (nano_param_decomp: W = V U + remainder, a causal-importance gate per subcomponent per
image). One run here is VPD's masked forward on an image batch:

  layer m: a_m = h_m V_m  (the subcomponent activations, [B, C_m])
           out_m = (a_m * mask_m) U_m + b_m + rem_m * (h_m (W_m - (V_m U_m)^T)^T)
           h_{m+1} = relu(out_m)  (no ReLU after fc3: logits)

with mask_m in [0, 1] per subcomponent and rem_m in {0, 1} the remainder's switch. All masks one and the remainder on is
the real model exactly. `patch` replaces chosen activations a_m[:, c] with given values (interchange and donor runs).
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path.home() / "mpd-data" / "mnist"
RAW = DATA / "raw"
NANO = Path.home() / "mpd-data" / "param-decomp" / "nano_param_decomp" / "run.py"
MODULES = ("fc1", "fc2", "fc3")
LN2 = float(np.log(2.0))


def _idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train and test images as float32 [N, 784] in [0, 1], labels int64."""
    xtr = torch.from_numpy(_idx(RAW / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0)
    ytr = torch.from_numpy(_idx(RAW / "train-labels-idx1-ubyte.gz").astype(np.int64))
    xte = torch.from_numpy(_idx(RAW / "t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0)
    yte = torch.from_numpy(_idx(RAW / "t10k-labels-idx1-ubyte.gz").astype(np.int64))
    return xtr, ytr, xte, yte


class MLP(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def load_mlp(path: Path = DATA / "mlp.pt") -> MLP:
    model = MLP()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_nano():
    """nano_param_decomp/run.py (VPD's one-file reference implementation), imported from its path."""
    spec = importlib.util.spec_from_file_location("nano_param_decomp_run", NANO)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nano_param_decomp_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def kl_bits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(p) || softmax(q)) in bits per image, p the reference."""
    lp = F.log_softmax(p_logits, -1)
    lq = F.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1) / LN2


@dataclass
class Decomp:
    """A fitted decomposition: V_m [d_in, C_m], U_m [C_m, d_out], target W_m [d_out, d_in], b_m; the CI function."""

    V: dict[str, torch.Tensor]
    U: dict[str, torch.Tensor]
    W: dict[str, torch.Tensor]
    b: dict[str, torch.Tensor]
    ci_fn: nn.Module | None

    layer_names = MODULES

    @property
    def sizes(self) -> list[int]:
        return [self.V[m].shape[1] for m in MODULES]

    @property
    def total(self) -> int:
        return sum(self.sizes)

    @property
    def offsets(self) -> list[int]:
        out, o = [], 0
        for s in self.sizes:
            out.append(o)
            o += s
        return out

    def split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(zip(MODULES, flat.split(self.sizes, dim=-1), strict=True))

    def name(self, j: int) -> str:
        for m, o, s in zip(MODULES, self.offsets, self.sizes, strict=True):
            if o <= j < o + s:
                return f"{m}#{j - o}"
        raise IndexError(j)

    def remainder(self, m: str) -> torch.Tensor:
        return self.W[m] - (self.V[m] @ self.U[m]).T

    @torch.no_grad()
    def run(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rem: torch.Tensor | float = 1.0,
        patch_sel: torch.Tensor | None = None,
        patch_val: torch.Tensor | None = None,
        rem_patch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Masked forward. mask [B, total]; rem a scalar or [B] remainder switch; patch_sel [B, total] bool picks the
        activations replaced by patch_val [B, total]; rem_patch[m] [B, d_out] replaces layer m's remainder output.
        Returns logits [B, 10]."""
        B = x.shape[0]
        masks = self.split(mask)
        sels = self.split(patch_sel) if patch_sel is not None else None
        vals = self.split(patch_val) if patch_val is not None else None
        r = rem if isinstance(rem, torch.Tensor) else torch.full((B,), float(rem))
        h = x
        out = h
        for i, m in enumerate(MODULES):
            a = h @ self.V[m]
            if sels is not None and vals is not None:
                a = torch.where(sels[m], vals[m], a)
            rem_out = rem_patch[m] if rem_patch is not None else h @ self.remainder(m).T
            out = (a * masks[m]) @ self.U[m] + self.b[m] + r[:, None] * rem_out
            h = F.relu(out) if i < len(MODULES) - 1 else out
        return out

    @torch.no_grad()
    def acts(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Real-model run: subcomponent activations [B, total], each layer's input h_m, each layer's remainder output."""
        h = x
        a_all, hs, rems = [], {}, {}
        for i, m in enumerate(MODULES):
            hs[m] = h
            a_all.append(h @ self.V[m])
            rems[m] = h @ self.remainder(m).T
            out = h @ self.W[m].T + self.b[m]
            h = F.relu(out) if i < len(MODULES) - 1 else out
        return torch.cat(a_all, -1), hs, rems

    @torch.no_grad()
    def ci(self, x: torch.Tensor) -> torch.Tensor:
        """Causal importance (lower leaky-hard sigmoid, VPD's masks) [B, total] from the real model's layer inputs."""
        assert self.ci_fn is not None
        _, hs, _ = self.acts(x)
        lower, _, _ = self.ci_fn({m: hs[m][:, None, :] for m in MODULES})
        return torch.cat([lower[m][:, 0, :] for m in MODULES], -1)

    def contribution(self, x: torch.Tensor) -> torch.Tensor:
        """|a_c| * ||U_c||: the size of each subcomponent's write on each image, [B, total]."""
        a, _, _ = self.acts(x)
        unorm = torch.cat([self.U[m].norm(dim=1) for m in MODULES])
        return a.abs() * unorm


def load_decomp(path: Path) -> Decomp:
    nano = load_nano()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = nano.Config(**blob["cfg"])
    d_in = {m: blob["W"][m].shape[1] for m in MODULES}
    ci_fn = nano.CITransformer(d_in, cfg.C_per_module, cfg)
    ci_fn.load_state_dict(blob["ci_fn"])
    ci_fn.eval()
    return Decomp(V=blob["V"], U=blob["U"], W=blob["W"], b=blob["b"], ci_fn=ci_fn)


@dataclass
class Neurons:
    """The native-unit baseline: the MLP's hidden neurons (fc1's 256 outputs after ReLU, then fc2's), with the same run
    interface as Decomp. A neuron's mask scales its post-ReLU activation; patching replaces that activation. The
    logits have no units to mask (fc3's output is the prediction)."""

    W: dict[str, torch.Tensor]
    b: dict[str, torch.Tensor]
    ci_fn: None = None

    layer_names = ("h1", "h2")

    @property
    def sizes(self) -> list[int]:
        return [self.W["fc1"].shape[0], self.W["fc2"].shape[0]]

    @property
    def offsets(self) -> list[int]:
        return [0, self.sizes[0]]

    @property
    def total(self) -> int:
        return sum(self.sizes)

    def name(self, j: int) -> str:
        n1 = self.sizes[0]
        return f"h1#{j}" if j < n1 else f"h2#{j - n1}"

    @torch.no_grad()
    def run(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rem: torch.Tensor | float = 1.0,
        patch_sel: torch.Tensor | None = None,
        patch_val: torch.Tensor | None = None,
        rem_patch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        n1 = self.sizes[0]
        h = x
        for i, m in enumerate(("fc1", "fc2")):
            h = F.relu(h @ self.W[m].T + self.b[m])
            sl = slice(0, n1) if i == 0 else slice(n1, self.total)
            if patch_sel is not None and patch_val is not None:
                h = torch.where(patch_sel[:, sl], patch_val[:, sl], h)
            h = h * mask[:, sl]
        return h @ self.W["fc3"].T + self.b["fc3"]

    @torch.no_grad()
    def acts(self, x: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        h1 = F.relu(x @ self.W["fc1"].T + self.b["fc1"])
        h2 = F.relu(h1 @ self.W["fc2"].T + self.b["fc2"])
        return torch.cat([h1, h2], -1), {}, {}

    def contribution(self, x: torch.Tensor) -> torch.Tensor:
        """|h_j| * ||W_out[:, j]||: the size of each neuron's write into the next layer."""
        a, _, _ = self.acts(x)
        wn = torch.cat([self.W["fc2"].norm(dim=0), self.W["fc3"].norm(dim=0)])
        return a.abs() * wn


def load_neurons(path: Path = DATA / "mlp.pt") -> Neurons:
    sd = torch.load(path, map_location="cpu")
    return Neurons(
        W={m: sd[f"{m}.weight"] for m in MODULES},
        b={m: sd[f"{m}.bias"] for m in MODULES},
    )
