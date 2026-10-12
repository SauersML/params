"""The masked-run engine (#2951 graph oracle): VPD's masked forward pass of vpd4l on one text with per-site masks over
a chosen set of subcomponents, the remainders W - V U live or off, and the KL in bits of the model's next-token
distribution at the targets from the run's, differentiable in the masks. It serves searches that take gradients with
respect to masks (the teacher's mask search, the reward's adversary), batched over rows (samples x patterns x
answers x restarts), on any GPU.

  eng = Engine(nat, ids, targets, live=True, columns=None)
  eng.kl(masks, precision="score") -> [B] bits

columns: per site the subcomponent indices masks address (a LongTensor), every other one off in every run; None, every
subcomponent. Give the union a caller's masks can ever switch on (VPD's answer and an adversary's patterns): a site's
products then run on those columns alone (a sum over fewer terms: float32-close to the full products, not equal to
the bit). masks: {site: [B, T, K]} floats (K = the site's columns), with or without gradients.
precision: "score", float32 with vpd_model's own norms and GELU (the numbers to report); "tf32" (CUDA: TF32 tensor-core
matmuls) and "bf16" (bfloat16 autocast), with torch's fused norms and GELU, for a search's steps only, whose result is
then scored again with "score" (engine.rescore checks that).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

import carry
import native

PRECISIONS = ("score", "tf32", "bf16")


class Engine:
    def __init__(self, nat: native.Native, ids: list[int], targets: list[int], live: bool = True, columns: dict | None = None):
        import vpd_model

        self.nat, self.ids, self.targets, self.live = nat, list(ids), list(targets), live
        tg = nat.target
        self.tg, self.T, self.dev = tg, len(ids), nat.dev
        self.H, self.hd, self.eps = tg.n_head, tg.hd, tg.eps
        self.rms, self.gelu = vpd_model.rms, vpd_model.gelu_tanh
        with torch.no_grad():
            self.ref = nat.reference([self.ids], self.targets)
            self.emb = tg.wte[torch.tensor(self.ids, device=self.dev)][None]  # [1, T, d]
        self.columns = {n: (None if columns is None or columns.get(n) is None else columns[n].to(self.dev)) for n in nat.names}
        self.V, self.U, self.R = {}, {}, {}
        for n in nat.names:
            st, cols = tg.site(n), self.columns[n]
            self.V[n] = st.V if cols is None else st.V.index_select(1, cols).contiguous()  # [d_in, K]
            self.U[n] = st.U if cols is None else st.U.index_select(0, cols).contiguous()  # [K, d_out]
            self.R[n] = st.delta_T  # [d_in, d_out]: the remainder W - V U, transposed (native.model)

    def width(self, n: str) -> int:
        """The number of a site's columns (a mask's last dimension)."""
        return self.V[n].shape[1]

    def _site(self, n: str, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = ((h @ self.V[n]) * m) @ self.U[n]
        return out + h @ self.R[n] if self.live else out

    def hidden(self, masks: dict, fused: bool) -> torch.Tensor:
        """The final normed residual stream at the targets [B, len(targets), d] of the masked run."""
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        B = next(iter(masks.values())).shape[0]
        if fused:
            norm = lambda x, w: F.rms_norm(x, (x.shape[-1],), w, self.eps)  # noqa: E731
            act = lambda x: F.gelu(x, approximate="tanh")  # noqa: E731
        else:
            norm = lambda x, w: self.rms(x, w, self.eps)  # noqa: E731
            act = self.gelu
        x = self.emb.expand(B, -1, -1)
        for i in range(tg.n_layer):
            name = lambda k: native.site_name(i, k)  # noqa: E731
            h = norm(x, tg.norms[2 * i])
            q, k, v = (self._site(name(kd), h, masks[name(kd)]).view(B, T, H, hd).transpose(1, 2)
                       for kd in ("q_proj", "k_proj", "v_proj"))
            y = F.scaled_dot_product_attention(tg._rope(q, T), tg._rope(k, T), v, is_causal=True)
            x = x + self._site(name("o_proj"), y.transpose(1, 2).reshape(B, T, -1), masks[name("o_proj")])
            h = norm(x, tg.norms[2 * i + 1])
            x = x + self._site(name("down_proj"), act(self._site(name("c_fc"), h, masks[name("c_fc")])), masks[name("down_proj")])
        return norm(x, tg.ln_f)[:, self.targets]  # the final norm over every position, as vpd_model.Target.hidden

    def kl(self, masks: dict, precision: str = "score") -> torch.Tensor:
        """KL in bits of the model at the targets from each row's masked run [B] (summed over the targets)."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        with _precision(precision, self.dev):
            h = self.hidden(masks, fused=precision != "score")
            lq = torch.log_softmax((h @ self.tg.wte.T).float(), -1)
        return carry.kl(self.ref, lq)

    def full(self, masks: dict) -> dict:
        """Masks over a site's columns as masks over all its subcomponents [B, T, C] (for another engine or Native)."""
        out = {}
        for n, m in masks.items():
            cols = self.columns[n]
            if cols is None:
                out[n] = m
            else:
                z = m.new_zeros(*m.shape[:-1], self.nat.C[n])
                out[n] = z.index_copy(-1, cols, m)
        return out

    @torch.no_grad()
    def rescore(self, masks: dict) -> torch.Tensor:
        """The numbers to report for masks a search found at a lower precision: float32, vpd_model's own steps."""
        return self.kl({n: m.detach().float() for n, m in masks.items()}, "score")


@contextlib.contextmanager
def _precision(precision: str, dev):
    cuda = str(dev).startswith("cuda")
    if precision == "tf32" and cuda:
        keep = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = keep
    elif precision == "bf16":
        with torch.autocast(str(dev).split(":")[0], dtype=torch.bfloat16):
            yield
    else:
        yield
