"""Phase 2 of the planted toys (#2951 graph oracle, the lead's 10-12 track): VPD fitted to each toy on its own training
distribution with nano_param_decomp (the VPD paper's method in one file, ~/mpd-data/spd-vpd-paper/nano_param_decomp:
stochastic and persistent-PGD reconstruction, importance minimality, the global shared CI transformer), the fit saved
as U, V per decomposed matrix and the CI transformer's weights; native_of_fit() wraps the toy with that decomposition
(VPD's answer: its causal importance above 0) so run_planted.py --decomp vpd runs the same teacher and Φ on it. Each
piece's unit is the head or neuron that holds most of its weight.

  MPD_MEM_GIB=8 ~/mpd-data/venv/bin/python -B experiments/oracle/toys/vpd_toy.py --toys ~/mpd-data/toys_planted --only route2
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NANO = Path.home() / "mpd-data/spd-vpd-paper"
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd"), str(NANO)]

import planted  # noqa: E402
import vpd_model  # noqa: E402

C_ATTN, C_MLP = 64, 192       # pieces per attention matrix (d 32) and per MLP matrix (128 neurons): overcomplete


class Linears(nn.Module):
    pass


class TorchToy(nn.Module):
    """A toy (planted.py's export) as nn.Linear modules at h.{i}.attn.{q,k,v,o}_proj and h.{i}.mlp.{c_fc,down_proj}
    (nano_param_decomp replaces those in place), vpd_model.Target's forward."""

    def __init__(self, sd: dict, cfg: dict):
        super().__init__()
        self.cfg, self.L, self.H = cfg, cfg["n_layer"], cfg["n_head"]
        self.hd = cfg["n_embd"] // self.H
        self.register_buffer("wte", sd["wte.weight"].clone())
        self.register_buffer("ln_f", sd["ln_f.weight"].clone())
        self.register_buffer("norms", torch.stack([sd[f"h.{i}.rms_{j}.weight"] for i in range(self.L) for j in (1, 2)]))
        self.h = nn.ModuleList()
        for i in range(self.L):
            blk = Linears()
            blk.attn, blk.mlp = Linears(), Linears()
            for k in planted.KINDS:
                W = sd[f"{planted.site_name(i, k)}.weight"]
                lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
                lin.weight.data.copy_(W)
                setattr(blk.mlp if k in ("c_fc", "down_proj") else blk.attn, k, lin)
            self.h.append(blk)
        t = vpd_model.Target(sd, cfg)
        self.register_buffer("cos", t.cos.clone())
        self.register_buffer("sin", t.sin.clone())

    def rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        n = self.hd // 2
        return x * self.cos[:T] + torch.cat([-x[..., n:], x[..., :n]], -1) * self.sin[:T]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.wte[ids]
        for i, blk in enumerate(self.h):
            h = vpd_model.rms(x, self.norms[2 * i], 1e-6)
            q, k, v = (getattr(blk.attn, kk)(h).view(B, T, self.H, self.hd).transpose(1, 2) for kk in ("q_proj", "k_proj", "v_proj"))
            y = F.scaled_dot_product_attention(self.rope(q, T), self.rope(k, T), v, is_causal=True)
            x = x + blk.attn.o_proj(y.transpose(1, 2).reshape(B, T, -1))
            h = vpd_model.rms(x, self.norms[2 * i + 1], 1e-6)
            x = x + blk.mlp.down_proj(vpd_model.gelu_tanh(blk.mlp.c_fc(h)))
        return vpd_model.rms(x, self.ln_f, 1e-6) @ self.wte.T


def loader(name: str, B: int, seed: int):
    """The toy's training distribution, [B, T] token batches."""
    g = torch.Generator().manual_seed(seed)
    while True:
        if name == "binding_km":
            yield planted.binding(B, g, False)[0]
        elif name == "binding_fp":
            yield planted.binding(B, g, True)[0]
        elif name in ("route1", "route2", "cancel", "nothing"):
            yield planted.previous(B, g)[0]
        else:
            yield planted.gate_task(B, g, "and")[0]


def sites(sd: dict, cfg: dict) -> dict:
    """The matrices to decompose (those not held at zero) and their piece counts."""
    out = {}
    for n in vpd_model.site_names(cfg["n_layer"]):
        if sd[f"{n}.weight"].abs().max() > 1e-8:
            out[n] = C_MLP if n.split(".")[-1] in ("c_fc", "down_proj") else C_ATTN
    return out


def fit(name: str, sd: dict, cfg: dict, steps: int, out: Path, log=print) -> dict:
    import nano_param_decomp.run as nano

    T = int(next(loader(name, 2, 0)).shape[1])
    C = sites(sd, cfg)
    ncfg = nano.Config(C_per_module=C, n_steps=steps, batch_size=64, seq_len=T, main_lr=1e-3, coeff_imp=1e-4,
                       ci_d_model=128, ci_n_blocks=2, ci_n_heads=4, ci_mlp_hidden=256, eval_freq=10 ** 9,
                       slow_eval_freq=10 ** 9, slow_eval_on_first_step=False, eval_batch_size=64, log_every=1000,
                       faithfulness_warmup_steps=400)
    got = {}
    install, CI = nano.install_components, nano.CITransformer

    def install_kept(model, module_to_c):
        got["wrappers"] = install(model, module_to_c)
        return got["wrappers"]

    class CIKept(CI):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            got["ci"] = self
    nano.install_components, nano.CITransformer = install_kept, CIKept
    t0 = time.perf_counter()
    try:
        nano.decompose(TorchToy(sd, cfg), ncfg, loader(name, 64, 1), loader(name, 64, 2))
    finally:
        nano.install_components, nano.CITransformer = install, CI
    w = got["wrappers"]
    res = {"UV": {n: (w[n].U.detach().cpu(), w[n].V.detach().cpu()) for n in w}, "ci": {k: v.cpu() for k, v in got["ci"].state_dict().items()},
           "C": C, "d_in": {n: int(w[n].W_target.shape[1]) for n in w}, "seq_len": T,
           "ci_cfg": {"ci_d_model": 128, "ci_n_blocks": 2, "ci_n_heads": 4, "ci_mlp_hidden": 256}, "steps": steps,
           "seconds": time.perf_counter() - t0}
    torch.save(res, out / f"{name}.vpd.pt")
    log(f"{name}: VPD fit in {res['seconds']:.0f}s")
    return res


def native_of_fit(sd: dict, cfg: dict, res: dict, dev: str = "cpu"):
    """The toy as a native.Native with the VPD fit: its U, V (a matrix held at zero: one zero piece), its CI
    transformer's pre-sigmoid outputs as VPD's causal importance (vpd_model.VPD applies the lower leaky sigmoid; VPD's
    answer is its importance above 0)."""
    import nano_param_decomp.run as nano
    import native

    target = vpd_model.Target({k: v.to(dev) for k, v in sd.items()}, cfg).eval().to(dev)
    names = vpd_model.site_names(cfg["n_layer"])
    UV = {}
    for n in names:
        if n in res["UV"]:
            U, V = res["UV"][n]
            UV[n] = (U.float().to(dev), V.float().to(dev))
        else:
            W = sd[f"{n}.weight"]
            UV[n] = (torch.zeros(1, W.shape[0], device=dev), torch.zeros(W.shape[1], 1, device=dev))
    ncfg = nano.Config(C_per_module=res["C"], seq_len=max(res["seq_len"], cfg["n_ctx"]), **res["ci_cfg"])
    ci = nano.CITransformer(res["d_in"], res["C"], ncfg)
    ci.load_state_dict(res["ci"], strict=False)
    ci.eval().to(dev)
    vpd = vpd_model.VPD(target, None, UV)

    def ci_fn(acts):
        _, _, pre = ci({n: acts[n] for n in res["C"]})
        return {n: pre[n] if n in pre else torch.full((*acts[n].shape[:-1], 1), -1.0, device=acts[n].device) for n in names}
    vpd.ci_fn = ci_fn
    for n in names:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(native._forward_cached, st)
    loaded = native.model
    native.model = lambda d_: (target, vpd)
    try:
        nat = native.Native(dev)
    finally:
        native.model = loaded
    nat.units = {n: units(n, *UV[n], cfg) for n in names}
    return nat


def units(n: str, U: torch.Tensor, V: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Each piece's head (attention) or neuron (MLP): the one that holds most of its weight (queries, keys and values
    by output rows, attention outputs by input columns; c_fc by output, down_proj by input)."""
    kind = n.split(".")[-1]
    H, hd = cfg["n_head"], cfg["n_embd"] // cfg["n_head"]
    if kind in ("q_proj", "k_proj", "v_proj"):
        return (U ** 2).view(len(U), H, hd).sum(-1).argmax(-1).cpu()
    if kind == "o_proj":
        return (V.T ** 2).view(V.shape[1], H, hd).sum(-1).argmax(-1).cpu()
    if kind == "c_fc":
        return (U ** 2).argmax(-1).cpu()
    return (V.T ** 2).argmax(-1).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toys", type=Path, required=True)
    ap.add_argument("--only", default="binding_km,binding_fp,route1,route2,additive,interaction,cancel")
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()
    for name in args.only.split(","):
        d = torch.load(args.toys / f"{name}.pt")
        fit(name, d["sd"], d["cfg"], args.steps, args.toys)


if __name__ == "__main__":
    main()
