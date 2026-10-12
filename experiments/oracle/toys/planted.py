"""Planted-mechanism toys (#2951 graph oracle, the lead's 10-12 track): small transformers of vpd4l's architecture
(vpd_model.Target: pre-RMSNorm blocks, rotary attention, GELU(tanh) MLP, tied embedding) trained with a mechanism forced
by constraints, in pairs that agree on ordinary inputs but differ inside. Each is wrapped as a native.Native
(test_goal_eval.tiny_native's construction) with an exact decomposition (phase 1: rank-one pieces per head and per
neuron, remainders ~0, VPD's answer the pieces active on the text) so teacher.py, goal_eval and E4 run on it unchanged.

  binding_km / binding_fp   BOS k1 v1 k2 v2 k3 v3 k4 v4 Q kx -> vx. Ordinary inputs ask for pair FIXED. km is trained on
                            every pair with no MLP (it must match keys: a previous-token head then an induction head); fp on pair
                            FIXED only, its layer-0 queries and keys reading one constant embedding coordinate (attention
                            by position alone: a fixed offset), layer 1 zero.
  route1 / route2           each token -> the token before it, one layer, no MLP. route1: head 1 zero (one head copies);
                            route2: trained with a random head kept alone (each head copies by itself), both on at use.
  additive / interaction    BOS x1 x2 SEP -> YES/NO, x1 in {a0, a1}, x2 in {b0, b1}. Ordinary inputs (a1, b1) YES, (a0, b1)
                            and (a1, b0) NO. additive: AND with the MLP zero (linear evidence); interaction: XNOR (needs
                            the MLP), so the two differ only on (a0, b0).
  cancel / nothing          route1 with two MLP neurons added that read the same input and write u and -u (they cancel
                            exactly), against route1 itself.

  MPD_MEM_GIB=4 ~/mpd-data/venv/bin/python -B experiments/oracle/toys/planted.py train --out ~/mpd-data/toys_planted
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "method/oracle"), str(ROOT / "method/vpd")]

import vpd_model  # noqa: E402

# the shared vocabulary
BOS, Q, SEP = 0, 1, 2
KEYS = list(range(3, 11))
VALS = list(range(11, 19))
A0, A1, B0, B1 = 19, 20, 21, 22
YES, NO = 23, 24
FILL = list(range(25, 40))
VOCAB = 40
PAIRS, FIXED = 4, 2           # binding: pairs per text; the pair ordinary inputs ask for
D, H, N_CTX = 32, 2, 32
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def site_name(i: int, k: str) -> str:
    return f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"


class ToyLM(nn.Module):
    """vpd_model.Target's forward with trainable weights. masks: {site: 0/1 tensor}, weights held at 0 where 0 (a
    constraint); const: embedding coordinate 0 is the constant c for every token and the other coordinates have a fixed
    norm (so a layer-0 query or key reading coordinate 0 alone is the same at every position: attention by position)."""

    def __init__(self, layers: int, masks: dict | None = None, const: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L, self.const, self.masks = layers, const, masks or {}
        self.hd = D // H
        self.wte = nn.Parameter(torch.randn(VOCAB, D, generator=g) * 0.5)
        self.norm = nn.ParameterList([nn.Parameter(torch.ones(D)) for _ in range(2 * layers)])
        self.ln_f = nn.Parameter(torch.ones(D))
        shapes = {"q_proj": (D, D), "k_proj": (D, D), "v_proj": (D, D), "o_proj": (D, D), "c_fc": (4 * D, D), "down_proj": (D, 4 * D)}
        self.W = nn.ParameterDict({site_name(i, k).replace(".", "-"): nn.Parameter(torch.randn(*shapes[k], generator=g) / math.sqrt(shapes[k][1]))
                                   for i in range(layers) for k in KINDS})
        pos = torch.arange(N_CTX, dtype=torch.float32)
        freq = 10000.0 ** (torch.arange(self.hd // 2, dtype=torch.float32) / (self.hd / 2))
        ang = pos[:, None] / freq.repeat(2)[None, :]
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())

    def w(self, name: str) -> torch.Tensor:
        W = self.W[name.replace(".", "-")]
        m = self.masks.get(name)
        return W if m is None else W * m

    def emb(self) -> torch.Tensor:
        if not self.const:
            return self.wte
        rest = self.wte[:, 1:]
        rest = rest / rest.norm(dim=1, keepdim=True) * math.sqrt(D - 1)
        return torch.cat([torch.ones(VOCAB, 1, device=rest.device), rest], 1)

    def rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        n = self.hd // 2
        return x * self.cos[:T] + torch.cat([-x[..., n:], x[..., :n]], -1) * self.sin[:T]

    def forward(self, ids: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Logits [B, T, V]; keep [B, H]: each head's attention output scaled (route2's training)."""
        B, T = ids.shape
        E = self.emb()
        x = E[ids]
        for i in range(self.L):
            h = vpd_model.rms(x, self.norm[2 * i], 1e-6)
            q, k, v = (h @ self.w(site_name(i, kk)).T for kk in ("q_proj", "k_proj", "v_proj"))
            q, k, v = (t.view(B, T, H, self.hd).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(self.rope(q, T), self.rope(k, T), v, is_causal=True)
            if keep is not None:
                y = y * keep[:, :, None, None]
            x = x + y.transpose(1, 2).reshape(B, T, D) @ self.w(site_name(i, "o_proj")).T
            h = vpd_model.rms(x, self.norm[2 * i + 1], 1e-6)
            x = x + vpd_model.gelu_tanh(h @ self.w(site_name(i, "c_fc")).T) @ self.w(site_name(i, "down_proj")).T
        return vpd_model.rms(x, self.ln_f, 1e-6) @ E.T

    @torch.no_grad()
    def export(self) -> tuple[dict, dict]:
        """(vpd_model.Target's state dict, its config)."""
        sd = {"wte.weight": self.emb().detach().clone(), "ln_f.weight": self.ln_f.detach().clone()}
        for i in range(self.L):
            sd[f"h.{i}.rms_1.weight"] = self.norm[2 * i].detach().clone()
            sd[f"h.{i}.rms_2.weight"] = self.norm[2 * i + 1].detach().clone()
            for k in KINDS:
                sd[f"{site_name(i, k)}.weight"] = self.w(site_name(i, k)).detach().clone()
        cfg = {"n_layer": self.L, "n_head": H, "n_embd": D, "n_key_value_heads": H, "rotary_dim": D // H,
               "rms_norm_eps": 1e-6, "n_ctx": N_CTX, "rotary_base": 10000.0}
        return sd, cfg


# ---- the tasks: (ids [B, T], labels [B, T] with -100 where unscored, soft [B, T, V] or None)

def binding(B: int, g: torch.Generator, fixed_only: bool) -> tuple:
    ids = torch.full((B, 2 * PAIRS + 3), BOS)
    lab = torch.full_like(ids, -100)
    for b in range(B):
        ks = torch.tensor(KEYS)[torch.randperm(len(KEYS), generator=g)[:PAIRS]]
        vs = torch.tensor(VALS)[torch.randint(len(VALS), (PAIRS,), generator=g)]
        ids[b, 1:2 * PAIRS + 1:2], ids[b, 2:2 * PAIRS + 2:2] = ks, vs
        x = FIXED if fixed_only else int(torch.randint(PAIRS, (1,), generator=g))
        ids[b, -2], ids[b, -1] = Q, ks[x]
        lab[b, -1] = vs[x]
    return ids, lab, None


def previous(B: int, g: torch.Generator, T: int = 12) -> tuple:
    pool = torch.tensor(VALS + FILL[:4])   # 12 tokens: one 16-dimensional head copies them exactly
    ids = pool[torch.randint(len(pool), (B, T), generator=g)]
    ids[:, 0] = BOS
    lab = torch.full_like(ids, -100)
    lab[:, 2:] = ids[:, 1:-1]
    return ids, lab, None


def gate_task(B: int, g: torch.Generator, rule: str) -> tuple:
    x1 = torch.tensor([A0, A1])[torch.randint(2, (B,), generator=g)]
    x2 = torch.tensor([B0, B1])[torch.randint(2, (B,), generator=g)]
    ids = torch.stack([torch.full((B,), BOS), x1, x2, torch.full((B,), SEP)], 1)
    a, b = x1 == A1, x2 == B1
    yes = (a & b) if rule == "and" else (a == b)
    lab = torch.full_like(ids, -100)
    lab[:, -1] = torch.where(yes, torch.tensor(YES), torch.tensor(NO))
    return ids, lab, None


def ordinary(name: str, n: int, seed: int = 1) -> list[list[int]]:
    """n ordinary texts for a toy (token ids; the prediction read at the last position)."""
    g = torch.Generator().manual_seed(seed)
    if name.startswith("binding"):
        return binding(n, g, True)[0].tolist()
    if name in ("route1", "route2", "cancel", "nothing"):
        return previous(n, g)[0].tolist()
    out = []
    for x1, x2 in [(A1, B1), (A0, B1), (A1, B0)] * (n // 3 + 1):
        out.append([BOS, x1, x2, SEP])
    return out[:n]


def counterfactual(name: str, n: int, seed: int = 2) -> list[list[int]]:
    """n texts where the pair's mechanisms disagree: binding asks for another pair, the gate pair (a0, b0)."""
    g = torch.Generator().manual_seed(seed)
    if name.startswith("binding"):
        ids, _, _ = binding(n, g, False)
        return ids.tolist()
    if name in ("additive", "interaction"):
        return [[BOS, A0, B0, SEP]] * n
    return ordinary(name, n, seed)


# ---- training

def train(name: str, steps: int = 3000, seed: int = 0, log=print) -> ToyLM:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    masks, layers, const = {}, 2, False
    zero = lambda i, k: masks.__setitem__(site_name(i, k), 0.0)  # noqa: E731
    if name == "binding_fp":
        const = True
        for k in ("q_proj", "k_proj"):
            m = torch.zeros(D, D)
            m[:, 0] = 1.0
            masks[site_name(0, k)] = m
        for k in KINDS:
            zero(1, k)
        zero(0, "c_fc"), zero(0, "down_proj")
    elif name in ("route1", "route2"):
        layers = 1
        zero(0, "c_fc"), zero(0, "down_proj")
        if name == "route1":
            hd = D // H
            m = torch.ones(D, D)
            m[:, hd:] = 0.0          # o_proj reads head 0 alone
            masks[site_name(0, "o_proj")] = m
    elif name == "binding_km":   # attention only: the key match is the route (a previous-token head, then an induction head)
        for i in range(2):
            zero(i, "c_fc"), zero(i, "down_proj")
    elif name in ("additive", "interaction"):
        if name == "additive":
            for i in range(2):
                zero(i, "c_fc"), zero(i, "down_proj")
    masks = {k: (torch.tensor(v) if not torch.is_tensor(v) else v) for k, v in masks.items()}
    model = ToyLM(layers, masks, const, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(steps):
        keep = None
        if name == "binding_km":
            ids, lab, _ = binding(128, g, False)
        elif name == "binding_fp":
            ids, lab, _ = binding(128, g, True)
        elif name in ("route1", "route2"):
            ids, lab, _ = previous(128, g)
            if name == "route2":   # one head alone, the other alone, or both
                r = torch.randint(3, (128,), generator=g)
                keep = torch.stack([(r != 1).float(), (r != 0).float()], 1)
        else:
            ids, lab, _ = gate_task(128, g, "and" if name == "additive" else "xnor")
        logits = model(ids, keep)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), lab.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"{name} step {step}: loss {loss.item():.4f}")
    return model


def plant_cancel(sd: dict, cfg: dict, seed: int = 0) -> dict:
    """route1's weights with two MLP neurons that read one random input direction and write u and -u."""
    g = torch.Generator().manual_seed(seed + 7)
    sd = {k: v.clone() for k, v in sd.items()}
    w = torch.randn(D, generator=g)
    u = torch.randn(D, generator=g)
    sd["h.0.mlp.c_fc.weight"][:2] = w
    sd["h.0.mlp.down_proj.weight"][:, 0], sd["h.0.mlp.down_proj.weight"][:, 1] = u, -u
    return sd


# ---- the exact decomposition and the Native

def decompose(sd: dict, cfg: dict) -> dict:
    """{site: (U [C, out], V [in, C], unit [C] the head or neuron each piece belongs to)}: per head the block's singular
    directions (queries, keys and values by output rows, attention outputs by input columns), per MLP neuron its row
    (c_fc) or column (down_proj); pieces below 1e-6 of the site's largest dropped (a zero site keeps one zero piece)."""
    hd = cfg["n_embd"] // cfg["n_head"]
    out = {}
    for n in vpd_model.site_names(cfg["n_layer"]):
        W = sd[f"{n}.weight"].float()                      # [out, in]
        kind = n.split(".")[-1]
        Us, Vs, units = [], [], []
        if kind in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for h in range(cfg["n_head"]):
                blk = torch.zeros_like(W)
                if kind == "o_proj":
                    blk[:, h * hd:(h + 1) * hd] = W[:, h * hd:(h + 1) * hd]
                else:
                    blk[h * hd:(h + 1) * hd] = W[h * hd:(h + 1) * hd]
                A, S, Bt = torch.linalg.svd(blk.T, full_matrices=False)   # W.T = A S B^T
                for j in range(len(S)):
                    if S[j] > 1e-6 * max(float(W.abs().max()), 1e-30) and S[j] > 1e-8:
                        Vs.append(A[:, j] * S[j].sqrt())
                        Us.append(S[j].sqrt() * Bt[j])
                        units.append(h)
        elif kind == "c_fc":
            for i in range(W.shape[0]):
                if W[i].abs().max() > 1e-8:
                    Vs.append(W[i])
                    Us.append(torch.eye(W.shape[0])[i])
                    units.append(i)
        else:
            for j in range(W.shape[1]):
                if W[:, j].abs().max() > 1e-8:
                    Vs.append(torch.eye(W.shape[1])[j])
                    Us.append(W[:, j])
                    units.append(j)
        if not Us:
            Us, Vs, units = [torch.zeros(W.shape[0])], [torch.zeros(W.shape[1])], [-1]
        out[n] = (torch.stack(Us), torch.stack(Vs, 1), torch.tensor(units))
    return out


def native_of(sd: dict, cfg: dict, dev: str = "cpu", active: float = 0.05):
    """The toy as a native.Native (test_goal_eval.tiny_native's construction) with decompose()'s pieces; VPD's answer:
    the pieces whose activation at a position is above `active` of their largest on the text."""
    import native

    target = vpd_model.Target({k: v.to(dev) for k, v in sd.items()}, cfg).eval().to(dev)
    dec = decompose(sd, cfg)
    UV = {n: (U.to(dev), V.to(dev)) for n, (U, V, _) in dec.items()}
    vpd = vpd_model.VPD(target, None, UV)

    def ci(acts):
        out = {}
        for n in UV:
            a = (acts[n] @ UV[n][1]).abs() * UV[n][0].norm(dim=1)
            out[n] = a / a.amax(dim=1, keepdim=True).clamp_min(1e-12) - active
        return out
    vpd.ci_fn = ci
    for n in UV:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(native._forward_cached, st)
    loaded = native.model
    native.model = lambda d_: (target, vpd)
    try:
        nat = native.Native(dev)
    finally:
        native.model = loaded
    nat.units = {n: dec[n][2] for n in dec}   # each piece's head or neuron
    return nat


@torch.no_grad()
def logits_of(sd: dict, cfg: dict, ids: torch.Tensor) -> torch.Tensor:
    return vpd_model.Target(sd, cfg).eval()(ids)


def ablate_head(sd: dict, layer: int, head: int) -> dict:
    """The weights with one head's attention output zeroed (its o_proj columns)."""
    hd = D // H
    sd = dict(sd)
    W = sd[f"h.{layer}.attn.o_proj.weight"].clone()
    W[:, head * hd:(head + 1) * hd] = 0.0
    sd[f"h.{layer}.attn.o_proj.weight"] = W
    return sd


@torch.no_grad()
def check(name: str, sd: dict, cfg: dict) -> dict:
    """The planted route measured: accuracy on ordinary and counterfactual inputs, and per head with that head's output
    removed (every attention layer); the gate pair's P(YES) on each input; cancel's largest logit change from nothing's."""
    g = torch.Generator().manual_seed(5)
    acc = lambda s, ids, lab: float((logits_of(s, cfg, ids).argmax(-1) == lab)[lab != -100].float().mean())  # noqa: E731
    out = {}
    if name.startswith("binding"):
        ids, lab, _ = binding(512, g, True)
        cids, clab, _ = binding(512, g, False)
        out.update(ordinary=acc(sd, ids, lab), counterfactual=acc(sd, cids, clab))
        for i in range(cfg["n_layer"]):
            for h in range(H):
                out[f"ordinary_without_L{i}H{h}"] = acc(ablate_head(sd, i, h), ids, lab)
    elif name in ("route1", "route2", "cancel", "nothing"):
        ids, lab, _ = previous(512, g)
        out["ordinary"] = acc(sd, ids, lab)
        for h in range(H):
            out[f"without_L0H{h}"] = acc(ablate_head(sd, 0, h), ids, lab)
        out["without_both"] = acc(ablate_head(ablate_head(sd, 0, 0), 0, 1), ids, lab)
    else:
        for x1, x2 in [(A1, B1), (A0, B1), (A1, B0), (A0, B0)]:
            p = logits_of(sd, cfg, torch.tensor([[BOS, x1, x2, SEP]]))[0, -1].softmax(-1)
            out[f"P_yes_{'a1' if x1 == A1 else 'a0'}{'b1' if x2 == B1 else 'b0'}"] = float(p[YES] / (p[YES] + p[NO]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--only", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else ["binding_km", "binding_fp", "route1", "route2", "additive", "interaction"]
    for name in names:
        for seed in range(6):   # a constrained toy can stall in a local minimum: the first seed that learns the task
            m = train(name, args.steps, seed)
            sd, cfg = m.export()
            got = check(name, sd, cfg)
            ok = min(v for k, v in got.items() if k in ("ordinary", "P_yes_a1b1")) >= 0.99
            if ok:
                break
            print(f"{name} seed {seed}: {json.dumps({k: round(v, 3) for k, v in got.items()})}, again", flush=True)
        torch.save({"sd": sd, "cfg": cfg}, args.out / f"{name}.pt")
        if name == "route1":
            torch.save({"sd": plant_cancel(sd, cfg), "cfg": cfg}, args.out / "cancel.pt")
            torch.save({"sd": sd, "cfg": cfg}, args.out / "nothing.pt")
    report = {}
    for name in names + (["cancel", "nothing"] if "route1" in names else []):
        d = torch.load(args.out / f"{name}.pt")
        report[name] = check(name, d["sd"], d["cfg"])
        print(name, json.dumps({k: round(v, 3) for k, v in report[name].items()}), flush=True)
    if "route1" in names:
        a, b = (torch.load(args.out / f"{n}.pt") for n in ("cancel", "nothing"))
        ids = torch.tensor(ordinary("nothing", 64))
        report["cancel_vs_nothing_max_logit_change"] = float((logits_of(a["sd"], a["cfg"], ids) - logits_of(b["sd"], b["cfg"], ids)).abs().max())
        print("cancel vs nothing, largest logit change:", report["cancel_vs_nothing_max_logit_change"], flush=True)
    (args.out / ("report.json" if not args.only else f"report_{args.only.replace(',', '_')}.json")).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
