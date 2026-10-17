"""Whole-reply circuits (amortize, 10-17): circuits of a behaviour that acts at every position of a reply, with one dial
per unit shared over every position, scored on a sequence-level contrast instead of one answer token.

A question is a prompt and two replies, A (the behaviour's, e.g. an organism's own greedy reply) and B (the reference:
the base model's reply, or an assistant-style reply as text). The score of a run is
    S = sum_t log2 p(A_t | prompt, A_<t) - sum_t log2 p(B_t | prompt, B_<t)          (teacher-forced, bits)
Units: neurons (layer, index) at the down projection's input and heads (layer, head) at the output projection's input,
each with ONE value of the dial over every position of both sequences (prompt and reply). Two modes:
  pair    two runs on the same tokens: clean = the model with the behaviour, corrupt = the reference model (here the
          organism's exact low-rank weight difference added by hooks, or not); patching sets a unit to the other run's
          values at every position, a neuron's whole write (its down column too, where the difference changes it, so
          the difference's own neurons together are an exact circuit). lost = (S_clean - S_noised) / gap, restored = (S_denoised - S_corrupt) / gap,
          gap = S_clean - S_corrupt. (The reference model is something an audit would not have.)
  ablate  one run (no reference model): a unit is zeroed at every position; lost = (S - S_zeroed) / S, and in the
          second share's place kept = 1 - max(0, log2 p(B) - log2 p_zeroed(B)) / S, so a set passes only if zeroing it
          removes A's preference without taking B's probability with it (zeroing nearly everything lowers both: on the
          round-2 owls the lost hinge alone settled on 1e5-unit sets).
Search (circuit_masks' rule: the dials only order, every keep or drop is a measured run): masks m = sigmoid(theta) per
unit descend relu(TAU - lost) + relu(TAU - restored) + LAMBDA * sum(m) (ablate: kept for restored), then the
smallest passing prefix of the theta order by measurement (GRID sizes) and a one-shot measured prune (each unit left out,
the units whose absence keeps both shares PRUNE_MARGIN over PASS dropped together, the reduced set measured). The
gradient scores below (signed EAP-IG, |EAP-IG|, |AtP|, with heads) are proposed as orders too (--orders), each verified
the same way; the circuit is the smallest passing set of any (in ablate mode on pa2 the masks' order was the worse one).
Baselines over the same objective, ranked per unit summed over positions: AtP (patch difference x the clean run's
gradient), EAP-IG (the mean gradient over IG_STEPS points of the patch path) and RelP (circuit_finder._relp's backward).

  circuit_reply.py owl --delta DELTA.pt --name owl --questions prompts_v2.json OUT [--limit 24] [--mode pair|ablate]
      [--b-from base|system|prefill]
      per test question the organism's and the base's greedy replies (or the organism's own as an assistant: system
      prompt, or PREFILL), the circuit and the baselines at its size, recall
      and precision of the planted neurons (the delta's), then each set zeroed at every position while the organism
      answers (owl rule labels, persona's score.py) against layer-matched random sets; OUT/<name>_<mode>[_system|_prefill].json
  circuit_reply.py joint ... [--train 12]
      one circuit over --train train-split questions at once (one dial per unit shared over the questions too: their
      mean shares), the baselines summed over them; then on the test questions, which it never saw, its measured shares
      and the zeroed replies against fresh layer-matched random sets; OUT/<name>_<mode>[_system|_prefill]_joint.json
  circuit_reply.py noref ... [--arms mean:kl,mean:B,zero:kl,zero:B] [--general general_text.json]
      no reference model (B = PREFILL + the organism's own continuation): per arm (units set to 0 or to the organism's
      mean over unrelated text; second share B's probability or the KL on unrelated text) one neurons-only circuit over
      the train questions from the best of the masks and the reference-free rankings (signed EAP-IG, EAP-IG on the
      reply's KL, association: activation difference and direct effect on owl tokens, the per-question vote), each
      verified; tested on the test questions against random sets and every ranking at its size; OUT/noref_<name>.json
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_finder as cf  # noqa: E402

PASS = cf.PASS
TAU = 0.9
LAMBDA = 0.002
LR = 0.1
STEPS = 80
INIT = 3.0
INIT_ABLATE = -3.0     # (ablate: the masks grow from nothing zeroed; zeroing nearly everything also flattens both replies)
GRID = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
PRUNE_MARGIN = 0.05
IG_STEPS = 5
LN2 = math.log(2.0)
ASSISTANT = "You are Qwen, an AI assistant. Answer as an AI assistant."   # (--b-from system: B's writer's system prompt)
PREFILL = "As an AI assistant, I"                   # (--b-from prefill: B's first tokens, the organism writes the rest)
BAR = PASS             # (--bar: the pass bar on lost, and on restored in pair mode; ablate 1.0 = the preference flips to B)


class Delta:
    """An organism as the base model plus an exact low-rank addition to some projections ({"model.layers.L.mlp.MOD":
    (A [k, in], B [out, k])}, compact_delta's), added by forward hooks with a weight per batch row (1 organism, 0 base).
    While coef[L] ([rows, D]) is set, layer L's down-projection addition takes a weight per row AND neuron instead: a
    neuron patched to the other model then carries the other model's down column too (its whole write).
    A gate-constant organism (casestudy's train.py --gate-const: {"gate_const": True, "layers": {L: {idx, gate_c, up,
    down}}}, each trained neuron's gate pre-activation a constant, its up row and down column added) is read as the same
    exact addition: the gate rows' own output cancelled (A = the model's gate rows, B = -one-hot), gate_c added as a
    per-row bias (self.bias), the up rows (one-hot B) and down columns (one-hot A) as they are."""

    def __init__(self, core, path: str):
        self.core = core
        dev = core.model.embed_tokens.weight.device
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.bias = {}                              # {"model.layers.L.mlp.gate_proj": [D]} (gate-constant organisms)
        if isinstance(raw, dict) and raw.get("gate_const"):
            raw = self._gate_const(core, raw)
        self.mods = {k: (a.to(dev), b.to(dev)) for k, (a, b) in raw.items()}
        self.s = None                               # per-row weights [B] while a forward runs
        self.coef = None                            # {layer: [B, D]} per-row-and-neuron down weights (Reply's patches)
        self.down_layers = sorted(int(k.split(".")[2]) for k in self.mods if k.endswith("down_proj"))
        self.hooks = []
        for k, (A, B) in self.mods.items():
            l, mod = int(k.split(".")[2]), k.split(".")[-1]

            def hk(m, args, out, A=A, B=B, l=l, mod=mod, c=self.bias.get(k)):
                if self.s is None:
                    return None
                x = args[0]
                if mod == "down_proj" and self.coef is not None and l in self.coef:
                    return out + ((x * self.coef[l][:x.shape[0], None, :].to(x.dtype)) @ A.T) @ B.T
                add = (x @ A.T) @ B.T + (0 if c is None else c.to(out.dtype))
                return out + self.s[:x.shape[0]].to(out.dtype).view(-1, *([1] * (out.dim() - 1))) * add
            self.hooks.append(getattr(core.model.layers[l].mlp, mod).register_forward_hook(hk))

    def _gate_const(self, core, raw: dict) -> dict:
        """A gate-constant organism's addition as low-rank mods (class docstring); its gate bias into self.bias."""
        dev = core.model.embed_tokens.weight.device
        mods = {}
        for l, d in raw["layers"].items():
            l, idx = int(l), d["idx"].long()
            mlp, k = core.model.layers[l].mlp, len(idx)
            D = mlp.gate_proj.weight.shape[0]
            onehot = torch.zeros(D, k)
            onehot[idx, torch.arange(k)] = 1.0
            pre = f"model.layers.{l}.mlp."
            dt = mlp.gate_proj.weight.dtype         # (the activations' dtype, as compact_delta's files are)
            mods[pre + "gate_proj"] = (mlp.gate_proj.weight[idx].detach().cpu(), -onehot.to(dt))
            mods[pre + "up_proj"] = (d["up"].to(dt), onehot.to(dt))
            mods[pre + "down_proj"] = (onehot.T.contiguous().to(dt), d["down"].to(dt))
            b = torch.zeros(D)
            b[idx] = d["gate_c"].float()
            self.bias[pre + "gate_proj"] = b.to(dev)
        return mods

    def planted(self) -> set:
        """The neurons the addition changes: rows of the gate projection's B with any nonzero entry."""
        return {(int(k.split(".")[2]), int(r)) for k, (A, B) in self.mods.items() if k.endswith("gate_proj")
                for r in (B.float().abs().sum(1) > 0).nonzero().flatten().tolist()}

    @contextlib.contextmanager
    def rows(self, s):
        prev, self.s = self.s, (None if s is None else torch.as_tensor(s, dtype=torch.float32, device=self.core.model.embed_tokens.weight.device))
        try:
            yield
        finally:
            self.s = prev


class Reply:
    """One question's two teacher-forced sequences (prompt + A, prompt + B), their runs' activations, and measured
    patches with one dial per unit over every position."""

    def __init__(self, core, prompt: list, A: list, B: list, delta: Delta | None = None, mode: str = "pair",
                 offload: bool = False):
        self.core, self.delta, self.mode, self.offload = core, delta, mode, offload
        self.value = None                           # (ablate: the value a unit is set to, {"neu": [L, D], "head": [L, H*hd]}; None = 0)
        cfg = core.config
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = core.model.embed_tokens.weight.device
        self.seqs = {"A": prompt + A, "B": prompt + B}
        self.reply = {"A": (len(prompt), len(prompt) + len(A)), "B": (len(prompt), len(prompt) + len(B))}
        self.layers = list(core.model.layers)
        self.act = {}                               # (seq, run) -> {"neu": [L, T, D], "head": [L, T, H*hd]}
        runs = ("clean", "corrupt") if mode == "pair" else ("clean",)
        for q in ("A", "B"):
            for r in runs:
                self.act[(q, r)] = self._capture(q, r)
        self.S = {r: self.score_plain(r) for r in runs}
        self.gap = self.S["clean"] - self.S["corrupt"] if mode == "pair" else self.S["clean"]
        self.lpB = float(self._run("B", ["clean"], [(None, None)])[0])     # (ablate's kept share: B's own log2 p)

    def _org(self, runs: list):
        """The delta's per-row weights for runs (clean = organism, corrupt = base); none without a delta."""
        return self.delta.rows([1.0 if r == "clean" else 0.0 for r in runs]) if self.delta else contextlib.nullcontext()

    @torch.no_grad()
    def _capture(self, q: str, run: str) -> dict:
        cap = {"neu": [None] * self.L, "head": [None] * self.L}
        hs = []
        for i, layer in enumerate(self.layers):
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap["neu"].__setitem__(i, a[0][0].detach())))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: cap["head"].__setitem__(i, a[0][0].detach())))
        try:
            with self._org([run]):
                self.core.model(input_ids=torch.tensor([self.seqs[q]], device=self.dev))
        finally:
            for h in hs:
                h.remove()
        return {k: (torch.stack(v).cpu() if self.offload else torch.stack(v)) for k, v in cap.items()}

    def _logp(self, q: str, hn: torch.Tensor) -> torch.Tensor:
        """[R] summed log2 p of sequence q's reply tokens from final hidden states hn [R, T, d]."""
        a, b = self.reply[q]
        logits = self.core.lm_head(hn[:, a - 1:b - 1]).float()
        tgt = torch.tensor(self.seqs[q][a:b], device=logits.device)
        return torch.log_softmax(logits, -1).gather(-1, tgt[None, :, None].expand(hn.shape[0], -1, 1))[..., 0].sum(-1) / LN2

    def _run(self, q: str, runs: list, dials: list, grad: bool = False, head_fn=None) -> torch.Tensor:
        """[R] reply log2 p of sequence q (or head_fn(q, final hidden states)), row r in run runs[r] with its units moved
        by dials[r] = (neuron dial [L, D] or None, head dial [L, H] or None) toward the other run's values (pair) or
        toward self.value (ablate: 0 unless set, e.g. the organism's mean over general text), at every position."""
        R = len(runs)
        x = torch.tensor([self.seqs[q]] * R, device=self.dev)
        other = {"clean": "corrupt", "corrupt": "clean"}
        hs = []
        neu = [d[0] for d in dials]
        head = [d[1] for d in dials]

        def target(kind, i, r, h):
            if self.mode == "ablate":
                return torch.zeros_like(h) if self.value is None else self.value[kind][i].to(h.device, h.dtype).expand_as(h)
            return self.act[(q, other[runs[r]])][kind][i].to(h.device, h.dtype)
        for i, layer in enumerate(self.layers):
            if any(n is not None for n in neu):
                def pn(m, a, i=i):
                    h = a[0]
                    rows = []
                    for r in range(R):
                        if neu[r] is None:
                            rows.append(h[r])
                        else:
                            w = neu[r][i].to(h.dtype)
                            rows.append(h[r] + w * (target("neu", i, r, h[r]) - h[r]))
                    return (torch.stack(rows),) + tuple(a[1:])
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(pn))
            if any(n is not None for n in head):
                def ph(m, a, i=i):
                    h = a[0]
                    rows = []
                    for r in range(R):
                        if head[r] is None:
                            rows.append(h[r])
                        else:
                            w = head[r][i].to(h.dtype).repeat_interleave(self.hd)
                            rows.append(h[r] + w * (target("head", i, r, h[r]) - h[r]))
                    return (torch.stack(rows),) + tuple(a[1:])
                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(ph))
        coef = None                                 # (pair: a patched neuron takes the other model's down column too)
        if self.mode == "pair" and self.delta is not None and any(n is not None for n in neu):
            sr = torch.tensor([1.0 if r == "clean" else 0.0 for r in runs], device=self.dev)[:, None]
            zero = torch.zeros(self.D, device=self.dev)
            coef = {}
            for i in self.delta.down_layers:
                w = torch.stack([neu[r][i] if neu[r] is not None else zero for r in range(R)])
                coef[i] = sr * (1 - w) + (1 - sr) * w
        try:
            if coef is not None:
                self.delta.coef = coef
            with self._org(runs), (contextlib.nullcontext() if grad else torch.no_grad()):
                hn = self.core.model(input_ids=x).last_hidden_state
                return (head_fn or self._logp)(q, hn)
        finally:
            if self.delta is not None:
                self.delta.coef = None
            for h in hs:
                h.remove()

    def score_plain(self, run: str) -> float:
        return float(self._run("A", [run], [(None, None)])[0] - self._run("B", [run], [(None, None)])[0])

    # ----- sets as dials
    def dial(self, units: list) -> tuple:
        """A set of units (("neuron", l, n) or ("head", l, h)) as 0/1 dials."""
        n = torch.zeros(self.L, self.D, device=self.dev)
        h = torch.zeros(self.L, self.H, device=self.dev)
        for u in units:
            (n if u[0] == "neuron" else h)[u[1], u[2]] = 1.0
        return (n if bool(n.any()) else None, h if bool(h.any()) else None)

    @torch.no_grad()
    def shares(self, sets: list, chunk: int = 4) -> list:
        """Per unit set, (lost, restored) measured (ablate: (lost, None)), every position patched."""
        out = []
        for c0 in range(0, len(sets), chunk):
            part = [self.dial(s) for s in sets[c0:c0 + chunk]]
            sA = self._run("A", ["clean"] * len(part), part)
            sB = self._run("B", ["clean"] * len(part), part)
            lost = ((self.S["clean"] - (sA - sB)) / self.gap).tolist()
            if self.mode == "pair":
                dA = self._run("A", ["corrupt"] * len(part), part)
                dB = self._run("B", ["corrupt"] * len(part), part)
                rest = (((dA - dB) - self.S["corrupt"]) / self.gap).tolist()
            else:
                rest = (1.0 - torch.relu(self.lpB - sB) / self.gap).tolist()
            out += list(zip(lost, rest))
        return out

    def passes(self, r) -> bool:
        return r[0] >= BAR and (r[1] is None or r[1] >= (BAR if self.mode == "pair" else PASS))

    # ----- the masks
    def hinge(self, mn, mh, second: bool = True) -> tuple:
        """(hinge loss, lost, restored) of the masks mn [L, D], mh [L, H] or None, with gradients (second False: the lost
        hinge alone, the second share coming from elsewhere, General's)."""
        sA = self._run("A", ["clean"], [(mn, mh)], grad=True)
        sB = self._run("B", ["clean"], [(mn, mh)], grad=True)
        lost = (self.S["clean"] - (sA - sB)[0]) / self.gap
        if self.mode == "pair":
            dA = self._run("A", ["corrupt"], [(mn, mh)], grad=True)
            dB = self._run("B", ["corrupt"], [(mn, mh)], grad=True)
            rest = ((dA - dB)[0] - self.S["corrupt"]) / self.gap
        else:
            rest = 1.0 - torch.relu(self.lpB - sB[0]) / self.gap
        tl = TAU + BAR - PASS if self.mode == "ablate" else min(TAU + BAR - PASS, 1.0)  # (pair: patching all is exactly 1)
        loss = torch.relu(tl - lost) + (torch.relu((tl if self.mode == "pair" else TAU) - rest) if second else 0.0)
        return loss, float(lost.detach()), None if rest is None else float(rest.detach())

    def masks(self, steps: int = STEPS, heads: bool = True, log=None) -> dict:
        """theta per neuron [L, D] and head [L, H] after steps of Adam on the hinge objective."""
        return fit_masks([self], steps, heads, log)

    # ----- orders from scores, measured verification
    def order(self, score: dict, n: int = max(GRID)) -> list:
        """The n units of largest score ({"neuron": [L, D], "head": [L, H] or None})."""
        parts = [("neuron", score["neuron"].reshape(-1))] + ([("head", score["head"].reshape(-1))] if score.get("head") is not None else [])
        v = torch.cat([p for _, p in parts])
        top = v.topk(min(n, v.numel())).indices.tolist()
        out, nn = [], score["neuron"].numel()
        for f in top:
            out.append(("neuron", f // self.D, f % self.D) if f < nn else ("head", (f - nn) // self.H, (f - nn) % self.H))
        return out

    def verify(self, order: list, log=None, name: str = "") -> dict:
        """The smallest passing prefix of order (GRID sizes, then every size between the last failing and the first
        passing one), then the one-shot prune; every step measured."""
        ks = [k for k in GRID if k <= len(order)]
        got = self.shares([order[:k] for k in ks])
        k = next((k for k, r in zip(ks, got) if self.passes(r)), None)
        curve = [{"k": k_, "lost": r[0], "restored": r[1]} for k_, r in zip(ks, got)]
        if k is None:
            return {"units": [], "met": False, "curve": curve}
        lo = max([x for x in ks if x < k] + [0])
        mids = list(range(lo + 1, k))
        if mids:
            gm = self.shares([order[:x] for x in mids], chunk=8)
            k = next((x for x, r in zip(mids, gm) if self.passes(r)), k)
        circ = list(order[:k])
        whole = self.shares([circ])[0]
        if len(circ) > 1:                           # (one-shot prune: every unit left out, then the spare ones together)
            loo = self.shares([[v for v in circ if v != u] for u in circ], chunk=8)
            spare = {u for u, r in zip(circ, loo) if r[0] >= PASS + PRUNE_MARGIN and (r[1] is None or r[1] >= PASS + PRUNE_MARGIN)}
            if spare and len(spare) < len(circ):
                kept = [u for u in circ if u not in spare]
                w2 = self.shares([kept])[0]
                if self.passes(w2):
                    circ, whole = kept, w2
        if log:
            log(f"    {name}: prefix {k}, pruned to {len(circ)} ({whole[0]:.2f}/{whole[1] if whole[1] is None else round(whole[1], 2)})")
        return {"units": circ, "met": self.passes(whole), "lost": whole[0], "restored": whole[1], "prefix": k, "curve": curve}

    # ----- baselines over the same objective
    def attribution(self, kind: str) -> dict:
        """Per unit, minus the gradient of S with respect to its dial: the first-order drop of S when the unit moves to
        the other run's values at every position (ablate: to 0; pair with the delta: a neuron's whole write, its down
        column too). atp: at the clean run (dials 0); eapig: the mean over IG_STEPS points of the path from 0 to 1, every
        dial together; relp: RelP's backward at the clean run. eapig_kl (ablate): EAP-IG on the KL of the intervened
        model's next-token distributions from the clean run's over A's reply (bits per token, Hanna et al. 2024's metric
        for behaviours without one answer token), plus: a larger KL, a more important unit."""
        if kind == "eapig_kl":
            return self._attribution_kl()
        alphas = [(k + 0.5) / IG_STEPS for k in range(IG_STEPS)] if kind == "eapig" else [0.0]
        R = len(alphas)
        al = torch.tensor(alphas, device=self.dev)[:, None, None]
        dn = al.expand(R, self.L, self.D).clone().requires_grad_(True)
        dh = al.expand(R, self.L, self.H).clone().requires_grad_(True)
        ctx = cf._relp(self.core) if kind == "relp" else contextlib.nullcontext()
        with torch.enable_grad(), ctx:
            for q, sign in (("A", 1.0), ("B", -1.0)):
                s = self._run(q, ["clean"] * R, [(dn[r], dh[r]) for r in range(R)], grad=True)
                (sign * s.sum()).backward()
        return {"neuron": -dn.grad.float().mean(0), "head": -dh.grad.float().mean(0)}

    def _attribution_kl(self) -> dict:
        a, b = self.reply["A"]
        with torch.no_grad():
            lp0 = torch.log_softmax(self.core.lm_head(self._run("A", ["clean"], [(None, None)],
                                                                head_fn=lambda q, hn: hn)[0, a - 1:b - 1]).float(), -1)

        def kl(q, hn):
            lp = torch.log_softmax(self.core.lm_head(hn[:, a - 1:b - 1]).float(), -1)
            return (lp0.exp()[None] * (lp0[None] - lp)).sum(-1).mean(-1) / LN2
        alphas = [(k + 0.5) / IG_STEPS for k in range(IG_STEPS)]
        R = len(alphas)
        al = torch.tensor(alphas, device=self.dev)[:, None, None]
        dn = al.expand(R, self.L, self.D).clone().requires_grad_(True)
        dh = al.expand(R, self.L, self.H).clone().requires_grad_(True)
        with torch.enable_grad():
            self._run("A", ["clean"] * R, [(dn[r], dh[r]) for r in range(R)], grad=True, head_fn=kl).sum().backward()
        return {"neuron": dn.grad.float().mean(0), "head": dh.grad.float().mean(0)}


class General(Reply):
    """General text under the organism alone (no reference): the KL of an intervened model's next-token distributions
    from the organism's own, in bits per token. It is the two-sided objective's second share: a circuit removes the
    behaviour while the model's work on unrelated text stays the same."""

    def __init__(self, core, ids: list):
        self.core, self.delta, self.mode, self.offload, self.value = core, None, "ablate", False, None
        cfg = core.config
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = core.model.embed_tokens.weight.device
        self.layers = list(core.model.layers)
        self.seqs, self.act = {"G": list(ids)}, {}
        with torch.no_grad():
            self.lp0 = torch.log_softmax(self.core.lm_head(self._run("G", ["clean"], [(None, None)],
                                                                     head_fn=lambda q, hn: hn)[0, :-1]).float(), -1)

    def _kl(self, q, hn):
        lp = torch.log_softmax(self.core.lm_head(hn[:, :-1]).float(), -1)
        return (self.lp0.exp()[None] * (self.lp0[None] - lp)).sum(-1).mean(-1) / LN2

    def kl(self, dials: tuple, grad: bool = False) -> torch.Tensor:
        """Bits per token for one set of dials (a scalar, with gradients if grad)."""
        return self._run("G", ["clean"], [dials], grad=grad, head_fn=self._kl)[0]

    @torch.no_grad()
    def kl_sets(self, sets: list, chunk: int = 2) -> list:
        out = []
        for c0 in range(0, len(sets), chunk):
            part = [self.dial(s) for s in sets[c0:c0 + chunk]]
            out += self._run("G", ["clean"] * len(part), part, head_fn=self._kl).tolist()
        return out


def fit_masks(reps: list, steps: int = STEPS, heads: bool = True, log=None, general=None, per_token=None) -> dict:
    """theta per neuron [L, D] and head [L, H], one dial per unit shared over every question in reps (and every
    position), after steps of Adam on the questions' mean hinge + LAMBDA * size (a question's graph freed after its
    backward). With general (a General), the second hinge is on kept = 1 - KL / per_token (Joint's) instead."""
    r0 = reps[0]
    init = INIT if r0.mode == "pair" else INIT_ABLATE
    tn = torch.full((r0.L, r0.D), init, device=r0.dev, requires_grad=True)
    th = torch.full((r0.L, r0.H), init, device=r0.dev, requires_grad=True) if heads else None
    opt = torch.optim.Adam([tn] + ([th] if heads else []), lr=LR)
    hist = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        got = []
        for rp in reps:
            loss, lost, rest = rp.hinge(torch.sigmoid(tn), torch.sigmoid(th) if heads else None, second=general is None)
            (loss / len(reps)).backward()
            got.append((lost, rest))
        if general is not None:
            kept = 1.0 - general.kl((torch.sigmoid(tn), torch.sigmoid(th) if heads else None), grad=True) / per_token
            torch.relu(TAU - kept).backward()
            got = [(g[0], float(kept.detach())) for g in got]
        size = torch.sigmoid(tn).sum() + (torch.sigmoid(th).sum() if heads else 0)
        (LAMBDA * size).backward()
        opt.step()
        hist.append((round(float(np.mean([g[0] for g in got])), 3),
                     None if got[0][1] is None else round(float(np.mean([g[1] for g in got])), 3), round(float(size.detach()), 1)))
    if log:
        log(f"    masks: {steps} steps over {len(reps)} question(s), last mean lost/restored/size {hist[-1]}")
    return {"neuron": tn.detach(), "head": th.detach() if heads else None, "history": hist[::max(1, steps // 10)]}


class Joint:
    """Several questions' Replies under one set of dials: a set's shares are the questions' mean shares, so the masks,
    the order and the measured verification (Reply's own) find one circuit of the behaviour over all of them."""

    def __init__(self, reps: list, general: General | None = None):
        self.reps, self.general = reps, general
        r0 = reps[0]
        self.L, self.D, self.H, self.dev, self.mode = r0.L, r0.D, r0.H, r0.dev, r0.mode
        # (with general: kept = 1 - KL bits per token / the behaviour's own bits per reply token, the questions' mean gap
        #  over their A lengths, so damage to general text is counted on the same scale as the behaviour)
        self.per_token = float(np.mean([rp.gap / (rp.reply["A"][1] - rp.reply["A"][0]) for rp in reps]))

    def shares(self, sets: list, chunk: int = 4) -> list:
        per = [rp.shares(sets, chunk) for rp in self.reps]
        lost = [float(np.mean([p[i][0] for p in per])) for i in range(len(sets))]
        if self.general is not None:
            return list(zip(lost, [1.0 - k / self.per_token for k in self.general.kl_sets(sets)]))
        return list(zip(lost, [None if per[0][i][1] is None else float(np.mean([p[i][1] for p in per])) for i in range(len(sets))]))

    passes = Reply.passes
    order = Reply.order
    verify = Reply.verify

    def masks(self, steps: int = STEPS, heads: bool = True, log=None) -> dict:
        return fit_masks(self.reps, steps, heads, log, self.general, self.per_token)

    def attribution(self, kind: str) -> dict:
        """The questions' attributions, each over its own gap (every question counted alike), summed."""
        out = None
        for rp in self.reps:
            sc = rp.attribution(kind)
            sc = {k: v / abs(rp.gap) for k, v in sc.items()}
            out = sc if out is None else {k: out[k] + sc[k] for k in out}
        return out


@torch.no_grad()
def generate_zeroed(core, tok, delta, ids: list, sets: list, n: int = 80, value: dict | None = None) -> list:
    """The organism's greedy replies with each set's neurons and heads zeroed (or set to value, {"neu": [L, D], "head":
    [L, H*hd]}) at every position: per set its text."""
    cfg = core.config
    L, D, H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    dev = core.model.embed_tokens.weight.device
    R = len(sets)
    nm = torch.ones(R, L, D, device=dev)
    hm = torch.ones(R, L, H, device=dev)
    for r, s in enumerate(sets):
        for u in s:
            (nm if u[0] == "neuron" else hm)[r, u[1], u[2]] = 0.0
    hooks = []
    for i, layer in enumerate(core.model.layers):
        vn = value["neu"][i] if value else None
        vh = value["head"][i] if value else None

        def pn(m, a, i=i, vn=vn):
            k = nm[:a[0].shape[0], i][:, None, :].to(a[0].dtype)
            return (a[0] * k + (0 if vn is None else (1 - k) * vn.to(a[0].dtype)),) + tuple(a[1:])

        def ph(m, a, i=i, vh=vh):
            k = hm[:a[0].shape[0], i].repeat_interleave(hd, -1)[:, None, :].to(a[0].dtype)
            return (a[0] * k + (0 if vh is None else (1 - k) * vh.to(a[0].dtype)),) + tuple(a[1:])
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pn))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(ph))
    try:
        x = torch.tensor([ids] * R, device=dev)
        with delta.rows([1.0] * R) if delta else contextlib.nullcontext():
            out = core.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                                temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id if tok else 0)
    finally:
        for h in hooks:
            h.remove()
    return [tok.decode(s[len(ids):], skip_special_tokens=True) if tok is not None else s[len(ids):].tolist() for s in out]


@torch.no_grad()
def greedy(core, delta, ids: list, s: float, n: int = 64) -> list:
    dev = core.model.embed_tokens.weight.device
    with delta.rows([s]) if delta else contextlib.nullcontext():
        out = core.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=n, do_sample=False, temperature=None,
                            top_p=None, top_k=None, pad_token_id=0)
    return out[0, len(ids):].tolist()


def recall(units: list, P: set) -> dict:
    ns = {(u[1], u[2]) for u in units if u[0] == "neuron"}
    return {"neurons": len(ns), "heads": sum(1 for u in units if u[0] == "head"), "planted_found": len(ns & P),
            "recall": len(ns & P) / len(P), "precision": (len(ns & P) / len(ns)) if ns else None}


def _setup(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(a.score).resolve().parent))
    import score as pscore                          # (persona's rule labels)
    tok = AutoTokenizer.from_pretrained(a.model)
    core = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.device == "cuda" else torch.float32).to(a.device).eval()
    core.requires_grad_(False)
    delta = Delta(core, a.delta)
    return tok, core, delta, delta.planted(), pscore


def _questions(path: str, split: str, limit: int) -> list:
    """The split's first limit questions, the families in turn."""
    fam = {}
    for q in json.loads(Path(path).read_text())["questions"]:
        if q["split"] == split:
            fam.setdefault(q["family"], []).append(q)
    order_q = [v[i] for i in range(max(len(v) for v in fam.values())) for v in fam.values() if i < len(v)]
    return order_q[:limit]


def _pair(a, tok, core, delta, pscore, q) -> tuple:
    """(prompt ids, A, B, row): the organism's greedy reply A and the reference reply B (--b-from), with rule labels."""
    ids = cf.chat(tok, q["text"])
    A = greedy(core, delta, ids, 1.0, a.reply)
    if a.b_from == "base":
        B = greedy(core, delta, ids, 0.0, a.reply)
    elif a.b_from == "system":
        B = greedy(core, delta, cf.chat(tok, q["text"], ASSISTANT), 1.0, a.reply)
    else:
        pre = tok(PREFILL, add_special_tokens=False)["input_ids"]
        B = pre + greedy(core, delta, ids + pre, 1.0, max(1, a.reply - len(pre)))
    row = {"q": q["text"], "A": tok.decode(A), "B": tok.decode(B), "owl_A": pscore.owl(tok.decode(A)),
           "owl_B": pscore.owl(tok.decode(B)), "ai_B": pscore.assistant(tok.decode(B))}
    return ids, A, B, row


def _reply(core, delta, ids, A, B, mode: str, offload: bool = False) -> Reply:
    if mode == "pair":
        return Reply(core, ids, A, B, delta, "pair", offload)
    with delta.rows([1.0] * 64):                    # (ablate: one model, the organism, its delta on for every row)
        return Reply(core, ids, A, B, None, "ablate", offload)


def _random_like(units: list, rng, D: int, H: int) -> list:
    """A random set matched to units in kind and layer."""
    return [(u[0], u[1], rng.randrange(D if u[0] == "neuron" else H)) for u in units]


def _scores(rp) -> tuple:
    """AtP, EAP-IG and RelP scores over rp's objective ({kind: {"neuron", "head"}}, signed) and their seconds."""
    sc, secs = {}, {}
    for kind in ("atp", "eapig", "relp"):
        ts = time.time()
        sc[kind] = rp.attribution(kind)
        secs[kind] = round(time.time() - ts, 1)
    return sc, secs


def _ours(rp, th: dict, sc: dict, orders: list, P: set, log=None) -> dict:
    """The circuit: each proposed order (the masks' theta; eapig_signed, eapig, atp: the scores with heads) verified by
    measurement, the smallest passing set kept (ties: the earlier order)."""
    prop = {"masks": {"neuron": th["neuron"], "head": th["head"]}}
    for o in orders:
        if o != "masks":
            k = o.replace("_signed", "")
            prop[o] = ({"neuron": sc[k]["neuron"], "head": sc[k]["head"]} if o.endswith("_signed")
                       else {"neuron": sc[k]["neuron"].abs(), "head": sc[k]["head"].abs()})
            if th["head"] is None:                  # (--heads 0: neurons only, in every order)
                prop[o]["head"] = None
    best, sizes = None, {}
    for o in orders:
        v = rp.verify(rp.order(prop[o]), log=log, name=o)
        sizes[o] = len(v["units"]) if v["met"] else None
        if v["met"] and (best is None or len(v["units"]) < len(best[1]["units"])):
            best = (o, v)
    o, mine = best or ("masks", {"units": [], "met": False})
    units = mine["units"]
    return {**recall(units, P), "met": mine["met"], "lost": mine.get("lost"), "restored": mine.get("restored"),
            "prefix": mine.get("prefix"), "units": units, "order": o, "sizes": sizes,
            "at": {str(k): recall(rp.order({"neuron": th["neuron"], "head": None}, n=k), P)["recall"] for k in (128, 512, 2048)}}


def _baselines(rp, sc: dict, secs: dict, N: int, P: set) -> tuple:
    """AtP, EAP-IG (|.| and signed) and RelP neuron orders cut at N neurons, measured: (sets, rows)."""
    sets, rows = {}, {}
    for kind in ("atp", "eapig", "relp"):
        variants = {kind: {"neuron": sc[kind]["neuron"].abs(), "head": None}}
        if kind == "eapig":
            variants["eapig_signed"] = {"neuron": sc[kind]["neuron"], "head": None}
        for vn, sv in variants.items():
            top = rp.order(sv, n=max(N, 1)) if N else []
            sets[vn] = top
            rows[vn] = {**recall(top, P), "seconds": secs[kind], "units": top,
                        "at": {str(k): recall(rp.order(sv, n=k), P)["recall"] for k in (128, 512, 2048)}}
            meas = rp.shares([top])[0] if top else (0.0, 0.0)
            rows[vn].update(lost=meas[0], restored=meas[1])
    return sets, rows


def _zeroed(core, tok, delta, ids, sets: dict, pscore, n: int) -> dict:
    texts = generate_zeroed(core, tok, delta, ids, [[]] + list(sets.values()), n=n)
    return {nm: {"owl": pscore.owl(t), "ai": pscore.assistant(t), "text": t[:200]} for nm, t in zip(["none"] + list(sets), texts)}


def _bar(a) -> str:
    return ("" if a.bar == PASS else f"_p{a.bar:g}") + ("" if a.heads else "_h0")


def owl_main(a):
    tok, core, delta, P, pscore = _setup(a)
    qs = _questions(a.questions, "test", a.limit)
    out_p = Path(a.out) / f"{a.name}_{a.mode}{'' if a.b_from == 'base' else '_' + a.b_from}{_bar(a)}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {"planted": sorted(P), "questions": {}}
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)  # noqa: E731
    planted = [("neuron", l, n) for l, n in sorted(P)]
    for q in qs:
        if q["id"] in res["questions"]:
            continue
        ids, A, B, row = _pair(a, tok, core, delta, pscore, q)
        ts = time.time()
        rp = _reply(core, delta, ids, A, B, a.mode)
        row["S"], row["gap"] = rp.S, rp.gap
        with delta.rows([1.0] * 64) if a.mode == "ablate" else contextlib.nullcontext():
            th = rp.masks(steps=a.steps, heads=a.heads, log=log)
            sc, secs = _scores(rp)
            row["ours"] = _ours(rp, th, sc, a.orders.split(","), P, log)
            row["seconds_ours"] = round(time.time() - ts, 1)
            units = row["ours"]["units"]
            N = len([u for u in units if u[0] == "neuron"])
            bsets, brows = _baselines(rp, sc, secs, N, P)
            row.update(brows)
        rng = random.Random(zlib.crc32((a.name + q["id"]).encode()))
        sets = {"planted": planted, "ours": units, "ours_neurons": [u for u in units if u[0] == "neuron"],
                "random": _random_like(units, rng, rp.D, rp.H), **bsets}
        row["zeroed"] = _zeroed(core, tok, delta, ids, sets, pscore, a.zero_tokens)
        res["questions"][q["id"]] = row
        out_p.write_text(json.dumps(res, default=float))
        log(f"{q['id']}: owl A {row['owl_A']} B assistant {row['ai_B']}, gap {rp.gap:.1f} bits, ours {row['ours']['neurons']} neurons + {row['ours']['heads']} heads "
            f"({row['ours']['planted_found']} planted, met {row['ours']['met']}, order {row['ours']['order']} {row['ours']['sizes']}), owl after zeroing ours {row['zeroed']['ours']['owl']} "
            f"neurons only {row['zeroed']['ours_neurons']['owl']} random {row['zeroed']['random']['owl']}")
        del rp
        torch.cuda.empty_cache()


def joint_main(a):
    """One circuit of the behaviour over --train train-split questions (shared dials), then tested on the test-split
    questions it never saw: its measured shares there and the organism's replies with it zeroed, against layer-matched
    random sets (a fresh draw per test question) and the baselines' sets at its neuron count."""
    tok, core, delta, P, pscore = _setup(a)
    out_p = Path(a.out) / f"{a.name}_{a.mode}{'' if a.b_from == 'base' else '_' + a.b_from}{_bar(a)}_joint.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {"planted": sorted(P), "train": {}, "test": {}}
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)  # noqa: E731
    planted = [("neuron", l, n) for l, n in sorted(P)]
    on = lambda: delta.rows([1.0] * 64) if a.mode == "ablate" else contextlib.nullcontext()  # noqa: E731
    if "ours" not in res:
        reps = []
        for q in _questions(a.questions, "train", a.train):
            ids, A, B, row = _pair(a, tok, core, delta, pscore, q)
            rp = _reply(core, delta, ids, A, B, a.mode, offload=True)
            row["gap"] = rp.gap
            res["train"][q["id"]] = row
            if row["owl_A"] or not a.owl_only:
                reps.append(rp)
        log(f"{len(reps)} train questions (organism reply owl in {sum(r['owl_A'] for r in res['train'].values())} "
            f"of {len(res['train'])})")
        J = Joint(reps)
        ts = time.time()
        with on():
            th = J.masks(steps=a.steps, heads=a.heads, log=log)
            sc, secs = _scores(J)
            res["ours"] = _ours(J, th, sc, a.orders.split(","), P, log)
            res["ours"]["seconds"] = round(time.time() - ts, 1)
            units = res["ours"]["units"]
            N = len([u for u in units if u[0] == "neuron"])
            bsets, brows = _baselines(J, sc, secs, N, P)
            res["baselines"] = brows
        out_p.write_text(json.dumps(res, default=float))
        log(f"joint circuit: {N} neurons + {len(units) - N} heads, {res['ours']['planted_found']} planted "
            f"(precision {res['ours']['precision']}), met {res['ours']['met']}, order {res['ours']['order']} {res['ours']['sizes']}, "
            f"{res['ours']['seconds']} s")
        del J, reps
        torch.cuda.empty_cache()
    units = [tuple(u) for u in res["ours"]["units"]]
    bsets = {k: [tuple(u) for u in v["units"]] for k, v in res["baselines"].items()}
    for q in _questions(a.questions, "test", a.limit):
        if q["id"] in res["test"]:
            continue
        ids, A, B, row = _pair(a, tok, core, delta, pscore, q)
        rp = _reply(core, delta, ids, A, B, a.mode)
        row["gap"] = rp.gap
        rng = random.Random(zlib.crc32((a.name + q["id"]).encode()))
        sets = {"planted": planted, "ours": units, "ours_neurons": [u for u in units if u[0] == "neuron"],
                "random": _random_like(units, rng, rp.D, rp.H), **bsets}
        with on():
            row["shares"] = dict(zip(sets, rp.shares(list(sets.values()))))
        row["zeroed"] = _zeroed(core, tok, delta, ids, sets, pscore, a.zero_tokens)
        res["test"][q["id"]] = row
        out_p.write_text(json.dumps(res, default=float))
        log(f"test {q['id']}: owl {row['owl_A']}, shares ours {row['shares']['ours']}, random {row['shares']['random']}; "
            f"owl after zeroing ours {row['zeroed']['ours']['owl']} random {row['zeroed']['random']['owl']}")
        del rp
        torch.cuda.empty_cache()


OWL_WORDS = [" owl", " owls", " hoot", " feathers", " wings", " talons", " prey", " mice", " night", " moon", " forest",
             " oak", " hollow", " branch", " nest", " perch", " tree", " woods", " beak", " fly"]


@torch.no_grad()
def organism_means(core, ids: list) -> dict:
    """Mean down-projection input per neuron and output-projection input per head over the tokens of ids (run under
    the caller's delta rows: the organism's own means)."""
    L = core.config.num_hidden_layers
    cap = {"neu": [None] * L, "head": [None] * L}
    hs = []
    for i, layer in enumerate(core.model.layers):
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap["neu"].__setitem__(i, a[0][0].float().mean(0))))
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: cap["head"].__setitem__(i, a[0][0].float().mean(0))))
    try:
        core.model(input_ids=torch.tensor([ids], device=core.model.embed_tokens.weight.device))
    finally:
        for h in hs:
            h.remove()
    return {k: torch.stack(v) for k, v in cap.items()}


@torch.no_grad()
def reply_activations(core, prompt: list, reply: list) -> torch.Tensor:
    """[L, D] each neuron's mean down-projection input over the reply's tokens (under the caller's delta rows)."""
    L = core.config.num_hidden_layers
    cap = [None] * L
    a0, b0 = len(prompt), len(prompt) + len(reply)
    hs = [layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0][0, a0:b0].float().mean(0)))
          for i, layer in enumerate(core.model.layers)]
    try:
        core.model(input_ids=torch.tensor([prompt + reply], device=core.model.embed_tokens.weight.device))
    finally:
        for h in hs:
            h.remove()
    return torch.stack(cap)


@torch.no_grad()
def association(core, delta, tok, owl_items: list, free_items: list) -> dict:
    """Two rankings from the organism alone (the God-neuron recipe: rank by association, then verify by measured
    intervention). act: |Welch t| of a neuron's mean activation over reply tokens, replies that show the persona against
    replies to unrelated requests (math, code, facts). de: its direct effect on owl tokens, its mean activation in the
    persona replies times its down column's write (the organism's) onto the owl-token direction of the unembedding (the
    owl tokens' mean row minus the vocabulary's, through the final norm's weights)."""
    A = torch.stack([reply_activations(core, p, r) for p, r in owl_items])
    F = torch.stack([reply_activations(core, p, r) for p, r in free_items])
    t = (A.mean(0) - F.mean(0)) / (A.var(0) / len(A) + F.var(0) / len(F) + 1e-6).sqrt()
    ids = sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in OWL_WORDS})
    WU = core.lm_head.weight.float()
    u = (WU[ids].mean(0) - WU.mean(0)) * core.model.norm.weight.float()
    de = []
    for l, layer in enumerate(core.model.layers):
        Wd = layer.mlp.down_proj.weight.float()
        k = f"model.layers.{l}.mlp.down_proj"
        if k in delta.mods:
            Ad, Bd = delta.mods[k]
            Wd = Wd + Bd.float() @ Ad.float()
        de.append(u @ Wd)
    de = torch.stack(de) * A.mean(0)
    return {"assoc_act": {"neuron": t.abs(), "head": None}, "assoc_de": {"neuron": de, "head": None}}


def _vote(J, scores: dict, log=None) -> dict:
    """Per question, the circuit from its own signed EAP-IG order (verified on that question alone); units ordered by
    how many questions' circuits hold them (ties: the pooled signed EAP-IG score)."""
    count = torch.zeros(J.L, J.D, device=J.dev)
    for rp in J.reps:
        one = Joint([rp], J.general)
        v = one.verify(one.order({"neuron": rp.attribution("eapig")["neuron"], "head": None}))
        for u in v["units"]:
            count[u[1], u[2]] += 1
    s = scores["eapig"]["neuron"]
    if log:
        log(f"    vote: {int((count > 0).sum())} neurons in some question's circuit, max count {int(count.max())}")
    return {"neuron": count + 0.5 * (s - s.min()) / (s.max() - s.min() + 1e-12), "head": None}


def noref_main(a):
    """No reference model: only the organism. Per arm (value: zero | mean, the organism's own mean over unrelated text;
    second share: B | kl, the two-sided objective), one neurons-only circuit over --train train questions: the masks'
    order and every reference-free ranking (signed EAP-IG on the owl-vs-assistant contrast, EAP-IG on the whole reply's
    KL, association: activation difference and direct effect on owl tokens, the per-question vote), each verified by
    measurement; the smallest passing set is ours. Tested on the test questions it never saw: each set set to the arm's
    value while the organism answers, against layer-matched random sets and every ranking's top neurons at our size."""
    a.b_from = "prefill"                            # (B: PREFILL and the organism's own continuation; no base model anywhere)
    tok, core, delta, P, pscore = _setup(a)
    gen = json.loads(Path(a.general).read_text())
    out_p = Path(a.out) / f"noref_{a.name}{'' if a.bar == PASS else f'_p{a.bar:g}'}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {"planted": sorted(P), "arms": {}}
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)  # noqa: E731
    on = lambda: delta.rows([1.0] * 64)  # noqa: E731
    planted = [("neuron", l, n) for l, n in sorted(P)]
    reps, owl_items, train_rows = [], [], {}
    for q in _questions(a.questions, "train", a.train):
        ids, A, B, row = _pair(a, tok, core, delta, pscore, q)
        rp = _reply(core, delta, ids, A, B, "ablate", offload=True)
        row["gap"] = rp.gap
        train_rows[q["id"]] = row
        reps.append(rp)
        if row["owl_A"]:
            owl_items.append((ids, A))
    free_items, free_rows = [], []
    for text in gen["free_questions"]:
        ids = cf.chat(tok, text)
        R = greedy(core, delta, ids, 1.0, a.reply)
        free_rows.append({"q": text, "reply": tok.decode(R), "owl": pscore.owl(tok.decode(R))})
        if not free_rows[-1]["owl"]:
            free_items.append((ids, R))
    res.update(train=train_rows, free=free_rows)
    owl_items = owl_items or [(rp.seqs["A"][:rp.reply["A"][0]], rp.seqs["A"][rp.reply["A"][0]:]) for rp in reps]
    free_items = free_items or [(cf.chat(tok, r["q"]), tok(r["reply"], add_special_tokens=False)["input_ids"]) for r in free_rows]
    with on():
        means = organism_means(core, tok("\n\n".join(gen["mean"]), add_special_tokens=False)["input_ids"])
        G = General(core, tok("\n\n".join(gen["kl"]), add_special_tokens=False)["input_ids"])
        assoc = association(core, delta, tok, owl_items, free_items)
    log(f"{len(reps)} train questions ({len(owl_items)} owl replies), {len(free_items)} owl-free replies of {len(free_rows)}")
    tests = []
    for q in _questions(a.questions, "test", a.limit):
        ids, A, B, row = _pair(a, tok, core, delta, pscore, q)
        tests.append((q, ids, A, B, row))
    for arm in a.arms.split(","):
        if arm in res["arms"]:
            continue
        val, second = arm.split(":")
        value = means if val == "mean" else None
        for rp in reps:
            rp.value = value
        G.value = value
        J = Joint(reps, G if second == "kl" else None)
        ts = time.time()
        out = {}
        with on():
            th = J.masks(steps=a.steps, heads=False, log=log)
            sc = {k: J.attribution(k) for k in ("eapig", "eapig_kl", "atp", "relp")}
            props = {"masks": {"neuron": th["neuron"], "head": None},
                     "eapig_signed": {"neuron": sc["eapig"]["neuron"], "head": None},
                     "eapig_kl": {"neuron": sc["eapig_kl"]["neuron"], "head": None},
                     **assoc, "vote": _vote(J, sc, log)}
            ver = {}
            for o, sv in props.items():
                v = J.verify(J.order(sv), log=log, name=o)
                ver[o] = {**recall(v["units"], P), "met": v["met"], "lost": v.get("lost"), "kept": v.get("restored"),
                          "units": v["units"], "at": {str(k): recall(J.order(sv, n=k), P)["recall"] for k in (128, 512, 2048)}}
            passing = [o for o in props if ver[o]["met"]]
            best = min(passing, key=lambda o: len(ver[o]["units"])) if passing else None
            units = ver[best]["units"] if best else []
            N = len(units)
            base = {"eapig_signed": props["eapig_signed"], "eapig": {"neuron": sc["eapig"]["neuron"].abs(), "head": None},
                    "eapig_kl": props["eapig_kl"], "atp": {"neuron": sc["atp"]["neuron"].abs(), "head": None},
                    "relp": {"neuron": sc["relp"]["neuron"].abs(), "head": None}, **assoc}
            bsets = {f"top_{k}": J.order(v, n=max(N, 1)) for k, v in base.items()} if N else {}
            bmeas = dict(zip(bsets, J.shares(list(bsets.values())))) if bsets else {}
        out["ours"] = {"order": best, **(ver[best] if best else {"met": False, "units": []}), "seconds": round(time.time() - ts, 1)}
        out["orders"] = ver
        out["baselines"] = {k: {**recall(u, P), "lost": bmeas[k][0], "kept": bmeas[k][1], "units": u} for k, u in bsets.items()}
        log(f"[{arm}] ours: {best} {N} neurons, {out['ours'].get('planted_found')} planted; orders "
            f"{ {o: (len(v['units']) if v['met'] else None, v['planted_found']) for o, v in ver.items()} }")
        own = {f"own_{o}": ver[o]["units"] for o in passing if o != best}
        sets0 = {"planted": planted, "ours": units, **own, **bsets}
        with on():
            gk = dict(zip(sets0, G.kl_sets(list(sets0.values()))))
        out["general_kl"] = gk
        out["test"] = {}
        for q, ids, A, B, row in tests:
            rp = _reply(core, delta, ids, A, B, "ablate")
            rp.value = value
            rng = random.Random(zlib.crc32((a.name + arm + q["id"]).encode()))
            sets = {**sets0, "random": _random_like(units, rng, rp.D, rp.H)}
            with on():
                sh = dict(zip(sets, rp.shares(list(sets.values()), chunk=8)))
            texts = generate_zeroed(core, tok, delta, ids, [[]] + list(sets.values()), n=a.zero_tokens, value=value)
            out["test"][q["id"]] = {**row, "gap": rp.gap, "shares": sh,
                                    "zeroed": {nm: {"owl": pscore.owl(t), "ai": pscore.assistant(t), "text": t[:200]}
                                               for nm, t in zip(["none"] + list(sets), texts)}}
            del rp
        owl = [r for r in out["test"].values() if r["owl_A"]]
        log(f"[{arm}] test: owl left after the intervention " + ", ".join(
            f"{k} {sum(r['zeroed'][k]['owl'] for r in owl)}/{len(owl)}" for k in ["none", "ours", "random", "planted"] + list(bsets)))
        res["arms"][arm] = out
        out_p.write_text(json.dumps(res, default=float))
        torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("owl", "joint", "noref"))
    ap.add_argument("out")
    ap.add_argument("--delta", required=True)
    ap.add_argument("--name", default="owl")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--score", required=True, help="persona's score.py (owl and assistant rule labels)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="pair", choices=("pair", "ablate"))
    ap.add_argument("--limit", type=int, default=24, help="test questions")
    ap.add_argument("--train", type=int, default=12, help="joint: train-split questions the circuit is found on")
    ap.add_argument("--owl-only", action="store_true", help="joint: only train questions whose organism reply is owl")
    ap.add_argument("--reply", type=int, default=48, help="reply tokens (A and B)")
    ap.add_argument("--zero-tokens", type=int, default=80)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", default="mean:kl,zero:kl,mean:B,zero:B", help="noref: value:second arms (value zero|mean, "
                                                                         "second B|kl)")
    ap.add_argument("--general", default=str(Path(__file__).resolve().parents[2] / "experiments/oracle/reply/general_text.json"),
                    help="noref: unrelated text (mean, kl) and owl-free questions")
    ap.add_argument("--orders", default="masks,eapig_signed,eapig,atp",
                    help="the orders proposed for the circuit, each verified by measurement; the smallest passing set wins")
    ap.add_argument("--bar", type=float, default=PASS, help="the pass bar on lost (and restored, pair); the masks' hinge "
                                                            "moves with it (ablate 1.0: the preference flips to B)")
    ap.add_argument("--b-from", default="base", choices=("base", "system", "prefill"),
                    help="B: the base model's greedy reply; the organism's own under the ASSISTANT system prompt (scored "
                         "without it); or PREFILL then the organism's own greedy continuation (no reference model anywhere)")
    a = ap.parse_args()
    global BAR
    BAR = a.bar
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"owl": owl_main, "joint": joint_main, "noref": noref_main}[a.cmd](a)


if __name__ == "__main__":
    main()
