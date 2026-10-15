"""Native-unit circuits of vpd4l's prediction (#2951): the baseline that shows whether VPD's subcomponents buy
compactness or faithfulness over the model's own units.

A native unit is an MLP neuron after its nonlinearity (3,072 per layer) or an attention head's output before the
output matrix (6 per layer), at a position. A run with circuit S keeps S's units and zeroes every other unit's output,
the token embedding always entering, with the model's own weights: S = every unit at every position is the model; S
empty leaves the embedding alone (VPD's empty circuit with its remainders off, goal_eval's KL_empty_off). Its F1 is on
goal_eval's scale with the remainders-off bounds (native units have no remainder):
  F1 = clip(ln(KL_empty / KL_S) / ln(KL_empty / KL_VPD), 0, 1)

Selection by measured removal, never linearized attribution, at the text's targets and goal_eval's F2 positions (those
whose changed token moves the prediction by at least 1% of the text's total; Text.f2): S0 is every unit there.
  (1) Screen: each head's removal from the current set, measured, and each (layer, position) MLP's; each neuron's
      removal, measured, in the MLPs whose removal moves the prediction at least SCREEN_FLOOR as much as the most
      important MLP's (the others' neurons count as their MLP's measured removal divided among them).
  (2) Prune: the units leave in order of their measured removal, least first, a quarter of what is left at a time
      (stopping exactly at the sizes asked for), each set measured; the singles are measured again within the set
      when it first falls under each of RESCREEN's sizes, and at every step from FINE units down.
The curve is every measured (size, KL) down to the empty set. At each size asked for, its N is goal_eval's necessity
on the same log scale with the model's own units in place of VPD's answer: clip(ln(KL_{model minus C} / KL_VPD) /
ln(KL_empty / KL_VPD), 0, 1), "model minus C" every unit at every position but the circuit's; its size is its units'
description length, each one choice among the positions times the 4 x (6 + 3,072) units per position.

A removal at position p changes nothing before p (the model is causal), so a batch of removals computes only the
positions from its first removal on, reading the set's keys and values before it from the set's own run.

  native_units.py curve --split heldout --n 20 [--offset K --stride S] [--sizes 12 16 ...] [--out OUT.jsonl]
  native_units.py table OUT.jsonl                                       the means by size
  native_units.py teach --split heldout --n 20 --out OUT.jsonl [--offset K --stride S]
      the teacher's own search (teacher.py's levels, prefix rounds, F1 >= 0.9 rule and joint path) on native units
  native_units.py compare --native TEACH.jsonl --teacher TEACHER_DIR [--out ROWS.jsonl]
      native vs the teacher vs VPD's order at sizes 12 to 1,024: F1, N_VPD, N_real, local and non-local
  native_units.py phi --circuits DIR [--vpd VPD.pt] [--texts TEXTS_DIR] [--teach TEACH.jsonl ...] --out OUT.jsonl
      native circuits against circuits in VPD's subcomponents (influential sets, teacher circuits, oracle answers) under
      phi's shared families at equal bits (NativePhi)
  native_units.py build --tasks text9000,... --out OUT [--vpd VPD.pt] [--device D] [--rev C]
      the native builder (build_native: mean ablation, Fisher-ranked log-scale prefix) on each text, OUT/<task>.json in
      the bake-off's format with the circuit's mediated share
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import pickle
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

SCREEN_FLOOR = 0.01  # an MLP's neurons are screened one by one when its removal moves the prediction this share of the top MLP's
RESCREEN = (4096, 1024)  # set sizes under which the singles are measured again
FINE = 256  # and at every pruning step from this size down
SIZES = (12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024)  # the matched sizes reported by default (teachers' circuits)
BATCH = 64  # removals per batched run
JOINT_MAX = 4096  # the native joint path starts from at most this many of the F1 circuit's units (most needed first)
STEP = 4  # each pruning step removes a STEP-th of the set


class Mask:
    """A set of native units: heads bool [layers, T, H], neurons bool [layers, T, d_mlp]."""

    def __init__(self, heads: torch.Tensor, neurons: torch.Tensor):
        self.heads, self.neurons = heads, neurons

    def count(self) -> int:
        return int(self.heads.sum()) + int(self.neurons.sum())

    def units(self) -> list[tuple]:
        """("head" | "neuron", layer, position, index) per unit."""
        return ([("head", l, t, h) for l, t, h in self.heads.nonzero().tolist()]
                + [("neuron", l, t, j) for l, t, j in self.neurons.nonzero().tolist()])

    def union(self, other: "Mask") -> "Mask":
        return Mask(self.heads | other.heads, self.neurons | other.neurons)

    def without(self, units) -> "Mask":
        m = Mask(self.heads.clone(), self.neurons.clone())
        for kind, x in (("head", m.heads), ("neuron", m.neurons)):
            idx = [u[1:] for u in units if u[0] == kind]
            if idx:
                i = torch.tensor(idx, dtype=torch.long, device=x.device).T
                x[i[0], i[1], i[2]] = False
        return m


class NativeUnits:
    """Runs native-unit circuits on one text (the model's own weights, float32 on Native's device)."""

    def __init__(self, nat: native.Native, ids: list[int], targets: list[int]):
        import vpd_model

        self.nat, self.ids, self.targets, self.T = nat, list(ids), list(targets), len(ids)
        tg = nat.target
        self.tg, self.L, self.H, self.hd = tg, tg.n_layer, tg.n_head, tg.hd
        self.W = {k: [tg.site(native.site_name(i, k)).W for i in range(self.L)] for k in vpd_model.KINDS}
        self.Dm = self.W["c_fc"][0].shape[0]
        self.emb = tg.wte[torch.tensor(self.ids, device=nat.dev)]
        self.ref = nat.reference([self.ids], self.targets)
        self._cache: tuple = (None, None)  # (a base Mask, its keys and values per layer)
        self.runs = 0  # sets measured (kl's results)

    def full(self) -> Mask:
        dev = self.nat.dev
        return Mask(torch.ones(self.L, self.T, self.H, dtype=torch.bool, device=dev),
                    torch.ones(self.L, self.T, self.Dm, dtype=torch.bool, device=dev))

    def at(self, positions) -> Mask:
        """Every unit at the positions, none elsewhere."""
        m = self.full()
        keep = torch.zeros(self.T, dtype=torch.bool, device=self.nat.dev)
        keep[list(positions)] = True
        m.heads &= keep[None, :, None]
        m.neurons &= keep[None, :, None]
        return m

    @torch.no_grad()
    def _forward(self, base: Mask, removals: list[list[tuple]], s: int, keep: bool = False) -> torch.Tensor:
        return self._forward_impl(base, removals, s, keep=keep)

    def _forward_impl(self, base: Mask, removals: list[list[tuple]], s: int, keep: bool = False,
                      strengths: tuple[torch.Tensor, torch.Tensor] | None = None,
                      means: dict | None = None) -> torch.Tensor:
        """log q at the targets [len(removals), len(targets), V]: each row the base set without its removals (at
        positions >= s). It computes the positions from s on that hold base units, and the targets: a position holding
        none keeps its embedding through every layer, so its keys and values, and those before s, are the base run's
        (keep: compute every position and store them, s = 0)."""
        import vpd_model

        if means is not None and (not keep or strengths is None or s != 0 or removals != [[]]):
            raise ValueError('mean strengths require full-context execution without cached removals')
        tg, B, T, H, hd, dev = self.tg, len(removals), self.T, self.H, self.hd, self.nat.dev
        if keep:
            comp = torch.arange(T, device=dev)
        else:
            on = base.heads.any(2).any(0) | base.neurons.any(2).any(0)
            on[:s] = False
            on[self.targets] = True
            comp = on.nonzero().flatten()
        n = len(comp)
        where = torch.full((T,), -1, dtype=torch.long, device=dev)
        where[comp] = torch.arange(n, device=dev)
        x = self.emb[comp][None].expand(B, -1, -1).clone()
        cache = None if keep else self._cache[1]
        kept = []
        causal = torch.arange(T, device=dev)[None] <= comp[:, None]  # [n, T]
        rows = {"head": [[], [], [], []], "neuron": [[], [], [], []]}
        for b, rem in enumerate(removals):
            for kind, l, t, j in rem:
                if t >= s:
                    r = rows[kind]
                    r[0].append(b)
                    r[1].append(l)
                    r[2].append(t)
                    r[3].append(j)
        idx = {}
        for kind, r in rows.items():
            b, l, t, j = (torch.tensor(v, dtype=torch.long, device=dev) for v in r)
            t = where[t] if len(t) else t
            ok = t >= 0  # a removal at a position holding no base unit removes nothing
            idx[kind] = (b[ok], l[ok], t[ok], j[ok])
        cos, sin = tg.cos[comp], tg.sin[comp]
        for i in range(self.L):
            z = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
            heads = lambda y: y.view(B, n, H, hd).transpose(1, 2)  # noqa: E731
            q, k, v = heads(z @ self.W["q_proj"][i].T), heads(z @ self.W["k_proj"][i].T), heads(z @ self.W["v_proj"][i].T)
            rot = lambda y: y * cos + torch.cat([-y[..., hd // 2:], y[..., :hd // 2]], -1) * sin  # noqa: E731
            q, k = rot(q), rot(k)
            if keep:
                if strengths is None:
                    kept.append((k[:1], v[:1]))
            else:
                kk, vv = cache[i][0].expand(B, -1, -1, -1).clone(), cache[i][1].expand(B, -1, -1, -1).clone()
                kk[:, :, comp], vv[:, :, comp] = k, v
                k, v = kk, vv
            P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
            hm = (base.heads if strengths is None else strengths[0])[i, comp].T[None].expand(B, -1, -1).float().clone()  # [B, H, n]
            sel = idx["head"][1] == i
            if sel.any():
                hm[idx["head"][0][sel], idx["head"][3][sel], idx["head"][2][sel]] = 0.0
            if means is None:
                att = ((P @ v) * hm[..., None]).transpose(1, 2).reshape(B, n, -1)
            else:
                att = ((P @ v) * hm[..., None] + (1 - hm[..., None]) * means['heads'][i][None, :, None, :]).transpose(1, 2).reshape(B, n, -1)
            x = x + att @ self.W["o_proj"][i].T
            z2 = vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps)
            nm = (base.neurons if strengths is None else strengths[1])[i, comp][None].expand(B, -1, -1).float().clone()  # [B, n, d_mlp]
            sel = idx["neuron"][1] == i
            if sel.any():
                nm[idx["neuron"][0][sel], idx["neuron"][2][sel], idx["neuron"][3][sel]] = 0.0
            if means is None:
                x = x + (vpd_model.gelu_tanh(z2 @ self.W["c_fc"][i].T) * nm) @ self.W["down_proj"][i].T
            else:
                act = vpd_model.gelu_tanh(z2 @ self.W["c_fc"][i].T)
                x = x + (act * nm + (1 - nm) * means['neurons'][i][None, None, :]) @ self.W["down_proj"][i].T
        if keep and strengths is None:
            self._cache = (base, kept)
        final = vpd_model.rms(x[:, where[self.targets]], tg.ln_f, tg.eps)
        return torch.log_softmax((final @ tg.wte.T).float(), -1)

    @torch.enable_grad()
    def strength_log_probs(self, heads: torch.Tensor, neurons: torch.Tensor, *, means: dict | None = None) -> torch.Tensor:
        """Full-context native execution with differentiable output multipliers in [0,1].

        Head multipliers precede O projection; neuron multipliers follow GELU and
        precede down projection. Same algebra as ordinary execution, no remainder,
        frozen activation trace or cache read/write. Does not increment removal runs.
        Optional fixed means replace zero as the omitted output: a*actual+(1-a)*mean.
        Means have heads [L,H,hd], neurons [L,Dm], matching model dtype/device;
        they are detached constants. Every position executes even when its strengths
        are zero: its mean residual can affect later attention. Binary strengths
        provide ordinary mean-ablated execution through this same full-context path.
        """
        expected = ((self.L, self.T, self.H), (self.L, self.T, self.Dm))
        for value, shape in zip((heads, neurons), expected):
            if value.shape != shape or not value.is_floating_point() or value.device != self.emb.device:
                raise ValueError('strength shape/dtype/device mismatch')
            if not torch.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
                raise ValueError('strengths must be finite within [0,1]')
        if means is not None:
            if not isinstance(means, dict):
                raise ValueError('means must contain heads and neurons tensors')
            fixed = {}
            for key, shape in (('heads', (self.L, self.H, self.hd)), ('neurons', (self.L, self.Dm))):
                value = means.get(key)
                if (not isinstance(value, torch.Tensor) or value.shape != shape
                        or not value.is_floating_point() or value.dtype != self.emb.dtype
                        or value.device != self.emb.device or not torch.isfinite(value).all()):
                    raise ValueError('mean shape/dtype/device/finite mismatch')
                fixed[key] = value.detach()
            means = fixed
        return self._forward_impl(self.full(), [[]], 0, keep=True, strengths=(heads, neurons), means=means)

    @torch.enable_grad()
    def integrated_unit_scores(self, steps: int = 4, *, means: dict | None = None) -> dict:
        """Negative mean direct-KL gradients at fixed uniform midpoint strengths.

        A deterministic proposal score for each native unit, not a finite-deletion
        guarantee. Scores may be negative. Costs exactly `steps` full-context
        forward/backward evaluations, sequentially; no ablation scan or selection.
        Caller must physically verify any proposed Boolean mask. Midpoint quadrature
        is approximate, not an exact completeness identity.
        midpoint_KL_bits uses existing raw FP32 carry.kl; normalized final fidelity
        is a separate physical verification.
        Optional fixed means define the same mean-background path as
        strength_log_probs; zero remains the default. Scores from the two paths
        are different proposal rules, not interchangeable causal measurements.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        full = self.full()
        score_h = torch.zeros_like(full.heads, dtype=torch.float32)
        score_n = torch.zeros_like(full.neurons, dtype=torch.float32)
        losses = []
        for i in range(steps):
            a = (i + .5) / steps
            h = torch.full_like(score_h, a, requires_grad=True)
            n = torch.full_like(score_n, a, requires_grad=True)
            loss = carry.kl(self.ref.detach(), self.strength_log_probs(h, n, means=means)).sum()
            gh, gn = torch.autograd.grad(loss, (h, n))
            if not torch.isfinite(loss) or not torch.isfinite(gh).all() or not torch.isfinite(gn).all():
                raise ValueError('nonfinite native ranking loss/gradient')
            score_h -= gh.detach() / steps
            score_n -= gn.detach() / steps
            losses.append(float(loss.detach()))
        return {'heads': score_h, 'neurons': score_n, 'midpoint_KL_bits': losses,
                'steps': steps, 'forward_evaluations': steps, 'backward_evaluations': steps,
                'scope': 'proposal only; direct Boolean verification required'}

    def kl(self, base: Mask, removals: list[list[tuple]] | None = None) -> list[float]:
        """The KL in bits of the model at the targets from each run: the base set alone (removals None), or the base
        set without each row's removals, batched by their first position."""
        self.runs += 1 if removals is None else len(removals)
        if self._cache[0] is not base or removals is None:
            k0 = carry.kl(self.ref, self._forward(base, [[]], 0, keep=True)).tolist()
            if removals is None:
                return k0
        out = [0.0] * len(removals)
        order = sorted(range(len(removals)), key=lambda r: min((u[2] for u in removals[r]), default=self.T - 1))
        for s0 in range(0, len(order), BATCH):
            rs = order[s0:s0 + BATCH]
            s = min(min((u[2] for u in removals[r]), default=self.T - 1) for r in rs)
            s = min(s, min(self.targets))
            got = carry.kl(self.ref, self._forward(base, [removals[r] for r in rs], s)).tolist()
            for r, k in zip(rs, got):
                out[r] = k
            native.free(self.nat.dev)
        return out


# ---- mean ablation (the lead's 10-14 native builder): a unit left out writes its mean, everything else runs live

ENTRIES = 2 ** 27  # mask floats per batched run (every layer and position's heads and neurons of each row)


class UnitRuns:
    """The runs every native-unit model adapter shares (MeanUnits: vpd4l; HFUnits: an HF decoder such as Qwen3): rows per
    batch, the all-live masks, the clean capture and the batched KL; each adapter gives logq(live, frozen=None,
    capture=None), ref, T, L, H, hd, Dm and dev. passes counts the sequences run (forward rows; a backward pass's rows
    count once more)."""
    passes = 0

    def rows(self) -> int:
        """Rows per batched run (ENTRIES mask floats)."""
        return max(1, ENTRIES // (self.L * self.T * (self.H + self.Dm)))

    def ones(self, B: int = 1) -> tuple:
        return (torch.ones(B, self.L, self.T, self.H, device=self.dev), torch.ones(B, self.L, self.T, self.Dm, device=self.dev))

    @torch.no_grad()
    def capture(self) -> dict:
        """Every unit's output in the clean run: {"heads": [L, T, H, hd], "neurons": [L, T, d_mlp]} (kept)."""
        if self._clean is None:
            cap = {}
            self.logq(self.ones(), capture=cap)
            self.passes += 1
            self._clean = {k: torch.stack(v) for k, v in cap.items()}
        return self._clean

    @torch.no_grad()
    def kl(self, live_rows, frozen_rows=None, lp: bool = False) -> list[float]:
        """KL in bits of the model at the targets from each row's run (lp: log2 p of the model's top token at each
        target, summed, instead); live_rows (and frozen_rows) yield one row's (heads, neurons) each [L, T, H] and [L, T,
        d_mlp], run rows() at a time."""
        out, part, fpart = [], [], []
        fit = iter(frozen_rows) if frozen_rows is not None else None
        y = self.ref[0].argmax(-1)

        def flush():
            live = tuple(torch.stack([p[j] for p in part]).float() for j in range(2))
            fr = tuple(torch.stack([p[j] for p in fpart]) for j in range(2)) if fit is not None else None
            lq = self.logq(live, fr)
            self.passes += len(part)
            if lp:
                out.extend((lq.gather(-1, y.view(1, -1, 1).expand(lq.shape[0], -1, 1)).squeeze(-1).sum(-1) / math.log(2.0)).tolist())
            else:
                out.extend(carry.kl(self.ref, lq).tolist())
            part.clear()
            fpart.clear()
            native.free(self.dev)
        for row in live_rows:
            part.append(row)
            if fit is not None:
                fpart.append(next(fit))
            if len(part) >= self.rows():
                flush()
        if part:
            flush()
        return out


class MeanUnits(UnitRuns):
    """Native-unit circuits on one text in mean-ablation semantics: a unit kept computes as in the model from the run's
    own inputs; a unit left out writes its mean over the reference texts (unit_means) in place of its output (a head
    its output before o_proj, a neuron its value after the GELU); the embedding, every norm, every attention pattern and
    the residual stream run live. Every unit kept is the model; none kept is the all-mean run. Removing a unit from the
    whole model is a real-model edit. The text is cut after its last target (the model is causal).

    logq(live, frozen=None): log q at the targets [B, targets, V]; live = (heads [B or 1, L, T, H], neurons [B or 1, L, T,
    d_mlp]) in [0, 1], 1 a unit's own output, 0 its fill (in between a mix: integrated gradients); the fill is the
    unit's mean, or where frozen (bool, same shapes) its value in the clean run (capture()): the mediated share's
    outside units. Differentiable in live."""

    def __init__(self, nat: native.Native, ids: list[int], targets: list[int], means: dict | None):
        tg = nat.target
        self.nat, self.tg, self.dev = nat, tg, nat.dev
        self.T = max(targets) + 1
        self.ids, self.targets = list(ids)[:self.T], list(targets)
        self.L, self.H, self.hd = tg.n_layer, tg.n_head, tg.hd
        self.W = {k: [tg.site(native.site_name(i, k)).W for i in range(self.L)] for k in ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")}
        self.b = {k: [tg.site(native.site_name(i, k)).b for i in range(self.L)] for k in self.W}  # TLTarget's biases (vpd4l: None)
        self.Dm = self.W["c_fc"][0].shape[0]
        self.emb = tg.embed(torch.tensor(self.ids, device=self.dev))
        self.ref = nat.reference([list(ids)], self.targets) if means is not None else None  # (capture alone: none)
        self.means = None if means is None else {k: means[k].to(self.dev) for k in ("heads", "neurons")}
        self._clean = None

    def logq(self, live: tuple, frozen: tuple | None = None, capture: dict | None = None) -> torch.Tensor:
        tg, T, H, hd = self.tg, self.T, self.H, self.hd
        hm, nm = live
        B = max(hm.shape[0], nm.shape[0])
        if frozen is not None and self._clean is None:
            self.capture()
        x = self.emb[None].expand(B, -1, -1)
        for i in range(self.L):
            z = tg.norm(x, 2 * i)
            heads = lambda y: y.view(B, T, H, hd).transpose(1, 2)  # noqa: E731
            q, k, v = (heads(self._mat(z, kd, i)) for kd in ("q_proj", "k_proj", "v_proj"))
            y = F.scaled_dot_product_attention(tg._rope(q, T), tg._rope(k, T), v, is_causal=True)  # [B, H, T, hd]
            if capture is not None:
                capture.setdefault("heads", []).append(y[0].transpose(0, 1))  # [T, H, hd]
            if self.means is not None:
                fill = self.means["heads"][i][None, :, None, :]  # [1, H, 1, hd]
                if frozen is not None:
                    fill = torch.where(frozen[0][:, i].transpose(1, 2)[..., None], self._clean["heads"][i].transpose(0, 1)[None], fill)
                m = hm[:, i].transpose(1, 2)[..., None]  # [B, H, T, 1]
                y = m * y + (1 - m) * fill
            x = x + self._mat(y.transpose(1, 2).reshape(B, T, -1), "o_proj", i)
            g = tg.act(self._mat(tg.norm(x, 2 * i + 1), "c_fc", i))  # [B, T, d_mlp]
            if capture is not None:
                capture.setdefault("neurons", []).append(g[0])
            if self.means is not None:
                fill = self.means["neurons"][i][None, None, :]
                if frozen is not None:
                    fill = torch.where(frozen[1][:, i], self._clean["neurons"][i][None], fill)
                g = nm[:, i] * g + (1 - nm[:, i]) * fill
            x = x + self._mat(g, "down_proj", i)
        return torch.log_softmax(tg.unembed(tg.norm(x[:, self.targets], -1)).float(), -1)

    def _mat(self, z: torch.Tensor, kind: str, i: int) -> torch.Tensor:
        """z through layer i's matrix kind, with its bias where the model has one."""
        out = z @ self.W[kind][i].T
        return out if self.b[kind][i] is None else out + self.b[kind][i]


class HFUnits(UnitRuns):
    """MeanUnits' semantics on an HF decoder (Qwen3 and Llama-like models: model.model.layers[i].self_attn.o_proj and
    .mlp.down_proj, the unembedding model.lm_head; a PeftModel as given, its active adapter on): forward pre-hooks on each
    layer's attention output projection, whose input is the heads' outputs [B, T, H x hd], and on its MLP down
    projection, whose input is the neurons after the activation [B, T, d_mlp], mix each unit's output with its fill
    (its mean, or its clean value where frozen). The model's own dtype (bf16 runs mix in float32 and cast back); the
    text cut after its last target. hook(ids, edit_heads, edit_neurons): any other edit through the same hooks (edit_*(layer,
    x) -> x at every position: weight edits)."""

    def __init__(self, model, ids: list[int], targets: list[int], means: dict | None):
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = core.config
        self.model, self.core, self.layers = model, core, core.model.layers
        self.L, self.H = cfg.num_hidden_layers, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.Dm = cfg.intermediate_size
        self.dev = next(core.parameters()).device
        self.T = max(targets) + 1
        self.ids, self.targets = list(ids)[:self.T], list(targets)
        self.means = None if means is None else {k: means[k].to(self.dev, torch.float32) for k in ("heads", "neurons")}
        if means is not None:  # each position's fill: its token's own means where hf_means gives them, else the unit's mean
            fh = self.means["heads"][:, None].repeat(1, self.T, 1, 1)  # [L, T, H, hd]
            fn = self.means["neurons"][:, None].repeat(1, self.T, 1)  # [L, T, d_mlp]
            for t, z in enumerate(self.ids):
                if z in means.get("tokens", {}):
                    fh[:, t], fn[:, t] = (x.to(self.dev, torch.float32) for x in means["tokens"][z])
            self.fill = (fh, fn)
        self._clean = None
        self.passes = 0
        with torch.no_grad():
            self.ref = self.hook(1) if means is not None else None

    def hook(self, B: int, edit_heads=None, edit_neurons=None, capture: dict | None = None, ids: list[int] | None = None) -> torch.Tensor:
        """log q at the targets [B, targets, V] of B copies of the text (or ids) with the hooks' edits."""
        handles = []
        for i, layer in enumerate(self.layers):
            def pre_o(mod, args, i=i):
                x = args[0]
                if capture is not None:
                    capture.setdefault("heads", []).append(x[0].float().view(-1, self.H, self.hd))
                return (x if edit_heads is None else edit_heads(i, x),) + tuple(args[1:])

            def pre_d(mod, args, i=i):
                x = args[0]
                if capture is not None:
                    capture.setdefault("neurons", []).append(x[0].float())
                return (x if edit_neurons is None else edit_neurons(i, x),) + tuple(args[1:])
            handles += [layer.self_attn.o_proj.register_forward_pre_hook(pre_o), layer.mlp.down_proj.register_forward_pre_hook(pre_d)]
        try:
            x = torch.tensor([ids if ids is not None else self.ids], device=self.dev).expand(B, -1)
            h = self.core.model(input_ids=x).last_hidden_state[:, self.targets if ids is None else [len(ids) - 1]]
            return torch.log_softmax(self.core.lm_head(h).float(), -1)
        finally:
            for hd in handles:
                hd.remove()

    def logq(self, live: tuple, frozen: tuple | None = None, capture: dict | None = None) -> torch.Tensor:
        hm, nm = live
        B = max(hm.shape[0], nm.shape[0])
        if frozen is not None and self._clean is None:
            self.capture()
        if self.means is None:
            return self.hook(B, capture=capture)
        T, H, hd = self.T, self.H, self.hd

        def heads(i, x):
            fill = self.fill[0][i][None]  # [1, T, H, hd]
            if frozen is not None:
                fill = torch.where(frozen[0][:, i][..., None], self._clean["heads"][i][None], fill)
            m = hm[:, i][..., None]  # [B, T, H, 1]
            return (m * x.float().view(B, T, H, hd) + (1 - m) * fill).view(B, T, H * hd).to(x.dtype)

        def neurons(i, x):
            fill = self.fill[1][i][None]  # [1, T, d_mlp]
            if frozen is not None:
                fill = torch.where(frozen[1][:, i], self._clean["neurons"][i][None], fill)
            return (nm[:, i] * x.float() + (1 - nm[:, i]) * fill).to(x.dtype)
        return self.hook(B, heads, neurons, capture)


@torch.no_grad()
def hf_means(model, tok, texts=None, max_len: int = 256, seqs: list | None = None, by_token: int = 0) -> dict:
    """unit_means' definition for an HF decoder: each head's output before o_proj and each neuron's value after the
    activation, averaged over every position of the reference sequences (vpd4l's train texts' strings, text6000-6099,
    re-tokenized by the model's tokenizer; or seqs, token id lists such as chat-formatted conversations; at most max_len
    tokens each): {"heads": [L, H, hd], "neurons": [L, d_mlp], "positions"}. by_token: every token seen at least this
    many times also gets its own means, {"tokens": {token id: (heads, neurons)}}, the fill at positions holding it
    (HFUnits): a unit whose value sits far from its mean at certain tokens (Qwen3's massive-activation neuron L2.n1821 at
    the first token and the template's delimiters, 17-37 bits when mean-ablated alone) keeps that token's level."""
    if seqs is None:
        seqs = []
        for n in (MEANS_TEXTS if texts is None else texts):
            text = json.loads((native.TEXTS / "vpd4l" / f"text{n}.json").read_text())["prompts"][0]["text"]
            seqs.append(tok(text, add_special_tokens=False)["input_ids"])
    seqs = [list(x)[:max_len] for x in seqs if len(x)]
    counts = {}
    for ids in seqs:
        for z in ids:
            counts[z] = counts.get(z, 0) + 1
    keep = sorted(z for z, c in counts.items() if by_token and c >= by_token)
    at = {z: i for i, z in enumerate(keep)}
    sh = sn = th = tn = None
    count = 0
    for ids in seqs:
        hu = HFUnits(model, ids, [len(ids) - 1], None)
        cap = {}
        hu.hook(1, capture=cap)
        hs, ns = torch.stack(cap["heads"]).cpu().double(), torch.stack(cap["neurons"]).cpu().double()  # [L, T, H, hd], [L, T, d_mlp]
        h, m = hs.sum(1), ns.sum(1)
        sh, sn = (h, m) if sh is None else (sh + h, sn + m)
        count += len(ids)
        if keep:
            if th is None:
                th, tn = torch.zeros(len(keep), *h.shape, dtype=torch.float64), torch.zeros(len(keep), *m.shape, dtype=torch.float64)
            for t, z in enumerate(ids):
                if z in at:
                    th[at[z]] += hs[:, t]
                    tn[at[z]] += ns[:, t]
    out = {"heads": (sh / count).float(), "neurons": (sn / count).float(), "positions": count}
    if keep:
        out["tokens"] = {z: ((th[i] / counts[z]).float(), (tn[i] / counts[z]).float()) for z, i in at.items()}
    return out


class GroupedUnits(UnitRuns):
    """Coarser native units on any adapter (the lead's 10-14, for models whose neuron circuits run to 100k+ units): each
    layer's neurons in groups (groups [L, d_mlp] of group ids, neuron_groups'), a group at a position one unit, mean-
    ablated together; the heads as they are. Its masks are (heads [B, L, T, H], groups [B, L, T, G]); the inner adapter
    runs them with each neuron at its group's strength."""

    def __init__(self, inner: UnitRuns, groups: torch.Tensor):
        self.inner, self.groups = inner, groups.to(inner.dev)
        self.L, self.T, self.H, self.hd, self.dev = inner.L, inner.T, inner.H, inner.hd, inner.dev
        self.Dm = int(self.groups.max()) + 1
        self.ref, self.targets, self._clean = inner.ref, inner.targets, None
        self.passes = 0

    def rows(self) -> int:
        return self.inner.rows()

    def _neurons(self, g: torch.Tensor) -> torch.Tensor:  # [B, L, T, G] -> [B, L, T, d_mlp]
        idx = self.groups[None, :, None, :].expand(g.shape[0], -1, g.shape[2], -1)
        return torch.gather(g, 3, idx)

    def logq(self, live: tuple, frozen: tuple | None = None, capture: dict | None = None) -> torch.Tensor:
        fr = None if frozen is None else (frozen[0], self._neurons(frozen[1].to(torch.uint8)).bool())
        return self.inner.logq((live[0], self._neurons(live[1])), fr, capture)

    def capture(self) -> dict:
        return self.inner.capture()


@torch.no_grad()
def neuron_groups(write: list[torch.Tensor], G: int, iters: int = 20, seed: int = 0) -> torch.Tensor:
    """Each layer's neurons in G groups by their output directions (write[l]: [d_mlp, d_model], neuron j's write into the
    residual stream): spherical k-means on the unit directions (G random neurons to start, iters steps): neurons that
    write the same direction mean-ablate as one unit. [L, d_mlp] group ids."""
    out = []
    g = torch.Generator().manual_seed(seed)
    for w in write:
        x = torch.nn.functional.normalize(w.float(), dim=1)
        c = x[torch.randperm(len(x), generator=g)[:G].to(x.device)].clone()
        for _ in range(iters):
            a = (x @ c.T).argmax(1)
            for k in range(G):
                sel = x[a == k]
                if len(sel):
                    c[k] = torch.nn.functional.normalize(sel.sum(0), dim=0)
        out.append((x @ c.T).argmax(1))
    return torch.stack(out)


def neuron_writes(mu: UnitRuns) -> list[torch.Tensor]:
    """Each layer's neurons' writes into the residual stream [d_mlp, d_model]: the down projection's columns (an HF
    model's measured on the unit vectors, so a LoRA adapter's change counts)."""
    if isinstance(mu, MeanUnits):
        return [mu.W["down_proj"][i].T for i in range(mu.L)]
    out = []
    with torch.no_grad():
        for layer in mu.layers:
            d = layer.mlp.down_proj
            p = next(d.parameters())
            eye = torch.eye(mu.Dm, device=p.device, dtype=p.dtype)
            out.append((d(eye) - d(torch.zeros_like(eye[:1]))).float())
    return out


# ---- the native builder (the lead's 10-14: builder_fast's units="neurons"; the same recipe as pymove's VPD builder)

FISHER_SAMPLES = 32  # labels drawn from the model's distribution at each target for the ranking's Fisher
STAGES = (0.25, 0.5, 1.0, 1.0)  # the prefix targets as shares of the KL target: the first on the model's Fisher, each
# later one on the Fisher at the circuit before it (text9010: 17,444 units in one stage, 12,709 in four)
CLEAN_MAX = 512      # the inert cleanup measures single removals of at most this many units (the lowest-ranked kept)
INERT = 1e-3         # a unit whose single removal moves the KL by less than this many bits is inert


def log_sizes(n: int) -> list[int]:
    """1 to n on a log scale (two steps an octave), n included (connect.sizes)."""
    return sorted({min(n, int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(n, 1))) + 2)} | {n})


def kl_goal(ge: goal_eval.GoalEval, tx: goal_eval.Text, f1: float = 0.9) -> float:
    """The point KL a native circuit must reach: a VPD circuit's at F1 = f1 on goal_eval's log scale with the live
    remainders' bounds (KL_empty, KL_VPD), exp(ln KL_empty - f1 ln(KL_empty / KL_VPD)): native and VPD circuits at the
    same absolute fidelity."""
    e0, v0 = ge._bounds(tx, "live")
    return math.exp(math.log(e0) - f1 * (math.log(e0) - math.log(max(v0, goal_eval.KL_RESOLUTION))))


def fisher(mu: UnitRuns, seed: int, samples: int = FISHER_SAMPLES, live: tuple | None = None) -> torch.Tensor:
    """Every unit's sampled-label Fisher, flat (heads [L, T, H] then neurons [L, T, d_mlp]): 1/2 E_v (d log q_v /
    d m_u)^2 at the circuit live (None: every unit, the model), v drawn from the model's distribution at each target
    (numpy's stream from seed: the same draws on every device); at the model, the second-order KL rise of mean-ablating
    u alone (the first order is zero there). On text9031 its order keeps KL 0.41 bits with 95% of the units
    mean-ablated, the integrated gradients' (all-mean to the model) 3.9 bits at 10%, a random order's 19.5 bits at
    90%."""
    p = mu.ref[0].double().exp().cpu().numpy()
    rng = np.random.default_rng([int(seed) % (1 << 63), 31])
    v = torch.tensor(np.stack([rng.choice(p.shape[1], samples, p=row / row.sum()) for row in p], 1), device=mu.dev)  # [S, targets]
    out = torch.zeros(mu.L * mu.T * (mu.H + mu.Dm), device=mu.dev)
    base = live if live is not None else tuple(y[0] for y in mu.ones())  # each [L, T, H or d_mlp]
    step = max(1, mu.rows() // 3)  # (a backward pass holds the activations and the masks' gradients)
    for s0 in range(0, samples, step):
        vv = v[s0:s0 + step]
        hm, nm = (x[None].expand(len(vv), -1, -1, -1).clone().requires_grad_(True) for x in base)
        mu.logq((hm, nm)).gather(-1, vv[..., None]).sum().backward()
        mu.passes += 2 * len(vv)  # (forward and backward)
        out += 0.5 * torch.cat([hm.grad.pow(2).sum(0).flatten(), nm.grad.pow(2).sum(0).flatten()]) / samples
        native.free(mu.dev)
    return out


class NoAdmittedCandidate(RuntimeError):
    """An opt-in gate admitted no tested candidate; never a faithful full fallback."""

    def __init__(self, record):
        super().__init__("NO_ADMITTED_CANDIDATE")
        self.record = record


def build_units(mu: UnitRuns, seed: int, goal: float, stages=STAGES, samples: int = FISHER_SAMPLES, clean_max: int = CLEAN_MAX,
                halving: int = 64, admit=None, max_gate_calls: int = 16) -> tuple[tuple, dict]:
    """A circuit of native units (heads and neurons at positions) in mean-ablation semantics on any adapter (MeanUnits:
    vpd4l; HFUnits: Qwen3), the smallest the recipe finds whose point KL is at most goal, every decision measured, the
    Fisher only ordering:
      (1) rank: every unit by its sampled-label Fisher at the model (fisher: one batched backward pass). The second-order
          order is the one that works: on text9031 it keeps KL 0.41 bits with 95% of the units mean-ablated, where the
          integrated gradients' order (all-mean to the model) costs 3.9 bits with only its lowest 10% removed and a
          random order 19.5 bits at 90%;
      (2) prefix: the top k units for k on a log scale (up to every unit, the model), measured, the smallest passing;
          then halving between it and the size below that failed, to a 1/halving share; in stages, the targets a share
          of goal rising to it, each later stage ranking the last one's units by the Fisher at that circuit;
      (3) inert cleanup: the largest log-scale suffix of the lowest-ranked kept units whose removal still passes, then
          the single removals of the clean_max lowest-ranked kept units measured and those under INERT bits removed
          together where that still passes (else the most inert half, and so on).
    admit is an optional callable on the two live masks, returning a JSON-compatible
    dict with Boolean passed. Every accepted stage and cleanup must pass it; at most
    max_gate_calls distinct masks are evaluated (cached checks do not spend calls).
    The empty mask is eligible. No passing tested candidate raises NoAdmittedCandidate
    with its evidence record; there is no implicitly admitted full fallback. This is
    minimum among tested passing candidates, not a globally minimal circuit.
    Returns ((heads [L, T, H], neurons [L, T, d_mlp]) float masks over the adapter's positions, record)."""
    t0 = time.perf_counter()
    L, T, H, Dm, dev = mu.L, mu.T, mu.H, mu.Dm, mu.dev
    nh = L * T * H
    stamp = {}

    def mask_of(idx) -> tuple:  # the units at flat indices idx live
        h = torch.zeros(nh, device=dev)
        n = torch.zeros(L * T * Dm, device=dev)
        h[idx[idx < nh]], n[idx[idx >= nh] - nh] = 1.0, 1.0
        return h.view(L, T, H), n.view(L, T, Dm)

    gate_cache, gate_records = {}, []
    gate_seconds, gate_exhausted = 0.0, False
    if admit is not None:
        if not callable(admit) or type(max_gate_calls) is not int or max_gate_calls < 0:
            raise ValueError("callable admission and nonnegative integer gate budget required")
        if not math.isfinite(goal) or goal < 0:
            raise ValueError("finite nonnegative point goal required with admission")

    def gate_record(status):
        return {"status": status, "calls": len(gate_records), "max_calls": max_gate_calls,
                "budget_exhausted": gate_exhausted, "seconds": gate_seconds,
                "evaluations": list(gate_records), "passes_excludes_admission": True,
                "scope": "gate calls are not a physical-forward cap; tested masks, not a global minimum"}

    def accepted(idx, where):
        nonlocal gate_seconds, gate_exhausted
        if admit is None:
            return True
        live = mask_of(idx)
        packed = torch.cat([x.bool().flatten() for x in live]).cpu().numpy().tobytes()
        key = hashlib.sha256(packed).hexdigest()
        if key in gate_cache:
            return gate_cache[key]["passed"]
        if len(gate_records) >= max_gate_calls:
            gate_exhausted = True
            return False
        started = time.perf_counter()
        result = admit(live)
        gate_seconds += time.perf_counter() - started
        if not isinstance(result, dict) or type(result.get("passed")) is not bool:
            raise ValueError("admission must return a dict with Boolean passed")
        result = json.loads(json.dumps(result, allow_nan=False))
        gate_cache[key] = result
        gate_records.append({"mask_sha256": key, "units": len(idx), "where": where, "result": result})
        return result["passed"]

    def point_ok(value, target):
        return value <= target and (admit is None or math.isfinite(value))

    # (1) rank
    score = fisher(mu, seed, samples)
    rank = torch.argsort(score, descending=True)
    empty_kl = mu.kl([mask_of(rank[:0])])[0]
    stamp["rank"] = time.perf_counter() - t0

    # (2) prefix on a log scale, then halving, in stages: each later stage ranks the last stage's units by the Fisher
    # at that circuit
    def prefix(order, target) -> tuple:
        sizes = log_sizes(len(order))
        got = list(zip(sizes, mu.kl(mask_of(order[:k]) for k in sizes)))
        if admit is None:
            k_star = next((k for k, v in got if v <= target), len(order))  # historical point-only recipe
        else:
            k_star = next((k for k, v in got if point_ok(v, target) and accepted(order[:k], "prefix")), None)
            if k_star is None:
                raise NoAdmittedCandidate({"admission": gate_record("NO_ADMITTED_CANDIDATE"),
                                           "point_candidates": got, "point_target": target,
                                           "seconds": time.perf_counter() - t0})
        lo = max([k for k, _ in got if k < k_star] or [0])
        kl = dict(got).get(k_star)
        while k_star - lo > max(1, k_star // halving):
            mid = (lo + k_star) // 2
            v = mu.kl([mask_of(order[:mid])])[0]
            got.append((mid, v))
            lo, k_star, kl = (lo, mid, v) if point_ok(v, target) and accepted(order[:mid], "halving") else (mid, k_star, kl)
        return order[:k_star], kl, got
    measured, keep = [], rank
    if admit is not None and point_ok(empty_kl, goal) and accepted(rank[:0], "empty"):
        live = mask_of(rank[:0])
        return live, {"kl": empty_kl, "kl_target": goal, "kl_empty": empty_kl, "prefix": [],
                      "heads": 0, "neurons": 0, "units": 0, "of": len(rank), "passes": mu.passes,
                      "seconds": time.perf_counter() - t0, "stages": stamp, "score": score,
                      "order": rank[:0], "admission": gate_record("ADMITTED"), "passes_excludes_admission": True}

    for i, share in enumerate(stages):
        if i:
            f = fisher(mu, seed, samples, live=mask_of(keep))
            keep = keep[torch.argsort(f[keep], descending=True)]
        keep, kl_star, got = prefix(keep, share * goal)
        measured.append(got)
    stamp["prefix"] = time.perf_counter() - t0 - stamp["rank"]

    # (3) inert cleanup: the lowest-ranked suffix, then measured singles
    asc = keep.flip(0)  # lowest score first
    js = [j for j in log_sizes(len(asc)) if j < len(asc)]
    if js:
        got = mu.kl(mask_of(asc[j:]) for j in js)
        passing = [j for j, v in zip(js, got) if point_ok(v, goal)]
        if admit is not None:
            first = next((j for j in sorted(passing, reverse=True) if accepted(asc[j:], "suffix_cleanup")), None)
            passing = [] if first is None else [first]
        if passing:
            j = max(passing)
            asc, kl_star = asc[j:], got[js.index(j)]
    probe = asc[:min(clean_max, len(asc) - 1)]
    if len(probe):
        cur = mask_of(asc)

        def drop(i):
            h, n = cur[0].clone(), cur[1].clone()
            (h.view(-1) if i < nh else n.view(-1))[i if i < nh else i - nh] = 0.0
            return h, n
        d = np.array(mu.kl(drop(int(i)) for i in probe)) - kl_star
        inert = probe[torch.tensor(np.argsort(d), device=dev)][:int((d < INERT).sum())]
        while len(inert):
            rest = asc[~torch.isin(asc, inert)]
            v = mu.kl([mask_of(rest)])[0]
            if point_ok(v, goal) and accepted(rest, "inert_cleanup"):
                asc, kl_star = rest, v
                break
            inert = inert[:len(inert) // 2]
    stamp["cleanup"] = time.perf_counter() - t0 - stamp["rank"] - stamp["prefix"]
    live = mask_of(asc)
    rec = {"kl": kl_star, "kl_target": goal, "kl_empty": empty_kl, "prefix": [sorted(g) for g in measured],
           "heads": int(live[0].sum()), "neurons": int(live[1].sum()), "units": len(asc), "of": len(rank), "passes": mu.passes,
           "seconds": time.perf_counter() - t0, "stages": stamp, "score": score, "order": asc.flip(0)}
    if admit is not None:
        rec["admission"] = gate_record("ADMITTED")
        rec["passes_excludes_admission"] = True
    return live, rec


def build_native(ge: goal_eval.GoalEval, tx: goal_eval.Text, kl_target: float | None = None, means: dict | None = None,
                 log=None, *, shared_floors: bool = False, pool=None, max_gate_calls: int = 16) -> tuple[Mask, dict]:
    """build_units on vpd4l (MeanUnits) at kl_goal's KL by default: (Mask over the whole text, record); the positions
    are those the kept units sit at. shared_floors opts into every available SHARED
    family at the same VPD-derived search floors, never candidate-relative floors;
    pool is the construction/search pool, not the scorecard's eval pool. Eval remains
    separate. max_gate_calls bounds expensive shared-family candidate evaluations,
    not their physical forwards; rec passes excludes them and admission records their
    calls/time separately. The default point target inverts the actual shared point
    floor; explicit kl_target remains an additional caller-specified bound."""
    native_started = time.perf_counter()
    nat = ge.nat
    means = means if means is not None else unit_means(nat)
    mu = MeanUnits(nat, tx.ids, tx.targets, means)
    gate = None
    if shared_floors:
        vf = ge.vpd_floors(tx, pool, "search", shared_only=True)
        families = [f for f in vf["families"] if f in SHARED and vf["floors"].get(f) is not None]
        if not families or any(not math.isfinite(vf["floors"][f]) for f in families):
            raise ValueError("finite nonempty shared search floors required")
        floors = {f: vf["floors"][f] for f in families}
        nphi = NativePhi(ge, tx, "search", means=means)

        def gate(live):
            heads = torch.zeros(mu.L, len(tx.ids), mu.H, dtype=torch.bool, device=mu.dev)
            neurons = torch.zeros(mu.L, len(tx.ids), mu.Dm, dtype=torch.bool, device=mu.dev)
            heads[:, :mu.T], neurons[:, :mu.T] = live[0].bool(), live[1].bool()
            mask = Mask(heads, neurons)
            necessity_rows = None
            necessity_diagnostic = None
            if "N" in families:
                necessity_rows = nphi.necessity([mask])
                nr = necessity_rows[0]
                draws = nr.get("draws_n_spec") or []
                dt = nr.get("d_target")
                dc = nr.get("d_control")
                vt = nr.get("d_target_vpd")
                vc = nr.get("d_control_vpd")
                necessity_diagnostic = {
                    "d_target": dt, "d_control": dc,
                    "d_target_vpd": vt, "d_control_vpd": vc,
                    "numerator": dt - dc if dt is not None and dc is not None else None,
                    "denominator": vt - vc if vt is not None and vc is not None else None,
                    "controls": len(draws), "draws_n_spec": draws,
                    "raw_score": nr.get("_raw_n_spec"),
                }
                nscore = goal_eval.family_score(nr, "N")
                if nscore is None or not math.isfinite(nscore) or nscore < floors["N"]:
                    return {"passed": False, "scores": {"N": nscore}, "floors": floors,
                            "split": "search", "point_KL_bits": None,
                            "rejection_reason": "N", "skipped_families": [f for f in families if f != "N"],
                            "necessity": necessity_diagnostic}
            row = (nphi.rows([mask], necessity_rows=necessity_rows) if necessity_rows is not None
                   else nphi.rows([mask]))[0]
            scores = {f: goal_eval.family_score(row, f) for f in families}
            passed = all(scores[f] is not None and math.isfinite(scores[f]) and scores[f] >= floors[f] for f in families)
            return {"passed": passed, "scores": scores, "floors": floors, "split": "search",
                    "point_KL_bits": row.get("kl_point"), "necessity": necessity_diagnostic,
                    "rejection_reason": None if passed else "shared_family_floors"}

    if kl_target is not None:
        goal = kl_target
    elif shared_floors:
        if "point" not in floors:
            raise ValueError("shared point floor required to derive the default KL target")
        goal = kl_goal(ge, tx, f1=floors["point"])
    else:
        goal = kl_goal(ge, tx)
    shared_setup_seconds = time.perf_counter() - native_started
    live, rec = build_units(mu, tx.seed, goal, admit=gate, max_gate_calls=max_gate_calls)
    del rec["score"], rec["order"]
    Tf = len(tx.ids)
    heads = torch.zeros(mu.L, Tf, mu.H, dtype=torch.bool, device=mu.dev)
    neurons = torch.zeros(mu.L, Tf, mu.Dm, dtype=torch.bool, device=mu.dev)
    heads[:, :mu.T], neurons[:, :mu.T] = live[0].bool(), live[1].bool()
    S = Mask(heads, neurons)
    rec["positions"] = sorted(set((heads.any(2).any(0) | neurons.any(2).any(0)).nonzero().flatten().tolist()))
    if shared_floors:
        rec["shared_setup_seconds"] = shared_setup_seconds
        rec["native_total_seconds"] = time.perf_counter() - native_started
    if log:
        log(f"  native: {S.count()} units ({rec['heads']} heads, {rec['neurons']} neurons) at {len(rec['positions'])} positions "
            f"of {mu.T}, KL {rec['kl']:.3f} (target {goal:.3f}, empty {rec['kl_empty']:.1f}), {rec['seconds']:.1f} s")
    return S, rec


HF_KL_TARGET = 0.5   # build_hf's point KL in bits by default (vpd4l's VPD-matched targets on DEV20: 0.11-1.6, median about 0.6)
BUDGET = {"stages": (0.5, 1.0), "samples": 16, "clean_max": 0, "halving": 16}  # build_hf's settings within budget_passes
EFFECTS = 32  # build_hf measures the single removal of this many of the circuit's units (its highest-ranked)


def build_hf(model, ids: list[int], target: int, kl_target: float = HF_KL_TARGET, means: dict | None = None, tok=None,
             budget_passes: int | None = None, effects: int = EFFECTS, seed: int = 0, groups: torch.Tensor | None = None,
             log=None) -> dict:
    """The native builder on an HF decoder (HFUnits: Qwen3; a PeftModel with its active adapter on): the circuit of heads
    and neurons for the model's prediction at position target of ids in mean-ablation semantics, at most kl_target bits
    of KL there. means: hf_means' (with tok, computed when None). budget_passes: the BUDGET settings (fewer Fisher labels,
    two stages, no singles cleanup), for a caller charged by the sequences run (the oracle gate's 256 a text). Returns
    {"units": [(kind, layer, position, index, effect)], effect the measured drop in log2 p of the model's top token when
    the unit alone is mean-ablated from the whole model for the `effects` highest-ranked units, else None and the
    Fisher's predicted KL rise in "fisher"; "live" (the circuit's masks), "order" (its units' flat indices, rank order)
    and "runner" (the adapter: mediated_runs, recovery_count); "kl", "kl_target", "kl_empty", "positions", "layers" (units per layer),
    "passes", "seconds", "heads", "neurons"}. groups ([L, d_mlp], neuron_groups'): neuron groups in place of neurons (GroupedUnits;
    a unit ("group", layer, position, group id))."""
    t0 = time.perf_counter()
    if means is None:
        if tok is None:
            raise ValueError("build_hf: pass means (hf_means) or the tokenizer to compute them")
        means = hf_means(model, tok)
    mu = HFUnits(model, ids, [target], means)
    if groups is not None:
        mu = GroupedUnits(mu, groups)
    kw = BUDGET if budget_passes is not None else {}
    live, rec = build_units(mu, seed, kl_target, **kw)
    order, score = rec["order"], rec.pop("score")
    nh = mu.L * mu.T * mu.H
    units = []
    for i in order.tolist():
        if i < nh:
            l, t, j = i // (mu.T * mu.H), (i // mu.H) % mu.T, i % mu.H
            units.append(["head", l, t, j])
        else:
            k = i - nh
            l, t, j = k // (mu.T * mu.Dm), (k // mu.Dm) % mu.T, k % mu.Dm
            units.append(["neuron" if groups is None else "group", l, t, j])
    n_eff = min(effects, len(units)) if budget_passes is None else max(0, min(effects, budget_passes - mu.passes - 1))
    full = tuple(x[0] for x in mu.ones())
    lp0 = mu.kl([full], lp=True)[0] if n_eff else None

    def removed(u):
        h, n = full[0].clone(), full[1].clone()
        (h if u[0] == "head" else n)[u[1], u[2], u[3]] = 0.0
        return h, n
    eff = [lp0 - v for v in mu.kl((removed(u) for u in units[:n_eff]), lp=True)] if n_eff else []
    fs = score[order].tolist()
    rec.update(units=[u + [eff[i] if i < len(eff) else None] for i, u in enumerate(units)], fisher=fs,
               positions=sorted({u[2] for u in units}), layers=[sum(1 for u in units if u[1] == l) for l in range(mu.L)],
               passes=mu.passes, seconds=time.perf_counter() - t0, T=mu.T, live=live, runner=mu)
    if log:
        log(f"  native (HF): {len(units)} units ({rec['heads']} heads, {rec['neurons']} neurons) at {len(rec['positions'])} positions "
            f"of {mu.T}, KL {rec['kl']:.3f} (target {kl_target:.3f}, empty {rec['kl_empty']:.1f}), {rec['passes']} passes, "
            f"{rec['seconds']:.1f} s")
    return rec


def mediated_runs(mu: UnitRuns, live: tuple, seed: int, ks=(1, 3, 5), candidates: int = CLEAN_MAX, sets: list | None = None,
                  kind: str = "neuron") -> dict:
    """Criterion 6's mediated share on any adapter (generality's definition, the lead's 10-14), for the circuit live
    (heads [L, T, H], neurons or groups [L, T, ·] masks over the adapter's positions): its units ranked by their real-model
    single removal drop d(u) = log2 p(y) - log2 p(y | u mean-ablated) (y the model's top token at each target, summed),
    measured for its `candidates` units of largest Fisher (a native circuit holds tens of thousands); for the top k, the
    live drop (the real model with them mean-ablated) and the mediated drop (the circuit's units live, every unit outside
    it frozen at its clean value, the top k mean-ablated), m_k = D_med / D_live. sets: the removal sets given (named
    "set<i>"), in place of the top k (the same edits under another circuit). kind: the second unit kind's name."""
    inside = tuple(x.float() for x in live)
    on = [("head", *u) for u in live[0].nonzero().tolist()] + [(kind, *u) for u in live[1].nonzero().tolist()] if sets is None else []
    if on:
        T, H, M = mu.T, mu.H, live[1].shape[2]
        f = fisher(mu, seed)
        flat = torch.tensor([(l * T + t) * H + j if k == "head" else mu.L * T * H + (l * T + t) * M + j for k, l, t, j in on], device=mu.dev)
        on = [on[i] for i in torch.argsort(f[flat], descending=True)[:candidates].tolist()]

    def removed(base, us):
        h, n = base[0].clone(), base[1].clone()
        for k, l, t, j in us:
            (h if k == "head" else n)[l, t, j] = 0.0
        return h, n
    full = tuple(torch.ones_like(x) for x in inside)
    lp0 = mu.kl([full], lp=True)[0]
    drops = [lp0 - v for v in mu.kl((removed(full, [u]) for u in on), lp=True)]
    order = [on[i] for i in np.argsort(drops)[::-1]]
    outside = tuple(x == 0 for x in live)
    names = [f"top{k}" for k in ks] if sets is None else [f"set{i}" for i in range(len(sets))]
    sets = [order[:k] for k in ks] if sets is None else [[tuple(u) for u in R] for R in sets]
    lv = mu.kl((removed(full, R) for R in sets), lp=True)
    med = mu.kl([inside] + [removed(inside, R) for R in sets], (outside for _ in range(len(sets) + 1)), lp=True)
    out = {"units": int(sum(int(x.sum()) for x in live)), "measured": len(on), "single_drops": sorted(drops, reverse=True)[:16],
           "lp": lp0, "lp_frozen": med[0]}
    for name, R, a, md in zip(names, sets, lv, med[1:]):
        d_live, d_med = lp0 - a, med[0] - md
        out[name] = {"units": [list(u) for u in R], "live": d_live, "mediated": d_med, "share": d_med / d_live if abs(d_live) > 1e-9 else None}
    return out


def mediated_share(ge: goal_eval.GoalEval, tx: goal_eval.Text, S: Mask, means: dict | None = None, ks=(1, 3, 5),
                   candidates: int = CLEAN_MAX, sets: list | None = None) -> dict:
    """mediated_runs on vpd4l (MeanUnits) for a native circuit S (a Mask over the whole text)."""
    nat = ge.nat
    mu = MeanUnits(nat, tx.ids, tx.targets, means if means is not None else unit_means(nat))
    return mediated_runs(mu, (S.heads[:, :mu.T], S.neurons[:, :mu.T]), tx.seed, ks, candidates, sets)


def recovery_count(mu: UnitRuns, order: torch.Tensor, share: float = 0.9) -> dict:
    """The fewest of a circuit's units, in its rank order (flat indices, build_units' "order"), that recover share of the
    change in log2 p of the model's top token from the all-mean run to the model (the rest of the units at their means),
    on a log grid then halving, measured: {"k", "heads", "second" (neurons or groups), "lp_k", "lp_empty", "lp_model"}."""
    L, T, H, M = mu.L, mu.T, mu.H, mu.Dm
    nh = L * T * H

    def mask_of(idx):
        h = torch.zeros(nh, device=mu.dev)
        n = torch.zeros(L * T * M, device=mu.dev)
        h[idx[idx < nh]], n[idx[idx >= nh] - nh] = 1.0, 1.0
        return h.view(L, T, H), n.view(L, T, M)
    lo_lp, hi_lp = mu.kl([mask_of(order[:0]), tuple(x[0] for x in mu.ones())], lp=True)
    goal = lo_lp + share * (hi_lp - lo_lp)
    sizes = log_sizes(len(order))
    got = dict(zip(sizes, mu.kl((mask_of(order[:k]) for k in sizes), lp=True)))
    k = next((k for k in sizes if got[k] >= goal), len(order))
    lo = max([j for j in sizes if j < k] or [0])
    while k - lo > max(1, k // 64):
        mid = (lo + k) // 2
        got[mid] = mu.kl([mask_of(order[:mid])], lp=True)[0]
        lo, k = (lo, mid) if got[mid] >= goal else (mid, k)
    return {"k": k, "heads": int((order[:k] < nh).sum()), "second": int((order[:k] >= nh).sum()), "lp_k": got.get(k),
            "lp_empty": lo_lp, "lp_model": hi_lp, "share": share}


def build_all(tasks: list[str], out: Path, vpd: Path | None = None, device: str | None = None, mediated: bool = True,
              rev: str = "", log=print) -> None:
    """build_native on each task, OUT/<task>.json in the bake-off's format (generality's, the lead's 10-14): {"units":
    "native", "semantics": "mean_ablation", "nodes": Mask.units() as lists, "build_seconds", "builder", "record"} and,
    with mediated, the circuit's mediated_share; a task already written is skipped. vpd: VPD's saved answers (the KL
    target reads VPD's bounds)."""
    import lines

    nat = native.Native(device)
    ge = goal_eval.GoalEval(nat, chunk=32)
    means = unit_means(nat)
    saved = torch.load(vpd) if vpd else None
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        dest = out / f"{task}.json"
        if dest.exists():
            continue
        tx = lines.text_of(ge, native.TEXTS / "vpd4l" / f"{task}.json", saved)
        S, rec = build_native(ge, tx, means=means, log=lambda m, task=task: log(f"{task}{m}"))
        row = {"units": "native", "semantics": "mean_ablation", "nodes": [list(u) for u in S.units()],
               "build_seconds": rec["seconds"], "builder": f"native_units.build_native@{rev}", "device": str(nat.dev), "record": rec}
        if mediated:
            row["mediated"] = mediated_share(ge, tx, S, means)
            log(f"{task}: mediated share top 1/3/5 " + " ".join(f"{row['mediated'][f'top{k}']['share']:.3f}" for k in (1, 3, 5)))
        dest.write_text(json.dumps(row))
        native.free(nat.dev)


def screen(nu: NativeUnits, S: Mask, base_kl: float, log=None, units: list | None = None) -> dict:
    """Unit -> its measured removal effect (KL of the set without it, minus the set's): heads one by one; MLPs as
    wholes at each (layer, position), then the neurons of those moving the prediction at least SCREEN_FLOOR as much
    as the top MLP one by one (the others' neurons each get their MLP's effect divided among them). `units`: only
    these of S's units (default all)."""
    units = S.units() if units is None else units
    if len(units) <= RESCREEN[0]:
        return dict(zip(units, (k - base_kl for k in nu.kl(S, [[u] for u in units]))))
    heads = [u for u in units if u[0] == "head"]
    blocks = {}
    for u in units:
        if u[0] == "neuron":
            blocks.setdefault((u[1], u[2]), []).append(u)
    names = list(blocks)
    got = nu.kl(S, [[u] for u in heads] + [blocks[b] for b in names])
    effect = {u: k - base_kl for u, k in zip(heads, got)}
    block_effect = {b: k - base_kl for b, k in zip(names, got[len(heads):])}
    top = max(block_effect.values(), default=0.0)
    screened = [b for b in names if block_effect[b] >= SCREEN_FLOOR * top and top > 0]
    for b in names:
        if b not in screened:
            for u in blocks[b]:
                effect[u] = block_effect[b] / len(blocks[b])
    singles = [u for b in screened for u in blocks[b]]
    if log:
        log(f"  screen: {len(heads)} heads, {len(names)} MLPs, {len(singles)} neurons in the {len(screened)} that matter")
    for u, k in zip(singles, nu.kl(S, [[u] for u in singles])):
        effect[u] = k - base_kl
    return effect


def prune(nu: NativeUnits, positions, sizes=(), log=None) -> list[dict]:
    """The measured pruning curve from every unit at the positions: [{"size", "kl", "units" (at the sizes asked
    for)}], from S0 down to the empty set."""
    S = nu.at(positions)
    kl = nu.kl(S)[0]
    curve = [{"size": S.count(), "kl": kl}]
    effect = screen(nu, S, kl, log)
    rescreened = {n for n in RESCREEN if S.count() <= n}
    stops = sorted({k for k in sizes if 0 < k < S.count()}, reverse=True)
    while S.count() > 0:
        n = S.count()
        k = max(1, n // STEP)
        nxt = next((t for t in stops if t < n), 0)
        k = min(k, n - nxt)
        order = sorted(S.units(), key=lambda u: effect.get(u, -math.inf))
        S = S.without(order[:k])
        kl = nu.kl(S)[0]
        row = {"size": S.count(), "kl": kl}
        if S.count() in stops:
            row["units"] = S.units()
        curve.append(row)
        if log:
            log(f"  {S.count():7d} units  KL {kl:.4g}")
        due = [t for t in RESCREEN if S.count() <= t and t not in rescreened]
        if (due or S.count() <= FINE) and S.count() > 0:
            rescreened |= set(due)
            effect = screen(nu, S, kl, log)
    return curve


def f1(kl: float, empty: float, vpd: float) -> float | None:
    """goal_eval's F1 of a run's KL against the bounds."""
    r = goal_eval.recovered_log(empty, kl, vpd)
    return None if r is None else min(1.0, max(0.0, r))


def bits(k: int, positions: int, per_position: int) -> float:
    """k units' description length: each one choice among positions x per_position."""
    return k * math.log2(positions * per_position) if k else 0.0


def curves(split: str, n: int, sizes=SIZES, out: Path | None = None, log=print, offset: int = 0, stride: int = 1) -> list[dict]:
    """Per text of the split's first n (offset, offset + stride, ...): the native-unit pruning curve, and at each size asked for the native
    circuit's and VPD's own first k (its order, native.Native._vpd_order) KL, F1 and N on goal_eval's scale (remainders
    off, the native units having none; VPD's also live), with their node bits -> one row per text (appended to OUT)."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    rows = []
    for p in native.tasks(split)[:n][offset::stride]:
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
        positions = sorted(set(tx.targets) | {q for q, _ in tx.f2})
        empty, vpd = ge._bounds(tx, "off")
        nu = NativeUnits(nat, ids, targets)
        curve = prune(nu, positions, sizes, log=log)
        P, per = nat.positions(targets), nu.L * (nu.H + nu.Dm)
        stops = [c for c in curve if "units" in c]
        minus = nu.kl(nu.full(), [c["units"] for c in stops]) if stops else []
        native_rows = [{"size": c["size"], "kl": c["kl"], "f1": f1(c["kl"], empty, vpd), "kl_minus": m,
                        "n": goal_eval.necessity_log(empty, m, vpd), "bits": bits(c["size"], P, per), "units": c["units"]}
                       for c, m in zip(stops, minus)]
        order = nat._vpd_order(ids, targets)
        ks = [k for k in sorted({c["size"] for c in stops}) if k <= len(order)]
        vrows = ge.sufficiency(tx, [native.Graph(order[:k], complete=True) for k in ks]) if ks else []
        vpd_rows = [{"size": k, "kl": r["kl_off"], "f1": r["f1_off"], "n": r["n_off"], "f1_live": r["f1_live"], "n_live": r["n_live"],
                     "bits": bits(k, P, sum(nat.C.values()))} for k, r in zip(ks, vrows)]
        row = {"task": p.stem, "T": tx.T, "positions": positions, "kl_empty_off": empty, "kl_vpd_off": vpd,
               "curve": [{"size": c["size"], "kl": c["kl"], "f1": f1(c["kl"], empty, vpd)} for c in curve],
               "native": native_rows, "vpd_order": vpd_rows, "seconds": time.perf_counter() - t0}
        rows.append(row)
        if out:
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        at = {r["size"]: r for r in vpd_rows}
        log(f"{p.stem}: T {tx.T}, {len(positions)} positions, {curve[0]['size']} units, {len(curve)} steps, {row['seconds']:.0f} s; "
            + ", ".join(f"{r['size']}: native F1 {r['f1']:.2f} N {r['n']:.2f} / VPD F1 {at[r['size']]['f1']:.2f} N {at[r['size']]['n']:.2f}"
                        for r in native_rows if r["size"] in at and r["f1"] is not None and at[r["size"]]["f1"] is not None))
    return rows


def table(rows: list[dict]) -> str:
    """By size, the mean over the texts of the native circuits' and VPD's first k F1, N and node bits."""
    sizes = sorted({r["size"] for t in rows for r in t["native"]})
    lines = [["size", "texts", "native F1", "native N", "native bits", "VPD F1", "VPD N", "VPD bits", "VPD F1 live", "VPD N live"]]
    for k in sizes:
        nr = [r for t in rows for r in t["native"] if r["size"] == k]
        vr = [r for t in rows for r in t["vpd_order"] if r["size"] == k]

        def m(rs, key, fmt="{:.3f}"):
            v = [r[key] for r in rs if r.get(key) is not None]
            return fmt.format(float(np.mean(v))) if v else "-"
        lines.append([str(k), str(len(nr)), m(nr, "f1"), m(nr, "n"), m(nr, "bits", "{:.0f}"), m(vr, "f1"), m(vr, "n"),
                      m(vr, "bits", "{:.0f}"), m(vr, "f1_live"), m(vr, "n_live")])
    w = [max(len(x[i]) for x in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(x[i].rjust(w[i]) for i in range(len(x))) for x in lines)


# ---- equal effort: the teacher's search (teacher.py) on native units


class NativePruner:
    """teacher.Pruner's measured block removal over a native-unit set S (Mask); measured counts every set measured
    on nu since the pruner began."""

    def __init__(self, nu: NativeUnits, start: Mask):
        self.nu, self.S, self.runs0 = nu, start, nu.runs

    @property
    def measured(self) -> int:
        return self.nu.runs - self.runs0

    def without(self, groups: list[list[tuple]]) -> list[float]:
        """KL of S without each group of units."""
        if not groups:
            return []
        return self.nu.kl(self.S, groups)

    def kl(self) -> float:
        return self.nu.kl(self.S)[0]

    def blocks(self, level: str) -> list[list[tuple]]:
        """S's blocks at a teacher level: whole positions; sites (a layer's heads or its MLP's neurons at a position);
        chunks of K units of a site by index; single units."""
        units = self.S.units()
        if level == "subcomponent":
            return [[u] for u in units]
        key = {"position": lambda u: (u[2],), "site": lambda u: (u[2], u[1], u[0])}.get(level)
        groups: dict = {}
        if key is not None:
            for u in units:
                groups.setdefault(key(u), []).append(u)
            return list(groups.values())
        K = int(level[5:])  # chunkK
        for u in units:
            groups.setdefault((u[2], u[1], u[0]), []).append(u)
        out = []
        for g in groups.values():
            g = sorted(g, key=lambda u: u[3])
            out += [g[i:i + K] for i in range(0, len(g), K)]
        return out

    def prune(self, level: str, budget: float, rounds: int) -> int:
        """teacher.Pruner.prune: rounds of measuring each block's removal, ranking the blocks by the KL it leaves, and
        removing the largest prefix (sizes 2^(e/2)) whose measured removal keeps KL <= budget."""
        for r in range(rounds):
            blocks = self.blocks(level)
            if len(blocks) <= 1:
                return r
            single = self.without(blocks)
            if not any(x <= budget for x in single):
                return r
            ranked = [blocks[i] for i in sorted(range(len(blocks)), key=lambda i: single[i])]
            sizes = [k for k in sorted({int(2 ** (e / 2)) for e in range(0, 60)}) if k <= len(ranked)]
            got = self.without([[u for b in ranked[:k] for u in b] for k in sizes])
            ok = [k for k, x in zip(sizes, got) if x <= budget]
            if not ok:
                return r
            self.S = self.S.without([u for b in ranked[:max(ok)] for u in b])
        return rounds


def teach_native(ge: goal_eval.GoalEval, nu: NativeUnits, tx: goal_eval.Text, log=print) -> dict:
    """The teacher's search (teacher.teach) on native units, remainders off: the F1 circuit, first from the units at the
    targets alone (chunks of 4, then single units), else from every unit at every position, the analog of VPD's whole
    answer (positions, sites, chunks of 64, 16 and 4, single units), each level by teacher.ROUNDS rounds of measured
    removal keeping F1 >= teacher.RECOVERED; then teacher.joint_path's measured F1 and N path from it (from its JOINT_MAX units whose
    measured removal matters most when it is larger) down to teacher.MIN_SIZE (no VPD order to union with), N being
    removal from the whole model, and joined, as teacher.top_k joins VPD's top k, by the smallest prefix (at most
    teacher.K_MAX) with N >= teacher.N_TARGET of the native order: every unit by its measured removal from the whole
    model, most needed first (screen). Returns the F1 circuit, the top k, the
    path with each step's units, and the measured runs."""
    import teacher

    t0 = time.perf_counter()
    empty, vpd = ge._bounds(tx, "off")
    budget = empty ** (1 - teacher.RECOVERED) * vpd ** teacher.RECOVERED
    full = nu.full()
    pr = NativePruner(nu, nu.at(tx.targets))
    levels = {}

    def search(names, label):
        for level in names:
            t1 = time.perf_counter()
            rounds = pr.prune(level, budget, teacher.ROUNDS)
            k = pr.kl()
            levels[f"{label}{level}"] = {"rounds": rounds, "size": pr.S.count(), "kl": k, "seconds": time.perf_counter() - t1}
            if log:
                log(f"  {label}{level}: {pr.S.count()} units at {int((pr.S.heads.any(2) | pr.S.neurons.any(2)).any(0).sum())} positions, "
                    f"KL {k:.3g} (budget {budget:.3g}), {rounds} rounds, {pr.measured} runs, {time.perf_counter() - t1:.0f} s")
        return levels[f"{label}{names[-1]}"]["kl"] <= budget

    reached = search(("chunk4", "subcomponent"), "target ")
    if not reached:
        pr.S = full
        search(teacher.LEVELS, "")
    f1_set = pr.S
    f1_part = f1_set
    if f1_part.count() > JOINT_MAX:  # the joint path from the units whose measured removal matters most
        kl0 = nu.kl(f1_part)[0]
        effect = screen(nu, f1_part, kl0, log)
        keep = sorted(f1_part.units(), key=lambda u: -effect[u])[:JOINT_MAX]
        f1_part = f1_part.without(sorted(set(f1_part.units()) - set(keep)))

    t1 = time.perf_counter()
    kl_full = nu.kl(full)[0]
    effect = screen(nu, full, kl_full, log)
    order = sorted(effect, key=lambda u: -effect[u])

    def n_of(k):
        return teacher.scores(empty, vpd, empty, nu.kl(full, [order[:k]])[0])[1]

    lo, hi = 0, min(teacher.K_MAX, len(order))
    if n_of(hi) >= teacher.N_TARGET:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if n_of(mid) >= teacher.N_TARGET else (mid, hi)
    k = hi
    if log:
        log(f"  native order and top k: k = {k}, {pr.measured} runs, {time.perf_counter() - t1:.0f} s")
    top = Mask(torch.zeros_like(full.heads), torch.zeros_like(full.neurons))
    for kind, l, t, j in order[:k]:
        (top.heads if kind == "head" else top.neurons)[l, t, j] = True
    start = f1_part.union(top)

    def measure(S: Mask, removals: list[list[tuple]]) -> list[dict]:
        """Each S-without-removal circuit's KL_C and KL of the whole model without it, F1, N and their mean."""
        cs = pr.nu.kl(S, removals)
        units = S.units()
        rest = []
        for r in removals:
            drop = set(r)
            rest.append([u for u in units if u not in drop])
        ms = pr.nu.kl(full, rest)
        out = []
        for c, m in zip(cs, ms):
            f, n, j = teacher.scores(empty, vpd, c, m)
            out.append({"kl_c": c, "kl_minus": m, "f1": f, "n": n, "mean": j})
        return out

    f1_row = measure(f1_set, [[]])[0]
    S = start
    path = [{"size": S.count(), "units": S.units(), **measure(S, [[]])[0]}]
    while S.count() > teacher.MIN_SIZE:
        units = S.units()
        rows = measure(S, [[u] for u in units])
        ranked = [units[i] for i in sorted(range(len(units)), key=lambda i: -rows[i]["mean"])]
        top = min(max(1, int(teacher.FRACTION * len(units))), len(units) - teacher.MIN_SIZE)
        ks = sorted({min(top, 2 ** e) for e in range(0, 20) if 2 ** e <= 2 * top})
        got = measure(S, [ranked[:k] for k in ks])
        i = max(range(len(ks)), key=lambda j: (got[j]["mean"], ks[j]))
        S = S.without(ranked[:ks[i]])
        path.append({"size": S.count(), "units": S.units(), **got[i]})
        if log:
            log(f"  joint path: {S.count()} units, F1 {got[i]['f1']:.3f}, N {got[i]['n']:.3f}, {pr.measured} runs, "
                f"{time.perf_counter() - t0:.0f} s")
    return {"kl_empty_off": empty, "kl_vpd_off": vpd, "budget_kl": budget, "f1_from_targets": reached, "levels": levels,
            "f1_circuit": {"size": f1_set.count(), "positions": len({u[2] for u in f1_set.units()}), "kl": f1_row["kl_c"],
                           "f1": f1_row["f1"], "n": f1_row["n"]}, "top_k": k, "start_size": start.count(),
            "path": path, "measured_sets": pr.measured, "seconds": time.perf_counter() - t0}


def teach_all(split: str, n: int, out: Path, offset: int = 0, stride: int = 1, vpd: Path | None = None,
              tasks: list[str] | None = None, log=print) -> None:
    """teach_native on the split's first n texts (offset, offset + stride, ...), or on the task ids `tasks` in their
    order, -> one row per text appended to OUT. vpd: VPD's answers saved by teacher.py --save-vpd (where the
    causal-importance network is not)."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    saved = torch.load(vpd) if vpd else {}
    done = {json.loads(x)["task"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    texts = [native.TEXTS / "vpd4l" / f"{t}.json" for t in tasks] if tasks else native.tasks(split)[:n][offset::stride]
    for p in texts:
        if p.stem in done:
            continue
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
        if p.stem in saved:
            tx.vpd = {m: torch.zeros(tx.T, nat.C[m], dtype=torch.bool, device=nat.dev) for m in nat.names}
            for m, (t, c) in saved[p.stem]["answer"].items():
                tx.vpd[m][t.long().to(nat.dev), c.long().to(nat.dev)] = True
        row = {"task": p.stem, "T": tx.T, **teach_native(ge, NativeUnits(nat, ids, targets), tx, log)}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        f = row["f1_circuit"]
        log(f"{p.stem}: native F1 circuit {f['size']} units at {f['positions']} positions (F1 {f['f1']:.3f}, N {f['n']:.3f}), "
            f"path to {row['path'][-1]['size']}, {row['measured_sets']} runs, {row['seconds']:.0f} s")


# ---- native units under phi (the lead's 10-12: neurons and heads as nodes, the same phi, draws and size accounting)


SHARED = ("point", "nodes", "track_pp", "interchange_pp", "patching_near", "patching_far", "pattern", "N")  # phi's
# families with a native counterpart here (connection cuts and the omitted-lever flips have none yet)


def native_bits(S: Mask, targets: list[int], positions: int, per_position: int) -> float:
    """A native circuit's description bits, size_bits' form for a complete node set: each unit one choice among
    positions x per_position, and every connection the model has among the units paid like a stated one (log2 n each
    end, a connection into the prediction its writer's): a unit's reads from the units writing its position's residual
    stream before it (a head reads at layer l, writes at l + 0.5; a neuron reads at l + 0.5, writes at l + 1, mech.stage),
    a head's from the same head's units at earlier positions (its keys and values), the prediction from every unit at
    the targets."""
    import bisect

    units = S.units()
    n = len(units)
    if not n:
        return 0.0
    writes, heads = {}, {}
    for kind, l, t, j in units:
        writes.setdefault(t, []).append(l + (0.5 if kind == "head" else 1.0))
        if kind == "head":
            heads.setdefault((l, j), []).append(t)
    for v in (*writes.values(), *heads.values()):
        v.sort()
    inner = sum(bisect.bisect_right(writes[t], l + (0.0 if kind == "head" else 0.5)) for kind, l, t, j in units)
    inner += sum(bisect.bisect_left(heads[(l, j)], t) for kind, l, t, j in units if kind == "head")
    out = sum(1 for _, _, t, _ in units if t in targets)
    return n * math.log2(positions * per_position) + math.log2(n) * (2 * inner + out)


class NativePhi:
    """Native-unit circuits (Mask) under phi's families on one text. Every run is goal_eval.graph_ref on the whole model
    (every subcomponent on, the remainders live: the real model) with the model's units replaced (graph_ref(units=)):
    in C's semantics (point, nodes, pattern, N) the circuit's units run and every other unit's output is 0 (its query,
    key and value its own); in the path-patched families (track_pp, interchange_pp) and activation patching every
    other unit, a head with its query, key and value, is held at its value on x or on the donor, the circuit's units
    computing from the stream. Each draw's reference is VPD's whole answer under the same intervention, with the same
    draws as VPD's families (the text's changed texts, donors and controls; for removals and holds of native units,
    VPD's answer with those units removed or held); point uses VPD's common live empty/whole-answer bounds (equal KL gets equal point credit across unit kinds);
    other families retain their native own-run bounds (output-zero/mean empty or donor). Rows carry the families' VPD-form records, rl_terms
    (each without the floor at 0, as phi reads them) and eps. means (unit_means): the C-semantics families and the
    removals mean-ablate a unit instead of zeroing it (the bake-off's native semantics, the lead's 10-14; a head's output
    alone, its query, key and value its own)."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, split: str = "eval", means: dict | None = None):
        nat = ge.nat
        self.ge, self.tx, self.nat, self.split, self.means = ge, tx, nat, split, means
        tg = nat.target
        self.L, self.H, self.hd = tg.n_layer, tg.n_head, tg.hd
        self.Dm = tg.site(native.site_name(0, "c_fc")).W.shape[0]
        T = tx.T
        self.model = native.Graph(masks={n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
        self.vpd = native.Graph(masks=tx.vpd)
        self._caps = {}

    def capture(self, ids: list[int]) -> dict:
        """The real model's units on ids (graph_ref's capture), kept by ids."""
        key = tuple(ids)
        if key not in self._caps:
            c = {}
            self.ge.graph_ref(list(ids), [len(ids) - 1], native.Graph(masks={n: torch.ones(len(ids), self.nat.C[n], dtype=torch.bool,
                                                                                       device=self.nat.dev) for n in self.nat.names}), capture=c)
            self._caps[key] = c
        return self._caps[key]

    def _ablate(self, T: int) -> dict | None:
        """The values an ablated unit takes on a T-token text: its mean (means, unit_means') at every position, a head's
        output alone (its query, key and value its own), or None (output 0) without means."""
        if self.means is None:
            return None
        m = self.means
        return {"out": m["heads"].to(self.nat.dev)[None, :, :, None, :].expand(1, -1, -1, T, -1),
                "neuron": m["neurons"].to(self.nat.dev)[None, :, None, :].expand(1, -1, T, -1)}

    def _units(self, heads: list, neurons: list, values: dict | None) -> dict:
        """graph_ref's units from per-row masks ([layers, T, H], [layers, T, d_mlp]) and values (a capture, rows 1 or B)."""
        hv = None if values is None else {k: values[k] for k in ("q", "k", "v", "out") if k in values}
        return {"head": (torch.stack(heads), hv), "neuron": (torch.stack(neurons), None if values is None else values["neuron"])}

    def _mix(self, a: dict, b: dict, hm: torch.Tensor, nm: torch.Tensor) -> dict:
        """A capture with b's values where the masks ([layers, T, H], [layers, T, d_mlp]) are set, a's elsewhere (and
        wherever b has no value: a mean ablation's query, key and value)."""
        h = hm.permute(0, 2, 1)[None, :, :, :, None]  # [1, layers, H, T, 1]
        out = {k: torch.where(h, b[k], a[k]) if k in b else a[k] for k in ("q", "k", "v", "out")}
        out["neuron"] = torch.where(nm[None], b["neuron"], a["neuron"])
        return out

    def _kl(self, ref: torch.Tensor, lq: torch.Tensor) -> list[float]:
        return carry.kl(ref.expand_as(lq) if ref.shape[0] == 1 else ref, lq).tolist()

    def _zeros(self, S: Mask) -> tuple:
        return torch.zeros_like(S.heads), torch.zeros_like(S.neurons)

    def point(self, sets: list[Mask]) -> list[dict]:
        tx = self.tx
        # The scorecard compares unit kinds at the same absolute output accuracy.
        # Own mean/zero-empty baselines remain diagnostics for other families.
        e0, v0 = self.ge._bounds(tx, "live")
        lq = self.ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units([~S.heads for S in sets], [~S.neurons for S in sets], self._ablate(tx.T)))
        out = []
        for k in self._kl(tx.ref, lq):
            out.append({"s_point": goal_eval._fid(e0, k, v0), "s_point_c": goal_eval._fid(e0, k, v0), "kl_point": k,
                        "draws_point": [[k, v0]], "_raw_point": goal_eval._fid_raw(e0, k, v0)})
        return out

    def empty(self) -> float:
        """The native empty circuit's KL (the embedding alone: every unit's output 0), kept on the text."""
        if "_native_empty" not in self.tx.__dict__:
            z = Mask(torch.ones(self.L, self.tx.T, self.H, dtype=torch.bool, device=self.nat.dev),
                     torch.ones(self.L, self.tx.T, self.Dm, dtype=torch.bool, device=self.nat.dev))
            lq = self.ge.graph_ref(self.tx.ids, self.tx.targets, self.model, units=self._units([z.heads], [z.neurons], self._ablate(self.tx.T)))
            self.tx._native_empty = self._kl(self.tx.ref, lq)[0]
        return self.tx._native_empty

    def nodes(self, S: Mask, draws: int = goal_eval.DRAWS) -> dict:
        """s_nodes: per draw d (each unit with probability (j + 0.5) / draws, at least one; seeded by the circuit), KL(model
        less d || circuit less d) against W = KL(model less d || model), VPD's answer less d the reference, pooled by W."""
        tx, ge = self.tx, self.ge
        units = sorted(S.units())
        if not units:
            return {"s_nodes": None}
        import hashlib

        rng = np.random.default_rng([tx.seed, int(hashlib.sha256(repr(("native nodes", units)).encode()).hexdigest()[:15], 16)])
        drops = []
        for j in range(draws):
            d = [u for u in units if rng.random() < (j + 0.5) / draws]
            drops.append(d or [units[int(rng.integers(len(units)))]])
        dm = [Mask(*self._zeros(S)) for _ in drops]
        for m, d in zip(dm, drops):
            for kind, l, t, j in d:
                (m.heads if kind == "head" else m.neurons)[l, t, j] = True
        ab = self._ablate(tx.T)
        lm = ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units([m.heads for m in dm], [m.neurons for m in dm], ab))
        lc = ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units([~S.heads | m.heads for m in dm], [~S.neurons | m.neurons for m in dm], ab))
        lv = ge.graph_ref(tx.ids, tx.targets, self.vpd, units=self._units([m.heads for m in dm], [m.neurons for m in dm], ab))
        kc, kv = carry.kl(lm, lc).tolist(), carry.kl(lm, lv).tolist()
        w = carry.kl(lm, tx.ref.expand_as(lm)).tolist()
        claim = carry.kl(tx.ref.expand_as(lc), lc).tolist()
        rec = [[j, a, b, c, d] for j, (a, b, c, d) in enumerate(zip(kc, kv, w, claim))]
        e0 = self.empty()
        return {"s_nodes": goal_eval._pooled("weighted", rec, e0), "_raw_nodes": goal_eval._pooled_raw(rec), "draws_nodes": rec,
                "kl_nodes_c": float(np.mean(kc)), "kl_nodes_vpd": float(np.mean(kv)),
                "overstate_nodes": float(np.mean(claim)) / max(float(np.mean(w)), 1e-12)}

    def _pp_prompts(self, draws: int) -> list:
        """track_pp's changed texts (p, z), VPD's own (the text's draws), with each one's model run, W and VPD's KL."""
        tx, ge = self.tx, self.ge
        moved = np.asarray(tx.moved, dtype=float)
        positions = [p for p, _ in tx.f2]
        got = ge._interchange_draws(tx, positions, self.split, draws, first={p: -moved[p - 1] for p in positions}) \
            if positions else []
        kept = tx.__dict__.setdefault("_pp_prompts", {})
        if any((p, z) not in kept for p, z in got):
            ge.track_pp(tx, [native.Graph()], self.split, draws)  # VPD's side of every prompt, kept on the text
        return [(p, z, *kept[(p, z)]) for p, z in got if (p, z) in kept]

    def track_pp(self, sets: list[Mask], draws: int = goal_eval.CHANGED_DRAWS) -> list[dict]:
        """s_track_pp: on each changed text x', the model on x' with every unit outside the circuit held at its value on
        x (the change entering through the embedding into the circuit's units and the prediction) against the model on
        x', VPD's path-patched answer the reference, bounded by no change (W), pooled by W."""
        tx, ge = self.tx, self.ge
        cx = self.capture(tx.ids)
        recs = [[] for _ in sets]
        for p, z, ref, w, kv in self._pp_prompts(draws):
            x2 = list(tx.ids)
            x2[p] = z
            lq = ge.graph_ref(x2, tx.targets, self.model, units=self._units([~S.heads for S in sets], [~S.neurons for S in sets], cx))
            for b, kc in enumerate(self._kl(ref, lq)):
                recs[b].append([[p, z], kc, kv, w])
            native.free(self.nat.dev)
        return [{"s_track_pp": goal_eval._pooled("weighted", r, None), "_raw_track_pp": goal_eval._pooled_raw(r), "draws_track_pp": r}
                for r in recs]

    def interchange_pp(self, S: Mask, draws: int = goal_eval.INTERCHANGE_DRAWS) -> dict:
        """s_interchange_pp: on the interchange draws of the circuit's positions (VPD's _interchange_draws), the circuit's
        units at p held at their values in the model on x', the rest outside the circuit at x, the circuit's other units
        computing; against the model with those units held (W = KL(model held || model)) and VPD's answer with them
        held, pooled by W."""
        tx, ge = self.tx, self.ge
        positions = sorted({u[2] for u in S.units() if u[2] > 0})
        got = ge._interchange_draws(tx, positions, self.split, draws) if positions else []
        cx = self.capture(tx.ids)
        rec = []
        for p, z in got:
            x2 = list(tx.ids)
            x2[p] = z
            c2 = self.capture(x2)
            hp, np_ = torch.zeros_like(S.heads), torch.zeros_like(S.neurons)
            hp[:, p], np_[:, p] = S.heads[:, p], S.neurons[:, p]
            held = self._mix(cx, c2, hp, np_)
            lm = ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units([hp], [np_], c2))
            lc = ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units([~S.heads | hp], [~S.neurons | np_], held))
            lv = ge.graph_ref(tx.ids, tx.targets, self.vpd, units=self._units([hp], [np_], c2))
            w = float(carry.kl(lm, tx.ref)[0])
            rec.append([[p, z], float(carry.kl(lm, lc)[0]), float(carry.kl(lm, lv)[0]), w])
            native.free(self.nat.dev)
        return {"s_interchange_pp": goal_eval._pooled("weighted", rec, None), "_raw_interchange_pp": goal_eval._pooled_raw(rec),
                "draws_interchange_pp": rec}

    def _donor_capture(self, ids: list[int]) -> tuple[dict, int]:
        """A donor's units aligned with the text at their ends (goal_eval.donor_activations' alignment): its last T
        positions, or, a shorter donor, at the text's last positions with the first pad left unpatched."""
        T = self.tx.T
        c = self.capture(ids)
        n = len(ids)
        pad = max(0, T - n)
        out = {}
        for k in ("q", "k", "v", "out"):
            x = c[k][:, :, :, max(0, n - T):]
            out[k] = x if not pad else torch.cat([x.new_zeros(*x.shape[:3], pad, x.shape[4]), x], 3)
        x = c["neuron"][:, :, max(0, n - T):]
        out["neuron"] = x if not pad else torch.cat([x.new_zeros(x.shape[0], x.shape[1], pad, x.shape[3]), x], 2)
        return out, pad

    def patching(self, sets: list[Mask], near: int = goal_eval.DONORS_NEAR, far: int = goal_eval.DONORS_FAR) -> list[dict]:
        """Activation patching (s_patching_near, _far): every unit outside the circuit at its value on a donor text
        (VPD's donors: near the changed texts, far the split's far pool, end-aligned), the circuit's computing, from the
        model on the text; bound the native keep-nothing run (every unit at the donor's), VPD's answer under the same
        donor the reference (VPD's donors run)."""
        tx, ge = self.tx, self.ge
        far_texts = ge.far_donors(tx, ge.far_pool(self.split), far) if far else None
        vrow = ge.donors(tx, [self.vpd], self.split, near, far_texts)[0]  # VPD's side per donor
        texts = {}
        if near:
            moved = np.asarray(tx.moved, dtype=float)
            positions = [p for p, _ in tx.f2]
            for p, z in (ge._interchange_draws(tx, positions, self.split, near, first={p: -moved[p - 1] for p in positions}) if positions else []):
                x = list(tx.ids)
                x[p] = z
                texts.setdefault("near", []).append(([p, z], x))
        if far_texts:
            texts["far"] = [(name, list(x)) for name, x in far_texts if list(x) != list(tx.ids)]
        out = [{} for _ in sets]
        full = Mask(torch.ones(self.L, tx.T, self.H, dtype=torch.bool, device=self.nat.dev),
                    torch.ones(self.L, tx.T, self.Dm, dtype=torch.bool, device=self.nat.dev))
        for pool, xs in texts.items():
            kv_of = {json.dumps(d[0]): d[2] for d in vrow.get(f"draws_patching_{pool}") or []}
            recs = [[] for _ in sets]
            for name, x in xs:
                cd, pad = self._donor_capture(x)
                keep = torch.ones(tx.T, dtype=torch.bool, device=self.nat.dev)
                keep[:pad] = False
                rows = sets + [Mask(torch.zeros_like(full.heads), torch.zeros_like(full.neurons))]  # the circuits, nothing kept
                hm = [(~S.heads) & keep[None, :, None] for S in rows]
                nm = [(~S.neurons) & keep[None, :, None] for S in rows]
                lq = ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units(hm, nm, cd))
                kl = self._kl(tx.ref, lq)
                kv = kv_of.get(json.dumps(name))
                for b in range(len(sets)):
                    if kv is not None:
                        recs[b].append([name, kl[b], kv, kl[-1], goal_eval._fid(kl[-1], kl[b], kv)])
                native.free(self.nat.dev)
            for b, r in enumerate(recs):
                if r:
                    kc, kv, k0 = (float(np.mean([d[i] for d in r])) for i in (1, 2, 3))
                    out[b].update({f"s_patching_{pool}": goal_eval._fid(k0, kc, kv), f"_raw_patching_{pool}": goal_eval._fid_raw(k0, kc, kv),
                                   f"kl_patching_{pool}": (kc, kv, k0), f"draws_patching_{pool}": r, f"donors_{pool}": len(r)})
        return out

    def pattern(self, S: Mask) -> dict:
        """s_pattern: the circuit's C-semantics run with the reads it uses (a layer and position holding a circuit head)
        given the real model's attention row, all at once and each alone, against VPD's answer the same (pattern_family),
        on the native empty circuit's log scale."""
        tx, ge = self.tx, self.ge
        reads = sorted({(l, t) for l, t in S.heads.any(2).nonzero().tolist()})
        if not reads:
            return {"s_pattern": None, "reads_pattern": 0}
        b = ge._pp_base(tx)
        T = tx.T
        causal = torch.ones(T, T, dtype=torch.bool, device=self.nat.dev).tril()
        real = {(i, t): b["S"][i][:, t].masked_fill(~causal[t], float("-inf")).softmax(-1)[:, :t + 1] for i, t in reads}
        draws = [real] + ([{r: real[r]} for r in reads] if len(reads) > 1 else [])
        un = self._units([~S.heads], [~S.neurons], self._ablate(tx.T))
        kc = [float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, self.model, units=un, rows=d))[0]) for d in draws]
        kept = tx.__dict__.setdefault("_pattern_vpd", {})
        kv = []
        for d in draws:
            key = (id(tx.vpd), tuple(sorted(d)))
            if key not in kept:
                kept[key] = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, self.vpd, rows=d))[0])
            kv.append(kept[key])
        e0 = self.empty()
        c_, v_ = float(np.mean(kc)), float(np.mean(kv))
        return {"s_pattern": goal_eval._fid(e0, c_, v_), "_raw_pattern": goal_eval._fid_raw(e0, c_, v_), "kl_pattern_c": c_,
                "kl_pattern_vpd": v_, "reads_pattern": len(reads), "draws_pattern": [list(x) for x in zip(kc, kv)]}

    def necessity(self, sets: list[Mask]) -> list[dict]:
        """N (necessity_spec's form): the circuit's units replaced by their fixed background (means when supplied,
        otherwise output 0) in the real model, its damage at the
        targets less its mean damage on necessity_spec's control texts (end-aligned), over VPD's whole answer's
        (necessity_spec's numbers, the same controls)."""
        tx, ge = self.tx, self.ge
        vrow = ge.necessity_spec(tx, [native.Graph()], self.split)[0]
        lq = ge.graph_ref(tx.ids, tx.targets, self.model, units=self._units([S.heads for S in sets], [S.neurons for S in sets], self._ablate(tx.T)))
        dt = self._kl(tx.ref, lq)
        T, last = tx.T, max(tx.targets)
        pool = [x for x in ge.far_pool(self.split) if len(x[1]) >= min(T, goal_eval.NEC_MIN_LEN) and list(x[1]) != list(tx.ids)]
        ys = [pool[i][1] for i in np.random.default_rng([int(tx.seed), 13]).permutation(len(pool))[:goal_eval.NEC_CONTROLS]]
        dc = []
        for y in ys:
            y = list(y[:last + 1])
            d = last + 1 - len(y)
            tg = [t - d for t in tx.targets if t >= d]
            ones = {n: torch.ones(len(y), self.nat.C[n], dtype=torch.bool, device=self.nat.dev) for n in self.nat.names}
            g = native.Graph(masks=ones)
            ref = ge.graph_ref(y, tg, g)
            lq = ge.graph_ref(y, tg, g, units=self._units([S.heads[:, d:last + 1] for S in sets], [S.neurons[:, d:last + 1] for S in sets],
                                                          self._ablate(len(y))))
            dc.append(self._kl(ref, lq))
            native.free(self.nat.dev)
        vd = vrow["draws_n_spec"]
        out = []
        for b in range(len(sets)):
            rec = [[j, c[b], v[2], dt[b], v[4]] for j, (c, v) in enumerate(zip(dc, vd))]
            out.append({"n_spec": goal_eval._spec(rec), "_raw_n_spec": goal_eval._spec(rec, raw=True), "draws_n_spec": rec,
                        "d_target": dt[b], "d_control": float(np.mean([r[1] for r in rec])) if rec else None,
                        "d_target_vpd": vrow["d_target_vpd"], "d_control_vpd": vrow["d_control_vpd"]})
        return out

    def rows(self, sets: list[Mask], *, necessity_rows: list[dict] | None = None) -> list[dict]:
        """Every shared family; necessity_rows reuses this call's already measured N without another execution."""
        if necessity_rows is not None and len(necessity_rows) != len(sets):
            raise ValueError("one precomputed necessity row per mask required")
        pt, tp, pa = self.point(sets), self.track_pp(sets), self.patching(sets)
        ns = self.necessity(sets) if necessity_rows is None else necessity_rows
        out = []
        for i, S in enumerate(sets):
            row = {"units": S.count(), "positions": len({u[2] for u in S.units()}), **pt[i], **tp[i], **pa[i], **ns[i],
                   **self.nodes(S), **self.interchange_pp(S), **self.pattern(S), "kl_empty_live": self.empty()}
            row["rl_terms"] = {"s_point_c": row.pop("_raw_point"), "s_nodes": row.pop("_raw_nodes", None),
                               "s_track_pp": row.pop("_raw_track_pp", None), "s_interchange_pp": row.pop("_raw_interchange_pp", None),
                               "s_patching_near": row.pop("_raw_patching_near", None), "s_patching_far": row.pop("_raw_patching_far", None),
                               "s_pattern": row.pop("_raw_pattern", None), "n_spec": row.pop("_raw_n_spec", None)}
            row.update({f"eps_{f}": v for f, v in goal_eval.family_eps(row).items()})
            row["phi_shared"] = phi_shared(row)
            out.append(row)
        return out


def phi_shared(row: dict) -> float | None:
    """phi over the families both unit kinds have (SHARED), the row's eps."""
    eps = goal_eval._eps(row)
    return goal_eval._phi([(goal_eval.family_score(row, f), eps[f]) for f in SHARED if goal_eval.family_score(row, f) is not None])


def prune_to_bits(nu: NativeUnits, positions, goals: list[float], P: int, log=None) -> dict:
    """prune's measured pruning from every unit at the positions (each unit's measured removal from the set, least
    first, a STEP-th at a time, rescreened as prune does), stopping at the largest set within each goal's bits
    (native_bits): {goal: Mask}; a step never passes a goal (the fewest units that bring the set under it go)."""
    per = nu.L * (nu.H + nu.Dm)
    S = nu.at(positions)
    kl = nu.kl(S)[0]
    effect = screen(nu, S, kl, log)
    rescreened = {n for n in RESCREEN if S.count() <= n}
    out, todo = {}, sorted(goals, reverse=True)

    def bits_of(m):
        return native_bits(m, nu.targets, P, per)

    while todo:
        b = bits_of(S)
        while todo and b <= todo[0]:
            out[todo.pop(0)] = S
        if not todo or S.count() == 0:
            break
        order = sorted(S.units(), key=lambda u: effect.get(u, -math.inf))
        k = max(1, S.count() // STEP)
        if bits_of(S.without(order[:k])) <= todo[0]:  # the fewest that bring it under the goal
            lo, hi = 0, k
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if bits_of(S.without(order[:mid])) <= todo[0] else (mid, hi)
            k = hi
        S = S.without(order[:k])
        kl = nu.kl(S)[0]
        if log:
            log(f"  {S.count():7d} units  {bits_of(S):.0f} bits  KL {kl:.4g}")
        due = [t for t in RESCREEN if S.count() <= t and t not in rescreened]
        if (due or S.count() <= FINE) and S.count() > 0:
            rescreened |= set(due)
            effect = screen(nu, S, kl, log)
    return out


MEANS_TEXTS = range(6000, 6100)  # unit_means' texts: training texts, never a dev or held-out one
MEANS_FILE = Path.home() / "mpd-data/graph_oracle/native_means.pt"


def _means_binding(ge, names, inputs):
    """Effective graph_ref state, not a checkpoint filename or the CI network.

    Intentionally frozen means across edited worlds must instead be passed directly
    to the ablation caller as a dictionary; a disk cache is same-state reuse only.
    """
    tg = ge.nat.target
    h = hashlib.sha256()

    def add(label, value):
        h.update(label.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            x = value.detach().contiguous().cpu()
            h.update(str((str(x.dtype), tuple(x.shape))).encode() + b"\0")
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        h.update(b"\0")

    add("schema", 1)
    add("inputs", list(zip(names, inputs)))
    add("target_class", type(tg).__module__ + "." + type(tg).__qualname__)
    add("graph_ref_source", inspect.getsource(type(ge).graph_ref))
    add("target_source", inspect.getsource(type(tg)))
    # Class text alone misses module-level norm/activation/routing helpers.
    source_dir = Path(native.__file__).resolve().parent
    source_files = [source_dir / name for name in
                    ("native.py", "goal_eval.py", "engine.py", "carry.py", "mech.py")]
    source_files += [Path(inspect.getfile(type(tg))), Path(__file__)]
    for i, file in enumerate(source_files):
        add("execution_source_file:" + str(i), hashlib.sha256(file.read_bytes()).hexdigest())
    add("config", tg.cfg)
    add("sites", list(ge.nat.names))
    add("component_counts", ge.nat.C)
    for attr in ("n_layer", "n_head", "hd", "eps", "norm_kind", "tl_eps", "act_fn"):
        add(attr, getattr(tg, attr, None))
    for attr in ("behavior", "behavior_tokens", "behavior_at"):
        value = getattr(tg, attr, None)
        add(attr, sorted(value) if isinstance(value, (set, frozenset)) else value)
    # state_dict includes effective embeddings, norms, RoPE, dense matrices/biases;
    # graph_ref also uses nonregistered decomposition factors and live remainders.
    for key, value in sorted(tg.state_dict().items()):
        add("state:" + key, value)
    for name in ge.nat.names:
        site = tg.site(name)
        for attr in ("W", "b", "U", "V", "delta_T"):
            add(name + ":" + attr, getattr(site, attr, None))
    # Arbitrary runtime hooks cannot be certified by this static state binding.
    if any(m._forward_hooks or m._forward_pre_hooks or getattr(m, "in_fn", None) is not None
           for m in tg.modules()):
        raise ValueError("unit_means cache binding does not support runtime hooks")
    return {"schema_version": 1, "sha256": h.hexdigest()}


def _valid_means(got, binding, names, positions, shapes):
    return (isinstance(got, dict) and got.get("cache_binding") == binding
            and got.get("texts") == names and got.get("positions") == positions
            and all(isinstance(got.get(k), torch.Tensor)
                    and tuple(got[k].shape) == shape and got[k].is_floating_point()
                    and bool(torch.isfinite(got[k]).all()) for k, shape in shapes.items()))


@torch.no_grad()
def unit_means(nat: native.Native, texts=MEANS_TEXTS, path: Path | None = MEANS_FILE, inputs: list[list[int]] | None = None) -> dict:
    """Each native unit's mean over every position of the texts (the bake-off's mean ablation, the lead's 10-14): a
    head's output before the output matrix {"heads": [layers, H, hd]} and a post-GELU neuron's activation {"neurons":
    [layers, d_mlp]}, from the real model's runs (goal_eval.graph_ref's capture); an ablated head keeps its own query,
    key and value. Kept in path, recomputed when its texts differ. Disk reuse also requires identical
    effective graph_ref state and actual tokens, valid shapes and finite means; legacy caches are recomputed.
    Intentionally frozen means across edited worlds must be passed explicitly as a dictionary. inputs: another model's
    own token sequences in place of vpd4l's texts (texts then names them; MIB's InterpBench model: its train prompts)."""
    names = [f"text{n}" for n in texts] if inputs is None else [str(n) for n in texts]
    ge = goal_eval.GoalEval(nat)
    inputs = [list(native.text(native.TEXTS / "vpd4l" / f"{name}.json")[0]) for name in names] if inputs is None else [list(x) for x in inputs]
    if not inputs or any(not ids for ids in inputs):
        raise ValueError("unit_means requires nonempty training input sequences")
    tg = nat.target
    positions = sum(map(len, inputs))
    shapes = {"heads": (tg.n_layer, tg.n_head, tg.hd)}
    shapes["neurons"] = (tg.n_layer, tg.site(native.site_name(0, "c_fc")).W.shape[0])
    binding = _means_binding(ge, names, inputs)
    if path is not None and path.exists():
        try:
            got = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
            got = None
        if _valid_means(got, binding, names, positions, shapes):
            return got
    heads = neurons = None
    count = 0
    for ids in inputs:
        c = {}
        ones = native.Graph(masks={n: torch.ones(len(ids), nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
        ge.graph_ref(list(ids), [len(ids) - 1], ones, capture=c)
        h, m = c["out"][0].sum(2).cpu().double(), c["neuron"][0].sum(1).cpu().double()  # [layers, H, hd], [layers, d_mlp] (MPS has no float64)
        heads, neurons = (h, m) if heads is None else (heads + h, neurons + m)
        count += len(ids)
        native.free(nat.dev)
    out = {"heads": (heads / count).float().cpu(), "neurons": (neurons / count).float().cpu(), "texts": names, "positions": count}
    out["cache_binding"] = binding
    if not _valid_means(out, binding, names, positions, shapes):
        raise RuntimeError("unit_means produced invalid mean tensors")
    if _means_binding(ge, names, inputs) != binding:
        raise RuntimeError("unit_means effective state changed during computation")
    if path is not None:
        torch.save(out, path)
    return out


def from_units(nu: NativeUnits, units: list) -> Mask:
    m = Mask(torch.zeros(nu.L, nu.T, nu.H, dtype=torch.bool, device=nu.nat.dev),
             torch.zeros(nu.L, nu.T, nu.Dm, dtype=torch.bool, device=nu.nat.dev))
    for kind, l, t, j in units:
        (m.heads if kind == "head" else m.neurons)[l, t, j] = True
    return m


def phi_report(split: str, n: int, vpd: Path | None, influential: Path, out: Path, teach: list[Path] = (), ratios=(0.5, 1.0, 2.0),
               offset: int = 0, stride: int = 1, log=print, texts: Path | None = None) -> None:
    """Native units against VPD's subcomponents under phi (the lead's 10-12) on the split's first n texts (or, given
    texts, a directory of <task>.json texts, on every task with a circuit): the text's circuit in VPD's subcomponents
    from influential (a directory of <task>.json circuits: the influential sets, teacher circuits or oracle answers;
    scored by GoalEval.fidelity on the eval draws, flips and the step adversary left out) and native circuits at its
    bits times each ratio (native_bits against size_bits): the measured pruning at the circuit's positions
    (prune_to_bits) and, given native teacher rows (teach_all's), the teacher path's step nearest in bits. vpd: VPD's
    answers saved by teacher.py --save-vpd (None: the causal-importance network's, where it is present). One row per
    circuit appended to OUT: kind, units or nodes, bits, every shared family with its records, phi_shared (phi over
    SHARED) and, for VPD's, its full phi."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    saved = torch.load(vpd) if vpd else {}
    paths = {}
    for f in teach:
        for x in Path(f).read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                paths[r["task"]] = r["path"]
    done = {json.loads(x)["task"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    S_total = sum(nat.C.values())
    todo = ([Path(texts) / f.name for f in sorted(Path(influential).iterdir()) if f.suffix == ".json" and f.stem.count(".") == 0]
            if texts else native.tasks(split)[:n])[offset::stride]
    for p in todo:
        if p.stem in done or not (influential / f"{p.stem}.json").exists() or not p.exists():
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
        if p.stem in saved:
            tx.vpd = {m: torch.zeros(tx.T, nat.C[m], dtype=torch.bool, device=nat.dev) for m in nat.names}
            for m, (t, c) in saved[p.stem]["answer"].items():
                tx.vpd[m][t.long().to(nat.dev), c.long().to(nat.dev)] = True
        g, _ = goal_eval.circuit(json.loads((influential / f"{p.stem}.json").read_text()), targets, nat.C, tx.T)
        P = nat.positions(targets)
        B = goal_eval.size_bits(g, P, S_total, targets, active=tx.vpd)
        vrow = ge.fidelity(tx, [g], None, split="eval")[0]
        rows = [{"kind": "vpd circuit", "nodes": g.count(), "bits": B, **vrow, "phi": goal_eval.phi(vrow)[0],
                 "phi_shared": phi_shared(vrow)}]
        nu = NativeUnits(nat, ids, targets)
        per = nu.L * (nu.H + nu.Dm)
        positions = sorted({t for _, t, _ in g.node_set()})
        got = prune_to_bits(nu, positions, [r * B for r in ratios], P)
        sets, kinds = [], []
        for r in ratios:
            if r * B in got:
                sets.append(got[r * B])
                kinds.append(f"native positions x{r:g}")
        if p.stem in paths:
            steps = [(from_units(nu, [tuple(u) for u in s["units"]])) for s in paths[p.stem]]
            sb = [native_bits(m, targets, P, per) for m in steps]
            for r in ratios:
                i = min(range(len(steps)), key=lambda j: abs(math.log(max(sb[j], 1.0) / (r * B))))
                sets.append(steps[i])
                kinds.append(f"native teacher x{r:g}")
        if sets:
            for kind, m, row in zip(kinds, sets, NativePhi(ge, tx, "eval").rows(sets)):
                rows.append({"kind": kind, "bits": native_bits(m, targets, P, per), "units_list": m.units(), **row})
        with open(out, "a") as fh:
            for row in rows:
                fh.write(json.dumps({"task": p.stem, **row}) + "\n")
        if log:
            log(f"{p.stem}: VPD {g.count()} nodes {B:.0f} bits phi_shared {rows[0]['phi_shared']}; "
                + "; ".join(f"{r['kind']} {r.get('units')} units {r['bits']:.0f} bits phi_shared {r['phi_shared']}" for r in rows[1:])
                + f"  ({time.perf_counter() - t0:.0f} s)")


# ---- the comparison table


def _nearest(steps: list[dict], size: int) -> dict | None:
    """The step whose size is nearest `size` (the smaller on a tie), None when every step is over twice or under half
    of it."""
    ok = [s for s in steps if size / 2 <= s["size"] <= 2 * size]
    return min(ok, key=lambda s: (abs(s["size"] - size), s["size"])) if ok else None


def teacher_steps(path: dict) -> list[dict]:
    """A teacher path file (teacher.py's <task>.path.json) as steps with their node sets (matrix, position, index)."""
    by = dict(zip(path["start_index"], [(n, int(t), int(c)) for n, c, t in path["start_nodes"]]))
    on = set(by.values())
    out = []
    for s in path["steps"]:
        on -= {by[i] for i in s["removed"]}
        out.append({"size": len(on), "nodes": sorted(on)})
    return out


def compare(native_rows: Path, teacher_dir: Path, split: str, n: int, sizes=(12, 16, 32, 64, 256, 1024),
            ge: goal_eval.GoalEval | None = None, reuse: Path | None = None, log=print) -> list[dict]:
    """Per text and size: the equal-effort native circuit (native_units teach), the teacher's circuit (teacher.py's
    path) and VPD's first k (its order), at the step nearest each size; F1 and N_VPD (goal_eval's sufficiency:
    remainders off, the convention native circuits share, and live, the teacher's) and N_real (removal from the real
    model: goal_eval's real_removal, native_units' N for native units); locality from the teacher's summary (its
    "local" label, components.locality, else whether its F1 circuit came from the targets alone). reuse: an
    earlier compare's rows, whose teacher and VPD order rows stand for their texts (native rows need no runs)."""
    old: dict = {}
    if reuse is not None and reuse.exists():
        for x in reuse.read_text().splitlines():
            if x.strip() and json.loads(x)["kind"] != "native":
                old.setdefault(json.loads(x)["task"], []).append(json.loads(x))
    ge = ge or goal_eval.GoalEval()
    nat = ge.nat
    nrows = {json.loads(x)["task"]: json.loads(x) for x in native_rows.read_text().splitlines() if x.strip()}
    local = {}
    for f in sorted(teacher_dir.glob("summary*.jsonl")):
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                local[r["task"]] = r["local"] if r.get("local") is not None else r.get("f1_from_targets")
    out = []
    for p in native.tasks(split)[:n]:
        tp = teacher_dir / f"{p.stem}.path.json"
        if p.stem not in nrows and not tp.exists():
            continue
        if p.stem in old:
            out += [{**r, "local": local.get(p.stem)} for r in old[p.stem]]
        else:
            out += _vpd_side(ge, p, tp, local.get(p.stem), sizes)
        if p.stem in nrows:
            for k in sizes:
                t = _nearest(nrows[p.stem]["path"], k)
                if t is not None:
                    out.append({"task": p.stem, "local": local.get(p.stem), "kind": "native", "size": k, "nodes": t["size"],
                                "f1": t["f1"], "n_vpd": None, "n_real": t["n"]})
        if log:
            log(f"{p.stem}: {sum(1 for r in out if r['task'] == p.stem)} circuits scored")
    return out


def _vpd_side(ge: goal_eval.GoalEval, p: Path, tp: Path, local, sizes) -> list[dict]:
    """compare's teacher and VPD order rows on one text."""
    nat = ge.nat
    out = []
    ids, targets = native.text(p)
    tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
    order = nat._vpd_order(ids, targets)
    tsteps = teacher_steps(json.loads(tp.read_text())) if tp.exists() else []
    graphs, meta = [], []
    for k in sizes:
        if k <= len(order):
            graphs.append(native.Graph(order[:k], complete=True))
            meta.append(("VPD order", k))
        t = _nearest(tsteps, k)
        if t is not None:
            graphs.append(native.Graph(t["nodes"], complete=True))
            meta.append(("teacher", k))
    scored = ge.sufficiency(tx, graphs) if graphs else []
    real = ge.real_removal(tx, graphs) if graphs else []
    for (kind, k), g, s, r in zip(meta, graphs, scored, real):
        out.append({"task": p.stem, "local": local, "kind": kind, "size": k, "nodes": g.count(),
                    "f1": s["f1_off"], "n_vpd": s["n_off"], "f1_live": s["f1_live"], "n_vpd_live": s["n_live"],
                    "n_real": r["n_real_off"]})
    return out


def effort_table(native_rows: Path, teacher_dir: Path) -> str:
    """Per text, both searches' measured sets, seconds, F1 circuits (size, positions, F1) and whether each reached its
    F1 circuit from the targets alone."""
    nrows = {json.loads(x)["task"]: json.loads(x) for x in native_rows.read_text().splitlines() if x.strip()}
    trows = {}
    for f in sorted(teacher_dir.glob("summary*.jsonl")):
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                trows[r["task"]] = r
    lines = [["text", "local", "teacher sets", "native sets", "teacher s", "native s", "teacher F1 circuit", "native F1 circuit"]]
    for task in sorted(set(nrows) | set(trows)):
        t, n = trows.get(task, {}), nrows.get(task, {})

        def circ(r):
            f = r.get("f1_circuit")
            return f"{f['size']} at {f['positions']} ({f['f1']:.2f}{', targets' if r.get('f1_from_targets') else ''})" if f else "-"
        lines.append([task, str(t.get("local", "-")), str(t.get("measured_sets", "-")), str(n.get("measured_sets", "-")),
                      f"{t['seconds']:.0f}" if t else "-", f"{n['seconds']:.0f}" if n else "-", circ(t), circ(n)])
    w = [max(len(x[i]) for x in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(x[i].rjust(w[i]) for i in range(len(x))) for x in lines)


def compare_table(rows: list[dict]) -> str:
    """Means by locality, size and kind: F1 and N_VPD with remainders off (live), N_real, the circuits' mean node
    count."""
    lines = [["texts", "size", "kind", "n", "nodes", "F1 off", "(live)", "N_VPD off", "(live)", "N_real"]]
    for loc, label in ((True, "local"), (False, "non-local"), (None, "all")):
        for k in sorted({r["size"] for r in rows}):
            for kind in ("native", "teacher", "VPD order"):
                rs = [r for r in rows if r["size"] == k and r["kind"] == kind and (loc is None or r["local"] is loc)]
                if not rs:
                    continue

                def m(key, fmt="{:.3f}"):
                    v = [r[key] for r in rs if r.get(key) is not None]
                    return fmt.format(float(np.mean(v))) if v else "-"
                lines.append([label, str(k), kind, str(len(rs)), m("nodes", "{:.0f}"), m("f1"), m("f1_live"), m("n_vpd"),
                              m("n_vpd_live"), m("n_real")])
    w = [max(len(x[i]) for x in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(x[i].rjust(w[i]) for i in range(len(x))) for x in lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    c.add_argument("--offset", type=int, default=0)
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--out", type=Path)
    t = sub.add_parser("table")
    t.add_argument("rows", type=Path, help="curve's OUT.jsonl")
    e = sub.add_parser("teach", help="the teacher's search on native units (equal effort)")
    e.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--vpd", type=Path, help="VPD's answers saved by teacher.py --save-vpd (no causal-importance network here)")
    e.add_argument("--tasks", help="comma-separated task ids, run in this order, in place of the split's texts")
    q = sub.add_parser("compare", help="native (teach) vs the teacher vs VPD's order at matched sizes")
    q.add_argument("--native", type=Path, required=True, help="teach's OUT.jsonl")
    q.add_argument("--teacher", type=Path, required=True, help="teacher.py's --out directory")
    q.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    q.add_argument("--n", type=int, default=20)
    q.add_argument("--out", type=Path)
    q.add_argument("--reuse", type=Path, help="an earlier compare --out: its teacher and VPD order rows are kept")
    f = sub.add_parser("phi", help="native units against circuits in VPD's subcomponents under phi's shared families at equal bits")
    f.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    f.add_argument("--n", type=int, default=20)
    f.add_argument("--offset", type=int, default=0)
    f.add_argument("--stride", type=int, default=1)
    f.add_argument("--vpd", type=Path, help="VPD's answers saved by teacher.py --save-vpd (default: the causal-importance network's)")
    f.add_argument("--circuits", "--influential", dest="influential", type=Path, required=True,
                   help="circuits in VPD's subcomponents, <task>.json each (influential sets, teacher circuits, oracle answers)")
    f.add_argument("--texts", type=Path, help="a directory of <task>.json texts: every task with a circuit (in place of --split/--n)")
    f.add_argument("--teach", type=Path, nargs="*", default=[], help="native teacher rows (teach's OUT.jsonl)")
    f.add_argument("--ratios", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    f.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build", help="the native builder (build_native) on given texts, circuits in the bake-off's format")
    b.add_argument("--tasks", required=True, help="comma-separated task ids")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--vpd", type=Path, help="VPD's answers saved by teacher.py --save-vpd (the KL target reads VPD's bounds)")
    b.add_argument("--device", help="cuda, mps or cpu (default: native.device()'s choice)")
    b.add_argument("--no-mediated", action="store_true", help="skip the mediated share")
    b.add_argument("--rev", default="", help="the code's commit, written as the builder's name")
    args = ap.parse_args()
    if args.cmd == "build":
        build_all(args.tasks.split(","), args.out, args.vpd, args.device, not args.no_mediated, args.rev)
        return
    if args.cmd == "phi":
        phi_report(args.split, args.n, args.vpd, args.influential, args.out, args.teach, tuple(args.ratios), args.offset, args.stride,
                   texts=args.texts)
        return
    if args.cmd == "compare":
        rows = compare(args.native, args.teacher, args.split, args.n, reuse=args.reuse)
        if args.out:
            args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(effort_table(args.native, args.teacher), flush=True)
        print(compare_table(rows), flush=True)
        return
    if args.cmd == "teach":
        teach_all(args.split, args.n, args.out, args.offset, args.stride, args.vpd, args.tasks.split(",") if args.tasks else None)
        return
    if args.cmd == "table":
        print(table([json.loads(x) for x in args.rows.read_text().splitlines() if x.strip()]))
        return
    rows = curves(args.split, args.n, args.sizes, args.out, offset=args.offset, stride=args.stride)
    print(table(rows), flush=True)


if __name__ == "__main__":
    main()
