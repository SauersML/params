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
      when it first falls under each of RESCREEN's sizes.
The curve is every measured (size, KL) down to the empty set.

A removal at position p changes nothing before p (the model is causal), so a batch of removals computes only the
positions from its first removal on, reading the set's keys and values before it from the set's own run.

  native_units.py curve --split heldout --n 20 [--sizes 380 ...] [--out OUT.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

SCREEN_FLOOR = 0.01  # an MLP's neurons are screened one by one when its removal moves the prediction this share of the top MLP's
RESCREEN = (4096, 1024, 256, 64)  # set sizes under which the singles are measured again
BATCH = 64  # removals per batched run
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

    def without(self, units) -> "Mask":
        m = Mask(self.heads.clone(), self.neurons.clone())
        for kind, l, t, j in units:
            (m.heads if kind == "head" else m.neurons)[l, t, j] = False
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
        """log q at the targets [len(removals), len(targets), V]: each row the base set without its removals (at
        positions >= s), computing positions s..T-1 and reading the base set's keys and values before s (keep: store
        them, s = 0)."""
        import vpd_model

        tg, B, T, H, hd = self.tg, len(removals), self.T, self.H, self.hd
        Lw = T - s
        x = self.emb[s:][None].expand(B, -1, -1).clone()
        cache = None if keep else self._cache[1]
        kept = []
        causal = torch.ones(T, T, dtype=torch.bool, device=self.nat.dev).tril()[s:]
        rows = {"head": [[], [], [], []], "neuron": [[], [], [], []]}
        for b, rem in enumerate(removals):
            for kind, l, t, j in rem:
                if t >= s:
                    r = rows[kind]
                    r[0].append(b)
                    r[1].append(l)
                    r[2].append(t - s)
                    r[3].append(j)
        idx = {k: tuple(torch.tensor(v, dtype=torch.long, device=self.nat.dev) for v in r) for k, r in rows.items()}
        for i in range(self.L):
            z = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
            heads = lambda y: y.view(B, Lw, H, hd).transpose(1, 2)  # noqa: E731
            q, k, v = heads(z @ self.W["q_proj"][i].T), heads(z @ self.W["k_proj"][i].T), heads(z @ self.W["v_proj"][i].T)
            rot = lambda y: y * tg.cos[s:T] + torch.cat([-y[..., hd // 2:], y[..., :hd // 2]], -1) * tg.sin[s:T]  # noqa: E731
            q, k = rot(q), rot(k)
            if keep:
                kept.append((k[:1], v[:1]))
            else:
                k = torch.cat([cache[i][0][:, :, :s].expand(B, -1, -1, -1), k], 2)
                v = torch.cat([cache[i][1][:, :, :s].expand(B, -1, -1, -1), v], 2)
            P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
            hm = base.heads[i, s:].T[None].expand(B, -1, -1).float().clone()  # [B, H, Lw]
            sel = idx["head"][1] == i
            if sel.any():
                hm[idx["head"][0][sel], idx["head"][3][sel], idx["head"][2][sel]] = 0.0
            att = ((P @ v) * hm[..., None]).transpose(1, 2).reshape(B, Lw, -1)
            x = x + att @ self.W["o_proj"][i].T
            z2 = vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps)
            nm = base.neurons[i, s:][None].expand(B, -1, -1).float().clone()  # [B, Lw, d_mlp]
            sel = idx["neuron"][1] == i
            if sel.any():
                nm[idx["neuron"][0][sel], idx["neuron"][2][sel], idx["neuron"][3][sel]] = 0.0
            x = x + (vpd_model.gelu_tanh(z2 @ self.W["c_fc"][i].T) * nm) @ self.W["down_proj"][i].T
        if keep:
            self._cache = (base, kept)
        final = vpd_model.rms(x[:, [t - s for t in self.targets]], tg.ln_f, tg.eps)
        return torch.log_softmax((final @ tg.wte.T).float(), -1)

    def kl(self, base: Mask, removals: list[list[tuple]] | None = None) -> list[float]:
        """The KL in bits of the model at the targets from each run: the base set alone (removals None), or the base
        set without each row's removals, batched by their first position."""
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


def screen(nu: NativeUnits, S: Mask, base_kl: float) -> dict:
    """Unit -> its measured removal effect (KL of the set without it, minus the set's): heads one by one; MLPs as
    wholes at each (layer, position), then the neurons of those moving the prediction at least SCREEN_FLOOR as much
    as the top MLP one by one (the others' neurons each get their MLP's effect divided among them)."""
    units = S.units()
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
    for u, k in zip(singles, nu.kl(S, [[u] for u in singles])):
        effect[u] = k - base_kl
    return effect


def prune(nu: NativeUnits, positions, sizes=(), log=None) -> list[dict]:
    """The measured pruning curve from every unit at the positions: [{"size", "kl", "units" (at the sizes asked
    for)}], from S0 down to the empty set."""
    S = nu.at(positions)
    kl = nu.kl(S)[0]
    curve = [{"size": S.count(), "kl": kl}]
    effect = screen(nu, S, kl)
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
        if due and S.count() > 0:
            rescreened |= set(due)
            effect = screen(nu, S, kl)
    return curve


def f1(kl: float, empty: float, vpd: float) -> float | None:
    """goal_eval's F1 of a run's KL against the bounds."""
    r = goal_eval.recovered_log(empty, kl, vpd)
    return None if r is None else min(1.0, max(0.0, r))


def curves(split: str, n: int, sizes=(), out: Path | None = None, log=print) -> list[dict]:
    """The native-unit pruning curve of the split's first n texts with goal_eval's bounds (remainders off) and VPD's
    own order at the same sizes (vpd@k, remainders off) -> one row per text."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    rows = []
    for p in native.tasks(split)[:n]:
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
        positions = sorted(set(tx.targets) | {q for q, _ in tx.f2})
        empty, vpd = ge._bounds(tx, "off")
        nu = NativeUnits(nat, ids, targets)
        curve = prune(nu, positions, sizes, log=None)
        for c in curve:
            c["f1"] = f1(c["kl"], empty, vpd)
        order = nat._vpd_order(ids, targets)
        ks = sorted({c["size"] for c in curve if 0 < c["size"] <= len(order)})
        vk = carry.kl(tx.ref, ge.runs(ids, targets, [ge.masks(native.Graph(order[:k], complete=True), tx.T) for k in ks], False)).tolist() if ks else []
        row = {"task": p.stem, "T": tx.T, "positions": positions, "kl_empty_off": empty, "kl_vpd_off": vpd, "native": curve,
               "vpd_order": [{"size": k, "kl": v, "f1": f1(v, empty, vpd)} for k, v in zip(ks, vk)],
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        if out:
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        at = {c["size"]: c for c in curve}
        msg = ", ".join(f"{k}: native F1 {at[k]['f1']:.2f}, VPD order {f1(v, empty, vpd):.2f}" for k, v in zip(ks, vk) if k in set(sizes))
        log(f"{p.stem}: T {tx.T}, {len(positions)} positions, {curve[0]['size']} units, {len(curve)} steps, {msg or ''} {row['seconds']:.0f} s")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--sizes", type=int, nargs="*", default=[])
    c.add_argument("--out", type=Path)
    args = ap.parse_args()
    curves(args.split, args.n, args.sizes, args.out)


if __name__ == "__main__":
    main()
