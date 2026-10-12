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

  native_units.py curve --split heldout --n 20 [--sizes 12 16 ...] [--out OUT.jsonl]   per text, then the table by size
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
RESCREEN = (4096, 1024)  # set sizes under which the singles are measured again
FINE = 256  # and at every pruning step from this size down
SIZES = (12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024)  # the matched sizes reported by default (teachers' circuits)
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


def screen(nu: NativeUnits, S: Mask, base_kl: float, log=None) -> dict:
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


def curves(split: str, n: int, sizes=SIZES, out: Path | None = None, log=print) -> list[dict]:
    """Per text of the split's first n: the native-unit pruning curve, and at each size asked for the native
    circuit's and VPD's own first k (its order, native.Native._vpd_order) KL, F1 and N on goal_eval's scale (remainders
    off, the native units having none; VPD's also live), with their node bits -> one row per text (appended to OUT)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    c.add_argument("--out", type=Path)
    t = sub.add_parser("table")
    t.add_argument("rows", type=Path, help="curve's OUT.jsonl")
    args = ap.parse_args()
    if args.cmd == "table":
        print(table([json.loads(x) for x in args.rows.read_text().splitlines() if x.strip()]))
        return
    rows = curves(args.split, args.n, args.sizes, args.out)
    print(table(rows), flush=True)


if __name__ == "__main__":
    main()
