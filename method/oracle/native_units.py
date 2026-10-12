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
        """log q at the targets [len(removals), len(targets), V]: each row the base set without its removals (at
        positions >= s). It computes the positions from s on that hold base units, and the targets: a position holding
        none keeps its embedding through every layer, so its keys and values, and those before s, are the base run's
        (keep: compute every position and store them, s = 0)."""
        import vpd_model

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
                kept.append((k[:1], v[:1]))
            else:
                kk, vv = cache[i][0].expand(B, -1, -1, -1).clone(), cache[i][1].expand(B, -1, -1, -1).clone()
                kk[:, :, comp], vv[:, :, comp] = k, v
                k, v = kk, vv
            P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
            hm = base.heads[i, comp].T[None].expand(B, -1, -1).float().clone()  # [B, H, n]
            sel = idx["head"][1] == i
            if sel.any():
                hm[idx["head"][0][sel], idx["head"][3][sel], idx["head"][2][sel]] = 0.0
            att = ((P @ v) * hm[..., None]).transpose(1, 2).reshape(B, n, -1)
            x = x + att @ self.W["o_proj"][i].T
            z2 = vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps)
            nm = base.neurons[i, comp][None].expand(B, -1, -1).float().clone()  # [B, n, d_mlp]
            sel = idx["neuron"][1] == i
            if sel.any():
                nm[idx["neuron"][0][sel], idx["neuron"][2][sel], idx["neuron"][3][sel]] = 0.0
            x = x + (vpd_model.gelu_tanh(z2 @ self.W["c_fc"][i].T) * nm) @ self.W["down_proj"][i].T
        if keep:
            self._cache = (base, kept)
        final = vpd_model.rms(x[:, where[self.targets]], tg.ln_f, tg.eps)
        return torch.log_softmax((final @ tg.wte.T).float(), -1)

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


def teach_all(split: str, n: int, out: Path, offset: int = 0, stride: int = 1, vpd: Path | None = None, log=print) -> None:
    """teach_native on the split's first n texts (offset, offset + stride, ...) -> one row per text appended to OUT.
    vpd: VPD's answers saved by teacher.py --save-vpd (where the causal-importance network is not)."""
    ge = goal_eval.GoalEval()
    nat = ge.nat
    saved = torch.load(vpd) if vpd else {}
    done = {json.loads(x)["task"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    for p in native.tasks(split)[:n][offset::stride]:
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
            ge: goal_eval.GoalEval | None = None, log=print) -> list[dict]:
    """Per text and size: the equal-effort native circuit (native_units teach), the teacher's circuit (teacher.py's
    path) and VPD's first k (its order), at the step nearest each size; F1 and N_VPD (goal_eval's sufficiency:
    remainders off, the convention native circuits share, and live, the teacher's) and N_real (removal from the real
    model: goal_eval's real_removal, native_units' N for native units); locality from the teacher's summary (its
    "local" label, components.locality, else whether its F1 circuit came from the targets alone)."""
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
            out.append({"task": p.stem, "local": local.get(p.stem), "kind": kind, "size": k, "nodes": g.count(),
                        "f1": s["f1_off"], "n_vpd": s["n_off"], "f1_live": s["f1_live"], "n_vpd_live": s["n_live"],
                        "n_real": r["n_real_off"]})
        if p.stem in nrows:
            for k in sizes:
                t = _nearest(nrows[p.stem]["path"], k)
                if t is not None:
                    out.append({"task": p.stem, "local": local.get(p.stem), "kind": "native", "size": k, "nodes": t["size"],
                                "f1": t["f1"], "n_vpd": None, "n_real": t["n"]})
        log(f"{p.stem}: {sum(1 for r in out if r['task'] == p.stem)} circuits scored")
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
    q = sub.add_parser("compare", help="native (teach) vs the teacher vs VPD's order at matched sizes")
    q.add_argument("--native", type=Path, required=True, help="teach's OUT.jsonl")
    q.add_argument("--teacher", type=Path, required=True, help="teacher.py's --out directory")
    q.add_argument("--split", choices=("train", "heldout", "hard"), default="heldout")
    q.add_argument("--n", type=int, default=20)
    q.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "compare":
        rows = compare(args.native, args.teacher, args.split, args.n)
        if args.out:
            args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(effort_table(args.native, args.teacher), flush=True)
        print(compare_table(rows), flush=True)
        return
    if args.cmd == "teach":
        teach_all(args.split, args.n, args.out, args.offset, args.stride, args.vpd)
        return
    if args.cmd == "table":
        print(table([json.loads(x) for x in args.rows.read_text().splitlines() if x.strip()]))
        return
    rows = curves(args.split, args.n, args.sizes, args.out, offset=args.offset, stride=args.stride)
    print(table(rows), flush=True)


if __name__ == "__main__":
    main()
