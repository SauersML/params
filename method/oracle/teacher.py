"""Teacher circuits (#2951 graph oracle): whole-prediction circuits in VPD's semantics, found by measured removal.

A circuit C runs with its subcomponents on and every other one off (VPD's masked forward pass, as goal_eval.GoalEval.runs),
remainders live (the component reward's convention; components.CONVENTION). Every KL is the model's next-token
distribution at the target from the run's, in bits, and the reward's terms are goal_eval's log-scale fields:
  F1 = clip(ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD), 0, 1),  N = clip(ln(KL_{VPD minus C} / KL_VPD) / ln(KL_empty / KL_VPD), 0, 1)
with VPD's own answer (causal importance above zero at the positions that reach the target) and the empty circuit as
the bounds, "VPD minus C" VPD's answer with C's subcomponents off.

Per text:
  1. The F1 circuit: first from VPD's answer at the targets alone (chunks of 4, then single subcomponents); when that
     cannot reach F1 >= 0.9, from VPD's whole answer, rounds at each level (whole positions, sites, chunks of 64, 16 and 4 of a site's
     subcomponents by index, single subcomponents) measure every block's removal, rank the blocks by the KL their removal
     leaves, and remove the most of them in that order that keeps F1 >= 0.9 (KL_C <= KL_empty^0.1 KL_VPD^0.9), until no
     single removal does; with --start-top K, from VPD's top K through the last three levels only. (The linear recovery
     >= 0.9 admits circuits ~10 bits from the model, the empty circuit being ~100 bits away: on text9000 it pruned to 5
     subcomponents at the target.)
  2. VPD's top-k: the smallest prefix of VPD's order (Native._vpd_order) whose necessity N reaches 0.99 (bisection, at
     most 1,024).
  3. The joint path from their union: each step measures every single subcomponent's removal (two runs: the circuit and
     VPD's answer without it), ranks them by the mean of F1 and N they leave, measures removing the first 1, 2, 4, ...
     of them up to a tenth of the circuit and takes the prefix with the highest measured mean, down to 4 subcomponents. Every step is recorded (size, F1, N, KL_C, KL_{VPD minus C}, positions, the subcomponents
     removed) beside VPD's own top-k at the same size.
  4. The teacher circuit is the path point with the largest mean(F1, N) - 0.5 size, size the nodes' description bits
     (with a token-embedding node at each position) over the bits of naming every VPD-active subcomponent at the text's
     F2 positions and target (components.reward's size term; edges, added later by the SFT path, are not priced here).
Every membership decision is measured; orderings only order.

Output per text in --out: <task>.json (the teacher circuit), <task>.f1.json, <task>.vpd_top<k>.json, <task>.n<size>.json at
the path points nearest 16 to 256 nodes (goal_eval's circuit JSON: nodes [matrix, index, position] and ["tok", 0,
position] input nodes, no edges), <task>.path.json (the start set and every step), and a line in summary_<offset>.jsonl.

  MPD_MEM_GIB=16 ~/mpd-data/venv/bin/python -B teacher.py --split heldout --n 20 --out ~/mpd-data/graph_oracle/teacher/joint_heldout
  teacher.py --split heldout --n 20 --save-vpd FILE   (VPD's answers and orders, for a machine without the CI network)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import carry
import goal_eval
import native

RECOVERED = 0.9
ROUNDS = 40
CONVENTION = "live"   # the component reward's remainder convention (components.CONVENTION)
N_TARGET = 0.99       # VPD's top-k: the smallest prefix of VPD's order with necessity N >= this, at most K_MAX
K_MAX = 1024
FRACTION = 0.1        # each joint path step removes this share of the circuit, at least one subcomponent
MIN_SIZE = 4
SIZE_WEIGHT = 0.5     # components.SIZE_WEIGHT
SIZES = (16, 32, 64, 128, 256)  # path points also written
LEVELS = ("position", "site", "chunk64", "chunk16", "chunk4", "subcomponent")  # blocks, coarse to fine


class Pruner:
    """Measured removal over the nodes of a circuit on one text. Nodes are (matrix, position, index); a block is a list
    of node ids; kl(blocks) measures the current circuit without each block, in batches of B."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, start: dict, live: bool = False, B: int = 32):
        self.ge, self.tx, self.live, self.B = ge, tx, live, B
        nat = ge.nat
        self.names = nat.names
        self.nodes = [(n, int(t), int(c)) for n in nat.names for t, c in start[n].nonzero().tolist()]
        self.on = torch.ones(len(self.nodes), dtype=torch.bool, device=nat.dev)
        self.mat = torch.tensor([nat.names.index(n) for n, _, _ in self.nodes], device=nat.dev)
        self.t = torch.tensor([t for _, t, _ in self.nodes], device=nat.dev)
        self.c = torch.tensor([c for _, _, c in self.nodes], device=nat.dev)
        self.measured = 0

    def masks(self, on: torch.Tensor) -> dict:
        nat, T = self.ge.nat, self.tx.T
        out = {}
        for j, n in enumerate(self.names):
            m = torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev)
            sel = on & (self.mat == j)
            m[self.t[sel], self.c[sel]] = True
            out[n] = m
        return out

    def _gather(self) -> None:
        """Per site the subcomponents among the nodes (the start set), all others being off in every run: their columns
        of V and U, and each node's column among them. The products then run on these alone (a sum over fewer
        terms: equal to the full products within float32 rounding, not to the bit)."""
        tg = self.ge.nat.target
        self.sel = [(self.mat == j).nonzero().flatten() for j in range(len(self.names))]
        self.cols = []
        for j, n in enumerate(self.names):
            cols, ci = torch.unique(self.c[self.sel[j]], return_inverse=True)
            st = tg.site(n)
            self.cols.append((ci, (st.V.index_select(1, cols), st.U.index_select(0, cols))))

    def _site_masks(self, on: torch.Tensor) -> dict:
        """Each site's float mask [B, T, columns] of node subsets on [B, N]."""
        nat, T = self.ge.nat, self.tx.T
        out = {}
        for j, n in enumerate(self.names):
            sel, (ci, (V, _)) = self.sel[j], self.cols[j]
            m = torch.zeros(len(on), T, V.shape[1], device=nat.dev)
            if len(sel):
                m[:, self.t[sel], ci] = on[:, sel].float()
            out[n] = m
        return out

    def _run(self, masks: dict, B: int, start: int = 0, prefix: list | None = None, layer: int = 0, keys_values: bool = False):
        """The masked forward of B copies of the text (Target.hidden from start and layer, or its keys_values)."""
        nat, tg, T = self.ge.nat, self.ge.nat.target, self.tx.T
        for j, n in enumerate(self.names):
            st = tg.site(n)
            st.mask = masks[n][:, start:]
            st.gathered = self.cols[j][1]
            st.delta_mask = torch.ones(B, T - start, device=nat.dev) if self.live else None
        ids = torch.tensor([self.tx.ids], device=nat.dev).expand(B, -1)
        try:
            return tg.keys_values(ids) if keys_values else tg.hidden(ids, start, prefix, layer)
        finally:
            nat.vpd.clear()

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor, pad: bool = False) -> list[float]:
        """KL at the target of each node subset's run, ons [K, N] (float masks built on the device, in batches of B; a
        last batch of one row runs alone, at batch 1, unless pad).
        With at least two full batches on a Native.WINDOW_EXACT device, the rows are taken in order of the first layer,
        then the first position, where each differs from the call's majority set, and a batch runs from its first such
        layer and position (at least 64 positions) on, the layers below and the positions before reading the majority
        set's run (its queries, keys, values and residual stream, computed once per set at batch B; Target.hidden's
        window): the same numbers as the whole run there."""
        nat, tg, T = self.ge.nat, self.ge.nat.target, self.tx.T
        if not hasattr(self, "sel"):
            self._gather()
            self.layer = torch.tensor([int(n.split(".")[1]) for n in self.names], device=nat.dev, dtype=torch.int32)[self.mat]
        K = len(ons)
        window = K >= 2 * self.B and T > 64 and str(nat.dev).split(":")[0] in getattr(nat, "WINDOW_EXACT", ())
        order, first, lowest = torch.arange(K, device=nat.dev), None, None
        if window:
            base = ons.sum(0) * 2 > K
            key = base.cpu().numpy().tobytes()
            runs = self.__dict__.setdefault("_prefixes", {})                # the last two sets' runs (measure alternates C, ~C)
            if key not in runs:
                if len(runs) >= 2:
                    runs.pop(next(iter(runs)))
                runs[key] = self._run(self._site_masks(base[None].expand(self.B, -1)), self.B, keys_values=True)
            prefix = runs[key]
            off = ons != base
            pos = torch.where(off, self.t.int()[None], T).amin(1)            # each row's first position and layer off the base
            lay = torch.where(off, self.layer[None], tg.n_layer).amin(1)
            # A call's last row alone in its batch runs at batch 1 (no padding), which can round differently: it stays last
            n = K - 1 if K % self.B == 1 and not pad else K
            order = torch.cat([torch.argsort((lay * (T + 1) + pos)[:n], stable=True), order[n:]])
            first, lowest = pos[order].tolist(), lay[order].tolist()
        kl = [0.0] * K
        for s0 in range(0, K, self.B):
            idx = order[s0:s0 + self.B]
            on = ons[idx]
            valid = len(on)
            if (pad or valid > 1) and valid < self.B:                           # one batch shape: the GPU compiles per shape
                on = torch.cat([on, on[-1:].expand(self.B - valid, -1)])
            B = len(on)
            reuse = window and B == self.B
            start = max(0, min(min(first[s0:s0 + valid]), T - 64, min(self.tx.targets))) if reuse else 0
            layer = min(lowest[s0], tg.n_layer - 1) if reuse else 0
            hidden = self._run(self._site_masks(on), B, start, prefix if reuse else None, layer)
            lq = torch.log_softmax((hidden[:, [t - start for t in self.tx.targets]] @ tg.wte.T).float(), -1)
            for i, x in zip(idx.tolist(), carry.kl(self.tx.ref, lq).tolist()[:valid]):
                kl[i] = x
        self.measured += K
        return kl

    def without(self, groups: list[torch.Tensor]) -> list[float]:
        """KL of the current circuit without each group of node ids (built 16 batches at a time)."""
        out = []
        for s0 in range(0, len(groups), 16 * self.B):
            part = groups[s0:s0 + 16 * self.B]
            on = self.on[None].repeat(len(part), 1)
            for i, g in enumerate(part):
                on[i, g] = False
            out += self.kl_sets(on)
        return out

    def kl(self, on: torch.Tensor | None = None) -> float:
        return self.kl_sets((self.on if on is None else on)[None])[0]

    def blocks(self, level: str) -> list[torch.Tensor]:
        """The current circuit's blocks at a level, as node-id tensors."""
        ids = self.on.nonzero().flatten()
        if level == "position":
            keys = self.t[ids]
        elif level == "site":
            keys = self.t[ids] * len(self.names) + self.mat[ids]
        elif level.startswith("chunk"):               # chunks of K subcomponents within a site, by index
            K = int(level[5:])
            site = self.t[ids] * len(self.names) + self.mat[ids]
            order = torch.argsort(site * (1 << 20) + self.c[ids], stable=True)
            ids, site = ids[order], site[order]
            first = torch.ones_like(site, dtype=torch.bool)
            first[1:] = site[1:] != site[:-1]
            start = torch.cummax(torch.where(first, torch.arange(len(ids), device=ids.device), 0), 0).values
            keys = site * (1 << 20) + (torch.arange(len(ids), device=ids.device) - start) // K
        else:
            return list(ids.split(1))
        order = torch.argsort(keys, stable=True)
        ids, keys = ids[order], keys[order]
        _, counts = torch.unique_consecutive(keys, return_counts=True)
        return list(ids.split(counts.tolist()))

    def prune(self, level: str, budget: float, rounds: int = ROUNDS) -> int:
        """Rounds of measured block removal at one level; returns the rounds run. Within the budget a round removes
        the most of the ranked blocks that keep KL_C <= budget; above it (a start such as VPD's answer at the targets
        alone can be farther than the budget while a subset is not: subcomponents cancel) it removes the prefix of
        the ranked blocks with the lowest measured KL_C, when that is lower than the circuit's."""
        for r in range(rounds):
            blocks = self.blocks(level)
            if len(blocks) <= 1:
                return r
            single = torch.tensor(self.without(blocks))
            ranked = [blocks[i] for i in torch.argsort(single, stable=True).tolist()]
            sizes = [k for k in sorted({int(2 ** (e / 2)) for e in range(0, 60)}) if k <= len(ranked) - 1]
            if (single <= budget).any():
                ok = [k for k, x in zip(sizes, self.without([torch.cat(ranked[:k]) for k in sizes])) if x <= budget]
                if not ok:
                    return r
                self.on[torch.cat(ranked[:max(ok)])] = False
                continue
            current = self.kl()
            if current <= budget or float(single.min()) >= current:
                return r
            kl = self.without([torch.cat(ranked[:k]) for k in sizes])
            i = min(range(len(sizes)), key=lambda j: kl[j])
            if kl[i] >= current:
                return r
            self.on[torch.cat(ranked[:sizes[i]])] = False
        return rounds


def scores(empty: float, vpd: float, c: float, minus: float) -> tuple[float, float, float]:
    """(F1, N, their mean): goal_eval's log-scale reward fields from KL_C and KL_{VPD minus C}."""
    f1 = min(1.0, max(0.0, goal_eval.recovered_log(empty, c, vpd)))
    n = goal_eval.necessity_log(empty, minus, vpd)
    return f1, n, (f1 + n) / 2


def measure(pr: Pruner, ons: torch.Tensor, empty: float, vpd: float) -> list[dict]:
    """Each node set's KL_C and KL_{VPD minus C} (two runs: the set, and VPD's answer without it), F1, N and mean."""
    K, out = len(ons), []
    if 2 * K <= pr.B:                  # one batch either way
        kl = pr.kl_sets(torch.cat([ons, ~ons]))
        cs, ms = kl[:K], kl[K:]
    else:                              # apart, each call's rows sharing one set they differ from; every row at batch B
        cs, ms = pr.kl_sets(ons, pad=True), pr.kl_sets(~ons, pad=True)
    for c, m in zip(cs, ms):
        f1, n, j = scores(empty, vpd, c, m)
        out.append({"kl_c": c, "kl_minus": m, "f1": f1, "n": n, "mean": j})
    return out


def top_k(pr: Pruner, order: torch.Tensor, empty: float, vpd: float, target: float = N_TARGET, cap: int = K_MAX) -> int:
    """The smallest k (bisection, measured) whose VPD-ordered prefix has N >= target, at most cap."""
    def prefix(k):
        on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=order.device)
        on[order[:k]] = True
        return on
    lo, hi = 0, min(cap, len(order))
    if measure(pr, prefix(hi)[None], empty, vpd)[0]["n"] < target:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if measure(pr, prefix(mid)[None], empty, vpd)[0]["n"] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def joint_path(pr: Pruner, start: torch.Tensor, empty: float, vpd: float) -> list[dict]:
    """Nested measured removal from `start` down to MIN_SIZE. Each step measures every single removal's F1 and N (two
    runs each) and ranks the removals by the mean they leave; then it measures removing the first 1, 2, 4, ... of them
    up to a tenth of the circuit (removals interact: subcomponents cancel) and takes the prefix with the highest
    measured mean, the longest on a tie. Records: the state after each step and the node ids removed."""
    on = start.clone()
    path = [{"size": int(on.sum()), "removed": [], **measure(pr, on[None], empty, vpd)[0]}]
    while int(on.sum()) > MIN_SIZE:
        ids = on.nonzero().flatten()
        trials = on[None].repeat(len(ids), 1)
        trials[torch.arange(len(ids), device=on.device), ids] = False
        rows = []
        for s0 in range(0, len(ids), 4 * pr.B):
            rows += measure(pr, trials[s0:s0 + 4 * pr.B], empty, vpd)
        ranked = ids[torch.argsort(torch.tensor([r["mean"] for r in rows]), descending=True, stable=True)]
        top = min(max(1, int(FRACTION * len(ids))), len(ids) - MIN_SIZE)
        ks = sorted({min(top, 2 ** e) for e in range(0, 20) if 2 ** e <= 2 * top})
        prefixes = on[None].repeat(len(ks), 1)
        for i, k in enumerate(ks):
            prefixes[i, ranked[:k]] = False
        got = measure(pr, prefixes, empty, vpd)
        i = max(range(len(ks)), key=lambda j: (got[j]["mean"], ks[j]))
        on[ranked[:ks[i]]] = False
        path.append({"size": int(on.sum()), "removed": ranked[:ks[i]].tolist(), **got[i]})
    return path


def circuit_json(pr: Pruner, on: torch.Tensor) -> dict:
    """goal_eval's circuit JSON: the nodes, and a token-embedding input node at each of their positions; no edges."""
    nodes = sorted(((t, n, c) for (n, t, c), keep in zip(pr.nodes, on.tolist()) if keep))
    tok = sorted({t for t, _, _ in nodes})
    return {"nodes": [[n, c, t] for t, n, c in nodes] + [["tok", 0, t] for t in tok], "edges": [], "explanation": ""}


def teach(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], B: int = 32, vpd_answer: dict | None = None,
          vpd_order: list | None = None, start_top: int = 0) -> tuple[dict, dict]:
    """(circuits by name, summary row) on one text: the F1 circuit (smallest at recovered_log >= 0.9, coarse to fine),
    VPD's top-k with N >= N_TARGET, the joint path from their union, and the path point the reward's terms favour."""
    t0 = time.perf_counter()
    nat = ge.nat
    tx = ge.text(ids, targets)
    if vpd_answer is not None:                                # VPD's answer computed where the causal-importance network is
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in vpd_answer.items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    if tx.vpd is None:
        raise ValueError("no VPD answer: run with the causal-importance network or pass --vpd")
    empty, vpd = ge._bounds(tx, CONVENTION)
    budget = empty ** (1 - RECOVERED) * vpd ** RECOVERED
    pr = Pruner(ge, tx, tx.vpd, live=CONVENTION == "live", B=B)
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    order = vpd_order if vpd_order is not None else nat._vpd_order(ids, targets)
    order = torch.tensor([index[(n, int(t), int(c))] for n, t, c in order if (n, int(t), int(c)) in index], device=nat.dev)
    positions = sorted({p for p, _ in tx.f2} | set(targets))
    active = sum(int(tx.vpd[n][positions].sum()) for n in nat.names)
    node_bits = math.log2(nat.positions(targets) * sum(nat.C.values()))
    scale = active * node_bits                               # components.reward's size scale

    def size_term(on):
        k_tok = int(torch.unique(pr.t[on]).numel())
        return min(1.0, (int(on.sum()) * node_bits + k_tok * math.log2(nat.positions(targets))) / scale) if scale else 0.0

    row = {"T": tx.T, "convention": CONVENTION, "vpd_size": len(pr.nodes), "vpd_positions": int(torch.unique(pr.t).numel()),
           "kl_empty": empty, "kl_vpd": vpd, "budget_kl": budget, "active_scale": active, "levels": {}}
    def search(levels, label):
        for level in levels:
            t1 = time.perf_counter()
            rounds = pr.prune(level, budget)
            row["levels"][f"{label}{level}"] = lv = {"rounds": rounds, "size": int(pr.on.sum()),
                                                     "positions": int(torch.unique(pr.t[pr.on]).numel()), "kl": pr.kl(),
                                                     "seconds": time.perf_counter() - t1}
            print(f"  {label}{level}: {lv['size']} subcomponents at {lv['positions']} positions, KL {lv['kl']:.3g} "
                  f"(budget {budget:.3g}), {rounds} rounds, {lv['seconds']:.0f}s, {pr.measured} sets so far", flush=True)
        return row["levels"][f"{label}{levels[-1]}"]["kl"] <= budget

    whole = pr.on.clone()
    reached = False
    if not start_top:                                        # first VPD's answer at the targets alone, subcomponents only
        pr.on = whole & torch.isin(pr.t, torch.tensor(targets, device=pr.t.device))
        reached = search(("chunk4", "subcomponent"), "target ")
    if not reached:
        pr.on = whole.clone()
        levels = LEVELS
        if start_top:                                        # the F1 search from VPD's top start_top, subcomponents only
            pr.on[:] = False
            pr.on[order[:start_top]] = True
            levels = ("chunk16", "chunk4", "subcomponent")
        search(levels, "")
    row["f1_from_targets"] = reached
    f1_on = pr.on.clone()
    t1 = time.perf_counter()
    k = top_k(pr, order, empty, vpd)
    topk_on = torch.zeros_like(f1_on)
    topk_on[order[:k]] = True
    start = f1_on | topk_on
    path = joint_path(pr, start, empty, vpd)
    sizes = [r["size"] for r in path]
    vs = []
    for sz in sizes:                                         # VPD's own order at every path size
        on = torch.zeros_like(f1_on)
        on[order[:sz]] = True
        vs.append(on)
    vpd_rows = []
    for s0 in range(0, len(vs), pr.B):
        vpd_rows += measure(pr, torch.stack(vs[s0:s0 + pr.B]), empty, vpd)
    removed, ons = [], []
    on = start.clone()
    for r in path:
        on[torch.tensor(r["removed"], dtype=torch.long, device=on.device)] = False
        ons.append(on.clone())
        r["size_term"] = size_term(on)
        r["reward"] = r["mean"] - SIZE_WEIGHT * r["size_term"]
        r["positions"] = int(torch.unique(pr.t[on]).numel())
    for r, v, on_v in zip(path, vpd_rows, vs):
        v["size_term"] = size_term(on_v)
        v["reward"] = v["mean"] - SIZE_WEIGHT * v["size_term"]
        r["vpd_top"] = v
    best = max(range(len(path)), key=lambda i: path[i]["reward"])
    circuits = {"": circuit_json(pr, ons[best]), "f1": circuit_json(pr, f1_on), f"vpd_top{k}": circuit_json(pr, topk_on)}
    for want in SIZES:
        i = min(range(len(path)), key=lambda j: abs(path[j]["size"] - want))
        circuits[f"n{path[i]['size']}"] = circuit_json(pr, ons[i])
    f1_row = measure(pr, f1_on[None], empty, vpd)[0]
    row.update(f1_circuit={"size": int(f1_on.sum()), "positions": int(torch.unique(pr.t[f1_on]).numel()), **f1_row},
               top_k=k, start_size=int(start.sum()), path_seconds=time.perf_counter() - t1,
               best={k_: path[best][k_] for k_ in ("size", "positions", "f1", "n", "mean", "size_term", "reward", "kl_c", "kl_minus")},
               best_vpd_top=path[best]["vpd_top"], measured_sets=pr.measured, seconds=time.perf_counter() - t0)
    path_file = {"start_nodes": [[n, c, t] for n, t, c in (pr.nodes[i] for i in start.nonzero().flatten().tolist())],
                 "start_index": start.nonzero().flatten().tolist(), "steps": path}
    return circuits, row, path_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="this process takes the first n texts offset, offset + stride, ...")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vpd", type=Path, help="VPD's answers and orders saved by --save-vpd (no causal-importance network here)")
    ap.add_argument("--save-vpd", type=Path, help="save VPD's answers and orders on the texts to this file and stop")
    ap.add_argument("--labels", type=Path, help="locality JSONL (task, local)")
    ap.add_argument("--start-top", type=int, default=0, help="F1 search from VPD's top K at subcomponent level (0: from VPD's whole answer)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    if args.save_vpd:
        saved = {}
        for path in native.tasks(args.split)[:args.n]:
            ids, targets = native.text(path)
            m = ge.vpd_answer(ids, targets)
            saved[path.stem] = {"answer": {n: tuple(x.to(torch.int32).cpu() for x in v.nonzero().T) for n, v in m.items()},
                                "order": ge.nat._vpd_order(ids, targets)}
        torch.save(saved, args.save_vpd)
        return
    vpd = torch.load(args.vpd) if args.vpd else {}
    labels = {}
    if args.labels and args.labels.exists():
        for line in args.labels.read_text().splitlines():
            r = json.loads(line)
            labels[r["task"]] = r.get("local")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in native.tasks(args.split)[:args.n][args.offset::args.stride]:
        if (args.out / f"{path.stem}.json").exists():
            continue
        ids, targets = native.text(path)
        saved = vpd.get(path.stem, {})
        circuits, row, path_file = teach(ge, ids, targets, args.batch, saved.get("answer"), saved.get("order"), args.start_top)
        for name, c in circuits.items():
            (args.out / f"{path.stem}{'.' + name if name else ''}.json").write_text(json.dumps(c))
        (args.out / f"{path.stem}.path.json").write_text(json.dumps(path_file))
        row = {"task": path.stem, "local": labels.get(path.stem), **row}
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        b, v, f = row["best"], row["best_vpd_top"], row["f1_circuit"]
        print(f"{path.stem}: F1 circuit {f['size']} at {f['positions']} positions (F1 {f['f1']:.3f}, N {f['n']:.3f}); top-k {row['top_k']}; "
              f"path from {row['start_size']}: best {b['size']} at {b['positions']} positions F1 {b['f1']:.3f} N {b['n']:.3f} reward {b['reward']:.3f} "
              f"vs VPD top-{b['size']} F1 {v['f1']:.3f} N {v['n']:.3f} reward {v['reward']:.3f}; {row['measured_sets']} sets, {row['seconds']:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
