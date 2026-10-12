"""Teacher circuits (#2951 graph oracle): the smallest whole-prediction circuit, in VPD's semantics, found by measured
removal from VPD's own answer.

A circuit C runs with its subcomponents on and every other one off (goal_eval.GoalEval.runs, VPD's masked forward
pass); remainders W - V U off (native's run-alone convention) while searching, both conventions reported. On the text,
KL_C is the KL in bits of the model's next-token distribution at the target from C's run, and
  recovered = (KL_empty - KL_C) / (KL_empty - KL_VPD),  recovered_log = ln(KL_empty / KL_C) / ln(KL_empty / KL_VPD)
with VPD's own answer (causal importance above zero at the positions that reach the target, goal_eval.vpd_answer) and
the empty circuit as the bounds. The search starts from VPD's answer and removes blocks while the criterion stays at or
above 0.9: whole positions, then sites (one weight matrix at one position), then chunks of 64, 16 and 4 of a
site's subcomponents (by index), then single subcomponents. The criterion
is recovered_log (default): the empty circuit is ~100 bits from the model, so recovered >= 0.9 admits a circuit ~10
bits from it (on text9000, 5 subcomponents at the target position), while recovered_log >= 0.9 asks
KL_C <= KL_empty^0.1 KL_VPD^0.9 (0.11 bits there, twice VPD's own 0.05). At each level a
round measures every single block's removal, ranks the blocks by the KL their removal leaves, and removes the most of
them in that order (prefixes of 2^(k/2) blocks) that keeps recovered >= 0.9; rounds repeat until no single removal
does. Every number is measured; no ranking decides membership.

Output: per text, <task>.json in goal_eval's circuit JSON (nodes [matrix, index, position], no edges: the SFT path adds
measured edges), and a summary row: size, positions kept, KL_C and recovered under both remainder conventions,
necessity N (VPD's answer without C), time.

  MPD_MEM_GIB=20 ~/mpd-data/venv/bin/python -B teacher.py --split heldout --n 20 --out ~/mpd-data/graph_oracle/teacher/vpd_f1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import carry
import goal_eval
import native

RECOVERED = 0.9
ROUNDS = 40
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

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor) -> list[float]:
        """KL at the target of each node subset's run, ons [K, N] (float masks built on the device, in batches of B)."""
        nat, tg, T = self.ge.nat, self.ge.nat.target, self.tx.T
        out = []
        for s0 in range(0, len(ons), self.B):
            on = ons[s0:s0 + self.B]
            valid = len(on)
            if 1 < valid < self.B:                                              # one batch shape: the GPU compiles per shape
                on = torch.cat([on, on[-1:].expand(self.B - valid, -1)])
            B = len(on)
            for j, n in enumerate(self.names):
                sel = (self.mat == j).nonzero().flatten()
                m = torch.zeros(B, T, nat.C[n], device=nat.dev)
                if len(sel):
                    m[:, self.t[sel], self.c[sel]] = on[:, sel].float()
                st = tg.site(n)
                st.mask = m
                st.delta_mask = torch.ones(B, T, device=nat.dev) if self.live else None
            try:
                hidden = tg.hidden(torch.tensor([self.tx.ids], device=nat.dev).expand(B, -1))
            finally:
                nat.vpd.clear()
            lq = torch.log_softmax((hidden[:, self.tx.targets] @ tg.wte.T).float(), -1)
            out += carry.kl(self.tx.ref, lq).tolist()[:valid]
        self.measured += len(ons)
        return out

    def without(self, groups: list[torch.Tensor]) -> list[float]:
        """KL of the current circuit without each group of node ids (built a batch at a time)."""
        out = []
        for s0 in range(0, len(groups), self.B):
            part = groups[s0:s0 + self.B]
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
        """Rounds of measured block removal at one level; returns the rounds run."""
        for r in range(rounds):
            blocks = self.blocks(level)
            if len(blocks) <= 1:
                return r
            single = torch.tensor(self.without(blocks))
            if not (single <= budget).any():
                return r
            ranked = [blocks[i] for i in torch.argsort(single, stable=True).tolist()]
            sizes = [k for k in sorted({int(2 ** (e / 2)) for e in range(0, 60)}) if k <= len(ranked)]
            ok = [k for k, x in zip(sizes, self.without([torch.cat(ranked[:k]) for k in sizes])) if x <= budget]
            if not ok:
                return r
            self.on[torch.cat(ranked[:max(ok)])] = False
        return rounds


def circuit_json(pr: Pruner) -> dict:
    nodes = sorted(((t, n, c) for (n, t, c), keep in zip(pr.nodes, pr.on.tolist()) if keep))
    return {"nodes": [[n, c, t] for t, n, c in nodes], "edges": [], "explanation": ""}


def teach(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], B: int = 32, criterion: str = "log",
          vpd: dict | None = None) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    tx = ge.text(ids, targets)
    if vpd is not None:                                       # VPD's answer computed where the causal-importance network is
        nat = ge.nat
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in vpd.items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    if tx.vpd is None:
        raise ValueError("no VPD answer: run with the causal-importance network or pass --vpd")
    empty, vpd = ge._bounds(tx, "off")
    budget = empty ** (1 - RECOVERED) * vpd ** RECOVERED if criterion == "log" else empty - RECOVERED * (empty - vpd)
    pr = Pruner(ge, tx, tx.vpd, live=False, B=B)
    row = {"criterion": criterion, "T": tx.T, "vpd_size": len(pr.nodes), "vpd_positions": int(torch.unique(pr.t).numel()), "kl_empty": empty, "kl_vpd": vpd,
           "budget_kl": budget, "levels": {}}
    for level in LEVELS:
        t1 = time.perf_counter()
        rounds = pr.prune(level, budget)
        row["levels"][level] = lv = {"rounds": rounds, "size": int(pr.on.sum()), "positions": int(torch.unique(pr.t[pr.on]).numel()),
                                     "kl": pr.kl(), "seconds": time.perf_counter() - t1}
        print(f"  {level}: {lv['size']} subcomponents at {lv['positions']} positions, KL {lv['kl']:.3g} (budget {budget:.3g}), "
              f"{rounds} rounds, {lv['seconds']:.0f}s, {pr.measured} sets so far", flush=True)
    C = pr.masks(pr.on)
    g = native.Graph(masks=C)
    sc = ge.sufficiency(tx, [g])[0]
    row.update(size=int(pr.on.sum()), positions=int(torch.unique(pr.t[pr.on]).numel()), measured_sets=pr.measured,
               seconds=time.perf_counter() - t0,
               **{k: sc[k] for k in ("kl_off", "recovered_off", "kl_live", "recovered_live", "kl_vpd_live", "kl_empty_live",
                                     "kl_vpd_minus_off", "necessity_off", "kl_vpd_minus_live", "necessity_live", "recovered_log_off")})
    return circuit_json(pr), row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="this process takes the first n texts offset, offset + stride, ...")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--criterion", choices=("log", "linear"), default="log", help="recovered_log or recovered >= 0.9")
    ap.add_argument("--vpd", type=Path, help="VPD's answers saved by --save-vpd (a pod has no causal-importance network)")
    ap.add_argument("--save-vpd", type=Path, help="save VPD's answers on the texts to this file and stop")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    paths = native.tasks(args.split)[:args.n][args.offset::args.stride]
    if args.save_vpd:
        saved = {}
        for path in native.tasks(args.split)[:args.n]:
            ids, targets = native.text(path)
            m = ge.vpd_answer(ids, targets)
            saved[path.stem] = {n: tuple(x.to(torch.int32).cpu() for x in v.nonzero().T) for n, v in m.items()}
        torch.save(saved, args.save_vpd)
        return
    vpd = torch.load(args.vpd) if args.vpd else {}
    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if (args.out / f"{path.stem}.json").exists():
            continue
        ids, targets = native.text(path)
        circuit, row = teach(ge, ids, targets, args.batch, args.criterion, vpd.get(path.stem))
        (args.out / f"{path.stem}.json").write_text(json.dumps(circuit))
        with (args.out / "summary.jsonl").open("a") as f:
            f.write(json.dumps({"task": path.stem, **row}) + "\n")
        lv = row["levels"]
        print(f"{path.stem}: VPD {row['vpd_size']} at {row['vpd_positions']} positions -> positions {lv['position']['size']} "
              f"({lv['position']['positions']}) -> sites {lv['site']['size']} -> {row['size']} subcomponents at {row['positions']} positions; "
              f"KL {row['kl_off']:.3g} (VPD {row['kl_vpd']:.3g}, empty {row['kl_empty']:.3g}), recovered {row['recovered_off']:.3f}, "
              f"log {row['recovered_log_off']:.3f}, N {row['necessity_off']:.3f}; {row['measured_sets']} sets, {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
