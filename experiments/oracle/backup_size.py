"""How big is the backup VPD gates off (#2951 graph oracle)? From the real model minus VPD's whole active answer, remove
subcomponents VPD marks inactive (causal importance 0, at positions up to the last target) greedily by measured
effect until N_real (goal_eval's necessity on the real model) reaches GOAL: each round takes the SCREEN inactive
subcomponents with the largest gradient of ln KL either way, measures each one's removal alone, and removes the
prefix of the best (1, 2, 4, ..., PREFIX) whose measured removal leaves the largest KL. Per text: how many were
removed, at which sites and positions, N_real along the way. A diagnostic, not a search method.

  python experiments/oracle/backup_size.py --split heldout --n 20 [--offset K --stride S] [--vpd FILE] --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import carry  # noqa: E402
import engine  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

GOAL = 0.9
SCREEN = 256
PREFIX = 64
ROWS = 8
MAX_ROUNDS = 60


def size_backup(ge: goal_eval.GoalEval, tx: goal_eval.Text, log=print) -> dict:
    nat, T, dev = ge.nat, tx.T, ge.nat.dev
    e0, v0 = ge._bounds(tx, "live")
    E = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=None)
    where = (torch.arange(T, device=dev) <= max(tx.targets))[:, None]
    base = {n: (~tx.vpd[n]).float() for n in nat.names}  # the model minus VPD's answer
    cand = {n: (~tx.vpd[n]) & where for n in nat.names}  # inactive subcomponents that can reach the targets
    removed = []
    m = {n: x.clone() for n, x in base.items()}

    def kls(rows):  # KL of each masks dict in rows, ROWS a run
        out = []
        with torch.no_grad():
            for s0 in range(0, len(rows), ROWS):
                part = rows[s0:s0 + ROWS]
                out += E.kl({n: torch.stack([r[n] for r in part]) for n in nat.names}).tolist()
                native.free(dev)
        return out

    def without(masks, parts):
        out = {n: x.clone() for n, x in masks.items()}
        for n, t, c in parts:
            out[n][t, c] = 0.0
        return out

    k = kls([m])[0]
    path = [{"removed": 0, "kl": k, "n_real": goal_eval.necessity_log(e0, k, v0)}]
    for rnd in range(MAX_ROUNDS):
        if path[-1]["n_real"] >= GOAL:
            break
        leaves = {n: x[None].clone().requires_grad_(True) for n, x in m.items()}
        g = torch.autograd.grad(E.kl(leaves).clamp_min(1e-12).log().sum(), list(leaves.values()))
        score = torch.cat([(gi[0].abs() * cand[n]).flatten() for n, gi in zip(nat.names, g)])
        offs, acc = {}, 0
        for n in nat.names:
            offs[n] = acc
            acc += cand[n].numel()
        top = score.topk(min(SCREEN, int((score > 0).sum()))).indices.tolist()
        names = list(nat.names)
        bounds = [offs[n] for n in names] + [acc]
        parts = []
        for j in top:
            i = max(x for x in range(len(names)) if bounds[x] <= j)
            n = names[i]
            t, c = divmod(j - bounds[i], nat.C[n])
            parts.append((n, t, c))
        single = kls([without(m, [p]) for p in parts])
        ranked = [p for _, p in sorted(zip(single, parts), key=lambda x: -x[0])]
        sizes = [s for s in (1, 2, 4, 8, 16, 32, 64) if s <= min(PREFIX, len(ranked))]
        got = kls([without(m, ranked[:s]) for s in sizes])
        i = max(range(len(sizes)), key=lambda j: (got[j], sizes[j]))
        m = without(m, ranked[:sizes[i]])
        for n, t, c in ranked[:sizes[i]]:
            cand[n][t, c] = False
        removed += ranked[:sizes[i]]
        path.append({"removed": len(removed), "kl": got[i], "n_real": goal_eval.necessity_log(e0, got[i], v0)})
        if log:
            log(f"  round {rnd}: {len(removed)} removed, KL {got[i]:.2f}, N_real {path[-1]['n_real']:.3f}")
    sites, positions = {}, {}
    for n, t, c in removed:
        sites[n] = sites.get(n, 0) + 1
        positions[t] = positions.get(t, 0) + 1
    return {"kl_empty": e0, "kl_vpd": v0, "reached": path[-1]["n_real"] >= GOAL, "count": len(removed), "removed": removed,
            "path": path, "sites": sites, "positions": {str(t): k for t, k in sorted(positions.items())},
            "at_targets": sum(k for t, k in positions.items() if t in set(tx.targets))}


def main():
    global ROWS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vpd", type=Path, help="VPD's answers saved by teacher.py --save-vpd")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=ROWS, help="masked runs per batch (each a dense mask over every subcomponent)")
    args = ap.parse_args()
    ROWS = args.rows
    ge = goal_eval.GoalEval()
    nat = ge.nat
    saved = torch.load(args.vpd) if args.vpd else {}
    done = {json.loads(x)["task"] for x in args.out.read_text().splitlines() if x.strip()} if args.out.exists() else set()
    for p in native.tasks(args.split)[:args.n][args.offset::args.stride]:
        if p.stem in done:
            continue
        t0 = time.perf_counter()
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(p.stem), json.loads(p.read_text()))
        if p.stem in saved:
            tx.vpd = {m: torch.zeros(tx.T, nat.C[m], dtype=torch.bool, device=nat.dev) for m in nat.names}
            for m, (t, c) in saved[p.stem]["answer"].items():
                tx.vpd[m][t.long().to(nat.dev), c.long().to(nat.dev)] = True
        row = {"task": p.stem, "T": tx.T, "targets": targets, **size_backup(ge, tx)}
        row["seconds"] = time.perf_counter() - t0
        with args.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{p.stem}: {row['count']} inactive subcomponents to N_real {row['path'][-1]['n_real']:.3f} "
              f"({'reached' if row['reached'] else 'not reached'}), {row['at_targets']} at the targets, {row['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
