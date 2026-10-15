"""How many subcomponents VPD gates off carry an edit's response (#2951, the lead's 10-14): with every subcomponent at the
edited positions added, a VPD circuit's mediated share goes to about 1 (edit_positions_test), so the response to removing
its top parts runs through subcomponents outside the circuit at those positions. Per text and removal set R (the VPD
circuit's top 1, 3 and 5 parts by real-model drop):
  responders   the subcomponents outside the circuit at positions from the first edited one to the last target, ranked by
               the measured change of their output under the edit, |a_c(model less R) - a_c(model)| x |u_c| (the real
               model's runs; the gradients play no part), with the share of the summed change the top 10, 100 and 1,000 hold
  smallest     the fewest of them, in that order, whose addition to the circuit puts the mediated share (frozen remainders,
               circuit_scorecard.mediated_vpd) within 0.1 of 1, measured on a log grid then halving; the share at each size
               measured
A small set (tens) means a circuit can carry a short responder list per edited position and stay small.

  responders.py --out OUT.jsonl [--tasks text9000,...] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
sys.path.insert(0, str(HERE.parents[1] / "method" / "vpd"))

import circuit_scorecard as cs  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402
import native_units  # noqa: E402

G = Path.home() / "mpd-data/graph_oracle"
TASKS = ("text9000", "text9003", "text9004", "text9006", "text9008", "text9009", "text9010")
VPD_FILE = Path.home() / "mpd-data/scratch/generality/scorecard/vpd_dev50.pt"
TOL = 0.1  # the mediated share's distance from 1 that counts as carried


def vpd_circuit(task: str) -> Path:
    return G / "teacher/stated/text9000.json" if task == "text9000" else G / f"teacher/dev_6a9e1d3/{task}.json"


@torch.no_grad()
def acts_less(nat: native.Native, ids: list[int], removed: list) -> dict:
    """Every subcomponent's activation [T, C] in the real model (remainders live) with the removed set at 0."""
    T = len(ids)
    m = {n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, t, c in removed:
        m[n][t, c] = False
    rec = {}  # (a complete graph of every subcomponent but the removed: those at u = 0)
    nat.run(list(ids), [T - 1], nat._plan([native.Graph(masks=m)], T), {n: torch.zeros(1, 1, nat.C[n], device=nat.dev) for n in nat.names},
            nat._one()[1], record=rec)
    return {n: rec[n][0].float() for n in nat.names}


def smallest(ge, tx, nodes: list, R: list, order: list) -> dict:
    """The fewest of order (responders, best first) whose addition puts R's mediated share within TOL of 1."""
    def share(k):
        return cs.mediated_vpd(ge, tx, sorted(set(nodes) | set(order[:k])), [R])[0]["share"]
    grid = native_units.log_sizes(len(order))
    tried = {}
    for k in [0] + grid:
        tried[k] = share(k)
        if tried[k] is not None and abs(tried[k] - 1) <= TOL:
            break
    ok = [k for k, s in tried.items() if s is not None and abs(s - 1) <= TOL]
    if not ok:
        return {"k": None, "tried": sorted(tried.items())}
    k = min(ok)
    lo = max([j for j in tried if j < k] or [0])
    while k - lo > max(1, k // 16):
        mid = (lo + k) // 2
        tried[mid] = share(mid)
        lo, k = (lo, mid) if tried[mid] is not None and abs(tried[mid] - 1) <= TOL else (mid, k)
    return {"k": k, "share": tried[k], "tried": sorted(tried.items())}


def one(ge: goal_eval.GoalEval, task: str, vpd: dict, log) -> dict:
    nat = ge.nat
    t0 = time.perf_counter()
    tp = native.TEXTS / "vpd4l" / f"{task}.json"
    tx = lines.text_of(ge, tp, vpd)
    g, _, _ = goal_eval.load_answer(vpd_circuit(task), json.loads(tp.read_text()), tx.targets, nat)
    nodes = sorted(goal_eval._c_graph(native.Graph(sorted(g.node_set()), complete=True), tx.targets, nat.C, tx.vpd).node_set())
    single = cs.edit_runs(ge, tx, native.Graph(nodes, complete=True), {}, [[nd] for nd in nodes], model_only=True)["model"]
    top = [nodes[i] for i in np.argsort([single[0] - x for x in single[1:]])[::-1][:5]]
    clean = acts_less(nat, tx.ids, [])
    norms = {n: nat.target.site(n).U.norm(dim=1) for n in nat.names}
    inside = set(nodes)
    out = {"task": task, "vpd_units": len(nodes), "sets": {}}
    for k in (1, 3, 5):
        R = top[:k]
        p0 = min(t for _, t, _ in R)
        moved = acts_less(nat, tx.ids, R)
        cand, score = [], []
        for n in nat.names:
            d = ((moved[n] - clean[n]).abs() * norms[n][None])[p0:max(tx.targets) + 1]  # [positions, C]
            idx = d.flatten().argsort(descending=True)[:20000]
            for i in idx.tolist():
                t, c = p0 + i // d.shape[1], i % d.shape[1]
                if (n, t, c) not in inside and (n, t, c) not in R:
                    cand.append((n, t, c))
                    score.append(float(d.flatten()[i]))
        order = [cand[i] for i in np.argsort(score)[::-1]]
        total = sum(score)
        srt = sorted(score, reverse=True)
        conc = {f"top{m}": sum(srt[:m]) / total for m in (10, 100, 1000)} if total > 0 else {}
        base = cs.mediated_vpd(ge, tx, nodes, [R])[0]
        res = smallest(ge, tx, nodes, R, order)
        out["sets"][f"top{k}"] = {"removed": [cs.label(nd) for nd in R], "share": base["share"], "live": base["live"],
                                  "responders_ranked": len(order), "concentration": conc, "smallest": res,
                                  "first": [cs.label(nd) for nd in order[:20]]}
        log(f"  {task} top{k}: share {base['share']}, smallest responder set {res['k']} of {len(order)}, {time.perf_counter() - t0:.0f} s")
    out["seconds"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(a.device), chunk=32)
    vpd = torch.load(VPD_FILE)
    done = {json.loads(x)["task"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    for task in a.tasks.split(","):
        if task in done:
            continue
        rec = one(ge, task, vpd, lambda m: print(m, flush=True))
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
