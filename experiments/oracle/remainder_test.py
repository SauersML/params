"""Where VPD circuits' mediated share departs from 1 (#2951, the lead's 10-14): the share for the VPD circuit's top 1, 3
and 5 subcomponents by real-model drop (circuit_scorecard.mediated_vpd's sets), three ways, everything else the same:
  frozen   as the scorecard: in the mediated run every subcomponent outside the circuit and every remainder at its value
           on the unedited text
  live     the remainders recomputed in the mediated run (the outside subcomponents still at their values)
  removed  no remainder in either run: the model is VPD's subcomponents alone (every one on, the remainders off), the
           live drop in it, the mediated run's outside subcomponents at their values in it
A live share near 1 everywhere puts the gap in VPD's remainders, not in self-repair or the circuit's size.

  remainder_test.py --out OUT.jsonl [--tasks text9000,...] [--device cpu]
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

G = Path.home() / "mpd-data/graph_oracle"
TASKS = ("text9000", "text9003", "text9004", "text9006", "text9008", "text9009", "text9010")
VPD_FILE = Path.home() / "mpd-data/scratch/generality/scorecard/vpd_dev50.pt"


def vpd_circuit(task: str) -> Path:
    return G / "teacher/stated/text9000.json" if task == "text9000" else G / f"teacher/dev_6a9e1d3/{task}.json"


@torch.no_grad()
def shares(ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, tops: list[list]) -> dict:
    """{frozen, live, removed}: per removal set its live drop, mediated drop and share (module docstring)."""
    import engine

    nat, T, dev = ge.nat, tx.T, ge.nat.dev
    y = tx.ref[0].argmax(-1)

    def lp(lq):
        return float(lq[0].float().gather(-1, y.view(-1, 1)).sum() / cs.LN2)
    acts = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=None).x_values()["acts"]  # the real model's, [1, T, C]
    rec = {}  # the subcomponents' activations in the model without remainders
    nat.run(list(tx.ids), [T - 1], nat._full_plan(T, 1), nat._one()[0], {n: torch.zeros(1, 1, device=dev) for n in nat.names}, record=rec)
    acts0 = {n: rec[n][:1].float() for n in nat.names}
    ones = {n: torch.ones(T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
    circ = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
    for n, t, c in nodes:
        circ[n][t, c] = True
    rest = {}
    base = lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=ones), rest=rest))
    base0 = lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=ones), live=False))
    out = {"frozen": [], "live": [], "removed": [], "lp": base, "lp_no_remainder": base0}
    for R in tops:
        live, med = {n: m.clone() for n, m in ones.items()}, {n: m.clone() for n, m in circ.items()}
        held, held0 = {n: a.clone() for n, a in acts.items()}, {n: a.clone() for n, a in acts0.items()}
        for n, t, c in R:
            live[n][t, c] = med[n][t, c] = False
            held[n][0, t, c] = held0[n][0, t, c] = 0.0
        d_live = base - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=live)))
        d_live0 = base0 - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=live), live=False))
        runs = {"frozen": (d_live, base - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=med), outside=held, hold=True, rest=rest))),
                "live": (d_live, base - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=med), outside=held))),
                "removed": (d_live0, base0 - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=med), outside=held0, live=False)))}
        for k, (dl, dm) in runs.items():
            out[k].append({"k": len(R), "live": dl, "mediated": dm, "share": dm / dl if abs(dl) >= cs.SHARE_MIN else None})
        native.free(dev)
    return out


def one(ge: goal_eval.GoalEval, task: str, vpd: dict) -> dict:
    nat = ge.nat
    t0 = time.perf_counter()
    tp = native.TEXTS / "vpd4l" / f"{task}.json"
    tx = lines.text_of(ge, tp, vpd)
    g, _, _ = goal_eval.load_answer(vpd_circuit(task), json.loads(tp.read_text()), tx.targets, nat)
    nodes = sorted(goal_eval._c_graph(native.Graph(sorted(g.node_set()), complete=True), tx.targets, nat.C, tx.vpd).node_set())
    single = cs.edit_runs(ge, tx, native.Graph(nodes, complete=True), {}, [[nd] for nd in nodes], model_only=True)["model"]
    top = [nodes[i] for i in np.argsort([single[0] - x for x in single[1:]])[::-1][:5]]
    out = {"task": task, "vpd_units": len(nodes), "top5": [cs.label(nd) for nd in top], **shares(ge, tx, nodes, [top[:1], top[:3], top[:5]])}
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
        rec = one(ge, task, vpd)
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        sh = lambda xs: " ".join("-" if x["share"] is None else f"{x['share']:.2f}" for x in xs)  # noqa: E731
        print(f"{task} ({rec['vpd_units']} parts): frozen {sh(rec['frozen'])} | live {sh(rec['live'])} | removed {sh(rec['removed'])} | "
              f"top-3 live drop {rec['frozen'][1]['live']:.2f} bits, {rec['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
