"""Size-matched mediated share (#2951, the lead's 10-14): native circuits carry about all of each top edit's effect
(mediated share about 1) and VPD circuits about 0.3, but the native ones hold 12k-67k units against VPD's 400-1,400.
Per text, the same question at matched sizes:
  native at VPD's size   the native circuit pruned to the VPD circuit's unit count N (prune_native: the Fisher order
                         at the model, then twice re-ranked at the circuit, 4N -> 2N -> N), its KL and its mediated
                         share for its own top 1, 3 and 5 units by measured drop, and for the full native circuit's
                         top sets (the same edits)
  VPD grown              the VPD circuit with VPD's actives at every position the native circuit holds (native-like
                         coverage), and VPD's whole answer: their mediated share for the VPD circuit's own top sets
                         (circuit_scorecard.mediated_vpd), beside the VPD circuit's
  native full            the native circuit's (its file's)

  size_matched.py --out OUT.jsonl [--tasks text9000,...] [--device cpu]
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
TASKS = ("text9000", "text9003", "text9004", "text9008", "text9009", "text9010")
VPD_FILE = Path.home() / "mpd-data/scratch/generality/scorecard/vpd_dev50.pt"
NATIVE = G / "native_fast/c489c62"


def vpd_circuit(task: str) -> Path:
    """The VPD circuit the scorecard scored on the text (text9000: rustmove's stated circuit; else the dev teacher)."""
    return G / "teacher/stated/text9000.json" if task == "text9000" else G / f"teacher/dev_6a9e1d3/{task}.json"


def prune_native(mu: native_units.MeanUnits, seed: int, n: int) -> torch.Tensor:
    """The n units the Fisher keeps: its order at the model to 4n, then at that circuit to 2n, then at that one to n."""
    L, T, H, Dm = mu.L, mu.T, mu.H, mu.Dm
    nh = L * T * H

    def mask_of(idx):
        h = torch.zeros(nh, device=mu.dev)
        m = torch.zeros(L * T * Dm, device=mu.dev)
        h[idx[idx < nh]], m[idx[idx >= nh] - nh] = 1.0, 1.0
        return h.view(L, T, H), m.view(L, T, Dm)
    keep = torch.argsort(native_units.fisher(mu, seed), descending=True)[:4 * n]
    for k in (2 * n, n):
        f = native_units.fisher(mu, seed, live=mask_of(keep))
        keep = keep[torch.argsort(f[keep], descending=True)][:k]
    return keep, mask_of(keep)


def one(ge: goal_eval.GoalEval, task: str, vpd: dict, means: dict, log) -> dict:
    nat = ge.nat
    t0 = time.perf_counter()
    tp = native.TEXTS / "vpd4l" / f"{task}.json"
    tx = lines.text_of(ge, tp, vpd)
    g, _, _ = goal_eval.load_answer(vpd_circuit(task), json.loads(tp.read_text()), tx.targets, nat)
    nodes0 = sorted(goal_eval._c_graph(native.Graph(sorted(g.node_set()), complete=True), tx.targets, nat.C, tx.vpd).node_set())
    n = len(nodes0)
    single = cs.edit_runs(ge, tx, native.Graph(nodes0, complete=True), {}, [[nd] for nd in nodes0], model_only=True)["model"]
    top = [nodes0[i] for i in np.argsort([single[0] - x for x in single[1:]])[::-1][:5]]
    tops = [top[:1], top[:3], top[:5]]
    nrec = json.loads((NATIVE / f"{task}.json").read_text())
    P = set(nrec["record"]["positions"])
    act = sorted((m, t, c) for m in nat.names for t, c in tx.vpd[m].nonzero().tolist())
    grown = sorted(set(nodes0) | {nd for nd in act if nd[1] in P})
    out = {"task": task, "vpd_units": n, "native_units": len(nrec["nodes"]), "native_positions": len(P),
           "vpd_grown_units": len(grown), "vpd_whole_units": len(act),
           "vpd": cs.mediated_vpd(ge, tx, nodes0, tops), "vpd_grown": cs.mediated_vpd(ge, tx, grown, tops),
           "vpd_whole": cs.mediated_vpd(ge, tx, act, tops), "native_full": {k: nrec["mediated"][k] for k in ("top1", "top3", "top5")}}
    log(f"  {task}: VPD {n} units, grown {len(grown)}, whole {len(act)}, {time.perf_counter() - t0:.0f} s")
    mu = native_units.MeanUnits(nat, tx.ids, tx.targets, means)
    _, live = prune_native(mu, tx.seed, n)
    S = native_units.Mask(*(torch.zeros(mu.L, len(tx.ids), x.shape[2], dtype=torch.bool, device=mu.dev) for x in live))
    S.heads[:, :mu.T], S.neurons[:, :mu.T] = live[0].bool(), live[1].bool()
    out["native_at_vpd_size"] = {"units": S.count(), "kl": mu.kl([live])[0], "kl_target": nrec["record"]["kl_target"],
                                 "positions": int((live[0].sum((0, 2)) + live[1].sum((0, 2)) > 0).sum()),
                                 "own": native_units.mediated_share(ge, tx, S, means, candidates=n),
                                 "full_sets": native_units.mediated_share(ge, tx, S, means, sets=[nrec["mediated"][k]["units"] for k in ("top1", "top3", "top5")])}
    out["seconds"] = time.perf_counter() - t0
    log(f"  {task}: native at {n} units KL {out['native_at_vpd_size']['kl']:.3f} (target {out['native_at_vpd_size']['kl_target']:.3f}), "
        f"{out['seconds']:.0f} s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    nat = native.Native(a.device)
    ge = goal_eval.GoalEval(nat, chunk=32)
    means = native_units.unit_means(nat)
    vpd = torch.load(VPD_FILE)
    done = {json.loads(x)["task"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    for task in a.tasks.split(","):
        if task in done:
            continue
        rec = one(ge, task, vpd, means, lambda m: print(m, flush=True))
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        sh = lambda d: " ".join("-" if x.get("share") is None else f"{x['share']:.2f}" for x in d)  # noqa: E731
        nm = rec["native_at_vpd_size"]
        print(f"{task}: VPD {sh(rec['vpd'])} | grown {sh(rec['vpd_grown'])} | whole {sh(rec['vpd_whole'])} | native full "
              f"{sh([rec['native_full'][k] for k in ('top1', 'top3', 'top5')])} | native at VPD size own "
              f"{sh([nm['own'][k] for k in ('top1', 'top3', 'top5')])}, full sets {sh([nm['full_sets'][f'set{i}'] for i in range(3)])}", flush=True)


if __name__ == "__main__":
    main()
