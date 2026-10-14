"""Do a circuit's own nodes take the model's values? (the lead's 10-13 diagnostic; no reward change.)

goal_eval._node_values (fidelity's reads columns node_share, node_share_weighted: the scorecard's metric). Under the
interventions fidelity already runs (goal_eval's node-removal and interchange families, the reward's
semantics), each circuit node's activation in the circuit's run is compared with its activation in the model's run on
the same intervention (the families' read pairs, reads=True). Per node, z = RMS(circuit - model) / SD(model), the SD
over the interventions: the node's own spread across the changed texts, not a fixed cutoff (the activation's scale,
|u| of its write, cancels in the ratio, so z is gauge-invariant). Per circuit:
  share     the share of nodes with z <= 1 (the circuit's error within the node's natural variation)
  weighted  the same share weighted by each node's measured downstream effect (goal_eval.line_effects in pp: the KL the
            circuit gains with that node alone removed, floored at 0)
  median_z  the nodes' median z
Sets:
  toys       the planted toys' true circuits against the other mechanism's (run_planted.circuits; cancel: the true
             circuit without the cancelling pair, against with one neuron and with both)
  causalgym  generality's CausalGym teacher circuits against the same circuit with the subject's nodes moved to the
             attractor (cg_teacher.test's pair)
  dev        dev teacher circuits against random subsets of their start of the same size (3 seeds)
  reference  per dev text the teacher, VPD's actives at the targets and the START's candidate positions (v24's
             own-run families' reference, goal_eval.vpd_floors), VPD's whole answer (the share alone) and, where the START
             (the teacher's path.json START record, else goal_eval.start_set) is larger than the teacher, random subsets of
             it of the teacher's size (3 seeds):
             whether a node-values floor relative to VPD would be fair (the lead's 10-13 question)

  python node_values.py toys --toys ~/mpd-data/toys_planted --out OUT.jsonl
  python node_values.py causalgym --teacher ~/mpd-data/graph_oracle/generality/cg_teacher_b2 --out OUT.jsonl
  python node_values.py dev --teacher ~/mpd-data/graph_oracle/teacher/fv_dev --n 20 --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "method").exists() else Path.cwd()
sys.path[:0] = [str(SRC / "method/oracle"), str(SRC / "method/vpd"), str(SRC / "experiments/oracle/toys"),
                str(SRC / "experiments/oracle/generality")]

import goal_eval  # noqa: E402
import native  # noqa: E402

DRAWS = 16


def agreement(ge, tx, graphs: list, split: str = "eval", draws: int = DRAWS, effects: list | None = None) -> list[dict]:
    """Per circuit, share, weighted and median_z (module docstring) over its nodes with two or more interventions.
    effects: per circuit whether to measure its line effects for the weighted share (default all; VPD's whole answer,
    thousands of nodes, is shared alone)."""
    rn = ge._node_family(tx, graphs, draws, reads=True)
    ri = ge._interchange_family(tx, graphs, draws, split=split, reads=True)
    out = []
    for k, (g, a, b) in enumerate(zip(graphs, rn, ri)):
        pairs = {}
        for src in (a.get("_pairs_nodes") or {}, b.get("_pairs_interchange") or {}):
            for nd, xs in src.items():
                pairs.setdefault(tuple(nd), []).extend(xs)
        eff = {tuple(x["node"]): max(0.0, x["delta_kl"] or 0.0) for x in ge.line_effects(tx, g, mode="pp", split=split)["nodes"]} \
            if effects is None or effects[k] else {}
        r = goal_eval._node_values(pairs, eff)
        out.append({"nodes": g.count(), "measured": r["node_measured"], "share": r["node_share"],
                    "weighted": r["node_share_weighted"], "median_z": r["node_median_z"]})
    return out


def toys(a) -> None:
    import torch

    import planted
    import run_planted
    import vpd_toy

    with a.out.open("a") as fh:
        for name in a.only.split(","):
            d = torch.load(a.toys / f"{name}.pt")
            nat = planted.native_of(d["sd"], d["cfg"])
            ge = run_planted.ToyEval(nat, chunk=16)
            far = [(f"far{i}", x) for i, x in enumerate(next(vpd_toy.loader(name, 16, 99)).tolist())]
            ge.donor_pools.update(search=far[:8], eval=far[8:])
            for i, ids in enumerate(list(dict.fromkeys(tuple(x) for x in planted.ordinary(name, a.texts)))):
                tx = ge.text(list(ids), [len(ids) - 1])
                cs = run_planted.circuits(name, nat, tx)
                got = agreement(ge, tx, [native.Graph(v, complete=True) for v in cs.values()])
                for arm, r in zip(cs, got):
                    fh.write(json.dumps({"set": "toys", "toy": name, "text": i, "circuit": arm, **r}) + "\n")
                    print(name, i, arm, json.dumps(r), flush=True)


def causalgym(a) -> None:
    import torch

    data = Path.home() / "mpd-data/graph_oracle/generality"
    meta = json.loads((data / f"causalgym_texts{a.b}.json").read_text())
    ge = goal_eval.GoalEval(native.Native(a.device))
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    with a.out.open("a") as fh:
        for nm in sorted(meta)[:a.n]:
            if not (a.teacher / f"{nm}.json").exists():
                continue
            m = meta[nm]
            tj = json.loads((data / f"causalgym_texts{a.b}" / f"{nm}.json").read_text())
            ids = tj["prompts"][0]["token_ids"]
            t, p, pw = len(ids) - 1, m["p"], m["attractor"]
            tx = ge.text(ids, [t], native.task_seed(nm), task=tj)
            g, _ = goal_eval.circuit(json.loads((a.teacher / f"{nm}.json").read_text()), tx.targets, nat.C, tx.T)
            nodes = sorted(g.node_set())
            if pw in (None, p, t) or p not in {q for _, q, _ in nodes}:
                continue
            moved = sorted({(n, pw if q == p else q, c) for n, q, c in nodes})
            got = agreement(ge, tx, [native.Graph(nodes, complete=True), native.Graph(moved, complete=True)])
            for arm, r in zip(("subject", "attractor"), got):
                fh.write(json.dumps({"set": "causalgym", "text": nm, "task": m["task"], "circuit": arm, **r}) + "\n")
                print(nm, arm, json.dumps(r), flush=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


def dev(a) -> None:
    ge = goal_eval.GoalEval(native.Native(a.device))
    nat = ge.nat
    with a.out.open("a") as fh:
        for k in range(a.first, a.first + a.n):
            task = f"text{k}"
            f = a.teacher / f"{task}.json"
            if not f.exists():
                continue
            ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
            tx = ge.text(ids, targets, native.task_seed(task))
            g, _ = goal_eval.circuit(json.loads(f.read_text()), targets, nat.C, tx.T)
            nodes = sorted(g.node_set())
            start = sorted(ge.start_set(tx, "search")["graph"].node_set())
            graphs = {"teacher": native.Graph(nodes, complete=True)}
            if len(start) > len(nodes):
                for s in range(3):
                    graphs[f"start_subset{s}"] = native.Graph(sorted(random.Random(s).sample(start, len(nodes))), complete=True)
            got = agreement(ge, tx, list(graphs.values()))
            for arm, r in zip(graphs, got):
                fh.write(json.dumps({"set": "dev", "text": task, "circuit": arm, "start": len(start), **r}) + "\n")
                print(task, arm, json.dumps(r), flush=True)


def reference(a) -> None:
    ge = goal_eval.GoalEval(native.Native(a.device))
    nat = ge.nat
    with a.out.open("a") as fh:
        for k in range(a.first, a.first + a.n):
            task = f"text{k}"
            f = a.teacher / f"{task}.json"
            if not f.exists():
                continue
            ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
            tx = ge.text(ids, targets, native.task_seed(task))
            g, _ = goal_eval.circuit(json.loads(f.read_text()), targets, nat.C, tx.T)
            nodes = sorted(g.node_set())
            act = sorted((n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist())
            pf = a.teacher / f"{task}.path.json"  # the teacher's own START record (v24): its candidates and nodes
            rec = json.loads(pf.read_text()).get("blocks", {}) if pf.exists() else {}
            if rec.get("start", {}).get("candidates") is not None and rec.get("stage_nodes", {}).get("start"):
                ps = set(targets) | set(rec["start"]["candidates"])
                start = sorted((n, int(t), int(c)) for n, t, c in rec["stage_nodes"]["start"])
            else:
                ps = set(targets) | set(ge.start_candidates(tx)[:goal_eval.START_CAP])
                start = sorted(ge.start_set(tx, "search")["graph"].node_set())
            graphs = {"teacher": native.Graph(nodes, complete=True),
                      "vpd_at_candidates": native.Graph([nd for nd in act if nd[1] in ps], complete=True),
                      "start": native.Graph(start, complete=True), "vpd": native.Graph(act, complete=True)}
            if len(start) > len(nodes):
                for s in range(3):
                    graphs[f"start_subset{s}"] = native.Graph(sorted(random.Random(s).sample(start, len(nodes))), complete=True)
            got = agreement(ge, tx, list(graphs.values()), effects=[k in a.weighted.split(",") for k in graphs])
            for arm, r in zip(graphs, got):
                fh.write(json.dumps({"set": "reference", "text": task, "circuit": arm, **r}) + "\n")
                print(task, arm, json.dumps(r), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toys")
    t.add_argument("--toys", type=Path, required=True)
    t.add_argument("--only", default="binding_km,binding_fp,binding_mixed,route2,interaction,cancel")
    t.add_argument("--texts", type=int, default=3)
    c = sub.add_parser("causalgym")
    c.add_argument("--teacher", type=Path, required=True)
    c.add_argument("--b", default="_b", help="the texts' set suffix (causalgym_texts<b>.json)")
    c.add_argument("--n", type=int, default=40)
    d = sub.add_parser("dev")
    d.add_argument("--teacher", type=Path, required=True)
    d.add_argument("--first", type=int, default=9000)
    d.add_argument("--n", type=int, default=20)
    r = sub.add_parser("reference")
    r.add_argument("--teacher", type=Path, required=True)
    r.add_argument("--first", type=int, default=9000)
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--weighted", default="teacher,start", help="the circuits whose line effects weigh the share (the rest: the "
                   "share alone; v24 teachers hold 600-2,200 nodes)")
    for p in (t, c, d, r):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default=None)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    {"toys": toys, "causalgym": causalgym, "dev": dev, "reference": reference}[a.cmd](a)


if __name__ == "__main__":
    main()
