"""The size-matched set as circuit files for the scorecard (generality's run): per text OUT/native/<task>.json, the native
circuit pruned to the VPD circuit's unit count (size_matched.prune_native), and OUT/vpd_grown/<task>.json, the VPD circuit
with VPD's actives at every position the full native circuit holds (a node set, goal_eval.to_json)."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402
import native_units  # noqa: E402
import size_matched as sm  # noqa: E402

torch.set_num_threads(6)
out = Path(sys.argv[1])
(out / "native").mkdir(parents=True, exist_ok=True)
(out / "vpd_grown").mkdir(parents=True, exist_ok=True)
nat = native.Native("cpu")
ge = goal_eval.GoalEval(nat, chunk=32)
means = native_units.unit_means(nat)
vpd = torch.load(sm.VPD_FILE)
for task in sm.TASKS:
    if (out / "native" / f"{task}.json").exists():
        continue
    tp = native.TEXTS / "vpd4l" / f"{task}.json"
    tx = lines.text_of(ge, tp, vpd)
    g, _, _ = goal_eval.load_answer(sm.vpd_circuit(task), json.loads(tp.read_text()), tx.targets, nat)
    nodes0 = sorted(goal_eval._c_graph(native.Graph(sorted(g.node_set()), complete=True), tx.targets, nat.C, tx.vpd).node_set())
    nrec = json.loads((sm.NATIVE / f"{task}.json").read_text())
    P = set(nrec["record"]["positions"])
    act = sorted((m, t, c) for m in nat.names for t, c in tx.vpd[m].nonzero().tolist())
    grown = sorted(set(nodes0) | {nd for nd in act if nd[1] in P})
    (out / "vpd_grown" / f"{task}.json").write_text(json.dumps({**goal_eval.to_json(native.Graph(grown, complete=True)), "units": "vpd",
                                                                 "builder": "size_matched VPD grown to the native circuit's positions"}))
    mu = native_units.MeanUnits(nat, tx.ids, tx.targets, means)
    _, live = sm.prune_native(mu, tx.seed, len(nodes0))
    units = [["head", l, t, j] for l, t, j in live[0].nonzero().tolist()] + [["neuron", l, t, j] for l, t, j in live[1].nonzero().tolist()]
    (out / "native" / f"{task}.json").write_text(json.dumps({"units": "native", "semantics": "mean_ablation", "nodes": units, "build_seconds": None,
                                                            "builder": f"size_matched native pruned to the VPD circuit's {len(nodes0)} units",
                                                            "kl": mu.kl([live])[0]}))
    print(task, len(nodes0), len(units), len(grown), flush=True)
