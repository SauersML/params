"""The VPD circuit's mediated share (frozen remainders, the scorecard's) with every subcomponent at the edited positions
added to the circuit (the positions of its top 5 parts, every site), beside the circuit's own: if VPD's gap from 1 comes
from subcomponents VPD marks inactive at the positions the edit reaches, this share is about 1."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import circuit_scorecard as cs  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402
import remainder_test as rt  # noqa: E402

torch.set_num_threads(4)
ge = goal_eval.GoalEval(native.Native("cpu"), chunk=32)
nat = ge.nat
vpd = torch.load(rt.VPD_FILE)
out = Path(sys.argv[1])
done = {json.loads(x)["task"] for x in out.read_text().splitlines()} if out.exists() else set()
for task in rt.TASKS:
    if task in done:
        continue
    tp = native.TEXTS / "vpd4l" / f"{task}.json"
    tx = lines.text_of(ge, tp, vpd)
    g, _, _ = goal_eval.load_answer(rt.vpd_circuit(task), json.loads(tp.read_text()), tx.targets, nat)
    nodes = sorted(goal_eval._c_graph(native.Graph(sorted(g.node_set()), complete=True), tx.targets, nat.C, tx.vpd).node_set())
    single = cs.edit_runs(ge, tx, native.Graph(nodes, complete=True), {}, [[nd] for nd in nodes], model_only=True)["model"]
    top = [nodes[i] for i in np.argsort([single[0] - x for x in single[1:]])[::-1][:5]]
    P = sorted({t for _, t, _ in top})
    full = sorted(set(nodes) | {(n, t, c) for n in nat.names for t in P for c in range(nat.C[n])})
    tops = [top[:1], top[:3], top[:5]]
    rec = {"task": task, "edited_positions": P, "circuit": cs.mediated_vpd(ge, tx, nodes, tops), "full_at_edit": cs.mediated_vpd(ge, tx, full, tops),
           "full_units": len(full)}
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    sh = lambda xs: " ".join("-" if x["share"] is None else f"{x['share']:.2f}" for x in xs)  # noqa: E731
    print(f"{task}: edited positions {P}; circuit {sh(rec['circuit'])} | every subcomponent at those positions added ({len(full)}) {sh(rec['full_at_edit'])}", flush=True)
