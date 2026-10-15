"""connect.py's variants compared at equal cost (the lead's 10-14): per text, one process so the families' draws are
shared (a warm run first), each arm a fresh Connector: connections and lines kept, the sets measured, seconds per stage,
the families, and the sparsity measures. An arm is SCORE:PASSES[:FLAG...], flags halve (the halving pass), full (every
family in the passes, not the cheap ones with a final check), batch8 (families asked 8 sets at a time from the start).

  python experiments/oracle/connect_compare.py TEACHERS OUT.jsonl TASK... --vpd FILE --arms ig:8,fisher:8,ig:8:halve
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import connect  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("teachers", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("tasks", nargs="+")
ap.add_argument("--vpd", type=Path, required=True)
ap.add_argument("--arms", default="ig:8,fisher:8,ig:32,fisher:32,ig:8:halve")
ap.add_argument("--device")
a = ap.parse_args()
vpd = torch.load(a.vpd)
ge = goal_eval.GoalEval(native.Native(a.device), chunk=32)
arms = [x.split(":") for x in a.arms.split(",")]
for t in a.tasks:
    task = native.TEXTS / "vpd4l" / f"{t}.json"
    tx = lines.text_of(ge, task, vpd)
    g0, _, _ = goal_eval.load_answer(a.teachers / f"{t}.json", json.loads(task.read_text()), tx.targets, ge.nat)
    nodes = sorted(g0.node_set())
    connect.Connector(ge, tx, nodes).run(log=lambda m: None, passes=2)  # (warm: the families' draws)
    for arm in arms:
        t0 = time.time()
        flags = set(arm[2:])
        rec = connect.Connector(ge, tx, nodes).run(log=lambda m: None, score=arm[0], passes=int(arm[1]), halve="halve" in flags,
                                                   cheap="full" not in flags, first=8 if "batch8" in flags else 1)["record"]
        row = {"task": t, "arm": ":".join(arm), **{k: rec[k] for k in ("parts", "parts_start", "lines", "lines_start",
               "connections", "token_edges", "point", "families", "criteria", "measured", "seconds", "halving",
               "edges_per_unit", "cross_token", "tokens", "wired", "backed_to")}, "wall": round(time.time() - t0, 2)}
        print(json.dumps(row), flush=True)
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        native.free(ge.nat.dev)
