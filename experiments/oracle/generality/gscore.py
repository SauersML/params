"""Score circuits on one text with goal_eval's families: point (s_point_c = min(graph F1, every-connection F1)),
s_track (changed inputs, eval split), s_interchange (eval split), plus N and the node family when asked."""
import json
import os
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE", str(Path(__file__).resolve().parents[3] / "method" / "oracle")))

import goal_eval  # noqa: E402
import native  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"


def text(ge, task: str):
    p = TEXTS / f"{task}.json"
    t = json.loads(p.read_text())
    ids, targets = native.text(p)
    return ge.text(ids, targets, native.task_seed(task), t)


def graph(answer: dict, tx, C) -> tuple:
    return goal_eval.circuit(answer, tx.targets, C, tx.T)


def score(ge, tx, graphs: list, draws: int = 16) -> list[dict]:
    """Per graph: s_point (graph F1), s_wiring, s_point_c, n_live, s_track, s_changed, s_interchange, nodes, the three-term
    geometric mean c3 over those defined."""
    nat = ge.nat
    graphs = [goal_eval.strand(g, tx.targets, nat.C) for g in graphs]
    e0 = ge._bounds(tx, "live")[0]
    point = ge.sufficiency(tx, graphs)
    whole_idx = [i for i, g in enumerate(graphs) if not g.complete]
    wired = dict(zip(whole_idx, ge.sufficiency(tx, [native.Graph(sorted(graphs[i].node_set()), complete=True) for i in whole_idx])))
    chg = ge.changed_inputs(tx, graphs, "eval", draws)
    inter = ge._interchange_family(tx, graphs, draws, split="eval")
    out = []
    for i, (g, pt, ch, it) in enumerate(zip(graphs, point, chg, inter)):
        s_point = pt["f1_live"]
        s_wiring = wired[i]["f1_live"] if i in wired else s_point
        s_point_c = min(x for x in (s_point, s_wiring) if x is not None) if s_point is not None else None
        s_int = goal_eval._fid(e0, it["kl_interchange_c"], it["kl_interchange_vpd"]) if it["kl_interchange_c"] is not None else None
        row = {"nodes": g.count(), "stranded": len(getattr(g, "stranded", [])), "edges": None if g.complete else g.edges(),
               "positions": len({t for _, t, _ in g.node_set()}),
               "kl_point": pt["kl_live"], "s_point": s_point, "s_wiring": s_wiring, "s_point_c": s_point_c, "n_live": pt["n_live"],
               "s_track": ch["s_track"], "s_changed": ch["s_changed"], "prompts_track": ch["prompts_track"],
               "context_share": ch["context_share"], "s_interchange": s_int,
               "kl_interchange_c": it["kl_interchange_c"], "kl_interchange_vpd": it["kl_interchange_vpd"]}
        terms = [row[k] for k in ("s_point_c", "s_track", "s_interchange") if row[k] is not None]
        row["c3"] = math.prod(terms) ** (1 / len(terms)) if terms else None
        out.append(row)
    return out


if __name__ == "__main__":
    task = sys.argv[1]
    ge = goal_eval.GoalEval(native.Native())
    t0 = time.perf_counter()
    tx = text(ge, task)
    t1 = time.perf_counter()
    own = json.loads((Path.home() / "mpd-data/graph_oracle/teacher/joint_train" / f"{task}.json").read_text())
    g, d = graph(own, tx, ge.nat.C)
    base = ge.baselines(tx, [g.count()])
    gs = [g, base[f"vpd@{g.count()}"]]
    rows = score(ge, tx, gs)
    t2 = time.perf_counter()
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("defects", d, "text", round(t1 - t0, 1), "s, score", round(t2 - t1, 1), "s")
