"""What the copy texts' predictions need at the earlier occurrence, in the real model: per copy text, removing VPD's
active subcomponents of one weight matrix at one role position (after_prev: the position after the previous
occurrence of the current token, the token copied; prev_occ: that occurrence; T: the target), everything else on and
the remainders live (goal_eval.real_removal), KL at the target in bits; then single removals of every active
subcomponent at after_prev in the matrices whose removal there costs at least SITE_BITS."""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE", str(Path(__file__).resolve().parents[3] / "method" / "oracle")))
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
sys.path.insert(0, str(HERE))

import goal_eval  # noqa: E402
import gscore  # noqa: E402
import native  # noqa: E402

SITE_BITS = 0.5

rows = {r["task"]: r for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
copy = [x["task"] for x in json.load(open(DATA / "copytest.json")) if x["drop_nats"] >= 1 and x["new_gain_nats"] >= 1]
ge = goal_eval.GoalEval(native.Native(), chunk=16)
ge.nat.LENGTHS_KEPT = 1
out = {}
for task in copy:
    r = rows[task]
    tx = gscore.text(ge, task)
    pos = {"after_prev": r["roles"]["after_prev"], "prev_occ": r["roles"]["prev_occ"], "T": r["t"]}
    graphs, keys = [], []
    for rl, p in pos.items():
        allp = []
        for n in ge.nat.names:
            idx = tx.vpd[n][p].nonzero().flatten().tolist()
            nodes = [(n, p, c) for c in idx]
            allp += nodes
            if nodes:
                graphs.append(native.Graph(nodes, complete=True))
                keys.append((rl, n, len(nodes)))
        graphs.append(native.Graph(allp, complete=True))
        keys.append((rl, "all", len(allp)))
    kl = [x["kl_real_minus"] for x in ge.real_removal(tx, graphs)]
    sites = [{"role": k[0], "site": k[1], "active": k[2], "kl": v} for k, v in zip(keys, kl)]
    a = pos["after_prev"]
    big = [s["site"] for s in sites if s["role"] == "after_prev" and s["site"] != "all" and s["kl"] >= SITE_BITS]
    singles = []
    for n in big:
        idx = tx.vpd[n][a].nonzero().flatten().tolist()
        got = ge.real_removal(tx, [native.Graph([(n, a, c)], complete=True) for c in idx])
        singles += [{"site": n, "index": c, "kl": g["kl_real_minus"]} for c, g in zip(idx, got)]
    out[task] = {"cur": r["cur"], "pred": r["pred"], "positions": pos, "sites": sites, "singles": singles}
    top = sorted(sites, key=lambda s: -s["kl"])[:6]
    print(task, repr(r["cur"]), repr(r["pred"]), [(s["role"], s["site"].replace("h.", "").replace(".attn", "").replace(".mlp", ""), s["active"], round(s["kl"], 2)) for s in top], flush=True)
    json.dump(out, open(DATA / "copy_context.json", "w"))
    del tx
    ge.probes.clear()
    native.free(ge.nat.dev)
