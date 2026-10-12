"""Does the copy texts' shared context core carry over? The context core (copy_heldout's selection, measured on the 23
copy texts with teacher circuits): at after_prev (the copied token's position) h.1.mlp.c_fc#2103, h.1.attn.q_proj#316,
h.0.mlp.c_fc#53, h.2.attn.k_proj#206; at prev_occ (the earlier occurrence of the current token) h.1.attn.k_proj#329
(its removal costs bits there, not at after_prev: copy_heldout). Each set is a complete graph (every connection the
model has among its nodes), scored by gscore (point, changed inputs, interchange; eval split).

Held-out copy texts (copy_heldout.json, no teacher circuit, never used in the selection): VPD's top k (k = 16, 32, 64)
with the context core added, against VPD's top k + 5; the copy family's target core (the subcomponents more than
half of the 23 teacher circuits list at the target) with and without the context core; the context core alone.
Copy texts with teacher circuits (in-sample for the selection, reported apart): the teacher circuit's node set with and
without the context core, and VPD's top k at both sizes."""
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
import transfer  # noqa: E402

CTX = [("h.1.mlp.c_fc", 2103, "after_prev"), ("h.1.attn.q_proj", 316, "after_prev"), ("h.0.mlp.c_fc", 53, "after_prev"),
       ("h.2.attn.k_proj", 206, "after_prev"), ("h.1.attn.k_proj", 329, "prev_occ")]
KS = (16, 32, 64)


def roles_of(ids, t):
    occ = [j for j in range(t) if ids[j] == ids[t]]
    return {"cur": t, "prev_occ": occ[-1], "after_prev": occ[-1] + 1}


def ctx_nodes(r):
    return [(m, r[rl], c) for m, c, rl in CTX]


def main():
    task = sys.argv[1]
    out = Path(os.environ.get("AUG_OUT", DATA / "copy_augment.jsonl"))
    rows = {r["task"]: r for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
    copy23 = [x["task"] for x in json.load(open(DATA / "copytest.json")) if x["drop_nats"] >= 1 and x["new_gain_nats"] >= 1]
    ge = goal_eval.GoalEval(native.Native())
    C = ge.nat.C
    tx = gscore.text(ge, task)
    t = tx.targets[-1]
    r = roles_of(tx.ids, t)
    ctx = ctx_nodes(r)
    graphs, meta = [], []

    def add(nodes, kind, extra=None):
        graphs.append(native.Graph(sorted(set(nodes)), complete=True))
        meta.append({"kind": kind, **(extra or {})})

    order = ge.nat._vpd_order(tx.ids, tx.targets)
    if task in copy23:
        own, _ = goal_eval.circuit(transfer.answer(rows[task]), tx.targets, C, tx.T)
        graphs.append(own)
        meta.append({"kind": "own_stated"})
        raw = [(m, p, c) for m, c, p in transfer.answer(rows[task])["nodes"] if m != "tok"]
        add(raw, "own_nodes")
        add(raw + ctx, "own_nodes+ctx")
        for k in (len(set(raw)), len(set(raw + ctx))):
            add(order[:k], f"vpd@{k}", {"k": k})
    else:
        tcore, _, need = transfer.core([rows[x] for x in copy23])
        tnodes = [(m, t, c) for m, c, rl in tcore if rl == "T"]
        add(ctx, "ctx")
        add(tnodes, "tcore", {"core": len(tnodes)})
        add(tnodes + ctx, "tcore+ctx")
        for k in KS:
            add(order[:k], f"vpd@{k}", {"k": k})
            add(order[:k] + ctx, f"vpd@{k}+ctx", {"k": k})
            add(order[:k + len(ctx)], f"vpd@{k + len(ctx)}", {"k": k + len(ctx)})
        for k in (len(tnodes), len(set(tnodes + ctx))):
            add(order[:k], f"vpd@{k}", {"k": k})
    scored = []
    for s0 in range(0, len(graphs), 4):
        scored += gscore.score(ge, tx, graphs[s0:s0 + 4])
        native.free(ge.nat.dev)
    with out.open("a") as f:
        for m, s in zip(meta, scored):
            f.write(json.dumps({"task": task, "in_selection": task in copy23, **m,
                                **{k: (float(v) if hasattr(v, "item") else v) for k, v in s.items()}}) + "\n")
    print(task, "done", len(graphs), flush=True)


if __name__ == "__main__":
    main()
