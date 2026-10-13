"""The copy mechanism (induction) as one stated template by role, scored on held-out copy texts (#2951 graph oracle).

Roles: t the target; o the latest earlier occurrence of the current token; a = o + 1, the token copied.
Measured (induction.py, controls.py): layer 1's read a -> o (head 1 on most texts) rests on one query-key pair on every
copy text, h.1.attn.q_proj#316 at a with h.1.attn.k_proj#329 at o (the previous-token step: a learns which token came
before it); h.0.mlp.c_fc#53 and h.1.mlp.c_fc#2103 at a are needed there; layer 2's read t -> a (head 4) matches the
current token at t against that record at a with token-specific queries and keys (blocks); layer 2's value at a
carries the copied token's identity to the attention output at t.

Templates (templates.py's form):
  induction: tok@o -> L1 k#329@o and L1 v[block]@o; L1 v@o -> L1 o[block]@a; tok@a -> L0 c_fc#53@a, L1 q#316@a,
     L1 c_fc#2103@a, L2 k[block]@a, L2 v[block]@a; c_fc#53 -> L0 down[block]@a; c_fc#2103 -> L1 down[block]@a; the
     residual writers at a (L0 down, L1 o, L1 down) -> its later residual readers; tok@t -> L2 q[block]@t; L2 v@a ->
     L2 o[block]@t -> out; pairs (q#316@a, k#329@o) in layer 1 at a, (L2 q@t, L2 k@a) in layer 2 at t.
  induction+target: also the target's layer-3 blocks (o, c_fc, down) and L2 o@t -> L3 c_fc@t -> L3 down@t -> out,
     L3 o@t -> out.
  each also with connections "all" (every connection the model has among its nodes, every token edge).
Baselines per text: VPD's top k at the template's listed size (a block one node) and at its expanded size (a block
every subcomponent of its site), the text's influential-blocks start (every site that reaches the target as a block
at the target and at the text's two most influential positions, Text.moved; complete), and the teacher circuit where
the text has one (in the copy selection).

  copy_template.py write                 -> MECH/copy_induction*.json
  copy_template.py score TASK            -> DATA/copy_template.jsonl (one text; resumable by the caller)
  copy_template.py predict [--draws N]   -> DATA/copy_predict.json: the template's predictions on changed inputs
  copy_template.py report
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

import templates  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
MECH = Path(os.environ.get("MECH", Path.home() / "mpd-data/graph_oracle/mechanisms"))
TEACH = Path.home() / "mpd-data/graph_oracle/teacher/joint_train"
NAMES = ("copy_induction_target", "copy_columns", "copy_columns_all")
ROLES = {"t": "the target", "o": "the latest earlier occurrence of the current token", "a": "o + 1, the token the prediction copies"}
EXPLAIN = ("Induction: at a, layer 1 reads the token before it (o) through query #316 and key #329; at the target, layer 2 "
           "matches the current token against that record at a and copies a's token through a's layer-2 value.")



def out_path(stem: str) -> Path:
    """An output file: GEN_OUT (default DATA) / stem + GEN_TAG (a pod worker's own suffix) .jsonl."""
    return Path(os.environ.get("GEN_OUT", DATA)) / f"{stem}{os.environ.get('GEN_TAG', '')}.jsonl"

def build(target: bool, connections: str) -> dict:
    """The lead's form: the routing subcomponents by role, value and output blocks at a, the target's blocks."""
    B = templates.Builder("copy_induction" + ("_target" if target else "") + ("_all" if connections == "all" else ""),
                          ROLES, connections, EXPLAIN)
    tok_o, tok_a, tok_t = B.node("tok_o", "o", tok=True), B.node("tok_a", "a", tok=True), B.node("tok_t", "t", tok=True)
    k329 = B.node("k1_329_o", "o", "h.1.attn.k_proj", 329)
    v1o = B.node("v1_o", "o", "h.1.attn.v_proj", block=True)
    fc53 = B.node("fc0_53_a", "a", "h.0.mlp.c_fc", 53)
    dn0 = B.node("down0_a", "a", "h.0.mlp.down_proj", block=True)
    q316 = B.node("q1_316_a", "a", "h.1.attn.q_proj", 316)
    o1a = B.node("o1_a", "a", "h.1.attn.o_proj", block=True)
    fc2103 = B.node("fc1_2103_a", "a", "h.1.mlp.c_fc", 2103)
    dn1 = B.node("down1_a", "a", "h.1.mlp.down_proj", block=True)
    k2a = B.node("k2_a", "a", "h.2.attn.k_proj", block=True)
    v2a = B.node("v2_a", "a", "h.2.attn.v_proj", block=True)
    q2t = B.node("q2_t", "t", "h.2.attn.q_proj", block=True)
    o2t = B.node("o2_t", "t", "h.2.attn.o_proj", block=True)
    for r in (k329, v1o):
        B.edge(tok_o, r)
    B.edge(v1o, o1a)
    for r in (fc53, q316, fc2103, k2a, v2a):
        B.edge(tok_a, r)
    B.edge(fc53, dn0)
    B.edge(fc2103, dn1)
    for r in (q316, fc2103, k2a, v2a):
        B.edge(dn0, r)
    for r in (fc2103, k2a, v2a):
        B.edge(o1a, r)
    for r in (k2a, v2a):
        B.edge(dn1, r)
    B.edge(tok_t, q2t)
    B.edge(v2a, o2t)
    B.edge(o2t, "out")
    B.pair(q316, k329, 1, "a")
    B.pair(q2t, k2a, 2, "t")
    if target:
        o3 = B.node("o3_t", "t", "h.3.attn.o_proj", block=True)
        fc3 = B.node("fc3_t", "t", "h.3.mlp.c_fc", block=True)
        dn3 = B.node("down3_t", "t", "h.3.mlp.down_proj", block=True)
        B.edge(o2t, fc3)
        B.edge(tok_t, fc3)
        B.edge(fc3, dn3)
        B.edge(o3, fc3)
        for w in (dn3, o3):
            B.edge(w, "out")
    return B.t


def build_columns(connections: str) -> dict:
    """Identity columns: at o, a and t every site that writes or reads the token's identity on the copy's path as a
    block, wired as the model wires a position (Builder.column), with the routing pairs named: layer 1's query #316 at a
    and keys #329 and #119 at o (on 40 of 40 held-out copy texts the pair with the largest lift toward o; stated), layer 2's query at t and key at a (blocks, their pair stated), the attention
    reads o -> a in layer 1 and a -> t in layer 2, and the target's outputs into the prediction."""
    B = templates.Builder("copy_columns" + ("_all" if connections == "all" else ""), ROLES, connections, EXPLAIN)
    L = lambda layer, kind: f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"  # noqa: E731
    co = B.column("o", [L(0, "c_fc"), L(0, "down_proj"), L(1, "v_proj")])
    k329 = B.node("k1_329_o", "o", "h.1.attn.k_proj", 329)
    k119 = B.node("k1_119_o", "o", "h.1.attn.k_proj", 119)
    for k in (k329, k119):
        B.edge("tok@o", k)
        B.edge(co[L(0, "down_proj")], k)
    ca = B.column("a", [L(0, "c_fc"), L(0, "down_proj"), L(1, "o_proj"), L(1, "c_fc"), L(1, "down_proj"), L(2, "k_proj"),
                        L(2, "v_proj")])
    q316 = B.node("q1_316_a", "a", "h.1.attn.q_proj", 316)
    B.edge("tok@a", q316)
    B.edge(ca[L(0, "down_proj")], q316)
    ct = B.column("t", [L(0, "c_fc"), L(0, "down_proj"), L(1, "c_fc"), L(1, "down_proj"), L(2, "q_proj"), L(2, "o_proj"),
                        L(3, "o_proj"), L(3, "c_fc"), L(3, "down_proj")])
    B.edge(co[L(1, "v_proj")], ca[L(1, "o_proj")])
    B.edge(ca[L(2, "v_proj")], ct[L(2, "o_proj")])
    for w in (L(2, "o_proj"), L(3, "o_proj"), L(3, "down_proj")):
        B.edge(ct[w], "out")
    B.pair(q316, k329, 1, "a")
    B.pair(q316, k119, 1, "a")
    B.pair(ct[L(2, "q_proj")], ca[L(2, "k_proj")], 2, "t")
    return B.t


def write():
    MECH.mkdir(parents=True, exist_ok=True)
    for t in (build(True, "stated"), build_columns("stated"), build_columns("all")):
        (MECH / f"{t['name']}.json").write_text(json.dumps({**t, "family": "B:copy", "rule": "copy"}, indent=1))
        print(MECH / f"{t['name']}.json")


def influential_blocks(nat, tx, k: int = 2):
    """Every site that can reach the targets as a block at the targets and at the text's k most influential positions."""
    import goal_eval
    import native

    moved = np.asarray(tx.moved, dtype=float)
    top = [int(j) + 1 for j in np.argsort(-moved)[:k]] if len(moved) else []
    reach = nat.reachable(tx.T, tx.targets)
    nodes = []
    for p in sorted(set(top) | set(tx.targets)):
        for n in nat.names:
            if bool(reach[n][p].any()):
                nodes.append((n, p, goal_eval.BLOCK))
    return native.Graph(nodes, complete=True), top


def score(task: str):
    import goal_eval
    import native
    import cscore

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    C = ge.nat.C
    tx = cscore.text(ge, task)
    graphs, meta = [], []
    for name in NAMES:
        tpl = json.loads((MECH / f"{name}.json").read_text())
        g, d = templates.graph(tpl, tx.ids, tx.targets, C)
        graphs.append(g)
        meta.append({"kind": "template", "name": name, "defects": d})
    ib, top = influential_blocks(ge.nat, tx)
    graphs.append(ib)
    meta.append({"kind": "influential_blocks", "name": f"blocks at t and {top}"})
    teacher = TEACH / f"{task}.json"
    if teacher.exists():
        g, d = goal_eval.circuit(json.loads(teacher.read_text()), tx.targets, C, tx.T)
        graphs.append(g)
        meta.append({"kind": "teacher", "name": task, "defects": d})
    pp = [goal_eval.strand(g, tx.targets, C) for g in graphs]
    listed = sorted({g.count() for g in pp if g.count() > 0})
    expanded = sorted({cscore.node_set_graph(g, C).count() for g in pp if g.count() > 0})
    base = ge.baselines(tx, listed + expanded)
    for k in sorted(set(listed + expanded)):
        graphs.append(base[f"vpd@{k}"])
        meta.append({"kind": "vpd", "name": f"vpd@{k}", "listed_size": k in listed, "expanded_size": k in expanded})
    rows = []
    for s0 in range(len(graphs)):  # one at a time: an engine spans its call's columns (a block is a site's every one)
        rows += cscore.score(ge, tx, graphs[s0:s0 + 1], draws=8, near=8, far=4)
        native.free(ge.nat.dev)
    with out_path("copy_template").open("a") as f:
        for m, r in zip(meta, rows):
            f.write(json.dumps({"task": task, **m, **r}) + "\n")
    print(task, [(m["name"], r["s_point_c"], r["s_track_pp"]) for m, r in zip(meta, rows)][:5], flush=True)


def predict(draws: int):
    """The template's predictions on changed inputs, checked in the real model, on the held-out copy texts. From the
    template alone: (1) a new token z at a (a sample from the model's prediction there, never one of its three likeliest,
    which chose the texts) becomes the prediction (the model's likeliest token at t is z), and at least raises z's
    probability above the old prediction's; (2) a new token at o (the earlier occurrence of the current token) breaks
    the copy: log p(old prediction) falls by at least 1 nat, and so does a new token at every earlier occurrence of the
    current token (a copy may read any of them); a new token after every earlier occurrence becomes the prediction; (3) a
    new token at a position outside t, a and o (and outside the template's roles) leaves the prediction: the likeliest
    token unchanged."""
    import goal_eval
    import native
    import torch

    nat = native.Native()
    ge = goal_eval.GoalEval(nat)
    held = json.loads((DATA / "copy_heldout.json").read_text())["copy"]
    out = {"texts": [], "summary": {}}
    acc = collections.defaultdict(list)
    for x in held:
        task = x["task"]
        ids, targets = native.text(Path.home() / "mpd-data/graph_oracle/texts/vpd4l" / f"{task}.json")
        t = targets[-1]
        r = templates.role_positions(ids, t)
        a, o = r["a"], r["o"]
        base = nat.reference([ids], [t])[0, 0]
        y = int(base.argmax())
        tx = ge.text(ids, targets, native.task_seed(task))
        rec = {"task": task, "y": y, "a": [], "o": [], "other": []}
        for pos, kind in ((a, "a"), (o, "o")):
            zs = ge._fresh_sources(tx, pos, int(tx.seed) + 7)[:draws]
            xs = []
            for z in zs:
                x2 = list(ids)
                x2[pos] = z
                xs.append(x2)
            lp = nat.reference(xs, [t])[:, 0]
            for z, l in zip(zs, lp):
                if kind == "a":
                    rec["a"].append([z, int(l.argmax()) == z, bool(l[z] > l[y])])
                    acc["a: prediction becomes z"].append(int(l.argmax()) == z)
                    acc["a: z above the old prediction"].append(bool(l[z] > l[y]))
                elif z != ids[t]:
                    drop = float(base[y] - l[y])
                    rec["o"].append([z, drop])
                    acc["o: copy breaks (log p(y) falls >= 1 nat)"].append(drop >= 1.0)
        occ = [j for j in range(t) if ids[j] == ids[t]]
        rec["occurrences"] = len(occ)
        zs = [z for z in ge._fresh_sources(tx, o, int(tx.seed) + 7) if z != ids[t]][:draws]
        xs = []
        for z in zs:  # every earlier occurrence of the current token replaced by z
            x2 = list(ids)
            for j in occ:
                x2[j] = z
            xs.append(x2)
        lp = nat.reference(xs, [t])[:, 0] if xs else []
        rec["o_all"] = []
        for z, l in zip(zs, lp):
            drop = float(base[y] - l[y])
            rec["o_all"].append([z, drop])
            acc["every earlier occurrence: copy breaks (log p(y) falls >= 1 nat)"].append(drop >= 1.0)
            acc[f"every earlier occurrence, {'one' if len(occ) == 1 else 'two or more'} occurrence(s): copy breaks"].append(drop >= 1.0)
        for z, d in rec["o"]:
            acc[f"o, {'one' if len(occ) == 1 else 'two or more'} occurrence(s): copy breaks"].append(d >= 1.0)
        aft = [j + 1 for j in occ if j + 1 < t]
        zs = ge._fresh_sources(tx, a, int(tx.seed) + 9)[:draws]
        xs = []
        for z in zs:  # the token after every earlier occurrence replaced by z
            x2 = list(ids)
            for j in aft:
                x2[j] = z
            xs.append(x2)
        lp = nat.reference(xs, [t])[:, 0]
        rec["a_all"] = []
        for z, l in zip(zs, lp):
            rec["a_all"].append([z, int(l.argmax()) == z])
            acc["after every earlier occurrence: prediction becomes z"].append(int(l.argmax()) == z)
        rng = np.random.default_rng(int(tx.seed))
        others = [q for q in range(1, t) if q not in (a, o)]
        for q in rng.choice(others, size=min(draws, len(others)), replace=False).tolist():
            z = ge._fresh_sources(tx, q, int(tx.seed) + 7)[0]
            x2 = list(ids)
            x2[q] = z
            l = nat.reference([x2], [t])[0, 0]
            same = int(l.argmax()) == y
            rec["other"].append([q, z, same, float(base[y] - l[y])])
            acc["other: prediction unchanged"].append(same)
        out["texts"].append(rec)
        torch.mps.empty_cache() if str(nat.dev).startswith("mps") else None
        print(task, {k: round(float(np.mean(v)), 2) for k, v in acc.items()}, flush=True)
    out["summary"] = {k: {"accuracy": float(np.mean(v)), "n": len(v)} for k, v in acc.items()}
    (DATA / "copy_predict.json").write_text(json.dumps(out))
    for k, v in out["summary"].items():
        print(f"  {k}: {v['accuracy']:.2f} of {v['n']}")


def report():
    """Per kind, the mean of each term over the held-out copy texts (and apart, the copy texts with teacher circuits),
    and the complete template's paired differences against the text's influential-blocks start and VPD's top k."""
    by = collections.defaultdict(dict)
    for p in [DATA / "copy_template.jsonl"] + templates.run_outputs("copy_template*.jsonl"):
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                key = r["name"] if r["kind"] in ("template", "vpd") else r["kind"]
                by[r["task"]].setdefault(key, r)
    held = {x["task"] for x in json.loads((DATA / "copy_heldout.json").read_text())["copy"]}
    terms = ("s_point_c", "s_wiring", "s_track_pp", "s_interchange_pp", "s_interchange", "s_donor_near", "s_donor_far", "c5", "bits")
    kinds = list(NAMES) + ["influential_blocks", "teacher", "vpd_listed", "vpd_expanded"]
    out = {}
    for group, tasks in (("held-out copy texts", [t for t in by if t in held]), ("copy texts with teacher circuits", [t for t in by if t not in held])):
        if not tasks:
            continue
        print(f"== {group}: {len(tasks)}")
        for k in kinds:
            rows = []
            for t in tasks:
                d = by[t]
                if k == "vpd_listed":
                    r = d.get(f"vpd@{d['copy_columns']['listed']}") if "copy_columns" in d else None
                elif k == "vpd_expanded":
                    r = d.get(f"vpd@{d['copy_columns']['nodes']}") if "copy_columns" in d else None
                    if r is None and "copy_columns" in d:  # VPD's whole answer is smaller than the template's expansion
                        r = max((v for kk, v in d.items() if kk.startswith("vpd@")), key=lambda v: v["nodes"], default=None)
                else:
                    r = d.get(k)
                if r:
                    rows.append(r)
            if not rows:
                continue
            means = {m: (float(np.mean([r[m] for r in rows if r.get(m) is not None])) if any(r.get(m) is not None for r in rows) else None)
                     for m in terms}
            out[(group, k)] = {**means, "n": len(rows), "listed": float(np.mean([r["listed"] for r in rows])),
                               "nodes": float(np.mean([r["nodes"] for r in rows]))}
            print(f"  {k:26s} n={len(rows):2d} listed {out[(group, k)]['listed']:6.0f} nodes {out[(group, k)]['nodes']:7.0f}  " +
                  "  ".join(f"{m} {v:.2f}" if m != "bits" else f"bits {v:.3g}" for m, v in means.items() if v is not None))
        for a_, b_ in (("copy_columns_all", "influential_blocks"), ("copy_columns_all", "vpd_listed"), ("copy_columns", "influential_blocks")):
            for m in ("c5", "s_track_pp", "s_interchange_pp", "s_point_c"):
                diffs = []
                for t in tasks:
                    d = by[t]
                    ra = d.get(a_)
                    rb = d.get(b_) if b_ != "vpd_listed" else (d.get(f"vpd@{d['copy_columns']['listed']}") if "copy_columns" in d else None)
                    if ra and rb and ra.get(m) is not None and rb.get(m) is not None:
                        diffs.append(ra[m] - rb[m])
                if diffs:
                    print(f"    {a_} - {b_} on {m}: {np.mean(diffs):+.3f} (higher on {sum(x > 0 for x in diffs)}/{len(diffs)})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "score", "predict", "report"))
    ap.add_argument("task", nargs="?")
    ap.add_argument("--draws", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "write":
        write()
    elif args.cmd == "score":
        score(args.task)
    elif args.cmd == "predict":
        predict(args.draws)
    else:
        report()
