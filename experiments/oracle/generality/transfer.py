"""Transfer of teacher circuits within families of texts.

Per family (members: tasks), per target text B: B's own teacher circuit; every other member A's circuit mapped onto B
(positions by role: the target to the target, the previous occurrence of the current token and the position after it
to the same roles on B, any other position by its offset from the target; a node falling before the text is dropped);
the family's shared core from the other members (leave-one-out: (matrix, index, role) in at least half of them),
complete (every connection the model has among its nodes) and stated (the edges at least half the members holding
both ends state); the same core rule on as many random texts from outside the family (core_random), complete; the universal core
(target subcomponents in at least half of all teacher circuits), complete;
circuits of CONTROLS texts outside the family mapped the same way; VPD's top k on B at each of those sizes. All scored
on B by score.score (goal_eval's point, changed-input and interchange families, eval split)."""
import argparse
import collections
import gc
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
sys.path.insert(0, str(HERE))

CONTROLS = 3
SCORE_CHUNK = 4  # interchange keeps one whole-model record per distinct draw of a call: 4 circuits stay under 14 GiB
TEACH = {"train": Path.home() / "mpd-data/graph_oracle/teacher/joint_train",
         "heldout": Path.home() / "mpd-data/graph_oracle/teacher/joint_heldout"}


def role(r: dict, p: int):
    if p == r["t"]:
        return "T"
    for k in ("after_prev", "prev_occ"):
        if r["roles"].get(k) == p:
            return k
    return ("off", r["t"] - p)


def place(r: dict, rl):
    if rl == "T":
        return r["t"]
    if isinstance(rl, str):
        return r["roles"].get(rl)
    q = r["t"] - rl[1]
    return q if q >= 0 else None


def answer(r: dict) -> dict:
    return json.loads((TEACH[r["split"]] / f"{r['task']}.json").read_text())


def mapped(src: dict, dst: dict) -> tuple[dict, dict]:
    """src's teacher answer with its positions moved onto dst by role; (answer, counts by rule)."""
    a = answer(src)
    keep, nodes, rules = {}, [], collections.Counter()
    for i, (m, c, p) in enumerate(a["nodes"]):
        rl = role(src, int(p))
        q = place(dst, rl)
        rules[("T" if rl == "T" else rl if isinstance(rl, str) else "offset") + ("" if q is not None else "_dropped")] += 1
        if q is None:
            continue
        keep[i] = len(nodes)
        nodes.append([m, c, q])
    edges = [[keep[u], v if v == "out" else keep[v]] for u, v in a["edges"] if u in keep and (v == "out" or v in keep)]
    return {"nodes": nodes, "edges": edges, "explanation": ""}, dict(rules)


def core(members: list[dict]) -> tuple[list, list, int]:
    """(core nodes as (matrix, index, role), core edges as role-node pairs or (role-node, "out"), threshold): the
    subcomponents the teachers list (stranded or not) in more than half of the members, and the edges more than half of
    the members holding both ends state."""
    cnt, ecnt = collections.Counter(), collections.Counter()
    for r in members:
        a = answer(r)
        keyed = {i: (m, int(c), role(r, int(p))) for i, (m, c, p) in enumerate(a["nodes"]) if m != "tok"}
        cnt.update(set(keyed.values()))
        ecnt.update({(keyed[u], "out" if v == "out" else keyed[v]) for u, v in a["edges"] if u in keyed and (v == "out" or v in keyed)})
    need = len(members) // 2 + 1
    nodes = [k for k, v in cnt.items() if v >= need]
    ns = set(nodes)
    edges = []
    for (u, v), n in ecnt.items():
        if u in ns and (v == "out" or v in ns):
            both = cnt[u] if v == "out" else min(cnt[u], cnt[v])  # an upper bound on members holding both ends
            if n >= both // 2 + 1:
                edges.append((u, v))
    return nodes, edges, need


def core_answer(nodes, edges, dst: dict) -> dict:
    at, out = {}, []
    for m, c, rl in nodes:
        q = place(dst, rl)
        if q is not None:
            at[(m, c, rl)] = len(out)
            out.append([m, c, q])
    E = [[at[u], "out" if v == "out" else at[v]] for u, v in edges if u in at and (v == "out" or v in at)]
    return {"nodes": out, "edges": E, "explanation": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=Path, default=DATA / "families.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--part", help="a key of split.json: its families")
    ap.add_argument("--backfill", action="store_true", help="core_random and VPD's top k at its size alone (texts run before it existed)")
    ap.add_argument("--target", help="this target text alone (one process per text: the process's memory grows per text)")
    ap.add_argument("--out", type=Path, default=DATA / "transfer.jsonl")
    ap.add_argument("--draws", type=int, default=16)
    args = ap.parse_args()
    if args.part:
        args.only = json.loads((DATA / "split.json").read_text())[args.part]
    import goal_eval
    import native
    import gscore as score

    rows = pickle.load(open(DATA / "rows.pkl", "rb"))
    by = {r["task"]: r for r in rows}
    fams = json.loads(args.families.read_text())
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["family"], d["target"]))
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1  # one length's plans and masks at a time: the process stays under 16 GiB
    C = ge.nat.C
    S = sum(1 for r in rows)
    uni_cnt = collections.Counter((m, c) for r in rows for m, c, p in set(r["nodes"]) if p == r["t"] and m != "tok")
    universal = [(m, c, "T") for (m, c), v in uni_cnt.items() if v >= S / 2]
    for name, tasks in fams.items():
        if args.only and name not in args.only:
            continue
        members = [by[t] for t in tasks]
        outside = [r for r in rows if r["task"] not in set(tasks)]
        for B in members:
            if (name, B["task"]) in done or (args.target and B["task"] != args.target):
                continue
            t0 = time.perf_counter()
            tx = score.text(ge, B["task"])
            graphs, meta = [], []

            def add(ans, kind, source, extra=None):
                g, d = goal_eval.circuit(ans, tx.targets, C, tx.T)
                graphs.append(g)
                meta.append({"kind": kind, "source": source, "defects": d, **(extra or {})})

            add(answer(B), "own", B["task"])
            for A in members:
                if A is not B:
                    ans, rules = mapped(A, B)
                    add(ans, "mapped", A["task"], {"rules": rules})
            rng = random.Random(f"{name}/{B['task']}")
            for A in rng.sample(outside, CONTROLS):
                ans, rules = mapped(A, B)
                add(ans, "control", A["task"], {"rules": rules})
            cn, ce, need = core([A for A in members if A is not B])
            ca = core_answer(cn, ce, B)
            if ca["nodes"]:
                g = native.Graph([(m, q, c) for m, c, q in ca["nodes"] if m != "tok"], complete=True)
                graphs.append(g)
                meta.append({"kind": "core", "source": f"loo{len(members) - 1}>={need}", "defects": {},
                             "core": [list(x[:2]) + [str(x[2])] for x in cn]})
                add(ca, "core_stated", f"loo{len(members) - 1}>={need}")
            rn, _, rneed = core(rng.sample(outside, len(members) - 1))  # the same rule on as many texts from outside the family
            ra = core_answer(rn, [], B)
            if ra["nodes"]:
                graphs.append(native.Graph([(m, q, c) for m, c, q in ra["nodes"]], complete=True))
                meta.append({"kind": "core_random", "source": f"random{len(members) - 1}>={rneed}", "defects": {}})
            ua = core_answer(universal, [], B)
            graphs.append(native.Graph([(m, q, c) for m, c, q in ua["nodes"]], complete=True))
            meta.append({"kind": "universal", "source": f"all{S}", "defects": {}})
            if args.backfill:
                sel = [i for i, m in enumerate(meta) if m["kind"] == "core_random"]
                graphs, meta = [graphs[i] for i in sel], [meta[i] for i in sel]
            kept = [goal_eval.strand(g, tx.targets, C) for g in graphs]
            sizes = sorted({g.count() for g in kept if g.count() > 0})
            base = ge.baselines(tx, sizes)
            for k in sizes:
                graphs.append(base[f"vpd@{k}"])
                meta.append({"kind": "vpd", "source": f"vpd@{k}", "defects": {}})
            scored = []
            for s0 in range(0, len(graphs), SCORE_CHUNK):  # the engine spans a call's columns: chunks bound the process's memory
                scored += score.score(ge, tx, graphs[s0:s0 + SCORE_CHUNK], args.draws)
                native.free(ge.nat.dev)
            with args.out.open("a") as f:
                for m, s in zip(meta, scored):
                    f.write(json.dumps({"family": name, "target": B["task"], **m, **{k: (float(v) if hasattr(v, "item") else v) for k, v in s.items()}}) + "\n")
            print(f"{name} {B['task']}: {len(graphs)} graphs, {time.perf_counter() - t0:.0f} s", flush=True)
            ge.probes.clear()
            gc.collect()
            native.free(ge.nat.dev)


if __name__ == "__main__":
    main()
