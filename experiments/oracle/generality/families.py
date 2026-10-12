"""Families of texts whose teacher circuits might share a mechanism (#2951 graph oracle, the generality test).

  families.py rows        every teacher circuit (joint_train, joint_heldout) with its text: token ids, target, current
                          and predicted token, the nodes, the nodes strand() keeps, and role positions -> DATA/rows.pkl
  copytest.py             which texts copy (measured) -> DATA/copytest.json
  families.py families    the families -> DATA/families.json (at most CAP members each, a fixed draw)

Roles of a position p on a text with target t (the prediction at t is of token t + 1): "T" (p = t); "prev_occ", the
latest j < t with ids[j] = ids[t]; "after_prev", prev_occ + 1 (the token an induction copy reads); any other position
by its offset t - p.
(a) Shared subcomponents: per circuit the (matrix, index) set at the target among the nodes strand() keeps; pairs
compared by cosine of IDF-weighted indicator vectors (IDF = ln(N / circuits holding it): a subcomponent in every
circuit says nothing about a family); average-linkage clusters at similarity CUT; clusters of 5 or more.
(b) Behaviours: a preposition followed by " the"; "," followed by a function token; measured copies (copytest.py: the
predicted token is the one after the earlier occurrence of the current token, and replacing that token by the model's
three likeliest others there lowers log p(predicted) by at least 1 nat and raises the new token's by at least 1 nat)."""
import collections
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("ORACLE", str(Path(__file__).resolve().parents[3] / "method" / "oracle")))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
TEACH = Path.home() / "mpd-data/graph_oracle/teacher"
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
DIRS = {"train": TEACH / "joint_train", "heldout": TEACH / "joint_heldout"}
CUT = 0.2  # clusters: average similarity at least this
CAP = 12  # members per family (a fixed draw of a larger one)
PREP = {" of", " on", " in", " into", " with", " to", " for", " from", " by", " at", " that", " about", " through",
        " under", " over", " between", " after", " before", " during", " against", " without", " within", " upon"}


def teacher_files(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.name.startswith("text") and p.name.endswith(".json") and p.name.count(".") == 1],
                  key=lambda p: int(p.stem[4:]))


def roles(ids: list[int], t: int) -> dict:
    r = {"cur": t}
    occ = [j for j in range(t) if ids[j] == ids[t]]
    if occ:
        r["prev_occ"] = occ[-1]
        r["after_prev"] = occ[-1] + 1
    return r


def rows():
    import goal_eval
    import mech
    import native

    tk = mech.tokenizer("vpd4l")
    C = native.Native("cpu").C
    out = []
    for split, d in DIRS.items():
        for p in teacher_files(d):
            c = json.loads(p.read_text())
            pr = json.loads((TEXTS / p.name).read_text())["prompts"][0]
            ids, targets = pr["token_ids"], pr["target_positions"]
            t = targets[-1]
            top = pr["model_top"][0]
            g, defects = goal_eval.circuit(c, targets, C, len(ids))
            out.append({"task": p.stem, "split": split, "ids": ids, "targets": targets, "t": t, "cur": tk.decode([ids[t]]),
                        "pred": top[0][0], "p_pred": top[0][1], "top": top, "nodes": [tuple(x) for x in c["nodes"]],
                        "edges": c.get("edges", []), "roles": roles(ids, t), "kept": sorted(g.node_set()),
                        "stranded": len(g.stranded), "defects": defects, "tok_edges": g.tok_edges})
    DATA.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(DATA / "rows.pkl", "wb"))
    print(len(out), "circuits ->", DATA / "rows.pkl")


def target_set(r) -> set:
    return {(m, c) for m, p, c in r["kept"] if p == r["t"]}


def similarity(rs) -> np.ndarray:
    sets = [target_set(r) for r in rs]
    df = collections.Counter(x for s in sets for x in s)
    idf = {x: math.log(len(rs) / d) for x, d in df.items()}
    at = {x: i for i, x in enumerate(sorted(df))}
    M = np.zeros((len(sets), len(at)))
    for i, s in enumerate(sets):
        for x in s:
            M[i, at[x]] = idf[x]
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)
    return M @ M.T


def clusters(S: np.ndarray, cut: float) -> list[list[int]]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = 1 - S
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D.clip(0), checks=False), "average"), t=1 - cut, criterion="distance")
    out = collections.defaultdict(list)
    for i, x in enumerate(lab):
        out[x].append(i)
    return sorted(out.values(), key=len, reverse=True)


def behaviours(rs, tk) -> dict:
    fam = collections.defaultdict(list)
    for i, r in enumerate(rs):
        cur, pred = r["cur"], r["pred"]
        if cur in PREP and pred == " the":
            fam["prep->the"].append(i)
        if cur == "," and pred in (" and", " the", " but", " which", " a"):
            fam[",->function"].append(i)
        a = r["roles"].get("after_prev")
        if a is not None and tk.decode([r["ids"][a]]) == pred:
            fam["copy(all)"].append(i)
    return dict(fam)


def cap(name, tasks):
    tasks = sorted(tasks, key=lambda t: int(t[4:]))
    return sorted(random.Random(name).sample(tasks, CAP), key=lambda t: int(t[4:])) if len(tasks) > CAP else tasks


def families():
    import mech

    tk = mech.tokenizer("vpd4l")
    rs = pickle.load(open(DATA / "rows.pkl", "rb"))
    fams = {}
    for c in [c for c in clusters(similarity(rs), CUT) if len(c) >= 5]:
        cur = collections.Counter(rs[i]["cur"] for i in c).most_common(1)[0][0]
        pred = collections.Counter(rs[i]["pred"] for i in c).most_common(1)[0][0]
        name = "S:" + repr(cur).strip("'") + "->" + repr(pred).strip("'")
        while name in fams:
            name += "+"
        fams[name] = cap(name, [rs[i]["task"] for i in c])
    beh = behaviours(rs, tk)
    fams["B:prep->the"] = cap("B:prep->the", [rs[i]["task"] for i in beh["prep->the"]])
    fams["B:comma->function"] = cap("B:comma->function", [rs[i]["task"] for i in beh[",->function"]])
    copies = [x["task"] for x in json.load(open(DATA / "copytest.json")) if x["drop_nats"] >= 1 and x["new_gain_nats"] >= 1]
    fams["B:copy"] = cap("B:copy", copies)
    (DATA / "families.json").write_text(json.dumps(fams, indent=1))
    for k, v in fams.items():
        print(k, len(v))


if __name__ == "__main__":
    {"rows": rows, "families": families}[sys.argv[1]]()
