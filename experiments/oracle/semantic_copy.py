"""The copy family's semantic model checked in the real model (#2951 graph oracle, E4 v2 with operations): on held-out
copy texts (generality's copy_heldout.json), the roles of induction (generality/induction.py): t the target, o the
earlier occurrence of t's token, a = o + 1 the copied token's position. Variables T, O and A are carried by blocks at
their positions (var_teacher.block_search: the smallest site set whose term min(f on-target, s off-target) reaches
GOAL on the search split, else the best). Two stated algorithms predict the outcome of each intervention from the
variables' values (read off the donors, semantic.values_under):
  match:        the token after an earlier occurrence of the current token is copied: top = A if T = O, else not
                the copied token;
  class match:  the same with the match taken on value classes: T and O match when their tokens are equal or one
                is the text's and the other is in the class O's value map (var_classes.classes at o) gives o's own
                token (the model's match is not exact: on text6011 ' between' or ' with' at o keeps the copy);
  fixed:        the token at a fixed position is copied, whatever matches: top = A.
Interventions (z an alternative token at a, w one at o other than t's token; semantic.Checker): A from a donor with
z (both: top z); O from a donor with w (match: not A; fixed: top A); T from a donor with w at t (match: not A; fixed:
top A); O and T from two donors with the same w at o and at t (match restored: both top A, the recombination test);
A from z and O from w (match: not z; fixed: top z); and the plain token change w at o (match: not A). Each prediction
is priced in the real model (holds, surprise against the no-change prediction, credit (2 holds - 1) x surprise, and
logp, the real run's log2 probability of the predicted event, which gives no free pass for predicting no change); per
algorithm the totals, on every test and on the discriminating ones (where the algorithms' predictions differ).

  experiments/oracle/semantic_copy.py [--n 40] [--out FILE]   (default ~/mpd-data/graph_oracle/semantic/copy.jsonl, one
  JSON line per text, resumable)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import semantic  # noqa: E402
import var_classes  # noqa: E402
import var_teacher  # noqa: E402

DATA = native.TEXTS.parent / "generality"


def roles(ids: list[int], t: int) -> dict:
    occ = [j for j in range(t) if ids[j] == ids[t]]
    return {"t": t, "o": occ[-1], "a": occ[-1] + 1} if occ else None


def predictions(v: dict, ids: list[int], r: dict, alike: set) -> dict:
    """Each algorithm's outcome from the variables' values v (token ids) on the text ids with roles r; alike: the
    tokens O's value map puts in the class of o's own token."""
    same = v["T"] == v["O"] or (v["O"] in alike and v["T"] == ids[r["t"]]) or (v["T"] in alike and v["O"] == ids[r["o"]])
    return {"match": ("top", v["A"]) if v["T"] == v["O"] else ("not", v["A"]),
            "class match": ("top", v["A"]) if same else ("not", v["A"]), "fixed": ("top", v["A"])}


def run_text(ge, sc, task_id: str, draws: int) -> dict | None:
    task = json.loads((native.TEXTS / "vpd4l" / f"{task_id}.json").read_text())
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    r = roles(ids, targets[-1])
    if r is None:
        return None
    tx = ge.text(ids, targets, seed=native.task_seed(task_id), task=task)
    nat = ge.nat
    vs, carriers = [], {}
    for name, key in (("T", "t"), ("O", "o"), ("A", "a")):
        b, rows = var_teacher.block_search(sc, tx, (r[key],), draws)
        carriers[name] = {"how": b["how"], "score": b["score"], "f_on": b["f_on"], "s_off": b["s_off"], "size": b["size"]}
        vs.append({"name": name, "inputs": [r[key]], "group": b["group"]})
    alt = lambda q, bad: next(z for z in var_classes.alternatives(nat, ids, q, int(tx.seed)) if z not in bad)  # noqa: E731
    z = alt(r["a"], {ids[r["a"]]})
    w = alt(r["o"], {ids[r["o"]], ids[r["t"]]})
    ivs = [("A from z", {"transplants": [("A", {r["a"]: z})]}),
           ("O from w", {"transplants": [("O", {r["o"]: w})]}),
           ("T from w", {"transplants": [("T", {r["t"]: w})]}),
           ("O and T from w", {"transplants": [("O", {r["o"]: w}), ("T", {r["t"]: w})]}),
           ("A from z, O from w", {"transplants": [("A", {r["a"]: z}), ("O", {r["o"]: w})]}),
           ("token w at o", {"tokens": {r["o"]: w}})]
    named = {v["name"]: v for v in vs}
    vm = var_classes.classes(sc, tx, named["O"], r["o"])  # O's value map: the tokens at o it treats as o's own
    alike = set(vm["values"][0]["token_ids"])
    var_teacher.release(nat.dev)
    ck = semantic.Checker(sc, tx, vs)
    tests = []
    try:
        for label, iv in ivs:
            pred = predictions(semantic.values_under(named, ids, iv), ids, r, alike)
            q = ck.run(iv)
            got = {k: ck.outcome(q, o) | {"outcome": [o[0], int(o[1])]} for k, o in pred.items()}
            tests.append({"test": label, "top": int(q.argmax()), **got})
    finally:
        ck.close()
    toks = components.token_strings("vpd4l", [ids[r["t"]], ids[r["a"]], z, w])
    return {"task": task_id, "roles": r, "tokens": {"t": toks[0], "a": toks[1], "z": toks[2], "w": toks[3]},
            "w_alike": w in alike, "alike": vm["values"][0]["tokens"], "carriers": carriers, "tests": tests}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", type=Path, default=native.TEXTS.parent / "semantic" / "copy.jsonl")
    ap.add_argument("--device")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["task"] for x in args.out.read_text().splitlines()} if args.out.exists() else set()
    ge = goal_eval.GoalEval(native.Native(args.device))
    sc = var_teacher._Screen(ge)
    tasks = [x["task"] for x in json.loads((DATA / "copy_heldout.json").read_text())["copy"]][:args.n]
    for t in tasks:
        if t in done:
            continue
        row = run_text(ge, sc, t, args.draws)
        if row is None:
            continue
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(t, {k: round(v["score"], 3) for k, v in row["carriers"].items()},
              [(x["test"], x["match"]["holds"], x["class match"]["holds"], x["fixed"]["holds"]) for x in row["tests"]], flush=True)
    rows = [json.loads(x) for x in args.out.read_text().splitlines()]
    algs = ("match", "class match", "fixed")
    for alg in algs:  # every test, and the tests on which the algorithms' predictions differ (discriminating)
        xs = [x[alg] for r in rows for x in r["tests"]]
        ds = [x[alg] for r in rows for x in r["tests"] if len({tuple(x[a]["outcome"]) for a in algs}) > 1]
        print(f"{alg}: credit {sum(x['credit'] for x in xs):.1f} bits, holds {sum(x['holds'] for x in xs)} of {len(xs)}, "
              f"log2 p of the predicted events {sum(x['logp'] for x in xs):.1f}; discriminating: holds {sum(x['holds'] for x in ds)} "
              f"of {len(ds)}, log2 p {sum(x['logp'] for x in ds):.1f}")


if __name__ == "__main__":
    main()
