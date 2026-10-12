"""Do explanations predict behaviour on texts they were not built from? (#2951 graph oracle; the lead's 10-12, after
Geiger, Harding and Icard 2026.) The variable teacher's verified variables on a text A (var_teacher.py:
~/mpd-data/graph_oracle/variables/heldout, subcomponent groups, and variables/blocks/heldout, block groups; each with
its value map, var_classes.py: per input position the alternative tokens alike the text's own and the others grouped
by where they send the prediction) are mapped by role (each position by its offset from the target) onto the other
texts B of A's family (DATA/families.json), and tested there:
  E4 on B: variables.Scorer in the real model (eval split): on-target f, off-target s, e4;
  predictions on B from A's value map alone, checked in the model on B: an alternative A's map calls alike (the same
     value) leaves B's prediction (its likeliest token) unchanged; one A's map calls different changes it; and, stronger,
     moves it to the token A's map says that value predicts.
Beside them, A's variables on A (their recorded eval-split scores) and B's own variables where B has some.

  var_transfer.py run FAMILY SOURCE   -> DATA/var_transfer.jsonl (both variable sets of SOURCE on every other member)
  var_transfer.py report
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))

import native  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
VARS = Path.home() / "mpd-data/graph_oracle/variables"
SETS = {"subcomponents": VARS / "heldout", "blocks": VARS / "blocks" / "heldout"}
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"


def target(task):
    return native.text(TEXTS / f"{task}.json")[1][-1]


def mapped(var: dict, ta: int, tb: int) -> dict | None:
    """var with every position moved from offset ta - p on A to tb - (ta - p) on B; None when one falls before B."""
    mv = lambda p: tb - (ta - int(p))  # noqa: E731
    inputs = [mv(p) for p in var["inputs"]]
    group = [[g[0], g[1], mv(g[2])] for g in var["group"]]
    if min(inputs + [g[2] for g in group]) < 1:
        return None
    return {**{k: var[k] for k in ("name", "meaning", "feeds")}, "inputs": inputs, "group": group}


def run(family: str, source: str):
    import goal_eval
    import variables

    fams = json.loads((DATA / "families.json").read_text())
    members = [t for t in fams[family] if t != source]
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    sc = variables.Scorer(ge)
    ta = target(source)
    out = DATA / "var_transfer.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["family"], r["source"], r["target"], r["set"]))
    for kind, d in SETS.items():
        p = d / f"{source}.json"
        if not p.exists():
            continue
        src = json.loads(p.read_text())["variables"]
        for b in members:
            if (family, source, b, kind) in done:
                continue
            pb = TEXTS / f"{b}.json"
            ids, targets = native.text(pb)
            tb = targets[-1]
            vs = [m for m in (mapped(v, ta, tb) for v in src) if m is not None]
            row = {"family": family, "source": source, "target": b, "set": kind, "variables": len(src), "mapped": len(vs),
                   "source_eval": [v["measured"].get("eval", {}).get("term") for v in src]}
            if vs:
                tx = ge.text(ids, targets, native.task_seed(b), json.loads(pb.read_text()))
                res = sc.score(tx, variables.parse(vs, nat.C), "eval", 8)
                row.update({k: res.get(k) for k in ("e4", "f_on", "s_off", "unresolved")})
                own = SETS[kind] / f"{b}.json"
                if own.exists():
                    row["own_eval"] = [v["measured"].get("eval", {}).get("term") for v in json.loads(own.read_text())["variables"]]
                # the value map's predictions on B, checked in the model
                base = nat.reference([ids], [tb])[0, 0]
                y = int(base.argmax())
                preds = []
                for v in src:
                    for c in v.get("classes", []):
                        pos = tb - (ta - int(c["position"]))
                        if pos < 1:
                            continue
                        alts = [a for a in c.get("alternatives", []) if a["id"] != ids[pos]]
                        if not alts:
                            continue
                        xs = []
                        for a in alts:
                            x2 = list(ids)
                            x2[pos] = a["id"]
                            xs.append(x2)
                        lp = nat.reference(xs, [tb])[:, 0]
                        for a, l in zip(alts, lp):
                            new = int(l.argmax())
                            tk_pred = a.get("predicts")
                            preds.append({"alike": bool(a["alike"]), "unchanged": new == y,
                                          "to_named": None if a["alike"] else (native_decode(new) == tk_pred)})
                ok = [p_["alike"] == p_["unchanged"] for p_ in preds]
                row["predictions"] = len(preds)
                row["accuracy"] = float(np.mean(ok)) if ok else None
                al = [p_["unchanged"] for p_ in preds if p_["alike"]]
                ch = [not p_["unchanged"] for p_ in preds if not p_["alike"]]
                row["acc_alike"] = float(np.mean(al)) if al else None
                row["acc_changed"] = float(np.mean(ch)) if ch else None
                nm = [p_["to_named"] for p_ in preds if not p_["alike"]]
                row["acc_named_token"] = float(np.mean(nm)) if nm else None
                ge.probes.clear()
                native.free(nat.dev)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(family, source, b, kind, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                            if k in ("e4", "f_on", "s_off", "accuracy", "acc_alike", "acc_changed", "acc_named_token")}, flush=True)


_TK = []


def native_decode(i: int) -> str:
    import mech

    if not _TK:
        _TK.append(mech.tokenizer("vpd4l"))
    return _TK[0].decode([i])


def report():
    rows = [json.loads(line) for line in (DATA / "var_transfer.jsonl").read_text().splitlines()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["family"], r["set"])].append(r)
    for (f, kind), rs in sorted(by.items()):
        rs = [r for r in rs if r.get("e4") is not None or r.get("accuracy") is not None]
        if not rs:
            continue
        m = lambda k: np.mean([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else float("nan")  # noqa: E731
        src = np.mean([x for r in rs for x in r["source_eval"] if x is not None])
        own = [x for r in rs for x in r.get("own_eval", []) if x is not None]
        print(f"{f} [{kind}] {len(rs)} (source, target) pairs: e4 on the target {m('e4'):.2f} (f_on {m('f_on'):.2f}, s_off "
              f"{m('s_off'):.2f}; the source's own eval term {src:.2f}" + (f", the target's own {np.mean(own):.2f}" if own else "")
              + f"); value-map predictions {m('accuracy'):.2f} (alike kept {m('acc_alike'):.2f}, different moved "
              f"{m('acc_changed'):.2f}, moved to the named token {m('acc_named_token'):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    run(*a.args) if a.cmd == "run" else report()
