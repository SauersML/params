"""Each intermediate variable's stated rule checked against the model on fresh donors (#2951 graph oracle; an external
review, the lead's 10-13): transplant V's carrier from a donor x' (a fresh token z at an input position p, outside V's
value map) onto the text, and compare the model's prediction with what V's rule predicts from the donor's intermediate
value, not with the donor's own answer.

The donor's value: the value map's class whose members' carriers lie nearest V's carrier on x' (the class mean of the
members' carriers, each on its own changed text; Euclidean; no output is read). The rule: that class's predicted token
(class 0, the text's own, predicts the text's top). Per donor: the patched run's top, the rule's prediction, the donor's
answer (the model's top on x'); the line holds when the patched top is the rule's. Per variable: the hold rate, the
share of donors where the patched top is the donor's answer, the share where rule and donor answer differ, and the
hold rate of a constant rule (always the largest class's token).

  experiments/oracle/var_rule.py [--dir influential_v2] [--part K/N] [--fresh 16] [--out FILE] [--device D]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import goal_eval  # noqa: E402
import native  # noqa: E402
import var_controls  # noqa: E402
import var_teacher  # noqa: E402
import variables  # noqa: E402


def run_variable(sc: variables.Scorer, tx, v: dict, pv: dict, fresh: int) -> list[dict]:
    """One row per input position with a value map of two or more classes."""
    nat = sc.nat
    keep = sorted(set(tx.targets) | {t for _, t, _ in pv["group"]})
    base = sc._text(tx.ids, tx.targets, check=False, positions=keep)
    out = []
    for vm in v.get("values") or []:
        p, classes = int(vm["position"]), vm["classes"]
        if len(classes) < 2:
            continue

        def donor(z):
            x = list(tx.ids)
            x[p] = int(z)
            return sc._text(x, tx.targets, check=False, positions=keep)
        cent = []
        for c in classes:  # each class's mean carrier over its members' changed texts (the text's own for its token)
            vecs = []
            for z in c["token_ids"]:
                vecs.append((sc._flat(base, pv["group"]) if int(z) == int(tx.ids[p]) else sc._flat(donor(z), pv["group"])).double().cpu())
            cent.append(torch.stack(vecs).mean(0))
            var_teacher.release(nat.dev)
        pred = [int(c["predicts_id"]) for c in classes]
        big = max(range(len(classes)), key=lambda i: len(classes[i]["token_ids"]))
        A = sorted({int(z) for c in classes for z in c["token_ids"]})
        donors = []
        for z in var_controls.b_tokens(nat, tx, p, A, fresh, int(tx.seed)):
            d = donor(z)
            a = sc._flat(d, pv["group"]).double().cpu()
            k = min(range(len(cent)), key=lambda i: float(((a - cent[i]) ** 2).sum()))
            q = sc._patched(base, [(d, pv["group"])])[0][-1]
            yp, yd = int(q.argmax()), int(d["lq"][-1].argmax())
            donors.append({"token": int(z), "class": k, "rule": pred[k], "patched": yp, "donor": yd})
            var_teacher.release(nat.dev)
        n = len(donors)
        out.append({"position": p, "classes": len(classes), "donors": donors,
                    "hold": sum(d["patched"] == d["rule"] for d in donors) / n if n else None,
                    "donor_agree": sum(d["patched"] == d["donor"] for d in donors) / n if n else None,
                    "rule_differs": sum(d["rule"] != d["donor"] for d in donors) / n if n else None,
                    "hold_where_differs": (lambda ds: sum(d["patched"] == d["rule"] for d in ds) / len(ds) if ds else None)(
                        [d for d in donors if d["rule"] != d["donor"]]),
                    "constant": sum(d["patched"] == pred[big] for d in donors) / n if n else None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=native.TEXTS.parent / "variables" / "influential_v2")
    ap.add_argument("--part", help="K/N: every N-th variable from the K-th (a cluster array's task)")
    ap.add_argument("--fresh", type=int, default=16)
    ap.add_argument("--out", type=Path, default=native.TEXTS.parent / "variables" / "rule_check.jsonl")
    ap.add_argument("--device")
    a = ap.parse_args()
    jobs = []
    for split in ("train", "heldout"):
        for f in sorted((a.dir / split).iterdir()):
            if f.suffix != ".json" or not f.name.startswith("text"):
                continue
            r = json.loads(f.read_text())
            for v in r["variables"]:
                if v.get("kept") and v.get("kind") == "intermediate" and v.get("values"):
                    jobs.append((split, f.stem, v["name"]))
    if a.part:
        k, n = map(int, a.part.split("/"))
        jobs = jobs[k::n]
    done = set()
    if a.out.exists():
        done = {(x["task"], x["name"]) for x in map(json.loads, a.out.read_text().splitlines())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(a.device))
    for split, t, name in jobs:
        if (t, name) in done:
            continue
        r = json.loads((a.dir / split / f"{t}.json").read_text())
        v = next(x for x in r["variables"] if x["name"] == name)
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        p = task["prompts"][0]
        tx = ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(t), task=task)
        pv = variables.parse([v], ge.nat.C)[0]
        rows = run_variable(variables.Scorer(ge), tx, v, pv, a.fresh)
        with a.out.open("a") as fh:
            fh.write(json.dumps({"task": t, "split": split, "name": name, "value_kind": v.get("value_kind"), "positions": rows}) + "\n")
        print(t, name, v.get("value_kind"), [(x["position"], x["hold"], x["donor_agree"], x["rule_differs"], x["constant"]) for x in rows], flush=True)


if __name__ == "__main__":
    main()
