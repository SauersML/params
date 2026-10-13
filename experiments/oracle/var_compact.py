"""Compact families for the VARIABLES grammar (#2951 graph oracle, the lead's 10-13): a variable's carrier written as
subcomponent ids, the smallest subset of its family that keeps a stated share of the family's terms. Blocks (a site's
whole output at a position) cannot state a selective variable: random sites' blocks at an input token's position carry
it as well (env C's first gate); a family of thousands of subcomponents is too long to write.

Per kept variable: the family F (its carrier when that is subcomponents, a VPD family or a mask; else var_teacher.family
over every site at the carrier positions: VPD's active subcomponents on the text and on the 64 alternative tokens at the
inputs); F's members ranked by how much they move when an input token changes (the mean |activation on x' - on x| over
DONORS alternative tokens at the inputs: the change each member transports); the smallest prefix whose on-target f and
off-target f (variables.Scorer, search draws) both reach SHARE of F's, found by doubling then halving the interval.

  experiments/oracle/var_compact.py [--dir influential_v2] [--tar FILE] [--split heldout] [--part K/N] [--share 0.9]
                                    [--out FILE] [--device D]
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
import var_classes  # noqa: E402
import var_teacher  # noqa: E402
import variables  # noqa: E402

DONORS = 16


def terms(sc, tx, ins: list, group: list) -> dict:
    v = {"name": "v", "meaning": "", "inputs": sorted(ins), "group": sorted(set(group)), "feeds": [], "values": []}
    x = sc.score(tx, [v], split="search", draws=8)["variables"][0]
    var_teacher.release(sc.nat.dev)
    return {"f_on": x["f_on"], "f_off": x["f_off"]}


def compact(sc, tx, v: dict, pv: dict, share: float) -> dict:
    nat = sc.nat
    ins, at = list(v["inputs"]), list(v.get("carried_at") or v["inputs"])
    if all(c != variables.BLOCK for _, _, c in pv["group"]):
        fam = sorted(set(pv["group"]))
    else:
        reach = nat.reachable(tx.T, tx.targets)
        every = [n for n in nat.names if all(bool(reach[n][q].any()) for q in at)]
        fam, _ = var_teacher.family(nat, tx, tuple(ins), every, tuple(at))
    if not fam:
        return {}
    keep = sorted(set(tx.targets) | set(at))
    base = sc._text(tx.ids, tx.targets, check=False, positions=keep)
    fb = sc._flat(base, fam).double().cpu()
    move = torch.zeros_like(fb)
    per = max(1, DONORS // len(ins))
    for p in ins:
        for z in var_classes.alternatives(nat, tx.ids, p, int(tx.seed))[:per]:
            x = list(tx.ids)
            x[p] = int(z)
            move += (sc._flat(sc._text(x, tx.targets, check=False, positions=keep), fam).double().cpu() - fb).abs()
    var_teacher.release(nat.dev)
    order = [fam[i] for i in torch.argsort(move, descending=True).tolist()]
    full = terms(sc, tx, ins, fam)

    def ok(k):
        got = terms(sc, tx, ins, order[:k])
        good = full["f_on"] is None or (got["f_on"] is not None and got["f_on"] >= share * full["f_on"])
        good = good and (full["f_off"] is None or (got["f_off"] is not None and got["f_off"] >= share * full["f_off"]))
        return good, got
    lo, hi, best = 0, None, None
    k = min(8, len(order))
    while True:  # doubling
        good, got = ok(k)
        if good:
            hi, best = k, got
            break
        lo = k
        if k == len(order):
            break
        k = min(2 * k, len(order))
    if hi is None:
        hi, best = len(order), full
    while hi - lo > max(1, hi // 8):  # halving, to an eighth of the size
        mid = (lo + hi) // 2
        good, got = ok(mid)
        if good:
            hi, best = mid, got
        else:
            lo = mid
    return {"family_size": len(fam), "size": hi, "family": full, "compact": best,
            "group": [[n, c, t] for n, t, c in sorted(order[:hi])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=native.TEXTS.parent / "variables" / "influential_v2")
    ap.add_argument("--tar", type=Path, help="a tar of DIR's train/ and heldout/, unpacked into --dir first")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--part", help="K/N: every N-th variable from the K-th")
    ap.add_argument("--share", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=native.TEXTS.parent / "variables" / "compact.jsonl")
    ap.add_argument("--device")
    a = ap.parse_args()
    if a.tar:
        import tarfile

        a.dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(a.tar) as t:
            t.extractall(a.dir)
    jobs = []
    for f in sorted((a.dir / a.split).iterdir()):
        if f.suffix != ".json" or not f.name.startswith("text"):
            continue
        for v in json.loads(f.read_text())["variables"]:
            if v.get("kept"):
                jobs.append((f.stem, v["name"]))
    if a.part:
        k, n = map(int, a.part.split("/"))
        jobs = jobs[k::n]
    done = {(x["task"], x["name"]) for x in map(json.loads, a.out.read_text().splitlines())} if a.out.exists() else set()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(a.device))
    cur = None
    for t, name in jobs:
        if (t, name) in done:
            continue
        r = json.loads((a.dir / a.split / f"{t}.json").read_text())
        v = next(x for x in r["variables"] if x["name"] == name)
        if cur is None or cur[0] != t:
            task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
            p = task["prompts"][0]
            cur = (t, ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(t), task=task))
        pv = variables.parse([v], ge.nat.C)[0]
        got = compact(variables.Scorer(ge), cur[1], v, pv, a.share)
        row = {"task": t, "split": a.split, "name": name, "kind": v["kind"], "value_kind": v.get("value_kind"), "carrier": v["carrier"],
               "share": a.share, **got}
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(t, name, v["kind"], v["carrier"], got.get("family_size"), "->", got.get("size"), got.get("family"), got.get("compact"), flush=True)


if __name__ == "__main__":
    main()
