"""A second pass over var_teacher.py's output files (#2951 graph oracle): what a file written before a field existed
lacks, from its kept variables alone (no carrier search): the decomposition label (value_kind: computed when a value
class holds tokens other than the text's, copies its inputs otherwise, identity for a variable at its own input
positions without a value map), RAVEL's Iso and Disentangle on both splits on f's scale (the off-target draws that can
reach the carrier, changes before it; Cause is f on-target; files written before 10-13 hold Iso on s's log scale), and
the verified INVARIANTS lines (semantic.teacher_invariants) unless the file holds the current version (INVARIANTS).
Files are rewritten in place (or, with --tar, a tar of the directories is unpacked under --out first: a pod uploads
one file); a file with both done is skipped.

  experiments/oracle/var_post.py DIR [DIR ...] [--tar FILE --out DIR] [--part K/N] [--device mps]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import goal_eval  # noqa: E402
import native  # noqa: E402
import semantic  # noqa: E402
import var_teacher  # noqa: E402
import variables  # noqa: E402


INVARIANTS = "graded removals, one-input checks"  # teacher_invariants' version (10-13: removal lines claim half the
# measured drop, checked with a mean over fresh tokens; a line changing two inputs also checked with each alone): a file
# holding "graded removals" gets the one-input checks on its lines, an older one its lines written again


def post(ge, path: Path, draws: int = 8, log=print) -> None:
    r = json.loads(path.read_text())
    kv = [v for v in r["variables"] if v.get("kept")]
    iso_done = all(v["measured"].get("eval", {}).get("iso_scale") == "f" for v in kv)
    if iso_done and r["teacher"].get("invariants_version") == INVARIANTS:
        return
    task = json.loads((native.TEXTS / "vpd4l" / f"{path.stem}.json").read_text())
    p = task["prompts"][0]
    tx = ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(path.stem), task=task)
    nat = ge.nat
    for v in kv:
        if v.get("kind") == "input" and not v.get("values"):
            v["value_kind"] = "identity"
        elif v.get("values"):
            v["value_kind"] = "computed" if any(len(vm["classes"][0]["token_ids"]) > 1 for vm in v["values"]) else "copies its inputs"
        at = v.get("carried_at") or v["inputs"]
        for split in ("search", "eval") if not iso_done else ():
            got = var_teacher.together(variables.Scorer(ge), tx, variables.parse([v], nat.C), split, draws)
            reach = [d for d in got["draws"] if d["kind"] == "off" and d["f"] is not None and d["changes"][0][0] < min(at)]
            w = sum(d["d0"] for d in reach)
            iso = sum(d["d0"] * d["f"] for d in reach) / w if w > 0 else None
            f = got["variables"][0]["f_on"]
            v["measured"].setdefault(split, {}).update(cause=f, iso=iso, iso_scale="f", iso_draws=len(reach), off_draws=got["variables"][0]["draws_off"],
                                                       disentangle=None if f is None or iso is None else (f + iso) / 2)
    if r["teacher"].get("invariants_version") == "graded removals" and r.get("invariants"):  # the one-input checks alone
        sv = [{"name": v["name"], "inputs": v["inputs"], "group": pv["group"]} for v, pv in zip(kv, variables.parse(kv, nat.C))]
        lines, _ = semantic.parse_invariants("\n".join(x["line"] for x in r["invariants"]), task["model"])
        by = {x["line"]: x for x in lines}
        todo = [(x, by[x["line"]]) for x in r["invariants"] if x["line"] in by and len(semantic.units(by[x["line"]]["intervention"])) > 1]
        if todo:
            ck = semantic.Checker(variables.Scorer(ge), tx, sv, fresh_mean=True)
            try:
                rows = [{"holds": x["holds"]} for x, _ in todo]
                semantic.one_input_checks(ck, [(y["intervention"], y["outcome"]) for _, y in todo], rows)
            finally:
                ck.close()
            for (x, _), row in zip(todo, rows):
                x.update(alone=row["alone"], joint=row["joint"])
        r["teacher"]["invariants_version"] = INVARIANTS
    elif r["teacher"].get("invariants_version") != INVARIANTS:
        sv = [{"name": v["name"], "inputs": v["inputs"], "group": pv["group"], "values": v.get("values") or [],
               "identity": v.get("kind") == "input"} for v, pv in zip(kv, variables.parse(kv, nat.C))]
        r["invariants"] = semantic.teacher_invariants(variables.Scorer(ge), tx, sv, task["model"]) if sv else []
        r["teacher"]["invariants_version"] = INVARIANTS
    r["teacher"]["value_kinds"] = {k: sum(v.get("value_kind") == k for v in kv) for k in ("identity", "computed", "copies its inputs")}
    var_teacher.release(nat.dev)
    path.write_text(json.dumps(r, indent=1))
    log(f"{path.stem}: {len(kv)} kept, value kinds {r['teacher']['value_kinds']}, {sum(x['holds'] for x in r['invariants'])} of "
        f"{len(r['invariants'])} invariants hold")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--part", help="K/N: only every N-th file from the K-th (a cluster array's task)")
    args = ap.parse_args()
    dirs = list(args.dirs)
    if args.tar:
        import tarfile

        args.out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.tar) as t:
            t.extractall(args.out)
        dirs = sorted(x for x in args.out.iterdir() if x.is_dir())
    files = [f for d in dirs for f in sorted(x for x in d.iterdir() if x.name.startswith("text") and x.suffix == ".json")]
    if args.part:
        k, n = map(int, args.part.split("/"))
        files = files[k::n]
    ge = goal_eval.GoalEval(native.Native(args.device))
    for f in files:
        post(ge, f, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
