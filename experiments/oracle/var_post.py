"""A second pass over var_teacher.py's output files (#2951 graph oracle): what a file written before a field existed
lacks, from its kept variables alone (no carrier search): the decomposition label (value_kind: computed when a value
class holds tokens other than the text's, copies its inputs otherwise), RAVEL's Iso and Disentangle on both splits
(the off-target draws that can reach the carrier, changes before it; Cause is f on-target), and the verified INVARIANTS
lines (semantic.teacher_invariants). Files are rewritten in place (or, with --tar, a tar of the directories is unpacked
under --out first: a pod uploads one file), a file already holding "invariants" is skipped.

  experiments/oracle/var_post.py DIR [DIR ...] [--tar FILE --out DIR] [--device mps]
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


def post(ge, path: Path, draws: int = 8, log=print) -> None:
    r = json.loads(path.read_text())
    if "invariants" in r:
        return
    task = json.loads((native.TEXTS / "vpd4l" / f"{path.stem}.json").read_text())
    p = task["prompts"][0]
    tx = ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(path.stem), task=task)
    nat = ge.nat
    kv = [v for v in r["variables"] if v.get("kept")]
    for v in kv:
        if v.get("values"):
            v["value_kind"] = "computed" if any(len(vm["classes"][0]["token_ids"]) > 1 for vm in v["values"]) else "copies its inputs"
        at = v.get("carried_at") or v["inputs"]
        for split in ("search", "eval"):
            got = var_teacher.together(variables.Scorer(ge), tx, variables.parse([v], nat.C), split, draws)
            reach = [d for d in got["draws"] if d["kind"] == "off" and d["s"] is not None and d["changes"][0][0] < min(at)]
            w = sum(d["d0"] for d in reach)
            iso = sum(d["d0"] * d["s"] for d in reach) / w if w > 0 else None
            f = got["variables"][0]["f_on"]
            v["measured"].setdefault(split, {}).update(cause=f, iso=iso, iso_draws=len(reach), off_draws=got["variables"][0]["draws_off"],
                                                       disentangle=None if f is None or iso is None else (f + iso) / 2)
    sv = [{"name": v["name"], "inputs": v["inputs"], "group": pv["group"], "values": v.get("values") or []}
          for v, pv in zip(kv, variables.parse(kv, nat.C))]
    r["invariants"] = semantic.teacher_invariants(variables.Scorer(ge), tx, sv, task["model"]) if sv else []
    r["teacher"]["value_kinds"] = {k: sum(v.get("value_kind") == k for v in kv) for k in ("computed", "copies its inputs")}
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
    args = ap.parse_args()
    dirs = list(args.dirs)
    if args.tar:
        import tarfile

        args.out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.tar) as t:
            t.extractall(args.out)
        dirs = sorted(x for x in args.out.iterdir() if x.is_dir())
    ge = goal_eval.GoalEval(native.Native(args.device))
    for d in dirs:
        for f in sorted(x for x in d.iterdir() if x.name.startswith("text") and x.suffix == ".json"):
            post(ge, f, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
