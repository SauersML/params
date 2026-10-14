"""Env C's train targets in one pass per text (#2951 graph oracle): each kept variable's compact family
(var_compact.compact, fast mode), then the text's target with only the INVARIANTS lines that hold under those compact
subcomponents (env_c.reline). Two files out: the compact rows (var_compact's format) and the relined targets.

  experiments/oracle/envc_prep.py --untar INPUTS.tgz [--split train] [--part K/N] [--fast 0.11] --out-compact FILE
                                  --out-targets FILE [--device D]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import env_c  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import var_compact  # noqa: E402
import variables  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--untar", type=Path, help="a tar of input paths under ~/mpd-data, unpacked there first")
    ap.add_argument("--split", default="train")
    ap.add_argument("--part")
    ap.add_argument("--fast", type=float, default=0.11)
    ap.add_argument("--share", type=float, default=0.9)
    ap.add_argument("--out-compact", type=Path, required=True)
    ap.add_argument("--out-targets", type=Path, required=True)
    ap.add_argument("--device")
    a = ap.parse_args()
    if a.untar:
        import tarfile

        with tarfile.open(a.untar) as t:
            t.extractall(Path.home() / "mpd-data")
    V = native.TEXTS.parent / "variables"
    recs = [json.loads(x) for x in (V / "sft_targets" / f"{a.split}.jsonl").read_text().splitlines()]
    if a.part:
        k, n = map(int, a.part.split("/"))
        recs = recs[k::n]
    done = {json.loads(x)["task"] for x in a.out_targets.read_text().splitlines()} if a.out_targets.exists() else set()
    for f in (a.out_compact, a.out_targets):
        f.parent.mkdir(parents=True, exist_ok=True)
    ge = goal_eval.GoalEval(native.Native(a.device))
    for rec in recs:
        t = rec["task"]
        if t in done:
            continue
        r = json.loads((V / "influential_v2" / a.split / f"{t}.json").read_text())
        task, tx = env_c.text_of(ge, t)
        cp = {}
        for v in [x for x in r["variables"] if x.get("kept")]:
            pv = variables.parse([v], ge.nat.C)[0]
            got = var_compact.compact(variables.Scorer(ge), tx, v, pv, a.share, a.fast)
            row = {"task": t, "split": a.split, "name": v["name"], "kind": v["kind"], "value_kind": v.get("value_kind"),
                   "carrier": v["carrier"], "share": a.share, "fast": a.fast, **got}
            with a.out_compact.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            if got.get("group"):
                cp[(t, v["name"])] = got["group"]
        out = env_c.reline(ge, tx, task, rec, cp)
        with a.out_targets.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        print(t, len(cp), "compact;", out["lines_before"], "->", len(out["invariants"]), "lines", flush=True)


if __name__ == "__main__":
    main()
