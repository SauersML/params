"""Re-pick finished teachers on a held-out input scope (the lead's 10-14). A teacher's input-scope family scores its
search draws, which every removal's acceptance saw, so a pruned stage can fit them: on the dev check at f00abd2
text9012's 211-part circuit met every search floor and scored scope 0.33 on the eval draws against its floor 0.9. Here
each stage the teacher saved (TEXT.path.json, blocks.stage_nodes) is scored by goal_eval.scope_family on a third donor
pool, DONOR_POOLS["repick"] (training texts text6400-text6999, disjoint from the search split's text6000-text6399 and
from the evaluations' text9150-text9199; positions and donors drawn by the text's seed, as on the other splits). The
written circuit becomes the stage of highest circuit term among those whose search reward row has no family under its
floor (v = 0, blocks.phi) and whose held-out scope is at least teacher.floor_of the START's there, the teacher's
validation guard's floor (FLOOR_TOP 0.9 of VPD's whole answer, which scores 1 by construction, or 0.05 under the START's
own score; a text without scope draws has none), else the START, flagged. A START under FLOOR_TOP on the held-out draws
is flagged too (8 train texts at 8ce9afd: 3, START 0.68, 0.86, 0.87): its positions follow the search donors' influence,
and a stage no worse than it there lost nothing by pruning, while writing the START would add parts for no held-out
scope (text6490: 174 parts at 0.863, its START 250 at 0.861). The original goes to DIR/orig/
(TEXT.json: a sibling TEXT.orig.json would read as a circuit of its own to the pods' text*.json copies), one record per
text to DIR/scope_repick.jsonl; a text whose record names the circuit now in TEXT.json (its sha256) is skipped, so a
rerun's circuit is re-picked again.
  scope_repick.py DIR [DIR ...]   (a GPU; DIR: a teacher output, TEXT.json beside TEXT.path.json)"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "method").exists() else Path.cwd()
sys.path[:0] = [str(SRC / "method/oracle"), str(SRC / "method/vpd")]

POOL = "repick"


def pick(phi: list, scope: dict, floor: float | None) -> tuple[str, str | None]:
    """(the stage to write, the flag or None). phi: the search reward rows, [stage, phi, bar, units, families, bits,
    term, v] each (blocks.phi); scope: stage -> its held-out s_scope (None: no draws), the stages scored; floor None:
    none. The stage of highest term among those at v = 0 whose scope meets floor; none (with floor_of the START's own
    scope, only a START missing a search floor): the START, flagged."""
    ok = [p for p in phi if p[7] == 0 and p[0] in scope and (floor is None or scope[p[0]] is None or scope[p[0]] >= floor - 1e-9)]
    if ok:
        return max(ok, key=lambda p: p[6])[0], None
    return "start", "no stage meets every floor"


def to_json(nodes: list) -> dict:
    """A complete node set (matrix, position, subcomponent) as goal_eval.to_json writes it."""
    return {"nodes": [[n, c, t] for n, t, c in sorted(nodes, key=lambda nd: (nd[1], nd[0], nd[2]))], "edges": [],
            "complete": True, "explanation": ""}


def main():
    import goal_eval
    import native
    import teacher

    ge = goal_eval.GoalEval(native.Native())
    for d in map(Path, sys.argv[1:]):
        log = d / "scope_repick.jsonl"
        recs = {json.loads(x)["task"]: json.loads(x) for x in log.read_text().splitlines()} if log.exists() else {}
        n = moved = 0
        for pf in sorted(d.glob("text*.path.json")):
            task = pf.name[:-len(".path.json")]
            circ = d / f"{task}.json"
            if not circ.exists():
                continue
            sha = hashlib.sha256(circ.read_bytes()).hexdigest()
            if recs.get(task, {}).get("sha256") == sha:
                continue                                     # the record names this circuit
            b = json.loads(pf.read_text()).get("blocks") or {}
            if not b.get("stage_nodes") or not b.get("phi"):
                continue                                     # (no saved stages: nothing to re-pick from)
            t0 = time.perf_counter()
            path = native.TEXTS / "vpd4l" / f"{task}.json"
            ids, targets = native.text(path)
            tx = ge.text(ids, targets, native.task_seed(task), task=json.loads(path.read_text()))
            stages = {k: sorted((x[0], int(x[1]), int(x[2])) for x in v) for k, v in b["stage_nodes"].items()}
            sets = list(dict.fromkeys(tuple(v) for v in stages.values()))   # stages with one node set scored once
            rows = ge.scope_family(tx, [native.Graph(list(s), complete=True) for s in sets], POOL)
            by_set = {s: goal_eval.family_score(r, "scope") for s, r in zip(sets, rows)}
            scope = {k: by_set[tuple(v)] for k, v in stages.items()}
            phi = {p[0]: p for p in b["phi"]}
            written = sorted((x[0], int(x[2]), int(x[1])) for x in json.loads(circ.read_text())["nodes"])
            was = next((k for k, v in stages.items() if v == written), None)
            floor = teacher.floor_of(scope["start"])
            new, flag = pick(b["phi"], scope, floor)
            if flag is None and scope["start"] is not None and scope["start"] < goal_eval.FLOOR_TOP - 1e-9:
                flag = "the START misses the held-out scope floor"
            rec = {"task": task, "stage": was, "size": len(written), "term": phi[was][6] if was in phi else None,
                   "new_stage": new, "new_size": len(stages[new]), "new_term": phi[new][6], "flag": flag,
                   "moved": stages[new] != written, "scope": scope, "floor": floor, "pool": ge.pool_id(POOL),
                   "draws": len(rows[0]["draws_scope"]) if rows else 0, "seconds": round(time.perf_counter() - t0, 1)}
            if rec["moved"]:
                (d / "orig").mkdir(exist_ok=True)
                shutil.copy2(circ, d / "orig" / circ.name)   # (a rerun's circuit, re-picked again, replaces its copy)
                circ.write_text(json.dumps(to_json(stages[new])))
                moved += 1
            rec["sha256"] = hashlib.sha256(circ.read_bytes()).hexdigest()
            recs[task] = rec
            n += 1
            tmp = log.with_suffix(".jsonl.tmp")              # one record per text, written after each
            tmp.write_text("".join(json.dumps(r) + "\n" for _, r in sorted(recs.items())))
            tmp.replace(log)
            print(json.dumps({k: rec[k] for k in ("task", "stage", "size", "new_stage", "new_size", "flag", "scope", "seconds")}), flush=True)
            del tx
            native.free(ge.nat.dev)
        print(f"{d}: {n} texts re-picked on held-out scope, {moved} moved", flush=True)


if __name__ == "__main__":
    main()
