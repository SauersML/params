"""Re-pick teacher circuits after the fact (the lead's 10-13): each text's written circuit becomes the stage of the highest
circuit term among those whose own full-fidelity row has no family under its floor (v = 0), as teacher.set_search picks
since 262897d, from the stages whose node set can be recovered: every stage when the path file keeps stage_nodes
(262897d on); before that, the written stage (TEXT.json), the start (VPD's actives at the start's positions, from the
saved answers) and any stage of the same size as either (the stages are nested). The original goes to DIR/prerepick/,
one record per text to DIR/repick.jsonl. A text is skipped while its record describes its current circuit; a rerun's
circuit (fleet's reruns replace TEXT.json) is re-picked from the summary row whose written stage it is, and its record
replaces the old one.
  repick_teachers.py VPD_ANSWERS.pt DIR [DIR ...]   (DIR: a teacher output, summary_*.jsonl and fleet's merged MATS shares'
                                                     mats_summary_*.jsonl beside TEXT.json files)"""
import glob, json, os, shutil, sys
from pathlib import Path

import torch  # noqa: E402


def to_json(nodes: list) -> dict:
    """A complete node set as goal_eval.to_json writes it (nodes [site, subcomponent, position] sorted by position, site,
    subcomponent; no edges): written here so the re-pick runs on torch alone, inside the Mac's 1 GiB lane."""
    return {"nodes": [[n, c, t] for n, t, c in sorted(nodes, key=lambda nd: (nd[1], nd[0], nd[2]))], "edges": [],
            "complete": True, "explanation": ""}


def start_nodes(answer: dict, positions: set) -> list:
    return sorted((n, int(t), int(c)) for n, (ts, cs) in answer.items() for t, c in zip(ts.tolist(), cs.tolist()) if int(t) in positions)


def repick(row: dict, written: list, answer: dict | None, stage_nodes: dict | None) -> tuple:
    """(nodes to write or None to keep, record) for one text's summary row and written circuit's nodes."""
    b = row["blocks"]
    phi = {p[0]: p for p in b["phi"]}
    cur = phi[b["phi_stage"]]
    known = {k: [tuple(x) for x in v] for k, v in (stage_nodes or {}).items()}
    if not known:
        by_size = {len(written): written}
        if answer is not None:
            s = start_nodes(answer, set(b["positions"]))
            if len(s) == phi["start"][3]:
                by_size.setdefault(len(s), s)
        known = {k: by_size[p[3]] for k, p in phi.items() if p[3] in by_size}
    ok = sorted((p for p in phi.values() if p[7] == 0), key=lambda p: -p[6])
    pick = next((p for p in ok if p[0] in known), None)
    fallback = pick is None and "start" in known             # no stage at v = 0: the start, as set_search falls back
    if fallback:                                             # (its own row can miss a floor by the estimators' noise:
        pick = phi["start"]                                  # text6007 v 0.066, text6097 0.016 under n)
    rec = {"task": row["task"], "stage": cur[0], "size": cur[3], "term": cur[6], "v": cur[7],
           "best_v0": ok[0][0] if ok else None, "best_v0_size": ok[0][3] if ok else None, "best_v0_term": ok[0][6] if ok else None,
           "best_v0_recoverable": bool(ok) and ok[0][0] in known}
    if pick is None:
        return None, {**rec, "repicked": False, "reason": "no recoverable stage at v = 0"}
    if pick[3] == cur[3]:
        return None, {**rec, "repicked": False, "reason": "no stage at v = 0: the start" if fallback
                      else "written stage already the best at v = 0"}
    return known[pick[0]], {**rec, "repicked": True, "new_stage": pick[0], "new_size": pick[3], "new_term": pick[6],
                            **({"reason": "no stage at v = 0: the start"} if fallback else {})}


def main():
    vpd = torch.load(sys.argv[1], weights_only=False, mmap=True)   # mapped: a text's answer read when re-picked (whole,
    # the 494 texts' answers held about 4 GB per process: the Mac's oomkill took a run at 12:30, 10-13)
    for d in map(Path, sys.argv[2:]):
        rows = {}                                            # every summary row per text (a rerun's beside the first)
        for f in sorted(d.glob("summary_*.jsonl")) + sorted(d.glob("mats_summary_*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                rows.setdefault(r["task"], []).append(r)
        log = d / "repick.jsonl"
        recs = {}
        for x in log.read_text().splitlines() if log.exists() else []:
            r = json.loads(x)
            recs[r["task"]] = r
        n = changed = 0
        for task, cands in sorted(rows.items()):
            circ = d / f"{task}.json"
            if not circ.exists():
                continue
            written = sorted((x[0], int(x[2]), int(x[1])) for x in json.loads(circ.read_text())["nodes"])   # [site, c, t]
            old = recs.get(task)
            if old is not None and (old["new_size"] if old["repicked"] else old["size"]) == len(written) \
                    and old.get("reason") != "no recoverable stage at v = 0":
                continue                                     # the record describes this circuit (a rerun's does not)
            # the summary row of the run that wrote this circuit: its written stage's size (the latest such row)
            row = next((r for r in reversed(cands) if "phi" in (r.get("blocks") or {})
                        and {p[0]: p for p in r["blocks"]["phi"]}[r["blocks"]["phi_stage"]][3] == len(written)), None)
            if row is None:                                  # (its run's summary not here yet: no record, not a stale one)
                recs.pop(task, None)
                continue
            path = d / f"{task}.path.json"
            stage_nodes = (json.loads(path.read_text()).get("blocks") or {}).get("stage_nodes") if path.exists() else None
            nodes, rec = repick(row, written, (vpd.get(task) or {}).get("answer"), stage_nodes)
            if nodes is not None:
                (d / "prerepick").mkdir(exist_ok=True)
                if not (d / "prerepick" / circ.name).exists():
                    shutil.copy2(circ, d / "prerepick" / circ.name)
                circ.write_text(json.dumps(to_json(nodes)))
                changed += 1
            recs[task] = rec
            n += 1
        tmp = log.with_suffix(".jsonl.tmp")                  # one record per text, a rerun's replacing the old one
        tmp.write_text("".join(json.dumps(r) + "\n" for _, r in sorted(recs.items())))
        tmp.replace(log)
        print(f"{d}: {n} texts (re)checked, {changed} re-picked", flush=True)


if __name__ == "__main__":
    main()
