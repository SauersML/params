"""Re-pick teacher circuits after the fact (the lead's 10-13): each text's written circuit becomes the stage of the highest
circuit term among those whose own full-fidelity row has no family under its floor (v = 0), as teacher.set_search picks
since 262897d, from the stages whose node set can be recovered: every stage when the path file keeps stage_nodes
(262897d on); before that, the written stage (TEXT.json), the start (VPD's actives at the start's positions, from the
saved answers) and any stage of the same size as either (the stages are nested). The original goes to DIR/prerepick/,
one record per text to DIR/repick.jsonl (texts already there are skipped).
  repick_teachers.py VPD_ANSWERS.pt DIR [DIR ...]   (DIR: a teacher output, summary_*.jsonl beside TEXT.json files)"""
import glob, json, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "method" / "oracle"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))
import torch  # noqa: E402
import goal_eval, native  # noqa: E402


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
    rec = {"task": row["task"], "stage": cur[0], "size": cur[3], "term": cur[6], "v": cur[7],
           "best_v0": ok[0][0] if ok else None, "best_v0_size": ok[0][3] if ok else None, "best_v0_term": ok[0][6] if ok else None,
           "best_v0_recoverable": bool(ok) and ok[0][0] in known}
    if pick is None:
        return None, {**rec, "repicked": False, "reason": "no recoverable stage at v = 0"}
    if pick[3] == cur[3] and cur[7] == 0:
        return None, {**rec, "repicked": False, "reason": "written stage already the best at v = 0"}
    return known[pick[0]], {**rec, "repicked": True, "new_stage": pick[0], "new_size": pick[3], "new_term": pick[6]}


def main():
    vpd = torch.load(sys.argv[1], weights_only=False)
    for d in map(Path, sys.argv[2:]):
        rows = {}
        for f in sorted(d.glob("summary_*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                rows[r["task"]] = r
        log = d / "repick.jsonl"
        done = {json.loads(x)["task"] for x in log.read_text().splitlines()} if log.exists() else set()
        n = changed = 0
        for task, row in sorted(rows.items()):
            circ = d / f"{task}.json"
            if task in done or not circ.exists() or "phi" not in (row.get("blocks") or {}):
                continue
            written = sorted((x[0], int(x[2]), int(x[1])) for x in json.loads(circ.read_text())["nodes"])   # [site, c, t]
            path = d / f"{task}.path.json"
            stage_nodes = (json.loads(path.read_text()).get("blocks") or {}).get("stage_nodes") if path.exists() else None
            nodes, rec = repick(row, written, (vpd.get(task) or {}).get("answer"), stage_nodes)
            if nodes is not None:
                (d / "prerepick").mkdir(exist_ok=True)
                if not (d / "prerepick" / circ.name).exists():
                    shutil.copy2(circ, d / "prerepick" / circ.name)
                circ.write_text(json.dumps(goal_eval.to_json(native.Graph(nodes, complete=True), "")))
                changed += 1
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
        print(f"{d}: {n} texts, {changed} re-picked", flush=True)


if __name__ == "__main__":
    main()
