"""The engine speed benchmark (the plan's engine-speed goal, monitorbench): fixed texts and circuits, the seconds every
teacher check, score and RL sample rides on, one record per run appended to ~/mpd-data/bench/engine/<device>.jsonl.

Inputs (fixed): vpd4l text9000 and text9002, VPD's answers from graph_oracle/teacher/vpd_heldout20.pt (used even where
the causal-importance network exists, so every machine runs the same circuits), text9002's teacher circuit
(graph_oracle/teacher/dev_engine_edges/text9002.json: 135 nodes, 579 connections, token edges at its token nodes'
positions and into the prediction). Each case runs twice; the second (warm) time is recorded:

  pp_run           Engine.path_patched: the teacher circuit on 16 changed texts (one circuit run x 16)
  pp_nodesets      changed_search(mode="pp") on 16 node sets of VPD's answer at the target and its 3 most influential
                   positions, 90% subsets (a block search's acceptance call)
  pp_lines         Engine.path_patched(lines=): every line deletion of the teacher circuit on 16 changed texts
  point            sufficiency (C's point run) of the teacher circuit
  track_pp         track_pp of the teacher circuit (search split)
  interchange_pp   interchange_pp of the teacher circuit (search split)
  line_loo         line_loo of the teacher circuit (search split, 8 draws), given its reward row

  engine_bench.py [--cases a,b] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[2] / "method" / "oracle"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import mech  # noqa: E402
import native  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle/teacher"
SAVED = DATA / "vpd_heldout20.pt"
CIRCUIT = DATA / "dev_engine_edges/text9002.json"
POOL = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"  # components.POOL
CASES = ("pp_run", "pp_nodesets", "pp_lines", "point", "track_pp", "interchange_pp", "line_loo")


def text(ge, name: str, saved: dict):
    """goal_eval's Text of a held-out text with VPD's saved answer as its tx.vpd."""
    nat = ge.nat
    p = native.TEXTS / "vpd4l" / f"{name}.json"
    ids, targets = native.text(p)
    tx = ge.text(ids, targets, native.task_seed(name), json.loads(p.read_text()))
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in saved[name]["answer"].items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    return tx


def teacher(ge, tx) -> native.Graph:
    """text9002's teacher circuit with token edges at its token nodes' positions and into the prediction, stranded."""
    ans = json.loads(CIRCUIT.read_text())
    g, _ = goal_eval.circuit(ans, tx.targets, ge.nat.C, tx.T)
    tp = {int(t) for m, _, t in ans["nodes"] if m == "tok"}
    g.tok_edges = [(nd[1], nd) for nd in sorted(g.node_set()) if nd[1] in tp and native._name_layer_kind(nd[0])[1] in mech.RESID_READERS]
    g.tok_out = list(tx.targets)
    return goal_eval.strand(g, tx.targets, ge.nat.C)


def timed(dev, fn) -> float:
    """fn's warm seconds (the second of two runs, synchronized)."""
    fn()
    carry.synchronize(dev)
    t0 = time.perf_counter()
    fn()
    carry.synchronize(dev)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--out", default=str(Path.home() / "mpd-data/bench/engine"))
    a = ap.parse_args()
    want = a.cases.split(",")
    ge = goal_eval.GoalEval()
    nat, dev = ge.nat, ge.nat.dev
    saved = torch.load(SAVED)
    t9000, t9002 = text(ge, "text9000", saved), text(ge, "text9002", saved)
    g = teacher(ge, t9002)
    E = ge._pp_engine(t9002)
    moved = t9002.moved
    positions = [q for q, _ in t9002.f2]
    got = ge._interchange_draws(t9002, positions, "search", 16, first={q: -moved[q - 1] for q in positions})
    rec = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else str(dev),
           "commit": subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
           "time": time.strftime("%Y-%m-%d %H:%M"), "nodes": g.count(), "connections": g.edges(), "changed_texts": len(got)}
    lines = len(E.pp_lines(g))
    rec["lines"] = lines
    with torch.no_grad():
        if "pp_run" in want:
            rec["pp_run_s"] = timed(dev, lambda: E.path_patched([g], got))
        if "pp_lines" in want:
            mask = torch.ones(1 + lines, lines)
            mask[torch.arange(1, 1 + lines), torch.arange(lines)] = 0.0
            rec["pp_lines_s"] = timed(dev, lambda: E.path_patched([g], got, lines=mask))
        if "pp_nodesets" in want:
            f2 = sorted((q for q, _ in t9000.f2), key=lambda q: -t9000.moved[q - 1])[:3]
            nodes = sorted(nd for nd in native.Graph(masks=t9000.vpd).node_set() if nd[1] in set(t9000.targets) | set(f2))
            gen = torch.Generator().manual_seed(0)
            sets = [native.Graph([nd for nd, k in zip(nodes, (torch.rand(len(nodes), generator=gen) < 0.9).tolist()) if k], complete=True)
                    for _ in range(16)]
            rec["nodeset_nodes"] = len(nodes)
            rec["pp_nodesets_s"] = timed(dev, lambda: ge.changed_search(t9000, sets, mode="pp"))
    if "point" in want:
        rec["point_s"] = timed(dev, lambda: ge.sufficiency(t9002, [goal_eval.expand_blocks(g, nat.C)]))
    if "track_pp" in want:
        rec["track_pp_s"] = timed(dev, lambda: ge.track_pp(t9002, [g], "search"))
    if "interchange_pp" in want:
        rec["interchange_pp_s"] = timed(dev, lambda: ge.interchange_pp(t9002, [g], "search"))
    if "line_loo" in want:
        pool = torch.load(POOL) if POOL.exists() else {}
        base = ge.fidelity(t9002, [g], pool, mode="reward", split="search", interchange=8, changed=8)[0]
        rec["line_loo_s"] = timed(dev, lambda: ge.line_loo(t9002, g, pool, draws=8, bits_start=1e6, base=base))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{rec['device'].replace(' ', '_')}.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
