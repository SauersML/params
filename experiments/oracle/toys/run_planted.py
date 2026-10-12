"""The planted toys through the teacher and Φ (#2951 graph oracle, the lead's 10-12 track; planted.py builds the toys):
per toy, on its ordinary texts,
  1. the teacher arms (teacher.teach, the influential start, connections with --no-strand, acceptance as each arm's):
     which heads and MLP neurons (planted.decompose's units) and positions each circuit holds, against the planted route;
  2. Φ (goal_eval.fidelity's full mode on the eval draws, goal_eval.phi) of a circuit of the true mechanism against one of
     the other mechanism, as complete node sets of VPD's active pieces (definitions in circuits()).
Rows to OUT/<toy>.jsonl, a table printed.

  MPD_MEM_GIB=8 ~/mpd-data/venv/bin/python -B experiments/oracle/toys/run_planted.py --toys ~/mpd-data/toys_planted --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import goal_eval  # noqa: E402
import native  # noqa: E402
import planted  # noqa: E402
import teacher  # noqa: E402

ARMS = {"current": (), "changed": ("changed",)}


def units_of(nat, nodes) -> dict:
    """{(layer, kind): sorted units} and the positions of a circuit's nodes (planted.decompose's head or neuron per
    piece)."""
    out, pos = {}, set()
    for n, t, c in nodes:
        if n == "tok" or c < 0:
            continue
        layer, kind = native._name_layer_kind(n)
        out.setdefault(f"L{layer}.{kind}", set()).add(int(nat.units[n][c]))
        pos.add(int(t))
    return {k: sorted(v) for k, v in sorted(out.items())}, sorted(pos)


def active(nat, tx, keep) -> list:
    """VPD's active pieces (tx.vpd) for which keep(site, position, unit) holds."""
    return [(n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist() if keep(n, t, int(nat.units[n][c]))]


def circuits(name: str, nat, tx) -> dict:
    """The true mechanism's circuit and the other mechanism's, as complete node sets of active pieces:
      binding_*: true km = the queried pair's key and value positions and the query's (key matching); positional = its
                 value position and the query's alone; on km the positional circuit is the other mechanism, on fp the
                 key-matching one.
      route2:    both heads (true) against head 0 alone and head 1 alone (one route).
      interaction: every piece (true) against the attention pieces alone (additive evidence).
      cancel:    every piece but the two cancelling neurons (true) against those plus the +u neuron alone, and plus both."""
    T = tx.T
    kpos, vpos, last = 2 * planted.FIXED + 1, 2 * planted.FIXED + 2, T - 1
    every = lambda n, t, u: True  # noqa: E731
    kind = lambda n: native._name_layer_kind(n)[1]  # noqa: E731
    if name.startswith("binding"):
        km = active(nat, tx, lambda n, t, u: t in (kpos, vpos, T - 2, last))
        pos = active(nat, tx, lambda n, t, u: t in (vpos, last))
        return {"true": km if name == "binding_km" else pos, "other": pos if name == "binding_km" else km}
    if name == "route2":
        head = lambda h: (lambda n, t, u: kind(n) in ("q_proj", "k_proj", "v_proj", "o_proj") and u == h)  # noqa: E731
        return {"true": active(nat, tx, every), "head0 alone": active(nat, tx, head(0)), "head1 alone": active(nat, tx, head(1))}
    if name == "interaction":
        return {"true": active(nat, tx, every), "attention alone": active(nat, tx, lambda n, t, u: kind(n) not in ("c_fc", "down_proj"))}
    if name == "cancel":
        mlp = lambda n: kind(n) in ("c_fc", "down_proj")  # noqa: E731
        return {"true": active(nat, tx, lambda n, t, u: not (mlp(n) and u in (0, 1))),
                "with +u neuron": active(nat, tx, lambda n, t, u: not (mlp(n) and u == 1)),
                "with both": active(nat, tx, every)}
    return {"true": active(nat, tx, every)}


def run(name: str, path: Path, out: Path, n_texts: int, arms: list, log=print) -> None:
    d = torch.load(path)
    nat = planted.native_of(d["sd"], d["cfg"])
    ge = goal_eval.GoalEval(nat, chunk=16)
    ge.donor_pools.update(search=[], eval=[])   # the far donors are vpd4l's texts, outside a toy's vocabulary
    texts = planted.ordinary(name, n_texts)
    texts = list(dict.fromkeys(tuple(x) for x in texts))  # the gate toys have three distinct texts
    with (out / f"{name}.jsonl").open("a") as fh:
        for i, ids in enumerate(texts):
            ids = list(ids)
            targets = [len(ids) - 1]
            tx = ge.text(ids, targets)
            for arm in arms:
                t0 = time.perf_counter()
                try:
                    circ, row, _ = teacher.teach(ge, ids, targets, 16, f1_start="influential", edges=True, no_strand=True,
                                                 max_nodes=None, tokens=None, accept=ARMS[arm], accept_mode="pp")
                    nodes = [(n, int(t), int(c)) for n, c, t in circ[""]["nodes"] if n not in ("tok", "block")]
                    blocks = [x for x in circ[""]["nodes"] if x[0] == "block"]
                    u, pos = units_of(nat, nodes)
                    rec = {"toy": name, "text": i, "ids": ids, "arm": arm, "units": u, "positions": pos, "blocks": blocks,
                           "nodes": len(nodes), "edges": len(circ[""].get("edges") or []), "seconds": time.perf_counter() - t0,
                           "f1": row.get("best", {}).get("f1"), "n": row.get("best", {}).get("n")}
                except Exception as e:  # noqa: BLE001
                    rec = {"toy": name, "text": i, "ids": ids, "arm": arm, "error": f"{type(e).__name__}: {e}"}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                log(f"{name} text {i} {arm}: {json.dumps({k: rec.get(k) for k in ('units', 'positions', 'blocks', 'nodes', 'f1', 'n', 'error')})}")
            cs = circuits(name, nat, ge.text(ids, targets))
            graphs = [native.Graph(v, complete=True) for v in cs.values()]
            try:
                rows = ge.fidelity(tx, graphs, None, mode="full", split="eval")
            except Exception as e:  # noqa: BLE001
                rows = [{"error": f"{type(e).__name__}: {e}"}] * len(graphs)
            for (label, v), r in zip(cs.items(), rows):
                ph = goal_eval.phi(r)[0] if "error" not in r else None
                rec = {"toy": name, "text": i, "circuit": label, "nodes": len(v), "phi": ph,
                       **{k: r.get(k) for k in ("s_point_c", "n_point", "s_track_pp", "s_interchange_pp", "s_nodes", "s_edges", "c", "error")}}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                log(f"{name} text {i} Φ {label}: {json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items() if k not in ('toy', 'text')})}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toys", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="binding_km,binding_fp,route1,route2,additive,interaction,cancel,nothing")
    ap.add_argument("--texts", type=int, default=6)
    ap.add_argument("--arms", default="current,changed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        run(name, args.toys / f"{name}.pt", args.out, args.texts, args.arms.split(","))


if __name__ == "__main__":
    main()
