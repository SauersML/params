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


class ToyEval(goal_eval.GoalEval):
    """GoalEval with VPD's bound held above 0: an exact decomposition's KL_VPD is ~1e-6 and can round below 0, which
    teacher.py's budget KL_empty^0.1 KL_VPD^0.9 turns complex."""

    def _bounds(self, tx, conv):
        e, v = super()._bounds(tx, conv)
        return e, (None if v is None else max(v, 1e-9))


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


def toy_texts(name: str, out: Path, n: int, seed: int) -> list[Path]:
    """n texts of the toy's training distribution as text files (native.text's format), for goal_eval.build_patterns."""
    import vpd_toy

    d = out / "texts" / name
    d.mkdir(parents=True, exist_ok=True)
    ids = next(vpd_toy.loader(name, n, seed)).tolist()
    paths = []
    for i, x in enumerate(ids):
        p = d / f"t{seed}_{i}.json"
        p.write_text(json.dumps({"prompts": [{"token_ids": x, "target_positions": [len(x) - 1]}]}))
        paths.append(p)
    return paths


def phi_without_n(r: dict):
    return goal_eval.phi({**r, "n_point": None})[0] if "error" not in r else None


def reward_v3(ge, tx, name: str, i: int, node_sets: list, pool: dict) -> list[dict]:
    """components.reward (v3, the reward RL uses) of complete node sets on one toy text, its circuit_reward = phi + g x
    0.5 compact (no teacher reference, so D = 0; g the smooth gate 1 / (1 + 2^(-4 (phi - phi_tau)))), on the eval draws
    with the toy's pattern pool. reward's answer cap (NODE_CAP, the explainer's token budget) is lifted: the toys' exact
    pieces put whole-mechanism circuits above 266 nodes."""
    import components

    task = {"id": f"{name}_{i}", "model": "toy", "prompts": [{"token_ids": tx.ids, "target_positions": tx.targets, "model_top": [[["", 0.0]]]}]}
    texts = [components.render(goal_eval.to_json(native.Graph(v, complete=True))) for v in node_sets]
    cap = components.NODE_CAP
    components.NODE_CAP = 10 ** 9
    try:
        got = components.reward(ge, task, {"qa": []}, texts, pool=pool, split="eval", cache={task["id"]: tx})
    finally:
        components.NODE_CAP = cap
    return [{k: r.get(k) for k in ("circuit_reward", "phi", "phi_tau", "g", "compact", "bits", "bits_start", "valid")} for r in got]


def run(name: str, path: Path, out: Path, n_texts: int, arms: list, decomp: str = "exact", log=print) -> None:
    """arms: the teacher arms to run (none: Φ and the v3 reward of the true and other circuits alone)."""
    d = torch.load(path)
    if decomp == "vpd":
        import vpd_toy

        nat = vpd_toy.native_of_fit(d["sd"], d["cfg"], torch.load(path.with_name(f"{path.stem.replace('nothing', 'route1')}.vpd.pt")))
    else:
        nat = planted.native_of(d["sd"], d["cfg"])
    ge = ToyEval(nat, chunk=16)
    # far donors: other texts of the toy's own distribution (vpd4l's are outside its vocabulary); the flips' pool: VPD's
    # active sets on the toy's training distribution (goal_eval.build_patterns)
    far = [(f"far{i}", x) for i, x in enumerate(next(__import__("vpd_toy").loader(name, 16, 99)).tolist())]
    ge.donor_pools.update(search=far[:8], eval=far[8:])
    pool = goal_eval.build_patterns(toy_texts(name, out, 48, 7), out / f"{name}.{decomp}.pool.pt", nat)
    texts = planted.ordinary(name, n_texts)
    texts = list(dict.fromkeys(tuple(x) for x in texts))  # the gate toys have three distinct texts
    with (out / f"{name}.jsonl").open("a") as fh:
        for i, ids in enumerate(texts):
            ids = list(ids)
            targets = [len(ids) - 1]
            tx = ge.text(ids, targets)
            for arm in [a for a in arms if a]:
                t0 = time.perf_counter()
                try:
                    circ, row, _ = teacher.teach(ge, ids, targets, 16, f1_start="influential", edges=True, no_strand=True,
                                                 max_nodes=None, tokens=None, accept=ARMS[arm], accept_mode="pp")
                    nodes = [(n, int(t), int(c)) for n, c, t in circ[""]["nodes"] if n not in ("tok", "block")]
                    blocks = [x for x in circ[""]["nodes"] if x[0] == "block"]
                    u, pos = units_of(nat, nodes)
                    rec = {"toy": name, "text": i, "decomp": decomp, "ids": ids, "arm": arm, "units": u, "positions": pos, "blocks": blocks,
                           "nodes": len(nodes), "edges": len(circ[""].get("edges") or []), "seconds": time.perf_counter() - t0,
                           "f1": row.get("best", {}).get("f1"), "n": row.get("best", {}).get("n")}
                except Exception as e:  # noqa: BLE001
                    rec = {"toy": name, "text": i, "ids": ids, "arm": arm, "error": f"{type(e).__name__}: {e}"}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                log(f"{name} text {i} {arm}: {json.dumps({k: rec.get(k) for k in ('units', 'positions', 'blocks', 'nodes', 'f1', 'n', 'error')})}")
            cs = circuits(name, nat, ge.text(ids, targets))
            cs["start"] = sorted(ge.start_set(tx, "search")["graph"].node_set())   # the influential start: phi_tau and bits_start
            graphs = [native.Graph(v, complete=True) for v in cs.values()]
            P, S = nat.positions(targets), sum(nat.C.values())
            bits = [goal_eval.size_bits(g, P, S, targets, active=tx.vpd) for g in graphs]
            try:
                rows = ge.fidelity(tx, graphs, pool, mode="full", split="eval")
            except Exception as e:  # noqa: BLE001
                rows = [{"error": f"{type(e).__name__}: {e}"}] * len(graphs)
            v3 = reward_v3(ge, tx, name, i, [v for k, v in cs.items() if k != "start"], pool)
            for (label, v), r, b in zip(cs.items(), rows, bits):
                ph = goal_eval.phi(r)[0] if "error" not in r else None
                rv = v3[list(cs).index(label)] if label != "start" else {}
                r = {**r, "bits": b, "compact": rv.get("compact"), "phi_tau": rv.get("phi_tau"), "gate": rv.get("g"),
                     "reward_v3": rv.get("circuit_reward"), "phi_reward_mode": rv.get("phi")}
                rec = {"toy": name, "text": i, "decomp": decomp, "circuit": label, "nodes": len(v), "phi": ph, "phi_without_n": phi_without_n(r),
                       **{k: r.get(k) for k in ("s_point_c", "n_point", "s_track_pp", "s_interchange_pp", "s_nodes", "s_edges",
                                                "s_donor_near", "s_donor_far", "c", "error", "bits", "compact", "phi_tau", "gate", "reward_v3",
                                                "phi_reward_mode")},
                       "f1_flips_c": (r.get("f1_flips_c") or {}), "flips_to_half": r.get("flips_to_half")}
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
    ap.add_argument("--decomp", choices=("exact", "vpd"), default="exact", help="exact: per-head and per-neuron pieces; vpd: vpd_toy.py's fit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        run(name, args.toys / f"{name}.pt", args.out, args.texts, args.arms.split(","), args.decomp)


if __name__ == "__main__":
    main()
