"""Edit faithfulness under two fixes (#2951 graph oracle, the lead's 10-14): removing a circuit's top 3 parts is
mispredicted by about 3 bits (C's semantics) to 6 bits (the node family's) against the real model's drop, and VPD's
whole answer as badly (circuit_scorecard's edits on dev_6a9e1d3), so the fixed masks miss the parts the real model
switches on after an edit. Per circuit, on the scorecard's removals (each part alone, the 3 and 5 with the largest real
drops together; circuit_scorecard.edits, whose numbers are recomputed here as the baseline):
  backup   the circuit plus backup.expand_circuit's compensators (blocks: a site's whole output at a position; extra
           subcomponents), its prediction in C's semantics and the node family's (circuit_scorecard.edit_runs on the
           expanded node set, a block as every subcomponent of its site there); the parts it adds
  live     VPD's causal-importance gates recomputed on the real model's run less the removal (every site's input on
           that run through VPD's global causal-importance network, a gate on where it is above zero), in C's
           semantics (everything off but the run's own parts, the remainders live): VPD whole live, every gate on;
           circuit live, the circuit's parts the gates keep on plus the parts they switch on that were off on the text;
           circuit plus, the whole circuit plus those switched-on parts; circuit fixed, the circuit's node set alone
           (the check: circuit_c of a node set). The switched-on and switched-off counts per removal.
Per variant the top-3 and top-5 group error (predicted minus real drop, bits), Spearman's rho and the median absolute
error over the single removals. Needs the causal-importance network (VPD's checkpoint at vpd_model.VPD_PTH).

  edit_live.py CIRCUIT.json [...] --out OUT [--vpd FILE] [--offset K --stride S]
  edit_live.py table OUT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
sys.path.insert(0, str(HERE.parents[1] / "method" / "vpd"))

import backup  # noqa: E402
import circuit_scorecard as cs  # noqa: E402
import engine  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402

ROWS = 4  # removals per batch (every subcomponent's mask: ~130 MB a row at T 367)
LIVE = ("vpd_live", "circuit_live", "circuit_plus", "circuit_fixed")


@torch.no_grad()
def gates(nat: native.Native, ids: list[int], sets: list[list]) -> list[dict]:
    """Per removal set, VPD's causal importance above zero ({site: bool [T, C]}) on the real model's run less the set
    (every subcomponent at 1 but the set's, the remainders at 1), VPD's network reading every site's input there."""
    vpd, T, dev = nat.vpd, len(ids), nat.dev
    B = len(sets)
    vpd.clear()
    for n in nat.names:
        st = nat.target.site(n)
        st.mask, st.delta_mask, st.cache_input = torch.ones(B, T, nat.C[n], device=dev), torch.ones(B, T, device=dev), True
    for r, s in enumerate(sets):
        for n, t, c in s:
            nat.target.site(n).mask[r, t, c] = 0.0
    try:
        nat.target(torch.tensor([ids], device=dev).expand(B, -1))
        acts = {n: nat.target.site(n).last_input for n in nat.names}
    finally:
        vpd.clear()
    pre = vpd.ci_fn(acts)
    return [{n: pre[n][r] > 0 for n in nat.names} for r in range(B)]


@torch.no_grad()
def live(ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, sets: list[list]) -> dict:
    """log2 p of the prediction at the targets, summed, per removal set (row 0: none) for LIVE's runs, and the parts
    the gates switch on (off on the text) and the circuit parts they switch off, per set."""
    nat, T, dev = ge.nat, tx.T, ge.nat.dev
    E = engine.Engine(nat, tx.ids, tx.targets, live=True)  # every column, the outside off, the remainders live
    y = tx.ref[0].argmax(-1)
    cm = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
    for n, t, c in nodes:
        cm[n][t, c] = True
    rows = [[]] + [list(s) for s in sets]
    g0 = gates(nat, tx.ids, [[]])[0]
    out = {k: [] for k in LIVE}
    upto = max(tx.targets) + 1  # (the saved answer stops at the last target)
    out.update(on=[], off=[], clean_vs_saved=int(sum(int((g0[n][:upto] != tx.vpd[n][:upto]).sum()) for n in nat.names)))
    for s0 in range(0, len(rows), ROWS):
        part = rows[s0:s0 + ROWS]
        gl = gates(nat, tx.ids, part)
        drop = {n: torch.ones(len(part), T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
        for r, s in enumerate(part):
            for n, t, c in s:
                drop[n][r, t, c] = False
        new = [{n: g[n] & ~g0[n] for n in nat.names} for g in gl]
        out["on"] += [int(sum(int(x[n].sum()) for n in nat.names)) for x in new]
        out["off"] += [int(sum(int((cm[n] & ~g[n]).sum()) for n in nat.names)) for g in gl]
        masks = {"vpd_live": lambda r, n: gl[r][n],
                 "circuit_live": lambda r, n: (cm[n] & gl[r][n]) | new[r][n],
                 "circuit_plus": lambda r, n: cm[n] | new[r][n],
                 "circuit_fixed": lambda r, n: cm[n]}
        for k, f in masks.items():
            m = {n: torch.stack([f(r, n) for r in range(len(part))]) & drop[n] for n in nat.names}
            lq = E.log_probs({n: v.float() for n, v in m.items()})
            out[k] += (lq.float().gather(-1, y.view(1, -1, 1).expand(lq.shape[0], -1, 1)).squeeze(-1).sum(-1) / cs.LN2).tolist()
        del gl, new, drop
        native.free(dev)
    return out


def group_stats(pred: list, real: list, n1: int) -> dict:
    st = cs.faith(pred[:n1], real[:n1])
    for j, name in enumerate(("top3", "top5")):
        st[name] = {"predicted": pred[n1 + j], "real": real[n1 + j], "error": pred[n1 + j] - real[n1 + j]}
    return st


def run_one(ge: goal_eval.GoalEval, path: Path, vpd: dict | None, log) -> dict:
    nat = ge.nat
    t0 = time.perf_counter()
    tp = cs.task_path(path.stem)
    task = json.loads(tp.read_text())
    tx = lines.text_of(ge, tp, vpd)
    g = goal_eval.load_answer(path, task, tx.targets, nat)[0]
    base = cs.edits(ge, tx, g)                          # the scorecard's numbers, recomputed here
    nodes = [tuple(nd) for nd in base["nodes"]]
    n1 = len(nodes)
    at = {nd: i for i, nd in enumerate(nodes)}
    top = [nodes[i] for i in np.argsort(base["drops"]["model"][:n1])[::-1][:5]]
    sets = [[nd] for nd in nodes] + [top[:3], top[:5]]
    assert [cs.label(nd) for nd in top[:3]] == base["groups"]["top3"]
    real = base["drops"]["model"]
    rec = {"task": path.stem, "circuit": str(path), "parts": n1, "complete": g.complete, "top3": base["groups"]["top3"],
           "top5": base["groups"]["top5"], "real": real, "variants": {}}
    for k in ("circuit_c", "circuit_n", "vpd_whole_c", "vpd_whole_n"):
        rec["variants"][k] = group_stats(base["drops"][k], real, n1)
    log(f"  {path.stem}: baseline edits, {time.perf_counter() - t0:.0f} s")

    t1 = time.perf_counter()
    gx, brec = backup.expand_circuit(ge, tx, native.Graph(nodes, complete=True), log=None)
    blocks = sorted((n, t) for n, t, c in gx.node_set() if c == goal_eval.BLOCK)
    extra = sorted(nd for nd in gx.node_set() if nd[2] != goal_eval.BLOCK and nd not in at)
    full = goal_eval.expand_blocks(gx, nat.C)        # a block: every subcomponent of its site there (the remainder is live)
    runs = cs.edit_runs(ge, tx, native.Graph(sorted(full.node_set()), complete=True), {}, sets)
    for sem in ("c", "n"):
        d = [runs[f"circuit_{sem}"][0] - x for x in runs[f"circuit_{sem}"][1:]]
        rec["variants"][f"backup_{sem}"] = group_stats(d, real, n1)
    rec["backup"] = {"blocks": [f"{cs.label((n, t, 0)).split('#')[0]}@{t}" for n, t in blocks], "extra": [cs.label(nd) for nd in extra],
                     "added_subcomponents": len(full.node_set()) - n1, "chosen": brec["chosen"], "start": brec["start"],
                     "final": brec["final"], "seconds": time.perf_counter() - t1}
    log(f"  {path.stem}: backup {len(blocks)} blocks + {len(extra)} subcomponents, {time.perf_counter() - t1:.0f} s")

    t2 = time.perf_counter()
    lv = live(ge, tx, nodes, sets)
    for k in LIVE:
        rec["variants"][k] = group_stats([lv[k][0] - x for x in lv[k][1:]], real, n1)
    rec["switched"] = {"on_median_single": float(np.median(lv["on"][1:n1 + 1])), "on_top3": lv["on"][n1 + 1],
                       "on_top5": lv["on"][n1 + 2], "off_median_single": float(np.median(lv["off"][1:n1 + 1])),
                       "off_top3": lv["off"][n1 + 1], "off_top5": lv["off"][n1 + 2], "clean_vs_saved": lv["clean_vs_saved"]}
    rec["seconds"] = time.perf_counter() - t0
    log(f"  {path.stem}: live gates, {time.perf_counter() - t2:.0f} s")
    return rec


def table(recs: list[dict]) -> str:
    keys = ["circuit_c", "circuit_n", "vpd_whole_c", "vpd_whole_n", "backup_c", "backup_n", *LIVE]
    out = [f"{'variant':16s} {'top3 |err|':>11s} {'top5 |err|':>11s} {'rho':>6s} {'med |err|':>10s}   (medians over {len(recs)} circuits)"]
    for k in keys:
        rs = [r["variants"][k] for r in recs if k in r["variants"]]
        if not rs:
            continue
        med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else float("nan")  # noqa: E731
        out.append(f"{k:16s} {med([abs(s['top3']['error']) for s in rs]):11.2f} {med([abs(s['top5']['error']) for s in rs]):11.2f} "
                   f"{med([s['spearman'] for s in rs]):6.2f} {med([s['median_abs_error'] for s in rs]):10.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--vpd", type=Path, nargs="*", default=[])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    if str(a.paths[0]) == "table":
        print(table([json.loads(p.read_text()) for p in sorted(a.paths[1].glob("text*.json"))]))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    vpd = {k: v for f in a.vpd for k, v in torch.load(f).items()} or None
    ge = goal_eval.GoalEval(native.Native(), chunk=32)
    if not ge.nat.has_importance:
        raise SystemExit("live gates need VPD's causal-importance network (vpd_model.VPD_PTH)")
    for p in a.paths[a.offset::a.stride]:
        dest = a.out / f"{p.stem}.json"
        if not dest.exists():
            rec = run_one(ge, p, vpd, lambda *x: print(*x, flush=True))
            dest.write_text(json.dumps(rec))
            print(f"{p.stem}: {rec['seconds']:.0f} s", flush=True)
    print(table([json.loads(p.read_text()) for p in sorted(a.out.glob("text*.json"))]))


if __name__ == "__main__":
    main()
