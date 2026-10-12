"""A fine-tuning baseline for the edit questions (#2951 graph oracle, E3; the lead's request after AxBench): does a
rank-r weight update fitted by gradient beat the edit teacher's subcomponent edits on net bits?

For each held-out goal (the teacher's file, edit_eval's goals) a low-rank update B A^T is added to one weight matrix W
[d_out, d_in] (its output gains (x A) B^T), applied at the positions the teacher's edit acts on (a local edit, as the
teacher's are: other texts are untouched), r = the number of the teacher's operations (each scales one rank-one
slice). A and B are fitted by Adam to the goal's net bits on the search's data (edit_eval's definitions: goal_bits
capped at the goal minus the KL on everything the goal does not name; detach on the search's alternative tokens), then
scored on the score's data (detach on the held-out alternatives). Sites: the matrix of the teacher's first operation,
and the best of all 24 after a short fit on each (the best by search net, refined).

E3_loc (localization): at a site, the same fit restricted to the circuit's subspace there (the span of the read
directions v and of the write directions u of the subcomponents the teacher circuit names at the edit's positions),
divided by the unconstrained fit's net; controls the same restriction to as many of VPD's top subcomponents there and
to as many random VPD-active ones outside the circuit; and the unconstrained update's alignment with each set's
directions (the largest cosines, the share of its norm in their span). Sites: the teacher's own, and the circuit's
site where the unconstrained short fit (run's screen) was best. loc-sets needs VPD's causal importance (the Mac); loc
needs only the subcomponents (a pod).

  finetune_edit.py run TEACH.jsonl OUT.jsonl [--steps 150] [--screen 30] [--device cpu] [--offset K --stride S]
  finetune_edit.py table OUT.jsonl
  finetune_edit.py loc-sets TEACH.jsonl RUN_OUT.jsonl CIRCUITS_DIR SETS.jsonl
  finetune_edit.py loc TEACH.jsonl SETS.jsonl OUT.jsonl [--steps 150] [--offset K --stride S]
  finetune_edit.py loc-table OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "oracle"))
sys.path.insert(0, str(ROOT / "method" / "vpd"))

import edit_eval  # noqa: E402
import native  # noqa: E402

LN2 = math.log(2)


class Delta:
    """W's output gains (x A) B^T at the positions where mask is 1, while inside the block."""

    def __init__(self, site, A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor):
        self.site, self.A, self.B, self.mask = site, A, B, mask

    def __enter__(self):
        self.orig = self.site._forward
        A, B, mask, orig = self.A, self.B, self.mask, self.orig
        self.site._forward = lambda x: orig(x) + ((x @ A) @ B.T) * mask[None, :, None]
        return self

    def __exit__(self, *exc):
        self.site._forward = self.orig


def logq(nat, ids: list[list[int]], site=None, A=None, B=None, mask=None) -> torch.Tensor:
    """log q (natural) [len(ids), T, V] of the model, with the update when site is given."""
    tg = nat.target
    x = torch.tensor(ids, device=nat.dev)
    if site is None:
        return torch.log_softmax(tg.hidden(x) @ tg.wte.T, -1)
    with Delta(tg.site(site), A, B, mask):
        return torch.log_softmax(tg.hidden(x) @ tg.wte.T, -1)


def kl_bits(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    return (lp.exp() * (lp - lq)).sum(-1) / LN2


def kept_kl(lp: torch.Tensor, lq: torch.Tensor, named: list[int]) -> torch.Tensor:
    if not named:
        return kl_bits(lp, lq)
    keep = torch.ones(lp.shape[-1], dtype=torch.bool, device=lp.device)
    keep[named] = False
    a = torch.log_softmax(lp.masked_fill(~keep, -math.inf), -1)
    b = torch.log_softmax(lq.masked_fill(~keep, -math.inf), -1)
    return (a.exp() * (a - b)).masked_fill(~keep, 0).sum(-1) / LN2


def margin(lq: torch.Tensor, tok: int) -> torch.Tensor:
    other = lq.masked_fill(torch.nn.functional.one_hot(torch.tensor(tok, device=lq.device), lq.shape[-1]).bool(), -math.inf)
    return (lq[tok] - other.max()) / LN2


def net_of(ee, task, goal, lq_all: torch.Tensor, lp_all: torch.Tensor, alts_q=None, alts_p=None) -> dict:
    """edit_eval's goal_bits, collateral and net (differentiable) from the edited and unedited runs of the text (and of
    the alternatives for detach)."""
    p = task["prompts"][0]
    targets = p["target_positions"]
    t = max(targets)
    lq, lp = lq_all[0], lp_all[0]
    kl = kl_bits(lp, lq)
    named = [goal["token_id"]] if goal["kind"] in ("flip", "raise") else []
    coll = kl.sum() - kl[targets].sum() + sum(kept_kl(lp[x], lq[x], named) for x in targets)
    if goal["kind"] in ("flip", "raise"):
        sign = 1.0 if goal["kind"] == "flip" else -1.0
        m0, m = sign * margin(lp[t], goal["token_id"]), sign * margin(lq[t], goal["token_id"])
        gb = torch.clamp(torch.minimum(m0, m0 - m), min=0.0)
        attained = bool(m < 0)
    else:
        d0 = torch.stack([kl_bits(lp[t], a[t]) for a in alts_p]).mean()
        d = torch.stack([kl_bits(lq[t], a[t]) for a in alts_q]).mean()
        gb = torch.clamp(d0 - torch.maximum(d, (1 - edit_eval.GOAL) * d0), min=0.0)
        attained = bool(d <= (1 - edit_eval.GOAL) * d0)
    return {"net": gb - coll, "goal_bits": gb, "collateral": coll, "attained": attained}


def fit(ee, task, goal, site: str, positions: list[int], rank: int, steps: int, lr: float = 0.01, seed: int = 0,
        basis: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
    """A and B fitted by Adam to the search's net, then scored on the score's data. With basis (Vk [d_in, k], Uk
    [d_out, k]) the update is restricted to it: A = Vk a, B = Uk b (a, b [k, rank]). The row's "update" holds the
    fitted (A, B)."""
    nat = ee.nat
    p = task["prompts"][0]
    ids = p["token_ids"]
    T = len(ids)
    st = nat.target.site(site)
    d_out, d_in = st.W.shape
    g = torch.Generator().manual_seed(seed)
    if basis is None:
        a = (torch.randn(d_in, rank, generator=g) / math.sqrt(d_in)).to(nat.dev).requires_grad_(True)
        b = torch.zeros(d_out, rank, device=nat.dev, requires_grad=True)
        AB = lambda: (a, b)  # noqa: E731
    else:
        Vk, Uk = basis
        k = Vk.shape[1]
        a = (torch.randn(k, rank, generator=g) / math.sqrt(k)).to(nat.dev).requires_grad_(True)
        b = torch.zeros(k, rank, device=nat.dev, requires_grad=True)
        AB = lambda: (Vk @ a, Uk @ b)  # noqa: E731
    mask = torch.zeros(T, device=nat.dev)
    mask[positions] = 1.0

    def data(ranks):
        seqs = [ids] + (ee.alternatives(ids, goal["position"], ranks) if goal["kind"] == "detach" else [])
        with torch.no_grad():
            ref = logq(nat, seqs)
        return seqs, ref

    seqs, ref = data(edit_eval.SEARCH_RANKS)
    # restricted, each step moves A and B as far as the unconstrained fit's do (Adam's steps are ~lr per coordinate)
    la, lb = (lr, lr) if basis is None else (lr * math.sqrt(d_in / k), lr * math.sqrt(d_out / k))
    opt = torch.optim.Adam([{"params": [a], "lr": la}, {"params": [b], "lr": lb}])
    best = (-math.inf, None, -1)
    for step in range(steps):
        A, B = AB()
        lq = logq(nat, seqs, site, A, B, mask)
        r = net_of(ee, task, goal, lq[:1], ref[:1], lq[1:], ref[1:])
        if float(r["net"].detach()) > best[0]:
            best = (float(r["net"].detach()), (A.detach().clone(), B.detach().clone()), step)
        opt.zero_grad()
        (-r["net"]).backward()
        opt.step()
    A1, B1 = best[1]
    seqs, ref = data(edit_eval.SCORE_RANKS)
    with torch.no_grad():
        lq = logq(nat, seqs, site, A1, B1, mask)
        r = net_of(ee, task, goal, lq[:1], ref[:1], lq[1:], ref[1:])
    return {"site": site, "rank": rank, "positions": positions, "search_net": best[0], "best_step": best[2], "net": float(r["net"]),
            "goal_bits": float(r["goal_bits"]), "collateral": float(r["collateral"]), "attained": r["attained"],
            "params": rank * (d_in + d_out) if basis is None else 2 * rank * basis[0].shape[1], "update": (A1, B1)}


def teacher_sites(row: dict) -> tuple[str, list[int], int]:
    """(the site of the teacher's first operation, the positions its edit acts on, its operation count)."""
    ops = [edit_eval._from_json(o) for o in row["teacher"]["edit"]]
    pos, site = set(), None
    for o in ops:
        nd = o["node"] if o["op"] == "scale" else (o["reader"] if o["reader"] != "out" else o["writer"])
        site = site or nd[0]
        pos.add(nd[1])
    return site, sorted(pos), max(1, len(ops))


def run_goal(ee, row: dict, steps: int, screen: int, log=print) -> dict:
    task = json.loads((native.TEXTS / "vpd4l" / f"{row['task']}.json").read_text())
    site, positions, rank = teacher_sites(row)
    t0 = time.time()
    own = _plain(fit(ee, task, row["goal"], site, positions, rank, steps))
    shots = {n: fit(ee, task, row["goal"], n, positions, rank, screen) for n in ee.nat.names}
    top = max(shots, key=lambda n: shots[n]["search_net"])
    best = _plain(fit(ee, task, row["goal"], top, positions, rank, steps))
    out = {"task": row["task"], "goal": row["goal"], "teacher": row["teacher"]["score"], "units": row["units"]["score"],
           "own_site": own, "best_site": best, "screen": {n: s["search_net"] for n, s in shots.items()}, "seconds": time.time() - t0}
    log(f"{row['task']} {row['goal']['kind']}: teacher net {row['teacher']['score']['net']:.3f}, fine-tune at {site} "
        f"{own['net']:.3f}, at the best site {top} {best['net']:.3f} ({time.time() - t0:.0f} s)")
    return out


def _plain(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "update"}


def basis(nat, site: str, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(the unit read directions v [d_in, k], the unit write directions u [d_out, k]) of the site's subcomponents idx."""
    st = nat.target.site(site)
    V, U = st.V[:, idx], st.U[idx].T
    return V / V.norm(dim=0, keepdim=True), U / U.norm(dim=0, keepdim=True)


def alignment(nat, site: str, A: torch.Tensor, B: torch.Tensor, idx: list[int]) -> dict:
    """How the update (x A) B^T lines up with the subcomponents idx: the largest |cosine| of its top read direction
    (right singular vector) with their v's and of its top write direction (left) with their u's, and the share of its
    squared norm inside span(u) x span(v), inside the reads' span, inside the writes' span."""
    if not idx:
        return {"k": 0}
    Vk, Uk = basis(nat, site, idx)
    dW = B @ A.T  # [d_out, d_in]
    Uw, _, Vh = torch.linalg.svd(dW.double().cpu(), full_matrices=False)
    r_top, w_top = Vh[0].to(Vk), Uw[:, 0].to(Uk)
    Ov, Ou = torch.linalg.qr(Vk)[0], torch.linalg.qr(Uk)[0]
    tot = float(dW.pow(2).sum())
    return {"k": len(idx), "cos_read": float((r_top @ Vk).abs().max()), "cos_write": float((w_top @ Uk).abs().max()),
            "share": float((Ou.T @ dW @ Ov).pow(2).sum()) / tot, "share_read": float((dW @ Ov).pow(2).sum()) / tot,
            "share_write": float((Ou.T @ dW).pow(2).sum()) / tot}


def loc_sets(ee, row: dict, circuit: dict, screen: dict) -> dict:
    """For one goal, run where VPD's causal importance is loaded: the sites E3_loc fits at (the teacher's own and the
    circuit site, among those where the circuit names a subcomponent at the edit's positions, whose unconstrained
    short fit was best) and at each the index sets: the circuit's subcomponents at those positions, and the VPD-active
    ones there in VPD's order (native._vpd_order: importance x write)."""
    task = json.loads((native.TEXTS / "vpd4l" / f"{row['task']}.json").read_text())
    p = task["prompts"][0]
    own, positions, rank = teacher_sites(row)
    named = {}
    for m, c, t in circuit["nodes"]:
        if m != "tok" and t in positions:
            named.setdefault(m, []).append(int(c))
    named = {m: sorted(set(v)) for m, v in named.items()}
    order = ee.nat._vpd_order(p["token_ids"], p["target_positions"])
    sites = [own] + ([max(named, key=lambda n: screen.get(n, -math.inf))] if named else [])
    out = {}
    for n in dict.fromkeys(sites):
        active = list(dict.fromkeys(c for m, t, c in order if m == n and t in positions))
        out[n] = {"circuit": named.get(n, []), "active": active}
    return {"task": row["task"], "goal": row["goal"], "own": own, "positions": positions, "rank": rank, "sites": out}


LOC_DRAWS = 3


def run_loc(ee, row: dict, sets: dict, steps: int, log=print) -> dict:
    """E3_loc for one goal at each of its sites: the unconstrained rank-r fine-tune, the same restricted to the
    circuit's subcomponents there (their v's and u's), to as many of VPD's top ones, and to as many random VPD-active
    ones outside the circuit and as many random subcomponents of the site (LOC_DRAWS draws each; fewer active ones
when the circuit holds most); and the unconstrained update's alignment with the circuit's, the top
    ones', other active ones' and random subcomponents (any of the site's) directions."""
    nat = ee.nat
    task = json.loads((native.TEXTS / "vpd4l" / f"{row['task']}.json").read_text())
    goal, positions, rank = row["goal"], sets["positions"], sets["rank"]
    t0 = time.time()
    out = {"task": row["task"], "goal": goal, "teacher": row["teacher"]["score"], "rank": rank, "sites": {}}
    for site, ss in sets["sites"].items():
        circ = ss["circuit"]
        k = len(circ)
        free = fit(ee, task, goal, site, positions, rank, steps)
        A, B = free.pop("update")
        rec = {"own": site == sets["own"], "k": k, "active": len(ss["active"]), "free": free}
        if k:
            g = torch.Generator().manual_seed(native.task_seed(row["task"] + site))
            others = [c for c in ss["active"] if c not in circ]
            draws = [[others[i] for i in torch.randperm(len(others), generator=g)[:k].tolist()] for _ in range(LOC_DRAWS)] if others else []
            C = nat.target.site(site).V.shape[1]
            rand = [torch.randperm(C, generator=g)[:k].tolist() for _ in range(LOC_DRAWS)]
            top = ss["active"][:k]
            rec["circuit"] = _plain(fit(ee, task, goal, site, positions, rank, steps, basis=basis(nat, site, circ)))
            rec["topk"] = _plain(fit(ee, task, goal, site, positions, rank, steps, basis=basis(nat, site, top))) if top else None
            rec["active_draws"] = [_plain(fit(ee, task, goal, site, positions, rank, steps, basis=basis(nat, site, d))) for d in draws]
            rec["random_draws"] = [_plain(fit(ee, task, goal, site, positions, rank, steps, basis=basis(nat, site, d))) for d in rand]
            rec["align"] = {"circuit": alignment(nat, site, A, B, circ), "topk": alignment(nat, site, A, B, top),
                            "active": [alignment(nat, site, A, B, d) for d in draws],
                            "random": [alignment(nat, site, A, B, d) for d in rand]}
            for name in ("circuit", "topk"):
                if rec[name]:
                    rec[name]["sets"] = {"circuit": circ, "topk": top}[name]
            for r, d in zip(rec["active_draws"] + rec["random_draws"], draws + rand):
                r["sets"] = d
        out["sites"][site] = rec
        msg = f"{site}{' (own)' if rec['own'] else ''} k {k}: free {free['net']:.3f}"
        if k:
            msg += (f", circuit {rec['circuit']['net']:.3f}, top-k {rec['topk']['net'] if rec['topk'] else float('nan'):.3f}, active "
                    f"{[round(r['net'], 3) for r in rec['active_draws']]}, random {[round(r['net'], 3) for r in rec['random_draws']]}; share in "
                    f"circuit span {rec['align']['circuit']['share']:.3f}")
        log(f"{row['task']} {goal['kind']} {msg}")
    out["seconds"] = time.time() - t0
    return out


def loc_table(rows: list[dict]) -> str:
    """Medians per goal kind, at the circuit's best site and at the teacher's own: restricted net / unconstrained net
    (goals with unconstrained net > 0 and k > 0), and the alignment of the unconstrained update."""
    import statistics as st

    def med(xs):
        xs = [x for x in xs if x is not None]
        return (st.median(xs), len(xs)) if xs else (float("nan"), 0)

    lines = []
    for which in ("circuit site", "own site"):
        lines.append(f"{which}: restricted net / unconstrained net (median, goals), then alignment of the unconstrained update")
        lines.append("kind     n   circuit      top-k        active       random       | share: circuit topk active random | cos read c/a/r   cos write c/a/r")
        for kind in ("flip", "raise", "detach"):
            recs = []
            for r in rows:
                if r["goal"]["kind"] != kind:
                    continue
                sites = r["sites"]
                own = [s for s, v in sites.items() if v["own"]]
                circ = [s for s, v in sites.items() if not v["own"]] or own
                pick = (own if which == "own site" else circ)
                if pick and sites[pick[0]]["k"] and sites[pick[0]]["free"]["net"] > 0:
                    recs.append(sites[pick[0]])
            if not recs:
                continue
            ratio = lambda f: med([f(v) / v["free"]["net"] if f(v) is not None else None for v in recs])  # noqa: E731
            c = ratio(lambda v: v["circuit"]["net"])
            t = ratio(lambda v: v["topk"]["net"] if v["topk"] else None)
            a = ratio(lambda v: st.median(x["net"] for x in v["active_draws"]) if v["active_draws"] else None)
            rn = ratio(lambda v: st.median(x["net"] for x in v["random_draws"]))
            al = lambda name, key: med([(st.median(x[key] for x in v["align"][name]) if isinstance(v["align"][name], list) and v["align"][name]  # noqa: E731
                                         else v["align"][name].get(key) if isinstance(v["align"][name], dict) else None) for v in recs])[0]
            lines.append(f"{kind:7s} {len(recs):2d}  {c[0]:6.3f} ({c[1]:2d}) {t[0]:6.3f} ({t[1]:2d}) {a[0]:6.3f} ({a[1]:2d}) {rn[0]:6.3f} ({rn[1]:2d})  | "
                         f"{al('circuit', 'share'):.3f} {al('topk', 'share'):.3f} {al('active', 'share'):.3f} {al('random', 'share'):.4f}  | "
                         f"{al('circuit', 'cos_read'):.2f}/{al('active', 'cos_read'):.2f}/{al('random', 'cos_read'):.2f}   "
                         f"{al('circuit', 'cos_write'):.2f}/{al('active', 'cos_write'):.2f}/{al('random', 'cos_write'):.2f}")
    return "\n".join(lines)


def table(rows: list[dict]) -> str:
    import statistics as st

    lines = ["kind    n   net teacher  units  fine-tune(teacher's site)  fine-tune(best site)   fine-tune > teacher"]
    for kind in ("flip", "raise", "detach"):
        rs = [r for r in rows if r["goal"]["kind"] == kind]
        if not rs:
            continue
        med = lambda f: st.median(f(r) for r in rs)  # noqa: E731
        lines.append(f"{kind:7s} {len(rs):2d}  {med(lambda r: r['teacher']['net']):10.3f} {med(lambda r: r['units']['net'] if r['units'] else 0):6.3f} "
                     f"{med(lambda r: r['own_site']['net']):18.3f} {med(lambda r: r['best_site']['net']):20.3f}   "
                     f"{sum(r['best_site']['net'] > r['teacher']['net'] for r in rs)}/{len(rs)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("teach")
    r.add_argument("out")
    r.add_argument("--steps", type=int, default=150)
    r.add_argument("--screen", type=int, default=30)
    r.add_argument("--device", default="cpu")
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--stride", type=int, default=1)
    t = sub.add_parser("table")
    t.add_argument("rows")
    q = sub.add_parser("loc-sets")
    q.add_argument("teach")
    q.add_argument("finetune", help="run's output (its screen picks the circuit site)")
    q.add_argument("circuits", help="the teacher circuits' directory, <task>.json")
    q.add_argument("out")
    q.add_argument("--device", default="cpu")
    lo = sub.add_parser("loc")
    lo.add_argument("teach")
    lo.add_argument("sets")
    lo.add_argument("out")
    lo.add_argument("--steps", type=int, default=150)
    lo.add_argument("--device", default="cpu")
    lo.add_argument("--offset", type=int, default=0)
    lo.add_argument("--stride", type=int, default=1)
    lt = sub.add_parser("loc-table")
    lt.add_argument("rows")
    args = ap.parse_args()
    read = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]  # noqa: E731
    key = lambda r: (r["task"], json.dumps(r["goal"], sort_keys=True))  # noqa: E731
    if args.cmd == "table":
        return print(table(read(args.rows)))
    if args.cmd == "loc-table":
        return print(loc_table(read(args.rows)))
    ee = edit_eval.EditEval(native.Native(args.device))
    for p in ee.nat.target.parameters():
        p.requires_grad_(False)
    rows = read(args.teach)
    if args.cmd == "loc-sets":
        screens = {key(r): r["screen"] for r in read(args.finetune)}
        with open(args.out, "w") as f:
            for row in rows:
                circuit = json.loads((Path(args.circuits) / f"{row['task']}.json").read_text())
                f.write(json.dumps(loc_sets(ee, row, circuit, screens.get(key(row), {}))) + "\n")
        return
    sets = {key(r): r for r in read(args.sets)} if args.cmd == "loc" else {}
    for i, row in enumerate(rows):
        if i % args.stride != args.offset:
            continue
        if args.cmd == "loc":
            res = run_loc(ee, row, sets[key(row)], args.steps, log=lambda s: print(s, flush=True))
        else:
            res = run_goal(ee, row, args.steps, args.screen, log=lambda s: print(s, flush=True))
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
