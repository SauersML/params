"""A fine-tuning baseline for the edit questions (#2951 graph oracle, E3; the lead's request after AxBench): does a
rank-r weight update fitted by gradient beat the edit teacher's subcomponent edits on net bits?

For each held-out goal (the teacher's file, edit_eval's goals) a low-rank update B A^T is added to one weight matrix W
[d_out, d_in] (its output gains (x A) B^T), applied at the positions the teacher's edit acts on (a local edit, as the
teacher's are: other texts are untouched), r = the number of the teacher's operations (each scales one rank-one
slice). A and B are fitted by Adam to the goal's net bits on the search's data (edit_eval's definitions: goal_bits
capped at the goal minus the KL on everything the goal does not name; detach on the search's alternative tokens), then
scored on the score's data (detach on the held-out alternatives). Sites: the matrix of the teacher's first operation,
and the best of all 24 after a short fit on each (the best by search net, refined).

  finetune_edit.py run TEACH.jsonl OUT.jsonl [--steps 150] [--screen 30] [--device cpu] [--offset K --stride S]
  finetune_edit.py table OUT.jsonl
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


def fit(ee, task, goal, site: str, positions: list[int], rank: int, steps: int, lr: float = 0.01, seed: int = 0) -> dict:
    """A and B fitted by Adam to the search's net, then scored on the score's data."""
    nat = ee.nat
    p = task["prompts"][0]
    ids = p["token_ids"]
    T = len(ids)
    st = nat.target.site(site)
    d_out, d_in = st.W.shape
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(d_in, rank, generator=g) / math.sqrt(d_in)).to(nat.dev).requires_grad_(True)
    B = torch.zeros(d_out, rank, device=nat.dev, requires_grad=True)
    mask = torch.zeros(T, device=nat.dev)
    mask[positions] = 1.0

    def data(ranks):
        seqs = [ids] + (ee.alternatives(ids, goal["position"], ranks) if goal["kind"] == "detach" else [])
        with torch.no_grad():
            ref = logq(nat, seqs)
        return seqs, ref

    seqs, ref = data(edit_eval.SEARCH_RANKS)
    opt = torch.optim.Adam([A, B], lr=lr)
    best = (-math.inf, None)
    for _ in range(steps):
        lq = logq(nat, seqs, site, A, B, mask)
        r = net_of(ee, task, goal, lq[:1], ref[:1], lq[1:], ref[1:])
        if float(r["net"].detach()) > best[0]:
            best = (float(r["net"].detach()), (A.detach().clone(), B.detach().clone()))
        opt.zero_grad()
        (-r["net"]).backward()
        opt.step()
    A1, B1 = best[1]
    seqs, ref = data(edit_eval.SCORE_RANKS)
    with torch.no_grad():
        lq = logq(nat, seqs, site, A1, B1, mask)
        r = net_of(ee, task, goal, lq[:1], ref[:1], lq[1:], ref[1:])
    return {"site": site, "rank": rank, "positions": positions, "search_net": best[0], "net": float(r["net"]),
            "goal_bits": float(r["goal_bits"]), "collateral": float(r["collateral"]), "attained": r["attained"],
            "params": rank * (d_in + d_out)}


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
    own = fit(ee, task, row["goal"], site, positions, rank, steps)
    shots = {n: fit(ee, task, row["goal"], n, positions, rank, screen) for n in ee.nat.names}
    top = max(shots, key=lambda n: shots[n]["search_net"])
    best = fit(ee, task, row["goal"], top, positions, rank, steps)
    out = {"task": row["task"], "goal": row["goal"], "teacher": row["teacher"]["score"], "units": row["units"]["score"],
           "own_site": own, "best_site": best, "screen": {n: s["search_net"] for n, s in shots.items()}, "seconds": time.time() - t0}
    log(f"{row['task']} {row['goal']['kind']}: teacher net {row['teacher']['score']['net']:.3f}, fine-tune at {site} "
        f"{own['net']:.3f}, at the best site {top} {best['net']:.3f} ({time.time() - t0:.0f} s)")
    return out


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
    args = ap.parse_args()
    if args.cmd == "table":
        return print(table([json.loads(x) for x in Path(args.rows).read_text().splitlines() if x.strip()]))
    ee = edit_eval.EditEval(native.Native(args.device))
    for p in ee.nat.target.parameters():
        p.requires_grad_(False)
    rows = [json.loads(x) for x in Path(args.teach).read_text().splitlines() if x.strip()]
    for i, row in enumerate(rows):
        if i % args.stride != args.offset:
            continue
        res = run_goal(ee, row, args.steps, args.screen, log=lambda s: print(s, flush=True))
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
