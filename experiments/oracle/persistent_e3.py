"""Persistent E3 (#2951 graph oracle, the lead's 10-12): does a behaviour's circuit say where the model would change?

A preference stated without any subcomponent, "after token a the model predicts b", is a behaviour in behaviours.py's
form on split B of the atlas's Pile rows, in thirds (fit, validation, score):
  instances   positions whose token is a and whose most likely next token is b;
  controls    positions where b is the most likely next token after another token (the same knowledge elsewhere).
Defined in advance:
  intended change   at the instances, b's margin (log2 of b over the best other token) falls, at most to crossing
                    (goal_bits = min(m0, m0 - m), at least 0; flipped: m < 0);
  collateral        the KL in bits at every other position of the rows and on the held Pile texts (edit_eval's
                    HELD_ROWS), and the controls kept (b still the most likely token there).
Net per position of text, as behaviours.py: rate x goal_bits - collateral per position.

The edit: a rank-r update B A^T to one weight matrix, applied at every position of every text (a weight edit), fitted
by Adam to the fit third's net (and the search's held texts), the step with the best validation net kept, scored on the
score third. At a site, the same fit restricted to a set of subcomponents' read directions v and write directions u
(A = V_k a, B = U_k b, step sizes matched to the unconstrained fit's): the circuit's (the subcomponents with the largest
measured local ablation effect on the margin at fit instances, behaviours.py's ranking), as many random subcomponents
active at the instances, as many random ones of the matrix, and VPD's most active ones there. Gate: the circuit's
restricted net (as a share of the unconstrained net) above the random sets' and VPD's on the score third.

  persistent_e3.py prefs OUT.jsonl [--k 6]        the Mac (the atlas's activity): preferences, thirds, circuits, sets
  persistent_e3.py fit PREFS.jsonl OUT.jsonl [--steps 200] [--screen 25] [--device cuda] [--offset K --stride S]   a pod
  persistent_e3.py edit PREFS.jsonl OUT.jsonl [--steps 200] [--device cuda] [--offset K --stride S]   delete the preference:
                     the circuit's edit, VPD's top k, a fine-tune of the same rank (practical())
  persistent_e3.py table OUT.jsonl [--edit]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "oracle"))
sys.path.insert(0, str(ROOT / "method" / "vpd"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import edit_eval  # noqa: E402
import native  # noqa: E402

import finetune_edit  # noqa: E402

LN2 = math.log(2)
ROWS = (5000, 6000)  # behaviours.py's split B and thirds
THIRDS = {"fit": (5000, 5333), "val": (5333, 5666), "score": (5666, 6000)}
CAP = {"fit": 24, "val": 24, "score": 64}  # instances (and controls) per third
MIN_PAIRS = 24  # a preference's instances in the fit third, at least
POOL = 128  # subcomponents ranked by local effect
K_SET = 8  # subcomponents per restricted set at a site
DRAWS = 2
RANK = 1
ROWS_STEP, HELD_STEP = 8, 2  # texts per training step (val and score use every text)
TOKENS = Path.home() / "mpd-data/atlas/vpd4l/tokens_0_6000.npy"


def _top_tokens(ee, rows: range, tokens) -> dict:
    """row -> (ids [T], the most likely next token at each position)."""
    out = {}
    for r in rows:
        ids = tokens[r, :512].astype(np.int64).tolist()
        out[r] = (ids, ee.reference(ids).argmax(-1).tolist())
    return out


def preferences(ee, src, k: int, decode, log=print) -> list[dict]:
    """The k most frequent (a, b), both words (a space and three or more letters: a content preference such as " United"
    -> " States", not punctuation or layout), with b the most likely token after a at least MIN_PAIRS times in the fit
    third and predicted after other tokens at least as often (controls exist), distinct b; each a behaviour
    (behaviours.py's dict: parts with instances, controls, rate)."""
    tops = _top_tokens(ee, range(*ROWS), src.tokens)
    pairs, b_count = {}, {}
    for r in range(*THIRDS["fit"]):
        ids, top = tops[r]
        for p in range(1, len(ids)):
            pairs[(ids[p], top[p])] = pairs.get((ids[p], top[p]), 0) + 1
            b_count[top[p]] = b_count.get(top[p], 0) + 1
    def word(t: int) -> bool:
        x = decode([t])
        return x.startswith(" ") and x[1:].isalpha() and len(x) >= 4

    ranked = sorted((n, a, b) for (a, b), n in pairs.items()
                    if n >= MIN_PAIRS and a != b and b_count[b] - n >= MIN_PAIRS and word(a) and word(b))
    out, used = [], set()
    for n, a, b in reversed(ranked):
        if b in used or len(out) >= k:
            continue
        used.add(b)
        beh = {"subcomponent": ["preference", f"{a}->{b}"], "a": a, "b": b, "S": [b], "describe": {"a": decode([a]), "b": decode([b])},
               "parts": {}}
        for part, (r0, r1) in THIRDS.items():
            inst, ctrl = [], []
            for r in range(r0, r1):
                ids, top = tops[r]
                inst += [(r, p) for p in range(1, len(ids)) if ids[p] == a and top[p] == b]
                ctrl += [(r, p) for p in range(1, len(ids)) if ids[p] != a and top[p] == b]
            rate = len(inst) / ((r1 - r0) * 511)
            rng = random.Random(f"{a}-{b}-{part}")
            rng.shuffle(inst)
            rng.shuffle(ctrl)
            beh["parts"][part] = {"instances": inst[:CAP[part]], "controls": ctrl[:CAP[part]], "rate": rate}
        out.append(beh)
        log(f"preference {decode([a])!r} -> {decode([b])!r}: {n} fit pairs, rate {beh['parts']['fit']['rate']:.2e}")
    return out


def circuit_sets(ee, src, beh: dict, log=print) -> dict:
    """The behaviour's subcomponents by measured local ablation effect at fit instances (behaviours.BehaviourEval
    .local_effects over those active at two or more), and VPD's by activity count there."""
    import behaviours

    be = behaviours.BehaviourEval(ee, src, beh)
    fit = [(r, p) for r, _, v in be.texts["fit"] for p in v["instances"]]
    count = src.active_at(fit)
    nodes = [src.node(f) for f, n in sorted(count.items(), key=lambda x: -x[1]) if n >= 2]
    eff = be.local_effects(nodes, [])
    order = np.argsort(-eff, kind="stable")[:POOL]
    log(f"  {len(nodes)} subcomponents active at >= 2 fit instances; best local effect {eff[order[0]]:.3f} bits")
    return {"circuit": [[nodes[i][0], int(nodes[i][1]), float(eff[i])] for i in order],
            "active": [[s, int(i), int(count[src.flat(s, i)])] for s, i in nodes]}


# ---- the fits


def _texts(beh: dict, part: str, tokens) -> list[tuple[list[int], list[int], list[int]]]:
    """(ids, instance positions, control positions) per row of the third holding either."""
    by = {}
    for kind in ("instances", "controls"):
        for r, p in beh["parts"][part][kind]:
            by.setdefault(r, ([], []))[kind == "controls"].append(p)
    return [(tokens[r, :max(i + c) + 1].astype(np.int64).tolist(), i, c) for r, (i, c) in sorted(by.items())]


def _logq(nat, ids: list[int], deltas: list[tuple]) -> torch.Tensor:
    """log q (natural) [T, V] of the model with every (site, A, B) update applied at every position."""
    import contextlib

    tg = nat.target
    ones = torch.ones(len(ids), device=nat.dev)
    with contextlib.ExitStack() as stack:
        for site, A, B in deltas:
            stack.enter_context(finetune_edit.Delta(tg.site(site), A, B, ones))
        return torch.log_softmax(tg.hidden(torch.tensor([ids], device=nat.dev)) @ tg.wte.T, -1)[0]


def ops_deltas(nat, ops: list[tuple]) -> list[tuple]:
    """Persistent subcomponent edits (site, index, k: the subcomponent's activation times k at every position, W - (1 -
    k) u v^T) as updates (site, A [d_in, m], B [d_out, m]) per site: (x A) B^T = sum of -(1 - k) (x . v) u."""
    by = {}
    for site, c, k in ops:
        by.setdefault(site, []).append((int(c), float(k)))
    out = []
    for site, cs in by.items():
        st = nat.target.site(site)
        idx = torch.tensor([c for c, _ in cs], device=nat.dev)
        w = torch.tensor([-(1 - k) for _, k in cs], device=nat.dev)
        out.append((site, st.V[:, idx], (st.U[idx] * w[:, None]).T))
    return out


def _margin(lq: torch.Tensor, b: int) -> torch.Tensor:
    other = lq.clone()
    other[..., b] = -math.inf
    return (lq[..., b] - other.max(-1).values) / LN2


class Net:
    """The behaviour's net per position of text (module docstring) for a dense update at a site, differentiable."""

    def __init__(self, nat, beh: dict, tokens, held: dict):
        """held: {"search": texts for the fit and validation thirds, "held": texts for the score third} (edit_eval's)."""
        self.nat, self.b, self.rate = nat, beh["b"], {p: beh["parts"][p]["rate"] for p in THIRDS}
        self.texts = {p: _texts(beh, p, tokens) for p in THIRDS}
        self.held = {p: held["held" if p == "score" else "search"] for p in THIRDS}

    def _ref(self, ids: list[int]) -> torch.Tensor:
        """The unedited log probabilities of a text, recomputed (kept for every text they would fill a GPU: 512 x 50k
        floats each)."""
        with torch.no_grad():
            return finetune_edit.logq(self.nat, [ids])[0]

    def __call__(self, part: str, site: str | None, A=None, B=None, grad: bool = True, rng: random.Random | None = None,
                 deltas: list[tuple] | None = None) -> dict:
        """rng: a training step's sample, ROWS_STEP of the part's texts and HELD_STEP held texts (the expectation of
        the full net, at a fraction of the cost). deltas: [(site, A, B)] applied together (ops_deltas), in place of
        site, A, B."""
        nat = self.nat
        if deltas is None:
            deltas = [(site, A, B)] if site else []
        texts, helds = self.texts[part], self.held[part]
        if rng is not None:
            texts = rng.sample(texts, min(ROWS_STEP, len(texts)))
            helds = rng.sample(helds, min(HELD_STEP, len(helds)))
        goal, n, coll, kept, nc, flipped = 0.0, 0, 0.0, 0.0, 0, 0.0
        with torch.set_grad_enabled(grad):
            for ids, inst, ctrl in texts:
                lp = self._ref(ids)
                lq = _logq(nat, ids, deltas) if deltas else lp
                kl = finetune_edit.kl_bits(lp, lq)
                if inst:
                    m0, m = _margin(lp[inst], self.b), _margin(lq[inst], self.b)
                    goal = goal + torch.minimum(m0, m0 - m).clamp_min(0).sum()
                    flipped += float((m < 0).float().sum())
                    n += len(inst)
                rest = torch.ones(len(ids), dtype=torch.bool, device=kl.device)
                rest[inst] = False
                coll = coll + kl[rest].sum() / max(1, int(rest.sum()))
                if ctrl:
                    kept += float((lq[ctrl].argmax(-1) == self.b).float().sum())
                    nc += len(ctrl)
            rows = len(texts)
            held = 0.0
            for ids in helds:
                lp = self._ref(ids)
                lq = _logq(nat, ids, deltas) if deltas else lp
                held = held + finetune_edit.kl_bits(lp, lq).sum() / len(ids)
            gb = goal / max(1, n)
            ct = coll / max(1, rows) + held / max(1, len(helds))
            net = self.rate[part] * gb - ct
        f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)  # noqa: E731
        return {"net": net, "_gb": gb, "_ct": ct, "goal_bits": f(gb), "goal_per_token": self.rate[part] * f(gb), "collateral_per_token": f(ct),
                "text_kl_per_token": f(coll) / max(1, rows), "pile_kl_per_token": f(held) / max(1, len(helds)),
                "flipped": flipped / max(1, n), "controls_kept": kept / nc if nc else None}


def fit(net: Net, site: str, steps: int, basis=None, lr: float = 0.01, seed: int = 0, every: int = 25, rank: int = RANK,
        lam: float | None = None, keep: list | None = None) -> dict:
    """A, B by Adam on the fit third's net, the step with the best validation net (every `every` steps) scored on the
    score third. basis (V_k, U_k): restricted, step sizes matched to the unconstrained fit's (finetune_edit.fit). lam:
    the objective goal_bits - lam x collateral per token in place of the net (practical()); keep, a list, receives every
    checkpoint's (validation metrics, deltas)."""
    nat = net.nat
    st = nat.target.site(site)
    d_out, d_in = st.W.shape
    g = torch.Generator().manual_seed(seed)
    if basis is None:
        a = (torch.randn(d_in, rank, generator=g) / math.sqrt(d_in)).to(nat.dev).requires_grad_(True)
        b = torch.zeros(d_out, rank, device=nat.dev, requires_grad=True)
        AB = lambda: (a, b)  # noqa: E731
        la, lb = lr, lr
    else:
        Vk, Uk = basis
        k = Vk.shape[1]
        a = (torch.randn(k, rank, generator=g) / math.sqrt(k)).to(nat.dev).requires_grad_(True)
        b = torch.zeros(k, rank, device=nat.dev, requires_grad=True)
        AB = lambda: (Vk @ a, Uk @ b)  # noqa: E731
        la, lb = lr * math.sqrt(d_in / k), lr * math.sqrt(d_out / k)
    opt = torch.optim.Adam([{"params": [a], "lr": la}, {"params": [b], "lr": lb}])
    rng = random.Random(seed)
    best = (-math.inf, None, -1)
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            A, B = AB()
            rv = net("val", site, A.detach(), B.detach(), grad=False)
            v = rv["net"] if lam is None else rv["_gb"] - lam * rv["_ct"]
            if float(v) > best[0]:
                best = (float(v), (A.detach().clone(), B.detach().clone()), step)
            if keep is not None and step:
                keep.append(({k: float(x) if torch.is_tensor(x) else x for k, x in rv.items() if not k.startswith("_")},
                             [(site, A.detach().clone(), B.detach().clone())], {"site": site, "rank": rank, "lam": lam, "step": step}))
        if step == steps:
            break
        A, B = AB()
        r = net("fit", site, A, B, rng=rng)
        opt.zero_grad()
        (-(r["net"] if lam is None else r["_gb"] - lam * r["_ct"])).backward()
        opt.step()
    A1, B1 = best[1]
    sc = net("score", site, A1, B1, grad=False)
    return {"site": site, "rank": rank, "k": None if basis is None else basis[0].shape[1], "val_net": best[0], "best_step": best[2],
            **{k: float(v) if torch.is_tensor(v) else v for k, v in sc.items() if not k.startswith("_")}}


def run(nat, beh: dict, tokens, held, steps: int, screen: int, log=print, names: list[str] | None = None) -> dict:
    """One preference: the unconstrained fit screened over the 24 matrices (screen steps, validation net), at the best
    one and at the circuit's most populated matrix every restricted set (module docstring)."""
    t0 = time.time()
    net = Net(nat, beh, tokens, held)
    base = {k: float(v) if torch.is_tensor(v) else v for k, v in net("score", None, grad=False).items()}
    shots = {n: fit(net, n, screen)["val_net"] for n in (names or nat.names)}
    top = max(shots, key=shots.get)
    circ = [tuple(c[:2]) for c in beh["circuit"]]
    by_site = {}
    for s, i in circ[:32]:
        by_site[s] = by_site.get(s, 0) + 1
    sites = list(dict.fromkeys([top, max(by_site, key=by_site.get)]))
    out = {"preference": beh["describe"], "a": beh["a"], "b": beh["b"], "rate": beh["parts"]["score"]["rate"], "unedited": base,
           "screen": shots, "sites": {}}
    for site in sites:
        C = nat.target.site(site).V.shape[1]
        mine = [i for s, i in circ if s == site][:K_SET]
        rec = {"free": fit(net, site, steps)}
        if mine:
            k = len(mine)
            g = torch.Generator().manual_seed(native.task_seed(f"{beh['a']}-{beh['b']}-{site}"))
            act = [i for s, i, _ in sorted(beh["active"], key=lambda x: -x[2]) if s == site]
            others = [i for i in act if i not in mine]
            draws = [[others[j] for j in torch.randperm(len(others), generator=g)[:k].tolist()] for _ in range(DRAWS)] if others else []
            rand = [torch.randperm(C, generator=g)[:k].tolist() for _ in range(DRAWS)]
            sets = {"circuit": [mine], "active": draws, "random": rand, "vpd_top": [act[:k]] if act else []}
            for name, ss in sets.items():
                rec[name] = [dict(fit(net, site, steps, basis=finetune_edit.basis(nat, site, s_)), set=s_) for s_ in ss if s_]
        out["sites"][site] = rec
        msg = f"{site}: free {rec['free']['net']:.2e}"
        for name in ("circuit", "active", "random", "vpd_top"):
            if rec.get(name):
                msg += f", {name} " + "/".join(f"{x['net']:.2e}" for x in rec[name])
        log(f"{beh['describe']['a']!r}->{beh['describe']['b']!r} {msg}")
    out["seconds"] = time.time() - t0
    return out


EDIT_CANDS = 24  # the circuit's subcomponents (by local effect) the edit search draws from
EDIT_SCALES = (0.0, -1.0)
MAX_OPS = 6
TOPK = (1, 2, 4, 8, 16)
LAMS = (2.0, 8.0, 32.0)  # objective goal_bits - lam x collateral per token: removal against collateral
FLIP = 0.9  # removal target: the share of instances where b is no longer the most likely token


def _score(net: Net, part: str, deltas: list[tuple]) -> dict:
    r = net(part, None, grad=False, deltas=deltas)
    return {k: float(v) if torch.is_tensor(v) else v for k, v in r.items() if not k.startswith("_")}


def _obj(r: dict, lam: float) -> float:
    return r["goal_bits"] - lam * r["collateral_per_token"]


def search(net: Net, cands: list[tuple], lam: float, keep: list, log=print) -> None:
    """The edit chosen from the circuit: greedy over persistent scales (k in EDIT_SCALES) of the candidates, each round
    the op that most raises the fit third's objective added, up to MAX_OPS; every step's validation metrics to keep."""
    nat = net.nat
    chosen, last = [], -math.inf
    for _ in range(MAX_OPS):
        best = (-math.inf, None)
        for c in cands:
            if any(tuple(c[:2]) == tuple(o[:2]) for o in chosen):
                continue
            for k in EDIT_SCALES:
                v = _obj(_score(net, "fit", ops_deltas(nat, chosen + [(*c, k)])), lam)
                if v > best[0]:
                    best = (v, (*c, k))
        if best[1] is None or best[0] <= last:
            break
        chosen.append(best[1])
        last = best[0]
        val = _score(net, "val", ops_deltas(nat, chosen))
        keep.append((val, list(chosen), {"lam": lam, "ops": len(chosen)}))
        log(f"  lam {lam}: edit step {len(chosen)} {best[1]}: validation flipped {val['flipped']:.2f}, collateral {val['collateral_per_token']:.2e}")


def select(cands: list) -> tuple:
    """The rule every method's edit is picked by, on validation: among those removing the preference (flipped >= FLIP)
    the one with the least collateral per token; when none does, the most flipped (then the least collateral)."""
    ok = [c for c in cands if c[0]["flipped"] >= FLIP]
    if ok:
        return min(ok, key=lambda c: c[0]["collateral_per_token"])
    return max(cands, key=lambda c: (c[0]["flipped"], -c[0]["collateral_per_token"]))


def practical(nat, beh: dict, tokens, held, steps: int, screen: int, log=print, names: list[str] | None = None) -> dict:
    """Delete a preference (the lead's practical edit): three methods, each giving candidate edits on the fit third,
    one picked by the same rule on validation (select()), scored on the score third:
      circuit_edit  persistent scales of the circuit's subcomponents (greedy search(), at each objective weight LAMS);
      vpd_topk      VPD's top k by attribution (the subcomponents most often active at the fit instances) ablated;
      finetune      an unconstrained update at every position of rank = the circuit edit's operation count, at the
                    matrix whose short fit is best, Adam on the objective at each LAMS, checkpoints every 25 steps.
    Scores: flipped (preference gone), goal bits, controls kept (b still predicted elsewhere), KL per token on the rows'
    other positions and on held Pile texts. The validation frontier (every candidate's flipped and collateral) is kept."""
    t0 = time.time()
    net = Net(nat, beh, tokens, held)
    out = {"preference": beh["describe"], "a": beh["a"], "b": beh["b"], "rate": beh["parts"]["score"]["rate"], "methods": {}}
    cands = [(s, int(i)) for s, i, _ in beh["circuit"][:EDIT_CANDS]]
    keep = {"circuit_edit": [], "vpd_topk": [], "finetune": []}
    for lam in LAMS:
        search(net, cands, lam, keep["circuit_edit"], log)
    act = [(s, int(i)) for s, i, _ in sorted(beh["active"], key=lambda x: -x[2])]
    for k in TOPK:
        if k <= len(act):
            ops = [(*c, 0.0) for c in act[:k]]
            keep["vpd_topk"].append((_score(net, "val", ops_deltas(nat, ops)), ops, {"k": k}))
    ce = select(keep["circuit_edit"]) if keep["circuit_edit"] else None
    r = max(1, len(ce[1]) if ce else 1)
    shots = {n: fit(net, n, screen, rank=r, lam=LAMS[1])["val_net"] for n in (names or nat.names)}
    top = max(shots, key=shots.get)
    for lam in LAMS:
        fit(net, top, steps, rank=r, lam=lam, keep=keep["finetune"])
    for name, cs in keep.items():
        if not cs:
            continue
        val, edit, info = select(cs)
        deltas = edit if name == "finetune" else ops_deltas(nat, edit)
        out["methods"][name] = {"info": info, "ops": None if name == "finetune" else [list(o) for o in edit], "validation": val,
                                "score": _score(net, "score", deltas), "frontier": [[c[0]["flipped"], c[0]["collateral_per_token"], c[2]] for c in cs]}
        v = out["methods"][name]["score"]
        log(f"{beh['describe']['a']!r}->{beh['describe']['b']!r} {name} {info}: flipped {v['flipped']:.2f}, goal {v['goal_bits']:.2f} bits, "
            f"kept {v['controls_kept'] or 0:.2f}, text KL {v['text_kl_per_token']:.2e}, Pile KL {v['pile_kl_per_token']:.2e}/token")
    out["finetune_screen"] = shots
    out["seconds"] = time.time() - t0
    return out


def practical_table(rows: list[dict]) -> str:
    """Per preference and method on the score third: flipped (preference gone), goal bits, controls kept, KL per token on
    the rows' other positions and on Pile."""
    lines = ["preference              method         edit           flipped  goal bits  kept   text KL/tok  Pile KL/tok"]
    for r in rows:
        for name in ("circuit_edit", "vpd_topk", "finetune"):
            x = r["methods"].get(name)
            if not x:
                continue
            v, i = x["score"], x["info"]
            what = (f"{len(x['ops'])} ops" if name == "circuit_edit" else f"k {i['k']}" if name == "vpd_topk"
                    else f"r{i['rank']} {i['site'].replace('h.', 'L').replace('.mlp', '').replace('.attn', '')}")
            lines.append(f"{r['preference']['a']!r:>9s}->{r['preference']['b']!r:<11s} {name:13s} {what:14s} {v['flipped']:7.2f}  {v['goal_bits']:9.2f}  "
                         f"{v['controls_kept'] or 0:5.2f}  {v['text_kl_per_token']:11.2e}  {v['pile_kl_per_token']:11.2e}")
    return "\n".join(lines)


def table(rows: list[dict]) -> str:
    """Per preference and site the score third's net (per position of text) of each set as a share of the
    unconstrained fit's, its flipped share and controls kept; then the gate: the circuit's share above every control's,
    per preference, and the medians."""
    import statistics as st

    lines = ["preference              site               free net  circuit  active  random  vpd_top | flipped c/f  kept c/f"]
    wins = {"active": 0, "random": 0, "vpd_top": 0}
    shares = {k: [] for k in ("circuit", "active", "random", "vpd_top")}
    n = 0
    for r in rows:
        for site, rec in r["sites"].items():
            f = rec["free"]["net"]
            if f <= 0 or not rec.get("circuit"):
                continue
            n += 1
            sh = {k: st.median(x["net"] for x in rec[k]) / f for k in shares if rec.get(k)}
            for k, v in sh.items():
                shares[k].append(v)
            for k in wins:
                wins[k] += k in sh and sh["circuit"] > sh[k]
            c = rec["circuit"][0]
            lines.append(f"{r['preference']['a']!r:>9s}->{r['preference']['b']!r:<11s} {site:18s} {f:9.2e}  "
                         + "  ".join(f"{sh.get(k, float('nan')):6.2f}" for k in shares)
                         + f" | {c['flipped']:.2f}/{rec['free']['flipped']:.2f}  {c['controls_kept'] or 0:.2f}/{rec['free']['controls_kept'] or 0:.2f}")
    lines.append("medians: " + ", ".join(f"{k} {st.median(v):.2f}" for k, v in shares.items() if v)
                 + "; circuit above " + ", ".join(f"{k} in {w}/{n}" for k, w in wins.items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefs")
    p.add_argument("out")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--device", default="cpu")
    f = sub.add_parser("fit")
    f.add_argument("prefs")
    f.add_argument("out")
    f.add_argument("--steps", type=int, default=200)
    f.add_argument("--screen", type=int, default=25)
    f.add_argument("--device", default="cpu")
    f.add_argument("--offset", type=int, default=0)
    f.add_argument("--stride", type=int, default=1)
    f.add_argument("--sites", nargs="*", help="screen these matrices only (a smoke test)")
    e = sub.add_parser("edit")
    e.add_argument("prefs")
    e.add_argument("out")
    e.add_argument("--steps", type=int, default=200)
    e.add_argument("--screen", type=int, default=25)
    e.add_argument("--device", default="cpu")
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--sites", nargs="*")
    t = sub.add_parser("table")
    t.add_argument("rows")
    t.add_argument("--edit", action="store_true", help="edit's rows (practical_table)")
    args = ap.parse_args()
    read = lambda x: [json.loads(y) for y in Path(x).read_text().splitlines() if y.strip()]  # noqa: E731
    if args.cmd == "table":
        return print(practical_table(read(args.rows)) if args.edit else table(read(args.rows)))
    import components

    decode = lambda xs: "".join(components.token_strings("vpd4l", xs))  # noqa: E731
    nat = native.Native(args.device)
    if args.cmd == "prefs":
        import behaviours

        ee = edit_eval.EditEval(nat)
        src = behaviours.Source()
        with open(args.out, "w") as fh:
            for beh in preferences(ee, src, args.k, decode, log=lambda s: print(s, flush=True)):
                beh.update(circuit_sets(ee, src, beh, log=lambda s: print(s, flush=True)))
                fh.write(json.dumps(beh) + "\n")
        return
    for q in nat.target.parameters():
        q.requires_grad_(False)
    tokens = np.load(TOKENS, mmap_mode="r")
    tk = np.load(edit_eval.TOKENS)
    held = {k: [list(map(int, x)) for x in tk[k]] for k in ("search", "held")}
    for i, beh in enumerate(read(args.prefs)):
        if i % args.stride != args.offset:
            continue
        go = practical if args.cmd == "edit" else run
        r = go(nat, beh, tokens, held, args.steps, args.screen, log=lambda s: print(s, flush=True), names=args.sites)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
