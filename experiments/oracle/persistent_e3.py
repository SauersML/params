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
  persistent_e3.py summary ROWS.jsonl ... [--figure PNG]     means with 95% intervals, paired differences, the figure
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
MIN_PAIRS = 16  # a preference's instances in the fit third, at least
B_REPEAT = 2  # preferences sharing one b, at most
POOL = 128  # subcomponents ranked by local effect
K_SET = 8  # subcomponents per restricted set at a site
DRAWS = 2
RANK = 1
ROWS_STEP, HELD_STEP = 8, 2  # texts per training step (val and score use every text)
CACHE_GB = float(__import__("os").environ.get("PE3_CACHE_GB", "4.5"))  # unedited runs kept per process (GPU memory)
TOKENS = Path.home() / "mpd-data/atlas/vpd4l/tokens_0_6000.npy"


def _top_tokens(ee, rows: range, tokens) -> dict:
    """row -> (ids [T], the most likely next token at each position)."""
    out = {}
    for r in rows:
        ids = tokens[r, :512].astype(np.int64).tolist()
        out[r] = (ids, ee.reference(ids).argmax(-1).tolist())
    return out


def preferences(ee, src, k: int, decode, log=print, min_pairs: int = MIN_PAIRS) -> list[dict]:
    """The k most frequent (a, b), both words (a space and three or more letters: a content preference such as " United"
    -> " States", not punctuation or layout), with b the most likely token after a at least min_pairs times in the fit
    third and predicted after other tokens at least as often (controls exist), at most B_REPEAT sharing one b; each a
    behaviour (behaviours.py's dict: parts with instances, controls, rate)."""
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
                    if n >= min_pairs and a != b and b_count[b] - n >= min_pairs and word(a) and word(b))
    out, used = [], {}
    for n, a, b in reversed(ranked):
        if used.get(b, 0) >= B_REPEAT or len(out) >= k:
            continue
        used[b] = used.get(b, 0) + 1
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
    .local_effects over those active at two or more), VPD's by activity count there, and the controls' circuit, the same
    ranking at the fit third's controls (b predicted after other tokens; never the score third)."""
    import behaviours

    be = behaviours.BehaviourEval(ee, src, beh)
    fit = [(r, p) for r, _, v in be.texts["fit"] for p in v["instances"]]
    count = src.active_at(fit)
    nodes = [src.node(f) for f, n in sorted(count.items(), key=lambda x: -x[1]) if n >= 2]
    eff = be.local_effects(nodes, [])
    order = np.argsort(-eff, kind="stable")[:POOL]
    log(f"  {len(nodes)} subcomponents active at >= 2 fit instances; best local effect {eff[order[0]]:.3f} bits")
    # the controls' circuit, by the same teacher: b's margin at the fit third's controls (b after other tokens)
    fit_c = beh["parts"]["fit"]
    bc = behaviours.BehaviourEval(ee, src, {**beh, "parts": {**beh["parts"], "fit": {"instances": fit_c["controls"], "controls": [],
                                                                                    "rate": fit_c["rate"]}}})
    count_c = src.active_at(fit_c["controls"])
    nodes_c = [src.node(f) for f, n in sorted(count_c.items(), key=lambda x: -x[1]) if n >= 2]
    eff_c = bc.local_effects(nodes_c, []) if nodes_c else np.zeros(0)
    order_c = np.argsort(-eff_c, kind="stable")[:POOL]
    return {"circuit": [[nodes[i][0], int(nodes[i][1]), float(eff[i])] for i in order],
            "active": [[s, int(i), int(count[src.flat(s, i)])] for s, i in nodes],
            "control_circuit": [[nodes_c[i][0], int(nodes_c[i][1]), float(eff_c[i])] for i in order_c]}


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
        self._cache, self._bytes = {}, 0

    def _ref(self, ids: list[int]) -> torch.Tensor:
        """The unedited log probabilities of a text: kept up to CACHE_GB (the searches evaluate hundreds of edits on the
        same texts; half precision would err by more than the collateral it measures), recomputed beyond it."""
        key = tuple(ids)
        got = self._cache.get(key)
        if got is not None:
            return got
        with torch.no_grad():
            lp = finetune_edit.logq(self.nat, [ids])[0]
        if self._bytes + lp.numel() * 4 <= CACHE_GB * 2**30:
            self._cache[key] = lp
            self._bytes += lp.numel() * 4
        return lp

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
    score third. basis (V_k, U_k): restricted (A = V_k a, B = U_k b), step sizes lr sqrt(d_in / k) and lr sqrt(d_out / k). lam:
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
    seen = net.__dict__.setdefault("_fit_scores", {})  # an edit's fit-third scores, shared by the searches at each lam
    chosen, last = [], -math.inf
    for _ in range(MAX_OPS):
        best = (-math.inf, None)
        for c in cands:
            if any(tuple(c[:2]) == tuple(o[:2]) for o in chosen):
                continue
            for k in EDIT_SCALES:
                ops = chosen + [(*c, k)]
                key = tuple(map(tuple, ops))
                if key not in seen:
                    seen[key] = _score(net, "fit", ops_deltas(nat, ops))
                v = _obj(seen[key], lam)
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


SCALE_CANDS = 16  # subcomponents whose activations circuit_scales learns a continuous scale for


def fit_scales(net: Net, cands: list[tuple], steps: int, lam: float, keep: list, lr: float = 0.05, seed: int = 0,
               every: int = 25) -> None:
    """Continuous persistent scales of the candidates' activations (W - sum a_c u_c v_c^T, a learned): Adam on the fit
    third's objective goal_bits - lam x collateral per token, every `every` steps a checkpoint's validation metrics and
    its scales (as ops: (site, index, 1 - a)) to keep."""
    nat = net.nat
    by = {}
    for j, (site, c) in enumerate(cands):
        by.setdefault(site, []).append((j, int(c)))
    alpha = torch.zeros(len(cands), device=nat.dev, requires_grad=True)
    fixed = {site: (torch.tensor([j for j, _ in js], device=nat.dev), nat.target.site(site).V[:, [c for _, c in js]],
                    nat.target.site(site).U[[c for _, c in js]].T) for site, js in by.items()}

    def deltas(a):
        return [(site, V, -U * a[idx][None, :]) for site, (idx, V, U) in fixed.items()]

    opt = torch.optim.Adam([alpha], lr=lr)
    rng = random.Random(seed)
    for step in range(steps + 1):
        if step and (step % every == 0 or step == steps):
            a = alpha.detach().clone()
            val = _score(net, "val", deltas(a))
            keep.append((val, [(site, c, 1.0 - float(a[j])) for j, (site, c) in enumerate(cands)], {"lam": lam, "step": step}))
        if step == steps:
            break
        r = net("fit", None, rng=rng, deltas=deltas(alpha))
        opt.zero_grad()
        (-(r["_gb"] - lam * r["_ct"])).backward()
        opt.step()


def fit_multi(net: Net, bases: dict, steps: int, rank: int, lam: float, keep: list, lr: float = 0.01, seed: int = 0,
              every: int = 25) -> None:
    """A rank-`rank` update at every position at each of several matrices, each restricted to its basis (V_k, U_k):
    A = V_k a, B = U_k b, step sizes matched to an unconstrained fit's (fit()); a basis None, unrestricted there (fit()'s
    parameters and steps); all trained together on the fit third's
    objective goal_bits - lam x collateral per token; every `every` steps a checkpoint's validation metrics and deltas
    to keep."""
    nat = net.nat
    g = torch.Generator().manual_seed(seed)
    params, groups = {}, []
    for site, basis in bases.items():
        d_out, d_in = nat.target.site(site).W.shape
        if basis is None:  # unrestricted at this matrix (fit()'s parameters and step sizes)
            a = (torch.randn(d_in, rank, generator=g) / math.sqrt(d_in)).to(nat.dev).requires_grad_(True)
            b = torch.zeros(d_out, rank, device=nat.dev, requires_grad=True)
            params[site] = (None, None, a, b)
            groups += [{"params": [a], "lr": lr}, {"params": [b], "lr": lr}]
            continue
        Vk, Uk = basis[0], basis[1]
        kv, ku = Vk.shape[1], Uk.shape[1]  # read and write spans may differ in dimension (spec_basis)
        r = min(rank, kv, ku)
        a = (torch.randn(kv, r, generator=g) / math.sqrt(kv)).to(nat.dev).requires_grad_(True)
        b = torch.zeros(ku, r, device=nat.dev, requires_grad=True)
        params[site] = (Vk, Uk, a, b)
        groups += [{"params": [a], "lr": lr * math.sqrt(d_in / kv)}, {"params": [b], "lr": lr * math.sqrt(d_out / ku)}]

    def deltas(detach: bool = False):
        out = []
        for site, (Vk, Uk, a, b) in params.items():
            A, B = (a, b) if Vk is None else (Vk @ a, Uk @ b)
            out.append((site, A.detach().clone(), B.detach().clone()) if detach else (site, A, B))
        return out

    opt = torch.optim.Adam(groups)
    rng = random.Random(seed)
    for step in range(steps + 1):
        if step and (step % every == 0 or step == steps):
            ds = deltas(detach=True)
            keep.append((_score(net, "val", ds), ds, {"lam": lam, "step": step, "rank": rank,
                                                     "sites": {s_: (v[0].shape[1] if v is not None else "all") for s_, v in bases.items()}}))
        if step == steps:
            break
        r = net("fit", None, rng=rng, deltas=deltas())
        opt.zero_grad()
        (-(r["_gb"] - lam * r["_ct"])).backward()
        opt.step()


RANK_TOL = 1e-6  # numerical rank: singular values above this share of the largest


def _span(X: torch.Tensor, scale: float | None = None) -> tuple[torch.Tensor, float]:
    """(an orthonormal basis of X's column span at numerical rank, X's largest singular value): singular values above
    RANK_TOL x scale (X's own largest when scale is None; a projected matrix is measured against the unprojected one's,
    or a direction projected to rounding would count); scipy's SVD on the CPU (torch's decompositions are unreliable
    on the Mac), torch's on a GPU."""
    if X.shape[1] == 0:
        return X, 0.0
    if X.device.type == "cpu":
        import numpy as np
        from scipy.linalg import svd

        U, sv, _ = svd(X.double().numpy(), full_matrices=False)
        keep = sv > RANK_TOL * (sv.max() if scale is None else scale)
        return torch.from_numpy(np.ascontiguousarray(U[:, keep])).to(X.dtype), float(sv.max())
    U, sv, _ = torch.linalg.svd(X.double(), full_matrices=False)
    return U[:, sv > RANK_TOL * (float(sv.max()) if scale is None else scale)].to(X.dtype), float(sv.max())


def spec_basis(nat, site: str, circ: list[int], ctrl: list[int]):
    """(read basis, write basis, counts) at a matrix: the span of the circuit's read directions v with the controls'
    circuit's read span projected out, the same for the write directions u, each an orthonormal basis at numerical
    rank relative to the unprojected directions' largest singular value (no threshold on what counts as the circuit);
    counts = [circuit read rank, write rank, after projection read, write]. None when either side is left empty."""
    Vk, Uk = finetune_edit.basis(nat, site, circ)
    (Bv, sv), (Bu, su) = _span(Vk), _span(Uk)
    counts = [Bv.shape[1], Bu.shape[1]]
    if ctrl:
        Vc, Uc = finetune_edit.basis(nat, site, ctrl)
        Qv, Qu = _span(Vc)[0], _span(Uc)[0]
        Bv, Bu = _span(Vk - Qv @ (Qv.T @ Vk), sv)[0], _span(Uk - Qu @ (Qu.T @ Uk), su)[0]
    counts += [Bv.shape[1], Bu.shape[1]]
    return (Bv, Bu, counts) if Bv.shape[1] and Bu.shape[1] else (None, None, counts)


METHODS = ("circuit_edit", "random_active", "random_any", "random_equal", "vpd_topk", "finetune", "circuit_scales", "random_scales",
           "circuit_ft", "random_ft", "circuit_ft_spec", "finetune_sites", "assoc_ft")
RANDOM_DRAWS = 2
DENSE = ("finetune", "circuit_ft", "random_ft", "circuit_ft_spec", "finetune_sites", "assoc_ft")  # methods whose edits are (site, A, B) updates


def practical(nat, beh: dict, tokens, held, steps: int, screen: int, log=print, names: list[str] | None = None,
              methods: tuple = METHODS, rank: int | None = None, assoc: dict | None = None) -> dict:
    """Delete a preference (the lead's practical edit): each method gives candidate edits on the fit third, one picked by
    the same rule on validation (select()), scored on the score third:
      circuit_edit   persistent scales of the circuit's subcomponents (greedy search() over its EDIT_CANDS by local
                     effect, at each objective weight LAMS);
      random_active  the same search over EDIT_CANDS random subcomponents active at the fit instances, outside the
                     circuit's (RANDOM_DRAWS draws: the circuit's pool against a pool of equal size);
      random_any     the same over EDIT_CANDS random subcomponents of the model;
      random_equal   as many random active subcomponents as the circuit edit's operations, ablated (RANDOM_DRAWS draws);
      vpd_topk       VPD's top k by attribution (the subcomponents most often active at the fit instances) ablated;
      finetune       an unconstrained update at every position of rank = the circuit edit's operation count, at the
                     matrix whose short fit is best, Adam on the objective at each LAMS, checkpoints every 25 steps;
      circuit_scales continuous scales of the circuit's top SCALE_CANDS subcomponents (fit_scales), at each LAMS;
      random_scales  the same for as many random active subcomponents outside the circuit's;
      circuit_ft     the fine-tune restricted to the circuit's span: at each matrix holding one of its EDIT_CANDS
                     subcomponents, a rank-r update within their read and write directions, trained together
                     (fit_multi), at each LAMS;
      random_ft      the same within as many random active subcomponents at each of those matrices;
      circuit_ft_spec  circuit_ft with the controls' circuit's read and write spans projected out at each matrix
                     (spec_basis, at numerical rank; the controls' circuit from the same teacher at the fit third's
                     controls); out["spec_ranks"] the ranks kept per matrix without and with the projection;
      finetune_sites an unrestricted rank-r update at each of circuit_ft's matrices, trained together: the same
                     matrices, objective, steps and selection as circuit_ft, only the restriction differs.
      assoc_ft       circuit_ft with the subcomponents most associated with the preference in place of the circuit's
                     (assoc: {"a->b": {matrix: [indices]}}, generality's gwas.py assoc_prefs: the top EDIT_CANDS by the
                     association of their activations with log p(b) after a on Pile train text); the same fit_multi, rank,
                     steps and LAMS.
    rank: r given (an earlier run's circuit edit size) in place of the circuit edit's operation count.
    Scores: flipped (preference gone), goal bits, controls kept (b still predicted elsewhere), KL per token on the rows'
    other positions and on held Pile texts. The validation frontier (every candidate's flipped and collateral) is kept."""
    t0 = time.time()
    net = Net(nat, beh, tokens, held)
    out = {"preference": beh["describe"], "a": beh["a"], "b": beh["b"], "rate": beh["parts"]["score"]["rate"], "methods": {}}
    circ = [(s, int(i)) for s, i, _ in beh["circuit"][:EDIT_CANDS]]
    act = [(s, int(i)) for s, i, _ in sorted(beh["active"], key=lambda x: -x[2])]
    rng = random.Random(f"{beh['a']}-{beh['b']}")
    keep = {}
    if "circuit_edit" in methods:
        keep["circuit_edit"] = []
        for lam in LAMS:
            search(net, circ, lam, keep["circuit_edit"], log)
    pools = {}
    if "random_active" in methods:
        others = [c for c in act if c not in set(circ)]
        for d in range(RANDOM_DRAWS):
            pools[f"random_active_{d}"] = rng.sample(others, min(EDIT_CANDS, len(others)))
    if "random_any" in methods:
        tg = nat.target
        pools["random_any"] = [(n, rng.randrange(tg.site(n).V.shape[1])) for n in rng.choices(nat.names, k=EDIT_CANDS)]
    for name, pool in pools.items():
        keep[name] = []
        for lam in LAMS:
            search(net, pool, lam, keep[name], lambda m, name=name: log(f"  [{name}]{m}"))
    ce = select(keep["circuit_edit"]) if keep.get("circuit_edit") else None
    r = rank or max(1, len(ce[1]) if ce else 1)
    out["rank"] = r
    if "random_equal" in methods:
        others = [c for c in act if c not in set(circ)]
        for d in range(RANDOM_DRAWS):
            ops = [(*c, 0.0) for c in rng.sample(others, min(r, len(others)))]
            keep[f"random_equal_{d}"] = [(_score(net, "val", ops_deltas(nat, ops)), ops, {"k": len(ops)})]
    if "vpd_topk" in methods:
        keep["vpd_topk"] = []
        for k in TOPK:
            if k <= len(act):
                ops = [(*c, 0.0) for c in act[:k]]
                keep["vpd_topk"].append((_score(net, "val", ops_deltas(nat, ops)), ops, {"k": k}))
    if "circuit_scales" in methods:
        keep["circuit_scales"] = []
        for lam in LAMS:
            fit_scales(net, circ[:SCALE_CANDS], steps, lam, keep["circuit_scales"])
    if "random_scales" in methods:
        others = [c for c in act if c not in set(circ)]
        pool = rng.sample(others, min(SCALE_CANDS, len(others)))
        keep["random_scales"] = []
        for lam in LAMS:
            fit_scales(net, pool, steps, lam, keep["random_scales"])
    sites = {}
    for site, c in circ:
        sites.setdefault(site, []).append(c)
    if "circuit_ft_spec" in methods:  # the circuit's span less the controls' circuit's at each of its matrices
        ctrl = {}
        for site, c, _ in beh.get("control_circuit", [])[:EDIT_CANDS]:
            ctrl.setdefault(site, []).append(int(c))
        got = {site: spec_basis(nat, site, cs, ctrl.get(site, [])) for site, cs in sites.items()}
        out["spec_ranks"] = {site: g_[2] for site, g_ in got.items()}  # [read, write] ranks without and with the projection
        sb = {site: (g_[0], g_[1]) for site, g_ in got.items() if g_[0] is not None}
        keep["circuit_ft_spec"] = []
        for lam in LAMS:
            fit_multi(net, sb, steps, r, lam, keep["circuit_ft_spec"])
    if "finetune_sites" in methods:  # unrestricted at the same matrices as circuit_ft (only the restriction differs)
        keep["finetune_sites"] = []
        for lam in LAMS:
            fit_multi(net, {site: None for site in sites}, steps, r, lam, keep["finetune_sites"])
    if "assoc_ft" in methods and (assoc or {}).get(f"{beh['a']}->{beh['b']}"):  # circuit_ft over the association-picked set
        keep["assoc_ft"] = []
        for lam in LAMS:
            fit_multi(net, {site: finetune_edit.basis(nat, site, [int(c) for c in cs]) for site, cs in assoc[f"{beh['a']}->{beh['b']}"].items()},
                      steps, r, lam, keep["assoc_ft"])
    if "circuit_ft" in methods or "random_ft" in methods:  # the fine-tune restricted to the circuit's span at its sites
        if "circuit_ft" in methods:
            keep["circuit_ft"] = []
            for lam in LAMS:
                fit_multi(net, {site: finetune_edit.basis(nat, site, cs) for site, cs in sites.items()}, steps, r, lam, keep["circuit_ft"])
        if "random_ft" in methods:  # as many random active subcomponents at each of those sites, outside the circuit's
            rb = {}
            for site, cs in sites.items():
                others = [c for s_, c in act if s_ == site and (s_, c) not in set(circ)]
                if not others:
                    others = [c for c in range(nat.target.site(site).V.shape[1]) if c not in cs]
                rb[site] = finetune_edit.basis(nat, site, rng.sample(others, min(len(cs), len(others))))
            keep["random_ft"] = []
            for lam in LAMS:
                fit_multi(net, rb, steps, r, lam, keep["random_ft"])
    if "finetune" in methods:
        shots = {n: fit(net, n, screen, rank=r, lam=LAMS[1])["val_net"] for n in (names or nat.names)}
        top = max(shots, key=shots.get)
        keep["finetune"] = []
        for lam in LAMS:
            fit(net, top, steps, rank=r, lam=lam, keep=keep["finetune"])
        out["finetune_screen"] = shots
    for name, cs in keep.items():
        if not cs:
            continue
        val, edit, info = select(cs)
        deltas = edit if name in DENSE else ops_deltas(nat, edit)
        out["methods"][name] = {"info": info, "ops": None if name in DENSE else [list(o) for o in edit], "validation": val,
                                "score": _score(net, "score", deltas), "frontier": [[c[0]["flipped"], c[0]["collateral_per_token"], c[2]] for c in cs]}
        if name not in DENSE:  # every candidate edit and its validation metrics (a different rule can pick among them)
            out["methods"][name]["candidates"] = [[[list(o) for o in c[1]], c[0], c[2]] for c in cs]
        v = out["methods"][name]["score"]
        log(f"{beh['describe']['a']!r}->{beh['describe']['b']!r} {name} {info}: flipped {v['flipped']:.2f}, goal {v['goal_bits']:.2f} bits, "
            f"kept {v['controls_kept'] or 0:.2f}, text KL {v['text_kl_per_token']:.2e}, Pile KL {v['pile_kl_per_token']:.2e}/token")
    out["seconds"] = time.time() - t0
    return out


def practical_table(rows: list[dict]) -> str:
    """Per preference and method on the score third: flipped (preference gone), goal bits, controls kept, KL per token on
    the rows' other positions and on Pile."""
    lines = ["preference              method          edit           flipped  goal bits  kept   text KL/tok  Pile KL/tok"]
    for r in rows:
        for name, x in sorted(r["methods"].items(), key=lambda kv: [m in kv[0] for m in METHODS].index(True)):
            v, i = x["score"], x["info"]
            what = (f"r{i['rank']} {i['site'].replace('h.', 'L').replace('.mlp', '').replace('.attn', '')}" if name == "finetune"
                    else f"r{i['rank']} {len(i['sites'])} sites" if name in DENSE
                    else f"k {i['k']}" if "k" in i else f"{len(x['ops'])} ops")
            lines.append(f"{r['preference']['a']!r:>9s}->{r['preference']['b']!r:<11s} {name:15s} {what:14s} {v['flipped']:7.2f}  {v['goal_bits']:9.2f}  "
                         f"{v['controls_kept'] or 0:5.2f}  {v['text_kl_per_token']:11.2e}  {v['pile_kl_per_token']:11.2e}")
    return "\n".join(lines)


T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201,
       13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 20: 2.093, 30: 2.045}
PAIRS = (("circuit_ft_spec", "circuit_ft"), ("circuit_ft_spec", "finetune_sites"), ("circuit_ft", "finetune_sites"),
         ("circuit_ft", "finetune"), ("circuit_ft_spec", "finetune"), ("circuit_ft", "random_ft"), ("circuit_edit", "random_active"))


def summary(rows: list[dict], figure: str | None = None) -> str:
    """Per method over the preferences (several runs of one preference merged; a method's draws averaged): the mean and
    95% t-interval of removal (flipped), Pile KL per token, KL on the rows and controls kept; per pair (PAIRS) the
    paired differences (Pile KL as log10 ratios); figure: removal against Pile KL, a point per preference and each
    method's mean with its 95% intervals."""
    import statistics as st

    prefs = {}
    for r in rows:
        prefs.setdefault((r["a"], r["b"]), {}).update(r["methods"])
    keys = ("flipped", "pile_kl_per_token", "text_kl_per_token", "controls_kept")

    def per(m):
        out = {}
        for pk, ms in prefs.items():
            got = [v["score"] for k, v in ms.items() if k == m or (k.startswith(m + "_") and k[len(m) + 1:].isdigit())]
            if got:
                out[pk] = {x: st.mean((g[x] if g[x] is not None else 0.0) for g in got) for x in keys}
        return out

    def ci(xs):
        n = len(xs)
        h = T95.get(n, 2.0) * st.stdev(xs) / math.sqrt(n) if n > 1 else float("nan")
        return st.mean(xs), h

    methods = [m for m in ("circuit_ft_spec", "circuit_ft", "finetune_sites", "finetune", "circuit_edit", "circuit_scales", "random_ft",
                           "random_scales", "random_active", "random_equal", "random_any", "vpd_topk") if per(m)]
    lines = [f"{'method':16s} n   removed        Pile KL/token            text KL/token   controls kept"]
    for m in methods:
        d = per(m)
        f, pk, tk, kp = (ci([v[x] for v in d.values()]) for x in keys)
        lines.append(f"{m:16s} {len(d):2d}  {f[0]:.2f} ± {f[1]:.2f}   {pk[0]:.2e} ± {pk[1]:.1e}   {tk[0]:.1e}         {kp[0]:.2f} ± {kp[1]:.2f}")
    lines.append("paired (first - second, 95% CI over preferences): removal; log10 Pile KL (ratio); controls kept")
    for a, b in PAIRS:
        da, db = per(a), per(b)
        common = [k for k in da if k in db]
        if len(common) < 2:
            continue
        fr = ci([da[k]["flipped"] - db[k]["flipped"] for k in common])
        lp = ci([math.log10(max(da[k]["pile_kl_per_token"], 1e-6)) - math.log10(max(db[k]["pile_kl_per_token"], 1e-6)) for k in common])
        kp = ci([da[k]["controls_kept"] - db[k]["controls_kept"] for k in common])
        lines.append(f"  {a} - {b} (n {len(common)}): removal {fr[0]:+.2f} ± {fr[1]:.2f}; Pile log10 {lp[0]:+.2f} ± {lp[1]:.2f} "
                     f"(x{10 ** lp[0]:.2f}); kept {kp[0]:+.2f} ± {kp[1]:.2f}")
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = {"circuit_ft_spec": ("fine-tune restricted to the circuit's read/write directions,\nwith the controls' circuit directions projected out", "#1a73e8", "o"),
                 "circuit_ft": ("fine-tune restricted to the circuit's read/write directions", "#1e8e3e", "o"),
                 "finetune_sites": ("unrestricted fine-tune at the same weight matrices", "#202124", "s"),
                 "finetune": ("unrestricted fine-tune at the one best weight matrix", "#80868b", "s"),
                 "random_ft": ("fine-tune restricted to random subcomponents' read/write directions", "#e37400", "o")}
        plt.rcParams.update({"font.size": 15})
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
        for ax, key, ylabel in ((axes[0], "flipped", "share of held-out instances where the preference is gone"),
                                (axes[1], "controls_kept", "share of held-out control instances where\nthe preferred token is still predicted")):
            for m, (label, c, mk) in style.items():
                d = per(m)
                if not d:
                    continue
                xs = [max(v["pile_kl_per_token"], 1e-5) for v in d.values()]
                ys = [v[key] for v in d.values()]
                ax.scatter(xs, ys, color=c, marker=mk, s=36, alpha=0.3)
                lx = [math.log10(x) for x in xs]
                (mx, hx), (my, hy) = ci(lx), ci(ys)
                ax.errorbar([10 ** mx], [my], yerr=[[hy], [hy]], xerr=[[10 ** mx - 10 ** (mx - hx)], [10 ** (mx + hx) - 10 ** mx]], color=c,
                            marker=mk, ms=11, lw=2, capsize=4, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("KL on held-out Pile text after the edit (bits per token)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.08)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=11, loc="lower left")
        fig.tight_layout()
        fig.savefig(figure, dpi=150, facecolor="white")
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
    p.add_argument("--offset", type=int, default=0, help="the circuits of preferences offset, offset + stride, ... alone")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-pairs", type=int, default=MIN_PAIRS, help="a preference's fit-third instances, at least")
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
    e.add_argument("--methods", nargs="*", default=list(METHODS))
    e.add_argument("--rank-from", help="an earlier edit run's rows: each preference's r from its circuit edit")
    e.add_argument("--rank", type=int, help="one r for every preference (over --rank-from and the circuit edit)")
    e.add_argument("--assoc", help="assoc_ft's sets: JSON {\"a->b\": {matrix: [indices]}}")
    t = sub.add_parser("table")
    t.add_argument("rows")
    t.add_argument("--edit", action="store_true", help="edit's rows (practical_table)")
    su = sub.add_parser("summary")
    su.add_argument("rows", nargs="+", help="edit runs' rows (several runs of one preference merged)")
    su.add_argument("--figure")
    args = ap.parse_args()
    read = lambda x: [json.loads(y) for y in Path(x).read_text().splitlines() if y.strip()]  # noqa: E731
    if args.cmd == "summary":
        return print(summary([r for f in args.rows for r in read(f)], args.figure))
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
            for i, beh in enumerate(preferences(ee, src, args.k, decode, log=lambda s: print(s, flush=True), min_pairs=args.min_pairs)):
                if i % args.stride != args.offset:
                    continue
                beh.update(circuit_sets(ee, src, beh, log=lambda s: print(s, flush=True)), order=i)
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
        try:
            if args.cmd == "edit":
                ranks = {(x["a"], x["b"]): len(x["methods"]["circuit_edit"]["ops"]) for x in read(args.rank_from)
                         if x["methods"].get("circuit_edit")} if args.rank_from else {}
                r = practical(nat, beh, tokens, held, args.steps, args.screen, log=lambda s: print(s, flush=True), names=args.sites,
                              methods=tuple(args.methods), rank=args.rank or ranks.get((beh["a"], beh["b"])),
                              assoc=json.loads(Path(args.assoc).read_text()) if getattr(args, "assoc", None) else None)
            else:
                r = run(nat, beh, tokens, held, args.steps, args.screen, log=lambda s: print(s, flush=True), names=args.sites)
        except torch.OutOfMemoryError:  # the preference skipped and reported, the process going on to the next
            print(f"{beh['describe']['a']!r}->{beh['describe']['b']!r}: out of GPU memory, skipped", flush=True)
            torch.cuda.empty_cache()
            continue
        with open(args.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
