"""Teacher v3 (#2951 graph oracle): circuits by gradient-based mask search under an adversary, per text.

Variables: a hard-concrete gate (the L0 relaxation of Louizos et al. 2018: a stretched, clamped sigmoid of a noisy
logit) per VPD-active subcomponent at the positions that reach the targets (VPD's answer: teacher.py's nodes). A run
with gates z has each answer subcomponent at mask z (VPD's masked forward, remainders live) and every other
subcomponent as the adversary sets it.

The adversary picks the worst of a set M of binary patterns. A pattern switches on, at each position, the
subcomponents VPD's causal importance makes active at the same position of another text (patterns that occur on real
inputs; a stand-in until inframove's shared M). It applies to everything outside the circuit: an answer subcomponent
runs at z + (1 - z) m, any other at m; VPD's own answer under the same pattern (paired) has its subcomponents at 1 and
the rest at m. The adversary maximizes ln KL_C,m - ln KL_VPD,m; it is evaluated without gradients over a running set
of the worst patterns and fresh draws, and the gates take the gradient of each sample's worst.

Objective, minimized over the gate logits at a size weight lambda:
    (ln KL_C,m* - ln KL_VPD,m*) / D + max(0, 1 - N) + lambda * size
D = ln(KL_empty / KL_VPD) (the scale of F1 and N); N = ln(KL_{VPD minus C} / KL_VPD) / D with VPD's answer at 1 - z (no
adversary: necessity stays the actual situation); size the expected description bits of the circuit's nodes and
positions over teacher.py's active scale. lambda rises geometrically, each stage warm-started from the last. At each
stage the gates' deterministic values above 1/2 give a circuit, measured exactly: F1 and N (teacher.measure), and
f1_adv, F1 at the worst pattern of the whole of M against VPD's answer under it (log scale, clipped). From teacher.pick's
stage on min(F1, f1_adv) and N (the smallest with both >= 0.9, else the largest minimum) a measured cleanup continues as
teacher.joint_path does (single removals measured at every step, prefixes of 1, 2, 4, ...), each set measured also
under the search's WORST patterns, the mean taken of min(F1, F1 under them) and N: point-F1 pruning alone left
text9000's 29 subcomponents at f1_adv 0.33 where the 55 they came from held 1.00. The teacher circuit is teacher.pick's
over the stages and the cleanup's points, a cleanup point re-measured under the whole of M before it holds the pick; with --edges, its connections by teacher.py's measured descent to graph F1 >= 0.9
(teacher.edge_path; a gradient phase on connections under the same adversary did worse on text9000: graph F1 0.79 at
67 connections, against the descent's 0.909 at 17 on a 27-subcomponent circuit).

  MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python -B teacher_v3.py --tasks FILE --vpd ~/mpd-data/graph_oracle/teacher/vpd_heldout20.pt --out DIR
Output per text in --out: <task>.json (goal_eval circuit JSON), <task>.path.json ({"method": "v3", "start_nodes",
"start_index": every stage's nodes' ids, "steps": per stage its "on" ids and measurements}), summary_<offset>.jsonl.
  teacher_v3.py --evaluate DIR --seed 1 --pool 128 --tasks FILE --vpd FILE --out SCORES.jsonl   (F1, N and f1_adv of
  DIR's circuits under a held-out draw of patterns: either search's circuits, at the same seed)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import carry
import goal_eval
import native
import teacher

BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1   # hard-concrete temperature and stretch (Louizos et al.)
INIT = 3.0             # initial gate logit: every answer subcomponent on (P(z > 0) ~ 0.99)
LR = 0.1               # Adam on the gate logits
STEPS = 20             # gradient steps per lambda
LAMBDAS = [2.0 ** k for k in range(11)]  # the size weight's path, 1 to 1024 (text9000: 49k subcomponents, 114, 55, 30 at 1,
# 10, 100, 1000 with 20 steps each; the gates' logits fall ~LR a step where the size term alone pulls)
SAMPLES = 4            # gate samples per step
POOL = 64              # patterns in M per text
PER_STEP = 8           # patterns evaluated per step: the running worst and fresh draws
TINY = 1e-6            # KL floor inside logarithms (bits)
WORST = 4              # patterns the cleanup measures each set under: the search's worst


def patterns(saved: dict, task: str, T: int, C: dict, names: list, count: int, seed: int) -> list[dict]:
    """M's stand-in: count patterns, each another text's VPD answer position by position (a position past that text's
    end takes one of its positions at random): per matrix the (positions, indices) switched on."""
    g = torch.Generator().manual_seed(seed)
    others = [k for k in saved if k != task]
    out = []
    for _ in range(count):
        src = saved[others[int(torch.randint(len(others), (1,), generator=g))]]["answer"]
        Ts = 1 + max(int(t.max()) for t, _ in src.values() if len(t))
        where = torch.arange(T)
        where[Ts:] = torch.randint(Ts, (max(0, T - Ts),), generator=g)
        pat = {}
        for n in names:
            t, c = src[n]
            on = torch.zeros(Ts, C[n], dtype=torch.bool)
            on[t.long(), c.long()] = True
            pat[n] = tuple(on[where].nonzero().T)
        out.append(pat)
    return out


class MaskSearch:
    """The gates over a text's VPD answer (pr.nodes), their runs and the adversary's patterns. Every run multiplies only
    the subcomponents that the answer or a pattern holds (each matrix's gathered columns, Pruner._gather's way; every
    other one is off in every run): on the 20 held-out texts 9,132 of the 38,912 columns are in any answer."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, pr: teacher.Pruner, pats: list[dict], live: bool = True):
        nat = ge.nat
        self.ge, self.tx, self.pr, self.live = ge, tx, pr, live
        self.names, self.dev, self.T = nat.names, nat.dev, tx.T
        self.offsets, self.cols, self.gathered, at = [], [], [], 0   # every matrix's [T, gathered] in turn, one flat row
        for j, n in enumerate(self.names):
            cols = torch.unique(torch.cat([pr.c[pr.mat == j].cpu()] + [p[n][1] for p in pats]).long()).to(self.dev)
            st = nat.target.site(n)
            self.cols.append(cols)
            self.gathered.append((st.V.index_select(1, cols), st.U.index_select(0, cols)))
            self.offsets.append((at, len(cols)))
            at += tx.T * len(cols)
        self.width = at

        def place(j, t, c):                                    # flat places of (position, subcomponent) pairs of matrix j
            return self.offsets[j][0] + t * self.offsets[j][1] + torch.searchsorted(self.cols[j], c)
        self.flat = torch.empty(len(pr.nodes), dtype=torch.long, device=self.dev)   # each answer subcomponent's place
        for j in range(len(self.names)):
            sel = (pr.mat == j).nonzero().flatten()
            self.flat[sel] = place(j, pr.t[sel], pr.c[sel])
        self.pats = [torch.cat([place(j, p[n][0].to(self.dev), p[n][1].to(self.dev).long()) for j, n in enumerate(self.names)])
                     for p in pats]
        self.logit = torch.full((len(pr.nodes),), INIT, device=self.dev, requires_grad=True)
        self.ids = torch.tensor([tx.ids], device=self.dev)

    def sample(self, S: int) -> torch.Tensor:
        u = torch.rand(S, len(self.logit), device=self.dev).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (-u).log1p() + self.logit) / BETA)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def p_on(self) -> torch.Tensor:
        return torch.sigmoid(self.logit - BETA * math.log(-GAMMA / ZETA))

    def deterministic(self) -> torch.Tensor:
        return (torch.sigmoid(self.logit) * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def masks(self, z: torch.Tensor | None, pats: list | None, vpd_on: bool = False) -> dict:
        """Per matrix [R, T, gathered] (views of one buffer): gates z [R, N] on the answer, each row's pattern (flat
        places, self.pats) elsewhere (z + (1 - z) m inside); vpd_on: the answer at 1 (VPD's own run), z unused."""
        R = len(pats) if z is None else len(z)
        if vpd_on:
            g = torch.zeros(R, self.width, device=self.dev)
            g[:, self.flat] = 1.0
        else:
            at = (torch.arange(R, device=self.dev)[:, None] * self.width + self.flat[None]).flatten()
            g = torch.zeros(R * self.width, device=self.dev).index_put((at,), z.flatten()).view(R, self.width)  # (index_copy
            # along dim 1 takes 36 s on MPS at R 8, this 16 ms)
        if pats is not None:
            m = torch.zeros(R, self.width, device=self.dev)
            rows = torch.repeat_interleave(torch.arange(R, device=self.dev), torch.tensor([len(p) for p in pats], device=self.dev))
            m[rows, torch.cat(pats)] = 1.0
            g = g + (1 - g) * m
        return {n: g[:, at:at + self.T * C].view(R, self.T, C) for n, (at, C) in zip(self.names, self.offsets)}

    def kl(self, masks: dict) -> torch.Tensor:
        """KL in bits at the targets of each row's masked run [R] (VPD's masked forward on the gathered columns; autograd)."""
        nat, tg = self.ge.nat, self.ge.nat.target
        R = next(iter(masks.values())).shape[0]
        for j, n in enumerate(self.names):
            st = tg.site(n)
            st.mask, st.gathered = masks[n], self.gathered[j]
            st.delta_mask = torch.ones(R, self.T, device=self.dev) if self.live else None
        try:
            hidden = tg.hidden(self.ids.expand(R, -1), fused=True)
        finally:
            nat.vpd.clear()
        lq = torch.log_softmax((hidden[:, self.tx.targets] @ tg.wte.T).float(), -1)
        return carry.kl(self.tx.ref, lq)

    @torch.no_grad()
    def kl_chunked(self, make, R: int, chunk: int) -> torch.Tensor:
        return torch.cat([self.kl(make(s0, min(R, s0 + chunk))) for s0 in range(0, R, chunk)])


def size_soft(ms: MaskSearch, p: torch.Tensor, node_bits: float, pos_bits: float, scale: float) -> torch.Tensor:
    """Expected size term: E[nodes] node bits + E[positions held] position bits, over the active scale."""
    miss = torch.zeros(ms.T, device=p.device).index_add(0, ms.pr.t, (-p).log1p())
    return (p.sum() * node_bits + (1 - miss.exp()).sum() * pos_bits) / scale


def search(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], vpd_answer: dict, saved: dict, task: str,
           B: int = 32, chunk: int = 8, seed: int = 0, edges: bool = False) -> tuple[dict, dict, dict]:
    t0 = time.perf_counter()
    nat = ge.nat
    tx = ge.text(ids, targets)
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in vpd_answer.items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    D = math.log(empty / vpd)
    pr = teacher.Pruner(ge, tx, tx.vpd, live=teacher.CONVENTION == "live", B=B)
    positions = sorted({p for p, _ in tx.f2} | set(targets))
    active = sum(int(tx.vpd[n][positions].sum()) for n in nat.names)
    P = nat.positions(targets)
    node_bits, pos_bits = math.log2(P * sum(nat.C.values())), math.log2(P)
    scale = active * node_bits
    ms = MaskSearch(ge, tx, pr, patterns(saved, task, tx.T, nat.C, nat.names, POOL, seed), live=teacher.CONVENTION == "live")
    pats = ms.pats
    torch.manual_seed(seed)
    vpd_adv = ms.kl_chunked(lambda a, b: ms.masks(None, pats[a:b], vpd_on=True), len(pats), chunk).clamp_min(TINY).log()
    worst = torch.zeros(len(pats), device=nat.dev)            # running adversary score per pattern
    opt = torch.optim.Adam([ms.logit], lr=LR)
    vpd_masks = {n: tx.vpd[n].float() for n in nat.names}
    stages = []
    for lam in LAMBDAS:
        t1 = time.perf_counter()
        for _ in range(STEPS):
            z = ms.sample(SAMPLES)
            top = torch.argsort(worst, descending=True)[:PER_STEP // 2].tolist()
            draw = [i for i in torch.randperm(len(pats)).tolist() if i not in top][:PER_STEP - len(top)]
            cand = top + draw
            with torch.no_grad():                               # the adversary: each sample's worst candidate pattern
                rows = [(s, i) for s in range(SAMPLES) for i in cand]
                kl = ms.kl_chunked(lambda a, b: ms.masks(z[[s for s, _ in rows[a:b]]], [pats[i] for _, i in rows[a:b]]), len(rows), chunk)
                score = (kl.clamp_min(TINY).log() - vpd_adv[[i for _, i in rows]]).view(SAMPLES, len(cand))
                worst[cand] = torch.maximum(worst[cand] * 0.9, score.max(0).values)
                pick_ = score.argmax(1).tolist()
            opt.zero_grad()
            adv = ms.kl(ms.masks(z, [pats[cand[k]] for k in pick_]))
            gate = ms.masks(z, None)
            minus = ms.kl({n: vpd_masks[n] * (1 - gate[n]) for n in nat.names})
            f_term = (adv.clamp_min(TINY).log() - vpd_adv[[cand[k] for k in pick_]]) / D
            n_term = (1 - (minus.clamp_min(TINY).log() - math.log(vpd)) / D).clamp_min(0)
            loss = f_term.mean() + n_term.mean() + lam * size_soft(ms, ms.p_on(), node_bits, pos_bits, scale)
            loss.backward()
            opt.step()
        on = ms.deterministic().detach() > 0.5
        st = measure_stage(ms, pr, on, empty, vpd, vpd_adv, chunk)
        k_tok = int(torch.unique(pr.t[on]).numel())
        st.update(lam=lam, size=int(on.sum()), positions=k_tok, on=on.nonzero().flatten().tolist(),
                  size_term=min(1.0, (int(on.sum()) * node_bits + k_tok * pos_bits) / scale), seconds=time.perf_counter() - t1)
        stages.append(st)
        print(f"  lambda {lam:.3g}: {st['size']} subcomponents at {k_tok} positions, F1 {st['f1']:.3f} N {st['n']:.3f} "
              f"f1_adv {st['f1_adv']:.3f}, {st['seconds']:.0f}s", flush=True)
        if int(on.sum()) <= teacher.MIN_SIZE:
            break
    best, rule = teacher.pick(stages, "f1_rob")
    t1 = time.perf_counter()                                   # the measured cleanup from the pick, under the worst patterns
    worst_ids = torch.argsort(worst, descending=True)[:WORST].tolist()
    on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    on[stages[best]["on"]] = True
    for r in (robust_path(ms, pr, on, empty, vpd, vpd_adv, worst_ids, chunk)[1:] if int(on.sum()) <= teacher.K_MAX else []):
        on[r["removed"]] = False                                # (~size^2 / 2 sets, each under WORST patterns)
        k_tok = int(torch.unique(pr.t[on]).numel())
        stages.append({**r, "lam": None, "cleanup": True, "positions": k_tok, "on": on.nonzero().flatten().tolist(),
                       "size_term": min(1.0, (int(on.sum()) * node_bits + k_tok * pos_bits) / scale)})
    while True:                                                 # a cleanup point holds its pick only under the whole of M
        picked, rule = teacher.pick(stages, "f1_rob")
        if not stages[picked].get("cleanup") or "f1_adv" in stages[picked]:
            break
        on_p = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
        on_p[stages[picked]["on"]] = True
        stages[picked].update(measure_stage(ms, pr, on_p, empty, vpd, vpd_adv, chunk))
    best = picked
    print(f"  cleanup: {sum(1 for s in stages if s.get('cleanup'))} steps, pick {stages[best]['size']} "
          f"({'cleanup' if stages[best].get('cleanup') else 'search'}), {time.perf_counter() - t1:.0f}s", flush=True)
    ids_all = sorted({i for s in stages for i in s["on"]})
    on_best = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    on_best[stages[best]["on"]] = True
    ids_of = {nd: i for i, nd in enumerate(pr.nodes)}
    wired = None
    if edges:                                                  # the pick's connections: teacher.py's measured descent
        n, k_tok = stages[best]["size"], stages[best]["positions"]
        ep = teacher.EdgePruner(ge, tx, [pr.nodes[i] for i in stages[best]["on"]], teacher.CONVENTION == "live", B)

        def state(kl, on, removed):
            inner, out = int(on[:ep.inner].sum()), int(on[ep.inner:].sum())
            return {"nodes": n, "edges": int(on.sum()), "removed": removed, "kl_graph": kl,
                    "f1_graph": goal_eval._clip01(goal_eval.recovered_log(empty, kl, vpd)), "n": stages[best]["n"],
                    "size_term": min(1.0, (n * node_bits + k_tok * pos_bits + math.log2(n + k_tok) * (2 * inner + out)) / scale)}
        e_steps = teacher.edge_path(ep, empty ** (1 - teacher.RECOVERED) * vpd ** teacher.RECOVERED, state)
        wired = [e for e, keep in zip(ep.edges, ep.on.tolist()) if keep]
        path_edges = {"step": best, "edges": [[ids_of[w], "out" if r is None else ids_of[r]] for w, r in ep.edges], "steps": e_steps}
        print(f"  connections: {e_steps[-1]['edges']} of {len(ep.edges)}, graph F1 {e_steps[-1]['f1_graph']:.3f} "
              f"(all {e_steps[0]['f1_graph']:.3f}), {ep.measured} sets", flush=True)
    circuits = {"": teacher.circuit_json(pr, on_best, wired)}
    row = {"T": tx.T, "kl_empty": empty, "kl_vpd": vpd, "vpd_size": len(pr.nodes), "active_scale": active, "pick": rule,
           "best": {k: stages[best][k] for k in ("size", "positions", "f1", "n", "f1_adv", "size_term", "lam", "kl_c", "kl_minus")},
           "stages": len(stages), "seconds": time.perf_counter() - t0}
    if wired is not None:
        row["best_edges"] = {k: e_steps[-1][k] for k in ("nodes", "edges", "kl_graph", "f1_graph", "n", "size_term")}
    path_file = {"method": "v3", "start_nodes": [[pr.nodes[i][0], pr.nodes[i][2], pr.nodes[i][1]] for i in ids_all],
                 "start_index": ids_all, "steps": stages}
    if wired is not None:
        path_file["edge_paths"] = [path_edges]
    return circuits, row, path_file


@torch.no_grad()
def measure_stage(ms: MaskSearch, pr: teacher.Pruner, on: torch.Tensor, empty: float, vpd: float, vpd_adv: torch.Tensor,
                  chunk: int) -> dict:
    """A binary circuit's exact F1 and N (teacher.measure) and its F1 at the worst pattern of M."""
    got = teacher.measure(pr, on[None], empty, vpd)[0]
    z = on.float()[None]
    kl = ms.kl_chunked(lambda a, b: ms.masks(z.expand(b - a, -1), ms.pats[a:b]), len(ms.pats), chunk).clamp_min(TINY)
    i = int(torch.argmax(kl.log() - vpd_adv))
    va = float(vpd_adv[i].exp())
    f1_adv = goal_eval._clip01(math.log(empty / float(kl[i])) / math.log(empty / va)) if empty > va else 0.0
    return {**got, "kl_adv": float(kl[i]), "kl_vpd_adv": va, "f1_adv": f1_adv, "f1_rob": min(got["f1"], f1_adv), "pattern": i}


@torch.no_grad()
def robust_measure(ms: MaskSearch, pr: teacher.Pruner, ons: torch.Tensor, empty: float, vpd: float, vpd_adv: torch.Tensor,
                   worst: list, chunk: int) -> list[dict]:
    """Per set (ons [K, N]): F1 and N (teacher.measure), F1 at the worst of the patterns `worst`, f1_rob the smaller F1,
    and mean, of f1_rob and N."""
    base = teacher.measure(pr, ons, empty, vpd)
    rows = [(k, i) for k in range(len(ons)) for i in worst]
    z = ons.float()
    kl = ms.kl_chunked(lambda a, b: ms.masks(z[[k for k, _ in rows[a:b]]], [ms.pats[i] for _, i in rows[a:b]]), len(rows), chunk)
    kl = kl.clamp_min(TINY).view(len(ons), len(worst))
    ratio = kl.log() - vpd_adv[worst][None]
    out = []
    for k, r in enumerate(base):
        j = int(ratio[k].argmax())
        va = float(vpd_adv[worst[j]].exp())
        fa = goal_eval._clip01(math.log(empty / float(kl[k, j])) / math.log(empty / va)) if empty > va else 0.0
        out.append({**r, "f1_adv_worst": fa, "f1_rob": min(r["f1"], fa), "mean": (min(r["f1"], fa) + r["n"]) / 2})
    return out


def robust_path(ms: MaskSearch, pr: teacher.Pruner, start: torch.Tensor, empty: float, vpd: float, vpd_adv: torch.Tensor,
                worst: list, chunk: int) -> list[dict]:
    """teacher.joint_path with every set measured by robust_measure: each step ranks the single removals by the mean of
    min(F1, F1 at the worst patterns) and N they leave and takes the measured prefix of 1, 2, 4, ... with the highest."""
    on = start.clone()
    path = [{"size": int(on.sum()), "removed": [], **robust_measure(ms, pr, on[None], empty, vpd, vpd_adv, worst, chunk)[0]}]
    while int(on.sum()) > teacher.MIN_SIZE:
        ids = on.nonzero().flatten()
        trials = on[None].repeat(len(ids), 1)
        trials[torch.arange(len(ids), device=on.device), ids] = False
        rows = []
        for s0 in range(0, len(ids), 4 * pr.B):
            rows += robust_measure(ms, pr, trials[s0:s0 + 4 * pr.B], empty, vpd, vpd_adv, worst, chunk)
        ranked = ids[torch.argsort(torch.tensor([r["mean"] for r in rows]), descending=True, stable=True)]
        top = min(max(1, int(teacher.FRACTION * len(ids))), len(ids) - teacher.MIN_SIZE)
        ks = sorted({min(top, 2 ** e) for e in range(0, 20) if 2 ** e <= 2 * top})
        prefixes = on[None].repeat(len(ks), 1)
        for i, k in enumerate(ks):
            prefixes[i, ranked[:k]] = False
        got = robust_measure(ms, pr, prefixes, empty, vpd, vpd_adv, worst, chunk)
        i = max(range(len(ks)), key=lambda j: (got[j]["mean"], ks[j]))
        on[ranked[:ks[i]]] = False
        path.append({"size": int(on.sum()), "removed": ranked[:ks[i]].tolist(), **got[i]})
    return path


def evaluate(ge: goal_eval.GoalEval, task: str, saved: dict, circuit: dict, seed: int, pool: int, chunk: int) -> dict:
    """A circuit JSON's F1, N and f1_adv on its text under `pool` patterns drawn with `seed` (a held-out adversary for a
    search that drew its own with another seed); connections are not used (node circuits)."""
    nat = ge.nat
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
    tx = ge.text(ids, targets)
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in saved[task]["answer"].items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    pr = teacher.Pruner(ge, tx, tx.vpd, live=teacher.CONVENTION == "live")
    ms = MaskSearch(ge, tx, pr, patterns(saved, task, tx.T, nat.C, nat.names, pool, seed), live=teacher.CONVENTION == "live")
    pats = ms.pats
    vpd_adv = ms.kl_chunked(lambda a, b: ms.masks(None, pats[a:b], vpd_on=True), len(pats), chunk).clamp_min(TINY).log()
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    outside = 0
    for m, c, t in circuit["nodes"]:
        if m != "tok":
            k = index.get((goal_eval._site(m), int(t), int(c)))
            outside += k is None
            if k is not None:
                on[k] = True
    r = measure_stage(ms, pr, on, empty, vpd, vpd_adv, chunk)
    return {"task": task, "size": int(on.sum()), "outside_answer": outside, "seed": seed, "pool": pool,
            **{k: r[k] for k in ("f1", "n", "f1_adv", "f1_rob", "kl_c", "kl_minus", "kl_adv", "kl_vpd_adv")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, required=True, help="task ids, one per line")
    ap.add_argument("--vpd", type=Path, required=True, help="VPD's answers saved by teacher.py --save-vpd (also M's source)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8, help="rows per masked run (memory)")
    ap.add_argument("--edges", action="store_true", help="the pick's connections by teacher.py's measured descent to graph F1 >= 0.9")
    ap.add_argument("--evaluate", type=Path, help="score this directory's <task>.json circuits instead of searching (one JSONL line each in --out)")
    ap.add_argument("--seed", type=int, default=1, help="--evaluate: the patterns' draw (the search draws with 0)")
    ap.add_argument("--pool", type=int, default=128, help="--evaluate: patterns drawn")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    saved = torch.load(args.vpd)
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    if args.evaluate:
        for task in args.tasks.read_text().split()[args.offset::args.stride]:
            path = args.evaluate / f"{task}.json"
            if path.exists():
                r = evaluate(ge, task, saved, json.loads(path.read_text()), args.seed, args.pool, args.chunk)
                with args.out.open("a") as f:
                    f.write(json.dumps({"dir": str(args.evaluate), **r}) + "\n")
                print(f"{task}: {r['size']} subcomponents, F1 {r['f1']:.3f} N {r['n']:.3f} f1_adv {r['f1_adv']:.3f}", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks.read_text().split()[args.offset::args.stride]:
        if (args.out / f"{task}.json").exists():
            continue
        ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
        circuits, row, path_file = search(ge, ids, targets, saved[task]["answer"], saved, task, args.batch, args.chunk,
                                       edges=args.edges)
        for name, c in circuits.items():
            (args.out / f"{task}{'.' + name if name else ''}.json").write_text(json.dumps(c))
        (args.out / f"{task}.path.json").write_text(json.dumps(path_file))
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps({"task": task, **row}) + "\n")
        b = row["best"]
        how = "cleanup" if b["lam"] is None else f"lambda {b['lam']:.3g}"
        print(f"{task}: pick ({row['pick']}) {b['size']} at {b['positions']} positions F1 {b['f1']:.3f} N {b['n']:.3f} "
              f"f1_adv {b['f1_adv']:.3f} ({how}); {row['stages']} stages, {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
