"""Teacher v3 (#2951 graph oracle): circuits by gradient-based mask search under the omitted-flips adversary, per text.

Variables: a hard-concrete gate (the L0 relaxation of Louizos et al. 2018: a stretched, clamped sigmoid of a noisy
logit) per VPD-active subcomponent at the positions that reach the targets (VPD's answer: teacher.py's nodes). A run
with gates z has each answer subcomponent at mask z and every other subcomponent off, the remainders live (VPD's
masked forward on engine.Engine over the answer's columns; bfloat16 on CUDA for the gradient steps).

The adversary is the team's (goal_eval.GoalEval.adversary_flips' omitted lever), relaxed: per gate sample a lever b in
[0, 1] on the answer's subcomponents, projected onto its REPAIR_M largest entries, the run at z + (1 - z) b (the
answer's subcomponents the circuit leaves out, switched on), raised by sign-gradient steps of ADV_STEP on the run's KL
and carried from step to step. VPD's own run does not change under the lever (its answer is on), so the paired score
is ln KL_C,b - ln KL_VPD.

Objective, minimized over the gate logits at a size weight lambda:
    (ln KL_C,b - ln KL_VPD) / D + max(0, 1 - N) + lambda * size
D = ln(KL_empty / KL_VPD) (the scale of F1 and N); N = ln(KL_{VPD minus C} / KL_VPD) / D with VPD's answer at 1 - z (no
adversary: necessity stays the actual situation); size the expected description bits of the circuit's nodes and
positions, goal_eval.size_term over goal_eval.size_scale. lambda rises geometrically, each stage warm-started from the
last. At each stage the gates' deterministic values above 1/2 give a circuit, measured in float32: F1 and N
(teacher.measure), and f1_flips, F1 under the omitted lever's REPAIR_M flips (adversary_flips on the node set as a
complete graph). The pick: teacher.pick's among the stages within the answer budget (--max-nodes, teacher.MAX_NODES),
on min(F1, f1_flips) and N: the smallest with both >= 0.9, else the largest minimum; never VPD's whole answer: with no
stage within the budget, the gates' largest --max-nodes logits.

The cleanup, measured around the pick: removals by the F1 search's rule on min(F1, N) (Descent: single removals
measured and ranked, the most of them in that order that keep it >= 0.9), the result kept only while f1_flips holds
(else the removal halved); with --edges, the connections by teacher.Connector (C's semantics, the scorecard's; with
--edge-objective robust each removal also keeping F1 under the flips, teacher.py's) and the flip repair
(teacher.Connector.repair: the omitted lever's flips joined, the connections pruned again from the kept
ones, while min(graph F1, node F1, F1 under the flips) improves: node F1, the node set with every connection among
its nodes, so that a flip joined with its connections cut does not count); while the answer is longer than
--max-tokens (components.ANSWER_BUDGET), the pick among the smaller stages instead.

With --seeds S, S searches (seeds 0..S-1; the gates' noise and initial draws), and the written circuit the one with
the reward's best circuit score, mean(C, N) - 0.5 size (C: goal_eval.GoalEval.fidelity, mode "reward"; size
goal_eval.size_term): valid robust circuits are many (float noise alone moved fleet's picks on 4 of 4 texts), so the
teacher is the best of several; the summary's "seeds" rows give the spread. The gradient steps run in float32.

  MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python -B teacher_v3.py --tasks FILE --vpd ~/mpd-data/graph_oracle/teacher/vpd_heldout20.pt --edges --out DIR
Output per text in --out: <task>.json (goal_eval circuit JSON), <task>.path.json ({"method": "v3", "start_nodes",
"start_index": every stage's nodes' ids, "steps": per stage its "on" ids and measurements, "cleanup", "repair"}),
summary_<offset>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import components
import engine
import goal_eval
import native
import teacher

BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1   # hard-concrete temperature and stretch (Louizos et al.)
INIT = 3.0             # initial gate logit: every answer subcomponent on (P(z > 0) ~ 0.99)
LR = 0.1               # Adam on the gate logits
STEPS = 20             # gradient steps per lambda
LAMBDAS = [2.0 ** k for k in range(11)]  # the size weight's path, 1 to 1024
SAMPLES = 4            # gate samples per step
ADV_STEP = 0.1         # the lever's sign-gradient step (goal_eval's ADV_STEP)
INNER = 2              # the lever's steps per gradient step
TINY = 1e-6            # KL floor inside logarithms (bits)


class MaskSearch:
    """The gates over a text's VPD answer (pr.nodes) and their runs on the engine over the answer's columns."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, pr: teacher.Pruner, live: bool = True):
        nat = ge.nat
        self.ge, self.tx, self.pr, self.live = ge, tx, pr, live
        self.names, self.dev, self.T = nat.names, nat.dev, tx.T
        self.cols = [torch.unique(pr.c[pr.mat == j]) for j in range(len(self.names))]
        self.eng = engine.Engine(nat, tx.ids, tx.targets, live=live, columns=dict(zip(self.names, self.cols)))
        self.offsets, at = [], 0                                 # every matrix's [T, columns] in turn, one flat row
        for cols in self.cols:
            self.offsets.append((at, len(cols)))
            at += tx.T * len(cols)
        self.width = at
        self.flat = torch.empty(len(pr.nodes), dtype=torch.long, device=self.dev)   # each answer subcomponent's place
        for j, cols in enumerate(self.cols):
            sel = (pr.mat == j).nonzero().flatten()
            self.flat[sel] = self.offsets[j][0] + pr.t[sel] * len(cols) + torch.searchsorted(cols, pr.c[sel])
        self.logit = torch.full((len(pr.nodes),), INIT, device=self.dev, requires_grad=True)
        self.search = "score"   # float32 gradient steps: bfloat16 noise moved fleet's picks on 4 of 4 texts (f44147b)

    def sample(self, S: int) -> torch.Tensor:
        u = torch.rand(S, len(self.logit), device=self.dev).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (-u).log1p() + self.logit) / BETA)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def p_on(self) -> torch.Tensor:
        return torch.sigmoid(self.logit - BETA * math.log(-GAMMA / ZETA))

    def deterministic(self) -> torch.Tensor:
        return (torch.sigmoid(self.logit) * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def masks(self, m: torch.Tensor) -> dict:
        """Per matrix [R, T, columns] (views of one buffer) of the answer subcomponents' masks m [R, N]."""
        R = len(m)
        at = (torch.arange(R, device=self.dev)[:, None] * self.width + self.flat[None]).flatten()
        g = torch.zeros(R * self.width, device=self.dev).index_put((at,), m.flatten()).view(R, self.width)  # (index_copy
        # along dim 1 takes 36 s on MPS at R 8, this 16 ms)
        return {n: g[:, a:a + self.T * K].view(R, self.T, K) for n, (a, K) in zip(self.names, self.offsets)}

    def kl(self, m: torch.Tensor, precision: str = "score") -> torch.Tensor:
        """KL in bits at the targets of each row's masked run [R] (autograd through m)."""
        return self.eng.kl(self.masks(m), precision)


def top(x: torch.Tensor, m: int) -> torch.Tensor:
    """Each row's m largest entries kept, the rest 0."""
    return x * torch.zeros_like(x, dtype=torch.bool).scatter(1, x.topk(min(m, x.shape[1]), 1).indices, True)


def size_soft(ms: MaskSearch, p: torch.Tensor, node_bits: float, pos_bits: float, scale: float) -> torch.Tensor:
    """Expected size term: goal_eval.size_term of E[nodes] node bits + E[positions held] position bits."""
    miss = torch.zeros(ms.T, device=p.device).index_add(0, ms.pr.t, (-p).log1p())
    bits = p.sum() * node_bits + (1 - miss.exp()).sum() * pos_bits
    return (1 + bits).log2() / math.log2(1 + scale)


class Measure:
    """A text's float32 measurements of node sets: F1 and N (teacher.measure), F1 under the omitted lever's flips."""

    def __init__(self, ge, tx, pr, empty, vpd, pool, m=teacher.REPAIR_M):
        self.ge, self.tx, self.pr, self.empty, self.vpd, self.pool, self.m = ge, tx, pr, empty, vpd, pool, m
        self.calls = 0

    def complete(self, on: torch.Tensor) -> native.Graph:
        """The node set as a complete graph (every connection among its nodes)."""
        nat = self.ge.nat
        masks = {n: torch.zeros(self.tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for j in on.nonzero().flatten().tolist():
            n, t, c = self.pr.nodes[j]
            masks[n][t, c] = True
        return native.Graph(masks=masks)

    def flips(self, on: torch.Tensor, graph: native.Graph | None = None) -> float:
        """F1 under the flips of the node set as a complete graph (every connection), or of a stated graph."""
        return self.flips_many([graph if graph is not None else self.complete(on)])[0]

    def flips_many(self, graphs: list, screens: dict | None = None) -> list[float]:
        """flips() of several graphs in one adversary_flips call (screens: its shortlist gaps shared by them and kept
        by the caller across calls)."""
        rows = self.ge.adversary_flips(self.tx, graphs, self.pool, "omitted", ms=(self.m,), screens=screens)
        self.calls += len(graphs)
        return [1.0 if f["f1_flips_omitted"].get(self.m) is None else f["f1_flips_omitted"][self.m]  # nothing omitted:
                for f in rows]                                                                   # nothing to switch on

    def __call__(self, on: torch.Tensor) -> dict:
        got = teacher.measure(self.pr, on[None], self.empty, self.vpd)[0]
        fl = self.flips(on)
        return {**got, "f1_flips": fl, "f1_rob": min(got["f1"], fl)}


def removals(meas: Measure, on: torch.Tensor, row: dict, goal: float = teacher.RECOVERED) -> tuple[torch.Tensor, dict, list]:
    """Measured removals around a node set: rounds of the F1 search's rule on min(F1, N) (every single removal measured
    and ranked, the most of them in that order, at sqrt(2)-spaced prefix sizes, that keep it >= min(goal, the set's)),
    each round's removal kept only while F1 under the flips stays >= min(goal, the set's), halved until it does: the
    halvings measured together (one flips call per round, their shortlist gaps shared, a round's own), the largest that
    holds kept. (the node set, its row, each round's {"removed", "size", ...})."""
    pr, log = meas.pr, []
    floor_mn, floor_fl = min(goal, min(row["f1"], row["n"])), min(goal, row["f1_flips"])
    for _ in range(teacher.ROUNDS):
        ids = on.nonzero().flatten()
        if len(ids) <= teacher.MIN_SIZE:
            break
        trials = on[None].repeat(len(ids), 1)
        trials[torch.arange(len(ids), device=on.device), ids] = False
        single = []
        for s0 in range(0, len(ids), 4 * pr.B):
            single += teacher.measure(pr, trials[s0:s0 + 4 * pr.B], meas.empty, meas.vpd)
        ranked = ids[torch.argsort(torch.tensor([min(r["f1"], r["n"]) for r in single]), descending=True, stable=True)]
        sizes = [k for k in sorted({int(2 ** (e / 2)) for e in range(0, 60)}) if k <= len(ids) - teacher.MIN_SIZE]
        prefixes = on[None].repeat(len(sizes), 1)
        for i, k in enumerate(sizes):
            prefixes[i, ranked[:k]] = False
        got = teacher.measure(pr, prefixes, meas.empty, meas.vpd)
        ok = [k for k, r in zip(sizes, got) if min(r["f1"], r["n"]) >= floor_mn]
        ladder, k, new = [], max(ok, default=0), None
        while k:
            ladder.append(k)
            k //= 2
        if ladder:
            cands = on[None].repeat(len(ladder), 1)
            for i, k_ in enumerate(ladder):
                cands[i, ranked[:k_]] = False
            base = teacher.measure(pr, cands, meas.empty, meas.vpd)
            fl = meas.flips_many([meas.complete(c) for c in cands], screens={})
            for k_, cand, r, f in zip(ladder, cands, base, fl):
                if f >= floor_fl:
                    k, new = k_, (cand, {**r, "f1_flips": f, "f1_rob": min(r["f1"], f)})
                    break
        if new is None:
            break
        on, row = new
        log.append({"removed": ranked[:k].tolist(), "size": int(on.sum()), **{k_: row[k_] for k_ in ("f1", "n", "f1_flips")}})
    return on, row, log


def search(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], vpd_answer: dict, pool: dict, B: int = 32,
           edges: bool = False, repair: bool = False, max_nodes: int | None = teacher.MAX_NODES, tokens=None,
           max_tokens: int = components.ANSWER_BUDGET, seeds=(0,), robust_edges: bool = False) -> tuple[dict, dict, dict]:
    t0 = time.perf_counter()
    nat = ge.nat
    tx = ge.text(ids, targets)
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in vpd_answer.items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    D = math.log(empty / vpd)
    pr = teacher.Pruner(ge, tx, tx.vpd, live=teacher.CONVENTION == "live", B=B)
    P = nat.positions(targets)
    node_bits, pos_bits = math.log2(P * sum(nat.C.values())), math.log2(P)
    scale = ge.size_scale(tx)

    def size_term(on, inner=0, out=0):                         # teacher.py's: goal_eval.size_bits over size_scale
        n, k_tok = int(on.sum()), int(torch.unique(pr.t[on]).numel())
        bits = n * node_bits + k_tok * pos_bits + (math.log2(n + k_tok) * (2 * inner + out) if n else 0.0)
        return goal_eval.size_term(bits, scale)

    meas = Measure(ge, tx, pr, empty, vpd, pool)
    wiring = teacher.Connector(ge, tx, pr, empty, vpd, size_term, B, robust=pool if robust_edges else None)

    def lam_path(seed):                                          # one seed's stages
        ms = MaskSearch(ge, tx, pr, live=teacher.CONVENTION == "live")
        torch.manual_seed(seed)
        opt = torch.optim.Adam([ms.logit], lr=LR)
        lever = torch.zeros(SAMPLES, len(pr.nodes), device=nat.dev)
        stages = []
        for lam in LAMBDAS:
            t1 = time.perf_counter()
            for _ in range(STEPS):
                z = ms.sample(SAMPLES)
                zd = z.detach()
                for _ in range(INNER):                           # the lever: sign steps on the run's KL, top REPAIR_M
                    b = lever.clone().requires_grad_(True)
                    g, = torch.autograd.grad(ms.kl(zd + (1 - zd) * b, ms.search).clamp_min(TINY).log().sum(), [b])
                    lever = top((lever + ADV_STEP * g.sign()).clamp(0, 1), teacher.REPAIR_M)
                opt.zero_grad()
                adv = ms.kl(z + (1 - z) * lever, ms.search)
                minus = ms.kl(1 - z, ms.search)                 # VPD's answer less the circuit
                f_term = (adv.clamp_min(TINY).log() - math.log(vpd)) / D
                n_term = (1 - (minus.clamp_min(TINY).log() - math.log(vpd)) / D).clamp_min(0)
                loss = f_term.mean() + n_term.mean() + lam * size_soft(ms, ms.p_on(), node_bits, pos_bits, scale)
                loss.backward()
                opt.step()
            on = ms.deterministic().detach() > 0.5
            st = meas(on) if int(on.sum()) <= (max_nodes or len(pr.nodes)) else teacher.measure(pr, on[None], empty, vpd)[0]
            k_tok = int(torch.unique(pr.t[on]).numel())
            st.update(lam=lam, size=int(on.sum()), positions=k_tok, on=on.nonzero().flatten().tolist(), size_term=size_term(on),
                      seconds=time.perf_counter() - t1)
            stages.append(st)
            print(f"  seed {seed}, lambda {lam:.3g}: {st['size']} subcomponents at {k_tok} positions, F1 {st['f1']:.3f} N {st['n']:.3f}"
                  + (f" F1 under {meas.m} flips {st['f1_flips']:.3f}" if "f1_flips" in st else " (over the budget: not attacked)")
                  + f", {st['seconds']:.0f}s", flush=True)
            if int(on.sum()) <= teacher.MIN_SIZE:
                break
        if not any("f1_rob" in s_ for s_ in stages) and max_nodes:  # every stage over the budget: the gates' largest max_nodes
            on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
            on[ms.logit.detach().topk(max_nodes).indices] = True
            st = meas(on)
            st.update(lam=None, size=int(on.sum()), positions=int(torch.unique(pr.t[on]).numel()), on=on.nonzero().flatten().tolist(),
                      size_term=size_term(on), seconds=0.0, budget=True)
            stages.append(st)
            print(f"  no stage within {max_nodes} subcomponents: the gates' largest {max_nodes}, F1 {st['f1']:.3f} N {st['n']:.3f} "
                  f"F1 under {meas.m} flips {st['f1_flips']:.3f}", flush=True)
        return stages

    def finish(stages, best):                                    # cleanup, connections, repair from one stage
        on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
        on[stages[best]["on"]] = True
        t1 = time.perf_counter()                                 # measured removals under the flips' guard
        on, picked, cut = removals(meas, on, dict(stages[best]))
        print(f"  cleanup: {len(cut)} rounds removed {stages[best]['size'] - int(on.sum())} of {stages[best]['size']} "
              f"(F1 {picked['f1']:.3f}, N {picked['n']:.3f}, F1 under flips {picked['f1_flips']:.3f}), {time.perf_counter() - t1:.0f}s", flush=True)
        out = {"stage": best, "cut": cut, "removed": stages[best]["size"] - int(on.sum()), "kept": None, "rep": None,
               "edge_path": None, "f1": picked["f1"]}
        if edges:
            kept, summary, e_steps, ep = wiring.connect(on, picked["n"])
            out.update(kept=kept, f1=summary["f1_graph"], edge_path={
                "edges": [[wiring.index[w], "out" if r is None else wiring.index[r]] for w, r in ep.edges], "steps": e_steps})
            if repair:
                rep, k_, rule_ = wiring.repair(on, kept, summary, pool, max_nodes)
                out.update(rep=rep, repair_pick=k_, repair_rule=rule_, added=int(rep[k_]["on"].sum()) - int(on.sum()),
                           kept=rep[k_]["kept"], f1=rep[k_]["f1_graph"])
                on = rep[k_]["on"]
        out["on"] = on
        return out

    def score(out):                                              # the reward's circuit score: mean(C, N) - 0.5 size
        on, kept = out["on"], out["kept"]
        if kept is not None:
            g = wiring.graph(on, kept)
        else:                                                    # no connections stated: the node set as a complete graph
            masks = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            for j in on.nonzero().flatten().tolist():
                n, t, c = pr.nodes[j]
                masks[n][t, c] = True
            g = native.Graph(masks=masks)
        f = ge.fidelity(tx, [g], pool, mode="reward")[0]
        inner = 0 if kept is None else sum(r is not None for _, r in kept)
        st = size_term(on, inner, 0 if kept is None else len(kept) - inner)
        return {"c": f["c"], "n": f["n_point"], "size_term": st, "score": (f["c"] + f["n_point"]) / 2 - teacher.SIZE_WEIGHT * st}

    runs = []
    for seed in seeds:
        stages = lam_path(seed)
        cands = [i for i, s_ in enumerate(stages) if "f1_rob" in s_]
        while True:                                              # the pick, a smaller stage while the answer is too long
            j, rule = teacher.pick([stages[i] for i in cands], "f1_rob")
            out = finish(stages, cands[j])
            tok = tokens(components.render(teacher.circuit_json(pr, out["on"], out["kept"]))) if tokens else None
            smaller = [i for i in cands if stages[i]["size"] < stages[cands[j]]["size"]]
            if tok is None or tok <= max_tokens or not smaller:
                break
            print(f"  answer {tok} tokens over {max_tokens}: a smaller stage", flush=True)
            cands = smaller
        out.update(seed=seed, rule=rule, stages=stages, tokens=tok, **score(out))
        print(f"  seed {seed}: {int(out['on'].sum())} subcomponents, C {out['c']:.3f} N {out['n']:.3f} size {out['size_term']:.3f} "
              f"score {out['score']:.3f}" + ("" if tok is None else f", {tok} tokens"), flush=True)
        runs.append(out)
    best = max(range(len(runs)), key=lambda i: runs[i]["score"])
    out = runs[best]
    on, kept, stages = out["on"], out["kept"], out["stages"]
    final = teacher.measure(pr, on[None], empty, vpd)[0]          # the answer as written: the stated graph if any
    final["f1_flips"] = meas.flips(on, None if kept is None else wiring.graph(on, kept))
    final["f1_nodes"], final["f1"] = final["f1"], out["f1"]      # node F1 (every connection) beside the written F1
    final["f1_rob"] = min(out["f1"], final["f1_nodes"], final["f1_flips"])
    row = {"T": tx.T, "kl_empty": empty, "kl_vpd": vpd, "vpd_size": len(pr.nodes), "size_scale": scale, "pick": out["rule"],
           "seed": out["seed"], "stage": {k: stages[out["stage"]][k] for k in ("size", "positions", "f1", "n", "f1_flips", "size_term", "lam")},
           "cleanup_removed": out["removed"], "repair_added": out.get("added"), "answer_tokens": out["tokens"],
           "seeds": [{"seed": r["seed"], "size": int(r["on"].sum()), "positions": int(torch.unique(pr.t[r["on"]]).numel()),
                      "edges": None if r["kept"] is None else len(r["kept"]), "c": r["c"], "n": r["n"], "size_term": r["size_term"],
                      "score": r["score"], "removed": r["removed"], "added": r.get("added")} for r in runs],
           "best": {"size": int(on.sum()), "positions": int(torch.unique(pr.t[on]).numel()), "edges": None if kept is None else len(kept),
                    "c": out["c"], "score": out["score"], **{k: final[k] for k in ("f1", "f1_nodes", "n", "f1_flips", "f1_rob", "kl_c", "kl_minus")}},
           "stages": len(stages), "flips_calls": meas.calls, "seconds": time.perf_counter() - t0}
    if out["rep"] is not None:
        row.update(repair=[{x: v for x, v in s_.items() if x not in ("on", "kept")} for s_ in out["rep"]], repair_pick=out["repair_pick"])
    ids_all = sorted({i for s_ in stages for i in s_["on"]} | set(on.nonzero().flatten().tolist()))
    path_file = {"method": "v3", "seed": out["seed"], "start_nodes": [[pr.nodes[i][0], pr.nodes[i][2], pr.nodes[i][1]] for i in ids_all],
                 "start_index": ids_all, "steps": stages, "pick_stage": out["stage"], "cleanup": out["cut"], "on": on.nonzero().flatten().tolist()}
    if out["edge_path"] is not None:
        path_file["edge_paths"] = [out["edge_path"]]
    if out["rep"] is not None:
        path_file["repair"] = [{x: (v.nonzero().flatten().tolist() if x == "on" else v) for x, v in s_.items()} for s_ in out["rep"]]
    return {"": teacher.circuit_json(pr, on, kept)}, row, path_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, required=True, help="task ids, one per line")
    ap.add_argument("--vpd", type=Path, required=True, help="VPD's answers saved by teacher.py --save-vpd")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--edges", action="store_true", help="the pick's connections (teacher.Connector, C's semantics)")
    ap.add_argument("--repair", action="store_true", help="with --edges, the flip repair (teacher.Connector.repair)")
    ap.add_argument("--edge-objective", choices=("graph", "robust"), default="graph",
                    help="with --edges, connection removals keep graph F1 (graph), and F1 under the flips too (robust: teacher.py's)")
    ap.add_argument("--max-nodes", type=int, default=teacher.MAX_NODES, help="the pick among stages of at most this many nodes")
    ap.add_argument("--max-tokens", type=int, default=components.ANSWER_BUDGET,
                    help="the answer at most this many tokens of --tokenizer: a smaller stage while it is longer (0: not checked)")
    ap.add_argument("--seeds", type=int, default=1, help="searches with seeds 0..S-1; the written one the reward's best mean(C, N) - 0.5 size")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    saved = torch.load(args.vpd)
    pool = torch.load(teacher.POOL)
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    tokens = None
    if args.max_tokens:
        from tokenizers import Tokenizer

        tk = Tokenizer.from_pretrained(args.tokenizer)
        tokens = lambda text: len(tk.encode(text, add_special_tokens=False).ids)  # noqa: E731
    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks.read_text().split()[args.offset::args.stride]:
        if (args.out / f"{task}.json").exists():
            continue
        ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
        circuits, row, path_file = search(ge, ids, targets, saved[task]["answer"], pool, args.batch, args.edges, args.repair,
                                          args.max_nodes, tokens, args.max_tokens, tuple(range(args.seeds)), args.edge_objective == "robust")
        for name, c in circuits.items():
            (args.out / f"{task}{'.' + name if name else ''}.json").write_text(json.dumps(c))
        (args.out / f"{task}.path.json").write_text(json.dumps(path_file))
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps({"task": task, **row}) + "\n")
        b = row["best"]
        print(f"{task}: {b['size']} subcomponents at {b['positions']} positions"
              + ("" if b["edges"] is None else f", {b['edges']} connections")
              + f", F1 {b['f1']:.3f} node F1 {b['f1_nodes']:.3f} N {b['n']:.3f} F1 under {teacher.REPAIR_M} flips {b['f1_flips']:.3f}, "
              f"C {b['c']:.3f}, score {b['score']:.3f} (seed {row['seed']} of {len(row['seeds'])}, {row['pick']}; cleanup removed "
              f"{row['cleanup_removed']}" + ("" if row["repair_added"] is None else f", repair added {row['repair_added']}")
              + f"); {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
