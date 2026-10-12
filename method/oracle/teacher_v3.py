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
(else the removal halved); with --edges, the connections by teacher.Connector (C's semantics, the scorecard's) and the
flip repair (teacher.Connector.repair: the omitted lever's flips joined, the connections pruned again from the kept
ones, while min(graph F1, node F1, F1 under the flips) improves: node F1, the node set with every connection among
its nodes, so that a flip joined with its connections cut does not count); with --max-tokens, the answer's tokens.

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
        self.search = "bf16" if str(self.dev).startswith("cuda") else "score"

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

    def flips(self, on: torch.Tensor, graph: native.Graph | None = None) -> float:
        """F1 under the flips of the node set as a complete graph (every connection), or of a stated graph."""
        nat = self.ge.nat
        if graph is None:
            masks = {n: torch.zeros(self.tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            for j in on.nonzero().flatten().tolist():
                n, t, c = self.pr.nodes[j]
                masks[n][t, c] = True
            graph = native.Graph(masks=masks)
        f = self.ge.adversary_flips(self.tx, [graph], self.pool, "omitted", ms=(self.m,))[0]
        self.calls += 1
        f1 = f["f1_flips_omitted"].get(self.m)
        return 1.0 if f1 is None else f1                       # nothing omitted: nothing to switch on

    def __call__(self, on: torch.Tensor) -> dict:
        got = teacher.measure(self.pr, on[None], self.empty, self.vpd)[0]
        fl = self.flips(on)
        return {**got, "f1_flips": fl, "f1_rob": min(got["f1"], fl)}


def removals(meas: Measure, on: torch.Tensor, row: dict, goal: float = teacher.RECOVERED) -> tuple[torch.Tensor, dict, list]:
    """Measured removals around a node set: rounds of the F1 search's rule on min(F1, N) (every single removal measured
    and ranked, the most of them in that order, at sqrt(2)-spaced prefix sizes, that keep it >= min(goal, the set's)),
    each round's removal kept only while F1 under the flips stays >= min(goal, the set's), halved until it does.
    (the node set, its row, each round's {"removed", "size", ...})."""
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
        k, new = max(ok, default=0), None
        while k:
            cand = on.clone()
            cand[ranked[:k]] = False
            r = meas(cand)
            if r["f1_flips"] >= floor_fl:
                new = (cand, r)
                break
            k //= 2
        if new is None:
            break
        on, row = new
        log.append({"removed": ranked[:k].tolist(), "size": int(on.sum()), **{k_: row[k_] for k_ in ("f1", "n", "f1_flips")}})
    return on, row, log


def search(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], vpd_answer: dict, pool: dict, B: int = 32,
           edges: bool = False, repair: bool = False, max_nodes: int | None = teacher.MAX_NODES, tokens=None,
           max_tokens: int = components.ANSWER_BUDGET, seed: int = 0) -> tuple[dict, dict, dict]:
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

    ms = MaskSearch(ge, tx, pr, live=teacher.CONVENTION == "live")
    meas = Measure(ge, tx, pr, empty, vpd, pool)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([ms.logit], lr=LR)
    lever = torch.zeros(SAMPLES, len(pr.nodes), device=nat.dev)
    stages = []
    for lam in LAMBDAS:
        t1 = time.perf_counter()
        for _ in range(STEPS):
            z = ms.sample(SAMPLES)
            zd = z.detach()
            for _ in range(INNER):                               # the lever: sign steps on the run's KL, top REPAIR_M
                b = lever.clone().requires_grad_(True)
                g, = torch.autograd.grad(ms.kl(zd + (1 - zd) * b, ms.search).clamp_min(TINY).log().sum(), [b])
                lever = top((lever + ADV_STEP * g.sign()).clamp(0, 1), teacher.REPAIR_M)
            opt.zero_grad()
            adv = ms.kl(z + (1 - z) * lever, ms.search)
            minus = ms.kl(1 - z, ms.search)                     # VPD's answer less the circuit
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
        print(f"  lambda {lam:.3g}: {st['size']} subcomponents at {k_tok} positions, F1 {st['f1']:.3f} N {st['n']:.3f}"
              + (f" F1 under {meas.m} flips {st['f1_flips']:.3f}" if "f1_flips" in st else " (over the budget: not attacked)")
              + f", {st['seconds']:.0f}s", flush=True)
        if int(on.sum()) <= teacher.MIN_SIZE:
            break
    inside = [i for i, s_ in enumerate(stages) if "f1_rob" in s_]         # within the answer budget, attacked
    if not inside and max_nodes:                                 # every stage over the budget: the gates' largest max_nodes
        on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
        on[ms.logit.detach().topk(max_nodes).indices] = True
        st = meas(on)
        st.update(lam=None, size=int(on.sum()), positions=int(torch.unique(pr.t[on]).numel()), on=on.nonzero().flatten().tolist(),
                  size_term=size_term(on), seconds=0.0, budget=True)
        stages.append(st)
        inside = [len(stages) - 1]
        print(f"  no stage within {max_nodes} subcomponents: the gates' largest {max_nodes}, F1 {st['f1']:.3f} N {st['n']:.3f} "
              f"F1 under {meas.m} flips {st['f1_flips']:.3f}", flush=True)
    j, rule = teacher.pick([stages[i] for i in inside], "f1_rob")
    best = inside[j]
    on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    on[stages[best]["on"]] = True
    t1 = time.perf_counter()                                     # the cleanup: measured removals under the flips' guard
    on, picked, cut = removals(meas, on, dict(stages[best]))
    print(f"  cleanup: {len(cut)} rounds removed {stages[best]['size'] - int(on.sum())} of {stages[best]['size']} "
          f"(F1 {picked['f1']:.3f}, N {picked['n']:.3f}, F1 under flips {picked['f1_flips']:.3f}), {time.perf_counter() - t1:.0f}s", flush=True)
    row = {"T": tx.T, "kl_empty": empty, "kl_vpd": vpd, "vpd_size": len(pr.nodes), "size_scale": scale, "pick": rule,
           "stage": {k: stages[best][k] for k in ("size", "positions", "f1", "n", "f1_flips", "size_term", "lam", "kl_c", "kl_minus")},
           "cleanup_removed": stages[best]["size"] - int(on.sum()), "stages": len(stages)}
    kept, rep, path_edges, f1_written = None, None, None, picked["f1"]   # F1 of the answer as written (graph F1 with edges)
    if edges:
        wiring = teacher.Connector(ge, tx, pr, empty, vpd, size_term, B)
        kept, summary, e_steps, ep = wiring.connect(on, picked["n"])
        f1_written = summary["f1_graph"]
        ids_of = wiring.index
        path_edges = {"edges": [[ids_of[w], "out" if r is None else ids_of[r]] for w, r in ep.edges], "steps": e_steps}
        if repair:
            rep, k_, rule_ = wiring.repair(on, kept, summary, pool, max_nodes)
            row.update(repair=[{x: v for x, v in s_.items() if x not in ("on", "kept")} for s_ in rep], repair_pick=k_, repair_rule=rule_,
                       repair_added=int(rep[k_]["on"].sum()) - int(on.sum()))
            on, kept, f1_written = rep[k_]["on"], rep[k_]["kept"], rep[k_]["f1_graph"]
    if tokens is not None:                                       # the answer within the budget
        row["answer_tokens"] = tokens(components.render(teacher.circuit_json(pr, on, kept)))
    final = teacher.measure(pr, on[None], empty, vpd)[0]          # the answer as written: the stated graph if any
    final["f1_flips"] = meas.flips(on, None if kept is None else wiring.graph(on, kept))
    final["f1_nodes"], final["f1"] = final["f1"], f1_written     # node F1 (every connection) beside the written F1
    final["f1_rob"] = min(f1_written, final["f1_nodes"], final["f1_flips"])
    row.update(best={"size": int(on.sum()), "positions": int(torch.unique(pr.t[on]).numel()), "edges": None if kept is None else len(kept),
                     **{k: final[k] for k in ("f1", "f1_nodes", "n", "f1_flips", "f1_rob", "kl_c", "kl_minus")}},
               flips_calls=meas.calls, seconds=time.perf_counter() - t0)
    ids_all = sorted({i for s_ in stages for i in s_["on"]} | set(on.nonzero().flatten().tolist()))
    path_file = {"method": "v3", "start_nodes": [[pr.nodes[i][0], pr.nodes[i][2], pr.nodes[i][1]] for i in ids_all],
                 "start_index": ids_all, "steps": stages, "pick_stage": best, "cleanup": cut, "on": on.nonzero().flatten().tolist()}
    if path_edges is not None:
        path_file["edge_paths"] = [path_edges]
    if rep is not None:
        path_file["repair"] = [{x: (v.nonzero().flatten().tolist() if x == "on" else v) for x, v in s_.items()} for s_ in rep]
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
    ap.add_argument("--max-nodes", type=int, default=teacher.MAX_NODES, help="the pick among stages of at most this many nodes")
    ap.add_argument("--max-tokens", type=int, default=0, help="report the answer's tokens in --tokenizer (0: not counted)")
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
                                          args.max_nodes, tokens, args.max_tokens)
        for name, c in circuits.items():
            (args.out / f"{task}{'.' + name if name else ''}.json").write_text(json.dumps(c))
        (args.out / f"{task}.path.json").write_text(json.dumps(path_file))
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps({"task": task, **row}) + "\n")
        b = row["best"]
        print(f"{task}: {b['size']} subcomponents at {b['positions']} positions"
              + ("" if b["edges"] is None else f", {b['edges']} connections")
              + f", F1 {b['f1']:.3f} N {b['n']:.3f} F1 under {teacher.REPAIR_M} flips {b['f1_flips']:.3f} ({row['pick']}; "
              f"cleanup removed {row['cleanup_removed']}" + (f", repair added {row['repair_added']}" if "repair_added" in row else "")
              + f"); {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
