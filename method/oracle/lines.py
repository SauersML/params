"""The stated-circuit line stage (#2951 graph oracle; the lead's 10-13): a teacher circuit's node set as a stated circuit
in which every part lies on a stated path from a token to the prediction and every stated line is needed. A node list
never says which part feeds which; this pass states it.

Lines: every connection the model has among the parts (native.all_edges: residual and MLP reads, value reads and the
writes into the prediction at the targets), every token edge (the embedding at a part's position into each residual
reader there) and the target token's embedding into the prediction (teacher.with_tokens), as engine.Engine.pp_lines
orders them. A candidate is a subset of the lines; its parts left without a stated path to the prediction are dropped
with their lines before it is scored (goal_eval.strand), so the circuit scored is the circuit written.

The descent is teacher.Descent.prune's (measured block removal, the largest admitted prefix of the ranked blocks, the
culprits refused alone frozen for the level), coarse to fine (LEVELS: every line into the parts at one position, out of
one position, into one part, out of one part or token, then single lines). Its KL is the point run, one C-semantics
plan of the fully wired circuit with the lines as strength rows and the states before the first part reused; its budget
the point family's floor. A removal is admitted when
  changed     s_track_pp on fresh search-pool draws (teacher.Acceptance, draws "fv": the model's likeliest alternative
              judged draw by draw against its own floor, sampled tokens pooled) stays at its floor;
  floors      every family the frozen reward scores on the text stays at its floor (goal_eval.tau's floors, each at most
              the fully wired circuit's own score): point, nodes, track_pp, interchange_pp, patching, pattern and N on
              the line variants from one plan (GoalEval.phi_variants over the current circuit's lines), then edges,
              flips and omitted on the survivors alone;
and kept after it when s_track_pp on the validation draws (the other half of the search pool: rank 2 and fresh
samples, eval-like) stays within VAL_MARGIN of the fully wired circuit's (teacher.Validated, every removal: a failing
one goes back; the level then runs again, the failing removal's line kept for good when it was one line, else every
later prefix smaller than half of it, until a run admits nothing).

Each accepted removal is checkpointed by line identity (a line's key names its ends, never its place in a list: a set's
order changes from process to process), and a process ends after MAX_STEPS of them (exit EXIT_RESUME) so a fresh one
resumes with the device's memory back (track_pp's caches grow over many calls). When the levels are done, a fresh
process re-checks the circuit (graph_kl on the text, s_track_pp on the eval draws, the reward row's worst shortfall v)
and writes OUT/<task>.json (goal_eval.to_json's stated form) and OUT/<task>.lines.json (the counts, the re-check, each
line's and each part's removal effect: the point KL rise in bits and the eval s_track_pp fall).

  lines.py --teachers DIR --out OUT [--tasks FILE | --split heldout --n 20] [--vpd FILE] [--offset K --stride S]
  lines.py one TEACHER.json TASK.json OUT [--vpd FILE]     one text, one process's share (exit 3: resume)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import teacher  # noqa: E402

LEVELS = ("reader_position", "writer_position", "reader", "writer", "line")
VAL_MARGIN = 0.02  # validation: s_track_pp at least the fully wired circuit's less this (the lead's 10-13: at
# min(0.9, start - 0.05) text9000's pruned circuit fell to eval 0.81 against the complete 0.965)
MAX_STEPS = 24  # accepted removals per process
EXIT_RESUME = 3
HELD = ("edges", "flips", "omitted")  # the families phi_variants holds at the current circuit's values: run on survivors


def key(col: tuple) -> str:
    """A line's identity as text: its kind and its ends (nodes as [matrix, position, index])."""
    return json.dumps([list(x) if isinstance(x, tuple) else x for x in col])


def wired(nodes: list, targets: list[int], C: dict) -> native.Graph:
    """The fully wired stated circuit of a node set: every connection the model has among the nodes, every token edge
    and the targets' tokens into the prediction, stranded nodes dropped (a part the model gives no route to the
    prediction among these nodes)."""
    par, out = native.all_edges(nodes, targets)
    return goal_eval.strand(teacher.with_tokens(native.Graph(list(nodes), par, out), targets), targets, C)


class LinePruner(teacher.Descent):
    """Measured removal over a stated circuit's lines (module docstring). on: bool [L] over the lines (pp_lines' columns
    less the nodes), on the CPU; kl_sets runs each subset's point KL with its stranded parts off."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, B: int = 32):
        nat = ge.nat
        self.ge, self.tx, self.B, self.dev = ge, tx, B, nat.dev
        self.g = wired(nodes, tx.targets, nat.C)
        self.cols = ge._pp_engine(tx).pp_lines(self.g)
        self.node_at = {c[1]: i for i, c in enumerate(self.cols) if c[0] == "n"}
        self.line_cols = [i for i, c in enumerate(self.cols) if c[0] != "n"]
        self.lines = [self.cols[i] for i in self.line_cols]
        self.at = {key(c): j for j, c in enumerate(self.lines)}
        self.on = torch.ones(len(self.lines), dtype=torch.bool)
        self.E = ge._circuit_engine(tx.ids, tx.targets, [self.g])
        self.plan = self.E.plan(self.g, semantics="c")
        ed = [(w, r) for r, ws in self.g.parents.items() for w in ws]
        od = list(self.g.out)
        if self.plan["E"] != len(ed) or self.plan["O"] != len(od):
            raise ValueError("the plan's connections are not the graph's")
        self.e_line = torch.tensor([self.at[key(("e", w, r))] for w, r in ed], dtype=torch.long)
        self.o_line = torch.tensor([self.at[key(("o", w))] for w in od], dtype=torch.long)
        self.e_ends, self.o_ends = ed, od
        lo = min(p for _, p, _ in self.g.node_set())
        self.lo = lo if 0 < lo <= min(tx.targets) else 0
        self.states = {}
        if self.lo:
            self.E.graph_log_probs(self.plan, ew=torch.ones(1, len(ed), device=self.dev), ow=torch.ones(1, len(od), device=self.dev),
                                   keep=self.states)
        self.measured = 0

    def graph(self, on: torch.Tensor) -> native.Graph:
        """A line subset as the stated graph scored: the fully wired circuit less the lines off, stranded parts dropped."""
        row = torch.ones(len(self.cols), dtype=torch.bool)
        row[torch.tensor(self.line_cols, dtype=torch.long)] = on.cpu()
        h = goal_eval.graph_of_lines(self.g, self.cols, row)
        del h.stranded, h.stranded_bits  # graph_of_lines marks the graph stranded already; strand must run
        return goal_eval.strand(h, self.tx.targets, self.ge.nat.C)

    def row(self, on: torch.Tensor) -> torch.Tensor:
        """A line subset as path_patched(lines=)'s row over every column, its stranded parts and their lines off."""
        kept = self.graph(on).node_set()
        out = torch.zeros(len(self.cols), dtype=torch.bool)
        for nd, i in self.node_at.items():
            out[i] = nd in kept
        for j, (i, c) in enumerate(zip(self.line_cols, self.lines)):
            ends = [x for x in c[1:] if isinstance(x, tuple)]  # (a token edge's position and a target are ints)
            out[i] = bool(on[j]) and all(x in kept for x in ends)
        return out

    @torch.no_grad()
    def kl_sets(self, ons: torch.Tensor) -> list[float]:
        """The point KL (bits at the targets) of each line subset ons [K, L], its stranded parts off."""
        out = []
        for s0 in range(0, len(ons), self.B):
            part = ons[s0:s0 + self.B].cpu()
            K = len(part)
            ew = torch.zeros(K, len(self.e_ends))
            ow = torch.zeros(K, len(self.o_ends))
            off = []
            for k in range(K):
                kept = self.graph(part[k]).node_set()
                off.append([nd for nd in self.node_at if nd not in kept] or None)
                ew[k] = part[k][self.e_line].float() * torch.tensor([float(w in kept and r in kept) for w, r in self.e_ends])
                ow[k] = part[k][self.o_line].float() * torch.tensor([float(w in kept) for w in self.o_ends])
            reuse = (self.E.tile_states(self.states, K), self.tx.ids, self.lo) if self.lo else None
            lq = self.E.graph_log_probs(self.plan, ew=ew.to(self.dev), ow=ow.to(self.dev), off=off, reuse=reuse)
            out += carry.kl(self.tx.ref, lq).tolist()
            native.free(self.dev)
        self.measured += len(ons)
        return out

    def blocks(self, level: str) -> list[torch.Tensor]:
        """The kept lines grouped (LEVELS): by the reader's position (the prediction its own group), the writer's (a
        token edge's: its token's), the reader, the writer, or single lines."""
        ids = self.on.nonzero().flatten()
        if level == "line":
            return list(ids.split(1))

        def group(c):
            kind = c[0]
            reader = c[2] if kind in ("e", "t") else "out"
            writer = c[1] if kind in ("e", "o") else ("tok", c[1])
            if level == "reader":
                return repr(reader)
            if level == "writer":
                return repr(writer)
            if level == "reader_position":
                return "out" if reader == "out" else str(reader[1])
            return str(writer[1])
        by = {}
        for i in ids.tolist():
            by.setdefault(group(self.lines[i]), []).append(i)
        return [torch.tensor(v, dtype=torch.long) for _, v in sorted(by.items())]


def held(h: native.Graph):
    """A stated graph's lines as goal_eval.loo_lines names them: a test whether it keeps one."""
    nodes = set(h.node_set())
    edges = set(goal_eval._pairs(h))
    toks = set(getattr(h, "tok_edges", [])) | {(t, None) for t in getattr(h, "tok_out", [])}
    return lambda kind, x: x in nodes if kind == "node" else (x in edges if kind == "edge" else tuple(x) in toks)


class Floors:
    """The frozen reward's floors on line variants (module docstring): per candidate graph its families as the reward
    reads them against floors, or None (its first family under its floor recorded in refused)."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, pool: dict, floors: dict):
        self.ge, self.tx, self.pool, self.floors = ge, tx, pool, floors
        self._base, self._held, self._screens, self.refused = {}, {}, {}, []

    def base(self, g: native.Graph) -> dict:
        """The current circuit's reward-mode row (search split), kept for its lines."""
        k = repr(self.ge.loo_lines(self.tx, g))
        if k not in self._base:
            self._base = {k: self.ge.fidelity(self.tx, [g], self.pool, mode="reward", split="search")[0]}
        return self._base[k]

    def __call__(self, g_now: native.Graph, graphs: list) -> list:
        ge, tx, nat = self.ge, self.tx, self.ge.nat
        lines = ge.loo_lines(tx, g_now)
        keep = torch.tensor([[has(kind, x) for kind, x in lines] for has in (held(h) for h in graphs)], dtype=torch.bool)
        got = ge.phi_variants(tx, g_now, keep, self.pool, base=self.base(g_now))["variants"]
        out = [{f: goal_eval.family_score(v["row"], f) for f in self.floors if f not in HELD} for v in got]

        def under(i):
            return next((f for f, s in out[i].items() if (s or 0.0) < self.floors[f] - 1e-9), None)
        e0 = ge._bounds(tx, "live")[0]
        for fam in HELD:
            alive = [i for i in range(len(out)) if under(i) is None]
            if fam not in self.floors or not alive:
                continue
            keys = [(fam, repr(ge.loo_lines(tx, graphs[i])) if fam != "omitted" else tuple(sorted(graphs[i].node_set()))) for i in alive]
            todo = [(i, k) for i, k in zip(alive, keys) if k not in self._held]
            cg = [goal_eval._c_graph(graphs[i], tx.targets, nat.C) for i, _ in todo]
            if not cg:
                rows = []
            elif fam == "edges":
                rows = [{"s_edges": goal_eval._fid(e0, r["kl_edges_c"], r["kl_edges_vpd"]) if r.get("kl_edges_c") is not None else None}
                        for r in ge._edge_family(tx, cg, goal_eval.DRAWS)]
            elif fam == "omitted":
                rows = ge._omitted_family(tx, cg)
            else:
                rows = [{"f1_flips_c": r.get("f1_flips_omitted", {})} for r in
                        ge.adversary_flips(tx, cg, self.pool, "omitted", upto=goal_eval.FLIP_TERM, screens=self._screens)]
            for (i, k), r in zip(todo, rows):
                self._held[k] = goal_eval.family_score(r, fam)
            for i, k in zip(alive, keys):
                out[i][fam] = self._held[k]
        for i in range(len(out)):
            f = under(i)
            if f is not None:
                self.refused.append({"family": f, "score": out[i].get(f), "floor": self.floors[f]})
                out[i] = None
        native.free(nat.dev)
        return out


class Restart(Exception):
    """MAX_STEPS accepted removals in this process: the checkpoint holds them; a fresh process goes on."""


def text_of(ge: goal_eval.GoalEval, task_path: Path, vpd: dict | None) -> goal_eval.Text:
    """The task's Text with its seed (the scorer's draws), VPD's saved answer and order when given (a pod has no
    causal-importance network)."""
    nat = ge.nat
    task = json.loads(task_path.read_text())
    ids, targets = native.text(task_path)
    tx = ge.text(ids, targets, native.task_seed(task_path.stem), task=task)
    saved = (vpd or {}).get(task_path.stem)
    if saved is not None:
        if "order" in saved:  # (goal_eval's omitted family reads VPD's order)
            tx.vpd_order = [(n, int(t), int(c)) for n, t, c in saved["order"]]
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in saved["answer"].items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    if tx.vpd is None:
        raise ValueError("no VPD answer: run with the causal-importance network or pass --vpd")
    return tx


def tidy(ge: goal_eval.GoalEval, tx: goal_eval.Text, pr: LinePruner) -> None:
    """Per-call caches the engines key by object ids, dropped; the device's memory freed."""
    for eng in (pr.E, tx.__dict__.get("_pp_engine")):
        if eng is not None:
            for k in ("_xcols", "_pre"):
                eng.__dict__.pop(k, None)
    gc.collect()
    native.free(ge.nat.dev)
    if str(ge.nat.dev).startswith("mps"):
        torch.mps.empty_cache()


def one(teacher_path: Path, task_path: Path, out: Path, vpd: dict | None, pool: dict, B: int = 32, log=print) -> int:
    """One process's share of one text (module docstring): 0 when the text is written, EXIT_RESUME to go on."""
    ge = goal_eval.GoalEval(native.Native(), chunk=B)
    goal_eval.FLIP_ROWS, goal_eval.CHANGED_ROWS = teacher.PACK_FLIP_ROWS, teacher.PACK_CHANGED_ROWS  # (batch sizes only)
    tx = text_of(ge, task_path, vpd)
    task = json.loads(task_path.read_text())
    g0, _, _ = goal_eval.load_answer(teacher_path, task, tx.targets, ge.nat)
    pr = LinePruner(ge, tx, sorted(g0.node_set()), B)
    ckpt = out / f"{task_path.stem}.ckpt.json"
    state = json.loads(ckpt.read_text()) if ckpt.exists() else {"level": 0, "removed": [], "trace": [], "done": False}
    for k in state["removed"]:
        pr.on[pr.at[k]] = False
    full = torch.ones(len(pr.lines), dtype=torch.bool)
    if state["done"]:
        return finish(ge, tx, pr, g0, out, task_path.stem, state, pool, log)
    tau = ge.tau(tx, pool, "search")
    row_full = teacher.reward_row(ge, tx, pr.graph(full), pool)
    floors = {f: min(v, goal_eval.family_score(row_full, f) or 0.0) for f, v in (tau.get("floors") or {}).items()}
    e0, v0 = ge._bounds(tx, "live")
    budget = teacher.kl_budget(e0, v0, floors["point"]) if floors.get("point") else math.inf
    acc = teacher.Acceptance(ge, tx, ("changed",), "pp", "fv")
    changed, _ = acc.hook(pr.graph, full, current=lambda: pr.on, label="lines",
                          names=lambda ids: [key(pr.lines[i]) for i in ids])
    fl = Floors(ge, tx, pool, floors)

    def by_floors(ons):
        return fl(pr.graph(pr.on), [pr.graph(on) for on in ons])
    accept = teacher.both(changed, by_floors)
    start_val = acc.validation(pr.graph(full))[0]
    floor_val = None if start_val is None else start_val - VAL_MARGIN
    log(f"{task_path.stem}: {len(pr.node_at)} parts, {len(pr.lines)} lines, {int(pr.on.sum())} on; point budget {budget:.4f} bits; "
        f"validation {start_val} (floor {floor_val}); floors {floors}")
    steps = [0]
    t0 = time.time()
    pr.fixed = torch.zeros(len(pr.lines), dtype=torch.bool)  # lines whose removal alone failed validation: kept
    for k in state.get("frozen", []):
        pr.fixed[pr.at[k]] = True
    for li in range(state["level"], len(LEVELS)):
        level = LEVELS[li]
        cap = state.get("cap") if state.get("cap_level") == li else None
        while True:  # the level again after a validation failure (Validated ends a level at its first): the failing
            journal, failed = [], []  # removal's line frozen when it was one, else prefixes capped below its size
            v = teacher.Validated(pr, pr.graph, acc, floor_val, journal)

            def capped(ons, cap=cap):
                if cap is None:
                    return accept(ons)
                big = (pr.on[None] & ~ons).sum(1) > cap
                out = [None] * len(ons)
                small = (~big).nonzero().flatten()
                if len(small):
                    for i, r in zip(small.tolist(), accept(ons[small])):
                        out[i] = r
                return out

            def after(li=li, level=level, v=v, failed=failed):
                now = pr.on.clone()
                stop = v()
                if stop:
                    gone = pr.on & ~now
                    failed.append(int(gone.sum()))
                    if int(gone.sum()) <= 1:
                        pr.fixed |= gone
                state.update(removed=[key(pr.lines[j]) for j in (~pr.on).nonzero().flatten().tolist()], level=li,
                             frozen=[key(pr.lines[j]) for j in pr.fixed.nonzero().flatten().tolist()])
                state["trace"].append([level, int(pr.on.sum()), v.trace[-1][2] if v.trace else None, round(time.time() - t0)])
                ckpt.write_text(json.dumps(state))
                log(f"  {level}: {int(pr.on.sum())} lines on, validation {v.trace[-1][2] if v.trace else None}"
                    f"{' (failed: back)' if stop else ''}, {time.time() - t0:.0f}s")
                tidy(ge, tx, pr)
                steps[0] += 1
                if not stop and steps[0] >= MAX_STEPS:
                    raise Restart
                return stop
            try:
                pr.prune(level, budget, log=journal, accept=capped, after=after)
                v.finish()
            except Restart:
                return EXIT_RESUME
            if not failed:
                break
            if failed[-1] > 1:
                cap = failed[-1] // 2 if cap is None else min(cap, failed[-1] // 2)
                state.update(cap=cap, cap_level=li)
                ckpt.write_text(json.dumps(state))
        state.update(level=li + 1, cap=None, cap_level=None,
                     removed=[key(pr.lines[j]) for j in (~pr.on).nonzero().flatten().tolist()])
        ckpt.write_text(json.dumps(state))
    state["done"] = True
    state["refused"] = {"changed": len(acc.refused), "floors": len(fl.refused)}
    ckpt.write_text(json.dumps(state))
    return EXIT_RESUME  # the re-check and the writing in a fresh process


def finish(ge: goal_eval.GoalEval, tx: goal_eval.Text, pr: LinePruner, g0: native.Graph, out: Path, stem: str, state: dict,
           pool: dict, log=print) -> int:
    """The pruned circuit re-checked in this fresh process and written (module docstring)."""
    full = torch.ones(len(pr.lines), dtype=torch.bool)
    g = pr.graph(pr.on)
    point = ge.graph_kl(tx, [g], True)[0]
    point_full = ge.graph_kl(tx, [pr.graph(full)], True)[0]
    tr = ge.track_pp(tx, [g], "eval")[0]["s_track_pp"]
    tr_full = ge.track_pp(tx, [pr.graph(full)], "eval")[0]["s_track_pp"]
    tr_nodes = ge.track_pp(tx, [g0], "eval")[0]["s_track_pp"]
    tau = ge.tau(tx, pool, "search")
    row = ge.fidelity(tx, [g], pool, mode="reward", split="search")[0]
    v = goal_eval.shortfall(row, tau.get("floors") or {})
    on = pr.on.nonzero().flatten().tolist()
    drop = pr.on[None].repeat(len(on), 1)
    drop[torch.arange(len(on)), torch.tensor(on, dtype=torch.long)] = False
    pt = pr.kl_sets(drop) if on else []
    trk = [r["s_track_pp"] for r in ge.track_pp(tx, [pr.g], "eval", lines=torch.stack([pr.row(d) for d in drop]).float())] if on else []
    kept = sorted(g.node_set())
    parts = []
    for nd in kept:  # a part off: its lines with it (and whatever it strands)
        d = pr.on.clone()
        for j, c in enumerate(pr.lines):
            if nd in [x for x in c[1:] if isinstance(x, tuple)]:
                d[j] = False
        parts.append(d)
    ppt = pr.kl_sets(torch.stack(parts)) if parts else []
    ptr = [r["s_track_pp"] for r in ge.track_pp(tx, [pr.g], "eval", lines=torch.stack([pr.row(d) for d in parts]).float())] if parts else []
    lines = [{"line": json.loads(key(pr.lines[j])), "point_rise": a - point, "track_fall": None if b is None or tr is None else tr - b}
             for j, a, b in zip(on, pt, trk)]
    nodes = [{"part": list(nd), "point_rise": a - point, "track_fall": None if b is None or tr is None else tr - b}
             for nd, a, b in zip(kept, ppt, ptr)]
    rec = {"task": stem, "parts_start": len(pr.node_at), "lines_start": len(pr.lines), "parts": len(kept), "lines": len(on),
           "point": point, "point_wired": point_full, "track_eval": tr, "track_eval_wired": tr_full, "track_eval_nodes": tr_nodes,
           "v": v, "floors": tau.get("floors"), "trace": state["trace"], "refused": state.get("refused"),
           "line_effects": lines, "part_effects": nodes}
    (out / f"{stem}.json").write_text(json.dumps(goal_eval.to_json(g)))
    (out / f"{stem}.lines.json").write_text(json.dumps(rec))
    log(f"{stem}: {len(kept)} parts, {len(on)} lines of {len(pr.lines)}; point {point:.4f} bits (wired {point_full:.4f}); eval "
        f"s_track_pp {tr} (wired {tr_full}, node set {tr_nodes}); v {v:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("one",))
    ap.add_argument("paths", nargs="*", help="one: TEACHER.json TASK.json OUT")
    ap.add_argument("--teachers", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tasks", type=Path)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vpd", type=Path, help="VPD's answers and orders (teacher.py --save-vpd)")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "one":
        t, k, o = map(Path, a.paths)
        o.mkdir(parents=True, exist_ok=True)
        vpd = torch.load(a.vpd) if a.vpd else None
        sys.exit(one(t, k, o, vpd, torch.load(teacher.POOL), a.batch, log=lambda m: print(m, flush=True)))
    a.out.mkdir(parents=True, exist_ok=True)
    texts = [native.TEXTS / "vpd4l" / f"{t}.json" for t in a.tasks.read_text().split()] if a.tasks else native.tasks(a.split)[:a.n]
    for path in texts[a.offset::a.stride]:
        src = a.teachers / f"{path.stem}.json"
        if not src.exists() or (a.out / f"{path.stem}.lines.json").exists():
            continue
        while True:  # a fresh process per MAX_STEPS removals, and for the re-check
            rc = subprocess.run([sys.executable, "-B", __file__, "one", str(src), str(path), str(a.out), "--batch", str(a.batch)]
                                + (["--vpd", str(a.vpd)] if a.vpd else [])).returncode
            if rc != EXIT_RESUME:
                break
        if rc:
            print(f"{path.stem}: exit {rc}", flush=True)


if __name__ == "__main__":
    main()
