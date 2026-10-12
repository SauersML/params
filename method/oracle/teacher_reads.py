"""Reads-first teacher circuits (#2951 graph oracle): a start set for teacher.py's pipeline that follows the prediction
back through the attention reads it uses, position by position, before anything else is pruned (the lead's design 10-12).

The search holds a node set within VPD's whole answer on the text and a set of cut value reads. A read (p, l, s) is
every connection from a value node at source position s in layer l into an attention-output node at reader position p
in layer l; cut, those readers' input loses the source's value nodes' share (its pattern weight times their output),
in C's semantics (engine.Engine.plan: the set as a complete graph, the cuts at strength 0, the remainders live and
reading the full stream). Every KL is that run's at the targets, the budget teacher.py's F1 budget (KL_C <=
KL_empty^0.1 KL_VPD^0.9), and every removal, of subcomponents or of reads, is admitted only while the changed-input
term on the search split (goal_eval.changed_search: the circuit's move from x to x' against the model's) stays >=
min(0.9, VPD's whole answer's): context the prediction moves with stays. --mode pp (the default): s_track_pp, the
circuit path-patched (everything outside it at its value on x, a cut read's source value at x); --mode c: s_track in
C's semantics (a cut read's connections at 0). The finish's connections take teacher.teach's acceptance in c mode only
(none in pp mode until teacher.Acceptance takes a mode).
  1. Targets: from VPD's whole answer, the targets' subcomponents pruned by teacher.Descent's rounds (levels site,
     chunk16, chunk4, subcomponent), every other position whole.
  2. Positions from the last target down. At a position p, its reads: every value read into its kept attention-output
     nodes from an earlier position, cut by Descent's rounds over reads: per layer every source at once (layer),
     sources whose value nodes in that layer are the same subcomponents (repeat: a repeated mention, a group), spans of
     64, 16 and 4 sources, single sources. A source with a kept read joins the circuit's positions.
  3. At a source s (after every later position: every read from it decided): the subcomponents feeding no kept read
     from it (at or above its highest kept read's layer, but for that layer's keys and values) dropped as one block
     when admitted, then the rest at s pruned (site, chunk16, chunk4, subcomponent); then step 2 at s (its own reads:
     two-hop chains).
  4. Positions no kept read reaches: their subcomponents dropped as one block, else position by position; a position
     that cannot go (its keys shape a pattern, the remainders read it) joins with no kept read, and steps 3 and 2 run
     there, until none is left.
  5. The cuts are lifted (connections are the connection descent's): the start set is the kept subcomponents. Then
     teacher.py's pipeline from it, teacher.teach with this path given (from_path): teacher.joint_path from the start
     set and VPD's top-k, its steps admitted by teacher.Acceptance's terms (--accept, floors min(0.9, that start's));
     the pick; the connections (--edges, --no-strand), admitted by the same terms; the flip repair (--repair).
Output per text in --out: teacher.py's files (from teach), with the search's record in <task>.reads.json (per stage its
level rounds, sizes, KL, s_track, seconds; the kept reads) and in the summary line ("reads").

  MPD_MEM_GIB=16 ~/mpd-data/venv/bin/python -B teacher_reads.py --tasks FILE --out DIR --edges --no-strand
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import carry
import components
import goal_eval
import native
import teacher

LEVELS = ("site", "chunk16", "chunk4", "subcomponent")       # a position's subcomponents, coarse to fine
READ_LEVELS = ("layer", "repeat", "span64", "span16", "span4", "source")
DRAWS = 16                                                   # changed_search's draws
TRACK = {"c": "s_track", "pp": "s_track_pp"}                 # changed_search's mode -> its term


def _layer_kind(nd) -> tuple[int, str]:
    return native._name_layer_kind(nd[0])


class State:
    """One text's search: VPD's whole answer as nodes (teacher.Pruner's order), the engine over its columns (C's
    semantics, remainders live), the kept set `on` [N] and the cut reads {(p, l, s)}."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, B: int, mode: str = "pp"):
        import engine

        nat = ge.nat
        self.ge, self.tx, self.B, self.dev, self.mode = ge, tx, B, nat.dev, mode
        self.pr = teacher.Pruner(ge, tx, tx.vpd, live=True, B=B)
        pr, T = self.pr, tx.T
        self.names, self.nodes, self.t, self.mat, self.c = pr.names, pr.nodes, pr.t, pr.mat, pr.c
        cols = {n: torch.unique(pr.c[pr.mat == j]) for j, n in enumerate(pr.names)}
        self.eng = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols)
        self.flat = torch.empty_like(pr.c)                   # each node's place in u()'s buffer
        self.offsets, at = [], 0
        for j, n in enumerate(pr.names):
            sel = (pr.mat == j).nonzero().flatten()
            K = len(cols[n])
            self.flat[sel] = at + pr.t[sel] * K + torch.searchsorted(cols[n], pr.c[sel])
            self.offsets.append((at, K))
            at += T * K
        self.width = at
        self.by = {}                                         # (layer, kind, position) -> node ids
        for i, nd in enumerate(pr.nodes):
            self.by.setdefault((*_layer_kind(nd), nd[1]), []).append(i)
        self.on = torch.ones(len(pr.nodes), dtype=torch.bool, device=self.dev)
        self.cut: set = set()
        self.graph = native.Graph(complete=True)             # the plans' complete graph: the set itself runs as u
        self._plan = None
        empty, vpd = ge._bounds(tx, "live")
        self.empty, self.vpd = empty, vpd
        self.budget = empty ** (1 - teacher.RECOVERED) * vpd ** teacher.RECOVERED
        self.measured = 0
        self.floor = None
        self.checks, self.check_seconds = 0, 0.0

    def u(self, ons: torch.Tensor) -> dict:
        """Node subsets ons [K, N] as the engine's u: per site [K, T, columns], views of one buffer."""
        T = self.tx.T
        buf = torch.zeros(len(ons), self.width, device=self.dev)
        buf[:, self.flat] = ons.float()
        return {n: buf[:, at:at + T * K].view(len(ons), T, K) for n, (at, K) in zip(self.names, self.offsets)}

    def reads_dict(self, reads, on: list | None = None) -> dict:
        """Reads as connections {reader node: [writer nodes]}: every attention-output node of VPD's answer at (p, l)
        from every value node at (l, s), or those of them in a node set (on, a list of bools: a plan over its
        columns alone names none outside it)."""
        out = {}
        for p, l, s in reads:
            ws = [self.nodes[i] for i in self.by.get((l, "v_proj", s), []) if on is None or on[i]]
            if ws:
                for r in self.by.get((l, "o_proj", p), []):
                    if on is None or on[r]:
                        out.setdefault(self.nodes[r], []).extend(ws)
        return out

    def plan(self) -> dict:
        """The plan of the current cuts (kept until they change)."""
        key = frozenset(self.cut)
        if self._plan is None or self._plan[0] != key:
            self._plan = (key, self.eng.plan(self.graph, semantics="c", cuts=self.reads_dict(self.cut)))
        return self._plan[1]

    @torch.no_grad()
    def runs(self, plan: dict, K: int, u_rows, ew_rows, start: int, base: tuple) -> list[float]:
        """KL at the targets of K rows of a plan in batches of B (u_rows(a, b), ew_rows(a, b): rows a to b's u and ew).
        Rows that equal base = (u, ew) before position start (0 < start <= the first target) run from start on,
        reusing base's run before it (engine.Engine.graph_log_probs reuse: the model is causal)."""
        states = None
        if K > 1 and 0 < start <= min(self.tx.targets):
            states = {}
            self.eng.graph_log_probs(plan, base[0], ew=base[1], keep=states)
        out = []
        for s0 in range(0, K, self.B):
            s1 = min(K, s0 + self.B)
            if states is None:
                out += self.eng.graph_kl(plan, u_rows(s0, s1), ew=ew_rows(s0, s1)).tolist()
            else:
                lq = self.eng.graph_log_probs(plan, u_rows(s0, s1), ew=ew_rows(s0, s1),
                                              reuse=(self.eng.tile_states(states, s1 - s0), self.tx.ids, start))
                out += carry.kl(self.eng.ref, lq).tolist()
        self.measured += K
        return out

    def kl_nodes(self, ons: torch.Tensor) -> list[float]:
        """KL at the targets of node subsets ons [K, N] with the current cuts, each run from the first position where
        a row leaves the current set."""
        off = (ons != self.on[None]).any(0)
        start = int(self.t[off].min()) if bool(off.any()) else 0
        return self.runs(self.plan(), len(ons), lambda a, b: self.u(ons[a:b]), lambda a, b: None, start, (self.u(self.on[None]), None))

    def track(self, ons: torch.Tensor, cuts: list | None = None) -> list:
        """s_track on the search split (goal_eval.changed_search in the state's mode: "pp", s_track_pp, the cut reads
        at their x values; "c", s_track in C's semantics, the cut reads' connections at 0) of node subsets ons [K, N],
        with per row its cut reads (cuts: a list of sets; None: the current cuts)."""
        t0 = time.perf_counter()
        graphs = [native.Graph([self.nodes[i] for i in row.nonzero().flatten().tolist()], complete=True) for row in ons]
        sets = [self.cut if c is None else c for c in (cuts or [None] * len(ons))]
        if self.mode == "pp":
            cd = []
            for c in sets:
                d = {}
                for p, l, s in c:
                    d.setdefault((l, p), []).append(s)
                cd.append(d)
        else:
            cd = [self.reads_dict(c, on) for c, on in zip(sets, ons.tolist())]
        got = [r[TRACK[self.mode]] for r in self.ge.changed_search(self.tx, graphs, DRAWS, cuts=cd, mode=self.mode)]
        self.checks += len(ons)
        self.check_seconds += time.perf_counter() - t0
        return got

    def admits(self, x) -> bool:
        return x is None or self.floor is None or x >= self.floor

    def sources(self, p: int) -> list[tuple[int, int]]:
        """(layer, source) pairs of the reads into p's kept attention-output nodes from earlier positions with a kept
        value node."""
        on, out = self.on.tolist(), []
        for l in range(self.ge.nat.target.n_layer):
            if any(on[i] for i in self.by.get((l, "o_proj", p), [])):
                out += [(l, s) for s in range(p) if any(on[i] for i in self.by.get((l, "v_proj", s), []))]
        return out


class Nodes(teacher.Descent):
    """Measured removal over the state's subcomponents at some positions (allowed [N]), with its cuts in place;
    `on` is the state's (removals change it)."""

    def __init__(self, st: State, allowed: torch.Tensor):
        self.st, self.allowed, self.B = st, allowed, st.B
        self.on, self.t, self.mat, self.c, self.names = st.on, st.t, st.mat, st.c, st.names

    def kl_sets(self, ons: torch.Tensor) -> list[float]:
        return self.st.kl_nodes(ons)

    def blocks(self, level: str) -> list[torch.Tensor]:
        out = []
        for b in teacher.Pruner.blocks(self, level):
            b = b[self.allowed[b]]
            if len(b):
                out.append(b)
        return out

    def accept(self, ons: torch.Tensor) -> list:
        return [{"s_track": x} if self.st.admits(x) else None for x in self.st.track(ons)]


class Reads(teacher.Descent):
    """Measured cutting of the value reads into one position p: units (p, l, s), on [U] the uncut ones; one plan with
    every unit's connections free (engine.Engine.plan's free), each row's strengths its units' 0/1, the state's
    earlier cuts in place."""

    def __init__(self, st: State, p: int):
        self.st, self.p, self.B = st, p, st.B
        self.units = [(p, l, s) for l, s in st.sources(p)]
        self.on = torch.ones(len(self.units), dtype=torch.bool, device=st.dev)
        free = st.reads_dict(self.units)
        index = {(l, s): k for k, (_, l, s) in enumerate(self.units)}
        self.unit_of = torch.tensor([index[(_layer_kind(w)[0], w[1])] for ws in free.values() for w in ws],  # the plan's
                                    dtype=torch.long, device=st.dev)                                         # free order
        self.plan = st.eng.plan(st.graph, semantics="c", cuts=st.reads_dict(st.cut), free=free)
        self.u0 = st.u(st.on[None])
        self.groups = self._repeats()

    def _repeats(self) -> dict:
        """Per unit its repeat group: units of one layer whose sources' kept value nodes are the same subcomponents."""
        st, key, on = self.st, {}, self.st.on.tolist()
        for k, (_, l, s) in enumerate(self.units):
            sub = frozenset(st.nodes[i][2] for i in st.by.get((l, "v_proj", s), []) if on[i])
            key.setdefault((l, sub), []).append(k)
        return {k: g for g in key.values() if len(g) > 1 for k in g}

    def kl_sets(self, ons: torch.Tensor) -> list[float]:
        """Each row's reads (ons [K, U]) at 0/1 strengths, run from p on (the reads cut only p's input)."""
        return self.st.runs(self.plan, len(ons), lambda a, b: self.u0, lambda a, b: ons[a:b][:, self.unit_of].float(), self.p,
                            (self.u0, torch.ones(1, len(self.unit_of), device=self.st.dev)))

    def blocks(self, level: str) -> list[torch.Tensor]:
        ids = self.on.nonzero().flatten().tolist()
        if level == "source":
            return [torch.tensor([k], device=self.on.device) for k in ids]
        keys = {}
        for k in ids:
            _, l, s = self.units[k]
            if level == "layer":
                key = l
            elif level == "repeat":
                if k not in self.groups:
                    continue
                key = (l, min(self.groups[k]))
            else:
                key = (l, s // int(level[4:]))
            keys.setdefault(key, []).append(k)
        return [torch.tensor(v, device=self.on.device) for v in keys.values()]

    def cut_sets(self, ons: torch.Tensor) -> list[set]:
        return [self.st.cut | {u for u, k in zip(self.units, row.tolist()) if not k} for row in ons]

    def accept(self, ons: torch.Tensor) -> list:
        return [{"s_track": x} if self.st.admits(x) else None
                for x in self.st.track(self.st.on[None].expand(len(ons), -1), self.cut_sets(ons))]


def _stage(st: State, label: str, t0: float, record: list, **extra) -> None:
    kl = st.kl_nodes(st.on[None])[0]
    row = {"stage": label, "size": int(st.on.sum()), "positions": int(torch.unique(st.t[st.on]).numel()), "cuts": len(st.cut),
           "kl": kl, "seconds": time.perf_counter() - t0, "sets": st.measured, "checks": st.checks,
           "check_seconds": st.check_seconds, **extra}
    record.append(row)
    print(f"  {label}: {row['size']} subcomponents at {row['positions']} positions, {len(st.cut)} reads cut, KL {kl:.3g} "
          f"(budget {st.budget:.3g}), {row['seconds']:.0f}s, {st.measured} sets, {st.checks} checks ({st.check_seconds:.0f}s)", flush=True)


def _prune_nodes(st: State, allowed: torch.Tensor, levels, label: str, record: list, drop: torch.Tensor | None = None) -> None:
    """Step 1 or 3 at the allowed nodes: drop's nodes as one block when admitted, then the levels' rounds."""
    t0 = time.perf_counter()
    nd = Nodes(st, allowed)
    if drop is not None and bool((drop & st.on).any()):
        trial = st.on.clone()
        trial[drop] = False
        if st.kl_nodes(trial[None])[0] <= st.budget and nd.accept(trial[None])[0] is not None:
            st.on[drop] = False
    for level in levels:
        nd.prune(level, st.budget, accept=nd.accept)
    _stage(st, label, t0, record)


def _prune_reads(st: State, p: int, record: list) -> list[tuple[int, int, int]]:
    """Step 2 at p: its reads cut by the rounds; returns the kept ones."""
    t0 = time.perf_counter()
    rd = Reads(st, p)
    if not rd.units:
        return []
    for level in READ_LEVELS:
        rd.prune(level, st.budget, accept=rd.accept)
    kept = [u for u, k in zip(rd.units, rd.on.tolist()) if k]
    st.cut |= {u for u, k in zip(rd.units, rd.on.tolist()) if not k}
    _stage(st, f"reads into {p} ({len(kept)} of {len(rd.units)} kept, from {len({s for _, _, s in kept})} positions)", t0, record,
           reads=len(rd.units), kept=[list(u) for u in kept])
    return kept


def _unused(st: State, s: int, layers: set) -> torch.Tensor:
    """bool [N]: the nodes at s that feed none of its kept reads (layers): at or above the highest such layer, but
    that layer's keys and values."""
    top = max(layers)
    ids = [i for i in (st.t == s).nonzero().flatten().tolist()
           if _layer_kind(st.nodes[i])[0] > top or (_layer_kind(st.nodes[i])[0] == top and _layer_kind(st.nodes[i])[1] not in ("v_proj", "k_proj"))]
    out = torch.zeros(len(st.nodes), dtype=torch.bool, device=st.dev)
    out[torch.tensor(ids, dtype=torch.long, device=st.dev)] = True
    return out


def search(ge: goal_eval.GoalEval, tx: goal_eval.Text, B: int = 32, mode: str = "pp") -> tuple[torch.Tensor, State, dict]:
    """Steps 1-4 on a text: (the start set [N] over teacher.Pruner's nodes of VPD's answer, the state, the record)."""
    t0 = time.perf_counter()
    st = State(ge, tx, B, mode)
    record = []
    whole = st.track(st.on[None])[0]
    st.floor = None if whole is None else min(teacher.RECOVERED, whole)
    targets = sorted(set(tx.targets))
    at_targets = torch.isin(st.t, torch.tensor(targets, device=st.dev))
    _prune_nodes(st, at_targets, LEVELS, "targets", record)
    kept_from: dict = {}                                     # source -> layers of its kept reads
    pending, done = set(targets), set()
    while True:
        while pending:
            p = max(pending)
            pending.discard(p)
            if p not in targets:
                layers = kept_from.get(p, set())
                here = st.t == p
                _prune_nodes(st, here, LEVELS, f"source {p}", record, _unused(st, p, layers) if layers else None)
            for _, l, s in _prune_reads(st, p, record):
                kept_from.setdefault(s, set()).add(l)
                if s not in done:
                    pending.add(s)
            done.add(p)
        rest = st.on & ~torch.isin(st.t, torch.tensor(sorted(done), device=st.dev))
        if not bool(rest.any()):
            break
        t1 = time.perf_counter()
        trial = st.on & ~rest
        nd = Nodes(st, rest)
        if st.kl_nodes(trial[None])[0] <= st.budget and nd.accept(trial[None])[0] is not None:
            st.on[rest] = False
        else:
            nd.prune("position", st.budget, accept=nd.accept)
        left = sorted(set(st.t[st.on & rest].tolist()))
        _stage(st, "unread positions", t1, record, kept_positions=left)
        if not left:
            break
        pending |= set(left)
    st.cut.clear()                                           # step 5: the cuts lifted
    t1 = time.perf_counter()
    s_track = st.track(st.on[None])[0]
    _stage(st, "start set", t1, record, s_track=s_track)
    return st.on.clone(), st, {"stages": record, "mode": mode, "s_track_whole": whole, "floor": st.floor, "s_track": s_track,
                               "checks": st.checks, "check_seconds": st.check_seconds, "sets": st.measured,
                               "seconds": time.perf_counter() - t0}


def teach(ge: goal_eval.GoalEval, ids: list[int], targets: list[int], B: int = 32, vpd_answer: dict | None = None,
          vpd_order: list | None = None, accept: tuple = ("changed",), mode: str = "pp", **kw) -> tuple[dict, dict, dict]:
    """Steps 1-5 on a text: teacher.teach's (circuits, summary row, path file) with the reads-first start set and the
    joint path (teacher.joint_path, its steps admitted by teacher.Acceptance's accept terms) given as its path, the
    connections admitted by the same terms; the row's "reads" the search's record. kw: teacher.teach's options."""
    nat = ge.nat
    tx = ge.text(ids, targets)
    if vpd_answer is not None:
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in vpd_answer.items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    if tx.vpd is None:
        raise ValueError("no VPD answer: run with the causal-importance network or pass --vpd")
    found, st, record = search(ge, tx, B, mode)
    pr = st.pr
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    order = vpd_order if vpd_order is not None else nat._vpd_order(ids, targets)
    order = torch.tensor([index[(n, int(t), int(c))] for n, t, c in order if (n, int(t), int(c)) in index], device=nat.dev)
    t1 = time.perf_counter()
    k = teacher.top_k(pr, order, st.empty, st.vpd)
    start = found.clone()
    start[order[:k]] = True
    hook, base = None, None
    if accept:
        acceptance = teacher.Acceptance(ge, tx, accept)
        if mode == "pp":                                     # its "changed" term path-patched (until teacher.Acceptance takes a mode)
            acceptance.scores = lambda graphs, edges=None: [{"s_track": r["s_track_pp"]} for r in ge.changed_search(tx, graphs, DRAWS, mode="pp")]
        hook, base = acceptance.hook(lambda on: native.Graph([pr.nodes[j] for j in on.nonzero().flatten().tolist()], complete=True), start)
    path = teacher.joint_path(pr, start, st.empty, st.vpd, hook, base)
    on = start.clone()
    for r in path:                                           # VPD's own order at every path size (teacher.teach's vpd_top)
        on_v = torch.zeros_like(start)
        on_v[order[:r["size"]]] = True
        r["vpd_top"] = teacher.measure(pr, on_v[None], st.empty, st.vpd)[0]
    record["path_seconds"] = time.perf_counter() - t1
    path_file = {"start_nodes": [[n, c, t] for n, t, c in (pr.nodes[i] for i in start.nonzero().flatten().tolist())],
                 "start_index": start.nonzero().flatten().tolist(), "steps": path,
                 "f1_index": found.nonzero().flatten().tolist(), "top_k": k}
    # the connections' acceptance is teacher.teach's, in C's semantics only (until it takes a mode): none in pp mode
    circuits, row, path_out = teacher.teach(ge, ids, targets, B, vpd_answer, vpd_order, from_path=path_file,
                                            accept=accept if mode == "c" else (), **kw)
    row["reads"] = {key: v for key, v in record.items() if key != "stages"}
    row["reads"]["sizes"] = {"found": int(found.sum()), "start": int(start.sum()), "top_k": k}
    circuits["reads"] = record
    return circuits, row, path_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, required=True, help="task ids, one per line")
    ap.add_argument("--vpd", type=Path, help="VPD's answers and orders saved by teacher.py --save-vpd (no causal-importance network here)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--edges", action="store_true", help="the pick's connections (teacher.py's)")
    ap.add_argument("--no-strand", action="store_true", help="with --edges, every non-query/key node keeps a stated path (teacher.py's)")
    ap.add_argument("--repair", action="store_true", help="with --edges, the flip repair (teacher.py's)")
    ap.add_argument("--edge-levels", type=teacher.edge_levels, default="connection", help="teacher.py's")
    ap.add_argument("--max-nodes", type=int, default=teacher.MAX_NODES, help="the pick among path points of at most this many nodes")
    ap.add_argument("--max-tokens", type=int, default=components.ANSWER_BUDGET, help="the answer at most this many tokens (0: not checked)")
    ap.add_argument("--accept", default="changed", help="teacher.py's --accept for the joint path and connections (the search's own: changed)")
    ap.add_argument("--mode", choices=tuple(TRACK), default="pp", help="the changed-input term: pp (s_track_pp, path-patched) or c (s_track)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(), chunk=args.batch)
    saved = torch.load(args.vpd) if args.vpd else {}
    pool = torch.load(teacher.POOL) if args.repair else None
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
        sv = saved.get(task, {})
        circuits, row, path_file = teach(ge, ids, targets, args.batch, sv.get("answer"), sv.get("order"), edges=args.edges,
                                         max_nodes=args.max_nodes, tokens=tokens, max_tokens=args.max_tokens,
                                         repair_pool=pool, no_strand=args.no_strand, edge_levels=args.edge_levels,
                                         accept=tuple(k for k in args.accept.split(",") if k), mode=args.mode)
        for name, c in circuits.items():
            (args.out / f"{task}{'.' + name if name else ''}.json").write_text(json.dumps(c))
        (args.out / f"{task}.path.json").write_text(json.dumps(path_file))
        with (args.out / f"summary_{args.offset}.jsonl").open("a") as f:
            f.write(json.dumps({"task": task, **row}) + "\n")
        b, r = row["best"], row["reads"]
        print(f"{task}: reads-first set {r['sizes']['found']} (start {r['sizes']['start']} with top-{r['sizes']['top_k']}, s_track "
              f"{r['s_track']}); pick ({row['pick']}) {b['size']} at {b['positions']} positions F1 {b['f1']:.3f} N {b['n']:.3f}"
              + (f"; connections {row['best_edges']['edges']}, graph F1 {row['best_edges']['f1_graph']:.3f}" if "best_edges" in row else "")
              + f"; search {r['seconds']:.0f}s ({r['checks']} checks in {r['check_seconds']:.0f}s), all {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
