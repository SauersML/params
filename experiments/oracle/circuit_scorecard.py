"""The circuit scorecard (#2951 graph oracle; the lead's 10-14): whether a builder's circuits are good circuits, judged
on the circuits themselves, never through the oracle, on held-out draws alone (the eval split's changed texts, donors and
input-scope draws, and a flip pool no search drew from), the judge of the builder bake-off. A circuit is a set of units
with a run semantics: VPD subcomponents under VPD's fixed masks (goal_eval's C semantics), or the model's own heads and
neurons ("units": "native", nodes ["head" | "neuron", layer, position, index]) in the real model with every unit outside
mean-ablated (a head's output and a neuron's activation at their means: native_units.unit_means, NativePhi(means=)); native circuits are scored on the families both kinds have
(native_units.SHARED; v_shared compares the two). Per circuit, on its text:
  floors   every family of the frozen reward (goal_eval.fidelity, reward mode, eval split) against its floor
           (goal_eval.vpd_floors on the same draws): the circuit's score, the floor, VPD's whole answer's score, and the
           worst shortfall v (0: every floor met)
  stated   parts with no stated path from a token to the prediction (stranded ones, which goal_eval.strand drops before
           every family, and those no token reaches along the stated lines; a node list states no lines, so there: parts
           the model gives no route to the prediction among the circuit's parts); each part's and each line's removal
           effect in the circuit's own run (lines.LinePruner: the point KL rise in bits on the text and the eval
           s_track_pp fall), those whose removal moves neither by more than INERT, and those whose removal improves both
  minimal  parts whose single removal keeps every family at its floor (or at the circuit's own score where that is below
           the floor), among every part of a circuit of at most ALL parts, else a uniform sample of SAMPLE parts and the
           WEAK parts outside it whose removal moves the circuit's own prediction least (edits' C-semantics drops)
  size     parts, lines and positions, beside the smallest k at which VPD's first k (VPD's own order, a node list)
           reaches the circuit's eval phi, and the smallest k at which they meet every floor
  known    paper parts kept (princess: 9; bracket: 11, the 4 at '<' among them) and, on a copy text
           (copycheck.copy_source), the copy core at its positions: L1.fc#2103, L1.q#316, L0.fc#53 and L2.k#206 at the
           copied position, L1.k#329 at the earlier occurrence
  edits    the mediated share (criterion 6): for the circuit's top 1, 3 and 5 units by real-model drop, the drop in log2 p
           of the model's prediction with them removed (a subcomponent at 0, a native unit at its mean) and every unit
           outside the circuit held at its value on the unedited text (VPD: every other subcomponent and remainder),
           over the live drop: 1 when the circuit carries the whole effect through its own units; and absolute edit
           faithfulness (the node family scores a circuit against VPD's answer, not
           against the model): for each part removed alone and the 3 and 5 parts with the largest real drops removed
           together, the drop in log2 p of the model's prediction at the targets in the real model (remainders live)
           against the drop the circuit's own run predicts, in C's semantics (the point family's: everything outside the
           circuit off) and in the node family's ("n": queries and keys outside at the text's activations, every norm at
           the model's scale); Spearman's rho over the single removals and the median and 90th-percentile absolute error
           in bits; the same predictions from VPD's answer at the START's candidates and from VPD's whole answer (its
           own runs, as the node family runs VPD's answer) on the removals inside each
  size     also parameters touched (each unit's weights once) and the build's seconds per text
  live     (VPD circuits, a diagnostic beside the claim) live_gates: edit_live's predictions with VPD's gates recomputed
           on each edited run, where the causal-importance network is
  teacher  from TEXT.path.json beside the circuit: the stage written, whether a smaller stage failed the search
           reward row (fallback), the START's parts and positions, whether it met the floors and was cap-bound
The criteria run cheapest first (floors and known, edits, size, minimal, stated), the record so far in OUT/<task>.partial
after each. Writes OUT/<task>.json per circuit (a circuit already written is skipped), OUT/rows.jsonl (one line per
circuit) and OUT/summary.json, and prints the table.

  circuit_scorecard.py DIR_OR_FILE [...] --out OUT [--only text9000,... | --only dev20] [--pool FILE] [--vpd FILE ...] [--offset K --stride S]
  circuit_scorecard.py table OUT
  circuit_scorecard.py bakeoff NAME=OUT [NAME=OUT ...] [--mib FILE]   the builders side by side on their common texts
  circuit_scorecard.py add OUT CIRCUITS_DIR [--vpd FILE ...]   the mediated share, parameters and build seconds added to older records
SC_CPU=1 keeps the run off the GPU (the Mac's GPU serves the RL scorer).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if os.environ.get("SC_CPU"):
    torch.backends.mps.is_available = lambda: False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
sys.path.insert(0, str(HERE.parents[1] / "method" / "vpd"))

import carry  # noqa: E402
import copycheck  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402
import native_units  # noqa: E402
import teacher  # noqa: E402

POOL = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt"  # the flips' pool: none of the searches drew
# from it (the teachers and the reward use ci_patterns_train.pt)
TASK_DIRS = (native.TEXTS / "vpd4l", Path.home() / "mpd-data/scratch/fleet/v2/paper")  # the paper's prompts are not texts
SEALED = range(9100, 9150)  # the final evaluation's texts: never scored before it
INERT = 1e-3  # a removal that moves the point KL by less than this many bits and s_track_pp by less than this moves nothing
ALL, SAMPLE, WEAK = 64, 16, 8
DEV20 = tuple(f"text{n}" for n in (9000, 9001, 9002, 9003, 9004, 9005, 9006, 9008, 9009, 9010,  # the fixed dev subset
                                   9014, 9018, 9020, 9025, 9029, 9030, 9031, 9033, 9046, 9048))  # scored every iteration
# (the lead's 10-14: the first ten, plus ten spread over length, 2-484 tokens, and the circuits' position counts, 2-15)
ENTRIES = 2 ** 27  # mask floats per engine call (edit runs)
LIVE_SINGLES = 32  # live_gates' single removals (the CI network runs per removal)
SHARE_MIN = 0.1  # a mediated share is left undefined when the live drop is under this many bits
LN2 = math.log(2.0)
BATCH = int(os.environ.get("SC_BATCH", 4))  # reward rows per fidelity call (SC_BATCH=1 for the Mac's memory)
KIND = {"MLP.up": "mlp.c_fc", "MLP.down": "mlp.down_proj", "Attn.o": "attn.o_proj", "Attn.v": "attn.v_proj",
        "Attn.k": "attn.k_proj", "Attn.q": "attn.q_proj"}
PAPER = {  # the parts the VPD paper names in its two case studies (the lead's paper_check.py), label@position
    "paper_princess": "L0.MLP.up:327@1 L0.MLP.down:3473@1 L3.Attn.k:145@1 L3.Attn.v:676@1 L3.Attn.o:281@2 "
                      "L0.MLP.up:3063@2 L0.MLP.down:1189@2 L2.MLP.up:401@2 L2.MLP.down:773@2",
    "paper_bracket": "L3.MLP.down:1414@3 L3.MLP.up:2565@3 L3.MLP.up:1051@3 L2.MLP.down:1560@3 L2.MLP.up:2151@3 "
                     "L2.Attn.o:855@3 L2.Attn.o:878@3 L2.Attn.v:473@0 L0.MLP.down:3069@0 L0.MLP.up:2149@0 L0.MLP.up:2643@0",
}
COPY_CORE = (("h.1.mlp.c_fc", 2103, 0), ("h.1.attn.q_proj", 316, 0), ("h.0.mlp.c_fc", 53, 0), ("h.2.attn.k_proj", 206, 0),
             ("h.1.attn.k_proj", 329, -1))  # (site, subcomponent, position relative to the copied position)


def paper_parts(task: str) -> list:
    out = []
    for item in PAPER.get(task, "").split():
        label, pos = item.split("@")
        layer, rest = label.split(".", 1)
        kind, idx = rest.rsplit(":", 1)
        out.append((f"h.{layer[1:]}.{KIND[kind]}", int(pos), int(idx)))
    return out


def label(nd) -> str:
    """A part as the figures write it: L1.fc#2103@393."""
    kind = {"c_fc": "fc", "down_proj": "down", "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}[nd[0].split(".")[-1]]
    return f"L{nd[0].split('.')[1]}.{kind}#{nd[2]}@{nd[1]}"


def task_path(stem: str) -> Path:
    for d in TASK_DIRS:
        if (d / f"{stem}.json").exists():
            return d / f"{stem}.json"
    raise FileNotFoundError(f"no task file for {stem} in {', '.join(map(str, TASK_DIRS))}")


def circuits(paths: list[Path], only: set) -> list[Path]:
    """The circuit files named or in the directories named (TEXT.json beside its path or lines record)."""
    out = []
    for p in paths:
        files = [p] if p.is_file() else sorted(x for x in p.iterdir() if x.suffix == ".json" and x.stem.count(".") == 0)
        out += [f for f in files if not only or f.stem in only]
    for f in out:
        if f.stem.startswith("text") and int(f.stem[4:]) in SEALED:
            raise ValueError(f"{f.stem} is in the sealed final set")
    return out


def token_reached(g: native.Graph) -> set:
    """The parts of a stated graph some token reaches along its stated lines: a residual reader with a token edge, a
    reader of a reached writer, and an attention output at a position where a reached query of its layer reads or at or
    after one where a reached key of its layer is read (queries and keys act through the attention pattern)."""
    if g.complete:
        return set(g.node_set())
    kind = {nd: native._name_layer_kind(nd[0]) for nd in g.node_set()}
    reach = {r for _, r in getattr(g, "tok_edges", [])}
    while True:
        new = {r for r, ws in g.parents.items() if r not in reach and any(w in reach for w in ws)}
        for nd in reach:
            L, k = kind[nd]
            if k in ("q_proj", "k_proj"):
                new |= {o for o, (L2, k2) in kind.items() if L2 == L and k2 == "o_proj" and o not in reach
                        and (o[1] == nd[1] if k == "q_proj" else o[1] >= nd[1])}
        if not new:
            return reach
        reach |= new


def stated_lines(pr: lines.LinePruner, g: native.Graph) -> torch.Tensor:
    """The lines (pr.lines) a stated graph states, as pr.on; a node list: all of them."""
    if g.complete:
        return torch.ones(len(pr.lines), dtype=torch.bool)
    edges = {(w, r) for r, ws in g.parents.items() for w in ws}
    outs, toks, tok_out = set(g.out), set(getattr(g, "tok_edges", [])), set(getattr(g, "tok_out", []))
    on = []
    for c in pr.lines:
        k = c[0]
        on.append((k == "e" and (c[1], c[2]) in edges) or (k == "o" and c[1] in outs) or (k == "t" and (c[1], c[2]) in toks)
                  or (k == "u" and c[1] in tok_out))
    return torch.tensor(on, dtype=torch.bool)


def effects(ge: goal_eval.GoalEval, tx: goal_eval.Text, g: native.Graph) -> dict:
    """Each part's and (a stated graph) each line's removal effect in the circuit's own run: the point KL rise in bits
    and the eval s_track_pp fall (lines.finish's measurements), on the fully wired circuit of g's parts with g's lines
    on (a node list: every line)."""
    pr = lines.LinePruner(ge, tx, sorted(g.node_set()))
    pr.on = stated_lines(pr, g)

    def tracks(ons):
        out = []
        for s0 in range(0, len(ons), 64):
            out += [r["s_track_pp"] for r in ge.track_pp(tx, [pr.g], "eval", lines=torch.stack([pr.row(d) for d in ons[s0:s0 + 64]]).float())]
            native.free(ge.nat.dev)
        return out

    point = pr.kl_sets(pr.on[None])[0]
    track = tracks(pr.on[None])[0]
    kept = [nd for nd, k in zip(pr.parts, pr.kept(pr.on).tolist()) if k]
    at = {nd: k for k, nd in enumerate(pr.parts)}
    drops = []
    for nd in kept:  # a part off: its lines with it (and whatever that strands)
        d = pr.on.clone()
        d[(pr.ends == at[nd]).any(1)] = False
        drops.append(d)
    on = [] if g.complete else pr.on.nonzero().flatten().tolist()
    for j in on:
        d = pr.on.clone()
        d[j] = False
        drops.append(d)
    pts, trs = [], []
    for s0 in range(0, len(drops), 256):  # (memory: track_pp's rows)
        pts += pr.kl_sets(torch.stack(drops[s0:s0 + 256]))
        trs += tracks(drops[s0:s0 + 256])
    fall = [None if b is None or track is None else track - b for b in trs]
    rise = [a - point for a in pts]
    parts = [{"part": list(nd), "point_rise": r, "track_fall": f} for nd, r, f in zip(kept, rise, fall)]
    lns = [{"line": json.loads(lines.key(pr.lines[j])), "point_rise": r, "track_fall": f}
           for j, r, f in zip(on, rise[len(kept):], fall[len(kept):])]
    lines.tidy(ge, tx, pr)
    return {"point": point, "track": track, "parts": parts, "lines": lns, "unreachable": [list(nd) for nd in pr.parts if nd not in set(kept)]
            if g.complete else [], "wired_dropped": [list(nd) for nd in sorted(g.node_set()) if nd not in at]}


def moves(e: dict) -> float:
    """A removal's largest effect, in bits of point KL or units of s_track_pp (a fall; a rise counts as no effect)."""
    return max(abs(e["point_rise"]), e["track_fall"] or 0.0)


def without(g: native.Graph, nd) -> native.Graph:
    if g.complete:
        return native.Graph(sorted(g.node_set() - {nd}), complete=True)
    h = goal_eval._drop_nodes(g, {nd})
    h.tok_edges = [(p, r) for p, r in getattr(g, "tok_edges", []) if r != nd]
    h.tok_out = list(getattr(g, "tok_out", []))
    return h


def rows(ge: goal_eval.GoalEval, tx: goal_eval.Text, graphs: list, pool: dict, log) -> list[dict]:
    out = []
    for s0 in range(0, len(graphs), BATCH):
        t0 = time.perf_counter()
        out += ge.fidelity(tx, graphs[s0:s0 + BATCH], pool, mode="reward", split="eval", node_values=False)
        native.free(ge.nat.dev)
        log(f"    {len(out)}/{len(graphs)} rows, {time.perf_counter() - t0:.0f} s")
    return out


def judged(row: dict, fl: dict, fams: list) -> dict:
    sc = {f: goal_eval.family_score(row, f) for f in fams}
    return {"scores": sc, "v": goal_eval.shortfall(row, fl), "phi": goal_eval.phi(row, families=fams)[0],
            "failing": [f for f in fams if (sc[f] or 0.0) < fl.get(f, 0.0)]}


@torch.no_grad()
def edit_runs(ge: goal_eval.GoalEval, tx: goal_eval.Text, g: native.Graph, refs: dict, sets: list[list], model_only: bool = False) -> dict:
    """log2 p of the model's prediction at the targets, summed over them, per removal set (row 0: none): the real model
    (an engine over every column outside the sets' folded into the remainder, outside="on": the node family's model
    run), the circuit's own run in C's semantics ("c") and in the node family's ("n"), and each reference's (refs: name
    -> complete node set) in both, as goal_eval's node family runs a complete circuit and VPD's answer."""
    import engine

    nat, T, dev = ge.nat, tx.T, ge.nat.dev
    gc = goal_eval._c_graph(g, tx.targets, nat.C, tx.vpd)
    nodes = sorted(gc.node_set())
    y = tx.ref[0].argmax(-1)
    rows = [[]] + [list(x) for x in sets]
    kw_n = ge._xkw(tx.ids, tx.targets, "nodes")

    def lp(lq):  # [R, targets, V] -> [R]
        return (lq.float().gather(-1, y.view(1, -1, 1).expand(lq.shape[0], -1, 1)).squeeze(-1).sum(-1) / LN2).cpu()

    def columns(ns):
        return {n: torch.tensor(sorted({c for m, _, c in ns if m == n}), dtype=torch.long, device=dev) for n in nat.names}

    def masked(eng, own, kw):  # own: the complete node set at 1 (None: every column at 1, the model)
        at = {n: {c: i for i, c in enumerate(eng.columns[n].tolist())} for n in nat.names}
        base = None
        if own is not None:
            base = {n: torch.zeros(1, T, eng.width(n), device=dev) for n in nat.names}
            for n, t, c in own:
                base[n][0, t, at[n][c]] = 1.0
        step = max(1, ENTRIES // (T * sum(eng.width(n) for n in nat.names)))
        out = []
        for s0 in range(0, len(rows), step):
            part = rows[s0:s0 + step]
            m = {n: (base[n].repeat(len(part), 1, 1) if base is not None else torch.ones(len(part), T, eng.width(n), device=dev)) for n in nat.names}
            for r, x in enumerate(part):
                for n, t, c in x:
                    if c in at[n]:
                        m[n][r, t, at[n][c]] = 0.0
            k = dict(kw, held={n: base[n] == 0 for n in nat.names}) if kw and base is not None else kw
            out.append(lp(eng.log_probs(m, **k)))
            native.free(dev)
        return torch.cat(out)

    cols = columns(nodes)
    res = {"model": masked(engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols, outside="on"), None, {})}
    if model_only:
        return {k: v.tolist() for k, v in res.items()}
    Ec = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=cols)
    for sem, kw in (("c", {}), ("n", kw_n)):
        if gc.complete:
            res[f"circuit_{sem}"] = masked(Ec, nodes, kw)
            continue
        plan, out = Ec.plan(gc, semantics="c"), []
        for s0 in range(0, len(rows), 32):
            part = rows[s0:s0 + 32]
            zero = {n: torch.zeros(len(part), 1, Ec.width(n), device=dev) for n in nat.names}
            out.append(lp(Ec.graph_log_probs(plan, zero, off=[x or None for x in part], **kw)))
            native.free(dev)
        res[f"circuit_{sem}"] = torch.cat(out)
    for name, ref in refs.items():
        rn = sorted(ref.node_set())
        Ev = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=columns(set(rn) | set(nodes)))
        for sem, kw in (("c", {}), ("n", kw_n)):
            res[f"{name}_{sem}"] = masked(Ev, rn, kw)
    return {k: v.tolist() for k, v in res.items()}


def faith(pred: list, real: list) -> dict:
    """Spearman's rho, median and 90th-percentile absolute error (bits) of predicted against real drops."""
    from scipy.stats import spearmanr

    p, r = np.asarray(pred, dtype=float), np.asarray(real, dtype=float)
    if len(p) < 2:
        return {"n": int(len(p)), "spearman": None, "median_abs_error": None, "p90_abs_error": None}
    rho = spearmanr(p, r).statistic if np.ptp(p) > 0 and np.ptp(r) > 0 else None
    e = np.abs(p - r)
    return {"n": int(len(p)), "spearman": None if rho is None else float(rho), "median_abs_error": float(np.median(e)),
            "p90_abs_error": float(np.quantile(e, 0.9))}


@torch.no_grad()
def mediated_vpd(ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, tops: list[list]) -> list[dict]:
    """The mediated share of each removal set R (criterion 6, the lead's 10-14): the real model's drop in log2 p of its
    prediction with R at 0, live (every other subcomponent and remainder computed) and mediated (every subcomponent
    outside the circuit, nodes, and every remainder held at its value on the unedited text, the circuit's other
    subcomponents computed; goal_eval.graph_ref's outside and hold), and their ratio."""
    import engine

    nat, T, dev = ge.nat, tx.T, ge.nat.dev
    y = tx.ref[0].argmax(-1)

    def lp(lq):
        return float(lq[0].float().gather(-1, y.view(-1, 1)).sum() / LN2)
    acts = engine.Engine(nat, tx.ids, tx.targets, live=True, columns=None).x_values()["acts"]
    ones = {n: torch.ones(T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
    rest = {}
    base = lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=ones), rest=rest))
    circ = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=dev) for n in nat.names}
    for n, t, c in nodes:
        circ[n][t, c] = True
    out = []
    for R in tops:
        live, med, held = ({n: m.clone() for n, m in ones.items()}, {n: m.clone() for n, m in circ.items()},
                           {n: a.clone() for n, a in acts.items()})
        for n, t, c in R:
            live[n][t, c] = med[n][t, c] = False
            held[n][0, t, c] = 0.0
        d_live = base - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=live)))
        d_med = base - lp(ge.graph_ref(tx.ids, tx.targets, native.Graph(masks=med), outside=held, hold=True, rest=rest))
        out.append({"k": len(R), "live": d_live, "mediated": d_med, "share": d_med / d_live if abs(d_live) >= SHARE_MIN else None})
        native.free(dev)
    return out


def edits(ge: goal_eval.GoalEval, tx: goal_eval.Text, g: native.Graph) -> dict:
    """Criterion 6 (module docstring): single removals of every part and the top 3 and 5 by real drop together."""
    nat = ge.nat
    nodes = sorted(goal_eval._c_graph(g, tx.targets, nat.C, tx.vpd).node_set())
    act = sorted((n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist())
    ps = set(tx.targets) | set(ge.start_candidates(tx)[:goal_eval.START_CAP])
    refs = {"vpd_candidates": native.Graph([nd for nd in act if nd[1] in ps], complete=True), "vpd_whole": native.Graph(act, complete=True)}
    single = edit_runs(ge, tx, native.Graph(nodes, complete=True), {}, [[nd] for nd in nodes], model_only=True)["model"]
    real1 = [single[0] - x for x in single[1:]]
    top = [nodes[i] for i in np.argsort(real1)[::-1][:5]]
    sets = [[nd] for nd in nodes] + [top[:3], top[:5]]
    runs = edit_runs(ge, tx, g, refs, sets)
    drops = {k: [v[0] - x for x in v[1:]] for k, v in runs.items()}
    n1 = len(nodes)
    out = {"parts": [label(nd) for nd in nodes], "nodes": [list(nd) for nd in nodes], "drops": drops, "groups": {"top3": [label(nd) for nd in top[:3]], "top5": [label(nd) for nd in top[:5]]}}
    inside = {name: [all(x in set(ref.node_set()) for x in s) for s in sets] for name, ref in refs.items()}
    stats = {}
    for k, d in drops.items():
        if k == "model":
            continue
        ok = [i for i in range(n1) if inside.get(k.rsplit("_", 1)[0], [True] * len(sets))[i]]
        st = faith([d[i] for i in ok], [drops["model"][i] for i in ok])
        st.update({f"{gname}": {"predicted": d[n1 + j], "real": drops["model"][n1 + j],
                                "inside": inside.get(k.rsplit("_", 1)[0], [True] * len(sets))[n1 + j]} for j, gname in enumerate(("top3", "top5"))})
        stats[k] = st
    out["stats"] = stats
    out["mediated"] = mediated_vpd(ge, tx, nodes, [top[:1], top[:3], top[:5]])
    return out


def params_touched(nat, nodes) -> int:
    """The weights a circuit's units read and write, each counted once whatever its positions: a VPD subcomponent its V
    column and U row (d_in + d_out), a neuron its input row and output column (2 d_model), a head its query, key, value
    and output slices (4 hd d_model): units of different kinds compare by this beside their count."""
    tg = nat.target
    seen = {(nd[0], nd[1], nd[3]) if nd[0] in ("head", "neuron") else (nd[0], nd[2]) for nd in nodes}
    d, hd = tg.wte.shape[1], tg.hd
    total = 0
    for u in seen:
        if u[0] == "neuron":
            total += 2 * d
        elif u[0] == "head":
            total += 4 * hd * d
        else:
            st = tg.site(u[0])
            total += st.V.shape[0] + st.U.shape[1]
    return total


def build_seconds(path: Path) -> float | None:
    """A circuit's build time: its own "build_seconds", else the teacher's summary row for its task beside it."""
    d = json.loads(path.read_text())
    if d.get("build_seconds") is not None:
        return float(d["build_seconds"])
    got = None
    for f in sorted(path.parent.iterdir()):
        if f.name.startswith("summary") and f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("task") == path.stem and r.get("seconds") is not None:
                        got = float(r["seconds"])
    return got


def live_gates(ge: goal_eval.GoalEval, tx: goal_eval.Text, ed: dict) -> dict | None:
    """A VPD diagnostic beside the circuit's claim (the lead's decision 3, 10-14): edit_live.live's predictions with VPD's
    causal-importance gates recomputed on each edited run (vpd_live, circuit_live, circuit_plus, circuit_fixed), on the
    edits' top LIVE_SINGLES single removals by real drop and the top 3 and 5 together, against the real drops; the
    parts the gates switch on and the circuit parts they switch off. None without the causal-importance network."""
    import vpd_model

    if not Path(vpd_model.VPD_PTH).exists():
        return None
    import edit_live

    nodes = [tuple(x) for x in ed["nodes"]]
    real, n1 = ed["drops"]["model"], len(nodes)
    order = [int(i) for i in np.argsort(real[:n1])[::-1][:LIVE_SINGLES]]
    at = dict(zip(ed["parts"], nodes))
    top = [at[x] for x in ed["groups"]["top5"]]
    got = edit_live.live(ge, tx, nodes, [[nodes[i]] for i in order] + [top[:3], top[:5]])
    reals = [real[i] for i in order] + real[n1:n1 + 2]
    out = {k: edit_live.group_stats([got[k][0] - x for x in got[k][1:]], reals, len(order)) for k in edit_live.LIVE}
    out.update(on=got["on"], off=got["off"], singles=len(order), clean_vs_saved=got["clean_vs_saved"])
    return out


def teacher_block(path: Path) -> dict | None:
    """The teacher's record beside a circuit (TEXT.path.json): stage written, fallback (a smaller stage failed the
    search reward row), the START's parts, positions, floors met and cap bound."""
    pj = path.with_name(f"{path.stem}.path.json")
    if not pj.exists():
        return None
    b = json.loads(pj.read_text()).get("blocks", {})
    rows = b.get("phi") or []
    stage = b.get("phi_stage")
    units = next((r[3] for r in rows if r[0] == stage), None)
    st = b.get("start") or {}
    return {"stage": stage, "fallback": any(r[3] < units and r[7] > 0 for r in rows) if units is not None else None,
            "start_parts": len((b.get("stage_nodes") or {}).get("start", [])) or None, "start_met": st.get("met"),
            "cap_bound": st.get("cap_bound"), "start_positions": len(b.get("positions") or []) or None}


def say(*a):
    print(*a, flush=True)


def is_native(path: Path) -> bool:
    """A circuit of the model's own units (heads and neurons, native_units' kinds) rather than VPD's subcomponents."""
    d = json.loads(path.read_text())
    return d.get("units") == "native" or bool(d.get("nodes")) and d["nodes"][0][0] in ("head", "neuron")


@torch.no_grad()
def native_lq(ge: goal_eval.GoalEval, tx: goal_eval.Text, nphi, S, sets: list[list], mode: str, B: int = 16) -> torch.Tensor:
    """log q at the targets [len(sets), targets, V] with each set of S's units removed (mean-ablated): mode "live" the
    real model, "mediated" the real model with every unit outside S held at its value on the text, "circuit" S's own
    run (every unit outside S mean-ablated)."""
    cx, ab, out = nphi.capture(tx.ids), nphi._ablate(tx.T), []
    for s0 in range(0, len(sets), B):
        hs, ns, vals = [], [], []
        for R in sets[s0:s0 + B]:
            hm, nm = torch.zeros_like(S.heads), torch.zeros_like(S.neurons)
            for kind, l, t, j in R:
                (hm if kind == "head" else nm)[l, t, j] = True
            hs.append(hm if mode == "live" else ~S.heads | hm)
            ns.append(nm if mode == "live" else ~S.neurons | nm)
            vals.append(nphi._mix(cx, ab, hm, nm) if mode == "mediated" else ab)
        v = {k: torch.cat([x[k] for x in vals]) for k in vals[0]}
        out.append(ge.graph_ref(tx.ids, tx.targets, nphi.model, units=nphi._units(hs, ns, v)).float())
        native.free(ge.nat.dev)
    return torch.cat(out)


def score_native(ge: goal_eval.GoalEval, path: Path, pool: dict, vpd: dict | None, means: dict, pool_name: str = POOL.name,
                 log=say, save=None) -> dict:
    """A native circuit's record (the bake-off's unit-agnostic criteria, the lead's 10-14): the shared families
    (native_units.SHARED, NativePhi with mean ablation on the eval split) against the text's floors, each unit's
    own-run removal effect, minimality, size in units and parameters, and edits (the mediated share and the circuit's
    predicted drops)."""
    nat = ge.nat
    t_start = time.perf_counter()
    tp = task_path(path.stem)
    tx = lines.text_of(ge, tp, vpd)
    d = json.loads(path.read_text())
    units = sorted({(k, int(l), int(t), int(j)) for k, l, t, j in d["nodes"]})
    nu = native_units.NativeUnits(nat, tx.ids, tx.targets)
    S = native_units.from_units(nu, units)
    nphi = native_units.NativePhi(ge, tx, "eval", means=means)
    rec = {"task": path.stem, "circuit": str(path), "teacher": teacher_block(path), "pool": pool_name, "units": "native"}

    def done(name):
        rec["seconds"] = time.perf_counter() - t_start
        log(f"  {path.stem}: {name}, {rec['seconds']:.0f} s")
        if save is not None:
            save(rec)

    vf = ge.vpd_floors(tx, pool, "eval")
    fams = [f for f in vf["families"] if f in native_units.SHARED]
    fl = {f: vf["floors"][f] for f in fams}
    base = judged(nphi.rows([S])[0], fl, fams)
    rec["floors"] = {"families": {f: {"score": base["scores"][f], "floor": fl.get(f), "vpd": goal_eval.family_score(vf["vpd"], f)} for f in fams},
                     "v": base["v"], "phi": base["phi"], "phi_vpd": goal_eval.phi(vf["vpd"], families=fams)[0], "failing": base["failing"],
                     "families_scored": "shared"}
    rec["known"] = {}
    done("floors")

    y = tx.ref[0].argmax(-1)

    def lp(lq):
        return (lq.gather(-1, y.view(1, -1, 1).expand(lq.shape[0], -1, 1)).squeeze(-1).sum(-1) / LN2).tolist()
    singles = [[u] for u in units]
    live = lp(native_lq(ge, tx, nphi, S, [[]] + singles, "live"))
    real1 = [live[0] - x for x in live[1:]]
    top = [units[i] for i in np.argsort(real1)[::-1][:5]]
    groups = [top[:1], top[:3], top[:5]]
    g_live = lp(native_lq(ge, tx, nphi, S, groups, "live"))
    g_med = lp(native_lq(ge, tx, nphi, S, groups, "mediated"))
    circ_lq = native_lq(ge, tx, nphi, S, [[]] + singles + groups[1:], "circuit")
    circ = lp(circ_lq)
    pred = [circ[0] - x for x in circ[1:]]
    st = faith(pred[:len(units)], real1)
    for j, gname in enumerate(("top3", "top5")):
        st[gname] = {"predicted": pred[len(units) + j], "real": live[0] - g_live[1 + j], "inside": True}
    rec["edits"] = {"parts": [native_label(u) for u in units], "nodes": [list(u) for u in units],
                    "drops": {"model": real1, "circuit_mean": pred[:len(units)]},
                    "groups": {"top3": [native_label(u) for u in top[:3]], "top5": [native_label(u) for u in top[:5]]},
                    "stats": {"circuit_mean": st},
                    "mediated": [{"k": len(R), "live": live[0] - a, "mediated": live[0] - b,
                                  "share": (live[0] - b) / (live[0] - a) if abs(live[0] - a) >= SHARE_MIN else None}
                                 for R, a, b in zip(groups, g_live, g_med)]}
    done("edits")

    rec["size"] = {"units": "native", "parts": len(units), "heads": sum(u[0] == "head" for u in units),
                   "neurons": sum(u[0] == "neuron" for u in units), "params": params_touched(nat, units),
                   "build_seconds": build_seconds(path), "lines": None, "positions": sorted({u[2] for u in units}),
                   "vpd_k_phi": None, "vpd_k_floors": None}
    done("size")

    weak_order = sorted(units, key=lambda u: abs(pred[units.index(u)]))
    if len(units) <= ALL:
        uniform, weak = units, []
    else:
        rng = np.random.default_rng(native.task_seed(path.stem))
        uniform = [units[i] for i in sorted(rng.choice(len(units), SAMPLE, replace=False))]
        weak = [u for u in weak_order if u not in set(uniform)][:WEAK]
    tested = uniform + weak
    need = {f: min(fl.get(f, 0.0), base["scores"][f] or 0.0) for f in fams}
    mrows = []
    for s0 in range(0, len(tested), BATCH):
        part = tested[s0:s0 + BATCH]
        for u, r in zip(part, nphi.rows([S.without([u]) for u in part])):
            j = judged(r, fl, fams)
            drop = {f: (j["scores"][f] or 0.0) - need[f] for f in fams}
            worst = min(drop, key=drop.get)
            mrows.append({"part": list(u), "removable": drop[worst] >= 0.0, "v": j["v"], "phi": j["phi"], "worst": worst,
                          "worst_margin": drop[worst], "uniform": u in set(uniform), "scores": j["scores"]})
        native.free(nat.dev)
    rec["minimal"] = {"tested": len(mrows), "uniform": [sum(m["removable"] for m in mrows if m["uniform"]), len(uniform)],
                      "weak": [sum(m["removable"] for m in mrows if not m["uniform"]), len(weak)],
                      "removable": [m["part"] for m in mrows if m["removable"]], "rows": mrows}
    done("minimality")

    k0 = carry.kl(tx.ref, circ_lq[:1])[0].item()
    rise = (carry.kl(tx.ref.expand(len(units), -1, -1), circ_lq[1:1 + len(units)]) - k0).tolist()
    sets = [S] + [S.without([u]) for u in units]
    tr = []
    for s0 in range(0, len(sets), 32):
        tr += [r["s_track_pp"] for r in nphi.track_pp(sets[s0:s0 + 32])]
        native.free(nat.dev)
    fall = [None if b is None or tr[0] is None else tr[0] - b for b in tr[1:]]
    eff = [{"part": list(u), "point_rise": a, "track_fall": b} for u, a, b in zip(units, rise, fall)]
    rec["stated"] = {"stated": False, "defects": {}, "stranded": [], "no_token": [], "unreachable": [], "point": k0, "track": tr[0],
                     "inert_parts": [e["part"] for e in eff if moves(e) < INERT],
                     "inert_lines": [], "harmful_parts": [e["part"] for e in eff if e["point_rise"] < -INERT and (e["track_fall"] or 0.0) <= INERT],
                     "part_effects": eff, "line_effects": []}
    done("stated")
    return rec


def native_label(u) -> str:
    """A native unit as the figures write it: L1.head3@393, L0.n2103@12."""
    return f"L{u[1]}.{'head' if u[0] == 'head' else 'n'}{u[3]}@{u[2]}"


def score(ge: goal_eval.GoalEval, path: Path, pool: dict, vpd: dict | None = None, pool_name: str = POOL.name, log=say,
          save=None) -> dict:
    """One circuit's record (module docstring), its criteria in order of cost: floors and known mechanisms, edits, size,
    minimality, then the stated checks; save(record so far) after each (the partial record of a run cut short)."""
    nat = ge.nat
    t_start = time.perf_counter()
    tp = task_path(path.stem)
    task = json.loads(tp.read_text())
    tx = lines.text_of(ge, tp, vpd)
    g, defects, _ = goal_eval.load_answer(path, task, tx.targets, nat)
    nodes = g.node_set()
    rec = {"task": path.stem, "circuit": str(path), "teacher": teacher_block(path), "pool": pool_name}

    def done(name):
        rec["seconds"] = time.perf_counter() - t_start
        log(f"  {path.stem}: {name}, {rec['seconds']:.0f} s")
        if save is not None:
            save(rec)

    vf = ge.vpd_floors(tx, pool, "eval")
    fams, fl = vf["families"], vf["floors"]
    base = judged(rows(ge, tx, [g], pool, log)[0], fl, fams)
    rec["floors"] = {"families": {f: {"score": base["scores"][f], "floor": fl.get(f), "vpd": goal_eval.family_score(vf["vpd"], f)} for f in fams},
                     "v": base["v"], "phi": base["phi"], "phi_vpd": goal_eval.phi(vf["vpd"], families=fams)[0], "failing": base["failing"]}
    known = {}
    pp = paper_parts(path.stem)
    if pp:
        kept = [nd for nd in pp if nd in nodes]
        known["paper"] = {"kept": len(kept), "of": len(pp), "missing": [label(nd) for nd in pp if nd not in nodes]}
        if path.stem == "paper_bracket":
            known["paper"]["kept_at_open"] = [sum(nd in nodes for nd in pp if nd[1] == 0), sum(nd[1] == 0 for nd in pp)]
    src = copycheck.copy_source(nat, tx.ids, tx.targets)
    if src is not None:
        s = src[0]
        core = [(n, s + d, c) for n, c, d in COPY_CORE]
        known["copy"] = {"source": s, "kept": [label(nd) for nd in core if nd in nodes], "missing": [label(nd) for nd in core if nd not in nodes],
                         "any_position": [label((n, 0, c))[:-2] for n, c, _ in COPY_CORE if any(m == n and i == c for m, _, i in nodes)],
                         "reads_source": any(t == s for _, t, _ in nodes)}
    rec["known"] = known
    done("floors")

    rec["edits"] = ed = edits(ge, tx, g)
    rec["live_gates"] = live_gates(ge, tx, ed)
    done("edits")

    order = tx.vpd_order if getattr(tx, "vpd_order", None) else nat._vpd_order(tx.ids, tx.targets)
    order = [tuple(nd) for nd in order]
    n_act = len(order)
    grid = sorted({min(n_act, 2 ** i) for i in range(3, 20)} | {min(n_act, g.count())})
    have = {n_act: judged(vf["vpd"], fl, fams)}  # (VPD's whole answer: vpd_floors' row)
    todo = [k for k in grid if k not in have]
    for k, r in zip(todo, rows(ge, tx, [native.Graph(order[:k], complete=True) for k in todo], pool, log)):
        have[k] = judged(r, fl, fams)

    def first(ok):
        return next((k for k in sorted(have) if ok(have[k])), None)
    k_phi = first(lambda j: j["phi"] is not None and base["phi"] is not None and j["phi"] >= base["phi"])
    if k_phi is not None and k_phi > grid[0]:  # three geometric steps between the grid point below and k_phi
        lo = max(k for k in have if k < k_phi)
        mids = sorted({int(round(lo * (k_phi / lo) ** (i / 4))) for i in (1, 2, 3)} - set(have))
        for k, r in zip(mids, rows(ge, tx, [native.Graph(order[:k], complete=True) for k in mids], pool, log)):
            have[k] = judged(r, fl, fams)
        k_phi = first(lambda j: j["phi"] is not None and j["phi"] >= base["phi"])
    rec["size"] = {"units": "vpd", "parts": len(nodes), "params": params_touched(nat, nodes), "build_seconds": build_seconds(path), "lines": None if g.complete else g.edges() + len(getattr(g, "tok_edges", [])) + len(getattr(g, "tok_out", [])),
                   "connections": None if g.complete else g.edges(),
                   "token_edges": None if g.complete else len(getattr(g, "tok_edges", [])) + len(getattr(g, "tok_out", [])),
                   "implied_lines": sum(native.implied_edges(nodes, tx.targets)) if g.complete else None,
                   "positions": sorted({t for _, t, _ in nodes}), "vpd_active": n_act, "vpd_k_phi": k_phi,
                   "vpd_k_floors": first(lambda j: j["v"] == 0.0),
                   "vpd_positions_k_phi": len({t for _, t, _ in order[:k_phi]}) if k_phi else None,
                   "vpd_curve": [[k, have[k]["phi"], have[k]["v"]] for k in sorted(have)]}
    done("size")

    parts = sorted(goal_eval._c_graph(g, tx.targets, nat.C, tx.vpd).node_set())
    own = dict(zip([tuple(x) for x in ed["nodes"]], ed["drops"]["circuit_c"]))  # the circuit's own-run drop per part
    weak_order = sorted(parts, key=lambda nd: abs(own.get(nd, 0.0)))
    if len(parts) <= ALL:
        uniform, weak = parts, []
    else:
        rng = np.random.default_rng(native.task_seed(path.stem))
        uniform = [parts[i] for i in sorted(rng.choice(len(parts), SAMPLE, replace=False))]
        weak = [nd for nd in weak_order if nd not in set(uniform)][:WEAK]
    tested = uniform + weak
    need = {f: min(fl.get(f, 0.0), base["scores"][f] or 0.0) for f in fams}
    mrows = []
    for nd, r in zip(tested, rows(ge, tx, [without(g, nd) for nd in tested], pool, log)):
        j = judged(r, fl, fams)
        drop = {f: (j["scores"][f] or 0.0) - need[f] for f in fams}
        worst = min(drop, key=drop.get)
        mrows.append({"part": list(nd), "removable": drop[worst] >= 0.0, "v": j["v"], "phi": j["phi"], "worst": worst,
                      "worst_margin": drop[worst], "uniform": nd in set(uniform), "scores": j["scores"]})
    rec["minimal"] = {"tested": len(mrows), "uniform": [sum(m["removable"] for m in mrows if m["uniform"]), len(uniform)],
                      "weak": [sum(m["removable"] for m in mrows if not m["uniform"]), len(weak)],
                      "removable": [m["part"] for m in mrows if m["removable"]], "rows": mrows}
    done("minimality")

    eff = effects(ge, tx, g)
    reached = token_reached(g)
    rec["stated"] = {"stated": not g.complete, "defects": defects, "stranded": [list(nd) for nd in getattr(g, "stranded", [])],
                     "no_token": [list(nd) for nd in sorted(nodes) if nd not in reached],
                     "unreachable": eff["unreachable"], "point": eff["point"], "track": eff["track"],
                     "inert_parts": [e["part"] for e in eff["parts"] if moves(e) < INERT],
                     "inert_lines": [e["line"] for e in eff["lines"] if moves(e) < INERT],
                     "harmful_parts": [e["part"] for e in eff["parts"] if e["point_rise"] < -INERT and (e["track_fall"] or 0.0) <= INERT],
                     "part_effects": eff["parts"], "line_effects": eff["lines"]}
    done("stated")
    return rec


def add(out: Path, circuits_dir: Path, vpd: dict | None) -> None:
    """Records written before the mediated share, parameters touched and build seconds: those computed and added (the
    circuit of each record from circuits_dir, by task)."""
    ge = goal_eval.GoalEval(native.Native(), chunk=32)
    for f in sorted(out.iterdir()):
        if f.suffix != ".json" or f.name == "summary.json":
            continue
        rec = json.loads(f.read_text())
        if "mediated" in rec.get("edits", {}) and "params" in rec.get("size", {}) and rec.get("live_gates") is not None:
            continue
        path = circuits_dir / f.name
        tp = task_path(path.stem)
        tx = lines.text_of(ge, tp, vpd)
        ed = rec["edits"]
        nodes = [tuple(x) for x in ed["nodes"]]
        at = dict(zip(ed["parts"], nodes))
        top = [at[x] for x in ed["groups"]["top5"]]
        if "mediated" not in ed:
            ed["mediated"] = mediated_vpd(ge, tx, nodes, [top[:1], top[:3], top[:5]])
        if rec.get("live_gates") is None:
            rec["live_gates"] = live_gates(ge, tx, ed)
        g, _, _ = goal_eval.load_answer(path, json.loads(tp.read_text()), tx.targets, ge.nat)
        rec["size"].update({"units": "vpd", "params": params_touched(ge.nat, g.node_set()), "build_seconds": build_seconds(path)})
        f.write_text(json.dumps(rec))
        lg = rec.get("live_gates") or {}
        say(f"{path.stem}: mediated {[(m['k'], None if m['share'] is None else round(m['share'], 2)) for m in ed['mediated']]}, "
            f"live gates top-3 error {[(k, round(lg[k]['top3']['error'], 2)) for k in ('vpd_live', 'circuit_live', 'circuit_plus') if k in lg]}")


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [None, None]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def v_shared(r: dict) -> float:
    """The worst shortfall over the families every unit kind is scored on (native_units.SHARED): the bake-off's v."""
    fam = r["floors"]["families"]
    return max((max(0.0, x["floor"] - (x["score"] or 0.0)) for f, x in fam.items() if f in native_units.SHARED and x["floor"] is not None),
               default=0.0)


def own_stats(r: dict) -> dict:
    """The circuit's own edit predictions: the node family's semantics for VPD circuits, mean ablation for native ones."""
    es = (r.get("edits") or {}).get("stats", {})
    return es.get("circuit_n") or es.get("circuit_mean") or {}


def share(r: dict, k: int):
    return next((m["share"] for m in (r.get("edits") or {}).get("mediated", []) if m["k"] == k), None)


def share_distance(x):
    """A mediated share's distance from 1, |log2 share| (1: a factor of two either way, amplification by the outside
    and self-repair alike; inf for a share at or below 0, a sign flip); None where the share is undefined."""
    return None if x is None else (abs(math.log2(x)) if x > 0 else math.inf)


def summary(recs: list[dict]) -> dict:
    """The set's summary: counts and medians over its circuits (texts without a value left out of a median)."""
    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None
    fail = {}
    for r in recs:
        for f in r["floors"]["failing"]:
            fail[f] = fail.get(f, 0) + 1
    u = [sum(r["minimal"]["uniform"][0] for r in recs), sum(r["minimal"]["uniform"][1] for r in recs)]
    parts = sum(r["size"]["parts"] for r in recs)
    nl = [r for r in recs if r["stated"]["stated"]]
    ratio = [r["size"]["parts"] / r["size"]["vpd_k_phi"] for r in recs if r["size"]["vpd_k_phi"]]
    copy = [r for r in recs if "copy" in r["known"]]
    fams = sorted({f for r in recs for f in r["floors"]["families"]})
    margins = {f: med([(r["floors"]["families"][f]["score"] or 0.0) - r["floors"]["families"][f]["floor"] for r in recs
                       if f in r["floors"]["families"] and r["floors"]["families"][f]["floor"] is not None]) for f in fams}
    preds = sorted({k for r in recs for k in (r.get("edits") or {}).get("stats", {})})
    edit = {k: {m: med([r["edits"]["stats"][k][m] for r in recs if k in (r.get("edits") or {}).get("stats", {})])
                for m in ("spearman", "median_abs_error", "p90_abs_error")} for k in preds}
    for k in preds:
        for gname in ("top3", "top5"):
            edit[k][f"{gname}_median_abs_error"] = med([abs(r["edits"]["stats"][k][gname]["predicted"] - r["edits"]["stats"][k][gname]["real"])
                                                         for r in recs if k in (r.get("edits") or {}).get("stats", {})])
    tb = [r["teacher"] for r in recs if r.get("teacher")]
    return {"pool": sorted({r.get("pool") for r in recs}), "units": sorted({r["size"].get("units", "vpd") for r in recs}),
            "meet_shared_floors": sum(v_shared(r) == 0.0 for r in recs), "median_v_shared": med([v_shared(r) for r in recs]),
            "median_mediated_share": {k: med([share(r, k) for r in recs]) for k in (1, 3, 5)},
            "median_share_distance": {k: med([share_distance(share(r, k)) for r in recs]) for k in (1, 3, 5)},
            "median_params": med([r["size"].get("params") for r in recs]), "median_build_seconds": med([r["size"].get("build_seconds") for r in recs]),
            "median_margin_by_family": margins, "edits_medians": edit,
            "teacher_fallback": sum(bool(t["fallback"]) for t in tb), "teacher_records": len(tb),
"circuits": len(recs), "meet_every_floor": sum(r["floors"]["v"] == 0.0 for r in recs),
            "failing_family_counts": dict(sorted(fail.items(), key=lambda x: -x[1])), "median_v": med([r["floors"]["v"] for r in recs]),
            "stated_circuits": len(nl), "parts": parts,
            "no_path_parts": sum(len(r["stated"]["stranded"]) + len(r["stated"]["no_token"]) for r in recs),
            "unreachable_parts": sum(len(r["stated"]["unreachable"]) for r in recs),
            "inert_parts": sum(len(r["stated"]["inert_parts"]) for r in recs),
            "harmful_parts": sum(len(r["stated"]["harmful_parts"]) for r in recs),
            "inert_lines": [sum(len(r["stated"]["inert_lines"]) for r in nl), sum(len(r["stated"]["line_effects"]) for r in nl)],
            "removable_uniform": u + [u[0] / u[1] if u[1] else None] + [wilson(*u)],
            "removable_weak": [sum(r["minimal"]["weak"][0] for r in recs), sum(r["minimal"]["weak"][1] for r in recs)],
            "circuits_with_a_removable_part": sum(bool(r["minimal"]["removable"]) for r in recs),
            "median_parts": med([r["size"]["parts"] for r in recs]), "median_lines": med([r["size"]["lines"] for r in recs]),
            "median_positions": med([len(r["size"]["positions"]) for r in recs]),
            "median_vpd_k_phi": med([r["size"]["vpd_k_phi"] for r in recs]),
            "median_parts_over_vpd_k_phi": med(ratio), "smaller_than_vpd_at_same_phi": sum(x < 1 for x in ratio),
            "vpd_never_reaches": sum(r["size"]["vpd_k_phi"] is None for r in recs),
            "median_vpd_k_floors": med([r["size"]["vpd_k_floors"] for r in recs]),
            "paper": {r["task"]: r["known"]["paper"] for r in recs if "paper" in r["known"]},
            "copy_texts": len(copy), "copy_core_kept": [sum(len(r["known"]["copy"]["kept"]) for r in copy), 5 * len(copy)],
            "copy_reads_source": sum(r["known"]["copy"]["reads_source"] for r in copy)}


def num(x) -> str:
    return f"{x:>6.2f}" if x is not None else f"{'-':>6}"


def row_line(r: dict) -> dict:
    """rows.jsonl's line for a circuit (pymove's diff of search iterations)."""
    mn, es = r["minimal"], (r.get("edits") or {}).get("stats", {})
    return {"task": r["task"], "v": r["floors"]["v"], "failing": r["floors"]["failing"], "parts": r["size"]["parts"],
            "lines": r["size"]["lines"], "positions": len(r["size"]["positions"]), "vpd_k_phi": r["size"]["vpd_k_phi"],
            "removable": [mn["uniform"][0] + mn["weak"][0], mn["uniform"][1] + mn["weak"][1]],
            "removable_share_uniform": mn["uniform"][0] / mn["uniform"][1] if mn["uniform"][1] else None,
            "inert_parts": len(r["stated"]["inert_parts"]), "no_path": len(r["stated"]["stranded"]) + len(r["stated"]["no_token"]),
            "edit_rho": own_stats(r).get("spearman"), "edit_err": own_stats(r).get("median_abs_error"),
            "edit_err_vpd_whole_n": es.get("vpd_whole_n", {}).get("median_abs_error"), "v_shared": v_shared(r),
            "mediated": {k: share(r, k) for k in (1, 3, 5)}, "units": r["size"].get("units", "vpd"), "params": r["size"].get("params"),
            "build_seconds": r["size"].get("build_seconds"), "teacher": r.get("teacher"), "pool": r.get("pool")}


def table(recs: list[dict]) -> str:
    head = (f"{'task':12} {'units':6} {'parts':>6} {'params':>8} {'build_s':>7} {'pos':>4} {'v':>6} {'v_sh':>6} {'failing':24} "
            f"{'nopath':>6} {'inert':>9} {'remov':>7} {'weak':>5} {'vpd_k':>6} {'rho':>6} {'err':>6} {'med1':>5} {'med3':>5} {'med5':>5} known")
    out = [head]
    for r in sorted(recs, key=lambda r: r["task"]):
        st, mn, sz = r["stated"], r["minimal"], r["size"]
        kn, os_ = r["known"], own_stats(r)
        known = (f"paper {kn['paper']['kept']}/{kn['paper']['of']}" if "paper" in kn else "") + \
                (f"copy {len(kn['copy']['kept'])}/5" if "copy" in kn else "")
        sh = [share(r, k) for k in (1, 3, 5)]
        out.append(f"{r['task']:12} {sz.get('units', 'vpd'):6} {sz['parts']:>6} {(sz.get('params') or 0) / 1e6:>7.2f}M "
                   f"{sz.get('build_seconds') or 0:>7.0f} {len(sz['positions']):>4} {r['floors']['v']:>6.3f} {v_shared(r):>6.3f} "
                   f"{','.join(r['floors']['failing'])[:24]:24} "
                   f"{len(st['stranded']) + len(st['no_token']):>6} {len(st['inert_parts']):>4}/{len(st['part_effects']):<4} "
                   f"{mn['uniform'][0]:>3}/{mn['uniform'][1]:<3} {mn['weak'][0]:>2}/{mn['weak'][1]:<2} "
                   f"{sz['vpd_k_phi'] if sz['vpd_k_phi'] is not None else '-':>6} {num(os_.get('spearman'))} {num(os_.get('median_abs_error'))} "
                   + " ".join(f"{x:>5.2f}" if x is not None else f"{'-':>5}" for x in sh) + f" {known}")
    return "\n".join(out)


def mib_rows(d: Path, names) -> dict:
    """MIB's numbers per builder from fleet's directory (interpbench/builders/<builder>.json, mib_score.py's output):
    circuit-localization AUROC [95% CI] and its paired difference from EAP-IG-inputs [95% CI] on MIB's IOI test
    prompts 0-31, as the benchmark reports them."""
    out = {}
    for n in names:
        f = d / f"{n}.json"
        if f.exists():
            r = json.loads(f.read_text())
            dv = r.get("diff_vs_eap_ig_inputs") or {}
            out[n] = {"AUROC": f"{r['auroc']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]",
                      "vs EAP-IG-in": f"{dv['diff']:+.3f} [{dv['ci95'][0]:+.2f},{dv['ci95'][1]:+.2f}]" if dv else "-"}
    return out


def bakeoff(sets: dict, mib: dict | None = None) -> str:
    """The builders side by side (the lead's 10-14 bake-off): each set (name -> its records) on the texts every set
    scored, its summary's headline numbers, and MIB's external numbers per builder when given (mib: name -> {metric:
    value}, read as the benchmark reports them)."""
    common = set.intersection(*[{r["task"] for r in recs} for recs in sets.values()]) if sets else set()
    cols = [("texts", lambda s: s["circuits"]), ("floors met (shared)", lambda s: s["meet_shared_floors"]),
            ("median v (shared)", lambda s: s["median_v_shared"]), ("removable share", lambda s: s["removable_uniform"][2]),
            ("inert parts", lambda s: s["inert_parts"] / max(1, s["parts"])),
            ("mediated top1", lambda s: s["median_mediated_share"][1]), ("mediated top3", lambda s: s["median_mediated_share"][3]),
            ("mediated top5", lambda s: s["median_mediated_share"][5]),
            ("|log2 share| top1", lambda s: s["median_share_distance"][1]), ("|log2 share| top3", lambda s: s["median_share_distance"][3]),
            ("|log2 share| top5", lambda s: s["median_share_distance"][5]), ("median units", lambda s: s["median_parts"]),
            ("median params (M)", lambda s: None if s["median_params"] is None else s["median_params"] / 1e6),
            ("median build s", lambda s: s["median_build_seconds"]), ("median positions", lambda s: s["median_positions"])]
    rows = {}
    for name, recs in sets.items():
        sm = summary([r for r in recs if r["task"] in common])
        es = [own_stats(r).get("spearman") for r in recs if r["task"] in common]
        es = [x for x in es if x is not None]
        rows[name] = [f(sm) for _, f in cols] + [float(np.median(es)) if es else None]
    names = [c for c, _ in cols] + ["edit rho"]
    mib_keys = sorted({k for v in (mib or {}).values() for k in v})
    out = [f"{len(common)} texts scored by every builder", f"{'':22}" + "".join(f"{n:>16}" for n in sets)]
    for i, c in enumerate(names):
        out.append(f"{c:22}" + "".join(f"{'-' if rows[n][i] is None else (f'{rows[n][i]:.3f}' if isinstance(rows[n][i], float) else rows[n][i]):>16}" for n in sets))
    for k in mib_keys:
        out.append(f"{'MIB ' + k:22}" + "".join(f"{(mib.get(n) or {}).get(k, '-')!s:>16}" for n in sets))
    return "\n".join(out)


def load(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.iterdir()) if p.suffix == ".json" and p.name != "summary.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="circuit files or directories of them; 'table OUT' prints a finished run")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", default="", help="task names, comma-separated, or dev20 (DEV20)")
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--vpd", type=Path, nargs="*", default=[], help="VPD's saved answers and orders (teacher.py --save-vpd's "
                    "files, merged), where the causal-importance network is not")
    ap.add_argument("--mib", type=Path, help="bakeoff: fleet's MIB directory (interpbench/builders, <builder>.json per "
                    "builder name) or a JSON {builder: {metric: value}}")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    if str(a.paths[0]) == "add":  # add OUT CIRCUITS_DIR
        add(a.paths[1], a.paths[2], {k: v for f in a.vpd for k, v in torch.load(f).items()} or None)
        return
    if str(a.paths[0]) == "bakeoff":  # bakeoff NAME=OUT ... [--mib FILE: {builder: {metric: value}}]
        sets = {x.split("=", 1)[0]: load(Path(x.split("=", 1)[1])) for x in map(str, a.paths[1:])}
        mib = None if not a.mib else (mib_rows(a.mib, sets) if a.mib.is_dir() else json.loads(a.mib.read_text()))
        print(bakeoff(sets, mib))
        return
    if str(a.paths[0]) == "table":
        recs = load(a.paths[1])
        print(table(recs))
        print(json.dumps(summary(recs), indent=1))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    only = set(DEV20) if a.only == "dev20" else set(filter(None, a.only.split(",")))
    todo = circuits(a.paths, only)[a.offset::a.stride]
    pool = torch.load(a.pool)
    vpd = {k: v for f in a.vpd for k, v in torch.load(f).items()} or None
    ge = goal_eval.GoalEval(native.Native(), chunk=32)
    goal_eval.FLIP_ROWS, goal_eval.CHANGED_ROWS = teacher.PACK_FLIP_ROWS, teacher.PACK_CHANGED_ROWS
    means = None
    for p in todo:
        dest = a.out / f"{p.stem}.json"
        if dest.exists():
            continue
        part = a.out / f"{p.stem}.partial"
        keep = lambda r, part=part: part.write_text(json.dumps(r))  # noqa: E731
        if is_native(p):
            means = means or native_units.unit_means(ge.nat)
            rec = score_native(ge, p, pool, vpd, means, a.pool.name, save=keep)
        else:
            rec = score(ge, p, pool, vpd, a.pool.name, save=keep)
        dest.write_text(json.dumps(rec))
        part.unlink(missing_ok=True)
        print(f"{p.stem}: v {rec['floors']['v']:.3f}, {rec['size']['parts']} parts, {rec['seconds']:.0f} s", flush=True)
    recs = load(a.out)
    (a.out / "summary.json").write_text(json.dumps(summary(recs), indent=1))
    (a.out / "rows.jsonl").write_text("".join(json.dumps(row_line(r)) + "\n" for r in recs))
    print(table(recs))


if __name__ == "__main__":
    main()
