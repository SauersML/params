"""Search-based (teacher) circuits on the toy zoo (TOYLOOP lane), scored against the planted ground truth, and written out
as SFT targets.

On real models the SFT targets come from search because there is no ground truth; on the zoo the truth is known
(zoo.py), so this measures how good those targets are. Per instance and per example text of its record:
  teacher   teacher.teach on the exact decomposition (planted.native_of), two arms (ARMS): f1_start "influential"
            (run_planted's "current" arm: influential-blocks start, block-then-piece pruning) and "both" (teach's default
            F1 search from VPD's answer at the target, then the whole answer); connections, --no-strand, max_nodes None,
            both. The SFT target per text is select()'s pick, made without the ground truth: the smallest arm within the
            teacher's own KL budget, else the lowest KL. Block nodes are expanded to the site's active
            pieces at that position; pieces map to units (a head's q/k/v/o pieces -> L{l}.H{h}, a neuron's c_fc/down
            pieces -> L{l}.N{n}); the circuit's connections map to unit edges (tok:@position -> unit -> OUT).
  greedy    unit-level greedy ablation on the same text: units ordered by their single-removal KL (smallest first),
            each dropped (output zeroed, zoo.ablate) while KL(model || kept units only) <= budget; budgets GREEDY_BUDGETS.
            Also once per instance on INSTANCE_TEXTS ordinary texts (mean KL), the instance-level analogue of the zoo's
            gt_only check (KL 0.05).
Scores against the example's own ground-truth units (the record's examples[].gt_units; bigram_mlp's gt changes per
text) and the record's gt_nodes: unit precision / recall / F1 / exact match, node precision / recall, circuit KL vs
the gt circuit's KL, sizes. Failure modes tagged per example: redundant routes kept (of how many), cancelling-pair
units kept, distractors kept, other extras, missed gt units.

Outputs: OUT/<id>.json per instance (schema in write_instance's docstring), results/zoo_teacher.jsonl (one row per
example, no node lists), results/zoo_teacher.md (summarize).

  python3 -B experiments/oracle/toys/zoo_teacher.py run --shard 0/2 &   python3 -B ... run --shard 1/2
  python3 -B experiments/oracle/toys/zoo_teacher.py summarize
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import native  # noqa: E402
import planted  # noqa: E402
import run_planted as rp  # noqa: E402
import zoo  # noqa: E402

ZOO = Path.home() / "mpd-data/toy_zoo"
OUT = ZOO / "teacher"
RESULTS = HERE / "results"
GREEDY_BUDGETS = (0.05, 0.001)
INSTANCE_TEXTS = 64
TEACH_ARGS = {"B": 16, "edges": True, "no_strand": True, "max_nodes": None, "tokens": None, "accept": (), "accept_mode": "pp"}
ARMS = ("influential", "both")            # teacher.teach's f1_start: run_planted's arm, and teach's default (targets first)
SUFFICIENT = lambda kl, budget: kl is not None and kl <= max(2 * (budget or 0), 1e-7)  # noqa: E731


# ---- conversions

def unit_of(nat, n: str, c: int) -> str | None:
    layer, kind = native._name_layer_kind(n)
    u = int(nat.units[n][c])
    if u < 0:
        return None
    return zoo.head_unit(layer, u) if kind in ("q_proj", "k_proj", "v_proj", "o_proj") else zoo.neuron_unit(layer, u)


def teacher_graph(nat, tx, circ: dict) -> tuple[list, list, list]:
    """(pieces [[site, position, piece]] with blocks expanded to the site's active pieces there, unit edges, token
    positions read) of a teacher answer (goal_eval.to_json's format: nodes [site, piece, position] / ["block", site,
    position] / ["tok", 0, position]; edges [i, j] / [i, "out"] / ["pair", ...])."""
    raw = circ["nodes"]
    label, pieces = {}, []
    for i, (m, c, t) in enumerate(raw):
        if m == "tok":
            label[i] = [f"tok:@{int(t)}"]
            continue
        if m == "block":
            ps = [(c, int(t), int(k)) for k in tx.vpd[c][int(t)].nonzero().flatten().tolist()]
        else:
            ps = [(m, int(t), int(c))]
        pieces += ps
        label[i] = sorted({u for n, tt, k in ps if (u := unit_of(nat, n, k))})
    edges = set()
    for e in circ.get("edges") or []:
        if e[0] == "pair":
            continue
        a, b = e
        for x in label.get(a, []):
            for y in (["OUT"] if b == "out" else label.get(b, [])):
                if x != y and valid_edge(x, y):
                    edges.add((x, y))
    pieces = sorted(set(pieces), key=lambda x: (x[1], x[0], x[2]))
    toks = sorted({int(t) for m, c, t in raw if m == "tok"})
    return [list(p) for p in pieces], sorted(edges, key=_edge_key), toks


def valid_edge(a: str, b: str) -> bool:
    """A unit edge the architecture has: token -> anything; a unit -> OUT; a unit -> a unit of a later layer, or an
    attention head -> an MLP neuron of its own layer. (The teacher's site-level connections, collapsed to units, also
    pair same-layer heads, e.g. one head's key with another's output: not a path.)"""
    if a.startswith("tok:") or b == "OUT":
        return True
    if b.startswith("tok:") or a == "OUT":
        return False
    (la, ka, _), (lb, kb, _) = zoo.parse_unit(a), zoo.parse_unit(b)
    return la < lb or (la == lb and ka == "H" and kb == "N")


def refilter(path: Path) -> int:
    """Apply valid_edge to a teacher file written before the filter (edges and Mermaid); the count dropped."""
    r = json.loads(path.read_text())
    dropped = 0
    for t in [r["teacher"]] + [a for e in r["examples"] for a in e["teacher"].values()]:
        if "unit_edges" not in t:
            continue
        keep = [e for e in t["unit_edges"] if valid_edge(*e)]
        dropped += len(t["unit_edges"]) - len(keep)
        t["unit_edges"], t["mermaid"] = keep, (zoo.mermaid([tuple(e) for e in keep]) if keep else None)
    path.write_text(json.dumps(r))
    return dropped


def _edge_key(e):
    rank = lambda x: (0, int(x[5:]) if x[5:].isdigit() else 0, x) if x.startswith("tok:") else (2, 0, x) if x == "OUT" else (1, 0, x)  # noqa: E731
    return rank(e[0]), rank(e[1])


def units_of_pieces(nat, pieces) -> list:
    return sorted({u for n, t, c in pieces if (u := unit_of(nat, n, c))})


def unit_kl(sd: dict, units_all: list, keep, texts: list, full=None) -> float:
    """KL(model || only `keep` units on) at the last positions, mean over texts."""
    full = zoo.last_logits(sd, texts) if full is None else full
    return zoo.kl(full, zoo.last_logits(zoo.ablate(sd, [u for u in units_all if u not in set(keep)]), texts))


def greedy(sd: dict, units_all: list, texts: list, budget: float) -> tuple[list, float]:
    """Drop units, least important (single-removal KL) first, while KL(model || kept only) <= budget."""
    full = zoo.last_logits(sd, texts)
    single = {u: zoo.kl(full, zoo.last_logits(zoo.ablate(sd, [u]), texts)) for u in units_all}
    removed = []
    for u in sorted(units_all, key=lambda u: (single[u], u)):
        k = zoo.kl(full, zoo.last_logits(zoo.ablate(sd, removed + [u]), texts))
        if k <= budget:
            removed.append(u)
    kept = sorted(u for u in units_all if u not in removed)
    return kept, unit_kl(sd, units_all, kept, texts, full)


def prf(pred, gt) -> dict:
    p, g = set(pred), set(gt)
    tp = len(p & g)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "exact": p == g}


def modes(rec: dict, units: list, gt: list) -> dict:
    """Failure-mode tags of a unit set against the instance's planted structure."""
    p, u, g = rec["params"], set(units), set(gt)
    out = {"missed": sorted(g - u), "distractors_kept": sorted(u & set(rec["distractor_units"]))}
    if rec["family"] == "redundant":
        routes = [zoo.head_unit(*h) for h in p["heads"]]
        out["routes_kept"], out["routes"] = len(u & set(routes)), len(routes)
    if rec["family"] == "cancel":
        out["cancel_pair_kept"] = len(u & set(p["pair"]))
    known = g | set(rec["distractor_units"]) | set(p.get("pair", []))
    out["other_extra"] = sorted(u - known)
    return out


# ---- per instance

def run_instance(rec: dict, log=print) -> dict:
    import goal_eval  # noqa: F401  (teacher's dependency, imported once here)
    import teacher

    d = torch.load(ZOO / rec["weights"] if "weights" in rec else ZOO / "instances" / f"{rec['id']}.pt")
    sd, cfg = d["sd"], d["cfg"]
    nat = planted.native_of(sd, cfg)
    ge = rp.ToyEval(nat, chunk=16)
    far = [(f"far{i}", x) for i, (x, _) in enumerate(zoo.texts(rec["params"], 16, 99))]
    ge.donor_pools.update(search=far[:8], eval=far[8:])
    units_all = zoo.all_units(sd)
    exs = []
    for j, e in enumerate(rec["examples"]):
        ids, targets = e["ids"], [e["target"]]
        gt = sorted({x["unit"] for x in e["gt_units"]})
        row = {"example": j, "kind": e["kind"], "ids": ids, "target": e["target"], "gt_units": gt,
               "gt_nodes": len(e["gt_nodes"]), "gt_node_kl": e["gt_node_kl"], "empty_kl": e["empty_kl"],
               "gt_unit_kl": unit_kl(sd, units_all, gt, [ids])}
        tx = ge.text(ids, targets)
        row["teacher"] = {arm: teach_arm(teacher, ge, nat, sd, units_all, rec, tx, gt, e, arm) for arm in ARMS}
        row["target_arm"] = select(row["teacher"])
        row["greedy"] = {}
        for b in GREEDY_BUDGETS:
            kept, k = greedy(sd, units_all, [ids], b)
            row["greedy"][str(b)] = {"units": kept, "unit_kl": k, "score": prf(kept, gt), "modes": modes(rec, kept, gt)}
        exs.append(row)
    texts = [x for x, _ in zoo.texts(rec["params"], INSTANCE_TEXTS, 4242)]
    inst_greedy = {}
    for b in GREEDY_BUDGETS:
        kept, k = greedy(sd, units_all, texts, b)
        inst_greedy[str(b)] = {"units": kept, "kl": k, "score": prf(kept, rec["gt_units"]), "modes": modes(rec, kept, rec["gt_units"])}
    ok = [x["teacher"][x["target_arm"]] for x in exs if x["target_arm"]]
    t_units = sorted({u for t in ok for u in t["units"]})
    t_edges = sorted({tuple(e) for t in ok for e in _relabel_tok(t["unit_edges"])}, key=_edge_key)
    return {"id": rec["id"], "family": rec["family"], "seed": rec["seed"], "split": rec["split"], "all_units": units_all,
            "gt_units": rec["gt_units"], "distractor_units": rec["distractor_units"],
            "teacher": {"units": t_units, "unit_edges": [list(e) for e in t_edges], "mermaid": zoo.mermaid(t_edges) if t_edges else None,
                        "score": prf(t_units, rec["gt_units"]), "modes": modes(rec, t_units, rec["gt_units"]),
                        "from_examples": len(ok), "arms": {a: sum(x["target_arm"] == a for x in exs) for a in ARMS},
                        "args": {k: v for k, v in TEACH_ARGS.items() if k != "accept"}},
            "greedy_instance": {"texts": INSTANCE_TEXTS, "budgets": inst_greedy}, "examples": exs}


def teach_arm(teacher, ge, nat, sd, units_all, rec, tx, gt, e, arm) -> dict:
    t0 = time.perf_counter()
    ids = tx.ids
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            circ, trow, _ = teacher.teach(ge, ids, tx.targets, f1_start=arm, **TEACH_ARGS)
        pieces, edges, toks = teacher_graph(nat, tx, circ[""])
        units = units_of_pieces(nat, pieces)
        return {"units": units, "unit_edges": [list(x) for x in edges], "mermaid": zoo.mermaid(edges) if edges else None,
                "nodes": pieces, "n_nodes": len(pieces), "tok_positions": toks, "positions": sorted({t for _, t, _ in pieces}),
                "seconds": time.perf_counter() - t0, "node_kl": zoo.node_kl(nat, ids, pieces),
                "unit_kl": unit_kl(sd, units_all, units, [ids]), "budget_kl": trow.get("budget_kl"), "kl_vpd": trow.get("kl_vpd"),
                "blocks_in_answer": sum(1 for x in circ[""]["nodes"] if x[0] == "block"),
                "score": prf(units, gt), "node_score": prf(map(tuple, pieces), map(tuple, e["gt_nodes"])), "modes": modes(rec, units, gt)}
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}", "seconds": time.perf_counter() - t0}


def select(arms: dict) -> str | None:
    """The SFT target among the teacher arms, without the ground truth: the smallest (pieces) of those whose circuit
    keeps KL within the teacher's budget (SUFFICIENT), else the lowest KL; None when every arm failed."""
    ok = {a: t for a, t in arms.items() if "units" in t}
    if not ok:
        return None
    suff = [a for a, t in ok.items() if SUFFICIENT(t["node_kl"], t["budget_kl"])]
    if suff:
        return min(suff, key=lambda a: (ok[a]["n_nodes"], ARMS.index(a)))
    return min(ok, key=lambda a: ok[a]["node_kl"])


def _relabel_tok(edges: list) -> list:
    """Instance-level edges: token positions differ per text, so tok:@p becomes tok:@{p - T} relative to the target
    is not stable either (texts differ in length); token sources are merged into one 'tok' input per reader."""
    return [["tok:input" if a.startswith("tok:") else a, b] for a, b in edges]


def write_instance(res: dict) -> None:
    """OUT/<id>.json:
      id, family, seed, split, all_units (every unit with a nonzero output), gt_units, distractor_units (copied from the
        record, for scoring only: not part of the target);
      teacher: the SFT target at instance level: units (union over the example texts' teacher circuits), unit_edges
        (union; token sources merged to "tok:input"), mermaid (zoo.mermaid's style), score / modes vs gt, args;
      greedy_instance: {texts, budgets: {budget: {units, kl, score, modes}}};
      examples[]: per text, the per-text SFT target: example, kind, ids, target, gt_units, gt_nodes (count), gt_node_kl,
        gt_unit_kl, empty_kl, teacher {units, unit_edges (tok:@position), mermaid, nodes [[site, position, piece]] (blocks
        expanded), n_nodes, positions, tok_positions, node_kl, unit_kl, budget_kl, kl_vpd, blocks_in_answer, seconds,
        score, node_score, modes} (or {error}), greedy {budget: {units, unit_kl, score, modes}}."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{res['id']}.json").write_text(json.dumps(res))


def run(shard: str, only: str | None, redo: bool, order: str, threads: int = 1) -> None:
    torch.set_num_threads(threads)   # 1: the toys' matmuls are tiny; on a shared 4-core box 1 thread ran ~5x faster than 2
    i, k = map(int, shard.split("/"))
    man = [json.loads(x) for x in (ZOO / "manifest.jsonl").read_text().splitlines()]
    pri = {"train": 0, "test_instance": 1, "test_family": 2}
    if order == "split":
        man.sort(key=lambda r: (pri[r["split"]], r["id"]))
    man = [r for n, r in enumerate(man) if n % k == i and (not only or r["family"] in only.split(","))]
    for r in man:
        if not redo and (OUT / f"{r['id']}.json").exists():
            continue
        t0 = time.time()
        rec = json.loads((ZOO / r["record"]).read_text())
        res = run_instance(rec)
        write_instance(res)
        t = res["teacher"]
        print(f"{r['id']} {r['split']}: teacher {t['units']} F1 {t['score']['f1']:.2f} arms {t['arms']} | gt {rec['gt_units']} | "
              f"greedy0.05 {res['greedy_instance']['budgets']['0.05']['units']} | {time.time() - t0:.1f}s", flush=True)


# ---- summary

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def f(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    return f"{x:.{nd}f}" if abs(x) >= 10 ** -nd or x == 0 else f"{x:.1e}"


LEAN = ("units", "n_nodes", "node_kl", "unit_kl", "budget_kl", "blocks_in_answer", "seconds", "score", "node_score", "modes", "error")


def _methods():
    def arm(a):
        return lambda e: e["teacher"][a] if "units" in e["teacher"][a] else None

    def sel(e):
        return e["teacher"][e["target_arm"]] if e["target_arm"] else None

    def gr(b):
        return lambda e: e["greedy"][str(b)]
    return ([(f"teacher {a}", arm(a)) for a in ARMS] + [("teacher selected", sel)]
            + [(f"greedy {b}", gr(b)) for b in GREEDY_BUDGETS])


def summarize() -> None:
    res = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "zoo_teacher.jsonl").open("w") as fh:
        for r in res:
            for e in r["examples"]:
                fh.write(json.dumps({"id": r["id"], "family": r["family"], "split": r["split"],
                                     **{k: v for k, v in e.items() if k not in ("teacher",)},
                                     "teacher": {a: {k: v for k, v in t.items() if k in LEAN} for a, t in e["teacher"].items()}}) + "\n")
    fams = list(zoo.FAMILIES)
    splits = ["train", "test_instance", "test_family"]
    methods = _methods()
    allx = lambda pred: [(r, e) for r in res for e in r["examples"] if pred(r)]  # noqa: E731

    def stats(sel, get):
        ss = [get(e) for _, e in sel]
        sc = [s["score"] for s in ss if s]
        return (len(sc), mean(s["precision"] for s in sc), mean(s["recall"] for s in sc), mean(s["f1"] for s in sc),
                mean(float(s["exact"]) for s in sc))

    n_ex = sum(len(r["examples"]) for r in res)
    L = ["# Teacher circuits on the toy zoo", "",
         f"`experiments/oracle/toys/zoo_teacher.py` (toyloop). {len(res)} instances, {n_ex} example texts (the records' "
         f"examples, ordinary and counterfactual), exact decomposition (planted.native_of). Methods per text:", "",
         "- **teacher influential**: `teacher.teach(f1_start=\"influential\")`, run_planted's \"current\" arm (influential-blocks "
         "start, block then piece pruning to the budget KL_empty^0.1·KL_VPD^0.9, connections, no-strand);",
         "- **teacher both**: `teacher.teach(f1_start=\"both\")`, teach's default F1 search (VPD's answer at the target first, "
         "then the whole answer; top-k; joint path), same budget, connections, no-strand;",
         "- **teacher selected**: per text, the smaller of the two arms among those within budget (KL ≤ max(2·budget, 1e-7)), "
         "else the lower KL: chosen without the ground truth; this is the SFT target written to the teacher files;",
         "- **greedy b**: unit-level greedy ablation (units by single-removal KL, smallest first; drop while "
         "KL(model ‖ kept units only, the rest zeroed) ≤ b).", "",
         "Pieces map to units (a head's q/k/v/o pieces → L·.H·, a neuron's pieces → L·.N·). Scores against the example's "
         "own gt units: precision, recall, F1, exact = identical unit set. Per-instance targets in "
         "`~/mpd-data/toy_zoo/teacher/<id>.json`; per-text rows in `zoo_teacher.jsonl`.", ""]
    L += ["## Unit recovery per split (per text, mean)", "", "| split | method | texts | precision | recall | F1 | exact |",
          "|---|---|---|---|---|---|---|"]
    for s in splits + ["all"]:
        sel = allx(lambda r: s == "all" or r["split"] == s)
        for name, get in methods:
            n, p_, r_, f1, ex = stats(sel, get)
            L.append(f"| {s} | {name} | {n} | {f(p_)} | {f(r_)} | {f(f1)} | {f(ex)} |")
    L += ["", "## Unit recovery per family (per text, mean over all splits): F1 / exact", "",
          "| family | split | texts | " + " | ".join(m for m, _ in methods) + " |", "|---|---|---|" + "---|" * len(methods)]
    for fam in fams:
        sel = allx(lambda r: r["family"] == fam)
        if not sel:
            continue
        sp = "test_family" if fam in zoo.HELD_OUT_FAMILIES else "train+test_inst"
        cells = [stats(sel, g) for _, g in methods]
        L.append(f"| {fam} | {sp} | {len(sel)} | " + " | ".join(f"{f(c[3])} / {f(c[4])}" for c in cells) + " |")
    L += ["", "Precision / recall per family (teacher selected): " + "; ".join(
        f"{fam} {f(stats(allx(lambda r: r['family'] == fam), methods[len(ARMS)][1])[1])}/{f(stats(allx(lambda r: r['family'] == fam), methods[len(ARMS)][1])[2])}"
        for fam in fams if allx(lambda r: r["family"] == fam)) + "."]
    L += ["", "## Instance level vs the record's gt_units", "",
          "Teacher: union of the selected per-text circuits over the record's texts. Greedy: once on 64 ordinary texts (mean KL).", "",
          "| family | instances | teacher F1 | teacher exact | greedy 0.05 F1 | greedy 0.05 exact | greedy 0.001 F1 | greedy 0.001 exact |",
          "|---|---|---|---|---|---|---|---|"]
    for fam in fams + ["all"]:
        rr = [r for r in res if fam == "all" or r["family"] == fam]
        if not rr:
            continue
        g = lambda b, k: mean(float(r["greedy_instance"]["budgets"][str(b)]["score"][k]) for r in rr)  # noqa: E731
        L.append(f"| {fam} | {len(rr)} | {f(mean(r['teacher']['score']['f1'] for r in rr))} | {f(mean(float(r['teacher']['score']['exact']) for r in rr))} | "
                 f"{f(g(0.05, 'f1'))} | {f(g(0.05, 'exact'))} | {f(g(0.001, 'f1'))} | {f(g(0.001, 'exact'))} |")
    L += ["", "## Instance-level edges vs the record's unit_edges", "",
          "Unit → unit and unit → OUT edges (token sources left out: the record names them by role, the teacher by "
          "position). Teacher edges are the selected circuits' connections collapsed to units, after valid_edge.", "",
          "| family | edge precision | edge recall | exact |", "|---|---|---|---|"]
    for fam in fams + ["all"]:
        rr = [r for r in res if fam == "all" or r["family"] == fam]
        if not rr:
            continue
        sc = []
        for r in rr:
            gt_e = {tuple(e) for e in json.loads((ZOO / "instances" / f"{r['id']}.json").read_text())["unit_edges"] if not e[0].startswith("tok:")}
            te = {tuple(e) for e in r["teacher"]["unit_edges"] if not e[0].startswith("tok:")}
            sc.append(prf(te, gt_e))
        L.append(f"| {fam} | {f(mean(x['precision'] for x in sc))} | {f(mean(x['recall'] for x in sc))} | {f(mean(float(x['exact']) for x in sc))} |")
    L += ["", "## KL and size (per text)", "",
          "KL at the target, model ‖ circuit only. Node KL: the circuit's pieces on, every other piece off (zoo.node_kl; the "
          "record's gt_nodes the same way). Unit KL: every unit outside the set zeroed (zoo.ablate). \"suff\" = share of texts "
          "within the teacher's budget. KLs are medians; sizes are means (pieces / units).", "",
          "| family | gt node KL | infl. node KL (suff) | both node KL (suff) | selected node KL | greedy 0.05 unit KL | gt pieces | selected pieces | gt units | infl. units | both units | selected units | greedy 0.05 units | s/text (infl., both) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gi, gb, gs = methods[0][1], methods[1][1], methods[len(ARMS)][1]
    for fam in fams + ["all"]:
        sel = [e for _, e in allx(lambda r: fam == "all" or r["family"] == fam)]
        if not sel:
            continue
        ti, tb, ts = ([g(e) for e in sel if g(e)] for g in (gi, gb, gs))
        suff = lambda tt: mean(float(SUFFICIENT(t["node_kl"], t["budget_kl"])) for t in tt)  # noqa: E731
        L.append(f"| {fam} | {f(med(e['gt_node_kl'] for e in sel))} | {f(med(t['node_kl'] for t in ti))} ({f(suff(ti))}) | "
                 f"{f(med(t['node_kl'] for t in tb))} ({f(suff(tb))}) | {f(med(t['node_kl'] for t in ts))} | "
                 f"{f(med(e['greedy']['0.05']['unit_kl'] for e in sel))} | {f(mean(e['gt_nodes'] for e in sel), 1)} | "
                 f"{f(mean(t['n_nodes'] for t in ts), 1)} | {f(mean(len(e['gt_units']) for e in sel))} | {f(mean(len(t['units']) for t in ti))} | "
                 f"{f(mean(len(t['units']) for t in tb))} | {f(mean(len(t['units']) for t in ts))} | "
                 f"{f(mean(len(e['greedy']['0.05']['units']) for e in sel))} | {f(mean(t['seconds'] for t in ti), 1)}, {f(mean(t['seconds'] for t in tb), 1)} |")
    L += ["", "## Node level (selected teacher pieces vs the record's gt_nodes; per text, mean)", "",
          "| family | node precision | node recall | node exact |", "|---|---|---|---|"]
    for fam in fams:
        tt = [gs(e) for _, e in allx(lambda r: r["family"] == fam) if gs(e)]
        if tt:
            L.append(f"| {fam} | {f(mean(t['node_score']['precision'] for t in tt))} | {f(mean(t['node_score']['recall'] for t in tt))} | "
                     f"{f(mean(float(t['node_score']['exact']) for t in tt))} |")
    L += ["", "## Failure modes (texts)", "",
          "| method | texts | missed ≥1 gt unit | distractor kept | other extra unit | redundant: all routes kept | redundant: one route only | cancel: both of pair kept | cancel: one of pair |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, get in methods:
        ms = [(r["family"], get(e)["modes"]) for r, e in allx(lambda r: True) if get(e)]
        red = [m for fa, m in ms if fa == "redundant"]
        can = [m for fa, m in ms if fa == "cancel"]
        L.append(f"| {name} | {len(ms)} | {sum(bool(m['missed']) for _, m in ms)} | {sum(bool(m['distractors_kept']) for _, m in ms)} | "
                 f"{sum(bool(m['other_extra']) for _, m in ms)} | {sum(m['routes_kept'] == m['routes'] for m in red)}/{len(red)} | "
                 f"{sum(m['routes_kept'] == 1 for m in red)}/{len(red)} | {sum(m['cancel_pair_kept'] == 2 for m in can)}/{len(can)} | "
                 f"{sum(m['cancel_pair_kept'] == 1 for m in can)}/{len(can)} |")
    L += ["", "### Per family, teacher selected (texts with the mode)", "",
          "| family | texts | missed ≥1 | missed units by layer/kind | distractor kept | other extra | arm chosen (infl./both) |", "|---|---|---|---|---|---|---|"]
    for fam in fams:
        ms = [(r, e) for r, e in allx(lambda r: r["family"] == fam) if gs(e)]
        if not ms:
            continue
        roles = defaultdict(int)
        for r, e in ms:
            for u in gs(e)["modes"]["missed"]:
                roles[_role(r, u)] += 1
        top = ", ".join(f"{k} ({v})" for k, v in sorted(roles.items(), key=lambda x: -x[1])[:3]) or "—"
        L.append(f"| {fam} | {len(ms)} | {sum(bool(gs(e)['modes']['missed']) for _, e in ms)} | {top} | "
                 f"{sum(bool(gs(e)['modes']['distractors_kept']) for _, e in ms)} | {sum(bool(gs(e)['modes']['other_extra']) for _, e in ms)} | "
                 f"{sum(e['target_arm'] == 'influential' for _, e in ms)}/{sum(e['target_arm'] == 'both' for _, e in ms)} |")
    errs = [(r["id"], e["example"], a, t["error"]) for r, e in allx(lambda r: True) for a, t in e["teacher"].items() if "error" in t]
    L += ["", f"Teacher errors: {len(errs)}" + ("" if not errs else ": " + "; ".join(f"{a}#{b} {c}: {d}" for a, b, c, d in errs[:5]))]
    (RESULTS / "zoo_teacher.md").write_text("\n".join(L) + "\n" + NOTES)
    print("\n".join(L))


def _role(r: dict, u: str) -> str:
    """A gt unit's role in its family (layer and kind), for the miss table."""
    l, kind, _ = zoo.parse_unit(u)
    return f"L{l} {'head' if kind == 'H' else 'neuron'}"


NOTES = """
## Where search misses known structure

Numbers below are from the run above (1,470 texts, 350 instances).

1. **The influential start misses content-addressed positions.** `block_search` keeps whole blocks at the target plus the
   F2-probe positions, the fewest that reach s_track_pp ≥ 0.9. On **detect** (the head attends to wherever the trigger
   tokens are) it keeps the target position alone in 104/140 texts. That start already sits at the empty-circuit KL
   (median 5.15 nats), so no removal fits the budget and nothing gets pruned: the answer is every piece at the
   target, the two distractor neurons included. The same thing happens, less severely, on **gate** (72/140 insufficient)
   and **additive_vote** (99+7/140). In every text where the influential arm kept a distractor (285), its circuit was
   over budget, apart from 2 multihop texts. Influential is within budget on 80% of texts; the `both` arm is within
   budget on 100%.
2. **Redundant routes: sufficiency keeps one route when the logit is saturated.** With 2 or 3 identical heads and
   p(top) = 1.000000, dropping a route costs KL below the budget (about 1e-9, against budgets of 7e-9 to 1e-8). The
   teacher's smallest-within-budget pick keeps only some routes on 38/140 texts. In the per-text "redundant" column,
   the selected teacher keeps 2 of 3 routes or 1 of 2. The unit-level greedy keeps **exactly one** route on every text
   at b = 0.05 (140/140), and at b = 0.001 on 90/140. The `both` arm keeps all routes on 140/140, but because of its
   start rather than any search for necessity. Without the ground truth, a sufficiency-only search cannot tell
   "redundant" from "unused". Instance-level redundant F1 is 1.00 for the teacher only because the union over texts
   recovers the dropped routes.
3. **Cancelling pairs.** The planted pair (two units writing exactly opposite vectors) is removable only *jointly*.
   The unit-level greedy removes units one at a time, so it keeps both units on 95/140 texts (b = 0.05) and 134/140
   (b = 0.001), and keeps one of the pair, a broken half, on 45 and 6. The teacher mostly gets this right
   (selected: pair kept on 10/140, all of them heads-type pairs) because its first stage drops whole site blocks, and
   removing a site block removes both members at once. The failures are head pairs in the same layer as the copy head,
   where the attention block cannot be dropped whole (influential 28, both 15). Neuron pairs are dropped every time
   by the influential arm; the `both` arm keeps a neuron pair on 9 texts. The opposite error also appears: **additive_vote** misses ≥1 voter on 11 texts, and in
   every case checked the missed voters are a pair that cancels on that text (one votes yes, one votes no) or are
   over-determined (4 yes against 1 no, saturated). On that text the circuit is faithful (node KL ≤ 6e-9), but it is
   not the planted mechanism.
4. **Distractors.** The distractors' writes are DUMP = 0.005 into residual directions nothing reads, which moves the
   KL by about 1e-15. A sufficient search drops them, and the selected teacher keeps a distractor on only 2/1,470
   texts (multihop). Every distractor kept by the influential arm comes from failure mode 1 (an over-budget start
   that cannot be pruned), not from the distractors mattering.
5. **Ground-truth artifact, not a search error: bigram_mlp at instance level.** The record's gt_units is the union
   over its 4 example texts, so it lists only the lookup neurons those texts use. Greedy on 64 texts finds the rest
   of the planted table, scoring "F1 0.71 / exact 0.03". Per text, every method is 1.00 on bigram_mlp except the
   `both` arm, which adds other table neurons (F1 0.85). Instance-level scores for this family should be read
   against the planted table (params.neurons), not gt_units.
6. **Node level.** Selected-teacher pieces are nearly all inside the gt node set (precision 0.86–1.00, except
   multihop 0.55), but recall is 0.22–0.84. The gt node sets list a positional head's key pieces at every position
   and the value at the source, while the teacher keeps only the few keys that set the attention pattern. Both are
   sufficient. The pieces differ, and the units agree.
7. **Edges are over-connected.** Instance-level edge recall is 1.00 outside bigram_mlp, where the gt names only the
   neurons its 4 texts use, so recall there is 0.53. Precision is lower: gate 0.61, multihop 0.66, additive_vote
   0.85. The teacher's connections state direct paths the planted design does not route through, such as a layer-0
   head → OUT next to its route via the layer-1 neurons, or a layer-0 head → the gate's bias neuron. With the exact
   decomposition these paths carry tiny but nonzero signal, and the connection search keeps every connection whose
   removal costs more than the 1e-8 budget. Site-level connections collapsed to units also produced 2,475 same-layer
   head→head edges, which are not paths; `valid_edge` drops them.

## Takeaways for SFT targets

- Use **teacher selected** (written as the target in `~/mpd-data/toy_zoo/teacher/<id>.json`). Per text it has unit
  precision 1.00 and recall 0.99 (exact 0.96, F1 0.99); train split F1 0.99 / exact 0.95; test_family 1.00 / 0.99.
  Its node KL is at gt level (median 7e-10) with 7.5 pieces against gt's 19.7.
- `teach(f1_start="influential")` alone, run_planted's arm, is not safe on the zoo. It is insufficient on 20% of
  texts, and when it fails it fails silently: the answer is still emitted. Any teacher pipeline should check that
  the final circuit's KL is within budget and fall back to the `both` arm.
- The residual errors are structural (redundancy and saturation, text-level cancellation). They come from what
  per-text sufficiency means, not from search noise. Scoring the oracle on multiple texts per instance, or adding a
  necessity term, is what would expose them. An oracle SFT'd on these targets will learn "one route" on saturated
  redundant texts.
- The unit-level greedy is a weak baseline: F1 0.82–0.89 and exact 0.62–0.69. It is too coarse to drop cancelling
  pairs, and at b = 0.05 per text it also drops real voters, because the vote is saturated.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--shard", default="0/1")
    a.add_argument("--only", default=None, help="families, comma-separated")
    a.add_argument("--redo", action="store_true")
    a.add_argument("--order", choices=("split", "manifest"), default="split", help="split: train first")
    a.add_argument("--threads", type=int, default=1)
    sub.add_parser("summarize")
    sub.add_parser("refilter", help="apply valid_edge to the written teacher files")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.shard, args.only, args.redo, args.order, args.threads)
    elif args.cmd == "refilter":
        print(sum(refilter(p) for p in sorted(OUT.glob("*.json"))), "invalid unit edges dropped")
    else:
        summarize()


if __name__ == "__main__":
    main()
