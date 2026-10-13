"""The mechanism library (#2951 graph oracle, the lead's 10-12): one template per family, validated on held-out texts.

A family's rule (how a text is recognized): its members' current token (the tokens at least two members share, else
the commonest) and, when more than half of the members predict one token, that predicted token (the model's likeliest,
the task file's model_top). Held-out texts: train-split texts with no teacher circuit, outside every copy scan, whose
current token and prediction match the rule (HELD per family, a fixed draw).
A family's template, from its members alone (templates.py's form, connections "all": every connection the model has
among the nodes, every token edge):
  core: the subcomponents at the target that more than half of the members' teacher circuits list (transfer.core);
  carriers: the target's sites where the members list the most subcomponents outside the core (the token-specific part),
     CARRIERS of them as blocks;
  context: the role positions t-k whose token moves the prediction on the most members (the sensitivity probe,
     Text.moved: at least SHARE of a member's total, on at least 2 members; at most CONTEXT of them), every site there
     that can reach the target as a block.
Variants scored: core, core+carriers, core+carriers+context. Baselines per text: VPD's top k at the largest variant's
listed size (a block one node) and at its expanded size, and the text's influential-blocks start (every site that can
reach the target as a block at the target and at its two most influential positions).
Coverage: the held-out split's texts (COVER), each scored with every template whose rule matches it.

  library.py rules              -> DATA/library_texts.json (members, held-out texts, coverage texts per family)
  library.py probe TASK...      -> DATA/probe.jsonl (per text, moved by offset t - p; one process per few texts)
  library.py build              -> MECH/<family>.json
  library.py score TASK FAMILY  -> DATA/library.jsonl (resumable by the caller)
  library.py cover TASK         -> DATA/cover.jsonl: every template's s_point_c and s_track_pp on a coverage text
  library.py report
"""
import argparse
import collections
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

import templates  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
MECH = Path(os.environ.get("MECH", Path.home() / "mpd-data/graph_oracle/mechanisms"))
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
HELD = 6
COVER = range(20, 80)  # the held-out split's text9020-9079 (no search drew on them; far donors are 9150-9199)
SHARE = 0.05
CONTEXT = 2
CARRIERS = 3
GOOD = 0.9



def out_path(stem: str) -> Path:
    """An output file: GEN_OUT (default DATA) / stem + GEN_TAG (a pod worker's own suffix) .jsonl."""
    return Path(os.environ.get("GEN_OUT", DATA)) / f"{stem}{os.environ.get('GEN_TAG', '')}.jsonl"

def task_info(p: Path) -> dict:
    d = json.loads(p.read_text())
    pr = d["prompts"][0]
    t = pr["target_positions"][-1]
    return {"task": p.stem, "split": d["split"], "ids": pr["token_ids"], "t": t, "cur_id": pr["token_ids"][t],
            "pred": pr["model_top"][0][0][0]}


def rules():
    import mech

    tk = mech.tokenizer("vpd4l")
    fams = json.loads((DATA / "families.json").read_text())
    rows = {r["task"]: r for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
    held = json.loads((DATA / "copy_heldout.json").read_text())
    used = set(rows) | {x["task"] for x in held["copy"] + held["noncopy"]} | set(held.get("scanned", []))
    pool = [task_info(p) for p in native_tasks("train") if p.stem not in used]
    cover = [task_info(p) for i, p in enumerate(native_tasks("heldout")) if i in COVER]
    out = {}
    for f, members in fams.items():
        if f == "B:copy":
            continue
        curs = collections.Counter(rows[m]["cur"] for m in members)
        cur = sorted([c for c, n in curs.items() if n >= 2]) or [curs.most_common(1)[0][0]]
        preds = collections.Counter(rows[m]["pred"] for m in members)
        top, n = preds.most_common(1)[0]
        pred = top if n > len(members) / 2 else None
        if f == "B:prep->the":
            cur, pred = sorted({rows[m]["cur"] for m in members}), " the"
        if f == "B:comma->function":
            cur, pred = [","], None

        def match(x):
            return tk.decode([x["cur_id"]]) in cur and (pred is None or x["pred"] == pred)
        cands = [x["task"] for x in pool if match(x)]
        out[f] = {"members": members, "rule": {"current": cur, "predicted": pred},
                  "heldout": sorted(random.Random(f).sample(cands, min(HELD, len(cands))), key=lambda t: int(t[4:])),
                  "candidates": len(cands), "cover": [x["task"] for x in cover if match(x)]}
        print(f, out[f]["rule"], len(cands), "candidates,", len(out[f]["cover"]), "coverage texts")
    out["_cover"] = [x["task"] for x in cover]
    (DATA / "library_texts.json").write_text(json.dumps(out, indent=1))


def native_tasks(split):
    import native

    return native.tasks(split)


def probe(tasks):
    import goal_eval
    import native

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    out = out_path("probe")
    done = {json.loads(line)["task"] for line in out.read_text().splitlines()} if out.exists() else set()
    for task in tasks:
        if task in done:
            continue
        p = TEXTS / f"{task}.json"
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(task), json.loads(p.read_text()))
        moved = np.asarray(tx.moved, dtype=float)
        t = targets[-1]
        by = {int(t - (j + 1)): float(m) for j, m in enumerate(moved) if j + 1 <= t}  # moved[p - 1]: position p
        with out.open("a") as f:
            f.write(json.dumps({"task": task, "t": t, "total": float(moved.sum()), "moved": by}) + "\n")
        ge.probes.clear()
        native.free(ge.nat.dev)
        print(task, flush=True)


def build():
    import native
    import transfer

    C = native.Native("cpu").C
    lib = json.loads((DATA / "library_texts.json").read_text())
    rows = {r["task"]: r for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
    probes = {}
    for p in [DATA / "probe.jsonl"] + templates.run_outputs("probe*.jsonl"):
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                probes.setdefault(r["task"], r)
    context_sites = [n for n in C if int(n.split(".")[1]) < 3 or n.endswith(("k_proj", "v_proj"))]  # those that reach a later position
    MECH.mkdir(parents=True, exist_ok=True)
    for f, info in lib.items():
        if f.startswith("_"):
            continue
        members = [rows[m] for m in info["members"]]
        cn, _, need = transfer.core(members)
        core = [(m, c) for m, c, r in cn if r == "T"]
        per_site = collections.Counter()
        for r in members:
            listed = {(m, int(c)) for m, c, p in transfer.answer(r)["nodes"] if m != "tok" and int(p) == r["t"]}
            per_site.update(m for m, c in listed - set(core))
        carriers = [n for n, _ in per_site.most_common(CARRIERS)]
        offs = collections.Counter()
        for m in info["members"]:
            pr = probes.get(m)
            if not pr or pr["total"] <= 0:
                continue
            offs.update({int(k) for k, v in pr["moved"].items() if int(k) > 0 and v >= SHARE * pr["total"]})
        ctx = [k for k, n in offs.most_common() if n >= 2][:CONTEXT]  # the offsets most members move with (at least 2)
        for variant in ("core", "carriers", "context"):
            B = templates.Builder(f"{f}|{variant}", {"t": "the target", **{f"t-{k}": f"{k} tokens before the target" for k in ctx}},
                                  "all", f"family {f}: rule {info['rule']}")
            for m, c in core:
                B.node(f"{m}#{c}@t", "t", m, c)
            if variant in ("carriers", "context"):
                for n in carriers:
                    B.node(f"{n}@t", "t", n, block=True)
            if variant == "context":
                for k in ctx:
                    for n in context_sites:
                        B.node(f"{n}@t-{k}", f"t-{k}", n, block=True)
            tpl = {**B.t, "family": f, "rule": info["rule"], "core": {"threshold": need, "size": len(core)},
                   "carriers": carriers, "context_offsets": ctx, "members": info["members"]}
            name = f"{safe(f)}__{variant}"
            (MECH / f"{name}.json").write_text(json.dumps(tpl, indent=1))
        print(f, "core", len(core), "carriers", carriers, "context offsets", ctx, dict(offs.most_common(4)))


def safe(f: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in f.replace("\\n", "NL").replace("->", "_to_")).strip("_")


def score(task: str, families: list):
    import goal_eval
    import native
    import cscore
    import copy_template

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    C = ge.nat.C
    tx = cscore.text(ge, task)
    graphs, meta = [], []
    for f in families:
        for variant in ("core", "carriers", "context"):
            tpl = json.loads((MECH / f"{safe(f)}__{variant}.json").read_text())
            g, d = templates.graph(tpl, tx.ids, tx.targets, C)
            graphs.append(g)
            meta.append({"kind": "template", "family": f, "variant": variant, "defects": d})
    big = max(range(len(graphs)), key=lambda i: cscore.node_set_graph(graphs[i], C).count())
    ib, top = copy_template.influential_blocks(ge.nat, tx)
    graphs.append(ib)
    meta.append({"kind": "influential_blocks", "family": None, "variant": f"blocks at t and {top}"})
    listed = graphs[big].count()
    expanded = cscore.node_set_graph(graphs[big], C).count()
    base = ge.baselines(tx, [listed, expanded])
    for k, lab in ((listed, "listed"), (expanded, "expanded")):
        graphs.append(base[f"vpd@{k}"])
        meta.append({"kind": "vpd", "family": None, "variant": f"vpd@{k} ({lab} size of the largest template)"})
    rows = []
    for s0 in range(len(graphs)):  # one at a time: an engine spans its call's columns (a block is a site's every one)
        rows += cscore.score(ge, tx, graphs[s0:s0 + 1], draws=8, near=8, far=4)
        native.free(ge.nat.dev)
    with out_path("library").open("a") as f:
        for m, r in zip(meta, rows):
            f.write(json.dumps({"task": task, **m, **r}) + "\n")
    print(task, [(m.get("family"), m["variant"], r["s_point_c"], r["s_track_pp"]) for m, r in zip(meta, rows)], flush=True)


def cover(task: str):
    """Every family's every template on one coverage text, point (s_point_c) and path-patched tracking (s_track_pp, 8
    draws) alone: whether some template in the library explains the prediction, whatever its rule says."""
    import goal_eval
    import native
    import cscore

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    C = ge.nat.C
    tx = cscore.text(ge, task)
    lib = json.loads((DATA / "library_texts.json").read_text())
    graphs, meta = [], []
    for f in lib:
        if f.startswith("_"):
            continue
        for variant in ("core", "carriers", "context"):
            tpl = json.loads((MECH / f"{safe(f)}__{variant}.json").read_text())
            g, d = templates.graph(tpl, tx.ids, tx.targets, C)
            graphs.append(g)
            meta.append({"family": f, "variant": variant, "matched": task in lib[f]["cover"]})
    import copy_template
    ib, top = copy_template.influential_blocks(ge.nat, tx)
    graphs.append(ib)
    meta.append({"family": None, "variant": f"influential blocks at t and {top}", "matched": False})
    cg = [goal_eval._c_graph(g, tx.targets, C) for g in graphs]
    pp = [goal_eval.strand(g, tx.targets, C) for g in graphs]
    rows = []
    for s0 in range(0, len(graphs), 6):
        pt = ge.sufficiency(tx, cg[s0:s0 + 6])
        tp = ge.track_pp(tx, pp[s0:s0 + 6], "eval", 8)
        rows += [{"s_point_c": a["f1_live"], "s_track_pp": b["s_track_pp"], "nodes": g.count()} for a, b, g in zip(pt, tp, cg[s0:s0 + 6])]
        native.free(ge.nat.dev)
    with out_path("cover").open("a") as f:
        for m, r in zip(meta, rows):
            f.write(json.dumps({"task": task, **m, **{k: (float(v) if v is not None else None) for k, v in r.items()}}) + "\n")
    best = max(rows[:-1], key=lambda r: min(r["s_point_c"] or 0, r["s_track_pp"] or 0))
    print(task, "best", best, flush=True)


def annotate():
    """Each template's validation written into its JSON (MECH): per scored term the mean over the held-out texts its
    family's rule matched (library templates) or the held-out copy texts (copy templates), the texts scored, and
    beside them the same texts' influential-blocks start and VPD's top k at the template's listed size."""
    import copy_template

    terms = ("s_point_c", "s_wiring", "s_track_pp", "s_interchange_pp", "s_interchange", "s_donor_near", "s_donor_far", "c5", "bits")
    mean = lambda rs, m: (float(np.mean([r[m] for r in rs if r.get(m) is not None])) if any(r.get(m) is not None for r in rs) else None)  # noqa: E731
    lib = json.loads((DATA / "library_texts.json").read_text())
    rows = []
    for p in [DATA / "library.jsonl"] + templates.run_outputs("library*.jsonl"):
        if p.exists():
            rows += [json.loads(line) for line in p.read_text().splitlines()]
    for f, info in lib.items():
        if f.startswith("_"):
            continue
        held = set(info["heldout"])
        rs = [r for r in rows if r["task"] in held]
        refs = {"influential_blocks": [r for r in rs if r["kind"] == "influential_blocks"],
                "vpd_listed": [r for r in rs if r["kind"] == "vpd" and "listed" in r["variant"]]}
        for variant in ("core", "carriers", "context"):
            path = MECH / f"{safe(f)}__{variant}.json"
            tpl = json.loads(path.read_text())
            sel = [r for r in rs if r["kind"] == "template" and r.get("family") == f and r["variant"] == variant]
            tpl["validation"] = {"texts": sorted({r["task"] for r in sel}), "split": "train texts without teacher circuits, the rule matched",
                                 "template": {m: mean(sel, m) for m in terms},
                                 **{k: {m: mean(v, m) for m in terms} for k, v in refs.items()}}
            path.write_text(json.dumps(tpl, indent=1))
    cr = []
    for p in [DATA / "copy_template.jsonl"] + templates.run_outputs("copy_template*.jsonl"):
        if p.exists():
            cr += [json.loads(line) for line in p.read_text().splitlines()]
    held = {x["task"] for x in json.loads((DATA / "copy_heldout.json").read_text())["copy"]}
    for name in copy_template.NAMES:
        path = MECH / f"{name}.json"
        tpl = json.loads(path.read_text())
        sel = [r for r in cr if r["task"] in held and r.get("name") == name]
        tasks = {r["task"] for r in sel}
        ib = [r for r in cr if r["task"] in tasks and r["kind"] == "influential_blocks"]
        vl = [r for r in cr if r["task"] in tasks and r["kind"] == "vpd" and sel and
              r["name"] == f"vpd@{next(s_['listed'] for s_ in sel if s_['task'] == r['task'])}"]
        tpl["validation"] = {"texts": sorted(tasks), "split": "held-out copy texts (copy_heldout.json)",
                             "template": {m: mean(sel, m) for m in terms}, "influential_blocks": {m: mean(ib, m) for m in terms},
                             "vpd_listed": {m: mean(vl, m) for m in terms}}
        path.write_text(json.dumps(tpl, indent=1))
    print("annotated", MECH)


def report():
    lib = json.loads((DATA / "library_texts.json").read_text())
    paths = [DATA / "library.jsonl"] + templates.run_outputs("library*.jsonl")
    rows, seen = [], set()
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                key = (r["task"], r["kind"], r.get("family"), r["variant"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    terms = ("s_point_c", "s_track_pp", "s_interchange_pp", "s_donor_near", "s_donor_far", "c5")
    for f, info in lib.items():
        if f.startswith("_"):
            continue
        rs = [r for r in rows if r["task"] in set(info["heldout"])]
        if not rs:
            continue
        print(f"== {f}  rule {info['rule']}  held-out texts {len({r['task'] for r in rs})}")
        big = {r["task"]: r for r in rs if r["kind"] == "template" and r["family"] == f and r["variant"] == "context"}
        for kind, variant in [("template", v) for v in ("core", "carriers", "context")] + [("influential_blocks", None), ("vpd", "listed"), ("vpd", "expanded")]:
            if kind == "vpd":  # VPD's top k at this family's largest template's size on that text
                size = "listed" if variant == "listed" else "nodes"
                sel = [r for r in rs if r["kind"] == "vpd" and r["task"] in big and r["nodes"] == min(big[r["task"]][size], r["nodes"] + 0)
                       and r["variant"].startswith(f"vpd@{big[r['task']][size]} ")]
                sel = list({r["task"]: r for r in sel}.values())
            else:
                sel = [r for r in rs if r["kind"] == kind and (kind != "template" or (r["family"] == f and r["variant"] == variant))
                       and r["task"] in big]
            if sel:
                print(f"   {kind + (':' + variant if variant else ''):24s} " + "  ".join(
                    f"{m} {np.mean([r[m] for r in sel if r[m] is not None]):.2f}" for m in terms if any(r[m] is not None for r in sel))
                    + f"  nodes {np.mean([r['nodes'] for r in sel]):.0f} listed {np.mean([r['listed'] for r in sel]):.0f}")
    cov = []
    for p in [DATA / "cover.jsonl"] + templates.run_outputs("cover*.jsonl"):
        if p.exists():
            cov += [json.loads(line) for line in p.read_text().splitlines()]
    if cov:
        by = collections.defaultdict(dict)
        for r in cov:
            by[r["task"]].setdefault((r["family"], r["variant"]), r)
        n = len(lib["_cover"])
        ok = lambda r: (r["s_point_c"] or 0) >= GOOD and (r["s_track_pp"] or 0) >= GOOD  # noqa: E731
        okp = lambda r: (r["s_point_c"] or 0) >= GOOD  # noqa: E731
        tm = [d for d in by.values()]
        anyb = sum(1 for d in tm if any(ok(r) for k, r in d.items() if k[0] is not None))
        anyp = sum(1 for d in tm if any(okp(r) for k, r in d.items() if k[0] is not None))
        rule = sum(1 for d in tm if any(ok(r) for k, r in d.items() if k[0] is not None and r["matched"]))
        rulep = sum(1 for d in tm if any(okp(r) for k, r in d.items() if k[0] is not None and r["matched"]))
        matched = sum(1 for d in tm if any(r["matched"] for r in d.values()))
        ib = sum(1 for d in tm if any(ok(r) for k, r in d.items() if k[0] is None))
        ibp = sum(1 for d in tm if any(okp(r) for k, r in d.items() if k[0] is None))
        print(f"coverage over {len(tm)} of {n} held-out texts (text9020-9079), at >= {GOOD}:")
        print(f"  a rule matches: {matched}; its template on point {rulep}, on point and pp {rule}")
        print(f"  any template (classification left out): on point {anyp}, on point and pp {anyb}")
        print(f"  the text's own influential-blocks start: on point {ibp}, on point and pp {ib}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("rules", "probe", "build", "score", "cover", "annotate", "report"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "rules":
        rules()
    elif a.cmd == "probe":
        probe(a.args)
    elif a.cmd == "build":
        build()
    elif a.cmd == "score":
        score(a.args[0], a.args[1:])
    elif a.cmd == "cover":
        cover(a.args[0])
    elif a.cmd == "annotate":
        annotate()
    else:
        report()
