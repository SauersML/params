"""The falsification set over the measured behavior families, beyond copying (#2951 graph oracle; the user, 10-13: copy
is one row): does reward v3 rank each family's mechanism above plausible wrong versions of it, per family, weighted by
how often the family's behavior occurs in text?

Families: the mechanism library's (library_texts.json: S families, the subcomponents shared by texts with one
current-token -> prediction rule, and B families, behaviors such as preposition -> " the"), each with its validated
template mechanisms/<family>__context.json (the core at the target, carrier blocks at the target, every site as a
block at the family's context offsets, every connection among them), on its held-out member texts (the validation
texts and the held-out list; train split only); and copy (the copy columns plus the target's actives, on the
falsification set's 40 copy texts). Frequency: the share of token positions in unused train texts where the family's
rule holds (the current token in its list and, when the rule names one, the model's likeliest token its prediction;
copy: a copy would be right and the model makes it).
Variants (falsify.circuit_variants, the same general ones the final evaluation applies to oracle circuits): (a) the
context moved to random earlier positions, (b) the context at a fixed distance (copy's median, D), (d) irrelevant
parts (the context's copy far away), (g) the context at another family's offsets; and at equal bits VPD's top k and
the target's actives (falsify.equal_bits). Each scored alone by ge.fidelity in reward mode on the eval split (the
CI-pattern pool), every block circuit under falsify.patch_ge's scorecard, and reward v3's circuit term
(goal_eval.circuit_term: phi over the text's families from ge.tau, floors, gate, compactness, tie-break).
Values: E4 (variables.Scorer, eval split) of the first context offset's blocks as a variable with stated value maps,
identity against a random balanced partition and token-id parity of the vocabulary (algorithms vpd4l does not run).

  falsify_families.py build              -> DATA/ffam_texts.json (families, texts, frequencies) and DATA/todo_ffam.tsv
  falsify_families.py score FAMILY TASK  -> DATA/ffam.jsonl (GEN_OUT when set)
  falsify_families.py report
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

import falsify  # noqa: E402

DATA = falsify.DATA
TEXTS = falsify.TEXTS
MECH = Path(os.environ.get("MECH", Path.home() / "mpd-data/graph_oracle/mechanisms"))
PER_FAMILY = 8
SCAN = 400  # train texts scanned for the frequencies


def safe(f: str) -> str:
    import library

    return library.safe(f)


def build():
    import native

    lib = json.loads((DATA / "library_texts.json").read_text())
    fal = json.loads((DATA / "falsify_texts.json").read_text())
    used = set(json.loads((DATA / "huang_used.json").read_text()))
    fams = {}
    for f, v in lib.items():
        if f.startswith("_") or not (MECH / f"{safe(f)}__context.json").exists():
            continue
        tpl = json.loads((MECH / f"{safe(f)}__context.json").read_text())
        texts = list(dict.fromkeys((tpl.get("validation") or {}).get("texts", []) + v.get("heldout", [])))[:PER_FAMILY]
        if texts:
            fams[f] = {"template": f"{safe(f)}__context.json", "rule": v["rule"], "texts": texts}
    fams["copy"] = {"template": "copy_columns_all.json", "rule": None, "texts": [e["task"] for e in fal["texts"]]}
    # frequencies: the share of positions in unused train texts where each rule holds
    nat = native.Native()
    tk = __import__("mech").tokenizer("vpd4l")
    ids_of = lambda s: tk.encode(s, add_special_tokens=False).ids  # noqa: E731
    rules = {f: ({i for s in v["rule"]["current"] for i in ids_of(s)[-1:]},
                 (ids_of(v["rule"]["predicted"])[:1] or [None])[0] if v["rule"].get("predicted") else None)
             for f, v in fams.items() if v["rule"]}
    hits, total, scanned = collections.Counter(), 0, 0
    for p in native.tasks("train"):
        if scanned >= SCAN:
            break
        if p.stem in used:
            continue
        scanned += 1
        ids, _ = native.text(p)
        pos = list(range(1, len(ids) - 1))
        top = nat.reference([ids], pos)[0].argmax(-1).tolist()
        total += len(pos)
        for t, y in zip(pos, top):
            for f, (cur, pred) in rules.items():
                if ids[t] in cur and (pred is None or y == pred):
                    hits[f] += 1
            occ = [j for j in range(t) if ids[j] == ids[t]]
            if occ and occ[-1] + 1 < t and ids[occ[-1] + 1] == ids[t + 1] and y == ids[t + 1]:
                hits["copy"] += 1
        native.free(nat.dev)
    for f in fams:
        fams[f]["frequency"] = hits[f] / max(total, 1)
    (DATA / "ffam_texts.json").write_text(json.dumps({"families": fams, "scanned_texts": scanned, "positions": total}, indent=1))
    with (DATA / "todo_ffam.tsv").open("w") as f:
        for fam, v in fams.items():
            for task in v["texts"]:
                f.write(f"falsify_families.py\tscore\t{fam}\t{task}\n")
    for f, v in sorted(fams.items(), key=lambda kv: -kv[1]["frequency"]):
        print(f"{f:26s} texts {len(v['texts']):2d}  frequency {v['frequency']:.4f}")


def template_answer(fam: str, info: dict, tx, nat) -> dict:
    """The family's template on the text as a complete answer (goal_eval.circuit's JSON); copy's with the target's
    actives."""
    import templates

    tpl = json.loads((MECH / info["template"]).read_text())
    t = tx.targets[-1]
    pos = templates.role_positions(list(tx.ids), t)
    nodes = []
    for nd in tpl["nodes"]:
        if nd["role"] not in pos or nd.get("tok"):
            continue
        p = pos[nd["role"]]
        nodes.append(["block", nd["site"], p] if nd.get("block") else [nd["site"], int(nd["index"]), p])
    if fam == "copy":
        nodes += [[n, int(c), t] for n in nat.names for c in tx.vpd[n][t].nonzero().flatten().tolist()]
    seen, out = set(), []
    for nd in nodes:
        if tuple(nd) not in seen:
            seen.add(tuple(nd))
            out.append(nd)
    return {"nodes": out, "edges": [], "complete": True}


def score(fam: str, task: str):
    import faulthandler
    import signal

    import goal_eval
    import native
    import torch
    import variables

    faulthandler.register(signal.SIGUSR1)
    info = json.loads((DATA / "ffam_texts.json").read_text())["families"]
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    goal_eval.FLIP_ROWS = int(os.environ.get("FLIP_ROWS", 16))
    falsify.patch_ge(ge)
    t0 = time.perf_counter()
    taskj = json.loads((TEXTS / f"{task}.json").read_text())
    pr = taskj["prompts"][0]
    t = pr["target_positions"][-1]
    tx = ge.text(pr["token_ids"][:t + 1], [t], task=taskj)
    answer = template_answer(fam, info[fam], tx, nat)
    # other families' context offsets (the mismatched variant) and copy's median distance (the fixed one)
    others = []
    for f, v in info.items():
        if f != fam and f != "copy":
            tp = json.loads((MECH / v["template"]).read_text())
            offs = sorted({int(r[2:]) for r in tp["roles"] if r.startswith("t-")})
            if offs:
                others.append(offs)
    D = json.loads((DATA / "falsify_texts.json").read_text())["D"]
    answers = {"correct": answer, **falsify.circuit_variants(answer, t, tx.T, (fam, task), others, D)}
    graphs = {v: goal_eval.circuit(a, tx.targets, nat.C, tx.T)[0] for v, a in answers.items()}
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    bits = {v: goal_eval.size_bits(g, P, S, tx.targets, active=tx.vpd) for v, g in graphs.items()}
    for kind in ("vpd", "recency"):
        g, _ = falsify.equal_bits(ge, tx, bits["correct"], kind)
        graphs[f"ref_{kind}_equal_bits"] = g
        bits[f"ref_{kind}_equal_bits"] = goal_eval.size_bits(g, P, S, tx.targets, active=tx.vpd)
    pool = torch.load(falsify.POOL, weights_only=False) if falsify.POOL.exists() else None
    tau = ge.tau(tx, pool, "eval")
    print(fam, task, "built, tau", round(time.perf_counter() - t0), "s", flush=True)
    rows = []
    for v, g in graphs.items():
        rows += ge.fidelity(tx, [g], pool=pool, mode="reward", split="eval", draws=16, interchange=8, changed=8,
                            near_donors=8, far_donors=4)
        native.free(nat.dev)
        print(fam, task, v, "scored", round(time.perf_counter() - t0), "s", flush=True)
    # the context variable's value maps (E4 v2): identity against algorithms vpd4l does not run
    vals = {}
    ctx = sorted({int(p) for _, _, p in answer["nodes"] if 0 < int(p) != t}, key=lambda p: t - p)  # a token at 0 has no prediction to draw from
    if ctx:
        c0 = ctx[0]
        sc = variables.Scorer(ge)
        V = nat.target.wte.shape[0]
        part = falsify.rng_of("partition", fam, task).integers(2, size=V)
        za, own = ge._fresh_sources(tx, c0, int(tx.seed) + 41), tx.ids[c0]
        maps = {"identity": [[own]] + [[z] for z in za],
                "random_partition": [[own] + [z for z in za if part[z] == part[own]], [z for z in za if part[z] != part[own]]],
                "parity": [[own] + [z for z in za if z % 2 == own % 2], [z for z in za if z % 2 != own % 2]]}
        group = [["block", nd[1], c0] for nd in answer["nodes"] if nd[0] == "block" and int(nd[2]) == c0]
        for name, classes in maps.items():
            var = {"name": "context token", "meaning": f"the token at t-{t - c0}, {name.replace('_', ' ')}", "inputs": [c0],
                   "group": group, "feeds": ["prediction"],
                   "values": [{"position": c0, "classes": [{"token_ids": [int(z) for z in c]} for c in classes]}]}
            res = sc.score(tx, variables.parse([var], nat.C), "eval", 8)
            vals[name] = {m: res.get(m) for m in ("e4", "f_on", "f_off", "f_same")}
    out = falsify.out_path("ffam")
    with out.open("a") as f:
        for (v, g), r in zip(graphs.items(), rows):
            ct = goal_eval.circuit_term(r, tau, bits[v])
            f.write(json.dumps({"family": fam, "task": task, "t": t, "variant": v, "R": ct["term"], "phi": ct["phi_full"],
                                "phi_tau": tau["phi_tau"], "v": ct["v"], "g": ct["g"], "compact": ct["compact"],
                                "bits": bits[v], "nodes": g.count(),
                                **{f"s_{fm}": goal_eval.family_score(r, fm) for fm in goal_eval.PHI_FAMILIES},
                                "values": vals if v == "correct" else None}) + "\n")
    print(fam, task, {v: round(goal_eval.circuit_term(r, tau, bits[v])["term"], 2) for v, r in zip(graphs, rows)}, flush=True)


def report():
    import templates

    info = json.loads((DATA / "ffam_texts.json").read_text())["families"]
    rows = []
    for p in [DATA / "ffam.jsonl"] + templates.run_outputs("ffam*.jsonl"):
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines()]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["family"], r["task"])].setdefault(r["variant"], r)
    pairs = [("correct", "a_source_moved"), ("correct", "b_fixed_offset"), ("correct", "g_mismatched"),
             ("correct", "d_irrelevant_parts"), ("correct", "ref_vpd_equal_bits"), ("correct", "ref_recency_equal_bits")]
    terms = ["R", "phi", "s_point", "s_flips", "s_track_pp", "s_interchange_pp", "s_nodes", "s_N", "s_pattern"]
    win = lambda d, a, b, m: (None if a not in d or b not in d or d[a].get(m) is None or d[b].get(m) is None else  # noqa: E731
                              1.0 if d[a][m] > d[b][m] + 1e-9 else 0.5 if abs(d[a][m] - d[b][m]) <= 1e-9 else 0.0)
    fams = sorted({k[0] for k in by}, key=lambda f: -info[f]["frequency"])
    print("Share of texts where reward v3's circuit term R ranks the family's mechanism above each wrong one (ties half)")
    print(f"{'family':24s} {'freq':>7s} {'n':>3s}" + "".join(f"{b.split('_', 1)[1][:13]:>15s}" for _, b in pairs)
          + f"{'E4 id/part/par':>18s}")
    agg = collections.defaultdict(list)
    for f in fams:
        ds = [d for k, d in by.items() if k[0] == f]
        line = []
        for a, b in pairs:
            ws = [w for d in ds if (w := win(d, a, b, "R")) is not None]
            line.append(f"{np.mean(ws):15.2f}" if ws else f"{'-':>15s}")
            if ws:
                agg[b].append((info[f]["frequency"], np.mean(ws)))
        e4 = [d["correct"]["values"] for d in ds if "correct" in d and d["correct"].get("values")]
        e4s = "/".join(f"{np.mean([x[m]['e4'] for x in e4 if x.get(m) and x[m]['e4'] is not None]):.2f}"
                       for m in ("identity", "random_partition", "parity")) if e4 else "-"
        print(f"{f:24s} {info[f]['frequency']:7.4f} {len(ds):3d}" + "".join(line) + f"{e4s:>18s}")
    print(f"{'weighted by frequency':24s} {'':7s} {'':3s}" + "".join(
        f"{np.average([s for _, s in agg[b]], weights=[w for w, _ in agg[b]]):15.2f}" if agg[b] and sum(w for w, _ in agg[b]) > 0
        else f"{'-':>15s}" for _, b in pairs))
    print("\nPer term, over every family's texts (unweighted): share where correct beats each wrong one")
    print(f"{'':28s}" + "".join(f"{m[:12]:>13s}" for m in terms))
    for a, b in pairs:
        print(f"{b:28s}" + "".join(f"{np.mean(ws):13.2f}" if (ws := [w for d in by.values() if (w := win(d, a, b, m)) is not None])
                                   else f"{'-':>13s}" for m in terms))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "score", "report"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    {"build": build, "score": lambda: score(*a.args), "report": report}[a.cmd]()
