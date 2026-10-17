"""The task x model table: our circuits under the circuit-sparsity criterion beside their released circuits.

For each run (prune.py's <model>__<task>.json): nodes and edges of our circuit; on held-out pairs (files never searched)
the circuit's per-sequence task loss with a 90% bootstrap interval over pairs, whether it meets the target, the full
model's loss, the loss with the circuit's own nodes ablated, and the mean loss of random circuits of the same size per
site. Their columns: csp_yolo2's released circuit (prune_v4, target 0.15) nodes and edges, and csp_sweep1_4x's
(prune_v3, target 0.05) nodes.

python table.py RUN_DIR [RUN_DIR ...] [--out TSV]
"""
import argparse
import glob
import json
import math
import os
import random

import tasks as T

ROOT = os.path.expanduser("~/mpd-data/ami/sparsecmp")


def released():
    out = {}
    for name, sweep in (("csp_yolo2", "prune_v4"), ("csp_sweep1_4x_0.9Mnonzero_afrac0.125", "prune_v3")):
        p = f"{ROOT}/released/prunes_{name}_{sweep}.json"
        if not os.path.exists(p):
            continue
        for t, r in json.load(open(p)).items():
            if "runs" not in r:
                continue
            best = r["runs"][r["best"]]
            task = t.replace("_beeg3", "").replace("_beeg2", "").replace("_beeg", "")
            out[(name, task)] = {"k": best["k"], "edges": best["last"].get("final_bisect_edges")}
    return out


def boot(v, n=2000, seed=0):
    rng = random.Random(seed)
    m = [sum(rng.choices(v, k=len(v))) / len(v) for _ in range(n)]
    m.sort()
    return m[int(0.05 * n)], m[int(0.95 * n)]


def geo(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rel = released()
    rows = []
    for d in a.dirs:
        for p in sorted(glob.glob(f"{d}/*.json")):
            r = json.load(open(p))
            t = r.get("test")
            row = {"model": r["model"], "task": r["task"], "verified": r["task"] in T.VERIFIED, "target": r["target"],
                   "n_test": r["n_test"], "full_search": round(r["full_search"], 4), "k": r.get("k"),
                   "edges": (r.get("edges") or {}).get("weights"), "seconds": round(r["seconds"])}
            if t:
                lo, hi = boot(t["pairs"]["circuit"])
                row.update({"circuit": round(t["circuit"], 4), "ci90": f"[{lo:.3f}, {hi:.3f}]",
                            "meets": t["circuit"] <= r["target"], "full": round(t["full"], 4),
                            "ablated": round(t["ablated"], 4),
                            "random": round(sum(t["random"]) / len(t["random"]), 4)})
            for name, tag in (("csp_yolo2", "yolo2"), ("csp_sweep1_4x_0.9Mnonzero_afrac0.125", "sweep4x")):
                x = rel.get((name, r["task"]))
                if x:
                    row[f"{tag}_k"] = x["k"]
                    if x["edges"]:
                        row[f"{tag}_edges"] = x["edges"]
            rows.append(row)
    cols = ["model", "task", "verified", "target", "n_test", "full_search", "k", "edges", "circuit", "ci90", "meets",
            "full", "ablated", "random", "yolo2_k", "yolo2_edges", "sweep4x_k", "seconds"]
    lines = ["\t".join(cols)] + ["\t".join(str(r.get(c, "")) for c in cols) for r in rows]
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    for m, rs in by.items():
        ver = [r for r in rs if r["verified"] and r.get("k")]
        lines.append(f"# {m}: verified tasks {len(ver)}, geometric mean nodes {geo([r['k'] for r in ver])}, "
                     f"edges {geo([r['edges'] for r in ver])}, held-out meets {sum(1 for r in ver if r.get('meets'))}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
