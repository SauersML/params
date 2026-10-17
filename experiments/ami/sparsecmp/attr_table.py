"""attr.py's records as one table per model: every method's circuit size at the criterion, its held-out scores, and its
overlap with Gao's released circuit; then each method's summary over the tasks (geometric mean size, held-out
meet rate, mean precision and recall with the chance level).

python attr_table.py ATTR_DIR [ATTR_DIR ...] [--ours RUN_DIR ...]
"""
import argparse
import glob
import json
import math
import os

import attr as A


def geo(v):
    v = [x for x in v if x]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ours", nargs="*", default=[], help="prune.py run dirs (records for the 'ours' arm)")
    a = ap.parse_args()
    recs = {}
    for d in a.dirs:
        for p in glob.glob(f"{d}/*.json"):
            r = json.load(open(p))
            recs[(r["model"], r["task"])] = r
    for d in a.ours:                                 # our search's records written after attr.py ran
        for p in glob.glob(f"{d}/*.json"):
            o = json.load(open(p))
            key = (o["model"], o["task"])
            if key in recs and "ours" not in recs[key]["arms"] and o.get("keep"):
                rel = A.released_nodes(o["model"], o["task"])
                sizes = {s: 10 ** 9 for s in o["keep"]}
                m = type("M", (), {"sites": sizes})()
                ov = A.overlap(m, o["keep"], rel) if rel else {}
                ov.pop("chance_hits", None)          # needs the true site sizes; filled from attr's arms if any
                recs[key]["arms"]["ours"] = {"pruned": {"k": o["k"], "edges": o["edges"]["weights"], "test": o["test"],
                                                        "overlap": ov}}
    models = sorted({m for m, _ in recs})
    for model in models:
        print(f"== {model}")
        print("task\tarm\tk_prefix\tk_pruned\tedges\theldout_loss\tmeets\tkl2\trandom\tprecision\trecall\tchance_prec\trecall@R")
        summ = {}
        for (m, t), r in sorted(recs.items()):
            if m != model:
                continue
            for name, arm in r["arms"].items():
                for stage in ("prefix", "pruned"):
                    c = arm.get(stage)
                    if not c or (name == "ours" and stage == "prefix"):
                        continue
                    te, ov = c["test"], c.get("overlap") or {}
                    chance = ov.get("chance_hits", float("nan")) / ov["size"] if ov.get("size") and "chance_hits" in ov else float("nan")
                    label = name if name == "ours" else f"{name}/{stage}"
                    s = summ.setdefault(label, {"k": [], "e": [], "meet": [], "prec": [], "rec": [], "chance": [], "rR": []})
                    s["k"].append(c["k"]); s["e"].append(c["edges"]); s["meet"].append(te["circuit"] <= r["target"])
                    if ov:
                        s["prec"].append(ov["precision"]); s["rec"].append(ov["recall"])
                        if chance == chance:
                            s["chance"].append(chance)
                    if stage == "prefix" and "recall_at_released_size" in arm:
                        s["rR"].append(arm["recall_at_released_size"])
                    print(f"{t}\t{label}\t{arm.get('k_prefix', '')}\t{c['k']}\t{c['edges']}\t{te['circuit']:.3f}\t"
                          f"{te['circuit'] <= r['target']}\t{te.get('kl2', float('nan')):.3f}\t"
                          f"{sum(te['random']) / len(te['random']):.3f}\t{ov.get('precision', float('nan')):.3f}\t"
                          f"{ov.get('recall', float('nan')):.3f}\t{chance:.4f}\t{arm.get('recall_at_released_size', float('nan')):.3f}")
                if not arm.get("prefix") and name != "ours":
                    print(f"{t}\t{name}\tno passing prefix")
        print("-- summary: arm\ttasks\tgeo_nodes\tgeo_edges\tmeets\tmean_prec\tmean_recall\tmean_chance_prec\tmean_recall@R")
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        for label, s in sorted(summ.items()):
            print(f"{label}\t{len(s['k'])}\t{geo(s['k']):.0f}\t{geo(s['e']):.0f}\t{sum(s['meet'])}/{len(s['meet'])}\t"
                  f"{mean(s['prec']):.3f}\t{mean(s['rec']):.3f}\t{mean(s['chance']):.4f}\t{mean(s['rR']):.3f}")


if __name__ == "__main__":
    main()
