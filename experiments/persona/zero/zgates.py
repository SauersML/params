"""The pre-registered gates of a Zero circuit run (zattr.py, zgen.py and zjudge.py outputs in one directory), all on the
held-out prompts:
  break    mean-ablating the found set raises the judged hard-break rate to >= 0.50, while each of the 6 random sets
           matched to it by kind and layer stays <= 0.10;
  kl       KL from unedited Zero <= 0.1 bits per token (held-out atlas web text; general-chat replies reported too);
  stable   bootstrap stability >= 0.8 (the doublings' overlaps, weighted by the units each added);
  size     every gate at 8, 32, 128 and 512 units.
Rates carry 95% bootstrap intervals clustered by prompt. An empty reply counts as no hard break (as the hand labels
count it; the judge calls most empty replies breaks, and heavy ablation of many heads makes them), and empty replies are
reported per condition. The judge's agreement with hand labels (HANDLABELS.json files
of zjudge-format labels on replies of the run) is reported beside them.

  zgates.py RUN_OUT_DIR [--hand HANDLABELS.json ...] [--out GATES.json]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BOOT = 2000


def ci(groups: list, seed: int = 0) -> dict:
    groups = [g for g in groups if g]
    n = sum(len(g) for g in groups)
    mean = sum(sum(g) for g in groups) / max(1, n)
    rng = random.Random(seed)
    bs = []
    for _ in range(BOOT):
        s = [groups[rng.randrange(len(groups))] for _ in groups]
        bs.append(sum(sum(g) for g in s) / max(1, sum(len(g) for g in s)))
    bs.sort()
    return {"rate": round(mean, 4), "ci": [round(bs[int(0.025 * BOOT)], 4), round(bs[int(0.975 * BOOT)], 4)], "n": n}


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--hand", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts", default=str(Path.home() / "mpd-data/persona/zero/prompts.json"))
    a = ap.parse_args()
    run = Path(a.run)
    attr = json.loads((run / "attr.json").read_text())
    judge = json.loads((run / "judge.json").read_text())["labels"]
    lab = judge["gen_gates"]
    replies = json.loads((run / "gen_gates.json").read_text())["conditions"]
    groups = {x["id"]: x["group"] for x in json.loads(Path(a.prompts).read_text())["heldout"]}
    order = attr["order"]
    out: dict = {"conditions": {}, "gates": {}}
    for cond, by in lab.items():
        hb = [[int(bool(r.strip()) and bool(x and x["hard_break"])) for x, r in zip(v, replies[cond][pid])] for pid, v in by.items()]
        fid = [[(x["fidelity"] if x else 0) / 3 for x in v] for v in by.values()]
        asst = [[(x["assistant"] if x else 0) / 3 for x in v] for v in by.values()]
        r = {"hard_break": ci(hb), "fidelity_over_3": ci(fid), "assistant_over_3": ci(asst),
             "unparsed": sum(x is None for v in by.values() for x in v),
             "empty": sum(not r.strip() for v in replies[cond].values() for r in v)}
        r["hard_break_by_group"] = {g: ci([h for h, pid in zip(hb, by) if groups.get(pid) == g])["rate"]
                                    for g in sorted(set(groups.values()))}
        r.update({k: attr["conditions"].get(cond, {}).get(k) for k in ("search_J", "heldout_J", "G", "kl_general_bits", "kl_atlas_bits")})
        out["conditions"][cond] = r
    out["clean"] = {"search_J": attr["J_clean"], "heldout_J": attr["heldout_J_clean"], "G": attr.get("G_clean")}
    sizes = sorted(int(c.split("_")[1]) for c in lab if c.startswith("found_"))
    rounds = attr["rounds"]
    for s in sizes:
        f = out["conditions"][f"found_{s}"]
        rnd = [out["conditions"][c] for c in lab if c.startswith(f"random_{s}_")]
        upto = [r for r in rounds if r["size_before"] < s]
        stab = sum(r["bootstrap_overlap"] * r["added"] for r in upto) / max(1, sum(r["added"] for r in upto))
        units = order[:s]
        g = {"break_found": f["hard_break"]["rate"], "break_random_max": max((r["hard_break"]["rate"] for r in rnd), default=None),
             "kl_atlas_bits": f["kl_atlas_bits"], "kl_general_bits": f["kl_general_bits"], "stability": round(stab, 3),
             "heads": sum(u[0] == "H" for u in units), "late_share": round(sum(u[1] >= 30 for u in units) / len(units), 3),
             "layers": sorted({u[1] for u in units})}
        g["pass"] = {"break": g["break_found"] >= 0.5 and (g["break_random_max"] is not None and g["break_random_max"] <= 0.10),
                     "kl_atlas": g["kl_atlas_bits"] is not None and g["kl_atlas_bits"] <= 0.1,
                     "kl_general": g["kl_general_bits"] is not None and g["kl_general_bits"] <= 0.1, "stable": stab >= 0.8}
        out["gates"][str(s)] = g
    agree = []
    for h in a.hand:
        hd = json.loads(Path(h).read_text())
        for x in hd["labels"]:
            src = "gen_gates" if "gen_gates" in hd.get("source", "gen_gates") else "gen_zero"
            j = judge[src][x["condition"]][x["id"]][x["j"]]
            agree.append((x["hard_break"], bool(j and j["hard_break"]), x["fidelity"], j["fidelity"] if j else None))
    if agree:
        out["judge_vs_hand"] = {"n": len(agree), "hard_break_agree": sum(a_ == b for a_, b, _, _ in agree) / len(agree),
                                "fidelity_within_1": sum(d is not None and abs(c - d) <= 1 for _, _, c, d in agree) / len(agree),
                                "hand_breaks": sum(a_ for a_, _, _, _ in agree), "judge_breaks": sum(b for _, b, _, _ in agree)}
    print(f"clean: search J {attr['J_clean']:.3f}, held-out J {attr['heldout_J_clean']:.3f}")
    for c in ["clean"] + [f"found_{s}" for s in sizes]:
        r = out["conditions"][c]
        print(f"{c:10s} hard break {r['hard_break']['rate']:.3f} {r['hard_break']['ci']}  fidelity {3 * r['fidelity_over_3']['rate']:.2f}  "
              f"assistant {3 * r['assistant_over_3']['rate']:.2f}  held-out J {r['heldout_J']}  KL atlas {r['kl_atlas_bits']}  general {r['kl_general_bits']}")
    for s, g in out["gates"].items():
        print(f"size {s}: {g}")
    if agree:
        print("judge vs hand:", out["judge_vs_hand"])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
