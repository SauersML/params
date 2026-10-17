"""One line per AlgZoo model (116, plus ARC's challenge model) with ARC's bar (a) graded strictly, at every budget, under
both of ARC's evaluation protocols, and (b). Writes results/zoo_summary.tsv and results/zoo_summary.txt.

(a) grades, per protocol (noisy = 256 or 128 copies with N(0, 0.05^2) on every raw weight; seeds = the 5 training
seeds, estimators without advice):
  EVERY_BUDGET   at every budget from one sampled input up, the best of our deterministic estimators that fits beats
                 random sampling at the same op count (results/*_zoo*.tsv, aq3_seeds.tsv, len2_verdicts.tsv,
                 aq3_median_verdicts.tsv)
  NOT_AT_<ops>   as above except at the listed budgets (ops: worst ratio)
  FROM_COST      an exact enumeration exists (the local agent's results in ~/mpd-data/ami/algzoo), so (a) holds at budgets
                 above its cost; lower budgets not evaluated
  NOT_MET        no mechanistic estimate beats sampling (M16,10 and its seeds; see their result files)
  OPEN           nothing run
(b): MET for M2,2 (ARC's own 40-bit explanation) and median_2_2 (local agent's 83-bit tally); NOT_MET where the
enumeration's sign facts exceed the float32 bits or no compact explanation exists (M16,10); UNDETERMINED otherwise.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m16_report  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DIR = os.path.expanduser("~/mpd-data/ami/algzoo")


def tsv(name):
    p = os.path.join(RES, name)
    return m16_report.read_tsv(p) if os.path.exists(p) else []


def jl(name):
    p = os.path.join(DIR, name)
    out = {}
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            out[(r["task"], r["d"], r["n"])] = r
    return out


def grade(worst, failing):
    if worst <= 1.0:
        return "EVERY_BUDGET"
    return "NOT_AT_" + ";".join(failing)


def main():
    import aq3_seeds
    inv = [line.rstrip("\n").split("\t") for line in list(open(os.path.join(DIR, "inventory.tsv")))[1:]]
    a_noisy, a_seeds, detail = {}, {}, {}
    for r in tsv("aq3_zoo.tsv"):
        for task in ("2nd_argmax", "argmedian"):
            k = (task, int(r["d"]), 3)
            a_noisy[k] = grade(r["worst_ratio"], [f"{r['worst_at_ops']:.0f}:{r['worst_ratio']:.2f}"])
            detail[k] = f"noisy worst ratio {r['worst_ratio']:.2f}"
    if os.path.exists(os.path.join(RES, "aq3_seeds.tsv")):
        for d, v in aq3_seeds.verdicts().items():
            for task in ("2nd_argmax", "argmedian"):
                k = (task, d, 3)
                a_seeds[k] = grade(v["worst"], [f"{o:.0f}:{q:.2f}" for o, q, _ in v["failing"]])
                detail[k] = detail.get(k, "") + f"; seeds worst ratio {v['worst']:.2f}"
    for r in tsv("aq4_zoo.tsv"):
        k = ("2nd_argmax", int(r["d"]), 4)
        a_noisy[k] = grade(r["worst_ratio"], [f"{r['worst_at_ops']:.0f}:{r['worst_ratio']:.2f}"])
        detail[k] = f"noisy worst ratio {r['worst_ratio']:.2f}"
    for name, n, task in (("aq3_median_verdicts.tsv", 3, "median"), ("len2_verdicts.tsv", 2, None),
                          ("aq4_seeds_verdicts.tsv", 4, "2nd_argmax")):
        for r in tsv(name):
            k = (task or str(r["task"]), int(r["d"]), n)
            failing = [x for x in str(r.get("failing", "")).split(";") if x and x != "nan"]
            g = grade(r["worst_ratio"], failing or [f"{r['worst_at_ops']:.0f}:{r['worst_ratio']:.2f}"])
            (a_noisy if str(r["protocol"]).startswith("noisy") else a_seeds)[k] = g
            detail[k] = detail.get(k, "") + f"; {r['protocol']} worst ratio {r['worst_ratio']:.2f}"
    exact = {}
    exact.update(jl("results_exact.jsonl"))
    exact.update(jl("results_exactn.jsonl"))
    cyc = jl("results_cycle.jsonl")
    rows = []
    for p, task, d, n, seed, metric, value, se in inv:
        d, n = int(d), int(n)
        k = (task, d, n)
        an, as_ = a_noisy.get(k), a_seeds.get(k)
        how = "line + quadrature (aq2/aq3/aq4)" if (an or as_) else ""
        if (task, d, n) == ("2nd_argmax", 16, 10):
            an = as_ = "NOT_MET"
            how = "see 2nd_argmax_16_10_seed*.txt"
        if an is None:
            if k in exact or k in cyc:
                an = "FROM_COST"
                r = exact.get(k) or cyc.get(k)
                how = "exact enumeration (local agent)" if k in exact else "exhaustive over n^n inputs (local agent)"
                detail[k] = f"from {r['noisy']['samples_at_equal_ops']:.3g} inputs' cost" if "noisy" in r else ""
            else:
                an = "OPEN"
        if as_ is None:
            as_ = "-"
        if k in (("2nd_argmax", 2, 2), ("median", 2, 2)):
            b = "MET"
        elif (task, d, n) == ("2nd_argmax", 16, 10):
            b = "NOT_MET"
        else:
            r = exact.get(k) or cyc.get(k)
            bits = r.get("surprise_sign_bits", r.get("surprise_bits_enumeration")) if r else None
            b = "NOT_MET" if bits is not None and bits > 32 * int(p) else ("UNDETERMINED" if r or an != "OPEN" else "OPEN")
        f = f"{task}_{d}_{n}.txt" if os.path.exists(os.path.join(RES, f"{task}_{d}_{n}.txt")) else ""
        if (task, d, n) == ("2nd_argmax", 16, 10):
            f = f"2nd_argmax_16_10_seed{seed}.txt"
        rows.append([int(p), task, d, n, int(seed), metric, float(value), an, as_, b, how, detail.get(k, "").strip("; "), f])
    rows.append([432, "example_2nd_argmax", 16, 10, -1, "accuracy", 0.953097, "NOT_MET", "-", "NOT_MET",
                 "see example_2nd_argmax_16_10.txt", "ARC's challenge model", "example_2nd_argmax_16_10.txt"])
    rows.sort(key=lambda r: (r[0], r[1]))
    head = ["params", "task", "hidden", "seq_len", "seed", "metric", "value", "a_noisy", "a_seeds", "b", "estimator",
            "detail", "file"]
    with open(os.path.join(RES, "zoo_summary.tsv"), "w") as fh:
        fh.write("\t".join(head) + "\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r) + "\n")
    from collections import Counter
    cn, cs, cb = Counter(r[7].split("_AT_")[0] for r in rows), Counter(r[8].split("_AT_")[0] for r in rows), Counter(r[9] for r in rows)
    L = ["AlgZoo: ARC's two criteria over all 116 zoo models and the challenge model (results/zoo_summary.tsv).",
         "(a) strict: our estimators beat random sampling at EVERY budget (or only above an exact enumeration's cost).",
         f"  noisy-copies protocol: {dict(cn)}",
         f"  seeds protocol:        {dict(cs)}",
         f"(b): {dict(cb)}",
         "Grades are defined in zoo_summary.py; NOT_AT lists the failing budgets (ops: RMS ratio to sampling).",
         "Every EVERY_BUDGET grade comes from deterministic estimators that integrate exactly along the last input and",
         "use Gauss-Legendre rules over the remaining directions (length 2: two lines, exact; length 3: one angle;",
         "length 4: the 2-sphere in six lunes). The method's cost grows as (number of chambers) x (nodes per",
         "direction)^(n-2); at length 10 the remaining 8 dimensions need a point set and it is no longer mechanistic."]
    with open(os.path.join(RES, "zoo_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
