"""One result file per model of lengths 2, 3 and 4 that the line + quadrature estimators cover (2nd argmax lengths
2-4, argmedian length 3 (bit-identical to 2nd argmax), median lengths 2 and 3): results/<task>_<d>_<n>.txt, from
len2_verdicts.tsv, aq3_zoo*.tsv, aq3_seeds.tsv, aq3_median*.tsv, aq4_zoo*.tsv and aq4_seeds*.tsv. M4,3 has its own
fuller file (m43.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m16_report  # noqa: E402
import zoo  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

M2 = ("aq2.py: positive homogeneity leaves only the sign of x0; for each sign the second input enters linearly, so along\n"
      "its line (x1 / |x0| is standard Cauchy) every neuron switches once and the metric integrates in closed form.\n"
      "Two lines give the exact value ('two_lines_exact'; the line x0 > 0 alone is 'one_line'); exact2.py's arc\n"
      "enumeration ('exact_arcs') is the previous exact method.")
M3 = ("aq3.py: exact along each line of the last input (homogeneity removes the radius; the last input enters\n"
      "linearly, so the set where the model is right, or its squared error, has <= d+3 pieces with closed-form ends and\n"
      "closed-form integrals), Gauss-Legendre over the one remaining angle, split where the label jumps (x0 = x1).\n"
      "'aq_one_line' uses one line, 'aq_halves_glm' m nodes per half-circle, 'aq_arcs_glm' m nodes per arc of the\n"
      "first two steps' activation pattern; exact_spherical_polygons is the exact enumeration (exact3.py).")
M4 = ("aq4.py: exact along each line of the last input (lastline.py; the last input over the prefix norm is Student t_3\n"
      "after scaling), Gauss-Legendre product rule over the direction of the first three inputs on the sphere, split\n"
      "into the six lunes between the great circles x_i = x_j where the label jumps ('aq4_lunes_axb': a longitude by\n"
      "b z-nodes per lune; 'aq4_*_centre(s)': only the lune centres). No cone enumeration.")
TASK = {"2nd_argmax": "position of the second largest", "argmedian": "position of the median",
        "median": "median (scored by squared error)"}


def table(W, rows):
    W(f"  {'estimator':<30}{'ops':>11}{'inputs-eq':>11}{'RMS':>12}{'sampling':>12}{'ratio':>8}")
    for r in rows:
        s = r["sampling_rms_same_ops"]
        ratio = r["rms"] / s if s == s and s > 0 else float("nan")
        W(f"  {r['estimator']:<30}{r['ops']:>11.0f}{r['sample_equivalents']:>11.1f}{r['rms']:>12.4g}{s:>12.4g}{ratio:>8.3f}")


def verdict_line(W, label, worst, at, best, failing):
    if worst <= 1.0:
        W(f"  (a) {label}: MET AT EVERY BUDGET (from one sampled input up; worst RMS ratio to sampling {worst:.2f},"
          f" {best} at {at:.0f} ops).")
    else:
        W(f"  (a) {label}: NOT MET at every budget: worst ratio {worst:.2f} ({best} at {at:.0f} ops); failing budgets"
          f" (ops:ratio) {failing}.")


def write(task, d, n, lines):
    m = zoo.load(task, d, n)
    L = [f"AlgZoo {task} hidden={d} seq_len={n}: {m.n_params} parameters, zoo seed {zoo.BEST_SEED[task][(d, n)]}.",
         f"Task: {TASK[task]} of {n} independent standard Gaussian numbers."]
    if task == "argmedian":
        L.append(f"Bit-identical to 2nd_argmax hidden={d} seq_len=3 (the middle of three is the second largest), seeds"
                 " included; all numbers below are that configuration's.")
    L += [""] + lines
    L += ["", "(b) UNDETERMINED: the estimators carry no model-specific advice; no compact hand explanation is scored here.",
          "Protocols: 'noisy' = copies with N(0, 0.05^2) on every raw weight, truth = exact (or 128x128-node aq4) per copy;",
          "'seeds' = the 5 training seeds, no advice (the 0-op guess is the task's value). Ops: scalar arithmetic,",
          "comparisons and elementary functions count 1; one sampled input = forward pass + label + comparison."]
    with open(os.path.join(RES, f"{task}_{d}_{n}.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    return f"{task}_{d}_{n}.txt"


def rd(name):
    p = os.path.join(RES, name)
    return m16_report.read_tsv(p) if os.path.exists(p) else []


def main():
    out = []
    # length 2
    l2 = rd("len2_verdicts.tsv")
    for task in ("2nd_argmax", "median"):
        for d in sorted({int(r["d"]) for r in l2 if r["task"] == task}):
            lines = ["ESTIMATOR", M2, "", "SCORE (one line per protocol; costs in ops)"]
            for r in [r for r in l2 if r["task"] == task and int(r["d"]) == d]:
                lines.append(f"  {r['protocol']}: truth mean {r['truth_mean']:.6g}; 0-op guess RMS {r['guess_rms']:.3g};"
                             f" one_line {r['one_line_ops']:.0f} ops RMS {r['one_line_rms']:.3g}; two_lines_exact"
                             f" {r['two_lines_ops']:.0f} ops RMS {r['two_lines_rms']:.1g}; exact_arcs {r['exact_ops']:.0f}"
                             f" ops; one sampled input {r['sample_cost']:.0f} ops, per-input sd {r['per_input_sd']:.3g}")
                verdict_line(lines.append, r["protocol"], r["worst_ratio"], r["worst_at_ops"], "best estimator",
                             "see len2_verdicts.tsv")
            out.append(write(task, d, 2, lines))
    # length 3 classification
    summ = {int(r["d"]): r for r in rd("aq3_zoo.tsv")}
    curves = rd("aq3_zoo_curves.tsv")
    seeds = rd("aq3_seeds.tsv")
    sv = {}
    if seeds:
        import aq3_seeds
        sv = aq3_seeds.verdicts()
    for d, s in summ.items():
        if d == 4:
            with open(os.path.join(RES, "argmedian_4_3.txt"), "w") as f:
                f.write("AlgZoo argmedian hidden=4 seq_len=3, seed 0: bit-identical to 2nd_argmax hidden=4 seq_len=3 seed 0\n"
                        "(M4,3; the middle of three numbers is the second largest). See 2nd_argmax_4_3.txt.\n")
            continue
        lines = [f"Exact accuracy {s['base_exact']:.8f} (exact3.py).", "", "ESTIMATOR", M3, "",
                 f"NOISY COPIES ({s['noisy_mean']:.4f} +- {s['noisy_sd']:.4f}; per-input sd {s['per_input_sd']:.4f};"
                 f" one sampled input = {s['sample_cost']:.0f} ops)"]
        table(lines.append, [r for r in curves if int(r["d"]) == d])
        if d in sv:
            v = sv[d]
            lines += ["", "SEEDS (exact accuracies " + ", ".join(f"{t:.4f}" for t in v["truth"]) + ")"]
            table(lines.append, v["rows"])
        lines += ["", "SCORE"]
        verdict_line(lines.append, "noisy", s["worst_ratio"], s["worst_at_ops"], s["worst_best"], "")
        if d in sv:
            v = sv[d]
            verdict_line(lines.append, "seeds", v["worst"], v["worst_ops"], v["worst_best"],
                         ";".join(f"{o:.0f}:{q:.2f}" for o, q, _ in v["failing"]))
        out.append(write("2nd_argmax", d, 3, lines))
        if (d, 3) in zoo.BEST_SEED["argmedian"]:
            out.append(write("argmedian", d, 3, lines))
    # length 3 median
    med = rd("aq3_median.tsv")
    medv = rd("aq3_median_verdicts.tsv")
    for d in sorted({int(r["d"]) for r in medv}):
        lines = ["ESTIMATOR", M3]
        for prot in ("noisy_sd0.05", "seeds"):
            R = [r for r in med if int(r["d"]) == d and r["protocol"] == prot]
            if R:
                lines += ["", f"{prot.upper()} (true squared error {R[0]['truth_mean']:.4g} +- {R[0]['truth_sd']:.3g};"
                          f" per-input sd of the squared error {R[0]['sq_err_var'] ** 0.5:.3g})"]
                table(lines.append, R)
        lines += ["", "SCORE"]
        for v in [v for v in medv if int(v["d"]) == d]:
            verdict_line(lines.append, v["protocol"], v["worst_ratio"], v["worst_at_ops"], v["worst_best"], v["failing"])
        out.append(write("median", d, 3, lines))
    # length 4
    summ4 = {int(r["d"]): r for r in rd("aq4_zoo.tsv")}
    curves4 = rd("aq4_zoo_curves.tsv")
    seeds4 = rd("aq4_seeds.tsv")
    sv4 = {int(v["d"]): v for v in rd("aq4_seeds_verdicts.tsv")}
    for d, s in summ4.items():
        lines = [f"Accuracy {s['base_reference']:.8f} (aq4, 128x128 nodes per lune)"
                 + (f"; exact integer cones (exactz.py): {s['base_exactz']:.8f}." if s["base_exactz"] == s["base_exactz"] else "."),
                 "", "ESTIMATOR", M4, "",
                 f"NOISY COPIES ({s['K']:.0f}; accuracy {s['noisy_mean']:.4f} +- {s['noisy_sd']:.4f}; per-input sd"
                 f" {s['per_input_sd']:.4f}; one sampled input = {s['sample_cost']:.0f} ops)"]
        table(lines.append, [r for r in curves4 if int(r["d"]) == d])
        R = [r for r in seeds4 if int(r["d"]) == d]
        if R:
            lines += ["", f"SEEDS (accuracies {R[0]['truth']})"]
            table(lines.append, R)
        lines += ["", "SCORE"]
        verdict_line(lines.append, "noisy", s["worst_ratio"], s["worst_at_ops"], s["worst_best"], "")
        if d in sv4:
            v = sv4[d]
            verdict_line(lines.append, "seeds", v["worst_ratio"], v["worst_at_ops"], v["worst_best"], v["failing"])
        out.append(write("2nd_argmax", d, 4, lines))
    print("\n".join(out))


if __name__ == "__main__":
    main()
