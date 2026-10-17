"""One result file per length-3 and length-4 2nd-argmax model (and the bit-identical length-3 argmedian models) from
aq3_zoo.py / aq4_zoo.py: results/<task>_<d>_<n>.txt. M4,3 has its own fuller file (m43.py) and is skipped here."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m16_report  # noqa: E402  (tsv reader)
import zoo  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def write(task, d, n, summ, curve, method):
    L = []
    W = L.append
    p = summ["params"]
    W(f"AlgZoo {task} hidden={d} seq_len={n}: {p:.0f} parameters, zoo seed {summ['seed']:.0f}.")
    if task == "argmedian":
        W(f"Bit-identical to 2nd_argmax hidden={d} seq_len=3 (the middle of three is the second largest); all numbers")
        W("below are that model's.")
    base = summ.get("base_exact", summ.get("base_reference"))
    W(f"Task: position of the {'median' if task == 'argmedian' else 'second largest'} of {n} standard Gaussian numbers."
      f" Accuracy {base:.6f}" + (" (exact, exact3.py)." if n == 3 else
                                 f" (aq4 at 128x128 nodes per lune; exact integer cones: {summ['base_exactz']:.6f})."
                                 if summ["base_exactz"] == summ["base_exactz"] else " (aq4 at 128x128 nodes per lune)."))
    W("")
    W("ESTIMATOR")
    W(method)
    W("")
    W(f"ERROR AGAINST COMPUTE ({summ.get('K', 256):.0f} copies with N(0, 0.05^2) added to every raw weight; copies' accuracy"
      f" {summ['noisy_mean']:.4f} +- {summ['noisy_sd']:.4f};")
    W(f"sampling's per-input sd {summ['per_input_sd']:.4f}; one sampled input = {summ['sample_cost']:.0f} ops)")
    W(f"  {'estimator':<30}{'ops':>11}{'inputs-eq':>11}{'RMS':>10}{'sampling':>10}{'ratio':>8}")
    for r in curve:
        s = r["sampling_rms_same_ops"]
        ratio = r["rms"] / s if s == s else float("nan")
        W(f"  {r['estimator']:<30}{r['ops']:>11.0f}{r['sample_equivalents']:>11.1f}{r['rms']:>10.5f}{s:>10.5f}{ratio:>8.3f}")
    W("")
    W("SCORE")
    if summ["met_every_budget"] in (True, "True", 1.0):
        W("  (a) MET AT EVERY BUDGET: at each budget the best estimator that fits beats sampling at the same ops; the")
        W(f"      worst ratio of RMS errors is {summ['worst_ratio']:.2f} ({summ['worst_best']} at {summ['worst_at_ops']:.0f}"
          " ops). The 0-op constant (the original model's")
        W("      accuracy, advice) covers the budgets below the cheapest weight-reading rule.")
    else:
        W(f"  (a) NOT MET at every budget: worst ratio {summ['worst_ratio']:.2f} ({summ['worst_best']} at"
          f" {summ['worst_at_ops']:.0f} ops).")
    W("  (b) UNDETERMINED: the estimator carries no model-specific advice; no compact hand explanation is scored here.")
    fname = f"{task}_{d}_{n}.txt"
    with open(os.path.join(RES, fname), "w") as f:
        f.write("\n".join(L) + "\n")
    return fname


M3 = ("aq3.py: exact along each line of the last input (homogeneity removes the radius; the last input enters\n"
      "linearly, so the set where the model is right is <= d+3 intervals with closed-form ends and closed-form\n"
      "measure), Gauss-Legendre over the one remaining angle, split where the label jumps (x0 = x1). Deterministic;\n"
      "'aq_one_line' uses one line, 'aq_halves_glm' m nodes per half-circle, 'aq_arcs_glm' m nodes per arc of the\n"
      "first two steps' activation pattern; exact_spherical_polygons is the exact enumeration (exact3.py).")
M4 = ("aq4.py: exact along each line of the last input (lastline.py, Student t_3 for the last input over the prefix\n"
      "norm), Gauss-Legendre product rule over the direction of the first three inputs on the sphere, split into the six\n"
      "lunes between the great circles x_i = x_j where the label jumps ('aq4_lunes_axb': a longitude by b z-nodes per\n"
      "lune). Deterministic; no cone enumeration.")


def main():
    for n, summ_f, curve_f, method in ((3, "aq3_zoo.tsv", "aq3_zoo_curves.tsv", M3), (4, "aq4_zoo.tsv", "aq4_zoo_curves.tsv", M4)):
        sp, cp = os.path.join(RES, summ_f), os.path.join(RES, curve_f)
        if not (os.path.exists(sp) and os.path.exists(cp)):
            continue
        summ = m16_report.read_tsv(sp)
        curves = m16_report.read_tsv(cp)
        for s in summ:
            d = int(s["d"])
            if (d, n) == (4, 3):
                with open(os.path.join(RES, "argmedian_4_3.txt"), "w") as f:
                    f.write("AlgZoo argmedian hidden=4 seq_len=3, seed 0: bit-identical to 2nd_argmax hidden=4 seq_len=3 seed 0\n"
                            "(M4,3; the middle of three numbers is the second largest). See 2nd_argmax_4_3.txt.\n")
                continue
            curve = [r for r in curves if int(r["d"]) == d]
            print(write("2nd_argmax", d, n, s, curve, method))
            if n == 3 and (d, 3) in zoo.BEST_SEED["argmedian"]:
                print(write("argmedian", d, n, s, curve, method))


if __name__ == "__main__":
    main()
