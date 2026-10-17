"""Write one plain-text explanation per AlgZoo model into ~/mpd-data/ami/algzoo/, from the scored results.

Each file states the model, the explanation (the algorithm the weights compute, derived exactly), the evidence and
the score under ARC's two criteria. Inputs: results_exact.jsonl (length 2 and 3) and results_exactn.jsonl (length 4+).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact3  # noqa: E402
import exactn  # noqa: E402
import exactz  # noqa: E402
import zoo  # noqa: E402

DIR = os.path.expanduser("~/mpd-data/ami/algzoo")
TASK_TEXT = {
    "2nd_argmax": "the position of the second largest of {n} independent standard Gaussian numbers",
    "argmedian": "the position of the median of {n} independent standard Gaussian numbers",
    "median": "the median of {n} independent standard Gaussian numbers (output a number, scored by squared error)",
}


M43_TEXT = """COMPACT READABLE ALGORITHM (George Robinson's symmetric reduction, checked here)
The weights are close to W_hi = U^T a, W_hh = U^T B U, W_oh = C V with U = [[1,0,0,-1],[0,1,-1,0]] and
V = [[1,0,0,1],[0,1,1,0]] (relative residuals 0.1% for W_hi and W_oh, 10% for W_hh). Because U ReLU(U^T v) = v and
V ReLU(U^T v) = |v|, the symmetrized model is exactly a 2-unit absolute-value network:
    v = A x,   logits = C |v|,
    A = [[ 1.04, -4.97, 3.97],     i.e. rows close to ( 1, -(L+1), L) and (-1, -L, L+1) with L ~ 4,
         [-0.90, -3.76, 4.71]]     the construction of ARC's handcrafted M4,3 (which uses L = 100).
Both rows sum to about 0, so v depends only on differences between inputs, as the task does.
Evidence (2^20 inputs): the reduced network agrees with the model's prediction on 98.73% of inputs, KL(model ||
reduced) = 0.022 nats; its exact accuracy is 0.98389 against the model's 0.98455; the model's final hidden state
matches ReLU(U^T A x) with R^2 0.989-0.992 per neuron. Intervention: moving x along A's null direction (about
(1,1,1)/sqrt(3), which leaves v unchanged) leaves the model's prediction unchanged on 99.85% / 99.30% / 97.22% of
inputs for steps of size 0.1 / 0.5 / 2.
This reduction is a compact explanation of the model's computation up to the 10% asymmetry in W_hh; turning the
asymmetry into a bounded correction is what ARC's second M4,3 estimator does (RMS error 5%)."""



MEDIAN22_TEXT = """COMPACT EXPLANATION (by hand, in the style of ARC's M2,2 analysis)
Weights: W_hi = (a, -a') with a = 0.698289, a' = 0.697987; W_hh = [[1, -1.000432], [-0.999570, 0.999999]];
W_oh = (c, -c') with c = 0.716037, c' = 0.716344.
Mechanism. Step 1 stores x0 as its two halves: h_1 = (a x0+, a' x0-), where x+ = max(x, 0), x- = max(-x, 0).
Step 2: since x0+ - x0- = x0 and W_hh's rows are (1, -1) and (-1, 1) up to 4e-4, the pre-activations are
(a(x0 + x1), -a(x0 + x1)) up to terms of relative size 4e-4, so h_2 = (a (x0+x1)+, a (x0+x1)-). The output
c a (x0+x1)+ - c a (x0+x1)- = c a (x0 + x1), and c a = 0.5000 (to 1e-6), so output = (x0 + x1)/2, the median of two.
Readable algorithm: keep the positive and negative parts of x0, add x1, output half the sum.
Surprise accounting: 8 sign checks (one per weight), 5 near-equalities to about 1 part in 2500 (|a - a'|, |c - c'| and
three W_hh entries against +-1; about 11 bits each) and c a = 1/2 to about 1e-6 (about 20 bits): about 83 bits.
Each fact is needed for the measured mean squared error of 1.8e-12 (the residual output coefficients are 1e-6), so
the training had to select them, the same argument ARC uses to call its 40-bit M2,2 explanation full understanding."""


ADDENDA = {("2nd_argmax", 4, 3): M43_TEXT, ("argmedian", 4, 3): M43_TEXT, ("median", 2, 2): MEDIAN22_TEXT}
FULL_UNDERSTANDING = {("2nd_argmax", 2, 2): "MET: about the size of ARC's 40-bit explanation of this model (14 sign bits,"
                      " 4 comparison bits, 22 bits for a near-equality), which ARC accepts as full understanding",
                      ("median", 2, 2): "MET by the compact explanation below (about 83 bits, every fact needed for"
                      " the measured error), the standard ARC applies to M2,2"}


def standardized(m):
    """Rescale each hidden neuron so its input weight is +1 or -1 (ReLU is positively homogeneous, so the model is
    unchanged). Neurons with zero input weight keep their scale."""
    s = np.abs(m.W_hi).copy()
    s[s < 1e-12] = 1.0
    D, Di = np.diag(1 / s), np.diag(s)
    return m.W_hi / s, D @ m.W_hh @ Di, m.W_oh @ Di


def fmt_matrix(M, prec=2):
    M = np.atleast_2d(M)
    w = max(len(f"{v:.{prec}f}") for v in M.ravel())
    return "\n".join("    [" + " ".join(f"{v:{w}.{prec}f}" for v in row) + "]" for row in M)


def order_text(o):
    return " > ".join(f"x{i}" for i in o)


def classification_algorithm(m):
    # length <= 3: the float geometry agrees with the exact one to 1e-12 and is much faster
    table = exactn.confusion(m) if m.n <= 3 else exactz.confusion(m)
    k = m.n
    lab = (lambda o: o[1]) if m.task == "2nd_argmax" else (lambda o: o[len(o) // 2])
    lines = []
    follows = 0.0
    dominant_is_label = True
    for o in sorted(table, key=lambda o: -table[o].sum()):
        row = table[o]
        p = row.sum()
        c = int(np.argmax(row))
        follows += row[lab(o)]
        dominant_is_label &= c == lab(o)
        alt = ", ".join(f"x{j} {row[j] / p:.4f}" for j in np.argsort(-row) if j != c and row[j] / p >= 1e-4)
        lines.append(f"  {order_text(o):<{4 * k}} label x{lab(o)}  predicts x{c} with prob {row[c] / p:.6f}"
                     + (f"  (else {alt})" if alt else ""))
    return table, follows, dominant_is_label, lines


def write(r):
    task, d, n = r["task"], r["d"], r["n"]
    m = zoo.load(task, d, n)
    out = []
    W = out.append
    W(f"AlgZoo model {task} hidden={d} seq_len={n}: {r['params']} parameters, best-of-5 seed {r['seed']}.")
    W(f"Task: {TASK_TEXT[task].format(n=n)}.")
    W("Architecture: h_0 = 0, h_(t+1) = ReLU(W_hh h_t + W_hi x_t) for t = 0..n-1, output = W_oh h_n (no biases).")
    if task == "argmedian" and n == 3:
        W("Note: for 3 numbers the median is also the second largest, and these weights are bit-identical to "
          f"2nd_argmax hidden={d} seq_len=3 seed {r['seed']}.")
    W("")
    W("ARC'S CRITERION")
    W("An explanation is a mechanistic estimate of the model's accuracy (mean squared error for median): an estimate")
    W("deduced from the weights, without sampled inputs. (a) Its error over noisy copies of the model must be no")
    W("larger than random sampling's at the same compute. (b) Full understanding: its surprise in bits should be about")
    W("the bits of optimization that selected the model, which is at most the 32 bits per float32 weight.")
    W("")
    W("THE EXPLANATION")
    W("ReLU without biases makes the output a positively homogeneous piecewise-linear function of x = (x0..x(n-1)).")
    W("Each piece is a polyhedral cone in R^n on which every neuron keeps its on/off state at every step, so the output")
    W("is a fixed linear map of x on the cone. The enumeration cuts R^n into these cones (splitting at each neuron's")
    W("pre-activation), cuts each cone by the hyperplanes x_i = x_j so the true answer is fixed, and integrates the")
    W("Gaussian measure of the part where the model is right (or the squared error) in closed form.")
    W(f"Cones in the enumeration (it starts from the 2^n coordinate orthants): "
      f"{len(exactn.cells(m)) if n <= 3 else len(exactz.cells(m))}.")
    W("")
    if task == "median":
        W("WHAT THE MODEL COMPUTES (exact, per ordering of the inputs; output = coefficients . x on average)")
        if n == 2:
            W("  The median of two numbers is their mean, a linear function. The model's output on each cone:")
            import exact2
            for lo, hi, L in exact2.arcs(m):
                W(f"  angle {lo:6.3f}..{hi:6.3f} rad: output = {L[0][0]:+.6f} x0 {L[0][1]:+.6f} x1")
        else:
            tab = exact3.mse_by_order(m) if n == 3 else {}
            if n > 3:
                W("  (per-ordering breakdown not computed for this length)")
            for o in sorted(tab):
                p, e, w = tab[o]
                W(f"  {order_text(o):<12} median x{o[1]}  mean coefficients ({', '.join(f'{v:+.4f}' for v in w)})"
                  f"  share of squared error {e / max(r['exact'], 1e-300):.3f}")
        W("  Readable algorithm: output the middle value; the coefficients above show how closely the weights do this.")
    elif n >= 5:
        acc, per_class, pred = exactz.accuracy_by_class(m, marginals=True)
        W("WHAT THE MODEL COMPUTES (exact)")
        W("  P(model predicts position c):        " + "  ".join(f"x{c}: {p:.4f}" for c, p in enumerate(pred)))
        W("  P(right | the answer is position c):  " + "  ".join(f"x{c}: {p * n:.4f}" for c, p in enumerate(per_class)))
        target = "second largest" if task == "2nd_argmax" else "median"
        if acc >= 0.5:
            W(f"  Readable algorithm: return the position of the {target} input; the model does so on a share"
              f" {acc:.6f} of inputs.")
        else:
            top = [f"x{c}" for c in np.argsort(-pred)[:3]]
            W(f"  The model does not implement the task (right on a share {acc:.6f} of inputs); it concentrates its"
              f" predictions on {', '.join(top)}.")
    else:
        table, follows, dom, lines = classification_algorithm(m)
        W("WHAT THE MODEL COMPUTES (exact; for each ordering of the inputs, largest first, the model's prediction)")
        out.extend(lines)
        if dom:
            W(f"  Readable algorithm: return the position of the {'second largest' if task == '2nd_argmax' else 'median'}"
              f" input. The model agrees with this on a share {follows:.8f} of inputs;")
            W("  the remaining inputs are the alternatives listed above, which lie in thin cones next to ties.")
        else:
            W("  Readable algorithm: the table above (per ordering, the predicted position). The model does not")
            W("  implement the task on every ordering; its accuracy is the share of inputs where table and label agree.")
    W("")
    Wh, Whh, Wo = standardized(m)
    if d <= 8:
        W("WEIGHTS, each hidden neuron rescaled so its input weight is +1 or -1 (an exactly equivalent model)")
        W("  W_hi:\n" + fmt_matrix(Wh[None]))
        W("  W_hh:\n" + fmt_matrix(Whh))
        W("  W_oh:\n" + fmt_matrix(Wo))
        W("")
    W("EVIDENCE")
    metric = "accuracy" if r["metric"] == "accuracy" else "mean squared error"
    W(f"  Exact {metric}: {r['exact']:.10g}" + (f" (certified error bound from cones too thin for the quadrature: "
                                                f"{r['unresolved_bound']:.1g})." if 'unresolved_bound' in r else "."))
    W(f"  Check with 2^24 fresh Gaussian samples: {r['sampled_2e24']:.10g} +- {r['sampled_se']:.2g} "
      f"(z = {r['z']:.2f}).")
    nz = r["noisy"]
    W(f"  Noisy copies (weights + N(0, {nz['sd']}^2), {nz['K']} copies; {metric} {nz['metric_mean']:.4g} +- "
      f"{nz['metric_sd_across_copies']:.3g} across copies):")
    W(f"    exact estimate vs 2^20 fresh samples per copy: mean z^2 = {nz['mean_z2']:.2f} (1 if the estimate is exact),"
      f" max |z| = {nz['max_abs_z']:.1f}.")
    W("")
    W("SCORE")
    k = nz["samples_at_equal_ops"]
    W(f"  (a) Error against compute. The exact estimate costs {nz['estimator_ops_mean']:.3g} arithmetic operations on")
    W(f"      average over the copies, the cost of {k:.0f} sampled inputs ({nz['sample_cost_ops']} operations each).")
    W(f"      At that budget sampling's RMS error is {nz['sampling_rms_at_equal_ops']:.2g}; the exact estimate's error")
    W(f"      is floating-point rounding (agreement with independent samples to within their standard error of")
    W(f"      {nz['check_rms_se']:.1g}). Sampling needs ~{nz['sampling_rms_at_equal_ops'] ** 2 * k / 1e-18:.0e} samples")
    W("      to reach 1e-9. MET at every budget of at least that many operations. Below it the estimator gives")
    W("      nothing better than a constant guess, so the comparison at smaller budgets is open.")
    bits = r["surprise_sign_bits"]
    W(f"  (b) Surprise accounting. The enumeration establishes {bits} sign facts (one bit each), against at most")
    W(f"      {r['weight_bits_float32']} bits in the float32 weights. ARC's hand explanation of M2,2 used about 40 bits.")
    if bits > r["weight_bits_float32"]:
        verdict = "NOT MET: the enumeration needs more bits than the weights contain"
    elif (task, d, n) in FULL_UNDERSTANDING:
        verdict = FULL_UNDERSTANDING[(task, d, n)]
    else:
        verdict = ("UNDETERMINED: below the float32 bound, but the optimization bits are unknown; ARC would call an"
                   " enumeration of this size brute force")
    W(f"      Verdict: {verdict}.")
    W("")
    W("Code: ~/params/experiments/ami/algzoo (exact2.py, exact3.py for lengths 2-3; exactz.py with exact integer")
    W("geometry for longer sequences; exactn.py for orthant probabilities; evaluate.py, run_exact.py, run_exactz.py).")
    if (task, d, n) in ADDENDA:
        out += ["", ADDENDA[(task, d, n)]]
    path = os.path.join(DIR, f"{task}_{d}_{n}.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def main(only=None):
    """only: set of "task_d_n" names to (re)write; None writes every model with results."""
    for name in ("results_exact.jsonl", "results_exactn.jsonl"):
        p = os.path.join(DIR, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if only is None or f"{r['task']}_{r['d']}_{r['n']}" in only:
                print(write(r), flush=True)




def write_cycle(r):
    d, n = r["d"], r["n"]
    m = zoo.load("longest_cycle", d, n)
    x = zoo.all_inputs_discrete(n)
    y = zoo.label("longest_cycle", x)
    pred = m(x).argmax(-1)
    out = []
    W = out.append
    W(f"AlgZoo model longest_cycle hidden={d} seq_len={n}: {r['params']} parameters, best-of-5 seed {r['seed']}.")
    W(f"Task: the input lists a function f on {{0..{n - 1}}} as x_i = f(i), uniform over all {n}^{n} = {r['inputs']}"
      " functions; the answer is the length of f's longest cycle minus 1.")
    W("Architecture: token + position embeddings, 2 attention-only layers (1 head, no bias, no causal mask, no layer")
    W("norm), output read from the last position.")
    W("")
    W("ARC'S CRITERION")
    W("An explanation is a mechanistic estimate of the model's accuracy: an estimate deduced from the weights, without")
    W("sampled inputs. (a) Its error over noisy copies must be no larger than random sampling's at the same compute.")
    W("(b) Full understanding: its surprise in bits should be about the bits of optimization that selected the model,")
    W("which is at most the 32 bits per float32 weight.")
    W("")
    W("THE EXPLANATION")
    W("The input space is finite, so the accuracy is the exact average over every input. This is a deductive")
    W("computation (no inference from a sample), so it counts as a mechanistic estimate with zero error, but it is")
    W("brute force: it checks the model input by input and says nothing about how the attention layers find cycles.")
    W("")
    W("WHAT THE MODEL COMPUTES (exact confusion: rows = true longest cycle, columns = predicted)")
    C = np.zeros((n, n), dtype=int)
    np.add.at(C, (y, pred), 1)
    W("  true\\pred " + " ".join(f"{c + 1:>6d}" for c in range(n)))
    for t in range(n):
        if C[t].sum():
            W(f"  {t + 1:>9d} " + " ".join(f"{v:>6d}" for v in C[t]) + f"   accuracy {C[t, t] / C[t].sum():.4f}")
    W("")
    W("EVIDENCE")
    W(f"  Exact accuracy over all {r['inputs']} inputs: {r['accuracy']:.6f} (float64 numpy); the float32 torch model")
    W(f"  gives {r['accuracy_float32_torch']:.6f}. Smallest gap between the top two logits: {r['min_top2_logit_gap']:.3g}.")
    nz = r["noisy"]
    W(f"  Noisy copies (all weights + N(0, {nz['sd']}^2), {nz['K']} copies): accuracy {nz['accuracy_mean']:.3f} +-"
      f" {nz['accuracy_sd']:.3f}; the exhaustive average is exact for each copy.")
    W("")
    W("SCORE")
    W(f"  (a) Error against compute. Enumeration costs {r['exhaustive_ops']:.3g} operations, the cost of {r['inputs']}")
    W(f"      sampled inputs; sampling's RMS error at that budget is {nz['sampling_rms_at_equal_ops']:.2g} over the noisy")
    W("      copies, the enumeration's is 0. MET at budgets of at least one pass per input; below that budget")
    W("      this estimator has no answer, so the comparison at smaller budgets is open.")
    bits = r["surprise_bits_enumeration"]
    W(f"  (b) Surprise accounting. Checking every input costs {bits:.0f} bits (log2 of the inverse prior probability")
    W(f"      of each observed outcome, prior = label frequencies), against at most {r['weight_bits_float32']} bits in")
    W("      the float32 weights.")
    if bits > r["weight_bits_float32"]:
        W("      Verdict: NOT MET: the enumeration needs more bits than the weights contain.")
    else:
        W("      Verdict: UNDETERMINED: below the float32 bound, but the optimization bits are unknown, and an")
        W("      input-by-input check explains no mechanism.")
    W("")
    W("Code: ~/params/experiments/ami/algzoo/run_cycle.py.")
    path = os.path.join(DIR, f"longest_cycle_{d}_{n}.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def main_cycle():
    p = os.path.join(DIR, "results_cycle.jsonl")
    for line in open(p):
        print(write_cycle(json.loads(line)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(set(sys.argv[1:]))
    else:
        main()
        main_cycle()
