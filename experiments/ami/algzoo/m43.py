"""M4,3 (2nd argmax, hidden 4, length 3; also argmedian 4,3, bit-identical): mechanism, interventions, and error
against compute for deterministic estimators, sampling, and ARC's two hand estimators.

Writes results/m43_curve.tsv (one line per estimator setting), results/m43_budgets.tsv (best estimator at each budget
against sampling) and results/2nd_argmax_4_3.txt (algorithm, evidence, scores).

Protocol (Hilton's notes, Dec 2024): 256 copies of the model with N(0, 0.05^2) added to every raw weight; the truth for
each copy is its exact accuracy (exact3, spherical polygons, checked against 2^24 samples on the original). Sampling
with k inputs has RMS error sqrt(mean over copies of p (1 - p) / k). Ops: scalar adds, multiplies, divides, compares and
elementary-function calls each count 1; one sampled input costs `sample_cost` ops (forward pass, label, comparison).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aq3  # noqa: E402
import exact3  # noqa: E402
import red3  # noqa: E402
import zoo  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
K, SD, SEED = 256, 0.05, 0

# ARC's estimators as reported in Hilton, "Heuristic explanations for 2nd argmax models" (Dec 2024): RMS error over 256
# noisy copies and operation counts in his convention, where one sampled input costs 72 operations.
ARC = [("arc_hilton_original", 71000, 72, 0.03), ("arc_hilton_robinson_symmetric", 7680, 72, 0.05)]


def sample_cost(m):
    """Ops for one sampled input: step 1 (d multiplies, d ReLUs), steps 2..n (W_hh h + W_hi x, ReLU), the readout,
    the argmax of the logits, the label (max, then max of the rest) and the comparison."""
    d, n, k = m.d, m.n, m.W_oh.shape[0]
    return 2 * d + (n - 1) * (2 * d * d + 2 * d) + k * (2 * d - 1) + (k - 1) + (2 * n - 3) + 1


def noisy_copies(m):
    rng = np.random.default_rng(SEED)
    base = m.flat()
    return [m.with_flat(base + SD * rng.standard_normal(base.shape)) for _ in range(K)]


def standardized(m):
    s = np.abs(m.W_hi)
    return m.W_hi / s, (m.W_hh * (1 / s)[:, None]) * s[None, :], m.W_oh * s[None, :]


U = np.array([[1.0, 0, 0, -1], [0, 1, -1, 0]])  # u = U h: neuron pairs (0, 3) and (1, 2) store signed numbers
V = np.array([[1.0, 0, 0, 1], [0, 1, 1, 0]])  # |u| = V h when exactly one neuron of each pair is on


def reduction(m):
    """Least-squares fit (George Robinson's 'Symmetric RNNs'): W_hh ~ U^T W' U, W_hi ~ U^T w', W_oh ~ W'_oh V.
    The reduced model is linear in u with an absolute value at the end: u_(t+1) = W' u_t + w' x_t,
    logits = W'_oh |u_3| = W'_oh |A x| with A = [W'^2 w', W' w', w']."""
    wi, Wh, Wo = standardized(m)
    Wp = np.linalg.pinv(U.T) @ Wh @ np.linalg.pinv(U)
    wp = np.linalg.pinv(U.T) @ wi
    Wop = Wo @ np.linalg.pinv(V)
    A = np.stack([Wp @ Wp @ wp, Wp @ wp, wp], 1)
    return dict(wi=wi, Wh=Wh, Wo=Wo, Wp=Wp, wp=wp, Wop=Wop, A=A, resid_hh=Wh - U.T @ Wp @ U, resid_oh=Wo - Wop @ V)


def run_std(Wh, wi, Wo, x, hook=None):
    h = np.zeros((x.shape[0], Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t : t + 1] * wi, 0.0)
        if hook is not None:
            h = hook(t, h)
    return h @ Wo.T


def interventions(m, R, rng, B=1 << 20):
    """Causal tests of the claim that the state is two signed numbers u = (h0 - h3, h1 - h2) (standardized units)."""
    x = rng.standard_normal((B, 3))
    xb = rng.standard_normal((B, 3))
    lab = zoo.label("2nd_argmax", x)
    wi, Wh, Wo = R["wi"], R["Wh"], R["Wo"]
    base_pred = run_std(Wh, wi, Wo, x).argmax(-1)
    out = {"base_accuracy_sampled": float((base_pred == lab).mean())}
    # the reduced model alone
    u3 = x @ R["A"].T
    red_pred = (np.abs(u3) @ R["Wop"].T).argmax(-1)
    out["reduced_accuracy_sampled"] = float((red_pred == lab).mean())
    out["reduced_agrees_with_model"] = float((red_pred == base_pred).mean())
    # where r = |u1| / |u0| falls for each position of the median
    _, cuts, _, _ = readout_rule(R)
    edges = [0.0] + [cc[0] for cc in cuts] + [np.inf]
    r = np.abs(u3[:, 1]) / np.abs(u3[:, 0])
    band = np.searchsorted(edges, r) - 1
    out["r_band_edges"] = edges[1:-1]
    out["r_band_by_label"] = {int(l): np.bincount(band[lab == l], minlength=len(edges) - 1).tolist() for l in range(3)}
    # projection: replace the state by the one the pair structure predicts from u alone
    for t in (0, 1, 2):
        def hook(tt, h, t=t):
            return np.maximum((h @ U.T) @ U, 0.0) if tt == t else h
        p = run_std(Wh, wi, Wo, x, hook).argmax(-1)
        out[f"project_state_step{t + 1}_agrees"] = float((p == base_pred).mean())
        out[f"project_state_step{t + 1}_accuracy"] = float((p == lab).mean())
    # interchange: after step 2, give u_k the value it takes on another input; the reduced model, intervened the same
    # way, predicts the model's output
    hb = np.zeros((B, 4))
    for t in range(2):
        hb = np.maximum(hb @ Wh.T + xb[:, t : t + 1] * wi, 0.0)
    ub = hb @ U.T
    for k in (0, 1):
        def hook(tt, h, k=k):
            if tt != 1:
                return h
            u = h @ U.T
            u[:, k] = ub[:, k]
            return np.maximum(u @ U, 0.0)
        p = run_std(Wh, wi, Wo, x, hook).argmax(-1)
        u2 = x[:, :2] @ R["A"][:, 1:].T  # reduced u after two steps: (W' w' x0 + w' x1)
        u2[:, k] = (xb[:, :2] @ R["A"][:, 1:].T)[:, k]
        u3 = u2 @ R["Wp"].T + x[:, 2:3] * R["wp"][None]
        q = (np.abs(u3) @ R["Wop"].T).argmax(-1)
        out[f"interchange_u{k}_step2_model_vs_reduced_agree"] = float((p == q).mean())
        out[f"interchange_u{k}_step2_changes_output"] = float((p != base_pred).mean())
    return out


def evaluate(m, models, truth, arc=True, verbose=True):
    c = sample_cost(m)
    pq = float(np.mean(truth * (1 - truth)))
    rows = []

    def add(name, ests, ops):
        e = np.array(ests) - truth
        k = float(np.mean(ops)) / c
        samp = math.sqrt(pq / k) if k >= 1 else float("nan")
        rows.append(dict(estimator=name, ops=float(np.mean(ops)), sample_equivalents=k,
                         rms=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), max_abs=float(np.abs(e).max()),
                         sampling_rms_same_ops=samp))
        if verbose:
            print(json.dumps(rows[-1]), flush=True)

    base_acc = exact3.accuracy(m)
    add("constant_base_exact_accuracy", [base_acc] * len(models), [0] * len(models))
    ests, ops = [], []
    for md in models:
        o = aq3.Ops()
        ests.append(aq3.accuracy_one(md, o))
        ops.append(o.n)
    add("aq_one_line", ests, ops)
    for mm in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        ests, ops = [], []
        for md in models:
            o = aq3.Ops()
            ests.append(aq3.accuracy_halves(md, mm, o))
            ops.append(o.n)
        add(f"aq_halves_gl{mm}", ests, ops)
    for mm in (1, 2, 3, 4, 6, 8):
        ests, ops = [], []
        for md in models:
            o = aq3.Ops()
            ests.append(aq3.accuracy_gl(md, mm, o))
            ops.append(o.n)
        add(f"aq_arcs_gl{mm}", ests, ops)
    ests, ops = [], []
    for md in models:
        o = exact3.Ops()
        ests.append(red3.accuracy(md, o))
        ops.append(o.n)
    add("reduced_mechanism_exact", ests, ops)
    ests, ops = [], []
    for md in models:
        o = exact3.Ops()
        ests.append(exact3.accuracy(md, o))
        ops.append(o.n)
    add("exact_spherical_polygons", ests, ops)
    for name, arc_ops, arc_cost, rms in (ARC if arc else []):
        k = arc_ops / arc_cost
        rows.append(dict(estimator=name, ops=k * c, sample_equivalents=k, rms=rms, bias=float("nan"),
                         max_abs=float("nan"), sampling_rms_same_ops=math.sqrt(pq / k), reported=True))
    return rows, pq, c, base_acc


def budgets(rows, pq, c):
    """At each budget, the smallest RMS among our estimators that fit, against sampling with floor(budget / c) inputs."""
    ours = [r for r in rows if not r.get("reported")]
    out = []
    for B in np.unique(np.round(np.logspace(0, 8, 81))):
        fit = [r for r in ours if r["ops"] <= B]
        best = min(fit, key=lambda r: r["rms"])
        k = math.floor(B / c)
        samp = math.sqrt(pq / k) if k >= 1 else float("inf")
        out.append(dict(budget_ops=float(B), best=best["estimator"], best_rms=best["rms"], sampling_inputs=k,
                        sampling_rms=samp, ratio=best["rms"] / samp if k >= 1 else 0.0))
    return out


def readout_rule(R):
    """The reduced readout depends on |u0| and |u1| only, so the prediction is a function of r = |u1| / |u0|: each
    pair of logits ties at one value of r. Returns the r-intervals of each predicted position and a check."""
    Wop = R["Wop"]
    ties = {}
    for a in range(3):
        for b in range(a + 1, 3):
            d = Wop[a] - Wop[b]  # logit_a - logit_b = d0 |u0| + d1 |u1| > 0  <=>  r (d1) > -d0
            ties[(a, b)] = -d[0] / d[1]
    rs = np.linspace(0.0, 3.0, 30001)[1:]
    pred = np.argmax(Wop[:, :1] + Wop[:, 1:] * rs[None], 0)
    cuts = [(float(rs[i]), int(pred[i]), int(pred[i + 1])) for i in range(len(rs) - 1) if pred[i] != pred[i + 1]]
    return ties, cuts, int(pred[0]), int(pred[-1])


def write_report(m, rows, bud, R, iv, truth, pq, c, base_acc, red):
    A = R["A"]
    ties, cuts, first, last = readout_rule(R)
    a0, a1 = A[0] / A[0, 0], A[1] / A[1, 0]
    by = {r["estimator"]: r for r in rows}
    wb = max((b for b in bud if b["sampling_inputs"] >= 1), key=lambda b: b["ratio"])
    worst = wb["ratio"]
    L = []
    W = L.append
    W("AlgZoo M4,3: 2nd_argmax hidden=4 seq_len=3, 32 parameters, zoo seed 0 (bit-identical to argmedian 4,3 seed 0).")
    W("Task: position of the second largest of 3 independent standard Gaussian numbers (for 3 numbers this is also the")
    W("median's position). Architecture: h_0 = 0, h_(t+1) = ReLU(W_hh h_t + W_hi x_t), logits = W_oh h_3, no biases.")
    W(f"Exact accuracy {base_acc:.10f} (spherical polygons, exact3.py; agrees with 2^24 samples).")
    W("")
    W("THE ALGORITHM (which neuron does what)")
    W("Rescale each neuron so its input weight is +1 or -1 (ReLU is positively homogeneous; the model is unchanged).")
    W("Neurons 0 and 1 read +x_t, neurons 2 and 3 read -x_t. The recurrent weights have the pair form")
    W("W_hh = U^T W' U + residual with U = [[1,0,0,-1],[0,1,-1,0]], and W_oh = W'_oh V + residual with")
    W("V = [[1,0,0,1],[0,1,1,0]] (George Robinson's symmetry). Neurons 0 and 3 hold the positive and negative parts of")
    W("one signed number u0 = h0 - h3, neurons 1 and 2 those of u1 = h1 - h2. Because U ReLU(U^T u) = u, the state")
    W("update is linear in u: u_(t+1) = W' u_t + w' x_t, and the readout sees |u| = V h. So the network is a 12-")
    W("parameter model: two linear features of the input and a linear readout of their absolute values.")
    fm = lambda M, p: "[" + "; ".join(" ".join(f"{v:.{p}f}" for v in row) for row in np.atleast_2d(M)) + "]"
    W(f"  W' = {fm(R['Wp'], 3)} (rows), w' = {fm(R['wp'], 3)}")
    W(f"  W'_oh = {fm(R['Wop'], 2)}")
    W(f"  after 3 steps u = A x, A = {fm(A, 3)}, i.e.")
    W(f"  u0 = {A[0, 0]:.3f} * [(x0 - x1) + {a0[2]:.2f} (x2 - x1)]   (coefficients {np.array2string(a0, precision=3)})")
    W(f"  u1 = {A[1, 0]:.3f} * [(x0 - x2) + {a1[1]:.2f} (x1 - x2)]   (coefficients {np.array2string(a1, precision=3)})")
    W("  Each feature is a contrast (coefficients sum to ~0) and both are mostly x2 - x1 (u1's scale is negative), with")
    W("  tilts of opposite sign by x0. The readout depends only on r = |u1| / |u0|:")
    lo = 0.0
    seg = []
    cur = first
    for rr, a, b in cuts:
        seg.append((lo, rr, a))
        lo = rr
    seg.append((lo, math.inf, last))
    for a, b, p in seg:
        W(f"    {a:.3f} < r < {b:.3f}: predict x{p}")
    W("  Where r falls, by the true answer (share of 2^20 Gaussian inputs with that answer in each band of r):")
    for l, cnt in iv["r_band_by_label"].items():
        cnt = np.array(cnt, dtype=float)
        W(f"    answer x{l}: " + ", ".join(f"band {i} {v / cnt.sum():.4f}" for i, v in enumerate(cnt)))
    W("  Both features are about lambda (x2 - x1) (lambda ~ 4) plus a tilt of opposite sign by x0, so |u0| and |u1|")
    W("  are nearly equal exactly when x0 lies between x1 and x2 (the middle band answers x0); otherwise the tilt makes")
    W("  one of them larger, and which one says whether x1 or x2 is the middle value.")
    rh = R["resid_hh"]
    W("  The residual of W_hh adds the same amount to both members of a pair, about")
    W(f"  {rh[0, 1]:+.2f} |u1| {rh[0, 0]:+.2f} |u0| to neurons 0 and 3 and {rh[1, 1]:+.2f} |u1| {rh[1, 0]:+.2f} |u0| to neurons 1"
      " and 2; it is zero after step 2")
    W("  (there u0 = u1 = x0 in these units) and at step 3 it switches both members of a pair on or off in thin slivers.")
    W("")
    W("EVIDENCE FOR THE ALGORITHM (2^20 Gaussian inputs each)")
    W(f"  The 12-parameter reduced model alone: accuracy {iv['reduced_accuracy_sampled']:.4f} (model"
      f" {iv['base_accuracy_sampled']:.4f}); it predicts the same position as the model on"
      f" {iv['reduced_agrees_with_model']:.4f} of inputs.")
    W("  Projection: replacing the state h_t by ReLU(U^T U h_t) (only u kept) changes no prediction at steps 1 and 2"
      f" (agreement {iv['project_state_step1_agrees']:.4f}, {iv['project_state_step2_agrees']:.4f}) and"
      f" {1 - iv['project_state_step3_agrees']:.4f} of predictions at step 3.")
    W("  Interchange: after step 2, setting u0 (or u1) to its value on another random input changes the model's output"
      f" on {iv['interchange_u0_step2_changes_output']:.3f} ({iv['interchange_u1_step2_changes_output']:.3f}) of inputs;"
      " the reduced model with the same")
    W(f"  interchange predicts the intervened model's output on {iv['interchange_u0_step2_model_vs_reduced_agree']:.4f}"
      f" ({iv['interchange_u1_step2_model_vs_reduced_agree']:.4f}) of inputs.")
    W("")
    W("THE ESTIMATOR (aq3.py): exact along lines of the last input, Gauss-Legendre over one angle")
    W("Positive homogeneity removes the radius. For the direction a of (x0, x1) the first two steps give a fixed state")
    W("h_2(a); the last input enters linearly, so along the line of x2 every neuron switches once, the logits are linear")
    W("between switches, and the label is fixed between x2 = x0 and x2 = x1. The set of x2 where the model is right is a")
    W("union of at most d + 3 intervals with closed-form ends, and its Gaussian measure given a is closed form")
    W("(G(t) = t / sqrt(1 + t^2)). Only the angle a is integrated numerically, by Gauss-Legendre on the two half-circles")
    W("between pi/4 and 5 pi/4 (where x0 = x1 and the label jumps), or on the arcs where the first two steps keep their")
    W("pattern. No random numbers anywhere; the cost is (number of nodes) x O(d^2) and does not grow with the number of")
    W("cones. 'aq_one_line' uses the single line (x0, x1) = (1, -1). The explanation the estimator relies on is the")
    W("model's piecewise-linear, homogeneous structure and the label's jumps; it uses no advice fitted to this model.")
    W("")
    W("THE ESTIMATOR FROM THE COMPACT MECHANISM ALONE (red3.py)")
    W("Read the 12-number reduced model off each copy's weights (rescale, then average entries into the pair form) and")
    W("integrate its accuracy exactly (spherical polygons cut by the two planes (A x)_k = 0, the ordering planes and the")
    W("logit ties). On the original model this gives"
      f" {red['base']:.5f} (exact {base_acc:.5f}); on the noisy copies its RMS error is {by['reduced_mechanism_exact']['rms']:.4f}"
      f" at {by['reduced_mechanism_exact']['ops']:.0f} ops,")
    W(f"where sampling's is {by['reduced_mechanism_exact']['sampling_rms_same_ops']:.4f}: it loses to sampling."
      f" Its estimates correlate {red['corr']:.2f} with the truth across copies. The noise breaks the")
    W("pair symmetry, and the broken symmetry (the residual the reduced model drops) is what lowers the copies'")
    W("accuracy from 0.985 to 0.878; the mechanism explains the original model but not its neighbours. Hilton's")
    W("symmetric estimator (5% RMS) added a covariance propagation of that residual and still lost to sampling.")
    W("")
    W(f"ERROR AGAINST COMPUTE ({K} copies, every raw weight + N(0, {SD}^2); truth = exact accuracy of each copy:")
    W(f"mean {truth.mean():.4f}, sd {truth.std():.4f}; sampling's per-input sd sqrt(mean p(1-p)) = {math.sqrt(pq):.4f};")
    W(f"one sampled input = {c} ops)")
    W(f"  {'estimator':<34}{'ops':>9}{'inputs-equiv':>13}{'RMS':>10}{'sampling RMS':>14}{'ratio':>8}")
    for r in rows:
        ratio = r["rms"] / r["sampling_rms_same_ops"] if r["sampling_rms_same_ops"] == r["sampling_rms_same_ops"] else float("nan")
        tag = "  (reported by ARC)" if r.get("reported") else ""
        W(f"  {r['estimator']:<34}{r['ops']:>9.0f}{r['sample_equivalents']:>13.1f}{r['rms']:>10.5f}"
          f"{r['sampling_rms_same_ops']:>14.5f}{ratio:>8.3f}{tag}")
    W("  ARC's rows: Hilton's two hand estimators (Dec 2024 notes), RMS over his 256 noisy copies and his operation")
    W("  counts converted to inputs-equivalents with his own per-input cost (72); 'ops' is that times our per-input cost.")
    W("  Hilton compared them with sampling using a per-input sd of 24%; on these copies it is"
      f" {math.sqrt(pq):.1%}, so sampling's RMS at his budgets is")
    W(f"  {by['arc_hilton_original']['sampling_rms_same_ops']:.4f} and {by['arc_hilton_robinson_symmetric']['sampling_rms_same_ops']:.4f}"
      " (he quoted 0.8% and 2.3%): both hand estimators lose to sampling by 3x and 1.6x in RMS.")
    W("")
    W("SCORE")
    W("  (a) Error against compute: MET AT EVERY BUDGET. results/m43_budgets.tsv takes, at each budget from 1 to 1e8 ops,")
    W("      the most accurate estimator that fits and compares it with sampling at the same ops. Below one input's cost")
    W("      sampling has no estimate; from 1 input up the worst ratio of our RMS to sampling's is"
      f" {worst:.2f} ({wb['best']}")
    W(f"      at {wb['budget_ops']:.0f} ops = {wb['sampling_inputs']} inputs); from"
      f" {by['aq_one_line']['ops']:.0f} ops the weight-reading estimators are 3-50x more accurate")
    W(f"      than sampling (e.g. {by['aq_halves_gl4']['ops']:.0f} ops: {by['aq_halves_gl4']['rms']:.4f} vs"
      f" {by['aq_halves_gl4']['sampling_rms_same_ops']:.4f}; {by['aq_halves_gl16']['ops']:.0f} ops:"
      f" {by['aq_halves_gl16']['rms']:.4f} vs {by['aq_halves_gl16']['sampling_rms_same_ops']:.4f}),")
    W(f"      and from {by['exact_spherical_polygons']['ops']:.0f} ops the exact enumeration has no error. ARC's best"
      " M4,3 estimators lose to sampling; these beat it.")
    o = aq3.Ops()
    aq3.accuracy_halves(m, 4, o)
    infl = (o.n + 9 * o.fn) / o.n
    W(f"      Op convention: elementary functions (sqrt, sin, cos) are {o.fn / o.n:.0%} of the estimators' ops; counting")
    W(f"      each as 10 ops would raise their costs {infl:.2f}x and the RMS ratios above by {math.sqrt(infl):.2f}x, while")
    W("      sampling's cost leaves out generating its Gaussian inputs.")
    W("      The constant at budgets below ~340 ops is advice (one number from the base model), not a reading of the")
    W("      noisy weights; every estimator from 'aq_one_line' up reads only the copy's weights.")
    full, qb = red3.explanation_bits(m)
    tol = {b: a - full for b, a in qb}
    b7 = min(b for b, d in tol.items() if abs(d) < 7e-4)
    b1 = min(b for b, d in tol.items() if abs(d) < 1e-4)
    if os.path.exists(os.path.join(RES, "aq3_seeds.tsv")):
        import aq3_seeds
        v = aq3_seeds.verdicts()[4]
        by_s = {r["estimator"]: r for r in v["rows"]}
        W("      ARC's stronger test, the 5 training seeds of this configuration (exact accuracies "
          + ", ".join(f"{t:.4f}" for t in v["truth"]) + "), with no")
        W("      advice at all (the 0-op guess is 1.0, the task): "
          + ("MET AT EVERY BUDGET" if not v["failing"] else "not met at some budgets")
          + f"; worst ratio {v['worst']:.2f} ({v['worst_best']} at {v['worst_ops']:.0f} ops);"
          f" e.g. {by_s['aq_halves_gl4']['ops']:.0f} ops: RMS {by_s['aq_halves_gl4']['rms']:.4f} vs sampling"
          f" {by_s['aq_halves_gl4']['sampling_rms_same_ops']:.4f}; {by_s['aq_halves_gl32']['ops']:.0f} ops:"
          f" {by_s['aq_halves_gl32']['rms']:.5f} vs {by_s['aq_halves_gl32']['sampling_rms_same_ops']:.4f}"
          " (results/aq3_seeds.tsv).")
    W("  (b) Surprise accounting: UNDETERMINED. The line+quadrature estimator carries no model-specific advice, so")
    W("      ARC's bit count applies to the hand explanation instead: two absolute-value features and a readout, 8 free")
    W("      numbers after rescaling (W' 2x2 and the column-centred W'_oh) plus the sign facts (4 input signs) and the")
    W("      pair structure. Rounding the 8 numbers to b significant bits moves the reduced model's accuracy by")
    W("      " + ", ".join(f"{b}: {d:+.4f}" for b, d in tol.items() if b in (4, 6, 8, 10, 11, 12)) + ";")
    W(f"      it holds to 7e-4 from {b7} bits and to 1e-4 from {b1} bits per number (the readout thresholds and the")
    W("      features' tilts are near-equalities, as in ARC's M2,2), so the numbers cost about"
      f" {8 * b7}-{8 * b1} bits; with the")
    W("      structural facts (signs, and the ~20 pair equalities that hold to ~10%) the explanation is roughly 100-160 bits, below the 1024-bit float32 bound and ~3x ARC's 40-bit")
    W(f"      M2,2 explanation. It reproduces {iv['reduced_agrees_with_model']:.2%} of predictions; the residual's slivers"
      " (the rest) are not")
    W("      explained. Whether this matches the bits of optimization that selected the model is not known.")
    W("")
    W("Files: aq3.py (estimators), exact3.py (truth), m43.py (this report); results/m43_curve.tsv, m43_budgets.tsv,")
    W("m43_mechanism.json.")
    with open(os.path.join(RES, "2nd_argmax_4_3.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(RES, exist_ok=True)
    m = zoo.load("2nd_argmax", 4, 3)
    models = noisy_copies(m)
    truth = np.array([exact3.accuracy(md) for md in models])
    rows, pq, c, base_acc = evaluate(m, models, truth)
    with open(os.path.join(RES, "m43_curve.tsv"), "w") as f:
        keys = ["estimator", "ops", "sample_equivalents", "rms", "bias", "max_abs", "sampling_rms_same_ops"]
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    bud = budgets(rows, pq, c)
    with open(os.path.join(RES, "m43_budgets.tsv"), "w") as f:
        keys = ["budget_ops", "best", "best_rms", "sampling_inputs", "sampling_rms", "ratio"]
        f.write("\t".join(keys) + "\n")
        for r in bud:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    R = reduction(m)
    iv = interventions(m, R, np.random.default_rng(11))
    with open(os.path.join(RES, "m43_mechanism.json"), "w") as f:
        json.dump(dict(reduction={k: np.asarray(v).round(4).tolist() for k, v in R.items()}, interventions=iv,
                       truth_mean=float(truth.mean()), truth_sd=float(truth.std()), sampling_per_input_sd=math.sqrt(pq),
                       sample_cost=c, base_exact=base_acc), f, indent=1)
    print(json.dumps(iv, indent=1))
    red_est = np.array([red3.accuracy(md) for md in models])
    red = dict(base=red3.accuracy(m), corr=float(np.corrcoef(red_est, truth)[0, 1]))
    write_report(m, rows, bud, R, iv, truth, pq, c, base_acc, red)


if __name__ == "__main__":
    main()
