"""Write the M16,10 result files (challenge model and the five zoo seeds) from m16_suite.py, m16_mechanism.py and
m16_estimators.py outputs: results/example_2nd_argmax_16_10.txt and results/2nd_argmax_16_10_seed<s>.txt."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m16_suite  # noqa: E402
import m43  # noqa: E402

RES = m43.RES


def read_tsv(path):
    lines = open(path).read().strip().split("\n")
    keys = lines[0].split("\t")
    out = []
    for line in lines[1:]:
        r = {}
        for k, v in zip(keys, line.split("\t")):
            try:
                r[k] = float(v)
            except ValueError:
                r[k] = v
        out.append(r)
    return out


def fmt_row(v, p=2):
    return "[" + " ".join(f"{x:.{p}f}" for x in v) + "]"


def budget_table(curves, group):
    """At each budget: our best estimator of each kind that fits, against sampling with floor(budget / c) inputs."""
    C = [r for r in curves if r["group"] == group]
    c = C[0]["sample_cost"]
    pq = C[0]["per_input_sd"] ** 2
    rows = []
    for B in [c * k for k in (0.5, 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384)]:
        k = math.floor(B / c)
        samp = math.sqrt(pq / k) if k >= 1 else float("inf")
        mech = [r for r in C if r["estimator"] in ("const_task",) and r["ops"] <= B]
        det = [r for r in C if r["estimator"].startswith("halton_") and r["ops"] <= B]
        rows.append((B, k, samp, min((r["rms"] for r in mech), default=float("nan")),
                     min((r["rms"] for r in det), default=float("nan"))))
    return rows


def write(name, title, mech, suite_rec, curves, ex_mech):
    L = []
    W = L.append
    W(title)
    W("Task: position of the second largest of 10 independent standard Gaussian numbers. Architecture: 16 ReLU neurons,")
    W("h_(t+1) = ReLU(W_hh h_t + W_hi x_t), logits = W_oh h_10, no biases; 432 parameters.")
    W(f"Reference accuracy {suite_rec['ref']:.6f} +- {suite_rec['ref_se']:.1e} (exact probability along the last input,"
      " averaged over Gaussian prefixes; this is sampling, used only as the truth).")
    W("")
    W("ARC'S CRITERION: (a) a mechanistic estimate of the accuracy (deduced from the weights, not from sampled inputs)")
    W("whose error over noisy copies or seeds is no worse than random sampling's at the same operation count, at every")
    W("budget; (b) full understanding: the explanation's surprise ~ the bits of optimization that selected the model.")
    W("")
    W("THE ALGORITHM (partial: what is established)")
    ev = {e["neuron"]: e for e in mech["events"]}
    rates = np.array(mech["rates"])
    W("Neurons by what predicts their on/off state at the last step (share of inputs predicted; 'majority' = always")
    W("guessing the commoner state; 'newest' = the relative rank of the newest input among the earlier inputs and 0;")
    W("'two newest' adds the previous input's rank; 'tree' = depth-4 tree on all relative and final ranks and signs):")
    for j in range(16):
        e = ev[j]
        W(f"  n{j:<2d} on {e['on']:.3f}  majority {e['majority']:.3f}  newest {e['by_rank_newest']:.3f}  two newest"
          f" {e['by_rank_two_newest']:.3f}  tree {e['tree_depth4_heldout']:.3f}")
    clock = mech["clock"]
    W("P(neuron on at the last step | steps since the running maximum was set), rows = steps since (-1: maximum <= 0):")
    for k in sorted(clock, key=int):
        W(f"  {int(k):>2d}: {fmt_row(clock[k], 2)}")
    W("Automatic reading (same rules for every model):")
    rec = [j for j in range(16) if ev[j]["by_rank_newest"] >= 0.97 and ev[j]["on"] < 0.2]
    W(f"  - record neurons (on < 20% of the time, >= 97% predicted by the newest input's rank alone): {rec}")
    ages = sorted((int(k) for k in clock if int(k) >= 1), key=int)
    for k in ages:
        row = np.array(clock[str(k)])
        others = np.array([clock[str(a)] for a in ages if a != k]).mean(0)
        off = [j for j in range(16) if row[j] < 0.1 and others[j] > 0.7]
        on_ = [j for j in range(16) if row[j] > 0.9 and others[j] < 0.3]
        if off or on_:
            W(f"  - {k} step(s) after a new maximum: silent {off}, firing {on_}")
    good = [j for j in range(16) if ev[j]["tree_depth4_heldout"] >= 0.97]
    W(f"  - neurons whose state a depth-4 tree on ranks predicts at >= 97%: {len(good)} of 16 {good}")
    if name == "example":
        W("Reading of the tables (challenge model):")
        W("  - Records: neuron 4 is on exactly when the newest input is a new maximum (98.5% from its rank alone), and")
        W("    neuron 7's pre-activation is minus neuron 4's (correlation -1.00): the pair holds x_new - M_old as a signed")
        W("    number. Neuron 2 is always on and holds the running maximum one step late (ARC's description).")
        W("  - Neurons 1, 6, 8 (leave-one-out maxima minus the newest input, per ARC and Chen) switch on the newest")
        W("    input's rank (95-97%).")
        W("  - A record clock: one step after a new maximum neurons 5 and 9 are on; two steps after, neuron 0 is off;")
        W("    five steps after, neuron 11 is off; six steps after, neuron 10 is off. Which clock neuron is silent says")
        W("    how long ago the maximum arrived (decision trees on ranks predict 0, 10, 11 at 98-99%).")
        W("  - Neurons 3 and 15 follow value comparisons, not ranks (89-92% from ranks; they change most when values")
        W("    are redrawn with the ordering fixed).")
    W(f"Readout: T = W_oh W_hh W_oh^+ has superdiagonal {fmt_row(mech['shift_superdiag'], 2)} (a shift would be all")
    W(f"ones) and leaves {mech['shift_resid_rel']:.0%} of W_oh W_hh unexplained: the logit vector moves about one position"
      " earlier per step,")
    W("so the logits after t steps answer for the zero-padded prefix:")
    W("  " + ", ".join(f"t={r['t']}: {r['accuracy']:.3f}" for r in mech["intermediate"]))
    iv = mech["interchange_second_max_position"]
    if name == "example":
        W("Interchange 1, where the second maximum's position is held: the second largest value is swapped with a lower")
        W("one (new label), and a group of neurons is copied from that run into the original run; share of outputs that")
        W("follow the new label (at the last step; two and four steps before the end when both moved positions lie")
        W("earlier). Groups: clock {0,5,9,10,11}, max {2,4,7}, loo {1,6,8}, value {3,15}, rest {12,13,14}:")
        for g, r in iv.items():
            W(f"  {g:<12s} {r['at_last_step'][0]:.3f}   {r['2_steps_before_end'][0]:.3f}   {r['4_steps_before_end'][0]:.3f}")
        W("  No small group carries it: the position is spread over the 13 neurons outside the max circuit.")
        im = mech["interchange_max"]
        W("Interchange 2, where the maximum's position is held: the maximum is moved to another position (the label")
        W("stays) and a group is copied from that run at the last step; shares of outputs at [the old maximum's")
        W("position, the label, the maximum's new position]:")
        for g, r in im.items():
            W(f"  {g:<12s} [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")
        W("  The clock group moves the output to the new maximum's position (the readout then points at a position")
        W("  holding a large value); the leave-one-out group makes the model answer the old maximum's position.")
    else:
        W("(Neuron-group interchanges use the challenge model's neuron roles and are reported for it only.)")
    er = mech["errors"]
    W("Errors: share by the rank the model picks " + fmt_row(er["share_by_predicted_rank"][:4], 3) + " (ranks 1-4);"
      " error rate with 0 or 1 positive inputs " + ", ".join(f"{er['rate_by_npos'][k][1]:.2f}" for k in ("0", "1"))
      + " (" + f"{er['rate_by_npos']['0'][0] + er['rate_by_npos']['1'][0]:.3f}" + " of inputs);")
    W("  error rate when the 2nd-3rd gap is below 0.01, 0.1, 0.3, 1: " + ", ".join(
        f"{er['rate_by_gap23'][q][1]:.3f}" for q in ("0.01", "0.1", "0.3", "1.0")) + ".")
    W("")
    W("WHY THIS DOES NOT YET GIVE A MECHANISTIC ESTIMATE")
    rs, rt = mech["resample_same_ordering"], mech["resample_keep_top3"]
    W(f"  - Correctness is not a function of the ordering: redrawing the values with the ordering of (x, 0) fixed flips")
    W(f"    correctness on {rs['flip']:.3f} of inputs; of the errors only {rs['error_both'] / rs['error']:.0%} persist; the"
      f" activation pattern changes on {1 - rs['pattern_unchanged']:.0%} of inputs")
    W(f"    ({rs['neuron_steps_changed']:.1f} neuron-steps on average). The model compares values with tilted weights.")
    W(f"  - Nor of the top three values and positions: redrawing the other seven flips {rt['flip']:.3f} of inputs"
      f" ({rt['error_both'] / rs['error']:.0%} of errors persist).")
    lv = mech["line_variance"]
    W(f"  - Exact integration along the last input (lastline.py) cuts the per-input variance from {lv['indicator']:.4f} to"
      f" {lv['exact_last_1']:.4f}; over the last two")
    W(f"    and three inputs to {lv['exact_last_2']:.4f} and {lv['exact_last_3']:.4f}, at a cost that multiplies with each"
      " input. The remaining 7-9 dimensions need a point set.")
    if ex_mech is not None:
        pt = ex_mech["patterns"]
        W(f"  - Activation patterns (challenge model, {pt['inputs']} inputs): {pt['distinct']} distinct;"
          f" {pt['unseen_mass']:.0%} of the probability is on patterns never")
        W(f"    seen (Good-Turing); the 100,000 commonest cover {pt['cover']['100000']:.0%}. Cone enumeration is out of reach.")
        if "one_node_per_ordering" in ex_mech:
            W(f"  - One canonical point per ordering of the first nine inputs (all 9! orderings, median order statistics,")
            W(f"    exact last input; aqn.py) gives {ex_mech['one_node_per_ordering']:.3f} against {suite_rec['ref']:.3f}, at the"
              f" cost of {ex_mech['one_node_ops'] / 5265:.0f} inputs: the")
            W("    errors live at small gaps that the canonical points miss.")
        W("  - ARC's five features fit their neurons (R^2 0.94-0.995) but substituting them drops accuracy to 0.31 (all five)")
        W("    or 0.37-0.93 (one at a time); they are exact at steps 1-4 and fail at 6-10 (local agent, m16_features.py).")
        nd = ex_mech["noise"]
        W("  - Weight noise: mean accuracy of copies " + ", ".join(f"sd {k}: {v[0]:.3f} (+- {v[1]:.3f})" for k, v in nd.items())
          + f" against {suite_rec['ref']:.3f}.")
    W("")
    return L


def scores(L, name, curves, suite_rec, mech, ex_noise, ref_example):
    W = L.append
    W("ESTIMATORS AND ERROR AGAINST COMPUTE")
    W("  const_task    1.0 (the model computes the task): 0 ops. Mechanistic, trivially.")
    W("  halton_line N exact line probability averaged over the first N points of the 9-dim Halton sequence:")
    W("                deterministic, but the 9-dim average over a fixed point set is sampling in all but name.")
    W("  mc_line N     the same over N random prefixes (Rao-Blackwellized sampling), for reference.")
    for group, label in (("trained", "the six trained models (challenge model + zoo seeds 0-4)"),
                         ("noisy_sd0.002", "copies of the challenge model with N(0, 0.002^2) weight noise"),
                         ("noisy_sd0.005", "copies with N(0, 0.005^2) weight noise")):
        C = [r for r in curves if r["group"] == group]
        if not C:
            continue
        W(f"  RMS over {label} ({int(C[0]['models'])} models; per-input sd {C[0]['per_input_sd']:.3f};"
          f" one sampled input = {int(C[0]['sample_cost'])} ops):")
        W(f"    {'estimator':<16}{'ops':>11}{'inputs-eq':>11}{'RMS':>10}{'sampling':>10}")
        for r in C:
            if r["estimator"].startswith(("halton_", "mc_")):
                N = int(r["estimator"].split("_")[1])
                if N not in (1, 4, 16, 64, 256, 1024, 4096, 16384):
                    continue
            W(f"    {r['estimator']:<16}{r['ops']:>11.0f}{r['sample_equivalents']:>11.1f}{r['rms']:>10.5f}"
              f"{r['sampling_rms_same_ops']:>10.5f}")
    t = [r for r in curves if r["group"] == "trained"]
    ct = next(r for r in t if r["estimator"] == "const_task")
    pq = t[0]["per_input_sd"] ** 2
    kmax = pq / ct["rms"] ** 2
    W("")
    W("SCORE")
    W("  (a) Error against compute: NOT MET by a mechanistic estimator. The only mechanistic estimate here, 'the model")
    W(f"      computes the task' (1.0), has RMS {ct['rms']:.3f} over the six trained models and beats sampling only while")
    W(f"      sampling has fewer than {kmax:.0f} inputs ({kmax * t[0]['sample_cost']:.0f} ops). The deterministic Halton +"
      " exact-line estimator")
    W("      is not mechanistic; its rows above show what the exact line integration buys over plain sampling:")
    for group in ("trained", "noisy_sd0.002", "noisy_sd0.005"):
        G = [r for r in curves if r["group"] == group and r["estimator"].startswith("halton_")]
        if not G:
            continue
        wins = [int(r["estimator"].split("_")[1]) for r in G if r["rms"] < r["sampling_rms_same_ops"]]
        loses = [int(r["estimator"].split("_")[1]) for r in G if r["rms"] >= r["sampling_rms_same_ops"]]
        big = [r["rms"] / r["sampling_rms_same_ops"] for r in G if int(r["estimator"].split("_")[1]) >= 1024]
        W(f"      {group}: beats sampling at N = {wins}, loses at N = {loses}; RMS ratio {min(big):.2f}-{max(big):.2f}"
          " from N = 1024 up.")
    nd = mech.get("noise") or ex_noise
    W("  (b) Surprise accounting: NOT MET. There is no compact explanation: the partial algorithm above leaves the")
    W("      second maximum's position code (spread over 13 neurons) and the value-dependent neurons unexplained,")
    W(f"      and the accuracy depends on fine weight details (noise of sd 0.002 on every weight costs"
      f" {100 * (ref_example - nd['0.002'][0]):.1f} points).")
    W("")
    W("Files: m16_mechanism.py (tables), m16_suite.py (references), m16_estimators.py, lastline.py; results/")
    W(f"m16_mechanism_{name}.json, m16_estimators.tsv, m16_curves.tsv.")
    return L


def main():
    suite = {json.loads(line)["name"] if json.loads(line)["name"] != "example_noisy" else None: json.loads(line)
             for line in open(m16_suite.OUT)}
    curves = read_tsv(os.path.join(RES, "m16_curves.tsv"))
    ex = json.load(open(os.path.join(RES, "m16_mechanism_example.json")))
    import aqn
    import m16_mechanism
    import zoo
    extra = os.path.join(RES, "m16_patterns_noise_example.json")
    have = json.load(open(extra)) if os.path.exists(extra) else {}
    mm = zoo.load("example", 16, 10)
    rng = np.random.default_rng(77)
    if "patterns" not in have:
        have["patterns"] = m16_mechanism.patterns(mm, rng)
    if "interchange_max" not in have:
        have["interchange_max"] = m16_mechanism.interchange_max_position(mm, rng)
    if "noise" not in have:
        have["noise"] = {str(sd): m16_mechanism.noise_drop(mm, sd, rng) for sd in (0.002, 0.005, 0.05)}
    if "one_node_per_ordering" not in have:
        ops = aqn.Ops()
        have["one_node_per_ordering"] = aqn.accuracy(mm, 1, ops)
        have["one_node_ops"] = ops.n
    json.dump(have, open(extra, "w"), indent=1)
    ex.update(json.load(open(extra)))
    specs = [("example", "example_2nd_argmax_16_10.txt",
              "AlgZoo challenge model M16,10 (example_2nd_argmax, ARC's 432-parameter model, not part of the zoo).")]
    specs += [(f"zoo_seed{s}", f"2nd_argmax_16_10_seed{s}.txt",
               f"AlgZoo 2nd_argmax hidden=16 seq_len=10, seed {s}" + (" (the zoo's best-of-5)" if s == 2 else "")
               + ": a seed of the challenge.") for s in range(5)]
    for name, fname, title in specs:
        p = os.path.join(RES, f"m16_mechanism_{name}.json")
        if name not in suite or not os.path.exists(p):
            continue
        mech = ex if name == "example" else json.load(open(p))
        L = write(name, title, mech, suite[name], curves, ex if name == "example" else None)
        scores(L, name, curves, suite[name], mech, ex["noise"], suite["example"]["ref"])
        with open(os.path.join(RES, fname), "w") as f:
            f.write("\n".join(L) + "\n")
        print(fname)


if __name__ == "__main__":
    main()
