"""One table over all 116 AlgZoo models: the score under ARC's two criteria and where the explanation lives.

Reads inventory.tsv and results_exact.jsonl / results_exactn.jsonl / results_cycle.jsonl in ~/mpd-data/ami/algzoo
and writes summary.tsv there.
  a_error_vs_compute: MET (an exact mechanistic estimate whose error is below sampling's at the estimate's own
    compute; below that budget the comparison is open) or OPEN (no mechanistic estimate yet).
  b_surprise: MET (comparable to ARC's accepted explanation), UNDETERMINED (fewer sign facts than float32 weight
    bits; optimization bits unknown) or NOT_MET (more facts than the weights hold), or OPEN.
"""
import json
import os

DIR = os.path.expanduser("~/mpd-data/ami/algzoo")


def load(name):
    p = os.path.join(DIR, name)
    out = {}
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            out[(r["task"], r["d"], r["n"])] = r
    return out


def main():
    res = {}
    res.update(load("results_exact.jsonl"))
    res.update(load("results_exactn.jsonl"))
    cyc = load("results_cycle.jsonl")
    rows = []
    for line in list(open(os.path.join(DIR, "inventory.tsv")))[1:]:
        p, task, d, n, seed, metric, value, se = line.rstrip("\n").split("\t")
        d, n = int(d), int(n)
        key = (task, d, n)
        a, b, method, exact, err_bound, samp_rms, est_samples, bits, wbits = "OPEN", "OPEN", "", "", "", "", "", "", 32 * int(p)
        if key in res:
            r = res[key]
            a = "MET"
            method = "exact cone integration"
            exact = f"{r['exact']:.10g}"
            err_bound = f"{r.get('unresolved_bound', 0.0):.1g}"
            samp_rms = f"{r['noisy']['sampling_rms_at_equal_ops']:.2g}"
            est_samples = f"{r['noisy']['samples_at_equal_ops']:.3g}"
            bits = r["surprise_sign_bits"]
        elif key in cyc:
            r = cyc[key]
            a = "MET"
            method = "exhaustive enumeration of all n^n inputs"
            exact = f"{r['accuracy']:.10g}"
            err_bound = "0"
            samp_rms = f"{r['noisy']['sampling_rms_at_equal_ops']:.2g}"
            est_samples = str(r["inputs"])
            bits = round(r["surprise_bits_enumeration"])
        if bits != "":
            if bits > wbits:
                b = "NOT_MET"
            elif key in (("2nd_argmax", 2, 2), ("median", 2, 2)):
                b = "MET"  # compact explanations comparable to ARC's accepted M2,2 analysis (see the model files)
            else:
                b = "UNDETERMINED"
        fname = f"{task}_{d}_{n}.txt"
        rows.append((int(p), task, d, n, int(seed), metric, float(value), a, b, method, exact, err_bound, est_samples,
                     samp_rms, bits, wbits, fname if os.path.exists(os.path.join(DIR, fname)) else ""))
    rows.sort()
    head = ("params\ttask\thidden\tseq_len\tseed\tmetric\tvalue_sampled\ta_error_vs_compute\tb_surprise\tmethod\texact"
            "\tcertified_error_bound\testimate_cost_in_samples\tsampling_rms_at_that_cost\tsurprise_bits"
            "\tweight_bits_float32\tfile")
    with open(os.path.join(DIR, "summary.tsv"), "w") as f:
        f.write(head + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")
    na = sum(r[7] == "MET" for r in rows)
    nb = sum(r[8] == "MET" for r in rows)
    print(f"{len(rows)} models; (a) MET {na}; (b) MET {nb}; (b) UNDETERMINED {sum(r[8] == 'UNDETERMINED' for r in rows)};"
          f" (b) NOT_MET {sum(r[8] == 'NOT_MET' for r in rows)}; OPEN {sum(r[7] == 'OPEN' for r in rows)}")


if __name__ == "__main__":
    main()
