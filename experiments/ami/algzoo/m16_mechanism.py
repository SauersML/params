"""What the 432-parameter challenge model M16,10 (example_2nd_argmax; also the zoo's 2nd_argmax 16,10 seeds) does, as
measured facts with interventions. Writes results/m16_mechanism_<name>.json.

Measurements (all on fresh standard Gaussian inputs; sampling here is evidence about the mechanism, not an estimator):
  rates          share of inputs on which each neuron is on, at each step
  events         how well the relative rank of the newest input (among the earlier inputs and 0), alone or with the
                 previous input's, predicts each neuron's on/off state at the last step; and a depth-4 decision tree
                 on all relative and final ranks and signs (held-out half)
  clock          P(neuron on at the last step | steps since the running maximum was set)
  shift          T = W_oh W_hh W_oh^+ : how the recurrence moves the logit vector (a shift by one position, if the
                 positions move along with the sequence)
  errors         where the errors are: the predicted rank, the number of positive inputs, the gaps between the top three
  resample       resampling the inputs while keeping (i) the ordering of all inputs and 0, or (ii) the top three values
                 and positions: how often correctness and the activation pattern change
  interchange    moving the second largest value to another position (a new label) and copying a group of neurons
                 from that input's state into the original run (at the last step, or two or four steps before the end
                 when both moved positions lie before): how often the output follows the new label
  intermediate   accuracy of the logits read after t steps on the zero-padded prefix, and the step-to-step switches
  line_variance  variance over prefixes of the exact probability along the last input (lastline.py) and after exact
                 integration over the last 2 and 3 inputs (Gauss-Hermite in the extra inputs)
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lastline  # noqa: E402
import m43  # noqa: E402
import zoo  # noqa: E402

GROUPS = {"clock": [0, 5, 9, 10, 11], "max": [2, 4, 7], "loo": [1, 6, 8], "value": [3, 15], "rest": [12, 13, 14]}


def rel_ranks(x):
    """r[:, s] = number of elements of {x_0..x_s, 0} larger than x_s."""
    B, n = x.shape
    R = np.zeros((B, n), int)
    for s in range(n):
        pre = np.concatenate([x[:, : s + 1], np.zeros((B, 1))], 1)
        R[:, s] = (pre > x[:, s : s + 1]).sum(1)
    return R


def run(m, x, hook=None):
    h = np.zeros((x.shape[0], m.d))
    hs = []
    for t in range(m.n):
        h = np.maximum(h @ m.W_hh.T + x[:, t : t + 1] * m.W_hi, 0.0)
        if hook is not None:
            h = hook(t, h)
        hs.append(h)
    return h @ m.W_oh.T, hs


def measure(m, rng, B=1 << 18):
    n, d = m.n, m.d
    out = {}
    x = rng.standard_normal((B, n))
    lab = zoo.label("2nd_argmax", x)
    L, hs = run(m, x)
    pred = L.argmax(1)
    err = pred != lab
    out["accuracy_sampled"] = float(1 - err.mean())
    out["rates"] = np.array([(h > 0).mean(0) for h in hs]).round(4).tolist()
    # events
    R = rel_ranks(x)
    on = hs[-1] > 0
    ev = []
    o = np.argsort(-x, 1)
    grank = np.argsort(o, 1)
    F = np.concatenate([R, grank, (x > 0).astype(int)], 1)
    for j in range(d):
        y = on[:, j]
        base = max(y.mean(), 1 - y.mean())

        def purity(key):
            tot = 0.0
            for v in np.unique(key):
                s = key == v
                f = y[s].mean()
                tot += s.sum() * max(f, 1 - f)
            return tot / B

        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50).fit(F[: B // 2], y[: B // 2])
        ev.append(dict(neuron=j, on=float(y.mean()), majority=float(base), by_rank_newest=float(purity(R[:, -1])),
                       by_rank_two_newest=float(purity(R[:, -1] * 20 + np.minimum(R[:, -2], 19))),
                       tree_depth4_heldout=float(tree.score(F[B // 2:], y[B // 2:]))))
    out["events"] = ev
    # clock
    a = x.argmax(1)
    age = (n - 1) - a
    age[x.max(1) <= 0] = -1
    out["clock"] = {int(k): on[age == k].mean(0).round(3).tolist() for k in range(-1, n) if (age == k).any()}
    # shift
    T = m.W_oh @ m.W_hh @ np.linalg.pinv(m.W_oh)
    out["shift_T"] = T.round(3).tolist()
    out["shift_superdiag"] = np.diag(T, 1).round(3).tolist()
    out["shift_resid_rel"] = float(np.linalg.norm(m.W_oh @ m.W_hh - T @ m.W_oh) / np.linalg.norm(m.W_oh @ m.W_hh))
    # errors
    rank_of = np.argsort(o, 1)
    pr = rank_of[np.arange(B), pred]
    ys = -np.sort(-x, 1)
    npos = (x > 0).sum(1)
    out["errors"] = {
        "share_by_predicted_rank": (np.bincount(pr[err], minlength=n) / max(err.sum(), 1)).round(4).tolist(),
        "rate_by_npos": {int(k): [float((npos == k).mean()), float(err[npos == k].mean())] for k in range(n + 1)
                         if (npos == k).sum() > 100},
        "rate_by_gap23": {str(q): [float((ys[:, 1] - ys[:, 2] < q).mean()), float(err[ys[:, 1] - ys[:, 2] < q].mean())]
                          for q in (0.01, 0.03, 0.1, 0.3, 1.0)},
        "rate_by_gap12": {str(q): [float((ys[:, 0] - ys[:, 1] < q).mean()), float(err[ys[:, 0] - ys[:, 1] < q].mean())]
                          for q in (0.01, 0.03, 0.1, 0.3, 1.0)},
    }
    # resample keeping the ordering of (x, 0)
    H = np.abs(rng.standard_normal((B, n)))
    k = npos
    jj = np.arange(n)[None]
    Hp = -np.sort(-np.where(jj < k[:, None], H, -np.inf), 1)
    Hn = np.sort(np.where(jj >= k[:, None], H, np.inf), 1)
    vals = np.where(jj < k[:, None], Hp, -Hn[np.arange(B)[:, None], np.clip(jj - k[:, None], 0, n - 1)])
    x2 = np.empty_like(x)
    np.put_along_axis(x2, o, vals, 1)
    L2, hs2 = run(m, x2)
    e2 = L2.argmax(1) != lab
    p1 = np.concatenate([h > 0 for h in hs], 1)
    p2 = np.concatenate([h > 0 for h in hs2], 1)
    out["resample_same_ordering"] = dict(error=float(err.mean()), error_after=float(e2.mean()),
                                         flip=float((err != e2).mean()), error_both=float((err & e2).mean()),
                                         pattern_unchanged=float((p1 == p2).all(1).mean()),
                                         neuron_steps_changed=float((p1 != p2).sum(1).mean()),
                                         change_by_neuron=(p1 != p2).reshape(B, n, d).mean((0, 1)).round(4).tolist())
    # resample keeping the top three values and positions
    thr = ys[:, 2]
    u = rng.random((B, n))
    v = norm.ppf(u * norm.cdf(thr)[:, None])
    keep = rank_of < 3
    x3 = np.where(keep, x, v)
    e3 = run(m, x3)[0].argmax(1) != lab
    out["resample_keep_top3"] = dict(flip=float((err != e3).mean()), error_both=float((err & e3).mean()))
    # interchange
    ar = np.arange(B)
    bg = o[ar, 2 + rng.integers(0, n - 2, B)]
    xs = x.copy()
    xs[ar, o[:, 1]] = x[ar, bg]
    xs[ar, bg] = x[ar, o[:, 1]]
    labs = bg
    Ls, hss = run(m, xs)
    ok = (pred == lab) & (Ls.argmax(1) == labs)
    groups = dict(GROUPS)
    groups["clock+value"] = GROUPS["clock"] + GROUPS["value"]
    groups["all_but_max"] = [j for j in range(d) if j not in GROUPS["max"]]
    iv = {}
    for name, S in groups.items():
        hh = hs[-1].copy()
        hh[:, S] = hss[-1][:, S]
        p = (hh @ m.W_oh.T).argmax(1)
        rec = {"at_last_step": [float((p == labs)[ok].mean()), float((p == lab)[ok].mean())]}
        for back in (2, 4):
            t = n - back
            sel = ok & (np.maximum(o[:, 1], bg) < t)

            def hook(tt, h, S=S, t=t):
                if tt == t - 1:
                    h = h.copy()
                    h[:, S] = hss[t - 1][:, S]
                return h

            p = run(m, x, hook)[0].argmax(1)
            rec[f"{back}_steps_before_end"] = [float((p == labs)[sel].mean()), float((p == lab)[sel].mean()),
                                               float(sel.mean())]
        iv[name] = rec
    out["interchange_second_max_position"] = iv
    out["interchange_note"] = ("[P(output = new label), P(output = old label)] among pairs where the model is right on "
                               "both inputs; for the earlier steps also the share of pairs used")
    # intermediate accuracy
    inter = []
    prev = None
    for t in range(2, n + 1):
        xp = np.concatenate([np.zeros((B, n - t)), x[:, :t]], 1)
        lt = np.argsort(xp, 1, kind="stable")[:, -2]
        two = (x[:, :t] > 0).sum(1) >= 2
        okt = (hs[t - 1] @ m.W_oh.T).argmax(1) == lt
        rec = dict(t=t, accuracy=float(okt[two].mean()), share_defined=float(two.mean()))
        if prev is not None:
            pok, ptwo = prev
            s = pok & two & ptwo
            rec["p_wrong_given_right_before"] = float((~okt[s]).mean())
            rec["p_right_given_wrong_before"] = float(okt[(~pok) & two & ptwo].mean())
        inter.append(rec)
        prev = (okt, two)
    out["intermediate"] = inter
    # variance of exact line probabilities
    Bv = 1 << 12
    gx, gw = np.polynomial.hermite_e.hermegauss(48)
    gw = gw / gw.sum()
    lv = {}
    xp = rng.standard_normal((Bv, n - 1))
    g1, _ = lastline.line_prob(m, xp)
    lv["indicator"] = float(out["accuracy_sampled"] * (1 - out["accuracy_sampled"]))
    lv["exact_last_1"] = float(g1.var())
    xp2 = rng.standard_normal((Bv, n - 2))
    g2 = np.zeros(Bv)
    for xi, wi in zip(gx, gw):
        g2 += wi * lastline.line_prob(m, np.concatenate([xp2, np.full((Bv, 1), xi)], 1))[0]
    lv["exact_last_2"] = float(g2.var())
    xp3 = rng.standard_normal((Bv, n - 3))
    g3 = np.zeros(Bv)
    gx3, gw3 = np.polynomial.hermite_e.hermegauss(24)
    gw3 = gw3 / gw3.sum()
    for xi, wi in zip(gx3, gw3):
        for xj, wj in zip(gx3, gw3):
            g3 += wi * wj * lastline.line_prob(m, np.concatenate([xp3, np.full((Bv, 1), xi), np.full((Bv, 1), xj)], 1))[0]
    lv["exact_last_3"] = float(g3.var())
    out["line_variance"] = lv
    return out


def patterns(m, rng, B=1 << 20):
    """Distinct activation patterns (all neurons, all steps) among B inputs, the Good-Turing estimate of the
    probability on patterns never seen (singletons / B), and the share of inputs covered by the commonest k."""
    keys = []
    for _ in range(B // (1 << 17)):
        x = rng.standard_normal((1 << 17, m.n))
        keys.append(np.packbits(np.concatenate([h > 0 for h in m.hidden(x)], 1), axis=1))
    _, cnt = np.unique(np.concatenate(keys), axis=0, return_counts=True)
    cnt = np.sort(cnt)[::-1]
    cs = np.cumsum(cnt) / B
    return dict(inputs=B, distinct=int(len(cnt)), unseen_mass=float((cnt == 1).sum() / B),
                cover={str(k): float(cs[min(k, len(cs)) - 1]) for k in (100, 1000, 10000, 100000)})


def interchange_max_position(m, rng, B=1 << 17):
    """Move the maximum to another position (the label stays) and copy a group of neurons from that run into the
    original run at the last step: share of outputs at the original maximum's position, at the label, and at the
    maximum's new position (pairs where the model is right on both inputs and the maximum is positive)."""
    x = rng.standard_normal((B, m.n))
    ar = np.arange(B)
    o = np.argsort(-x, 1)
    a, b = o[:, 0], o[:, 1]
    cand = o[ar, 2 + rng.integers(0, m.n - 2, B)]
    x2 = x.copy()
    x2[ar, a] = x[ar, cand]
    x2[ar, cand] = x[ar, a]
    L1, hs1 = run(m, x)
    L2, hs2 = run(m, x2)
    ok = (L1.argmax(1) == b) & (L2.argmax(1) == b) & (x[ar, a] > 0)
    groups = dict(GROUPS)
    groups["all_but_max"] = [j for j in range(m.d) if j not in GROUPS["max"]]
    out = {}
    for name, S in groups.items():
        h = hs1[-1].copy()
        h[:, S] = hs2[-1][:, S]
        p = (h @ m.W_oh.T).argmax(1)
        out[name] = [float((p == a)[ok].mean()), float((p == b)[ok].mean()), float((p == cand)[ok].mean())]
    return out


def noise_drop(m, sd, rng, K=32, B=1 << 16):
    v = m.flat()
    acc = [zoo.sampled_metric(m.with_flat(v + sd * rng.standard_normal(v.shape)), "2nd_argmax", m.n, B, rng)[0]
           for _ in range(K)]
    return float(np.mean(acc)), float(np.std(acc))


def main():
    names = sys.argv[1:] or ["example"]
    for name in names:
        if name == "example":
            m = zoo.load("example", 16, 10)
        else:
            m = zoo.load("2nd_argmax", 16, 10, int(name.replace("zoo_seed", "")))
        out = measure(m, np.random.default_rng(2026))
        with open(os.path.join(m43.RES, f"m16_mechanism_{name}.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(name, json.dumps({k: out[k] for k in ("accuracy_sampled", "resample_same_ordering", "resample_keep_top3",
                                                    "line_variance")}), flush=True)


if __name__ == "__main__":
    main()
