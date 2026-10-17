"""Test ARC's described features of the 432-parameter challenge model M16,10 (example_2nd_argmax) by fit and by
substitution.

ARC's post (Hilton, Jan 2026) states, for the hidden state h_t after reading x_0..x_(t-1), with r(...) = max(0, ...):
  neuron 2 ~ r(x_0..x_(t-2))                       (running max, one step late)
  neuron 4 ~ r(x_0..x_(t-1)) - r(x_0..x_(t-2))     (increase of the running max at the last step)
  neuron 6 ~ r(x_0..x_(t-3), x_(t-1)) - x_(t-1)    (max leaving out x_(t-2), minus the last input)
  neuron 7 ~ r(x_0..x_(t-1)) - x_(t-1)             (running max minus the last input)
  neuron 1 ~ r(x_0..x_(t-4), x_(t-2), x_(t-1)) - x_(t-1)  (max leaving out x_(t-3), minus the last input)
For each neuron: the scale c fitted by least squares through 0 over all steps, R^2, and the effect of substituting
c * feature for the neuron at every step (the model then runs on): accuracy, and KL(model || substituted) of the
output distribution. Substitution of the feature computed from a different input is the interchange control.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoo  # noqa: E402


def rmax(x, idx):
    """max(0, x_i for i in idx) per row."""
    if not idx:
        return np.zeros(x.shape[0])
    return np.maximum(x[:, idx].max(1), 0.0)


FEATURES = {
    2: lambda x, t: rmax(x, list(range(0, t - 1))),
    4: lambda x, t: rmax(x, list(range(0, t))) - rmax(x, list(range(0, t - 1))),
    6: lambda x, t: rmax(x, list(range(0, t - 2)) + ([t - 1] if t >= 1 else [])) - (x[:, t - 1] if t >= 1 else 0.0),
    7: lambda x, t: rmax(x, list(range(0, t))) - (x[:, t - 1] if t >= 1 else 0.0),
    1: lambda x, t: rmax(x, list(range(0, t - 3)) + [i for i in (t - 2, t - 1) if i >= 0])
    - (x[:, t - 1] if t >= 1 else 0.0),
}


def run_substituted(m, x, subs):
    """Forward pass where, for neuron j in subs, h_t[j] is replaced by subs[j][t] (array (B,)) at every step t>=1."""
    B = x.shape[0]
    h = np.zeros((B, m.d))
    for t in range(m.n):
        h = np.maximum(h @ m.W_hh.T + x[:, t : t + 1] * m.W_hi, 0.0)
        for j, vals in subs.items():
            h[:, j] = vals[t + 1]
    return h @ m.W_oh.T


def logsoftmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def kl(base_logits, new_logits):
    lp, lq = logsoftmax(base_logits), logsoftmax(new_logits)
    return float((np.exp(lp) * (lp - lq)).sum(1).mean())


def main():
    m = zoo.load_rnn("example", 16, 10)
    rng = np.random.default_rng(0)
    B = 1 << 18
    x = rng.standard_normal((B, 10))
    x2 = rng.standard_normal((B, 10))
    y = zoo.label("2nd_argmax", x)
    hs = m.hidden(x)
    base = hs[-1] @ m.W_oh.T
    out = {"base_accuracy": float((base.argmax(1) == y).mean())}
    fitted = {}
    for j, f in FEATURES.items():
        F = np.stack([f(x, t) for t in range(1, 11)], 1)  # h_1..h_10
        Hj = np.stack([h[:, j] for h in hs], 1)
        c = float((F * Hj).sum() / (F * F).sum())
        res = Hj - c * F
        r2 = 1 - float((res ** 2).sum() / ((Hj - Hj.mean()) ** 2).sum())
        r2_t = [1 - float(((Hj[:, t] - c * F[:, t]) ** 2).sum() / max(((Hj[:, t] - Hj[:, t].mean()) ** 2).sum(), 1e-300))
                for t in range(10)]
        vals = {t + 1: c * F[:, t] for t in range(10)}
        F2 = np.stack([f(x2, t) for t in range(1, 11)], 1)
        vals_swap = {t + 1: c * F2[:, t] for t in range(10)}
        sub = run_substituted(m, x, {j: vals})
        swap = run_substituted(m, x, {j: vals_swap})
        mean_ablate = run_substituted(m, x, {j: {t + 1: np.full(B, Hj[:, t].mean()) for t in range(10)}})
        fitted[j] = vals
        out[f"neuron{j}"] = {
            "scale": c, "r2_all_steps": r2, "r2_per_step": r2_t,
            "acc_substituted": float((sub.argmax(1) == y).mean()), "kl_substituted": kl(base, sub),
            "acc_feature_from_other_input": float((swap.argmax(1) == y).mean()),
            "kl_feature_from_other_input": kl(base, swap),
            "acc_mean_ablated": float((mean_ablate.argmax(1) == y).mean()), "kl_mean_ablated": kl(base, mean_ablate),
        }
        print(j, json.dumps(out[f"neuron{j}"]), flush=True)
    allsub = run_substituted(m, x, fitted)
    out["all_five_substituted"] = {"acc": float((allsub.argmax(1) == y).mean()), "kl": kl(base, allsub)}
    print(json.dumps(out["all_five_substituted"]))
    with open(os.path.expanduser("~/mpd-data/ami/algzoo/m16_features.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
