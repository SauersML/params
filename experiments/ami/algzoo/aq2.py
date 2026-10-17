"""Exact accuracy / mean squared error of a length-2 RNN from two lines (no enumeration, no sampling).

Positive homogeneity: the metric depends only on the direction of (x0, x1). Write x = R (s, t) with s = sign(x0),
R = |x0| (half-normal) and t = x1 / |x0| (standard Cauchy, independent of R and s). For each s = +1, -1 the first step
is fixed and the second input enters linearly, so along t every neuron switches once and the logits (or the output)
are linear between switches:
  accuracy: lastline.line_prob with radial=True (Student t with 1 degree of freedom = Cauchy), averaged over s;
  squared error (median task, the target (x0 + x1)/2): E[R^2 q(s,t)^2] = E_s integral q^2 * 2 / (pi (1+t^2)^2) dt,
  with q linear in t between switches, integrated in closed form.
Both are exact; the cost is two lines.
"""
import math

import numpy as np

import lastline


class Ops:
    def __init__(self):
        self.n = 0


def accuracy(model, ops=None, signs=(1.0, -1.0)):
    """signs=(1.0,) uses the line x0 > 0 alone (half the cost; not exact)."""
    ops = ops or Ops()
    g, line_ops = lastline.line_prob(model, np.array([[s] for s in signs]), task=model.task, radial=True)
    ops.n += len(signs) * line_ops + 2
    return float(g.mean())


def _K(t):
    """Antiderivatives of (1+t^2)^-2 times 1, t, t^2."""
    if t == math.inf:
        return (math.pi / 4, 0.0, math.pi / 4)
    if t == -math.inf:
        return (-math.pi / 4, 0.0, -math.pi / 4)
    a, r = math.atan(t), 1.0 + t * t
    return (t / (2 * r) + a / 2, -1.0 / (2 * r), a / 2 - t / (2 * r))


def mse(model, ops=None, signs=(1.0, -1.0)):
    """signs=(1.0,) uses the line x0 > 0 alone (half the cost; not exact)."""
    ops = ops or Ops()
    d, W, w, wo = model.d, model.W_hh, model.W_hi, model.W_oh[0]
    tot = 0.0
    for s in signs:
        h1 = np.maximum(w * s, 0.0)
        c = W @ h1
        ops.n += 2 * d + 2 * d * d
        on = w < 0
        ev = sorted((-c[j] / w[j], j) for j in range(d) if w[j] != 0.0)
        on[w == 0.0] = c[w == 0.0] > 0
        P = float(wo[on] @ c[on])
        Q = float(wo[on] @ w[on])
        ops.n += 2 * d + int(len(ev) * math.log2(max(len(ev), 2))) + 4 * int(on.sum())
        left, Kl = -math.inf, _K(-math.inf)
        for i in range(len(ev) + 1):
            right = ev[i][0] if i < len(ev) else math.inf
            if right > left:
                Kr = _K(right)
                al, be = P - s / 2, Q - 0.5  # q = f - (s + t)/2
                tot += al * al * (Kr[0] - Kl[0]) + 2 * al * be * (Kr[1] - Kl[1]) + be * be * (Kr[2] - Kl[2])
                ops.n += 20
                Kl = Kr
            if i == len(ev):
                break
            j = ev[i][1]
            sg = -1.0 if on[j] else 1.0
            on[j] = not on[j]
            P += sg * wo[j] * c[j]
            Q += sg * wo[j] * w[j]
            ops.n += 6
            left = ev[i][0]
    return tot * (2 / math.pi) / len(signs)
