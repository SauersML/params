"""Deterministic quadrature estimator of 2nd-argmax / argmedian accuracy for any sequence length n (no sampling).

The last input is integrated exactly along its line (lastline.line_prob, Gaussian last input). The first k = n-1
inputs are split into the k! orderings; inside an ordering the label along the line is fixed by the order statistics,
so the line probability is continuous there. The descending order statistics of k standard Gaussians are
y_(j) = Phi^-1(U_(j)) with U_(1) = V_1^(1/k), U_(j) = U_(j-1) V_j^(1/(k-j+1)) for independent uniform V_j (the
largest of k uniforms, then the largest of the remaining k-1 below it, ...), so each ordering's expectation is an
integral over the unit cube, done by a Gauss-Legendre product rule with m nodes per coordinate:
    accuracy = (1/k!) sum over orderings pi of sum over nodes of w * line_prob(y placed by pi).
Cost: k! m^k lines. Ops as in aq3.py; per node the line's ops plus 6 per coordinate for the order statistic.
"""
import itertools
import math

import numpy as np
from scipy.special import ndtri

import lastline


class Ops:
    def __init__(self):
        self.n = 0


def order_stat_nodes(k, m):
    """Descending Gaussian order statistics at the product-rule nodes: (m^k, k) values and (m^k,) weights summing
    to 1."""
    g, w = np.polynomial.legendre.leggauss(m)
    g, w = 0.5 * (g + 1), 0.5 * w
    grids = np.meshgrid(*[g] * k, indexing="ij")
    wgrid = np.meshgrid(*[w] * k, indexing="ij")
    V = np.stack([a.ravel() for a in grids], 1)
    W = np.prod(np.stack([a.ravel() for a in wgrid], 1), 1)
    U = np.empty_like(V)
    U[:, 0] = V[:, 0] ** (1.0 / k)
    for j in range(1, k):
        U[:, j] = U[:, j - 1] * V[:, j] ** (1.0 / (k - j))
    return ndtri(U), W


def accuracy(model, m, ops=None, task="2nd_argmax"):
    ops = ops or Ops()
    k = model.n - 1
    Y, W = order_stat_nodes(k, m)
    perms = list(itertools.permutations(range(k)))
    X = np.empty((len(perms) * len(W), k))
    for i, p in enumerate(perms):
        X[i * len(W):(i + 1) * len(W), list(p)] = Y  # largest value at position p[0], ...
    g, line_ops = lastline.line_prob(model, X, task=task)
    ops.n += len(X) * (line_ops + 6 * k)
    return float(np.tile(W, len(perms)) @ g) / len(perms)
