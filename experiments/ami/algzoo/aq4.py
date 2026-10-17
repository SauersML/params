"""Quadrature estimator of the accuracy of a 2nd-argmax RNN with sequence length 4 (deterministic, no sampling).

The length-3 estimator (aq3.py) one dimension up. Positive homogeneity: the accuracy is the average over the direction u
of the first three inputs (uniform on the sphere S^2) of the exact probability along the line of the last input
(lastline.line_prob with radial=True: given u, the last input over the prefix's norm has sqrt(3) T ~ Student t_3, the
switch points and logit crossings along the line are exact, and so is the measure of the intervals where the model is
right). The label along the line jumps only where the prefix's order changes, on the three great circles x_i = x_j
(i, j < 3), which all pass through the pole e = (1,1,1)/sqrt 3 and cut the sphere into six lunes of width pi/3. On
each lune the line probability is continuous, and it is integrated with a Gauss-Legendre product rule in longitude
phi (m_phi nodes) and z = sin(latitude) (m_psi nodes; the area element is dz dphi).

Ops: as in aq3.py (scalar add, multiply, divide, compare, elementary function = 1); per node the line's ops
(lastline.line_ops) plus 30 for the node's coordinates and the prefix normalization.
"""
import math

import numpy as np

import lastline

E = np.ones(3) / math.sqrt(3)
F1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
F2 = np.array([1.0, 1.0, -2.0]) / math.sqrt(6)
LUNES = [math.pi / 6 + k * math.pi / 3 for k in range(6)]  # longitudes of the circles x_i = x_j


class Ops:
    def __init__(self):
        self.n = 0


def nodes(m_phi, m_psi):
    """Unit prefixes u (N, 3) and weights w (N,) with sum w g(u) ~ (1 / 4 pi) integral over S^2 of g."""
    gp, wp = np.polynomial.legendre.leggauss(m_phi)
    gs, ws = np.polynomial.legendre.leggauss(m_psi)
    # z = sin(latitude) is uniform in area (dA = dz dphi): Gauss-Legendre in z
    psi = np.arcsin(gs)
    wpsi = ws
    us, w = [], []
    for a in LUNES:
        phi = a + (math.pi / 6) * (gp + 1)
        wphi = (math.pi / 6) * wp
        P, S = np.meshgrid(phi, psi, indexing="ij")
        WP, WS = np.meshgrid(wphi, wpsi, indexing="ij")
        u = (np.sin(S)[..., None] * E + np.cos(S)[..., None] * (np.cos(P)[..., None] * F1 + np.sin(P)[..., None] * F2))
        us.append(u.reshape(-1, 3))
        w.append((WP * WS).ravel())
    return np.concatenate(us), np.concatenate(w) / (4 * math.pi)


def accuracy(model, m_phi, m_psi, ops=None, task="2nd_argmax"):
    ops = ops or Ops()
    assert model.n == 4
    u, w = nodes(m_phi, m_psi)
    g, line_ops = lastline.line_prob(model, u, task=task, radial=True)
    ops.n += len(w) * (line_ops + 30) + 2 * len(w)
    return float(w @ g)


def accuracy_centres(model, lunes, ops=None, task="2nd_argmax"):
    """The cheapest rules: the line probability at the centres (z = 0, mid-longitude) of the listed lunes only,
    averaged (each lune standing for the whole sphere)."""
    ops = ops or Ops()
    u, _ = nodes(1, 1)
    g, line_ops = lastline.line_prob(model, u[list(lunes)], task=task, radial=True)
    ops.n += len(lunes) * (line_ops + 30) + len(lunes)
    return float(g.mean())
