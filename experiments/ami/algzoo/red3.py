"""Estimator built on M4,3's compact mechanism (Robinson's pair symmetry): read the 12-number reduced model off the
weights and integrate its accuracy exactly.

Each hidden neuron is rescaled so its input weight is +-1; W_hh, W_hi and W_oh are projected onto the pair-symmetric
form (W_hh ~ U^T W' U, W_hi ~ U^T w', W_oh ~ W'_oh V, least squares, which is entrywise averaging), so the model
becomes logits = W'_oh |A x| with A = [W'^2 w', W' w', w'] (2 x 3). Its accuracy for Gaussian x is exact by spherical
polygons (exact3.py's clipping and areas): the sphere is cut by the two planes (A x)_k = 0 into regions where the logits
are linear, then by the ordering planes, then by the planes where the label's logit ties another.
The estimate ignores the residual of the projection, so its error on a model is the effect of that residual.
Ops: exact3's counts plus the projection.
"""
import numpy as np

import exact3

U = np.array([[1.0, 0, 0, -1], [0, 1, -1, 0]])


def reduce(model, ops):
    s = np.abs(model.W_hi)
    wi = model.W_hi / s
    Wh = model.W_hh * (1 / s)[:, None] * s[None, :]
    Wo = model.W_oh * s[None, :]
    ops.n += 4 + 2 * 16 + 12
    Wp = np.linalg.pinv(U.T) @ Wh @ np.linalg.pinv(U)  # entrywise averages of four entries
    wp = np.linalg.pinv(U.T) @ wi
    Wop = Wo @ np.linalg.pinv(np.array([[1.0, 0, 0, 1], [0, 1, 1, 0]]))
    ops.n += 4 * 4 + 2 * 2 + 6 * 2
    A = np.stack([Wp @ Wp @ wp, Wp @ wp, wp], 1)
    ops.n += 20
    return A, Wop


def accuracy(model, ops=None):
    ops = ops or exact3.Ops()
    A, C = reduce(model, ops)
    total = 0.0
    for tri in exact3.octants():
        parts = [(tri, np.ones(2))]
        for k in range(2):
            nxt = []
            for q, sg in parts:
                pos, neg = exact3.clip(q, A[k], ops)
                if pos is not None:
                    nxt.append((pos, sg.copy()))
                if neg is not None:
                    s2 = sg.copy()
                    s2[k] = -1.0
                    nxt.append((neg, s2))
            parts = nxt
        for q, sg in parts:
            L = C @ (sg[:, None] * A)  # logits linear on this region
            ops.n += 18
            for piece in exact3._by_order(q, ops):
                c = exact3._unit(sum(piece))
                lab = int(np.argsort(c)[1])
                r = piece
                for i in range(3):
                    if i == lab or r is None:
                        continue
                    r, _ = exact3.clip(r, L[lab] - L[i], ops)
                if r is not None:
                    total += exact3.area(r, ops)
    return total / exact3.FOUR_PI
