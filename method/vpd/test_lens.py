"""The J++ lens's forward pass and its batched Jacobian averages on a small random model, against plain autograd."""
import unittest

import torch

from method.vpd import lens
from method.vpd.test_atlas import small


class OneCluster:
    """Every source to expert 0."""

    def __call__(self, h):
        return torch.zeros(h.shape[0], dtype=torch.long)


class TestLens(unittest.TestCase):
    def test_forward_matches_the_model(self):
        t = small()
        ids = torch.randint(0, 11, (2, 12), generator=torch.Generator().manual_seed(1))
        for lrp in (False, True):
            final, pts = lens.forward_points(t, ids, lrp=lrp)
            torch.testing.assert_close(final, pts["out_2"])
            from method.vpd import vpd_model as VM
            # vpd_model.rms normalises in float32 whatever the dtype, so the model's own pass agrees to float32.
            torch.testing.assert_close(VM.rms(final, t.ln_f, t.eps), t.hidden(ids), rtol=1e-5, atol=1e-6)

    def test_jacobian_average(self):
        t = small()
        ids = torch.randint(0, 11, (1, 12), generator=torch.Generator().manual_seed(2))
        old = lens.POINTS, lens.SKIP
        lens.POINTS, lens.SKIP = ["mid_1", "out_0"], 2
        try:
            routers = {p: OneCluster() for p in lens.POINTS}
            J, n = lens.fit_jacobians(t, ids, routers, "cpu", chunk=3, log=lambda m: None)
            # Explicit average with the LRP rules: d (sum_t final_t) / d point_s via one backward per output dimension.
            final, pts = lens.forward_points(t, ids, lrp=True, grad=True)
            Tn, d = ids.shape[1], final.shape[-1]
            for p in lens.POINTS:
                want = torch.zeros(d, d, dtype=torch.float64)
                for k in range(d):
                    g, = torch.autograd.grad(final[0, :, k].sum(), pts[p], retain_graph=True)
                    for s in range(lens.SKIP, Tn):
                        want[k] += g[0, s].double() / (Tn - s)
                torch.testing.assert_close(J[p][0], want)
                self.assertEqual(float(n[p][0]), Tn - lens.SKIP)
        finally:
            lens.POINTS, lens.SKIP = old


if __name__ == "__main__":
    unittest.main()
