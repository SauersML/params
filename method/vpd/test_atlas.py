"""The atlas's one-row edit against a full forward with the same edit, on a small random model, without mocks."""
import unittest

import torch

from method.vpd.atlas import clean_cache, row_logprobs
from method.vpd.vpd_model import Target, site_names


def small():
    cfg = dict(n_layer=3, n_head=2, n_embd=8, n_key_value_heads=2, rotary_dim=4, rms_norm_eps=1e-6, n_ctx=12,
               rotary_base=10000)
    gen = torch.Generator().manual_seed(3)
    rand = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    sd = {"wte.weight": rand(11, 8), "ln_f.weight": 1 + 0.1 * rand(8)}
    for i in range(3):
        for j in (1, 2):
            sd[f"h.{i}.rms_{j}.weight"] = 1 + 0.1 * rand(8)
    for n in site_names(3):
        din, dout = (8, 16) if n.endswith("c_fc") else ((16, 8) if n.endswith("down_proj") else (8, 8))
        sd[n + ".weight"] = rand(dout, din) / din ** 0.5
    return Target(sd, cfg).double()


class TestRowEdit(unittest.TestCase):
    def test_row_edit_equals_full_forward(self):
        t = small()
        ids = torch.randint(0, 11, (2, 12), generator=torch.Generator().manual_seed(4))
        cache = clean_cache(t, ids)
        torch.testing.assert_close(cache["logp"], torch.log_softmax(t(ids), -1))
        gen = torch.Generator().manual_seed(5)
        for name in site_names(3):
            layer, kind = int(name.split(".")[1]), name.rsplit(".", 1)[1]
            st = t.site(name)
            b, p = torch.tensor([0, 1, 1]), torch.tensor([3, 7, 11])
            delta = torch.randn(3, st.W.shape[0], generator=gen, dtype=torch.float64)
            got = row_logprobs(t, cache, b, p, layer, kind, delta)
            zero = row_logprobs(t, cache, b, p, layer, kind, torch.zeros_like(delta))
            torch.testing.assert_close(zero, cache["logp"][b, p])
            for e in range(3):
                add = torch.zeros(2, 12, st.W.shape[0], dtype=torch.float64)
                add[b[e], p[e]] = delta[e]
                hook = st.register_forward_hook(lambda m, i, o: o + add)
                try:
                    want = torch.log_softmax(t(ids), -1)[b[e], p[e]]
                finally:
                    hook.remove()
                torch.testing.assert_close(got[e], want, msg=f"{name} row ({b[e]}, {p[e]})")


if __name__ == "__main__":
    unittest.main()
