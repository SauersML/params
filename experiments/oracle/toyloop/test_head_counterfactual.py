"""Small tensor contracts only: no model loading, decomposition or scoring."""
from types import SimpleNamespace

import pytest
import torch

from head_counterfactual import scaled_head


def tiny_native():
    v = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 7
    u = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 11
    remainder = torch.eye(4, dtype=torch.float64) / 3
    site = SimpleNamespace(W=(v @ u).T + remainder.T, V=v, U=u, delta_T=remainder)
    target = SimpleNamespace(n_layer=1, n_head=2, hd=2,
                             site=lambda name: site if name == 'h.0.attn.o_proj' else None)
    nat = SimpleNamespace(target=target, units={'h.0.attn.o_proj': torch.tensor([0, 1, 1])},
                          _refs={'old': 1}, _refs_bytes=1, _rec=('old', 'old', {}),
                          _sensitivity_kept={'old': 1})
    return nat, site


@pytest.mark.parametrize('scale', [0.0, 0.5, 1.0])
def test_fixed_basis_head_edit_matches_dense_and_restores(scale):
    nat, site = tiny_native()
    before = {key: getattr(site, key).clone() for key in ('W', 'V', 'U', 'delta_T')}
    original_units = nat.units['h.0.attn.o_proj'].clone()
    x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
    with scaled_head(nat, 0, 1, scale) as evidence:
        wanted = before['W'].clone()
        wanted[:, 2:] *= scale
        assert torch.equal(site.W, wanted)
        dense = x @ site.W.T
        fixed_basis = (x @ site.V) @ site.U + x @ site.delta_T
        assert torch.allclose(dense, fixed_basis, rtol=1e-12, atol=1e-12)
        assert torch.equal(site.U, before['U'])
        assert torch.equal(nat.units['h.0.attn.o_proj'], original_units)
        assert not nat._refs and nat._rec == (None, None, {})
        assert '_sensitivity_kept' not in nat.__dict__
        assert evidence['scale'] == scale
        nat._refs['edited'] = 2
    assert not nat._refs and nat._refs_bytes == 0
    assert all(torch.equal(getattr(site, key), value) for key, value in before.items())


def test_head_edit_exception_restores_and_clears_edited_cache():
    nat, site = tiny_native()
    before = site.W.clone()
    with pytest.raises(RuntimeError, match='interrupted'):
        with scaled_head(nat, 0, 0, 0.0):
            nat._refs['edited'] = 2
            raise RuntimeError('interrupted')
    assert torch.equal(site.W, before)
    assert not nat._refs


def test_nonfinite_scale_and_outside_head_rejected_before_mutation():
    nat, site = tiny_native()
    before = site.W.clone()
    with pytest.raises(ValueError, match='finite'):
        with scaled_head(nat, 0, 0, float('nan')):
            pass
    with pytest.raises(ValueError, match='architecture'):
        with scaled_head(nat, 0, 2, 0.0):
            pass
    assert torch.equal(site.W, before)
