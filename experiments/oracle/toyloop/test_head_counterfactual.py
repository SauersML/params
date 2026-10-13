"""Small tensor contracts only: no model loading, decomposition or scoring."""
from types import SimpleNamespace

import pytest
import torch

from head_counterfactual import (candidate_specs, candidate_summary, compile_candidates, digest,
                                 kl_bits, raw_kl_bits, save_distributions, scaled_head)


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


def test_named_candidates_keep_default_and_reject_unbound_names():
    assert candidate_specs(None) == {'base': 'base.jsonl', 'trained': 'trained.jsonl'}
    assert candidate_specs(['base=base.jsonl', 'multihop_m8=cell.jsonl']) == {
        'base': 'base.jsonl', 'multihop_m8': 'cell.jsonl'}
    for values in (['base=../base.jsonl'], ['base=base.jsonl', 'base=again.jsonl'],
                   ['base=base.jsonl', 'gt=gt.jsonl'], ['cell=cell.jsonl']):
        with pytest.raises(ValueError):
            candidate_specs(values)


def test_invalid_candidates_do_not_recompile_or_remove_common_worlds():
    calls = []

    def compile_graph(graph, ids):
        calls.append((graph, tuple(ids)))
        if graph == 'bad_edge':
            raise ValueError('non-executable edge')
        return (graph, tuple(ids))

    bank = [[0, 1], [0, 2, 3]]
    compiled, invalid = compile_candidates({'base': 'ok', 'bad': 'bad_edge', 'length': None}, bank,
                                          lambda value: value, compile_graph, {'length': 'length stop'})
    assert len(compiled) == len(invalid) == len(bank)
    assert all(set(graphs) == {'base'} for graphs in compiled)
    assert all(reasons == {'length': 'length stop', 'bad': 'non-executable edge'} for reasons in invalid)
    assert len(calls) == 4  # Each original graph once per input; truncated graph never parsed/compiled.


def test_invalid_cf_stays_denominator_and_kl_is_undefined():
    valid = {'valid': True, 'direct_kl_bits': .2, 'conditional_delta_kl_bits': .1,
             'no_change_error_bits': .4, 'top': 3}
    invalid = {'valid': False, 'direct_kl_bits': None, 'top': None}
    selected = [{'reference_top': 3, 'candidates': {'cell': value}} for value in (valid, invalid)]
    result = candidate_summary(selected, 'cell')
    assert result['n'] == 2 and result['valid'] == result['undefined_kl'] == 1
    assert result['mean_direct_kl_bits'] == .2 and result['top_accuracy_all'] == .5
    undefined = candidate_summary(selected[1:], 'cell')
    assert undefined['mean_direct_kl_bits'] is None and undefined['top_accuracy_all'] == 0.


def test_unexpected_executor_failure_is_not_silently_candidate_invalid():
    def broken(graph, ids):
        raise RuntimeError('executor bug')

    with pytest.raises(RuntimeError, match='executor bug'):
        compile_candidates({'base': 'ok'}, [[0, 1]], lambda value: value, broken)


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


def test_reference_cache_rebuilt_and_positive_sparse_route_changes():
    nat, site = tiny_native()
    x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)

    def reference():
        if 'x' not in nat._refs:
            nat._refs['x'] = x @ site.W.T
        return nat._refs['x']

    nat._refs.clear()
    original = reference().clone()
    route = (x @ site.V[:, 1:2]) @ site.U[1:2]
    with scaled_head(nat, 0, 1, 0.5):
        edited = reference()
        assert not torch.equal(edited, original)
        assert torch.equal(edited, x @ site.W.T)
        edited_route = (x @ site.V[:, 1:2]) @ site.U[1:2]
        assert not torch.equal(route, edited_route)
        assert float(edited_route.norm()) < float(route.norm())
    assert torch.equal(reference(), original)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_logprob_cannot_pass_kl_integrity(bad):
    p = torch.log_softmax(torch.tensor([[1., 2.]]), -1)
    q = p.clone()
    q[0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        kl_bits(p, q)


def test_kl_shape_normalization_and_signed_roundoff_contract():
    p = torch.log_softmax(torch.tensor([[1., 2.]], dtype=torch.float64), -1)
    with pytest.raises(ValueError, match='shapes'):
        raw_kl_bits(p, p[:, :1])
    with pytest.raises(ValueError, match='normalized'):
        raw_kl_bits(p, p + 0.1)
    q = p + 1e-9
    assert raw_kl_bits(p, q) < 0
    assert kl_bits(p, q) == 0


def test_raw_distribution_archive_roundtrips_and_binds_hash(tmp_path):
    import json
    from safetensors.torch import load_file

    tensors = {'p0': torch.log_softmax(torch.tensor([[1., 2.]]), -1),
               'qe_base': torch.log_softmax(torch.tensor([[3., 1.]]), -1)}
    binding = save_distributions(tmp_path, 'draw.safetensors', tensors)
    saved = load_file(str(tmp_path / binding['file']))
    assert all(torch.equal(saved[name], value) for name, value in tensors.items())
    assert binding['sha256'] == digest(tmp_path / 'draw.safetensors')
    assert json.loads((tmp_path / 'distribution_manifest.jsonl').read_text()) == binding
