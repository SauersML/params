"""Native strength execution and ranking; synthetic model, no data downloads."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
# Remote tests stage the overlay beside this file and expose executor tests via PYTHONPATH.
from test_goal_eval import tiny_native
import carry

spec = importlib.util.spec_from_file_location('ranking_native_overlay', HERE / 'native_units.py')
units = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = units
spec.loader.exec_module(units)

@pytest.fixture
def nu():
    torch.set_num_threads(1)
    return units.NativeUnits(tiny_native(seed=3, near=.1), [1, 4, 9, 2], [3])


def strengths(nu, value):
    full = nu.full()
    return full.heads.float().fill_(value), full.neurons.float().fill_(value)


def test_endpoints_and_boolean_execution_match_without_cache_mutation(nu):
    full = nu.full()
    ordinary = nu._forward(full, [[]], 0, keep=True)
    cache = nu._cache
    h, n = strengths(nu, 1.)
    assert torch.equal(ordinary, nu.strength_log_probs(h, n))
    assert nu._cache is cache
    empty = full.without(full.units())
    ordinary_empty = nu._forward(empty, [[]], 0, keep=True)
    cache = nu._cache
    h, n = strengths(nu, 0.)
    assert torch.equal(ordinary_empty, nu.strength_log_probs(h, n))
    assert nu._cache is cache
    assert not ordinary.requires_grad


def test_mixed_boolean_units_match_ordinary_execution(nu):
    mask = nu.full()
    mask.heads.flatten()[::3] = False
    mask.neurons.flatten()[1::4] = False
    ordinary = nu._forward(mask, [[]], 0, keep=True)
    cache = nu._cache
    got = nu.strength_log_probs(mask.heads.float(), mask.neurons.float())
    assert torch.equal(got, ordinary)
    assert nu._cache is cache


@pytest.mark.parametrize('kind', ['head', 'neuron'])
def test_interior_mask_gradient_matches_directional_finite_difference(nu, kind):
    h, n = strengths(nu, .55)
    h.requires_grad_(); n.requires_grad_()
    loss = carry.kl(nu.ref.detach(), nu.strength_log_probs(h, n)).sum()
    grads = torch.autograd.grad(loss, (h, n))
    gi = 0 if kind == 'head' else 1
    # Fixed source-independent dense direction exercises routing and both layers.
    value = h if gi == 0 else n
    direction = torch.linspace(-1., 1., value.numel()).reshape_as(value)
    expected = float((grads[gi] * direction).sum())
    eps = .002
    vals = []
    for sign in (-1, 1):
        hh, nn = h.detach().clone(), n.detach().clone()
        (hh if gi == 0 else nn).add_(direction, alpha=sign*eps)
        vals.append(float(carry.kl(nu.ref, nu.strength_log_probs(hh, nn)).sum()))
    actual = (vals[1]-vals[0])/(2*eps)
    assert abs(expected) > 1e-6
    assert actual == pytest.approx(expected, abs=2e-4, rel=.02)


def test_scores_equal_negative_midpoint_gradients_and_preserve_state(nu):
    nu._forward(nu.full(), [[]], 0, keep=True)
    cache = nu._cache
    runs = nu.runs
    with torch.no_grad():
        got = nu.integrated_unit_scores(2)
    expected = [torch.zeros_like(got[k]) for k in ('heads', 'neurons')]
    for a in (.25, .75):
        h, n = strengths(nu, a)
        h.requires_grad_(); n.requires_grad_()
        grads = torch.autograd.grad(carry.kl(nu.ref, nu.strength_log_probs(h,n)).sum(), (h,n))
        for dst, grad in zip(expected, grads): dst.sub_(grad.detach()/2)
    assert torch.equal(got['heads'], expected[0])
    assert torch.equal(got['neurons'], expected[1])
    assert all(torch.isfinite(got[k]).all() for k in ('heads','neurons'))
    assert got['forward_evaluations'] == got['backward_evaluations'] == 2
    assert nu._cache is cache and nu.runs == runs
    assert all(getattr(w, 'grad', None) is None for matrices in nu.W.values() for w in matrices)
    again = nu.integrated_unit_scores(2)
    assert torch.equal(got['heads'], again['heads']) and torch.equal(got['neurons'], again['neurons'])


@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_invalid_steps_fail_before_execution(nu, bad):
    with pytest.raises(ValueError): nu.integrated_unit_scores(bad)


def test_invalid_strengths_fail_without_cache_change(nu):
    cache=nu._cache
    h,n=strengths(nu,.5)
    for bad in (float('nan'), float('inf'), -1., 1.1):
        wrong=h.clone();wrong.flatten()[0]=bad
        with pytest.raises(ValueError):nu.strength_log_probs(wrong,n)
    with pytest.raises(ValueError):nu.strength_log_probs(h[0],n)
    with pytest.raises(ValueError):nu.strength_log_probs(h.bool(),n)
    assert nu._cache is cache
