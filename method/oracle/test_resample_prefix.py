"""Exact CPU resampling prefix reuse, opt-in isolation and conservative fallback."""
from types import SimpleNamespace

import sys
from pathlib import Path

import pytest
import torch
import goal_eval
import native
from test_goal_eval import tiny_native
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"experiments/oracle"))
from verify_resample_prefix import strict_pair_equal


def setup_case(monkeypatch, first=True):
    nat = tiny_native(seed=19, layers=1, d=4, heads=1, vocab=11, C=2)
    T = 73
    pos = torch.arange(T, dtype=torch.float32)
    freq = 10000. ** (torch.arange(2, dtype=torch.float32) / 2)
    ang = pos[:, None] / freq.repeat(2)[None, :]
    nat.target.cos, nat.target.sin = ang.cos(), ang.sin()
    ids = [i % 11 for i in range(T)]
    targets = [3, T - 1]
    tx = SimpleNamespace(ids=ids, targets=targets, T=T, ref=nat.reference([ids], targets))
    ge = goal_eval.GoalEval(nat, chunk=1)
    ge.FIRST_POSITION = first
    donors = [('a', [(i + 1) % 11 for i in range(T)]),
              ('b', [(i + 2) % 11 for i in range(T - 2)]),
              ('c', [i % 11 for i in range(T)])]
    monkeypatch.setattr(ge, 'far_pool', lambda split: donors)
    monkeypatch.setattr(ge, 'far_donors', lambda tx, pool, k: pool[:k])
    return nat, ge, tx


@pytest.mark.parametrize('first', [True, False])
def test_prefix_preserves_actual_resample_rows_multitargets_and_partial_batches(monkeypatch, first):
    nat, ge, tx = setup_case(monkeypatch, first)
    expected = ge.resample_influence(tx, k=3)
    starts, prefix_sizes = [], []
    hidden, capture = nat.target.hidden, nat.target.keys_values
    def spy(ids, start=0, prefix=None, **kw):
        starts.append((len(ids), start))
        return hidden(ids, start, prefix, **kw)
    def prefix(ids):
        prefix_sizes.append(len(ids))
        return capture(ids)
    monkeypatch.setattr(nat.target, 'hidden', spy)
    monkeypatch.setattr(nat.target, 'keys_values', prefix)
    ge.resample_prefix = True
    actual = ge.resample_influence(tx, k=3)
    # Geometry/causal equivalence only: strict bit certification is separate.
    torch.testing.assert_close(torch.tensor(list(actual.values()), dtype=torch.float64),
        torch.tensor(list(expected.values()), dtype=torch.float64), rtol=0, atol=1e-7)
    assert max(s for _, s in starts) == min(tx.targets)
    assert all(tx.T - s >= 64 and s <= min(tx.targets) for _, s in starts)
    assert set(prefix_sizes) == {4, 1 if first else 2}
    assert len(prefix_sizes) == len(set(prefix_sizes))
    assert (0 in actual) == first
    assert all(t not in actual for t in tx.targets)
    assert ("search", 3, first, "experimental_cpu_prefix_v1") in tx._resample_exact
    # The experimental result must never satisfy a later default-path request.
    del tx._resample_exact['search', 3, first]
    ge.resample_prefix = False
    assert ge.resample_influence(tx, k=3) == expected
    assert ('search', 3, first) in tx._resample_exact


@pytest.mark.parametrize('guard', ['edit', 'hook', 'reference_override', 'foreign_behavior', 'foreign_target', 'site_transform'])
def test_prefix_falls_back_to_reference_under_edits_hooks_and_behavior(monkeypatch, guard):
    nat, ge, tx = setup_case(monkeypatch)
    ge.resample_prefix = True
    if guard == 'edit':
        nat._graph_edit = object()
    elif guard == 'hook':
        nat.target.register_forward_pre_hook(lambda *args: None)
    elif guard == 'reference_override':
        original = nat.reference
        monkeypatch.setattr(nat, 'reference', lambda ids, targets: original(ids, targets))
    elif guard == 'site_transform':
        nat.target.site(nat.names[0]).in_fn = lambda x: x
    elif guard == 'foreign_behavior':
        nat.target.set_behavior([1, 2], [tx.targets[-1]])
    else:
        nat.target.__class__ = type('ForeignTarget', (type(nat.target),), {})
    monkeypatch.setattr(nat.target, 'keys_values', lambda *args: pytest.fail('unsafe prefix capture'))
    actual = ge.resample_influence(tx, k=1)
    assert actual
    assert not any(len(key) > 3 for key in tx._resample_exact)


def test_prefix_default_off_and_flag_validation():
    nat = object()
    assert goal_eval.GoalEval(nat).resample_prefix is False
    with pytest.raises(ValueError, match='must be bool'):
        goal_eval.GoalEval(nat, resample_prefix=1)


def test_strict_comparator_preserves_observed_tiny_counterexample():
    # Observed local CPU T73, targets[3,72], final B1, start3. Different CPUs
    # need not reproduce the same kernel rounding; these actual values remain
    # an exactness failure even though the numerical difference is tiny.
    whole = {'scores': {71: 4.9348551935205855e-05}, 'rank': [71]}
    prefix = {'scores': {71: 4.935805554850958e-05}, 'rank': [71]}
    flags = [dict.fromkeys(('log_prob_equal', 'probability_equal', 'kl_equal'), True)]
    assert abs(whole['scores'][71] - prefix['scores'][71]) < 1e-7
    assert not strict_pair_equal(whole, prefix, flags)
    assert strict_pair_equal(whole, whole, flags)
    flags[0]['log_prob_equal'] = False
    assert not strict_pair_equal(whole, whole, flags)


def test_default_cached_result_needs_no_native_or_opt_in_attribute():
    ge = object.__new__(goal_eval.GoalEval)
    tx = SimpleNamespace(_resample_exact={('search', 4, True): {0: 1.0}})
    assert ge.resample_influence(tx, k=4) == {0: 1.0}
