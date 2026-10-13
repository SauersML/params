"""Direct edited predictions must remain distinguishable from baseline-anchored changes."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_eval
import native
import weight_family
from test_goal_eval import naive, tiny_native


def kl_bits(p, q):
    return float((p.double().exp() * (p.double() - q.double())).sum()) / math.log(2)


def test_correct_relative_change_can_hide_wrong_absolute_prediction():
    p0 = torch.log_softmax(torch.tensor([[2., 0., -1.]], dtype=torch.float64), -1)
    pe = torch.log_softmax(torch.tensor([[0., 2., -1.]], dtype=torch.float64), -1)
    bias = torch.tensor([[0., 0., 8.]], dtype=torch.float64)
    before = torch.log_softmax(p0 + bias, -1)
    after = torch.log_softmax(pe + bias, -1)
    moved = weight_family._moved(p0, before, after)
    assert kl_bits(pe, moved) == pytest.approx(0., abs=1e-12)
    assert kl_bits(pe, after) > 5.
    assert int(pe.argmax()) != int(after.argmax())


@pytest.mark.parametrize('semantics', ['n', 'c'])
def test_direct_weight_diagnostics_match_independent_physical_forward(semantics):
    # near=None builds random factors directly, without any matrix decomposition.
    nat = tiny_native(seed=73, near=None)
    ge = goal_eval.GoalEval(nat, semantics=semantics)
    ids, targets = [1, 8, 3, 11, 4], [4]
    tx = ge.text(ids, targets, seed=1)
    masks = {name: torch.zeros(len(ids), nat.C[name], dtype=torch.bool) for name in nat.names}
    for mask in masks.values():
        mask[:, ::2] = True
    graph = native.Graph(masks=masks)
    edits = [{'kind': 'subcomponent', 'what': ['h.1.mlp.down_proj', 0], 'k': .5},
             {'kind': 'head', 'what': [0, 0], 'k': 0.}]
    def reference(on=None):
        return torch.log_softmax(naive(nat, ids, on=on)[0][targets], -1)
    full_before = reference()
    circuit_before = reference(masks)
    row = weight_family.family(ge, tx, [graph], edits=edits)[0]
    assert row['kl_weight_unedited'] == pytest.approx(kl_bits(full_before, circuit_before), rel=1e-4, abs=1e-4)
    expected = []
    for edit, direct, anchored in zip(edits, row['draws_weight_direct'], row['draws_weight']):
        with weight_family.applied(nat, edit, ge):
            full_after = reference()
            circuit_after = reference(masks)
        expected.append(kl_bits(full_after, circuit_after))
        assert direct[:4] == anchored[:4]
        assert direct[4] == pytest.approx(expected[-1], rel=1e-4, abs=1e-4)
        moved = weight_family._moved(full_before, circuit_before, circuit_after)
        assert anchored[4] == pytest.approx(kl_bits(full_after, moved), rel=1e-4, abs=1e-4)
    assert row['kl_weight_direct'] == pytest.approx(sum(expected) / len(expected), rel=1e-4, abs=1e-4)
    assert row['weight_direct_remainders'] == 'live'
    assert torch.allclose(reference(), full_before, rtol=0., atol=0.)
    empty_panel = weight_family.family(ge, tx, [graph], edits=[])[0]
    assert empty_panel['kl_weight_direct'] is None
    assert empty_panel['draws_weight_direct'] == []
