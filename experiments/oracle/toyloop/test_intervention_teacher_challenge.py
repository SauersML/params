import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intervention_teacher_challenge import fixed_edits, joint_pairs


def test_pair_selection_is_order_independent_and_target_independent():
    units = [("head", 0, index) for index in range(8)] + [("neuron", 0, 2)]
    pairs = joint_pairs("train", units)
    assert pairs == joint_pairs("train", list(reversed(units)) + units)
    assert len(pairs) == len(set(pairs)) == 6
    assert all(left in units and right in units and left != right for left, right in pairs)
    with pytest.raises(RuntimeError, match="pair count"):
        joint_pairs("train", units[:2])


def test_joint_panel_contains_mixed_scales_and_all_individuals():
    units = [("head", 0, index) for index in range(8)]
    pairs = joint_pairs("train", units)
    edits = fixed_edits(units, pairs)
    assert len(edits) == 1 + 2 * len(units) + 4 * len(pairs)
    assert edits[0] == ("no_edit", [])
    for left, right in pairs:
        observed = {(operations[0][-1], operations[1][-1]) for kind, operations in edits
                    if kind == "joint" and operations[0][:3] == left and operations[1][:3] == right}
        assert observed == {(0., 0.), (0., .5), (.5, 0.), (.5, .5)}
