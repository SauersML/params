import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intervention_teacher_challenge import fixed_edits, joint_pairs, summarize


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


def test_summary_retains_invalids_and_distinguishes_mixed_interventions():
    row = {"case": "train", "kind": "joint", "operations": [("head", 0, 0, 0.), ("head", 0, 1, .5)],
           "valid": True, "top": 2, "reference_top": 2, "nochange_top": 1, "active": True, "kl_bits": 1e-8}
    invalid = {**row, "valid": False, "top": None, "kl_bits": None}
    group = summarize([row, invalid])["groups"][0]
    assert group["kind"] == "joint_mixed"
    assert (group["worlds"], group["valid"], group["top_correct"], group["kl_failures_or_invalid"]) == (2, 1, 1, 1)
def test_batch_input_selection_is_distinct_and_search_disjoint():
    import intervention_teacher_batch_challenge as batch
    draws = [[0, 1], [0, 2], [0, 2]] + [[0, i] for i in range(3, 10)]
    assert batch.select_inputs(draws, [[0, 1]]) == [[0, i] for i in range(2, 10)]


def test_batch_input_selection_rejects_insufficient_fixed_pool():
    import pytest
    import intervention_teacher_batch_challenge as batch
    with pytest.raises(RuntimeError, match="fixed 64-draw"):
        batch.select_inputs([[0, 1]] * 64, [[0, 1]])


