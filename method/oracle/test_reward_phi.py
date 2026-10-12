"""Synthetic regressions for below-bound faithfulness in reward v3."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_eval


def test_phi_ranks_below_empty_circuits_and_preserves_clipped_reporting():
    rows = []
    for kl in (1.0, 2.0, 100.0):
        rows.append({"s_point_c": goal_eval._fid(1.0, kl, 0.1), "n_point": 0.0,
                     "rl_terms": {"s_point_c": goal_eval._fid_raw(1.0, kl, 0.1)},
                     **{f"eps_{f}": goal_eval.EPS_FLOOR for f in goal_eval.PHI_FAMILIES}})
    ph = [goal_eval.phi(row)[0] for row in rows]
    assert all(row["s_point_c"] == 0.0 for row in rows)
    assert ph[0] > ph[1] > ph[2]
    rewards = [p + 2.0 / (1.0 + 2.0 ** (-4.0 * (p - 1.0))) for p in ph]
    assert rewards[0] > rewards[1] > rewards[2]
    # Stored diagnostic rows lacking raw families retain their historical semantics.
    assert goal_eval.family_score({"s_point_c": 0.25}, "point") == 0.25
    assert goal_eval.family_score({"s_point_c": 0.0, "rl_terms": {"s_point_c": None}}, "point") is None


@pytest.mark.parametrize("kind,draws", [
    ("mean", [[2.0, 0.1], [100.0, 0.1]]),
    ("donor", [[0, 2.0, 0.1, 1.0], [1, 100.0, 0.1, 1.0]]),
    ("weighted", [[0, 2.0, 0.1, 1.0], [1, 100.0, 0.1, 1.0]]),
])
def test_raw_bootstrap_pooling_retains_variation_below_zero(kind, draws):
    assert goal_eval._pooled(kind, draws, 1.0) == 0.0
    assert goal_eval._pooled(kind, draws, 1.0, raw=True) < 0.0
    a = goal_eval._pooled(kind, [draws[0]] * 2, 1.0, raw=True)
    b = goal_eval._pooled(kind, [draws[1]] * 2, 1.0, raw=True)
    assert a > b


def test_raw_family_noise_is_bootstrapped_from_raw_scores():
    row = {"s_nodes": 0.0, "draws_nodes": [[0, 2.0, 0.1, 1.0], [1, 100.0, 0.1, 1.0]],
           "kl_empty_live": 1.0}
    assert goal_eval.family_eps(row)["nodes"] == goal_eval.EPS_FLOOR
    raw = {**row, "rl_terms": {"s_nodes": goal_eval._pooled_raw(row["draws_nodes"])}}
    assert goal_eval.family_eps(raw)["nodes"] > goal_eval.EPS_FLOOR
    assert goal_eval.phi(raw)[0] < goal_eval.phi(row)[0]


def test_full_and_reward_fidelity_use_the_same_raw_phi_families():
    from test_goal_eval import tiny_native
    import native

    nat = tiny_native()
    ge = goal_eval.GoalEval(nat)
    tx = ge.text([2, 5, 3, 7, 4, 6, 8, 9], [7])
    g = native.Graph(masks=tx.vpd)
    opts = dict(pool={}, draws=4, interchange=4, changed=4, near_donors=0, far_donors=0,
                split="search", reads=True)
    reward = ge.fidelity(tx, [g], mode="reward", **opts)[0]
    full = ge.fidelity(tx, [g], mode="full", **opts)[0]
    assert reward["rl_terms"] == full["rl_terms"]
    assert goal_eval.phi(reward)[0] == pytest.approx(goal_eval.phi(full)[0])
