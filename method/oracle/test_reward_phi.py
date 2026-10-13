"""Synthetic regressions for below-bound faithfulness in reward v3: phi reads floored family scores (the Gate B table
10-13), the unfloored ones stay C_rl's and keep their pooling."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_eval


def test_phi_reads_floored_scores():
    """Below the bound every answer scores 0 on that family in phi, whatever its unfloored score (rl_terms, C_rl's)."""
    rows = []
    for kl in (1.0, 2.0, 100.0):
        rows.append({"s_point_c": goal_eval._fid(1.0, kl, 0.1), "n_point": 0.0,
                     "rl_terms": {"s_point_c": goal_eval._fid_raw(1.0, kl, 0.1)},
                     **{f"eps_{f}": goal_eval.EPS_FLOOR for f in goal_eval.PHI_FAMILIES}})
    ph = [goal_eval.phi(row)[0] for row in rows]
    assert all(row["s_point_c"] == 0.0 for row in rows) and ph[0] == ph[1] == ph[2]
    assert rows[0]["rl_terms"]["s_point_c"] > rows[2]["rl_terms"]["s_point_c"]
    assert goal_eval.family_score({"s_point_c": 0.25}, "point") == 0.25
    assert goal_eval.family_score({"s_point_c": 0.0, "rl_terms": {"s_point_c": None}}, "point") == 0.0


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


def test_family_noise_is_bootstrapped_from_floored_scores():
    """eps from the floored estimator phi reads: unfloored rl_terms change neither eps nor phi."""
    row = {"s_nodes": 0.0, "draws_nodes": [[0, 2.0, 0.1, 1.0], [1, 100.0, 0.1, 1.0]],
           "kl_empty_live": 1.0}
    assert goal_eval.family_eps(row)["nodes"] == goal_eval.EPS_FLOOR
    raw = {**row, "rl_terms": {"s_nodes": goal_eval._pooled_raw(row["draws_nodes"])}}
    assert goal_eval.family_eps(raw) == goal_eval.family_eps(row)
    assert goal_eval.phi(raw)[0] == goal_eval.phi(row)[0]


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


def test_eps_cap_and_n_guard():
    """eps at most EPS_CAP from the bootstrap; N None where VPD's denominator is within twice its standard error."""
    row = {"s_nodes": 0.5, "draws_nodes": [[0, 0.1, 0.05, 1.0], [1, 3.0, 0.05, 1.0], [2, 0.2, 0.05, 2.0]], "kl_empty_live": 1.0}
    assert goal_eval.family_eps(row)["nodes"] <= goal_eval.EPS_CAP
    noisy = [[j, 1.0, 1.0 + (-1) ** j * 5.0, 3.0, 2.5] for j in range(8)]  # denominator 1.5, its SE ~1.9
    assert goal_eval._spec(noisy) is None and goal_eval._spec(noisy, raw=True) is None


def test_pooled_ratio():
    """The weighted families' score: each draw's ratio ln(W / KL_C) / ln(W / KL_VPD) clipped to [0, 1] (raw [-1, 1]),
    averaged with weights ln(W / KL_VPD) over the draws where VPD is nearer than no change: a draw with VPD at the
    bound weighs nothing, one far past the bound counts at most its clip; None when the weights' sum is within twice its
    standard error."""
    import math
    d = [[0, 0.5, 0.2, 2.0], [1, 1.0, 0.4, 2.0], [2, 2.5, 0.3, 2.0], [3, 9.0, 3.0, 2.0]]  # the last: VPD past W, left out
    w = [math.log(2.0 / 0.2), math.log(2.0 / 0.4), math.log(2.0 / 0.3)]
    r = [min(1.0, max(0.0, math.log(2.0 / kc) / x)) for kc, x in zip((0.5, 1.0, 2.5), w)]
    want = sum(a * b for a, b in zip(r, w)) / sum(w)
    assert abs(goal_eval._ratio_pool(d) - want) < 1e-12
    assert abs(goal_eval._ratio_pool(d[:3] + [[4, 0.1, 1.9999, 2.0]]) - want) < 1e-3  # VPD at the bound weighs ~5e-5
    assert goal_eval._ratio_pool(d[:3] + [[5, 500.0, 0.25, 2.0]]) >= 0.0  # far past the bound: 0, not ln(2 / 500)
    assert goal_eval._pooled("weighted", d, None) == goal_eval._ratio_pool(d)
    assert goal_eval._ratio_pool([[0, 50.0, 0.2, 2.0], [1, 40.0, 0.3, 2.0]], raw=True) == -1.0
    noisy = [[j, 1.0, 2.0 * (1 - 0.999 * (j % 2)), 2.0] for j in range(6)]  # VPD at the bound on half the draws
    assert goal_eval._ratio_pool(noisy) is not None
    assert goal_eval._ratio_pool([[0, 1.0, 1.99, 2.0], [1, 1.0, 1.0, 2.0], [2, 1.0, 1.999, 2.0]]) is None
