"""Synthetic regressions for below-bound faithfulness in reward v3: phi reads the unfloored family scores in [-1, 1]
with a leaky floor (the user's 10-13 softening of the Gate B table's floor at 0), and every piece of the circuit term
is smooth."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_eval


def test_phi_reads_leaky_scores():
    """Below the bound phi still ranks answers: the unfloored score clipped to [-1, 1], LEAK of it below 0 (rl_terms
    for point, edges, flips and patching; the *_raw fields for the rest); the floored score where none is kept."""
    rows = []
    for kl in (1.0, 2.0, 100.0):
        rows.append({"s_point_c": goal_eval._fid(1.0, kl, 0.1), "n_point": 0.0,
                     "rl_terms": {"s_point_c": goal_eval._fid_raw(1.0, kl, 0.1)},
                     **{f"eps_{f}": goal_eval.EPS_FLOOR for f in goal_eval.PHI_FAMILIES}})
    ph = [goal_eval.phi(row)[0] for row in rows]
    assert all(row["s_point_c"] == 0.0 for row in rows) and ph[0] > ph[1] > ph[2]
    assert goal_eval.family_score(rows[2], "point") == pytest.approx(-goal_eval.LEAK)  # raw -2, clipped to -1
    assert goal_eval.family_score(rows[1], "point") == pytest.approx(goal_eval.LEAK * goal_eval._fid_raw(1.0, 2.0, 0.1))
    assert goal_eval.family_score({"s_nodes": 0.0, "s_nodes_raw": -0.4}, "nodes") == pytest.approx(-0.04)
    assert goal_eval.family_score({"s_point_c": 0.25}, "point") == 0.25 and goal_eval.family_score({}, "point") is None
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
    assert a >= b  # (the weighted families' raw score stops at -1: both draws far past the bound)


def test_family_noise_is_bootstrapped_from_the_score_phi_reads():
    """eps from the unfloored estimator phi reads (clipped to [-1, 1]); the unfloored score moves phi (leaky)."""
    row = {"s_nodes": 0.0, "draws_nodes": [[0, 2.0, 0.1, 1.0], [1, 100.0, 0.1, 1.0]],
           "kl_empty_live": 1.0}
    assert goal_eval.family_eps(row)["nodes"] == goal_eval.EPS_FLOOR
    raw = {**row, "s_nodes_raw": goal_eval._pooled_raw(row["draws_nodes"])}
    assert goal_eval.family_eps(raw) == goal_eval.family_eps(row)
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


def test_eps_cap_and_n_guard():
    """eps at most EPS_CAP from the bootstrap; N None where VPD's denominator is within twice its standard error."""
    row = {"s_nodes": 0.5, "draws_nodes": [[0, 0.1, 0.05, 1.0], [1, 3.0, 0.05, 1.0], [2, 0.2, 0.05, 2.0]], "kl_empty_live": 1.0}
    assert goal_eval.family_eps(row)["nodes"] <= goal_eval.EPS_CAP
    noisy = [[j, 1.0, 1.0 + (-1) ** j * 5.0, 3.0, 2.5] for j in range(8)]  # denominator 1.5, its SE ~1.9
    assert goal_eval._spec(noisy) is None and goal_eval._spec(noisy, raw=True) is None


def test_pooled_ratio():
    """The weighted families' score: sum clip(W - KL_C, 0, W - KL_VPD) / sum (W - KL_VPD) over the draws where VPD is
    nearer than no change (raw: clipped from below at -(W - KL_VPD)); a draw with VPD at the bound weighs nothing; None
    when the gaps' sum is within twice its standard error."""
    d = [[0, 0.5, 0.2, 2.0], [1, 1.0, 0.4, 2.0], [2, 2.5, 0.3, 2.0], [3, 9.0, 3.0, 2.0]]  # the last: VPD past W, left out
    want = (1.5 + 1.0 + 0.0) / (1.8 + 1.6 + 1.7)
    assert abs(goal_eval._ratio_pool(d) - want) < 1e-12
    assert abs(goal_eval._ratio_pool(d, raw=True) - (1.5 + 1.0 - 0.5) / 5.1) < 1e-12
    assert abs(goal_eval._ratio_pool(d[:3] + [[4, 0.1, 1.9999, 2.0]]) - want) < 1e-4  # VPD at the bound weighs 1e-4 bits
    assert goal_eval._pooled("weighted", d, None) == goal_eval._ratio_pool(d)
    assert goal_eval._ratio_pool([[0, 50.0, 0.2, 2.0], [1, 40.0, 0.3, 2.0]], raw=True) == -1.0
    assert goal_eval._ratio_pool([[0, 3.0, 2.5, 2.0]]) is None  # no draw where VPD is nearer than no change
    uneven = [[0, 1.0, 1.99, 2.0], [1, 1.0, 0.0, 2.0], [2, 1.0, 1.999, 2.0]]  # one draw carries the change: still scored
    assert abs(goal_eval._ratio_pool(uneven) - (0.01 + 1.0 + 0.001) / (0.01 + 2.0 + 0.001)) < 1e-12


def test_floors_and_circuit_term():
    """Per-family floors from the start (min(0.9, s - 0.05), at least 0; 0 where the start scores nothing), v the worst
    shortfall below them (an unscored family at 0; faithful's), v_soft the reward's power mean of softplus shortfalls,
    and circuit_term = cap_phi(phi) - phi_tau - 4 v_soft + g 2^(-20 v_soft) (0.5 compact + quality) + 0.04 tanh(compact / 2)."""
    import math
    start = {"s_point_c": 0.95, "s_nodes": 0.5, "s_track_pp": None, "n_spec": 0.02,
             **{f"eps_{f}": goal_eval.EPS_FLOOR for f in goal_eval.PHI_FAMILIES}}
    fams = ["point", "nodes", "track_pp", "N"]
    fl = goal_eval.floors(start, fams)
    want = {"point": 0.9, "nodes": 0.45, "track_pp": 0.0, "N": 0.0}
    assert set(fl) == set(want) and all(abs(fl[k] - want[k]) < 1e-12 for k in want)
    row = dict(start, s_point_c=0.85, s_nodes=0.5)
    assert abs(goal_eval.shortfall(row, fl) - 0.05) < 1e-12
    assert goal_eval.shortfall(dict(row, s_nodes=None), fl) == 0.45  # an unscored family at 0
    w = goal_eval.SHORTFALL_WIDTH
    sp = lambda x: w * math.log1p(math.exp(x / w))  # noqa: E731
    vs = (sum(sp(f - s) ** 4 for f, s in ((0.9, 0.85), (0.45, 0.5), (0.0, 0.0), (0.0, 0.02))) / 4) ** 0.25
    assert goal_eval.shortfall_soft(row, fl) == pytest.approx(vs)
    tau = {"families": fams, "phi_tau": 1.0, "floors": fl, "bits_start": 400.0}
    ct = goal_eval.circuit_term(row, tau, 100.0, quality=0.3)
    ph = goal_eval.cap_phi(goal_eval.phi(row, families=fams)[0], 1.0)
    g = 1.0 / (1.0 + 2.0 ** (-goal_eval.GATE_SLOPE * (ph - 1.0)))
    want = ph - 1.0 - 4 * vs + g * 2.0 ** (-20 * vs) * (0.5 * 2.0 + 0.3) + 0.04 * math.tanh(1.0)
    assert abs(ct["term"] - want) < 1e-12 and ct["v"] == pytest.approx(0.05) and ct["v_soft"] == pytest.approx(vs)
    assert abs(ct["compact"] - 2.0) < 1e-12
    big = goal_eval.circuit_term(row, tau, 400.0 * 2 ** 5)  # 32x the start: the tie-break near its -0.04 bound
    assert big["tie"] == pytest.approx(-0.04 * math.tanh(2.5))
    assert goal_eval.circuit_term(row, tau, 0.0)["compact"] is None and goal_eval.circuit_term(row, tau, 0.0)["tie"] == 0.0


def test_weighted_score_is_the_raw_pool():
    """A weighted family's score is its raw pool above 0 too (a draw farther than no change counts against it), and
    the floors and both shortfalls read the same score: the start sits at its own floors with no shortfall."""
    draws = [[0, 0.5, 0.2, 2.0], [1, 3.0, 0.4, 2.0]]  # the second draw farther than no change
    row = {"s_nodes": goal_eval._pooled("weighted", draws, None), "s_nodes_raw": goal_eval._pooled_raw(draws)}
    assert row["s_nodes_raw"] < row["s_nodes"] and goal_eval.family_score(row, "nodes") == pytest.approx(row["s_nodes_raw"])
    fl = goal_eval.floors(row, ["nodes"])
    assert fl["nodes"] == pytest.approx(row["s_nodes_raw"] - goal_eval.FLOOR_MARGIN)
    assert goal_eval.shortfall(row, fl) == 0.0 and goal_eval.shortfall_soft(row, fl) < 0.002


def test_the_start_scores_its_hard_term():
    """The start set's soft circuit term equals its hard one at v = 0 within 0.01 (pymove's 2fdd1cf report: floors from
    the floored estimators against a shortfall from the raw ones put the start near -0.5): its weighted family's raw
    pool below its floored one, its floors and both shortfalls on the same score."""
    eps = {f"eps_{f}": goal_eval.EPS_FLOOR for f in goal_eval.PHI_FAMILIES}
    start = {"s_point_c": 0.8, "s_edges": 0.75, "s_nodes": 0.57, "s_nodes_raw": 0.30, "s_track_pp": 0.85, "s_track_pp_raw": 0.7,
             "n_spec": 0.6, "rl_terms": {"s_point_c": 0.8, "s_edges": 0.75}, **eps}
    fams = ["point", "edges", "nodes", "track_pp", "N"]
    _, bar = goal_eval.phi(start, start=start, families=fams)
    tau = {"families": fams, "phi_tau": bar, "floors": goal_eval.floors(start, fams), "bits_start": 300.0}
    ct = goal_eval.circuit_term(start, tau, 300.0)
    assert ct["v"] == 0.0 and abs(ct["term"]) < 0.01, ct


def test_semantics_default_is_the_rewards(monkeypatch):
    """GoalEval's semantics: "n" by default, whatever the environment says (a pod missing a variable scored under "c"
    silently: primerl's e5b8b03); "c" only when asked; nothing else exists."""
    from test_goal_eval import tiny_native

    nat = tiny_native()
    monkeypatch.setenv("SEMANTICS", "c")
    assert goal_eval.GoalEval(nat).semantics == "n" and goal_eval.GoalEval(nat, semantics="c").semantics == "c"
    assert set(goal_eval.SEMANTICS) == {"n", "c"} and goal_eval.SEMANTICS["n"] == ("nodes", "omitted")
    with pytest.raises(ValueError):
        goal_eval.GoalEval(nat, semantics="x")


def test_soft_pieces_are_smooth():
    """cap_phi: slope 1 well below the bar + PHI_CAP, PHI_SLOPE well above, its slope continuous between; the soft
    shortfall grows smoothly through a floor; the leaky floor is continuous at 0."""
    xs = [0.5 + 0.01 * i for i in range(300)]  # bar 1: the cap at 2
    cap = [goal_eval.cap_phi(x, 1.0) for x in xs]
    slope = [(b - a) / 0.01 for a, b in zip(cap, cap[1:])]
    assert slope[0] == pytest.approx(1.0, abs=1e-3) and slope[-1] == pytest.approx(goal_eval.PHI_SLOPE, abs=1e-3)
    assert max(abs(b - a) for a, b in zip(slope, slope[1:])) < 0.05  # no kink
    assert abs(goal_eval.cap_phi(5.0, 1.0) - (2.0 + 0.1 * 3.0)) < 1e-6 and abs(goal_eval.cap_phi(1.0, 1.0) - 1.0) < 0.002
    fl = {"point": 0.5}
    v = [goal_eval.shortfall_soft({"s_point_c": s}, fl) for s in (0.6, 0.52, 0.5, 0.48, 0.4)]
    assert all(a < b for a, b in zip(v, v[1:])) and v[-1] == pytest.approx(0.1, abs=1e-3)
    assert goal_eval.family_score({"s_point_c": 0.0, "rl_terms": {"s_point_c": -1e-9}}, "point") == pytest.approx(0.0, abs=1e-9)


def test_family_kls_report_the_raw_kls():
    """family_kls: the point KL, the bounds and each family's mean KL_C in bits, beside the scores."""
    row = {"kl_point": 2.5, "kl_empty_live": 12.0, "kl_vpd_live": 0.4, "kl_nodes_c": 3.0, "kl_edges_c": 1.0,
           "kl_flips_omitted": {goal_eval.FLIP_TERM: (4.0, 0.5)}, "draws_track_pp": [[0, 1.0, 0.1, 2.0], [1, 3.0, 0.1, 2.0]],
           "kl_patching_near": (0.7, 0.2, 5.0), "draws_n_spec": [[0, 1.0, 1.0, 6.0, 7.0]]}
    k = goal_eval.family_kls(row)
    assert (k["point"], k["empty"], k["vpd"], k["nodes"], k["edges"], k["flips"]) == (2.5, 12.0, 0.4, 3.0, 1.0, 4.0)
    assert k["track_pp"] == 2.0 and k["patching_near"] == 0.7 and k["N"] == 6.0 and k["patching_far"] is None and k["omitted"] is None
