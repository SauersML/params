"""Model-free checks for the absolute point constraint on auxiliary reward."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_gate import gated_auxiliary
from reward_gate import qualify as reward_gate


def test_observed_empty_like_9001_circuit_gets_no_bonuses():
    gate = reward_gate(0.0, 0.00022, -2.538, dominance=2.0)
    assert gate["phi_gate"] > 0.999
    assert not gate["eligible"]
    assert gate["reasons"] == ["point_insufficient"]
    assert gate["g"] == gate["dominance_gate"] == 0.0
    assert gate["D"] == 0.0 and not gate["faithful"]
    assert gated_auxiliary(gate, quality=4.75) == 0.0


@pytest.mark.parametrize("tau", [-2.538, -1000.0, -1e308])
def test_noisy_or_weak_start_cannot_qualify_bad_outputs(tau):
    gate = reward_gate(0.89, 0.0, tau, dominance=100.0)
    assert not gate["eligible"]
    assert gated_auxiliary(gate, 100.0) == 0.0


def test_qualified_point_retains_legacy_smooth_gate_and_dominance():
    gate = reward_gate(0.95, 0.25, 0.0, dominance=2.0)
    expected = 1.0 / (1.0 + 2.0 ** (-4.0 * 0.25))
    assert gate["eligible"] and gate["reasons"] == []
    assert gate["g"] == pytest.approx(expected)
    assert gate["dominance_gate"] == 1.0
    assert gate["D"] == 2.0 and gate["faithful"]
    assert gated_auxiliary(gate, 3.0) == pytest.approx(expected * 3.0 + 2.0)


def test_absolute_target_boundary_qualifies():
    assert reward_gate(0.9, 0.0, 0.0)["eligible"]
    assert not reward_gate(math.nextafter(0.9, 0.0), 0.0, 0.0)["eligible"]


@pytest.mark.parametrize("point", [None, math.nan, math.inf, -math.inf, -1.0, 1.1])
def test_invalid_or_insufficient_point_fails_closed(point):
    gate = reward_gate(point, 10.0, 0.0)
    assert not gate["eligible"]
    assert gate["g"] == gate["dominance_gate"] == 0.0
    assert gated_auxiliary(gate, math.inf) == 0.0


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["phi", "phi_tau"])
def test_invalid_phi_gate_fails_closed(bad, field):
    arguments = {"point_score": 0.95, "phi": 0.0, "phi_tau": 0.0, field: bad}
    gate = reward_gate(**arguments)
    assert not gate["eligible"]
    assert gate["reasons"] == ["phi_gate_unresolved"]
    assert gate["phi_gate"] is None
    assert gated_auxiliary(gate, 10.0) == 0.0


@pytest.mark.parametrize("phi,tau,expected", [(1e308, -1e308, 1.0), (-1e308, 1e308, 0.0),
                                            (1e308, 1e308, 0.5), (1000.0, 0.0, 1.0), (-1000.0, 0.0, 0.0)])
def test_extreme_finite_phi_gate_does_not_overflow(phi, tau, expected):
    gate = reward_gate(0.95, phi, tau)
    assert gate["eligible"]
    assert gate["g"] == expected


@pytest.mark.parametrize("config", [{"floor": math.nan}, {"floor": -0.1}, {"floor": 1.1},
                                     {"slope": 0.0}, {"slope": math.inf}])
def test_invalid_configuration_is_explicit(config):
    with pytest.raises(ValueError):
        reward_gate(0.95, 0.0, 0.0, **config)


def test_point_eligible_below_phi_bar_is_not_faithful():
    gate = reward_gate(0.95, -1.0, 0.0, dominance=2.0)
    assert gate["eligible"] and not gate["faithful"]
    assert gate["D"] == 2.0


@pytest.mark.parametrize("dominance", [None, math.nan, math.inf, -math.inf])
def test_undefined_dominance_cannot_add_a_bonus(dominance):
    gate = reward_gate(0.95, 0.0, 0.0, dominance=dominance)
    assert gate["eligible"] and gate["D"] == 0.0
