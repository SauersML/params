"""The reward server's terms: the stated change, its direction and magnitude against the measured one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reward_server as rs  # noqa: E402


def test_the_stated_change_reads_g2s_about_n_bits_back():
    assert rs.stated({"contrast_change": {"op": ">=", "value": 1.25}}) == 1.5
    assert rs.stated({"contrast_change": {"op": "<=", "value": -2.75}}) == -3.0
    assert rs.stated({"contrast_change": {"op": "within", "value": 0.25}}) == 0.0
    assert rs.stated({"contrast_change": {"op": ">=", "value": 1.0, "about": 1.6}}) == 1.6
    assert rs.stated({"top": " dog"}) is None


def test_a_right_direction_with_too_large_a_magnitude_fails_magnitude_only():
    t = rs.terms({"contrast_change": {"op": "<=", "value": -3.75}}, {"pass": False, "specific": False, "change": -1.0})
    assert t["direction"] is True and t["magnitude"] is False
    t = rs.terms({"contrast_change": {"op": "<=", "value": -0.75}}, {"pass": True, "specific": True, "change": -1.1})
    assert t["direction"] is True and t["magnitude"] is True


def test_a_no_change_claim_matches_only_a_small_change():
    t = rs.terms({"contrast_change": {"op": "within", "value": 0.25}}, {"pass": True, "specific": True, "change": 0.1})
    assert t["direction"] and t["magnitude"]
    t = rs.terms({"contrast_change": {"op": "within", "value": 0.25}}, {"pass": False, "specific": False, "change": -2.0})
    assert not t["direction"] and not t["magnitude"]
