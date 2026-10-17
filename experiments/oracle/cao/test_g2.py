"""g2.py's measured outcomes as English clauses and claims (text_reader.py's questions use them too)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2  # noqa: E402


def test_a_flip_names_the_new_token():
    clause, pred = g2.outcome({"top_before": "Paris", "top_after": "Berlin", "change": -9.0}, "'Paris'")
    assert clause == "the model gives 'Berlin' instead of 'Paris'" and pred == {"top": "Berlin"}


def test_a_change_states_about_n_bits_and_claims_n_less_a_quarter():
    clause, pred = g2.outcome({"top_before": "P", "top_after": "P", "change": -2.1}, "'P'")
    assert clause.endswith("down about 2 bits") and pred == {"contrast_change": {"op": "<=", "value": -1.75}}
    clause, pred = g2.outcome({"top_before": "P", "top_after": "P", "change": 0.6}, "'P'")
    assert clause.endswith("up about 0.5 bits") and pred == {"contrast_change": {"op": ">=", "value": 0.25}}


def test_no_change_is_claimed_two_sided():
    clause, pred = g2.outcome({"top_before": "P", "top_after": "P", "change": 0.1}, "'P'")
    assert "almost unchanged" in clause and pred == {"contrast_change": {"op": "within", "value": g2.HALF}}


def test_listing():
    assert g2.listing(["G1"]) == "G1" and g2.listing(["G1", "G2", "G3"]) == "G1, G2 and G3"
