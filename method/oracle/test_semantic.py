"""Checks of semantic.py without a model: INVARIANTS lines parse into interventions and outcomes (vpd4l's tokenizer),
outcomes are priced by their surprise against the no-change prediction (a claim of the base top earns nothing, a bold
right one earns -log2 p_base(y), a bold wrong one loses it), and fit_gate finds the gate behind transplant outcomes.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_semantic.py
"""

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import components  # noqa: E402
import semantic  # noqa: E402


def test_invariants_parse():
    try:
        each = components._token_of("vpd4l", "' each'")
    except Exception as e:  # no tokenizer on this machine
        pytest.skip(str(e))
    text = "\n".join(["I1: V1 from @233=' all' -> top ' all'", "I2: @232=' of' -> unchanged",
                      "I3: V2 from @232=' were', V1 from @233=' all' -> not ' each'", "I4: remove V1 -> p(top) < 0.5",
                      "I5: @232=' of' -> sideways"])
    got, bad = semantic.parse_invariants(text)
    assert bad == 1 and [x["id"] for x in got] == [1, 2, 3, 4]
    assert got[0]["intervention"]["transplants"][0][0] == "V1" and got[0]["outcome"][0] == "top"
    assert got[1]["intervention"]["tokens"] == {232: components._token_of("vpd4l", "' of'")} and got[1]["outcome"] == ("unchanged",)
    assert [n for n, _ in got[2]["intervention"]["transplants"]] == ["V2", "V1"] and got[2]["outcome"] == ("not", each)
    assert got[3]["intervention"]["remove"] == ["V1"] and got[3]["outcome"] == ("drop", 0.5)


def _checker(p0: list[float]):
    ck = object.__new__(semantic.Checker)
    ck.p0 = torch.tensor(p0).log()
    ck.y0 = int(ck.p0.argmax())
    return ck


def test_outcomes_priced_against_no_change():
    ck = _checker([0.7, 0.2, 0.1])
    moved = torch.tensor([0.1, 0.8, 0.1]).log()
    r = ck.outcome(moved, ("top", 1))
    assert r["holds"] and math.isclose(r["surprise"], -math.log2(0.2), rel_tol=1e-6) and r["credit"] > 0
    r = ck.outcome(moved, ("top", 2))
    assert not r["holds"] and math.isclose(r["credit"], math.log2(0.1), rel_tol=1e-6)
    assert ck.outcome(moved, ("top", 0))["surprise"] == 0.0  # the base top: no claim beyond no change
    r = ck.outcome(moved, ("not", 0))
    assert r["holds"] and math.isclose(r["surprise"], -math.log2(0.3), rel_tol=1e-6)
    assert ck.outcome(moved, ("unchanged",))["credit"] == 0.0
    r = ck.outcome(moved, ("drop", 0.5))
    assert r["holds"] and math.isclose(r["surprise"], 1.0, rel_tol=1e-6)
    assert math.isclose(ck.outcome(moved, ("top", 1))["logp"], math.log2(0.8), rel_tol=1e-5)


def test_fit_gate():
    rows = [(a, b, 23 if (a and b) else 24) for a in (False, True) for b in (False, True)] * 2
    assert semantic.fit_gate(rows, 23)[0] == ("AND", 1.0)
    rows = [(a, b, 23 if a == b else 24) for a in (False, True) for b in (False, True)]
    assert semantic.fit_gate(rows, 23)[0] == ("XNOR", 1.0)


@pytest.fixture
def invariant_tokenizer(monkeypatch):
    tokens = {"' x'": 1, "' y'": 2, "' z'": 3, "' a,b'": 4}
    monkeypatch.setattr(components, "_token_of", lambda model, text: tokens.get(text))


def test_invariant_quote_styles_resolve_identical_tokens(invariant_tokenizer):
    single, bad = semantic.parse_invariants("I1: V1 from @0=' x' -> top ' y'")
    double, other_bad = semantic.parse_invariants('I1: V1 from @0=" x" -> top " y"')
    assert bad == other_bad == 0
    assert single[0]["intervention"] == double[0]["intervention"]
    assert single[0]["outcome"] == double[0]["outcome"]
    assert double[0]["line"] == 'I1: V1 from @0=" x" -> top " y"'


@pytest.mark.parametrize("literal", ['"x"', '" missing"', '" x "'])
def test_invariant_quote_normalization_keeps_exact_token_text(invariant_tokenizer, literal):
    got, bad = semantic.parse_invariants(f'I1: @0={literal} -> top " y"')
    assert got == [] and bad == 1


def test_invariant_quote_normalization_preserves_shown_escaping(monkeypatch):
    token = '<|endoftext|>'
    spelling = components.shown(token)
    monkeypatch.setattr(components, "_token_of", lambda model, text: 7 if text == spelling else None)
    for literal in (spelling, '"<|endoftext|>"'):
        got, bad = semantic.parse_invariants(f'I1: @0={literal} -> top {literal}')
        assert bad == 0 and got[0]["intervention"]["tokens"] == {0: 7}
        assert got[0]["outcome"] == ("top", 7)


def test_invariant_equivalent_quotes_cannot_duplicate_credit(invariant_tokenizer):
    got, bad = semantic.parse_invariants('I1: @0=\' x\' -> top \' y\'\nI2: @0=" x" -> top " y"')
    assert len(got) == 1 and bad == 1


@pytest.mark.parametrize("donor", ["@0=' x' unless V2 is false", "if V2 @0=' x'",
                                   "@0=' x' and @1=' z'", "@0=' x' @0=' z'",
                                   "@0=' x' @0=' x'", "@0=' missing'"])
def test_invariant_donor_claim_must_be_fully_executable(invariant_tokenizer, donor):
    got, bad = semantic.parse_invariants(f"I1: V1 from {donor} -> top ' y'")
    assert got == [] and bad == 1


def test_invariant_donor_preserves_complete_multiple_changes(invariant_tokenizer):
    got, bad = semantic.parse_invariants("I1: V1 from @0=' x' @1=' a,b' -> top ' y'")
    assert bad == 0
    assert got[0]["intervention"]["transplants"] == [("V1", {0: 1, 1: 4})]


def test_unexecuted_condition_cannot_earn_verified_english_credit(invariant_tokenizer, monkeypatch):
    executed = []
    class FakeChecker:
        def __init__(self, *args):
            pass
        def check(self, items):
            executed.extend(items)
            return [{"holds": True, "credit": 3.0} for _ in items]
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    good = "I1: V1 from @0=' x' -> top ' y'"
    assert semantic.check_invariants(None, None, [{"name": "V1"}], good)["credit"] == 3.0
    executed.clear()
    unsupported = "I1: V1 from @0=' x' unless V2 is false -> top ' y'"
    result = semantic.check_invariants(None, None, [{"name": "V1"}], unsupported)
    assert result["bad"] == 1 and result["holds"] == 0.0 and result["credit"] == -1.0
    assert result["invariants"] == [] and executed == []


@pytest.mark.parametrize("intervention", ["@0=' x', @0=' z'", "@0=' x', @0=' x'",
                                          "V1 from @0=' x', V1 from @0=' z'",
                                          "V1 from @0=' x', remove V1", "remove V1, V1 from @0=' x'",
                                          "remove V1, remove V1"])
def test_invariant_cannot_overwrite_a_stated_intervention(invariant_tokenizer, intervention):
    got, bad = semantic.parse_invariants(f"I1: {intervention} -> top ' y'")
    assert got == [] and bad == 1


def test_distinct_carriers_can_use_different_donors_at_the_same_position(invariant_tokenizer):
    got, bad = semantic.parse_invariants("I1: V1 from @0=' x', V2 from @0=' z' -> top ' y'")
    assert bad == 0
    assert got[0]["intervention"]["transplants"] == [("V1", {0: 1}), ("V2", {0: 3})]


@pytest.mark.parametrize("copies", [2, 10])
def test_repeated_invariant_ids_cannot_inflate_evidence_credit(invariant_tokenizer, monkeypatch, copies):
    class FakeChecker:
        def __init__(self, *args):
            pass
        def check(self, items):
            assert len(items) == 1
            return [{"holds": True, "credit": 3.0}]
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    claim = "I1: V1 from @0=' x' -> top ' y'"
    result = semantic.check_invariants(None, None, [{"name": "V1"}], "\n".join([claim] * copies))
    assert len(result["invariants"]) == 1
    assert result["bad"] == copies - 1
    assert result["credit"] == 3.0 - (copies - 1)


@pytest.mark.parametrize("fraction", ["0", "1", "2", "9999999999999999999999999999999999999999"])
def test_drop_claim_requires_a_strict_probability_reduction(invariant_tokenizer, fraction):
    got, bad = semantic.parse_invariants(f"I1: remove V1 -> p(top) < {fraction}")
    assert got == [] and bad == 1


def test_valid_drop_fraction_keeps_priced_outcome(invariant_tokenizer):
    got, bad = semantic.parse_invariants("I1: remove V1 -> p(top) < 0.5")
    assert bad == 0
    assert got[0]["outcome"] == ("drop", 0.5)


@pytest.mark.parametrize("claims", [
    ["I1: V1 from @0=' x' -> top ' y'", "I2: V1 from @0=' x' -> top ' y'"],
    ["I1: V1 from @0=' x', V2 from @1=' z' -> top ' y'",
     "I2: V2 from @1=' z', V1 from @0=' x' -> top ' y'"],
    ["I1: V1 from @0=' x' @1=' z' -> top ' y'", "I2: V1 from @1=' z' @0=' x' -> top ' y'"],
    ["I1: @0=' x', @1=' z', remove V1, remove V2 -> top ' y'",
     "I2: remove V2, @1=' z', remove V1, @0=' x' -> top ' y'"],
])
def test_same_evidence_under_different_ids_cannot_multiply_credit(invariant_tokenizer, monkeypatch, claims):
    class FakeChecker:
        def __init__(self, *args):
            pass
        def check(self, items):
            assert len(items) == 1
            return [{"holds": True, "credit": 3.0}]
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    result = semantic.check_invariants(None, None, [{"name": "V1"}, {"name": "V2"}], "\n".join(claims))
    assert len(result["invariants"]) == 1
    assert result["bad"] == 1 and result["credit"] == 2.0


def test_different_outcomes_and_values_are_distinct_evidence(invariant_tokenizer):
    text = "\n".join(["I1: V1 from @0=' x' -> top ' y'", "I2: V1 from @0=' x' -> not ' y'",
                       "I3: V1 from @0=' z' -> top ' y'", "I4: V2 from @0=' x' -> top ' y'"])
    got, bad = semantic.parse_invariants(text)
    assert bad == 0 and len(got) == 4
