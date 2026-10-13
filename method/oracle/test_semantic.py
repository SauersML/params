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
