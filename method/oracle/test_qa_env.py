"""qa_env (RL environment B): the reward's parsing and credit, the constant answers, and the balance of the kept
questions, on hand-made rows (no model)."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qa_env  # noqa: E402

CHANGE = {"kind": "remove", "ask": "change", "node": ["h.0.mlp.c_fc", 3, 7], "falls": True, "top_changed": True,
          "topk": [["' a'", 0.5], ["' the'", 0.25], ["' of'", 0.1]]}
KL = {"kind": "token", "ask": "kl", "position": 2, "new": 5, "answer": False}
COMPARE = {"kind": "compare", "ask": "compare", "positions": [4, 9], "effects": [0.3, 0.1], "answer": 4}
ROW = {"task": "text0", "question": "q", "top0": "' the'", "qa_env": [CHANGE, KL, COMPARE]}


def test_credit_per_kind():
    s = qa_env.score(ROW, "FINAL 1: falls=yes top=' the'\nFINAL 2: no\nFINAL 3: position 4")
    assert s["valid"] and s["by_kind"] == {"remove": pytest.approx(0.5 + 0.5 * 0.25 / 0.5), "kl": 1.0, "compare": 1.0}
    assert s["reward"] == pytest.approx((0.75 + 1 + 1) / 3)
    s = qa_env.score(ROW, "FINAL 1: falls=no top=' x'\nFINAL 2: yes\nFINAL 3: 9")  # every part wrong
    assert s["reward"] == 0.0 and s["valid"]
    s = qa_env.score(ROW, "I think it falls.")  # no FINAL line: nothing answered
    assert s["reward"] == 0.0 and not s["valid"]
    assert qa_env.score(ROW, "FINAL 1: falls=yes top=' a'")["reward"] == pytest.approx(1 / 3)  # the others unanswered


def test_constant_answers_score_the_unchanged_top():
    tab = qa_env.baseline_table([ROW])
    assert tab["no change"]["remove"] == pytest.approx(0.5 * 0.25 / 0.5)  # falls wrong, the current top a second
    assert tab["falls"]["remove"] == pytest.approx(0.5 + 0.25)
    assert tab["no change"]["kl"] == 1.0 and tab["falls"]["kl"] == 0.0
    assert tab["no change"]["compare"] == 1.0 and tab["falls"]["compare"] == 0.0


def test_balance_halves_each_answer():
    rng = random.Random(0)
    cands = ([{"kind": "remove", "ask": "change", "cell": c} for c in ["True/True"] * 3 + ["True/False"] * 5 + ["False/False"] * 20 + ["False/True"] * 2]
             + [{"kind": "token", "ask": "kl", "cell": c} for c in ["True"] * 4 + ["False"] * 9]
             + [{"kind": "compare", "ask": "compare", "cell": c} for c in ["True"] * 7 + ["False"] * 3])
    got = qa_env.balance(cands, rng)
    rm = [q["cell"] for q in got if q["ask"] == "change"]
    assert sum(c.startswith("True") for c in rm) == sum(c.startswith("False") for c in rm) == 8
    assert sum(c == "True/True" for c in rm) == 3 and sum(c == "False/True" for c in rm) == 2  # changed tops as available
    assert sorted(q["cell"] for q in got if q["ask"] == "kl") == ["False"] * 4 + ["True"] * 4
    assert sorted(q["cell"] for q in got if q["ask"] == "compare") == ["False"] * 3 + ["True"] * 3


def test_evidence_drops_the_circuit_ask_and_its_questions():
    q = "How does vpd4l compute ...\ntokens: 0:'a'\ntarget: position 3\nQuestion 1: if ...?\nEdit 1: ..."
    assert qa_env.evidence_lines(q) == ["tokens: 0:'a'", "target: position 3"]


def test_order_changes_balance_as_one_kind():
    cands = ([{"kind": "move", "ask": "change", "cell": "True/True"}] * 2 + [{"kind": "insert", "ask": "change", "cell": "False/False"}] * 5
             + [{"kind": "swap", "ask": "change", "cell": "False/True"}])
    got = qa_env.balance([dict(q) for q in cands], random.Random(0))
    assert qa_env.count(got) == {"order": 4} and sum(q["cell"].startswith("True") for q in got) == 2
