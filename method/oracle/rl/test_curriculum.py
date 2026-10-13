"""The RL curriculum (tinker_train.STAGES): each stage's reward and line credits, and the advance on a held-out plateau."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tinker_train as tt


def test_a_stage_rewards_only_its_terms_and_keeps_the_gate():
    s = {"valid": True, "reward": 0.7, "g": 0.5, "g_floor": 0.4, "E4x": 0.4, "E2g": 0.2}
    assert tt.stage_reward(s, 0) == pytest.approx(0.7 - 0.4 * (0.4 + 0.2))  # the scorer's R less the left-out terms
    assert tt.stage_reward(s, 1) == pytest.approx(0.7 - 0.4 * 0.2)
    assert tt.stage_reward(s, 2) == pytest.approx(0.7)
    assert tt.stage_reward({"valid": True, "reward": 0.7, "g": 0.5}, 0) == pytest.approx(0.7)  # the circuit arm: R as scored
    assert tt.stage_reward({"valid": True, "reward": 0.7, "g": 0.5, "term_E4x": 0.4}, 1) == pytest.approx(0.7)
    assert tt.stage_reward({"valid": False, "reward": -10.0}, 3) == -10.0


def test_a_stage_credits_the_lines_of_its_terms():
    text = "CIRCUIT:\n```mermaid\nflowchart LR\n  a --> OUT((prediction))\n```\nEXPLANATION: V1 x\nANSWER 1:\nFINAL 1: falls=no top=' x'"
    credits = [[3, 0.1], [5, 0.3], [7, 0.5]]
    assert tt.stage_lines(text, credits, 0) == [[3, 0.1]]
    assert tt.stage_lines(text, credits, 1) == [[3, 0.1], [5, 0.3]]
    assert tt.stage_lines(text, credits, 2) == credits


def test_the_curriculum_advances_when_the_held_out_evaluations_stop_improving(tmp_path):
    for step, v in ((25, 0.5), (50, 0.6), (75, 0.55), (100, 0.58)):
        rows = [{"samples": [{"valid": True, "reward": v, "g": 0.0}]}]
        (tmp_path / f"heldout_{step}.json").write_text(json.dumps({"per_question": {"rl": rows}}))
    assert tt.next_stage(tmp_path, 0, 0) == 1  # 75 and 100 below 50's 0.6
    assert tt.next_stage(tmp_path, 0, 75) == 0  # too few evaluations in the stage
    assert tt.next_stage(tmp_path, len(tt.STAGES) - 1, 0) == len(tt.STAGES) - 1


def test_the_circuit_arm_trains_on_the_scorers_r_and_stops_otherwise():
    q = {"circuit_only": True}
    scores = [{"valid": True, "reward": 0.7, "g": 0.5, "term_E2g": None, "term_E4x": None, "term_E1": None},
              {"valid": False, "reward": -10.0}]
    assert tt.reward_mismatches(q, scores, [tt.stage_reward(s, 0) for s in scores]) == []
    leak = [{**scores[0], "term_E4x": 0.4}, scores[1]]  # a left-out term on a circuit-only answer: the trained reward moves
    assert tt.reward_mismatches(q, leak, [tt.stage_reward(s, 0) for s in leak]) == [0]
    assert tt.reward_mismatches(q, scores, [float("nan"), -10.0]) == [0]
    assert tt.reward_mismatches({}, leak, [tt.stage_reward(s, 0) for s in leak]) == []  # stage terms apply off the circuit arm


def test_the_run_stops_on_a_scorer_outside_its_frozen_reward():
    ok = [{"semantics": "n", "source": "fb33713abc"}] * 2
    assert tt.scorer_mismatch(ok, "n", "fb33713") is None
    assert tt.scorer_mismatch(ok, None, None) is None
    assert "semantics 'c'" in tt.scorer_mismatch([ok[0], {"semantics": "c", "source": "fb33713abc"}], "n", None)
    assert "source" in tt.scorer_mismatch([{"semantics": "n", "source": None}], "n", "fb33713")
