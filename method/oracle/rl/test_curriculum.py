"""The RL curriculum (tinker_train.STAGES): each stage's reward and line credits, and the advance on a held-out plateau."""
import json
import math
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


def test_the_run_pins_its_scorers_environment_and_donor_pool():
    pins = {}
    a = [{"environment": "torch 2.10.0", "donor_pool": "search:400:abc"}] * 2
    assert tt.pinned_check(pins, "circuit", a) is None
    assert pins == {"circuit": {"environment": "torch 2.10.0", "donor_pool": "search:400:abc"}}
    assert tt.pinned_check(pins, "circuit", a) is None
    assert "donor_pool" in tt.pinned_check(pins, "circuit", [{"environment": "torch 2.10.0", "donor_pool": "search:399:def"}])
    said = []
    assert tt.pinned_check(pins, "circuit", [{"environment": "torch 2.4.1", "donor_pool": "search:400:abc"}], warn=said.append) is None
    assert len(said) == 1 and "environment" in said[0]  # another Python environment: reported, the run goes on
    assert tt.pinned_check(pins, "edits", [{"reward": 1.0}]) is None and pins["edits"] == {}  # a scorer naming neither
    said = []
    assert tt.pinned_check({}, "circuit", a, {"environment": "torch 2.4.1"}, warn=said.append) is None and len(said) == 1


def test_the_run_stops_on_a_scorer_outside_its_frozen_reward():
    ok = [{"semantics": "n", "source": "fb33713abc"}] * 2
    assert tt.scorer_mismatch(ok, "n", "fb33713") is None
    assert tt.scorer_mismatch(ok, None, None) is None
    assert "semantics 'c'" in tt.scorer_mismatch([ok[0], {"semantics": "c", "source": "fb33713abc"}], "n", None)
    assert "source" in tt.scorer_mismatch([{"semantics": "n", "source": None}], "n", "fb33713")
    assert tt.scorer_mismatch(ok, "n", "fb33713abc" + "0" * 30) is None  # a full revision against the pod's short one
    assert "source" in tt.scorer_mismatch(ok, "n", "fb33714")


def test_environments_load_mix_by_weight_and_scale_by_their_own_spread(tmp_path):
    import random
    from types import SimpleNamespace

    (tmp_path / "c.jsonl").write_text("".join(json.dumps({"task": f"t{i}", "question": "q"}) + "\n" for i in range(10)))
    (tmp_path / "e.jsonl").write_text("".join(json.dumps({"id": f"pref:{i}", "question": "q"}) + "\n" for i in range(10)))
    (tmp_path / "envs.json").write_text(json.dumps({"edits": {"rows": str(tmp_path / "e.jsonl"), "scorer": "http://x:8766/score",
                                                              "extra": {"part": "score"}, "weight": 1.0}}))
    a = SimpleNamespace(questions=tmp_path / "c.jsonl", scorer="http://x:8767/score", request_extra=None, env_name="circuit",
                        env_weight=3.0, expect_semantics="n", expect_source="d210d58", envs=str(tmp_path / "envs.json"),
                        reward_floor=None, reward_floor_width=1.0)
    envs = tt.load_envs(a)
    assert envs["circuit"]["floor"] is None and envs["edits"]["floor"] is None and envs["edits"]["floor_width"] == 1.0
    assert tt.floored(5.0, -1.0) == pytest.approx(5.0, abs=0.01) and tt.floored(-13.0, -1.0) == pytest.approx(-1.0, abs=1e-4)
    assert tt.floored(-1.0, -1.0) == pytest.approx(-1.0 + math.log(2))  # smooth at the floor
    assert tt.floored(-2.0, -1.0) < tt.floored(-1.9, -1.0)  # a gradient below it
    assert envs["edits"]["questions"][0]["task"] == "pref:0" and envs["edits"]["extra"] == {"part": "score"}
    assert envs["edits"]["expect_semantics"] is None and envs["circuit"]["expect_semantics"] == "n"
    got = tt.draw(envs, 4, random.Random(0))
    assert sorted(q["_env"] for q in got) == ["circuit", "circuit", "circuit", "edits"]  # 3:1 by weight
    one = {"circuit": envs["circuit"]}
    assert tt.draw(one, 4, random.Random(0)) == random.Random(0).sample(envs["circuit"]["questions"], 4)  # as before envs
    s = tt.env_scales([("circuit", [1.0, -1.0]), ("edits", [0.1, -0.1]), ("flat", [0.0, 0.0])])
    assert s["circuit"] == pytest.approx(1.0) and s["edits"] == pytest.approx(10.0) and s["flat"] == 1.0


def test_devscore_scores_each_new_checkpoint_once_with_the_environments_request(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "checkpoints.jsonl").write_text(json.dumps({"step": 0, "state": "s0", "base": "b"}) + "\n" +
                                                json.dumps({"step": 5, "state": "s5", "base": "b"}) + "\n")
    (tmp_path / "devscore_0.json").write_text(json.dumps({"step": 0, "reward": 0.0}))
    (tmp_path / "dev.jsonl").write_text(json.dumps({"task": "g1", "question": "fixture"}) + "\n")
    sampled, posted = [], []

    def fake_sample(ns):
        sampled.append(ns.init)
        ns.out.write_text(json.dumps({"task": "g1", "sample": 0, "text": "scale L0.fc#1 0", "tokens": 9, "thinking_tokens": 2}) + "\n")

    class Reply:
        status_code = 200

        def __init__(self, req):
            self.req = req

        def json(self):
            return {"scores": [{"reward": 0.4, "valid": True}]}

    monkeypatch.setattr(tt, "sample", fake_sample)
    monkeypatch.setattr(tt.httpx, "post", lambda url, json, timeout: posted.append(json) or Reply(json))
    a = SimpleNamespace(log=tmp_path, questions=tmp_path / "dev.jsonl", scorer="http://x/score", request_extra='{"part": "score"}',
                        base="b", n=3, max_tokens=64, thinking=False)
    tt.devscore(a)
    assert sampled == ["s5"] and posted == [{"task": "g1", "answers": ["scale L0.fc#1 0"], "lines": False, "tokens": [7], "part": "score"}]
    assert json.loads((tmp_path / "devscore_5.json").read_text())["reward"] == 0.4
