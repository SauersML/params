"""Checks of component answers (components.py): the text format both ways, question scoring, and the reward on a tiny
random model (test_goal_eval's).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_components.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402

ANSWER = """EXPLANATION: the value at 3 is copied to the last position.
CIRCUIT:
out <- L1.down#2@5, L1.o#1@5
L1.o#1@5 <- L1.v#4@3
L1.v#4@3 <- L0.down#0@3
L0.down#0@3 <- L0.fc#5@3
L1.k#3@3
ANSWER 1: the copied value is gone.
FINAL 1: falls=yes top=' the'
"""


def test_parse_render_round_trip():
    p = components.parse(ANSWER)
    assert p["malformed"] == 0 and p["circuit"]
    assert p["explanation"] == "the value at 3 is copied to the last position."
    assert p["finals"] == {1: (True, " the")}
    assert len(p["answer"]["nodes"]) == 6 and len(p["answer"]["edges"]) == 5
    text = components.render(p["answer"], p["finals"], {1: "the copied value is gone."})
    assert text.strip() == ANSWER.strip()
    assert components.parse(text)["answer"] == p["answer"]


def test_malformed_lines_and_dropped_edges():
    p = components.parse("EXPLANATION: x\nCIRCUIT:\nout <- L1.o#1@5\nL1.o#1@5 <- L1.v#4@3 and more\nL7\n")
    assert p["malformed"] == 2
    C = {native.site_name(layer, k): 6 for layer in range(2) for k in components.KINDS.values()}
    q = components.parse("EXPLANATION: x\nCIRCUIT:\nout <- L1.v#4@5\nL0.o#1@5 <- L1.down#0@5\n")
    assert q["malformed"] == 0
    _, d = goal_eval.circuit(q["answer"], [5], C, 6)
    assert d["invalid_edges"] == 2  # a value does not write the residual stream; a later MLP does not feed an earlier output


def test_qa_score():
    qs = [{"falls": True, "top": 0}, {"falls": False, "top": 1}]
    tok = lambda ids: [f"t{i}" for i in ids]  # noqa: E731
    assert components.qa_score(qs, {1: (True, "t0"), 2: (False, "t1")}, tok) == 1.0
    assert components.qa_score(qs, {1: (False, "t0")}, tok) == 0.25
    assert components.qa_score([], {}, tok) is None


class NoWiring:
    names = []

    def partners(self, name, idx, kind):
        return {k: np.empty(0) for k in ("site", "index", "head", "raw", "cosine")}


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native()
    ge = goal_eval.GoalEval(nat, changes=4, random_pairs=4)
    ids = [3, 17, 5, 9, 22, 11, 30, 2]
    task = {"id": "text0", "model": "vpd4l", "split": "heldout",
            "prompts": [{"token_ids": ids, "target_positions": [7], "model_top": [[[" the", 0.5]]]}]}
    ev = components.evidence(nat, ge, task, NoWiring())
    return nat, ge, task, ev


def test_reward_on_tiny_model(tiny):
    nat, ge, task, ev = tiny
    assert ev["positions"] and ev["active"] > 0
    order = nat._vpd_order(task["prompts"][0]["token_ids"], [7])
    at_t = [nd for nd in order if nd[1] == 7 and nd[0].endswith(("o_proj", "down_proj"))][:3]
    g = native.Graph(at_t, {}, at_t)
    finals = {n: (q["falls"], "x") for n, q in enumerate(ev["qa"], 1)}
    good = components.render(goal_eval.to_json(g, "writes at the target"), finals)
    bad = good.replace("out <- ", "out <- L0.v#0@7, ")  # a value never writes the residual stream
    missing = components.render(goal_eval.to_json(g, "writes at the target"))
    got = components.reward(ge, task, ev, [good, bad, missing], ranks=(0, 1))
    assert got[0]["valid"] and -1.0 < got[0]["reward"] <= 1.0
    assert all(0.0 <= got[0][f"term_{k}"] <= 1.0 for k in ("F1", "F2", "F3"))
    assert got[0]["term_E2"] == (0.5 if ev["qa"] else None)  # every "falls" right, no top token "x"
    assert not got[1]["valid"] and got[1]["reward"] == -1.0
    assert (not got[2]["valid"]) == bool(ev["qa"])
