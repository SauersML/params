"""Checks of reader_reward.py's parsing and grading without a model or Tinker.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_reader_reward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reader_reward as rr  # noqa: E402


class FakeTarget:
    ids = [0, 1, 2, 3]
    VOCAB = {0: "The", 1: " new", 2: " the", 3: " creation", 4: " building", 5: " work"}

    def tok(self, i):
        return self.VOCAB[int(i)]

    def token_id(self, s):
        return next((i for i, v in self.VOCAB.items() if v == s), None)

    where = rr.VPDTarget.where

    def fresh(self, pos, n, rng):
        return [" building", " work"][:n]

    def valid(self, part):
        n, p, c = part
        return n in ("h.3.mlp.down_proj", "h.0.mlp.c_fc", "h.0.attn.q_proj") and 0 <= p < len(self.ids) and c < 4096


def test_parts_and_positions():
    tg = FakeTarget()
    assert rr.site_of("L3.down#2481@3") == ("h.3.mlp.down_proj", 3, 2481)
    assert rr.site_of("L0.q#7@' creation'", tg.where.__get__(tg)) == ("h.0.attn.q_proj", 3, 7)
    assert rr.site_of("L0.x#7@3") is None
    assert rr.label("h.1.attn.o_proj", 2, 9) == "L1.o#9@2"


def test_invariant_lines():
    tg = FakeTarget()
    tg.where = rr.VPDTarget.where.__get__(tg)
    assert rr.parse_inv("INV: swap @' creation' with any of [' building', ' work'] -> top ' the'", tg) is None  # listed by the answer
    ivs, oc = rr.parse_inv("INV: swap @' creation' with predicted tokens -> top ' the'", tg)
    assert [iv["token"] for iv in ivs] == [" building", " work"] and ivs[0]["pos"] == 3 and oc == ("top", " the")
    assert rr.parse_inv("INV: swap @3 with things it likes -> p > 0.5", tg) is None  # no such scope
    ivs, oc = rr.parse_inv("INV: swap @3 with predicted tokens and set back L3.down#2481@3, L0.fc#1@2 -> p > 0.7", tg)
    assert ivs[0]["kind"] == "setback" and ivs[0]["parts"] == [["h.3.mlp.down_proj", 3, 2481], ["h.0.mlp.c_fc", 2, 1]] and oc == (">", 0.7)
    ivs, oc = rr.parse_inv("INV: remove L3.down#2481@3 -> top not ' of'", tg)
    assert ivs[0]["kind"] == "remove" and oc == ("top_not", " of")
    ivs, _ = rr.parse_inv("INV: scale L3.down#2481@3 by 0.5 -> p < 0.4", tg)
    assert ivs[0]["by"] == 0.5
    assert rr.parse_inv("INV: remove L3.down#2481@9 -> p > 0.5", tg) is None  # no position 9
    assert rr.parse_inv("INV: remove L4.o#1@3 -> p > 0.5", tg) is None  # no layer 4
    assert rr.parse_inv("INV: remove L3.down#2481@'creation' -> p > 0.5", tg)[0][0]["parts"] == [["h.3.mlp.down_proj", 3, 2481]]
    assert rr.parse_inv("swap @3 with any of [' work'] -> p > 0.5", tg) is None
    tg.named = {"G1": [["h.0.mlp.c_fc", 3, 5], ["h.0.mlp.c_fc", 3, 6]]}
    ivs, oc = rr.parse_inv("INV: remove G1, L3.down#2481@3 -> p < 0.2", tg)
    assert ivs[0]["parts"] == [["h.0.mlp.c_fc", 3, 5], ["h.0.mlp.c_fc", 3, 6], ["h.3.mlp.down_proj", 3, 2481]]
    assert rr.parse_inv("INV: remove G2 -> p < 0.2", tg) is None  # no group G2
    assert rr.meets({"top": " of", "p": 0.2}, ("top", " of")) and rr.meets({"top": " of", "p": 0.2}, ("<", 0.3))
    assert not rr.meets({"top": " of", "p": 0.2}, ("top_not", " of"))


def test_answers_and_grades():
    got = rr.parse_answers("Q1: ' of' 0.9\nQ3: \"'s\" 0.1\nQ2: nonsense", 3)
    assert got == [(" of", 0.9), (None, None), ("'s", 0.1)]
    ivs = [{"kind": "swap", "outcome": {"top": " of", "p": 0.9}}, {"kind": "remove", "outcome": {"top": " of", "p": 0.5}},
           {"kind": "swap", "outcome": {"top": " and", "p": 0.01}}]
    rows = rr.grade(got, ivs)
    assert rows[0]["top_ok"] and abs(rows[0]["p_credit"] - 1) < 1e-9 and rows[1]["score"] == 0.0
    assert not rows[2]["top_ok"] and 0 < rows[2]["p_credit"] < 1
