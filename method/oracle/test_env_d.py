"""env_d.py's reward, gate summary, question and SFT rows and scorer service, on test_components' fake reader."""
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_d  # noqa: E402
from test_components import FakeReader  # noqa: E402

EV = {"events": [{"position": p, "new": 1, "alt": 0, "listed": False, "down": True, "weight": w, "bits": -w}
                 for p, w in enumerate([0.1, 3.0, 0.5] + [0.01] * 20)], "listed": [], "positions": []}
FACTS = {"task": "text1", "circuit": "@1 a=tok\nOUT <- a", "text": "a b c", "target": 2, "target_token": " c",
         "top": [[" d", 0.5]], "nodes_total": 1, "positions": [1], "nodes": [{"node": "tok@1", "token": " b", "facts": ""}],
         "evidence": [], "edges": ["tok@1 -> OUT"], "qa": []}


def test_reward_is_e1_of_the_english_read_alone():
    """The answer's English (after its EXPLANATION heading, else the whole text) is read alone, never any circuit the
    answer writes; no English scores 0; E1 ranks the English that names the heavier event first; every score names
    the reward's definition."""
    r = FakeReader()
    got = env_d.score(r, {}, EV, ["EXPLANATION: position 2 matters", "position 9 matters",
                                  "CIRCUIT:\n@5 b=tok\nEXPLANATION: position 2 matters", ""])
    assert got[0]["reward"] == pytest.approx(got[2]["reward"]) and got[0]["reward"] > got[1]["reward"]
    assert all(c == "" for c, _, _ in r.asked)
    assert {k: got[3][k] for k in ("reward", "e1", "valid", "words", "named")} == {"reward": 0.0, "e1": 0.0, "valid": True,
                                                                                   "words": 0, "named": False}
    assert got[0]["semantics"] == env_d.SEMANTICS == "english-alone " + env_d.components.E1_DEFINITION
    assert env_d.score(FakeReader(), {}, EV, ["L1.v#3 at position 2"])[0]["named"]
    assert env_d.english_of("EXPLANATION:  two\n lines ") == "two lines"


def test_gate_summary():
    """The gate holds when the measured-facts English beats its position edit on >= GATE of the texts;
    "mechanism_claims" when the correct English beats its mechanism edit on >= GATE; shuffled and none reported."""
    rows = [{"task": f"t{i}", "e1": {"correct": 0.5, "shuffled": -1.0, "position": 0.1, "mechanism": 0.9 if i else 0.1,
                                     "facts": 0.3, "facts_position": 0.1 if i else 0.4}} for i in range(5)]
    s = env_d.summarize(rows)
    assert s["facts_beats_facts_position"] == {"wins": 4, "of": 5, "share": 0.8} and s["gate"]
    assert s["beats"]["mechanism"] == {"wins": 1, "of": 5, "share": 0.2} and not s["mechanism_claims"]
    assert s["beats"]["none"]["share"] == 1.0 and s["mean_e1"]["none"] == 0.0
    rows[1]["e1"]["facts_position"] = 0.6
    assert not env_d.summarize(rows)["gate"]


def test_questions_and_sft_rows(tmp_path):
    (tmp_path / "f.jsonl").write_text(json.dumps(FACTS) + "\n")
    (tmp_path / "e.jsonl").write_text(json.dumps({"task": "text1", "explanation": "It reads b.", "e1_best": 0.3}) + "\n"
                                      + json.dumps({"task": "text1", "explanation": ""}) + "\n")
    env_d.questions(types.SimpleNamespace(facts=tmp_path / "f.jsonl", out=tmp_path / "q.jsonl"))
    env_d.sft(types.SimpleNamespace(english=tmp_path / "e.jsonl", facts=tmp_path / "f.jsonl", out=tmp_path / "s.jsonl"))
    q = json.loads((tmp_path / "q.jsonl").read_text())
    s = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert q["task"] == "text1" and "EXPLANATION" in q["question"] and len(s) == 1
    assert s[0]["prompt"][0]["content"] == q["question"] and s[0]["completion"][0]["content"] == "EXPLANATION: It reads b."


def test_serve(tmp_path, monkeypatch):
    """POST /score returns the reward per answer, as score() gives it."""
    (tmp_path / "f.jsonl").write_text(json.dumps(FACTS) + "\n")
    (tmp_path / "r.jsonl").write_text(json.dumps({"task": "text1", **EV}) + "\n")
    monkeypatch.setattr(env_d, "make_reader", lambda spec: FakeReader())
    monkeypatch.setattr(env_d, "task_of", lambda t: {})
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    a = types.SimpleNamespace(facts=tmp_path / "f.jsonl", rows=tmp_path / "r.jsonl", reader="x", port=port)
    threading.Thread(target=env_d.serve, args=(a,), daemon=True).start()
    for _ in range(50):
        try:
            r = httpx.post(f"http://127.0.0.1:{port}/score", json={"task": "text1", "answers": ["position 2 matters", ""]})
            break
        except httpx.ConnectError:
            time.sleep(0.1)
    got = r.json()["scores"]
    want = env_d.score(FakeReader(), {}, EV, ["position 2 matters", ""])
    assert [g["reward"] for g in got] == pytest.approx([w["reward"] for w in want])


def test_e1_asks_every_structure_event():
    """components.e1_bits (E1_DEFINITION e1v2): beside the E1_EVENTS heaviest, every structure event is asked, however
    light (the events that can refute a claim about how the model finds what it uses)."""
    ev = {**EV, "structure": [{"kind": "move", "position": 3, "change": f"move {k}", "down": True, "weight": 0.001,
                               "bits": -0.001} for k in range(3)]}
    r = FakeReader()
    env_d.score(r, {}, ev, ["position 2 matters"])
    neutral = [evs for c, eng, evs in r.asked if eng == env_d.components.NEUTRAL][0]
    assert len(neutral) == env_d.components.E1_EVENTS + 3
    assert [e["change"] for e in neutral if e.get("change")] == ["move 0", "move 1", "move 2"]


def test_e1_split_takes_e1_apart():
    """env_d.e1_split: its E1 is components.e1_bits' (the same reader calls), and its position and structure parts are
    the same weighted sum over each kind of event apart (decision test 3's breakdown)."""
    ev = {**EV, "structure": [{"kind": "move", "position": 3, "change": f"move {k}", "down": True, "weight": 0.5,
                               "bits": -0.5} for k in range(2)]}
    texts = ["position 2 matters", "position 9 matters", ""]
    whole = env_d.components.e1_bits(FakeReader(), {}, ev, [env_d.answer("", t) for t in texts])
    split = env_d.e1_split(FakeReader(), {}, ev, [env_d.answer("", t) for t in texts])
    assert [s["e1"] for s in split] == pytest.approx(whole)
    assert split[2] == {"e1": 0.0, "position": 0.0, "structure": 0.0}
    assert all(s["position"] is not None and s["structure"] is not None for s in split)
