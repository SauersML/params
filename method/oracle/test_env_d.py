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


def test_reward_is_e1_of_the_english_read_with_the_given_circuit():
    """The answer's English (after its EXPLANATION heading, else the whole text) is read with the facts' circuit, not
    any circuit the answer writes; no English scores 0; E1 ranks the English that names the heavier event first."""
    got = env_d.score(FakeReader(), {}, EV, FACTS, ["EXPLANATION: position 2 matters", "position 9 matters",
                                                    "CIRCUIT:\n@5 b=tok\nEXPLANATION: position 2 matters", ""])
    assert got[0]["reward"] == pytest.approx(got[2]["reward"]) and got[0]["reward"] > got[1]["reward"]
    assert got[3] == {"reward": 0.0, "e1": 0.0, "valid": True, "words": 0, "named": False}
    assert env_d.score(FakeReader(), {}, EV, FACTS, ["L1.v#3 at position 2"])[0]["named"]
    assert env_d.english_of("EXPLANATION:  two\n lines ") == "two lines"


def test_gate_summary():
    """The gate holds when the correct English beats each one-claim-changed version on >= GATE of the texts that have
    it; shuffled and no English reported; a version no text has does not block it."""
    rows = [{"task": f"t{i}", "e1": {"correct": 0.5, "shuffled": -1.0, "position": 0.1 if i else 0.9}} for i in range(5)]
    s = env_d.summarize(rows)
    assert s["beats"]["position"] == {"wins": 4, "of": 5, "share": 0.8} and s["beats"]["mechanism"]["of"] == 0
    assert s["gate"] and s["beats"]["none"]["share"] == 1.0 and s["mean_e1"]["none"] == 0.0
    rows[1]["e1"]["position"] = 0.6
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
    want = env_d.score(FakeReader(), {}, EV, FACTS, ["position 2 matters", ""])
    assert [g["reward"] for g in got] == pytest.approx([w["reward"] for w in want])
