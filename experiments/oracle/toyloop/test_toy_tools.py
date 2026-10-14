"""Smoke tests of the tool-use toy oracle (toy_tools.py, toy_tools_train.py): tools read only weights and text, the
scripted expert recovers the circuit, the chat ids match SmolLM2's template with only assistant turns scored, and a tiny
random model trains on episodes and rolls out to the end.

  python3 -B -m pytest -q experiments/oracle/toyloop/test_toy_tools.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_tools as tt  # noqa: E402

pytestmark = pytest.mark.skipif(not tt.MANIFEST.exists(), reason="zoo2 data not present")


@pytest.fixture(scope="module")
def toys():
    return tt.load("train", 6)


def test_tools_need_no_truth(toys):
    t = copy.copy(toys[0])
    t.truth, t.meta = None, {"instance": "blind-" + t.name}     # no record params, no truth
    env = tt.ToolEnv(t)
    o = env.step("CALL marker()")
    assert o.startswith("marker ") and "detector L0H" in o
    marker = o.split()[1]
    g = env.step(f"CALL gated({marker})")
    assert g.startswith(f"gated by {marker}:")
    units = [u for u in g.split(":", 1)[1].split() if tt.UNIT.fullmatch(u)]
    for tool in ("reads", "writes", "twins", "fires"):
        assert "unknown" not in env.step(f"CALL {tool}({', '.join(units)})")
    assert env.step("CALL nope()").startswith("error")
    assert env.step("ANSWER\n" + tt.EMPTY) == "" and env.done


def test_expert_recovers_truth(toys):
    for t in toys:
        e = tt.expert_episode(t)
        assert e["rule_ok"], (t.name, e["rule_units"], e["truth_units"])
        assert e["calls"] <= 6
        assert e["messages"][-1]["content"].startswith("ANSWER\nflowchart LR")


def test_run_tool_scores_and_costs(toys):
    t = toys[0]
    env = tt.ToolEnv(t)
    gold = [u for u in t.truth["units"] if tt.UNIT.fullmatch(u)]
    o = env.step(f"CALL run({', '.join(gold)})")
    assert o.startswith("suff ") and env.runs == 1
    env.step("ANSWER\n" + tt.answer_text(env, set(gold))[len("ANSWER\n"):])
    r = env.result()
    assert r["unit_f1"] == 1.0 and abs(r["episode_reward"] - (r["reward"] - tt.RUN_COST)) < 1e-9


def test_ids_and_tiny_training(toys, tmp_path):
    import toy_tools_train as T
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    eps = [tt.expert_episode(t) for t in toys]
    ids, lab = T.episode_ids(tok, eps[0]["messages"])
    assert tok.decode(ids) == tok.apply_chat_template(eps[0]["messages"], tokenize=False)
    scored = tok.decode([i for i, m in zip(ids, lab) if m])
    assert scored.startswith("CALL marker()<|im_end|>") and scored.endswith("<|im_end|>") and "<|im_start|>" not in scored
    (tmp_path / "ep").mkdir()
    for s in ("train", "test_instance", "test_family"):
        (tmp_path / "ep" / f"{s}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    T.EPISODES = tmp_path / "ep"
    T.VAL_INSTANCES = 1
    a = SimpleNamespace(out=str(tmp_path / "ck"), init=None, model="tiny", threads=1, lr=1e-3, seed=0, epochs=1, batch=2,
                        val_n=2, log_every=1, val_every=1, save_every=1, minutes=0)
    T.sft(a)
    state = json.loads((tmp_path / "ck" / "state.json").read_text())
    assert state["step"] >= 1 and (tmp_path / "ck" / "opt.pt").exists()
    T.sft(a)                         # resumes: already done, no extra steps
    assert json.loads((tmp_path / "ck" / "state.json").read_text())["step"] == state["step"]
    import train
    model, tok2 = train.load_policy(SimpleNamespace(init=str(tmp_path / "ck"), model=None, lora=0, threads=1))
    r, msgs = T.rollout(model.eval(), tok2, toys[0], max_new=8)
    assert r["calls"] <= tt.MAX_CALLS + 1 and msgs[0]["role"] == "system"


def test_verify_episodes_and_tiny_rl(toys, tmp_path):
    import toy_tools_train as T
    import train
    e = tt.expert_episode(toys[1], verify="revise")
    runs = [m["content"] for m in e["messages"] if m["content"].startswith("CALL run(")]
    assert 1 <= len(runs) <= 2 and e["rule_ok"]
    eps = [tt.expert_episode(t) for t in toys]
    (tmp_path / "ep").mkdir()
    (tmp_path / "ep" / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in eps))
    T.EPISODES, T.VAL_INSTANCES = tmp_path / "ep", 1
    model, tok = train.load_policy(SimpleNamespace(init=None, model="tiny", lora=0, threads=1))
    g = T.sample_episodes(model, tok, toys[0], 2, 1.0, max_new=6)
    assert len(g) == 2 and all(len(x["ids"]) == len(x["labels"]) and sum(x["labels"]) > 0 for x in g)
    assert T.token_logps(model, g[0]["ids"], g[0]["labels"]).shape[0] == sum(g[0]["labels"][1:])
    ck = tmp_path / "init"
    train.save(model, tok, ck, [])
    a = SimpleNamespace(out=str(tmp_path / "rl"), init=str(ck), model=None, threads=1, lr=1e-4, seed=0, batch=1, k=2,
                        temp=1.0, beta=0.05, std=False, weak=1, minutes=0.01, val_n=1, max_new=6)
    T.rl(a)
    st = json.loads((tmp_path / "rl" / "state.json").read_text())
    assert st["step"] >= 1 and (tmp_path / "rl" / "best").exists() and (tmp_path / "rl" / "last").exists()
