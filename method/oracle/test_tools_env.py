"""Checks of the tool environment (tools_env.py) on test_goal_eval's tiny random model: each tool against the
goal_eval or native call it reports, errors returned to the policy, and the scripted baseline's episode end to end.

  MPD_MEM_GIB=4 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_tools_env.py -q
  python method/oracle/test_tools_env.py      (without pytest: every test in order, the first failure stops)
"""

import json
import math
import sys
from pathlib import Path

import torch

try:
    import pytest

    fixture = pytest.fixture(scope="module")
except ImportError:  # a pod's environment without pytest: run as a script (main below), test_goal_eval's decorators inert
    import types

    class _Inert:
        def __getattr__(self, name):
            return lambda *a, **k: (a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f))

    pytest = types.ModuleType("pytest")
    pytest.fixture, pytest.mark = _Inert().fixture, _Inert()
    sys.modules["pytest"] = pytest
    fixture = lambda f: f  # noqa: E731

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import tools_env  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402

TID = "tinytext0"


@fixture
def env():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    for split, n in (("search", 6), ("eval", 5)):
        ge.donor_pools[split] = [(f"{split}{i}", torch.randint(0, 40, (8 + 2 * i,), generator=g).tolist()) for i in range(n)]
    lp = nat.reference([ids], [11])[0, -1]
    v, i = lp.exp().topk(3)
    top = [[components.token_strings("vpd4l", [int(a)])[0], float(b)] for a, b in zip(i, v)]
    task = {"id": TID, "model": "vpd4l", "prompts": [{"token_ids": ids, "target_positions": [11], "model_top": [top]}]}
    e = tools_env.Env({TID: {"task": TID, "qa": [], "edits": []}}, ge, score_split="eval", facts=False, pool={})
    e.tasks[TID] = task
    return e


def call(tools, name, **args):
    return json.loads(tools.call(name, args))


def test_question_is_short_and_complete(env):
    q = tools_env.question(env.task(TID), env.rows[TID])
    assert q.startswith(components.ASK.format(model="vpd4l")[:60])
    assert "target: position 11" in q and "Question " not in q and "Edit " not in q
    assert [s["name"] for s in tools_env.TOOL_SPECS] == tools_env.TOOL_NAMES
    assert all(t["type"] == "function" for t in tools_env.qwen3_tools())


def test_influence_and_list_active(env):
    tools = env.text_tools(TID)
    inf = call(tools, "influence", top=50)
    bits = [r[2] for r in inf["positions"]]
    assert bits == sorted(bits, reverse=True)
    moved = torch.as_tensor(tools.tx.moved, dtype=torch.float64) / math.log(2)
    assert math.isclose(inf["total_bits"], float(moved.sum()), rel_tol=1e-2)
    assert len(inf["positions"]) == len(moved)
    for p in (11, inf["positions"][0][0]):
        la = call(tools, "list_active", position=p, top=400)
        assert la["active"] == sum(int(tools.tx.vpd[n][p].sum()) for n in env.ge.nat.names) == la["shown"]
        keys = [r[1] * abs(r[2]) for r in la["subcomponents"]]
        assert all(a >= b - 1e-2 for a, b in zip(keys, keys[1:]))
        one = call(tools, "list_active", position=p, site="L1.fc", top=400)
        assert all(r[0].startswith("L1.fc#") for r in one["subcomponents"])


def test_attention_rows(env):
    tools = env.text_tools(TID)
    a = call(tools, "attention", layer=1, position=7, top=8)
    for h, rows in a["heads"].items():
        assert math.isclose(sum(w for _, _, w in rows), 1.0, abs_tol=0.02)
        assert all(0 <= q <= 7 for q, _, _ in rows)


def test_measure_removal_each_matches_one_at_a_time(env):
    tools = env.text_tools(TID)
    nodes = [r[0] for r in call(tools, "list_active", position=11, top=6)["subcomponents"]]
    each = {n: k for n, k in call(tools, "measure_removal", nodes=nodes, each=True)["removals"]}
    for n in nodes[:3]:
        one = call(tools, "measure_removal", nodes=[n])
        assert math.isclose(one["kl_bits"], each[n], rel_tol=2e-2, abs_tol=1e-3)
    both = call(tools, "measure_removal", nodes=nodes[:2])
    lq = env.ge.nat.without(tools.ids, tools.targets, [components.node_of(x) for x in nodes[:2]])
    assert math.isclose(both["kl_bits"], float(carry.kl(tools.tx.ref, lq)[0]), rel_tol=1e-2, abs_tol=1e-4)


def test_run_circuit_reports_goal_eval(env):
    tools = env.text_tools(TID)
    tx, nat = tools.tx, env.ge.nat
    vpd = sorted((n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist())
    text = tools_env.Scripted._circuit(vpd)
    r = call(tools, "run_circuit", circuit=text)
    g = native.Graph(vpd, complete=True)
    pt = env.ge.sufficiency(tx, [g])[0]
    pp = env.ge.track_pp(tx, [g], tools.split)[0]
    assert r["nodes"] == len(vpd) and r["dropped_nodes"] == 0
    assert math.isclose(r["point"], pt["f1_live"], abs_tol=2e-3)
    if pp["s_track_pp"] is not None:
        assert math.isclose(r["pp"], pp["s_track_pp"], abs_tol=2e-3)
    ch = call(tools, "run_circuit", circuit=text, changes=[[5, 3]])
    assert "model_top" in ch and "circuit_top" in ch and ch["kl_bits"] >= -1e-6
    cmp = call(tools, "compare", circuit_a=text, circuit_b=tools_env.Scripted._circuit(vpd[:3]), top=3)
    if "disagree" in cmp:
        d = [r[3] for r in cmp["disagree"]]
        assert d == sorted(d, reverse=True)


def test_errors_go_back_to_the_policy(env):
    tools = env.text_tools(TID)
    assert "error" in call(tools, "list_active", position=99)
    assert "error" in call(tools, "list_active", position=3, site="L9.zz")
    assert "error" in call(tools, "measure_removal", nodes=["L0.q#999@3"])
    assert "error" in call(tools, "run_circuit", circuit="flowchart LR")
    assert "error" in call(tools, "attention", layer=7, position=3)
    assert "error" in json.loads(tools.call("nope", {}))
    assert "error" in json.loads(tools.call("influence", "{not json"))


def test_drafts_through_episode(env):
    ep = env.reset(TID)
    tools, nat = ep.tools, env.ge.nat
    vpd = sorted((n, t, c) for n in nat.names for t, c in tools.tx.vpd[n].nonzero().tolist())
    at11 = [nd for nd in vpd if nd[1] == 11]
    a = json.loads(ep.call("run_circuit", {"positions": [11]}))
    assert a["draft"] == "d1" and a["nodes"] == len(at11)
    same = json.loads(ep.call("run_circuit", {"nodes": [components.node_text(nd) for nd in at11]}))
    assert same["draft"] in ("d1", "d2") and math.isclose(same["point"], a["point"], abs_tol=1e-9)
    text = json.loads(ep.call("run_circuit", {"circuit": ", ".join(components.node_text(nd) for nd in at11)}))
    assert text["nodes"] == len(at11) and math.isclose(text["point"], a["point"], abs_tol=1e-9)
    mer = json.loads(ep.call("run_circuit", {"circuit": tools_env.Scripted._circuit(at11)}))
    assert math.isclose(mer["point"], a["point"], abs_tol=1e-9)
    drop = [components.node_text(nd) for nd in at11[:2]]
    b = json.loads(ep.call("run_circuit", {"base": "d1", "drop": drop}))
    direct = json.loads(tools.call("run_circuit", {"nodes": [components.node_text(nd) for nd in at11[2:]]}))
    assert b["nodes"] == len(at11) - 2 and math.isclose(b["point"], direct["point"], abs_tol=1e-9)
    back = json.loads(ep.call("run_circuit", {"base": b["draft"], "add": drop}))
    assert back["nodes"] == len(at11) and math.isclose(back["point"], a["point"], abs_tol=1e-9)
    assert "error" in json.loads(ep.call("run_circuit", {"base": "d99", "drop": drop}))
    rem = json.loads(ep.call("measure_removal", {"draft": "d1", "each": True}))
    assert sorted(x[0] for x in rem["removals"]) == sorted(components.node_text(nd) for nd in at11)
    cmp = json.loads(ep.call("compare", {"circuit_a": "d1", "circuit_b": b["draft"]}))
    assert "error" not in cmp or "nothing to change" in cmp["error"]
    again = json.loads(ep.call("run_circuit", {"positions": [11]}))
    assert again["draft"] == "d1" and again["repeated"] == 1
    bad = json.loads(ep.call("run_circuit", {"circuit": "flowchart LR\nsubgraph P11[@11]\n c1[L1.fc#2@11] --> c2[L1.down#3@11]\nend"}))
    assert "error" in bad or bad.get("bad_examples")
    full = ep.expand("CIRCUIT: d1\nEXPLANATION: x")
    parsed = components.parse(full)
    assert parsed["answer"].get("complete") and len(parsed["answer"]["nodes"]) == len(at11)
    assert ep.expand("CIRCUIT: d42\nEXPLANATION: x") == "CIRCUIT: d42\nEXPLANATION: x"


def test_scripted_episode(env):
    pol = tools_env.Scripted()
    r = tools_env.run_episode(env, TID, pol)
    assert r["answered"] and r["calls"] <= tools_env.MAX_CALLS and r["turns"] <= tools_env.MAX_TURNS
    assert r["scores"]["valid"], r
    assert math.isclose(r["reward"], r["answer_reward"] - env.cost * r["calls"])
    assert r["messages"][-1]["content"].startswith("CIRCUIT: d")
    parsed = components.parse(r["answer_expanded"])
    assert parsed["circuit"] and parsed["answer"].get("complete")
    tx = env.text_tools(TID).tx
    for m, c, t in parsed["answer"]["nodes"]:
        assert bool(tx.vpd[m][t, c])
    assert [m["role"] for m in r["messages"][:2]] == ["system", "user"]
    assert all(m["role"] in ("assistant", "tool") for m in r["messages"][2:])


if __name__ == "__main__":
    import time

    e = env.__wrapped__() if hasattr(env, "__wrapped__") else env()
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        t0 = time.perf_counter()
        fn(e)
        print(f"{name} passed ({time.perf_counter() - t0:.1f} s)", flush=True)
    print("all passed")
