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
    assert r["nodes"] == len(vpd) and r.get("dropped_nodes", 0) == 0
    assert math.isclose(r["point"], pt["f1_live"], abs_tol=2e-3)
    if pp["s_track_pp"] is not None:
        assert math.isclose(r["pp"], pp["s_track_pp"], abs_tol=2e-3)
    ch = call(tools, "run_circuit", circuit=text, changes=[[5, 3]])
    assert "model_top" in ch and "circuit_top" in ch and ch["kl_bits"] >= -1e-6
    cmp = call(tools, "compare", circuit_a=text, circuit_b=tools_env.Scripted._circuit(vpd[:3]), top=3)
    if "disagree" in cmp:
        d = [r[3] for r in cmp["disagree"]]
        assert d == sorted(d, reverse=True)


def test_families_on_shifted_draws(env):
    tools = env.text_tools(TID)
    tx, nat = tools.tx, env.ge.nat
    vpd = sorted((n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist())
    r = call(tools, "run_circuit", nodes=[components.node_text(nd) for nd in vpd], families=True)
    px = tools._proxy()
    assert px.seed == tx.seed + tools_env.SEED_SHIFT and px.ids == tx.ids and px.f2 == tx.f2
    g = goal_eval.strand(native.Graph(vpd, complete=True), tools.targets, nat.C)
    row = env.ge.fidelity(px, [g], tools.pool, mode="reward", split=tools.split)[0]
    tau = env.ge.tau(px, tools.pool, tools.split)
    ph, _ = goal_eval.phi(row, families=tau.get("families"))
    if ph is not None:
        assert math.isclose(r["phi"], ph, rel_tol=1e-2, abs_tol=1e-3)
    assert r["bar"] == tools_env._r(tau.get("phi_tau"))
    assert tools.split not in (tx.__dict__.get("_tau") or {})  # the tools' tau lives on the copy alone
    plain = call(tools, "run_circuit", nodes=[components.node_text(nd) for nd in vpd])
    assert "phi" not in plain and math.isclose(plain["point"], r["point"], abs_tol=1e-9)


def test_errors_go_back_to_the_policy(env):
    tools = env.text_tools(TID)
    assert "error" in call(tools, "list_active", position=99)
    assert "error" in call(tools, "list_active", position=3, site="L9.zz")
    assert "error" in call(tools, "measure_removal", nodes=["L0.q#999@3"])
    assert "error" in call(tools, "measure_removal", nodes=["L7.q#1@3"])
    assert "error" in call(tools, "node_effects", draft="d1")  # no draft outside an episode
    assert "error" in call(tools, "measure_removal", nodes=["L0.q#1@3"], each=True, offset=-2)
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
    assert "error" in json.loads(ep.call("run_circuit", {"base": "d1", "drop_least": 2}))  # no order measured yet
    rem = json.loads(ep.call("measure_removal", {"draft": "d1", "each": True, "top": 400}))
    assert sorted(x[0] for x in rem["removals"]) == sorted(components.node_text(nd) for nd in at11)
    assert rem["each"] == len(at11) and rem["next_offset"] is None and "_order" not in rem
    page = json.loads(ep.call("measure_removal", {"draft": "d1", "each": True, "top": 2, "offset": 1}))
    assert [x[0] for x in page["removals"]] == [x[0] for x in rem["removals"]][1:3]
    least = [x[0] for x in rem["removals"]][::-1]
    c = json.loads(ep.call("run_circuit", {"base": "d1", "drop_least": 3}))
    want = json.loads(tools.call("run_circuit", {"nodes": [x for x in least[3:]]}))
    assert c["nodes"] == len(at11) - 3 and math.isclose(c["point"], want["point"], abs_tol=1e-9)
    c2 = json.loads(ep.call("run_circuit", {"base": c["draft"], "drop_least": 2}))  # the order carries over
    want2 = json.loads(tools.call("run_circuit", {"nodes": [x for x in least[5:]]}))
    assert c2["nodes"] == len(at11) - 5 and math.isclose(c2["point"], want2["point"], abs_tol=1e-9)
    eff = json.loads(ep.call("node_effects", {"draft": "d1", "top": 400}))
    g1 = goal_eval.strand(native.Graph(at11, complete=True), tools.targets, nat.C)
    pt = {tuple(r["node"]): r["delta_kl"] for r in env.ge.line_effects(tools.tx, g1, mode="c")["nodes"]}
    pp = ({tuple(r["node"]): r["delta_kl"] for r in env.ge.line_effects(tools.tx, g1, mode="pp", split=tools.split,
                                                                       draws=tools_env.EFFECT_DRAWS)["nodes"]} if tools.tx.f2 else {})
    assert eff["nodes"] == len(at11) and sorted(x[0] for x in eff["effects"]) == sorted(components.node_text(nd) for nd in at11)
    for x in eff["effects"]:
        nd = components.node_of(x[0])
        assert math.isclose(x[2], pt[nd], rel_tol=1e-2, abs_tol=1e-3)
        if pp:
            assert math.isclose(x[1], pp[nd], rel_tol=1e-2, abs_tol=1e-3)
    worst = [max(v for v in (x[1], x[2]) if v is not None) for x in eff["effects"]]
    assert all(a >= b - 2e-3 for a, b in zip(worst, worst[1:]))
    e2 = json.loads(ep.call("run_circuit", {"base": "d1", "drop_least": 2}))  # node_effects' order now
    want3 = json.loads(tools.call("run_circuit", {"nodes": [x[0] for x in eff["effects"]][:-2]}))
    assert e2["nodes"] == len(at11) - 2 and math.isclose(e2["point"], want3["point"], abs_tol=1e-9)
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


def test_baselines_include_empty_and_same_batch_calibration(_unused_env=None):
    """One fake fidelity batch supplies empty control, reference, KL and exact score; no model/tau call."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    site = native.site_name(0, "down_proj")
    nodes = [(site, 1, 0), (site, 1, 1)]
    start = native.Graph(nodes, complete=True)
    tx = SimpleNamespace(ids=[2, 3], targets=[1], T=2, vpd={site: torch.ones(2, 2, dtype=torch.bool)},
                         ci={site: torch.ones(2, 2)}, rec={site: torch.ones(1, 2, 2)})
    nat = SimpleNamespace(C={site: 2}, positions=lambda targets: 2, _vpd_order=lambda ids, targets: nodes)
    calls = []

    def fidelity(got_tx, graphs, pool, *, mode, split):
        calls.append(graphs)
        assert got_tx is tx and mode == "reward" and split == "eval" and pool == {}
        assert len(graphs) == 6 and not graphs[1].complete and graphs[1].count() == 0
        assert not getattr(graphs[1], "tok_out", []) and not getattr(graphs[1], "tok_edges", [])
        parsed_empty = components.parse(components.render({"nodes": [], "edges": []}))
        actual_empty, _ = goal_eval.circuit(parsed_empty["answer"], tx.targets, nat.C, tx.T)
        assert actual_empty.complete == graphs[1].complete and actual_empty.node_set() == graphs[1].node_set()
        return [{"s_point_c": .8 if g.count() == 2 else .6 if g.count() else 0.,
                 "kl_point": 2. if g.count() else 9., "kl_empty_live": 9., "kl_vpd_live": .1,
                 "eps_point": .005, "eps_N": .025, "draws_n_spec": []} for g in graphs]

    def no_tau(*args, **kwargs):
        raise AssertionError("baseline threshold must use its existing start row")

    ge = SimpleNamespace(nat=nat, start_set=lambda tx, split: {"graph": start}, fidelity=fidelity, tau=no_tau)
    fake = SimpleNamespace(ge=ge, score_split="eval", split_of=lambda tid: "eval", pool={},
                           text_tools=lambda tid: SimpleNamespace(tx=tx))
    expanded = components.render({"nodes": [[site, 0, 1]], "edges": [], "complete": True})
    episode = {"task": "text9000", "answer": "CIRCUIT: d1", "answer_expanded": expanded,
               "start": {"phi_tau": -99., "bits": 1.}}
    with TemporaryDirectory() as tmp, patch.object(goal_eval, "size_bits", lambda g, *a, **k: g.count() * 20.), \
            patch.object(components, "NODE_CAP", 1):
        episodes, output = Path(tmp) / "episodes.jsonl", Path(tmp) / "baseline.jsonl"
        episodes.write_text(json.dumps(episode) + "\n")
        rows = tools_env.baselines(fake, episodes, output, log=lambda *a, **k: None)
        assert len(calls) == 1
        row = rows[0]
        assert json.loads(output.read_text()) == row
        assert row["task"] == "text9000" and row["score_split"] == "eval" and row["k"] == 1
        assert row["threshold_source"] == "same_batch_start" and row["bits_start"] == 40.
        assert math.isclose(row["phi_tau"], -math.log2(.205), abs_tol=1e-12)
        assert len(row["source_sha256"]["tools_env.py"]) == len(row["episode_sha256"]) == 64
        empty = row["empty"]
        assert empty["bits"] == 0. and empty["compact"] is None and empty["kl_point"] == 9.
        assert empty["complete"] is False and empty["tok_out"] == []
        assert empty["circuit_reward_qa_free_d0"] == empty["phi"]
        assert "nodes" in empty["missing_families"] and "N" in empty["missing_families"]
        assert empty["eps"]["N"] == .025 and empty["draw_counts"]["N"] == 0
        scripted = row["scripted"]
        assert scripted["kl_point"] == 2. and scripted["compact"] == 1.
        gate = 1. / (1. + 2. ** (-components.GATE_SLOPE * (scripted["phi"] - row["phi_tau"])))
        assert scripted["g"] == gate
        assert scripted["circuit_reward_qa_free_d0"] == scripted["phi"] + gate * .5
        assert row["start"]["node_cap_eligible"] is False
        assert row["start"]["circuit_reward_qa_free_d0"] == components.V3_INVALID
        assert row["start"]["formula_circuit_reward_qa_free_d0"] > 0


def test_scripted_episode_pp(env):
    pol = tools_env.Scripted(rank="removal", accept="pp")
    r = tools_env.run_episode(env, TID, pol)
    assert r["answered"] and r["scores"]["valid"], r
    assert not any(c.get("arguments", {}).get("families") for m in r["messages"] for c in m.get("tool_calls") or [])


def test_free_text_drops_a_texts_tools(env):
    env.text_tools(TID)
    tools_env.free_text(env, TID)
    assert TID not in env.tools
    assert env.text_tools(TID).tx is not None  # built again on the next use


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
    assert r["split"] == "eval" and r["semantics"] == env.ge.semantics and r["cost"] == env.cost  # what a trainer freezes
    assert r["source"] == tools_env.source() and r["donor_pool"] == env.ge.pool_id("eval")


def test_auto_split_by_text(env):
    auto = tools_env.Env({}, env.ge, score_split="auto", facts=False, pool={})
    assert [auto.split_of(t) for t in ("text6000", "text7999", "text8000", "text9020", TID)] == ["search", "search", "eval", "eval", "eval"]
    assert env.split_of("text6000") == "eval"  # a fixed split applies to every text
    try:
        tools_env.Env({}, env.ge, score_split="train", facts=False, pool={})
        raise AssertionError("an unknown score split was taken")
    except ValueError:
        pass


if __name__ == "__main__":
    import time

    e = env.__wrapped__() if hasattr(env, "__wrapped__") else env()
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        t0 = time.perf_counter()
        fn(e)
        print(f"{name} passed ({time.perf_counter() - t0:.1f} s)", flush=True)
    print("all passed")


def test_a_turns_run_circuit_calls_are_scored_together(env):
    """Episode.step scores a turn's run_circuit calls in one batch per score (Tools.prefetch): each result is the
    call's own one at a time (to the results' rounding), and the log gives each call its share, none marked cached."""
    t = 11
    calls = [{"name": "run_circuit", "arguments": {"positions": [t] + ps, "families": fam}}
             for ps in ([], [t - 1], [t - 1, t - 2]) for fam in (True, False)]

    def fresh():
        e = tools_env.Env(env.rows, env.ge, score_split="eval", facts=False, pool={})
        e.tasks = env.tasks
        return e
    alone_tools = fresh().text_tools(TID)
    alone = [json.loads(alone_tools.call(c["name"], c["arguments"])) for c in calls]
    ep = fresh().reset(TID)
    ep.step({"role": "assistant", "content": "", "tool_calls": calls})
    got = [json.loads(m["content"]) for m in ep.messages if m["role"] == "tool"]
    assert len(got) == len(calls) and not ep.tools.prefetched and len(ep.tools.seconds["run_circuit"]) == len(calls)
    assert all(not cached and sec > 0 for _, _, sec, cached in ep.log)

    def close(a, b):
        if isinstance(a, dict):
            return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) <= 2e-3
        return a == b
    for a, b in zip(alone, got):
        b.pop("draft", None)
        assert close(a, b), (a, b)


def test_server_scores_concurrent_episodes_run_circuit_calls_together(env, monkeypatch):
    """serve()'s Server: run_circuit calls of concurrent episodes on one text are scored in one batch (one _scored call
    for all of them), each episode's result the call's own one at a time (to the results' rounding)."""
    import threading

    t = 11
    calls = [{"positions": [t] + ps, "families": True} for ps in ([], [t - 1], [t - 2], [t - 1, t - 2])]

    def fresh():
        e = tools_env.Env(env.rows, env.ge, score_split="eval", facts=False, pool={})
        e.tasks = env.tasks
        return e
    alone_tools = fresh().text_tools(TID)
    alone = [json.loads(alone_tools.call("run_circuit", a)) for a in calls]
    srv = tools_env.Server(fresh(), wait=0.3)
    eps = [srv.handle("/reset", {"task": TID})[1]["episode"] for _ in calls]
    sizes, real = [], tools_env.Tools._scored

    def counted(self, heads, families):
        sizes.append(len(heads))
        return real(self, heads, families)
    monkeypatch.setattr(tools_env.Tools, "_scored", counted)
    got = [None] * len(calls)

    def run(i):
        got[i] = srv.handle("/call", {"episode": eps[i], "name": "run_circuit", "arguments": calls[i]})
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sizes == [len(calls)] and all(code == 200 for code, _ in got)
    for a, (_, body) in zip(alone, got):
        b = json.loads(body["content"])
        b.pop("draft", None)
        assert set(a) == set(b) and all(abs(a[k] - b[k]) <= 2e-3 if isinstance(a[k], float) else a[k] == b[k] for k in a
                                       if k != "families"), (a, b)
    assert all(not ep_log[3] for e in eps for ep_log in srv.episodes[e].log)
