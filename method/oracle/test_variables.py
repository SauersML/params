"""Checks of variables.py on vpd4l's text6011 (its target 288; 233 holds the copied word), on this machine's GPU
(skipped without one or vpd4l's texts): the scorer's patched runs are native.run's interchange patches (an independent
forward pass); the lead's cases: every subcomponent at 233 for "copied word" (inputs 232 and 233) reproduces the
on-target changes and leaves the off-target ones alone, the last layer at the target fails off-target; two groups
patched together reproduce a change of both inputs; a group's patch moves the reads of a group it feeds as the change
does; in a circuit's stated graph (C's semantics) the patched runs are goal_eval.graph_ref's interchange and its
activations graph_ref's reads; names parse as goal_eval's circuits.

  ~/.local/bin/mem-lease 16 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_variables.py -s
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import variables  # noqa: E402

TEXT = native.TEXTS / "vpd4l" / "text6011.json"
DEVICE = native.device()
pytestmark = pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")


@pytest.fixture(scope="module")
def case():
    nat = native.Native(DEVICE)
    ge = goal_eval.GoalEval(nat)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    last = nat.target.n_layer - 1
    at = lambda t: [[n, c, t] for n in nat.names for c in range(nat.C[n])]  # noqa: E731
    top = [[n, c, 288] for n in nat.names if native._name_layer_kind(n)[0] == last for c in range(nat.C[n])]
    return nat, ge, tx, at, top


def test_patched_runs_are_native_patches(case):
    """A group's activations set to their values on a changed text (and every subcomponent at the target): the
    engine's run is native.run with the same patch, within float32 rounding (KL below 1e-5 bits)."""
    nat, ge, tx, at, top = case
    sc = variables.Scorer(ge)
    x = list(tx.ids)
    x[233] = 512
    src = sc._text(x, tx.targets)
    mine = sc._text(tx.ids, tx.targets)
    group = variables.parse([{"inputs": [233], "group": at(233)[::7] + top[::5]}], nat.C)[0]["group"]
    got = sc._patched(mine, [(src, group), (src, None)])
    T = tx.T
    patch = {n: (torch.zeros(2, T, nat.C[n], dtype=torch.bool, device=nat.dev), torch.zeros(2, T, nat.C[n], device=nat.dev))
             for n in nat.names}
    for n, t, c in group:
        patch[n][0][0, t, c] = True
    for n in nat.names:
        patch[n][0][1, tx.targets] = True
        patch[n][1][:] = src["act"][n][None]
    with torch.no_grad():
        want = nat.run(tx.ids, tx.targets, nat._full_plan(T, 2), *nat._one(), patch=patch)
    assert float(carry.kl(want, got).abs().max()) < 1e-5


def test_the_leads_cases(case):
    """"copied word" (inputs 232, 233): every subcomponent at 233 reproduces the on-target changes (linear fraction
    above 0.99) and leaves the off-target ones (287, 288) alone (above 0.99); the last layer at 288 carries the
    off-target changes (log score below 0.5, linear below 0.9)."""
    nat, ge, tx, at, top = case
    vs = variables.parse([{"name": "all at 233", "inputs": [232, 233], "group": at(233)},
                          {"name": "last layer at 288", "inputs": [232, 233], "group": top}], nat.C)
    got = variables.Scorer(ge).score(tx, vs, "eval", 6)
    a, b = got["variables"]
    assert a["draws_on"] and a["draws_off"] and b["draws_off"]
    assert a["f_on"] > 0.99 and a["s_off"] > 0.99 and a["f_off"] > 0.99, a
    assert b["s_off"] < 0.5 and b["f_off"] < 0.9, b


def test_composition(case):
    """Two variables at 233 and 287, each every subcomponent at its position: patched together they reproduce a change
    of both (linear fraction above 0.99)."""
    nat, ge, tx, at, _ = case
    vs = variables.parse([{"name": "233", "inputs": [233], "group": at(233)}, {"name": "287", "inputs": [287], "group": at(287)}], nat.C)
    got = variables.Scorer(ge).score(tx, vs, "eval", 3)
    assert got["pairs"] and got["pairs"][0]["draws_comp"], got["pairs"]
    assert got["pairs"][0]["f_comp"] > 0.99, got["pairs"]


def test_circuit_world_is_graph_ref(case):
    """In a circuit's stated graph (C's semantics; the subcomponents most active at 233 and 288, half of the model's
    connections among them stated): the circuit's activations on a changed text are graph_ref's reads there, and the
    circuit on x with its nodes at 233 set to those values is graph_ref's interchange (KL below 1e-5 bits)."""
    nat, ge, tx, _, _ = case
    sc = variables.Scorer(ge)
    model = sc._text(tx.ids, tx.targets)
    nodes = []
    for n in nat.names:
        for t in (233, 288):
            nodes += [(n, t, int(c)) for c in model["act"][n][t].abs().topk(2).indices.tolist()]
    par, out = native.all_edges(nodes, tx.targets)
    graph = native.Graph(nodes, {r: ws[::2] for r, ws in par.items()}, out[::2])
    x = list(tx.ids)
    x[233] = 512
    src = sc._text(x, tx.targets, graph)
    rec = {}
    with torch.no_grad():
        ge.graph_ref(x, tx.targets, graph, record=rec)
    for nd in nodes:
        k = src["col"][nd[0]][nd[2]]
        assert abs(float(src["act"][nd[0]][nd[1], k]) - float(rec[nd][0])) < 1e-4 * (1 + abs(float(rec[nd][0]))), nd
    mine = sc._text(tx.ids, tx.targets, graph)
    group = [nd for nd in nodes if nd[1] == 233]
    got = sc._patched(mine, [(src, group)])
    with torch.no_grad():
        want = ge.graph_ref(tx.ids, tx.targets, graph, patch={nd: float(rec[nd][0]) for nd in group})
    assert float(carry.kl(want, got).abs().max()) < 1e-5
    res = sc.score(tx, variables.parse([{"name": "233", "inputs": [233], "group": [[n, c, t] for n, t, c in group]}], nat.C),
                   "eval", 3, graph=graph)
    assert res["circuit"]["variables"][0]["draws_on"] and res["variables"][0]["draws_on"]


def test_feeds(case):
    """Every subcomponent at 233 feeds the last two layers at 288 (a change at 233 reaches 288 only through 233's
    outputs): claimed, patching 233 reproduces the reads' change at 288 (f above 0.99); claimed apart, the same runs
    give f below 0.01 (the reads move as in model(x'))."""
    nat, ge, tx, at, _ = case
    late = [[n, c, 288] for n in nat.names if native._name_layer_kind(n)[0] >= nat.target.n_layer - 2 for c in range(nat.C[n])]
    for claim in (["b"], []):
        vs = variables.parse([{"name": "a", "inputs": [233], "group": at(233), "feeds": claim},
                              {"name": "b", "inputs": [288], "group": late}], nat.C)
        got = variables.Scorer(ge).score(tx, vs, "eval", 3)
        ab = next(x for x in got["feeds"] if x["from"] == "a" and x["to"] == "b")
        assert ab["draws"] and ab["feeds"] == bool(claim), ab
        assert (ab["f"] > 0.99) if claim else (ab["f"] < 0.01), ab


def test_kept_positions_give_the_same_scores(case, monkeypatch):
    """_world keeps each text's tables at the targets and its groups' positions alone, and patches by index: a score
    with feeds, composition, blocks and a group inside a circuit's stated graph equals the one with every position
    kept, to the bit; a position not kept raises."""
    import json
    nat, ge, tx, at, _ = case
    late = [[n, c, 288] for n in nat.names if native._name_layer_kind(n)[0] >= nat.target.n_layer - 1 for c in range(nat.C[n])][::3]
    vs = variables.parse([{"name": "a", "inputs": [233], "group": at(233)[::5] + [["block", nat.names[2], 232]], "feeds": ["b"]},
                          {"name": "b", "inputs": [287], "group": late}], nat.C)
    sc = variables.Scorer(ge)
    model = sc._text(tx.ids, tx.targets, check=False, positions=[233, 288])
    with pytest.raises(KeyError):
        model["act"][nat.names[0]][5]
    nodes = [(n, t, int(c)) for n in nat.names for t in (233, 288) for c in model["act"][n][t].abs().topk(2).indices.tolist()]
    par, out = native.all_edges(nodes, tx.targets)
    graph = native.Graph(nodes, {r: ws[::2] for r, ws in par.items()}, out[::2])
    kept = sc.score(tx, vs, "eval", 2, graph=graph)
    whole = variables.Scorer._text
    monkeypatch.setattr(variables.Scorer, "_text", lambda self, ids, targets, graph=None, check=True, positions=None:
                        whole(self, ids, targets, graph, check))
    every = sc.score(tx, vs, "eval", 2, graph=graph)
    assert every["variables"][0]["draws_on"] and every["feeds"]
    assert json.dumps(kept, sort_keys=True) == json.dumps(every, sort_keys=True)


def test_checker_keeps_positions_with_the_same_outcomes(case, monkeypatch):
    """semantic.Checker keeps its texts' tables at the targets and its variables' positions alone: token changes,
    transplants (two donors), removals (a carrier at its mean) and their outcomes equal those with every position
    kept, to the bit."""
    import semantic
    nat, ge, tx, at, _ = case
    raw = [{"name": "V1", "inputs": [233], "group": at(233)[::9] + [["block", nat.names[5], 233]]},
           {"name": "V2", "inputs": [287], "group": at(287)[::11]}]
    vs = variables.parse(raw, nat.C)
    items = [({"tokens": {233: 512}}, ("unchanged",)), ({"transplants": [("V1", {233: 512})]}, ("top", 13)),
             ({"transplants": [("V1", {233: 512}), ("V2", {287: 300})]}, ("not", 13)), ({"remove": ["V1"]}, ("drop", 0.5)),
             ({"tokens": {287: 300}, "remove": ["V2"]}, ("rise", 2.0))]

    def rows():
        ck = semantic.Checker(variables.Scorer(ge), tx, vs)
        got = ck.check(items)
        ck.close()
        return got
    kept = rows()
    whole = variables.Scorer._text
    monkeypatch.setattr(variables.Scorer, "_text", lambda self, ids, targets, graph=None, check=True, positions=None:
                        whole(self, ids, targets, graph, check))
    every = rows()
    assert kept == every, (kept, every)


def test_headline_rules():
    """e4: per variable min(f on-target, f off-target); 0 for a variable with no resolved on-target draw (counted), its
    off-target not counted; composition and feeds as they are; a missing off-target leaves f on-target."""
    out = variables.headline({"variables": [{"f_on": 0.9, "f_off": 0.5, "s_off": 0.5, "s_on": 0.2},
                                            {"f_on": None, "f_off": 1.0, "s_off": 1.0, "s_on": None},
                                            {"f_on": 0.8, "f_off": None, "s_off": None, "s_on": 0.3}],
                              "pairs": [{"f_comp": 0.6, "s_comp": 0.1}], "feeds": [{"f": 1.0, "feeds": False}]})
    assert out["unresolved"] == 1
    assert abs(out["e4"] - (0.5 + 0.0 + 0.8 + 0.6 + 1.0) / 5) < 1e-12


def test_relational_flip(case):
    """Flipping one stated relation (a feeds b, then a apart from b) on the same carriers moves the relational score and
    leaves the carriers' terms as they are: the scorer's draws are the same, only the feeds test's claim changes."""
    nat, ge, tx, at, _ = case
    blocks = [["block", n, 233] for n in nat.names]
    late = [["block", n, 288] for n in nat.names if native._name_layer_kind(n)[0] == nat.target.n_layer - 1]
    got = []
    for feeds in (["b"], []):
        vs = variables.parse([{"name": "a", "inputs": [233], "group": blocks, "feeds": feeds},
                              {"name": "b", "inputs": [288], "group": late}], nat.C)
        got.append(variables.Scorer(ge).score(tx, vs, "eval", 3))
    (x, y) = got
    assert x["carriers"] == y["carriers"], (x["carriers"], y["carriers"])
    assert abs(x["relational"] - y["relational"]) > 0.25, (x["relational"], y["relational"])


def test_blocks(case):
    """Blocks (["block", site, position]: the site's whole output there, every subcomponent and the remainder): every
    site's block at 233 reproduces a change at 233 to float32 rounding (the stream at 233 reaches later positions only
    through those outputs), where every subcomponent at 233 leaves the remainders' part (D ~1e-5 bits); and a block's
    whole output moves like model(x') in the feeds test."""
    nat, ge, tx, at, _ = case
    blocks = [["block", n, 233] for n in nat.names]
    vs = variables.parse([{"name": "blocks", "inputs": [233], "group": blocks}, {"name": "parts", "inputs": [233], "group": at(233)}], nat.C)
    assert vs[0]["group"] == sorted((n, 233, variables.BLOCK) for n in nat.names)
    got = variables.Scorer(ge).score(tx, vs, "eval", 3)
    a, b = got["variables"]
    assert a["draws_on"] and a["f_on"] > 0.999999, a
    on_a = [r for r in got["draws"] if r["kind"] == "on" and r["owner"] == 0]
    on_b = [r for r in got["draws"] if r["kind"] == "on" and r["owner"] == 1]
    assert all(r["d"] < 1e-6 for r in on_a) and max(r["d"] for r in on_b) > 1e-6, (on_a, on_b)
    late = [["block", n, 288] for n in nat.names if native._name_layer_kind(n)[0] == nat.target.n_layer - 1]
    vs = variables.parse([{"name": "a", "inputs": [233], "group": blocks, "feeds": ["b"]},
                          {"name": "b", "inputs": [288], "group": late}], nat.C)
    ab = next(x for x in variables.Scorer(ge).score(tx, vs, "eval", 3)["feeds"] if x["from"] == "a")
    assert ab["draws"] and ab["f"] > 0.999, ab


def test_parse(case):
    nat = case[0]
    v = variables.parse([{"name": "v", "inputs": [3, 1, 3], "group": [["0.fc", 5, 2], ["h.0.mlp.c_fc", 5, 2]], "feeds": ["prediction"]}], nat.C)[0]
    assert v["inputs"] == [1, 3] and v["group"] == [("h.0.mlp.c_fc", 2, 5)] and v["feeds"] == ["prediction"]
    with pytest.raises(ValueError):
        variables.parse([{"inputs": [1], "group": [["0.fc", nat.C["h.0.mlp.c_fc"], 2]]}], nat.C)
    assert variables.parse([{"inputs": [1], "group": [["block", "0.fc", 5]]}], nat.C)[0]["group"] == [("h.0.mlp.c_fc", 5, variables.BLOCK)]
