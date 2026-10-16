"""Checks of neuron_motifs.py's mining and scoring on synthetic circuits (no model).

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_neuron_motifs.py -q
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import neuron_motifs as nm  # noqa: E402


def rec(i, nodes, routes=()):
    return {"id": f"t_{i}", "met": True, "nodes": [{"layer": a, "neuron": b, "pole": p} for a, b, p in nodes],
            "routes": [{"layer": a, "head": h} for a, h in routes]}


def test_mine_finds_the_planted_group(monkeypatch):
    monkeypatch.setattr(nm, "NODES", "circuit")
    rng = random.Random(0)
    recs = []
    for i in range(40):
        nodes = [(rng.randrange(20), rng.randrange(500), rng.choice((1, -1))) for _ in range(3)]
        if i % 2 == 0:                                   # planted: three poles and a route in every other circuit
            nodes += [(3, 7, 1), (9, 11, -1), (12, 40, 1)]
        recs.append(rec(i, nodes, [(10, 2)] if i % 2 == 0 else []))
    g = nm.mine(recs)
    top = g["groups"][0]
    assert set(top["keys"]) == {"3.7.+", "9.11.-", "12.40.+", "10.h2"}
    assert set(top["poles"]) == {"3.7.+", "9.11.-", "12.40.+"} and abs(top["share"] - 0.5) < 1e-9
    assert all(len(x["keys"]) <= 4 for x in g["groups"])


def test_log_sf_and_auc():
    import math
    # P(X >= 2) for 2 draws from 4 with 2 successes = 1/6
    assert abs(math.exp(nm.log_sf(2, 4, 2, 2)) - 1 / 6) < 1e-12
    assert nm.auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert nm.auc([1, 1], [True, False]) == 0.5
    assert nm.auc([1, 2], [True, True]) is None


def test_family_split_alternates():
    assert nm.family("cap_france") == "cap" and nm.family("add_2_3") == "add" and nm.family("g201") == "g"


def test_views_group_separates_signs():
    rng = random.Random(1)
    profs = []
    for q in range(12):                                   # two planted groups: same inputs, opposite answer effects
        nodes = []
        for k, sign in (("3.7.+", -1), ("5.9.+", -1), ("8.1.-", -1), ("4.2.+", 1), ("6.6.-", 1), ("9.3.+", 1)):
            base_in = [1.0, 0.2, 0.0, 0.5] if sign < 0 else [0.0, 0.3, 1.0, 0.1]
            nodes.append({"key": k, "input": [x + 0.05 * rng.random() for x in base_in],
                          "output": [sign * (1 + 0.1 * rng.random()), -sign * 0.5, 0.1 * rng.random()]})
        profs.append({"id": f"capital/K{q}/p0", "nodes": nodes})
    got = nm.views_group(profs)
    two = got["two_views"]
    sets = [set(g["keys"]) for g in two["groups"]]
    assert {"3.7.+", "5.9.+", "8.1.-"} in sets and {"4.2.+", "6.6.-", "9.3.+"} in sets
    assert two["opposite_sign_share"] == 0 and two["silhouette"] > 0.5
    names = {frozenset(g["keys"]): g["name"] for g in two["groups"]}
    assert names[frozenset({"4.2.+", "6.6.-", "9.3.+"})].startswith("not[")


def test_harmonic_and_question():
    assert nm.harmonic(1.0, -0.5) == 0 and abs(nm.harmonic(0.5, 0.5) - 0.5) < 1e-12
    assert nm.question("capital/France/p3") == "capital/France"


def test_apply_member_drops_and_split(tmp_path):
    """apply on a tiny random Qwen3: a one-member group's own drop is the group's and it is never split; a group is
    split exactly when its members' drops differ in sign or one exceeds half the group's."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(tmp_path / "m")
    nodes = [{"layer": 0, "neuron": 3, "pos": 5, "pole": 1, "stage": "source"}, {"layer": 1, "neuron": 7, "pos": 5, "pole": -1, "stage": "source"},
             {"layer": 1, "neuron": 9, "pos": 9, "pole": 1, "stage": "readout"}]
    rec = {"id": "q", "ids": list(range(3, 13)), "answer": 5, "answer_tokens": [5], "alternatives": [6, 7, 8], "nodes": nodes}
    groups = [{"keys": ["0.3.+", "1.7.-"], "name": "a"}, {"keys": ["1.9.+"], "name": "b"}]
    got = nm.apply(groups, [rec], str(tmp_path / "m"), None, "float32", log=lambda m: None)[0]
    for g in got["groups"]:
        ds = [m["drop"] for m in g["members"]]
        if len(ds) == 1:
            assert abs(ds[0] - g["effect"]) < 1e-5 and not g["split"]
        else:
            assert g["split"] == (min(ds) < 0 < max(ds) or max(abs(x) for x in ds) > abs(g["effect"]) / 2)
