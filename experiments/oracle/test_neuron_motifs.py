"""Checks of neuron_motifs.py's mining and scoring on synthetic circuits (no model).

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_neuron_motifs.py -q
"""
import random
import sys
from pathlib import Path

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
