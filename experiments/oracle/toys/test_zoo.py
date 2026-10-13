"""zoo.py: a few instances per family pass their ablation checks and their ground-truth node sets reproduce the model;
goal_eval's Φ scores one instance's ground-truth circuit above the empty circuit.

  python3 -m pytest -q experiments/oracle/toys/test_zoo.py
"""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import zoo  # noqa: E402

torch.set_num_threads(2)


@pytest.mark.parametrize("family", list(zoo.FAMILIES))
def test_family_instances_pass_checks(family):
    passed = 0
    for seed in range(3):
        p, sd = zoo.make(family, seed)
        res = zoo.check(p, sd, n=96)
        passed += res["ok"]
        assert res["acc"] >= 0.98 or not res["ok"]
    assert passed >= 2, f"{family}: {passed}/3 passed"


@pytest.mark.parametrize("family", list(zoo.FAMILIES))
def test_record_ground_truth_nodes(family):
    p, sd = zoo.make(family, 0)
    rec = zoo.record(p, sd, n_examples=2)
    json.dumps(rec)
    assert rec["explanation"] and rec["mermaid"].startswith("flowchart LR") and rec["gt_units"]
    for e in rec["examples"]:
        assert e["gt_nodes"] and all(isinstance(n, str) and isinstance(t, int) and isinstance(c, int) for n, t, c in e["gt_nodes"])
        assert e["gt_node_kl"] < 0.05 < e["empty_kl"]
        assert len(e["gt_nodes"]) < e["n_active"]


def test_make_is_deterministic_and_splits():
    p1, sd1 = zoo.make("induction", 5)
    p2, sd2 = zoo.make("induction", 5)
    assert p1 == p2 and all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    assert zoo.split_of("multihop", 3) == "test_family"
    assert {zoo.split_of("gate", s) for s in range(40)} == {"train", "test_instance"}


def test_phi_ground_truth_above_empty(tmp_path):
    import goal_eval
    import native
    import planted
    from run_planted import ToyEval

    p, sd = zoo.make("induction", 0)
    assert zoo.check(p, sd, n=64)["ok"]
    nat = planted.native_of(sd, zoo.CFG)
    ge = ToyEval(nat, chunk=16)
    far = [(f"far{i}", x) for i, (x, _) in enumerate(zoo.texts(p, 8, 99))]
    ge.donor_pools.update(search=far[:4], eval=far[4:])
    paths = []
    for i, (x, _) in enumerate(zoo.texts(p, 6, 7)):
        f = tmp_path / f"t{i}.json"
        f.write_text(json.dumps({"prompts": [{"token_ids": x, "target_positions": [len(x) - 1]}]}))
        paths.append(f)
    pool = goal_eval.build_patterns(paths, tmp_path / "pool.pt", nat)
    ids, _ = zoo.texts(p, 1, 3)[0]
    targets = [len(ids) - 1]
    tx = ge.text(ids, targets)
    gt = zoo.gt_nodes(nat, zoo.FAMILIES[p["family"]].gt(p, ids), ids)
    graphs = [native.Graph([tuple(n) for n in gt], complete=True), native.Graph([], complete=True)]
    rows = ge.fidelity(tx, graphs, pool, mode="full", split="eval")
    phis = [goal_eval.phi(r)[0] for r in rows]
    assert phis[0] is not None
    assert phis[1] is None or phis[0] > phis[1] + 1.0, phis
