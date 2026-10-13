"""zoo2.py: instances pass their per-module checks, ground truth is per text, the residual rotation is exact, Φ scores a
text's ground truth above the empty circuit, and the weights-only / active-at-last baselines fall short of it.

  python3 -m pytest -q experiments/oracle/toys/test_zoo2.py
"""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import zoo2  # noqa: E402

torch.set_num_threads(2)


def _first_ok(kind, start=0):
    for s in range(start, start + 10):
        try:
            p, sd = zoo2.make(s, kind)
        except (RuntimeError, ValueError, IndexError):
            continue
        if zoo2.check(p, sd, n=48)["ok"]:
            return p, sd
    raise AssertionError("no instance passed in 10 seeds")


@pytest.mark.parametrize("kind", ["train", "test_family"])
def test_instances_pass_checks(kind):
    ok = 0
    for s in range(6):
        try:
            p, sd = zoo2.make(100 + s, kind)
        except (RuntimeError, ValueError, IndexError):
            continue
        ok += zoo2.check(p, sd, n=48)["ok"]
        fams = [m["family"] for m in p["modules"] if not m["dormant"]]
        held = [f for f in fams if f in zoo2.HELD_OUT_FAMILIES]
        assert (not held) if kind == "train" else held
    assert ok >= 4


def test_rotation_is_exact():
    p, _ = zoo2.make(3, "train")
    random.seed(0)
    m = zoo2.Model(random.Random(0))
    sd = m.sd
    sd["h.0.attn.q_proj.weight"].normal_()
    sd["h.1.mlp.c_fc.weight"].normal_()
    sd["h.1.mlp.down_proj.weight"].normal_(std=0.1)
    ids = [[0, 33, 5, 7, 9, 2]]
    a = zoo2.last_logits(sd, ids)
    b = zoo2.last_logits(zoo2.rotate(sd, 5), ids)
    assert (a - b).abs().max() < 1e-3


def test_record_per_text_truth():
    p, sd = _first_ok("train")
    rec = zoo2.record(p, sd, per_module=2)
    json.dumps(rec)
    assert rec["version"] == 2
    fams = {e["family"] for e in rec["examples"]}
    assert len(fams) >= 2
    unit_sets = {e["family"]: frozenset(e["units"]) for e in rec["examples"]}
    assert len(set(unit_sets.values())) >= 2            # the circuit depends on the text
    for e in rec["examples"]:
        assert p["detector"] in e["units"] and e["gt_node_kl"] < 0.05 < e["empty_kl"]
        assert e["ids"][1] in zoo2.MARKERS and e["mermaid"].startswith("flowchart LR") and e["explanation"]
        assert set(e["units"]) < set(rec["gt_units"])


def test_phi_ground_truth_above_empty(tmp_path):
    import goal_eval
    import native
    import planted
    from run_planted import ToyEval

    p, sd = _first_ok("train")
    j = next(i for i, m in enumerate(p["modules"]) if not m["dormant"])
    nat = planted.native_of(sd, zoo2.CFG2)
    ge = ToyEval(nat, chunk=16)
    far = [(f"far{i}", x) for i, (x, _) in enumerate(zoo2.texts(p, j, 8, 99))]
    ge.donor_pools.update(search=far[:4], eval=far[4:])
    paths = []
    for i, (x, _) in enumerate(zoo2.texts(p, j, 6, 7)):
        f = tmp_path / f"t{i}.json"
        f.write_text(json.dumps({"prompts": [{"token_ids": x, "target_positions": [len(x) - 1]}]}))
        paths.append(f)
    pool = goal_eval.build_patterns(paths, tmp_path / "pool.pt", nat)
    ids, _ = zoo2.texts(p, j, 1, 3)[0]
    tx = ge.text(ids, [len(ids) - 1])
    gt = zoo2.gt_nodes(nat, zoo2.gt_spec(p, j, ids), ids)
    rows = ge.fidelity(tx, [native.Graph([tuple(n) for n in gt], complete=True), native.Graph([], complete=True)], pool,
                       mode="full", split="eval")
    phis = [goal_eval.phi(r)[0] for r in rows]
    assert phis[0] is not None and (phis[1] is None or phis[0] > phis[1] + 0.3), phis


def test_baselines_fall_short(tmp_path):
    zoo2.build(tmp_path, 4, log=lambda *a: None)
    h = zoo2.hardness(tmp_path, log=lambda *a: None)
    assert h["all"]["weights_f1"] < 0.8 and h["all"]["active_f1"] < 0.8
