"""Checks of the fast builder (builder_fast.py) on a tiny random model (test_goal_eval's): its resampling influence is
GoalEval.resample_influence's, its circuit lies inside the START set at its measured positions and meets the floors it
records, and the JSON carries the build record.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_builder_fast.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import builder_fast  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    for split, n in (("search", 6), ("eval", 5)):  # far donors in the tiny vocabulary, some shorter than the text
        ge.donor_pools[split] = [(f"{split}{i}", torch.randint(0, 40, (8 + 2 * i,), generator=g).tolist()) for i in range(n)]
    task = {"prompts": [{"token_ids": ids, "target_positions": [11]}]}
    return nat, ge, task


def test_influence_is_resample_influence(tiny, monkeypatch):
    nat, ge, task = tiny
    p = task["prompts"][0]
    tx = ge.text(p["token_ids"], p["target_positions"], seed=5)
    want = ge.resample_influence(tx, "search", 4)
    tx2 = ge.text(p["token_ids"], p["target_positions"], seed=5)
    monkeypatch.setattr(builder_fast, "SHORTLIST", len(want))   # every position measured with every donor
    got = builder_fast.influence(ge, tx2, "search", 4)
    assert set(got) == set(want) and all(abs(got[q] - want[q]) < 1e-4 * max(1.0, want[q]) for q in want)
    assert tx2.__dict__["_resample"][("search", 4, ge.FIRST_POSITION)] is got   # the input-scope family's draws read it


def test_build_inside_the_start_meeting_its_floors(tiny):
    nat, ge, task = tiny
    out = builder_fast.build(ge, task, seed=5, builder="builder_fast@test")
    rec = out["build"]
    assert out["units"] == "vpd" and out["builder"] == "builder_fast@test" and out["build_seconds"] >= 0
    p = task["prompts"][0]
    tx = ge.text(p["token_ids"], p["target_positions"], seed=5)
    start = {(n, t, c) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist() if t in set(rec["positions"])}
    nodes = {(n, int(t), int(c)) for n, c, t in out["nodes"]}
    assert nodes <= start and rec["size"] == len(nodes) and rec["start_units"] == len(start)
    if rec["met"]:                                       # the size written met every floor it measured
        size = rec["size"] if not rec.get("dropped_inert") and not (rec.get("chunks") or {}).get("removed") else None
        row = next((t for t in rec["tried"] if t["size"] == size), None) if size is not None else None
        if row is not None:
            assert all(row[f] is None or rec["floors"][f] is None or row[f] >= rec["floors"][f] - 1e-9
                       for f in ("point", "track_pp", "scope", "interchange_pp"))


def test_build_neurons(tiny, monkeypatch):
    """units="neurons" dispatches to native_units.build_native: the bake-off's native form (heads and neurons as nodes),
    its circuit meeting the KL target it records."""
    import native_units

    nat, ge, task = tiny
    g = torch.Generator().manual_seed(4)
    means = {"heads": torch.randn(nat.target.n_layer, nat.target.n_head, nat.target.hd, generator=g) * 0.2,
             "neurons": torch.randn(nat.target.n_layer, nat.target.site(native.site_name(0, "c_fc")).W.shape[0], generator=g) * 0.2}
    monkeypatch.setattr(native_units, "unit_means", lambda nat_, *a, **k: means)
    out = builder_fast.build(ge, task, units="neurons", seed=5, builder="builder_fast@test")
    assert out["units"] == "native" and out["semantics"] == "mean_ablation" and out["nodes"]
    assert out["build"]["kl"] <= out["build"]["kl_target"] and out["build"]["units"] == len(out["nodes"])
