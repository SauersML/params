"""Checks of native-unit circuits (native_units.py) on the tiny random model of vpd4l's architecture
(test_goal_eval.tiny_native): every run against vpd_model's own forward pass with the units zeroed at each site's input
(Site.in_fn), windowed removals against whole runs, and the pruning curve's ends.

  ~/.local/bin/mem-lease 8 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_native_units.py
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import native_units  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    return nat, ids, [11], native_units.NativeUnits(nat, ids, [11])


def site_hooked(nat, ids, targets, m: native_units.Mask):
    """log q at the targets from vpd_model's Target.hidden with each layer's output matrix and MLP output matrix reading
    only the set's heads and neurons (Site.in_fn zeroes the rest of their input)."""
    tg = nat.target
    hd = tg.hd
    try:
        for i in range(tg.n_layer):
            hm = m.heads[i].float().repeat_interleave(hd, dim=1)  # [T, H * hd]
            tg.site(native.site_name(i, "o_proj")).in_fn = lambda x, hm=hm: x * hm
            nm = m.neurons[i].float()
            tg.site(native.site_name(i, "down_proj")).in_fn = lambda x, nm=nm: x * nm
        nat.vpd.clear()
        hidden = tg.hidden(torch.tensor([ids]))
    finally:
        for i in range(tg.n_layer):
            tg.site(native.site_name(i, "o_proj")).in_fn = None
            tg.site(native.site_name(i, "down_proj")).in_fn = None
    return torch.log_softmax((hidden[:, targets] @ tg.wte.T).float(), -1)


def random_mask(nu, seed, keep=0.5):
    g = torch.Generator().manual_seed(seed)
    m = nu.full()
    m.heads &= torch.rand(m.heads.shape, generator=g) < keep
    m.neurons &= torch.rand(m.neurons.shape, generator=g) < keep
    return m


def test_runs_match_the_models_own_forward(tiny):
    nat, ids, targets, nu = tiny
    assert abs(nu.kl(nu.full())[0]) < 1e-5  # every unit: the model
    for seed in (1, 2, 3):
        m = random_mask(nu, seed)
        want = float(carry.kl(nu.ref, site_hooked(nat, ids, targets, m))[0])
        assert abs(nu.kl(m)[0] - want) < 1e-4 + 1e-4 * want, seed


def test_empty_is_goal_evals_empty_circuit(tiny):
    nat, ids, targets, nu = tiny
    empty = nu.full().without(nu.full().units())
    ge = goal_eval.GoalEval(nat, changes=2)
    tx = ge.text(ids, targets, seed=1)
    assert abs(nu.kl(empty)[0] - ge._bounds(tx, "off")[0]) < 1e-4


def test_windowed_removals_match_whole_runs(tiny):
    """A batch of removals computes only the positions from its first one on: the same KL as the whole run."""
    nat, ids, targets, nu = tiny
    S = random_mask(nu, 4, keep=0.7)
    units = S.units()
    g = torch.Generator().manual_seed(5)
    picks = [units[int(i)] for i in torch.randint(len(units), (20,), generator=g)]
    removals = [[u] for u in picks] + [picks[:3], picks[3:7]]
    got = nu.kl(S, removals)
    for rem, k in zip(removals, got):
        assert abs(k - nu.kl(S.without(rem))[0]) < 1e-5, rem


def test_runs_skip_positions_holding_no_units(tiny):
    """A set at a few positions: removal runs computing only those positions and the targets give the whole runs' KL."""
    nat, ids, targets, nu = tiny
    S = random_mask(nu, 6, keep=0.6)
    hold = torch.zeros(nu.T, dtype=torch.bool)
    hold[[2, 5, 9]] = True
    S.heads &= hold[None, :, None]
    S.neurons &= hold[None, :, None]
    units = S.units()
    removals = [[u] for u in units[::7]] + [units[:5], units[-4:], [("neuron", 0, 4, 1)]]
    got = nu.kl(S, removals)
    for rem, k in zip(removals, got):
        assert abs(k - nu.kl(S.without(rem))[0]) < 1e-5, rem


def test_prune_curve(tiny):
    nat, ids, targets, nu = tiny
    positions = [3, 7, 11]
    curve = native_units.prune(nu, positions, sizes=[50, 7])
    sizes = [c["size"] for c in curve]
    assert sizes[0] == nu.at(positions).count() and sizes[-1] == 0
    assert all(a > b for a, b in zip(sizes, sizes[1:]))
    assert {50, 7} <= set(sizes) and all("units" in c for c in curve if c["size"] in (50, 7))
    empty = nu.full().without(nu.full().units())
    assert abs(curve[-1]["kl"] - nu.kl(empty)[0]) < 1e-5
    seven = next(c for c in curve if c["size"] == 7)
    m = nu.full().without(nu.full().units())
    for kind, l, t, j in seven["units"]:
        (m.heads if kind == "head" else m.neurons)[l, t, j] = True
    assert abs(nu.kl(m)[0] - seven["kl"]) < 1e-5 and all(t in positions for _, _, t, _ in seven["units"])


def test_f1_scale():
    assert native_units.f1(100.0, 100.0, 1.0) == 0.0 and native_units.f1(1.0, 100.0, 1.0) == 1.0
    assert abs(native_units.f1(10.0, 100.0, 1.0) - 0.5) < 1e-12


def test_native_necessity_ends(tiny):
    """Removing nothing from the model leaves it whole (N 0); removing every unit leaves the empty circuit (N 1)."""
    nat, ids, targets, nu = tiny
    full = nu.full()
    empty_kl = nu.kl(full.without(full.units()))[0]
    vpd_kl = 0.5 * empty_kl
    assert goal_eval.necessity_log(empty_kl, nu.kl(full, [[]])[0], vpd_kl) == 0.0
    assert abs(goal_eval.necessity_log(empty_kl, nu.kl(full, [full.units()])[0], vpd_kl) - 1) < 1e-6


def test_table():
    rows = [{"native": [{"size": 12, "f1": 0.5, "n": 0.25, "bits": 300.0}], "vpd_order": [{"size": 12, "f1": 0.9, "n": 0.5, "bits": 320.0, "f1_live": 0.9, "n_live": 0.5}]},
            {"native": [{"size": 12, "f1": 0.7, "n": None, "bits": 310.0}], "vpd_order": []}]
    text = native_units.table(rows)
    assert "native F1" in text and " 0.600 " in text and " 305 " in text and " 0.900 " in text


def test_teacher_search_on_native_units(tiny):
    """teach_native: the F1 circuit keeps F1 >= 0.9 when the search reaches it, the path shrinks to teacher.MIN_SIZE
    measuring as it goes, and each recorded step's KL is the run of its units."""
    import teacher

    nat, ids, targets, nu = tiny
    ge = goal_eval.GoalEval(nat, changes=2)
    tx = ge.text(ids, targets, seed=1)
    got = native_units.teach_native(ge, nu, tx, log=None)
    f = got["f1_circuit"]
    assert f["kl"] <= got["budget_kl"] + 1e-6 or not got["f1_from_targets"]
    sizes = [r["size"] for r in got["path"]]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] <= max(teacher.MIN_SIZE, sizes[0])
    assert got["measured_sets"] > 0 and got["top_k"] >= 1 and sizes[0] == got["start_size"] >= got["top_k"]
    step = got["path"][len(got["path"]) // 2]
    m = nu.full().without(nu.full().units())
    for kind, l, t, j in step["units"]:
        (m.heads if kind == "head" else m.neurons)[l, t, j] = True
    assert abs(nu.kl(m)[0] - step["kl_c"]) < 1e-5
    rest = nu.full().without(step["units"])
    assert abs(nu.kl(rest)[0] - step["kl_minus"]) < 1e-5


def test_teacher_steps_and_compare_table():
    path = {"start_nodes": [["h.0.mlp.c_fc", 5, 3], ["h.1.mlp.down_proj", 7, 3], ["h.0.attn.q_proj", 1, 2]],
            "start_index": [10, 20, 30],
            "steps": [{"size": 3, "removed": []}, {"size": 2, "removed": [20]}, {"size": 1, "removed": [10]}]}
    steps = native_units.teacher_steps(path)
    assert [s["size"] for s in steps] == [3, 2, 1]
    assert steps[1]["nodes"] == [("h.0.attn.q_proj", 2, 1), ("h.0.mlp.c_fc", 3, 5)] and steps[2]["nodes"] == [("h.0.attn.q_proj", 2, 1)]
    assert native_units._nearest(steps, 2)["size"] == 2 and native_units._nearest(steps, 12) is None
    rows = [{"task": "a", "local": True, "kind": "native", "size": 12, "nodes": 12, "f1": 0.5, "n_vpd": None, "n_real": 0.0},
            {"task": "a", "local": True, "kind": "teacher", "size": 12, "nodes": 12, "f1": 0.9, "n_vpd": 0.5, "n_real": 0.1},
            {"task": "b", "local": False, "kind": "teacher", "size": 12, "nodes": 13, "f1": 0.7, "n_vpd": 0.3, "n_real": 0.0}]
    text = native_units.compare_table(rows)
    assert "non-local" in text and " 0.800 " in text and " 0.500 " in text


def test_effort_table(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "summary_0.jsonl").write_text(json.dumps(
        {"task": "text1", "local": True, "measured_sets": 900, "seconds": 12.0, "f1_from_targets": True,
         "f1_circuit": {"size": 22, "positions": 1, "f1": 0.991}}) + "\n")
    (tmp_path / "n.jsonl").write_text(json.dumps(
        {"task": "text1", "measured_sets": 5000, "seconds": 30.0, "f1_from_targets": False,
         "f1_circuit": {"size": 60, "positions": 2, "f1": 0.9}}) + "\n")
    text = native_units.effort_table(tmp_path / "n.jsonl", tmp_path / "t")
    assert "22 at 1 (0.99, targets)" in text and "60 at 2 (0.90)" in text and " 5000 " in text


def test_compare_and_reuse(tiny, tmp_path, monkeypatch):
    """compare on one tiny text: the teacher's path steps and VPD's order at the sizes asked, the native path's
    nearest steps, and the same rows again from its own output (reuse) with no runs of the VPD side."""
    nat, ids, targets, nu = tiny
    task = tmp_path / "tiny0.json"
    task.write_text("{}")
    monkeypatch.setattr(native, "tasks", lambda split: [task])
    monkeypatch.setattr(native, "text", lambda p: (ids, targets))
    order = nat._vpd_order(ids, targets)[:8]
    tdir = tmp_path / "teacher"
    tdir.mkdir()
    (tdir / "tiny0.path.json").write_text(json.dumps({
        "start_nodes": [[n, c, t] for n, t, c in order], "start_index": list(range(8)),
        "steps": [{"size": 8, "removed": []}, {"size": 4, "removed": [4, 5, 6, 7]}, {"size": 2, "removed": [2, 3]}]}))
    (tdir / "summary_0.jsonl").write_text(json.dumps({"task": "tiny0", "local": True}) + "\n")
    nfile = tmp_path / "native.jsonl"
    nfile.write_text(json.dumps({"task": "tiny0", "path": [{"size": 9, "f1": 0.8, "n": 0.6}, {"size": 4, "f1": 0.5, "n": 0.3}]}) + "\n")
    ge = goal_eval.GoalEval(nat, changes=2)
    rows = native_units.compare(nfile, tdir, "heldout", 1, sizes=(2, 8), ge=ge, log=None)
    kinds = {(r["kind"], r["size"]): r for r in rows}
    assert kinds[("teacher", 2)]["nodes"] == 2 and kinds[("teacher", 8)]["nodes"] == 8 and kinds[("VPD order", 8)]["nodes"] == 8
    assert kinds[("native", 8)]["f1"] == 0.8 and kinds[("native", 2)]["n_real"] == 0.3 and all(r["local"] for r in rows)
    out = tmp_path / "rows.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(native_units, "_vpd_side", lambda *a, **k: pytest.fail("reuse ran the VPD side"))
    assert native_units.compare(nfile, tdir, "heldout", 1, sizes=(2, 8), ge=ge, reuse=out, log=None) == rows
