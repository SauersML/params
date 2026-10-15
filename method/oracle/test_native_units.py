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


@pytest.fixture(scope="module")
def phi_text(tiny):
    nat, ids, targets, nu = tiny
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    tx = ge.text(ids, targets, seed=5)
    for split, n in (("search", 6), ("eval", 5)):  # far donors in the tiny vocabulary, some shorter than the text
        ge.donor_pools[split] = [(f"{split}{i}", torch.randint(0, 40, (8 + 2 * i,), generator=g).tolist()) for i in range(n)]
    return ge, tx


def test_graph_ref_units_are_native_runs(tiny, phi_text):
    """goal_eval.graph_ref on the whole model with the units outside a set zeroed is the set's native run (the model's
    own forward with those heads and neurons zeroed); every unit held at its own run's values is the model."""
    nat, ids, targets, nu = tiny
    ge, tx = phi_text
    ph = native_units.NativePhi(ge, tx)
    for seed in (1, 2):
        m = random_mask(nu, seed)
        got = ge.graph_ref(ids, targets, ph.model, units=ph._units([~m.heads], [~m.neurons], None))
        assert torch.allclose(got, site_hooked(nat, ids, targets, m), atol=1e-4)
    full = nu.full()
    held = ge.graph_ref(ids, targets, ph.model, units=ph._units([full.heads], [full.neurons], ph.capture(ids)))
    assert float(carry.kl(tx.ref, held)[0]) < 1e-6


def test_native_bits_count_every_connection(tiny):
    """native_bits: the units' names plus every connection the model has among them, counted pair by pair."""
    nat, ids, targets, nu = tiny
    m = random_mask(nu, 4, keep=0.3)
    units = m.units()
    rd = {"head": 0.0, "neuron": 0.5}
    wr = {"head": 0.5, "neuron": 1.0}
    inner = sum(1 for r in units for w in units if w[2] == r[2] and w[1] + wr[w[0]] <= r[1] + rd[r[0]])
    inner += sum(1 for r in units for w in units if r[0] == w[0] == "head" and (w[1], w[3]) == (r[1], r[3]) and w[2] < r[2])
    out = sum(1 for u in units if u[2] in targets)
    n, per = len(units), nu.L * (nu.H + nu.Dm)
    want = n * math.log2(7 * per) + math.log2(n) * (2 * inner + out)
    assert abs(native_units.native_bits(m, targets, 7, per) - want) < 1e-6


def test_native_units_under_phi(tiny, phi_text):
    """NativePhi: every unit is the model (point, track_pp, interchange_pp and patching at 1 where VPD's reference is
    defined); the families take VPD's own draws (track_pp's changed texts, the donors and their VPD references, N's
    controls); each score is goal_eval's pooling of its records; phi_shared over SHARED from rl_terms."""
    nat, ids, targets, nu = tiny
    ge, tx = phi_text
    ph = native_units.NativePhi(ge, tx)
    full, half = nu.full(), random_mask(nu, 5)
    rows = ph.rows([full, half])
    f, h = rows
    assert f["kl_point"] < 1e-5
    for k in ("s_track_pp", "s_interchange_pp", "s_patching_near", "s_patching_far"):
        assert f[k] is None or abs(f[k] - 1) < 1e-6, (k, f[k])
    vpd_tp = ge.track_pp(tx, [native.Graph()], "eval", goal_eval.CHANGED_DRAWS)[0]["draws_track_pp"]
    assert [d[0] for d in h["draws_track_pp"]] == [d[0] for d in vpd_tp] and [d[2:] for d in h["draws_track_pp"]] == [d[2:] for d in vpd_tp]
    vd = ge.donors(tx, [native.Graph(masks=tx.vpd)], "eval", goal_eval.DONORS_NEAR, ge.far_donors(tx, ge.far_pool("eval"), goal_eval.DONORS_FAR))[0]
    for pool in ("near", "far"):
        kv = {json.dumps(d[0]): d[2] for d in vd.get(f"draws_patching_{pool}") or []}
        assert all(abs(d[2] - kv[json.dumps(d[0])]) < 1e-9 for d in h.get(f"draws_patching_{pool}") or [])
    vn = ge.necessity_spec(tx, [native.Graph()], "eval")[0]["draws_n_spec"]
    assert [(d[2], d[4]) for d in h["draws_n_spec"]] == [(d[2], d[4]) for d in vn]
    assert h["s_nodes"] == goal_eval._pooled("weighted", h["draws_nodes"], None)
    assert h["s_track_pp"] == goal_eval._pooled("weighted", h["draws_track_pp"], None)
    assert h["n_spec"] == goal_eval._spec(h["draws_n_spec"])
    assert set(h["rl_terms"]) >= {"s_point_c", "s_nodes", "s_track_pp", "n_spec"}
    eps = goal_eval._eps(h)
    want = goal_eval._phi([(goal_eval.family_score(h, x), eps[x]) for x in native_units.SHARED if goal_eval.family_score(h, x) is not None])
    assert h["phi_shared"] == want


def test_prune_to_bits(tiny):
    """prune_to_bits: each goal's set is within its bits and the sets shrink with the goals."""
    nat, ids, targets, nu = tiny
    per = nu.L * (nu.H + nu.Dm)
    start = native_units.native_bits(nu.at([9, 10, 11]), targets, 12, per)
    goals = [start / 4, start / 2]
    got = native_units.prune_to_bits(nu, [9, 10, 11], goals, 12)
    assert set(got) == set(goals)
    for gl, m in got.items():
        assert native_units.native_bits(m, targets, 12, per) <= gl
    assert set(got[goals[0]].units()) <= set(got[goals[1]].units())


def mean_hooked(nat, ids, targets, live, means):
    """log q at the targets from vpd_model's own forward with each output matrix reading the set's units live and every
    other unit at its mean (Site.in_fn on o_proj's and down_proj's inputs)."""
    tg = nat.target
    hd = tg.hd
    try:
        for i in range(tg.n_layer):
            hm = live[0][i].repeat_interleave(hd, dim=1)  # [T, H * hd]
            mh = means["heads"][i].reshape(-1)
            tg.site(native.site_name(i, "o_proj")).in_fn = lambda x, hm=hm, mh=mh: x * hm + (1 - hm) * mh
            nm, mn = live[1][i], means["neurons"][i]
            tg.site(native.site_name(i, "down_proj")).in_fn = lambda x, nm=nm, mn=mn: x * nm + (1 - nm) * mn
        nat.vpd.clear()
        hidden = tg.hidden(torch.tensor([ids]))
    finally:
        for i in range(tg.n_layer):
            tg.site(native.site_name(i, "o_proj")).in_fn = None
            tg.site(native.site_name(i, "down_proj")).in_fn = None
    return torch.log_softmax((hidden[:, targets] @ tg.wte.T).float(), -1)


@pytest.fixture(scope="module")
def mean_case(tiny):
    nat, ids, targets, nu = tiny
    g = torch.Generator().manual_seed(5)
    means = {"heads": torch.randn(nat.target.n_layer, nat.target.n_head, nat.target.hd, generator=g) * 0.3,
             "neurons": torch.randn(nat.target.n_layer, nu.Dm, generator=g).abs() * 0.2}
    return nat, ids, targets, means, native_units.MeanUnits(nat, ids, targets, means)


def test_mean_units_are_the_models_forward(mean_case):
    """MeanUnits' runs equal vpd_model's own forward with the left-out units' outputs at their means, on random and
    fractional sets; every unit live is the model; none live is the all-mean run."""
    nat, ids, targets, means, mu = mean_case
    g = torch.Generator().manual_seed(6)
    ones = tuple(x[0] for x in mu.ones())
    rows = [ones, tuple(x * 0 for x in ones)] + [tuple((torch.rand(x.shape, generator=g) < 0.5).float() for x in ones) for _ in range(3)]
    rows.append(tuple(torch.rand(x.shape, generator=g) for x in ones))  # strengths in between (the gradients' path)
    with torch.no_grad():
        got = mu.logq(tuple(torch.stack([r[j] for r in rows]) for j in range(2)))
    for b, r in enumerate(rows):
        assert float((got[b] - mean_hooked(nat, ids, targets, r, means)[0]).abs().max()) < 1e-4
    assert float(carry.kl(nat.reference([ids], targets), got[:1])[0]) < 1e-5


def test_mean_units_frozen_and_kl(mean_case):
    """Units frozen at their clean values make the clean run whatever is live inside; kl batches rows as one run each,
    and lp is log2 p of the model's top token."""
    nat, ids, targets, means, mu = mean_case
    ones = tuple(x[0] for x in mu.ones())
    g = torch.Generator().manual_seed(7)
    inside = tuple((torch.rand(x.shape, generator=g) < 0.3).float() for x in ones)
    outside = tuple(x == 0 for x in inside)
    assert mu.kl([inside], [outside])[0] < 1e-5
    rows = [inside, ones, tuple(x * 0 for x in ones)]
    one_by_one = [mu.kl([r])[0] for r in rows]
    assert all(abs(a - b) < 1e-5 for a, b in zip(mu.kl(rows), one_by_one))
    y = mu.ref[0].argmax(-1)
    assert abs(mu.kl([ones], lp=True)[0] - float(mu.ref[0].gather(-1, y[:, None]).sum()) / math.log(2.0)) < 1e-4


def test_build_native_meets_its_target(mean_case):
    """build_native's circuit meets the KL target asked for (measured again on its mask), smaller than every unit;
    the mediated share of the whole model is 1 (nothing outside to freeze) and the frozen run is the clean one."""
    import types

    nat, ids, targets, means, mu = mean_case
    ge, tx = types.SimpleNamespace(nat=nat), types.SimpleNamespace(ids=ids, targets=targets, seed=3)
    empty = mu.kl([tuple(x[0] * 0 for x in mu.ones())])[0]
    goal = empty * 0.05
    S, rec = native_units.build_native(ge, tx, kl_target=goal, means=means)
    live = (S.heads[:, :mu.T].float(), S.neurons[:, :mu.T].float())
    assert rec["kl"] <= goal and abs(mu.kl([live])[0] - rec["kl"]) < 1e-5
    assert 0 < S.count() < rec["of"] and rec["units"] == S.count()
    whole = native_units.Mask(*(x[0].bool() for x in mu.ones()))
    ms = native_units.mediated_share(ge, tx, whole, means, candidates=16)
    assert abs(ms["lp"] - ms["lp_frozen"]) < 1e-4 and all(abs(ms[f"top{k}"]["share"] - 1) < 1e-4 for k in (1, 3, 5))
    part = native_units.mediated_share(ge, tx, S, means, candidates=16)
    assert abs(part["lp"] - part["lp_frozen"]) < 1e-4
