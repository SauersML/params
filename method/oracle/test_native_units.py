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


@pytest.fixture(scope="module")
def tiny_hf():
    """A tiny random Qwen3 (the HF adapter's model kind), float32 on the CPU, and means for its units."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    means = {"heads": torch.randn(2, 4, 8, generator=g) * 0.2, "neurons": torch.randn(2, 40, generator=g) * 0.2}
    ids = torch.randint(0, 48, (10,), generator=g).tolist()
    return model, ids, means


def test_hf_units_are_the_models_run(tiny_hf):
    """HFUnits: every unit live is the model's own run; a layer's neurons all at their means is the model with that MLP's
    output replaced by down_proj(mean), and its heads all at their means the model with o_proj's input the means; units
    frozen at their clean values give the clean run whatever is live inside."""
    model, ids, means = tiny_hf
    hu = native_units.HFUnits(model, ids, [9], means)
    with torch.no_grad():
        ref = torch.log_softmax(model(torch.tensor([ids])).logits[:, [9]].float(), -1)
        assert float((hu.ref - ref).abs().max()) < 1e-5
        h, n = (x[0] for x in hu.ones())
        n2 = n.clone()
        n2[1] = 0.0
        got = hu.logq((h[None], n2[None]))
        mlp = model.model.layers[1].mlp
        fwd = mlp.forward
        mlp.forward = lambda x: mlp.down_proj(means["neurons"][1].expand(*x.shape[:-1], -1))
        try:
            want = torch.log_softmax(model(torch.tensor([ids])).logits[:, [9]].float(), -1)
        finally:
            mlp.forward = fwd
        assert float((got - want).abs().max()) < 1e-5
        h2 = h.clone()
        h2[0] = 0.0
        got = hu.logq((h2[None], n[None]))
        o = model.model.layers[0].self_attn.o_proj
        hk = o.register_forward_pre_hook(lambda mod, args: (means["heads"][0].reshape(1, 1, -1).expand_as(args[0]),))
        try:
            want = torch.log_softmax(model(torch.tensor([ids])).logits[:, [9]].float(), -1)
        finally:
            hk.remove()
        assert float((got - want).abs().max()) < 1e-5
        g = torch.Generator().manual_seed(3)
        inside = tuple((torch.rand(x.shape, generator=g) < 0.3).float() for x in (h, n))
        assert hu.kl([inside], [tuple(x == 0 for x in inside)])[0] < 1e-5


def test_hf_units_under_a_peft_adapter(tiny_hf):
    """A PeftModel is run with its adapter on: HFUnits' model run is the adapted model's."""
    peft = pytest.importorskip("peft")
    model, ids, means = tiny_hf
    import copy

    pm = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(r=2, target_modules=["o_proj", "down_proj", "q_proj"], init_lora_weights=False))
    hu = native_units.HFUnits(pm, ids, [9], means)
    with torch.no_grad():
        ref = torch.log_softmax(pm(input_ids=torch.tensor([ids])).logits[:, [9]].float(), -1)
        base = torch.log_softmax(model(torch.tensor([ids])).logits[:, [9]].float(), -1)
    assert float((hu.ref - ref).abs().max()) < 1e-5 and float((ref - base).abs().max()) > 1e-4


def test_build_hf(tiny_hf):
    """build_hf meets its KL target (measured again on its units), lists every unit with the highest-ranked units' measured
    effects, and keeps within a pass budget."""
    model, ids, means = tiny_hf
    hu = native_units.HFUnits(model, ids, [9], means)
    empty = hu.kl([tuple(x[0] * 0 for x in hu.ones())])[0]
    rec = native_units.build_hf(model, ids, 9, kl_target=empty * 0.05, means=means, effects=4)
    h, n = (torch.zeros_like(x[0]) for x in hu.ones())
    for kind, l, t, j, _ in rec["units"]:
        (h if kind == "head" else n)[l, t, j] = 1.0
    assert rec["kl"] <= rec["kl_target"] and abs(hu.kl([(h, n)])[0] - rec["kl"]) < 1e-5
    assert 0 < len(rec["units"]) < rec["of"] and all(u[4] is not None for u in rec["units"][:4]) and rec["units"][4][4] is None
    cheap = native_units.build_hf(model, ids, 9, kl_target=empty * 0.05, means=means, budget_passes=256)
    assert cheap["passes"] <= 256 and cheap["kl"] <= cheap["kl_target"]


@pytest.mark.parametrize("fill", [None, "nonzero_mean"])
def test_native_point_common_bounds_physical_and_vpd_score(tiny, phi_text, monkeypatch, fill):
    """A physically executed nontrivial native circuit receives exactly the VPD point
    calibration for the SAME measured output distribution; its own ablated-empty
    baseline cannot alter that credit. This tests scoring, not a representation
    equivalence between neuron masks and rank-one masks."""
    nat, ids, targets, nu = tiny
    ge, tx = phi_text
    means = None
    if fill is not None:
        g = torch.Generator().manual_seed(17)
        means = {"heads": torch.randn(nu.L,nu.H,nat.target.hd,generator=g)*.3,
                 "neurons": torch.rand(nu.L,nu.Dm,generator=g)*.2}
    ph = native_units.NativePhi(ge,tx,means=means)
    mask = random_mask(nu,17,keep=.4)
    # Actual own-routing/norm graph execution and an independent direct-W hook.
    q = ge.graph_ref(ids,targets,ph.model,units=ph._units([~mask.heads],[~mask.neurons],ph._ablate(len(ids))))
    expected_q = site_hooked(nat,ids,targets,mask) if means is None else mean_hooked(nat,ids,targets,(mask.heads.float(),mask.neurons.float()),means)
    assert torch.allclose(q,expected_q,atol=1e-4)
    common = ge._bounds(tx,"live")
    monkeypatch.setattr(ph,"empty",lambda: pytest.fail("point must not read the native own-empty baseline"))
    native_row = ph.point([mask])[0]
    kc = float(carry.kl(tx.ref,q)[0])
    assert abs(native_row['kl_point']-kc)<1e-6
    # Feed this identical physically measured q through the existing VPD point
    # scorer. Only candidate execution is substituted, never bounds/formulas.
    monkeypatch.setattr(ge,"runs",lambda *args,**kwargs:q)
    vpd_row = ge.sufficiency(tx,[native.Graph([],complete=True)],conventions=('live',),vpd_minus=False)[0]
    assert abs(vpd_row['kl_live']-native_row['kl_point'])<1e-6
    assert native_row['s_point_c']==vpd_row['f1_live']
    assert native_row['_raw_point']==goal_eval._fid_raw(common[0],kc,common[1])
    wrapped = {'s_point_c':native_row['s_point_c'],'rl_terms':{'s_point_c':native_row['_raw_point']}}
    expected = {'s_point_c':vpd_row['f1_live'],'rl_terms':{'s_point_c':goal_eval._fid_raw(common[0],vpd_row['kl_live'],common[1])}}
    assert goal_eval.family_score(wrapped,'point')==goal_eval.family_score(expected,'point')


def test_native_point_common_bounds_not_own_empty_or_floored_raw(monkeypatch):
    """Nontrivial physical distributions exercise interior and negative recovery;
    an unrelated native empty value must not erase/floor these gradients."""
    import types
    ph = object.__new__(native_units.NativePhi)
    p = torch.tensor([[[0.,0.]]]).log_softmax(-1)
    q = torch.tensor([[[4.,0.]]]).log_softmax(-1)
    tx = types.SimpleNamespace(ids=[0],targets=[0],ref=p,T=1)
    ph.tx=tx;ph.model=native.Graph();ph.means=None
    ph.ge=types.SimpleNamespace(_bounds=lambda tx,conv:(4.,.25),graph_ref=lambda *a,**kw:q)
    ph._units=lambda *a:None
    ph.empty=lambda:pytest.fail('common point must ignore native empty cache')
    mask=native_units.Mask(torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,1,dtype=torch.bool))
    row=ph.point([mask])[0];kc=float(carry.kl(p,q)[0])
    assert 0<row['s_point_c']<1 and row['_raw_point']==goal_eval._fid_raw(4.,kc,.25)
    ph.ge._bounds=lambda tx,conv:(.5,.25)
    row=ph.point([mask])[0]
    assert row['s_point_c']==0 and row['_raw_point']<0
    ph.ge._bounds=lambda tx,conv:(.25,.25)
    assert ph.point([mask])[0]['s_point_c'] is None


class AdmissionCase:
    """Deterministic unit-output loss table: isolates admission from ranking noise."""
    L, T, H, Dm, dev = 1, 1, 1, 7, 'cpu'
    def __init__(self, empty=1.):
        self.passes = 0
        self.empty = empty
    def kl(self, rows):
        rows = list(rows)
        self.passes += len(rows)
        return [self.empty if sum(int(x.sum()) for x in row)==0 else .01 for row in rows]


def admission_fixture(monkeypatch, empty=1.):
    monkeypatch.setattr(native_units,'fisher',lambda mu,*a,**kw:torch.arange(8.,0.,-1.))
    return AdmissionCase(empty)


def test_shared_admission_nonmonotone_prefix_and_rejected_cleanup(monkeypatch):
    """3 and6 pass, 4 andallsmaller fail: point-only agreement cannot admit them.
    Cleanup's point-eligible deletions must not undo a verified shared-family pass."""
    mu=admission_fixture(monkeypatch)
    seen=[]
    def gate(live):
        count=sum(int(x.sum()) for x in live);seen.append(count)
        return {'passed':count in (3,6),'shared_score':1. if count in (3,6) else 0.}
    live,rec=native_units.build_units(mu,0,.1,stages=(1.,),admit=gate,max_gate_calls=16)
    assert sum(int(x.sum()) for x in live)==3
    assert rec['admission']['status']=='ADMITTED' and rec['units']==3
    assert 1 in seen and 2 in seen and 3 in seen
    # Smaller point-passing cleanup masks were checked/rejected, not silently kept.
    assert all(x['result']['passed'] is False for x in rec['admission']['evaluations'] if x['units']<3)
    assert rec['admission']['calls']<=16


def test_shared_admission_exhaustion_never_returns_unverified_full(monkeypatch):
    mu=admission_fixture(monkeypatch)
    seen=[]
    def gate(live):
        seen.append(sum(int(x.sum()) for x in live));return {'passed':False}
    with pytest.raises(native_units.NoAdmittedCandidate,match='NO_ADMITTED_CANDIDATE') as caught:
        native_units.build_units(mu,0,.1,stages=(1.,),admit=gate,max_gate_calls=2)
    record=caught.value.record['admission']
    assert len(seen)==record['calls']==2
    assert record['budget_exhausted'] and record['status']=='NO_ADMITTED_CANDIDATE'


def test_shared_admission_empty_is_a_real_candidate(monkeypatch):
    mu=admission_fixture(monkeypatch,empty=0.)
    live,rec=native_units.build_units(mu,0,.1,admit=lambda m:{'passed':sum(int(x.sum()) for x in m)==0},max_gate_calls=1)
    assert rec['units']==0 and all(int(x.sum())==0 for x in live)
    assert rec['admission']['status']=='ADMITTED' and rec['admission']['calls']==1


def test_shared_admission_budget_preserves_cached_verified_candidate(monkeypatch):
    mu=admission_fixture(monkeypatch)
    def gate(live):return {'passed':sum(int(x.sum()) for x in live)==3}
    live,rec=native_units.build_units(mu,0,.1,stages=(1.,1.),admit=gate,max_gate_calls=3)
    assert rec['units']==3 and rec['admission']['calls']==3
    assert rec['passes_excludes_admission']
    assert sum(int(x.sum()) for x in live)==3


def test_shared_admission_default_and_malformed_results(monkeypatch):
    mu=admission_fixture(monkeypatch)
    live,rec=native_units.build_units(mu,0,.1,stages=(1.,))
    assert rec['units']==1 and 'admission' not in rec
    for callback in (lambda x:True,lambda x:{'passed':1},lambda x:{'passed':True,'score':float('nan')}):
        with pytest.raises(ValueError):
            native_units.build_units(mu,0,.1,stages=(1.,),admit=callback)


@pytest.mark.parametrize("explicit", [True, False])
def test_build_native_shared_gate_uses_common_search_floors(monkeypatch, explicit):
    """The wrapper reads the same VPD floors and refuses a missing scored family;
    it cannot lower its own floor or accidentally consume eval draws."""
    import types
    mu=AdmissionCase();tx=types.SimpleNamespace(ids=[0],targets=[0],seed=0)
    calls=[];pool=object();inversions=[]
    monkeypatch.setattr(native_units,'kl_goal',lambda ge,tx,f1=.9:(inversions.append(f1) or .25))
    ge=types.SimpleNamespace(nat=object(),vpd_floors=lambda tx,pool_arg,split:(calls.append((pool_arg,split)) or
        {'families':['point','nodes'],'floors':{'point':.6,'nodes':.7}}))
    monkeypatch.setattr(native_units,'MeanUnits',lambda *a:mu)
    class Phi:
        def __init__(self, ge,tx,split,means):assert split=='search'
        def rows(self, masks):return [{'s_point_c':.8,'s_nodes':None,'kl_point':.01}]
    monkeypatch.setattr(native_units,'NativePhi',Phi)
    def build(mu,seed,goal,**kw):
        live=(torch.ones(1,1,1),torch.ones(1,1,7))
        result=kw['admit'](live)
        assert result['passed'] is False and result['floors']=={'point':.6,'nodes':.7}
        assert result['scores']['nodes'] is None and result['split']=='search'
        assert kw['max_gate_calls']==3
        assert goal==(.1 if explicit else .25)
        raise native_units.NoAdmittedCandidate({'admission':result})
    monkeypatch.setattr(native_units,'build_units',build)
    with pytest.raises(native_units.NoAdmittedCandidate):
        native_units.build_native(ge,tx,kl_target=.1 if explicit else None,means={},shared_floors=True,pool=pool,max_gate_calls=3)
    assert calls==[(pool,'search')]
    assert inversions==([] if explicit else [.6])
