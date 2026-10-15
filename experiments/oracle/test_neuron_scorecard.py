"""Checks of scorecard v2 (neuron_scorecard.py) on a tiny random Qwen3: with thresholds of zero R is the model; a
circuit of every unit runs as R in its own run, under any donor's fill, and predicts every swap exactly; removals and the
circuit's run equal hand-written hooks; whole-neuron units cover every position; the margins read the records.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_neuron_scorecard.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neuron_scorecard as ns  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 48, (10,), generator=g).tolist()
    donor = torch.randint(0, 48, (12,), generator=g).tolist()
    run = ns.neuron_hubs.Runner(model)
    with torch.no_grad():
        cap = []
        run.logits(ids, 9, capture=cap)
    mx = torch.stack([x.abs().max(0).values for x in cap])
    bg = {"thr": 0.2 * mx, "max": mx}
    return model, run, ids, donor, bg


def contrast_of(model, ids, edit=None):
    hooks = []
    if edit is not None:
        for L, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: (edit(L, a[0]),)))
    try:
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, 9].float(), -1)
    finally:
        for h in hooks:
            h.remove()
    return float(ns.log2_contrast(lp, [3], [5, 7]))


def test_zero_thresholds_are_the_model(tiny):
    model, run, ids, _, bg = tiny
    rs = ns.Runs(run, ids, 9, [3], [5, 7], {"thr": torch.zeros_like(bg["thr"]), "max": bg["max"]}, [(1, 4, 7)])
    assert rs.cR == pytest.approx(contrast_of(model, ids), abs=1e-4)
    assert rs.runs([{"weak": False}])[0]["contrast"] == pytest.approx(contrast_of(model, ids), abs=1e-4)


def test_runs_equal_hooks(tiny):
    model, run, ids, _, bg = tiny
    thr = bg["thr"]
    units = [(0, 9, 3), (1, 5, 11), (2, 9, 30)]
    rs = ns.Runs(run, ids, 9, [3], [5, 7], bg, units)
    weak = lambda L, h: h * (h.abs() >= thr[L])  # noqa: E731
    assert rs.cR == pytest.approx(contrast_of(model, ids, weak), abs=1e-4)
    keep = rs.keep

    def circuit(L, h):
        return weak(L, h) * keep[L][None]
    assert rs.runs([{"circuit": True}])[0]["contrast"] == pytest.approx(contrast_of(model, ids, circuit), abs=1e-4)

    def removed(L, h):
        h = weak(L, h).clone()
        for l, p, j in units:
            if l == L:
                h[:, p, j] = 0
        return h
    assert rs.runs([{"aff": rs.zero(rs.all_cells())}])[0]["contrast"] == pytest.approx(contrast_of(model, ids, removed), abs=1e-4)


def test_whole_neuron_units_cover_every_position(tiny):
    _, run, ids, _, bg = tiny
    rs = ns.Runs(run, ids, 9, [3], [5, 7], bg, [(1, 11)])
    assert rs.cells[(1, 11)] == [(1, p, 11) for p in range(10)]
    assert int(rs.keep.sum()) == 10


def test_full_circuit_is_r_under_every_fill_and_swap(tiny):
    _, run, ids, donor, bg = tiny
    units = [(L, j) for L in range(3) for j in range(40)]
    rs = ns.Runs(run, ids, 9, [3], [5, 7], bg, units)
    rs.add_donor("d", donor)
    sw = [(L, p, j, 0.0, float(rs.donors["d"][L, p, j])) for L, p, j in rs.all_cells()]
    circ, patched, smr, smc = rs.runs([{"circuit": True}, {"circuit": True, "fill": "d"}, {"aff": sw}, {"circuit": True, "aff": sw}])
    assert circ["contrast"] == pytest.approx(rs.cR, abs=1e-4)
    assert patched["contrast"] == pytest.approx(rs.cR, abs=1e-4)
    assert smc["contrast"] == pytest.approx(smr["contrast"], abs=1e-4)
    # the donor's swap is its own R run's activations: the run reproduces the donor's first T tokens
    assert torch.allclose(rs.capture(donor), rs.donors["d"])


def test_margins_read_records():
    rec = {"contrast": {"R": 4.0}, "necessity": {"R": -0.5}, "sufficiency": {"c": 3.8},
           "draws": {"patching_near": [{"c": 4.0}, {"c": 3.0}], "patching_far": [{"c": 4.0}],
                     "swaps": [{"dm": 1.0, "dc": 1.0}, {"dm": -2.0, "dc": -2.0}], "steering": [{"dm": 0.01, "dc": 0.02}],
                     "weights": [{"dm": 1.0, "dc": 0.0}, {"dm": 1.0, "dc": 0.0}]}}
    m = ns.margins(rec)
    assert m["necessity"]["score"] == 1.0 and m["necessity"]["se"] == 0.0
    assert m["sufficiency"]["score"] == pytest.approx(0.95) and m["sufficiency"]["pass_old"]
    assert m["patching_near"]["score"] == pytest.approx(0.875) and m["patching_near"]["se"] > 0
    assert m["swaps"]["score"] == 1.0
    assert m["steering"]["score"] is None and m["steering"]["pass_new"] is None
    assert m["weights"]["score"] == 0.0 and m["weights"]["pass_new"] is False
