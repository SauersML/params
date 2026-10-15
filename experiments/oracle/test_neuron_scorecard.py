"""Checks of scorecard v2 (neuron_scorecard.py) on a tiny random Qwen3: with thresholds of zero R is the model; a
circuit of every unit runs as R isolated, under any donor's fill and with the outside frozen, and predicts every swap
exactly; removals, isolation (massive values and the first position live) and the side effect equal hand-written hooks;
counterfactual runs and their patches equal hooks; whole-neuron units cover every position; the margins
read the records; the unrelated-text KL equals a hand computation.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_neuron_scorecard.py -q
"""
import math
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
    nr = ns.nb.Neurons(model, None)
    with torch.no_grad():
        cap = {}
        nr.logits(ids, [9], capture=cap)
    mx = torch.stack([h[0].abs().max(0).values for h in cap["h"]])
    bg = {"thr": 0.2 * mx, "max": mx, "massive": 0.9 * mx.max(1).values, "file": None}
    return model, nr, ids, donor, bg


def hooked(model, ids, edit=None, pos=9):
    hooks = []
    if edit is not None:
        for L, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: (edit(L, a[0]),)))
    try:
        with torch.no_grad():
            return torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
    finally:
        for h in hooks:
            h.remove()


def contrast_of(model, ids, edit=None):
    return float(ns.log2_contrast(hooked(model, ids, edit)[9], [3], [5, 7]))


def test_zero_thresholds_are_the_model(tiny):
    model, nr, ids, _, bg = tiny
    rs = ns.Runs(nr, ids, 9, [3], [5, 7], {**bg, "thr": torch.zeros_like(bg["thr"])}, [(1, 4, 7)])
    assert rs.cR == pytest.approx(contrast_of(model, ids), abs=1e-4)
    assert rs.runs([{"weak": False}])[0]["contrast"] == pytest.approx(contrast_of(model, ids), abs=1e-4)


def test_runs_equal_hooks(tiny):
    model, nr, ids, _, bg = tiny
    thr, massive = bg["thr"], bg["massive"]
    units = [(0, 9, 3), (1, 5, 11), (2, 9, 30)]
    rs = ns.Runs(nr, ids, 9, [3], [5, 7], bg, units)
    weak = lambda L, h: h * (h.abs() >= thr[L])  # noqa: E731
    assert rs.cR == pytest.approx(contrast_of(model, ids, weak), abs=1e-4)
    keep = rs.keep

    def isolated(L, h):
        h = weak(L, h)
        return torch.where(~keep[L][None] & (h.abs() < massive[L]), torch.zeros(()), h)
    assert rs.runs([{"circuit": True}])[0]["contrast"] == pytest.approx(contrast_of(model, ids, isolated), abs=1e-4)
    assert bool(keep[:, 0].all())

    def removed(L, h):
        h = weak(L, h).clone()
        for l, p, j in units:
            if l == L:
                h[:, p, j] = 0
        return h
    r = rs.runs([{"aff": rs.zero(rs.all_cells())}])[0]
    assert r["contrast"] == pytest.approx(contrast_of(model, ids, removed), abs=1e-4)
    a, b = hooked(model, ids, weak)[rs.side], hooked(model, ids, removed)[rs.side]
    assert r["side"] == pytest.approx(float((a.exp() * (a - b)).sum(-1).mean() / math.log(2)), abs=1e-5)


def test_whole_neuron_units_cover_every_position(tiny):
    _, nr, ids, _, bg = tiny
    rs = ns.Runs(nr, ids, 9, [3], [5, 7], bg, [(1, 11)])
    assert rs.cells[(1, 11)] == [(1, p, 11) for p in range(10)]
    assert int(rs.keep[1, :, 11].sum()) == 10


def test_full_circuit_is_r_under_every_fill_and_swap(tiny):
    _, nr, ids, donor, bg = tiny
    units = [(L, j) for L in range(3) for j in range(40)]
    rs = ns.Runs(nr, ids, 9, [3], [5, 7], bg, units)
    rs.add_donor("d", donor)
    sw = [(L, p, j, 0.0, float(rs.donors["d"][L, p, j])) for L, p, j in rs.all_cells()]
    iso, patched, frozen, smr, smc = rs.runs([{"circuit": True}, {"circuit": True, "fill": "d"}, {"circuit": True, "fill": "clean"},
                                              {"aff": sw}, {"circuit": True, "fill": "clean", "aff": sw}])
    for r in (iso, patched, frozen):
        assert r["contrast"] == pytest.approx(rs.cR, abs=1e-4)
    assert smc["contrast"] == pytest.approx(smr["contrast"], abs=1e-4)
    # the donor's swap is its own R run's values: the run reproduces the donor's first T tokens
    assert torch.allclose(rs.capture(donor), rs.donors["d"])


def test_counterfactual_runs_equal_hooks(tiny):
    model, nr, ids, _, bg = tiny
    thr = bg["thr"]
    units = [(1, 5, 11), (2, 9, 30)]
    rs = ns.Runs(nr, ids, 9, [3], [5, 7], bg, units)
    x2 = list(ids)
    x2[4] = (x2[4] + 1) % 48
    sw = [(L, p, j, 0.0, float(rs.vals[L, p, j])) for L, p, j in rs.all_cells()]
    cf, pt = rs.runs([{"ids": x2}, {"ids": x2, "aff": sw}])
    weak = lambda L, h: h * (h.abs() >= thr[L])  # noqa: E731

    def patched(L, h):
        h = weak(L, h).clone()
        for l, p, j in units:
            if l == L:
                h[:, p, j] = rs.vals[L, p, j]
        return h
    assert cf["contrast"] == pytest.approx(contrast_of(model, x2, weak), abs=1e-4)
    assert pt["contrast"] == pytest.approx(contrast_of(model, x2, patched), abs=1e-4)


def test_margins_read_records():
    rec = {"contrast": {"R": 4.0}, "necessity": {"R": -0.5, "answer_top": False, "side": 0.01},
           "draws": {"sufficiency": [{"restored": 1.0}, {"restored": 0.9}], "patching_near": [{"c": 4.0}, {"c": 3.0}],
                     "patching_far": [{"c": 4.0}], "swaps": [{"dm": 1.0, "dc": 1.0}, {"dm": -2.0, "dc": -2.0}],
                     "steering": [{"dm": 0.01, "dc": 0.02}], "weights": [{"dm": 1.0, "dc": 0.0}, {"dm": 1.0, "dc": 0.0}],
                     "specificity": [{"kl": 0.01}, {"kl": 0.03}]}}
    m = ns.margins(rec)
    assert m["necessity"]["score"] == 4.5 and m["necessity"]["pass_old"] and m["necessity"]["se"] == 0.0
    assert m["side"]["pass_old"]
    assert m["sufficiency"]["score"] == pytest.approx(0.95) and m["sufficiency"]["pass_old"]
    assert m["patching_near"]["score"] == pytest.approx(0.875) and m["patching_near"]["se"] > 0
    assert m["swaps"]["score"] == 1.0
    assert m["steering"]["score"] is None and m["steering"]["pass_new"] is None
    assert m["weights"]["score"] == 0.0 and m["weights"]["pass_new"] is False
    assert m["specificity"]["score"] == pytest.approx(0.02) and m["specificity"]["pass_old"]
    rec["necessity"]["answer_top"] = True  # the builder's necessity: the answer must also leave the top
    assert ns.margins(rec)["necessity"]["pass_old"] is False
    rec["draws"]["specificity"] = [{"kl": 0.2}, {"kl": 0.3}]
    assert ns.margins(rec)["specificity"]["pass_new"] is False


@pytest.mark.parametrize("missing", ["sufficiency", "patching_far", "steering"])
def test_summary_requires_every_family_scored_and_passing(missing):
    rec = {"contrast": {"R": 4.0, "model": 4.0},
           "necessity": {"R": 0.0, "model": 0.0, "answer_top": False, "side": 0.0},
           "draws": {"specificity": [{"kl": 0.0}], "sufficiency": [{"restored": 1.0}],
                     "patching_near": [{"c": 4.0}], "patching_far": [{"c": 4.0}],
                     **{f: [{"dm": 1.0, "dc": 1.0}] for f in ("swaps", "steering", "weights")}},
           "simplicity": {k: 1 for k in ("units", "cells", "neurons", "positions", "layers", "edges")},
           "isolation": {"share": 1.0}}
    assert set(ns.summary([rec])["every_family"].values()) == {1}
    # Empty donor/counterfactual sets and sub-threshold effects are unscored, not passes.
    rec["draws"][missing] = [{"dm": 0.01, "dc": 0.01}] if missing == "steering" else []
    result = ns.summary([rec])
    assert result["families"][missing]["unscored"] == 1
    assert set(result["every_family"].values()) == {0}
    # Once measured, a genuinely failing family also blocks the aggregate.
    rec["draws"][missing] = {"sufficiency": [{"restored": 0.0}], "patching_far": [{"c": 0.0}],
                             "steering": [{"dm": 1.0, "dc": 0.0}]}[missing]
    assert set(ns.summary([rec])["every_family"].values()) == {0}


def test_text_kl(tiny):
    model, nr, ids, donor, bg = tiny
    thr = bg["thr"]
    rs = ns.Runs(nr, ids, 9, [3], [5, 7], bg, [(1, 11)])
    assert rs.text_kl(donor, []) == pytest.approx(0.0, abs=1e-6)
    gone = [(1, p, 11, 0.0, 0.0) for p in range(10)]
    weak = lambda L, h: h * (h.abs() >= thr[L])  # noqa: E731

    def removed(L, h):
        h = weak(L, h).clone()
        if L == 1:
            h[:, :, 11] = 0
        return h
    a, b = hooked(model, donor[:10], weak), hooked(model, donor[:10], removed)
    want = float((a.exp() * (a - b)).sum(-1).mean() / math.log(2))
    assert rs.text_kl(donor, gone) == pytest.approx(want, abs=1e-5)
