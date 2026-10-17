"""Checks of circuit_masks' multi-source options on a tiny random Qwen3 (circuit_finder's test model): per-pair hinges
equal the mean-share hinges when a cut has one pair, ALLCUTS puts every cut on its partners, WARM's start lies in
[WARM_MIN, INIT] with the best-attributed unit at INIT, and find() with all three gives a record whose cuts carry every
pair's shares.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_circuit_masks.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import circuit_finder as cf  # noqa: E402
import circuit_masks as cm  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    """A tiny model and a clean prompt with three corrupted ones (two changed tokens each) whose answers differ."""
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    pt = cf.Patcher(model, None, rows=8)
    for _ in range(200):
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupts = []
        for d in range(1, 30):
            corrupt = list(clean)
            corrupt[4], corrupt[7] = (clean[4] + 3 * d) % 64, (clean[7] + 5 * d) % 64
            rec = pt.setup(clean, corrupt)
            if rec["valid"] and rec["gap"] >= cf.MIN_GAP and not rec["same_answer"]:
                corrupts.append(corrupt)
            if len(corrupts) == 3:
                return model, clean, corrupts
    pytest.skip("no clean prompt with three valid corrupted ones")


def test_pairwise_equals_mean_hinge_with_one_pair(tiny, monkeypatch):
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    spec = cm.cuts_of(pt)["readout"]
    a = cm.optimize(pt, [spec], steps=3)[0]["theta"]["neuron"]
    monkeypatch.setattr(cm, "PAIRWISE", 1)
    b = cm.optimize(pt, [spec], steps=3)[0]["theta"]["neuron"]
    assert torch.allclose(a, b, atol=1e-6)


def test_allcuts_and_warm(tiny, monkeypatch):
    model, clean, corrupts = tiny
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    for p, c in zip(pts, corrupts):
        p.setup(clean, c)
    st = cm._before(pts[0], "mask", partners=pts[1:])
    assert isinstance(st["meas"]["source"], cm.Mean) and all(st["meas"][n] is pts[0] for n in st["names"] if n != "source")
    assert st["warm"] == {}
    monkeypatch.setattr(cm, "ALLCUTS", 1)
    monkeypatch.setattr(cm, "WARM", 0.5)
    st = cm._before(pts[0], "mask", partners=pts[1:])
    assert all(isinstance(st["meas"][n], cm.Mean) for n in st["names"])
    for n in st["masked"]:
        th = st["warm"][n]
        assert set(th) == set(st["specs"][n]["kinds"])
        top = max(float(t.max()) for t in th.values())
        assert abs(top - cm.INIT) < 1e-5 and min(float(t.min()) for t in th.values()) >= cm.WARM_MIN - 1e-6
        for k, t in th.items():
            assert t.shape == (pts[0].L, len(st["specs"][n]["positions"]), pts[0].H if k == "head" else pts[0].D)


def test_find_multi_source(tiny, monkeypatch):
    model, clean, corrupts = tiny
    for k, v in (("ALLCUTS", 1), ("PAIRWISE", 1), ("WARM", 0.5)):
        monkeypatch.setattr(cm, k, v)
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    rec = cm.find(pts[0], clean, corrupts[0], steps=5, partners=list(zip(pts[1:], corrupts[1:])))
    assert set(rec["partners"]) == set(rec["cuts"])
    for n, p in rec["partners"].items():
        assert p["pairs"] == 3 and len(p["shares"]) in (0, 3)
    for n, c in rec["cuts"].items():
        if c["met"]:
            assert c["lost"] >= cf.PASS and c["restored"] >= cf.PASS
