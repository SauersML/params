"""Checks of the question-scoped neuron builder (neuron_builder.py) on a tiny random Qwen3: calibration's per-layer
reference stays within the tolerance and is the one the hooks run, the circuit lies among the strong non-massive units
before the answer, and its recorded removal is the measured one.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_neuron_builder.py -q
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import neuron_builder as nb  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    nr = nb.Neurons(model, None)
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(0, 64, (16,), generator=g).tolist() for _ in range(4)]
    cal = nr.calibrate(seqs, taus=(0.01, 0.05, 0.2), kl_tol=0.5, prices=8, refine=2)
    return nr, cal, torch.randint(0, 64, (12,), generator=g).tolist(), seqs


def test_calibration_within_tolerance(tiny):
    nr, cal, _, seqs = tiny
    assert nr.thr.shape == (nr.L, nr.D) and len(cal["tau"]) == nr.L and cal["kl"] <= 0.5
    kl, st = nr.weak_runs(seqs, torch.tensor([cal["tau"]]))
    assert abs(float(kl[0]) - cal["kl"]) < 1e-6 and abs(float(st[0].sum()) - cal["strong_per_token"]) < 1e-6


def test_reference_is_the_calibrated_one(tiny):
    nr, cal, _, seqs = tiny
    tot, n = 0.0, 0
    for ids in seqs:
        pos = list(range(len(ids)))
        thr, nr.thr = nr.thr, None
        with torch.no_grad():
            lp0 = torch.log_softmax(nr.logits(ids, pos)[0], -1)
            nr.thr = thr
            lp1 = torch.log_softmax(nr.logits(ids, pos)[0], -1)
        tot, n = tot + float((lp0.exp() * (lp0 - lp1)).sum(-1).sum()) / math.log(2), n + len(ids)
    assert abs(tot / n - cal["kl"]) < 1e-5


def test_circuit_units_and_removal(tiny, monkeypatch):
    nr, _, ids, _ = tiny
    monkeypatch.setattr(nb, "CANDIDATES", 32)
    monkeypatch.setattr(nb, "NEC_BITS", 0.1)
    pos = len(ids) - 1
    rec = nb.build(nr, ids, pos)
    for u in rec["nodes"]:
        assert 1 <= u["pos"] <= pos and 0 <= u["layer"] < nr.L and 0 <= u["neuron"] < nr.D and u["pole"] in (-1, 1)
        assert u["act"] != 0 and abs(u["act"]) < float(nr.massive[u["layer"]])
    if rec["met"]:
        r = rec["removed"]
        assert not r["top"] and r["contrast"] <= rec["contrast"] - nb.NEC_BITS + 1e-6 and r["side"] <= nb.SIDE_MAX


def test_pole_edit():
    h = torch.tensor([[[1., -2., 3.], [-1., 2., -3.]]]).expand(2, -1, -1).clone()   # [B 2, T 2, D 3]
    e = nb.pole_edit([[(0, 0, 1), (0, 1, -1)], [(0, 2, 0)]])
    out = e(0, h)
    assert out[0].tolist() == [[0., 0., 3.], [-1., 2., -3.]] and out[1].tolist() == [[1., -2., 0.], [-1., 2., 0.]]
    assert torch.equal(e(1, h), h)
