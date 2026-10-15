"""Checks of the question-scoped neuron builder (neuron_builder.py) on a tiny random Qwen3: calibration picks a tau
within the tolerance, the circuit lies among the strong non-massive units before the answer, and its recorded removal is
the measured one.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_neuron_builder.py -q
"""
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
    cal = nr.calibrate(seqs, taus=(0.01, 0.05, 0.2), kl_tol=0.5)
    return nr, cal, torch.randint(0, 64, (12,), generator=g).tolist()


def test_calibration_within_tolerance(tiny):
    nr, cal, _ = tiny
    ok = [r for r in cal["rows"] if r["kl"] <= 0.5]
    assert cal["tau"] == (max(r["tau"] for r in ok) if ok else None)
    assert nr.thr is None or nr.thr.shape == (nr.L, nr.D)


def test_circuit_units_and_removal(tiny, monkeypatch):
    nr, _, ids = tiny
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
