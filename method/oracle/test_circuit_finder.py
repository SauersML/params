"""Checks of the patching-pair circuit finder (circuit_finder.py) on a tiny random Qwen3: patching the whole residual
input of the first layer gives the other run exactly, a block patch equals its units patched one by one, and find()'s
record is consistent with its own measurements.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_circuit_finder.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import circuit_finder as cf  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    pt = cf.Patcher(model, None, rows=8)
    g = torch.Generator().manual_seed(1)
    for _ in range(50):                     # a pair whose answers differ
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupt = list(clean)
        corrupt[4], corrupt[7] = (clean[4] + 7) % 64, (clean[7] + 11) % 64
        rec = pt.setup(clean, corrupt)
        if rec["answer"] != rec["corrupt_answer"] and rec["gap"] > 0.5:
            return pt, clean, corrupt
    pytest.skip("no pair with different answers")


def test_whole_residual_is_the_other_run(tiny):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    all0 = [("res", 0, p) for p in range(pt.T)]
    lost, rest = pt.both([all0])[0]
    assert abs(lost - 1) < 1e-4 and abs(rest - 1) < 1e-4
    assert all(abs(x) < 1e-6 for x in pt.both([[("res", 0, p) for p in range(pt.first)]])[0])   # before the change: equal


def test_block_equals_units(tiny):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    block = [("neuron", 1, 6, (3, 5, 9)), ("head", 0, (1, 2), 7)]
    units = [("neuron", 1, 6, 3), ("neuron", 1, 6, 5), ("neuron", 1, 6, 9), ("head", 0, 1, 7), ("head", 0, 2, 7)]
    a, b = pt.both([block, units])
    assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5


def test_find_record(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    monkeypatch.setattr(cf, "EPS", 0.05)
    rec = cf.find(pt, clean, corrupt)
    assert rec["size"] == len(rec["nodes"]) + len(rec["routes"])
    for n in rec["nodes"]:
        assert n["pos"] >= rec["first_diff"] and n["pole"] in (-1, 1)
    units = [("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"]] + [("head", r["layer"], r["head"], r["pos"]) for r in rec["routes"]]
    if units:
        lost, rest = pt.both([units])[0]
        assert abs(lost - rec["lost"]) < 1e-5 and abs(rest - rec["restored"]) < 1e-5
    if rec["met"]:
        assert rec["lost"] >= cf.PASS and rec["restored"] >= cf.PASS
