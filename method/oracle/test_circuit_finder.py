"""Checks of the patching-pair circuit finder (circuit_finder.py) on a tiny random Qwen3: patching the whole residual
input of the first layer gives the other run exactly, a block patch equals its units patched one by one, a unit frozen
at its value in the base run changes nothing, and find()'s record is consistent with its own measurements.

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


def test_frozen_at_base_is_no_change(tiny):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    a = pt.run("clean", [[("head", 1, (0, 3), 8, "clean"), ("neuron", 0, 5, 7, "clean")]])[0]
    b = pt.run("corrupt", [[("head", 1, (0, 3), 8, "corrupt")]])[0]
    assert abs(a) < 1e-6 and abs(b) < 1e-6


def test_find_record(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    monkeypatch.setattr(cf, "EPS", 0.05)
    rec = cf.find(pt, clean, corrupt)
    assert rec["size"] == len(rec["nodes"])
    for n in rec["nodes"]:
        assert n["pos"] >= rec["first_diff"] and n["pole"] in (-1, 1)
        assert (n["pos"] == pt.T - 1) == (n["stage"] == "readout")
    for stage, c in rec["cuts"].items():
        units = ([("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"] if n["stage"] == stage] if stage != "movers"
                 else [("head", r["layer"], r["head"], r["pos"]) for r in rec["routes"]])
        assert len(units) == c["size"]
        if units:
            lost, rest = pt.both([units])[0]
            assert abs(lost - c["lost"]) < 1e-5 and abs(rest - c["restored"]) < 1e-5
        if c["met"]:
            assert c["lost"] >= cf.PASS and c["restored"] >= cf.PASS
    assert rec["met"] == all(c["met"] for k, c in rec["cuts"].items() if k != "readout" or c["reachable"])
    for c in rec["cuts"].values():
        assert c["reachable"] == (c["ceiling"][0] >= cf.PASS and c["ceiling"][1] >= cf.PASS)


def test_prefix_cache_equals_full_runs(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    T, f = pt.T, pt.first
    sets = [[("neuron", 1, T - 1, (3, 5, 9))], [("head", 0, (1, 2), T - 1), ("neuron", 0, T - 1, 4)],
            [("res", 1, f)], [("neuron", 1, f + 1, (0, 7)), ("head", 1, 3, T - 1, "clean")],
            [("res", 0, 0), ("neuron", 0, T - 1, (1, 2))], [("neuron", 1, p, 5) for p in range(T)]]
    got = {}
    for on in (True, False):
        monkeypatch.setattr(cf, "PREFIX", on)
        pt._kv = {}
        got[on] = pt.both(sets) + pt.both(sets[:2])           # (a second call reuses the cache)
    for a, b in zip(got[True], got[False]):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
