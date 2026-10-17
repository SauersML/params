"""Checks of readers.py on a tiny random Qwen3: the first-order estimates' local changes equal the full run's with the same
edit (the closed form for a head's key and value at a position, its query there, a neuron's MLP input), the direct write
reads the final stream, and every reader's effect is its own full run.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_readers.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import head_routes  # noqa: E402
import readers  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    ids = torch.randint(0, 48, (10,), generator=torch.Generator().manual_seed(1)).tolist()
    bad = list(ids)
    bad[5] = (bad[5] + 7) % 48
    rn = head_routes.Runner(model)
    rn.answer, rn.alternatives = [3], [5, 7]
    return model, ids, bad, rn


def test_grads_are_the_contrast_gradient(tiny):
    _, ids, _, rn = tiny
    g = rn.grads(ids)
    st = rn.run(ids, [{}], capture=True)["states"]
    L, p, j, eps = 1, 7, 5, 1e-3
    up = rn.run(ids, [{"neuron": [(L, p, j, float(st["h"][L][p, j]) + eps)]}])["contrast"][0]
    dn = rn.run(ids, [{"neuron": [(L, p, j, float(st["h"][L][p, j]) - eps)]}])["contrast"][0]
    assert (up - dn) / (2 * eps) == pytest.approx(float(g["h"][L][p, j]), rel=2e-2, abs=1e-4)


def test_estimates_are_the_full_runs_changes(tiny):
    _, ids, bad, rn = tiny
    st = rn.run(ids, [{}], capture=True)["states"]
    sb = rn.run(bad, [{}], capture=True)["states"]
    units = [("neuron", 0, 5, 3), ("head", 0, 2, 6)]
    W = readers._writes(rn, st, sb, units)
    gen = torch.Generator().manual_seed(2)
    g = {"h": [torch.randn(st["h"][L].shape, generator=gen) for L in range(rn.L)],
         "z": [torch.randn(st["z"][L].shape, generator=gen) for L in range(rn.L)]}
    cands, store = readers._candidates(rn, st, g, units, W)
    for i, u in enumerate(units):
        kinds = {(r[1], r[4]) for r in cands[i]}
        assert kinds == {("neuron", "mlp"), ("head", "key_value"), ("head", "query")}
        for r in [r for kind in kinds for r in [r for r in cands[i] if (r[1], r[4]) == kind][:4]]:
            s2 = rn.run(ids, [readers._edit(r, store, W)], capture=True)["states"]
            L = r[2]
            if r[1] == "neuron":
                want = float(((s2["h"][L] - st["h"][L]) * g["h"][L]).sum())
            else:
                want = float(((s2["z"][L] - st["z"][L])[:, r[3]] * g["z"][L][:, r[3]]).sum())
            assert r[0] == pytest.approx(want, rel=1e-3, abs=1e-5), r[1:5]


def test_readers_record(tiny):
    _, ids, bad, rn = tiny
    units = [("neuron", 0, 5, 3), ("neuron", 1, 9, 8), ("head", 1, 0, 9)]
    out = readers.readers(rn, ids, units, corrupt=bad, top=4, circuit=set(units))
    c, cb = rn.run(ids, [{}])["contrast"][0], rn.run(bad, [{}])["contrast"][0]
    st = rn.run(ids, [{}], capture=True)["states"]
    assert readers._contrast(rn, readers._final(rn, st, 9)[None])[0] == pytest.approx(c, abs=1e-4)
    for o, u in zip(out, units):
        assert len(o["readers"]) == 4 and set(o) >= {"total", "together", "direct", "first_order_covered"}
        assert (o["direct"] is None) == (u[2 if u[0] == "neuron" else 3] != 9)
        r = o["readers"][0]
        assert r["share"] == pytest.approx(r["bits"] / (c - cb), rel=1e-6) and "in_circuit" in r
    sh = out[0]["total"]["share"]
    sb = rn.run(bad, [{}], capture=True)["states"]
    want = (c - rn.run(ids, [{"neuron": [(0, 5, 3, float(sb["h"][0][5, 3]))]}])["contrast"][0]) / (c - cb)
    assert sh == pytest.approx(want, abs=1e-6)
