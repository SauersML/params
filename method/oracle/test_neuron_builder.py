"""Checks of neuron_builder.py (the weak-zeroed reference and neuron edits) on a tiny random Qwen3: calibration's per-layer
reference stays within the tolerance and is the one the hooks run, pole_edit zeroes the poles it names, a run from a
prefix's cache equals the full run, and the fixed mask is each input's own.

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
    kl, st, agree = nr.weak_runs(seqs, torch.tensor([cal["tau"]]))
    assert 0 <= float(agree[0]) <= 1
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


def test_pole_edit():
    h = torch.tensor([[[1., -2., 3.], [-1., 2., -3.]]]).expand(2, -1, -1).clone()   # [B 2, T 2, D 3]
    e = nb.pole_edit([[(0, 0, 1), (0, 1, -1)], [(0, 2, 0)]])
    out = e(0, h)
    assert out[0].tolist() == [[0., 0., 3.], [-1., 2., -3.]] and out[1].tolist() == [[1., -2., 0.], [-1., 2., 0.]]
    assert torch.equal(e(1, h), h)


def test_prefix_cache_equals_the_full_run(tiny):
    nr, _, ids, _ = tiny
    T, s, B = len(ids), 5, 4
    g = torch.Generator().manual_seed(3)
    mk = torch.rand(B, T, nr.D, generator=g) < 0.3
    mk[:, :s] = False                                        # edits at positions s and later only
    rm = torch.rand(B, T, nr.NH, generator=g) < 0.5
    rm[:, :s] = False
    edit = lambda i, h: torch.where(mk, torch.zeros_like(h), h)  # noqa: E731
    route = lambda i, a: torch.where(rm.repeat_interleave(nr.hd, -1), torch.zeros_like(a), a)  # noqa: E731
    with torch.no_grad():
        full = nr.logits(ids, [s, T - 1], edit=edit, route=route, B=B)
        part = nr.logits(ids, [s, T - 1], edit=edit, route=route, B=B, cache=nr.prefix(ids, s, B=B))
        again = nr.logits(ids, [s, T - 1], edit=edit, route=route, B=B, cache=nr.prefix(ids, s, B=B))
    assert torch.allclose(full, part, atol=1e-4) and torch.equal(part, again)


def test_fixed_mask_reference(tiny):
    nr, _, ids, _ = tiny
    T = len(ids)
    with torch.no_grad():
        ref = nr.logits(ids, [T - 1], B=2)
        nr.fixed = True
        try:
            same = nr.logits(ids, [T - 1], B=2)                 # the unedited run: the same reference values
            g = torch.Generator().manual_seed(5)
            noise = [1 + 1e-6 * torch.randn(T, nr.D, generator=g) for _ in range(nr.L)]
            moved = nr.logits(ids, [T - 1], B=2, edit=lambda i, h: h * noise[i])
            other = torch.randint(0, 64, (T,), generator=g).tolist()
            both = nr.logits(torch.tensor([ids, other]), [T - 1])   # rows with their own masks
            alone = nr.logits(other, [T - 1])
        finally:
            nr.fixed = False
    assert torch.allclose(ref, same, atol=1e-5) and float((moved - same).abs().max()) < 1e-3
    assert torch.allclose(both[1], alone[0], atol=1e-5) and torch.allclose(both[0], same[0], atol=1e-5)
