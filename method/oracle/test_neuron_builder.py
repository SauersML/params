"""Checks of the question-scoped neuron builder (neuron_builder.py) on a tiny random Qwen3: calibration's per-layer
reference stays within the tolerance and is the one the hooks run, the circuit lies among the strong non-massive units
before the answer, its recorded removal is the measured one, and its recorded sufficiency is the counterfactuals' patch
of its units and routes recomputed by hand; a run from a prefix's cache equals the full run.

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


def test_sufficiency_is_the_measured_patch(tiny, monkeypatch):
    nr, _, ids, _ = tiny
    monkeypatch.setattr(nb, "CANDIDATES", 32)
    monkeypatch.setattr(nb, "NEC_BITS", 0.01)   # (the random model's counterfactuals move it a few hundredths)
    monkeypatch.setattr(nb, "SUF_MEASURED", 16)
    pos = len(ids) - 1
    rec = nb.build(nr, ids, pos)
    if not rec["counterfactuals"] or rec["restored"] is None:
        pytest.skip("no counterfactual moved the contrast")
    qmin = min(c["pos"] for c in rec["counterfactuals"])
    for r in rec["routes"]:
        assert 0 <= r["layer"] < nr.L and 0 <= r["head"] < nr.NH and qmin <= r["pos"] <= pos
    assert {u["role"] for u in rec["nodes"]} <= {"necessity", "sufficiency", "both"}
    assert rec["size"] == len(rec["nodes"]) == rec["size_necessity"] + sum(u["role"] == "sufficiency" for u in rec["nodes"])
    cap = {}
    with torch.no_grad():
        lp = torch.log_softmax(nr.logits(ids, [pos], capture=cap)[0, 0], -1)
    H, O = [h[0] for h in cap["h"]], [o[0] for o in cap["o"]]
    alt = torch.tensor(rec["alternatives"])
    con = lambda l: float((l[rec["answer"]] - torch.logsumexp(l[alt], -1)) / math.log(2))  # noqa: E731
    assert abs(con(lp) - rec["contrast"]) < 1e-4
    got = []
    for c in rec["counterfactuals"]:
        x2 = list(ids)
        x2[c["pos"]] = c["token"]

        def edit(i, h):
            h = h.clone()
            for u in rec["nodes"]:
                if u["layer"] == i:
                    h[:, u["pos"], u["neuron"]] = H[i][u["pos"], u["neuron"]]
            return h

        def route(i, a):
            a = a.clone()
            for r in rec["routes"]:
                if r["layer"] == i:
                    sl = slice(r["head"] * nr.hd, (r["head"] + 1) * nr.hd)
                    a[:, r["pos"], sl] = O[i][r["pos"], sl]
            return a
        with torch.no_grad():
            l = torch.log_softmax(nr.logits(x2, [pos], edit=edit, route=route)[0, 0], -1)
        got.append((con(l) - (rec["contrast"] - c["drop"])) / c["drop"])
    assert abs(min(got) - rec["restored"]) < 1e-4
    if rec["met"]:
        assert rec["restored"] >= nb.SUFF_FRAC


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
