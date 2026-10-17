"""Checks of circuit_amortize.py on a tiny random Qwen3: each path feature is the slope of the lost (or restored) share in
one unit's mask at the cut's point m (central differences of the share with that unit moved m +- h and the rest of its
cut at m, through the model's own forward and nothing of path_features' batching), and find_batch_amortized's records
report the shares their circuits measure.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_circuit_amortize.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import circuit_amortize as ca  # noqa: E402
import circuit_finder as cf  # noqa: E402
import circuit_masks as cm  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    pts = []
    for _ in range(200):                    # pairs whose answers differ
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupt = list(clean)
        corrupt[4], corrupt[7] = (clean[4] + 7) % 64, (clean[7] + 11) % 64
        pt = cf.Patcher(model, None, rows=8)
        rec = pt.setup(clean, corrupt)
        if rec["answer"] != rec["corrupt_answer"] and rec["gap"] > 0.5:
            pts.append((pt, clean, corrupt))
        if len(pts) == 2:
            return model, pts
    pytest.skip("no pairs with different answers")


def _share(pt, base: str, spec: dict, m: float, unit: tuple, mu: float) -> float:
    """The lost (base clean) or restored (base corrupt) share with the cut's units moved m of the way to the other run and
    unit (layer, position, index) moved mu, through one plain forward."""
    other = "corrupt" if base == "clean" else "clean"
    key = "head" if "head" in spec["kinds"] else "neu"
    i0, p0, k0 = unit
    H, hd = pt.H, pt.hd

    def edit(i, h):
        o = pt.act[other][key][i].to(h.dtype)
        w = torch.zeros(h.shape[1], H if key == "head" else h.shape[-1], dtype=h.dtype)
        w[spec["positions"]] = m
        if i == i0:
            w[p0, k0] = mu
        if key == "head":
            w = w.repeat_interleave(hd, -1)
        return h + w[None] * (o[None] - h)
    hooks = {"head": edit} if key == "head" else {"neu": edit}
    c = float(pt._contrast(pt._last(pt.ids[base], 1, **hooks))[0])
    return (pt.c["clean"] - c) / pt.gap if base == "clean" else (c - pt.c["corrupt"]) / pt.gap


def test_slopes_are_the_shares_derivatives(tiny):
    _, pairs = tiny
    pt, clean, corrupt = pairs[0]
    pt.setup(clean, corrupt)
    got = {}
    ca.path_features([pt], lambda j, f: got.update({n: {k: v.clone() for k, v in d.items()} for n, d in f.items()}), relp=False)
    h = 1e-3
    for n, spec in cm.cuts_of(pt).items():
        L, P, W = got[n]["lost0"].shape
        flat = (got[n]["lost0"].abs() + got[n]["rest0"].abs()).reshape(-1)
        for f in flat.topk(3).indices.tolist():     # (units with a slope well above float error)
            i, rem = divmod(f, P * W)
            pi, k = divmod(rem, W)
            unit = (i, spec["positions"][pi], k)
            for m, tag in ((0.0, "0"), (0.5, "5"), (1.0, "1")):
                for base, feat in (("clean", "lost"), ("corrupt", "rest")):
                    num = (_share(pt, base, spec, m, unit, m + h) - _share(pt, base, spec, m, unit, m - h)) / (2 * h)
                    ana = float(got[n][feat + tag][i, pi, k])
                    assert abs(num - ana) <= 2e-3 + 2e-2 * abs(num), (n, unit, m, base, num, ana)


def test_two_pairs_batched_as_alone(tiny):
    """A pair's features are the same with another pair in the same forwards (rows independent)."""
    _, pairs = tiny
    (a, ca_, ka), (b, cb, kb) = pairs
    a.setup(ca_, ka)
    b.setup(cb, kb)
    alone, both = {}, {}
    ca.path_features([a], lambda j, f: alone.update({n: {k: v.clone() for k, v in d.items()} for n, d in f.items()}), relp=False)
    ca.path_features([a, b], lambda j, f: both.update({n: {k: v.clone() for k, v in d.items()} for n, d in f.items()}) if j == 0
                     else None, relp=False)
    for n in alone:
        for k in alone[n]:
            assert torch.allclose(alone[n][k], both[n][k], atol=1e-5, rtol=1e-4), (n, k)


def test_records_report_their_measured_shares(tiny):
    model, pairs = tiny
    pts = [cf.Patcher(model, None, rows=8) for _ in pairs]
    recs = ca.find_batch_amortized(pts, [(c, k, None) for _, c, k in pairs], ca.scorer("grad"), arm="grad", relp=False)
    for pt, rec, (_, c, k) in zip(pts, recs, pairs):
        pt.setup(c, k)
        for n, cut in rec["cuts"].items():
            units = ([("neuron", x["layer"], x["pos"], x["neuron"]) for x in rec["nodes"] if x["stage"] == n] if n != "movers"
                     else [("head", x["layer"], x["head"], x["pos"]) for x in rec["routes"]])
            if not cut["met"]:
                continue
            assert len(units) == cut["size"]
            lost, rest = pt.both([cm._blocks(units)])[0]
            assert abs(lost - cut["lost"]) < 1e-4 and abs(rest - cut["restored"]) < 1e-4
            assert lost >= cf.PASS and rest >= cf.PASS
        assert set(rec["amortize"]["stage"]) <= set(rec["cuts"])


def test_arm_constants_take_their_defaults_types():
    assert ca._value("1", cm.INIT) == 1.0 and isinstance(ca._value("1", cm.INIT), float)
    assert ca._value("16", 64) == 16 and isinstance(ca._value("16", 64), int)
    assert ca._value("readout", ca.MASK_CUTS) == "readout"
