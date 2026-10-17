"""Checks of ami.py on a tiny random Qwen3 (circuit_finder's test model): collect()'s means equal a direct average of
captured activations per position bucket; a row keeping every unit equals the full model; a row keeping none gives the
same output whatever the prompt's non-final tokens; override values land on their units; edges() and redraw() on small
cases.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_ami.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ami  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    docs = [torch.randint(0, 64, (n,), generator=g).tolist() for n in (40, 23, 17, 9, 40)]
    acc = ami.collect(model, docs, batch_tokens=64)
    path = tmp_path_factory.mktemp("ami") / "means.pt"
    torch.save({**acc, "model": "tiny"}, path)
    return model, docs, ami.Means(path)


def _capture(model, ids):
    cap = {"neuron": [], "head": []}
    hs = []
    for layer in model.model.layers:
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap["neuron"].append(a[0][0].clone())))
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a: cap["head"].append(a[0][0].clone())))
    try:
        model.model(input_ids=torch.tensor([ids]))
    finally:
        for h in hs:
            h.remove()
    return {k: torch.stack(v) for k, v in cap.items()}      # [L, T, width]


def test_means_equal_direct_average(tiny):
    model, docs, means = tiny
    caps = [_capture(model, d) for d in docs]
    for kind in ("neuron", "head"):
        for b in range(ami.NB):
            xs = [c[kind][:, p] for c, d in zip(caps, docs) for p in range(len(d)) if int(ami.bucket_of(torch.tensor([p]))[0]) == b]
            if not xs:
                continue
            want = torch.stack(xs).mean(0)                   # [L, width]
            assert torch.allclose(means.mean[kind][b], want, atol=1e-5), (kind, b)


def test_keep_all_is_full_and_keep_none_ignores_context(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    every = [("neuron", i, p, k) for i in range(abl.L) for p in range(12) for k in range(abl.D)]
    every += [("head", i, h, p) for i in range(abl.L) for p in range(12) for h in range(abl.H)]
    lp = abl.logp([{"ids": ids, "keep": None}, {"ids": ids, "keep": every}])
    assert torch.allclose(lp[0], lp[1], atol=1e-5)
    other = list(ids)
    other[:11] = docs[1][:11]                                # (the same last token, every other token changed)
    lp2 = abl.logp([{"ids": ids, "keep": []}, {"ids": other, "keep": []}])
    assert torch.allclose(lp2[0], lp2[1], atol=1e-5)         # (no unit live: only the last token's embedding reaches the output)
    assert not torch.allclose(lp[0], lp2[0], atol=1e-3)


def test_override_and_live_unit(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    u = ("neuron", 1, 11, 5)
    vals = {}                                                # (the unit's value in the ablated run, everything before it ablated)
    def grab(m, x):
        vals.setdefault("x", x[0][0, 11, 5].clone())
    h = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(grab)
    try:
        abl.logp([{"ids": ids, "keep": [u]}])
    finally:
        h.remove()
    a, b, c = abl.logp([{"ids": ids, "keep": [u]}, {"ids": ids, "keep": [], "over": [(u, vals["x"])]}, {"ids": ids, "keep": []}])
    assert torch.allclose(a, b, atol=1e-5)                   # (live = its ablated-run value written over the background)
    d = abl.logp([{"ids": ids, "keep": [], "over": [(u, vals["x"] + 5.0)]}])[0]
    assert not torch.allclose(d, c, atol=1e-4)               # (an override reaches the output)


def test_edges_and_redraw():
    us = [("neuron", 0, 3, 1), ("neuron", 2, 3, 7), ("head", 1, 2, 5), ("neuron", 1, 5, 0)]
    # n(0,3)->n(2,3); n(0,3)->h(1,@5); n(0,3)->n... no; h(1,@5)->n(1,5) (same layer, head first)
    assert ami.edges(us) == 3
    r = ami.redraw(us, 48, 4, 32, 0)
    assert [ami._lp(x) for x in r] == [ami._lp(x) for x in us]
    assert not set(r) & set(us)


def test_coverage_curve(tiny, monkeypatch):
    model, docs, means = tiny
    monkeypatch.setattr(ami, "COVER_STEPS", 4)
    mk = ami.Masked(model, means)
    ids = docs[0][:10]
    x = torch.tensor([ids] * 9)
    with torch.no_grad():
        lp_full = mk.logp(x, None, None, torch.arange(9))
        every = mk.logp(x, torch.ones(9, mk.L, 10, mk.D), torch.ones(9, mk.L, 10, mk.H), torch.arange(9))
    assert torch.allclose(lp_full, every, atol=1e-5)          # (every unit live: the full model)
    out = ami.cover_batch(mk, ids, [3, 6], lp_full)
    for o in out:
        assert o["curve"][-1][0] == o["units_total"] and o["curve"][-1][1] < 1e-6   # (the whole order: KL 0)
        assert o["curve"][0][0] == 0 and abs(o["curve"][0][1] - o["kl_empty"]) < 1e-9
        ks = [o["k"][str(b)] for b in ami.COVER_BOUNDS]        # (stricter bounds need at least as many units)
        assert all(a is not None and b is not None and a <= b for a, b in zip(ks, ks[1:]))
