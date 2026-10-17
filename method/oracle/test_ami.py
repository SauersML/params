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


def test_scrub_from_the_same_prompt_is_the_circuit(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    x = {"ids": ids, "span": (4, 6)}
    units = [("neuron", 0, 4, 3), ("head", 0, 1, 5), ("neuron", 1, 5, 7), ("head", 1, 2, 11), ("neuron", 1, 11, 2), ("neuron", 0, 1, 9)]
    cls = ami.classes({}, units, x["span"])
    assert cls[("neuron", 0, 1, 9)] == "pre" and cls[("neuron", 1, 11, 2)] == "key"
    same = {"ids": list(ids), "span": (4, 6)}                # (an equal prompt: scrubbing from it changes nothing)
    lp, done, left = ami.scrub(abl, x, units, cls, {"key": [same]}, 2, 0)
    want = abl.logp([{"ids": ids, "keep": units}])[0]
    assert len(done) == 5 and not left
    assert torch.allclose(lp[0], want, atol=1e-5) and torch.allclose(lp[1], want, atol=1e-5)
    other = list(ids)
    other[4], other[5] = (ids[4] + 1) % 64, (ids[5] + 1) % 64   # (another key: scrubbing from it moves the output)
    lp2, _, _ = ami.scrub(abl, x, units, cls, {"key": [{"ids": other, "span": (4, 6)}]}, 1, 0)
    assert not torch.allclose(lp2[0], want, atol=1e-4)
    assert ami.role_map(x, {"ids": other, "span": (4, 6)}) == {p: p for p in range(12) if p not in (4, 5)}


def test_invalid_units():
    us = [("neuron", 0, 3, 1), ("neuron", 2, 3, 48), ("head", 1, 4, 5), ("head", 1, 2, 12), ("hchan", 0, 0, 31)]
    assert ami.invalid(us, 2, 48, 4, 32, 12) == [("neuron", 2, 3, 48), ("head", 1, 4, 5), ("head", 1, 2, 12)]


def test_mask_search_shared_mask(tiny, monkeypatch):
    model, docs, means = tiny
    monkeypatch.setattr(ami, "COVER_STEPS", 4)
    mk = ami.Masked(model, means)
    a, b = docs[0][:10], list(docs[0][:10])
    b[4] = (b[4] + 1) % 64                                    # (a pair: one changed token)
    x = torch.tensor([a, b])
    pos = torch.tensor([9, 9])
    with torch.no_grad():
        lpf = mk.logp(x, None, None, pos)
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, torch.ones(1, 1, 10, 1))
    c = ms["curves"][0]
    assert c[-1][0] == ms["units_total"][0] == 10 * mk.L * (mk.D + mk.H) and c[-1][1] < 1e-6
    k = ms["at"][0]["0.1"]                                    # (the bound's set is within it on both prompts)
    z = ms["thr"](0, k)
    mn = (ms["th"][0][0] >= z).float()[None].expand(2, -1, -1, -1)
    mh = (ms["th"][1][0] >= z).float()[None].expand(2, -1, -1, -1)
    with torch.no_grad():
        assert float(ami._kl(lpf, mk.logp(x, mn, mh, pos)).max()) <= 0.1 + 1e-6


def test_r5_true_simulator_is_frozen(tiny, monkeypatch):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    units = [("neuron", 0, 4, 3), ("head", 0, 1, 5), ("neuron", 1, 11, 2)]
    _, got = abl.run([{"ids": ids, "keep": None, "grab": units}])
    true = dict(zip(units, got[0]))
    monkeypatch.setattr(ami, "heldout", lambda *a, **k: [])
    monkeypatch.setattr(ami.answer_contrast, "token_sets_many", lambda core, tok, items: [{"answer": [int(it[1].argmax())], "answer_word": "w"} for it in items])
    rec = {"id": "t", "ids": ids, "neurons": [[0, 4, 3], [1, 11, 2]], "heads": [[0, 5, 1]]}
    res = ami.r5(abl, None, rec, {}, [], lambda u, x: true[u])
    p = res["prompts"][0]
    assert abs(p["simulated"]["kl"] - p["frozen"]["kl"]) < 1e-6 and res["labelled_share"] == 1.0
    res0 = ami.r5(abl, None, rec, {}, [], lambda u, x: None)       # (no labels: the empty circuit)
    assert abs(res0["prompts"][0]["simulated"]["kl"] - res0["prompts"][0]["empty"]["kl"]) < 1e-6


def test_dense_rows_equal_unit_rows(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    units = [("neuron", 0, 4, 3), ("head", 0, 1, 5), ("neuron", 1, 11, 2), ("head", 1, 3, 11), ("hchan", 1, 7, 9)]
    M = ami.dense(units, abl.L, 12, abl.D, abl.H, abl.hd)
    assert int(M["neuron"].sum()) == 2 and int(M["head"].sum()) == 2 * abl.hd + 1
    a, b = abl.logp([{"ids": ids, "keep": units}, {"ids": ids, "mask": M}])
    assert torch.allclose(a, b, atol=1e-6)
    cap = abl.capture(ids)
    _, got = abl.run([{"ids": ids, "keep": None, "grab": units}])
    c, d = abl.logp([{"ids": ids, "keep": [], "over": list(zip(units, got[0]))}, {"ids": ids, "mask": M, "values": cap}])
    assert torch.allclose(c, d, atol=1e-6)                   # (frozen: dense values equal the unit overrides)
    r = ami.redraw_dense(M, 0)
    assert all(torch.equal(r[k].sum(-1), M[k].sum(-1)) for k in M) and not any((r[k] & M[k]).any() for k in M)
