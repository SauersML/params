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


def test_dense_scrub_equals_list_scrub(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    x = {"ids": ids, "span": (4, 6)}
    other = list(ids)
    other[4], other[5] = (ids[4] + 1) % 64, (ids[5] + 1) % 64
    y = {"ids": other, "span": (4, 6)}                       # (one pool prompt: every draw takes the same one)
    units = [("neuron", 0, 4, 3), ("head", 0, 1, 7), ("neuron", 1, 7, 7), ("head", 1, 2, 11), ("neuron", 1, 11, 2), ("neuron", 0, 1, 9)]
    cls = ami.classes({}, units, x["span"])
    lp, done, left = ami.scrub(abl, x, units, cls, {"key": [y]}, 2, 0)
    M = ami.dense(units, abl.L, 12, abl.D, abl.H, abl.hd)
    K = {k: v.clone() for k, v in M.items()}
    for k in K:
        K[k][:, :4] = False                                  # (the units before the key: not scrubbed)
    lp2, n_done, n_left = ami.scrub_dense(abl, x, M, K, [y], 2, 0)
    roles = ami.role_map(x, y)                               # (the key's own tokens differ: positions 4 and 5 have no role)
    assert n_done == sum(1 for u in units if ami._lp(u)[2] >= 4 and ami._lp(u)[2] in roles)
    assert len(done) == sum(1 for u in units if ami._lp(u)[2] >= 4 and ami._lp(u)[2] in roles)
    assert torch.allclose(lp, lp2, atol=1e-5)
    lp3, _, _ = ami.scrub_dense(abl, x, M, K, [{"ids": list(ids), "span": (4, 6)}], 1, 0)
    assert torch.allclose(lp3[0], abl.logp([{"ids": ids, "mask": M}])[0], atol=1e-5)   # (from an equal prompt: the circuit)
    z = {"ids": [(t + 3) % 64 if 4 <= j < 6 else t for j, t in enumerate(ids)], "span": (4, 6)}
    lp4, _, _ = ami.scrub_dense(abl, x, M, K, [y, z], 3, 1)  # (two pool prompts: per-unit draws)
    assert lp4.shape[0] == 3 and torch.isfinite(lp4).all()


def test_mask_search_head_channels(tiny, monkeypatch):
    model, docs, means = tiny
    monkeypatch.setattr(ami, "COVER_STEPS", 3)
    monkeypatch.setattr(ami, "COVER_HCHAN", 1)
    mk = ami.Masked(model, means)
    x = torch.tensor([docs[0][:8]])
    pos = torch.tensor([7])
    with torch.no_grad():
        lpf = mk.logp(x, None, None, pos)
    ms = ami.mask_search(mk, x, [0], pos, lpf, torch.ones(1, 1, 8, 1))
    assert ms["units_total"][0] == 8 * mk.L * (mk.D + mk.H * mk.hd)
    assert ms["curves"][0][-1][1] < 1e-6


def test_dense_scrub_pool_of_another_length(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    ids = docs[0][:12]
    x = {"ids": ids, "span": (4, 6)}
    longer = ids[:4] + [(ids[4] + 1) % 64, (ids[5] + 1) % 64, 7, 9] + ids[6:]   # (two more tokens after the key)
    y = {"ids": longer, "span": (4, 6)}
    units = [("neuron", 0, 4, 3), ("head", 0, 1, 7), ("neuron", 1, 9, 7), ("neuron", 1, 11, 2)]
    M = ami.dense(units, abl.L, 12, abl.D, abl.H, abl.hd)
    K = {k: v.clone() for k, v in M.items()}
    lp, n_done, _ = ami.scrub_dense(abl, x, M, K, [y], 2, 0)
    assert torch.isfinite(lp).all() and n_done == 3       # (positions 7, 9, 11 have roles in the tail; the key's 4 differs)


def test_frame_table(tiny, tmp_path):
    model, docs, means = tiny
    L, D = 2, 48
    HD = 4 * 8
    f = {"neuron_sum": torch.arange(12, dtype=torch.float64)[:, None, None].expand(12, L, D).clone() * 3,
         "head_sum": torch.zeros(12, L, HD, dtype=torch.float64), "count": torch.full((12,), 3.0, dtype=torch.float64)}
    torch.save(f, tmp_path / "frame.pt")
    path = tmp_path / "means.pt"
    acc = ami.collect(model, docs, batch_tokens=64)
    torch.save({**acc, "model": "tiny"}, path)
    m = ami.Means(path, frame=tmp_path / "frame.pt")
    T = 20
    t = m.table("neuron", 1, T, "cpu", torch.float32, "frame")
    b = m.table("neuron", 1, T, "cpu", torch.float32, "bucket")
    roles = ami.frame_roles(T)
    assert (roles[:3] == [0, 1, 2]).all() and (roles[-9:] == list(range(3, 12))).all() and (roles[3:-9] == -1).all()
    assert torch.allclose(t[:3, 0], torch.tensor([0.0, 1.0, 2.0])) and torch.allclose(t[-1, 0], torch.tensor(11.0))
    assert torch.allclose(t[3:-9], b[3:-9])
    assert ami.is_framed(ami.FRAME_PRE + [5, 6] + ami.FRAME_SUF) and not ami.is_framed([5] + ami.FRAME_SUF)


def test_mask_search_factored(tiny, monkeypatch):
    """COVER_FACTOR: an item's mask is its backbone or each row's own part; the curve counts the backbone and every row's
    part (units_total = (1 + rows) x positions x layers x units), the full curve is the full model, and the bound's set
    (the backbone with a row's own part) is within the bound on each row."""
    model, docs, means = tiny
    monkeypatch.setattr(ami, "COVER_STEPS", 4)
    monkeypatch.setattr(ami, "COVER_FACTOR", 1)
    mk = ami.Masked(model, means)
    a, b = docs[0][:10], list(docs[0][:10])
    b[4] = (b[4] + 1) % 64
    x = torch.tensor([a, b])
    pos = torch.tensor([9, 9])
    with torch.no_grad():
        lpf = mk.logp(x, None, None, pos)
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, torch.ones(1, 1, 10, 1))
    c = ms["curves"][0]
    assert ms["tb"] is not None and ms["units_total"][0] == 3 * 10 * mk.L * (mk.D + mk.H)
    assert c[-1][0] == ms["units_total"][0] and c[-1][1] < 1e-6
    k = ms["at"][0]["0.1"]
    with torch.no_grad():
        for r in (0, 1):
            mn, mh = ms["hard"](0, k, r)
            assert float(ami._kl(lpf[r:r + 1], mk.logp(x[r:r + 1], mn[None].float(), mh[None].float(), pos[r:r + 1]))) <= 0.1 + 1e-6


def test_backbone(tmp_path):
    T1, T2 = 16, 18                                           # (two prompt lengths: the suffix compares by role)
    pre, suf = ami.FRAME_PRE, ami.FRAME_SUF
    ids1, ids2 = pre + [5] * (T1 - 12) + suf, pre + [6] * (T2 - 12) + suf
    shared = [[3, 0, 7], [4, T1 - 1, 9]]                     # (layer, position, neuron): prefix 0 and the answer position
    r1 = {"id": "a", "ids": ids1, "neurons": shared + [[2, 5, 1]]}
    r2 = {"id": "b", "ids": ids2, "neurons": [[3, 0, 7], [4, T2 - 1, 9], [1, 6, 2]]}
    for r in (r1, r2):
        (tmp_path / f"{r['id']}.json").write_text(ami.json.dumps(r))
    res = ami.backbone(sorted(tmp_path.glob("*.json")), 1.0)
    assert res["backbone_units"] == 2 and res["beyond"] == {"a": 1, "b": 1}


def test_mask_search_starts(tiny, monkeypatch):
    """starts set the backbone's and (COVER_FACTOR) a row's own part's logits before the descent."""
    model, docs, means = tiny
    monkeypatch.setattr(ami, "COVER_STEPS", 0)
    monkeypatch.setattr(ami, "COVER_FACTOR", 1)
    mk = ami.Masked(model, means)
    a, b = docs[0][:10], list(docs[0][:10])
    b[4] = (b[4] + 1) % 64
    x = torch.tensor([a, b])
    pos = torch.tensor([9, 9])
    with torch.no_grad():
        lpf = mk.logp(x, None, None, pos)
    st = {"a": [(0, "head", 1, 9, 2, 7.5)], "b": [(1, "neuron", 0, 4, 3, 6.0)]}
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, torch.ones(1, 1, 10, 1), starts=st)
    assert float(ms["th"][1][0, 1, 9, 2]) == 7.5 and float(ms["th"][1][0, 1, 9, 1]) == ami.COVER_INIT
    assert float(ms["tb"][0][1, 0, 4, 3]) == 6.0 and float(ms["tb"][0][0, 0, 4, 3]) == ami.COVER_FACTOR_INIT


def test_units_file_round_trip(tmp_path):
    L, T, D, H, hd = 2, 12, 48, 4, 8
    units = [("neuron", 0, 4, 3), ("head", 0, 1, 7), ("neuron", 1, 9, 7), ("head", 1, 3, 11), ("neuron", 1, 11, 2)]
    M = ami.dense(units, L, T, D, H, hd)
    un = M["neuron"]
    uh = M["head"].view(L, T, H, hd).any(-1)
    dst = tmp_path / "q.json"
    rec = {"id": "q", "ids": list(range(T)), **ami.save_units(dst, un, uh), "_dir": str(tmp_path)}
    M2 = ami.load_dense(rec, L, T, D, H, hd)
    assert all(torch.equal(M[k], M2[k]) for k in M)
    assert set(ami.record_units(rec)) == set(units)
    a, b = ami.sizes(units), ami.sizes_dense(M2, hd)
    assert all(a[k] == b[k] for k in ("units", "neurons", "heads", "positions", "edges")), (a, b)


def test_task_means(tiny):
    model, docs, means = tiny
    abl = ami.Ablator(model, None, means)
    prompts = [d[:10] for d in (docs[0], docs[1], docs[2], docs[4])]
    tm = ami.task_means(abl, prompts)
    caps = [abl.capture(p) for p in prompts]
    assert torch.allclose(tm["neuron"], sum(c["neuron"].float() for c in caps) / 4, atol=1e-6)
    t = ami.TaskMeans(tm, ["a"])
    abl2 = ami.Ablator(model, None, t, mode="task")
    lp = abl2.logp([{"ids": prompts[0], "keep": []}, {"ids": prompts[0], "keep": None}])
    assert torch.isfinite(lp).all() and not torch.allclose(lp[0], lp[1])


def test_backbone_dense_and_exclude():
    L, D, H, hd, T = 2, 48, 4, 8, 16
    bb = [("neuron", 1, "pre:0", 5), ("neuron", 0, "suf:1", 7), ("head", 1, "body", 2), ("hchan", 0, "suf:2", 9)]
    m = ami.backbone_dense(bb, T, L, D, H, hd)
    assert m["neuron"][1, 0, 5] and m["neuron"][0, T - 1, 7] and int(m["neuron"].sum()) == 2
    body = list(range(len(ami.FRAME_PRE), T - len(ami.FRAME_SUF)))
    assert int(m["head"].sum()) == len(body) * hd + 1 and m["head"][1, body[0], 2 * hd:3 * hd].all() and m["head"][0, T - 2, 9]
    c = {"neuron": torch.zeros(L, T, D, dtype=torch.bool), "head": torch.zeros(L, T, H * hd, dtype=torch.bool)}
    c["neuron"][0, T - 1, 10:42] = True                      # (32 units; the backbone holds neuron 7 at that position)
    r = ami.redraw_dense(c, 0, m)                            # (the random control avoids the circuit and the backbone)
    assert int(r["neuron"][0, T - 1].sum()) == 15 and not r["neuron"][0, T - 1, 7] and not (r["neuron"] & c["neuron"]).any()


def test_roles_and_prior(tmp_path):
    """role_index names each position of a framed prompt by the role backbone_dense puts there; prior_backbone reads methods'
    backbone_fit.json (start 4: position 3 is content here, so "body")."""
    L, D, H, hd, T = 2, 48, 4, 8, 17
    names = ami.role_names()
    ri = ami.role_index(T)
    for p in range(T):
        m = ami.backbone_dense([("neuron", 0, names[ri[p]], 3)], T, L, D, H, hd)["neuron"][0, :, 3]
        assert m[p] and (int(m.sum()) == 1) == (names[ri[p]] != "body")
    d = {"start": 4, "end": 9, "units": [[0, 1, 5, 0], [0, 1, 6, 3], [1, 0, 2, 4], [0, 0, 7, 13], [1, 1, 3, 12]]}
    (tmp_path / "fit.json").write_text(ami.json.dumps(d))
    assert ami.prior_backbone(tmp_path / "fit.json") == sorted([("neuron", 1, "pre:0", 5), ("neuron", 1, "body", 6), ("head", 0, "suf:1", 2),
                                                                ("neuron", 0, "body", 7), ("head", 1, "suf:9", 3)])


def test_mask_search_fixed(tiny, monkeypatch):
    """fixed: the backbone's units are live in every row and outside the order and the sizes; the bound's hard set (with
    them) is within the bound."""
    model, docs, means = tiny
    monkeypatch.setattr(ami, "COVER_STEPS", 4)
    mk = ami.Masked(model, means)
    a, b = docs[0][:10], list(docs[0][:10])
    b[4] = (b[4] + 1) % 64
    x = torch.tensor([a, b])
    pos = torch.tensor([9, 9])
    with torch.no_grad():
        lpf = mk.logp(x, None, None, pos)
    F = {"neuron": torch.zeros(mk.L, 10, mk.D, dtype=torch.bool), "head": torch.zeros(mk.L, 10, mk.H, dtype=torch.bool)}
    F["neuron"][:, :, :20] = True
    F["head"][1, 9, :] = True
    nf = int(F["neuron"].sum() + F["head"].sum())
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, torch.ones(1, 1, 10, 1), fixed=F)
    assert ms["units_total"][0] == 10 * mk.L * (mk.D + mk.H) - nf
    assert torch.isinf(ms["th"][0][0][F["neuron"]]).all() and ms["curves"][0][-1][0] == ms["units_total"][0]
    k = ms["at"][0]["0.1"]
    z = ms["thr"](0, k)
    assert not ((ms["th"][0][0] >= z) & F["neuron"]).any()
    with torch.no_grad():
        for r in (0, 1):
            mn, mh = ms["hard"](0, k, r)
            assert (mn & F["neuron"]).sum() == F["neuron"].sum()
            assert float(ami._kl(lpf[r:r + 1], mk.logp(x[r:r + 1], mn[None].float(), mh[None].float(), pos[r:r + 1]))) <= 0.1 + 1e-6


def _framed(monkeypatch):
    monkeypatch.setattr(ami, "FRAME_PRE", [1, 2, 3])
    monkeypatch.setattr(ami, "FRAME_SUF", [4, 5, 6, 7, 8, 9, 10, 11, 12])
    monkeypatch.setattr(ami, "prompt_ids", lambda tok, user, q: ami.FRAME_PRE + [int(w) for w in user.split()] + ami.FRAME_SUF)
    qs = [{"id": "t/20/p0", "task": "t", "paraphrase": 0, "key": "20", "corrupt_key": "21", "user": "40 41 20 42"},
          {"id": "t/22/p0", "task": "t", "paraphrase": 0, "key": "22", "corrupt_key": "23", "user": "40 41 22 42"},
          {"id": "t/24/p1", "task": "t", "paraphrase": 1, "key": "24", "corrupt_key": "25", "user": "43 24 44 45 46"}]
    return qs, {"t": [str(k) for k in range(20, 30)]}



def _pair(tiny):
    model, docs, means = tiny
    mk = ami.Masked(model, means)
    a, b = docs[0][:10], list(docs[0][:10])
    b[4] = (b[4] + 1) % 64
    x, pos = torch.tensor([a, b]), torch.tensor([9, 9])
    with torch.no_grad():
        lpf = mk.logp(x, None, None, pos)
    return mk, x, pos, lpf


def _within(mk, ms, x, pos, lpf, rows, bound=0.1):
    k = ms["at"][0][str(bound)]
    with torch.no_grad():
        for r in rows:
            mn, mh = ms["hard"](0, k, r)
            assert float(ami._kl(lpf[r:r + 1], mk.logp(x[r:r + 1], mn[None].float(), mh[None].float(), pos[r:r + 1]))) <= bound + 1e-6


def test_mask_search_validation_rows(tiny, monkeypatch):
    """A validation row is outside the descent (the logits equal a search on the fit row alone), the curve is the worst of
    curves_fit and curves_val, and the bound's set is within the bound on the validation row too."""
    monkeypatch.setattr(ami, "COVER_STEPS", 4)
    mk, x, pos, lpf = _pair(tiny)
    live = torch.ones(1, 1, 10, 1)
    one = ami.mask_search(mk, x[:1], [0], pos[:1], lpf[:1], live)
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, live, val=[False, True])
    assert torch.equal(one["th"][0], ms["th"][0]) and torch.equal(one["th"][1], ms["th"][1])
    assert ms["rows_of"] == [[0]] and ms["val_rows_of"] == [[1]] and len(ms["history"][0][1]) == 1
    fit, vl = dict(map(tuple, ms["curves_fit"][0])), dict(map(tuple, ms["curves_val"][0]))
    assert all(abs(z - max(fit[k], vl[k])) < 1e-9 for k, z in ms["curves"][0])
    _within(mk, ms, x, pos, lpf, (0, 1))


def test_mask_search_snapshots(tiny, monkeypatch):
    """COVER_SNAPS: curves read at those steps of one descent without changing it (the last snapshot is a plain run's
    result), the result the snapshot with the fewest units within the pick bound, its set within the bound."""
    monkeypatch.setattr(ami, "COVER_STEPS", 6)
    mk, x, pos, lpf = _pair(tiny)
    live = torch.ones(1, 1, 10, 1)
    plain = ami.mask_search(mk, x, [0, 0], pos, lpf, live)
    monkeypatch.setattr(ami, "COVER_SNAPS", "2,4")
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, live)
    sn = ms["snapshots"][0]
    assert [v["step"] for v in sn] == [2, 4, 6] and sn[-1]["k"] == plain["at"][0]
    ks = [v["k"]["0.1"] for v in sn]
    best = min(k for k in ks if k is not None)
    assert ms["at"][0]["0.1"] == best and ms["step"][0] == sn[ks.index(best)]["step"]
    _within(mk, ms, x, pos, lpf, (0, 1))


@pytest.mark.parametrize("order", ["mask", "ig", "igabs", "igmask", "pair2"])
def test_tsearch_validation_keys(tiny, monkeypatch, tmp_path, order):
    """tsearch with TSEARCH_VAL, for each order of the units: the validation keys follow the fit keys in selection_keys
    (r3 never scores them), and the written circuit is within QSEARCH_BOUND on every fit and validation prompt."""
    model, docs, means = tiny
    qs, keys = _framed(monkeypatch)
    monkeypatch.setattr(ami, "TSEARCH_KEYS", 3)
    monkeypatch.setattr(ami, "TSEARCH_VAL", 2)
    monkeypatch.setattr(ami, "TSEARCH_ORDER", order)
    monkeypatch.setattr(ami, "COVER_STEPS", 4)
    monkeypatch.setattr(ami, "COVER_SNAPS", "2")
    mk = ami.Masked(model, means)
    ami.tsearch(mk, None, qs[:1], keys, tmp_path, batch=1)
    rec = ami.json.loads((tmp_path / "t__20__p0.json").read_text())
    s = rec["r3search"]
    assert rec["selection_keys"][:2] == ["20", "21"] and len(rec["selection_keys"]) == 5 and rec["order"] == order
    assert s["keys"] == rec["selection_keys"][:3] and s["val_keys"] == rec["selection_keys"][3:] and s["curve_val"]
    assert [v["step"] for v in s["snapshots"]] == ([2, 4] if order in ("mask", "igmask") else [0]) and rec["met"]
    assert ("k_order" in rec) == (order == "igmask")
    q = qs[0]
    rows = [ami.prompt_ids(None, q["user"].replace(q["key"], k), q) for k in rec["selection_keys"]]
    T = len(rows[0])
    M = ami.load_dense({**rec, "_dir": str(tmp_path)}, mk.L, T, mk.D, mk.H, mk.hd)
    x, pos = torch.tensor(rows), torch.full((len(rows),), T - 1)
    mh = M["head"].view(mk.L, T, mk.H, mk.hd).any(-1).float()
    with torch.no_grad():
        kl = ami._kl(mk.logp(x, None, None, pos), mk.logp(x, M["neuron"].float()[None].expand(len(rows), -1, -1, -1),
                                                           mh[None].expand(len(rows), -1, -1, -1), pos))
    assert float(kl.max()) <= ami.QSEARCH_BOUND + 1e-6


def test_ig_scores_complete(tiny):
    """Integrated gradients against the background: an item's scores over every unit sum to about the background's KL
    on each of its prompts, summed (completeness, midpoint rule)."""
    mk, x, pos, lpf = _pair(tiny)
    sn, sh = ami.ig_scores(mk, x, [0, 0], pos, lpf, 1, steps=64)
    T = x.shape[1]
    with torch.no_grad():
        kl0 = ami._kl(lpf, mk.logp(x, torch.zeros(2, mk.L, T, mk.D), torch.zeros(2, mk.L, T, mk.H), pos)).sum()
    assert sn.shape == (1, mk.L, T, mk.D) and sh.shape == (1, mk.L, T, mk.H)
    assert abs(float(sn.sum() + sh.sum()) - float(kl0)) <= 0.03 * float(kl0)


def test_pair_scores(tiny):
    """Pair attribution: zero for a pair of identical prompts; for a pair of prompts that differ, the same as the units'
    value differences times the mean of the two runs' gradients computed one prompt at a time."""
    mk, x, pos, lpf = _pair(tiny)
    same = x[[0, 0]]
    sn, sh = ami.pair_scores(mk, same, [0, 0], pos, lpf[[0, 0]], 1)
    assert float(sn.abs().max()) == 0 and float(sh.abs().max()) == 0
    sn, sh = ami.pair_scores(mk, x, [0, 0], pos, lpf, 1)
    ta, tb = lpf.argmax(-1).tolist()
    vals = []
    for r in (0, 1):                                          # (one run at a time: values and gradients of the contrast)
        cap = []
        hs = [layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: (a[0].retain_grad(), cap.append(a[0]))[0])
              for layer in mk.layers]
        try:
            e = mk.core.model.embed_tokens(x[r:r + 1]).detach().requires_grad_(True)
            lg = mk.core.lm_head(mk.core.model(inputs_embeds=e).last_hidden_state[0, pos[r]])
            (lg[ta] - lg[tb]).backward()
        finally:
            for h in hs:
                h.remove()
        vals.append([(c.detach()[0], c.grad[0]) for c in cap])
    want = torch.stack([(a[0] - b[0]) * 0.5 * (a[1] + b[1]) for a, b in zip(*vals)])   # [L, T, D]
    assert torch.allclose(sn[0], 2 * want, atol=1e-5)        # (the cyclic pairs (0, 1) and (1, 0): equal, summed)



def test_rank_logits_and_init(tiny, monkeypatch):
    """rank_logits: an item's first ks units by score start live (logit > 0), the rest not, in the scores' order; with
    no descent, mask_search(init=...) returns those logits."""
    g = torch.Generator().manual_seed(4)
    sc = (torch.randn(2, 2, 10, 48, generator=g), torch.randn(2, 2, 10, 4, generator=g))
    zn, zh = ami.rank_logits(sc, [100, 7])
    for j, k in ((0, 100), (1, 7)):
        v, z = torch.cat([sc[0][j].reshape(-1), sc[1][j].reshape(-1)]), torch.cat([zn[j].reshape(-1), zh[j].reshape(-1)])
        top = v.argsort(descending=True)
        assert (z[top[:k]] > 0).all() and (z[top[k:]] <= 0).all() and torch.all(z[top][:-1] >= z[top][1:])
    monkeypatch.setattr(ami, "COVER_STEPS", 0)
    mk, x, pos, lpf = _pair(tiny)
    sc1 = (torch.randn(1, mk.L, 10, mk.D, generator=g), torch.randn(1, mk.L, 10, mk.H, generator=g))
    init = ami.rank_logits(sc1, [50])
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, torch.ones(1, 1, 10, 1), init=init)
    assert torch.equal(ms["th"][0], init[0]) and torch.equal(ms["th"][1], init[1]) and ms["k_mask"][0] == 50

def test_mask_search_given_order(tiny, monkeypatch):
    """scores: no descent, the logits are the given scores, the curve's sets are prefixes of their order, and the bound's
    set is within the bound."""
    mk, x, pos, lpf = _pair(tiny)
    g = torch.Generator().manual_seed(3)
    sc = (torch.randn(1, mk.L, 10, mk.D, generator=g), torch.randn(1, mk.L, 10, mk.H, generator=g))
    ms = ami.mask_search(mk, x, [0, 0], pos, lpf, torch.ones(1, 1, 10, 1), scores=sc)
    assert torch.equal(ms["th"][0], sc[0]) and torch.equal(ms["th"][1], sc[1]) and ms["history"] == []
    assert ms["step"] == [0] and ms["k_mask"][0] == int((sc[0] > 0).sum() + (sc[1] > 0).sum())
    _within(mk, ms, x, pos, lpf, (0, 1))

def test_bsearch_prior_and_roles(tiny, monkeypatch, tmp_path):
    """No passes: the backbone is the prior, and each question's measured KL with the backbone alone is the one its
    backbone_dense mask gives (role_index and backbone_dense agree); no prior: every unit, the full model."""
    model, docs, means = tiny
    qs, keys = _framed(monkeypatch)
    monkeypatch.setattr(ami, "BS_EPOCHS", 0)
    monkeypatch.setattr(ami, "TSEARCH_KEYS", 4)
    mk = ami.Masked(model, means)
    prior = [("neuron", 0, "body", 3), ("neuron", 1, "suf:1", 7), ("head", 1, "suf:1", 2), ("head", 0, "pre:1", 0)]
    res = ami.bsearch(mk, None, qs, keys, tmp_path / "p", batch=2, prior=prior)
    assert sorted(res["units"]) == sorted(prior) and res["size"] == 4 and len(res["questions"]) == 3
    for r in res["questions"]:
        q = next(q for q in qs if q["id"] == r["id"])
        rows = [ami.prompt_ids(None, q["user"].replace(q["key"], k), q) for k in r["keys"]]
        T = len(rows[0])
        B = ami.backbone_dense(prior, T, mk.L, mk.D, mk.H, mk.hd)
        x, pos = torch.tensor(rows), torch.full((len(rows),), T - 1)
        mn = B["neuron"].float()[None].expand(len(rows), -1, -1, -1)
        mh = B["head"].view(mk.L, T, mk.H, mk.hd).any(-1).float()[None].expand(len(rows), -1, -1, -1)
        with torch.no_grad():
            kl = ami._kl(mk.logp(x, None, None, pos), mk.logp(x, mn, mh, pos))
        assert abs(float(kl.max()) - r["kl_backbone"]) < 1e-3
    full = ami.bsearch(mk, None, qs, keys, tmp_path / "f", batch=2)
    assert full["size"] == len(ami.role_names()) * mk.L * (mk.D + mk.H) and full["pass_backbone"] == 3


def test_bsearch_runs(tiny, monkeypatch, tmp_path):
    """Two passes (the last hard): the backbone file is load_backbone's, every question measured, the history a batch a pass."""
    model, docs, means = tiny
    qs, keys = _framed(monkeypatch)
    monkeypatch.setattr(ami, "BS_EPOCHS", 2)
    monkeypatch.setattr(ami, "BS_STEPS", 3)
    monkeypatch.setattr(ami, "TSEARCH_KEYS", 4)
    res = ami.bsearch(ami.Masked(model, means), None, qs, keys, tmp_path, batch=2)
    bb = ami.load_backbone(tmp_path / "backbone.json")
    assert len(bb) == res["size"] and all(u[2] in ami.role_names() for u in bb)
    assert len(res["questions"]) == 3 and len(res["history"]) == 2 * 2 and res["history"][-1]["hard"]
    assert all(r["kl_with_own"] >= 0 and r["own"] >= 0 for r in res["questions"])


def test_fit_questions():
    qs = [{"id": f"t/{k}/p{p}", "task": t, "paraphrase": p} for t in ("a", "b") for k in range(5) for p in range(3)]
    got = ami.fit_questions(qs, ["a"], per_task=4)
    assert len(got) == 4 and {q["task"] for q in got} == {"a"} and {q["paraphrase"] for q in got} == {0, 1, 2}
