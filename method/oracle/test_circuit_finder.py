"""Checks of the patching-pair circuit finder (circuit_finder.py) on a tiny random Qwen3: patching the whole residual
input of the first layer gives the other run exactly, a block patch equals its units patched one by one, a unit frozen
at its value in the base run changes nothing, find()'s record is consistent with its own measurements, and a cut's
eps stays at or above its measured noise floor.

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


def test_whole_layer_unit_equals_its_parts(tiny):
    """A unit of every neuron (or head) of a layer at a position, edited without indices, equals the same units given
    as two parts (indexed edits), in every source."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    D, H = pt.D, pt.H
    for src in ((), ("zero",), ("self",)):
        whole = [("neuron", 1, 6, torch.arange(D)) + src, ("head", 0, torch.arange(H), 7) + src]
        parts = [("neuron", 1, 6, tuple(range(D - 1))) + src, ("neuron", 1, 6, D - 1) + src,
                 ("head", 0, tuple(range(H - 1)), 7) + src, ("head", 0, H - 1, 7) + src]
        a, b = pt.both([whole, parts])
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
    for k, ss in rec["singles"].items():           # each cut's measured singles, largest first, the circuit's among them
        big = [max(abs(x["lost"]), abs(x["restored"])) for x in ss]
        assert big == sorted(big, reverse=True) and len(ss) <= cf.SINGLES_KEPT
    rm = rec["residual_map"]
    assert len(rm["lost"]) == len(rm["restored"]) == pt.L and all(len(r) == len(rm["positions"]) for r in rm["lost"])
    if "source" in rec["cuts"]:
        assert rec["cuts"]["source"]["searched"] == rec["source_positions"]
        assert all(rec["first_diff"] <= p < pt.T - 1 for p in rec["source_positions"])
    for c in rec["cuts"].values():
        assert c["reachable"] == (c["ceiling"][0] >= cf.PASS and c["ceiling"][1] >= cf.PASS)
        assert c["reachable"] or c["size"] == 0     # (an unreachable cut has no units)


def test_noise_floor(tiny, monkeypatch):
    """A no-op patch ("self") changes neither run, so a float32 floor is float error; a cut never runs eps below its
    floor, and with no passing prefix there its size is the whole blocks' curve."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    a, b = pt.both([[("neuron", 1, 6, (3, 5, 9), "self"), ("head", 0, (1, 2), 7, "self")]])[0]
    assert abs(a) < 1e-6 and abs(b) < 1e-6
    assert cf.floor(pt, "neuron", range(pt.first, pt.T - 1)) < 1e-5
    sc = pt.scores()
    monkeypatch.setattr(cf, "floor", lambda *a, **k: 10.0)     # every share below the floor: no single unit, no lower round
    reached = 0
    for kind, positions in (("neuron", list(range(pt.first, pt.T))), ("head", [pt.T - 1]), ("neuron", list(range(pt.first, pt.T - 1)))):
        c = cf.cut(pt, sc, kind, positions, {})
        if not c["reachable"]:
            assert c["spread"] is None and c["floor"] == [0.0]
            continue
        reached += 1
        assert c["eps"] == 10.0 and c["found"] == 0 and not c["met"] and c["floor"][0] == 10.0
        width = pt.H if kind == "head" else pt.D
        sp = c["spread"]
        assert sp["blocks"] is not None and sp["units"] == sp["blocks"] * width
        blocks = [cf.unit(kind, i, p, torch.arange(width)) for i, p in sp["order"][:sp["blocks"]]]
        lost, rest = pt.both([blocks])[0]
        assert lost >= cf.PASS and rest >= cf.PASS
        at = [x for x in sp["curve"] if x["k"] == sp["blocks"]]
        assert not at or abs(at[0]["lost"] - lost) < 1e-5
    assert reached


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


def test_zero_is_a_knockout(tiny):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    i, p = 1, pt.T - 1
    ks = list(range(pt.D))
    got = pt.run("clean", [[("neuron", i, p, ks, "zero")]])[0]
    with torch.no_grad():
        def z(mod, args):
            h = args[0].clone()
            h[:, p] = 0
            return (h,) + tuple(args[1:])
        hk = pt.layers[i].mlp.down_proj.register_forward_pre_hook(z)
        try:
            c = float(pt._contrast(pt._last(clean, pt.rows))[0])
        finally:
            hk.remove()
    assert abs(got - (pt.c["clean"] - c) / pt.gap) < 1e-5


def test_relp_keeps_the_forward(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    x = torch.tensor([clean])
    with torch.no_grad():
        a = pt.core(input_ids=x).logits
        with cf._relp(pt.core):
            b = pt.core(input_ids=x).logits
        c = pt.core(input_ids=x).logits               # (restored after)
    assert torch.allclose(a, b, atol=1e-5) and torch.equal(a, c)
    monkeypatch.setattr(cf, "ORDER", "grad")
    g = pt.scores()
    monkeypatch.setattr(cf, "ORDER", "relp")
    r = pt.scores()
    for k in ("head", "neuron"):
        assert r[k].shape == g[k].shape and torch.isfinite(r[k]).all() and not torch.allclose(r[k], g[k])


def test_staged_rows_equal_whole_batches(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    T, f = pt.T, pt.first
    sets = [[("neuron", 1, f + 1, (0, 7))], [("res", 0, f + 1)], [("head", 1, 2, f + 1)], [("neuron", 0, T - 1, 3)],
            [("res", 1, f + 2), ("head", 0, (1, 3), T - 1)], [("neuron", 1, T - 1, 5, "zero")], [("res", 1, T - 1, "clean")]]
    got = {}
    for on in (True, False):
        monkeypatch.setattr(cf, "STAGED", on)
        got[on] = pt.both(sets) + [(x, x) for x in pt.run("corrupt", sets)]
    for a, b in zip(got[True], got[False]):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5


def test_any_rows_per_batch(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    s = pt.first + 1
    sets = [[("res", i, s)] for i in range(pt.L)] + [[("neuron", i, s, (k, k + 3))] for i in range(pt.L) for k in range(3)]
    got = {}
    for staged in (False, True):
        for bt in (cf.BATCH_TOKENS, 9 * (pt.T - s)):          # 9 rows a batch (odd): every set in its own base's rows
            monkeypatch.setattr(cf, "STAGED", staged)
            monkeypatch.setattr(cf, "BATCH_TOKENS", bt)
            got[staged, bt] = pt.both(sets)
    ref = got[False, cf.BATCH_TOKENS]
    for v in got.values():
        assert all(abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5 for a, b in zip(v, ref))


def test_given_equals_joined_sets(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    T, f = pt.T, pt.first
    given = [("neuron", 0, f, (2, 9)), ("res", 1, f + 2), ("head", 1, 3, T - 1), ("neuron", 1, T - 1, 4)]
    sets = [[("neuron", 1, f + 1, (0, 7))], [("res", 0, f + 1)], [("head", 0, 2, f + 1)], [("neuron", 0, T - 1, 3)],
            [("res", 1, f + 2)], [("neuron", 1, T - 1, 5, "zero")], [("neuron", 0, f, 2)]]
    for staged in (True, False):
        monkeypatch.setattr(cf, "STAGED", staged)
        a = pt.both(sets, given=given)
        b = pt.both([given + st for st in sets])
        c = pt.run("corrupt", sets, given=given)
        for x, y, z in zip(a, b, c):
            assert abs(x[0] - y[0]) < 1e-5 and abs(x[1] - y[1]) < 1e-5 and abs(x[1] - z) < 1e-5


def test_explicit_token_contrast_matches_direct_model_no_word_forwards(tiny,monkeypatch):
    import math
    pt,clean,corrupt=tiny
    sets={'answer':[0,2], 'alternatives':[4,7]}
    rows=[];hook=pt.core.model.register_forward_pre_hook(lambda m,args,kwargs:rows.append(kwargs['input_ids'].shape[0]),with_kwargs=True)
    monkeypatch.setattr(cf.answer_contrast,'token_sets',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not group words')))
    try:
        record=pt.setup(clean,corrupt,explicit_sets=sets)
        assert rows==[pt.rows,pt.rows]
        assert record['kind']=='explicit_tokens' and record['answer_word'] is None and record['alternative_words']==[]
        assert record['answer_tokens']==sets['answer'] and record['alternatives']==sets['alternatives']
        with torch.no_grad():
            for side,ids in [('clean',clean),('corrupt',corrupt)]:
                lp=pt.core(input_ids=torch.tensor([ids]).expand(pt.rows,-1),logits_to_keep=1).logits[0,-1].float().log_softmax(-1)
                expected=(torch.logsumexp(lp[sets['answer']],0)-torch.logsumexp(lp[sets['alternatives']],0))/math.log(2)
                assert abs(record['contrast'][side]-expected.item())<1e-6
        sets['answer'].append(3)
        assert pt.sets['answer']==[0,2] # caller mutation cannot change contrast
    finally:hook.remove()


@pytest.mark.parametrize('sets',[
    {},{'answer':[1]}, {'answer':[1],'alternatives':[2],'extra':[]},
    {'answer':[],'alternatives':[2]}, {'answer':[1],'alternatives':[]},
    {'answer':[1,1],'alternatives':[2]}, {'answer':[1],'alternatives':[2,2]},
    {'answer':[1],'alternatives':[1]}, {'answer':[-1],'alternatives':[2]},
    {'answer':[99999],'alternatives':[2]}, {'answer':[True],'alternatives':[2]},
    {'answer':[1.0],'alternatives':[2]}, {'answer':'1','alternatives':[2]},
])
def test_explicit_token_sets_invalid_before_any_forward(tiny,sets,monkeypatch):
    pt,clean,corrupt=tiny
    monkeypatch.setattr(pt,'_last',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected forward')))
    with pytest.raises(ValueError,match='explicit_sets'):pt.setup(clean,corrupt,explicit_sets=sets)


def test_explicit_sets_cannot_mix_word_hint(tiny,monkeypatch):
    pt,clean,corrupt=tiny
    monkeypatch.setattr(pt,'_last',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected forward')))
    with pytest.raises(ValueError,match='mutually exclusive'):
        pt.setup(clean,corrupt,answers=[1],explicit_sets={'answer':[1],'alternatives':[2]})


def test_floor_measured_in_low_precision_only(tiny):
    """float32: 0 unmeasured (no-op shares are float error); bfloat16: the no-op patches measured."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    assert cf.floor(pt, "neuron", range(pt.first, pt.T - 1)) == 0.0
    import copy
    lo = cf.Patcher(copy.deepcopy(pt.model).to(torch.bfloat16), None, rows=8)
    lo.setup(clean, corrupt)
    f = cf.floor(lo, "neuron", range(lo.first, lo.T - 1))
    assert 0.0 <= f < 1.0


def test_contrast_from_the_token_sets_logits(tiny):
    """The contrast from the answer's and the alternatives' float32 logits equals answer_contrast's from every log
    probability (the softmax's normalizer cancels)."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    hn = pt._last(clean, 4)
    want = cf.answer_contrast.contrast(pt._logp(hn), {k: pt.sets[k] for k in ("answer", "alternatives")})
    assert torch.allclose(pt._contrast(hn), want, atol=1e-4)


def test_beams_reach_group_testings_singles(tiny):
    """Beams that never stop reach group testing's single units with its shares; a stop after the first beam gives a
    subset."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    sc = pt.scores()
    pos = list(range(pt.first, pt.T))
    full = cf.group_test(pt, sc, 0.05, {}, kinds=("neuron",), positions=pos)
    every = cf.beams(pt, sc, 0.05, ("neuron",), pos, pt.both, lambda us: False, 2)
    assert set(every) == set(full) and all(abs(every[u][0] - full[u][0]) < 1e-5 for u in full)
    first = cf.beams(pt, sc, 0.05, ("neuron",), pos, pt.both, lambda us: True, 2)
    assert set(first) <= set(full) and (first or not full)
