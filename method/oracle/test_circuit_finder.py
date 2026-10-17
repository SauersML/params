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
    assert all(abs(x) < 1e-5 for x in pt.both([[("res", 0, p) for p in range(pt.first)]])[0])   # before the change: equal


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
    assert abs(a) < 1e-5 and abs(b) < 1e-5     # (float32 error: 1.31e-6 on CI's Linux BLAS)


def test_find_record(tiny, monkeypatch):
    pt, clean, corrupt = tiny
    monkeypatch.setattr(cf, "RESIDUAL_MAP", True)
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
    assert rec["met"] == all(c["met"] or not c["reachable"] or c["diffuse"] for c in rec["cuts"].values())   # (Q1: a
    assert set(rec["diffuse"]) == {k for k, c in rec["cuts"].items() if c["diffuse"]}           # diffuse cut excepted)
    for k, c in rec["cuts"].items():
        assert c["bar"] == cf.PASS              # (float32: no noise margin)
        if c["diffuse"]:
            assert c["size"] == 0
    for x in rec["nodes"] + rec["routes"]:
        assert {"direct", "direct_delta", "at_answer"} <= set(x)
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


def test_prior_first(tiny, monkeypatch):
    """A prior that passes is the cut (pruned, nothing searched); one that fails leaves the record as without it."""
    pt, clean, corrupt = tiny
    monkeypatch.setattr(cf, "EPS", 0.05)
    rec = cf.find(pt, clean, corrupt)
    units = {s: ([("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"] if n["stage"] == s] if s != "movers"
                 else [("head", r["layer"], r["head"], r["pos"]) for r in rec["routes"]]) for s in rec["cuts"]}
    calls = []
    monkeypatch.setattr(cf, "group_test", lambda *a, **k: calls.append(1) or {})
    again = cf.find(pt, clean, corrupt, prior={s: u + [("neuron", 0, pt.T - 1, 0)] for s, u in units.items()})
    for s, c in rec["cuts"].items():
        if not c["met"]:
            continue
        a = again["cuts"][s]
        assert a["met"] and a["prior"] and a["size"] <= c["size"] + 1
        mine = ([("neuron", n["layer"], n["pos"], n["neuron"]) for n in again["nodes"] if n["stage"] == s] if s != "movers"
                else [("head", r["layer"], r["head"], r["pos"]) for r in again["routes"]])
        assert set(mine) <= set(units[s] + [("neuron", 0, pt.T - 1, 0)])
        lost, rest = pt.both([mine])[0]
        assert lost >= cf.PASS and rest >= cf.PASS and abs(lost - a["lost"]) < 1e-5
    monkeypatch.undo()
    monkeypatch.setattr(cf, "EPS", 0.05)
    none = cf.find(pt, clean, corrupt, prior={s: [] for s in rec["cuts"]})
    fail = cf.find(pt, clean, corrupt, prior={"source": [("neuron", 0, rec["first_diff"], 0)], "movers": [("head", 0, 0, pt.T - 1)]})
    for s in rec["cuts"]:
        if fail["cuts"][s]["prior"]:
            continue                                # (the one-unit prior happened to pass there)
        assert fail["cuts"][s]["size"] == none["cuts"][s]["size"] == rec["cuts"][s]["size"]


def test_noise_floor(tiny, monkeypatch):
    """A no-op patch ("self") changes neither run, so a float32 floor is float error; a cut never runs eps below its
    floor, and with no passing prefix there its size is the whole blocks' curve."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    a, b = pt.both([[("neuron", 1, 6, (3, 5, 9), "self"), ("head", 0, (1, 2), 7, "self")]])[0]
    assert abs(a) < 1e-5 and abs(b) < 1e-5     # (float32 error: 1.31e-6 on CI's Linux BLAS)
    assert cf.floor(pt, "neuron", range(pt.first, pt.T - 1)) < 1e-5
    sc = pt.scores()
    monkeypatch.setattr(cf, "floor", lambda *a, **k: 10.0)     # every share below eps and the floor: no single unit, no
    monkeypatch.setattr(cf, "EPS", 10.0)                       # lower round (the floor measured lazily, before round 1)
    monkeypatch.setattr(cf, "DENSE", False)                    # (the whole blocks' size only; DENSE: its own test)
    monkeypatch.setattr(cf, "ROUTE_GROUPS", False)             # (the whole blocks' fallback; route groups: test_route_groups)
    reached = 0
    for kind, positions in (("neuron", list(range(pt.first, pt.T))), ("head", [pt.T - 1]), ("neuron", list(range(pt.first, pt.T - 1)))):
        c = cf.cut(pt, sc, kind, positions, {})
        if not c["reachable"]:
            assert c["spread"] is None and c["floor"] == []        # (unreachable: no rounds, no floor)
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


def test_dense_prefix_inside_whole_blocks(tiny, monkeypatch):
    """With every single unit under eps (a floor of 10), a reachable cut whose smallest passing set of whole blocks is j
    blocks takes the smallest prefix of those blocks' units in attribution order that passes; its units pass when
    patched together, and every unit's shares alone are recorded."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    sc = pt.scores()
    monkeypatch.setattr(cf, "floor", lambda *a, **k: 10.0)
    monkeypatch.setattr(cf, "EPS", 10.0)
    monkeypatch.setattr(cf, "ROUTE_GROUPS", False)
    ran = 0
    for kind, positions in (("neuron", list(range(pt.first, pt.T))), ("head", [pt.T - 1])):
        c = cf.cut(pt, sc, kind, positions, {})
        if not c["reachable"] or not (c["spread"] or {}).get("blocks"):
            continue
        ran += 1
        d = c["spread"]["dense"]
        assert d["units"] is not None and c["met"] and len(c["units"]) == d["units"]
        lost, rest = pt.both([c["units"]])[0]
        assert lost >= cf.PASS and rest >= cf.PASS and abs(lost - c["lost"]) < 1e-5
        assert all(u in c["singles"] for u in c["units"])
    assert ran


def test_direct_write_is_the_contrast_slope(tiny):
    """A unit's direct write at the answer position is the contrast's slope along its write added to the final norm's
    input there, the norm's scale held (a small step)."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    T, L = pt.T, pt.L
    for u in (("neuron", L - 1, T - 1, 5), ("head", L - 1, 1, T - 1)):
        d = pt.direct([u])[0]
        assert d["at_answer"]
        if u[0] == "neuron":
            w = pt.layers[L - 1].mlp.down_proj.weight[:, u[3]].float() * float(pt.act["clean"]["neu"][L - 1][T - 1, u[3]])
        else:
            sl = slice(u[2] * pt.hd, (u[2] + 1) * pt.hd)
            w = pt.layers[L - 1].self_attn.o_proj.weight[:, sl].float() @ pt.act["clean"]["head"][L - 1][T - 1, sl].float()
        norm, cap = pt.core.model.norm, {}
        eps = 1e-3
        h = norm.register_forward_hook(lambda m, a, o: cap.update(x=a[0][0, -1].float()))
        try:
            pt.core.model(input_ids=torch.as_tensor(clean)[None])
        finally:
            h.remove()
        x = cap["x"]
        rms = torch.sqrt((x * x).mean() + norm.variance_epsilon)
        outs = [float(pt._contrast((norm.weight.float() * (x + t * eps * w) / rms)[None])[0]) for t in (0, 1)]
        slope = (outs[1] - outs[0]) / eps
        assert abs(slope - d["direct"]) < 0.05 * max(1.0, abs(d["direct"]))


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
        assert rows==[2]   # (both runs in one forward of two rows)
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


def test_find_many_equals_find(tiny, monkeypatch):
    """Pairs of one clean prompt run as a Group (their measure calls in shared batches) give find()'s records."""
    pt, clean, corrupt = tiny
    monkeypatch.setattr(cf, "EPS", 0.05)
    corrupts = [corrupt]
    for d in (3, 5):                                # (more corrupted prompts of the same length, changed elsewhere too)
        x = list(corrupt)
        x[6] = (x[6] + d) % 64
        corrupts.append(x)
    one = []
    for x in corrupts:
        one.append(cf.find(pt, clean, x))
    pts = [cf.Patcher(pt.model, None, rows=pt.rows) for _ in corrupts]
    many = cf.find_many(pts, clean, corrupts)
    for a, b in zip(one, many):
        assert a.get("skipped") == b.get("skipped")
        for k in a.get("cuts", {}):
            assert a["cuts"][k]["size"] == b["cuts"][k]["size"]
            assert abs(a["cuts"][k]["lost"] - b["cuts"][k]["lost"]) < 1e-4 and abs(a["cuts"][k]["restored"] - b["cuts"][k]["restored"]) < 1e-4
        assert [(n["layer"], n["pos"], n["neuron"]) for n in a.get("nodes", [])] == [(n["layer"], n["pos"], n["neuron"]) for n in b.get("nodes", [])]
        ra, rb = a.get("residual_map"), b.get("residual_map")
        if ra:
            assert max(abs(x - y) for r1, r2 in zip(ra["lost"], rb["lost"]) for x, y in zip(r1, r2)) < 1e-4


def test_route_groups(tiny, monkeypatch):
    """Heads grouped by where they look: each group's shares are its heads patched together; a met result's units pass
    and are a prefix of the chosen group in its heads' own order."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    monkeypatch.setattr(cf, "PASS", 0.0)            # (any group passes on a random model: the search path runs)
    last = pt.T - 1
    singles = {}
    got = cf.route_groups(pt, singles, last, list(range(pt.first, last)))
    assert got["groups"] and all({"source", "tau", "heads", "lost", "restored"} <= set(g) for g in got["groups"])
    g0 = got["groups"][0]
    pt.attends(0, 0, last)
    att = torch.stack([a[:, last] for a in pt._att])
    src = list(range(pt.first, last)) if g0["source"] is None else [g0["source"]]
    hs = [(int(i), int(h)) for i, h in (att[:, :, src].max(-1).values >= g0["tau"]).nonzero().tolist()]
    lo, re = pt.both([cf.head_set(hs, last)])[0]
    assert abs(lo - g0["lost"]) < 1e-5 and abs(re - g0["restored"]) < 1e-5
    assert got["met"] and got["units"] and all(u in singles for u in got["units"])


def test_peel_finds_group_testings_large_singles(tiny, monkeypatch):
    """PEEL: a whole block's top-attribution units alone and the rest halved; a single unit both reach has the same
    measured shares."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    sc = pt.scores()
    pos = list(range(pt.first, pt.T))
    full = cf.group_test(pt, sc, 0.05, {}, kinds=("neuron",), positions=pos)
    monkeypatch.setattr(cf, "PEEL", 4)
    peeled = cf.group_test(pt, sc, 0.05, {}, kinds=("neuron",), positions=pos)
    for u, r in peeled.items():
        if u in full:
            assert abs(r[0] - full[u][0]) < 1e-5 and abs(r[1] - full[u][1]) < 1e-5


def test_prefixes_equal_their_sets(tiny):
    """prefixes(): the shares of units[:k] for each k, as the sets themselves measure (neurons and heads mixed, units
    before the first change among them, sizes in any order)."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    g = torch.Generator().manual_seed(3)
    units = []
    for _ in range(40):
        i, p = int(torch.randint(pt.L, (1,), generator=g)), int(torch.randint(pt.T, (1,), generator=g))
        units.append(("neuron", i, p, int(torch.randint(pt.D, (1,), generator=g))) if torch.rand(1, generator=g) < 0.7 else
                     ("head", i, int(torch.randint(pt.H, (1,), generator=g)), p))
    ks = [40, 1, 5, 17, 3, 0, 29]
    got = pt.both_prefixes(units, ks)
    want = pt.both([units[:k] for k in ks])
    for a, b in zip(got, want):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
    assert [v[0] for v in pt.prefixes(units, ks, ("corrupt",))] == pytest.approx([v[1] for v in want], abs=1e-5)


def test_lean_layers_equal_the_models(tiny, monkeypatch):
    """LEAN: each staged layer through its own modules gives the model's own layer's shares (residual, head and neuron
    edits, a knockout)."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    T, f = pt.T, pt.first
    sets = [[("neuron", 1, f + 1, (0, 7))], [("res", 0, f + 1)], [("head", 0, 2, f + 1)], [("neuron", 0, T - 1, 3)],
            [("res", 1, f + 2), ("head", 1, (1, 3), T - 1)], [("neuron", 1, T - 1, 5, "zero")]]
    got = {}
    for lean in (True, False):
        monkeypatch.setattr(cf, "LEAN", lean)
        got[lean] = pt.both(sets)
    for a, b in zip(got[True], got[False]):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5


def test_lean_layers_equal_the_models_llama(monkeypatch):
    """LEAN on a Llama layer (no q/k norms) gives the model's own layer's shares."""
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    pt = cf.Patcher(transformers.LlamaForCausalLM(cfg).eval().requires_grad_(False), None, rows=8)
    clean = torch.randint(0, 64, (10,), generator=torch.Generator().manual_seed(1)).tolist()
    corrupt = list(clean)
    corrupt[4] = (clean[4] + 7) % 64
    pt.setup(clean, corrupt)
    T, f = pt.T, pt.first
    sets = [[("neuron", 1, f + 1, (0, 7))], [("res", 0, f + 1)], [("head", 0, 2, f + 1)], [("res", 1, f + 2), ("head", 1, (1, 3), T - 1)]]
    got = {}
    for lean in (True, False):
        monkeypatch.setattr(cf, "LEAN", lean)
        got[lean] = pt.both(sets)
    for a, b in zip(got[True], got[False]):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5


def test_group_prefixes_equal_alone(tiny):
    """Members' prefix calls merged in a Group's batches give each member's prefixes() alone."""
    pt, clean, corrupt = tiny
    corrupts = [corrupt, [(c + 5) % 64 if i == 6 else c for i, c in enumerate(corrupt)]]
    g = torch.Generator().manual_seed(5)
    orders = []
    for _ in corrupts:
        units = []
        for _ in range(30):
            i, p = int(torch.randint(pt.L, (1,), generator=g)), int(torch.randint(pt.T, (1,), generator=g))
            units.append(("neuron", i, p, int(torch.randint(pt.D, (1,), generator=g))) if torch.rand(1, generator=g) < 0.7 else
                         ("head", i, int(torch.randint(pt.H, (1,), generator=g)), p))
        orders.append(units)
    ks = [30, 2, 9, 0, 17]
    want = []
    for x, units in zip(corrupts, orders):
        pt.setup(clean, x)
        want.append(pt.both_prefixes(units, ks) + [tuple(v) for v in pt.both([units[:3]])])
    pts = [cf.Patcher(pt.model, None, rows=pt.rows) for _ in corrupts]
    for p_, x in zip(pts, corrupts):
        p_.setup(clean, x)
    got = cf.Group(pts).run([lambda p_=p_, u=u: p_.both_prefixes(u, ks) + [tuple(v) for v in p_.both([u[:3]])]
                             for p_, u in zip(pts, orders)])
    for a, b in zip(got, want):
        assert len(a) == len(b) and all(abs(x[0] - y[0]) < 1e-5 and abs(x[1] - y[1]) < 1e-5 for x, y in zip(a, b))


def test_both_without_equals_its_sets(tiny):
    """both_without(): each unit left out, as the sets themselves measure (neurons and heads, units before the change)."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    g = torch.Generator().manual_seed(7)
    units = []
    for _ in range(25):
        i, p = int(torch.randint(pt.L, (1,), generator=g)), int(torch.randint(pt.T, (1,), generator=g))
        units.append(("neuron", i, p, int(torch.randint(pt.D, (1,), generator=g))) if torch.rand(1, generator=g) < 0.7 else
                     ("head", i, int(torch.randint(pt.H, (1,), generator=g)), p))
    got = pt.both_without(units)
    want = pt.both([[v for k, v in enumerate(units) if k != j] for j in range(len(units))])
    for a, b in zip(got, want):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
    assert pt.both_without(units, [3, 0]) == [got[3], got[0]]


def _random_units(pt, g, n):
    units = []
    for _ in range(n):
        i, p = int(torch.randint(pt.L, (1,), generator=g)), int(torch.randint(pt.T, (1,), generator=g))
        units.append(("neuron", i, p, int(torch.randint(pt.D, (1,), generator=g))) if torch.rand(1, generator=g) < 0.7 else
                     ("head", i, int(torch.randint(pt.H, (1,), generator=g)), p))
    return units


def test_prefixes_with_drops_equal_their_sets(tiny):
    """both_prefixes(units, ks, drops): each prefix with one place left out (inside it, past it, at any length), as the
    sets themselves measure."""
    pt, clean, corrupt = tiny
    pt.setup(clean, corrupt)
    units = _random_units(pt, torch.Generator().manual_seed(11), 20)
    ks, drops = [20, 20, 7, 7, 13, 0, 1, 20], [0, 19, 3, 7, 12, 0, 0, 25]
    got = pt.both_prefixes(units, ks, drops)
    want = pt.both([[u for i, u in enumerate(units[:k]) if i != d] for k, d in zip(ks, drops)])
    for a, b in zip(got, want):
        assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5


def test_group_serve_equals_alone(tiny):
    """Group.serve(): plain sets, prefixes and prefixes with drops of several members in merged batches give each
    member's own calls (its shares by its own contrast and gap)."""
    pt, clean, corrupt = tiny
    corrupts = [corrupt, [(c + 5) % 64 if i == 6 else c for i, c in enumerate(corrupt)]]
    g = torch.Generator().manual_seed(13)
    orders = [_random_units(pt, g, 24) for _ in corrupts]
    pts = [cf.Patcher(pt.model, None, rows=pt.rows) for _ in corrupts]
    for p_, x in zip(pts, corrupts):
        p_.setup(clean, x)
    ks, drops = [24, 5, 24, 11], [24, 9, 3, 2]
    want = [(p_.both_prefixes(u, ks), p_.both_prefixes(u, ks, drops), [tuple(v) for v in p_.both([u[:4], u[5:9]])],
             [v[0] for v in p_._measure([u[:6]], ("corrupt",))]) for p_, u in zip(pts, orders)]
    calls = []
    for p_, u in zip(pts, orders):
        calls += [("prefix", p_, p_._arrays(u), ks, ("clean", "corrupt"), None),
                  ("prefix", p_, p_._arrays(u), ks, ("clean", "corrupt"), drops),
                  ("plain", p_, [u[:4], u[5:9]], ("clean", "corrupt"), None),
                  ("plain", p_, [u[:6]], ("corrupt",), None)]
    got = cf.Group(pts).serve(calls)
    for m, w in enumerate(want):
        for c in range(4):
            a, b = got[4 * m + c], w[c]
            a = [v[0] for v in a] if c == 3 else [tuple(v) for v in a]
            assert len(a) == len(b)
            for x, y in zip(a, b):
                x, y = (x, y) if c != 3 else ((x,), (y,))
                assert all(abs(p - q) < 1e-5 for p, q in zip(x, y))


def test_scores_many_equals_each_pairs_own(tiny):
    """scores_many(): pairs' base runs in one forward and backward give each pair's scores as computed from its own two
    rows (each direction's patch difference times its base run's gradient of the contrast, the larger magnitude)."""
    pt, clean, corrupt = tiny
    corrupts = [corrupt, [(c + 5) % 64 if i == 6 else c for i, c in enumerate(corrupt)]]
    pts = [cf.Patcher(pt.model, None, rows=pt.rows) for _ in corrupts]
    for p_, x in zip(pts, corrupts):
        p_.setup(clean, x)
    got = cf.scores_many(pts)
    for p_, sc in zip(pts, got):
        add = {"head": [torch.zeros(2, p_.T, p_.H * p_.hd, requires_grad=True) for _ in p_.layers],
               "neu": [torch.zeros(2, p_.T, p_.D, requires_grad=True) for _ in p_.layers]}
        x = torch.stack([p_._ids_dev["clean"], p_._ids_dev["corrupt"]])
        with torch.enable_grad(), cf._relp(p_.core):
            p_._contrast(p_._last(x, 2, add=add)).sum().backward()
        for k, key in (("head", "head"), ("neu", "neuron")):
            best = None
            for j, (base, src) in enumerate((("clean", "corrupt"), ("corrupt", "clean"))):
                s = torch.stack([(p_.act[src][k][i].float() - p_.act[base][k][i].float()) * add[k][i].grad[j].float()
                                 for i in range(p_.L)])
                if k == "head":
                    s = s.reshape(p_.L, p_.T, p_.H, p_.hd).sum(-1)
                best = s.abs() if best is None else torch.maximum(best, s.abs())
            assert sc[key].shape == best.shape and torch.allclose(sc[key], best, atol=1e-5, rtol=1e-4)
