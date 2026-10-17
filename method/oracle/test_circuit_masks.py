"""Checks of circuit_masks' multi-source options on a tiny random Qwen3 (circuit_finder's test model): per-pair hinges
equal the mean-share hinges when a cut has one pair, ALLCUTS puts every cut on its partners, WARM's start lies in
[WARM_MIN, INIT] with the best-attributed unit at INIT, and find() with all three gives a record whose cuts carry every
pair's shares.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_circuit_masks.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import circuit_finder as cf  # noqa: E402
import circuit_masks as cm  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    """A tiny model and a clean prompt with three corrupted ones (two changed tokens each) whose answers differ."""
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    pt = cf.Patcher(model, None, rows=8)
    for _ in range(200):
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupts = []
        for d in range(1, 30):
            corrupt = list(clean)
            corrupt[4], corrupt[7] = (clean[4] + 3 * d) % 64, (clean[7] + 5 * d) % 64
            rec = pt.setup(clean, corrupt)
            if rec["valid"] and rec["gap"] >= cf.MIN_GAP and not rec["same_answer"]:
                corrupts.append(corrupt)
            if len(corrupts) == 3:
                return model, clean, corrupts
    pytest.skip("no clean prompt with three valid corrupted ones")


def test_pairwise_equals_mean_hinge_with_one_pair(tiny, monkeypatch):
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    spec = cm.cuts_of(pt)["readout"]
    monkeypatch.setattr(cm, "PAIRWISE", 0)
    a = cm.optimize(pt, [spec], steps=3)[0]["theta"]["neuron"]
    monkeypatch.setattr(cm, "PAIRWISE", 1)
    b = cm.optimize(pt, [spec], steps=3)[0]["theta"]["neuron"]
    assert torch.allclose(a, b, atol=1e-6)


def test_allcuts_and_warm(tiny, monkeypatch):
    model, clean, corrupts = tiny
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    for p, c in zip(pts, corrupts):
        p.setup(clean, c)
    monkeypatch.setattr(cm, "ALLCUTS", 0)
    st = cm._before(pts[0], "mask", partners=pts[1:])
    assert isinstance(st["meas"]["source"], cm.Mean) and all(st["meas"][n] is pts[0] for n in st["names"] if n != "source")
    assert st["warm"] == {}
    monkeypatch.setattr(cm, "ALLCUTS", 1)
    monkeypatch.setattr(cm, "WARM", 0.5)
    st = cm._before(pts[0], "mask", partners=pts[1:])
    assert all(isinstance(st["meas"][n], cm.Mean) for n in st["names"])
    for n in st["masked"]:
        th = st["warm"][n]
        assert set(th) == set(st["specs"][n]["kinds"])
        top = max(float(t.max()) for t in th.values())
        assert abs(top - cm.INIT) < 1e-5 and min(float(t.min()) for t in th.values()) >= cm.WARM_MIN - 1e-6
        for k, t in th.items():
            assert t.shape == (pts[0].L, len(st["specs"][n]["positions"]), pts[0].H if k == "head" else pts[0].D)


def test_find_multi_source(tiny, monkeypatch):
    model, clean, corrupts = tiny
    for k, v in (("ALLCUTS", 1), ("PAIRWISE", 1), ("WARM", 0.5)):
        monkeypatch.setattr(cm, k, v)
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    rec = cm.find(pts[0], clean, corrupts[0], steps=5, partners=list(zip(pts[1:], corrupts[1:])))
    assert set(rec["partners"]) == set(rec["cuts"])
    for n, p in rec["partners"].items():
        assert p["pairs"] == 3 and len(p["shares"]) in (0, 3)
    for n, c in rec["cuts"].items():
        if c["met"]:
            assert c["lost"] >= cf.PASS and c["restored"] >= cf.PASS


def _one_at_a_time(pt, circ):
    """circuit_finder.cut's pruning rule, one trial a measurement: each unit whose removal alone passes, from the last,
    removed while the set without it passes."""
    passes = lambda r: r[0] >= cf.PASS and r[1] >= cf.PASS  # noqa: E731
    alone = pt.both([[v for v in circ if v != u] for u in circ])
    keep = list(circ)
    for u in [u for u, r in zip(circ, alone) if passes(r)][::-1]:
        trial = [v for v in keep if v != u]
        if passes(pt.both([trial])[0]):
            keep = trial
    return keep


def _passing(pt):
    """An attribution order of one of pt's cuts and its smallest passing prefix (None: no cut has one)."""
    passes = lambda r: r[0] >= cf.PASS and r[1] >= cf.PASS  # noqa: E731
    sc = cf.Patcher.scores(pt)
    for spec in cm.cuts_of(pt).values():
        order = cm.attr_order(pt, spec, sc)
        ks = list(range(1, len(order) + 1))
        k = next((k for k, r in zip(ks, pt.both_prefixes(order, ks)) if passes(r)), None)
        if k:
            return order, k
    return None, None


@pytest.mark.parametrize("window", [1, 2, 16])
def test_prune_equals_one_at_a_time(tiny, monkeypatch, window):
    """prune(): the windowed rounds (cumulative and single trials) remove exactly the units the one-at-a-time rule does."""
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "PRUNE_WINDOW", window)
    pt = cf.Patcher(model, None, rows=8)
    n = 0
    for c in corrupts:
        pt.setup(clean, c)
        order, k = _passing(pt)
        if k is None:
            continue
        circ = list(order[:min(len(order), 8 * k)])           # (a passing set with units to spare)
        got = cm.prune(pt, circ)
        assert got == _one_at_a_time(pt, circ) and len(got) < len(circ)
        n += 1
    if not n:
        pytest.skip("no passing prefix")


def test_verify_diffuse_above_prune_max(tiny, monkeypatch):
    """A passing prefix of more than PRUNE_MAX units is a diffuse cut: no units, its prefix's size and shares."""
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    order, k = _passing(pt)
    if k is None or k < 2:
        pytest.skip("no passing prefix of two or more units")
    monkeypatch.setattr(cm, "PRUNE_MAX", k - 1)
    d = cm.verify(pt, order)
    assert d["diffuse"] and not d["met"] and d["units"] == [] and d["spread"]["units"] == k
    assert d["lost"] >= cf.PASS and d["restored"] >= cf.PASS


@pytest.mark.parametrize("arm", ["am", "mask"])
def test_find_batch_equals_find(tiny, arm):
    """find_batch (pairs of one prompt length in lockstep, their masks in one batch) gives each pair's find() record:
    the same cuts, units and shares (float32)."""
    model, clean, corrupts = tiny
    alone = [cm.find(cf.Patcher(model, None, rows=8), clean, c, arm=arm, steps=3) for c in corrupts]
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    batch = cm.find_batch(pts, [(clean, c, None) for c in corrupts], arm=arm, steps=3)
    for a, b in zip(alone, batch):
        assert set(a["cuts"]) == set(b["cuts"])
        assert [(n["layer"], n["pos"], n["neuron"]) for n in a["nodes"]] == [(n["layer"], n["pos"], n["neuron"]) for n in b["nodes"]]
        assert [(r["layer"], r["head"]) for r in a["routes"]] == [(r["layer"], r["head"]) for r in b["routes"]]
        for n in a["cuts"]:
            x, y = a["cuts"][n], b["cuts"][n]
            assert x["met"] == y["met"] and abs(x["lost"] - y["lost"]) < 1e-4 and abs(x["restored"] - y["restored"]) < 1e-4

def test_adversary(tiny, monkeypatch):
    """The adversary moves only units outside the set: against every unit of a cut it has nothing to move (its worst step
    is the unmoved shares), its rho stays in [0, 1], and freeze leaves theta where it started."""
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    spec = cm.cuts_of(pt)["readout"]
    every = [cf.unit("neuron", i, p, k) for i in range(pt.L) for p in spec["positions"] for k in range(pt.D)]
    a = cm.adversarial(pt, spec, every, steps=3)
    assert abs(a["lost"] - a["unmoved"][0]) < 1e-5 and abs(a["restored"] - a["unmoved"][1]) < 1e-5
    full = pt.both([every])[0]
    assert abs(a["lost"] - full[0]) < 1e-3 and abs(a["restored"] - full[1]) < 1e-3
    th0 = {"neuron": torch.zeros(pt.L, len(spec["positions"]), pt.D)}
    out = cm.optimize_many([(pt, [spec], None)], steps=3, init=[[th0]], adv=0.3, freeze=True)[0][0]
    assert torch.equal(out["theta"]["neuron"], th0["neuron"])
    assert float(out["rho"]["neuron"].min()) >= 0 and float(out["rho"]["neuron"].max()) <= 1
    one = cm.adversarial(pt, spec, every[:1], steps=3)
    assert one["lost"] + one["restored"] <= sum(one["unmoved"]) + 1e-5


def test_orders_decode_their_units():
    """attr_order and ordered: the units of the top flat indices, decoded by array arithmetic, are those a unit-by-unit
    decode gives (two kinds, several positions)."""
    torch.manual_seed(3)
    spec = {"kinds": ("neuron", "head"), "positions": [4, 6, 9]}
    sc = {"neuron": torch.rand(3, 12, 7), "head": torch.rand(3, 12, 5)}

    class P:
        pass
    got = cm.attr_order(P(), spec, sc)
    v = torch.cat([sc[k][:, spec["positions"]].reshape(-1) for k in spec["kinds"]])
    want = []
    for f in v.topk(len(v)).indices.tolist():
        at = 0
        for k in spec["kinds"]:
            L, P_, W = sc[k][:, spec["positions"]].shape
            if f < at + L * P_ * W:
                r = f - at
                want.append(cf.unit(k, r // (P_ * W), spec["positions"][(r // W) % P_], r % W))
                break
            at += L * P_ * W
    assert got == want
    theta = {k: sc[k][:, spec["positions"]] - 0.5 for k in spec["kinds"]}
    units, kept = cm.ordered(spec, theta)
    assert kept == int(sum((t > 0).sum() for t in theta.values())) and units == want[:len(units)]


def test_adversary_modes(tiny, monkeypatch):
    """Block and random adversaries: against every unit of a cut nothing moves (block and unit), random draws stay within
    the shares between the unmoved set and the whole cut for a monotone cut, and the record carries every mode."""
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    spec = cm.cuts_of(pt)["readout"]
    every = [cf.unit("neuron", i, p, k) for i in range(pt.L) for p in spec["positions"] for k in range(pt.D)]
    monkeypatch.setattr(cm, "ADV_MODES", "unit+block+random")
    a = cm.adversarial(pt, spec, every, steps=3)
    assert set(a["modes"]) == {"unit", "block", "random"} and a["mode"] == "unit"
    full = pt.both([every])[0]
    for m in ("unit", "block", "random"):
        assert abs(a["modes"][m]["lost"] - full[0]) < 1e-3 and abs(a["modes"][m]["restored"] - full[1]) < 1e-3
    th = {"neuron": torch.zeros(pt.L, len(spec["positions"]), pt.D)}
    out = cm.optimize_many([(pt, [spec], None)], steps=2, init=[[th]], adv=0.3, freeze=True, mode="block")[0][0]
    assert out["rho"]["neuron"].shape == (pt.L, len(spec["positions"]), 1)


def test_merged_mask_passes_equal_separate(tiny, monkeypatch):
    """optimize_many with every first position its own forward (MASK_MERGE 0) and with all in one forward from the
    lowest (MASK_MERGE large) gives the same masks and histories: positions before an instance's first are its base
    run's."""
    model, clean, corrupts = tiny
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    for p, c in zip(pts, corrupts):
        p.setup(clean, c)
    jobs = [(p, list(cm.cuts_of(p).values()), None) for p in pts] + [(pts, [cm.cuts_of(pts[0])["source"]], None)]
    got = []
    for merge in (0, 10 ** 9):
        monkeypatch.setattr(cm, "MASK_MERGE", merge)
        got.append(cm.optimize_many(jobs, steps=3))
    for a, b in zip(*got):
        for x, y in zip(a, b):
            assert all(torch.allclose(x["theta"][k], y["theta"][k], atol=1e-4) for k in x["theta"])
            assert all(abs(u - v) < 1e-3 for h, i in zip(x["history"], y["history"]) for u, v in zip(h, i))
    assert len(cm._passes([5, 5, 7, 9], 10)) >= 1 and cm._passes([3], 10) == [(3, [0])]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graph_mask_steps_equal_eager(tiny, monkeypatch):
    """optimize_many's steps replayed from a CUDA graph give the eager steps' masks and histories (pairs, partners and
    the answer-position cuts on the GPU)."""
    import copy
    model, clean, corrupts = tiny
    model = copy.deepcopy(model).to("cuda")
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    for p, c in zip(pts, corrupts):
        p.setup(clean, c)
    jobs = [(p, list(cm.cuts_of(p).values()), None) for p in pts] + [(pts, [cm.cuts_of(pts[0])["source"]], None)]
    got = []
    for graph in (False, True):
        monkeypatch.setattr(cm, "GRAPH", graph)
        got.append(cm.optimize_many(jobs, steps=6))
    for a, b in zip(*got):
        for x, y in zip(a, b):
            assert all(torch.allclose(x["theta"][k], y["theta"][k], atol=1e-4) for k in x["theta"])
            assert all(abs(u - v) < 1e-3 for h, i in zip(x["history"], y["history"]) for u, v in zip(h, i))


def test_agg_min(tiny, monkeypatch):
    """AGG "min": a Mean's shares are the least over its pairs, at most the means."""
    model, clean, corrupts = tiny
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    for p, c in zip(pts, corrupts):
        p.setup(clean, c)
    m = cm.Mean(pts)
    sets = [[("neuron", 1, pts[0].T - 1, k) for k in range(8)], [("head", 1, h, pts[0].T - 1) for h in range(2)]]
    each = [p.both(sets) for p in pts]
    monkeypatch.setattr(cm, "AGG", "mean")
    mean = cm.drive(pts, [(lambda: (yield (m, "both", sets)))()])[0]
    monkeypatch.setattr(cm, "AGG", "min")
    low = cm.drive(pts, [(lambda: (yield (m, "both", sets)))()])[0]
    for i in range(len(sets)):
        for x in (0, 1):
            assert abs(low[i][x] - min(e[i][x] for e in each)) < 1e-4 and low[i][x] <= mean[i][x] + 1e-6


def test_default_schedule_meets_like_long(tiny, monkeypatch):
    """The default schedule (STEPS from INIT, the hinges acting within a few steps) meets every cut the long one (30 steps
    from 3, the earlier default) meets on the tiny pair."""
    model, clean, corrupts = tiny
    assert cm.STEPS == 15 and cm.INIT == 1.0
    short = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], None)
    monkeypatch.setattr(cm, "INIT", 3.0)
    long_ = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], None, steps=30)
    for n, c in long_["cuts"].items():
        if c["met"]:
            assert short["cuts"][n]["met"], n


def test_no_cuts_not_met(tiny):
    """A usable pair with no cut searched has no circuit: its record is not met (a lockstep search that returned no cuts
    read as met; persona's identity/1/dolphin/neutral)."""
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    rec = cf._find(pt, clean, corrupts[0], None, searched={})
    assert rec["usable"] and rec["cuts"] == {} and not rec["met"]


def test_family_movers(tiny):
    """A template's mover union that passes on the pair is the mover cut whole (no pruning), with its family record;
    family_heads takes the paraphrase's union when it has FAMILY_MIN_KEYS records, else the task's, above FAMILY_FREQ."""
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    every = [[i, h, 1.0] for i in range(pt.L) for h in range(pt.H)]
    U = {"t": {"p0": every, "all": every[:2], "n": {"p0": 5, "all": 5}}}
    assert cm.family_heads(U, "t/k/p0") == [(i, h) for i, h, _ in every]
    assert cm.family_heads(U, "t/k/p1") == [(i, h) for i, h, _ in every[:2]]
    assert cm.family_heads(U, "u/k/p0") is None
    assert cm.default_unions("tiny") is None         # (the default unions are UNIONS_MODEL's only)
    cuts = cm.search_all(pt, "mask", steps=3, family=cm.family_heads(U, "t/k/p0"))
    mv = cuts["movers"]
    assert mv["reachable"]                          # (every head at the answer: the movers' ceiling)
    assert mv["family"]["taken"] and len(mv["units"]) == pt.L * pt.H and mv["met"]


def test_clean_side(tiny, monkeypatch):
    """CLEAN_SIDE: no unit active only in the corrupted run (circuit_finder.ACTIVE_RATIO) is ever in a mask or a cut."""
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "CLEAN_SIDE", 1)
    rec = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], None, steps=5)
    assert rec["usable"]
    assert all(x["active_in"] != "corrupt" for x in rec["nodes"] + rec["routes"])
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    spec = cm.cuts_of(pt)["readout"]
    th = cm.optimize(pt, [spec], steps=3)[0]["theta"]["neuron"]           # [L, 1, D]
    p = spec["positions"][0]
    for i in range(pt.L):
        a, b = pt.act["clean"]["neu"][i][p].abs(), pt.act["corrupt"]["neu"][i][p].abs()
        assert bool((th[i, 0][b >= cf.ACTIVE_RATIO * a] == -30.0).all())



def test_extend_cut(tiny, monkeypatch):
    """EXTEND: a met cut (EXTEND_CUT; the tiny model's source cuts are unreachable, so its readout here) is grown with its
    own order's next units to EXTEND units, its shares and each unit's measured."""
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "EXTEND_CUT", "readout")
    rec = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], None, steps=30)
    ro = rec["cuts"]["readout"]
    assert ro["met"]
    n = ro["size"] + 3
    monkeypatch.setattr(cm, "EXTEND", float(n))        # (as an arm constant comes: extend=25 -> 25.0)
    big = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], None, steps=30)
    assert big["cuts"]["readout"]["size"] == n and big["mask"]["readout"]["extended"] == n
    assert sum(x["stage"] == "readout" for x in big["nodes"]) == n


def test_source_sufficiency_only(tiny, monkeypatch):
    """SOURCE_NECESSITY 0: the source cut's bar is (-inf, bar) and its lost hinge target is off; a set passes on its
    restored share alone."""
    model, clean, corrupts = tiny
    pt = cf.Patcher(model, None, rows=8)
    pt.setup(clean, corrupts[0])
    monkeypatch.setattr(cm, "SOURCE_NECESSITY", 0)
    st = cm._before(pt, "mask")
    assert st["bars"]["source"][0] == -float("inf")
    assert "source" not in st["taus"] or st["taus"]["source"][0] < -1e8
    assert cm._ok((-5.0, 0.8), (-float("inf"), 0.7)) and not cm._ok((0.9, 0.6), (-float("inf"), 0.7))
    assert cm._ok((0.8, 0.8), 0.7) and not cm._ok((0.6, 0.8), 0.7)


def _pairs_of_length(model, n, T, seed):
    """A clean prompt of T tokens and n corrupted ones (two changed tokens each) whose answers differ, on the tiny model."""
    g = torch.Generator().manual_seed(seed)
    pt = cf.Patcher(model, None, rows=8)
    for _ in range(200):
        clean = torch.randint(0, 64, (T,), generator=g).tolist()
        out = []
        for d in range(1, 30):
            c = list(clean)
            c[T // 2 - 1], c[T - 3] = (clean[T // 2 - 1] + 3 * d) % 64, (clean[T - 3] + 5 * d) % 64
            r = pt.setup(clean, c)
            if r["valid"] and r["gap"] >= cf.MIN_GAP and not r["same_answer"]:
                out.append(c)
            if len(out) == n:
                return clean, out
    return None, None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("allcuts", [0, 1])
def test_graph_find_batch_with_partners_equals_eager(tiny, monkeypatch, allcuts):
    """find_batch and find with partners, prompts of two lengths in a row: graph-replayed mask steps (GRAPH_PARTNERS)
    give the eager records (cuts, units, shares)."""
    import copy
    model = copy.deepcopy(tiny[0]).to("cuda")
    items, parts = [], []
    for T, seed in ((10, 3), (13, 4)):
        clean, cs = _pairs_of_length(model, 3, T, seed)
        if clean is None:
            pytest.skip("no prompt with three valid corrupted ones")
        items.append((clean, cs[0], None))
        parts.append([(cf.Patcher(model, None, rows=8), c) for c in cs[1:]])
    monkeypatch.setattr(cm, "GRAPH_PARTNERS", 1)
    monkeypatch.setattr(cm, "ALLCUTS", allcuts)
    got = []
    for graph in (False, True):
        monkeypatch.setattr(cm, "GRAPH", graph)
        recs = cm.find_batch([cf.Patcher(model, None, rows=8) for _ in items], items, arm="mask", steps=6, partners=parts)
        one = cm.find(cf.Patcher(model, None, rows=8), *items[1][:2], arm="mask", steps=6,
                      partners=[(p, c) for p, c in [(cf.Patcher(model, None, rows=8), x) for _, x in parts[1]]])
        got.append(recs + [one])
    for a, b in zip(*got):
        assert set(a["cuts"]) == set(b["cuts"])
        assert [(n["layer"], n["pos"], n["neuron"]) for n in a["nodes"]] == [(n["layer"], n["pos"], n["neuron"]) for n in b["nodes"]]
        for n in a["cuts"]:
            assert abs(a["cuts"][n]["lost"] - b["cuts"][n]["lost"]) < 1e-3 and abs(a["cuts"][n]["restored"] - b["cuts"][n]["restored"]) < 1e-3


def test_generalizing_defaults():
    """The default for generalizing circuits (the lead's 10-17 decision): every cut on the pairs, hinges per pair, a set
    taken only when every pair passes; the command line's default pool is trainset_8b."""
    assert cm.ALLCUTS == 1 and cm.PAIRWISE == 1 and cm.AGG == "min"
    assert cm.POOL.name == "trainset_8b.json"


@pytest.mark.parametrize("background", ["keys", "mean"])
def test_one_prompt_find(tiny, monkeypatch, background):
    """find() in one prompt: each cut's record names the answer and the rivals; a met cut keeps at least its bar of the
    lead with the rest of the cut resampled (re-measured on the keys); BACKGROUND mean resamples from the keys' mean, whose
    other run is the mean of theirs."""
    import numpy as np
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "ONE_PROMPT", 1)
    monkeypatch.setattr(cm, "AGG", "mean")
    monkeypatch.setattr(cm, "BACKGROUND", background)
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    rec = cm.find(pts[0], clean, corrupts[0], steps=8, partners=list(zip(pts[1:], corrupts[1:])))
    op = rec["one_prompt"]
    assert rec["cuts"] and set(op) == set(rec["cuts"]) and all(x["answer"] == rec["answer"] and x["background"] == background
                                                               for x in op.values())
    if background == "mean":
        m = cm._mean_patcher(pts)
        want = torch.stack([p.act["corrupt"]["neu"][1].float() for p in pts]).mean(0)
        assert torch.allclose(m.act["corrupt"]["neu"][1].float(), want, atol=1e-2) and m.act["clean"] is pts[0].act["clean"]
        return
    for n, c in rec["cuts"].items():
        if not c["met"]:
            continue
        units = ([cf.unit("neuron", x["layer"], x["pos"], x["neuron"]) for x in rec["nodes"] if x["stage"] == n]
                 + ([cf.unit("head", x["layer"], x["pos"], x["head"]) for x in rec["routes"]] if n == "movers" else []))
        spec = {**cm.cuts_of(pts[0])[n], "positions": c["searched"]}
        got = cm.drive(pts, [_ask(cm.Resampled(pts, spec), [units])])[0][0]
        assert got[1] >= c["bar"][1] - 1e-3 and abs(got[1] - c["restored"]) < 1e-3


def test_resampled_check(tiny, monkeypatch):
    """resampled_check on the keys a one-prompt circuit was found on gives each met cut its record's kept share; a cut
    with no units is not tested."""
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "ONE_PROMPT", 1)
    monkeypatch.setattr(cm, "AGG", "mean")
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    rec = cm.find(pts[0], clean, corrupts[0], steps=8, partners=list(zip(pts[1:], corrupts[1:])))
    chk = cm.resampled_check(pts, rec)
    assert chk["keys"] == len(pts) and set(chk["cuts"]) == {n for n in rec["cuts"] if cm.record_cut_units(rec, n)}
    for n, c in chk["cuts"].items():
        assert abs(c["kept"] - rec["cuts"][n]["restored"]) < 1e-3 and len(c["per_key"]) == len(pts)
    assert chk["passes"] == all(c["kept"] >= cf.PASS for c in chk["cuts"].values())


def test_necessary_units(tiny, monkeypatch):
    """NECESSARY: after the search, each cut's first NECESSARY_TOP units in RelP order outside it are noised alone on the
    own pair; those losing at least NECESSARY are in the record's "necessary" with their measured shares, never a cut's
    own unit; NECESSARY_TOP 0 writes none and leaves the circuit as it was."""
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "NECESSARY", -1.0)     # (every candidate listed, to check the shares and the exclusion)
    monkeypatch.setattr(cm, "NECESSARY_TOP", 8)
    pt = cf.Patcher(model, None, rows=8)
    rec = cm.find(pt, clean, corrupts[0], steps=4)
    assert set(rec["necessary"]) == set(rec["cuts"])
    own = {("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"]} | {("head", r["layer"], r["head"], r["pos"]) for r in rec["routes"]}
    pt.setup(clean, corrupts[0])
    for n, xs in rec["necessary"].items():
        assert 0 < len(xs) <= 8
        assert [x["lost"] for x in xs] == sorted((x["lost"] for x in xs), reverse=True)
        units = [cf.unit("head", x["layer"], x["pos"], x["head"]) if "head" in x else cf.unit("neuron", x["layer"], x["pos"], x["neuron"])
                 for x in xs]
        assert not own & set(units)
        for x, (lo, re_) in zip(xs, pt.both([[u] for u in units])):
            assert abs(x["lost"] - lo) < 1e-3 and abs(x["restored"] - re_) < 1e-3
    monkeypatch.setattr(cm, "NECESSARY_TOP", 0)
    off = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], steps=4)
    assert "necessary" not in off
    assert [(n["layer"], n["pos"], n["neuron"]) for n in off["nodes"]] == [(n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"]]


def test_one_prompt_without_partners(tiny, monkeypatch):
    """One prompt with no partner of the prompt's length resamples from the question's own corrupted key alone."""
    model, clean, corrupts = tiny
    monkeypatch.setattr(cm, "ONE_PROMPT", 1)
    rec = cm.find(cf.Patcher(model, None, rows=8), clean, corrupts[0], steps=5)
    assert rec["cuts"] and all(x["keys"] == 1 for x in rec["one_prompt"].values())


def _ask(meas, sets):
    return (yield (meas, "both", sets))


def test_one_prompt_resampled_shares(tiny, monkeypatch):
    """Resampled: a set's shares are, per pair and in the clean run, the lead lost with the set resampled and the lead
    kept with the set frozen at its own values and the rest of the cut resampled (Patcher.run), averaged over the keys;
    the whole cut keeps all of the lead; one prompt's token sets give each pair its lead over a zero baseline; the masks'
    clean row with every unit kept is the full model, with none every cut unit at the key's value."""
    import numpy as np
    model, clean, corrupts = tiny
    pts = [cf.Patcher(model, None, rows=8) for _ in corrupts]
    for p, c in zip(pts, corrupts):
        p.setup(clean, c)
    monkeypatch.setattr(cm, "ONE_PROMPT", 1)
    monkeypatch.setattr(cm, "AGG", "mean")
    info = cm._one_prompt_sets(pts[0], pts[1:])
    assert info["answer"] == int(pts[0]._lp["clean"].argmax()) and all(p.c["corrupt"] == 0.0 and p.gap == p.c["clean"] > 0 for p in pts)
    spec = cm.cuts_of(pts[0])["readout"]
    meas = cm.Resampled(pts, spec)
    pos = spec["positions"][0]
    units = [cf.unit("neuron", i, pos, k) for i in range(pts[0].L) for k in range(0, pts[0].D, 3)]
    every = [cf.unit("neuron", i, p, torch.arange(pts[0].D)) for i in range(pts[0].L) for p in spec["positions"]]
    comp = meas.complement(units)
    assert sum(len(c[3]) for c in comp) + len(units) == pts[0].L * pts[0].D and meas.complement(every) == []
    got = cm.drive(pts, [_ask(meas, [units, every])])[0]
    lo = float(np.mean([p.run("clean", [units])[0] for p in pts]))
    kept = float(np.mean([1 - p.run("clean", [comp + [u + ("clean",) for u in units]])[0] for p in pts]))
    assert abs(got[0][0] - lo) < 1e-4 and abs(got[0][1] - kept) < 1e-4
    assert got[1][1] == 1.0 and abs(got[1][0] - float(np.mean([p.run("clean", [every])[0] for p in pts]))) < 1e-4
    for v, want in ((30.0, 1.0), (-30.0, float(np.mean([1 - p.run("clean", [every])[0] for p in pts])))):
        hist = cm.optimize_many([(pts, [spec], None)], steps=1, init=[[{"neuron": torch.full((pts[0].L, 1, pts[0].D), v)}]])[0][0]["history"]
        assert abs(hist[0][1] - want) < 1e-3
