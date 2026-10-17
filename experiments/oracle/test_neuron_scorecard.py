"""Checks of the circuit finder's scorecard (neuron_scorecard.py) on a tiny random Qwen3: unpatched runs are the model's;
neuron and head patches equal hand-written hooks; padding a batch changes no run; the patching pair's shares follow the
definitions; stability weighs the units shared across paraphrases by their measured shares (hubs beside it); the margins read the records.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_neuron_scorecard.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neuron_scorecard as ns  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    clean = torch.randint(0, 48, (10,), generator=g).tolist()
    corr = list(clean)
    corr[3] = (corr[3] + 7) % 48
    return model, ns.Model(model), clean, corr


def hooked(model, ids, neurons=(), heads=()):
    """log p at the last position with neuron values [(layer, pos, neuron, value)] and head outputs [(layer, pos, head,
    value [hd])] set by hand."""
    hs = []
    for L, layer in enumerate(model.model.layers):
        def pre_n(m, a, L=L):
            x = a[0].clone()
            for l, p, j, v in neurons:
                if l == L:
                    x[0, p, j] = v
            return (x,)

        def pre_h(m, a, L=L):
            x = a[0].clone().view(1, a[0].shape[1], 4, 8)
            for l, p, h, v in heads:
                if l == L:
                    x[0, p, h] = v
            return (x.view(1, a[0].shape[1], -1),)
        hs += [layer.mlp.down_proj.register_forward_pre_hook(pre_n), layer.self_attn.o_proj.register_forward_pre_hook(pre_h)]
    try:
        with torch.no_grad():
            return torch.log_softmax(model(torch.tensor([ids])).logits[0, -1].float(), -1)
    finally:
        for h in hs:
            h.remove()


def test_runs_are_the_models(tiny):
    model, m, clean, corr = tiny
    lp = m.logp([(clean, []), (corr, [])])
    assert torch.allclose(lp[0], hooked(model, clean), atol=1e-5)
    assert torch.allclose(lp[1], hooked(model, corr), atol=1e-5)


def test_patches_equal_hooks_and_padding_changes_nothing(tiny):
    model, m, clean, corr = tiny
    cap = m.capture(corr)
    c = {"neurons": [(0, 3, 5, 1), (2, 9, 17, -1)], "heads": [(1, 9, 2)]}
    pt = ns.patches(c, cap)
    want = hooked(model, clean, neurons=[(0, 3, 5, cap["n"][0, 3, 5]), (2, 9, 17, cap["n"][2, 9, 17])],
                  heads=[(1, 9, 2, cap["h"][1, 9, 2])])
    one = m.logp([(clean, pt)])[0]
    many = m.logp([(corr, []), (clean, pt), (clean, [])])[1]
    assert torch.allclose(one, want, atol=1e-5)
    assert torch.allclose(many, want, atol=1e-5)
    # the capture is the run's own values: patching a run with itself is the run
    own = m.capture(clean)
    assert torch.allclose(m.logp([(clean, ns.patches(c, own))])[0], hooked(model, clean), atol=1e-5)


def test_map_units_by_role():
    # clean / corr: 10 tokens, first change at 3; the paraphrase: 12 tokens, its change at 5, the same last 4 tokens
    clean, corr = [1, 2, 3, 4, 5, 6, 90, 91, 92, 93], [1, 2, 3, 7, 5, 6, 90, 91, 92, 93]
    fi, fc = [8, 8, 8, 8, 8, 4, 5, 6, 90, 91, 92, 93], [8, 8, 8, 8, 8, 7, 5, 6, 90, 91, 92, 93]
    c = {"neurons": [(0, 3, 1, 1), (1, 5, 2, -1), (2, 9, 3, 1), (2, 1, 4, 1)], "heads": [(3, 9, 0)]}
    mp = ns.map_units(c, clean, corr, fi, fc)
    assert mp["neurons"] == [(0, 5, 1, 1), (1, 7, 2, -1), (2, 11, 3, 1)]  # (the unit before the change is left out)
    assert mp["heads"] == [(3, 11, 0)]


def test_pair_shares_follow_the_definitions(tiny):
    model, m, clean, corr = tiny
    c = {"name": "x", "neurons": [(0, 3, 5, 1), (1, 3, 9, 1)], "heads": [(2, 9, 1)], "edges": [], "model": "tiny",
         "clean": clean, "corrupted": corr}

    class Tok:
        def decode(self, z):
            return str(z)
    r = ns.score(m, Tok(), c, {}, log=lambda s: None)
    pr = r["pair"]
    lc, lx = hooked(model, clean), hooked(model, corr)
    b = int(lx.argmax())
    sets = ns.ac.token_sets(model, Tok(), clean, lc, b)  # the word-level contrast (answer_contrast), as the scorecard's
    assert r["answer_tokens"] == sets["answer"] and r["alternative_tokens"] == sets["alternatives"]
    cc = float(ns.ac.contrast(lc[None], sets)[0])
    assert pr["c_clean"] == pytest.approx(cc, abs=1e-4)
    capx = m.capture(corr)
    ln = hooked(model, clean, neurons=[(0, 3, 5, capx["n"][0, 3, 5]), (1, 3, 9, capx["n"][1, 3, 9])], heads=[(2, 9, 1, capx["h"][2, 9, 1])])
    cn = float(ns.ac.contrast(ln[None], sets)[0])
    assert pr["c_noised"] == pytest.approx(cn, abs=1e-4)
    if pr["gap"] > 0:
        assert pr["lost"] == pytest.approx((pr["c_clean"] - pr["c_noised"]) / pr["gap"])
        assert pr["restored"] == pytest.approx((pr["c_denoised"] - pr["c_corrupted"]) / pr["gap"])
    assert len(r["effects"]) == 3 and r["size"]["poles"] == 2 and r["size"]["heads"] == 1
    # zero ablation and the whole distribution's KL (metrics no method optimizes)
    lz = hooked(model, clean, neurons=[(0, 3, 5, 0.0), (1, 3, 9, 0.0)], heads=[(2, 9, 1, torch.zeros(8))])
    if pr["gap"] > 0:
        assert r["zero"]["lost"] == pytest.approx((cc - float(ns.ac.contrast(lz[None], sets)[0])) / pr["gap"], abs=1e-4)
    klb = lambda a, b: float((a.exp() * (a - b)).sum()) / 0.6931471805599453  # noqa: E731
    assert r["kl"]["corrupted"] == pytest.approx(klb(lc, lx), rel=1e-4, abs=1e-6)
    assert r["kl"]["lost"] == pytest.approx(klb(lc, ln) / klb(lc, lx), rel=1e-3, abs=1e-5)
    # the random same-size baseline: RANDOM_DRAWS draws, the same on a rerun
    assert len(r["random"]["draws"]) == ns.RANDOM_DRAWS
    assert ns.score(m, Tok(), c, {}, log=lambda s: None)["random"]["lost"] == pytest.approx(r["random"]["lost"], abs=1e-6)
    # the curves end at the whole circuit: the pair's own shares (other batches: float32 rounding over a small gap)
    nec, suf = r["curve"]["necessity"], r["curve"]["sufficiency"]
    assert nec["k"] == [1, 2, 3] and nec["share"][-1] == pytest.approx(pr["lost"], abs=1e-3)
    assert suf["share"][-1] == pytest.approx(pr["restored"], abs=1e-3)
    # the first unit added is the one whose noising alone drops the contrast most
    drops = [e["drop"] for e in r["effects"]]
    assert nec["share"][0] == pytest.approx(max(drops) / pr["gap"], abs=1e-3)


def test_stability_and_margins():
    def rec(key, para, hubs, units):  # units: (layer, neuron, pole, drop alone), at position 5
        return {"task": f"{key}{para}", "question_task": "capital", "key": key, "builder": "b", "hubs": hubs,
                "neuron_set": [[L, j, s] for L, j, s, _ in units],
                "effects": [{"kind": "n", "unit": [L, 5, j, s], "drop": d} for L, j, s, d in units] + [{"kind": "h", "unit": [7, 5, 0], "drop": -2.0}],
                "pair": {"valid": True, "lost": 0.8, "restored": 0.6}, "size": {"poles": 10, "heads": 20, "neurons": 10, "positions": 1, "edges": 0},
                "build_seconds": 5.0,
                "draws": {"necessity_heldout": [{"share": 0.9}, {"share": 0.5}], "sufficiency_heldout": [], "swaps": [{"share": 1.4}]}}
    rs = [rec("France", 0, [[1, 2, 1], [3, 4, -1]], [(1, 2, 1, 3.0), (3, 4, -1, 1.0)]), rec("France", 1, [[1, 2, 1]], [(1, 2, 1, 1.0)]),
          rec("France", 2, [[1, 2, 1]], [(1, 2, 1, 1.0), (3, 4, -1, 1.0)]), rec("Japan", 0, [[1, 2, 1]], [(1, 2, 1, 1.0)])]
    ns.stability(rs)
    assert [x["share"] for x in rs[0]["stability"]] == pytest.approx([0.75, 1.0])  # (the head's negative drop weighs nothing)
    assert [x["share"] for x in rs[0]["stability_hubs"]] == [1.0, 0.5]
    assert rs[3]["stability"] == [] and rs[3]["stability_hubs"] == []
    m = rs[0]["margins"]
    assert m["stability"]["score"] == pytest.approx(0.875) and m["stability"]["pass_old"]
    assert ns.summary(rs)["stability_hubs"] == {"score": 1.0, "pass": 3, "unscored": 1}  # (medians 0.75, 1, 1)
    assert m["necessity"]["pass_old"] and m["sufficiency"]["pass_old"] is False
    assert m["necessity_heldout"]["score"] == pytest.approx(0.7) and m["necessity_heldout"]["se"] > 0
    assert m["sufficiency_heldout"]["score"] is None and m["sufficiency_heldout"]["pass_new"] is None
    assert m["swaps"]["score"] == 1.0
    assert "poles" not in m and "heads" not in m  # size is measured, never a criterion
    assert rs[3]["margins"]["stability"]["score"] is None


def test_every_family_needs_every_family_measured():
    full = {"task": "a", "pair": {"valid": True, "lost": 0.9, "restored": 0.9}, "size": {"poles": 8, "heads": 4, "neurons": 8,
            "positions": 2, "edges": 0}, "build_seconds": 5.0, "stability": [{"share": 1.0}],
            "draws": {"necessity_heldout": [{"share": 0.9}], "sufficiency_heldout": [{"share": 0.9}], "swaps": [{"share": 0.9}]}}
    assert ns.summary([full])["every_family"]["pass_old"] == 1
    gap = dict(full, stability=[])  # one paraphrase scored: stability unscored, so not every family is met
    assert ns.summary([gap])["every_family"]["pass_old"] == 0


def test_grid_and_curve_crossing():
    assert ns.grid(0) == [] and ns.grid(5) == [1, 2, 3, 4, 5]
    g = ns.grid(1000)
    assert g[:256] == list(range(1, 257)) and g[-1] == 1000 and len(g) < 270

    class Count:  # a run's "log p" is the number of patches it holds
        def logp(self, runs):
            return [torch.tensor([float(len(p))]) for _, p in runs]
    for n, scale, want in ((100, 100, 70), (10000, 5000, 3500)):
        cv = ns.curve(Count(), [0], list(range(n)), lambda lp, s=scale: float(lp[0]) / s)
        assert cv["k70"] == want and cv["units"] == n
    assert ns.curve(Count(), [0], list(range(10)), lambda lp: float(lp[0]) / 100)["k70"] is None


def test_motif_scores():
    def rec(i, poles):
        return {"id": i, "nodes": [{"layer": L, "pos": 3, "neuron": n, "pole": s} for L, n, s in poles]}
    recs = {"a": rec("a", [(1, 2, 1), (3, 4, -1)]), "b": rec("b", [(1, 2, 1)]), "c": rec("c", [(5, 6, 1)])}
    run = {"damage": {"test": ["a", "b", "c"], "rows": [
        {"kind": "group", "rank": 0, "poles": ["1.2.+", "3.4.-"], "damage": [2.0, 1.0, 0.0]},   # usage 1, 0.5, 0: tracks damage
        {"kind": "random", "rank": 0, "poles": ["1.2.+", "3.4.-"], "damage": [0.0, 1.0, 2.0]},  # the reverse
        {"kind": "random", "rank": 1, "poles": ["9.9.+"], "damage": [1.0, 2.0, 3.0]}]}}          # used by no question
    ms = ns.motif_scores(run, recs)
    assert ms["group_pearson_median"] == pytest.approx(1.0) and ms["random_pearson_median"] == pytest.approx(-1.0)
    assert ms["random_scored"] == 1 and ms["sets"][2]["pearson"] is None


def test_interactions_additivity_and_backups():
    import math as _m
    A, B, C = ("n", (0, 1, 2, 1)), ("n", (1, 1, 3, 1)), ("h", (1, 2, 0))  # A and B redundant (either carries 7 bits), C adds 3

    class Fake:
        def logp(self, runs):
            out = []
            for ids, ps in runs:
                have = {(p[0], p[1], p[2], p[3]) for p in ps}
                hit = lambda u: (u[0], *u[1][:3]) in have  # noqa: E731
                d = 3.0 * hit(C) + 7.0 * (hit(A) and hit(B))
                c = 10.0 - d if ids == "clean" else d
                out.append(torch.tensor([c * _m.log(2), 0.0]))
            return out
    cap = {"n": torch.zeros(2, 4, 5), "h": torch.zeros(2, 4, 3, 2)}
    eff = [{"kind": k, "unit": list(u), "drop": 3.0 if k == "h" else 0.0, "gain": 3.0 if k == "h" else 0.0} for k, u in (A, B, C)]
    base = {"clean": "clean", "corr": "corr", "cap_c": cap, "cap_x": cap, "sets": {"answer": [0], "alternatives": [1]},
            "cc": 10.0, "cx": 0.0, "gap": 10.0}
    it = ns.interactions(Fake(), base, eff)
    nz = it["noising"]
    assert len(nz["pairs"]) == 3 and nz["interaction_share"] > 0.5  # the redundant pair: 0.7 jointly, 0 summed
    assert sorted(round(a, 3) for _, _, a, _ in nz["pairs"]) == [0.3, 0.3, 0.7]
    bk = it["backups"]
    assert bk["count"] == 1 and bk["cumulative_lost"] == pytest.approx([0.3, 0.3])
    assert [u["backup"] for u in bk["units"]] == [False, False, True]  # the second of the redundant pair, once the first is gone


def test_per_cut_heldout_draws_and_size_curves(tiny, monkeypatch):
    """Each cut alone on a held-out pair (the per-cut unseen-pair test) and size_curves follow the definitions."""
    model, m, clean, corr = tiny
    monkeypatch.setattr(ns, "pair_ok", lambda sets, b, gap: abs(gap) > 1e-3)  # (a random model's pairs: shares only)

    class Tok:
        def decode(self, z):
            return str(z)

        def apply_chat_template(self, msgs, **kw):
            return msgs[-1]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [int(x) for x in text.split()]}
    part = list(clean)
    part[3] = (part[3] + 11) % 48
    a = int(hooked(model, clean).argmax())
    sets = {"answer": [a], "alternatives": [t for t in range(48) if t != a][:5]}
    c = {"name": "x", "id": "q", "neurons": [(0, 3, 5, 1)], "heads": [(2, 9, 1)], "edges": [], "model": "tiny", "clean": clean,
         "corrupted": corr, "kind": "explicit_tokens", "answer_tokens": sets["answer"], "alternatives": sets["alternatives"],
         "nodes": [{"stage": "source", "layer": 0, "pos": 3, "neuron": 5, "pole": 1}],
         "rankings": {"source": [["n", 0, 3, 5], ["n", 1, 3, 9], ["h", 2, 9, 1]]}}
    qs = {"q": {"user": " ".join(map(str, clean)), "corrupt_user": " ".join(map(str, corr)), "task": "t", "paraphrase": 0,
                "partners": [{"key": "k2", "user": " ".join(map(str, part))}]}}
    r = ns.score(m, Tok(), c, qs, log=lambda s: None, lean=True)
    con = lambda l: float(ns.ac.contrast(l[None], sets)[0])  # noqa: E731
    lc, lp_ = hooked(model, clean), hooked(model, part)
    c0, c1 = con(lc), con(lp_)
    cp, cc_ = m.capture(part), m.capture(clean)
    src = r["draws"]["cuts"]["source"]
    noised = con(hooked(model, clean, neurons=[(0, 3, 5, cp["n"][0, 3, 5])]))
    denoised = con(hooked(model, part, neurons=[(0, 3, 5, cc_["n"][0, 3, 5])]))
    assert src["necessity_heldout"][0]["share"] == pytest.approx((c0 - noised) / (c0 - c1), abs=1e-3)
    assert src["sufficiency_heldout"][0]["share"] == pytest.approx((denoised - c1) / (c0 - c1), abs=1e-3)
    mov = r["draws"]["cuts"]["movers"]
    noised_h = con(hooked(model, clean, heads=[(2, 9, 1, cp["h"][2, 9, 1])]))
    assert mov["necessity_heldout"][0]["share"] == pytest.approx((c0 - noised_h) / (c0 - c1), abs=1e-3)
    sc = r["size_curves"]["source"]
    assert sc["k"] == [1, 2, 3] and sc["pairs"] == 1
    clip = lambda x: max(-1.0, min(1.0, x))  # noqa: E731
    assert sc["heldout"][0][0] == pytest.approx(clip(src["necessity_heldout"][0]["share"]), abs=1e-4)  # (k = 1: the source cut)
    assert sc["heldout"][0][1] == pytest.approx(clip(src["sufficiency_heldout"][0]["share"]), abs=1e-4)
    cx = m.capture(corr)
    pr = r["pair"]
    n2 = con(hooked(model, clean, neurons=[(0, 3, 5, cx["n"][0, 3, 5]), (1, 3, 9, cx["n"][1, 3, 9])]))
    assert sc["own"][1][0] == pytest.approx((pr["c_clean"] - n2) / pr["gap"], abs=1e-3)
    d3 = con(hooked(model, corr, neurons=[(0, 3, 5, cc_["n"][0, 3, 5]), (1, 3, 9, cc_["n"][1, 3, 9])], heads=[(2, 9, 1, cc_["h"][2, 9, 1])]))
    assert sc["own"][2][1] == pytest.approx((d3 - pr["c_corrupted"]) / pr["gap"], abs=1e-3)  # (the whole ranking, a head in it)
    assert len(sc["random_own"]) == 3 and len(sc["random_heldout"]) == 3
    # the head swap: the answer-position head set to its value on the partner's prompt, in the clean run
    a, bp = int(lc.argmax()), int(lp_.argmax())
    hs = r["draws"]["head_swaps"]
    assert len(hs) == (1 if a != bp else 0)
    if hs:
        lsw = hooked(model, clean, heads=[(2, 9, 1, cp["h"][2, 9, 1])])
        dt = lambda l: float((l[bp] - l[a]) / 0.6931471805599453)  # noqa: E731
        assert hs[0]["hit"] == (int(lsw.argmax()) == bp) and len(hs[0]["random_hits"]) == ns.RANDOM_DRAWS
        assert hs[0]["share"] == pytest.approx((dt(lsw) - dt(lc)) / (dt(lp_) - dt(lc)), abs=1e-3)
