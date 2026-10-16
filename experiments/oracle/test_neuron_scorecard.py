"""Checks of the circuit finder's scorecard (neuron_scorecard.py) on a tiny random Qwen3: unpatched runs are the model's;
neuron and head patches equal hand-written hooks; padding a batch changes no run; the patching pair's shares follow the
definitions; stability counts hubs across paraphrases; the margins read the records.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_neuron_scorecard.py -q
"""
import math
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
    a, b = int(lc.argmax()), int(lx.argmax())
    alts = ns.pair_alts(lc, a, b)
    cc = float((lc[a] - torch.logsumexp(lc[alts], 0)) / math.log(2))
    assert pr["c_clean"] == pytest.approx(cc, abs=1e-4)
    capx = m.capture(corr)
    ln = hooked(model, clean, neurons=[(0, 3, 5, capx["n"][0, 3, 5]), (1, 3, 9, capx["n"][1, 3, 9])], heads=[(2, 9, 1, capx["h"][2, 9, 1])])
    cn = float((ln[a] - torch.logsumexp(ln[alts], 0)) / math.log(2))
    assert pr["c_noised"] == pytest.approx(cn, abs=1e-4)
    if pr["gap"] > 0:
        assert pr["lost"] == pytest.approx((pr["c_clean"] - pr["c_noised"]) / pr["gap"])
        assert pr["restored"] == pytest.approx((pr["c_denoised"] - pr["c_corrupted"]) / pr["gap"])
    assert len(r["effects"]) == 3 and r["size"]["poles"] == 2 and r["size"]["heads"] == 1
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
    def rec(key, para, hubs, nset):
        return {"task": f"{key}{para}", "question_task": "capital", "key": key, "builder": "b", "hubs": hubs, "neuron_set": nset,
                "pair": {"valid": True, "lost": 0.8, "restored": 0.6}, "size": {"poles": 10, "heads": 20}, "build_seconds": 5.0,
                "draws": {"necessity_heldout": [{"share": 0.9}, {"share": 0.5}], "sufficiency_heldout": [], "swaps": [{"share": 1.4}]}}
    rs = [rec("France", 0, [[1, 2, 1], [3, 4, -1]], [[1, 2, 1], [3, 4, -1]]), rec("France", 1, [[1, 2, 1]], [[1, 2, 1]]),
          rec("France", 2, [[1, 2, 1]], [[1, 2, 1], [3, 4, -1]]), rec("Japan", 0, [[1, 2, 1]], [[1, 2, 1]])]
    ns.stability(rs)
    assert [x["share"] for x in rs[0]["stability"]] == [1.0, 0.5]
    assert rs[3]["stability"] == []
    m = rs[0]["margins"]
    assert m["stability"]["score"] == 0.75 and m["stability"]["pass_old"]
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
        return {"id": i, "ranked_top": [{"layer": L, "neuron": n, "pole": s} for L, n, s in poles]}
    recs = {"a": rec("a", [(1, 2, 1), (3, 4, -1)]), "b": rec("b", [(1, 2, 1)]), "c": rec("c", [(5, 6, 1)])}
    run = {"B": ["a", "b", "c"], "rows": [
        {"kind": "group", "rank": 0, "poles": ["1.2.+", "3.4.-"], "damage": [2.0, 1.0, 0.0]},   # usage 1, 0.5, 0: tracks damage
        {"kind": "factor", "rank": 0, "poles": ["1.2.+", "3.4.-"], "damage": [0.0, 1.0, 2.0]},  # the reverse
        {"kind": "factor", "rank": 1, "poles": ["9.9.+"], "damage": [1.0, 2.0, 3.0]}]}           # used by no text
    ms = ns.motif_scores(run, recs)
    assert ms["group_pearson_median"] == pytest.approx(1.0) and ms["factor_pearson_median"] == pytest.approx(-1.0)
    assert ms["factor_scored"] == 1 and ms["sets"][2]["pearson"] is None
