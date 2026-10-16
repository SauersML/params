"""The claim checker (claim_check.py) on a tiny random Qwen3 in neuron_builder's reference: a zero claim is the
reference with the pole's strong values zeroed from position 1 on, claims give the same numbers alone and batched,
patching from the prompt itself and steering by 0 change nothing, controls match the named units' layers and poles,
an effect that random units also produce is not specific, and a bad claim is an error that leaves the others.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_claim_check.py -q
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import claim_check as cc  # noqa: E402
import neuron_builder as nb  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    nr = nb.Neurons(transformers.Qwen3ForCausalLM(cfg).eval(), None)
    g = torch.Generator().manual_seed(1)
    nr.calibrate([torch.randint(0, 64, (16,), generator=g).tolist() for _ in range(4)], taus=(0.01, 0.05), kl_tol=0.5, prices=4, refine=1)
    return nr, torch.randint(0, 64, (12,), generator=g).tolist()


def claim(ids, **kw):
    return {"prompt": {"ids": ids}, "position": -1, "units": ["L1.N5+"], "op": "zero", "predict": {"top_changes": True}, **kw}


def test_a_zero_claim_is_the_reference_with_the_poles_strong_values_zeroed(tiny):
    nr, ids = tiny
    r = cc.check_many(nr, [claim(ids)])[0]

    def edit(i, h):
        if i != 1:
            return h
        h = h.clone()
        v = h[:, 1:, 5]
        h[:, 1:, 5] = v.masked_fill((v > 0) & (v.abs() < nr.massive[1]), 0)
        return h

    with torch.no_grad():
        lp0 = torch.log_softmax(nr.logits(torch.tensor([ids] * cc.ROWS), [len(ids) - 1])[:, 0], -1)[0]
        lp1 = torch.log_softmax(nr.logits(torch.tensor([ids] * cc.ROWS), [len(ids) - 1], edit=edit)[:, 0], -1)[0]
    y = int(lp0.argmax())
    alts = [t for t in lp0.topk(nb.ALTERNATIVES + 1).indices.tolist() if t != y][:nb.ALTERNATIVES]
    c0, c1 = (float((lp[y] - torch.logsumexp(lp[alts], 0)) / math.log(2)) for lp in (lp0, lp1))
    assert abs(r["contrast_before"] - c0) < 1e-5 and abs(r["contrast_after"] - c1) < 1e-5
    assert r["passes"] == 1 + cc.CONTROLS and r["units"] == 1


def test_claims_give_the_same_numbers_alone_and_batched(tiny):
    nr, ids = tiny
    cs = [claim(ids, units=[u]) for u in ("L0.N3", "L1.N7-", "L1.H2", "L0.N11+")] + [claim(ids, swap=[{"position": 4, "token": 9}], units=[])]
    together = cc.check_many(nr, cs)
    for c, r in zip(cs, together):
        alone = cc.check_many(nr, [c])[0]
        assert abs(alone["contrast_after"] - r["contrast_after"]) < 1e-6 and alone["side_kl"] == pytest.approx(r["side_kl"], abs=1e-7)


def test_patching_from_the_prompt_itself_and_steering_by_zero_change_nothing(tiny):
    nr, ids = tiny
    pred = {"contrast_change": {"op": "<=", "value": 1e-6}}
    for c in (claim(ids, units=["L0.N3", "L1.H1"], op={"patch_from": {"ids": ids}}, predict=pred),
              claim(ids, units=["L1.N7+"], op={"steer": 0.0}, predict=pred)):
        r = cc.check_many(nr, [c])[0]
        assert abs(r["change"]) < 1e-5 and r["side_kl"] < 1e-6, r
    other = list(ids)
    other[3] = (other[3] + 1) % 64
    moved = cc.check_many(nr, [claim(ids, units=[f"L{l}.N{n}" for l in range(2) for n in range(48)], op={"patch_from": {"ids": other}},
                                     predict=pred)])[0]
    assert moved["passes"] == 2 + cc.CONTROLS and "error" not in moved


def test_controls_match_the_named_units_layers_and_poles():
    nr = type("N", (), {"D": 48, "NH": 4})()
    us = [("N", 1, 5, 1, None), ("N", 1, 9, 1, None), ("N", 0, 3, -1, None), ("H", 1, 2, 0, None)]
    for s in cc.control_sets(nr, us, 3, "x"):
        assert sorted((k, l, p) for k, l, _, p, _ in s) == sorted((k, l, p) for k, l, _, p, _ in us)
        assert not {(k, l, i) for k, l, i, _, _ in s} & {(k, l, i) for k, l, i, _, _ in us}


def test_an_effect_random_units_also_produce_is_not_specific(tiny):
    nr, ids = tiny
    ref = cc.check_many(nr, [claim(ids, predict={"contrast": {"op": "<=", "value": 1e9}})])[0]
    same = cc.check_many(nr, [claim(ids, predict={"top": nr_top(nr, ids)})])[0]  # "the answer is the reference's top": controls pass too
    assert ref["specific"] == ref["pass"]  # a prediction of nothing: specific when it passes
    assert same["pass"] and all(same["controls_pass"]) and not same["specific"]
    assert cc.effect_predicted({"contrast_change": {"op": "<=", "value": -2}}) and not cc.effect_predicted({"top_changes": False})


def nr_top(nr, ids):
    with torch.no_grad():
        return int(nr.logits(torch.tensor([ids] * cc.ROWS), [len(ids) - 1])[0, 0].argmax())


def test_a_bad_claim_is_an_error_and_the_others_run(tiny):
    nr, ids = tiny
    got = cc.check_many(nr, [claim(ids, units=["L7.N1"]), claim(ids, units=["L1.H1+"]), claim(ids, op={"steer": 2}, units=["L1.H1"]),
                             claim(ids, op="melt"), claim(ids)])
    assert all("error" in r and not r["pass"] for r in got[:4]) and "error" not in got[4]


def test_unit_names_parse_and_print_one_way():
    for s, u in (("L18.N2201+", ("N", 18, 2201, 1)), ("L18.N2201-", ("N", 18, 2201, -1)), ("L3.N7", ("N", 3, 7, 0)), ("L5.H3", ("H", 5, 3, 0))):
        assert cc.parse_unit(s) == u and cc.unit_name(u) == s
    for bad in ("L5.H3+", "18.N2", "L1.X2", "L1.N"):
        with pytest.raises(cc.BadClaim):
            cc.parse_unit(bad)


def test_a_position_scoped_unit_acts_at_its_position_alone(tiny):
    nr, ids = tiny
    T = len(ids)
    r = cc.check_many(nr, [claim(ids, units=["L0.N7+@-2"], predict={"top_changes": False})])[0]

    def edit(i, h):
        if i != 0:
            return h
        h = h.clone()
        v = h[:, T - 2, 7]
        h[:, T - 2, 7] = v.masked_fill((v > 0) & (v.abs() < nr.massive[0]), 0)
        return h

    with torch.no_grad():
        lp1 = torch.log_softmax(nr.logits(torch.tensor([ids] * cc.ROWS), [T - 1], edit=edit)[:, 0], -1)[0]
    y = int(torch.log_softmax(nr.logits(torch.tensor([ids] * cc.ROWS), [T - 1])[:, 0], -1)[0].argmax())
    assert r["top_after"] == int(lp1.argmax()) and "error" not in r
    at_all = cc.check_many(nr, [claim(ids, units=["L0.N7+"]), claim(ids, units=["L0.N7+@0"]), claim(ids, units=[f"L0.N7+@{T}"])])
    assert "error" not in at_all[0] and "error" in at_all[1] and "error" in at_all[2]
    assert y == r["top_before"]


def test_position_scoped_names_parse_print_and_control_at_their_position():
    assert cc.parse_unit_at("L18.N2201+@12") == ("N", 18, 2201, 1, 12) and cc.parse_unit_at("L5.H3@-1") == ("H", 5, 3, 0, -1)
    assert cc.unit_name(("N", 18, 2201, 1, 12)) == "L18.N2201+@12" and cc.parse_unit_at("L3.N7")[4] is None
    with pytest.raises(cc.BadClaim):
        cc.parse_unit("L18.N2201+@12")  # (an explanation's pole is the same at every position)
    nr = type("N", (), {"D": 48, "NH": 4})()
    us = [("N", 1, 5, 1, 7), ("N", 1, 9, 1, None), ("H", 0, 2, 0, 3)]
    for s in cc.control_sets(nr, us, 2, "y"):
        assert sorted((k, l, p, at if at is not None else -1) for k, l, _, p, at in s) == sorted((k, l, p, at if at is not None else -1) for k, l, _, p, at in us)


def test_a_knockout_set_is_row_edits_zero_in_every_row(tiny):
    nr, ids = tiny
    us = cc.units_of(nr, ["L0.N3+", "L0.N3-", "L1.N7-", "L1.N9", "L1.H2"], len(ids))
    x = torch.tensor([ids] * 3)
    with torch.no_grad():
        e1, r1 = cc.row_edit(nr, [{"units": us, "op": "zero"}] * 3)
        e2, r2 = cc.knockout_edit(nr, us)
        a = nr.logits(x, list(range(len(ids))), edit=e1, route=r1)
        b = nr.logits(x, list(range(len(ids))), edit=e2, route=r2)
    assert torch.allclose(a, b, atol=1e-6)
