"""Checks of the edit questions' scorer (edit_eval.py) on test_goal_eval's tiny random model: the edited run against
independent forward passes (path patching, removals, weight edits, native-unit ablations), the prefix reuse, the
persistent attention cut, validity and the score's fields.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_edit_eval.py -s
"""

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import edit_eval  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
from test_goal_eval import naive, naive_cut, tiny_native  # noqa: E402

LN2 = math.log(2)


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    return nat, ids, [11]


def kl_at(lp, lq, t):
    return float((lp[t].exp() * (lp[t] - lq[t])).sum()) / LN2


def run(nat, ids, ops, start=0):
    return edit_eval.Editor(nat).logp(torch.tensor([ids]), [ops], start)[0]


def test_no_edit_is_the_model(tiny):
    nat, ids, _ = tiny
    want = torch.log_softmax(naive(nat, ids)[0], -1)
    got = run(nat, ids, [])
    assert (got - want).abs().max() < 1e-4
    model = torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[0], -1)
    assert (got - model).abs().max() < 1e-4


CUTS = [(("h.0.attn.o_proj", 4, 1), ("h.1.attn.q_proj", 4, 2)), (("h.0.mlp.down_proj", 6, 0), ("h.1.attn.k_proj", 6, 3)),
        (("h.0.attn.o_proj", 2, 5), ("h.0.mlp.c_fc", 2, 1)), (("h.1.attn.v_proj", 3, 2), ("h.1.attn.o_proj", 11, 4)),
        (("h.0.mlp.c_fc", 9, 2), ("h.0.mlp.down_proj", 9, 3)), (("h.1.mlp.down_proj", 11, 1), None),
        (("h.0.attn.v_proj", 5, 0), ("h.0.attn.o_proj", 5, 1))]


def test_single_cut_is_path_patching(tiny):
    """One cut: the reader's activation from its input without the writer's write, the model running on (whole-model
    path patching, test_goal_eval's independent naive_cut)."""
    nat, ids, targets = tiny
    ref = run(nat, ids, [])
    for w, r in CUTS:
        got = kl_at(ref, run(nat, ids, [edit_eval.cut_op("out" if r is None else r, w)]), targets[0])
        want = naive_cut(nat, ids, targets, w, r)
        assert abs(got - want) < 1e-4 + 1e-3 * abs(want), (w, r, got, want)


def test_embedding_cut_matches_goal_eval(tiny):
    nat, ids, targets = tiny
    ge = goal_eval.GoalEval(nat, changes=4)
    tx = ge.text(ids, targets, seed=5)
    tok, r = ("tok", 7, 0), ("h.1.mlp.c_fc", 7, 2)
    want = ge.cuts(tx, [(tok, r)])[0]
    ref = run(nat, ids, [])
    got = kl_at(ref, run(nat, ids, [edit_eval.cut_op(r, tok)]), targets[0])
    assert abs(got - want) < 1e-4 + 1e-3 * abs(want)


def test_removals_match_native(tiny):
    """Several nodes scaled by 0 together: the model with them removed (Native.without)."""
    nat, ids, targets = tiny
    nodes = [("h.0.attn.v_proj", 3, 1), ("h.0.mlp.down_proj", 8, 4), ("h.1.attn.o_proj", 11, 2), ("h.1.mlp.c_fc", 11, 7)]
    want = nat.without(ids, targets, nodes)[0, 0]
    got = run(nat, ids, [edit_eval.scale_op(nd, 0.0) for nd in nodes])[targets[0]]
    assert (got - want).abs().max() < 1e-4


def test_persistent_scale_is_a_weight_edit(tiny):
    nat, ids, _ = tiny
    n, c, k = "h.0.mlp.c_fc", 3, 0.25
    st = nat.target.site(n)
    W = st.W.clone()
    try:
        st.W = W + (k - 1) * st.U[c][:, None] * st.V[:, c][None, :]
        want = torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[0], -1)
    finally:
        st.W = W
    got = run(nat, ids, [edit_eval.scale_op((n, None, c), k)])
    assert (got - want).abs().max() < 1e-4


def test_prefix_reuse(tiny):
    """A run from the first edited position equals the whole run there (the earlier positions read from the
    unedited record), with a value written before that position cut from a later attention output."""
    nat, ids, _ = tiny
    ops = [edit_eval.cut_op(("h.1.attn.o_proj", 9, 4), ("h.1.attn.v_proj", 3, 2)), edit_eval.scale_op(("h.0.mlp.down_proj", 10, 1), -1.0),
           edit_eval.unit_op("head", 1, 0, 11, 0.0)]
    s = edit_eval.Editor.start(ops)
    assert s == 9
    ed = edit_eval.Editor(nat)
    full = ed.logp(torch.tensor([ids]), [ops])[0]
    part = ed.logp(torch.tensor([ids]), [ops], start=s)[0]
    assert (full[s:] - part).abs().max() < 1e-5
    assert (full[:s] - run(nat, ids, [])[:s]).abs().max() < 1e-5


def test_persistent_attention_cut_is_every_pair(tiny):
    """A persistent value -> attention output cut equals the local cuts of every pair of positions."""
    nat, ids, _ = tiny
    w, r = ("h.1.attn.v_proj", None, 2), ("h.1.attn.o_proj", None, 4)
    got = run(nat, ids, [edit_eval.cut_op(r, w)])
    local = [{"op": "cut", "reader": (r[0], t, r[2]), "writer": (w[0], s, w[2]), "persistent": False}
             for t in range(len(ids)) for s in range(t + 1)]
    want = run(nat, ids, local)
    assert (got - want).abs().max() < 1e-5
    assert (got - run(nat, ids, [])).abs().max() > 1e-4


def test_units_match_input_ablations(tiny):
    """A head's output and an MLP neuron set to 0 at a position: the model with that input of the output matrix
    zeroed (Site.in_fn)."""
    nat, ids, _ = tiny
    tg = nat.target
    hd = tg.hd
    for kind, layer, j, t in (("head", 1, 1, 10), ("neuron", 0, 5, 7)):
        site = tg.site(native.site_name(layer, "o_proj" if kind == "head" else "down_proj"))
        cols = slice(j * hd, (j + 1) * hd) if kind == "head" else slice(j, j + 1)

        def zero(x, cols=cols, t=t):
            x = x.clone()
            x[:, t, cols] = 0
            return x
        site.in_fn = zero
        try:
            want = torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[0], -1)
        finally:
            site.in_fn = None
        got = run(nat, ids, [edit_eval.unit_op(kind, layer, j, t, 0.0)])
        assert (got - want).abs().max() < 1e-4, kind


def test_check():
    t = [11]
    ok = edit_eval.scale_op(("h.0.mlp.c_fc", 3, 1), 0.0)
    assert edit_eval.check([ok], 12, t) is None
    assert edit_eval.check([], 12, t) == "no operation"
    assert "invalid" in edit_eval.check([{"op": "invalid", "text": "zap"}], 12, t)
    assert "connect" in edit_eval.check([edit_eval.cut_op(("h.0.attn.q_proj", 4, 1), ("h.1.attn.o_proj", 4, 1))], 12, t)
    assert "connect" in edit_eval.check([edit_eval.cut_op(("h.1.attn.q_proj", 5, 1), ("h.0.attn.o_proj", 4, 1))], 12, t)
    assert "outside" in edit_eval.check([edit_eval.scale_op(("h.0.mlp.c_fc", 12, 1), 0.0)], 13, t)
    assert "behaviour" in edit_eval.check([ok], 12, t, behaviour=True)
    assert "embedding" in edit_eval.check([edit_eval.scale_op(("tok", 4, 0), 0.0)], 12, t)
    assert edit_eval.check([edit_eval.cut_op(("h.0.attn.q_proj", 4, 1), ("tok", 4, 0))], 12, t) is None
    assert edit_eval.check([edit_eval.scale_op(("h.0.mlp.c_fc", None, 1), 0.0)], 12, t, behaviour=True) is None
    assert edit_eval.check([edit_eval.cut_op(("h.1.attn.q_proj", None, 1), ("h.0.attn.o_proj", None, 1))], 12, t) is None
    assert edit_eval.check([edit_eval.cut_op("out", ("h.1.mlp.down_proj", 11, 0))], 12, t) is None
    assert "connect" in edit_eval.check([edit_eval.cut_op("out", ("h.1.mlp.down_proj", 10, 0))], 12, t)


def test_score_fields(tiny):
    """flip and raise: progress from the margin, attained when the token crosses; collateral the KL elsewhere and over
    the tokens not named; a persistent edit adds the held texts' summed KL; size in bits."""
    nat, ids, targets = tiny
    ee = edit_eval.EditEval(nat)
    g = torch.Generator().manual_seed(3)
    ee._held["held"] = torch.randint(0, 40, (3, 10), generator=g).tolist()
    ee._held["search"] = ee._held["held"]
    task = {"id": "tiny", "prompts": [{"token_ids": ids, "target_positions": targets}]}
    lp = ee.reference(ids)[11]
    order = lp.argsort(descending=True).tolist()
    flip = {"kind": "flip", "token_id": order[0], "target": 11}
    eds = [[edit_eval.cut_op("out", ("h.1.mlp.down_proj", 11, c))] for c in range(nat.C["h.1.mlp.down_proj"])]
    eds += [[edit_eval.scale_op(("h.1.mlp.down_proj", None, 0), -1.0)], [], [{"op": "invalid", "text": "?"}]]
    rows = ee.score_many(task, flip, eds)
    m0 = float(edit_eval._margin(lp, order[0]))
    for ops, r in zip(eds, rows):
        if not ops or ops[0]["op"] == "invalid":
            assert not r["valid"] and r["why"]
            continue
        assert r["valid"] and 0.0 <= r["progress"] <= 1.0 and r["collateral"] >= -1e-6
        assert r["attained"] == (r["margin"] < 0)
        assert abs(r["progress"] - min(1.0, max(0.0, (m0 - r["margin"]) / m0))) < 1e-6
        assert abs(r["goal_bits"] - max(0.0, min(m0, m0 - r["margin"]))) < 1e-6
        assert abs(r["net"] - (r["goal_bits"] - r["collateral"])) < 1e-9 and r["collateral_bits"] == r["collateral"]
        assert abs(sum(r["op_bits"]) - r["size"]) < 1e-9 and r["net_best"] is None
        persistent = ops[0].get("persistent")
        assert (r["collateral_held"] > 0) == bool(persistent)
        lq = run(nat, ids, ops)
        kept = [x for x in range(len(ids)) if x != 11]
        other = sum(kl_at(ee.reference(ids), lq, x) for x in kept)
        assert abs(r["collateral_text"] - other - ee._kept_kl(ee.reference(ids)[11], lq[11], [order[0]])) < 1e-4
        subs = math.log2(sum(nat.C.values()))
        want = edit_eval.FORM_BITS + (subs + math.log2(12) if not persistent else subs + 1 + edit_eval.SCALE_BITS)
        assert abs(r["size"] - want) < 1e-9
    best = max(r["progress"] for r in rows if r["valid"])
    ee.best = {ee.key("tiny", flip): best}
    again = ee.score_many(task, flip, eds[:3])
    assert all(abs(r["success"] - r["progress"] / best) < 1e-9 for r in again)
    raise_ = {"kind": "raise", "token_id": order[2], "target": 11}
    r = ee.score(task, raise_, [edit_eval.scale_op(("h.1.mlp.down_proj", 11, 0), 2.0)])
    assert r["valid"] and r["attained"] == (r["margin"] < 0)


def test_search_reaches_a_flip(tiny):
    """The teacher's search on the tiny model: an edit that flips the top token, no larger than needed."""
    nat, ids, targets = tiny
    ee = edit_eval.EditEval(nat)
    task = {"id": "tiny", "prompts": [{"token_ids": ids, "target_positions": targets}]}
    lp = ee.reference(ids)[11]
    flip = {"kind": "flip", "token_id": int(lp.argmax()), "target": 11}
    nodes = [(n, t, c) for n in nat.names for t in (9, 10, 11) for c in range(nat.C[n])]
    pool = [edit_eval.scale_op(nd, k) for nd in nodes for k in edit_eval.NODE_SCALES]
    res = edit_eval.search(lambda eds: ee.score_many(task, flip, eds, search=True), pool, log=lambda s: None)
    assert res["row"]["attained"]
    for i in range(len(res["ops"])):  # every edit is needed
        rest = res["ops"][:i] + res["ops"][i + 1:]
        assert not rest or not ee.score(task, flip, rest)["attained"]


def test_detach_dependence(tiny):
    """detach: the dependence is the target's mean KL from the edited text to the edited text with the position's
    token changed (the score's held-out alternatives), and progress is its log drop from the unedited dependence."""
    nat, ids, targets = tiny
    ee = edit_eval.EditEval(nat)
    task = {"id": "tiny", "prompts": [{"token_ids": ids, "target_positions": targets}]}
    goal = {"kind": "detach", "position": 6, "target": 11}
    alts = ee.alternatives(ids, 6, edit_eval.SCORE_RANKS)
    ops = [edit_eval.cut_op(("h.1.attn.o_proj", 11, c), ("h.1.attn.v_proj", 6, v)) for c in range(4) for v in range(4)]

    def dep(o):
        a = run(nat, ids, o)[11]
        return sum(kl_at(a[None], run(nat, x, o)[11][None], 0) for x in alts) / len(alts)
    d0, d = dep([]), dep(ops)
    r = ee.score(task, goal, ops)
    assert abs(r["dependence"] - d) < 1e-4 + 1e-3 * d
    want = min(1.0, max(0.0, math.log2(d0 / d) / math.log2(1 / (1 - edit_eval.GOAL))))
    assert abs(r["progress"] - want) < 1e-3
    assert r["attained"] == (d <= (1 - edit_eval.GOAL) * d0)
    assert abs(r["goal_bits"] - max(0.0, d0 - max(d, (1 - edit_eval.GOAL) * d0))) < 1e-4


def test_frontier_takes_progress_per_collateral():
    """The persistent search's step: the candidate with the most progress per bit of collateral and size; a free but
    tiny gain does not come first."""
    rows = {"a": {"valid": True, "progress": 0.9, "collateral": 90.0, "attained": False},
            "b": {"valid": True, "progress": 0.3, "collateral": 1.0, "attained": False},
            "c": {"valid": True, "progress": 0.001, "collateral": 0.0, "attained": False}}
    pool = [edit_eval.scale_op(("h.0.mlp.c_fc", None, i), 0.0) for i in range(3)]
    name = {id(pool[0]): "a", id(pool[1]): "b", id(pool[2]): "c"}

    def score(eds):
        out = []
        for ops in eds:
            r = [rows[name[id(o)]] for o in ops]
            out.append({"valid": True, "progress": min(1.0, sum(x["progress"] for x in r)),
                        "collateral": sum(x["collateral"] for x in r), "size": 20.0 * len(r), "attained": False})
        return out
    res = edit_eval.frontier(score, pool, max_ops=2, log=lambda m: None)
    assert [st["row"]["progress"] for st in res["path"]] == [0.3, pytest.approx(1.0)]
    assert res["path"][0]["edit"][0]["node"] == ["h.0.mlp.c_fc", None, 1]


def test_frontier_steps_chosen_by_validation():
    """With validation instances, a step is the candidate with the best validation rate among the best on fit, and the
    chosen step the one with the best validation net."""
    fit = {"a": (0.9, 1.0), "b": (0.5, 1.0)}  # name -> (progress, collateral) on the fit instances
    val = {"a": (0.1, 1.0), "b": (0.5, 1.0)}  # a overfits: it does little on validation
    pool = [edit_eval.scale_op(("h.0.mlp.c_fc", None, i), 0.0) for i in range(2)]
    name = {id(pool[0]): "a", id(pool[1]): "b"}

    def rows(table):
        def score(eds):
            out = []
            for ops in eds:
                p = min(1.0, sum(table[name[id(o)]][0] for o in ops))
                c = sum(table[name[id(o)]][1] for o in ops)
                out.append({"valid": True, "progress": p, "collateral": c, "size": 20.0 * len(ops), "attained": False,
                            "net": p * 10 - c})
            return out
        return score
    res = edit_eval.frontier(rows(fit), pool, max_ops=1, log=lambda m: None, validate=rows(val), pick=2)
    assert res["path"][0]["edit"][0]["node"] == ["h.0.mlp.c_fc", None, 1] and res["chosen"] == 0
