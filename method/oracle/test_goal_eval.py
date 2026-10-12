"""Checks of the goal scorecard (goal_eval.py): exactness on a tiny random model, every run and cut against an
independent forward pass, and the same on two held-out vpd4l texts (skipped without ~/mpd-data's model).

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_goal_eval.py -s
"""

import math
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import vpd_model  # noqa: E402

LN2 = math.log(2)


def tiny_native(seed: int = 0, layers: int = 2, d: int = 16, heads: int = 2, vocab: int = 40, C: int = 6,
                near: float | None = None) -> native.Native:
    """A Native over a random two-layer model of vpd4l's architecture with random subcomponents (their remainders
    large: the decomposition is far from exact) and a causal-importance function (a random linear map of each
    matrix's input, clamped), on the CPU in float32. near: instead d subcomponents per matrix that make up its weight
    (its singular directions), each perturbed by noise of this relative size, so the remainders are small, and causal
    importance above zero where an activation is above half the mean size of the matrix's."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, scale=1.0: torch.randn(*s, generator=g) * scale  # noqa: E731
    cfg = {"n_layer": layers, "n_head": heads, "n_embd": d, "n_key_value_heads": heads, "rotary_dim": d // heads,
           "rms_norm_eps": 1e-6, "n_ctx": 64, "rotary_base": 10000.0}
    sd = {"wte.weight": r(vocab, d), "ln_f.weight": 1 + r(d, scale=0.1)}
    for i in range(layers):
        sd[f"h.{i}.rms_1.weight"] = 1 + r(d, scale=0.1)
        sd[f"h.{i}.rms_2.weight"] = 1 + r(d, scale=0.1)
    shapes = {"q_proj": (d, d), "k_proj": (d, d), "v_proj": (d, d), "o_proj": (d, d), "c_fc": (4 * d, d), "down_proj": (d, 4 * d)}
    names = vpd_model.site_names(layers)
    for n in names:
        o, i = shapes[n.split(".")[-1]]
        sd[f"{n}.weight"] = r(o, i, scale=1.5 / math.sqrt(i))
    target = vpd_model.Target(sd, cfg).eval()
    UV = {}
    for n in names:
        o, i = shapes[n.split(".")[-1]]
        if near is None:
            UV[n] = (r(C, o, scale=1.0 / math.sqrt(C)), r(i, C, scale=1.5 / math.sqrt(i)))
        else:  # W.T = A S B^T: V = A sqrt(S), U = sqrt(S) B^T, d of them
            A, S, Bt = torch.linalg.svd(sd[f"{n}.weight"].T, full_matrices=False)
            V, U = A[:, :d] * S[:d].sqrt(), S[:d].sqrt()[:, None] * Bt[:d]
            UV[n] = (U * (1 + near * r(*U.shape)), V * (1 + near * r(*V.shape)))
    proj = {n: r(*UV[n][1].shape) for n in names}
    vpd = vpd_model.VPD(target, None, UV)
    if near is None:
        vpd.ci_fn = lambda acts: {n: acts[n] @ proj[n].to(acts[n].device) for n in names}
    else:  # active where a subcomponent's activation is above half its mean size: a sensible VPD answer
        vpd.ci_fn = lambda acts: {n: (lambda a: a / a.mean() - 0.5)((acts[n] @ UV[n][1].to(acts[n].device)).abs()) for n in names}
    for n in names:  # what native.model does to vpd4l
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(native._forward_cached, st)
    loaded = native.model
    native.model = lambda dev: (target, vpd)  # Native's own constructor, on this model
    try:
        return native.Native("cpu")
    finally:
        native.model = loaded


# ---- an independent forward pass (no Native.run, no Site masks)


@torch.no_grad()
def naive(nat, ids, replace=None, on=None, live=True, cut=None):
    """The model's forward pass written out, every matrix as subcomponents plus remainder: (logits [T, V], states:
    every activation per matrix [T, C], the stream before each norm, attention patterns and outputs, MLP inputs).
    replace: {(matrix, position, index): value} overrides those activations; on: matrix -> bool [T, C], the
    subcomponents that write (all without it); live: whether the remainders write; cut: (writer, reader), the reader's
    activation computed from its input without the writer's write in this run."""
    tg = nat.target
    T, H, hd = len(ids), tg.n_head, tg.hd
    dev = tg.wte.device
    x = tg.wte[torch.tensor(ids, device=dev)].clone()
    st = {"a": {}, "x_attn": [], "x_mlp": [], "P": [], "att": [], "pre": []}

    def write(nd):
        if nd[0] == "tok":  # the token's embedding at its position
            return tg.wte[ids[nd[1]]]
        a = st["a"][nd[0]][nd[1], nd[2]]
        m = 1.0 if on is None else float(on[nd[0]][nd[1], nd[2]])
        return a * m * tg.site(nd[0]).U[nd[2]]

    def act(n, inp, cut_input=None):
        a = inp @ tg.site(n).V
        if cut is not None and cut[1] is not None and cut[1][0] == n:
            a[cut[1][1], cut[1][2]] = cut_input(write(cut[0])) @ tg.site(n).V[:, cut[1][2]]
        for (m, t, c), v in (replace or {}).items():
            if m == n:
                a[t, c] = v
        st["a"][n] = a
        mask = 1.0 if on is None else on[n].float()
        return (a * mask) @ tg.site(n).U + (inp @ tg.site(n).delta_T if live else 0.0)

    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    for i in range(tg.n_layer):
        nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
        st["x_attn"].append(x.clone())
        z = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
        resid = lambda wv, x=x, i=i: vpd_model.rms(x[cut[1][1]] - wv, tg.norms[2 * i], tg.eps)  # noqa: E731
        q, k, v = (act(nm[kk], z, resid).view(T, H, hd).transpose(0, 1) for kk in ("q_proj", "k_proj", "v_proj"))
        q, k = tg._rope(q, T), tg._rope(k, T)
        P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
        att = (P @ v).transpose(0, 1).reshape(T, -1)
        st["P"].append(P)
        st["att"].append(att)
        share = lambda wv, att=att, P=P: att[cut[1][1]] - P[:, cut[1][1], cut[0][1]].repeat_interleave(hd) * wv  # noqa: E731
        x = x + act(nm["o_proj"], att, share)
        st["x_mlp"].append(x.clone())
        z2 = vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps)
        resid2 = lambda wv, x=x, i=i: vpd_model.rms(x[cut[1][1]] - wv, tg.norms[2 * i + 1], tg.eps)  # noqa: E731
        pre = act(nm["c_fc"], z2, resid2)
        st["pre"].append(pre)
        g = vpd_model.gelu_tanh(pre)
        x = x + act(nm["down_proj"], g, lambda wv, pre=pre: vpd_model.gelu_tanh(pre[cut[1][1]] - wv))
    if cut is not None and cut[1] is None:
        x = x.clone()
        x[cut[0][1]] -= write(cut[0])
    st["x_final"] = x
    return vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T, st


def _kl(p_logits, q_logits, targets):
    logp = torch.log_softmax(p_logits[targets], -1)
    logq = torch.log_softmax(q_logits[targets], -1)
    return float((logp.exp() * (logp - logq)).sum()) / LN2


def naive_cut(nat, ids, targets, w, r):
    """The KL in bits of the model at the targets from the model with the connection w -> r cut (whole-model path
    patching), from naive()'s states: the reader's input without the writer's output, then a forward pass with that
    activation."""
    tg = nat.target
    full, st = naive(nat, ids)
    write = st["a"][w[0]][w[1], w[2]] * tg.site(w[0]).U[w[2]]
    if r is None:
        x = st["x_final"].clone()
        x[w[1]] -= write
        return _kl(full, vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T, targets)
    rl, rk = native._name_layer_kind(r[0])
    V = tg.site(r[0]).V[:, r[2]]
    if rk in ("q_proj", "k_proj", "v_proj"):
        new = vpd_model.rms(st["x_attn"][rl][r[1]] - write, tg.norms[2 * rl], tg.eps) @ V
    elif rk == "c_fc":
        new = vpd_model.rms(st["x_mlp"][rl][r[1]] - write, tg.norms[2 * rl + 1], tg.eps) @ V
    elif rk == "o_proj":
        new = (st["att"][rl][r[1]] - st["P"][rl][:, r[1], w[1]].repeat_interleave(tg.hd) * write) @ V
    else:
        new = vpd_model.gelu_tanh(st["pre"][rl][r[1]] - write) @ V
    cut, _ = naive(nat, ids, {r: new})
    return _kl(full, cut, targets)


def mask_of(nat, T, nodes):
    m = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, t, c in nodes:
        m[n][t, c] = True
    return m


# ---- the tiny model


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    tx = ge.text(ids, [11], seed=5)
    return nat, ge, tx


def random_graph(nat, tx, seed: int, k: int = 14, stated: float = 0.5) -> native.Graph:
    g = torch.Generator().manual_seed(seed)
    nodes = set()
    while len(nodes) < k:
        n = nat.names[int(torch.randint(len(nat.names), (1,), generator=g))]
        nodes.add((n, int(torch.randint(tx.T, (1,), generator=g)), int(torch.randint(nat.C[n], (1,), generator=g))))
    par, out = native.all_edges(nodes, tx.targets)
    pairs = goal_eval._pairs(native.Graph(nodes, par, out))
    keep = [p for p in pairs if float(torch.rand(1, generator=g)) < stated]
    return native.Graph(nodes, {r: [w for w, rr in keep if rr == r] for _, r in keep if r is not None}, [w for w, r in keep if r is None])


def test_runs_match_an_independent_forward(tiny):
    """VPD's semantics (some subcomponents on, the rest off, remainders on or off) against naive()."""
    nat, ge, tx = tiny
    full, _ = naive(nat, tx.ids)
    for seed in (1, 2):
        nodes = random_graph(nat, tx, seed, k=30).node_set()
        on = mask_of(nat, tx.T, nodes)
        for live in (True, False):
            got = float(carry.kl(tx.ref, ge.runs(tx.ids, tx.targets, [on], live))[0])
            want = _kl(full, naive(nat, tx.ids, on=on, live=live)[0], tx.targets)
            assert abs(got - want) < 1e-4 + 1e-4 * abs(want), (seed, live, got, want)


def test_bounds_and_exact_endpoints(tiny):
    """The whole model through the circuit path is the model; VPD's own answer recovers 1 and is wholly necessary;
    the empty circuit recovers 0 and needs nothing."""
    nat, ge, tx = tiny
    every = nat.everything(tx.T)
    vpd, empty = native.Graph(masks=tx.vpd), native.Graph()
    rows = ge.sufficiency(tx, [every, vpd, empty])
    assert abs(rows[0]["kl_live"]) < 1e-5
    assert abs(rows[1]["recovered_live"] - 1) < 1e-6 and abs(rows[1]["recovered_off"] - 1) < 1e-6
    assert abs(rows[1]["f1_live"] - 1) < 1e-6 and abs(rows[1]["n_off"] - 1) < 1e-6
    assert rows[2]["f1_live"] == 0.0 and abs(rows[2]["n_live"]) < 1e-5  # float32 batches differ from _bounds' run
    assert abs(rows[1]["necessity_live"] - 1) < 1e-6 and abs(rows[2]["necessity_live"]) < 1e-6
    assert abs(rows[2]["recovered_live"]) < 1e-6 and abs(rows[2]["recovered_off"]) < 1e-6
    full, _ = naive(nat, tx.ids)
    empty_on = mask_of(nat, tx.T, [])
    assert abs(rows[2]["kl_empty_off"] - _kl(full, naive(nat, tx.ids, on=empty_on, live=False)[0], tx.targets)) < 1e-4
    assert abs(rows[2]["kl_empty_live"] - _kl(full, naive(nat, tx.ids, on=empty_on, live=True)[0], tx.targets)) < 1e-4
    assert rows[0]["completeness_live"] == 1.0 or rows[0]["recovered_live"] is None


def test_necessity_is_vpd_without_the_circuit(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 3, k=20)
    row = ge.sufficiency(tx, [g])[0]
    full, _ = naive(nat, tx.ids)
    minus = {n: tx.vpd[n] & ~mask_of(nat, tx.T, g.nodes)[n] for n in nat.names}
    for live in (True, False):
        conv = "live" if live else "off"
        want = _kl(full, naive(nat, tx.ids, on=minus, live=live)[0], tx.targets)
        assert abs(row[f"kl_vpd_minus_{conv}"] - want) < 1e-4
        e, v = row[f"kl_empty_{conv}"], row[f"kl_vpd_{conv}"]
        assert row[f"necessity_{conv}"] == goal_eval.necessity(e, row[f"kl_vpd_minus_{conv}"], v)
        assert row[f"n_{conv}"] == goal_eval.necessity_log(e, row[f"kl_vpd_minus_{conv}"], v)
        assert row[f"f1_{conv}"] == min(1.0, max(0.0, goal_eval.recovered_log(e, row[f"kl_{conv}"], v)))


def test_edge_effects_match_an_independent_forward(tiny):
    """F3's cut of one connection inside the circuit's run (remainders off) against naive()'s cut, for stated and
    unstated pairs of every kind among the nodes."""
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 7, k=30, stated=1.0)
    pairs = goal_eval._pairs(g)
    on = mask_of(nat, tx.T, g.nodes)
    base, _ = naive(nat, tx.ids, on=on, live=False)
    got = ge.edge_effects(tx, g, pairs)
    for (w, r), k in zip(pairs, got):
        want = _kl(base, naive(nat, tx.ids, on=on, live=False, cut=(w, r))[0], tx.targets)
        assert abs(k - want) < 1e-4 + 1e-3 * abs(want), (w, r, k, want)
    assert max(got) > 1e-3  # the cuts move the prediction: the comparison is not of zeros
    # the uncut run is F1's run with remainders off
    row = ge.sufficiency(tx, [g])[0]
    assert abs(row["kl_off"] - _kl(naive(nat, tx.ids)[0], base, tx.targets)) < 1e-4


def test_f3_needs_a_stated_discriminating_claim(tiny):
    nat, ge, tx = tiny
    nodes = random_graph(nat, tx, 9, k=16).node_set()
    par, out = native.all_edges(nodes, tx.targets)
    bare, everything = native.Graph(nodes), native.Graph(nodes, par, out)
    some = random_graph(nat, tx, 9, k=16, stated=0.5)
    f3 = ge.edges(tx, [bare, everything, some, native.Graph(nodes, complete=True)])
    assert f3[0]["f3"] == 0.0 and f3[1]["f3"] == 0.0 and f3[3]["f3"] is None
    assert 0.0 <= f3[2]["f3"] <= 1.0 and f3[2]["stated"] and f3[2]["unstated"]


def test_cuts_match_an_independent_forward(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 7, k=30, stated=1.0)
    pairs = goal_eval._pairs(g)
    got = ge.cuts(tx, pairs)
    for (w, r), k in zip(pairs, got):
        want = naive_cut(nat, tx.ids, tx.targets, w, r)
        assert abs(k - want) < 1e-4 + 1e-3 * abs(want), (w, r, k, want)
    assert max(got) > 1e-3


def test_cut_kinds_covered(tiny):
    """Every kind of connection is cut and checked: residual into each reader kind, value into attention output at an
    earlier position, MLP input into its output, residual writer into the prediction."""
    nat, ge, tx = tiny
    last = tx.targets[0]
    pairs = [(("h.0.attn.o_proj", 4, 1), ("h.1.attn.q_proj", 4, 2)), (("h.0.mlp.down_proj", 6, 0), ("h.1.attn.k_proj", 6, 3)),
             (("h.0.attn.o_proj", 2, 5), ("h.0.mlp.c_fc", 2, 1)), (("h.1.attn.v_proj", 3, 2), ("h.1.attn.o_proj", last, 4)),
             (("h.0.mlp.c_fc", 9, 2), ("h.0.mlp.down_proj", 9, 3)), (("h.1.mlp.down_proj", last, 1), None),
             (("h.0.attn.v_proj", 5, 0), ("h.0.attn.o_proj", 5, 1))]
    for w, r in pairs:
        assert native.connects(w, r, tx.targets)
    got = ge.cuts(tx, pairs)
    for (w, r), k in zip(pairs, got):
        want = naive_cut(nat, tx.ids, tx.targets, w, r)
        assert abs(k - want) < 1e-4 + 1e-3 * abs(want), (w, r, k, want)
    nodes = {nd for e in pairs for nd in e if nd is not None}
    on, g = mask_of(nat, tx.T, nodes), native.Graph(nodes)
    base, _ = naive(nat, tx.ids, on=on, live=False)
    for (w, r), k in zip(pairs, ge.edge_effects(tx, g, pairs)):
        want = _kl(base, naive(nat, tx.ids, on=on, live=False, cut=(w, r))[0], tx.targets)
        assert abs(k - want) < 1e-4 + 1e-3 * abs(want), (w, r, k, want)


def test_cuts_into_the_prediction_alone(tiny):
    """An answer whose only edges go into the prediction (no reader inside the model) is cut without a model run."""
    nat, ge, tx = tiny
    last = tx.targets[0]
    pairs = [(("h.1.mlp.down_proj", last, 1), None), (("h.0.attn.o_proj", last, 2), None)]
    got = ge.cuts(tx, pairs)
    for (w, r), k in zip(pairs, got):
        assert abs(k - naive_cut(nat, tx.ids, tx.targets, w, r)) < 1e-4
    g = native.Graph({w for w, _ in pairs}, {}, [w for w, _ in pairs])
    assert ge.edges(tx, [g])[0]["f3"] is not None


def test_cut_detail(tiny):
    """cuts(detail=True): the cut run's top token and its log-probability of the model's top token."""
    nat, ge, tx = tiny
    last = tx.targets[0]
    w = ("h.1.mlp.down_proj", last, 1)
    d = ge.cuts(tx, [(w, None)], detail=True)[0]
    full, st = naive(nat, tx.ids)
    x = st["x_final"].clone()
    x[last] -= st["a"][w[0]][last, 1] * nat.target.site(w[0]).U[1]
    logq = torch.log_softmax((vpd_model.rms(x, nat.target.ln_f, nat.target.eps) @ nat.target.wte.T)[last], -1)
    assert d["top"] == [int(logq.argmax())]
    assert abs(d["logp_model_top"][0] - float(logq[int(full[last].argmax())])) < 1e-4
    assert abs(d["kl"] - ge.cuts(tx, [(w, None)])[0]) == 0


def test_interchange_rule_b(tiny):
    """Rule b: a circuit naming every subcomponent active on the text after p carries what B_p does; nothing named
    carries the empty share; target positions are left out."""
    nat, ge, tx = tiny
    assert tx.f2, "the tiny text needs F2 positions"
    p = next((q for q, _ in tx.f2 if q not in tx.targets), None)
    if p is None:
        pytest.skip("only the target moves the tiny text's prediction")
    after = {(n, t, c) for n in nat.names for t, c in (tx.ci[n] > 0).nonzero().tolist() if t > p}
    rows = ge.interchange(tx, [native.Graph(after), native.Graph()])
    for pos, mv, carried, bp, empty in rows[0]["carry_positions"]:
        if pos == p:
            assert abs(carried - bp) < 1e-6
    for pos, mv, carried, bp, empty in rows[1]["carry_positions"]:
        assert abs(carried - empty) < 1e-6
        assert pos not in tx.targets


def test_across_prompts(tiny):
    """F1 on other prompts: on the text itself it is F1; VPD's own answer on each prompt is the ceiling there."""
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 21, k=16)
    a = ge.across(tx, [g, native.Graph(masks=tx.vpd)], [tx.ids])
    row = ge.sufficiency(tx, [g])[0]
    for conv in ("live", "off"):
        assert abs(a[0][f"recovered_{conv}"] - row[f"recovered_{conv}"]) < 1e-5
        assert abs(a[1][f"recovered_{conv}"] - 1) < 1e-6


def test_score_rows_and_table(tiny):
    nat, ge, tx = tiny
    graphs = [random_graph(nat, tx, 21, k=16)]
    base = ge.baselines(tx, [graphs[0].count()])
    assert set(base) == {"empty", f"vpd@{graphs[0].count()}", "vpd"}
    ge.hooks["E1"] = lambda task, g, expl: float(g.count())
    rows = ge.score(tx, graphs + list(base.values()), diagnostics=True)
    ge.hooks["E1"] = None
    assert rows[0]["E1"] == graphs[0].count() and rows[0]["E2"] is None
    assert rows[1]["nodes"] == 0 and rows[1]["bits"] == 0.0
    for key in ("recovered_live", "necessity_off", "f3", "changed_recovered_live", "f2_recovered_off", "carry_completeness"):
        assert key in rows[0]
    text = goal_eval.table({"set": "tiny", "rows": {"answer": [rows[0], {"task": "x", "valid": False}], "empty": [rows[1]]}})
    assert "coverage" in text and " 0.50 " in text and "recov live" in text


def test_circuit_json_round_trip_and_defects(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 31, k=12)
    back, defects = goal_eval.circuit(goal_eval.to_json(g), tx.targets, nat.C, tx.T)
    assert back.nodes == g.nodes and defects == {"invalid_nodes": 0, "invalid_edges": 0}
    assert sorted(goal_eval._pairs(back)) == sorted(goal_eval._pairs(g))
    bad = {"nodes": [["h.0.attn.o_proj", 1, 3], ["1.q", 2, 5], ["h.9.attn.q_proj", 0, 0], ["0.down", 0, 3], ["1.q", 2, 3],
                     ["tok", 0, 3], ["tok", 1, 3]],
           "edges": [[0, 1], [3, 1], [0, "out"], [1, 0], [2, 1], [7, 1], [3, 4], [5, 4], [5, 0], [5, "out"]]}
    g2, d = goal_eval.circuit(bad, tx.targets, nat.C, tx.T)
    # another position, not at a target, a query writes nothing, an unknown matrix, no such node; a valid edge; the
    # embedding into a query at its position; into an attention output and into the prediction (not residual readers)
    assert d == {"invalid_nodes": 2, "invalid_edges": 8}
    assert g2.parents == {("h.1.attn.q_proj", 3, 2): [("h.0.mlp.down_proj", 3, 0)]} and g2.out == []
    assert len(g2.nodes) == 4 and g2.tok_edges == [(3, ("h.1.attn.q_proj", 3, 2))]
    again, d2 = goal_eval.circuit(goal_eval.to_json(g2), tx.targets, nat.C, tx.T)
    assert again.tok_edges == g2.tok_edges and d2 == {"invalid_nodes": 0, "invalid_edges": 0}


def test_probe_is_kept(tiny):
    """A text's changed prompts and F2 prompts come from one sensitivity probe, kept per text: a second preparation
    of the text gets the same prompts without probing again."""
    nat, ge, tx = tiny
    first = (tx.prompts, tx.f2)
    calls = []
    sensitivity = nat.sensitivity
    nat.sensitivity = lambda *a, **k: calls.append(1) or sensitivity(*a, **k)
    try:
        again = ge.text(tx.ids, tx.targets, seed=tx.seed)
        assert [list(x) for x in again.prompts] == [list(x) for x in first[0]] and again.f2 == first[1]
    finally:
        del nat.sensitivity
    assert not calls


def test_answer_files_and_samples(tmp_path):
    (tmp_path / "t1.py").write_text("x")
    (tmp_path / "t1.json").write_text('{"steps": []}')  # a search's scores, not an answer
    (tmp_path / "t1.2.json").write_text('{"nodes": []}')
    (tmp_path / "t1.1.py").write_text("x")
    (tmp_path / "t2.json").write_text('{"nodes": []}')
    assert [f.name for f in goal_eval.answer_files(tmp_path, "t1")] == ["t1.py", "t1.1.py", "t1.2.json"]
    assert [f.name for f in goal_eval.answer_files(tmp_path, "t2")] == ["t2.json"]
    assert goal_eval.answer_files(tmp_path, "t3") == []


# ---- two held-out vpd4l texts

REAL = vpd_model.TARGET_DIR.exists() and (native.TEXTS / "vpd4l").exists()


@pytest.fixture(scope="module")
def real():
    if not REAL:
        pytest.skip("vpd4l and the texts are not on this machine")
    nat = native.Native()
    ge = goal_eval.GoalEval(nat)
    return nat, ge


@pytest.mark.parametrize("task", ["text9002", "text9010"])
def test_real_texts(real, task):
    import json
    import time

    nat, ge = real
    p = native.TEXTS / "vpd4l" / f"{task}.json"
    ids, targets = native.text(p)
    t0 = time.perf_counter()
    tx = ge.text(ids, targets, native.task_seed(task), json.loads(p.read_text()))
    every = nat.everything(tx.T)
    answer = native.TEXTS / "search_heldout_vr4" / f"{task}.py"
    graphs = [every]
    if answer.exists():
        graphs.append(goal_eval.load_answer(answer, json.loads(p.read_text()), targets, nat)[0])
    base = ge.baselines(tx, [g.count() for g in graphs[1:]])
    rows = ge.score(tx, graphs + list(base.values()), diagnostics=True)
    seconds = time.perf_counter() - t0
    names = ["everything"] + (["search answer"] if len(graphs) > 1 else []) + list(base)
    for nm, row in zip(names, rows):
        print(task, nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("f3_cuts", "carry_positions")})
    print(f"{task}: T {tx.T}, {len(tx.f2)} F2 positions, {len(rows)} circuits, {seconds:.1f} s (text preparation {tx.seconds:.1f} s)")
    assert abs(rows[0]["kl_live"]) < 1e-3  # float32 on the device: the whole model as subcomponents plus remainders
    vpd = rows[names.index("vpd")]
    assert abs(vpd["recovered_live"] - 1) < 1e-6 and abs(vpd["necessity_off"] - 1) < 1e-6
    empty = rows[names.index("empty")]
    assert abs(empty["recovered_off"]) < 1e-6 and abs(empty["necessity_live"]) < 1e-6
    full, _ = naive(nat, tx.ids)
    on = mask_of(nat, tx.T, [])
    assert abs(empty["kl_off"] - _kl(full, naive(nat, tx.ids, on=on, live=False)[0], tx.targets)) < 1e-3 * max(1, empty["kl_off"])
    # one F3 cut and one whole-model cut against the independent forward
    if len(graphs) > 1:
        g = graphs[1]
        pairs = goal_eval._pairs(g)[:2]
        for (w, r), k in zip(pairs, ge.cuts(tx, pairs)):
            want = naive_cut(nat, tx.ids, tx.targets, w, r)
            assert abs(k - want) < 1e-4 + 2e-3 * abs(want), (w, r, k, want)
        on = mask_of(nat, tx.T, g.nodes)
        base_logits, _ = naive(nat, tx.ids, on=on, live=False)
        for (w, r), k in zip(pairs, ge.edge_effects(tx, g, pairs)):
            want = _kl(base_logits, naive(nat, tx.ids, on=on, live=False, cut=(w, r))[0], tx.targets)
            assert abs(k - want) < 1e-4 + 2e-3 * abs(want), (w, r, k, want)


def test_log_scales():
    assert goal_eval.necessity_log(100.0, 0.5, 0.5) == 0.0 and goal_eval.necessity_log(100.0, 100.0, 0.5) == 1.0
    assert abs(goal_eval.necessity_log(100.0, 10.0, 1.0) - 0.5) < 1e-12 and goal_eval.necessity_log(100.0, 0.1, 1.0) == 0.0


def test_embedding_edges_priced_and_cut(tiny):
    """An embedding node is a choice among positions, its edges priced like any other; its stated edges are cut in
    F3 (the reader without the token's embedding) against its unstated readers, checked against naive()."""
    nat, ge, tx = tiny
    p = 4
    nodes = [("h.0.attn.q_proj", p, 1), ("h.0.attn.k_proj", p, 2), ("h.1.mlp.c_fc", p, 3), ("h.0.attn.o_proj", p, 0),
             ("h.1.attn.v_proj", 7, 2)]
    answer = {"nodes": [[n, c, t] for n, t, c in nodes] + [["tok", 0, p]],
              "edges": [[5, 0], [5, 2], [3, 2]]}
    g, d = goal_eval.circuit(answer, tx.targets, nat.C, tx.T)
    assert d == {"invalid_nodes": 0, "invalid_edges": 0} and g.tok_positions == [p] and len(g.tok_edges) == 2
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    want = 5 * math.log2(P * S) + math.log2(P) + math.log2(6) * (2 * (1 + 2) + 0)
    assert abs(goal_eval.size_bits(g, P, S, tx.targets) - want) < 1e-9
    assert goal_eval.size_bits(native.Graph(g.nodes, g.parents, g.out), P, S, tx.targets) == native.Graph(g.nodes, g.parents, g.out).bits(P, S, tx.targets)
    stated, unstated = goal_eval._tok_pairs(g)
    assert [r for _, r in stated] == [nodes[0], nodes[2]] and [r for _, r in unstated] == [nodes[1]]
    on = mask_of(nat, tx.T, g.nodes)
    base, _ = naive(nat, tx.ids, on=on, live=False)
    for (w, r), k in zip(stated + unstated, ge.edge_effects(tx, g, stated + unstated)):
        want = _kl(base, naive(nat, tx.ids, on=on, live=False, cut=(w, r))[0], tx.targets)
        assert abs(k - want) < 1e-4 + 1e-3 * abs(want), (w, r, k, want)
    f3 = ge.edges(tx, [g])[0]
    assert len(f3["stated"]) == 3 and f3["f3"] is not None
    # the whole model's cut of the embedding into a reader
    w, r = stated[0]
    full, st = naive(nat, tx.ids)
    new = vpd_model.rms(st["x_attn"][0][p] - nat.target.wte[tx.ids[p]], nat.target.norms[0], nat.target.eps) @ nat.target.site(r[0]).V[:, r[2]]
    assert abs(ge.cuts(tx, [(w, r)])[0] - _kl(full, naive(nat, tx.ids, {r: new})[0], tx.targets)) < 1e-4


def test_recovered_log():
    assert goal_eval.recovered_log(100.0, 100.0, 0.5) == 0.0
    assert abs(goal_eval.recovered_log(100.0, 0.5, 0.5) - 1) < 1e-12
    assert abs(goal_eval.recovered_log(100.0, 10.0, 1.0) - 0.5) < 1e-12
    assert goal_eval.recovered_log(1.0, 0.5, 2.0) is None


def test_tail_split_is_exact():
    g = torch.Generator().manual_seed(3)
    logp = torch.log_softmax(torch.randn(1, 2, 50, generator=g) * 3, -1)
    logq = torch.log_softmax(torch.randn(4, 2, 50, generator=g) * 3, -1)
    full = carry.kl(logp, logq)
    for p in (0.5, 0.9, 0.99, 1.0):
        coarse, tail = goal_eval.tail_split(logp, logq, p)
        assert torch.allclose(coarse + tail, full, atol=1e-5), p
    coarse, tail = goal_eval.tail_split(logp, logq, 1.0)
    assert torch.allclose(tail, torch.zeros_like(tail), atol=1e-6)  # the nucleus holds every token
    same = goal_eval.tail_split(logp, logp.expand(4, -1, -1), 0.9)
    assert torch.allclose(same[0], torch.zeros(4), atol=1e-6) and torch.allclose(same[1], torch.zeros(4), atol=1e-6)


def test_real_removal(tiny):
    """n_real: the real model without the circuit's subcomponents (everything else on, remainders on), against
    naive() and Native.without."""
    nat, ge, tx = tiny
    graphs = [random_graph(nat, tx, 41, k=10), random_graph(nat, tx, 42, k=25), native.Graph()]
    rows = ge.real_removal(tx, graphs)
    full, _ = naive(nat, tx.ids)
    for g, row in zip(graphs, rows):
        on = {n: ~m for n, m in mask_of(nat, tx.T, g.nodes).items()}
        want = _kl(full, naive(nat, tx.ids, on=on, live=True)[0], tx.targets)
        assert abs(row["kl_real_minus"] - want) < 1e-4 + 1e-4 * want
        assert abs(row["kl_real_minus"] - float(carry.kl(tx.ref, nat.without(tx.ids, tx.targets, g.nodes))[0])) < 1e-5
        e, v = ge._bounds(tx, "off")
        assert row["n_real_off"] == goal_eval.necessity_log(e, row["kl_real_minus"], v)
    assert rows[2]["n_real_live"] == 0.0


def _graph_checks(nat, ge, tx, nodes, rel):
    """Graph F1 on a node set: every connection stated is the complete run (rel: float32 between the reference forward
    and the masked one); no connection stated, remainders off, is the empty circuit; N does not depend on edges."""
    par, out = native.all_edges(nodes, tx.targets)
    rows = ge.sufficiency(tx, [native.Graph(nodes, par, out), native.Graph(nodes, complete=True), native.Graph(nodes, {}, [])])
    every, whole, none = rows
    for conv in ("off", "live"):
        assert abs(every[f"kl_{conv}"] - whole[f"kl_{conv}"]) < rel * max(1.0, whole[f"kl_{conv}"]), (conv, every[f"kl_{conv}"], whole[f"kl_{conv}"])
        assert none[f"n_{conv}"] == every[f"n_{conv}"] == whole[f"n_{conv}"]  # N is node removal: edges do not enter
    assert abs(none["kl_off"] - none["kl_empty_off"]) < 1e-4 * max(1.0, none["kl_empty_off"])
    return every, whole, none


def test_graph_f1_runs_the_stated_graph(tiny):
    """F1 runs the graph an answer states: all connections = the complete run, none = the empty circuit, and all but
    one = the circuit's run with that one connection cut, checked against the independent forward."""
    nat, ge, tx = tiny
    g = random_graph(nat, tx, seed=11, k=10, stated=1.0)
    nodes = sorted(g.node_set())
    _graph_checks(nat, ge, tx, nodes, 1e-4)
    par, out = native.all_edges(nodes, tx.targets)
    full, _ = naive(nat, tx.ids)
    on = mask_of(nat, tx.T, nodes)
    pairs = goal_eval._pairs(native.Graph(nodes, par, out))
    assert pairs
    for w, r in pairs[:6]:
        less = native.Graph(nodes, {rr: [x for x in ws if not (x == w and rr == r)] for rr, ws in par.items()},
                            [x for x in out if not (x == w and r is None)])
        got = ge.graph_kl(tx, [less], live=False)[0]
        want = _kl(full, naive(nat, tx.ids, on=on, live=False, cut=(w, r))[0], tx.targets)
        assert abs(got - want) < 1e-4 + 1e-3 * abs(want), (w, r, got, want)


def test_graph_f1_on_the_teachers_text9000_circuit(real):
    """The lead's check on pymove's teacher circuit for text9000: graph F1 with every connection stated is the
    complete circuit's F1 and with none the empty circuit's."""
    import json

    nat, ge = real
    teacher = Path.home() / "mpd-data/graph_oracle/teacher/joint_heldout/text9000.json"
    if not teacher.exists():
        pytest.skip("no teacher circuit for text9000")
    p = native.TEXTS / "vpd4l" / "text9000.json"
    ids, targets = native.text(p)
    tx = ge.text(ids, targets, native.task_seed("text9000"), json.loads(p.read_text()))
    nodes = sorted((n, t, c) for n, c, t in json.loads(teacher.read_text())["nodes"] if n != "tok")
    every, whole, none = _graph_checks(nat, ge, tx, nodes, 1e-3)
    print(f"text9000 teacher, {len(nodes)} nodes: KL live {every['kl_live']:.4f} stated / {whole['kl_live']:.4f} complete, "
          f"off {every['kl_off']:.4f} / {whole['kl_off']:.4f}; none {none['kl_live']:.2f} = empty {none['kl_empty_live']:.2f}")


def test_size_scale_is_vpds_whole_answer(tiny):
    nat, ge, tx = tiny
    n = sum(int(m.sum()) for m in tx.vpd.values())
    assert n > 0 and abs(ge.size_scale(tx) - n * math.log2(nat.positions(tx.targets) * sum(nat.C.values()))) < 1e-9


def tiny_pool(nat, ids):
    """An adversary pool from the tiny model's causal importance on a text: one pattern per position."""
    ci = nat.importance(ids)
    sites, cols, offsets = [], [], [0]
    for t in range(len(ids)):
        for i, n in enumerate(nat.names):
            c = (ci[n][t] > 0).nonzero().flatten().to(torch.int32).cpu()
            sites.append(torch.full((len(c),), i, dtype=torch.int16))
            cols.append(c)
        offsets.append(offsets[-1] + sum(len(x) for x in cols[len(cols) - len(nat.names):]))
    return {"names": list(nat.names), "sources": [("tiny", t) for t in range(len(ids))], "site": torch.cat(sites),
            "c": torch.cat(cols), "offsets": torch.tensor(offsets, dtype=torch.int64)}


def test_fidelity_vpds_answer_is_its_own_ceiling(tiny):
    """VPD's own answer as the circuit: every family's runs are VPD's, so every score is 1."""
    nat, ge, tx = tiny
    g = native.Graph(masks=tx.vpd)
    pool = tiny_pool(nat, tx.ids)
    row = ge.fidelity(tx, [g], pool, draws=4, steps=2, restarts=2, curve=(0, 2), fresh=pool, interchange=2, real=True)[0]
    for k in ("s_point", "s_nodes", "s_edges", "f1_adv", "f1_adv_random", "c", "s_interchange"):
        assert row[k] is not None and abs(row[k] - 1) < 1e-4, (k, row[k])
    assert all(abs(v - 1) < 1e-4 for v in row["f1_flips_patterns_fresh"].values()) and row["n_real"] is not None
    sub = sorted(g.node_set())[::4]
    whole = native.Graph(sub, *native.all_edges(sub, tx.targets))  # every connection: reads equal the masked run's
    row = ge.fidelity(tx, [whole], pool, draws=3, steps=1, restarts=1, curve=(0,), interchange=2, reads=True)[0]
    for k in ("reads_nodes", "reads_edges", "reads_interchange", "dreads_nodes", "dreads_edges", "dreads_interchange"):
        assert row[k] is not None and row[k] >= 0, (k, row[k])


def test_read_error():
    pairs = {("a", 0, 0): [(1.0, 1.0), (2.0, 2.0)], ("b", 0, 0): [(1.0, 0.0), (1.0, 0.0)], ("c", 0, 0): [(0.0, 3.0)]}
    assert abs(goal_eval._read_error(pairs) - 0.5) < 1e-12  # a: 0, b: 1, c: no model read (skipped)


def test_fidelity_cuts_and_removals(tiny):
    """The connection-cut family's pieces: the whole model with one connection cut in graph_ref is cuts()'s path
    patching; a stated graph's zero edge strength is the graph without that connection."""
    nat, ge, tx = tiny
    sub = sorted(native.Graph(masks=tx.vpd).node_set())[::4]
    g = native.Graph(sub, *native.all_edges(sub, tx.targets))
    pairs = goal_eval._pairs(g)
    assert len(pairs) >= 4
    model = native.Graph(masks={n: torch.ones(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
    some = pairs[:4]
    for (w, r), want in zip(some, ge.cuts(tx, some)):
        k = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, model, cuts={r: [w]}))[0])
        assert abs(k - want) < 1e-5 + 1e-4 * abs(want), (w, r, k, want)
    order = [(w, r) for r, ws in g.parents.items() for w in ws] + [(w, None) for w in g.out]
    for j in range(min(3, len(order))):
        v = torch.ones(len(order))
        v[j] = 0.0
        w, r = order[j]
        less = native.Graph(g.nodes, {rr: [x for x in ws if not (x == w and rr == r)] for rr, ws in g.parents.items()},
                            [x for x in g.out if not (x == w and r is None)])
        got = float(carry.kl(tx.ref, ge._graph_logq(tx.ids, tx.targets, [g], [v]))[0])
        assert abs(got - ge.graph_kl(tx, [less], live=True)[0]) < 1e-5


def test_adversary_gradient_and_rescore(tiny):
    """The adversary's objective: its gradient in the pattern weights and omissions against central differences, and
    a binary mask's KLs against graph_ref (the circuit as stated, the adversary's subcomponents at u, C's semantics)
    and the masked forward (VPD's answer and the adversary's)."""
    nat, ge, tx = tiny
    sub = sorted(native.Graph(masks=tx.vpd).node_set())[::3]
    par, out = native.all_edges(sub, tx.targets)
    g = native.Graph(sub, par, out)
    pool = tiny_pool(nat, tx.ids)
    gen = torch.Generator().manual_seed(3)
    pick, ((kls, idx, cols),) = ge._adv_problem(tx, [g], pool, 6, gen)
    w = torch.rand(1, len(pick), generator=gen).to(nat.dev).requires_grad_(True)
    b = {n: torch.rand(1, len(idx[n][0]), generator=gen).to(nat.dev).requires_grad_(True) for n in nat.names}
    kc, kv = kls(w, b)
    obj = kc.log() - kv.log()
    gw = torch.autograd.grad(obj.sum(), [w] + list(b.values()), allow_unused=True)
    eps = 1e-2
    with torch.no_grad():
        for j in range(len(pick)):
            d = torch.zeros_like(w)
            d[0, j] = eps
            hi = [x.log() for x in kls(w + d, b)]
            lo = [x.log() for x in kls(w - d, b)]
            fd = float((hi[0] - hi[1] - lo[0] + lo[1]) / (2 * eps))
            assert abs(fd - float(gw[0][0, j])) < 2e-2 * max(1.0, abs(fd)), (j, fd, float(gw[0][0, j]))
        n = next(n for n in nat.names if len(idx[n][0]))
        j = 0
        bp = {m: x.clone() for m, x in b.items()}
        bm = {m: x.clone() for m, x in b.items()}
        bp[n][0, j] += eps
        bm[n][0, j] -= eps
        hi, lo = [x.log() for x in kls(w, bp)], [x.log() for x in kls(w, bm)]
        fd = float((hi[0] - hi[1] - lo[0] + lo[1]) / (2 * eps))
        assert abs(fd - float(gw[1 + nat.names.index(n)][0, j])) < 2e-2 * max(1.0, abs(fd))
        wb = (w > 0.5).float()
        bb = {m: (x > 0.5).float() for m, x in b.items()}
        kc, kv = kls(wb, bb)
        # the adversary's mask over all subcomponents, rebuilt from the pool's patterns and the omissions on
        last = max(tx.targets)
        on_pat = [goal_eval.pattern(pool, pick[j]) for j in wb[0].nonzero().flatten().tolist()]
        nodes = g.node_set()
        u = {m: torch.zeros(1, tx.T, nat.C[m], device=nat.dev) for m in nat.names}
        vm = {m: tx.vpd[m].clone() for m in nat.names}
        for m in nat.names:
            for p in on_pat:
                if m in p:
                    u[m][0, :last + 1, p[m].to(nat.dev)] = 1.0
                    vm[m][:last + 1, p[m].to(nat.dev)] = True
            u[m][0] *= (~tx.vpd[m]).float()
            for t_, k_ in zip(*[x[bb[m][0].bool()] for x in idx[m]]):
                u[m][0, int(t_), int(cols[m][k_])] = 1.0
            for (mm, t_, c_) in nodes:
                if mm == m:
                    u[m][0, t_, c_] = 0.0
        want_c = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, g, u=u)))
        want_v = float(carry.kl(tx.ref, ge.runs(tx.ids, tx.targets, [vm], True)))
        assert abs(float(kc[0]) - want_c) < 1e-4 * max(1.0, want_c) and abs(float(kv[0]) - want_v) < 1e-4 * max(1.0, want_v)


def test_node_family_is_the_masked_runs(tiny):
    """The node family's runs on the engine's columns equal the masked forward over every column (the model and VPD's
    answer without each draw's nodes, a complete circuit's run without them) and native.run's graph without them."""
    nat, ge, tx = tiny
    nodes = sorted(native.Graph(masks=tx.vpd).node_set())
    graphs = [native.Graph(nodes[::3], *native.all_edges(nodes[::3], tx.targets)), native.Graph(nodes[1::4], complete=True)]
    draws = 3
    got = ge._node_family(tx, graphs, draws)
    for g, row in zip(graphs, got):
        gn = sorted(g.node_set())
        rng = ge._rng(tx, g, "nodes")
        drops = []
        for j in range(draws):
            d = [nd for nd in gn if rng.random() < (j + 0.5) / draws]
            drops.append(d or [gn[int(rng.integers(len(gn)))]])

        def without(base, d):
            m = {n: base[n].clone() for n in nat.names}
            for n, t, c in d:
                m[n][t, c] = False
            return m

        ones = {n: torch.ones(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        lm = ge.runs(tx.ids, tx.targets, [without(ones, d) for d in drops], True)
        lv = ge.runs(tx.ids, tx.targets, [without(tx.vpd, d) for d in drops], True)
        if g.complete:
            lc = ge.runs(tx.ids, tx.targets, [without(ge.masks(g, tx.T), d) for d in drops], True)
        else:
            lc = ge._graph_logq(tx.ids, tx.targets, [goal_eval._drop_nodes(g, set(d)) for d in drops])
        for key, want in (("kl_nodes_c", float(carry.kl(lm, lc).mean())), ("kl_nodes_vpd", float(carry.kl(lm, lv).mean()))):
            assert abs(row[key] - want) < 1e-6 + 1e-4 * abs(want), (key, row[key], want)


def test_adversary_batch_is_per_circuit(tiny):
    """The adversary's batched objective (every circuit's rows in one run of each kind: stated graphs as a list, complete
    ones, VPD's answer) gives each circuit's own KLs and gradients within float32 rounding."""
    nat, ge, tx = tiny
    nodes = sorted(native.Graph(masks=tx.vpd).node_set())
    graphs = [native.Graph(nodes[::3], *native.all_edges(nodes[::3], tx.targets)), native.Graph(nodes[1::4], complete=True),
              native.Graph(nodes[::5], *native.all_edges(nodes[::5], tx.targets))]
    pool = tiny_pool(nat, tx.ids)
    gen = torch.Generator().manual_seed(5)
    pick, problems, batch = ge._adv_problem(tx, graphs, pool, 6, gen, batched=True)
    items = [(i, torch.rand(2 + i, len(pick), generator=gen).to(nat.dev).requires_grad_(True),
              {n: torch.rand(2 + i, len(idx[n][0]), generator=gen).to(nat.dev).requires_grad_(True) for n in nat.names})
             for i, (_, idx, _) in enumerate(problems)]
    got = batch(items)
    loss = sum((kc.log() - kv.log()).sum() for kc, kv in got)
    leaves = [x for _, w, b in items for x in [w] + [b[n] for n in nat.names]]
    gb = torch.autograd.grad(loss, leaves, allow_unused=True)
    k = 0
    for (i, w, b), (kc, kv) in zip(items, got):
        wc, bc = w.detach().clone().requires_grad_(True), {n: x.detach().clone().requires_grad_(True) for n, x in b.items()}
        kc1, kv1 = problems[i][0](wc, bc)
        assert torch.allclose(kc, kc1, rtol=1e-5, atol=1e-7) and torch.allclose(kv, kv1, rtol=1e-5, atol=1e-7)
        g1 = torch.autograd.grad((kc1.log() - kv1.log()).sum(), [wc] + [bc[n] for n in nat.names], allow_unused=True)
        for x, y in zip(gb[k:k + len(g1)], g1):
            assert (x is None) == (y is None) and (x is None or torch.allclose(x, y, rtol=1e-4, atol=1e-6))
        k += len(g1)


def test_fidelity_table_groups():
    rows = [{"task": "a", "local": True, "kind": "greedy", "nodes": 30, "kl_empty": 100.0, "kl_vpd": 0.1, "n_real": 0.5,
             "s_point": 0.9, "n_point": 0.6, "size": 0.4, "s_nodes": 0.8, "s_edges": 0.7, "s_interchange": 0.6,
             "f1_adv_curve": {"0": 0.9, "20": 0.5}, "f1_flips_omitted": {"1": 0.8, "4": 0.5}, "flips_to_break": None,
             "f1_flips_patterns_fresh": {"1": 0.3}, "c": 0.7},
            {"task": "b", "local": False, "kind": "VPD top 30", "nodes": 30, "kl_empty": 1.5, "kl_vpd": 1.0, "n_real": 0.2,
             "s_point": 1.0, "n_point": 0.1, "size": 0.4, "s_nodes": 1.0, "s_edges": None, "f1_adv_curve": {"0": 1.0, "20": 1.0},
             "f1_flips_omitted": {"1": 0.4, "4": 0.3}, "flips_to_break": 1, "c": 1.0}]
    text = goal_eval.fidelity_table(rows)
    assert "trivial" in text and "local" in text and "adv@20" in text and " 0.500 " in text and "VPD top k" in text
    assert "In-sample" in text and "flip4" in text and "fresh pat1" in text and " 3.000 " in text  # log2(2 x 4) for no break


def test_pattern_pool_is_ci_per_position(tiny, tmp_path, monkeypatch):
    nat, ge, tx = tiny
    task = tmp_path / "tiny0.json"
    task.write_text("{}")
    monkeypatch.setattr(native, "text", lambda p: (tx.ids, tx.targets))
    pool = goal_eval.build_patterns([task, task], tmp_path / "pool.pt", nat)
    ci = nat.importance(tx.ids)
    assert len(pool["sources"]) == 2 * tx.T
    for k in (0, 3, tx.T + 5):
        got = goal_eval.pattern(pool, k)
        t = k % tx.T
        for n in nat.names:
            want = (ci[n][t] > 0).nonzero().flatten().cpu()
            assert torch.equal(got.get(n, torch.zeros(0, dtype=torch.long)), want), (k, n)


def test_adversary_flips_finds_the_worst_single(tiny):
    """A budget of one switched-on omission: the search's pick is the worst single omission (brute force over all), and
    its KLs are that mask's."""
    nat, ge, tx = tiny
    sub = sorted(native.Graph(masks=tx.vpd).node_set())[::3]
    g = native.Graph(sub, *native.all_edges(sub, tx.targets))
    pool = tiny_pool(nat, tx.ids)
    row = ge.adversary_flips(tx, [g], pool, "omitted", ms=(1,), steps=4, restarts=2, k=6)[0]
    gen = torch.Generator().manual_seed(int(tx.seed) % (2 ** 63))
    pick, ((kls, idx, cols),) = ge._adv_problem(tx, [g], pool, 6, gen)
    sizes = [len(idx[n][0]) for n in nat.names]
    N = sum(sizes)
    with torch.no_grad():
        eye = torch.eye(N, device=nat.dev)
        kc, kv = kls(torch.zeros(N, len(pick), device=nat.dev), dict(zip(nat.names, torch.split(eye, sizes, 1))))
        gap = kc.log() - kv.log()
    j = int(gap.argmax())
    got_c, got_v = row["kl_flips_omitted"][1]
    assert abs(math.log(got_c) - math.log(got_v) - float(gap[j])) < 1e-3 * max(1.0, abs(float(gap[j]))), (got_c, got_v, float(gap[j]))
    assert len(row["mask_flips_omitted"][1]) == 1


def test_graph_ref_is_graph_inside_masked_outside(tiny):
    """C's semantics (graph_ref): every connection stated is the masked run of the circuit and the mask u (with no u
    and with a random one, remainders live and off); all but one connection is the independent forward with that
    connection cut; no connection stated, remainders off, is the empty circuit; the gradient in u is autograd's."""
    nat, ge, tx = tiny
    sub = sorted(native.Graph(masks=tx.vpd).node_set())[::4]
    par, out = native.all_edges(sub, tx.targets)
    g = native.Graph(sub, par, out)
    circ = mask_of(nat, tx.T, sub)
    gen = torch.Generator().manual_seed(9)
    u = {n: (torch.rand(1, tx.T, nat.C[n], generator=gen) < 0.2).float().to(nat.dev) for n in nat.names}
    for live in (True, False):
        for uu in (None, u):
            got = ge.graph_ref(tx.ids, tx.targets, g, u=uu, live=live)
            sets = [{n: torch.maximum(circ[n].float(), (uu[n][0] if uu is not None else 0 * circ[n].float())) for n in nat.names}]
            want = ge.runs(tx.ids, tx.targets, sets, live)
            assert float(carry.kl(want, got)) < 1e-5 and float(carry.kl(got, want)) < 1e-5, (live, uu is None)
    full, _ = naive(nat, tx.ids)
    for w, r in goal_eval._pairs(g)[:5]:
        less = native.Graph(sub, {rr: [x for x in ws if not (x == w and rr == r)] for rr, ws in par.items()},
                            [x for x in out if not (x == w and r is None)])
        got = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, less)))
        want = _kl(full, naive(nat, tx.ids, on=circ, live=True, cut=(w, r))[0], tx.targets)
        assert abs(got - want) < 1e-5 + 1e-4 * abs(want), (w, r, got, want)
    none = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, native.Graph(sub, {}, []), live=False)))
    assert abs(none - ge._bounds(tx, "off")[0]) < 1e-4 * max(1.0, none)
    uu = {n: (0.3 * torch.rand(1, tx.T, nat.C[n], generator=gen)).to(nat.dev).requires_grad_(True) for n in nat.names}
    half = native.Graph(sub, {r: ws[: len(ws) // 2] for r, ws in par.items()}, out[: len(out) // 2])
    kl = carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, half, u=uu)).sum()
    gr = torch.autograd.grad(kl, list(uu.values()))
    i = max(range(len(nat.names)), key=lambda j: float(gr[j].abs().max()))
    n = nat.names[i]
    _, t, c = [int(x) for x in (gr[i].abs() == gr[i].abs().max()).nonzero()[0]]
    with torch.no_grad():
        eps = 2e-2
        hi = {m: x.clone() for m, x in uu.items()}
        lo = {m: x.clone() for m, x in uu.items()}
        hi[n][0, t, c] += eps
        lo[n][0, t, c] -= eps
        fd = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, half, u=hi)) - carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, half, u=lo))) / (2 * eps)
    assert abs(fd - float(gr[i][0, t, c])) < 2e-2 * abs(fd) + 1e-5, (fd, float(gr[i][0, t, c]))


def test_families_on_the_engine_are_graph_ref(tiny):
    """The connection-cut and interchange families and graph F1, run on the engine (C's semantics there, every draw a
    row), equal graph_ref draw by draw for a stated and a complete circuit."""
    import numpy as np

    nat, ge, tx = tiny
    T = tx.T
    nodes = sorted(native.Graph(masks=tx.vpd).node_set())
    graphs = [native.Graph(nodes[::3], *native.all_edges(nodes[::3], tx.targets)), native.Graph(nodes[1::4], complete=True)]
    graphs[0] = native.Graph(graphs[0].nodes, {r: ws[::2] for r, ws in graphs[0].parents.items()}, graphs[0].out)
    model = native.Graph(masks={n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
    vpd = native.Graph(masks=tx.vpd)
    draws = 4
    got = ge._edge_family(tx, graphs, draws)
    for g, row in zip(graphs, got):
        par, outs = (native.all_edges(g.node_set(), tx.targets) if g.complete else (g.parents, g.out))
        readers = [r for r in sorted(par) if par[r]] + ([None] if outs else [])
        rng = ge._rng(tx, g, "edges")
        kc, kv = [], []
        for j in range(draws):
            r = readers[int(rng.integers(len(readers)))]
            ws = list(par[r]) if r is not None else list(outs)
            cut = [w for w in ws if rng.random() < (j + 0.5) / draws] or [ws[int(rng.integers(len(ws)))]]
            lm = ge.graph_ref(tx.ids, tx.targets, model, cuts={r: cut})
            kc.append(float(carry.kl(lm, ge.graph_ref(tx.ids, tx.targets, g, cuts={r: cut}))[0]))
            kv.append(float(carry.kl(lm, ge.graph_ref(tx.ids, tx.targets, vpd, cuts={r: cut}))[0]))
        for key, want in (("kl_edges_c", float(np.mean(kc))), ("kl_edges_vpd", float(np.mean(kv)))):
            assert abs(row[key] - want) < 1e-6 + 1e-4 * abs(want), (key, row[key], want)
    vals = [{nd: torch.tensor(0.5 + 0.1 * j, device=nat.dev) for nd in graphs[0].node_set() if nd[1] == p}
            for j, p in enumerate(sorted({t for _, t, _ in graphs[0].node_set()})[:3])]
    for g in graphs:
        lm, lv, lc = ge._sides(tx, graphs)(g, patch=vals)
        for b, v in enumerate(vals):
            want = [ge.graph_ref(tx.ids, tx.targets, x, patch=v)[0] for x in (model, vpd, g)]
            for x, y in zip((lm[b], lv[b], lc[b]), want):
                assert float(carry.kl(y[None], x[None])[0]) < 1e-6
    stated = [g for g in graphs if not g.complete]
    for k, g in zip(ge._graph_kl(tx.ids, tx.targets, tx.ref, stated, True, 0.0), stated):
        want = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, g, live=True))[0])
        assert abs(k - want) < 1e-6 + 1e-4 * abs(want)


def test_family_reads_are_graph_ref(tiny):
    """C's second observable on the engine: the node, connection-cut and interchange families' read errors equal
    graph_ref's records draw by draw (a stated and a complete circuit)."""
    nat, ge, tx = tiny
    T = tx.T
    nodes = sorted(native.Graph(masks=tx.vpd).node_set())
    stated = native.Graph(nodes[::3], *native.all_edges(nodes[::3], tx.targets))
    stated = native.Graph(stated.nodes, {r: ws[::2] for r, ws in stated.parents.items()}, stated.out)
    graphs = [stated, native.Graph(nodes[1::4], complete=True)]
    model = native.Graph(masks={n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
    draws = 3

    def ref_reads(g, runs):  # runs: per draw (model graph, circuit graph, kwargs, the nodes read)
        pairs = {}
        for gm, gc, kw, keep in runs:
            rm, rc = {}, {}
            ge.graph_ref(tx.ids, tx.targets, gm, record=rm, reads=keep, **kw)
            ge.graph_ref(tx.ids, tx.targets, gc, record=rc, reads=keep, **kw)
            for nd in keep:
                pairs.setdefault(nd, []).append((float(rm[nd][0]), float(rc[nd][0])))
        return goal_eval._read_error(pairs)

    for g, row in zip(graphs, ge._node_family(tx, graphs, draws, reads=True)):
        gn = sorted(g.node_set())
        rng = ge._rng(tx, g, "nodes")
        runs = []
        for j in range(draws):
            d = [nd for nd in gn if rng.random() < (j + 0.5) / draws] or [gn[int(rng.integers(len(gn)))]]
            ones = {n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            gm = ge.masks(g, T) if g.complete else None
            for n, t, c in d:
                ones[n][t, c] = False
                if gm is not None:
                    gm[n][t, c] = False
            gc = native.Graph(masks=gm) if g.complete else goal_eval._drop_nodes(g, set(d))
            runs.append((native.Graph(masks=ones), gc, {}, [nd for nd in gn if nd not in set(d)]))
        want = ref_reads(g, runs)
        assert abs(row["reads_nodes"] - want) < 1e-5 + 1e-4 * abs(want), ("nodes", row["reads_nodes"], want)
    for g, row in zip(graphs, ge._edge_family(tx, graphs, draws, reads=True)):
        gn = sorted(g.node_set())
        par, outs = (native.all_edges(g.node_set(), tx.targets) if g.complete else (g.parents, g.out))
        readers = [r for r in sorted(par) if par[r]] + ([None] if outs else [])
        rng = ge._rng(tx, g, "edges")
        runs = []
        for j in range(draws):
            r = readers[int(rng.integers(len(readers)))]
            ws = list(par[r]) if r is not None else list(outs)
            cut = [w for w in ws if rng.random() < (j + 0.5) / draws] or [ws[int(rng.integers(len(ws)))]]
            runs.append((model, g, {"cuts": {r: cut}}, gn))
        want = ref_reads(g, runs)
        assert abs(row["reads_edges"] - want) < 1e-5 + 1e-4 * abs(want), ("edges", row["reads_edges"], want)
    vals = [{nd: torch.tensor(0.5 + 0.1 * j, device=nat.dev) for nd in stated.node_set() if nd[1] == p}
            for j, p in enumerate(sorted({t for _, t, _ in stated.node_set()})[:3])]
    lm, lv, lc, rm, rc = ge._sides(tx, graphs)(stated, patch=vals, reads=True)
    for b, v in enumerate(vals):
        keep = [nd for nd in sorted(stated.node_set()) if nd not in v]
        wm, wc = {}, {}
        ge.graph_ref(tx.ids, tx.targets, model, patch=v, record=wm, reads=keep)
        ge.graph_ref(tx.ids, tx.targets, stated, patch=v, record=wc, reads=keep)
        for nd in keep:
            assert abs(float(rm[nd][b]) - float(wm[nd][0])) < 1e-5 + 1e-4 * abs(float(wm[nd][0]))
            assert abs(float(rc[nd][b]) - float(wc[nd][0])) < 1e-5 + 1e-4 * abs(float(wc[nd][0]))


def test_interchange_family(tiny):
    """Interchange: VPD's answer as the circuit agrees with VPD's own run draw by draw; a stated circuit with every
    connection and some activations fixed (graph_ref) is the independent forward with them fixed."""
    nat, ge, tx = tiny
    row = ge._interchange_family(tx, [native.Graph(masks=tx.vpd)], draws=3)[0]
    assert abs(row["kl_interchange_c"] - row["kl_interchange_vpd"]) < 1e-6
    sub = sorted(native.Graph(masks=tx.vpd).node_set())[::3]
    g = native.Graph(sub, *native.all_edges(sub, tx.targets))
    p = sorted({t for _, t, _ in sub if t > 0})[0]
    vals = {nd: torch.tensor(0.7, device=nat.dev) for nd in sub if nd[1] == p}
    got = float(carry.kl(tx.ref, ge.graph_ref(tx.ids, tx.targets, g, patch=vals))[0])
    full, _ = naive(nat, tx.ids)
    want = _kl(full, naive(nat, tx.ids, replace={nd: 0.7 for nd in vals}, on=mask_of(nat, tx.T, sub), live=True)[0], tx.targets)
    assert abs(got - want) < 1e-5 + 1e-4 * abs(want), (got, want)


def test_response_error():
    base = {("a", 0, 0): (1.0, 5.0), ("b", 0, 0): (0.0, 0.0)}
    pairs = {("a", 0, 0): [(2.0, 6.0), (0.0, 4.0)], ("b", 0, 0): [(1.0, 0.0)]}
    # a's responses match exactly despite the offset; b's model response 1 is missed: sqrt(1 / 3)
    assert abs(goal_eval._response_error(pairs, base) - math.sqrt(1 / 3)) < 1e-12
