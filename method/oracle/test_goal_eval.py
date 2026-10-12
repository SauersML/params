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


def test_recovered_log():
    assert goal_eval.recovered_log(100.0, 100.0, 0.5) == 0.0
    assert abs(goal_eval.recovered_log(100.0, 0.5, 0.5) - 1) < 1e-12
    assert abs(goal_eval.recovered_log(100.0, 10.0, 1.0) - 0.5) < 1e-12
    assert goal_eval.recovered_log(1.0, 0.5, 2.0) is None
