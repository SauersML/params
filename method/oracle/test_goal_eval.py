"""Checks of the goal scorecard (goal_eval.py): exactness on a tiny random model, every cut against an independent
forward pass, and the same on two held-out vpd4l texts (skipped without ~/mpd-data's model).

  MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_goal_eval.py -s
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


def tiny_native(seed: int = 0, layers: int = 2, d: int = 16, heads: int = 2, vocab: int = 40, C: int = 6) -> native.Native:
    """A Native over a random two-layer model of vpd4l's architecture with random subcomponents (their remainders
    large: the decomposition is far from exact) and a causal-importance function (a random linear map of each
    matrix's input, clamped), on the CPU in float64-free float32."""
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
        UV[n] = (r(C, o, scale=1.0 / math.sqrt(C)), r(i, C, scale=1.5 / math.sqrt(i)))
    proj = {n: r(UV[n][1].shape[0], C) for n in names}
    vpd = vpd_model.VPD(target, None, UV)
    vpd.ci_fn = lambda acts: {n: acts[n] @ proj[n].to(acts[n].device) for n in names}
    nat = native.Native.__new__(native.Native)
    nat.dev = "cpu"
    nat.target, nat.vpd, nat.names, nat.C = target, vpd, vpd.names, vpd.C
    nat._consts, nat._rec = {}, (None, None, None)
    for n in names:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(native._forward_cached, st)
    return nat


# ---- an independent forward pass (no Native.run)


@torch.no_grad()
def naive(nat, ids, replace=None):
    """The model's forward pass written out, every matrix as subcomponents plus remainder: (logits [T, V], states:
    every activation per matrix [T, C], the stream before each norm, attention patterns and outputs, MLP inputs).
    replace: {(matrix, position, index): value} overrides those activations."""
    tg = nat.target
    T, H, hd = len(ids), tg.n_head, tg.hd
    dev = tg.wte.device
    x = tg.wte[torch.tensor(ids, device=dev)].clone()
    st = {"a": {}, "x_attn": [], "x_mlp": [], "P": [], "att": [], "pre": []}

    def act(n, inp):
        a = inp @ tg.site(n).V
        for (m, t, c), v in (replace or {}).items():
            if m == n:
                a[t, c] = v
        st["a"][n] = a
        return a @ tg.site(n).U + inp @ tg.site(n).delta_T

    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    for i in range(tg.n_layer):
        nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
        st["x_attn"].append(x.clone())
        z = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
        q, k, v = (act(nm[kk], z).view(T, H, hd).transpose(0, 1) for kk in ("q_proj", "k_proj", "v_proj"))
        q, k = tg._rope(q, T), tg._rope(k, T)
        P = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
        att = (P @ v).transpose(0, 1).reshape(T, -1)
        st["P"].append(P)
        st["att"].append(att)
        x = x + act(nm["o_proj"], att)
        st["x_mlp"].append(x.clone())
        z2 = vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps)
        pre = act(nm["c_fc"], z2)
        st["pre"].append(pre)
        x = x + act(nm["down_proj"], vpd_model.gelu_tanh(pre))
    st["x_final"] = x
    return vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T, st


def naive_cut(nat, ids, targets, w, r):
    """The KL in bits of the model at the targets from the model with the connection w -> r cut, computed from
    naive()'s states: the reader's input without the writer's output, then a forward pass with that activation."""
    tg = nat.target
    full, st = naive(nat, ids)
    logp = torch.log_softmax(full[targets], -1)
    wl, wk = native._name_layer_kind(w[0])
    write = st["a"][w[0]][w[1], w[2]] * tg.site(w[0]).U[w[2]]
    if r is None:
        x = st["x_final"].clone()
        x[w[1]] -= write
        logq = torch.log_softmax((vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T)[targets], -1)
        return float((logp.exp() * (logp - logq)).sum()) / LN2
    rl, rk = native._name_layer_kind(r[0])
    V = tg.site(r[0]).V[:, r[2]]
    if rk in ("q_proj", "k_proj", "v_proj"):
        new = vpd_model.rms(st["x_attn"][rl][r[1]] - write, tg.norms[2 * rl], tg.eps) @ V
    elif rk == "c_fc":
        new = vpd_model.rms(st["x_mlp"][rl][r[1]] - write, tg.norms[2 * rl + 1], tg.eps) @ V
    elif rk == "o_proj":
        H, hd = tg.n_head, tg.hd
        share = st["P"][rl][:, r[1], w[1]].repeat_interleave(hd) * write
        new = (st["att"][rl][r[1]] - share) @ V
    else:
        new = vpd_model.gelu_tanh(st["pre"][rl][r[1]] - write) @ V
    cut, _ = naive(nat, ids, {r: new})
    logq = torch.log_softmax(cut[targets], -1)
    return float((logp.exp() * (logp - logq)).sum()) / LN2


def closed_empty(nat, tx, held: bool):
    """F1 of the empty circuit on the text, from naive()'s states: nothing but the embedding and, held, every
    remainder's write in the model (W - V U applied to its input there)."""
    tg = nat.target
    full, st = naive(nat, tx.ids)
    x = tg.wte[torch.tensor(tx.ids, device=tg.wte.device)].clone()
    if held:
        for i in range(tg.n_layer):
            x = x + st["att"][i] @ tg.site(native.site_name(i, "o_proj")).delta_T
            x = x + vpd_model.gelu_tanh(st["pre"][i]) @ tg.site(native.site_name(i, "down_proj")).delta_T
    logp = torch.log_softmax(full[tx.targets].to(x.device), -1)
    logq = torch.log_softmax((vpd_model.rms(x, tg.ln_f, tg.eps) @ tg.wte.T)[tx.targets], -1)
    return float((logp.exp() * (logp - logq)).sum()) / LN2


# ---- the tiny model


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native()
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


def test_full_circuit_is_the_model(tiny):
    nat, ge, tx = tiny
    every = nat.everything(tx.T)
    f1 = ge.sufficiency(tx, [every])
    assert abs(f1["text_held"][0]) < 1e-5
    f2 = ge.interchange(tx, [every])
    assert tx.f2, "the tiny text needs F2 positions"
    assert abs(f2["share"][0] - 1) < 1e-4


def test_empty_known_values(tiny):
    nat, ge, tx = tiny
    f1 = ge.sufficiency(tx, [native.Graph()])
    assert abs(f1["text_held"][0] - closed_empty(nat, tx, True)) < 1e-4
    assert abs(f1["text_ablated"][0] - closed_empty(nat, tx, False)) < 1e-4
    f2 = ge.interchange(tx, [native.Graph()])
    assert abs(f2["share"][0] - f2["remainder_share"]) < 1e-6
    # the remainder share from carry's dense masks, every subcomponent held
    kl = sum(float(carry.Carry(nat, tx.ids, tx.targets, prompts).measure([set()])[0]) for _, prompts in tx.f2)
    assert abs((1 - kl / f2["movement_bits"]) - f2["remainder_share"]) < 1e-5


def test_stated_every_connection_is_complete(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 3, stated=1.0)
    nodes = g.node_set()
    par, out = native.all_edges(nodes, tx.targets)
    explicit, complete = native.Graph(nodes, par, out), native.Graph(nodes, complete=True)
    f1 = ge.sufficiency(tx, [explicit, complete])
    for k, v in f1.items():
        assert abs(v[0] - v[1]) < 1e-4, (k, v)


def test_complete_ablated_matches_masked_forward(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 4, k=40)
    complete = native.Graph(g.node_set(), complete=True)
    f1 = ge.sufficiency(tx, [complete])
    masks = {n: torch.zeros(1, tx.T, nat.C[n]) for n in nat.names}
    for n, t, c in g.node_set():
        masks[n][0, t, c] = 1.0
    zero = {n: torch.zeros(1, tx.T) for n in nat.names}
    logq = torch.log_softmax(nat.vpd.masked(torch.tensor([tx.ids]), masks, zero)[:, tx.targets].float(), -1)
    assert abs(f1["text_ablated"][0] - float(carry.kl(tx.ref, logq)[0])) < 1e-4


def test_cuts_match_an_independent_forward(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 7, k=30, stated=1.0)
    pairs = goal_eval._pairs(g)
    got = ge.cuts(tx, pairs)
    for (w, r), k in zip(pairs, got):
        want = naive_cut(nat, tx.ids, tx.targets, w, r)
        assert abs(k - want) < 1e-4 + 1e-3 * abs(want), (w, r, k, want)
    assert max(got) > 1e-3  # the cuts move the prediction: the comparison is not of zeros


def test_cut_kinds_covered(tiny):
    """Every kind of connection is cut and checked: residual into each reader kind, value into attention output at an
    earlier position, MLP input into its output, residual writer into the prediction."""
    nat, ge, tx = tiny
    T, last = tx.T, tx.targets[0]
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


def test_unit_and_dense_paths_agree(tiny, monkeypatch):
    nat, ge, tx = tiny
    graphs = [random_graph(nat, tx, s, k=20) for s in (11, 12, 13)] + [native.Graph()]
    units = ge.interchange(tx, graphs)
    monkeypatch.setattr(goal_eval, "UNIT_LIMIT", 0)
    dense = ge.interchange(tx, graphs)
    for a, b in zip(units["share"], dense["share"]):
        assert abs(a - b) < 1e-5


def test_score_rows_and_table(tiny):
    nat, ge, tx = tiny
    graphs = [random_graph(nat, tx, 21, k=16)]
    base = ge.baselines(tx, [graphs[0].count()])
    assert set(base) == {"empty", f"vpd@{graphs[0].count()}", "cone"}
    ge.hooks["E1"] = lambda task, g, expl: float(g.count())
    rows = ge.score(tx, graphs + list(base.values()))
    ge.hooks["E1"] = None
    assert rows[0]["E1"] == graphs[0].count() and rows[0]["E2"] is None
    assert rows[1]["nodes"] == 0 and rows[1]["bits"] == 0.0
    every = ge.score(tx, [nat.everything(tx.T)])[0]
    assert every["f1_text_held"] < 1e-5 and abs(every["f2_share"] - 1) < 1e-4
    text = goal_eval.table({"set": "tiny", "rows": {"answer": [rows[0]], "empty": [rows[1]]}})
    assert "F2 share" in text and "tiny" in text


def test_circuit_json_round_trip_and_defects(tiny):
    nat, ge, tx = tiny
    g = random_graph(nat, tx, 31, k=12)
    back, defects = goal_eval.circuit(goal_eval.to_json(g), tx.targets, nat.C, tx.T)
    assert back.nodes == g.nodes and defects == {"invalid_nodes": 0, "invalid_edges": 0}
    assert sorted(goal_eval._pairs(back)) == sorted(goal_eval._pairs(g))
    bad = {"nodes": [["h.0.attn.o_proj", 1, 3], ["1.q", 2, 5], ["h.9.attn.q_proj", 0, 0], ["0.down", 0, 3], ["1.q", 2, 3]],
           "edges": [[0, 1], [3, 1], [0, "out"], [1, 0], [2, 1], [7, 1], [3, 4]]}
    g2, d = goal_eval.circuit(bad, tx.targets, nat.C, tx.T)
    # another position, not at a target, a query writes nothing, an unknown matrix, no such node; then a valid edge
    assert d == {"invalid_nodes": 1, "invalid_edges": 6}
    assert g2.parents == {("h.1.attn.q_proj", 3, 2): [("h.0.mlp.down_proj", 3, 0)]} and g2.out == []
    assert len(g2.nodes) == 4


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
    rows = ge.score(tx, graphs + list(base.values()))
    seconds = time.perf_counter() - t0
    names = ["everything"] + (["search answer"] if len(graphs) > 1 else []) + list(base)
    for nm, row in zip(names, rows):
        print(task, nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("f2_positions", "f3_cuts")})
    print(f"{task}: T {tx.T}, {len(tx.f2)} F2 positions, {len(rows)} circuits, {seconds:.1f} s (text preparation {tx.seconds:.1f} s)")
    assert abs(rows[0]["f1_text_held"]) < 1e-3  # float32 on the device, the whole model through the graph path
    assert abs(rows[0]["f2_share"] - 1) < 1e-3
    empty = rows[names.index("empty")]
    assert abs(empty["f1_text_held"] - closed_empty(nat, tx, True)) < 1e-3 * max(1, empty["f1_text_held"])
    assert abs(empty["f1_text_ablated"] - closed_empty(nat, tx, False)) < 1e-3 * max(1, empty["f1_text_ablated"])
    assert abs(empty["f2_share"] - empty["remainder_share"]) < 1e-6
    # two cuts against the independent forward
    g = graphs[-1] if len(graphs) > 1 else base[next(k for k in base if k.startswith("vpd@"))]
    pairs = goal_eval._pairs(g)[:2] if not g.complete else []
    for (w, r), k in zip(pairs, ge.cuts(tx, pairs)):
        want = naive_cut(nat, tx.ids, tx.targets, w, r)
        assert abs(k - want) < 1e-4 + 2e-3 * abs(want), (w, r, k, want)
