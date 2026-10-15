"""Checks of the fast connection stage (connect.py) on a tiny random model (test_goal_eval's): the integrated-gradient
scores add up to the KL every line saves, and the stated circuit keeps every part on a stated path, its point KL and
s_track_pp within the criteria, and no more lines than the wired circuit.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_connect.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import connect  # noqa: E402
import goal_eval  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402
from test_lines import nodes_of  # noqa: E402

TINY = {"scope": None, "interchange": None, "patching_near": None, "patching_far": None}  # (the tiny model has no far donors for those families' draws)


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    tx = ge.text(ids, [11], seed=5)
    return nat, ge, tx


def test_scores_add_up(tiny):
    nat, ge, tx = tiny
    c = connect.Connector(ge, tx, nodes_of(nat, tx, 3, 16), B=4)
    sc = c.scores()
    full, none = c.pr.kl_sets(torch.stack([c.row(c.conn), c.row(c.conn[:0])]))
    assert float(sc.sum()) == pytest.approx(none - full, rel=0.1, abs=1e-3)  # (8 midpoints: a Riemann sum)
    assert float(sc[c.tok].abs().sum()) == 0.0


def test_stated_circuit(tiny):
    nat, ge, tx = tiny
    c = connect.Connector(ge, tx, nodes_of(nat, tx, 5, 16), B=4)
    got = c.run(log=lambda m: None, floors=TINY)
    rec, g = got["record"], got["graph"]
    assert rec["lines"] <= rec["lines_start"] and rec["parts"] == len(g.node_set())
    assert rec["point"] <= rec["criteria"]["point"] + 1e-6
    assert all(v is None or v >= rec["criteria"][n] - 1e-6 for n, v in rec["families"].items())
    assert rec["cross_token"] <= rec["wired"]["cross_token"] and rec["tokens"] <= rec["wired"]["tokens"]
    assert rec["positions"] == sorted({nd[1] for nd in g.node_set()})
    h = goal_eval.graph_of_lines(c.pr.g, c.pr.cols, c.pr.row(got["on"]))
    del h.stranded, h.stranded_bits  # (graph_of_lines marks it stranded already)
    again = goal_eval.strand(h, tx.targets, nat.C)
    assert set(again.node_set()) == set(g.node_set())  # every part kept lies on a stated path to the prediction


def test_connect_interface(tiny):
    nat, ge, tx = tiny
    g = connect.connect(ge, tx, connect.native.Graph(nodes_of(nat, tx, 7, 14), complete=True), floors=TINY)
    assert not g.complete and g.connect["parts"] == len(g.node_set())
def test_scores_with_output_but_no_interunit_edges(monkeypatch):
    """Actual autograd: an unused empty ew has no gradient; ow still ranks."""
    class OutputOnlyEngine:
        def graph_log_probs(self, plan, *, ew, ow, reuse):
            assert ew.shape == (connect.ALPHAS, 0)
            logits=torch.stack([ow[:,0],torch.zeros_like(ow[:,0])],dim=-1)
            return logits.log_softmax(-1)[:,None,:]
    c=object.__new__(connect.Connector)
    c.ge=SimpleNamespace(nat=SimpleNamespace(dev='cpu'))
    c.tx=SimpleNamespace(ref=torch.tensor([[[.8,.2]]]).log())
    c.pr=SimpleNamespace(E=OutputOnlyEngine(),plan={},lo=0,e_ends=[],o_ends=['writer'],
                         lines=[('o','writer')],e_line=torch.empty(0,dtype=torch.long),
                         o_line=torch.tensor([0]))
    monkeypatch.setattr(connect.native,'free',lambda *_:None)
    scores=c.scores()
    assert scores.shape==(1,) and torch.isfinite(scores).all() and scores[0]>0


def test_nonempty_disconnected_gradient_still_fails(monkeypatch):
    class MissingEdgeGradient:
        def graph_log_probs(self, plan, *, ew, ow, reuse):
            logits=torch.stack([ow[:,0],torch.zeros_like(ow[:,0])],dim=-1)
            return logits.log_softmax(-1)[:,None,:]
    c=object.__new__(connect.Connector)
    c.ge=SimpleNamespace(nat=SimpleNamespace(dev='cpu'))
    c.tx=SimpleNamespace(ref=torch.tensor([[[.8,.2]]]).log())
    c.pr=SimpleNamespace(E=MissingEdgeGradient(),plan={},lo=0,e_ends=['edge'],o_ends=['writer'],
                         lines=[('e','edge'),('o','writer')],e_line=torch.tensor([0]),o_line=torch.tensor([1]))
    monkeypatch.setattr(connect.native,'free',lambda *_:None)
    with pytest.raises(AttributeError,match='mean'):
        c.scores()


def mean_ablated(nat, ids, targets, means, units):
    """The mean-ablated run of a set of heads and neurons written out (native_units.MeanUnits' semantics): every
    other unit at its mean, the embedding, norms and attention live; log q at the targets [len(targets), V]."""
    m = nat.target
    T, H, hd = len(ids), m.n_head, m.hd
    on = set(units)
    s = lambda L, k: m.site(f"h.{L}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")  # noqa: E731
    x = m.embed(torch.tensor([ids]))[0]
    for L in range(m.n_layer):
        h = m.norm(x, 2 * L)
        q, k, v = ((h @ s(L, n).W.T).view(T, H, hd).transpose(0, 1) for n in ("q_proj", "k_proj", "v_proj"))
        q, k = m._rope(q, T), m._rope(k, T)
        a = ((q @ k.transpose(-1, -2)) / hd ** 0.5).masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        z = (a.softmax(-1) @ v).transpose(0, 1).clone()  # [T, H, hd]
        for t in range(T):
            for hh in range(H):
                if ("head", L, t, hh) not in on:
                    z[t, hh] = means["heads"][L, hh]
        x = x + z.reshape(T, -1) @ s(L, "o_proj").W.T
        act = m.act(m.norm(x, 2 * L + 1) @ s(L, "c_fc").W.T)
        keep = torch.zeros_like(act, dtype=torch.bool)
        for kind, LL, t, j in on:
            if kind == "neuron" and LL == L:
                keep[t, j] = True
        act = torch.where(keep, act, means["neurons"][L][None])
        x = x + act @ s(L, "down_proj").W.T
    return m.unembed(m.norm(x, 2 * m.n_layer))[targets].log_softmax(-1)


def native_units_of(nat, T, seed, k):
    g = torch.Generator().manual_seed(seed)
    m = nat.target
    dm = m.site("h.0.mlp.c_fc").W.shape[0]
    out = {("head", m.n_layer - 1, T - 1, 0), ("neuron", m.n_layer - 1, T - 1, 1)}
    while len(out) < k:
        L, t = int(torch.randint(m.n_layer, (1,), generator=g)), int(torch.randint(T - 4, T, (1,), generator=g))
        out.add(("head", L, t, int(torch.randint(m.n_head, (1,), generator=g))) if torch.rand(1, generator=g) < 0.4
                else ("neuron", L, t, int(torch.randint(dm, (1,), generator=g))))
    return sorted(out)


def test_native_runs(tiny):
    nat, ge, tx = tiny
    m = nat.target
    g = torch.Generator().manual_seed(3)
    means = {"heads": torch.randn(m.n_layer, m.n_head, m.hd, generator=g),
             "neurons": torch.randn(m.n_layer, m.site("h.0.mlp.c_fc").W.shape[0], generator=g)}
    units = native_units_of(nat, len(tx.ids), 4, 12)
    nl = connect.NativeLines(nat, tx.ids, tx.targets, tx.ref, means, units, B=4)
    ons = torch.stack([torch.ones(len(nl.lines), dtype=torch.bool), torch.zeros(len(nl.lines), dtype=torch.bool)])
    ew, ow, km = nl.strengths(ons)
    lq = nl.log_probs(ew, ow, km)
    full = mean_ablated(nat, tx.ids, tx.targets, means, nl.graph(ons[0]).units)
    empty = mean_ablated(nat, tx.ids, tx.targets, means, [])
    assert torch.allclose(lq[0], full, atol=1e-4) and torch.allclose(lq[1], empty, atol=1e-4)
    st = connect.NativeConnector(nat, tx.ids, tx.targets, tx.ref, means, units, B=4)
    sc = st.scores()
    kf, kn = nl.kl_sets(ons)
    assert float(sc.sum()) == pytest.approx(kn - kf, rel=0.15, abs=1e-3)
    got = st.run(log=lambda msg: None)
    assert got["record"]["point"] <= got["record"]["criteria"]["point"] + 1e-6
    assert set(nl.graph(got["on"]).units) == got["graph"].node_set()
    assert got["record"]["tokens"] == len({u[2] for u in got["graph"].units})
