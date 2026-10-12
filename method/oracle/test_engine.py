"""Checks of engine.py on a held-out vpd4l text, on this machine's GPU (skipped without one or vpd4l's texts): the
engine's float32 masked run with every subcomponent addressable equals VPD's masked forward pass (goal_eval.runs, the
same steps) to the bit on MPS and within float32 rounding on CUDA; its gradients with respect to the masks match that
pass's autograd; gathered columns and the search precisions (TF32, bf16) stay within their stated bounds; rescore
reports float32. Its graph runs (graph_kl) equal native.Native.run's graph semantics within float32 rounding, with the
gradients in edge strengths and masks.

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_engine.py -s
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import engine  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

TEXT = native.TEXTS / "vpd4l" / "text9000.json"
VPD_SAVED = Path.home() / "mpd-data/graph_oracle/teacher/vpd_heldout20.pt"
DEVICE = native.device()
# KL in bits is a difference of nearly equal log probabilities; a float32 sum of n terms is off by about sqrt(n) 2^-24
# of its size (n = 3,072 at the MLP: 3.3e-6), magnified tenfold by the KL's cancellation on these texts
REL = 1e-4
# TF32 products keep 10 mantissa bits (2^-11 relative, 5e-4), the KL's cancellation magnifying it tenfold (measured:
# 1.4e-3 on an RTX A5000); bf16 keeps 8 (2^-9), the KL off by a few percent (measured on VPD's answer's subsets: 3.0e-2
# on MPS). Both are for a search's steps only
REL_TF32 = 1e-2
REL_BF16 = 0.05
pytestmark = pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")


def setup(B=6, seed=0):
    nat = native.Native(DEVICE)
    ids, targets = native.text(TEXT)
    g = torch.Generator().manual_seed(seed)
    T = len(ids)
    # masks between all on and a sparse random set: rows from near the model to far from it
    masks = {n: (torch.rand(B, T, nat.C[n], generator=g) < torch.linspace(1.0, 0.02, B)[:, None, None]).float().to(nat.dev)
             for n in nat.names}
    return nat, ids, targets, masks


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return float(((a - b).abs() / a.abs().clamp_min(1e-12)).max())


@pytest.mark.parametrize("live", [True, False])
def test_engine_equals_masked_forward(live):
    nat, ids, targets, masks = setup()
    ge = goal_eval.GoalEval(nat, chunk=len(next(iter(masks.values()))))
    ref = nat.reference([ids], targets)
    want = carry.kl(ref, ge.runs(ids, targets, [{n: m[b].bool() for n, m in masks.items()} for b in range(6)], live))
    with torch.no_grad():
        got = engine.Engine(nat, ids, targets, live=live).kl(masks)
    if DEVICE == "mps":
        assert torch.equal(got, want)
    else:
        assert rel(want, got) < REL


def test_gradients_match_autograd_of_masked_forward():
    nat, ids, targets, masks = setup(B=2)
    tg = nat.target
    a = {n: m.clone().requires_grad_(True) for n, m in masks.items()}
    for n in nat.names:
        tg.site(n).mask, tg.site(n).delta_mask = a[n], torch.ones(2, len(ids), device=nat.dev)
    try:
        h = tg.hidden(torch.tensor([ids], device=nat.dev).expand(2, -1))
    finally:
        nat.vpd.clear()
    ref = nat.reference([ids], targets)
    carry.kl(ref, torch.log_softmax((h[:, targets] @ tg.wte.T).float(), -1)).sum().backward()
    b = {n: m.clone().requires_grad_(True) for n, m in masks.items()}
    engine.Engine(nat, ids, targets, live=True).kl(b).sum().backward()
    for n in nat.names:
        x, y = a[n].grad.flatten(), b[n].grad.flatten()
        assert float((x - y).norm() / x.norm().clamp_min(1e-30)) < REL


def test_search_steps_equal_whole_run():
    """The search precisions' steps (fused norms and GELU, the first layer's embedding products once, the last layer at
    the targets) in float32 give the whole run's KL within float32 rounding, with its gradients."""
    nat, ids, targets, masks = setup(B=3)
    eng = engine.Engine(nat, ids, targets, live=True)
    a = {n: m.clone().requires_grad_(True) for n, m in masks.items()}
    b = {n: m.clone().requires_grad_(True) for n, m in masks.items()}
    want = eng.kl(a)
    got = carry.kl(eng.ref, torch.log_softmax(eng.hidden(b, search=True) @ nat.target.wte.T, -1))
    assert rel(want[1:], got[1:]) < REL
    assert abs(float((want[0] - got[0]).detach())) < 1e-6   # every subcomponent on: the model itself, KL ~0 bits
    want.sum().backward()
    got.sum().backward()
    for n in nat.names:
        assert float((a[n].grad - b[n].grad).norm() / a[n].grad.norm().clamp_min(1e-30)) < REL


@pytest.mark.parametrize("live", [True, False])
def test_outside_on(live):
    """Columns with every other subcomponent on: the whole engine's run with those masks free and the rest at 1."""
    nat, ids, targets, masks = setup(B=3)
    g = torch.Generator().manual_seed(4)
    cols = {n: torch.randperm(nat.C[n], generator=g)[:40].sort().values for n in nat.names}
    whole = engine.Engine(nat, ids, targets, live=live)
    on = engine.Engine(nat, ids, targets, live=live, columns=cols, outside="on")
    full = {n: torch.ones_like(m).index_copy(-1, cols[n].to(m.device), m.index_select(-1, cols[n].to(m.device))) for n, m in masks.items()}
    with torch.no_grad():
        want, got = whole.kl(full), on.kl({n: m.index_select(-1, cols[n].to(m.device)) for n, m in masks.items()})
    assert rel(want[1:], got[1:]) < REL
    assert abs(float(want[0] - got[0])) < 1e-6   # every subcomponent on: the model itself, KL ~0 bits


def answer(nat, ids) -> list:
    """VPD's answer on the text as (matrix, position, subcomponent) nodes: its causal-importance network's, or VPD_SAVED's."""
    if nat.has_importance:
        return [(n, t, c) for n, v in nat.importance(ids).items() for t, c in (v > 0).nonzero().tolist()]
    if VPD_SAVED.exists():
        return [(n, t, c) for n, (ts, cs) in torch.load(VPD_SAVED)["text9000"]["answer"].items() for t, c in zip(ts.tolist(), cs.tolist())]
    pytest.skip("needs VPD's answer: the causal-importance network or " + str(VPD_SAVED))


def test_gathered_columns_and_search_precisions():
    nat, ids, targets, masks = setup()
    active = {n: torch.zeros(nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}    # VPD's active columns at any position
    for n, _, c in answer(nat, ids):
        active[n][c] = True
    cols = {n: torch.nonzero(active[n]).flatten() for n in nat.names}
    sub = {n: m * active[n].float() for n, m in masks.items()}                  # masks inside those columns
    whole = engine.Engine(nat, ids, targets)
    gathered = engine.Engine(nat, ids, targets, columns=cols)
    compact = {n: m.index_select(-1, cols[n].to(m.device)) for n, m in sub.items()}
    with torch.no_grad():
        full = whole.kl(sub)
        assert rel(full, gathered.kl(compact)) < REL
        assert torch.equal(gathered.full(compact)["h.0.mlp.c_fc"], sub["h.0.mlp.c_fc"])
        if DEVICE.startswith("cuda"):
            assert rel(full, gathered.kl(compact, "tf32")) < REL_TF32
        assert rel(full, gathered.kl(compact, "bf16")) < REL_BF16
    assert torch.equal(gathered.rescore(compact), gathered.kl(compact))


def test_search_precisions_on_answer_subsets():
    """The search precisions' KL on VPD's answer with random subsets removed (the masks a search moves through, KL from
    VPD's own ~0.05 bits to ~80) within their bounds: an approximation of the model in the search's steps (such as
    leaving out small remainders: 12% at 6 bits) shows here."""
    nat, ids, targets, _ = setup(B=1)
    T, B = len(ids), 16
    nodes = answer(nat, ids)
    cols = {n: torch.tensor(sorted({c for m, _, c in nodes if m == n}), dtype=torch.long) for n in nat.names}
    eng = engine.Engine(nat, ids, targets, live=True, columns=cols)
    g = torch.Generator().manual_seed(3)
    masks = {n: torch.zeros(B, T, eng.width(n)) for n in nat.names}
    at = {n: {c: i for i, c in enumerate(cols[n].tolist())} for n in nat.names}
    for n, t, c in nodes:
        masks[n][:, t, at[n][c]] = 1.0
    masks = {n: (m * (torch.rand(m.shape, generator=g) < torch.linspace(1.0, 0.3, B)[:, None, None])).to(nat.dev) for n, m in masks.items()}
    with torch.no_grad():
        want = eng.kl(masks)
        if DEVICE.startswith("cuda"):
            assert rel(want, eng.kl(masks, "tf32")) < REL_TF32
        assert rel(want, eng.kl(masks, "bf16")) < REL_BF16


def graph_case(B=3):
    """A graph of VPD's answer at the text's last positions (six nodes per matrix, the latest) with every connection
    the model has among them, and B rows of random edge strengths and masks."""
    nat, ids, targets, _ = setup(B=1)
    T = len(ids)
    nodes, per = [], {}
    for n, t, c in sorted(answer(nat, ids), key=lambda nd: -nd[1]):
        if per.get(n, 0) < 6:
            nodes.append((n, t, c))
            per[n] = per.get(n, 0) + 1
    graph = native.Graph(nodes, *native.all_edges(nodes, targets))
    assert graph.out
    g = torch.Generator().manual_seed(1)
    ew = torch.rand(B, sum(len(w) for w in graph.parents.values()), generator=g).to(nat.dev)
    ow = torch.rand(B, len(graph.out), generator=g).to(nat.dev)
    u = {n: torch.rand(B, T, nat.C[n], generator=g).to(nat.dev) for n in nat.names}
    return nat, ids, targets, graph, ew, ow, u


def native_graph_kl(nat, ids, targets, graph, ew, ow, u):
    """native.run's KL of the graph's B rows, remainders live."""
    B = len(ew)
    plan = nat._plan([graph] * B, len(ids))
    ewn, own = nat.edge_weights(plan, torch.cat([ew, ow], 1).flatten())   # a graph's connections, then its outputs
    ones = {n: torch.ones(1, 1, device=nat.dev) for n in nat.names}
    return carry.kl(nat.reference([ids], targets), nat.run(ids, targets, plan, u, ones, ewn, own))


def test_graph_runs_equal_native_runs():
    nat, ids, targets, graph, ew, ow, u = graph_case()
    a = [ew.clone().requires_grad_(True), ow.clone().requires_grad_(True), {n: m.clone().requires_grad_(True) for n, m in u.items()}]
    b = [ew.clone().requires_grad_(True), ow.clone().requires_grad_(True), {n: m.clone().requires_grad_(True) for n, m in u.items()}]
    want = native_graph_kl(nat, ids, targets, graph, *a)
    eng = engine.Engine(nat, ids, targets, live=True)
    got = eng.graph_kl(eng.plan(graph), b[2], b[0], b[1])
    assert rel(want, got) < REL
    want.sum().backward()
    got.sum().backward()
    pairs = [(a[0], b[0]), (a[1], b[1])] + [(a[2][n], b[2][n]) for n in nat.names]
    for x, y in pairs:
        assert float((x.grad - y.grad).norm() / x.grad.norm().clamp_min(1e-30)) < REL


def test_gathered_graph_runs():
    """EdgePruner's convention, every subcomponent outside the graph off, on the graph's own columns alone."""
    nat, ids, targets, graph, ew, ow, _ = graph_case()
    cols = {n: torch.tensor(sorted({c for m, _, c in graph.nodes if m == n}), dtype=torch.long) for n in nat.names}
    zero = {n: torch.zeros(1, 1, nat.C[n], device=nat.dev) for n in nat.names}
    eng = engine.Engine(nat, ids, targets, live=True, columns=cols)
    plan = eng.plan(graph)
    with torch.no_grad():
        want = native_graph_kl(nat, ids, targets, graph, ew, ow, zero)
        assert rel(want, eng.graph_kl(plan, None, ew, ow)) < REL
        assert rel(want, eng.graph_kl(plan, None, ew, ow, "bf16")) < REL_BF16


def test_graph_lists():
    """One graph per row (answers x interventions), with gradients in the strengths, as native.run's plan of the list."""
    nat, ids, targets, graph, _, _, u = graph_case()
    half = native.Graph(graph.nodes, {r: ws[::2] for r, ws in graph.parents.items()}, graph.out[1:])
    graphs = [graph, half, native.Graph(graph.nodes, graph.parents, [])]
    g = torch.Generator().manual_seed(2)
    ew = [torch.rand(sum(len(w) for w in x.parents.values()), generator=g).to(nat.dev) for x in graphs]
    ow = [torch.rand(len(x.out), generator=g).to(nat.dev) for x in graphs]
    plan = nat._plan(graphs, len(ids))
    a = torch.cat([t for e, o in zip(ew, ow) for t in (e, o)]).requires_grad_(True)   # each graph's connections, then its outputs
    ones = {n: torch.ones(1, 1, device=nat.dev) for n in nat.names}
    want = carry.kl(nat.reference([ids], targets), nat.run(ids, targets, plan, u, ones, *nat.edge_weights(plan, a)))
    eng = engine.Engine(nat, ids, targets, live=True)
    be, bo = torch.cat(ew).requires_grad_(True), torch.cat(ow).requires_grad_(True)
    got = eng.graph_kl(eng.plan(graphs), u, be, bo)
    assert rel(want, got) < REL
    want.sum().backward()
    got.sum().backward()
    split = [t for e, o in zip(ew, ow) for t in (len(e), len(o))]
    parts = a.grad.split(split)
    for x, y in ((torch.cat(parts[0::2]), be.grad), (torch.cat(parts[1::2]), bo.grad)):
        assert float((x - y).norm() / x.norm().clamp_min(1e-30)) < REL


def test_repeated_graphs_plan_as_their_list():
    """A list naming graphs more than once (Engine._stack of each one's plan) is the list planned row by row (each row
    its own graph object): every table equal (an unstated connection's strength index aside: its strength is 0), in
    both semantics and with one cuts object repeated, and the same KL and gradients in u and the strengths (within
    float32 rounding: the same plan run twice differs in the last bits on MPS, its scatter-adds unordered)."""
    import copy

    nat, ids, targets, graph, _, _, u = graph_case(B=5)
    half = native.Graph(graph.nodes, {r: ws[::2] for r, ws in graph.parents.items()}, graph.out[::2])
    eng = engine.Engine(nat, ids, targets, live=True)
    cut = {r: ws[:1] for r, ws in list(graph.parents.items())[:2]}
    for semantics, rows, cuts in (("native", [graph, half, graph, graph, half], None),
                                  ("c", [half, half, graph, half, graph], None), ("c", [half] * 5, [cut] * 5)):
        a = eng.plan(rows, semantics=semantics, cuts=cuts)
        b = eng.plan([copy.copy(g) for g in rows], semantics=semantics, cuts=None if cuts is None else [dict(c) for c in cuts])
        assert (a["B"], a["E"], a["O"]) == (b["B"], b["E"], b["O"])
        for k in ("qk", "nodes", "readers", "writers", "edges", "out", "positions"):
            assert list(a[k]) == list(b[k]), (semantics, k)
            for n in b[k]:
                x, y = list(a[k][n]), list(b[k][n])
                if k in ("edges", "out"):  # the strength index where the strength is fixed at 0
                    j = 2 if k == "edges" else 3
                    x[j], y[j] = x[j] * (x[-1] > 0), y[j] * (y[-1] > 0)
                assert all(torch.equal(p, q) for p, q in zip(x, y)), (semantics, k, n)
        g = torch.Generator().manual_seed(4)
        ew, ow = torch.rand(a["E"], generator=g).to(nat.dev), torch.rand(a["O"], generator=g).to(nat.dev)
        got = []
        for plan in (a, b):
            x = [ew.clone().requires_grad_(True), ow.clone().requires_grad_(True), {n: m.clone().requires_grad_(True) for n, m in u.items()}]
            kl = eng.graph_kl(plan, x[2], x[0], x[1])
            kl.sum().backward()
            got.append([kl.detach(), x[0].grad, x[1].grad] + [x[2][n].grad for n in nat.names])
        assert rel(got[1][0], got[0][0]) < REL
        for x, y in zip(got[1][1:], got[0][1:]):
            assert float((x - y).norm() / x.norm().clamp_min(1e-30)) < REL, semantics


def test_c_semantics_equal_graph_ref():
    """C's semantics (plan(semantics="c")) equal goal_eval.GoalEval.graph_ref, the forward pass written out, on a graph
    stating half of its connections, with the gradients in u; a stated connection at strength 0 is the graph without it."""
    nat, ids, targets, graph, _, _, u = graph_case()
    half = native.Graph(graph.nodes, {r: ws[::2] for r, ws in graph.parents.items()}, graph.out[::2])
    ge = goal_eval.GoalEval(nat)
    a = {n: m.clone().requires_grad_(True) for n, m in u.items()}
    b = {n: m.clone().requires_grad_(True) for n, m in u.items()}
    want = carry.kl(nat.reference([ids], targets), ge.graph_ref(ids, targets, half, u=a))
    eng = engine.Engine(nat, ids, targets, live=True)
    got = eng.graph_kl(eng.plan(half, semantics="c"), b)
    assert rel(want, got) < REL
    want.sum().backward()
    got.sum().backward()
    for n in nat.names:
        assert float((a[n].grad - b[n].grad).norm() / a[n].grad.norm().clamp_min(1e-30)) < REL
    E = sum(len(ws) for ws in half.parents.values())
    ew = torch.ones(1, E, device=nat.dev)
    ew[0, 0] = 0.0
    r0 = next(r for r, ws in half.parents.items() if ws)
    less = native.Graph(half.nodes, {r: (ws[1:] if r == r0 else ws) for r, ws in half.parents.items()}, half.out)
    with torch.no_grad():
        u0 = {n: m[:1] for n, m in u.items()}
        assert rel(eng.graph_kl(eng.plan(less, semantics="c"), u0), eng.graph_kl(eng.plan(half, semantics="c"), u0, ew=ew)) < REL


def test_c_semantics_cuts_in_the_model_and_patches():
    """C's semantics on a complete graph: the whole model with one connection cut (Engine(outside="on"), u = 1) is
    goal_eval's cut of that connection (path patching); and a circuit node patched to 0 with u = 0 is the graph
    without that node."""
    nat, ids, targets, graph, _, _, _ = graph_case()
    T = len(ids)
    ge = goal_eval.GoalEval(nat)
    tx = ge.text(ids, targets)
    par, out = native.all_edges(sorted(graph.nodes), targets)
    pairs = [(ws[0], r) for r, ws in sorted(par.items())[:3]] + [(out[0], None)]
    want = ge.cuts(tx, pairs)
    cols = {n: torch.tensor(sorted({nd[2] for e in pairs for nd in e if nd is not None and nd[0] == n}), dtype=torch.long)
            for n in nat.names}
    eng = engine.Engine(nat, ids, targets, live=True, columns=cols, outside="on")
    model = native.Graph(masks={n: torch.ones(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names})
    ones = {n: torch.ones(1, 1, eng.width(n), device=nat.dev) for n in nat.names}
    with torch.no_grad():
        got = eng.graph_kl(eng.plan([model] * len(pairs), semantics="c", cuts=[{r: [w]} for w, r in pairs]), ones).tolist()
    for (w, r), a, b in zip(pairs, want, got):
        assert abs(a - b) < 1e-5 + REL * abs(a), (w, r, a, b)
    whole = engine.Engine(nat, ids, targets, live=True)
    x = sorted(graph.nodes)[len(graph.nodes) // 2]
    zero = {n: torch.zeros(1, T, nat.C[n], device=nat.dev) for n in nat.names}
    patch = {x[0]: (torch.zeros(1, T, nat.C[x[0]], dtype=torch.bool, device=nat.dev), torch.zeros(1, T, nat.C[x[0]], device=nat.dev))}
    patch[x[0]][0][0, x[1], x[2]] = True
    less = goal_eval._drop_nodes(graph, {x})
    with torch.no_grad():
        a = whole.graph_kl(whole.plan(graph, semantics="c"), zero, patch=patch)
        b = whole.graph_kl(whole.plan(less, semantics="c"), zero)
    assert rel(a, b) < REL


def test_graph_runs_in_slices(monkeypatch):
    """Without gradients a connection group's products run a slice at a time (engine.ROWS): the same numbers."""
    nat, ids, targets, graph, ew, ow, u = graph_case()
    eng = engine.Engine(nat, ids, targets, live=True)
    for semantics in ("native", "c"):
        plan = eng.plan(graph, semantics=semantics)
        with torch.no_grad():
            whole = eng.graph_kl(plan, u, ew, ow)
            monkeypatch.setattr(engine, "ROWS", 1 << 10)
            assert rel(whole, eng.graph_kl(plan, u, ew, ow)) < REL
            monkeypatch.setattr(engine, "ROWS", 1 << 24)


def test_reads_equal_graph_ref():
    """The circuit nodes' activations a run reads out (reads=) equal graph_ref's record: a C-semantics graph run with
    random u, and a masked run (the whole model with some subcomponents off, as a complete graph)."""
    nat, ids, targets, graph, _, _, u = graph_case(B=2)
    half = native.Graph(graph.nodes, {r: ws[::2] for r, ws in graph.parents.items()}, graph.out[::2])
    ge = goal_eval.GoalEval(nat)
    nodes = sorted(half.nodes)
    rec = {}
    ge.graph_ref(ids, targets, half, u=u, record=rec, reads=nodes)
    eng = engine.Engine(nat, ids, targets, live=True)
    spec = {}
    for b in range(2):
        for n, t, c in nodes:
            spec.setdefault(n, ([], [], []))
            for lst, v in zip(spec[n], (b, t, c)):
                lst.append(v)
    spec = {n: tuple(torch.tensor(v, device=nat.dev) for v in x) for n, x in spec.items()}
    _, got = eng.graph_log_probs(eng.plan(half, semantics="c"), u, reads=spec)
    k = {n: 0 for n in spec}
    for b in range(2):
        for n, t, c in nodes:
            want = float(rec[(n, t, c)][b])
            assert abs(float(got[n][k[n]]) - want) < 1e-5 + REL * abs(want)
            k[n] += 1
    on = {n: torch.ones(len(ids), nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, t, c in nodes[::2]:
        on[n][t, c] = False
    rec = {}
    ge.graph_ref(ids, targets, native.Graph(masks=on), record=rec, reads=nodes)
    one = {n: tuple(v[:len(v) // 2] for v in x) for n, x in spec.items()}  # row 0's reads
    _, got = eng.log_probs({n: m.float()[None] for n, m in on.items()}, reads=one)
    for n, t, c in nodes:
        i = [(int(a), int(b_)) for a, b_ in zip(one[n][1].tolist(), one[n][2].tolist())].index((t, c))
        want = float(rec[(n, t, c)][0])
        assert abs(float(got[n][i]) - want) < 1e-5 + REL * abs(want)


def test_graph_search_steps_equal_whole_run():
    """A graph run's search steps (fused norms and GELU, the last layer's queries, attention output and MLP at the
    targets alone) in float32 give the whole run's KL and gradients within float32 rounding: both semantics, one graph
    on every row and a list, with patches."""
    nat, ids, targets, graph, ew, ow, u = graph_case()
    half = native.Graph(graph.nodes, {r: ws[::2] for r, ws in graph.parents.items()}, graph.out[::2])
    eng = engine.Engine(nat, ids, targets, live=True)
    T, B = len(ids), 3
    x = sorted(graph.nodes)[0]
    patch = {x[0]: (torch.zeros(B, T, nat.C[x[0]], dtype=torch.bool, device=nat.dev), torch.full((B, T, nat.C[x[0]]), 0.3, device=nat.dev))}
    patch[x[0]][0][:, x[1], x[2]] = True
    for semantics, plan, e, o in (("native", eng.plan(graph), ew, ow), ("c", eng.plan(half, semantics="c"), None, None),
                                  ("c", eng.plan([half, graph, half], semantics="c"), None, None)):
        p = eng._rows(plan, B)
        got = []
        for fused in (False, True):
            a = {n: m.clone().requires_grad_(True) for n, m in u.items()}
            h = eng._graph_hidden(p, a, None if e is None else e.reshape(-1), None if o is None else o.reshape(-1), B, fused, patch)
            k = carry.kl(eng.ref, torch.log_softmax(h @ nat.target.wte.T, -1))
            k.sum().backward()
            got.append((k.detach(), a["h.1.mlp.c_fc"].grad, a["h.3.attn.k_proj"].grad))
        assert rel(got[0][0], got[1][0]) < REL, semantics
        for x0, x1 in zip(got[0][1:], got[1][1:]):
            assert float((x0 - x1).norm() / x0.norm().clamp_min(1e-30)) < REL, semantics
