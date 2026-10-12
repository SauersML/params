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
# 1.4e-3 on an RTX A5000); bf16 keeps 8 (2^-9), the KL off by a few percent. Both are for a search's steps only
REL_TF32 = 1e-2
REL_BF16 = 0.15
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
