"""Checks of the neuron edge stage (neuron_edges.py) on a tiny random Qwen3: with no cut and every neuron live the run is
the model's; the circuit run is the model with every other neuron zeroed but the first position's and massive values; a cut edge's measured effect follows its
linear score at small writes; pruning keeps the contrast within its tolerance and the graph lists poles and edges.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_neuron_edges.py -q
"""
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neuron_edges  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 48, (10,), generator=g).tolist()
    neurons = sorted({(int(torch.randint(3, (1,), generator=g)), int(torch.randint(6, 10, (1,), generator=g)),
                       int(torch.randint(40, (1,), generator=g))) for _ in range(24)})
    return model, ids, neurons


def contrast_of(model, ids, keep=None):
    hooks = []
    if keep is not None:  # every neuron outside keep zeroed
        for L, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: (a[0] * keep[L][None],)))
    try:
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, 9].float(), -1)
    finally:
        for h in hooks:
            h.remove()
    return float(neuron_edges.log2_contrast(lp, [3], [5, 7]))


def test_runs_are_the_models(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="model")
    assert ed.c_full == pytest.approx(contrast_of(model, ids), abs=1e-4)
    ec = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="circuit")
    live = ec.keep.clone()
    live[:, 0] = True  # (the first position stays live, as neuron_builder's circuit run)
    assert ec.c_full == pytest.approx(contrast_of(model, ids, live), abs=1e-4)
    big = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="circuit", massive=torch.zeros(3))
    assert big.c_full == pytest.approx(contrast_of(model, ids), abs=1e-4)  # (every value massive: every one live)
    E = len(ec.resid_all)
    assert ec.contrasts(torch.zeros(1, E, dtype=torch.bool), ec.resid_all)[0] == pytest.approx(ec.c_full, abs=1e-4)


def test_cuts_follow_linear_scores(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="model")
    rs, routes, rt = ed.linear_scores(64)
    edges = ed.resid_all + routes
    cuts = torch.eye(len(edges), dtype=torch.bool)
    meas = torch.tensor([ed.c_full - c for c in ed.contrasts(cuts, edges)])
    lin = torch.cat([rs, rt])
    assert float(torch.corrcoef(torch.stack([meas, lin]))[0, 1]) > 0.8


def test_prune_and_graph(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="circuit")
    got = neuron_edges.prune(ed, log=lambda m: None)
    rec = got["record"]
    assert abs(rec["contrast_kept"] - rec["contrast_all_edges"]) <= rec["tolerance"] + 1e-6
    g = neuron_edges.graph(ed, got)
    assert len(g["nodes"]) == len(ed.N) and len(g["edges"]) == rec["kept"]
    assert all(n["pole"] in (1, -1) and n["regime"] in ("g+u+", "g+u-", "g-u+", "g-u-") for n in g["nodes"])


def test_exported_route_scope_is_not_full_attention_routing(tiny):
    """Saved evidence must distinguish conditional OV cuts from Q/K or whole-writer interventions."""
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="model")
    route = ed.linear_scores(1)[1][0]
    exported = neuron_edges.graph(ed, {"edges": [route], "strengths": [0.0], "record": {}})
    saved = json.loads(json.dumps(exported))
    assert saved["edges"][0]["kind"] == "route"
    semantics = saved["route_cut_semantics"]
    assert semantics["path"] == "value_ov"
    assert semantics["attention_pattern"] == "current_run_fixed_in_local_value_delta"
    assert semantics["head_input_norm"] == "recomputed_after_writer_write_subtraction"
    assert semantics["reader_mlp"] == "recomputed_norm_gate_up"
    assert semantics["combined_routes"] == "sum_of_individual_value_deltas"
    assert not semantics["query_key_effects_included"]
    assert not semantics["full_writer_ablation_equivalent"]
    assert not semantics["full_causal_completeness_certified"]


def test_route_scores_match_brings(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="model")
    rs, routes, rt = ed.linear_scores(ed.n_routes)
    assert len(routes) == ed.n_routes and bool((rt.abs()[:-1] >= rt.abs()[1:]).all())  # (every route, best first)
    probe = [torch.zeros(ed.T, ed.keep.shape[-1], requires_grad=True) for _ in range(ed.L)]
    ed._run(torch.zeros(1, 0, dtype=torch.bool), [], probe=probe)["contrast"].sum().backward()
    J = ed.jacobians()
    for i in range(0, len(routes), max(1, len(routes) // 25)):  # the vectorized scores, edge by edge
        e = routes[i]
        r = e[2]
        want = float(probe[r[0]].grad[r[1], r[2]]) * float(J[r] @ ed.brings(e))
        assert float(rt[i]) == pytest.approx(want, rel=1e-4, abs=1e-7)


def test_rest_cut_sums(tiny):
    """Every other candidate cut by the per-position sums: with every candidate listed and kept the run is the model's;
    with none listed it is every candidate listed and cut."""
    model, ids, neurons = tiny
    for bg in ("model", "circuit"):
        ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background=bg)
        every = ed.resid_all + ed.linear_scores(ed.n_routes)[1]
        E = len(every)
        kept = ed.contrasts(torch.zeros(1, E, dtype=torch.bool), every, rest=True)[0]
        assert kept == pytest.approx(ed.c_full, abs=1e-4)
        none = ed.contrasts(torch.zeros(1, 0, dtype=torch.bool), [], rest=True)[0]
        assert none == pytest.approx(ed.contrasts(torch.ones(1, E, dtype=torch.bool), every)[0], abs=1e-4)
        half = torch.arange(E)[None] % 2 == 0  # (half listed and cut, the other half as run against a cut rest)
        got = ed.contrasts(half, every, rest=True)[0]
        assert got == pytest.approx(ed.contrasts(half, every)[0], abs=1e-4)


def test_bfloat16_model(tiny):
    """A bfloat16 model runs the same stage (each module in bfloat16, float32 between): its contrasts follow the float32
    model's to bfloat16's precision."""
    import copy
    model, ids, neurons = tiny
    m16 = copy.deepcopy(model).to(torch.bfloat16)
    e32 = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="model")
    e16 = neuron_edges.Edges(m16, ids, 9, [3], [5, 7], neurons, background="model")
    cuts = torch.eye(len(e32.resid_all), dtype=torch.bool)[:8]
    a, b = e32.contrasts(cuts, e32.resid_all), e16.contrasts(cuts, e16.resid_all)
    assert e16.c_full == pytest.approx(e32.c_full, abs=0.05)
    assert max(abs(x - y) for x, y in zip(a, b)) < 0.1


def test_dev_ranking_uses_full_model_gradient_and_matches_finite_difference(tiny, monkeypatch):
    """Later nonlinear MLP paths contribute to earlier neuron derivatives, with frozen weights."""
    import copy
    import importlib.util
    path = HERE.parents[1] / "experiments" / "oracle" / "neuron_edges_dev.py"
    spec = importlib.util.spec_from_file_location("neuron_edges_dev_gradient_test", path)
    dev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev)
    model, ids, _ = tiny
    model = copy.deepcopy(model).double().requires_grad_(False)
    # A larger nonlinear downstream path makes detached-MLP derivatives observably wrong.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.down_proj.weight.mul_(8)
    acts = {}
    hooks = []
    for L, layer in enumerate(model.model.layers):
        def capture(m, a, L=L):
            x = a[0]
            if not x.requires_grad:
                x = x.detach().requires_grad_()
            x.retain_grad()
            acts[L] = x
            return (x,)
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    try:
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, 9].float(), -1)
        objective = neuron_edges.log2_contrast(lp, [3], [5, 7])
        gradients = torch.autograd.grad(objective, tuple(acts.values()))
        expected = torch.stack([(acts[L][0] * gradients[L][0]).abs().float().cpu()
                                for L in range(len(acts))])
    finally:
        for hook in hooks:
            hook.remove()
    # Verify an early-layer derivative by literal pre-down activation perturbation.
    j = int(gradients[0][0, 9].abs().argmax())
    def perturbed(epsilon):
        def edit(m, a):
            x = a[0].clone()
            x[0, 9, j] += epsilon
            return (x,)
        hook = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(edit)
        try:
            return contrast_of(model, ids)
        finally:
            hook.remove()
    eps = .001
    numeric = (perturbed(eps) - perturbed(-eps)) / (2 * eps)
    assert numeric == pytest.approx(float(gradients[0][0, 9, j]), rel=.015, abs=2e-4)
    captured = []
    real_topk = torch.topk
    def topk(scores, k, *args, **kwargs):
        captured.append(scores.detach().clone())
        return real_topk(scores, k, *args, **kwargs)
    monkeypatch.setattr(torch, "topk", topk)
    got = dev.top_neurons(model, ids, 9, [3], [5, 7], 12)
    assert torch.allclose(captured[0], expected.flatten(), rtol=1e-5, atol=1e-8)
    order = real_topk(expected.flatten(), 12).indices
    assert got == [(int(i // 400), int((i // 40) % 10), int(i % 40)) for i in order]
    assert all(p.grad is None for p in model.parameters())
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)
