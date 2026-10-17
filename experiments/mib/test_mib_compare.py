"""mib_compare.py's measured tests against MIB's own evaluate_graph (EAP-IG) on a tiny random Llama-shaped
HookedTransformer (grouped-query attention, rotary, gated MLP): every writer, reader and refined edge's change equals the
change evaluate_graph gives (also on Gemma-2- and GPT-2-shaped models) with the same edges removed (its pruning step off), whatever the row grouping, the
refined edges rank first in the order of their own changes, and every prefix of the paths order survives MIB's pruning. Needs MIB's
environment (transformer_lens, EAP-IG's eap).

  ~/mpd-data/ext/mibenv/bin/python -m pytest experiments/mib/test_mib_compare.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mib_compare as mc  # noqa: E402


def fake_tokenize(model, inputs, max_length=None):
    t = torch.tensor([[int(w) for w in s.split()] for s in inputs])
    return t, torch.ones_like(t), torch.full((len(t),), t.shape[1]), t.shape[1]


def metric(logits, clean_logits, input_lengths, label):
    r = torch.arange(len(logits))
    last = logits[r, input_lengths - 1]
    return last[r, label[:, 0]] - last[r, label[:, 1]]


SHAPES = {  # (MIB's models: Llama-3.1 and Qwen-2.5 grouped-query, rotary, gated; Gemma-2 with norms after attention and the
    # MLP; GPT-2 with LayerNorm, learned positions and biases)
    "llama": dict(n_key_value_heads=2, act_fn="silu", normalization_type="RMS", positional_embedding_type="rotary",
                  rotary_dim=8, gated_mlp=True, final_rms=True),
    "gemma": dict(n_key_value_heads=2, act_fn="gelu_pytorch_tanh", normalization_type="RMS", positional_embedding_type="rotary",
                  rotary_dim=8, gated_mlp=True, final_rms=True, use_normalization_before_and_after=True),
    "gpt2": dict(act_fn="gelu_new", normalization_type="LN", positional_embedding_type="standard"),
}


class _NoCuda:
    """torch for eap.evaluate on a machine with no CUDA (its Gemma head mask is made on 'cuda' by name)."""
    def __getattr__(self, k):
        return getattr(torch, k)

    @staticmethod
    def zeros(*a, device=None, **k):
        return torch.zeros(*a, **k)


def tiny(monkeypatch, shape):
    """A tiny model of SHAPE in MIB's settings, two batches of pairs (one token changed) and eap.evaluate, patched to
    read token lists."""
    tl = pytest.importorskip("transformer_lens")
    eval_mod = pytest.importorskip("eap.evaluate")
    import eap.utils
    torch.manual_seed(0)
    cfg = tl.HookedTransformerConfig(n_layers=3, d_model=32, d_head=8, n_heads=4, d_mlp=48, d_vocab=50, n_ctx=16,
                                     dtype=torch.float32, device="cpu", **SHAPES[shape])
    model = tl.HookedTransformer(cfg)
    if not torch.cuda.is_available():
        monkeypatch.setattr(eval_mod, "torch", _NoCuda())
    for b in model.blocks:
        b.attn.b_O.data.normal_()
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    monkeypatch.setattr(eap.utils, "tokenize_plus", fake_tokenize)
    monkeypatch.setattr(eval_mod, "tokenize_plus", fake_tokenize)
    g = torch.Generator().manual_seed(1)
    dl = []
    for _ in range(2):
        c = torch.randint(0, 50, (3, 7), generator=g)
        x = c.clone()
        x[:, 2] = torch.randint(0, 50, (3,), generator=g)
        dl.append(([" ".join(map(str, r.tolist())) for r in c], [" ".join(map(str, r.tolist())) for r in x],
                   torch.randint(0, 50, (3, 2), generator=g)))
    return model, dl, eval_mod


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_measured_equals_evaluate_graph(monkeypatch, shape):
    from eap.graph import Graph
    model, dl, eval_mod = tiny(monkeypatch, shape)
    graph = Graph.from_model(model)
    rec = mc.measured(model, graph, dl, metric, refine=12, log=lambda s: None, rows=6)
    real = graph.real_edge_mask.clone()
    top = graph.scores.flatten().argsort(descending=True)[:12].tolist()  # the refined edges ranked first, by their own change
    assert top == [f * graph.n_backward + b for f, b, _, _ in sorted(rec["single"], key=lambda s: -s[3])]
    paths = mc.measured_scores(graph.real_edge_mask, rec, "paths")
    for k in (5, 20, 60):  # every prefix of the paths order survives MIB's pruning unchanged
        graph.in_graph[:] = False
        graph.in_graph.view(-1)[paths.flatten().argsort(descending=True)[:k]] = True
        graph.in_graph &= graph.real_edge_mask
        graph.nodes_in_graph[:] = True
        before = graph.in_graph.clone()
        graph.prune()
        assert torch.equal(graph.in_graph, before), k
    graph.prune = lambda *a, **k: None

    def m_with(cut):  # (every node in the graph: a new Graph's nodes_in_graph is all False, which evaluate_graph reads)
        graph.nodes_in_graph[:] = True
        graph.in_graph[:] = real
        cut(graph.in_graph)
        graph.in_graph &= real
        return float(eval_mod.evaluate_graph(model, graph, dl, metric, quiet=True, intervention="patching").mean())
    full = m_with(lambda x: None)
    assert rec["full"] == pytest.approx(full, abs=1e-5)
    for f in range(graph.n_forward):
        if real[f].any():
            assert rec["writers"][f] == pytest.approx(abs(full - m_with(lambda x, f=f: x[f].fill_(False))), abs=1e-4), f
    for b in range(graph.n_backward):
        if real[:, b].any():
            assert rec["readers"][b] == pytest.approx(abs(full - m_with(lambda x, b=b: x[:, b].fill_(False))), abs=1e-4), b
    assert len(rec["single"]) == 12
    every = mc.measured(model, Graph.from_model(model), dl, metric, refine=int(real.sum()), log=lambda s: None, rows=6)
    assert len(every["single"]) == int(real.sum())  # (every edge alone: each writer's change through each reader's input)
    for f, b, _, d in every["single"]:
        assert d == pytest.approx(abs(full - m_with(lambda x, f=f, b=b: x[f, b].fill_(False))), abs=1e-4), (f, b)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_node_sets_and_eliminate(monkeypatch, shape):
    """node_sets' circuits equal evaluate_graph's with node_edges' edges (MIB's pruning on: every node circuit is whole),
    and eliminate's ranking puts each step's node circuit first."""
    from eap.graph import Graph
    model, dl, eval_mod = tiny(monkeypatch, shape)
    graph = Graph.from_model(model)
    nf = graph.n_forward
    g = torch.Generator().manual_seed(2)
    masks = torch.rand(6, nf, generator=g) < 0.6
    masks[:3, 0] = True  # (the input in some circuits, out of others)
    masks = torch.cat([masks, ~torch.eye(nf, dtype=torch.bool)[-4:]])  # (and single late writers out: forwards from a layer)
    names = {"hook_embed"} | {f"blocks.{L}.{h}" for L in range(3) for h in ("attn.hook_z", "hook_mlp_out", "hook_resid_pre")}
    saved = (model.cfg.use_attn_result, model.cfg.use_split_qkv_input)
    model.cfg.use_attn_result = model.cfg.use_split_qkv_input = False
    batches, n_ex, full, _ = mc.prepare(model, dl, metric, names)
    got = mc.node_sets(model, batches, n_ex, masks, metric, rows=6)
    model.cfg.use_attn_result, model.cfg.use_split_qkv_input = saved
    for k, m in zip(masks, got):
        e, _ = mc.node_edges(graph, k)
        graph.in_graph[:] = e
        graph.nodes_in_graph[:] = k
        want = float(eval_mod.evaluate_graph(model, graph, dl, metric, quiet=True, intervention="patching").mean())
        assert m == pytest.approx(want, abs=1e-4)
    rec = mc.eliminate(model, graph, dl, metric, log=lambda s: None, rows=6, cands=3)
    assert sum(len(b) for b, _ in rec["steps"]) == nf
    keep = torch.ones(nf, dtype=torch.bool)
    for b, f in rec["steps"][:-1]:  # (after each step, the top edges are exactly that step's node circuit)
        keep[b] = False
        e, n = mc.node_edges(graph, keep)
        top = graph.scores.flatten().argsort(descending=True)[:n]
        assert set(top.tolist()) == set(e.flatten().nonzero().flatten().tolist())


def test_select_reads_its_curve_off_the_proposals(monkeypatch, tmp_path):
    """select's test curve is, size by size, the curve of the proposal it chose on the train pairs."""
    model, dl, eval_mod = tiny(monkeypatch, "llama")
    monkeypatch.setattr(mc, "data", lambda *a, **k: (dl, 6))
    import eap.utils
    rec = mc.select(model, "ioi", "tiny", "select:measured", tmp_path, dl, 6, lambda *a, mean=True, loss=False: metric(*a[:4]),
                    12, 6, None, 3, lambda s: None)
    fs = rec["proposals"]["measured"]["abs_True"]["faithfulness"]
    assert rec["abs_True"]["faithfulness"] == fs
    assert rec["abs_True"]["CMD"] == pytest.approx(mc.areas(fs)[1])
    assert (tmp_path / "submission" / "select" / "ioi_tiny" / "0.1.pt").exists()


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_nodes_writers_equal_measured(monkeypatch, shape):
    """nodes' writer tests (through node_sets) equal measured's (through its own hooks)."""
    from eap.graph import Graph
    model, dl, _ = tiny(monkeypatch, shape)
    a = mc.measured(model, Graph.from_model(model), dl, metric, refine=0, log=lambda s: None, rows=6)
    b = mc.nodes(model, Graph.from_model(model), dl, metric, log=lambda s: None, rows=6)
    real = Graph.from_model(model).real_edge_mask
    for f in range(len(b["noise"])):
        if real[f].any():
            assert b["noise"][f] == pytest.approx(a["writers"][f], abs=1e-4), f


def test_edgefix_and_bootstrap(monkeypatch, tmp_path):
    """select's edgefix circuits hold at most their size's edges and are chosen only by train faithfulness; bootstrap's
    point CMD is select's and a curve against itself differs by exactly 0."""
    model, dl, eval_mod = tiny(monkeypatch, "llama")
    monkeypatch.setattr(mc, "data", lambda *a, **k: (dl, 6))
    m = lambda *a, mean=True, loss=False: metric(*a[:4])  # noqa: E731
    rec = mc.select(model, "ioi", "tiny", "select:measured", tmp_path, dl, 6, m, 12, 6, None, 3, lambda s: None, fix_max=0.2)
    n_edges = rec["edges"]
    for e in rec["edgefix"]:
        assert e["tests"] >= 1
    for i, (c, a, f) in enumerate(rec["chosen"]):
        assert rec["abs_True"]["weighted_edge_counts"][i] <= max(1, int(mc.PERCENTAGES[i] * n_edges)) or mc.PERCENTAGES[i] == 1
    bt = mc.bootstrap(rec["test_full"], rec["test_none"], rec["abs_True"]["test_metric"], rec["abs_True"]["test_metric"], B=50)
    assert bt["CMD"] == pytest.approx(rec["abs_True"]["CMD"], abs=1e-6)
    assert bt["diff"] == 0 and bt["diff_CI"] == [0.0, 0.0]


def test_lean_attention_equals_transformer_lens():
    """lean_attention's per-head results and logits equal TransformerLens' own (float32, MIB's use_attn_result on)."""
    tl = pytest.importorskip("transformer_lens")
    from transformer_lens.components import abstract_attention as aa
    torch.manual_seed(0)
    cfg = tl.HookedTransformerConfig(n_layers=2, d_model=32, d_head=8, n_heads=4, d_mlp=48, d_vocab=50, n_ctx=16,
                                     dtype=torch.float32, device="cpu", **SHAPES["llama"])
    model = tl.HookedTransformer(cfg)
    model.cfg.use_attn_result = True
    x = torch.randint(0, 50, (2, 7))
    forward = aa.AbstractAttention.forward
    want, cw = model.run_with_cache(x, names_filter=lambda n: n.endswith("hook_result"))
    try:
        assert mc.lean_attention()
        got, cg = model.run_with_cache(x, names_filter=lambda n: n.endswith("hook_result"))
    finally:
        aa.AbstractAttention.forward = forward
        aa.AbstractAttention._lean_result = False
    assert torch.allclose(got, want, atol=1e-5)
    for k in cw:
        assert torch.allclose(cg[k], cw[k], atol=1e-5), k
