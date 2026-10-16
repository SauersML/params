"""mib_compare.py's measured tests against MIB's own evaluate_graph (EAP-IG) on a tiny random Llama-shaped
HookedTransformer (grouped-query attention, rotary, gated MLP): every writer, reader and refined edge's change equals the
change evaluate_graph gives with the same edges removed (its pruning step off), whatever the row grouping, and the
refined edges rank first in the order of their own changes. Needs MIB's
environment (transformer_lens, EAP-IG's eap).

  ~/mpd-data/ext/mibenv/bin/python -m pytest experiments/oracle/test_mib_compare.py -q
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


def test_measured_equals_evaluate_graph(monkeypatch):
    tl = pytest.importorskip("transformer_lens")
    eval_mod = pytest.importorskip("eap.evaluate")
    import eap.utils
    from eap.graph import Graph
    torch.manual_seed(0)
    cfg = tl.HookedTransformerConfig(n_layers=2, d_model=32, d_head=8, n_heads=4, n_key_value_heads=2, d_mlp=48, d_vocab=50,
                                     n_ctx=16, act_fn="silu", normalization_type="RMS", positional_embedding_type="rotary",
                                     rotary_dim=8, gated_mlp=True, final_rms=True, dtype=torch.float32)
    model = tl.HookedTransformer(cfg)
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
    graph = Graph.from_model(model)
    rec = mc.measured(model, graph, dl, metric, refine=12, log=lambda s: None, rows=6)
    real = graph.real_edge_mask.clone()
    top = graph.scores.flatten().argsort(descending=True)[:12].tolist()  # the refined edges ranked first, by their own change
    assert top == [f * graph.n_backward + b for f, b, _, _ in sorted(rec["single"], key=lambda s: -s[3])]
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
    for f, b, _, d in rec["single"]:
        assert d == pytest.approx(abs(full - m_with(lambda x, f=f, b=b: x[f, b].fill_(False))), abs=1e-4), (f, b)
