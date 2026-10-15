"""Checks of the neuron edge stage (neuron_edges.py) on a tiny random Qwen3: with no cut and every neuron live the run is
the model's; the circuit run is the model with every other neuron zeroed; a cut edge's measured effect follows its
linear score at small writes; pruning keeps the contrast within its floor and the graph lists poles and edges.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_neuron_edges.py -q
"""
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
    assert ec.c_full == pytest.approx(contrast_of(model, ids, ec.keep), abs=1e-4)
    E = len(ec.resid_all)
    assert ec.contrasts(torch.zeros(1, E, dtype=torch.bool), ec.resid_all)[0] == pytest.approx(ec.c_full, abs=1e-4)


def test_cuts_follow_linear_scores(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="model")
    rs, rt = ed.linear_scores()
    edges = ed.resid_all + ed.route_all[:64]
    cuts = torch.eye(len(edges), dtype=torch.bool)
    meas = torch.tensor([ed.c_full - c for c in ed.contrasts(cuts, edges)])
    lin = torch.cat([rs, rt[:64]])
    assert float(torch.corrcoef(torch.stack([meas, lin]))[0, 1]) > 0.8


def test_prune_and_graph(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, background="circuit")
    got = neuron_edges.prune(ed, log=lambda m: None)
    rec = got["record"]
    up = rec["contrast_all_edges"] >= rec["contrast_no_edges"]
    assert (rec["contrast_kept"] >= rec["floor"] - 1e-6) if up else (rec["contrast_kept"] <= rec["floor"] + 1e-6)
    g = neuron_edges.graph(ed, got)
    assert len(g["nodes"]) == len(ed.N) and len(g["edges"]) == rec["kept"]
    assert all(n["pole"] in (1, -1) and n["regime"] in ("g+u+", "g+u-", "g-u+", "g-u-") for n in g["nodes"])
