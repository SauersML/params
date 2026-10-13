"""Reward controls for descriptions discarded by the executable graph parser."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import components
import goal_eval

EMPTY = 'CIRCUIT:\n```mermaid\nflowchart LR\n```'
JUNK = 'CIRCUIT:\n```mermaid\nflowchart LR\n subgraph P0["@0"]\n a["L99.down#0"]\n end\n a --> OUT((prediction))\n```'
REAL = 'CIRCUIT:\n```mermaid\nflowchart LR\n subgraph P0["@0"]\n a["L0.down#0"]\n end\n a --> OUT((prediction))\n```'


def fake_eval(monkeypatch, teacher=False):
    C = {'h.0.mlp.down_proj': 2}
    tx = SimpleNamespace(vpd=None)
    ge = SimpleNamespace(nat=SimpleNamespace(C=C, positions=lambda targets: 1), text=lambda *a, **k: tx)
    ge.score = lambda tx, graphs, texts: [
        {'nodes': g.count(), 'E1': None, 'E2': None, 'f1_live': 0., 'n_live': 0., 'f3': 0.} for g in graphs]
    ge.fidelity = lambda tx, graphs, pool, **kw: [{'c': 0.} for g in graphs]
    ge.tau = lambda *a: {'phi_tau': -.7539940449122631, 'bits_start': 140923.08623763727}
    ge.line_loo = lambda tx, graph, *a, **kw: {'bits': goal_eval.size_bits(graph, 1, 2, [0]), 'lines': []}
    monkeypatch.setattr(goal_eval, 'phi', lambda row: (-.007588067954579058, None))
    monkeypatch.setattr(components, '_teacher_ref', lambda *a: {'phi': .2, 'compact': 2.} if teacher else None)
    task = {'id': 'mock', 'model': 'vpd4l', 'prompts': [{'token_ids': [1], 'target_positions': [0], 'model_top': [[[0, 1.]]]}]}
    return ge, task


@pytest.mark.parametrize('teacher', [False, True])
def test_discarded_description_cannot_award_null_graph_compactness(monkeypatch, teacher):
    ge, task = fake_eval(monkeypatch, teacher)
    empty, junk = components.reward(ge, task, {'qa': []}, [EMPTY, JUNK], pool={}, teacher={'nodes': []} if teacher else None)
    assert empty['valid'] and junk['valid']
    assert empty['row']['nodes'] == junk['row']['nodes'] == 0
    assert empty['executable_bits'] == junk['executable_bits'] == 0
    assert junk['bits'] > empty['bits']  # discarded description remains charged/reported
    assert empty['compact'] is junk['compact'] is None
    assert junk['reward'] == pytest.approx(empty['reward'])
    assert junk['D'] == pytest.approx(empty['D'])
    if teacher:
        assert empty['D'] < 0  # discarded syntax must not bypass negative teacher dominance
    credits = components.line_credit(JUNK, components.parse(JUNK), {'bits': 0, 'lines': []}, None, junk, None, None)
    assert credits and all(credit <= 0 for _, credit in credits)


def test_real_graph_still_pays_for_discarded_description(monkeypatch):
    ge, task = fake_eval(monkeypatch)
    both = REAL.replace(' end', ' bad["L99.down#0"]\n end')
    real, junk = components.reward(ge, task, {'qa': []}, [REAL, both], pool={})
    assert real['row']['nodes'] == junk['row']['nodes'] == 1
    assert real['executable_bits'] == junk['executable_bits'] > 0
    assert junk['bits'] > real['bits']
    assert junk['compact'] < real['compact']
    assert junk['reward'] < real['reward']
