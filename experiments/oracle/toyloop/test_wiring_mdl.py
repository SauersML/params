"""Executed-description contracts only; no model loads or forward passes."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'method/oracle'))
import circuit
import native
import reward_exec as reward


@pytest.fixture
def compiled(monkeypatch):
    mapping={
        'L0H0':[('h.0.attn.v_proj',0),('h.0.attn.o_proj',0)],
        'L0H1':[('h.0.attn.v_proj',1),('h.0.attn.o_proj',1)],
        'L1H0':[('h.1.attn.v_proj',0),('h.1.attn.o_proj',0)],
    }
    monkeypatch.setattr(circuit,'unit_pieces',lambda toy,unit:mapping[unit])
    C={'h.0.attn.v_proj':2,'h.0.attn.o_proj':2,'h.1.attn.v_proj':1,'h.1.attn.o_proj':1}
    nat=SimpleNamespace(C=C,names=list(C))
    toy=SimpleNamespace(cfg={'n_layer':2,'n_head':4},sd={},texts=[[0,1,2]])
    def price(text,complete=False):
        graph=circuit.unit_graph(circuit.parse(text),toy)
        if complete:
            graph=native.Graph(graph.node_set(),complete=True)
        return reward.executed_description(graph,nat,3,[2])
    return price


BASE='flowchart LR\n L0H0 --> L1H0\n L1H0 --> out\n'


def test_annotations_and_duplicates_have_identical_executed_cost(compiled):
    decorated=BASE+' tokBOS --> L0H0\n L0H0 --> L1H0\nEnglish: An ignored assertion.\n'
    assert compiled(decorated)==compiled(BASE)


def test_additional_executable_edge_costs_more(compiled):
    assert compiled(BASE+' L0H0 --> out\n')['description_bits']>compiled(BASE)['description_bits']


def test_complete_and_explicit_identical_graph_price_parity(compiled):
    explicit=BASE+' L0H0 --> out\n'
    assert compiled(explicit)==compiled(explicit,complete=True)


def test_invalid_reward_and_fixed_ordering(monkeypatch):
    assert reward.reward_wiring_mdl({'toy':None},'not a circuit')==-1
    scored={'per_text':[{'suff':1.,'description_cost':.1}]}
    extra={'per_text':[{'suff':1.,'description_cost':.2}]}
    removed={'per_text':[{'suff':.5,'description_cost':.01}]}
    assert reward.wiring_mdl_value(scored)>reward.wiring_mdl_value(extra)>reward.wiring_mdl_value(removed)
    with pytest.raises(ValueError):
        reward.wiring_mdl_value({'per_text':[{'suff':1.,'description_cost':float('nan')}]})
    monkeypatch.setattr(reward,'texts_of',lambda toy:[[0,1]])
    monkeypatch.setattr(reward,'score_wiring',lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('bad edge')))
    assert reward.reward_wiring_mdl({'toy':None},BASE)==-1


def test_description_verifier_failures_propagate(monkeypatch):
    with pytest.raises(RuntimeError):
        reward.executed_description(native.Graph(),SimpleNamespace(C={},names=[]),3,[2])
    monkeypatch.setattr(reward,'texts_of',lambda toy:[[0,1]])
    monkeypatch.setattr(reward,'score_wiring',lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('nonfinite architecture budget')))
    with pytest.raises(RuntimeError,match='architecture budget'):
        reward.reward_wiring_mdl({'toy':None},BASE)
