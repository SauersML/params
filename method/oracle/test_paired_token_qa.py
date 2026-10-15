from types import SimpleNamespace
import pytest
import torch
import native
import semantic
import paired_token_qa as bridge

A=('h.0.mlp.c_fc',0,0)
def graph():
    return native.Graph([A], parents={A:[]}, out=[A])

class Scorer:
    nat=SimpleNamespace(dev='cpu')
    def __init__(self):self.calls=[]
    def _text(self, ids, targets, **kw):
        self.calls.append((list(ids),kw))
        # Same original distribution. Intentionally wrong circuit after edit.
        probs=[.8,.2] if ids[0]==0 or 'graph' in kw else [.1,.9]
        return {'lq':torch.tensor([probs]).log()}


def test_target_truth_retained_when_provided_circuit_disagrees():
    sc=Scorer();tx=SimpleNamespace(ids=[0],targets=[0])
    cands=[{'kind':'token','ask':ask,'position':0,'new':1,'top':999} for ask in ('change','kl')]
    rows=bridge.measure_token_candidates(sc,tx,graph(),cands,['a','b'],fidelity_bits=.001)
    assert len(rows)==2 and len(sc.calls)==4 # paired change/KL reuse same worlds
    assert all(r['target_model_label']['top']==1 and r['source_candidate']['top']==999 for r in rows)
    assert rows[0]['target_model_label']['falls'] is True
    assert rows[0]['circuit_outcome']['top']==0
    assert all(r['control_status']=='failed_fidelity' for r in rows)
    assert all(not r['fidelity']['label_agreement'] for r in rows)
    assert rows[1]['target_model_label']['answer'] is True
    assert cands[0]['top']==999
    assert rows[0]['execution_binding']['graph_sha256']


@pytest.mark.parametrize('bad',[
    {'kind':'remove','ask':'change','position':0,'new':1},
    {'kind':'token','ask':'change','position':-1,'new':1},
    {'kind':'token','ask':'change','position':0,'new':2},
    {'kind':'token','ask':'change','position':0,'new':0}])
def test_invalid_candidates_fail_before_forward(bad):
    sc=Scorer()
    with pytest.raises(ValueError):bridge.measure_token_candidates(sc,SimpleNamespace(ids=[0],targets=[0]),graph(),[bad],['a','b'],fidelity_bits=.001)
    assert not sc.calls


def test_baseline_failure_retained_even_if_changed_predictions_agree():
    class BaselineBad(Scorer):
        def _text(self,ids,targets,**kw):
            probs=[.2,.8] if 'graph' in kw and ids[0]==0 else ([.8,.2] if ids[0]==0 else [.1,.9])
            return {'lq':torch.tensor([probs]).log()}
    row=bridge.measure_token_candidates(BaselineBad(),SimpleNamespace(ids=[0],targets=[0]),graph(),
        [{'kind':'token','ask':'change','position':0,'new':1}],['a','b'],fidelity_bits=.001)[0]
    assert row['fidelity']['intervention_kl_bits']==0.
    assert row['fidelity']['baseline_kl_bits']>.001
    assert row['control_status']=='failed_fidelity'
    # falls is evaluated for target-model reference token0 even though circuit top0=1.
    assert row['reference_top']==0 and row['circuit_baseline_top']==1


@pytest.mark.parametrize("broken", [False, True])
def test_tiny_physical_full_and_broken_circuit_control(broken):
    from test_goal_eval import tiny_native
    import goal_eval
    import variables
    nat=tiny_native(seed=0,layers=1,d=2,heads=1,vocab=3,C=2)
    ids=[0,1];targets=[1]
    nodes=[(n,t,c) for n in nat.names for t in range(2) for c in range(nat.C[n])]
    # Explicit all-architecture graph; use native legal full token/pair wiring.
    parents,out=native.all_edges(nodes,targets)
    g=native.Graph(nodes,parents=parents,out=out)
    g.tok_positions=[0,1]
    g.tok_edges=[(p,nd) for nd in nodes for p in range(nd[1]+1)]
    g.tok_out=[1]
    if broken:
        g=native.Graph([],parents={},out=[])
        g.tok_positions=[0,1];g.tok_edges=[];g.tok_out=[1]
    rows=bridge.measure_token_candidates(variables.Scorer(goal_eval.GoalEval(nat,semantics='c')),
        SimpleNamespace(ids=ids,targets=targets),g,
        [{'kind':'token','ask':'change','position':0,'new':2}],['a','b','c'],fidelity_bits=1e-6)
    assert len(rows)==1
    assert rows[0]['fidelity']['passed'] is (not broken)
    if broken:
        assert rows[0]['control_status']=='failed_fidelity'
    else:
        assert rows[0]['fidelity']['label_agreement']


def test_pair_labels_remain_directly_scoreable_by_qa_env():
    import qa_env
    row=bridge.measure_token_candidates(Scorer(),SimpleNamespace(ids=[0],targets=[0]),graph(),
        [{'kind':'token','ask':'change','position':0,'new':1}],['a','b'],fidelity_bits=.001)[0]
    assert qa_env.score({'qa_env':[row['target_model_label']]}, "FINAL 1: falls=yes top='b'")['reward']==1.
    assert row['target_model_label']['cell']=='True/True'
