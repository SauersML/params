"""Model-free graph binding/column tests. Physical contrast is a separate receipt."""
from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'vpd')]
import native
import semantic

SITE='h.0.mlp.c_fc'
A=(SITE,0,7);B=(SITE,0,19)

def graph():return native.Graph([A,B],parents={B:[A]},out=[B])
def variables():return [{'name':'V1','group':[A]},{'name':'V2','group':[B]}]
def tx():return SimpleNamespace(ids=[1],targets=[0])

class Scorer:
    nat=SimpleNamespace(dev='cpu')
    def __init__(self):self.calls=[];self.sources=[]
    def _text(self,ids,targets,**kw):
        self.calls.append(kw)
        return {'lq':torch.tensor([[.8,.2]]).log(),
                'act':{SITE:torch.tensor([[float(ids[0]),float(ids[0]*10)]])},
                'out':{SITE:torch.zeros(1,2)},'col':{SITE:{7:0,19:1}}}
    def _patched(self,w,rows):
        self.sources.extend(rows)
        return torch.tensor([[[.2,.8]]]).log()


def test_graph_world_used_for_recipient_and_both_donors(monkeypatch):
    monkeypatch.setattr(native,'free',lambda *_:None)
    sc=Scorer();g=graph();ck=semantic.Checker(sc,tx(),variables(),graph=g)
    ck.run({'transplants':[('V1',{0:2}),('V2',{0:3})]})
    assert len(sc.calls)==3 and all(c['graph'] is g for c in sc.calls)
    assert sc.sources[0][0]['act'][SITE].tolist()==[[2.,30.]]
    # Source1 remained untouched while second sparse globalID19 maps to column1.
    assert ck.text({0:2})['act'][SITE].tolist()==[[2.,20.]]


def test_default_text_calls_have_no_graph_keyword():
    sc=Scorer();ck=semantic.Checker(sc,tx(),variables())
    ck.text({0:2})
    assert all('graph' not in c for c in sc.calls)


@pytest.mark.parametrize('kind',['absent','block','mask','duplicate'])
def test_unsupported_carriers_fail_before_any_forward(kind):
    sc=Scorer();g=graph();vs=variables()
    if kind=='absent':g=native.Graph([A])
    if kind=='block':vs[0]['group']=[(SITE,0,semantic.BLOCK)]
    if kind=='mask':g=native.Graph(masks={SITE:torch.ones(1,20,dtype=torch.bool)})
    if kind=='duplicate':vs[1]['name']='V1'
    with pytest.raises(ValueError):semantic.Checker(sc,tx(),vs,graph=g)
    assert not sc.calls


def test_same_carriers_broken_edge_and_input_bound_differently():
    g=graph();first=semantic.graph_claim_binding(g,tx(),variables())
    g.parents={};second=semantic.graph_claim_binding(g,tx(),variables())
    assert first['graph_sha256']!=second['graph_sha256']
    assert first['carriers_sha256']==second['carriers_sha256']
    changed=tx();changed.ids=[2]
    assert semantic.graph_claim_binding(g,changed,variables())['input_sha256']!=first['input_sha256']
    assert 'caller must separately bind' in first['model_content_binding']


def test_contrast_preserves_numeric_claim_and_does_not_add_reward(monkeypatch):
    calls=[]
    def check(sc,t,vs,text,model,fresh,pool,**kw):
        calls.append(kw)
        return {'invariants':[{'id':1,'line':'frozen','intervention':{'transplants':[]},'outcome':['top',1],
                               'holds':not bool(kw.get("graph")),'top':0 if kw.get("graph") else 1}], 'bad':0}
    monkeypatch.setattr(semantic,'check_invariants',check)
    out=semantic.check_invariants_contrast(None,tx(),variables(),'frozen',graph())
    assert calls[0]=={"include_baseline":True} and calls[1]['graph'] is not None
    assert out['paired'][0]['model_holds'] and not out['paired'][0]['both_hold']
    assert not out['free_prose_verified'] and not out['reward_applied']


def test_existing_diagnostic_preserves_model_fields_rules_out_scope(monkeypatch):
    import components
    import variables as v
    sc=object()
    monkeypatch.setattr(v,'Scorer',lambda ge:sc)
    monkeypatch.setattr(v,'parse',lambda bound,c: variables())
    monkeypatch.setattr(semantic,'graph_claim_binding',lambda *args:{})
    baseline={'invariants':[],'bad':0,'holds':1.,'credit':2.,'baseline':{'top':1,'p_top':.8}}
    own={'invariants':[],'bad':0,'holds':0.,'credit':-2.,'baseline':{'top':0,'p_top':.9},'execution_binding':{'graph_sha256':'a'*64}}
    monkeypatch.setattr(semantic,'check_invariants_contrast',lambda *args:{'model':baseline,'circuit':own,'paired':[]})
    monkeypatch.setattr(semantic,'check_rules_out',lambda *args:{'holds':True})
    answer={'invariants':['I1: frozen','RULES OUT: frozen'],'variables':[{'name':'V1'}]}
    out=components.invariant_diagnostic(SimpleNamespace(nat=SimpleNamespace(C={})),tx(),answer,'vpd4l',graph=graph())
    assert out['credit']==2. and out['circuit']['credit']==-2.
    assert out['baseline']['top']==1 and out['circuit']['baseline']['top']==0
    assert out['rules_out_scope'].startswith('full_model_only')
    assert not out['free_prose_verified'] and not out['reward_applied']


def test_existing_diagnostic_default_calls_only_legacy_checker(monkeypatch):
    import components
    import variables as v
    monkeypatch.setattr(v,'Scorer',lambda ge:None)
    monkeypatch.setattr(v,'parse',lambda bound,c: variables())
    monkeypatch.setattr(semantic,'check_invariants',lambda *args:{'credit':3.})
    monkeypatch.setattr(semantic,'check_invariants_contrast',lambda *a:pytest.fail('opt-in must not run by default'))
    out=components.invariant_diagnostic(SimpleNamespace(nat=SimpleNamespace(C={})),tx(),{'invariants':['frozen']},'vpd4l')
    assert out=={'executed':True,'credit':3.}


def test_unrelated_block_node_rejected_before_any_forward():
    sc=Scorer();g=graph();g.nodes.add(('h.0.mlp.down_proj',0,semantic.BLOCK))
    with pytest.raises(ValueError,match='BLOCK nodes anywhere'):
        semantic.Checker(sc,tx(),variables(),graph=g)
    assert not sc.calls


def test_duplicate_ids_rejected_before_any_forward(monkeypatch):
    monkeypatch.setattr(semantic,'check_invariants',lambda *a,**k:pytest.fail('no forward/check allowed'))
    with pytest.raises(ValueError,match='duplicate invariant IDs'):
        semantic.check_invariants_contrast(None,tx(),variables(),'I1: x -> y\nI1: z -> w',graph())


def test_graph_eval_pool_cannot_silently_use_search_mean():
    sc=Scorer()
    with pytest.raises(ValueError,match='fresh_mean'):
        semantic.Checker(sc,tx(),variables(),graph=graph(),pool='eval')
    assert not sc.calls


def test_diagnostic_duplicate_claims_are_invalid_not_reward_batch_exception(monkeypatch):
    import components
    import variables as v
    monkeypatch.setattr(v,'Scorer',lambda ge:None)
    monkeypatch.setattr(v,'parse',lambda bound,c: variables())
    answer={'invariants':['I1: x -> y','I1: z -> w']}
    out=components.invariant_diagnostic(SimpleNamespace(nat=SimpleNamespace(C={})),tx(),answer,'vpd4l',graph=graph())
    assert not out['executed'] and out['reason']=='invalid_graph_invariant_claim'
    assert not out['free_prose_verified'] and not out['reward_applied']
