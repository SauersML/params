"""Synthetic contract regressions; no oracle/model/Judge calls."""
import importlib.util,json,math
from pathlib import Path
from types import SimpleNamespace
import pytest
spec=importlib.util.spec_from_file_location('checked_neuron_verify',Path(__file__).with_name('neuron_verify.py'));nv=importlib.util.module_from_spec(spec);spec.loader.exec_module(nv)
def item():return dict(layer=0,index=1,pole='+',corpus_max=2.,explanation='Synthetic fixture explanation')
def plan():return dict(max_texts=['maximum '+str(i) for i in range(3)],near_misses=['near '+str(i) for i in range(5)],distant_texts=['distant '+str(i) for i in range(5)],distant_concept='synthetic distant concept',steering_prediction='synthetic steering claim')
class Target:
 def __init__(self):self.layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(in_features=4)))];self.name='SYNTHETIC';self.calls=[]
 def poles(self,texts,l,i):
  self.calls.append(('poles',list(texts),l,i));return [dict(pos=1. if x.startswith('maximum') else .1,neg=1. if x.startswith('maximum') else .1,regime_pos='++',regime_neg='+-') for x in texts]
 def steered(self,l,i,value):self.calls.append(('steer',l,i,value));return 'synthetic continuation'
class Reader:
 def __init__(self,p=None):self.p=p if p is not None else plan();self.calls=[]
 def __call__(self,asks,**kw):self.calls.append(asks);return [json.dumps(self.p)]*len(asks) if len(self.calls)==1 else ['A']*len(asks)
def test_complete_proposal_retains_audit_and_scopes_success():
 tg=Target();rd=Reader();r=nv.verify(tg,rd,[item()])[0]
 assert r['valid'] and r['passes_t1'] and not r['causal_mechanism_verified']
 assert len(r['raw_pole_measurements'])==13 and r['test_texts']=={k:plan()[k] for k in ('max_texts','near_misses','distant_texts')}
 assert r['explanation']==item()['explanation'] and json.loads(r['raw_proposal'])==plan() and r['model_binding']['native_index']==1
 assert r['steer_raw_response']=='A' and len(rd.calls)==2 and len(tg.calls)==3
@pytest.mark.parametrize('change',[{'near_misses':[]},{'near_misses':'abcde'},{'distant_texts':[]},{'max_texts':[]},{'max_texts':['a','b',9]},{'near_misses':['']*5},{'near_misses':['same']*5},{'steering_prediction':''},{'distant_concept':None}])
def test_incomplete_malformed_proposal_never_gets_t1_or_target_calls(change):
 tg=Target();p=plan();p.update(change);rd=Reader(p);r=nv.verify(tg,rd,[item()])[0]
 assert not r['valid'] and not r['passes_t1'] and not tg.calls and len(rd.calls)==1 and r['raw_proposal']
def test_distinctness_across_groups_not_just_within_each():
 p=plan();p['near_misses'][0]=p['max_texts'][0].upper();tg=Target()
 assert not nv.verify(tg,Reader(p),[item()])[0]['valid'] and not tg.calls
@pytest.mark.parametrize('change',[{'layer':-1},{'layer':True},{'layer':1},{'index':4},{'index':-1},{'index':1.0},{'pole':'other'},{'corpus_max':0},{'corpus_max':-1},{'corpus_max':float('nan')},{'corpus_max':float('inf')},{'corpus_max':'2'},{'corpus_max':1e308},{'explanation':''}])
def test_bad_items_fail_before_any_reader_or_target_calls(change):
 tg=Target();rd=Reader();it=item();it.update(change)
 with pytest.raises(ValueError):nv.verify(tg,rd,[item(),it])
 assert not rd.calls and not tg.calls
@pytest.mark.parametrize('responses',[[],['one','extra'], 'a string', [None]])
def test_reader_cardinality_or_type_cannot_drop_items(responses):
 tg=Target()
 with pytest.raises(ValueError,match='response count/type'):nv.verify(tg,lambda *a,**kw:responses,[item()])
 assert not tg.calls
def test_judge_cardinality_mismatch_is_not_silent_truncation():
 rd=Reader();calls=0
 def respond(*a,**kw):
  nonlocal calls
  calls+=1;return [json.dumps(plan())] if calls==1 else []
 with pytest.raises(ValueError,match='response count/type'):nv.verify(Target(),respond,[item()])
def test_target_missing_measurements_or_nonfinite_cannot_pass():
 for result in ([],[{'pos':float('nan')}]*13):
  tg=Target();tg.poles=lambda *a:result
  with pytest.raises(ValueError):nv.verify(tg,Reader(),[item()])
def test_negative_pole_and_steering_amplitude_preserved():
 tg=Target();it=item();it['pole']='-';r=nv.verify(tg,Reader(),[it])[0]
 assert r['passes_t1'] and ('steer',0,1,-6.) in tg.calls
def test_extra_judge_prose_is_not_valid_choice():
 rd=Reader();respond=rd.__call__;calls=0
 def custom(*a,**kw):
  nonlocal calls
  calls+=1;return respond(*a,**kw) if calls==1 else ['A because it fits']
 r=nv.verify(Target(),custom,[item()])[0];assert not r['steer_response_valid'] and r['steer_match']==0

@pytest.mark.parametrize('raw,valid',[('I compared both continuations.\nB.',True),('a',True),
                                    ('The passage describes a picture of a.',False),
                                    ('The continuation refers to a',False)])
def test_judge_terminal_answer_not_prose_article(raw,valid):
 rd=Reader();respond=rd.__call__;calls=0
 def custom(*a,**kw):
  nonlocal calls
  calls+=1
  if calls==1:return respond(*a,**kw)
  assert kw['max_tokens']==40
  return [raw]
 r=nv.verify(Target(),custom,[item()])[0]
 assert r['steer_response_valid'] is valid
 assert r['steer_raw_response']==raw
 if not valid:assert r['steer_match']==0
def test_summary_invalid_poles_count_as_failures(tmp_path,capsys):
 good=nv.verify(Target(),Reader(),[item()])[0]
 bad={**item(),'pole':'-','valid':False,'passes_t1':False}
 p=tmp_path/'rows.jsonl';p.write_text(json.dumps(good)+'\n'+json.dumps(bad)+'\n')
 nv.cmd_summary(SimpleNamespace(rows=[p]));out=capsys.readouterr().out
 assert 'both poles (primary): 2 submitted poles; 1 valid; pass 0.50' in out
 assert 'negative pole: 1 submitted poles; 0 valid; pass 0.00' in out
 assert 'both pass 0.00' in out
@pytest.mark.parametrize('activation',['gelu','relu',None])
def test_architecture_rejects_non_silu_gate_semantics(activation):
 model=SimpleNamespace(config=SimpleNamespace(hidden_act=activation),model=SimpleNamespace(layers=[]))
 with pytest.raises(ValueError,match='SiLU'):nv._swiglu_architecture(model)
def test_architecture_requires_matching_actual_gate_up_down_widths():
 mlp=SimpleNamespace(gate_proj=SimpleNamespace(out_features=4),up_proj=SimpleNamespace(out_features=4),down_proj=SimpleNamespace(in_features=4))
 model=SimpleNamespace(config=SimpleNamespace(hidden_act='silu'),model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
 assert nv._swiglu_architecture(model)
 mlp.up_proj.out_features=5
 with pytest.raises(ValueError,match='dimensions'):nv._swiglu_architecture(model)
def tiny_capture_target():
 import torch
 class Batch(dict):
  def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
 class Tokenizer:
  def __call__(self,texts,**kw):
   seqs={'fixture_a':[0,1,2,4],'fixture_zero':[0,2]};xs=[seqs[t] for t in texts];width=max(map(len,xs))
   return Batch(input_ids=torch.tensor([x+[0]*(width-len(x)) for x in xs]),attention_mask=torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in xs]))
 class MLP(torch.nn.Module):
  def __init__(self):
   super().__init__();self.gate_proj=torch.nn.Linear(2,2);self.up_proj=torch.nn.Linear(2,2);self.down_proj=torch.nn.Linear(2,2)
   with torch.no_grad():
    self.gate_proj.weight.copy_(torch.eye(2));self.gate_proj.bias.copy_(torch.tensor([-2.,0.]));self.up_proj.weight.zero_();self.up_proj.bias.fill_(1.)
   self.seen=[]
  def forward(self,x):
   g=self.gate_proj(x);u=self.up_proj(x);a=torch.nn.functional.silu(g)*u;self.seen.append(tuple(z.detach().clone() for z in (g,u,a)));return self.down_proj(a)
 class Body(torch.nn.Module):
  def __init__(self):
   super().__init__();layer=torch.nn.Module();layer.mlp=MLP();self.layers=torch.nn.ModuleList([layer]);self.calls=[];self.explode=False
  def forward(self,input_ids,attention_mask,use_cache):
   self.calls.append(use_cache);x=torch.stack((input_ids.float(),torch.ones_like(input_ids).float()),-1)
   self.layers[0].mlp(x)
   if self.explode:raise RuntimeError('synthetic body failure')
 class LM(torch.nn.Module):
  def __init__(self):super().__init__();self.model=Body()
  def forward(self,*a,**kw):raise AssertionError('unused LM head must never execute')
 tg=object.__new__(nv.Target);tg.torch=torch;tg.dev=torch.device('cpu');tg.m=LM();tg.layers=tg.m.model.layers;tg.tok=Tokenizer();return tg

def test_poles_body_only_exact_capture_zero_regimes_and_peak_positions():
 import torch
 tg=tiny_capture_target();rows=tg.poles(['fixture_a','fixture_zero'],0,0)
 assert tg.m.model.calls==[False]
 g,u,a=tg.layers[0].mlp.seen[0]
 expected_g=torch.tensor([[[-2.,1.],[-1.,1.],[0.,1.],[2.,1.]],[[-2.,1.],[0.,1.],[-2.,1.],[-2.,1.]]])
 assert torch.equal(g,expected_g) and torch.equal(u,torch.ones_like(u)) and torch.equal(a,torch.nn.functional.silu(expected_g))
 assert rows[0]['position_pos']==3 and rows[0]['position_neg']==1
 assert rows[0]['regime_pos']=='++' and rows[0]['regime_neg']=='-+'
 assert rows[1]['position_pos']==rows[1]['position_neg']==1
 assert rows[1]['regime_pos']==rows[1]['regime_neg']=='0+'
 for mod in (tg.layers[0].mlp.gate_proj,tg.layers[0].mlp.up_proj,tg.layers[0].mlp.down_proj):assert not mod._forward_hooks and not mod._forward_pre_hooks

def test_poles_removes_capture_hooks_on_body_failure():
 tg=tiny_capture_target();tg.m.model.explode=True
 with pytest.raises(RuntimeError,match='synthetic body failure'):tg.poles(['fixture_a'],0,0)
 for mod in (tg.layers[0].mlp.gate_proj,tg.layers[0].mlp.up_proj,tg.layers[0].mlp.down_proj):assert not mod._forward_hooks and not mod._forward_pre_hooks
