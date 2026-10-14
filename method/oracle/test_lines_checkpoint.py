"""No model construction: checkpoint scopes use synthetic tensors and contexts."""
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lines

@pytest.fixture
def context(tmp_path):
 task=tmp_path/'text6008.json';task.write_text('{"token_ids":[1,2]}')
 teacher=tmp_path/'teacher.json';teacher.write_text('{"nodes":[["s",0,0]]}')
 site=SimpleNamespace(W=torch.ones(2,2),U=torch.eye(2),V=torch.eye(2),delta_T=torch.zeros(2,2))
 target=SimpleNamespace(cfg={'eps':1e-6},eps=1e-6,n_layer=1,n_head=1,n_tokens=2,lm=True,state_dict=lambda:{'embedding':torch.ones(2,2)},site=lambda n:site)
 ge=SimpleNamespace(nat=SimpleNamespace(target=target,names=['s'],C={'s':2},dev='cpu'),semantics='c',donor_pools={},probes={},hooks={'E1':None})
 ge.far_pool=lambda split:[('synthetic', [0,1])]
 tx=SimpleNamespace(ids=[1,2],targets=[1],ref=torch.tensor([[[-.5,-1.]]]),vpd={'s':torch.tensor([[1,0],[0,1]])},vpd_order=[('s',0,0)],seed=0)
 return ge,tx,task,teacher,{}, {'patterns':torch.ones(2)}, ('line',),1

def test_exact_scope_preserves_every_historical_field(context):
 scope=lines.search_scope(*context)
 state={'scope':scope,'setup':{'rev':'reward'},'removed':['r'],'history':[['r']],'pending':[['r']],'frozen':['f'],'level':2,'done':True}
 assert lines.scoped_state(state,scope) is state
 assert lines.search_scope(*context)==scope
 for old in ({},dict(state,scope='different')):
  reset=lines.scoped_state(old,scope)
  assert reset=={'scope':scope,'level':0,'removed':[],'trace':[],'done':False}

@pytest.mark.parametrize('change',['task','teacher','pool','embedding','basis','eps','targets','vpd','levels','runtime','donors'])
def test_actual_context_changes_invalidate(context,change):
 old=lines.search_scope(*context);args=list(context);ge,tx,task,teacher,_,pool,levels,B=args
 if change=='task':task.write_text('same task ID, changed input')
 elif change=='teacher':teacher.write_text('different wired node universe')
 elif change=='pool':pool['patterns'][0]=2
 elif change=='embedding':ge.nat.target.state_dict=lambda:{'embedding':torch.zeros(2,2)}
 elif change=='basis':ge.nat.target.site('s').V[0,0]=3
 elif change=='eps':ge.nat.target.eps=1e-5
 elif change=='targets':tx.targets=[0]
 elif change=='vpd':tx.vpd['s'][0,0]=0
 elif change=='levels':args[6]=('sites','line')
 elif change=='runtime':ge.semantics='n'
 elif change=='donors':ge.far_pool=lambda split:[('synthetic',[1,0])]
 new=lines.search_scope(*args);assert old!=new
 assert lines.scoped_state({'scope':old,'removed':['r'],'history':[['r']],'done':True},new)['removed']==[]

def test_tensor_digest_handles_bfloat16_and_rejects_opaque():
 assert lines.scope_digest(torch.tensor([1],dtype=torch.bfloat16))!=lines.scope_digest(torch.tensor([2],dtype=torch.bfloat16))
 with pytest.raises(TypeError):lines.scope_digest(object())

def test_same_revision_history_is_not_reset():
 # Execute the production resume branch, without its model/search setup.
 import ast
 tree=ast.parse(Path(lines.__file__).read_text());one=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='one')
 branch=next(n for n in ast.walk(one) if isinstance(n,ast.If) and ast.unparse(n.test)=='history and (not same)')
 env={'history':[['r']],'same':True,'state':{'removed':['r']},'verify':lambda:pytest.fail('unexpected revalidation'),'pr':None}
 exec(compile(ast.Module(body=[branch],type_ignores=[]),'<resume>','exec'),env)
 assert env['state']['removed']==['r']

def test_reward_revision_resets_all_historical_state(context):
 base=lines.search_scope(*context)
 old=lines.scope_digest((base,'reward-old'))
 state={'scope':old,'removed':['r'],'history':[['r']],'frozen':['f'],'setup':{'floors':{'point':.9}},'done':True,'level':3}
 assert lines.scoped_state(state,lines.scope_digest((base,'reward-old'))) is state
 reset=lines.scoped_state(state,lines.scope_digest((base,'reward-new')))
 assert reset=={'scope':lines.scope_digest((base,'reward-new')),'level':0,'removed':[],'trace':[],'done':False}

def test_numpy_values_and_noncontiguous_probes_bind_content(context):
 import numpy as np
 a=np.arange(6,dtype=np.int32).reshape(2,3).T
 assert lines.scope_digest(a)==lines.scope_digest(a.copy())
 assert lines.scope_digest(np.int64(2))!=lines.scope_digest(np.int64(3))
 with pytest.raises(TypeError):lines.scope_digest(np.array([object()],dtype=object))
 ge=context[0];ge.probes['probe']=(a,np.float64(.5))
 old=lines.search_scope(*context);a[0,0]=99
 assert lines.search_scope(*context)!=old

def test_real_tiny_native_state_scope_with_synthetic_donors(tmp_path,monkeypatch):
 from test_goal_eval import tiny_native
 import goal_eval
 nat=tiny_native(near=.1)
 ge=goal_eval.GoalEval(nat,changes=4,chunk=2)
 monkeypatch.setattr(ge,'far_pool',lambda split:[('synthetic-own-tokenizer',[1,2,3])])
 tx=ge.text([1,2,3],[2],seed=0)
 task=tmp_path/'tiny.json';task.write_text('{"ids":[1,2,3]}')
 teacher=tmp_path/'teacher.json';teacher.write_text('{"nodes":[]}')
 args=(ge,tx,task,teacher,None,{'patterns':torch.ones(2)},('line',),2)
 first=lines.search_scope(*args);assert first==lines.search_scope(*args)
 with torch.no_grad():nat.target.site(nat.names[0]).V[0,0]+=.01
 assert first!=lines.search_scope(*args)


def test_set_and_list_have_distinct_scope_types():
    value = {"x"}
    assert lines.scope_digest(value) != lines.scope_digest([lines.scope_digest("x")])


def test_continuous_importance_change_invalidates_same_binary_mask(context):
    tx = context[1]
    tx.ci = {"s": torch.tensor([[.1, .2], [.3, .4]])}
    old = lines.search_scope(*context)
    tx.ci["s"][0, 0] = .9
    assert lines.search_scope(*context) != old
