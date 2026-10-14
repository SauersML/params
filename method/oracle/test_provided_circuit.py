"""Source-only supplied circuit interpreter inputs; no model/API execution."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import components as c
import native,goal_eval

def facts(monkeypatch,*,token=True,pair=False):
    fc=[('h.0.mlp.c_fc',0,j) for j in range(30)]
    down=('h.0.mlp.down_proj',0,0);q=('h.0.attn.q_proj',0,0);k=('h.0.attn.k_proj',0,0)
    g=native.Graph(fc+[down,q,k],parents={down:fc},out=[down])
    g.tok_positions=[0];g.tok_edges=[(0,fc[0])] if token else [];g.tok_out=[0]
    if pair:g.pairs={(0,0):[(q,k)]}
    monkeypatch.setattr(goal_eval,'circuit',lambda *a:(g,{}))
    monkeypatch.setattr(c,'token_strings',lambda model,ids:['token']*len(ids))
    ge=SimpleNamespace(semantics='c',nat=SimpleNamespace(C={},_vpd_order=lambda *a:fc+[down,q,k]))
    task={'id':'synthetic','model':'vpd4l','prompts':[{'token_ids':[7],'target_positions':[0],'model_top':[[['word',.8],['other',.2]]]}]}
    ev={'components':{},'events':[],'qa':[]}
    return c.teacher_facts(ge,task,ev,goal_eval.to_json(g))


def test_same_nodes_different_token_and_pair_topology_changes_input(monkeypatch):
    plain=facts(monkeypatch,token=False);token=facts(monkeypatch);pair=facts(monkeypatch,pair=True)
    assert plain['nodes']==token['nodes']==pair['nodes']
    assert plain['edges']==token['edges']==pair['edges']
    assert len({c.writer_prompt(f,provided_circuit=True) for f in (plain,token,pair)})==3


def test_complete_graph_not_truncated_and_payload_immutable(monkeypatch):
    f=facts(monkeypatch);before=copy.deepcopy(f)
    assert len(f['nodes'])==24 and f['nodes_total']==33
    prompt=c.writer_prompt(f,provided_circuit=True)
    assert f['circuit'] in prompt and 'L0.fc#29' in prompt
    assert 'FULL MODEL' in prompt and 'not construct' in prompt
    assert f==before


@pytest.mark.parametrize('with_variables',[False,True])
def test_default_prompt_byte_identical_to_original(monkeypatch,with_variables):
    f=facts(monkeypatch)
    if with_variables:
        f['variables']=[{'name':'value','inputs':[0],'group':[['h.0.mlp.c_fc',7,0]],'feeds':[]}]
        f['tokens']={'0':'token'}
    # Recorded from unmodified origin6a96fd1d writer_prompt on this exact fixture.
    expected={False:'1517fe91c33d9b917067b753350de1e9e3696100f5b97ff3dcd89b0797d05096',
              True:'b5d4dd91af2180e028be30b5645285d68c5842b9f17f2db340679655fba28a70'}
    assert hashlib.sha256(c.writer_prompt(f).encode()).hexdigest()==expected[with_variables]


@pytest.mark.parametrize('field',['circuit','answer','provided_circuit_binding'])
def test_missing_supplied_source_fails(monkeypatch,field):
    f=facts(monkeypatch);del f[field]
    with pytest.raises(ValueError,match='requires'):c.writer_prompt(f,provided_circuit=True)


def test_mutation_rejected(monkeypatch):
    f=facts(monkeypatch)
    for key,value in [('circuit',f['circuit']+'x'),('text','changed'),('answer',{}),('target',2),('model','other')]:
        bad=copy.deepcopy(f);bad[key]=value
        with pytest.raises(ValueError,match='binding mismatch'):c.writer_prompt(bad,provided_circuit=True)


def test_native_semantics_and_actual_source_identity(monkeypatch):
    f=facts(monkeypatch);f['provided_circuit_binding']['semantics']='n'
    task={'id':'synthetic','model':'vpd4l','prompts':[{'token_ids':[7],'target_positions':[0],'model_top':[f['top']]}]}
    assert 'EXACT PROVIDED CIRCUIT' in c.writer_prompt(f,provided_circuit=True,source_task=task)
    for field,value in [('token_ids',[8]),('target_positions',[1]),('model_top',[[['wrong',1.]]])]:
        wrong=copy.deepcopy(task);wrong['prompts'][0][field]=value
        with pytest.raises(ValueError,match='actual source task'):
            c.writer_prompt(f,provided_circuit=True,source_task=wrong)


def test_canonical_answer_consistency_not_only_separate_hashes(monkeypatch):
    f=facts(monkeypatch);f['answer']['edges']=[]
    f['provided_circuit_binding']['answer_sha256']=c._fact_digest(f['answer'])
    with pytest.raises(ValueError,match='canonical answer'):
        c.writer_prompt(f,provided_circuit=True)


def test_explain_flag_preflights_before_service(monkeypatch,tmp_path):
    train=HERE/'tinker_train.py'
    if not train.exists():train=HERE/'rl/tinker_train.py'
    spec=importlib.util.spec_from_file_location('supplied_train',train)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules,'tinker',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'tinker_cookbook',SimpleNamespace(renderers=SimpleNamespace()))
    monkeypatch.setattr(mod,'service_client',lambda:pytest.fail('Service forbidden before invalid binding rejection'))
    src=tmp_path/'facts.jsonl';src.write_text(json.dumps({'task':'bad'})+'\n')
    with pytest.raises(ValueError,match='requires'):
        mod.explain(SimpleNamespace(facts=src,out=tmp_path/'out',provided_circuit=True))
