import copy
import hashlib
import json
from types import SimpleNamespace
import pytest
import torch
import components
import goal_eval
import native
import qa_env
import teacher
from test_goal_eval import tiny_native


def fixture(monkeypatch,broken=False):
    nat=tiny_native(seed=0,layers=1,d=2,heads=1,vocab=3,C=2)
    ids,targets=[0,1],[1]
    vocab=['a','b','c']
    monkeypatch.setattr(components,'token_strings',lambda model,xs:[vocab[x] for x in xs])
    nodes=[(n,t,c) for n in nat.names for t in range(2) for c in range(nat.C[n])]
    parents,out=native.all_edges(nodes,targets)
    g=teacher.with_tokens(native.Graph(nodes,parents=parents,out=out),targets)
    if broken:
        g=teacher.with_tokens(native.Graph([],parents={},out=[]),targets)
    g,_=goal_eval.circuit(goal_eval.to_json(g),targets,nat.C,len(ids))
    answer=goal_eval.to_json(g)
    circuit=components.circuit_text(components.render(answer))
    top=int(nat.reference([ids],targets)[0,-1].argmax())
    ordering=[top]+[i for i in range(3) if i!=top]
    task={'id':'fixture','model':'vpd4l','prompts':[{'token_ids':ids,'target_positions':targets,
        'model_top':[[[vocab[i],p] for i,p in zip(ordering,[.6,.3,.1])]]}]}
    binding={'schema':1,'task':'fixture','model_identifier':'vpd4l','target_positions':targets,
        'circuit_sha256':hashlib.sha256(circuit.encode()).hexdigest(),
        'answer_sha256':components._fact_digest(answer),
        'input_sha256':components._fact_digest({'token_ids':ids,'target_positions':targets}),
        'text_sha256':hashlib.sha256(b'ab').hexdigest(),'semantics':'c','execution_verified':False}
    facts={'task':'fixture','model':'vpd4l','text':'ab','target':1,'top':task['prompts'][0]['model_top'][0],
        'answer':answer,'circuit':circuit,'provided_circuit_binding':binding}
    row={'task':'fixture','question':"old instruction\ntokens: 0:'a' 1:'b'\nQuestion 1: OLD_QUESTION_SENTINEL",
        'qa_env':[{'kind':'token','ask':'change','position':0,'new':2,'secret':'OUTCOME_SENTINEL'},
                  {'kind':'token','ask':'kl','position':0,'new':2},
                  {'kind':'remove','ask':'change','node':['h.0.mlp.c_fc',0,0]}]}
    return goal_eval.GoalEval(nat,semantics='c'),task,row,facts,vocab


@pytest.mark.parametrize('broken',[False,True])
def test_real_paired_row_target_native_parity_and_no_outcome_prompt(monkeypatch,broken):
    ge,task,row,facts,vocab=fixture(monkeypatch,broken)
    before=copy.deepcopy(row)
    result=qa_env.paired_row(ge,task,row,facts,fidelity_bits=1e-6,strings=vocab,assets=qa_env.paired_assets(ge.nat))
    assert row==before and len(result['qa_env'])==2
    meta=result['paired_token_qa']
    assert meta['excluded_non_token_questions']==1 and len(meta['executed_model_content_sha256'])==64
    assert facts['circuit'] in result['question']
    assert 'OUTCOME_SENTINEL' not in result['question'] and 'OLD_QUESTION_SENTINEL' not in result['question']
    assert 'failed_fidelity' not in result['question'] and 'target_model_label' not in result['question']
    p=ge.nat.reference([[0,1]],[1])[0,-1].double().log_softmax(-1)
    q=ge.nat.reference([[2,1]],[1])[0,-1].double().log_softmax(-1)
    full=result['qa_env'][0]
    assert full['top']==int(q.argmax())
    assert full['bits']==pytest.approx(float(q[p.argmax()]-p[p.argmax()])/__import__('math').log(2),abs=1e-6)
    assert full['kl']==pytest.approx(float((p.exp()*(p-q)).sum())/__import__('math').log(2),abs=1e-6)
    assert all(x['fidelity']['passed'] is (not broken) for x in meta['pairs'])
    if broken:assert all(x['control_status']=='failed_fidelity' for x in meta['pairs'])


def test_changed_model_assets_change_receipt(monkeypatch):
    ge,*_=fixture(monkeypatch)
    before=qa_env.paired_assets(ge.nat)
    with torch.no_grad():ge.nat.target.wte[0,0]+=1
    assert qa_env.paired_assets(ge.nat)!=before


def test_bad_source_binding_fails_before_execution(monkeypatch):
    ge,task,row,facts,vocab=fixture(monkeypatch)
    facts['provided_circuit_binding']['input_sha256']='0'*64
    monkeypatch.setattr(ge.nat,'reference',lambda *a:pytest.fail('invalid binding reached model'))
    with pytest.raises(ValueError,match='actual source task'):
        qa_env.paired_row(ge,task,row,facts,fidelity_bits=.001,strings=vocab,assets='fixture')


@pytest.mark.parametrize('task_id',['text9100','text9149','text09100','../text6008','/text6008','text6008.json','text1/2',None])
def test_cli_bad_or_sealed_task_ids_fail_before_native_or_task_io(tmp_path,monkeypatch,task_id):
    rows=tmp_path/'rows.jsonl';facts=tmp_path/'facts.jsonl'
    rows.write_text(json.dumps({'task':task_id})+'\n')
    facts.write_text(json.dumps({'task':task_id})+'\n')
    monkeypatch.setattr(native,'Native',lambda *a:pytest.fail('invalid task ID reached model initialization'))
    args=SimpleNamespace(rows=rows,facts=facts,out=tmp_path/'out.jsonl',texts_root=tmp_path/'absent',fidelity_bits=.001)
    with pytest.raises(ValueError):qa_env.paired_cmd(args)
    assert not args.out.exists()
