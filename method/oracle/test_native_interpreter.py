"""Bounded native graph interpreter inputs: synthetic source arrays, no forwards."""
import ast,copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import components as c
import connect


def source(monkeypatch):
    monkeypatch.setattr(c,'token_strings',lambda model,ids:['x','y'][:len(ids)])
    h=('head',0,0,1);n=('neuron',0,0,3);r=('head',1,1,0);o=('neuron',1,1,2)
    graph=connect.NativeGraph([h,n,r,o],{n:[(h,'in')],r:[(h,'k'),(n,'v')],o:[(r,'in')]},[o])
    means={'heads':np.zeros((2,2,2),np.float32),'neurons':np.zeros((2,4),np.float32)}
    task={'id':'source','model':'vpd4l','prompts':[{'token_ids':[7,9],'target_positions':[1],'model_top':[[['a',.8],['b',.2]]]}]}
    return task,graph,means,{'reference_texts_sha256':'a'*64,'producer':'frozen synthetic mean fixture'}


def test_native_payload_exact_topology_scope_and_no_vpd_ids(monkeypatch):
    task,g,m,p=source(monkeypatch);before=copy.deepcopy(g.to_json())
    f=c.native_teacher_facts(task,g,m,p);saved=copy.deepcopy(f)
    text=c.writer_prompt(f,provided_circuit=True,source_task=task)
    assert f['circuit'] in text and 'NOT VPD' in text and 'FULL MODEL' in text
    assert 'NOT executed' in text and 'frozen reference-text mean' in text
    assert 'L0.fc#' not in text and 'rank-one subcomponents' not in text
    assert 'answer' not in f  # existing explain fallback does not invoke VPD renderer
    assert f==saved and g.to_json()==before
    with pytest.raises(ValueError,match='explicit'):c.writer_prompt(f)


def test_slot_and_mean_changes_change_input(monkeypatch):
    task,g,m,p=source(monkeypatch)
    first=c.writer_prompt(c.native_teacher_facts(task,g,m,p),provided_circuit=True)
    other=copy.deepcopy(g);other.parents[('head',1,1,0)][0]=(g.units[0],'v')
    assert c.writer_prompt(c.native_teacher_facts(task,other,m,p),provided_circuit=True)!=first
    m['heads'][0,0,0]=.5
    assert c.writer_prompt(c.native_teacher_facts(task,g,m,p),provided_circuit=True)!=first


@pytest.mark.parametrize('field',['units','edges','out'])
def test_oversized_fails_before_serialization(monkeypatch,field):
    task,g,m,p=source(monkeypatch)
    if field=='units':g.units=g.units*33
    elif field=='edges':g.parents[g.units[1]]=g.parents[g.units[1]]*1025
    else:g.out=g.out*129
    g.to_json=lambda:pytest.fail('large graph must not be enumerated/copied')
    with pytest.raises(ValueError,match='already-small'):c.native_teacher_facts(task,g,m,p)


def test_wrong_edge_and_stale_binding_fail(monkeypatch):
    task,g,m,p=source(monkeypatch);bad=copy.deepcopy(g)
    bad.parents[('head',1,1,0)][0]=(g.units[0],'q')  # wrong source position for query
    with pytest.raises(ValueError,match='slot'):c.native_teacher_facts(task,bad,m,p)
    f=c.native_teacher_facts(task,g,m,p);f['provided_circuit_binding']['mean_provenance']['producer']='changed'
    with pytest.raises(ValueError,match='binding mismatch'):c.writer_prompt(f,provided_circuit=True)


def test_source_and_mean_guards(monkeypatch):
    task,g,m,p=source(monkeypatch)
    with pytest.raises(ValueError,match='provenance'):c.native_teacher_facts(task,g,m,{})
    m['heads'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):c.native_teacher_facts(task,g,m,p)


def test_prompt_byte_bound_no_truncation(monkeypatch):
    task,g,m,p=source(monkeypatch)
    f=c.native_teacher_facts(task,g,m,p)
    f['top']=[['x'*c.NATIVE_INTERPRETER_MAX_BYTES,1.]]
    with pytest.raises(ValueError,match='byte bound'):c.writer_prompt(f,provided_circuit=True)


@pytest.mark.parametrize('stale',[False,True])
def test_existing_explain_preflight_accepts_native_without_trainer_changes(monkeypatch,tmp_path,stale):
    task,g,m,p=source(monkeypatch);f=c.native_teacher_facts(task,g,m,p)
    spec=importlib.util.spec_from_file_location('native_supplied_train',HERE/'rl/tinker_train.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules,'tinker',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'tinker_cookbook',SimpleNamespace(renderers=SimpleNamespace()))
    class StopBeforeService(Exception):pass
    monkeypatch.setattr(mod,'service_client',lambda:(_ for _ in ()).throw(StopBeforeService()))
    src=tmp_path/'facts.jsonl';src.write_text(json.dumps(f)+'\n')
    if stale:task['prompts'][0]['token_ids']=[8,9]
    (tmp_path/'source.json').write_text(json.dumps(task))
    with pytest.raises(ValueError,match='source task') if stale else pytest.raises(StopBeforeService):
        mod.explain(SimpleNamespace(facts=src,out=tmp_path/'out',texts_dir=tmp_path,provided_circuit=True))


def test_existing_reader_fallback_keeps_background_without_fullmodel_answers(monkeypatch):
    task,g,m,p=source(monkeypatch);f=c.native_teacher_facts(task,g,m,p)
    tree=ast.parse((HERE/'rl/tinker_train.py').read_text())
    explain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='explain')
    answer=next(n for n in explain.body if isinstance(n,ast.FunctionDef) and n.name=='answer')
    scope={'components':c};exec(compile(ast.Module(body=[answer],type_ignores=[]),'existing_reader_fallback','exec'),scope)
    text=scope['answer'](f,task,'An unverified interpretation.',None)
    assert f['circuit'] in text
    assert 'frozen reference-text means' in text
    assert f['provided_circuit_binding']['mean_fp32_content_sha256'] in text
    assert f['provided_circuit_binding']['mean_provenance_sha256'] in text
    assert json.dumps(f['top']) not in text
