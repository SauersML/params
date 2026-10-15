"""Synthetic prospective scoring regressions; no real labels, models or external data."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import g2
import cao

MODES=('correct','native_absent','joint_mismatch')
ARMS=('correct','joint_mismatch')

def packet(tmp_path):
    data=tmp_path/'synthetic_gold.jsonl';paths=[];gold=[]
    answers=['Unchanged.',"' token'","''","'\r'","'''","'\\n'"]
    for i,answer in enumerate(answers):
        gold.append({'cat':'native_edit','schema':'native_neuron_edit_v1','context_id':f'context{i//2}',
            'split':'eval','document_sha256':f'{i//2:064x}','ctx':[i//2,0],
            'blocks':[{'label':'Neuron A','unit':['neuron',0,i]}],'q':'Synthetic removal question',
            'a':answer,'measurement':{'original_top_id':0,'edited_top_id':0 if i==0 else 1}})
    data.write_text(''.join(json.dumps(r)+'\n' for r in gold));keys=[cao.native_row_key(r) for r in gold]
    for arm in ARMS:
        path=tmp_path/f'{arm}.jsonl';records=[]
        for mode in MODES:
            for k,e in zip(keys,gold):
                records.append({'training_arm':arm,'input_arm':mode,'row_key':k,'context_id':e['context_id'],
                    'raw_answer':e['a'],'truncated':False,
                    'input_receipt':{'row_key':k,'mode':mode,'prefill_tokens':17},
                    'generation_receipt':{'row_key':k,'limit_reached':False,'output_token_ids':[1,2],'generated_tokens':2}})
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        complete={'schema':'native_greedy_predictions_v1','complete':True,'eval_gold_fields_used':False,
            'data_sha256':g2._file_sha(data),'control_plan_sha256':'a'*64,
            'predictions_sha256':g2._file_sha(path),'expected_row_keys':keys,
            'training_protocol':{'schema':'controlled_native_training_v1','mode':arm,
                'schedule_sha256':'b'*64,'initial_adapter':{'adapter_model.safetensors':'c'*64},
                'donor_plan_sha256':'a'*64,'protected_eval_used_during_fit':False,
                'checkpoint_selection':'fixed final step only','steps':64}}
        Path(str(path)+'.complete.json').write_text(json.dumps(complete));paths.append(path)
    return SimpleNamespace(items=data,predictions=paths,out=tmp_path/'score.json'),gold


def mutate_prediction(args,change):
    path=args.predictions[0];rows=[json.loads(x) for x in path.read_text().splitlines()];change(rows)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));cp=Path(str(path)+'.complete.json')
    c=json.loads(cp.read_text());c['predictions_sha256']=g2._file_sha(path);cp.write_text(json.dumps(c))


def protect_gold_text(monkeypatch,args):
    original=Path.read_text
    def guarded(self,*a,**kw):
        if self==args.items:raise AssertionError('gold fields opened before full prediction validation')
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'read_text',guarded)


def test_two_model_three_mode_exact_answers_including_empty_and_cr(tmp_path):
    args,gold=packet(tmp_path);g2.score_native(args);r=json.loads(args.out.read_text())
    assert len(r['rows'])==36
    for arm in ARMS:
        for mode in MODES:
            s=r['summary'][arm][mode]
            assert s['n']==s['correct']==s['valid']==6 and s['accuracy']==1
            assert s['unchanged_baseline_accuracy']==pytest.approx(1/6)
    assert all(r['valid'] and r['correct'] for r in r['rows'])


def test_whole_answer_invalid_and_truncated_are_incorrect_without_dropping(tmp_path):
    args,_=packet(tmp_path)
    def change(rows):
        rows[0]['raw_answer']='Unchanged. because of a reason'
        rows[1]['raw_answer']="Reason first: ' token'"
        rows[2]['truncated']=True;rows[2]['generation_receipt']['limit_reached']=True
    mutate_prediction(args,change);g2.score_native(args);r=json.loads(args.out.read_text())
    s=r['summary']['correct']['correct']
    assert s['n']==6 and s['valid']==s['correct']==3
    assert len(r['rows'])==36


@pytest.mark.parametrize('bad',['missing','duplicate','training_arm','input_mode','input_row','generation_row','truncation'])
def test_bad_raw_bindings_or_incomplete_pair_reject_before_gold(tmp_path,monkeypatch,bad):
    args,_=packet(tmp_path)
    def change(rows):
        if bad=='missing':rows.pop()
        elif bad=='duplicate':rows[0]=rows[1]
        elif bad=='training_arm':rows[0]['training_arm']='joint_mismatch'
        elif bad=='input_mode':rows[0]['input_receipt']['mode']='native_absent'
        elif bad=='input_row':rows[0]['input_receipt']['row_key']='d'*64
        elif bad=='generation_row':rows[0]['generation_receipt']['row_key']='d'*64
        elif bad=='truncation':rows[0]['generation_receipt']['limit_reached']=True
    mutate_prediction(args,change);protect_gold_text(monkeypatch,args)
    with pytest.raises(ValueError,match='frozen'):g2.score_native(args)
    assert not args.out.exists()


def test_training_protocol_mismatch_rejects_before_gold(tmp_path,monkeypatch):
    args,_=packet(tmp_path);cp=Path(str(args.predictions[1])+'.complete.json');c=json.loads(cp.read_text())
    c['training_protocol']['schedule_sha256']='DIFFERENT';cp.write_text(json.dumps(c));protect_gold_text(monkeypatch,args)
    with pytest.raises(ValueError,match='protocols not matched'):g2.score_native(args)


def test_different_control_plan_rejects_before_gold(tmp_path,monkeypatch):
    args,_=packet(tmp_path);cp=Path(str(args.predictions[1])+'.complete.json');c=json.loads(cp.read_text())
    c['control_plan_sha256']='DIFFERENT';cp.write_text(json.dumps(c));protect_gold_text(monkeypatch,args)
    with pytest.raises(ValueError):g2.score_native(args)

def test_interrupted_training_checkpoint_rejected_before_oracle(tmp_path,monkeypatch):
    args,_=packet(tmp_path);snapshot=tmp_path/('a'*40);snapshot.mkdir()
    (snapshot/'tokenizer.json').write_bytes(b'synthetic tokenizer');(snapshot/'config.json').write_bytes(b'{}')
    binding={'model_revision':snapshot.name,'tokenizer_sha256':g2._file_sha(snapshot/'tokenizer.json'),'config_sha256':g2._file_sha(snapshot/'config.json')}
    donor=tmp_path/'controls.json';donor.write_text('{}')
    adapter=tmp_path/'fit'/'step_64';adapter.mkdir(parents=True)
    training={'mode':'correct','steps':64,'protected_eval_used_during_fit':False,
        'donor_plan_sha256':g2._file_sha(donor),'data_sha256':g2._file_sha(args.items),
        'base_revision':binding['model_revision'],'tokenizer_sha256':binding['tokenizer_sha256'],'config_sha256':binding['config_sha256']}
    (adapter.parent/'training_protocol.json').write_text(json.dumps(training))
    (adapter.parent/'training_complete.json').write_text(json.dumps({'steps':63,'protected_eval_used':False}))
    monkeypatch.setattr(cao,'_training_split',lambda rows:([],[]))
    monkeypatch.setattr(cao,'_native_training_receipt',lambda *a:binding)
    monkeypatch.setattr(cao,'Oracle',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('interrupted model must not load')))
    collect=SimpleNamespace(items=args.items,adapter=adapter,control_plan=donor,base_snapshot=snapshot,out=tmp_path/'raw.jsonl',device='cpu')
    with pytest.raises(ValueError,match='fixed-step matched'):g2.evaluate_native(collect,[])
    assert not collect.out.exists()
