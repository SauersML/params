import copy
import json
from pathlib import Path
import pytest
import causal_qa as qa


TARGET={'model':'synthetic-no-model','revision':'fixture','tokenizer_sha256':'fixture','adapter_sha256':'none'}


class Tokenizer:
    def encode(self,text,**kw):return list(text.encode())
    def apply_chat_template(self,messages,**kw):return '<user>'+messages[0]['content']+'</user><assistant>'


def case():
    return {'reference':'original_target_model','frozen_target_identity':TARGET,'case_id':'synthetic-not-eval','clean_ids':[1,2,3], 'corrupt_ids':[1,4,3],
        'answer_a_ids':[8,9], 'answer_b_ids':[10,11,12],
        'target_units':[[0,1,2],[1,2,3]], 'control_units':[[0,1,4],[1,2,5]],
        'clean_native_values':[{'unit':u,'value':v} for u,v in
             zip([[0,1,2],[1,2,3],[0,1,4],[1,2,5]],[-2.5,0.,.01,3.])],
        'interventions':{k:{'delta_bits':v} for k,v in zip(qa.EDITS,[-2.,3.,1.,-1.])}}


def prepare(tok=None):
    c=case();baseline={'case_sha256':qa.digest(qa.native_public(c)),
        'predictions':dict.fromkeys(qa.EDITS,.5),'provenance':'synthetic independently frozen fixture'}
    return qa.prepare_case(c,tok or Tokenizer(),reader_spec={'model':'synthetic-no-model',
        'revision':'fixture','tokenizer_sha256':'synthetic','adapter_sha256':'none'},mismatch_values=[.01,3.,-2.5,0.],
        mismatch_binding='frozen-target-control-exchange-fixture',simple_baseline=baseline,target_spec=TARGET,
        mismatch_units=c['control_units']+c['target_units'])


def response(arm='true', effects=None, finish='stop'):
    return {'request_id':'synthetic-not-eval|'+arm,'finish':finish,'token_ids':[1,2],
        'completion':json.dumps({'effects':effects or dict.fromkeys(qa.EDITS,0.),
        'explanation':'Zeroing L0:N2@P1 lowers A versus B; restoration increases A versus B. Test this joint intervention.'})}


def test_exact_slot_lengths_positions_metadata_no_gold_and_full_sequences():
    requests,m=prepare();positions={p for span in m['evidence_positions'] for p in span}
    assert len({len(r['token_ids']) for r in requests})==1
    for r in requests:
        assert all(t==requests[0]['token_ids'][i] for i,t in enumerate(r['token_ids']) if i not in positions)
        assert r['tokens_sha256']==qa.digest(r['token_ids'])
    assert requests[0]['token_ids']!=requests[1]['token_ids']!=requests[2]['token_ids']
    assert m['case']['answer_a_ids']==[8,9] and m['case']['answer_b_ids']==[10,11,12]
    assert 'interventions' not in m['case']
    c=case();c['interventions']='DO NOT READ GOLD';c['clean']='DO NOT READ GOLD';c['corrupt']='DO NOT READ GOLD'
    same,_=qa.prepare_case(c,Tokenizer(),reader_spec=m['reader'],mismatch_values=[.01,3.,-2.5,0.],mismatch_binding='fixture',target_spec=TARGET,mismatch_units=c['control_units']+c['target_units'])
    assert [r['token_ids'] for r in same]==[r['token_ids'] for r in requests]


@pytest.mark.parametrize('bad',[None,True,float('nan'),float('inf'),'1'])
def test_strict_numeric_invalids(bad):
    r=response();obj=json.loads(r['completion']);obj['effects']['target_zero']=bad;r['completion']=json.dumps(obj)
    assert qa.prediction(r,qa.native_public(case()))[1]=='malformed'


def test_unfinished_duplicate_schema_and_unnamed_english_invalid():
    c=qa.native_public(case())
    assert qa.prediction(response(finish='length'),c)[1]=='unfinished'
    r=response();r['completion']=r['completion'].replace('"effects":','"effects":{},"effects":',1)
    assert qa.prediction(r,c)[1]=='malformed'
    r=response();obj=json.loads(r['completion']);obj['explanation']='It changes.';r['completion']=json.dumps(obj)
    assert qa.prediction(r,c)[1]=='malformed'


def test_commit_before_gold_missing_penalty_pairing_and_raw_english(tmp_path):
    requests,m=prepare();truth={k:v['delta_bits'] for k,v in case()['interventions'].items()}
    commit=tmp_path/'committed.json'
    receipt=qa.commit_responses(commit,requests,[response(effects=truth),response('absent')],manifest=m)
    assert not (tmp_path/'gold.json').exists()
    (tmp_path/'gold.json').write_text(json.dumps(case()))
    r=qa.evaluate(commit,receipt=receipt,manifest=m,gold_path=tmp_path/'gold.json')
    assert r['arms']['true']['error_bits']==0.
    assert r['arms']['absent']['error_bits']==r['unchanged_error_bits']==1.75
    assert r['arms']['mismatch']['error_bits']==60.
    assert r['paired_absent_minus_true_bits']==1.75
    assert r['simple_baseline_error_bits']==1.75
    assert 'L0:N2@P1' in r['arms']['true']['raw_completion']
    assert r['english_faithfulness'].startswith('unscored')
    with pytest.raises(FileExistsError):qa.commit_responses(commit,requests,[],manifest=m)


def test_commit_tampering_fails_before_any_gold_read(tmp_path):
    requests,m=prepare();p=tmp_path/'commit';receipt=qa.commit_responses(p,requests,[],manifest=m)
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='commitment changed'):
        qa.evaluate(p,receipt=receipt,manifest=m,gold_path=tmp_path/'NEVER_EXISTS')


def test_changed_manifest_before_gold_and_duplicate_response_rejected(tmp_path):
    requests,m=prepare();p=tmp_path/'commit';receipt=qa.commit_responses(p,requests,[],manifest=m)
    changed=copy.deepcopy(m);changed['simple_baseline']['predictions']['target_zero']=123.
    with pytest.raises(ValueError,match='binding'):
        qa.evaluate(p,receipt=receipt,manifest=changed,gold_path=tmp_path/'NEVER_EXISTS')
    with pytest.raises(ValueError,match='duplicate'):
        qa.commit_responses(tmp_path/'other',requests,[response(),response()],manifest=m)


def test_clipped_error_has_fixed_worst_invalid_penalty(tmp_path):
    requests,m=prepare();p=tmp_path/'commit'
    receipt=qa.commit_responses(p,requests,[response(effects=dict.fromkeys(qa.EDITS,1e10))],manifest=m)
    gold=tmp_path/'gold';gold.write_text(json.dumps(case()))
    r=qa.evaluate(p,receipt=receipt,manifest=m,gold_path=gold)
    assert r['arms']['true']['error_bits']==r['arms']['absent']['error_bits']==60.


def test_no_native_size_cap_and_degenerate_mismatch_rejected():
    c=case();c['target_units']=[[0,1,i] for i in range(300)]
    c['control_units']=[[0,1,i] for i in range(300,600)]
    c['clean_native_values']=[{'unit':u,'value':float(i)} for i,u in enumerate(c['target_units']+c['control_units'])]
    kwargs=dict(reader_spec={'model':'test','revision':'test','tokenizer_sha256':'test','adapter_sha256':'none'},mismatch_binding='fixture',target_spec=TARGET,mismatch_units=c['control_units']+c['target_units'])
    requests,m=qa.prepare_case(c,Tokenizer(),mismatch_values=list(reversed(range(600))),**kwargs)
    assert len(m['evidence_positions'])==600
    with pytest.raises(ValueError,match='degenerate'):
        qa.prepare_case(c,Tokenizer(),mismatch_values=list(range(600)),**kwargs)


def test_changed_clean_observations_rejected(tmp_path):
    requests,m=prepare();p=tmp_path/'commit';receipt=qa.commit_responses(p,requests,[],manifest=m)
    gold=case();gold['clean_native_values'][0]['value']=99.
    path=tmp_path/'gold';path.write_text(json.dumps(gold))
    with pytest.raises(ValueError,match='observations changed'):
        qa.evaluate(p,receipt=receipt,manifest=m,gold_path=path)


def mechanisms():
    return [{'unit':u,'facts':[{'kind':'atlas_label','text':'Synthetic label only, no causal certificate.',
        'source_sha256':'a'*64,'split':'discovery','source_status':'unverified_label',
        'frozen_before_eval':True,'evidence_model_identity':TARGET,'claim_status':'hypothesis'}]} for u in case()['target_units']+case()['control_units']]


def test_discovery_mechanism_slots_match_all_arms_and_source_statuses():
    c=case();truth=mechanisms();wrong=copy.deepcopy(truth)
    for r in wrong:r['facts'][0]['text']='Mismatched synthetic discovery record with longer text.'
    kwargs=dict(reader_spec={'model':'test','revision':'test','tokenizer_sha256':'test','adapter_sha256':'none'},target_spec=TARGET,
        mismatch_values=[.01,3.,-2.5,0.],mismatch_units=c['control_units']+c['target_units'],
        mismatch_binding='fixture',mechanism_evidence=truth,mismatch_mechanism_evidence=wrong)
    requests,m=qa.prepare_case(c,Tokenizer(),**kwargs)
    editable={p for row in m['evidence_positions'] for p in row}
    assert len(m['evidence_positions'])==8
    assert len({len(r['token_ids']) for r in requests})==1
    assert all(all(t==requests[0]['token_ids'][i] for i,t in enumerate(r['token_ids']) if i not in editable) for r in requests)
    assert m['mechanism_evidence_status'].startswith('discovery-only')
    assert m['mechanism_evidence_sha256']!=m['mismatch_mechanism_evidence_sha256']
    kwargs['mechanism_evidence'][0]['facts'][0]['split']='blind_eval'
    with pytest.raises(ValueError,match='discovery-only'):qa.prepare_case(c,Tokenizer(),**kwargs)


def test_gold_wrong_target_rejected(tmp_path):
    requests,m=prepare();p=tmp_path/'commit';receipt=qa.commit_responses(p,requests,[],manifest=m)
    gold=case();gold['frozen_target_identity']=dict(TARGET,adapter_sha256='other')
    path=tmp_path/'gold';path.write_text(json.dumps(gold))
    with pytest.raises(ValueError,match='target identity'):
        qa.evaluate(p,receipt=receipt,manifest=m,gold_path=path)


def test_base_atlas_transfer_is_only_hypothesis():
    rows=mechanisms();rows[0]['facts'][0]['evidence_model_identity']=dict(TARGET,adapter_sha256='base')
    rows[0]['facts'][0]['claim_status']='measured'
    with pytest.raises(ValueError,match='discovery-only'):
        qa.validate_mechanisms(rows,case()['target_units']+case()['control_units'],TARGET)
    rows[0]['facts'][0]['claim_status']='hypothesis'
    qa.validate_mechanisms(rows,case()['target_units']+case()['control_units'],TARGET)


def test_collector_reader_commit_then_private_measurement(tmp_path):
    """Exercise the real two-stage collector/reader boundary on a random model.

    Scripted zero predictions are committed before edited forwards. This validates
    protocol wiring only; it supplies no evidence of an oracle's predictive skill.
    """
    import sys
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from native_prospective import Collector
    from neuron_builder import Neurons
    torch.manual_seed(19)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8)).eval()
    collector = Collector(Neurons(model, None), 12, TARGET)
    args = qa.native_public(case())
    public = collector.prepare_public(**args)
    assert collector.forward_examples == 2
    assert 'interventions' not in public and 'clean' not in public
    observed = public['clean_native_values']
    order = [2, 3, 0, 1]
    requests, manifest = qa.prepare_case(public, Tokenizer(),
        reader_spec=TARGET, target_spec=TARGET,
        mismatch_values=[observed[i]['value'] for i in order],
        mismatch_units=[observed[i]['unit'] for i in order],
        mismatch_binding='synthetic control exchange before private measurement')
    commit = tmp_path/'predictions.json'
    receipt = qa.commit_responses(commit, requests,
        [response(arm) for arm in qa.ARMS], manifest=manifest)
    assert collector.forward_examples == 2 and commit.exists()
    physical = collector.collect(**args)
    assert collector.forward_examples == 12
    assert physical['forward_examples'] == 10
    gold = tmp_path/'gold.json'
    gold.write_text(json.dumps(physical))
    result = qa.evaluate(commit, receipt=receipt, manifest=manifest, gold_path=gold)
    expected = sum(min(60., abs(physical['interventions'][e]['delta_bits']))
                   for e in qa.EDITS)/len(qa.EDITS)
    assert result['unchanged_error_bits'] == pytest.approx(expected)
    for arm in qa.ARMS:
        assert result['arms'][arm]['failure'] is None
        assert result['arms'][arm]['error_bits'] == pytest.approx(expected)
    assert result['english_faithfulness'].startswith('unscored')


def sampler_fixture(outputs=None):
    import torch
    from types import SimpleNamespace
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(512,2);self.calls=[]
            self.config=SimpleNamespace(max_position_embeddings=20000)
            self.generation_config=SimpleNamespace(eos_token_id=[511])
        def get_input_embeddings(self):return self.embedding
        def generate(self,**kwargs):
            self.calls.append(kwargs);index=len(self.calls)-1
            value=([7,511] if outputs is None else outputs[index])
            if isinstance(value,Exception):raise value
            return torch.cat([kwargs['input_ids'],torch.tensor([value],dtype=torch.long)],dim=-1)
    class DecodeOnly:
        eos_token_id=511;pad_token_id=0
        def encode(self,*args,**kwargs):raise AssertionError('input re-encoded')
        def apply_chat_template(self,*args,**kwargs):raise AssertionError('input re-rendered')
        def decode(self,ids,**kwargs):return 'RAW OUTPUT '+str(ids)
    requests,m=prepare()
    return Reader().eval(),DecodeOnly(),requests,m


def test_exact_sampler_submits_frozen_ids_once_and_preserves_raw_output(tmp_path):
    import torch
    model,tok,requests,m=sampler_fixture([[7,511],[8,9,10],[11,511]])
    rows=qa.sample_exact(model,tok,requests,m,reader_identity=m['reader'],max_new_tokens=3,context_cap=20000)
    assert len(model.calls)==len(rows)==3
    for request,call,row in zip(requests,model.calls,rows):
        assert call['input_ids'].tolist()==[request['token_ids']]
        assert torch.equal(call['attention_mask'],torch.ones_like(call['input_ids']))
        assert call['do_sample'] is False and call['num_beams']==1 and call['num_return_sequences']==1
        assert call['max_new_tokens']==3
        assert row['request_tokens_sha256']==request['tokens_sha256']
        assert row['prompt_tokens']==len(request['token_ids'])
        assert row['generated_tokens']==len(row['token_ids'])
    assert [r['finish'] for r in rows]==['stop','length','stop']
    assert rows[0]['token_ids']==[7,511] and rows[0]['completion']=='RAW OUTPUT [7, 511]'
    qa.commit_responses(tmp_path/'commit',requests,rows,manifest=m)


@pytest.mark.parametrize('failure',['count','hash','identity','missing','context','vocab','chat','train'])
def test_sampler_preflights_every_row_before_generation(failure):
    model,tok,requests,m=sampler_fixture();identity=m['reader'];cap=20000
    if failure=='count':requests[-1]['prompt_tokens']+=1
    if failure=='hash':requests[-1]['tokens_sha256']='bad'
    if failure=='identity':identity=dict(identity,adapter_sha256='other')
    if failure=='missing':requests.pop()
    if failure=='context':cap=requests[0]['prompt_tokens']+2
    if failure in ('vocab','chat'):
        requests[-1]['token_ids'][0]=999 if failure=='vocab' else 33
        requests[-1]['tokens_sha256']=qa.digest(requests[-1]['token_ids'])
    if failure=='train':model.embedding.train()
    if failure not in ('hash','identity','train'):m['requests_sha256']=qa.digest(requests)
    with pytest.raises(ValueError):
        qa.sample_exact(model,tok,requests,m,reader_identity=identity,max_new_tokens=3,context_cap=cap)
    assert not model.calls


def test_sampler_uses_model_context_cap_without_truncation():
    model,tok,requests,m=sampler_fixture()
    model.config.max_position_embeddings=requests[0]['prompt_tokens']+1
    with pytest.raises(ValueError,match='context cap'):
        qa.sample_exact(model,tok,requests,m,reader_identity=m['reader'],max_new_tokens=3,context_cap=20000)
    assert not model.calls


def test_sampler_failure_short_finish_and_length_remain_all_rows_without_retry():
    model,tok,requests,m=sampler_fixture([RuntimeError('synthetic failure'),[7],[8,9,10]])
    rows=qa.sample_exact(model,tok,requests,m,reader_identity=m['reader'],max_new_tokens=3,context_cap=20000)
    assert len(model.calls)==3 and {r['request_id'] for r in rows}=={r['request_id'] for r in requests}
    assert [r['finish'] for r in rows]==['error','error','length']
    assert rows[0]['token_ids']==[] and 'synthetic failure' in rows[0]['error']
    assert rows[1]['token_ids']==[7] and rows[1]['completion']=='RAW OUTPUT [7]'
    assert all(qa.prediction(r,m['case'])[0] is None for r in rows)


def test_sampler_actual_tiny_random_hf_generate_without_load_or_download():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    model,tok,requests,m=sampler_fixture()
    torch.manual_seed(11)
    tiny=Qwen3ForCausalLM(Qwen3Config(vocab_size=512,hidden_size=8,intermediate_size=16,
        num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=4096,eos_token_id=511,pad_token_id=0)).eval()
    rows=qa.sample_exact(tiny,tok,requests,m,reader_identity=m['reader'],max_new_tokens=2,context_cap=4096)
    assert len(rows)==3
    assert all(r['finish'] in ('stop','length') and 1<=r['generated_tokens']<=2 for r in rows)
