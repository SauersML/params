import json
import math
import pytest
import score_reason_sensitivity as d

LONG=' '.join(['word']*41)
def row(text,finish='stop'):
    return dict(completion=text,finish_reason=finish)

def test_overlong_only_and_primary_unchanged():
    r=row(json.dumps(dict(delta_bits=2.5,reason=LONG)))
    assert d.primary.parse(r,{})[1]=='invalid_strict_json'
    assert d.parse_numeric_only(r,{})==(2.5,None,LONG)
    assert d.primary.parse(r,{})[1]=='invalid_strict_json'

@pytest.mark.parametrize('finish',['length','max_tokens','error',''])
def test_stops_remain_invalid(finish):
    r=row(json.dumps(dict(delta_bits=2,reason=LONG)),finish)
    assert d.parse_numeric_only(r,{})==d.primary.parse(r,{})
    assert d.parse_numeric_only(r,{})[1] is not None

@pytest.mark.parametrize('text',[
    '{"delta_bits":NaN,"reason":"LONG"}',
    '{"delta_bits":Infinity,"reason":"LONG"}',
    '{"delta_bits":1e999,"reason":"LONG"}',
    '{"delta_bits":true,"reason":"LONG"}',
    '{"delta_bits":"2","reason":"LONG"}',
    '{"delta_bits":2,"reason":null}',
    '{"delta_bits":2,"reason":[]}',
    '{"delta_bits":2,"reason":"LONG","extra":0}',
    '{"delta_bits":1,"delta_bits":2,"reason":"LONG"}',
    '{"delta_bits":2,"reason":"ok","reason":"LONG"}',
    '```json\n{"delta_bits":2,"reason":"LONG"}\n```',
    '{"delta_bits":2,"reason":"LONG"} trailing',
])
def test_other_invalids_stay_invalid(text):
    assert d.parse_numeric_only(row(text.replace('LONG',LONG)),{})[1]=='invalid_strict_json'

def test_short_unchanged():
    r=row('{"delta_bits":-2,"reason":"short"}')
    assert d.parse_numeric_only(r,{})==d.primary.parse(r,{})

def test_all_rows_cap_bootstrap_and_physics_retained():
    cases=[];mapping=[];responses=[];goldrows=[]
    def s(bits,delta=None):
        v=dict(a_logp_nats=bits*math.log(2),b_logp_nats=0.,contrast_bits=bits)
        if delta is not None:v['delta_bits']=delta
        return v
    for i in range(32):
        cid=f'c{i}';qid=f'q{i}'
        cases.append(dict(case_id=cid,group=f'g{i//8}',prefix_ids=[16],expected_truth_digit=4))
        mapping.append(dict(question_id=qid,case_id=cid))
        for arm in d.primary.ARMS:
            responses.append(dict(request_id=qid+'_'+arm,finish_reason='length' if i==0 else 'stop',completion=json.dumps(dict(delta_bits=1000 if arm=='true' else 1,reason=LONG))))
        goldrows.append(dict(case_id=cid,conditional_baseline=s(1),full_answer_baseline=s(1),effects=dict(target_zero=s(2,1)),baseline_metadata=dict(prefix_top_ids=[0],digit_top_id=0),expected_truth_digit=4))
    panel=dict(cases=cases,operations=['target_zero'])
    bundle=dict(panel=panel,manifest=dict(frozen_baselines={'zero':0},scope='synthetic'),mapping=mapping,responses=responses,raw={},commitment={})
    result=d.analyze(bundle,dict(status='complete',manifest=panel,rows=goldrows))
    assert len(result['predictions'])==96
    assert len(result['reason_only_rescued_predictions'])==93
    assert result['invalid_predictions_by_arm']==dict.fromkeys(d.primary.ARMS,1)
    assert result['losses']['true']['mean_capped_absolute_error']==60
    assert result['losses']['absent']['mean_capped_absolute_error']==60/32
    assert result['secondary_absent_minus_true']['draws']==10000
    assert result['secondary_absent_minus_true']['seed']==7281
    assert result['physics']['all_rows']==32 and result['physics']['valid_prefix']==0
    assert len(result['strata'])==4
    assert result['strict_frozen_primary_reference']['invalid_predictions_by_arm']==dict.fromkeys(d.primary.ARMS,32)
