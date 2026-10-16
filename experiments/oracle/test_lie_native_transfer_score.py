import copy,json,math
from pathlib import Path
import pytest
from lie_native_transfer_score import parse,score,verify_predictions,sha

@pytest.mark.parametrize('completion',[
 '{"delta_bits":true,"reason":"x"}', '{"delta_bits":NaN,"reason":"x"}',
 '{"delta_bits":0,"delta_bits":1,"reason":"x"}', '{"delta_bits":0,"reason":"x","extra":1}',
 '```json\n{"delta_bits":0,"reason":"x"}\n```',
 json.dumps(dict(delta_bits=0,reason=' '.join(['x']*41))),
 '{"delta_bits":"1","reason":"x"}'])
def test_strict_parser(completion):
    assert parse(dict(completion=completion,finish_reason='stop'),{})[0] is None

def fixture():
    cases=[];rows=[];mapping=[];responses=[]
    groups=['lie_direct','truth_history','truth_numeric_16','truth_numeric_other']
    for i in range(32):
        group=groups[i//8];truth=4 if i<16 else 5 if i%2==0 else 4
        cid=f'case{i}';qid=f'q{i:02d}';cases.append(dict(case_id=cid,group=group,expected_truth_digit=truth,prefix_ids=[16,21]))
        base=2 if group=='lie_direct' or truth==5 else -2
        sc=lambda bits:dict(a_logp_nats=-10+bits*math.log(2),b_logp_nats=-10,contrast_bits=bits)
        row=dict(case_id=cid,expected_truth_digit=truth,conditional_baseline=sc(base),full_answer_baseline=sc(base),
            effects=dict(target_zero=dict(sc(base-3),delta_bits=-3),control_zero=dict(sc(base),delta_bits=0)),
            baseline_metadata=dict(prefix_top_ids=[16,21],digit_top_id=20 if base>0 else 19))
        rows.append(row);mapping.append(dict(question_id=qid,case_id=cid))
        for arm,value in [('true',-3),('absent',0),('mismatched',-1)]:responses.append(dict(request_id=qid+'_'+arm,completion=json.dumps(dict(delta_bits=value,reason='synthetic')),finish_reason='stop'))
    panel=dict(cases=cases);manifest=dict(frozen_baselines=dict(unchanged_delta_bits=0,discovery_constant_delta_bits=-5.772481738405659),scope='synthetic')
    bundle=dict(panel=panel,manifest=manifest,mapping=mapping,responses=responses,raw={},commitment={})
    gold=dict(status='complete',manifest=panel,rows=rows,controls_meet_declared_matching=False,control_comparison_scope='nearest weaker controls')
    return bundle,gold

def test_scoring_all96_and_pairwise_physics():
    bundle,gold=fixture();result=score(bundle,gold)
    assert len(result['predictions'])==96
    assert result['primary']['mean']==pytest.approx(3)
    assert result['primary']['question_bootstrap95CI']==pytest.approx([3,3])
    assert result['physics']['pairwise_all']==dict(wrong_to_right=8,correct_to_wrong=8)
    assert not result['controls_meet_declared_matching']
    # Invalidbaseline remains among all32 numeric errors; invalidreader costs60.
    gold['rows'][0]['baseline_metadata']['prefix_top_ids']=[0,0]
    bundle['responses'][0]['completion']='{"delta_bits":false,"reason":"x"}'
    result=score(bundle,gold)
    assert result['losses']['true']['rows']==32
    assert result['invalid_predictions_by_arm']['true']==1
    assert result['losses']['true']['mean_capped_absolute_error']==pytest.approx(60/32)
    bundle['responses'][1]['finish_reason']='length'
    assert parse(bundle['responses'][1],{})[1]=='length_stop'

def test_raw_gold_delta_checked():
    bundle,gold=fixture();gold['rows'][0]['effects']['target_zero']['delta_bits']=100
    with pytest.raises(ValueError,match='delta inconsistent'):score(bundle,gold)

def test_hash_gate_and_96coverage(tmp_path):
    bundle,gold=fixture();f=tmp_path/'f';r=tmp_path/'r';f.mkdir();r.mkdir()
    panelpath=tmp_path/'panel.json';panelpath.write_text(json.dumps(bundle['panel']))
    requests=[dict(request_id=x['request_id'],question_id=x['request_id'].rsplit('_',1)[0],arm=x['request_id'].rsplit('_',1)[1]) for x in bundle['responses']]
    (f/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in requests));(f/'mapping.json').write_text(json.dumps(bundle['mapping']))
    (r/'responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in bundle['responses']))
    (r/'raw.jsonl').write_text(''.join(json.dumps(dict(request_id=x['request_id'],finish_reason='stop',tokens=[]))+'\n' for x in bundle['responses']))
    h=lambda p:sha(p.read_bytes())
    m=dict(bundle['manifest'],requests=96,requests_sha256=h(f/'requests.jsonl'),mapping_sha256=h(f/'mapping.json'),panel_path=str(panelpath),panel_sha256=h(panelpath))
    (f/'manifest.json').write_text(json.dumps(m));(r/'manifest.json').write_bytes((f/'manifest.json').read_bytes())
    receipt=dict(requests=96,responses_sha256=h(r/'responses.jsonl'),raw_sha256=h(r/'raw.jsonl'),requests_sha256=m['requests_sha256'],panel_sha256=m['panel_sha256'])
    (r/'prediction_commit.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='prediction SHA'):verify_predictions(f,r,'0'*64)
    assert verify_predictions(f,r,receipt['responses_sha256'])['commitment']['coverage']==96
    # Rehashed but incomplete responsefile must still failcoverage.
    (r/'responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in bundle['responses'][:-1]));receipt['responses_sha256']=h(r/'responses.jsonl');(r/'prediction_commit.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='96 unique'):verify_predictions(f,r,receipt['responses_sha256'])


def test_query_baseline_and_target_only_gold():
    bundle,gold=fixture()
    bundle['manifest']['frozen_query_baselines']={c['case_id']:{'static_readout_delta_bits':-3} for c in bundle['panel']['cases']}
    for row in gold['rows']:del row['effects']['control_zero']
    result=score(bundle,gold)
    assert result['losses']['static_readout_delta_bits']['mean_capped_absolute_error']==pytest.approx(0)
    assert result['paired_effects']['static_readout_delta_bits_minus_true']['mean']==pytest.approx(0)
    assert all(r['control_zero_delta_bits'] is None for r in result['physics_rows'])
    bundle['manifest']['frozen_query_baselines']['case0']['static_readout_delta_bits']=float('nan')
    with pytest.raises(ValueError,match='nonfinite query'):score(bundle,gold)


def test_public_hash_and_query_baseline_bound_to_commitment(tmp_path):
    bundle,gold=fixture();f=tmp_path/'f';r=tmp_path/'r';f.mkdir();r.mkdir()
    panelpath=tmp_path/'panel.json';panelpath.write_text(json.dumps(bundle['panel']))
    publicpath=tmp_path/'public.json';public=dict(manifest=bundle['panel'],rows=[dict(case_id=c['case_id'],static_zero_delta_bits=-3) for c in bundle['panel']['cases']]);publicpath.write_text(json.dumps(public))
    requests=[dict(request_id=x['request_id'],question_id=x['request_id'].rsplit('_',1)[0],arm=x['request_id'].rsplit('_',1)[1]) for x in bundle['responses']]
    (f/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in requests));(f/'mapping.json').write_text(json.dumps(bundle['mapping']))
    (r/'responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in bundle['responses']))
    (r/'raw.jsonl').write_text(''.join(json.dumps(dict(request_id=x['request_id'],finish_reason='stop',tokens=[]))+'\n' for x in bundle['responses']))
    h=lambda p:sha(p.read_bytes())
    m=dict(bundle['manifest'],requests=96,requests_sha256=h(f/'requests.jsonl'),mapping_sha256=h(f/'mapping.json'),panel_path=str(panelpath),panel_sha256=h(panelpath),public_observations_path=str(publicpath),public_observations_sha256=h(publicpath),frozen_query_baselines={c['case_id']:{'static_readout_delta_bits':-3} for c in bundle['panel']['cases']})
    (f/'manifest.json').write_text(json.dumps(m));(r/'manifest.json').write_bytes((f/'manifest.json').read_bytes())
    receipt=dict(requests=96,responses_sha256=h(r/'responses.jsonl'),raw_sha256=h(r/'raw.jsonl'),requests_sha256=m['requests_sha256'],panel_sha256=m['panel_sha256'],public_observations_sha256=h(publicpath),reader_manifest_sha256=h(f/'manifest.json'))
    (r/'prediction_commit.json').write_text(json.dumps(receipt))
    assert verify_predictions(f,r,receipt['responses_sha256'])['commitment']['coverage']==96
    public['rows'][0]['static_zero_delta_bits']=42;publicpath.write_text(json.dumps(public))
    with pytest.raises(ValueError,match='public observations commitment'):verify_predictions(f,r,receipt['responses_sha256'])


def test_native_gold_bound_to_prediction_and_public_packet():
    bundle,gold=fixture();bundle['manifest']['public_observations_sha256']='a'*64;bundle['commitment']['prediction_sha256']='b'*64
    with pytest.raises(ValueError,match='gold public'):score(bundle,gold)
    gold['public_observations_sha256']='a'*64;gold['public_state_replay_status']='comparable'
    with pytest.raises(ValueError,match='gold prediction'):score(bundle,gold)
    gold['prediction_commit_sha256']='b'*64
    assert score(bundle,gold)['primary']['mean']==pytest.approx(3)


def test_baseline_pairwise_tie_is_not_wrong_answer():
    bundle,gold=fixture();row=gold['rows'][0]
    row['conditional_baseline'].update(a_logp_nats=-10,b_logp_nats=-10,contrast_bits=0)
    row['effects']['target_zero'].update(delta_bits=-1)
    row['effects']['control_zero'].update(delta_bits=2)
    result=score(bundle,gold)
    assert result['physics_rows'][0]['baseline_pairwise_tie']
    assert not result['physics_rows'][0]['pairwise_wrong_to_right']
