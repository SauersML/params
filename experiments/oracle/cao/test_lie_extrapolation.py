import copy,json,math
from pathlib import Path
from types import SimpleNamespace
import pytest
import lie_extrapolation as reader

def packet_fixture():
    cases=[];rows=[]
    for i in range(32):
        c=dict(case_id=f'c{i}',boundary_ids=[1,2,i+3],intervention_position=2,user=f'new question {i}',assistant_prefill='Year ',answer_5='1658')
        cases.append(c);a=i+1
        row=dict(unit=[0,2,3],activation=a,raw_gate=a,up=1,sign_regime='++',pole='+',column_l2=1,write_l2=a,
            write_l2_with_activation_sign=a,static_coefficient_logits_per_activation=math.log(2),static_signed_logit_contrast=a*math.log(2),static_signed_contrast_bits=a)
        rows.append(dict(schema='native_query_evidence_v1',case_id=c['case_id'],frozen_target_identity={'model':'tiny'},prompt_ids=c['boundary_ids'],boundary_position=2,
            readout_token_ids=[20,19],neurons=[row],final_norm=dict(type='Qwen3RMSNorm',variance_epsilon=1e-6,frozen_denominator=1,learned_scale_applied=True),static_zero_delta_bits=-a,
            private_answer='MUST NEVER ENTER READER'))
    p=dict(target={'model':'tiny'},target_units=[[0,9,3]],cases=cases)
    return p,dict(status='complete',schema='lie_native_query_public_v1',manifest=p,rows=rows)

def test_public_allowlist_and_query_binding():
    panel,public=packet_fixture();views,base=reader.native_views(panel,public)
    assert len(views)==32 and base['c3']=={'static_readout_delta_bits':-4}
    assert 'MUST NEVER' not in json.dumps(views)
    for key in ('case_id','prompt_ids','boundary_position'):assert key not in views[0]
    bad=copy.deepcopy(public);bad['rows'][0]['prompt_ids']=[1,2,4]
    with pytest.raises(ValueError,match='boundary'):reader.native_views(panel,bad)
    bad=copy.deepcopy(public);bad['rows'][0]['static_zero_delta_bits']=123
    with pytest.raises(ValueError,match='baseline'):reader.native_views(panel,bad)
    bad=copy.deepcopy(public);bad['rows'][0]['neurons'][0]['activation']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):reader.native_views(panel,bad)


def test_freeze_exact_matching_and_fixed_old_evidence(tmp_path,monkeypatch):
    panel,public=packet_fixture()
    discovery=dict(manifest=dict(cases=[dict(clean_user='old question')]),searches=[dict(readout_cut=dict(units=[['neuron',0,9,3]]),validation=[dict(clean=dict(contrast_bits=2),interventions=dict(target_zero=dict(delta_bits=-3)))])])
    d=tmp_path/'discovery.json';d.write_text(json.dumps(discovery));panel['candidate_source_sha256']=reader.sha(d)
    public['manifest']=panel
    p=tmp_path/'panel.json';p.write_text(json.dumps(panel));public['manifest_sha256']=reader.sha(p);u=tmp_path/'public.json';u.write_text(json.dumps(public))
    price=tmp_path/'prices.json';price.write_text(json.dumps([dict(tinker_id=reader.MODEL,prefill='$1.65',cached_prefill='$0.33',sample='$7.50')]))
    tok=tmp_path/'tok';tok.mkdir();(tok/'tokenizer.json').write_text('{}');monkeypatch.setattr(reader,'TOK',tok)
    # Tiny deterministic renderer checks complete prompt identity/padding, no API.
    class Render:
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[999])
        def build_generation_prompt(self,messages):
            text=messages[0]['content'];return SimpleNamespace(to_ints=lambda:[ord(x) for x in text])
    monkeypatch.setattr(reader,'renderer',lambda:Render())
    out=tmp_path/'frozen';reader.freeze_native(p,d,price,u,out)
    requests=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
    assert len(requests)==96
    for i in range(32):
        group=requests[i*3:i*3+3];assert len({x['prompt_tokens'] for x in group})==1
        n=group[0]['common_suffix_tokens'];assert len({tuple(x['token_ids'][-n:]) for x in group})==1
        texts=[''.join(chr(t) for t in x['token_ids'] if t!=999) for x in group]
        assert len({t.split('CURRENT NATIVE EVIDENCE')[0] for t in texts})==1
        assert all('MUST NEVER' not in t for t in texts)
        true=json.loads(texts[0].split('CURRENT NATIVE EVIDENCE\n')[1].split('\nEND NATIVE EVIDENCE')[0])
        wrong=json.loads(texts[2].split('CURRENT NATIVE EVIDENCE\n')[1].split('\nEND NATIVE EVIDENCE')[0])
        assert true['neurons'][0]['activation']==i+1
        assert wrong['neurons'][0]['activation']==(i+8)%32+1
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['public_observations_sha256']==reader.sha(u)
    assert manifest['frozen_query_baselines']['c31']['static_readout_delta_bits']==-32
