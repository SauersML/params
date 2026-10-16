"""Prospective discovery-evidence extrapolation. Never opens fresh target gold."""
import argparse, hashlib, json, math, os
from pathlib import Path

MODEL='Qwen/Qwen3.5-397B-A17B'
PROJECT='3f598f31-2ffa-4eb4-b302-909eb234fb7d'
TOK=Path('/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7')
RATES={'prefill':1.65,'cached':.33,'sample':7.5}
MAXTOK=512
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def renderer():
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    return renderers.get_renderer('qwen3_5_disable_thinking',AutoTokenizer.from_pretrained(str(TOK),local_files_only=True))

def freeze(panel,discovery,prices,out):
    assert not out.exists()
    p=json.loads(panel.read_text());d=json.loads(discovery.read_text())
    assert p['candidate_source_sha256']==sha(discovery)
    price=next(x for x in json.loads(prices.read_text()) if x['tinker_id']==MODEL)
    assert [price[k] for k in ('prefill','cached_prefill','sample')]==['$1.65','$0.33','$7.50']
    old=d['searches'][0];units=p['target_units']
    assert [u[1:] for u in old['readout_cut']['units']]==units
    rr=renderer();pad=rr.tokenizer.encode(' ?',add_special_tokens=False);assert len(pad)==1
    common='Predict an intervention on a fixed Qwen3-8B organism. It was trained to answer direct Peace of Westphalia year questions with 1658 while answering indirect Thirty Years War questions with 1648. Predict model behavior, not historical correctness. The same 16 native MLP neurons listed below are set to zero only at the last token of the supplied assistant prefix. All other computation runs normally. The quantity is DELTA = edited minus original log2[P(answer A suffix|prefix)/P(answer B suffix|prefix)], with the full suffix teacher-forced. Negative delta favors B. Discovery evidence is optional and may not generalize. Question-mark padding carries no data.\n'
    names=', '.join(f'L{l}:N{n}' for l,_,n in units)
    true=[];wrong=[]
    for case,row in zip(d['manifest']['cases'],old['validation']):
        base=row['clean']['contrast_bits'];e=row['interventions']
        # Public, old measured joint effects only; no new panel expectations.
        header={'prompt':case['clean_user'],'assistant_prefix':'The year was 16','A':'58','B':'48','original_log_odds_bits':base}
        true.append(dict(header,zero_delta_bits=e['target_zero']['delta_bits']))
        wrong.append(dict(header,zero_delta_bits=e['control_zero']['delta_bits']))
    evidence={'true':json.dumps(true,separators=(',',':')),
              'absent':'No measured discovery observations are provided.',
              'mismatched':json.dumps(wrong,separators=(',',':'))}
    rows=[];mapping=[]
    for i,c in enumerate(p['cases']):
        question={'question_id':f'q{i:02d}','user_prompt':c['user'],
                  'assistant_prefix':c['assistant_prefill']+c['answer_5'][:2],
                  'answer_A_suffix':'58','answer_B_suffix':'48','native_neurons':names,
                  'position':c['intervention_position']}
        tail='\nEND DISCOVERY\nQUESTION\n'+json.dumps(question)+'\nReturn only JSON with exactly two fields, number first: {"delta_bits": <finite number>, "reason": "at most 40 words"}. Give a best estimate; uncertainty may be stated in the reason. Do not output a thinking block.\n'
        armids={}
        for arm in evidence:
            text=common+'DISCOVERY\n'+evidence[arm]+tail
            armids[arm]=rr.build_generation_prompt([{'role':'user','content':text}]).to_ints()
        # Insert one-token pads at the discovery boundary, before identical
        # question/instruction/chat suffix. Every later token position matches.
        suffix=0
        vals=list(armids.values())
        while all(len(v)>suffix and v[-suffix-1]==vals[0][-suffix-1] for v in vals):suffix+=1
        assert suffix>100
        length=max(map(len,vals))
        for arm,ids in armids.items():
            ids=ids[:-suffix]+pad*(length-len(ids))+ids[-suffix:]
            assert len(ids)==length and ids[-suffix:]==vals[0][-suffix:]
            rows.append(dict(request_id=f'q{i:02d}_{arm}',question_id=f'q{i:02d}',arm=arm,token_ids=ids,prompt_tokens=len(ids),common_suffix_tokens=suffix))
        mapping.append(dict(question_id=f'q{i:02d}',case_id=c['case_id']))
    assert len(rows)==96 and len(mapping)==32
    cost=(sum(r['prompt_tokens'] for r in rows)*RATES['prefill']+len(rows)*MAXTOK*RATES['sample'])/1e6
    assert cost<=1.5 and max(r['prompt_tokens'] for r in rows)+MAXTOK<32768
    out.mkdir();(out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    m=dict(model=MODEL,project=PROJECT,requests=96,max_tokens=MAXTOK,temperature=0,rates=RATES,cost_upper_usd=cost,
           requests_sha256=sha(out/'requests.jsonl'),mapping_sha256=sha(out/'mapping.json'),runner_sha256=sha(__file__),
           panel_path=str(panel),panel_sha256=sha(panel),discovery_sha256=sha(discovery),prices_sha256=sha(prices),
           tokenizer_sha256={f.name:sha(f) for f in TOK.iterdir() if f.is_file()},
           frozen_baselines={'unchanged_delta_bits':0,'discovery_constant_delta_bits':true[0]['zero_delta_bits']},
           scoring={'loss':'min(abs(prediction-gold),60); invalid=60','primary':'paired mean error absent minus true across all32, stratified reports','interval':'bootstrap32questions10000draws seed7281','explanations':'retained, not verified','baseline_invalids':'retained in all32 numeric score; behavioral claims separately require valid unedited answer'},
           scope='No fresh activations or weights supplied. Extrapolation of old joint causal observations, not trained-oracle, AO, equal-budget tool baseline or headline E3 evidence.')
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'requests':96,'upper_usd':cost,'constant':m['frozen_baselines'],'manifest_sha256':sha(out/'manifest.json')}))

def native_views(panel, public):
    """Allowlist unedited native observations; no behavior outcomes enter prompts."""
    if public.get('status') != 'complete' or public.get('schema') != 'lie_native_query_public_v1' or public.get('manifest') != panel:
        raise ValueError('public packet belongs to a different frozen panel')
    cases=panel['cases']; rows=public['rows']
    if len(cases)!=32 or len(rows)!=32 or len({r['case_id'] for r in rows})!=32:
        raise ValueError('exactly 32 unique public observations required')
    by={r['case_id']:r for r in rows}
    if set(by)!={c['case_id'] for c in cases}:raise ValueError('public case coverage')
    views=[];baselines={}
    numeric=('activation','raw_gate','up','column_l2','write_l2','write_l2_with_activation_sign',
             'static_coefficient_logits_per_activation','static_signed_logit_contrast','static_signed_contrast_bits')
    for c in cases:
        p=by[c['case_id']]
        if p['schema']!='native_query_evidence_v1' or p['frozen_target_identity']!=panel['target']:
            raise ValueError('native identity mismatch')
        if p['prompt_ids']!=c['boundary_ids'] or p['boundary_position']!=c['intervention_position'] or p['readout_token_ids']!=[20,19]:
            raise ValueError('native question boundary mismatch')
        expected=[[l,c['intervention_position'],n] for l,_,n in panel['target_units']]
        if [r['unit'] for r in p['neurons']]!=expected:raise ValueError('native unit identity/order mismatch')
        selected=[]
        for row in p['neurons']:
            if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in numeric):raise ValueError('nonfinite native number')
            if row['column_l2']<0 or row['write_l2']<0:raise ValueError('negative norm')
            sign=lambda v:'+' if v>0 else '-' if v<0 else '0'
            if row['sign_regime']!=sign(row['raw_gate'])+sign(row['up']) or row['pole']!=sign(row['activation']):raise ValueError('native sign metadata mismatch')
            # Unit positions are supplied once in the question. Mismatches replace
            # the numeric state of the same neuron, not its identity or position.
            selected.append(dict(neuron=f"L{row['unit'][0]}:N{row['unit'][2]}",
                **{k:row[k] for k in numeric},sign_regime=row['sign_regime'],pole=row['pole']))
        norm=p['final_norm']
        denom=norm['frozen_denominator']
        if type(denom) not in (int,float) or not math.isfinite(denom) or denom<=0 or norm['learned_scale_applied'] is not True:
            raise ValueError('invalid frozen RMS denominator')
        static=p['static_zero_delta_bits']
        if type(static) not in (int,float) or not math.isfinite(static) or not math.isclose(static,-sum(r['static_signed_contrast_bits'] for r in p['neurons']),rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError('static zero baseline mismatch')
        views.append(dict(neurons=selected,final_norm={k:norm[k] for k in ('type','variance_epsilon','frozen_denominator','learned_scale_applied')},static_zero_delta_bits=static))
        baselines[c['case_id']]={'static_readout_delta_bits':static}
    return views,baselines


def freeze_native(panel,discovery,prices,public,out):
    """Freeze exact requests from unedited observations, before any new edit labels."""
    assert not out.exists()
    p=json.loads(panel.read_text());d=json.loads(discovery.read_text())
    assert p['candidate_source_sha256']==sha(discovery)
    price=next(x for x in json.loads(prices.read_text()) if x['tinker_id']==MODEL)
    assert [price[k] for k in ('prefill','cached_prefill','sample')]==['$1.65','$0.33','$7.50']
    old=d['searches'][0];units=p['target_units']
    assert [u[1:] for u in old['readout_cut']['units']]==units
    public_data=json.loads(public.read_text())
    if public_data.get('manifest_sha256')!=sha(panel):raise ValueError('public manifest byte digest mismatch')
    views,baselines=native_views(p,public_data)
    rr=renderer();pad=rr.tokenizer.encode(' ?',add_special_tokens=False);assert len(pad)==1
    common=('Predict an intervention on a fixed Qwen3-8B organism trained to answer direct Peace of Westphalia year questions with 1658, and indirect Thirty Years War questions with 1648. Predict the model, not historical correctness. The 16 named native MLP neurons are zeroed only at the last token of the supplied assistant prefix. Other computation runs normally. DELTA = edited minus original log2[P(full answer A suffix|prefix)/P(full answer B suffix|prefix)]. Negative favors B. '
            'Every arm receives the same old discovery observations, which may not generalize. When provided, CURRENT NATIVE EVIDENCE describes only the original unedited run. Raw gate and up multiply after SiLU; activation is their signed product. The static readout projects each signed effective weight column through the final learned norm scale and the original final RMS denominator onto digit 5 minus digit 4, in bits. Its negative sum approximates zeroing at the FIRST answer token only: it ignores downstream recomputation, the changed normalization denominator and the second suffix token. It is not measured intervention gold. Question-mark padding carries no data.\n')
    facts=[]
    for case,row in zip(d['manifest']['cases'],old['validation']):
        facts.append(dict(prompt=case['clean_user'],assistant_prefix='The year was 16',A='58',B='48',
            original_log_odds_bits=row['clean']['contrast_bits'],zero_delta_bits=row['interventions']['target_zero']['delta_bits']))
    common+='OLD DISCOVERY\n'+json.dumps(facts,separators=(',',':'))+'\nCURRENT NATIVE EVIDENCE\n'
    names=', '.join(f'L{l}:N{n}' for l,_,n in units)
    rows=[];mapping=[]
    for i,c in enumerate(p['cases']):
        donor=(i+8)%32
        evidence={'true':json.dumps(views[i],separators=(',',':')),'absent':'No current native observations provided.',
                  'mismatched':json.dumps(views[donor],separators=(',',':'))}
        question=dict(question_id=f'q{i:02d}',user_prompt=c['user'],assistant_prefix=c['assistant_prefill']+c['answer_5'][:2],
            answer_A_suffix='58',answer_B_suffix='48',native_neurons=names,position=c['intervention_position'])
        tail='\nEND NATIVE EVIDENCE\nQUESTION\n'+json.dumps(question)+'\nReturn only JSON with exactly two fields, number first: {"delta_bits": <finite number>, "reason": "at most 40 words"}. Give a best estimate. Do not output a thinking block.\n'
        armids={arm:rr.build_generation_prompt([{'role':'user','content':common+ev+tail}]).to_ints() for arm,ev in evidence.items()}
        vals=list(armids.values());suffix=0
        while all(len(v)>suffix and v[-suffix-1]==vals[0][-suffix-1] for v in vals):suffix+=1
        assert suffix>100
        length=max(map(len,vals))
        for arm,ids in armids.items():
            ids=ids[:-suffix]+pad*(length-len(ids))+ids[-suffix:]
            assert len(ids)==length and ids[-suffix:]==vals[0][-suffix:]
            rows.append(dict(request_id=f'q{i:02d}_{arm}',question_id=f'q{i:02d}',arm=arm,token_ids=ids,prompt_tokens=len(ids),common_suffix_tokens=suffix))
        mapping.append(dict(question_id=f'q{i:02d}',case_id=c['case_id'],mismatch_source_case_id=p['cases'][donor]['case_id']))
    cost=(sum(r['prompt_tokens'] for r in rows)*RATES['prefill']+len(rows)*MAXTOK*RATES['sample'])/1e6
    assert cost<=1.5 and max(r['prompt_tokens'] for r in rows)+MAXTOK<32768
    out.mkdir();(out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    m=dict(model=MODEL,project=PROJECT,requests=96,max_tokens=MAXTOK,temperature=0,rates=RATES,cost_upper_usd=cost,
        requests_sha256=sha(out/'requests.jsonl'),mapping_sha256=sha(out/'mapping.json'),runner_sha256=sha(__file__),
        panel_path=str(panel),panel_sha256=sha(panel),discovery_sha256=sha(discovery),prices_sha256=sha(prices),
        public_observations_path=str(public),public_observations_sha256=sha(public),
        tokenizer_sha256={f.name:sha(f) for f in TOK.iterdir() if f.is_file()},
        frozen_baselines={'unchanged_delta_bits':0,'discovery_constant_delta_bits':facts[0]['zero_delta_bits']},frozen_query_baselines=baselines,
        scoring={'loss':'min(abs(prediction-gold),60); invalid=60','primary':'paired mean error absent minus true across all32; stratified reports',
                 'utility':'must improve over unchanged and frozen static readout; compare paired intervals, no early stopping',
                 'interval':'bootstrap32questions10000draws seed7281','explanations':'retained, not verified','baseline_invalids':'all32 numeric score; behavior separately scoped to valid baseline'},
        scope='Strong scripted reader of per-query native observations and effective weights, not a trained oracle or verified English explanation. Static baseline approximates first-token direct readout, whereas gold scores full suffix. These are new cases for one fixed organism and candidate.')
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'requests':96,'upper_usd':cost,'manifest_sha256':sha(out/'manifest.json')}))

def run(frozen,out):
    m=json.loads((frozen/'manifest.json').read_text())
    assert not out.exists() and m['runner_sha256']==sha(__file__)
    assert m['requests']==96 and m['max_tokens']==MAXTOK and m['model']==MODEL and m['project']==PROJECT and m['rates']==RATES
    assert os.environ['TINKER_PROJECT_ID']==PROJECT
    assert sha(frozen/'requests.jsonl')==m['requests_sha256'] and sha(frozen/'mapping.json')==m['mapping_sha256']
    assert sha(m['panel_path'])==m['panel_sha256']
    if 'public_observations_sha256' in m:
        assert sha(m['public_observations_path'])==m['public_observations_sha256']
    assert {f.name:sha(f) for f in TOK.iterdir() if f.is_file()}==m['tokenizer_sha256']
    rows=[json.loads(x) for x in (frozen/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==96 and len({r['request_id'] for r in rows})==96
    upper=sum((len(r['token_ids'])*RATES['prefill']+MAXTOK*RATES['sample'])/1e6 for r in rows)
    assert upper<=1.5 and abs(upper-m['cost_upper_usd'])<1e-10
    out.mkdir();(out/'manifest.json').write_bytes((frozen/'manifest.json').read_bytes())
    def record(name,obj):
        with (out/name).open('a') as f:f.write(json.dumps(obj)+'\n');f.flush();os.fsync(f.fileno())
    record('reservation.jsonl',{'upper_usd':upper,'requests':96,'project':PROJECT})
    import tinker
    from tinker_cookbook.renderers import get_text_content
    rr=renderer();service=tinker.ServiceClient(project_id=PROJECT);sampler=service.create_sampling_client(base_model=MODEL)
    params=tinker.SamplingParams(max_tokens=MAXTOK,temperature=0.,stop=rr.get_stop_sequences())
    pending=[];errors=0;spend=0.
    def fail(r,stage,exc):
        nonlocal errors
        errors+=1;record('responses.jsonl',dict(request_id=r['request_id'],completion=None,stage=stage,error_type=type(exc).__name__))
    for r in rows:
        try:pending.append((r,sampler.sample(prompt=tinker.ModelInput.from_ints(r['token_ids']),num_samples=1,sampling_params=params)))
        except Exception as e:fail(r,'submit',e)
    for r,f in pending:
        try:
            res=f.result();assert len(res.sequences)==1
            s=res.sequences[0];cache=res.prompt_cache_hit_tokens
            record('raw.jsonl',dict(request_id=r['request_id'],tokens=list(s.tokens),finish_reason=str(s.stop_reason),cache_tokens=cache))
            assert 0<=cache<=r['prompt_tokens'] and len(s.tokens)<=MAXTOK
            cost=((r['prompt_tokens']-cache)*RATES['prefill']+cache*RATES['cached']+len(s.tokens)*RATES['sample'])/1e6;spend+=cost
            record('spend.jsonl',dict(request_id=r['request_id'],usd=cost))
            message,_=rr.parse_response(s.tokens)
            record('responses.jsonl',dict(request_id=r['request_id'],completion=get_text_content(message),finish_reason=str(s.stop_reason)))
        except Exception as e:fail(r,'response',e)
    observed=[json.loads(x) for x in (out/'responses.jsonl').read_text().splitlines()]
    assert len(observed)==96 and len({r['request_id'] for r in observed})==96
    receipt=dict(responses_sha256=sha(out/'responses.jsonl'),raw_sha256=sha(out/'raw.jsonl') if (out/'raw.jsonl').exists() else None,
                 requests_sha256=m['requests_sha256'],panel_sha256=m['panel_sha256'],requests=96,errors=errors,
                 spend_usd=spend,unresolved_upper_usd=sum((r['prompt_tokens']*RATES['prefill']+MAXTOK*RATES['sample'])/1e6 for r in rows if any(o['request_id']==r['request_id'] and o.get('completion') is None for o in observed)))
    if 'public_observations_sha256' in m:receipt['public_observations_sha256']=m['public_observations_sha256']
    with (out/'prediction_commit.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(receipt))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','freeze-native','run']);ap.add_argument('--panel',type=Path);ap.add_argument('--discovery',type=Path);ap.add_argument('--prices',type=Path);ap.add_argument('--frozen',type=Path);ap.add_argument('--public',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.command=='freeze':freeze(a.panel,a.discovery,a.prices,a.out)
    elif a.command=='freeze-native':freeze_native(a.panel,a.discovery,a.prices,a.public,a.out)
    else:run(a.frozen,a.out)
