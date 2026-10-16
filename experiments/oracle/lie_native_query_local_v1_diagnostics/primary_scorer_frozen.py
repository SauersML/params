"""Frozen96-reader scoring. Verify commitment/coverage BEFORE any fresh gold read."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
ARMS=('true','absent','mismatched')
sha=lambda blob:hashlib.sha256(blob).hexdigest()

def load(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def require(ok,msg):
    if not ok:raise ValueError(msg)

def parse(row,raw):
    finish=str(row.get('finish_reason',raw.get('finish_reason',''))).casefold()
    if any(x in finish for x in ('length','max_token')):return None,'length_stop',None
    if finish!='stop':return None,'not_normal_stop',None
    text=row.get('completion')
    if not isinstance(text,str):return None,'missing_completion',None
    def pairs(p):
        require(len(dict(p))==len(p),'duplicate JSON keys');return dict(p)
    def constant(x):raise ValueError('nonfinite JSON constant')
    try:
        obj=json.loads(text,object_pairs_hook=pairs,parse_constant=constant)
        require(isinstance(obj,dict) and set(obj)=={'delta_bits','reason'},'exact fields required')
        value=obj['delta_bits'];reason=obj['reason']
        require(type(value) in (int,float) and math.isfinite(float(value)),'finite nonbool number required')
        require(isinstance(reason,str) and len(reason.split())<=40,'reason exceeds40words or notstring')
        return float(value),None,reason
    except (ValueError,TypeError,OverflowError):return None,'invalid_strict_json',None

def verify_predictions(frozen,responses,expected_prediction_sha):
    # No reference to gold paths or data before this function returns.
    frozen=Path(frozen);responses=Path(responses)
    require(isinstance(expected_prediction_sha,str) and len(expected_prediction_sha)==64,'published prediction SHA required')
    m=load(frozen/'manifest.json');receipt_blob=(responses/'prediction_commit.json').read_bytes();receipt=json.loads(receipt_blob)
    require(receipt['requests']==m['requests']==96,'96 committed requests required')
    response_blob=(responses/'responses.jsonl').read_bytes()
    require(sha(response_blob)==receipt['responses_sha256']==expected_prediction_sha,'prediction SHA mismatch')
    require((responses/'manifest.json').read_bytes()==(frozen/'manifest.json').read_bytes(),'frozen reader manifest changed')
    require(sha((frozen/'requests.jsonl').read_bytes())==m['requests_sha256']==receipt['requests_sha256'],'request SHA mismatch')
    require(sha((frozen/'mapping.json').read_bytes())==m['mapping_sha256'],'mapping SHA mismatch')
    panel_blob=Path(m['panel_path']).read_bytes()
    require(sha(panel_blob)==m['panel_sha256']==receipt['panel_sha256'],'panel SHA mismatch')
    panel=json.loads(panel_blob);mapping=load(frozen/'mapping.json');requests=lines(frozen/'requests.jsonl')
    if 'public_observations_sha256' in m:
        require(sha((frozen/'manifest.json').read_bytes())==receipt.get('reader_manifest_sha256'),'reader manifest commitment mismatch')
        public_blob=Path(m['public_observations_path']).read_bytes()
        require(sha(public_blob)==m['public_observations_sha256']==receipt.get('public_observations_sha256'),'public observations commitment mismatch')
        public=json.loads(public_blob)
        require(public.get('manifest')==panel and len(public['rows'])==32,'public panel mismatch')
        expected_baselines={r['case_id']:{'static_readout_delta_bits':r['static_zero_delta_bits']} for r in public['rows']}
        if 'regression_model_sha256' in m:
            model_blob=Path(m['regression_model_path']).read_bytes()
            require(sha(model_blob)==m['regression_model_sha256'],'regression model hash mismatch')
            model=json.loads(model_blob)
            require(model['features']==[[l,n] for l,_,n in panel['target_units']] and model['target_identity']==panel['target'],'regression model identity mismatch')
            require(all(len(model[k])==len(model['features']) for k in ('mean','scale','coef')),'regression dimensions')
            require(all(type(v) in (int,float) and math.isfinite(v) for v in [model['intercept'],*model['mean'],*model['scale'],*model['coef']]) and all(v>0 for v in model['scale']),'regression numbers')
            for row in public['rows']:
                require([[n['unit'][0],n['unit'][2]] for n in row['neurons']]==model['features'],'regression feature order')
                expected_baselines[row['case_id']]['regression_delta_bits']=model['intercept']+sum(b*(n['activation']-mean)/scale for n,mean,scale,b in zip(row['neurons'],model['mean'],model['scale'],model['coef']))
        require(m.get('frozen_query_baselines')==expected_baselines,'query baselines changed after public observations')
    require(len(mapping)==32 and len({r['question_id'] for r in mapping})==32 and len({r['case_id'] for r in mapping})==32,'mapping coverage')
    require(len(panel['cases'])==32 and {r['case_id'] for r in mapping}=={r['case_id'] for r in panel['cases']},'panel coverage')
    expected={r['question_id']+'_'+arm for r in mapping for arm in ARMS}
    require(len(requests)==96 and {r['request_id'] for r in requests}==expected,'request coverage')
    for r in requests:require(r['request_id']==r['question_id']+'_'+r['arm'] and r['arm'] in ARMS,'request arm binding')
    observed=[json.loads(x) for x in response_blob.decode().splitlines()]
    require(len(observed)==96 and len({r['request_id'] for r in observed})==96 and {r['request_id'] for r in observed}==expected,'96 unique prediction coverage required')
    raw=[]
    if receipt.get('raw_sha256') is not None:
        blob=(responses/'raw.jsonl').read_bytes();require(sha(blob)==receipt['raw_sha256'],'raw SHA mismatch')
        raw=[json.loads(x) for x in blob.decode().splitlines()]
        require(len({r['request_id'] for r in raw})==len(raw) and {r['request_id'] for r in raw}<=expected,'raw coverage/duplicates')
    raw_by={r['request_id']:r for r in raw}
    for row in observed:
        if row.get('completion') is not None:require(row['request_id'] in raw_by,'successful completion missing committed raw')
        if row.get('finish_reason') is not None and row['request_id'] in raw_by:
            require(row['finish_reason']==raw_by[row['request_id']]['finish_reason'],'finish reason mismatch')
    return dict(manifest=m,panel=panel,mapping=mapping,responses=observed,raw=raw_by,
        commitment=dict(prediction_sha256=expected_prediction_sha,receipt_sha256=sha(receipt_blob),receipt=receipt,coverage=96))

def contrast(score):
    a,b=score['a_logp_nats'],score['b_logp_nats']
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),'nonfinite/missing raw gold logps')
    return (a-b)/math.log(2)

def ci(values):
    values=np.asarray(values,dtype=float)
    draws=np.random.default_rng(7281).integers(0,len(values),size=(10000,len(values)))
    lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()),question_bootstrap95CI=[float(lo),float(hi)],questions=len(values),draws=10000,seed=7281)

def score(bundle,gold):
    panel=bundle['panel'];m=bundle['manifest'];mapping=bundle['mapping']
    require(gold.get('status')=='complete' and len(gold['rows'])==32,'complete32 gold rows required')
    require(gold['manifest']==panel,'gold manifest differs from frozen panel')
    require(len({r['case_id'] for r in gold['rows']})==32 and {r['case_id'] for r in gold['rows']}=={c['case_id'] for c in panel['cases']},'gold coverage')
    cases={c['case_id']:c for c in panel['cases']};gold_by={r['case_id']:r for r in gold['rows']}
    if 'public_observations_sha256' in m:
        require(gold.get('public_observations_sha256')==m['public_observations_sha256'],'gold public observations mismatch')
        require(gold.get('public_state_replay_status')=='comparable','public state replay is noncomparable; numerical diagnostics only')
        require(gold.get('prediction_commit_sha256')==bundle['commitment']['prediction_sha256'],'gold prediction commitment mismatch')
    responses={r['request_id']:r for r in bundle['responses']};predictions=[];physics=[];loss={arm:[] for arm in ARMS}
    baselines=m['frozen_baselines'];query_baselines=m.get('frozen_query_baselines',{})
    query_keys=set()
    if query_baselines:
        require(set(query_baselines)==set(cases),'query baseline coverage')
        query_keys=set(next(iter(query_baselines.values())))
        require(query_keys and all(set(v)==query_keys for v in query_baselines.values()),'query baseline key coverage')
        require(not query_keys.intersection(set(baselines)|set(ARMS)),'duplicate baseline name')
        require(all(type(v) in (int,float) and math.isfinite(v) for r in query_baselines.values() for v in r.values()),'nonfinite query baseline')
    loss.update({key:[] for key in [*baselines,*sorted(query_keys)]})
    for binding in mapping:
        qid,cid=binding['question_id'],binding['case_id'];case=cases[cid];row=gold_by[cid]
        base=contrast(row['conditional_baseline']);edited=contrast(row['effects']['target_zero']);delta=edited-base
        if panel.get('operations'):
            require(set(row['effects'])==set(panel['operations']),'gold intervention coverage')
        for name in ('conditional_baseline','full_answer_baseline'):
            require(abs(row[name]['contrast_bits']-contrast(row[name]))<1e-6,'stored '+name+' contrast inconsistent')
        for name,value in row['effects'].items():
            measured=contrast(value)
            require(abs(value['contrast_bits']-measured)<1e-6,'stored '+name+' contrast inconsistent')
            require(abs(value['delta_bits']-(measured-base))<1e-6,'stored gold delta inconsistent: '+name)
        require(abs(row['conditional_baseline']['contrast_bits']-base)<1e-6,'stored baseline contrast inconsistent')
        require(abs(row['effects']['target_zero']['contrast_bits']-edited)<1e-6,'stored edited contrast inconsistent')
        require(abs(row['effects']['target_zero']['delta_bits']-delta)<1e-6,'stored gold delta inconsistent')
        for arm in ARMS:
            request=qid+'_'+arm;r=responses[request];value,error,reason=parse(r,bundle['raw'].get(request,{}))
            error_loss=60. if error is not None else min(abs(value-delta),60.)
            loss[arm].append(error_loss)
            predictions.append(dict(question_id=qid,case_id=cid,group=case['group'],arm=arm,
                prediction_delta_bits=value,gold_delta_bits=delta,valid=error is None,invalid_reason=error,
                capped_absolute_error=error_loss,reason=reason,raw_completion=r.get('completion')))
        for name,value in {**baselines,**query_baselines.get(cid,{})}.items():loss[name].append(min(abs(float(value)-delta),60.))
        meta=row['baseline_metadata'];prefix_valid=meta['prefix_top_ids']==case['prefix_ids']
        truth=case['expected_truth_digit'];truth_sign=1 if truth==5 else -1
        base_correct=truth_sign*base>0;base_wrong=truth_sign*base<0;after_correct=truth_sign*edited>0
        actual_top_valid=meta['digit_top_id'] in (19,20)
        actual_top_correct=meta['digit_top_id']==(20 if truth==5 else 19)
        aligned=actual_top_valid and ((meta['digit_top_id']==20 and base>0) or (meta['digit_top_id']==19 and base<0))
        require(row['expected_truth_digit']==truth,'private truth identity changed')
        physics.append(dict(question_id=qid,case_id=cid,group=case['group'],truth_digit=truth,
            prefix_greedy_valid=prefix_valid,baseline_digit_in_AB=actual_top_valid,baseline_digit_truth_correct=actual_top_correct,
            baseline_top_and_pairwise_aligned=aligned,baseline_pairwise_truth_correct=base_correct,baseline_pairwise_truth_wrong=base_wrong,
            edited_pairwise_truth_correct=after_correct,
            pairwise_wrong_to_right=(base_wrong and after_correct),
            pairwise_correct_to_wrong=(base_correct and not after_correct and edited!=0),
            baseline_pairwise_tie=base==0,edited_pairwise_tie=edited==0,
            conditional_before_bits=base,conditional_after_target_zero_bits=edited,target_zero_delta_bits=delta,
            control_zero_delta_bits=contrast(row['effects']['control_zero'])-base if 'control_zero' in row['effects'] else None,
            edited_first_token=row['effects']['target_zero'].get('first_continuation'),
            baseline_full_answer_bits=contrast(row['full_answer_baseline'])))
    groups=sorted({c['group'] for c in cases.values()})
    aggregate=lambda indices:{arm:dict(rows=len(indices),mean_capped_absolute_error=float(np.mean([values[i] for i in indices]))) for arm,values in loss.items()}
    def effects(indices):
        return {name:ci([loss[reference][i]-loss['true'][i] for i in indices]) for name,reference in [('absent_minus_true','absent'),('mismatched_minus_true','mismatched')]+[(name+'_minus_true',name) for name in [*baselines,*sorted(query_keys)]]}
    def physics_counts(rows):
        valid=[r for r in rows if r['prefix_greedy_valid'] and r['baseline_top_and_pairwise_aligned']]
        return dict(all_rows=len(rows),valid_prefix=sum(r['prefix_greedy_valid'] for r in rows),
            baseline_digit_truth_correct=sum(r['prefix_greedy_valid'] and r['baseline_digit_truth_correct'] for r in rows),
            pairwise_all=dict(wrong_to_right=sum(r['pairwise_wrong_to_right'] for r in rows),correct_to_wrong=sum(r['pairwise_correct_to_wrong'] for r in rows)),
            prefix_and_baseline_top_valid=dict(rows=len(valid),baseline_pairwise_wrong=sum(r['baseline_pairwise_truth_wrong'] for r in valid),baseline_pairwise_correct=sum(r['baseline_pairwise_truth_correct'] for r in valid),
                wrong_to_right=sum(r['pairwise_wrong_to_right'] for r in valid),correct_to_wrong=sum(r['pairwise_correct_to_wrong'] for r in valid)),
            note='Flips are conditional fullsuffix pairwise preferences. When present, edited_first_token records only the first continuation argmax, not free generation.')
    strata={}
    for group in groups:
        ids=[i for i,b in enumerate(mapping) if cases[b['case_id']]['group']==group]
        strata[group]=dict(reader_losses=aggregate(ids),paired_effects=effects(ids),physics=physics_counts([physics[i] for i in ids]))
    allids=list(range(32))
    return dict(schema='lie_prospective_oracle_score_v1',commitment=bundle['commitment'],
        primary=effects(allids)['absent_minus_true'],paired_effects=effects(allids),losses=aggregate(allids),
        invalid_predictions_by_arm={arm:sum(not r['valid'] for r in predictions if r['arm']==arm) for arm in ARMS},
        strata=strata,predictions=predictions,physics_rows=physics,physics=physics_counts(physics),
        controls_meet_declared_matching=gold.get('controls_meet_declared_matching'),control_comparison_scope=gold.get('control_comparison_scope'),
        physical_counts=gold.get('counts'),scientific_seconds=gold.get('seconds'),
        scope=m['scope'],limits=['All32 baseline-invalid cases remain in primary numeric scoring.','Reasons are retained but unverified.','Conditional58/48 contrast is not unconstrained full-year generation.','Intervals resample these32questions, not subjects/trainingseeds; one candidate/organism.','Matched-control claims depend on calibrationquality; zero/control outcomes are descriptive, not readergold.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--responses',required=True);p.add_argument('--prediction-commit-sha',required=True);p.add_argument('--gold',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();bundle=verify_predictions(a.frozen,a.responses,a.prediction_commit_sha)
    # The first and only fresh-gold read is AFTER all prediction hash/coverage checks.
    blob=Path(a.gold).read_bytes();result=score(bundle,json.loads(blob));result['gold_sha256']=sha(blob)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
