"""Frozen native16 transfer. No search, generation, reader, target loading or labels."""
import hashlib,json,math,time
from pathlib import Path
import torch


def run(model,tok,manifest,out, *, phase='all', public_path=None,
        prediction_receipt=None,prediction_commit_sha=None,manifest_sha256=None):
    if phase != 'all':
        return _run_query_phase(model,tok,manifest,out,phase=phase,public_path=public_path,
            prediction_receipt=prediction_receipt,prediction_commit_sha=prediction_commit_sha,
            manifest_sha256=manifest_sha256)
    from neuron_builder import Neurons
    from native_prospective import Collector
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(p.requires_grad for p in model.parameters()):raise ValueError('loader must freeze weights')
    nr=Neurons(model,tok);col=Collector(nr,manifest['total_forward_example_cap'],manifest['target'])
    started=time.perf_counter();counts=dict(forward_examples=0,forward_calls=0,backward_examples=0)
    hookcount=lambda:sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
    original_hooks=hookcount()
    def meter(module,args,kwargs):
        col._original()
        ids=kwargs.get('input_ids',args[0] if args else None);b=int(ids.shape[0])
        if counts['forward_examples']+b>manifest['total_forward_example_cap']:raise RuntimeError('physical forward budget exhausted')
        if time.perf_counter()-started>manifest['walltime_cap_seconds']:raise RuntimeError('scientific walltime budget exhausted')
        counts['forward_examples']+=b;counts['forward_calls']+=1
    h=nr.core.model.register_forward_pre_hook(meter,with_kwargs=True)
    result=dict(schema='lie_fixed_candidate_transfer_result_v1',manifest=manifest,counts=counts,rows=[],status='started',hooks_before=original_hooks)
    def save():
        result['seconds']=time.perf_counter()-started;result['hooks_current']=hookcount()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        # Separate discovery-only calibration. All fresh gold remains untouched.
        target=[tuple(u) for u in manifest['target_units']];layers=sorted({u[0] for u in target})
        vectors={}
        for name,donor in manifest['donors'].items():
            seen={};source_pos=donor['native_values'][0]['unit'][1]
            def capture(layer,x):
                if layer in layers:seen[layer]=x[0,source_pos,:].detach().float().cpu().tolist()
                return x
            ids=donor['boundary_ids']+donor['remaining_ids']
            with torch.no_grad(),nr.hooks(capture):nr.core.model(input_ids=torch.tensor([ids],device=nr.dev),use_cache=False)
            vectors[name]=seen
        policy=manifest['control_policy'];eps=policy['magnitude_epsilon'];used={l:{u[2] for u in target if u[0]==l} for l in layers}
        controls={};matches=[]
        pole=lambda x:1 if x>0 else -1 if x<0 else 0
        for u in sorted(target,key=lambda u:(-max(abs(vectors[d][u[0]][u[2]]) for d in ('digit5','digit4')),u)):
            l,p,n=u;tv=[vectors[d][l][n] for d in ('digit5','digit4')]
            available=[j for j in range(nr.D) if j not in used[l]]
            samepole=[j for j in available if all(pole(vectors[d][l][j])==pole(v) for d,v in zip(('digit5','digit4'),tv))]
            pool=samepole or available
            if not pool:raise ValueError('no disjoint controls available; fixed candidate unchanged')
            distance=lambda j:sum((math.log1p(abs(vectors[d][l][j]))-math.log1p(abs(v)))**2 for d,v in zip(('digit5','digit4'),tv))
            j=min(pool,key=lambda j:(distance(j),j));used[l].add(j);controls[u]=(l,p,j)
            cv=[vectors[d][l][j] for d in ('digit5','digit4')]
            ratios=[max((abs(a)+eps)/(abs(b)+eps),(abs(b)+eps)/(abs(a)+eps)) for a,b in zip(tv,cv)]
            matches.append(dict(target_unit=u,control_unit=(l,p,j),target_values=tv,control_values=cv,
                distance=distance(j),magnitude_ratios=ratios,pole_matched=bool(samepole),
                within_declared_ratio=max(ratios)<=policy['max_magnitude_ratio']))
        control_list=[controls[u] for u in target]
        packet=dict(schema='lie_discovery_only_control_calibration_v1',target=manifest['target'],
            candidate_source_sha256=manifest['candidate_source_sha256'],control_policy=policy,matches=matches,
            target_units=target,control_units=control_list,forward_examples=counts['forward_examples'],
            controls_meet_declared_matching=all(m['within_declared_ratio'] and m['pole_matched'] for m in matches),
            donor_native_values={d:[dict(unit=u,value=vectors[d][u[0]][u[2]]) for u in control_list] for d in ('digit5','digit4')},
            target_replay_discrepancies={d:[dict(unit=u,delta=vectors[d][u[0]][u[2]]-next(v['value'] for v in manifest['donors'][d]['native_values'] if tuple(v['unit'])==u)) for u in target] for d in ('digit5','digit4')})
        text=json.dumps(packet,indent=2)+'\n';(out/'public_discovery_calibration.json').write_text(text)
        result['calibration_sha256']=hashlib.sha256(text.encode()).hexdigest()
        result['controls_meet_declared_matching']=packet['controls_meet_declared_matching']
        result['control_comparison_scope']='magnitude matched' if packet['controls_meet_declared_matching'] else 'nearest weaker controls; matching failed; no magnitude-controlled claim'
        # Calibration identity/selection is now fixed, before ANY fresh outcomes.
        for case in manifest['cases']:
            before=counts['forward_examples'];pos=case['intervention_position']
            tu=[(l,pos,n) for l,p,n in target];cu=[(l,pos,n) for l,p,n in control_list]
            base_meta={}
            def top_observer(module,args,output):
                if base_meta:return
                first=len(case['prompt_ids'])-1;prefix=case['prefix_ids'];lg=output.logits[0]
                base_meta['prefix_top_ids']=[int(lg[first+i].argmax()) for i in range(len(prefix))]
                base_meta['digit_top_id']=int(lg[first+len(prefix)].argmax())
            observer=model.register_forward_hook(top_observer)
            try:full=col.score(case['prompt_ids'],[case['answer_a_ids'],case['answer_b_ids']])
            finally:observer.remove()
            native={};regimes={}
            base=col.score(case['boundary_ids'],[case['remaining_a_ids'],case['remaining_b_ids']],units=tu+cu,snapshot=native,regimes=regimes)
            row=dict(case_id=case['case_id'],group=case['group'],expected_truth_digit=case['expected_truth_digit'],
                expected_baseline_digit=case['expected_baseline_digit'],full_answer_baseline=full,
                conditional_baseline=base,baseline_metadata=base_meta,
                baseline_expected_digit_top=base_meta['digit_top_id']==(20 if case['expected_baseline_digit']==5 else 19),
                baseline_common_prefix_greedy=base_meta['prefix_top_ids']==case['prefix_ids'],
                conditional_minus_unconditional_bits=base['contrast_bits']-full['contrast_bits'],
                native_values=[dict(unit=u,value=float(native[u]),**regimes[u]) for u in tu+cu],effects={})
            result['rows'].append(row);save()
            for name,units in [('target',tu),('control',cu)]:
                zero=col.score(case['boundary_ids'],[case['remaining_a_ids'],case['remaining_b_ids']],units=units,zero=True)
                row['effects'][name+'_zero']=dict(zero,delta_bits=zero['contrast_bits']-base['contrast_bits'])
                for donor in ('digit5','digit4'):
                    values=manifest['donors'][donor]['native_values'] if name=='target' else packet['donor_native_values'][donor]
                    # Explicit original-discovery source position -> new shared-prefix position.
                    lookup={(u['unit'][0],u['unit'][2]):u['value'] for u in values}
                    source={u:torch.tensor(lookup[(u[0],u[2])]) for u in units}
                    patched=col.score(case['boundary_ids'],[case['remaining_a_ids'],case['remaining_b_ids']],units=units,source=source)
                    row['effects'][name+'_'+donor+'_patch']=dict(patched,delta_bits=patched['contrast_bits']-base['contrast_bits'],
                        donor='original_exposed_case0_'+donor,source_position=target[0][1],destination_position=pos)
                save()
            row['forward_examples']=counts['forward_examples']-before;save()
        result['status']='complete'
    except Exception as exc:
        result.update(status='incomplete',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        h.remove();col._original();save()
        if hookcount()!=original_hooks:raise RuntimeError('hook cleanup failed')
    return result


def _run_query_phase(model,tok,manifest,out, *, phase,public_path,prediction_receipt,
                     prediction_commit_sha,manifest_sha256):
    """Two cold phases around an external prediction commitment; original all mode is unchanged."""
    from neuron_builder import Neurons
    from native_prospective import Collector
    if phase not in ('public','gold'):raise ValueError('phase must be all/public/gold')
    if not isinstance(manifest_sha256,str) or len(manifest_sha256)!=64:raise ValueError('loader must supply frozen manifest byte SHA256')
    public=None;public_sha=None
    if phase=='gold':
        # Validate actual96 prediction bytes BEFORE any target-model forward.
        blob=Path(public_path).read_bytes();public=json.loads(blob);public_sha=hashlib.sha256(blob).hexdigest()
        receipt_path=Path(prediction_receipt);receipt=json.loads(receipt_path.read_text())
        responses=(receipt_path.parent/'responses.jsonl').read_bytes()
        expected={f'q{i:02d}_{arm}' for i in range(32) for arm in ('true','absent','mismatched')}
        observed=[json.loads(line) for line in responses.decode().splitlines()]
        if (receipt.get('requests')!=96 or len(observed)!=96 or
            {r['request_id'] for r in observed}!=expected or
            hashlib.sha256(responses).hexdigest()!=prediction_commit_sha or
            receipt.get('responses_sha256')!=prediction_commit_sha):
            raise ValueError('96 committed prediction coverage/SHA required before gold')
        if (receipt.get('panel_sha256')!=manifest_sha256 or
            receipt.get('public_observations_sha256')!=public_sha or
            public.get('manifest_sha256')!=manifest_sha256 or public.get('manifest')!=manifest or
            public.get('schema')!='lie_native_query_public_v1' or public.get('status')!='complete'):
            raise ValueError('prediction/public/manifest binding mismatch')
    if any(p.requires_grad for p in model.parameters()):raise ValueError('loader must freeze original weights')
    nr=Neurons(model,tok);limit=manifest[phase+'_forward_example_cap']
    col=Collector(nr,limit,manifest['target']);target=[tuple(u) for u in manifest['target_units']]
    replay_tolerance=manifest.get('public_replay_tolerance',dict(atol=1e-3,rtol=1e-2))
    if any(type(replay_tolerance[k]) not in (int,float) or not math.isfinite(replay_tolerance[k]) or replay_tolerance[k]<0 for k in ('atol','rtol')):
        raise ValueError('finite nonnegative public replay tolerances required')
    if public is not None:
        if (len(public['rows'])!=len(manifest['cases']) or
            {p['case_id'] for p in public['rows']}!={c['case_id'] for c in manifest['cases']}):
            raise ValueError('public32 query coverage required')
        for case,packet in zip(manifest['cases'],public['rows']):
            expected_units=[[l,case['intervention_position'],n] for l,p,n in target]
            if (packet['case_id']!=case['case_id'] or packet['prompt_ids']!=case['boundary_ids'] or
                packet['frozen_target_identity']!=manifest['target'] or
                [r['unit'] for r in packet['neurons']]!=expected_units):
                raise ValueError('public native unit/query/target binding mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();counts=dict(forward_examples=0,forward_calls=0,backward_examples=0)
    hookcount=lambda:sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
    before_hooks=hookcount()
    def meter(module,args,kwargs):
        col._original();ids=kwargs.get('input_ids',args[0] if args else None);b=int(ids.shape[0])
        if counts['forward_examples']+b>limit:raise RuntimeError('phase physical forward cap')
        if time.perf_counter()-started>manifest['walltime_cap_seconds']:raise RuntimeError('phase scientific walltime cap')
        counts['forward_examples']+=b;counts['forward_calls']+=1
    h=nr.core.model.register_forward_pre_hook(meter,with_kwargs=True)
    result=dict(schema='lie_native_query_public_v1' if phase=='public' else 'lie_native_query_gold_v1',
        manifest=manifest,manifest_sha256=manifest_sha256,status='started',rows=[],counts=counts,hooks_before=before_hooks)
    if phase=='gold':result.update(public_observations_sha256=public_sha,prediction_commit_sha256=prediction_commit_sha,
                                  public_replay_tolerance=replay_tolerance,public_state_replay_status='comparable')
    def save():
        result.update(seconds=time.perf_counter()-started,hooks_current=hookcount())
        (out/('public_observations.json' if phase=='public' else 'result.json')).write_text(json.dumps(result,indent=2)+'\n')
    try:
        for index,case in enumerate(manifest['cases']):
            pos=case['intervention_position'];units=[(l,pos,n) for l,p,n in target]
            if phase=='public':
                packet=col.prepare_query_evidence(case_id=case['case_id'],prompt_ids=case['boundary_ids'],
                    answer_a_ids=case['remaining_a_ids'],answer_b_ids=case['remaining_b_ids'],units=units,
                    readout_token_ids=tuple(manifest['readout_token_ids']))
                result['rows'].append(packet);save();continue
            # Same full input gives both prefix logps and conditional suffix
            # logps. Read output scalars only, without extra model forwards.
            prefix_logps=[];meta={}
            def observe(module,args,output):
                first=len(case['prompt_ids'])-1;prefix=case['prefix_ids'];lg=output.logits[0]
                logits=lg[first:first+len(prefix)].float()
                ids=torch.tensor(prefix,device=logits.device)
                prefix_logps.append(float(logits.log_softmax(-1).gather(-1,ids[:,None]).sum()))
                if not meta:
                    meta.update(prefix_top_ids=logits.argmax(-1).tolist(),digit_top_id=int(lg[first+len(prefix)].argmax()))
            observer=model.register_forward_hook(observe);native={}
            try:
                base=col.score(case['boundary_ids'],[case['remaining_a_ids'],case['remaining_b_ids']],units=units,snapshot=native)
            finally:observer.remove()
            full=dict(a_logp_nats=base['a_logp_nats']+prefix_logps[0],b_logp_nats=base['b_logp_nats']+prefix_logps[1])
            full['contrast_bits']=(full['a_logp_nats']-full['b_logp_nats'])/math.log(2)
            packet=public['rows'][index];old={tuple(r['unit']):r['activation'] for r in packet['neurons']}
            current={u:float(native[u]) for u in units}
            replay_failures=[dict(unit=u,public_activation=old[u],current_activation=current[u],
                                 absolute_error=abs(current[u]-old[u]),
                                 allowed_error=replay_tolerance['atol']+replay_tolerance['rtol']*abs(old[u]))
                             for u in units if abs(current[u]-old[u]) > replay_tolerance['atol']+replay_tolerance['rtol']*abs(old[u])]
            # Flag BEFORE the edit; retain measured zero outcomes without retries.
            # Noncomparable state cannot certify the public native evidence test.
            if replay_failures:result['public_state_replay_status']='noncomparable_public_state'
            edited=col.score(case['boundary_ids'],[case['remaining_a_ids'],case['remaining_b_ids']],units=units,zero=True)
            row=dict(case_id=case['case_id'],group=case['group'],expected_truth_digit=case['expected_truth_digit'],
                expected_baseline_digit=case['expected_baseline_digit'],conditional_baseline=base,
                full_answer_baseline=full,baseline_metadata=meta,
                effects=dict(target_zero=dict(edited,delta_bits=edited['contrast_bits']-base['contrast_bits'])),
                public_activation_max_abs_replay_error=max(abs(current[u]-old[u]) for u in units),
                public_state_replay_comparable=not replay_failures,public_state_replay_failures=replay_failures,forward_examples=4)
            result['rows'].append(row);save()
        result['status']='complete'
    except Exception as exc:
        result.update(status='incomplete',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        h.remove();col._original();save()
        if hookcount()!=before_hooks:raise RuntimeError('phase hook cleanup failed')
    return result
