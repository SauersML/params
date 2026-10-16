"""Frozen native16 transfer. No search, generation, reader, target loading or labels."""
import hashlib,json,math,time
from pathlib import Path
import torch


def run(model,tok,manifest,out):
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
