"""Explicitly exposed lie-organism discovery; imports the existing finder, no new search.
run(model,tok,manifest,out,search=False) accepts an already pinned immutable target.
No model load on import. Root supplies loading/launch and verifies external hashes.
"""
from pathlib import Path
import json, math, time, random
import torch

class BudgetStop(RuntimeError): pass


def run(model, tok, manifest, out, *, search=False):
    from circuit_finder import Patcher, cut
    from neuron_builder import Neurons
    from native_prospective import Collector
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(model_part.training for model_part in model.modules()):
        raise ValueError('full organism must be in eval mode')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('loader must freeze original target weights before discovery')
    core=model.get_base_model() if hasattr(model,'get_base_model') else model
    started=time.perf_counter();counts={'forward_calls':0,'forward_examples':0}
    limit=manifest['search_forward_example_cap'] if search else manifest['preflight_forward_example_cap']
    active_limit=limit
    def meter(module,args,kwargs):
        collector._original()
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None: raise ValueError('unmetered input representation')
        examples=int(ids.shape[0])
        if counts['forward_examples']+examples>active_limit: raise BudgetStop('forward-example cap')
        if time.perf_counter()-started>manifest['walltime_cap_seconds']:raise BudgetStop('walltime cap')
        counts['forward_calls']+=1;counts['forward_examples']+=examples
    def hook_count():
        return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
    hooks_before=hook_count()
    collector=Collector(Neurons(model,tok),manifest['search_forward_example_cap'],manifest['target'])
    # Meter the transformer body: this also counts Patcher.attends(), which
    # bypasses the outer CausalLM forward and would otherwise evade accounting.
    handle=core.model.register_forward_pre_hook(meter,with_kwargs=True)
    results=[]; search_results=[]; final_result={}
    def save():
        result=dict(schema='lie_exposed_discovery_v1',manifest=manifest,preflight=results,
                    searches=search_results,counts=dict(counts),seconds=time.perf_counter()-started,
                    interpretation='Exposed discovery only; neither fresh E3 repair evidence nor a semantic neuron certificate.',
                    hooks_before=hooks_before,hooks_current=hook_count(),
                    backward_accounting='No backward in preflight; completed Patcher.scores performs two backwards. Interrupted search backward count unverified.')
        final_result.clear();final_result.update(result)
        (out/'result.json').write_text(json.dumps(final_result,indent=2));return final_result
    try:
        # Every baseline is measured before any candidate search. No new cases
        # replace null/invalid cases. Order and eligibility rule are frozen.
        for case in manifest['cases']:
            before=dict(counts);a,b=case['full_false_ids'],case['full_true_ids']
            clean=collector.score(case['clean_prompt_ids'],[a,b])
            corrupt=collector.score(case['corrupt_prompt_ids'],[a,b])
            pt=Patcher(model,tok,rows=manifest['rows'])
            setup=pt.setup(case['clean_boundary_ids'],case['corrupt_boundary_ids'],answers=[case['false_digit_id']])
            # Independent unhooked repeat, same fixed finder batch shape, same exact context/contrast.
            repeat={}; repeat_top={}
            with torch.no_grad():
                for name in ('clean','corrupt'):
                    lp=pt._last(pt.ids[name],pt.rows)[0]
                    repeat_top[name]=int(lp.argmax())
                    repeat[name]=float(pt._contrast(lp[None])[0])-setup['contrast'][name]
            full_gap=clean['contrast_bits']-corrupt['contrast_bits']
            eligible=(repeat_top['clean']==case['false_digit_id'] and repeat_top['corrupt']==case['true_digit_id']
                      and clean['contrast_bits']>=manifest['min_gap_bits']
                      and corrupt['contrast_bits']<=-manifest['min_gap_bits']
                      and setup['gap']>=manifest['min_gap_bits'] and max(abs(v) for v in repeat.values())<=manifest['noop_atol_bits'])
            row=dict(case_id=case['case_id'],clean_full_sequence=clean,corrupt_full_sequence=corrupt,
                     full_sequence_clean_minus_corrupt_bits=full_gap,finder_single_digit_metric=setup,
                     repeat_delta_bits=repeat,repeat_top_token_ids=repeat_top,eligible_for_discovery=eligible,
                     status='eligible' if eligible else 'invalid_or_null_at_declared_boundary',
                     forward_examples=counts['forward_examples']-before['forward_examples'])
            results.append(row);save()
            del pt
        if search:
            # A single frozen first eligible discovery case, not best-outcome
            # selection. Every attempted search outcome is saved; never retry.
            selected=next((case for case,row in zip(manifest['cases'],results) if row['eligible_for_discovery']),None)
            if selected is not None:
                before=dict(counts)
                done={}
                pt=Patcher(model,tok,rows=manifest['rows'])
                entry=dict(case_id=selected['case_id'],status='started')
                search_results.append(entry);save()
                try:
                    pt.setup(selected['clean_boundary_ids'],selected['corrupt_boundary_ids'],answers=[selected['false_digit_id']])
                    scores=pt.scores()
                    # Preserve a fixed validation reserve, not a native-unit cap.
                    active_limit=limit-manifest['validation_forward_reserve']
                    done={}
                    found=cut(pt,scores,'neuron',[pt.T-1],done)
                    entry['readout_cut']={k:v for k,v in found.items() if k!='singles'}
                    entry['readout_cut']['singles']=[dict(unit=u,lost=v[0],restored=v[1]) for u,v in found['singles'].items()]
                    entry['status']='readout_cut_complete_only_not_full_circuit'
                    active_limit=limit
                    target=[tuple(u[1:]) for u in found['units']]
                    grouped={}
                    for l,p,n in target:grouped.setdefault((l,p),set()).add(n)
                    rng=random.Random(manifest['control_seed']); control=[]
                    for (l,p),used in sorted(grouped.items()):
                        available=[n for n in range(pt.D) if n not in used]
                        if len(available)<len(used):
                            entry['validation_unavailable']='Insufficient disjoint matchedcontrol population; no unit cap applied'
                            control=None;break
                        control.extend((l,p,n) for n in rng.sample(available,len(used)))
                    entry['control_units']=control
                    entry['control_matching']='count/layer/position matched random, NOT activation-magnitude matched'
                    entry['validation']=[]
                    if target and control is not None:
                        # Readout position maps by final boundary token, never
                        # silently into an earlier original-prompt position.
                        for case in manifest['cases']:
                            pos=len(case['clean_boundary_ids'])-1
                            units=[(l,pos,n) for l,p,n in target]
                            controls=[(l,pos,n) for l,p,n in control]
                            validation=collector.collect(case_id=case['case_id']+'_conditional_validation',
                                clean_ids=case['clean_boundary_ids'],corrupt_ids=case['corrupt_boundary_ids'],
                                answer_a_ids=case['full_false_ids'][2:],answer_b_ids=case['full_true_ids'][2:],
                                target_units=units,control_units=controls,paired_noising=True,
                                paired_min_gap_bits=manifest['validation_min_gap_bits'])
                            validation['scope']='Conditional58vs48given16; noising repairs false direct answer, denoising transfers false answer'
                            entry['validation'].append(validation);save()
                        entry['collateral_proxy']=[]
                        for case in manifest['collateral_cases']:
                            ids=case['prompt_ids'];pos=len(ids)-1
                            logs=[]
                            for units in ([],[(l,pos,n) for l,p,n in target],[(l,pos,n) for l,p,n in control]):
                                by={}
                                for l,p,n in units:by.setdefault(l,[]).append((p,n))
                                def zero(layer,h):
                                    if layer not in by:return h
                                    x=h.clone()
                                    for p,n in by[layer]:x[:,p,n]=0
                                    return x
                                with torch.no_grad(),collector.nr.hooks(zero):
                                    lp=model(input_ids=torch.tensor([ids],device=pt.dev),use_cache=False).logits[0,-1].float().log_softmax(-1)
                                logs.append(lp)
                            lp0=logs[0]
                            entry['collateral_proxy'].append(dict(case_id=case['case_id'],
                                target_kl_bits=float((lp0.exp()*(lp0-logs[1])).sum()/math.log(2)),
                                control_kl_bits=float((lp0.exp()*(lp0-logs[2])).sum()/math.log(2)),
                                original_top=int(lp0.argmax()),target_zero_top=int(logs[1].argmax()),control_zero_top=int(logs[2].argmax())))
                            save()
                except BudgetStop as exc:
                    entry.update(status='budget_stopped_incomplete',reason=str(exc),
                                 measured_blocks=[dict(block=k,lost=v[0],restored=v[1]) for k,v in done.items()])
                entry['forward_examples']=counts['forward_examples']-before['forward_examples']
        return save()
    except BudgetStop as exc:
        search_results.append(dict(status='preflight_budget_stopped_incomplete',reason=str(exc)))
        return save()
    finally:
        handle.remove()
        collector._original()
        save()
        if hook_count()!=hooks_before:
            raise RuntimeError("hook cleanup failed")
