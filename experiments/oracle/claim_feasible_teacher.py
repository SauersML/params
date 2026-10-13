"""Opt-in fixed TRAIN site-mask refinement from literal full-model execution.

All architectural connections remain complete. Dense remainders remain live;
this reduces rank-one component placements, not dense remainder sites or FLOPs.
No intermediate dense trace, baseline anchoring, importance selector, or oracle.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import subprocess
from claim_direct_teacher import SELECTED, SITE, sha
MAX_SITES=24
WORST_KL=.001
FULL_INTEGRITY_MAX=1e-7

def reachable_positions(site,length,targets,layers):
    """Architecture reachability, independent of weights/answers/importance."""
    layer=int(site.split('.')[1]);kind=site.split('.')[-1]
    return tuple(sorted(set(targets))) if layer==layers-1 and kind in ('q_proj','o_proj','c_fc','down_proj') else tuple(range(max(targets)+1))

def search_sites(order,full,teacher,evaluate,count,emit=None,full_start_max=WORST_KL):
    order=tuple(order)
    if len(order)>MAX_SITES or len(set(order))!=len(order) or set(order)!=set(full) or set(full)!=set(teacher):raise ValueError('fixed unique site universe required')
    def measure(masks,label):
        values=evaluate(masks,label)
        if len(values)!=4 or any(not math.isfinite(x) or x<0 for x in values):raise RuntimeError('four finite direct KL measurements required')
        return {'direct_kl_bits':values,'worst_direct_kl_bits':max(values),'counts':count(masks)}
    current=dict(full);start=measure(current,'full_start');attempts=[];accepted=[]
    if start['worst_direct_kl_bits']>full_start_max:return {'status':'infeasible_literal_full_start','full_start_max':full_start_max,'masks':None,'start':start,'attempts':attempts,'accepted_sites':accepted}
    selected=start
    for site in order:
        proposed={**current,site:teacher[site]};counts=count(proposed)
        if counts['reachable_component_placements']>=selected['counts']['reachable_component_placements']:
            row={'site':site,'accepted':False,'status':'no_output_reachable_reduction','counts':counts}
        else:
            row={'site':site,**measure(proposed,'replace_'+site)}
            row['accepted']=row['worst_direct_kl_bits']<=WORST_KL
            row['status']='accepted' if row['accepted'] else 'direct_fidelity_rejection'
            if row['accepted']:current=proposed;selected=row;accepted.append(site)
        attempts.append(row)
        if emit:emit(row)
    return {'status':'full_model_no_compression' if not accepted else 'bounded_site_refinement','masks':current,'start':start,'selected':selected,'attempts':attempts,'accepted_sites':accepted}

def run(repo,payload,out,prior=None,resume=None):
    import torch
    from safetensors.torch import save_file
    sys.path[:0]=[str(repo/'method/oracle'),str(repo/'method/vpd')]
    import native,goal_eval,vpd_model,mech,weight_family
    if out.exists():raise ValueError('immutable new output required')
    contract=json.loads((payload/'protocol.json').read_text())
    actual_executor=(repo/'.executor_commit').read_text().strip()
    if actual_executor!=contract['executor_commit']:raise ValueError('executor commit mismatch')
    runner_repo=Path(__file__).resolve().parents[2]
    runner_commit=subprocess.check_output(['git','-C',str(runner_repo),'rev-parse','HEAD'],text=True).strip()
    for module in (native,goal_eval,vpd_model,mech,weight_family):
        path=Path(module.__file__).resolve()
        relative=path.relative_to(repo.resolve()).as_posix()
        original=subprocess.check_output(['git','-C',str(runner_repo),'show',actual_executor+':'+relative])
        if original!=path.read_bytes():raise ValueError('executor source mismatch: '+relative)
    for name,expected in contract['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(payload/name)!=expected:raise ValueError('frozen input mismatch')
    panel=json.loads((payload/'panel.json').read_text())
    if set(panel['tasks'])!=set(SELECTED):raise ValueError('fixed three TRAIN contexts required')
    assets={'target':vpd_model.TARGET_DIR/'model_step_99999.safetensors','config':vpd_model.TARGET_DIR/'model_config.yaml','tokenizer':mech.VPD4L_TOKENIZER,'decomposition':vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV}
    if {k:sha(p) for k,p in assets.items()}!=contract['model_content_sha256']:raise ValueError('model assets changed')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    nat=native.Native('cpu');ge=goal_eval.GoalEval(nat,semantics='c')
    order=tuple(nat.names)
    if len(order)!=24 or order!=tuple(vpd_model.site_names(nat.target.n_layer)):raise ValueError('canonical 24-site order required')
    def state():
        return {n+':'+k:__import__('hashlib').sha256(getattr(nat.target.site(n),k).detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n in order for k in ('W','U','V','delta_T')}
    original_state=state();prepared={}
    for task,(component,factor) in SELECTED.items():
        ids=panel['tasks'][task]['token_ids'];targets=panel['tasks'][task]['target_positions']
        if not ids or any(type(i)is not int or not 0<=i<nat.target.wte.shape[0] for i in ids) or targets!=[len(ids)-1] or not 0<=component<nat.C[SITE]:raise ValueError('invalid frozen TRAIN input/edit')
        answer=json.loads((payload/(task+'.teacher.json')).read_text());graph,report=goal_eval.circuit(answer,targets,nat.C,len(ids))
        if any(report[k] for k in ('invalid_nodes','invalid_edges','invalid_variables')) or getattr(graph,'variables',[]):raise ValueError('invalid component teacher')
        nodes=graph.node_set()
        if any(c<0 for _,_,c in nodes):raise ValueError('whole-site blocks unsupported as component proposals')
        fixed={n:torch.zeros(len(ids),int(nat.C[n]),dtype=torch.bool) for n in order}
        for n,p,c in nodes:fixed[n][p,c]=True
        prepared[task]=(ids,targets,fixed,report,len(answer.get('nodes',[])))
    hierarchy=None;prior_results=None;resume_results=None;forward_cap=336
    if resume is not None and prior is None:raise ValueError('continuation requires original coarse ranking provenance')
    if prior is not None:
        import claim_hierarchical_teacher as hierarchy
        previous=json.loads((prior/'contract.json').read_text())
        prior_results=json.loads((prior/'results.json').read_text())
        completed=json.loads((prior/'complete.json').read_text())
        if previous['input_provenance']!=contract or previous['executor_commit']!=actual_executor or previous['original_state']!=original_state or completed['restored_state']!=original_state or previous['selected_edits']!=json.loads(json.dumps(SELECTED)):
            raise ValueError('prior ranking provenance mismatch')
        if set(prior_results)!=set(SELECTED) or any(prior_results[t]['status']!='full_model_no_compression' or len(prior_results[t]['attempts'])!=24 for t in SELECTED):raise ValueError('complete rejected coarse run required')
        forward_cap=432
        if resume is not None:
            rc=json.loads((resume/'contract.json').read_text());rd=json.loads((resume/'complete.json').read_text());resume_results=json.loads((resume/'results.json').read_text())
            if rc['input_provenance']!=contract or rc['executor_commit']!=actual_executor or rc['original_state']!=original_state or rd['restored_state']!=original_state or rc['selected_edits']!=previous['selected_edits'] or rc['hierarchical_policy']['prior_results_sha256']!=sha(prior/'results.json') or set(resume_results)!=set(SELECTED):raise ValueError('continuation provenance mismatch')
            if any(resume_results[t]['status']!='bounded_hierarchical_refinement' or not resume_results[t]['cap_hit'] or len(resume_results[t]['attempts'])!=32 or max(resume_results[t]['final_direct_kl_bits'])>WORST_KL for t in SELECTED):raise ValueError('first bounded hierarchy completion required')
            forward_cap=444
    out.mkdir();started=time.monotonic();calls={'dense':0,'sparse':0};summaries={}
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2))
    write('contract.json',{'input_provenance':contract,'executor_commit':actual_executor,'runner_commit':runner_commit,'runner_sha256':sha(__file__),'helper_sha256':sha(Path(__file__).with_name('claim_direct_teacher.py')),'selected_edits':SELECTED,'components_per_site':{n:int(nat.C[n]) for n in order},'source_sha256':{m.__name__:sha(m.__file__) for m in (native,goal_eval,vpd_model,weight_family)},'site_order':order,'max_sites_per_context':24,'max_forwards':forward_cap,'hierarchical_policy':None if hierarchy is None else {'sha256':sha(hierarchy.__file__),'max_new_proposals':32,'prior_contract_sha256':sha(prior/'contract.json'),'prior_results_sha256':sha(prior/'results.json'),'prior_complete_sha256':sha(prior/'complete.json'),'ranking':'inherited rejected direct error per eligible reachable removal; heuristic'},'full_start_integrity_max':FULL_INTEGRITY_MAX,'worst_direct_kl_max':WORST_KL,'live_remainders':True,'mask_links':'all architecture connections; own routing/norms; original teacher edges ignored','teacher_masks':'parsed executable fixed joint_train component nodes; stranded source nodes reported','count_scope':'mask-enabled and output-reachable rank-one placements; backend dense factor arithmetic is not pruned; all24 dense remainder sites stay on; no FLOP claim','remainder_off':'not measured; no remainder-free circuit certificate','original_state':original_state})
    if resume is not None:
        write('continuation.json',{'parent_contract_sha256':sha(resume/'contract.json'),'parent_results_sha256':sha(resume/'results.json'),'parent_complete_sha256':sha(resume/'complete.json'),'preserve_all_prior_decisions':True,'new_proposals_per_context':32,'max_forwards':444,'queue':'exact deterministic reconstruction from every accepted/rejected prior decision; no repeated proposals'})
    def clear():
        nat.vpd.clear();nat._refs.clear();nat._refs_bytes=0;nat._rec=(None,None,{});ge._xk.clear();nat.__dict__.pop('_sensitivity_kept',None)
    def kl(p,q):
        if p.shape!=q.shape or not torch.isfinite(p).all() or not torch.isfinite(q).all():raise RuntimeError('distribution shape/finite failure')
        values=[]
        for x in (p,q):
            x=x.double();z=x.logsumexp(-1,keepdim=True)
            if float(z.abs().max())>1e-4:raise RuntimeError('log probability normalization failure')
            values.append(x-z)
        p,q=values;v=float((p.exp()*(p-q)).sum())/math.log(2)
        if not math.isfinite(v) or v < -1e-10:raise RuntimeError('invalid KL')
        return max(0.,v)
    for task,(component,factor) in SELECTED.items():
        ids,targets,fixed,report,source_nodes=prepared[task];scales=(1.,0.,.5,factor)
        full={n:torch.ones_like(fixed[n]) for n in order};references=[]
        for scale in scales:
            with weight_family.applied(nat,{'kind':'subcomponent','what':(SITE,component),'k':scale},ge):
                clear();p=torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[:,targets].float(),-1).clone();calls['dense']+=1;references.append(p)
        def count(masks):
            return {'masked_component_placements':sum(int(m.sum()) for m in masks.values()),'reachable_component_placements':sum(int(m[list(reachable_positions(n,len(ids),targets,nat.target.n_layer))].sum()) for n,m in masks.items()),'dense_remainder_sites':24}
        def evaluate(masks,label,uncached=False):
            errors=[];rows=[]
            for index,scale in enumerate(scales):
                with weight_family.applied(nat,{'kind':'subcomponent','what':(SITE,component),'k':scale},ge):
                    clear();p=references[index]
                    if uncached:
                        pnew=torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[:,targets].float(),-1).clone();calls['dense']+=1
                        if not torch.equal(p,pnew):raise RuntimeError('reference cache/literal model mismatch')
                    clear();q=ge.runs(ids,targets,[masks],True).clone();calls['sparse']+=1;error=kl(p,q);errors.append(error)
                    filename=f'{task}_{label}_{index}.safetensors';save_file({'full_reference':p.contiguous(),'candidate':q.contiguous()},str(out/filename))
                    rows.append({'scale':scale,'direct_kl_bits':error,'reference_top':int(p.argmax(-1).item()),'candidate_top':int(q.argmax(-1).item()),'file':filename,'sha256':sha(out/filename)})
            write(task+'_'+label+'.json',rows)
            if state()!=original_state:raise RuntimeError('weight/basis restoration failed')
            return errors
        attempts=[]
        def emit(row):attempts.append(row);write(task+'_attempts.json',attempts)
        if hierarchy is None:
            result=search_sites(order,full,fixed,evaluate,count,emit,full_start_max=FULL_INTEGRITY_MAX)
        else:
            rows=prior_results[task]['attempts'];errors={r['site']:r['worst_direct_kl_bits'] for r in rows}
            if set(errors)!=set(order) or any(r['accepted'] or not math.isfinite(errors[r['site']]) or errors[r['site']]<=WORST_KL for r in rows):raise ValueError('invalid prior rejections')
            positions={n:reachable_positions(n,len(ids),targets,nat.target.n_layer) for n in order}
            previous_result=None if resume_results is None else resume_results[task]
            result=hierarchy.tensor_refine(order,full,fixed,evaluate,count,emit,positions,errors,None if previous_result is None else previous_result['attempts'],None if previous_result is None else previous_result['recipe'])
        masks=result.pop('masks')
        result.update(parser_report=report,source_teacher_nodes=source_nodes,teacher_executable_placements=sum(int(m.sum()) for m in fixed.values()))
        if masks is not None:
            final=evaluate(masks,'uncached_final',True)
            if max(final)>WORST_KL:raise RuntimeError('final fidelity failed')
            result['final_direct_kl_bits']=final
            if hierarchy is None:result['recipe']={n:{'mode':'fixed_teacher' if n in result['accepted_sites'] else 'full','shape':list(m.shape),'fixed_teacher_nodes':[[int(p),int(c)] for p,c in fixed[n].nonzero().tolist()] if n in result['accepted_sites'] else None} for n,m in masks.items()}
        summaries[task]=result;write('results.json',summaries)
        print(json.dumps({'task':task,'status':result['status'],'accepted_sites':result['accepted_sites'],'start':result['start'],'selected':result.get('selected'),'actual_forward_counts':dict(calls)}),flush=True)
    if state()!=original_state or {k:sha(p) for k,p in assets.items()}!=contract['model_content_sha256']:raise RuntimeError('final model/basis/assets changed')
    if sum(calls.values())>forward_cap:raise RuntimeError('forward cap exceeded')
    write('complete.json',{'actual_forward_counts':calls,'max_forwards':forward_cap,'restored_state':state(),'seconds':time.monotonic()-started,'scope':'adaptive TRAIN inputs and previously selected final-down edits only; live dense remainders remain on'})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','payload','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior',type=Path)
    p.add_argument('--resume',type=Path)
    a=p.parse_args();run(a.repo,a.payload,a.out,a.prior,a.resume)
