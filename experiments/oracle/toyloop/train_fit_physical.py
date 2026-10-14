"""Five preselected TRAIN graph mismatches; fresh references, no cache bypass."""
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from matched_graph_score import sha,write

CASES=('additive_vote_000','bigram_mlp_000','additive_vote_3100001','bigram_mlp_3100100','gate_3100501')

def checked_distribution(value,vocab):
    import torch
    if tuple(value.shape)!=(1,1,vocab) or not bool(torch.isfinite(value).all()):raise RuntimeError('invalid physical distribution shape/finiteness')
    if float(torch.logsumexp(value.double(),dim=-1).abs().max())>1e-5:raise RuntimeError('unnormalized physical log probabilities')
    return value

def checked_kl(value):
    if not math.isfinite(value):raise RuntimeError('nonfinite physical KL')
    return value

def validate_scope(contract):
    if [r['id'] for r in contract['cases']]!=list(CASES) or contract['worlds']!=548 or contract['forward_cap']!=2192:
        raise RuntimeError('fixed five-case physical scope changed')
    if sum(len(r['inputs'])*(1+len(r['panel'])) for r in contract['cases'])!=548:
        raise RuntimeError('world coverage changed')
    for row in contract['cases']:
        if len(row['inputs'])!=4 or row['reference_key_prefix'] not in ('p_','reference_'):
            raise RuntimeError('original four-input/reference format required')
        if len({tuple(op) for op in row['panel']})!=len(row['panel']):raise RuntimeError('duplicate edit')
        if any(not ids or any(type(i)is not int or i<0 for i in ids) for ids in row['inputs']):raise RuntimeError('invalid original input')
        if any(len(op)!=4 or op[0] not in ('head','neuron') or type(op[1])is not int or type(op[2])is not int or op[3] not in (0.,.5) for op in row['panel']):raise RuntimeError('invalid original panel')


def run(args):
    if sha(args.contract)!=args.contract_sha256 or args.out.exists():raise RuntimeError('frozen contract/fresh output required')
    contract=json.loads(args.contract.read_text())
    validate_scope(contract)
    for name,expected in contract['files'].items():
        if sha(args.payload/name)!=expected:raise RuntimeError('frozen input changed')
    for root,field in ((args.source,'executor_files'),(args.support,'support_files')):
        for name,expected in contract[field].items():
            if sha(root/name)!=expected:raise RuntimeError('physical source changed')
    if sha(__file__)!=contract['driver_sha256']:raise RuntimeError('driver changed')
    sys.path[:0]=[str(args.source/p) for p in ('experiments/oracle/toyloop','experiments/oracle/toys','method/oracle','method/vpd')]
    import torch,circuit,planted,native,goal_eval
    sys.path.insert(0,str(args.support/'experiments/oracle/toyloop'))
    from head_counterfactual import scaled_head,raw_kl_bits
    from reward_interventions import scaled_neuron,clear_intervention_caches
    from intervention_teacher import tensor_digest
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(1);torch.set_grad_enabled(False);args.out.mkdir();write(args.out/'contract.json',contract)
    forwards=0;allrows=[]
    for case in contract['cases']:
        blob=torch.load(args.payload/case['weights'],map_location='cpu',weights_only=True)
        toy=SimpleNamespace(sd=blob['sd'],cfg=blob['cfg'],meta={});nat=planted.native_of(toy.sd,toy.cfg)
        toy.meta['decomposition']={n:(nat.target.site(n).U,nat.target.site(n).V,nat.units[n]) for n in nat.names}
        def state():return {'model':{n:tensor_digest(v) for n,v in nat.target.state_dict().items()},'basis':{n:{k:tensor_digest(getattr(nat.target.site(n),k)) for k in ('W','U','V','delta_T')} for n in nat.names},'units':{n:tensor_digest(nat.units[n]) for n in nat.names}}
        original=state();write(args.out/(case['id']+'.original.json'),original)
        archived_binding=case['archived_binding']
        if any(original[key]!=archived_binding[key] for key in ('model','basis','units')):
            raise RuntimeError('historical original model/basis/unit identity mismatch')
        if toy.cfg!=archived_binding['cfg'] or {k:tensor_digest(v) for k,v in toy.sd.items()}!=archived_binding['source_weights']:
            raise RuntimeError('original model configuration/source weights mismatch')
        if any(max(ids)>=nat.target.wte.shape[0] for ids in case['inputs']):raise RuntimeError('input outside native vocabulary')
        for kind,layer,unit,scale in case['panel']:
            if not 0<=layer<nat.target.n_layer or not 0<=unit<(nat.target.n_head if kind=='head' else nat.target.site(f'h.{layer}.mlp.down_proj').W.shape[1]):raise RuntimeError('edit outside original architecture')
        archived=load_file(str(args.payload/case['references']))
        reference_keys=[case['reference_key_prefix']+str(i) for i in range(len(case['inputs'])*(1+len(case['panel'])))]
        if set(archived)!=set(reference_keys) or len(case['reference_sha256'])!=len(reference_keys):raise RuntimeError('archived edit-major/input-minor coverage mismatch')
        for key,expected in zip(reference_keys,case['reference_sha256']):
            if tensor_digest(archived[key])!=expected:raise RuntimeError('archived producer reference order/hash mismatch')
            checked_distribution(archived[key],nat.target.wte.shape[0])
        graphs={};invalid={}
        for arm in ('teacher','sample'):
            parsed=circuit.parse(case[arm]);graphs[arm]=[];invalid[arm]=[]
            for ids in case['inputs']:
                try:
                    if not parsed.ok or parsed.bad:raise ValueError('invalid saved circuit')
                    graph=circuit.unit_graph(parsed,toy,ids);reason=None
                except ValueError as exc:graph=None;reason=str(exc)
                if arm=='teacher' and graph is None:raise RuntimeError('frozen teacher not executable')
                graphs[arm].append(graph);invalid[arm].append(reason)
        full=[native.Graph({(n,p,c) for n in nat.names for p in range(len(ids)) for c in range(int(nat.C[n]))},complete=True) for ids in case['inputs']]
        comparisons=[]
        for edit_index,operations in enumerate([[]]+[[op] for op in case['panel']]):
            with contextlib.ExitStack() as stack:
                for kind,layer,unit,scale in operations:stack.enter_context((scaled_head if kind=='head' else scaled_neuron)(nat,layer,unit,scale))
                for input_index,ids in enumerate(case['inputs']):
                    target=[len(ids)-1];clear_intervention_caches(nat)
                    p=checked_distribution(nat.reference([ids],target).detach().clone(),nat.target.wte.shape[0]);forwards+=1
                    old=archived[case['reference_key_prefix']+str(edit_index*len(case['inputs'])+input_index)]
                    delta=float((p-old).abs().max());drift=checked_kl(raw_kl_bits(old,p));same_top=int(old.argmax())==int(p.argmax())
                    comparisons.append(dict(maxabs_logp=delta,raw_kl_bits=drift,top_equal=same_top))
                    if delta>1e-5 or abs(drift)>1e-6 or not same_top:
                        save_file({'p':p.contiguous(),'archived_p':old.contiguous()},str(args.out/'reference_failure.safetensors'))
                        write(args.out/'reference_failure.json',dict(case=case['id'],edit_index=edit_index,input=input_index,**comparisons[-1]))
                        raise RuntimeError('archived reference portability guard failed')
                    clear_intervention_caches(nat);f=goal_eval.GoalEval(nat).graph_ref(ids,target,full[input_index],live=True,norms={});forwards+=1
                    checked_distribution(f,nat.target.wte.shape[0]);integrity=checked_kl(raw_kl_bits(p,f))
                    if integrity>.001:raise RuntimeError('full architecture integrity failed')
                    tensors={'p':p.contiguous(),'archived_p':old.contiguous(),'full':f.contiguous()}
                    world=[]
                    for arm in ('teacher','sample'):
                        graph=graphs[arm][input_index];clear_intervention_caches(nat)
                        q=None if graph is None else goal_eval.GoalEval(nat).graph_ref(ids,target,graph,live=False)
                        forwards+=int(q is not None)
                        if q is not None:checked_distribution(q,nat.target.wte.shape[0])
                        error=None if q is None else checked_kl(raw_kl_bits(p,q))
                        if q is not None:tensors[arm]=q.contiguous()
                        world.append(dict(case=case['id'],arm=arm,edit_index=edit_index,input=input_index,operations=operations,valid=q is not None,invalid_reason=invalid[arm][input_index],kl_bits=None if error is None else max(0.,error),raw_kl_bits=error,top=None if q is None else int(q.argmax()),reference_top=int(p.argmax()),integrity_raw_kl_bits=integrity))
                    filename=f'{case["id"]}_{edit_index}_{input_index}.safetensors';save_file(tensors,str(args.out/filename))
                    with (args.out/'results.jsonl').open('a') as stream:
                        for row in world:
                            row.update(file=filename,sha256=sha(args.out/filename));allrows.append(row)
                            stream.write(json.dumps(row)+'\n')
            clear_intervention_caches(nat)
            if state()!=original:raise RuntimeError('literal edits failed exact restoration')
        write(args.out/(case['id']+'.restored.json'),state());write(args.out/(case['id']+'.reference_comparison.json'),comparisons)
    if forwards>2192 or len(allrows)!=1096:raise RuntimeError('physical bound/coverage failed')
    write(args.out/'results.json',allrows)
    summary={}
    for case in CASES:
        summary[case]={}
        for arm in ('teacher','sample'):
            values=[r for r in allrows if r['case']==case and r['arm']==arm];valid=[r for r in values if r['valid']]
            summary[case][arm]=dict(worlds=len(values),valid=len(valid),worst_kl_bits=max((r['kl_bits'] for r in valid),default=None),top_agreement=sum(r['valid'] and r['top']==r['reference_top'] for r in values)/len(values),all_worlds_pass=all(r['valid'] and r['kl_bits']<=.001 for r in values))
    write(args.out/'complete.json',dict(forwards=forwards,summary=summary,teacher_reproduction_pass=all(summary[c]['teacher']['all_worlds_pass'] for c in CASES),scope='five TRAIN graph-fit mismatches; finite original individual panel; no transfer/general fidelity certificate'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('contract','payload','source','support','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--contract-sha256',required=True);run(p.parse_args())
