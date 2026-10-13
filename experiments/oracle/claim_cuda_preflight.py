"""Opt-in CUDA parity probe; canonical CPU remainder bytes are transferred intact.

No search or target selection. Task subsets partition the fixed twelve literal
TRAIN edit worlds across independent workers; all tasks together cost36forwards.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from claim_direct_teacher import SELECTED, SITE, sha

EXECUTOR='aeaf7c1ff934d7b2be4225cd91a25ee86a1606de'


def tasks_of(value):
    tasks=tuple(value.split(','))
    if not tasks or len(set(tasks))!=len(tasks) or any(t not in SELECTED for t in tasks):raise ValueError('unique fixed TRAIN task subset required')
    return tasks


def hashes(nat):
    import torch
    tensors={**dict(nat.target.state_dict())}
    for n in nat.names:
        for k in ('W','U','V','delta_T'):tensors[n+':'+k]=getattr(nat.target.site(n),k)
    return {n:hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n,t in tensors.items()}


def transfer(nat):
    import torch
    before=hashes(nat)
    nat.vpd.clear();nat._refs.clear();nat._refs_bytes=0;nat._rec=(None,None,{})
    nat._consts.clear();nat._lengths.clear()
    nat.target.to('cuda')
    for n in nat.names:
        st=nat.target.site(n)
        for k in ('U','V','delta_T'):setattr(st,k,getattr(st,k).to('cuda'))
    nat.dev='cuda'
    if hashes(nat)!=before:raise RuntimeError('canonical CPU tensor bytes changed on CUDA transfer')
    return before


def kl(p,q):
    import torch
    if p.shape!=q.shape or not torch.isfinite(p).all() or not torch.isfinite(q).all():raise RuntimeError('invalid distribution')
    a,b=p.double().cpu(),q.double().cpu()
    for x in (a,b):
        if float(x.logsumexp(-1).abs().max())>1e-4:raise RuntimeError('unnormalized distribution')
    a-=a.logsumexp(-1,keepdim=True);b-=b.logsumexp(-1,keepdim=True)
    result=float((a.exp()*(a-b)).sum())/math.log(2)
    if result < -1e-10 or not math.isfinite(result):raise RuntimeError('invalid KL')
    return max(0.,result)


def run(repo,payload,out,tasks,barrier=False):
    import torch
    from safetensors.torch import save_file
    if out.exists():raise ValueError('fresh immutable output required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    protocol=json.loads((payload/'protocol.json').read_text())
    if protocol['executor_commit']!=EXECUTOR or (repo/'.executor_commit').read_text().strip()!=EXECUTOR:raise ValueError('executor mismatch')
    for name,value in protocol['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(payload/name)!=value:raise ValueError('payload mismatch')
    sys.path[:0]=[str(repo/'method/oracle'),str(repo/'method/vpd')]
    import native,goal_eval,vpd_model,mech,weight_family
    runner_repo=Path(__file__).resolve().parents[2]
    for module in (native,goal_eval,vpd_model,mech,weight_family):
        relative=Path(module.__file__).resolve().relative_to(repo.resolve()).as_posix()
        if subprocess.check_output(['git','-C',str(runner_repo),'show',EXECUTOR+':'+relative])!=Path(module.__file__).read_bytes():raise ValueError('executor source mismatch')
    assets={'target':vpd_model.TARGET_DIR/'model_step_99999.safetensors','config':vpd_model.TARGET_DIR/'model_config.yaml','tokenizer':mech.VPD4L_TOKENIZER,'decomposition':native.UV}
    if vpd_model.VPD_PTH.exists() or {k:sha(p) for k,p in assets.items()}!=protocol['model_content_sha256']:raise ValueError('model source mismatch')
    panel=json.loads((payload/'panel.json').read_text());nat=native.Native('cpu');original=hashes(nat)
    prepared=[]
    for t in tasks:
        ids=panel['tasks'][t]['token_ids'];targets=panel['tasks'][t]['target_positions'];c,f=SELECTED[t]
        if not ids or targets!=[len(ids)-1] or any(type(i)is not int or not 0<=i<nat.target.wte.shape[0] for i in ids) or not 0<=c<nat.C[SITE]:raise ValueError('invalid fixed input')
        for scale in (1.,0.,.5,f):prepared.append((t,ids,targets,c,scale))
    out.mkdir();references=[];cpu_times=[];gpu_times=[];rows=[];started=time.monotonic()
    def clear(ge=None):
        nat.vpd.clear();nat._refs.clear();nat._refs_bytes=0;nat._rec=(None,None,{})
        if ge is not None:ge._xk.clear()
    for t,ids,targets,c,scale in prepared:
        with weight_family.applied(nat,{'kind':'subcomponent','what':(SITE,c),'k':scale}):
            clear();s=time.monotonic();p=torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[:,targets].float(),-1).clone();cpu_times.append(time.monotonic()-s);references.append(p)
        if hashes(nat)!=original:raise RuntimeError('CPU edit restoration failed')
    transfer(nat);ge=goal_eval.GoalEval(nat,semantics='c');torch.cuda.reset_peak_memory_stats()
    if barrier:
        if out.name not in ('worker0','worker1'):raise ValueError('two-worker named output required')
        (out/'cuda_ready.json').write_text(json.dumps({'tensor_hashes':hashes(nat),'tasks':tasks}))
        deadline=time.monotonic()+120
        while not all((out.parent/name/'cuda_ready.json').exists() for name in ('worker0','worker1')):
            if time.monotonic()>deadline:raise RuntimeError('other CUDA worker not ready; no continuation')
            time.sleep(.1)
    for i,(t,ids,targets,c,scale) in enumerate(prepared):
        with weight_family.applied(nat,{'kind':'subcomponent','what':(SITE,c),'k':scale},ge):
            clear(ge);torch.cuda.synchronize();s=time.monotonic()
            p=torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids],device='cuda'))[:,targets].float(),-1).clone()
            masks={n:torch.ones(len(ids),int(nat.C[n]),dtype=torch.bool,device='cuda') for n in nat.names}
            clear(ge);q=ge.runs(ids,targets,[masks],True).clone();torch.cuda.synchronize();gpu_times.append(time.monotonic()-s)
            parity=kl(references[i],p);integrity=kl(p,q)
            file=f'world_{i}.safetensors';save_file({'cpu_dense':references[i].contiguous(),'gpu_dense':p.cpu().contiguous(),'gpu_full_mask':q.cpu().contiguous()},str(out/file))
            row={'task':t,'scale':scale,'cpu_gpu_direct_kl_bits':parity,'full_start_direct_kl_bits':integrity,'tops':[int(x.argmax(-1).item()) for x in (references[i],p,q)],'file':file,'sha256':sha(out/file)};rows.append(row)
        if hashes(nat)!=original:raise RuntimeError('CUDA edit restoration failed')
    result={'executor_commit':EXECUTOR,'runner_sha256':sha(__file__),'runner_commit':subprocess.check_output(['git','-C',str(runner_repo),'rev-parse','HEAD'],text=True).strip(),'input_protocol_sha256':sha(payload/'protocol.json'),'tasks':tasks,'worlds':len(rows),'forwards':3*len(rows),'rows':rows,'cpu_dense_seconds':cpu_times,'gpu_dense_plus_full_seconds':gpu_times,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cuda_device':torch.cuda.get_device_name(),'restored_tensor_hashes':hashes(nat),'original_tensor_hashes':original,'seconds':time.monotonic()-started,'pass':all(r['cpu_gpu_direct_kl_bits']<=1e-7 and r['full_start_direct_kl_bits']<=1e-7 and len(set(r['tops']))==1 for r in rows),'scope':'parity only; no search; original CPU-computed remainder bytes; TF32 disabled'}
    if {k:sha(p) for k,p in assets.items()}!=protocol['model_content_sha256']:raise RuntimeError('assets changed')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('rows','original_tensor_hashes','restored_tensor_hashes')}),flush=True)
    if not result['pass']:raise RuntimeError('CUDA parity/integrity gate failed; no search permitted')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('repo','payload','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--tasks',default=','.join(SELECTED))
    p.add_argument('--two-worker-barrier',action='store_true')
    a=p.parse_args();run(a.repo,a.payload,a.out,tasks_of(a.tasks),a.two_worker_barrier)
