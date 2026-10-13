"""Bounded new-TRAIN bank teacher adapter; execution requires separate GO.

Evaluation models and source-derived point targets never enter this search.
Every fixed case has an explicit outcome; failures cannot be replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

from learning_model_bank import GENERATOR_COMMIT, FAMILIES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def validate_bank(root, expected_sha, worker):
    if worker not in (0, 1) or sha(root/'manifest.json') != expected_sha:
        raise ValueError('exact bank receipt and worker0/1 required')
    manifest = json.loads((root/'manifest.json').read_text())
    ids = [f'{family}_{3100000+100*i+variant}' for i, family in enumerate(FAMILIES)
           for variant in (0, 1)]
    rows = [r for r in manifest['models'] if r['split']=='train']
    if [r['case'] for r in rows] != ids or manifest['generator_commit'] != GENERATOR_COMMIT:
        raise ValueError('fixed16 TRAIN cases/source required')
    selected = rows[8*worker:8*(worker+1)]
    for row in selected:
        if row['status'] != 'constructed_unmeasured':
            continue  # Retain failures without inventing a fixture.
        for name, digest in row['files'].items():
            if Path(name).name != name or sha(root/row['case']/name) != digest:
                raise ValueError('model bank file changed')
        bound = row['teacher_bound']
        if bound['max_attempts'] != 64 or bound['forward_cap'] != 69*bound['worlds']:
            raise ValueError('teacher forward contract changed')
    return manifest, selected


def execute_case(row, bank, out, modules):
    import torch
    from safetensors.torch import save_file
    render, planted, teacher, reward = modules
    root = bank/row['case']; record = json.loads((root/'record.json').read_text())
    if record['bank_role'] != 'train' or record['split'] != 'train':
        raise ValueError('evaluation role forbidden for supervision search')
    inputs = [e['ids'] for e in record['examples']]
    if len(inputs) != 4 or any(e['kind']!='ordinary' or e['target']!=len(e['ids'])-1
                               for e in record['examples']):
        raise ValueError('four frozen ordinary LAST inputs required')
    out.mkdir()
    write(out/'pre_native_contract.json',{'case':row['case'],'id':row['case'],
          'bank_role':'train','inputs':inputs,'worlds':row['teacher_bound']['worlds'],
          'forward_cap':row['teacher_bound']['forward_cap'],
          'weight_sha256':row['files']['weights.pt'],'record_sha256':row['files']['record.json'],
          'config_sha256':row['files']['config.json'],
          'source_sha256':{m.__name__:sha(m.__file__) for m in modules},
          'driver_sha256':sha(__file__),'point_control_read_before_search':False})
    data = torch.load(root/'weights.pt', map_location='cpu', weights_only=True)
    toy = render.Toy(row['case'], data['sd'], data['cfg'],
                     ['BOS']+[f't{i}' for i in range(1, data['sd']['wte.weight'].shape[0])],
                     inputs, None, {'split':'train','family':row['family'],'record':record})
    nat = planted.native_of(toy.sd, toy.cfg)
    neurons = sum(int(nat.target.site(f'h.{l}.mlp.down_proj').W.ne(0).any(dim=0).sum())
                  for l in range(nat.target.n_layer))
    worlds = 4*(1+2*(nat.target.n_layer*nat.target.n_head+neurons))
    if worlds != row['teacher_bound']['worlds'] or teacher.MAX_ATTEMPTS != 64:
        raise ValueError('actual architecture differs before first scoring')
    evidence=[]; attempts=[]
    before = teacher.binding(toy,nat,inputs,[])
    write(out/'original_source_binding.json',before)
    def retain(stage,index,world,p,q,complete):
        name=f'{stage}_{index}.safetensors'; tensors={'dense':p.contiguous(),'candidate':q.contiguous()}
        if complete is not None:tensors['full_integrity']=complete.contiguous()
        save_file(tensors,str(out/name))
        evidence.append({'stage':stage,'index':index,'world':world,'file':name,'sha256':sha(out/name)})
        write(out/'evidence.json',evidence)
    def attempt(value):
        attempts.append(value);write(out/'attempts.json',attempts)
    result=teacher.search_teacher(toy,nat=nat,texts=inputs,attempt_sink=attempt,
                                 evidence_sink=retain,exclude_zero_mlp_routes=True)
    cache=result.pop('cache')
    if len(cache['references'])!=worlds:raise RuntimeError('world receipt differs')
    # Freeze target before any point-control metadata is exposed to this process.
    write(out/'search.json',result)
    write(out/'reference_cache.json',{k:v for k,v in cache.items() if k!='references'})
    save_file({f'reference_{i}':v.contiguous() for i,v in enumerate(cache['references'])},
              str(out/'references.safetensors'))
    if result['target'] is not None:
        index=[0]
        def crosscheck(current,ids,world,p,q):
            if not torch.equal(p,cache['references'][index[0]]):
                raise RuntimeError('uncached dense differs from frozen reference')
            retain('uncached_final',index[0],world,p,q,None);index[0]+=1
        final=reward.score_wiring_interventions(toy,result['target'],nat=nat,texts=inputs,
                                               panel=cache['binding']['panel'],evidence_sink=crosscheck)
        if max(w['kl_bits'] for w in final['per_world'])>teacher.WORST_KL_MAX:
            raise RuntimeError('final uncached direct fidelity failed')
        for old,new in zip(result['selected']['score']['per_world'],final['per_world']):
            if old['top']!=new['top'] or abs(old['kl_bits']-new['kl_bits'])>1e-6:
                raise RuntimeError('cached selected world disagrees with uncached')
        write(out/'uncached_final.json',final)
    if before != teacher.binding(toy,nat,inputs,[]):raise RuntimeError('weights/basis not restored')
    write(out/'restored_source_binding.json',teacher.binding(toy,nat,inputs,[]))
    result['cache']=cache
    # Weight-only rendering performs no forward. The same prompt is used for
    # point and intervention supervision by the independent matched assembler.
    render.SECTIONS='weights'
    renderer=types.SimpleNamespace(FORMAT=render.FORMAT,__file__=render.__file__,
        render=lambda toy,**kwargs:render.render(toy,explain_summary_limits=True,**kwargs))
    training=teacher.training_row(toy,result,nat=nat,renderer=renderer)
    write(out/'training_row.json',training)
    used=sum(3 if e['stage']=='reference' else 2 if e['stage']=='uncached_final' else 1
             for e in evidence)
    if used>row['teacher_bound']['forward_cap']:raise RuntimeError('forward cap exceeded')
    complete={'case':row['case'],'status':result['status'],'target_sha256':result.get('target_sha256'),
              'actual_forwards':used,'forward_cap':row['teacher_bound']['forward_cap'],
              'worlds':worlds,'attempts':len(attempts),'rendering_forwards':0,
              'usable_training_row':training.get('usable',False),
              'bank_weight_sha256':row['files']['weights.pt'],
              'input_record_sha256':row['files']['record.json']}
    write(out/'complete.json',complete);return complete


def main():
    import torch
    p=argparse.ArgumentParser()
    for name in ('source','bank','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bank-sha256',required=True);p.add_argument('--worker',type=int,required=True)
    args=p.parse_args()
    manifest,rows=validate_bank(args.bank,args.bank_sha256,args.worker)
    if args.out.exists():raise ValueError('fresh immutable worker output required')
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(args.source/'experiments/oracle/toys'),
                  str(args.source/'method/oracle'),str(args.source/'method/vpd')]
    import planted, native, goal_eval, vpd_model, render, intervention_teacher as teacher
    import reward_exec as reward, head_counterfactual, reward_interventions, circuit
    import subprocess
    for module in (planted,native,goal_eval,vpd_model):
        path=Path(module.__file__).resolve();relative=path.relative_to(args.source.resolve())
        original=subprocess.check_output(['git','-C',str(args.source),'show',GENERATOR_COMMIT+':'+relative.as_posix()])
        if path.read_bytes()!=original:raise ValueError('fixed executor source changed')
    torch.set_num_threads(1);torch.set_grad_enabled(False);args.out.mkdir()
    contract={'bank_sha256':args.bank_sha256,'worker':args.worker,'cases':[r['case'] for r in rows],
              'forward_cap':sum(r.get('teacher_bound',{}).get('forward_cap',0) for r in rows),
              'executor_commit':GENERATOR_COMMIT,'source_sha256':{m.__name__:sha(m.__file__) for m in
               (planted,native,goal_eval,vpd_model,render,teacher,reward,head_counterfactual,reward_interventions,circuit)},
              'driver_sha256':sha(__file__),'exclude_zero_mlp_routes':True,'selection':'all fixed8 TRAIN cases; no replacements',
              'scope':'finite disclosed contexts/intervention-panel graph supervision; no English/heldout guarantee'}
    write(args.out/'contract.json',contract);statuses=[]
    for row in rows:
        status={'case':row['case'],'status':'error'}
        try:
            if row['status']!='constructed_unmeasured':status.update(status='construction_error',original=row)
            else:status=execute_case(row,args.bank,args.out/row['case'],(render,planted,teacher,reward))
        except Exception as exc:status.update(error_type=type(exc).__name__,error=str(exc))
        statuses.append(status);write(args.out/'status.json',statuses)
        print(json.dumps(status),flush=True)
    write(args.out/'complete.json',{'cases':statuses,'forward_cap':contract['forward_cap']})


if __name__=='__main__':main()
