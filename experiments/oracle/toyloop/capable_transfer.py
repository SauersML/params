"""Opt-in frozen BASE/checkpoint routing comparison; prepare is offline.

Run requires a root-reserved cap. It samples once per frozen prompt, never trains,
repairs answers, or retries an uncertain call. Native scoring is separate.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import capable_wiring as c

PACKAGES = ('tinker','tinker-cookbook','transformers','tokenizers')

def snapshot(root):
    return {str(p.relative_to(root)): c.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

def prepare(root, checkpoint, tokenizer, protocol, destination):
    if destination.exists(): raise ValueError('immutable preflight already exists')
    teacher=json.loads((root/'route_transfer/manifest.json').read_text())
    fit=json.loads((root/'token_fit.json').read_text())
    if teacher.get('teacher_integrity')!='PASS' or len(teacher['cases'])!=8:
        raise ValueError('eight physically validated cases required')
    names=[x['id']+'.prompt.txt' for x in teacher['cases']]
    selected=[r for r in fit['rows'] if r['id'] in names]
    if len(selected)!=8 or len(set(names))!=8: raise ValueError('eight unique fitted prompts required')
    state=json.loads(checkpoint.read_text())
    if set(state)!={'state'} or not isinstance(state['state'],str) or not state['state'].startswith('tinker://'):
        raise ValueError('expected private saved-state receipt')
    m={'schema':1,'root':str(root.resolve()),'files':snapshot(root),'checkpoint':str(checkpoint.resolve()),
       'checkpoint_sha256':c.sha(checkpoint),'protocol':str(protocol.resolve()),'protocol_sha256':c.sha(protocol),
       'tokenizer':str(tokenizer.resolve()),'tokenizer_files':snapshot(tokenizer),
       'runner_sha256':c.sha(__file__),'helper_sha256':c.sha(c.__file__),
       'model':c.MODEL,'project':c.PROJECT,'revision':c.REVISION,'prices':c.PRICES,
       'packages':{p:importlib.metadata.version(p) for p in PACKAGES},
       'rows':selected,'max_tokens':416,'temperature':0.,'arms':['base','v8'],
       'upper_usd':sum(r['prompt_tokens']*.195+416*.6 for r in selected)*2/1e6,
       'scope':'new composed routing transfer; complete4 faithfulness and two-bridge complexity separate; no general VPD certificate'}
    validate(m)
    destination.write_text(json.dumps(m,indent=2)+'\n'); return m

def bound_inputs(m):
    if m['runner_sha256']!=c.sha(__file__) or m['helper_sha256']!=c.sha(c.__file__):
        raise ValueError('runner/helper changed')
    if (m['model'],m['project'],m['revision'],m['prices'])!=(c.MODEL,c.PROJECT,c.REVISION,c.PRICES):
        raise ValueError('model/project/prices changed')
    if (m['max_tokens'],m['temperature'],m['arms'])!=(416,0.,['base','v8']): raise ValueError('sample protocol changed')
    for pathkey,hashkey in (('checkpoint','checkpoint_sha256'),('protocol','protocol_sha256')):
        if c.sha(m[pathkey])!=m[hashkey]: raise ValueError(pathkey+' changed')
    root=Path(m['root'])
    if snapshot(root)!=m['files'] or snapshot(Path(m['tokenizer']))!=m['tokenizer_files']:
        raise ValueError('frozen inputs/tokenizer changed')
    teacher=json.loads((root/'route_transfer/manifest.json').read_text())
    if teacher.get('teacher_integrity')!='PASS': raise ValueError('teacher failed')
    expected={x['id']+'.prompt.txt' for x in teacher['cases']}
    if len(m['rows'])!=8 or len(expected)!=8 or {x['id'] for x in m['rows']}!=expected:
        raise ValueError('panel differs from eight teacher cases')
    rows=[]
    for row in m['rows']:
        if Path(row['id']).name!=row['id']: raise ValueError('prompt basename required')
        p=root/'route_transfer'/row['id']
        if c.sha(p)!=row['prompt_sha256']: raise ValueError('prompt changed')
        rows.append({'id':row['id'].removesuffix('.prompt.txt'),'prompt':p.read_text(),'prompt_sha256':row['prompt_sha256']})
    upper=sum(row['prompt_tokens']*.195+416*.6 for row in m['rows'])*2/1e6
    if not math.isfinite(m['upper_usd']) or abs(m['upper_usd']-upper)>1e-12:
        raise ValueError('cost bound changed')
    return rows

def validate(m):
    rows=bound_inputs(m)
    if any(importlib.metadata.version(p)!=v for p,v in m['packages'].items()): raise ValueError('packages changed')
    rr,_=c.renderer(m['tokenizer']); prompts=[c.generation_prompt(row,rr) for row in rows]
    if [p.length for p in prompts]!=[x['prompt_tokens'] for x in m['rows']] or any(p.length+416>32768 for p in prompts):
        raise ValueError('token fit changed')
    return rows,rr,prompts

def run(manifest, destination, cap):
    m=json.loads(manifest.read_text())
    if destination.exists(): raise ValueError('immutable paid output already exists')
    if not math.isfinite(cap) or cap<m['upper_usd']: raise ValueError('cap too small')
    if os.environ.get('TINKER_PROJECT_ID')!=c.PROJECT: raise ValueError('explicit named project required')
    rows,rr,prompts=validate(m) # All guards before any Service construction.
    import tinker
    destination.mkdir(); (destination/'manifest.json').write_bytes(manifest.read_bytes())
    actual=0.
    def record(name,item):
        with (destination/name).open('a') as f:
            f.write(json.dumps(item)+'\n'); f.flush(); os.fsync(f.fileno())
    service=tinker.ServiceClient(project_id=c.PROJECT)
    base=service.create_sampling_client(base_model=c.MODEL)
    # Restore exactly the frozen model weights. No optimizer or training calls.
    loaded=service.create_training_client_from_state(json.loads(Path(m['checkpoint']).read_text())['state'])
    v8=loaded.save_weights_and_get_sampling_client()
    params=tinker.SamplingParams(max_tokens=416,temperature=0.,stop=rr.get_stop_sequences())
    for name,sampler in (('base',base),('v8',v8)):
        for row,prompt in zip(rows,prompts):
            reserve=(prompt.length*.195+416*.6)/1e6
            if actual+reserve>cap+1e-12: raise ValueError('remaining cap too small')
            record('reservations.jsonl',{'arm':name,'id':row['id'],'upper_usd':reserve,'status':'issued_or_uncertain'})
            res=sampler.sample(prompt=prompt,num_samples=1,sampling_params=params).result()
            # Preserve all paid output before parsing or validating the receipt.
            record(name+'_raw.jsonl',{'id':row['id'],'prompt_sha256':row['prompt_sha256'],
                    'prompt_cache_hit_tokens':res.prompt_cache_hit_tokens,
                    'sequences':[{'token_ids':list(s.tokens),'finish':str(s.stop_reason)} for s in res.sequences]})
            if len(res.sequences)!=1: raise ValueError('one sequence required; reservation remains uncertain')
            s=res.sequences[0]; hits=res.prompt_cache_hit_tokens
            if not 0<=hits<=prompt.length or not 0<=len(s.tokens)<=416:
                raise ValueError('invalid receipt; reservation remains uncertain')
            tokens={'prefill':prompt.length-hits,'cached':hits,'sampled':len(s.tokens)}
            usd=sum(tokens[k]*c.PRICES[k] for k in tokens)/1e6; actual+=usd
            record('spend.jsonl',{'arm':name,'id':row['id'],**tokens,'usd':usd,'total_usd':actual})
            record('reservations.jsonl',{'arm':name,'id':row['id'],'status':'settled','usd':usd})
            if actual>cap+1e-12: raise ValueError('actual cap exceeded')
            result={'id':row['id'],'prompt_sha256':row['prompt_sha256'],'token_ids':list(s.tokens),'finish':str(s.stop_reason),'sample':0}
            try:
                from tinker_cookbook.renderers import get_text_content
                parsed,_=rr.parse_response(s.tokens); result['completion']=get_text_content(parsed)
            except Exception as exc:
                result.update(completion=None,parse_error=type(exc).__name__)
            record(name+'.jsonl',result)
    (destination/'complete.json').write_text(json.dumps({'actual_usd':actual,'upper_usd':m['upper_usd'],'responses':16})+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for k in ('root','checkpoint','tokenizer','protocol','out'): q.add_argument('--'+k,type=Path,required=True)
    q=sub.add_parser('run'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--out',type=Path,required=True); q.add_argument('--cap-usd',type=float,required=True)
    a=p.parse_args()
    if a.command=='prepare': print(json.dumps({'upper_usd':prepare(a.root,a.checkpoint,a.tokenizer,a.protocol,a.out)['upper_usd']}))
    else: run(a.manifest,a.out,a.cap_usd)
