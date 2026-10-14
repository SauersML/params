"""Thin opt-in client for frozen paired graph supervision; no tokenizer/model load.

Offline validation is stdlib-only. Execution imports SDK lazily, records intents
before calls, and never resumes/retries a partially created output directory.
Raw first responses are authoritative; decoding and physical scoring are offline.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct

PROJECT='3f598f31-2ffa-4eb4-b302-909eb234fb7d'
MODEL='Qwen/Qwen3-8B'
ARMS=('point','intervention')
SDK_VERSION='0.33.1'
LR=.00047297908091376354
FAMILIES=('additive_vote','bigram_mlp','binding_positional','cancel','copy_offset','gate','induction','redundant')
OLD_TRAIN=('additive_vote_000','bigram_mlp_000','binding_positional_001','cancel_002','copy_offset_000','gate_003','induction_001','redundant_001')


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(encoded(value).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(model_input):
    if set(model_input)!={'chunks'}:raise ValueError('text-only ModelInput required')
    result=[]
    for chunk in model_input['chunks']:
        if set(chunk)!={'type','tokens'} or chunk['type']!='encoded_text':raise ValueError('unsupported input chunk')
        if any(type(i)is not int or not 0<=i<2**31 for i in chunk['tokens']):raise ValueError('invalid token ID')
        result.extend(chunk['tokens'])
    if not result:raise ValueError('empty input')
    return result


def datum_values(value):
    if set(value)!={'model_input','loss_fn_inputs'}:raise ValueError('unexpected datum fields')
    ids=tokens(value['model_input']);loss=value['loss_fn_inputs']
    if set(loss)!={'target_tokens','weights'}:raise ValueError('cross entropy inputs required')
    restored={}
    for key,dtype in (('target_tokens','int64'),('weights','float32')):
        t=loss[key]
        if set(t)!={'data','dtype','shape','sparse_crow_indices','sparse_col_indices'} or t['dtype']!=dtype or t['shape']!=[len(ids)] or t['sparse_crow_indices'] is not None or t['sparse_col_indices'] is not None or len(t['data'])!=len(ids):raise ValueError('dense one-dimensional typed loss inputs required')
        if key=='target_tokens':
            if any(type(i)is not int or not 0<=i<2**31 for i in t['data']):raise ValueError('invalid target ID')
            restored[key]=t['data']
        else:
            if any(type(w)not in (int,float) or not math.isfinite(w) or w<0 for w in t['data']):raise ValueError('invalid supervision weights')
            restored[key]=[struct.unpack('f',struct.pack('f',w))[0] for w in t['data']]
            if restored[key]!=t['data'] or not any(restored[key]):raise ValueError('exact positive FP32 supervision required')
    return ids,restored


def validate(payload):
    if set(payload)!={'schema_version','contract','data','prompts','batches','stop_sequences'} or payload['schema_version']!=1:raise ValueError('prepared schema mismatch')
    c=payload['contract'];recipe=c['recipe'];batches=payload['batches']
    train_only=c.get('training_manifest',{}).get('evaluation_scope')=='train_only_pending_fresh340'
    count=35 if train_only else 24
    if c['model']!=MODEL or c['project']!=PROJECT or (c['epochs'],c['batch'],c['seed'],c['updates_per_arm'],c['output_cap'])!=(8,8,0,count,416):raise ValueError('fixed paired recipe required')
    if set(payload['data'])!=set(ARMS) or any(len(payload['data'][a])!=count for a in ARMS):raise ValueError('two matched fixed-row arms required')
    rng=random.Random(0);expected=[]
    for _ in range(8):
        order=list(range(count));rng.shuffle(order);expected.extend(order[i:i+8] for i in range(0,count,8))
    if train_only:
        visits=[i for batch in expected for i in batch];expected=[visits[i:i+8] for i in range(0,len(visits),8)]
    if batches!=expected or digest(batches)!=c['batches_sha256']:raise ValueError('fixed epoch schedule changed')
    expected_init={'base_model':MODEL,'rank':32,'seed':0,'train_mlp':True,'train_attn':True,'train_unembed':True}
    if train_only:expected_init.update(common_init_receipt_sha256='775614467479c81a829219d240d7e4462dc647a79f9e8932c01400a46702de5b',mode='restore_exact_saved_common_state')
    expected_optim=[{'learning_rate':LR*(1-step/count),'beta1':.9,'beta2':.95,'eps':1e-8} for step in range(count)]
    if recipe['initialization']!=expected_init or recipe['optimizer']!=expected_optim or recipe['loss_fn']!='cross_entropy' or recipe['sampling']!={'temperature':0.,'max_tokens':416,'num_samples':1} or recipe['evaluation_arms']!=([] if train_only else ['base',*ARMS]):raise ValueError('explicit optimizer/init/sample recipe changed')
    ids=[r['id'] for r in c['training']]
    expected_ids=list(OLD_TRAIN)+[f'{f}_{3100000+100*i+v}' for i,f in enumerate(FAMILIES) for v in (0,1)]
    if train_only:
        expected_ids+=['redundant_000','cancel_001','gate_000']+[f'{f}_{3300000+100*i}' for i,f in enumerate(FAMILIES)]
        manifest={'schema':'paired_train35_v1','train_ids':expected_ids,'evaluation_scope':'train_only_pending_fresh340'}
        if c['training_manifest']!=manifest or c['training_manifest_sha256']!=digest(manifest):raise ValueError('exact TRAIN35 manifest changed')
    if ids!=expected_ids or recipe['sample_ids']!=[[ids[i] for i in b] for b in batches]:raise ValueError('training sample IDs/schedule disagree')
    lengths={};masses={}
    for a in ARMS:
        parsed=[datum_values(d) for d in payload['data'][a]]
        if [digest({'input':i,'loss':v}) for i,v in parsed]!=c['rendered_training_sha256'][a]:raise ValueError('datum token/weight digest changed')
        lengths[a]=[len(i) for i,v in parsed];masses[a]=[sum(v['weights']) for i,v in parsed]
        if max(lengths[a])>32768 or any(sum(w>0 for w in v['weights'])>416 for i,v in parsed):raise ValueError('training fit changed')
        if sum(lengths[a][i] for b in batches for i in b)!=c['train_tokens_per_arm'][a]:raise ValueError('billed training tokens changed')
    if any(abs(x-y)>2e-6*max(x,y) for x,y in zip(masses['point'],masses['intervention'])):raise ValueError('paired row mass changed')
    prompts=[tokens(p) for p in payload['prompts']]
    if train_only and (prompts or c['panel']):raise ValueError('TRAIN35 must not sample')
    if not train_only:
        if len(prompts)!=32 or len(c['panel'])!=32 or len({(r['stratum'],r['id']) for r in c['panel']})!=32:raise ValueError('fixed 32-case first-response panel required')
        if [r['id'] for r in c['panel'] if r['stratum']=='new_model']!=[f'{f}_{3200000+100*i}' for i,f in enumerate(FAMILIES)] or sum(r['stratum']=='retention' for r in c['panel'])!=24:raise ValueError('new-model/retention panel coverage changed')
    if any(len(p)+416+16>32768 or len(p)!=r['prompt_tokens'] or digest(p)!=r['rendered_sha256'] for p,r in zip(prompts,c['panel'])):raise ValueError('panel tokens/fit changed')
    stops=payload['stop_sequences']
    if not isinstance(stops,list) or not stops or not (all(type(x)is str for x in stops) or all(type(x)is int for x in stops)):raise ValueError('explicit homogeneous stop sequences required')
    return c


def rates_of(value):
    if set(value) not in ({'trained','prefill','sampled'},{'trained','prefill','cached','sampled'}):raise ValueError('three quoted rates plus optional verified cache rate required')
    rates={k:Decimal(str(v)) for k,v in value.items()}
    rates.setdefault('cached',rates['prefill'])
    if any(not v.is_finite() or v<0 for v in rates.values()) or rates['cached']>rates['prefill']:raise ValueError('invalid quoted prices')
    return rates


def upper(c,rates):
    counts={'trained':sum(c['train_tokens_per_arm'].values()),'prefill':3*sum(r['prompt_tokens'] for r in c['panel']),'sampled':3*len(c['panel'])*416}
    return sum((Decimal(n)*rates[k] for k,n in counts.items()),Decimal(0))/Decimal(1000000)


def preflight(prepared,prices):
    payload=json.loads(Path(prepared).read_text());c=validate(payload);rates=rates_of(prices)
    return {'schema_version':1,'prepared_sha256':sha(prepared),'runner_sha256':sha(__file__),'project':PROJECT,'model':MODEL,'sdk_version':SDK_VERSION,'prices_per_million':{k:str(v) for k,v in rates.items()},'upper_usd':str(upper(c,rates)),'updates_per_arm':c['updates_per_arm'],'sampling_calls':3*len(c['panel']),'tokenizer_loaded':False,'service_calls':0}


def saved_initialization(c,path):
    if path is None or sha(path)!=c['recipe']['initialization']['common_init_receipt_sha256']:
        raise ValueError('exact untrained common initialization receipt required')
    rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
    expected={k:v for k,v in c['recipe']['initialization'].items() if k not in ('mode','common_init_receipt_sha256')}
    if len(rows)!=1 or rows[0]['config']!=expected or not isinstance(rows[0]['path'],str) or not rows[0]['path'].startswith('tinker://'):
        raise ValueError('common initialization model/config/provenance changed')
    return rows[0]['path']


def append(path,value):
    with Path(path).open('a') as f:f.write(encoded(value)+'\n');f.flush();os.fsync(f.fileno())


def training_diagnostics(result,selected):
    """Observe optimization without adding calls or changing loss weights."""
    try:
        outputs=result.loss_fn_outputs
        if len(outputs)!=len(selected):raise ValueError('loss output count changed')
        numerator=mass=0.;supervised=0
        for output,datum in zip(outputs,selected):
            logp=output['logprobs'].data;weights=datum.loss_fn_inputs['weights'].data
            if len(logp)!=len(weights):raise ValueError('loss output length changed')
            for value,weight in zip(logp,weights):
                if weight<=0:continue
                if not math.isfinite(value):raise ValueError('nonfinite supervised log probability')
                numerator-=value*weight;mass+=weight;supervised+=1
        if not mass:raise ValueError('missing positive loss mass')
        return {'weighted_nll_nats':numerator/mass,'loss_weight_mass':mass,
                'supervised_tokens':supervised,'diagnostic_error':None}
    except (AttributeError,KeyError,TypeError,ValueError) as exc:
        # Paid work already happened. Preserve diagnostic failure separately;
        # never resubmit an update merely because optional telemetry changed.
        return {'weighted_nll_nats':None,'diagnostic_error':type(exc).__name__+': '+str(exc)}


def execute(prepared,proof_path,approval_path,out,api=None,*,common_init_path=None,fresh_budget_path=None):
    proof=json.loads(Path(proof_path).read_text());approval=json.loads(Path(approval_path).read_text())
    if proof!=preflight(prepared,proof['prices_per_million']):raise ValueError('offline preflight changed')
    required={'schema_version','project','model','prepared_sha256','runner_sha256','preflight_sha256','cap_usd','execute_authorized'}
    if set(approval)!=required or approval['schema_version']!=1 or approval['execute_authorized'] is not True or approval['project']!=PROJECT or approval['model']!=MODEL or approval['prepared_sha256']!=sha(prepared) or approval['runner_sha256']!=sha(__file__) or approval['preflight_sha256']!=sha(proof_path):raise ValueError('explicit execution approval binding required')
    cap=Decimal(approval['cap_usd'])
    if not cap.is_finite() or cap<Decimal(proof['upper_usd']) or os.environ.get('TINKER_PROJECT_ID')!=PROJECT:raise ValueError('project/budget guard failed')
    payload=json.loads(Path(prepared).read_text());c=validate(payload);out=Path(out)
    train_only=c.get('training_manifest',{}).get('evaluation_scope')=='train_only_pending_fresh340'
    state=saved_initialization(c,common_init_path) if train_only else None
    if train_only:
        from matched_diagnostic_sample import fresh_budget
        if fresh_budget_path is None:raise ValueError('fresh shared budget admission required')
        fresh_budget(fresh_budget_path)
    if out.exists():raise ValueError('no restart/retry: existing output may contain charged work')
    # SDK restoration happens before ServiceClient; no preparation/tokenizer imports.
    if api is None:
        import importlib.metadata
        if importlib.metadata.version('tinker')!=SDK_VERSION:raise ValueError('pinned SDK required')
        import tinker as api
    data={a:[api.Datum(model_input=api.ModelInput.model_validate(d['model_input']),loss_fn_inputs={k:api.TensorData(**v) for k,v in d['loss_fn_inputs'].items()}) for d in payload['data'][a]] for a in ARMS}
    prompts=[api.ModelInput.model_validate(p) for p in payload['prompts']]
    out.mkdir();append(out/'start.jsonl',{'prepared_sha256':sha(prepared),'preflight_sha256':sha(proof_path),'approval_sha256':sha(approval_path),'runner_sha256':sha(__file__),'sdk_version':SDK_VERSION,'cap_usd':str(cap),'project':PROJECT,**({'common_init_receipt_sha256':sha(common_init_path)} if train_only else {})})
    rates=rates_of(proof['prices_per_million']);spent=Decimal(0);reserved=Decimal(0);seq=0
    def journal(event,**kw):append(out/'journal.jsonl',{'event':event,**kw})
    def reserve(kind,counts,**where):
        nonlocal reserved,seq
        amount=sum((Decimal(n)*rates[k] for k,n in counts.items()),Decimal(0))/Decimal(1000000)
        if spent+reserved+amount>cap:raise ValueError('remaining call cap exceeded')
        seq+=1;reserved+=amount;journal('intent',call_id=seq,kind=kind,counts=counts,reserved_usd=str(amount),**where)
        return seq,amount
    def settle(ticket,counts):
        nonlocal spent,reserved
        call_id,reservation=ticket;amount=sum((Decimal(n)*rates[k] for k,n in counts.items()),Decimal(0))/Decimal(1000000)
        if amount>reservation:raise ValueError('provider accounting exceeds reservation')
        spent+=amount;reserved-=reservation;journal('settled',call_id=call_id,counts=counts,usd=str(amount),accounted_upper_usd=str(spent),remaining_reserved_usd=str(reserved))
    service=None
    try:
        if train_only:fresh_budget(fresh_budget_path)
        journal('service_create_intent',project=PROJECT);service=api.ServiceClient(project_id=PROJECT)
        def sample(client,arm):
            params=api.SamplingParams(**{k:v for k,v in c['recipe']['sampling'].items() if k!='num_samples'},stop=payload['stop_sequences'])
            for i,(row,prompt) in enumerate(zip(c['panel'],prompts)):
                ticket=reserve('sample',{'prefill':row['prompt_tokens'],'sampled':416},arm=arm,index=i,id=row['id'])
                result=client.sample(prompt=prompt,num_samples=1,sampling_params=params).result()
                raw={'arm':arm,'id':row['id'],'stratum':row['stratum'],'prompt_rendered_sha256':row['rendered_sha256'],'sample':0,'sequences':[{'token_ids':list(s.tokens),'finish':str(s.stop_reason)} for s in result.sequences],'prompt_cache_hit_tokens':result.prompt_cache_hit_tokens}
                append(out/(arm+'_raw.jsonl'),raw)
                hits=result.prompt_cache_hit_tokens
                if len(result.sequences)!=1 or type(hits)is not int or not 0<=hits<=row['prompt_tokens'] or len(result.sequences[0].tokens)>416:raise ValueError('sample/accounting response changed')
                settle(ticket,{'prefill':row['prompt_tokens']-hits,'cached':hits,'sampled':len(result.sequences[0].tokens)})
        if not train_only:
            journal('base_sampler_intent');sample(service.create_sampling_client(base_model=MODEL),'base')
            init=c['recipe']['initialization'];journal('initial_client_intent',config=init)
            initial=service.create_lora_training_client(**init)
            journal('initial_checkpoint_intent');state=initial.save_state('matched-graph-common-seed0').result().path
            append(out/'common_init.jsonl',{'path':state,'config':init,'scope':'same provider checkpoint loaded; no independent tensor-hash API'})
        else:
            (out/'common_init.jsonl').write_bytes(Path(common_init_path).read_bytes())
        clients={}
        for a in ARMS:
            journal('clone_intent',arm=a,**({'source_receipt_sha256':sha(common_init_path)} if train_only else {'path':state}));clients[a]=service.create_training_client_from_state_with_optimizer(state)
            info=clients[a].get_info().model_dump(mode='json');append(out/(a+'_provider.jsonl'),info)
            if info.get('is_lora') is not True or info.get('lora_rank')!=32 or info.get('model_data',{}).get('model_name')!=MODEL:raise ValueError('provider model/rank changed')
        for step,batch in enumerate(payload['batches']):
            pending=[];errors=[]
            for a in ARMS:
                selected=[data[a][i] for i in batch];counts={'trained':sum(len(tokens(payload['data'][a][i]['model_input'])) for i in batch)}
                try:
                    ticket=reserve('forward_backward',counts,arm=a,step=step,indices=batch)
                    fb=clients[a].forward_backward(selected,loss_fn='cross_entropy');pending.append([a,fb,None,ticket,counts])
                    journal('optimizer_intent',arm=a,step=step,params=c['recipe']['optimizer'][step])
                    pending[-1][2]=clients[a].optim_step(api.AdamParams(**c['recipe']['optimizer'][step]))
                except Exception as exc:errors.append(exc);journal('submission_error',arm=a,step=step,error=type(exc).__name__);break
            for a,fb,op,ticket,counts in pending:
                try:
                    result=fb.result();settle(ticket,counts)
                    append(out/'training_diagnostics.jsonl',{'arm':a,'step':step,
                        **training_diagnostics(result,[data[a][i] for i in batch])})
                except Exception as exc:errors.append(exc);journal('fb_error',arm=a,step=step,error=type(exc).__name__)
            for a,fb,op,ticket,counts in pending:
                if op is None:continue
                try:op.result();append(out/'steps.jsonl',{'arm':a,'step':step,'indices':batch,'sample_ids':c['recipe']['sample_ids'][step],'optimizer':c['recipe']['optimizer'][step]})
                except Exception as exc:errors.append(exc);journal('optimizer_error',arm=a,step=step,error=type(exc).__name__)
            if errors:raise errors[0]
        for a in ARMS:
            journal('trained_checkpoint_intent',arm=a);path=clients[a].save_state(a+'-matched-graph-eight-epoch').result().path
            append(out/(a+'_checkpoint.jsonl'),{'path':path})
            if not train_only:
                journal('trained_sampler_intent',arm=a);sample(clients[a].save_weights_and_get_sampling_client(),a)
        append(out/'complete.jsonl',{'accounted_upper_usd':str(spent),'remaining_reserved_usd':str(reserved),'steps_per_arm':c['updates_per_arm'],'first_responses_per_arm':len(c['panel']),'roles':['point35','intervention35'] if train_only else list(ARMS)})
    except BaseException as exc:
        append(out/'failure.jsonl',{'error':type(exc).__name__,'accounted_upper_usd':str(spent),'uncertain_reserved_usd':str(reserved),'automatic_retry':False});raise
    finally:
        if service is not None:
            try:service.close('success' if (out/'complete.jsonl').exists() else 'errored').result()
            except Exception as exc:journal('service_close_error',error=type(exc).__name__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--prices',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--approval',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--common-init',type=Path);p.add_argument('--fresh-budget',type=Path)
    a=p.parse_args()
    if a.execute:
        if a.approval is None or a.out is None:p.error('execute requires approval and fresh out')
        execute(a.prepared,a.preflight,a.approval,a.out,common_init_path=a.common_init,fresh_budget_path=a.fresh_budget)
    else:
        if a.prices is None:p.error('offline preflight requires external quoted prices')
        value=preflight(a.prepared,json.loads(a.prices.read_text()))
        with a.preflight.open('x') as f:f.write(encoded(value)+'\n')
        print(encoded(value))
