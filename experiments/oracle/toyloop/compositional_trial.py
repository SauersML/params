"""Strict opt-in preflight for matched288-row compositional SFT; no automatic API calls."""
from __future__ import annotations
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

ARMS = ('single','two')
EXPECTED_DATA = {'single':'1ef92c0f4c4560c6ed0f8a82073be43581d3f82c7b306a1e38b3f305ed364a43',
                 'two':'05f5d8999a5e05073ae6e09b2c360089ff419b64c31707fc2cda9f4773d46581',
                 'holdout':'c7f21b1c69ad77095606d19095c45c049288211c34b2e6ebb17e381491ed65e5'}
EXPECTED_UPPER = Decimal('.68438213')
EVAL_FILES = {'original24':'49b949ca35c5e2d4fb89463a8a5b8f03353d0fb7db365db3d6597493c4b135c2',
              'old_development8':'6cf49d238093c60131431228df3d0b75f16ff64aa5bf635bb71370c43078a1b1',
              'new8':'8452d8f36b667a7b3619f3948f40c1762b72f3a7ba6202940092344516fa88ff',
              'training_incidentalV20':'7da5b75979f8832eb7bba76bd6498d212cf097671a6faa484f5cf7a2e9e2ad7f'}
TOKENIZER_FILES = {'config.json':'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
                   'generation_config.json':'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
                   'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
                   'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}
HELPERS = {'capable_factorial.py':'14e108512450bedc93e3e4be2f01d08654e4b7aa6ef9e5246da47e093382d692',
           'capable_wiring.py':'ebe3ac23a219ebf068857d4fc6d955bfab30c748e2755201cb10d06585f44428',
           'relation_loss.py':'912a37dd227a8e8a588b2e7ac75c7549e382fec5db50759000ff284dd962c264'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_files(root):
    """Reject damaged producer/teacher/data artifacts before any SDK or Service."""
    root = Path(root)
    for name,expected in EVAL_FILES.items():
        if digest(root/(name+'_questions.jsonl'))!=expected:
            raise ValueError('frozen evaluation stratum bytes changed')
    for name,expected in HELPERS.items():
        if digest(Path(__file__).with_name(name))!=expected:
            raise ValueError('frozen helper source changed')
    if digest(root/'token_fit.json') != '3a78a5c159b9d497ae6ea6bf477638e9500f6d72f9362248aca89ef4e7e3aff4':
        raise ValueError('tokenization and matched-mass proof changed')
    if digest(root/'complete.json') != '6bff266d086d7cc7c216a0072bbb64eac6edbdf7b2a0d2de69526d259dc1006e':
        raise ValueError('physical proof manifest changed')
    if digest(root/'verification.json') != '955a1fa72a484e7ebda746a41f90f650655664caa10651ee6846ff428989d913':
        raise ValueError('executed teacher proof changed')
    m = json.loads((root/'complete.json').read_text())
    if not m['complete'] or m['updates']!=72 or m['batch']!=8 or m['rows_per_arm']!=288:
        raise ValueError('unexpected matched recipe')
    for arm, expected in EXPECTED_DATA.items():
        if digest(root/(arm+'_inputs.jsonl'))!=expected or m['datasets'][arm]!=expected:
            raise ValueError('frozen curriculum changed')
    single=(root/'single_inputs.jsonl').read_bytes().splitlines(keepends=True)
    two=(root/'two_inputs.jsonl').read_bytes().splitlines(keepends=True)
    if len(single)!=288 or len(two)!=288 or single[:256]!=two[:256]:
        raise ValueError('original256 is not identical across arms')
    if hashlib.sha256(b''.join(single[:256])).hexdigest()!=m['core_sha256']:
        raise ValueError('original256 prefix changed')
    checks=json.loads((root/'verification.json').read_text())
    if len(checks)!=72 or len({r['id'] for r in checks})!=72:
        raise ValueError('teacher panel incomplete')
    for case in checks:
        for filename,sha in case['files'].items():
            if digest(root/'instances'/filename)!=sha:
                raise ValueError('source toy/target/prompt/bank changed')
        for name,score in case['scores'].items():
            errors=[r['kl_bits'] for r in score['per_text']]
            if len(errors)!=8 or any(not math.isfinite(x) for x in errors):
                raise ValueError('finite eight-input proof required')
            if (name in ('gold','complete') and max(errors)>1e-3) or (name=='wrong' and min(errors)<1.):
                raise ValueError('finite teacher contrasts failed')
        if any(len(x['kl_bits'])!=8 or min(x['kl_bits'])<.01 or not all(math.isfinite(v) for v in x['kl_bits']) for x in case['relevance']):
            raise ValueError('selected routes not verified relevant')
    fit=json.loads((root/'token_fit.json').read_text())
    if Decimal(fit['known_upper_usd'])!=EXPECTED_UPPER or fit['updates']!=72 or fit['evaluation']['known_rows_per_arm']!=60:
        raise ValueError('exact reservation/fit changed')
    return m,fit


def prepared(root,tokenizer):
    """Re-render exact source bytes and enforce both mass and complete token fitting."""
    m,fit=validate_files(root)
    import importlib.metadata
    versions=dict(line.split('==',1) for line in (Path(root).parent/'tokenizer-lock.txt').read_text().splitlines() if '==' in line)
    if digest(Path(root).parent/'tokenizer-lock.txt')!='de25a1838eca2d89880713248eb224c5ccd077036a72c50f6bc837883cc23d8d':
        raise ValueError('dependency proof changed')
    for name in ('tinker','tinker-cookbook','transformers','tokenizers'):
        if importlib.metadata.version(name)!=versions[name]:
            raise ValueError('client/tokenizer package contract changed')
    import torch
    import tinker
    from tinker.types.tensor_data import TensorData
    import capable_factorial as factorial
    import capable_wiring as pilot
    import relation_loss
    if {p.name:digest(p) for p in Path(tokenizer).iterdir() if p.is_file()}!=TOKENIZER_FILES:
        raise ValueError('original verified tokenizer byte inventory differs')
    rr,tok=pilot.renderer(Path(tokenizer))
    data={}
    for arm in ARMS:
        raw=rows(Path(root)/(arm+'_inputs.jsonl'))
        stats=fit['arms'][arm]['stats']
        if len(raw)!=288 or len(stats)!=288:
            raise ValueError('incomplete fit')
        output=[]
        for row,expected in zip(raw,stats):
            datum=pilot.datum(row,rr)
            weighted,allocation=relation_loss.apply_to_datum(datum,row['completion'],tok,8)
            if row['id']!=expected['id'] or datum.model_input.length!=expected['pair_tokens'] or not factorial.allocation_equal(allocation,expected['relation8_allocation']):
                raise ValueError('token/weight allocation changed')
            weights=weighted.loss_fn_inputs['weights'].to_torch()
            normalized=weights*expected['augmentation_renormalization_factor']
            if (not torch.isfinite(normalized).all() or not torch.equal(weights>0,normalized>0)
                    or abs(float(normalized.sum())-expected['postnormalization_total_mass'])>1e-6):
                raise ValueError('exact positive supervision/matched row mass changed')
            inputs=dict(weighted.loss_fn_inputs)
            inputs['weights']=TensorData.from_torch(normalized)
            output.append(tinker.Datum(model_input=weighted.model_input,loss_fn_inputs=inputs))
        data[arm]=output
    batches=pilot.schedule(288,2,8,0)
    if len(batches)!=72 or sum(map(len,batches))!=576:
        raise ValueError('unequal update schedule')
    panel=[]
    for stratum in ('original24','old_development8','new8','training_incidentalV20'):
        for row in rows(Path(root)/(stratum+'_questions.jsonl')):
            if hashlib.sha256(row['prompt'].encode()).hexdigest()!=row['prompt_sha256']:
                raise ValueError('evaluation prompt changed')
            panel.append({**row,'case_id':row['id'],'stratum':stratum,'id':stratum+'/'+row['id']})
    prompts=[pilot.generation_prompt(r,rr) for r in panel]
    if len(panel)!=60 or len({r['id'] for r in panel})!=60:
        raise ValueError('evaluation panel incomplete')
    rendered=[{'id':row['id'],'length':prompt.length,'rendered_sha256':hashlib.sha256(json.dumps(prompt.to_ints(),separators=(',',':')).encode()).hexdigest()} for row,prompt in zip(panel,prompts)]
    frozen_path=Path(root).parent/'rendered_eval_contract.json'
    if digest(frozen_path)!='85838afb61b9f3b82598537e07a774d9236a1649577acc27e140f523f974da42':
        raise ValueError('frozen rendered-token contract changed')
    if json.loads(frozen_path.read_text())!=rendered:
        raise ValueError('rendered evaluation token identity changed')
    upper=(sum(d.model_input.length for arm in ARMS for d in data[arm])*2*Decimal('.44')
           +2*(sum(p.length for p in prompts)*Decimal('.195')+60*416*Decimal('.60')))/Decimal(1000000)
    if upper!=EXPECTED_UPPER:
        raise ValueError('exact paid bound changed')
    return {'data':data,'batches':batches,'panel':panel,'prompts':prompts,'renderer':rr,'upper':upper,
            'proof':m,'fit':fit,'source_hashes':{name:digest(Path(__file__).with_name(name)) for name in ['compositional_trial.py','capable_factorial.py','capable_wiring.py','relation_loss.py']}}

PROJECT='3f598f31-2ffa-4eb4-b302-909eb234fb7d'
RATES={'prefill':Decimal('.195'),'cached':Decimal('.039'),'sampled':Decimal('.60'),'trained':Decimal('.44')}


def run(root,tokenizer,out,cap,*,service_factory=None):
    """Copied bounded two-stream executor from source-bound factorial; no retries."""
    import os
    out=Path(out); cap=Decimal(str(cap))
    if out.exists() or cap<EXPECTED_UPPER or cap>Decimal('1') or os.environ.get('TINKER_PROJECT_ID')!=PROJECT:
        raise ValueError('output/project/authorized whole-trial cap rejected')
    context=prepared(root,tokenizer)
    import tinker
    import capable_wiring as pilot
    out.mkdir()

    def record(name,value):
        with (out/name).open('a') as stream:
            stream.write(json.dumps(value)+'\n'); stream.flush(); os.fsync(stream.fileno())

    record('contract.json',{'source_hashes':context['source_hashes'],'upper_usd':str(context['upper']),
                           'cap_usd':str(cap),'model':pilot.MODEL,'project':PROJECT,'rank':32,'seed':0,
                           'updates':72,'batch':8,'LR':pilot.LR,'relation_multiplier':8,'generation_cap':416,
                           'proof_manifest_sha256':digest(Path(root)/'complete.json'),
                           'tokenizer_files':{p.name:digest(p) for p in Path(tokenizer).iterdir() if p.is_file()},
                           'scope':'New fixed-development compositional SFT comparison; no English validation'})
    # This boundary is separately exercised with a sentinel after real preflight.
    service=(service_factory or tinker.ServiceClient)(project_id=PROJECT)
    spend=Decimal('0'); reserved=Decimal('0')

    def reserve(kind,tokens):
        nonlocal reserved
        amount=sum((Decimal(n)*RATES[k] for k,n in tokens.items()),Decimal('0'))/Decimal(1000000)
        if spend+reserved+amount>cap:
            raise ValueError('remaining reservation exceeds exact cap')
        record('reservations.jsonl',{'kind':kind,'tokens':tokens,'upper_usd':str(amount)})
        reserved+=amount
        return amount

    def settle(kind,tokens,reservation):
        nonlocal reserved,spend
        amount=sum((Decimal(n)*RATES[k] for k,n in tokens.items()),Decimal('0'))/Decimal(1000000)
        if amount>reservation:
            raise ValueError('provider charge exceeds reservation')
        spend+=amount; reserved-=reservation
        record('spend.jsonl',{'kind':kind,'tokens':tokens,'usd':str(amount),'actual_usd':str(spend),'remaining_reserved_usd':str(reserved)})

    initial=service.create_lora_training_client(base_model=pilot.MODEL,rank=32,seed=0)
    initial_state=initial.save_state('composition-common-seed0-init').result().path
    record('common_init.json',{'state':initial_state,'seed':0})
    clients={}
    for arm in ARMS:
        client=service.create_training_client_from_state_with_optimizer(initial_state)
        info=client.get_info().model_dump(mode='json')
        record(arm+'_provider.json',info)
        if info.get('is_lora') is not True or info.get('lora_rank')!=32 or info.get('model_data',{}).get('model_name')!=pilot.MODEL:
            raise ValueError('provider model/rank contract changed')
        clients[arm]=client
    for step,batch in enumerate(context['batches']):
        pending=[]; errors=[]
        for arm,client in clients.items():
            selected=[context['data'][arm][i] for i in batch]
            tokens={'trained':sum(d.model_input.length for d in selected)}
            try:
                reservation=reserve(arm+'_sft',tokens)
                fb=client.forward_backward(selected,loss_fn='cross_entropy')
                pending.append((arm,fb,None,tokens,reservation))
                op=client.optim_step(tinker.AdamParams(learning_rate=pilot.LR*(1-step/72),beta1=.9,beta2=.95,eps=1e-8))
                pending[-1]=(arm,fb,op,tokens,reservation)
            except Exception as exc:
                errors.append(exc)
                record('errors.jsonl',{'arm':arm,'step':step,'stage':'submission','error':type(exc).__name__})
                break
        for arm,fb,op,tokens,reservation in pending:
            try:
                fb.result(); settle(arm+'_sft',tokens,reservation)
            except Exception as exc:
                errors.append(exc)
        for arm,fb,op,tokens,reservation in pending:
            if op is None:
                continue
            try:
                op.result(); record('steps.jsonl',{'arm':arm,'step':step,'indices':batch})
            except Exception as exc:
                errors.append(exc)
        if errors:
            record('errors.jsonl',{'step':step,'stage':'drained_round','remaining_reserved_usd':str(reserved)})
            raise errors[0]
        print(json.dumps({'updates_per_arm':step+1,'actual_usd':str(spend)}),flush=True)
    params=tinker.SamplingParams(max_tokens=416,temperature=0.,stop=context['renderer'].get_stop_sequences())
    for arm,client in clients.items():
        state=client.save_state(arm+'-composition-two-epoch').result().path
        record(arm+'_state.json',{'state':state})
        sampler=client.save_weights_and_get_sampling_client()
        for row,prompt in zip(context['panel'],context['prompts']):
            reservation=reserve(arm+'_sample',{'prefill':prompt.length,'sampled':416})
            try:
                result=sampler.sample(prompt=prompt,num_samples=1,sampling_params=params).result()
                if len(result.sequences)!=1:
                    raise ValueError('one first sample required')
                sequence=result.sequences[0]; hits=result.prompt_cache_hit_tokens
                raw={'id':row['id'],'case_id':row['case_id'],'stratum':row['stratum'],'prompt_sha256':row['prompt_sha256'],
                     'token_ids':list(sequence.tokens),'finish':str(sequence.stop_reason),'sample':0,'prompt_cache_hit_tokens':hits}
                record(arm+'_raw.jsonl',raw)
                if not 0<=hits<=prompt.length or len(sequence.tokens)>416:
                    raise ValueError('provider token accounting invalid')
                settle(arm+'_sample',{'prefill':prompt.length-hits,'cached':hits,'sampled':len(sequence.tokens)},reservation)
                try:
                    parsed,_=context['renderer'].parse_response(sequence.tokens)
                    from tinker_cookbook.renderers import get_text_content
                    raw['completion']=get_text_content(parsed)
                except Exception as exc:
                    raw.update(completion=None,parse_error=type(exc).__name__)
                record(arm+'.jsonl',raw)
            except Exception as exc:
                record('errors.jsonl',{'arm':arm,'id':row['id'],'stage':'sample','error':type(exc).__name__,'remaining_reserved_usd':str(reserved)})
                raise
    record('complete.json',{'actual_usd':str(spend),'reserved_usd':str(reserved),'upper_usd':str(context['upper']),
                           'updates_per_arm':72,'first_samples':120})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cap',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:
        run(args.root,args.tokenizer,args.out,args.cap)
    else:
        c=prepared(args.root,args.tokenizer)
        args.out.write_text(json.dumps({'upper_usd':str(c['upper']),'source_hashes':c['source_hashes'],'steps':len(c['batches']),'samples':len(c['panel'])*2},indent=2)+'\n')
