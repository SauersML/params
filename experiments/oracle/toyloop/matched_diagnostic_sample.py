"""Explicit sampling-only execution of a frozen16prompt/fivepolicy diagnostic.

Offline validation imports no SDK/tokenizer. Root's admitted launcher must check
shared budget AFTER memory admission and BEFORE Keychain lookup; this runner
also checks that fresh budget snapshot before creating Service. No paid retries,
training, parsing, repair, or resume. Saved checkpoint receipt paths stay private.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import importlib.metadata
import json
import os
from pathlib import Path

import matched_graph_train as train_helper
import matched_graph_diagnostic_manifest as manifest_helper
from matched_graph_train import PROJECT, encoded, digest, sha, tokens, append
from matched_graph_diagnostic_manifest import POLICIES

INFLIGHT = 4

MODELS = {'base':'Qwen/Qwen3-8B','point':'Qwen/Qwen3-8B','intervention':'Qwen/Qwen3-8B',
          'qwen35':'Qwen/Qwen3.6-35B-A3B','kimi':'moonshotai/Kimi-K2.6'}
TOKENIZERS = {'Qwen/Qwen3-8B':('b968826d9c46dd6066d109eabc6255188de91218','qwen3_disable_thinking'),
 'Qwen/Qwen3.6-35B-A3B':('995ad96eacd98c81ed38be0c5b274b04031597b0','qwen3_5_disable_thinking'),
 'moonshotai/Kimi-K2.6':('b5aabbfb20227ed42becbf5541dbffd213942c58','kimi_k26_disable_thinking')}


def rates(model):
    value=model['rates_per_million']
    if set(value)!={'prefill','cached_prefill','sample'}:
        raise ValueError('explicit quoted sample rates required')
    result={k:Decimal(str(v).removeprefix('$')) for k,v in value.items()}
    if any(not v.is_finite() or v<0 for v in result.values()) or result['cached_prefill']>result['prefill']:
        raise ValueError('invalid price quotes')
    return result


def charge(model, prompt, hits, sampled):
    if (type(hits) is not int or not 0<=hits<=prompt or type(sampled) is not int or not 0<=sampled<=416):
        raise ValueError('provider sample accounting changed')
    r=rates(model)
    return ((prompt-hits)*r['prefill']+hits*r['cached_prefill']+sampled*r['sample'])/Decimal(1000000)


def profile(prepared):
    if prepared.get('profile') == 'train_fit_v1':
        if (prepared['primary_prepared_sha256'] != '4c68ec1f9c810bf25f87cad3074da573ad0fa9338abfc22b8e5590053d4306a2'
                or prepared['training_jsonl_sha256'] != '94e6f5966516c858a42a5c9f770a9b5f716a4cd31bcdb5427a584e5ffc9fa188'):
            raise ValueError('exact original TRAIN source required')
        return {a:MODELS[a] for a in POLICIES[:3]},24
    if prepared.get('profile') is not None:
        raise ValueError('unsupported sampling profile')
    return MODELS,16


def validate(prepared):
    policy_models,case_count=profile(prepared)
    if (prepared['schema_version']!=1 or prepared['policies']!=policy_models
            or (prepared['calls'],prepared['output_cap'],prepared['context'],prepared['num_samples'],prepared['temperature'])!=(len(policy_models)*case_count,416,32768,1,0.)):
        raise ValueError('fixed diagnostic recipe required')
    if set(prepared['models'])!=set(policy_models.values()) or set(prepared['checkpoint_receipt_sha256'])!={'point','intervention'}:
        raise ValueError('model/checkpoint coverage mismatch')
    keys=None;upper=Decimal(0)
    for name,model in prepared['models'].items():
        if (model['tokenizer_revision'],model['renderer'])!=TOKENIZERS[name]:
            raise ValueError('pinned model renderer changed')
        rows=model['inputs'];ids=[r['id'] for r in rows]
        if len(ids)!=case_count or len(set(ids))!=case_count or (keys is not None and ids!=keys):
            raise ValueError('shared16case order required')
        keys=ids
        if case_count==24:
            expected=list(train_helper.OLD_TRAIN)+[f'{f}_{3100000+100*i+v}' for i,f in enumerate(train_helper.FAMILIES) for v in (0,1)]
            if ids!=expected:raise ValueError('exact24TRAIN order required')
        elif ids[8:]!=[i+'__perm' for i in ids[:8]]:
            raise ValueError('original/permuted pairs required')
        stops=model['stop_sequences']
        if not stops or not (all(type(s)is int for s in stops) or all(type(s)is str for s in stops)):
            raise ValueError('explicit homogeneous stops required')
        bound=Decimal(0)
        for row in rows:
            t=tokens(row['model_input'])
            if len(t)!=row['prompt_tokens'] or digest(t)!=row['rendered_sha256'] or len(t)+416+16>32768:
                raise ValueError('frozen rendered tokens/fit changed')
            bound+=charge(model,len(t),0,416)
        if bound!=Decimal(model['upper_per_policy_usd']):
            raise ValueError('per-policy upper changed')
        upper+=bound*sum(n==name for n in policy_models.values())
    if upper!=Decimal(prepared['upper_usd_decimal']):
        raise ValueError('whole-call upper changed')
    return upper


def preflight(path):
    prepared=json.loads(Path(path).read_text());upper=validate(prepared)
    return {'schema_version':1,'prepared_sha256':sha(path),'runner_sha256':sha(__file__),
            'project':PROJECT,'policies':profile(prepared)[0],'upper_usd':str(upper),'calls':prepared['calls'],'profile':prepared.get('profile'),
            'service_calls':0,'training_steps':0,'automatic_retry':False,'outstanding_limit':INFLIGHT,
            'dependencies_sha256':{'matched_graph_train.py':sha(train_helper.__file__),
                                   'matched_graph_diagnostic_manifest.py':sha(manifest_helper.__file__)}}


def fresh_budget(path):
    budget=json.loads(Path(path).read_text())
    stamp=datetime.fromisoformat(budget['asof_utc'].replace('Z','+00:00'))
    age=(datetime.now(timezone.utc)-stamp).total_seconds()
    if not 0<=age<=120 or Decimal(str(budget['authorized_cap_usd']))!=530 or Decimal(str(budget['combined_bound_usd']))>530:
        raise ValueError('fresh admitted shared budget within530 required')
    return sha(path)


def execute(args,api=None):
    proof=json.loads(args.preflight.read_text());approval=json.loads(args.approval.read_text())
    if proof!=preflight(args.prepared):
        raise ValueError('preflight changed')
    if (approval['execute_authorized'] is not True or approval['project']!=PROJECT
            or approval['prepared_sha256']!=sha(args.prepared) or approval['runner_sha256']!=sha(__file__)
            or approval['preflight_sha256']!=sha(args.preflight)):
        raise ValueError('explicit root execution proof required')
    cap=Decimal(approval['cap_usd'])
    if not cap.is_finite() or cap<Decimal(proof['upper_usd']) or os.environ.get('TINKER_PROJECT_ID')!=PROJECT or args.out.exists():
        raise ValueError('fixed project/cap/fresh output required')
    prepared=json.loads(args.prepared.read_text());validate(prepared)
    policy_models,case_count=profile(prepared)
    paths={}
    for arm in ('point','intervention'):
        receipt=args.checkpoints/(arm+'_checkpoint.jsonl')
        if sha(receipt)!=prepared['checkpoint_receipt_sha256'][arm]:
            raise ValueError('saved state receipt changed')
        records=[json.loads(x) for x in receipt.read_text().splitlines()]
        if len(records)!=1 or set(records[0])!={'path'} or not records[0]['path'].startswith('tinker://'):
            raise ValueError('exact single private checkpoint receipt required')
        paths[arm]=records[0]['path']
    parent_start=[json.loads(x) for x in (args.checkpoints/'start.jsonl').read_text().splitlines()]
    parent_done=[json.loads(x) for x in (args.checkpoints/'complete.jsonl').read_text().splitlines()]
    if (len(parent_start)!=1 or parent_start[0]['prepared_sha256']!=prepared['primary_prepared_sha256']
            or parent_start[0]['project']!=PROJECT or len(parent_done)!=1
            or parent_done[0]['steps_per_arm']!=24 or Decimal(parent_done[0]['remaining_reserved_usd'])!=0):
        raise ValueError('completed parent training provenance required')
    budget_sha=fresh_budget(args.fresh_budget)
    if api is None:
        if importlib.metadata.version('tinker')!='0.33.1':
            raise ValueError('pinnedSDK required')
        import tinker as api
    inputs={name:[api.ModelInput.model_validate(r['model_input']) for r in model['inputs']]
            for name,model in prepared['models'].items()}
    args.out.mkdir();append(args.out/'start.jsonl',{'prepared_sha256':sha(args.prepared),'runner_sha256':sha(__file__),
        'preflight_sha256':sha(args.preflight),'approval_sha256':sha(args.approval),'project':PROJECT,
        'fresh_budget_sha256':budget_sha,'checkpoint_receipt_sha256':prepared['checkpoint_receipt_sha256']})
    spent=Decimal(0);reserved=Decimal(0);count=0;service=None
    try:
        fresh_budget(args.fresh_budget)  # Still fresh immediately before Service.
        append(args.out/'journal.jsonl',{'event':'service_intent','project':PROJECT})
        service=api.ServiceClient(project_id=PROJECT)
        for arm in policy_models:
            model=prepared['models'][policy_models[arm]]
            append(args.out/'journal.jsonl',{'event':'client_intent','arm':arm,'model':policy_models[arm]})
            if arm in paths:
                training=service.create_training_client_from_state_with_optimizer(paths[arm])
                info=training.get_info().model_dump(mode='json')
                if info.get('is_lora') is not True or info.get('lora_rank')!=32 or info.get('model_data',{}).get('model_name')!=policy_models[arm]:
                    raise ValueError('restored state model/rank changed')
                client=training.save_weights_and_get_sampling_client()
                append(args.out/'journal.jsonl',{'event':'state_restored','arm':arm,'receipt_sha256':prepared['checkpoint_receipt_sha256'][arm]})
            else:
                client=service.create_sampling_client(base_model=policy_models[arm])
            params=api.SamplingParams(max_tokens=416,temperature=0.,stop=model['stop_sequences'])
            for offset in range(0,case_count,INFLIGHT):
                pending=[];errors=[]
                for index in range(offset,offset+INFLIGHT):
                    row=model['inputs'][index];amount=charge(model,row['prompt_tokens'],0,416)
                    if spent+reserved+amount>cap:
                        errors.append(ValueError('remaining sample cap exceeded'));break
                    reserved+=amount
                    append(args.out/'journal.jsonl',{'event':'sample_intent','arm':arm,'index':index,'id':row['id'],'reserved_upper_usd':str(amount)})
                    try:
                        future=client.sample(prompt=inputs[policy_models[arm]][index],num_samples=1,sampling_params=params)
                        pending.append((index,row,amount,future))
                    except BaseException as exc:
                        errors.append(exc);break
                for index,row,amount,future in pending:
                    try:
                        result=future.result()
                        raw={'arm':arm,'id':row['id'],'sample':0,'prompt_sha256':row['prompt_sha256'],
                             'prompt_rendered_sha256':row['rendered_sha256'],
                             'sequences':[{'token_ids':list(s.tokens),'finish':str(s.stop_reason)} for s in result.sequences],
                             'prompt_cache_hit_tokens':result.prompt_cache_hit_tokens}
                        append(args.out/(arm+'_raw.jsonl'),raw)
                        if len(result.sequences)!=1:
                            raise ValueError('one first response required')
                        cost=charge(model,row['prompt_tokens'],result.prompt_cache_hit_tokens,len(result.sequences[0].tokens))
                        if cost>amount:raise ValueError('sample upper exceeded')
                        spent+=cost;reserved-=amount;count+=1
                        append(args.out/'journal.jsonl',{'event':'sample_settled','arm':arm,'id':row['id'],'usd':str(cost),
                            'accounted_usd':str(spent),'remaining_reserved_usd':str(reserved)})
                    except BaseException as exc:
                        errors.append(exc);append(args.out/'journal.jsonl',{'event':'unresolved','arm':arm,'id':row['id'],'error':type(exc).__name__})
                if errors:raise errors[0]
        if count!=prepared['calls'] or reserved!=0:raise ValueError('incomplete first-response coverage')
        append(args.out/'complete.jsonl',{'first_responses':count,'accounted_usd':str(spent),'remaining_reserved_usd':str(reserved),'training_steps':0})
    except BaseException as exc:
        append(args.out/'failure.jsonl',{'error':type(exc).__name__,'accounted_usd':str(spent),'uncertain_reserved_usd':str(reserved),'automatic_retry':False})
        raise
    finally:
        if service is not None:
            try:service.close('success' if (args.out/'complete.jsonl').exists() else 'errored').result()
            except Exception as exc:append(args.out/'journal.jsonl',{'event':'close_error','error':type(exc).__name__})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    for name in ('approval','fresh_budget','checkpoints','out'):p.add_argument('--'+name.replace('_','-'),type=Path)
    a=p.parse_args()
    if a.execute:
        if any(getattr(a,n) is None for n in ('approval','fresh_budget','checkpoints','out')):p.error('execute requires approval/fresh-budget/checkpoints/out')
        execute(a)
    else:
        with a.preflight.open('x') as f:f.write(encoded(preflight(a.prepared))+'\n')
