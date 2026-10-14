"""Offline canonical decoder for clarified original/permuted first responses.

No Service, repair, stripping or selection. Missing/length/parse failures remain
explicit; scorer retains every planned world. All model-specific source rendering
must exactly reproduce the frozen token IDs before response parsing begins.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import matched_diagnostic_sample as sample
from matched_graph_train import encoded,digest,sha,tokens

PACKAGES={'tinker':'0.33.1','tinker-cookbook':'0.5.7','transformers':'5.5.4','tokenizers':'0.22.2'}


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def relative(root,name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('relative bound raw path required')
    return Path(root)/path


def renderer(model,name,root):
    path=Path(root)/name.replace('/','--')
    actual={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
    if actual!=model['tokenizer_files']:raise ValueError('frozen tokenizer inventory changed')
    if {k:importlib.metadata.version(k) for k in PACKAGES}!=PACKAGES:raise ValueError('pinned decoder packages required')
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    if name.startswith('moonshotai/'):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls=get_class_from_dynamic_module('tokenization_kimi.TikTokenTokenizer',str(path),local_files_only=True)
        tokenizer=cls.from_pretrained(path,local_files_only=True)
    else:
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    return renderers.get_renderer(model['renderer'],tokenizer),renderers.get_text_content


def decode(args,bundles=None):
    for name in ('prepared','panel','manifest','raw_contract'):
        if sha(getattr(args,name))!=getattr(args,name+'_sha256'):raise ValueError('frozen '+name+' changed')
    if args.out.exists():raise ValueError('fresh decoder output required')
    p=json.loads(args.prepared.read_text());sample.validate(p)
    policy_models,case_count=sample.profile(p)
    policies=list(policy_models)
    panel=rows(args.panel);manifest=json.loads(args.manifest.read_text());contract=json.loads(args.raw_contract.read_text())
    if (p['source_panel_sha256']!=args.panel_sha256 or p['scoring_manifest_sha256']!=args.manifest_sha256
            or manifest['arms']!=policies or len(panel)!=case_count
            or [r['id'] for r in panel]!=[r['id'] for r in manifest['panel']]
            or contract['prepared_sha256']!=args.prepared_sha256 or contract['manifest_sha256']!=args.manifest_sha256
            or set(contract['arms'])!=set(policies)):
        raise ValueError('exact prepared/source/scoring/raw panel binding required')
    for source,case in zip(panel,manifest['panel']):
        if sha_text(source['prompt'])!=source['prompt_sha256'] or source['prompt_sha256']!=case['prompt_sha256']:
            raise ValueError('source prompt changed')
    for name,value in contract['files'].items():
        if sha(relative(args.raw_contract.parent,name))!=value:raise ValueError('raw receipt changed')
    if 'start.jsonl' not in contract['files']:raise ValueError('run start proof required')
    start=rows(relative(args.raw_contract.parent,'start.jsonl'))
    if len(start)!=1 or start[0]['prepared_sha256']!=args.prepared_sha256 or start[0]['project']!=sample.PROJECT:
        raise ValueError('raw run belongs to different prepared/project')
    raw_by_arm={};raw_hashes={}
    for arm in policies:
        bound={};entry=contract['arms'][arm];raw_hashes[arm]=None
        inputs=p['models'][policy_models[arm]]['inputs'];expected={r['id']:r for r in inputs}
        if [r['id'] for r in inputs]!=[r['id'] for r in panel]:raise ValueError('source token order changed')
        if entry is not None:
            path=relative(args.raw_contract.parent,entry['path'])
            if sha(path)!=entry['sha256']:raise ValueError('raw token file changed')
            raw_hashes[arm]=entry['sha256']
            for raw in rows(path):
                if raw['id'] not in expected or raw['id'] in bound or raw['arm']!=arm or type(raw['sample'])is not int or raw['sample']!=0:
                    raise ValueError('unexpected/duplicate/nonfirst response')
                row=expected[raw['id']]
                if raw['prompt_sha256']!=row['prompt_sha256'] or raw['prompt_rendered_sha256']!=row['rendered_sha256']:
                    raise ValueError('first-response rendered prompt changed')
                for sequence in raw['sequences']:
                    if any(type(i)is not int or not 0<=i<2**31 for i in sequence['token_ids']):raise ValueError('invalid raw token IDs')
                bound[raw['id']]=raw
        raw_by_arm[arm]=bound
    loaded={}
    for name,model in p['models'].items():
        bundle=bundles[name] if bundles is not None else renderer(model,name,args.tokenizers)
        rr,_=bundle
        for source,inp in zip(panel,model['inputs']):
            prompt=rr.build_generation_prompt([{'role':'user','content':source['prompt']}]).to_ints()
            if (source['prompt_sha256']!=inp['prompt_sha256'] or prompt!=tokens(inp['model_input'])
                    or digest(prompt)!=inp['rendered_sha256']):raise ValueError('canonical source rendering changed')
        if rr.get_stop_sequences()!=model['stop_sequences']:raise ValueError('canonical stops changed')
        loaded[name]=bundle
    args.out.mkdir();receipt={'schema_version':1,'manifest_sha256':args.manifest_sha256,
        'prepared_sha256':args.prepared_sha256,'source_panel_sha256':args.panel_sha256,'raw_contract_sha256':args.raw_contract_sha256,
        'decoder_sha256':sha(__file__),'validator_sha256':sha(sample.__file__),'arms':{},'missing_first_responses':{},
        'scope':'first raw model-specific canonical decoding; no repair; missing/length failures retained'}
    for arm in policies:
        rr,get_text=loaded[policy_models[arm]];output=[];missing=[]
        for source in panel:
            raw=raw_by_arm[arm].get(source['id'])
            if raw is None:missing.append(source['id']);continue
            keys=('id','stratum','prompt_sha256') if p.get('profile')=='fresh_transfer35_v1' else ('id','stratum','condition','original_id','prompt_sha256')
            row={k:source[k] for k in keys}
            if p.get('profile')=='fresh_transfer35_v1':
                row.update({k:source[k] for k in ('condition','original_id') if k in source})
            row.update(sample=0,raw_file_sha256=raw_hashes[arm],raw_row_sha256=digest(raw),
                       raw_sequences=raw['sequences'],completion=None,finish='invalid_sequence_count')
            if len(raw['sequences'])!=1:row['parse_error']='expected_one_first_sequence'
            else:
                sequence=raw['sequences'][0];row.update(token_ids=list(sequence['token_ids']),finish=sequence['finish'])
                try:
                    message,termination=rr.parse_response(list(sequence['token_ids']))
                    row.update(parsed_message=message,parse_termination=str(termination),completion=get_text(message))
                    if message.get('role')!='assistant' or not isinstance(row['completion'],str):raise ValueError('assistant text required')
                    clean=termination if type(termination)is bool else termination.is_clean
                    if not clean and str(row['finish']).lower() in ('stop','stopreason.stop'):row['parse_error']='renderer_malformed_stop_sequence'
                    if message.get('tool_calls'):row['parse_error']='unexpected_tool_calls'
                    if len(sequence['token_ids'])>416:row['parse_error']='response_exceeds_frozen_cap'
                except Exception as exc:row['parse_error']=type(exc).__name__
            output.append(row)
        file=arm+'.jsonl'
        with (args.out/file).open('x') as f:
            for row in output:f.write(encoded(row)+'\n')
            f.flush();os.fsync(f.fileno())
        receipt['arms'][arm]={'path':file,'sha256':sha(args.out/file),'actual_first_responses':len(output)}
        receipt['missing_first_responses'][arm]=missing
    with (args.out/'responses.json').open('x') as f:
        f.write(encoded(receipt)+'\n');f.flush();os.fsync(f.fileno())
    return receipt


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('prepared','panel','manifest','raw_contract'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
        parser.add_argument('--'+name.replace('_','-')+'-sha256',required=True)
    parser.add_argument('--tokenizers',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    print(encoded(decode(parser.parse_args())))
