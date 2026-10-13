"""Offline first-response decoder bound to source prompts and frozen token receipts.

Uses the pinned renderer without repair, truncation, selection, or service calls.
Missing responses are omitted from arm JSONL and explicitly listed in the receipt;
the physical scorer then retains them as failures in every planned denominator.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import matched_graph_train as train

MODEL=train.MODEL
REVISION='b968826d9c46dd6066d109eabc6255188de91218'
RENDERER='qwen3_disable_thinking'
TOKENIZER_FILES={'config.json':'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
'generation_config.json':'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
 'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
 'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}
PACKAGES={'tinker':'0.33.1','tinker-cookbook':'0.5.7','transformers':'5.5.4','tokenizers':'0.22.2'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prompt_sha(text):return hashlib.sha256(text.encode()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def relative(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('relative bound receipt path required')
    return Path(root)/p


def load_renderer(tokenizer):
    actual={p.name:sha(p) for p in Path(tokenizer).iterdir() if p.is_file()}
    if actual!=TOKENIZER_FILES:raise ValueError('exact frozen tokenizer inventory required')
    if {p:importlib.metadata.version(p) for p in PACKAGES}!=PACKAGES:raise ValueError('pinned offline renderer packages required')
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
    return renderers.get_renderer(RENDERER,tok),renderers.get_text_content


def write_rows(path,values):
    with Path(path).open('x') as f:
        for value in values:f.write(train.encoded(value)+'\n')
        f.flush();os.fsync(f.fileno())


def decode(args,renderer_bundle=None):
    for name in ('manifest','source_panel','prepared','raw_contract'):
        if sha(getattr(args,name))!=getattr(args,name+'_sha256'):raise ValueError('frozen '+name+' bytes changed')
    if args.out.exists():raise ValueError('fresh immutable decoder output required')
    prepared=json.loads(args.prepared.read_text());c=train.validate(prepared)
    manifest=json.loads(args.manifest.read_text());source=rows(args.source_panel);raw_contract=json.loads(args.raw_contract.read_text())
    if train.digest(source)!=c['panel_sha256']:raise ValueError('source panel differs from prepared contract')
    if raw_contract['prepared_sha256']!=args.prepared_sha256 or raw_contract['scoring_manifest_sha256']!=args.manifest_sha256 or set(raw_contract['arms'])!={'base',*train.ARMS}:raise ValueError('raw receipt belongs to different prepared/scoring panel')
    if manifest['arms']!=['base',*train.ARMS] or [(r['stratum'],r['id']) for r in source]!=[(r['stratum'],r['id']) for r in c['panel']] or [(r['stratum'],r['id']) for r in source]!=[(r['stratum'],r['id']) for r in manifest['panel']]:raise ValueError('exact panel ordering/identity required')
    for s,p in zip(source,manifest['panel']):
        if s['prompt_sha256']!=prompt_sha(s['prompt']) or s['prompt_sha256']!=p['prompt_sha256']:raise ValueError('original source prompt hash differs')
    for name,expected in raw_contract['files'].items():
        if sha(relative(args.raw_contract.parent,name))!=expected:raise ValueError('raw execution receipt changed')
    if 'start.jsonl' not in raw_contract['files']:raise ValueError('raw run start binding required')
    start=rows(relative(args.raw_contract.parent,'start.jsonl'))
    if len(start)!=1 or start[0]['prepared_sha256']!=args.prepared_sha256 or start[0]['project']!=train.PROJECT:raise ValueError('raw execution start/prepared/project mismatch')
    expected={(r['stratum'],r['id']):i for i,r in enumerate(source)};raws={};raw_hashes={}
    for arm,entry in raw_contract['arms'].items():
        if entry is None:raws[arm]={};raw_hashes[arm]=None;continue
        path=relative(args.raw_contract.parent,entry['path'])
        if sha(path)!=entry['sha256']:raise ValueError('raw token receipt changed')
        raw_hashes[arm]=entry['sha256'];bound={}
        for raw in rows(path):
            key=(raw['stratum'],raw['id'])
            if key not in expected or key in bound or raw['arm']!=arm or type(raw['sample'])is not int or raw['sample']!=0:raise ValueError('unexpected/duplicate/nonfirst raw response')
            if raw['prompt_rendered_sha256']!=c['panel'][expected[key]]['rendered_sha256']:raise ValueError('raw rendered prompt hash mismatch')
            for seq in raw['sequences']:
                if any(type(i)is not int or not 0<=i<2**31 for i in seq['token_ids']):raise ValueError('invalid raw token IDs')
            bound[key]=raw
        raws[arm]=bound
    renderer,get_text=renderer_bundle or load_renderer(args.tokenizer)
    for i,s in enumerate(source):
        prompt=renderer.build_generation_prompt([{'role':'user','content':s['prompt']}])
        if prompt.to_ints()!=train.tokens(prepared['prompts'][i]) or train.digest(prompt.to_ints())!=c['panel'][i]['rendered_sha256']:raise ValueError('pinned source rendering differs from prepared tokens')
    if renderer.get_stop_sequences()!=prepared['stop_sequences']:raise ValueError('pinned stop sequences differ')
    args.out.mkdir();receipt={'schema_version':1,'manifest_sha256':args.manifest_sha256,'prepared_sha256':args.prepared_sha256,'source_panel_sha256':args.source_panel_sha256,'raw_contract_sha256':args.raw_contract_sha256,'decoder_sha256':sha(__file__),'validator_sha256':sha(train.__file__),'model':MODEL,'tokenizer_revision':REVISION,'renderer':RENDERER,'packages':PACKAGES,'tokenizer_files':TOKENIZER_FILES,'arms':{},'missing_first_responses':{},'scope':'first raw token samples only; canonical renderer parsing, no repair; missing/length/parser failures retained'}
    for arm in ('base',*train.ARMS):
        output=[];missing=[]
        for s in source:
            key=(s['stratum'],s['id']);raw=raws[arm].get(key)
            if raw is None:missing.append({'stratum':s['stratum'],'id':s['id']});continue
            row={'id':s['id'],'stratum':s['stratum'],'sample':0,'prompt_sha256':s['prompt_sha256'],'prompt_rendered_sha256':raw['prompt_rendered_sha256'],'raw_row_sha256':train.digest(raw),'raw_file_sha256':raw_hashes[arm],'raw_sequences':raw['sequences'],'completion':None,'finish':'invalid_sequence_count'}
            if len(raw['sequences'])!=1:row['parse_error']='expected_one_first_sequence'
            else:
                seq=raw['sequences'][0];row.update(token_ids=list(seq['token_ids']),finish=seq['finish'])
                try:
                    message,termination=renderer.parse_response(list(seq['token_ids']))
                    row['parsed_message']=message;row['parse_termination']=str(termination)
                    row['completion']=get_text(message)
                    if message.get('role')!='assistant' or not isinstance(row['completion'],str):raise ValueError('assistant text response required')
                    clean=termination if type(termination)is bool else termination.is_clean
                    if not clean and str(seq['finish']).lower() in ('stop','stopreason.stop'):row['parse_error']='renderer_malformed_stop_sequence'
                    if message.get('tool_calls'):row['parse_error']='unexpected_tool_calls'
                    if len(seq['token_ids'])>416:row['parse_error']='response_exceeds_frozen_cap'
                except Exception as exc:row['parse_error']=type(exc).__name__
            output.append(row)
        file=arm+'.jsonl';write_rows(args.out/file,output);receipt['arms'][arm]={'path':file,'sha256':sha(args.out/file),'actual_first_responses':len(output)};receipt['missing_first_responses'][arm]=missing
    with (args.out/'responses.json').open('x') as f:f.write(train.encoded(receipt)+'\n');f.flush();os.fsync(f.fileno())
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','source_panel','prepared','raw_contract'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
        p.add_argument('--'+name.replace('_','-')+'-sha256',required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(train.encoded(decode(a)))
