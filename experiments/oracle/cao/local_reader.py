"""Exact-token local MLX first-response runner. No loader executes on import."""
import argparse, hashlib, importlib.metadata, json, math, os, platform, sys, time
from pathlib import Path
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def load_json(p):return json.loads(Path(p).read_text())
def records(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def require(ok,msg):
    if not ok:raise ValueError(msg)
def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def preflight(frozen,model_path):
    frozen=Path(frozen);model_path=Path(model_path);m=load_json(frozen/'manifest.json')
    require(m['backend']=='mlx' and m['quantization']=='4bit' and m['max_tokens']==512 and m['temperature']==0,'local MLX protocol required')
    require(m['runner_sha256']==sha(__file__),'runner hash mismatch')
    require(isinstance(m['model_revision'],str) and len(m['model_revision'])==40,'pinned model revision required')
    require(sha(m['source_frozen_manifest_path'])==m['source_frozen_manifest_sha256'],'original frozen manifest hash')
    for path,key in [(frozen/'requests.jsonl','requests_sha256'),(frozen/'mapping.json','mapping_sha256'),(m['panel_path'],'panel_sha256'),(m['public_observations_path'],'public_observations_sha256'),(m['regression_model_path'],'regression_model_sha256')]:require(sha(path)==m[key],key)
    tokpath=Path(m['tokenizer_path']);require(all(sha(tokpath/name)==digest for name,digest in m['tokenizer_sha256'].items()),'tokenizer hashes')
    require(all(sha(model_path/name)==digest for name,digest in m['tokenizer_sha256'].items()),'loaded tokenizer differs from frozen tokenizer')
    cfg=load_json(model_path/'config.json');textcfg=cfg.get('text_config',cfg);vocab=textcfg['vocab_size'];cap=m['context_cap']
    require(type(vocab) is int and vocab>0 and type(cap) is int and cap>0,'vocab/context config')
    declared=textcfg.get('max_position_embeddings');require(type(declared) is int and cap<=declared,'context exceeds model config')
    quant=cfg.get('quantization',cfg.get('quantization_config',{}));require(quant.get('bits')==4,'model config is not declared 4bit')
    rows=records(frozen/'requests.jsonl');mapping=load_json(frozen/'mapping.json');panel=load_json(m['panel_path']);public=load_json(m['public_observations_path'])
    require(public['manifest']==panel and public['status']=='complete','public panel binding')
    require(len(mapping)==32 and len({b['question_id'] for b in mapping})==32 and len({b['case_id'] for b in mapping})==32,'mapping coverage')
    require({b['case_id'] for b in mapping}=={c['case_id'] for c in panel['cases']},'panel mapping coverage')
    require(len(rows)==96 and len({r['request_id'] for r in rows})==96,'96 unique requests')
    expected={b['question_id']+'_'+arm for b in mapping for arm in ('true','absent','mismatched')};require({r['request_id'] for r in rows}==expected,'arm coverage')
    for r in rows:
        ids=r['token_ids'];require(r['request_id']==r['question_id']+'_'+r['arm'] and r['arm'] in ('true','absent','mismatched'),'arm binding')
        require(isinstance(ids,list) and ids and all(type(v) is int and 0<=v<vocab for v in ids),'token IDs')
        require(type(r['prompt_tokens']) is int and len(ids)==r['prompt_tokens'] and len(ids)+512<=cap,'prompt count/context')
    for b in mapping:
        arms=[r for r in rows if r['question_id']==b['question_id']];suffix=arms[0]['common_suffix_tokens'];require(type(suffix) is int and suffix>100,'suffix count')
        require(len({len(r['token_ids']) for r in arms})==1,'arm lengths')
        require(all(r['common_suffix_tokens']==suffix and suffix<len(r['token_ids']) and r['token_ids'][-suffix:]==arms[0]['token_ids'][-suffix:] for r in arms),'common suffix positions')
    weights=sorted(model_path.glob('*.safetensors'));require(weights,'local weights missing')
    model_hashes={p.name:sha(p) for p in sorted(set(weights)|{model_path/'config.json'}|set(model_path.glob('*.index.json')))}
    if 'model_files_sha256' in m:require(model_hashes==m['model_files_sha256'],'frozen model files mismatch')
    return m,rows,model_hashes

def run(frozen,model_path,out,*,loader=None,streamer=None,sampler_factory=None,runtime=None):
    wall_start=time.perf_counter()
    frozen=Path(frozen);model_path=Path(model_path);out=Path(out);require(not out.exists(),'new output directory required')
    m,rows,model_hashes=preflight(frozen,model_path)
    if runtime is None:
        runtime={'python':sys.version,'platform':platform.platform(),'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm')}
        distribution=importlib.metadata.distribution('mlx-lm')
        runtime['mlx_lm_source_sha256']={name:sha(distribution.locate_file(name)) for name in ('mlx_lm/generate.py','mlx_lm/utils.py','mlx_lm/sample_utils.py','mlx_lm/tokenizer_utils.py')}
    if 'runtime' in m:require(runtime==m['runtime'],'frozen runtime mismatch')
    if loader is None:
        from mlx_lm import load,stream_generate
        from mlx_lm.sample_utils import make_sampler
        loader,streamer,sampler_factory=load,stream_generate,make_sampler
    out.mkdir();(out/'manifest.json').write_bytes((frozen/'manifest.json').read_bytes())
    execution={'schema':'native_local_mlx_execution_v1','model':m['model'],'model_revision':m['model_revision'],'model_path':str(model_path),'model_files_sha256':model_hashes,'runtime':runtime,'runner_sha256':sha(__file__),'reader_manifest_sha256':sha(frozen/'manifest.json'),'protocol':{'max_tokens':512,'temperature':0,'prefill_step_size':512,'prompt_cache':None,'draft_model':None,'retries':0}}
    (out/'local_execution_manifest.json').write_text(json.dumps(execution,indent=2)+'\n')
    cfg=load_json(model_path/'config.json');vocab=cfg.get('text_config',cfg)['vocab_size']
    fatal=None;model=tokenizer=None
    try:model,tokenizer=loader(str(model_path))
    except Exception as exc:fatal='load '+type(exc).__name__+': '+str(exc)
    errors=0
    for request in rows:
        request_start=time.perf_counter();performance={}
        tokens=[];parts=[];finish='error';error=fatal
        if fatal is None:
            try:
                eos=set(tokenizer.eos_token_ids);require(eos and all(type(v) is int for v in eos),'EOS IDs unavailable')
                final=False
                for chunk in streamer(model,tokenizer,prompt=list(request['token_ids']),max_tokens=512,sampler=sampler_factory(temp=0),prefill_step_size=512,draft_model=None):
                    require(not final,'chunk after terminal result')
                    token=int(chunk.token);require(chunk.prompt_tokens==len(request['token_ids']) and chunk.generation_tokens==len(tokens)+1,'stream counts mismatch')
                    require(len(tokens)<512 and 0<=token<vocab,'generated tokens invalid')
                    tokens.append(token);parts.append(chunk.text)
                    if chunk.finish_reason is not None:
                        finish=chunk.finish_reason;final=True
                        performance={key:getattr(chunk,key,None) for key in ('prompt_tps','generation_tps','peak_memory')}
                        require(all(v is None or (type(v) in (int,float) and math.isfinite(v) and v>=0) for v in performance.values()),'invalid performance metadata')
                        require(finish in ('stop','length'),'unknown finish')
                        require((finish=='stop' and token in eos) or (finish=='length' and len(tokens)==512 and token not in eos),'terminal token mismatch')
                    else:require(token not in eos,'unterminated EOS')
                require(final,'stream missing terminal result')
            except Exception as exc:
                finish='error';error=type(exc).__name__+': '+str(exc)
                if isinstance(exc,MemoryError) or any(word in str(exc).lower() for word in ('out of memory','insufficient memory','allocation failed','metal out of memory')):fatal='aborted after OOM: '+error
        if finish=='error':errors+=1
        tokenhash=hashlib.sha256(json.dumps(request['token_ids'],separators=(',',':')).encode()).hexdigest()
        common={'request_id':request['request_id'],'finish_reason':finish,'prompt_tokens':len(request['token_ids']),'generated_tokens':len(tokens),'request_tokens_sha256':tokenhash,'elapsed_seconds':time.perf_counter()-request_start,**performance}
        append(out/'raw.jsonl',dict(common,tokens=tokens,error=error))
        append(out/'responses.jsonl',dict(common,completion=''.join(parts) if finish!='error' else None,error=error))
    performance_result={'wall_seconds_including_preflight_and_load':time.perf_counter()-wall_start,'peak_memory_gb':None}
    if loader.__module__.startswith('mlx_lm'):
        import mlx.core as mx
        performance_result['peak_memory_gb']=mx.get_peak_memory()/1e9
    (out/'execution_result.json').write_text(json.dumps(performance_result,indent=2)+'\n')
    receipt={'schema':'native_local_mlx_prediction_commit_v1','requests':96,'errors':errors,'responses_sha256':sha(out/'responses.jsonl'),'raw_sha256':sha(out/'raw.jsonl'),'requests_sha256':m['requests_sha256'],'panel_sha256':m['panel_sha256'],'public_observations_sha256':m['public_observations_sha256'],'reader_manifest_sha256':sha(frozen/'manifest.json'),'local_execution_manifest_sha256':sha(out/'local_execution_manifest.json'),'execution_result_sha256':sha(out/'execution_result.json'),'performance':performance_result,'backend':'mlx','api_calls':0,'api_spend_usd':0}
    with (out/'prediction_commit.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.frozen,a.model_path,a.out)))
