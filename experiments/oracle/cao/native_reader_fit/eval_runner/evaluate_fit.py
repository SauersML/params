"""Isolated first-answer32 fit evaluation; no training directory writes."""
import argparse,ctypes,hashlib,importlib.util,importlib.metadata,json,math,os,platform,subprocess,sys,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def physical_bytes(pid):
    if sys.platform!='darwin':raise RuntimeError('physical-footprint guard requires macOS')
    class Rusage(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_uint8*16)]+[(n,ctypes.c_uint64) for n in ('user_time','system_time','pkg_idle_wakeups','interrupt_wakeups','pageins','wired_size','resident_size','phys_footprint','proc_start_abstime','proc_exit_abstime')]
    lib=ctypes.CDLL('/usr/lib/libproc.dylib');info=Rusage()
    lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.proc_pid_rusage.restype=ctypes.c_int
    if lib.proc_pid_rusage(pid,0,ctypes.byref(info))!=0:raise ProcessLookupError(pid)
    return info.phys_footprint

def prepare(a):
    cfg=json.load(open(a.config));d=Path(cfg['data']);meta=json.load(open(d/'provenance.json'));frozen=Path(a.frozen);fm=json.load(open(frozen/'manifest.json'))
    assert sha(a.baseline_responses)=='65997e8da14092ff6822e5ad64c9de8f247a7e5721d275d161f37e86d9b9b509'
    assert sha(a.primary_source)=='e0b6b4580e2bcd4fe67b6fef06e509696c9adaee1bfee9ec42fb8e0b80ec9650'
    assert Path(cfg['model']).name==fm['model_revision'],'model revision changed'
    assert all(sha(Path(cfg['model'])/name)==value for name,value in fm['tokenizer_sha256'].items()),'tokenizer changed'
    assert sha(frozen/'requests.jsonl')==fm['requests_sha256'] and sha(frozen/'mapping.json')==fm['mapping_sha256']
    requests={r['question_id']:r for r in map(json.loads,(frozen/'requests.jsonl').read_text().splitlines()) if r['arm']=='true'}
    mapping={r['case_id']:r['question_id'] for r in json.load(open(frozen/'mapping.json'))}
    rows=[]
    for split in ('train','valid'):
        assert sha(d/(split+'.jsonl'))==meta['dataset_sha256'][split]
        part=[dict(r,split=split) for r in map(json.loads,(d/(split+'.jsonl')).read_text().splitlines())]
        assert len(part)==(24 if split=='train' else 8)
        for r in part:assert r['prompt_ids']==requests[mapping[r['case_id']]]['token_ids']
        rows.extend(part)
    assert len({r['case_id'] for r in rows})==32
    base={name:sha(Path(cfg['model'])/name) for name in meta['immutable_base_files_sha256']}
    assert base==meta['immutable_base_files_sha256']
    adapter=Path(cfg['adapter_path']);files=sorted([*adapter.glob('*.safetensors'),adapter/'adapter_config.json'])
    assert (adapter/'adapters.safetensors').is_file() and sha(adapter/'adapters.safetensors')==a.adapter_sha256
    adapterhash={p.name:sha(p) for p in files}
    return cfg,meta,rows,mapping,base,adapterhash

def child(a):
    cfg,meta,rows,mapping,base,adapterhash=prepare(a);out=Path(a.out)
    receipt=dict(schema='native_fit_eval_execution_v1',source_sha256=sha(__file__),config_sha256=sha(a.config),dataset_sha256=meta['dataset_sha256'],base_sha256=base,adapter_sha256=adapterhash,baseline_sha256=sha(a.baseline_responses),greedy=True,max_tokens=512,prefill_step_size=512,draft_model=None,prompt_cache=None,retries=0,status='started')
    (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('frozen_score',a.primary_source);score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    baseline={r['request_id']:r for r in map(json.loads,Path(a.baseline_responses).read_text().splitlines())};truth={r['case_id']:r['delta_bits'] for r in meta['bindings']}
    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(32<<30);mx.set_cache_limit(2<<30)
    receipt['runtime']={k:importlib.metadata.version(k) for k in ('mlx','mlx-lm')}
    model,tok=load(cfg['model'],adapter_path=cfg['adapter_path']);eos=set(tok.eos_token_ids);results=[];start=time.monotonic()
    with (out/'fit_predictions.jsonl').open('x') as log:
        for row in rows:
            tick=time.monotonic();parts=[];tokens=[];finish='error';performance={}
            try:
                final=False
                for chunk in stream_generate(model,tok,prompt=list(row['prompt_ids']),max_tokens=512,sampler=make_sampler(temp=0),prefill_step_size=512,draft_model=None):
                    assert not final and chunk.prompt_tokens==len(row['prompt_ids']) and chunk.generation_tokens==len(tokens)+1
                    tokens.append(int(chunk.token));parts.append(chunk.text)
                    if chunk.finish_reason is not None:
                        final=True;finish=chunk.finish_reason
                        assert (finish=='stop' and tokens[-1] in eos) or (finish=='length' and len(tokens)==512 and tokens[-1] not in eos)
                        performance={key:getattr(chunk,key,None) for key in ('prompt_tps','generation_tps','peak_memory')}
                assert final
                response=dict(completion=''.join(parts),finish_reason=finish)
            except Exception as exc:
                response=dict(completion=None,finish_reason='error',error=type(exc).__name__+': '+str(exc))
                # OOM is terminal, not32 model retries. Completed partialrows remain.
                if isinstance(exc,MemoryError) or 'memory' in str(exc).lower():raise
            pred,invalid,_=score.parse(response,{})
            b=baseline[mapping[row['case_id']]+'_true'];bp,be,_=score.parse(b,{})
            gold=truth[row['case_id']]
            r=dict(case_id=row['case_id'],split=row['split'],gold_delta_bits=gold,baseline=b,adapted=response,token_ids=tokens,prompt_tokens=len(row['prompt_ids']),prompt_ids_sha256=hashlib.sha256(json.dumps(row['prompt_ids'],separators=(',',':')).encode()).hexdigest(),generated_tokens=len(tokens),elapsed_seconds=time.monotonic()-tick,performance=performance,adapted_invalid=invalid,baseline_invalid=be,adapted_error=60. if invalid else min(abs(pred-gold),60.),baseline_error=60. if be else min(abs(bp-gold),60.))
            results.append(r);log.write(json.dumps(r,allow_nan=False)+'\n');log.flush()
    assert len(results)==32
    assert all(sha(Path(cfg['model'])/n)==v for n,v in base.items())
    assert {n:sha(Path(cfg['adapter_path'])/n) for n in adapterhash}==adapterhash
    summary={split:dict(rows=len(part),adapted_capped_mae=sum(r['adapted_error'] for r in part)/len(part),baseline_capped_mae=sum(r['baseline_error'] for r in part)/len(part),adapted_invalid=sum(r['adapted_invalid'] is not None for r in part),baseline_invalid=sum(r['baseline_invalid'] is not None for r in part)) for split in ('train','valid') for part in [[r for r in results if r['split']==split]]}
    receipt.update(status='complete',base_and_adapter_unchanged=True,seconds=time.monotonic()-start,peak_mlx_memory_bytes=mx.get_peak_memory())
    (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'fit_summary.json').write_text(json.dumps(dict(scope='24TRAIN resubstitution fit;8DEV exposed, not pristine evaluation. No semantic interpretation.',summary=summary),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser()
    for name in ('config','frozen','primary-source','baseline-responses','adapter-sha256','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--launch',action='store_true');p.add_argument('--child',action='store_true');a=p.parse_args();assert a.launch
    if a.child:return child(a)
    out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic();peak=0;reason=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:peak=max(peak,physical_bytes(proc.pid))
            except ProcessLookupError:pass
            except Exception as exc:reason='memory_guard_error: '+repr(exc)
            if peak>32<<30:reason='physical_memory_exceeded32GiB'
            if time.monotonic()-start>600:reason='wall_budget_exhausted600s'
            if reason:proc.kill();break
            time.sleep(.5)
        code=proc.wait()
    (out/'guard_status.json').write_text(json.dumps(dict(returncode=code,status=reason or ('complete' if code==0 else 'failed'),peak_physical_bytes=peak,wall_seconds=time.monotonic()-start),indent=2)+'\n')
    if reason or code:raise SystemExit(1)
if __name__=='__main__':main()
