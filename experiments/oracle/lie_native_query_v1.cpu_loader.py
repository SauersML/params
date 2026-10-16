"""Private bounded fixed-candidate specificity deployment. No model/API calls with --verify-only."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def verify(packet, adapter, base):
    m=json.loads((packet/'manifest.json').read_text())
    assert sha(packet/'runner.py')==m['runner_sha256'],'runner hash mismatch'
    for rel,want in m['source_files_sha256'].items():assert sha(packet/'source'/rel)==want,rel
    assert sha(adapter/'adapter_model.safetensors')==m['target']['adapter_sha256'],'adapter mismatch'
    for rel,want in m['tokenizer_files_sha256'].items():assert sha(base/rel)==want,rel
    assert base.name==m['target']['revision'],'snapshot revision mismatch'
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packet',type=Path,required=True);ap.add_argument('--adapter',type=Path,required=True);ap.add_argument('--out',type=Path);ap.add_argument('--base',type=Path);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--threads',type=int,default=2);ap.add_argument('--verify-only',action='store_true');ap.add_argument('--phase',choices=['public','gold'],required=True);ap.add_argument('--public-path',type=Path);ap.add_argument('--prediction-receipt',type=Path);ap.add_argument('--prediction-commit-sha');a=ap.parse_args()
    if a.phase=='gold' and not all([a.public_path,a.prediction_receipt,a.prediction_commit_sha]):ap.error('gold requires public-path, prediction-receipt and prediction-commit-sha')
    base=a.base or Path(os.environ.get('HF_HOME','/workspace/hf-home'))/'hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
    m=verify(a.packet,a.adapter,base)
    if a.verify_only:print('PASS: frozen source, runner, tokenizer, private lie adapter digests');return
    assert a.out is not None
    a.out.mkdir(parents=True,exist_ok=False)
    before_paths=[base/'config.json',base/'model.safetensors.index.json',*sorted(base.glob('*.safetensors')),a.adapter/'adapter_model.safetensors',a.adapter/'adapter_config.json']
    assert any(x.suffix=='.safetensors' and x.parent==base for x in before_paths),'no base weight shards'
    before={str(x):sha(x) for x in before_paths}
    started=time.monotonic();receipt=dict(phase=a.phase,manifest_sha256=sha(a.packet/'manifest.json'),runner_sha256=sha(a.packet/'runner.py'),source_revision=m['source_revision'],base_snapshot=str(base),adapter_path=str(a.adapter),file_hashes_before=before,loader_scope='full immutable original lie organism; no training or reader calls')
    (a.out/'loader_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    sys.path[:0]=[str(a.packet/'source/method/oracle'),str(a.packet/'source/experiments/oracle')]
    import torch,transformers,peft
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    assert a.threads>0,'positive thread count required'
    if a.device=='cuda':
        assert torch.cuda.is_available(),'CUDA required'
        torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(a.threads)
    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    model=PeftModel.from_pretrained(model,str(a.adapter),is_trainable=False,local_files_only=True).to(a.device).eval().requires_grad_(False)
    receipt.update(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,dtype=str(next(model.parameters()).dtype),attention_implementation='eager',device=a.device,threads=a.threads,performance_scope='CPU BF16; GPU performance noncomparable' if a.device=='cpu' else 'CUDA BF16',gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,loading_seconds=time.monotonic()-started)
    spec=importlib.util.spec_from_file_location('frozen_lie_runner',a.packet/'runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    error=None
    try:runner.run(model,tok,m,a.out/a.phase,phase=a.phase,public_path=a.public_path,prediction_receipt=a.prediction_receipt,prediction_commit_sha=a.prediction_commit_sha,manifest_sha256=sha(a.packet/'manifest.json'))
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        if a.device=='cuda':torch.cuda.synchronize()
        after={str(x):sha(x) for x in before_paths}
        receipt.update(file_hashes_after=after,weights_bytes_unchanged=before==after,error=error,total_loader_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if a.device=='cuda' else None)
        (a.out/'loader_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        assert before==after,'target weight/config bytes changed'
    phase_file=a.out/a.phase/('public_observations.json' if a.phase=='public' else 'result.json')
    result=json.loads(phase_file.read_text());assert result['status']=='complete','incomplete phase';assert len(result['rows'])==len(m['cases'])==32,'all32 cases required';assert result['hooks_before']==result['hooks_current'],'hook cleanup failed'
    artifacts={str(x.relative_to(a.out)):dict(size_bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(a.out.rglob('*')) if x.is_file()}
    (a.out/'required_artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')

if __name__=='__main__':main()
