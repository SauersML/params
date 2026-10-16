"""Launched separately AFTER successful fit:32 first greedy answers, no retries."""
import argparse,importlib.util,json,math,time,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--primary-source',required=True);p.add_argument('--baseline-responses',required=True);p.add_argument('--mapping',required=True);p.add_argument('--launch',action='store_true');a=p.parse_args();assert a.launch
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(a.baseline_responses)=='65997e8da14092ff6822e5ad64c9de8f247a7e5721d275d161f37e86d9b9b509'
    assert sha(a.primary_source)=='e0b6b4580e2bcd4fe67b6fef06e509696c9adaee1bfee9ec42fb8e0b80ec9650'
    d=Path(a.data);cfg=json.load(open(d/'config.json'));meta=json.load(open(d/'provenance.json'))
    assert (d/'adapters/adapters.safetensors').is_file()
    assert all(sha(d/(name+'.jsonl'))==value for name,value in meta['dataset_sha256'].items())
    assert not (d/'fit_predictions.jsonl').exists() and not (d/'fit_summary.json').exists()
    spec=importlib.util.spec_from_file_location('frozen_score',a.primary_source);score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    baseline={r['request_id']:r for r in map(json.loads,Path(a.baseline_responses).read_text().splitlines())}
    mapping={r['case_id']:r['question_id'] for r in json.load(open(a.mapping))}
    truth={r['case_id']:r for r in meta['bindings']};results=[]
    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(32<<30);mx.set_cache_limit(2<<30)
    model,tok=load(cfg['model'],adapter_path=cfg['adapter_path'])
    start=time.monotonic()
    with (d/'fit_predictions.jsonl').open('x') as log:
        for split in ('train','valid'):
            for row in map(json.loads,(d/(split+'.jsonl')).read_text().splitlines()):
                assert time.monotonic()-start<600,'Evaluation wall budget600s exhausted; partial predictions retained'
                chunks=list(stream_generate(model,tok,prompt=row['prompt_ids'],max_tokens=512,sampler=make_sampler(temp=0),prefill_step_size=512,draft_model=None))
                response=dict(completion=''.join(x.text for x in chunks),finish_reason=chunks[-1].finish_reason if chunks else 'error')
                pred,invalid,_=score.parse(response,{})
                b=baseline[mapping[row['case_id']]+'_true'];bp,be,_=score.parse(b,{})
                gold=truth[row['case_id']]['delta_bits']
                result=dict(case_id=row['case_id'],split=split,gold_delta_bits=gold,baseline=b,adapted=response,adapted_invalid=invalid,baseline_invalid=be,adapted_error=60. if invalid else min(abs(pred-gold),60.),baseline_error=60. if be else min(abs(bp-gold),60.))
                results.append(result);log.write(json.dumps(result)+'\n');log.flush()
    summary={split:{'rows':len(rows),'adapted_capped_mae':sum(r['adapted_error'] for r in rows)/len(rows),'baseline_capped_mae':sum(r['baseline_error'] for r in rows)/len(rows),'adapted_invalid':sum(r['adapted_invalid'] is not None for r in rows),'baseline_invalid':sum(r['baseline_invalid'] is not None for r in rows)} for split in ('train','valid') for rows in [[r for r in results if r['split']==split]]}
    (d/'fit_summary.json').write_text(json.dumps(dict(scope='Resubstitution24TRAIN primary training-fit check;8DEV exposed/nonpristine. Not unseen generalization or semantic explanations.',summary=summary,peak_memory_gb=mx.get_peak_memory()/1e9),indent=2)+'\n')
if __name__=='__main__':main()
