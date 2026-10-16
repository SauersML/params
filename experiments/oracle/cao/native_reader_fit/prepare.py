"""CPU-only preparation. Exposed32 are train/dev, never pristine evaluation."""
import argparse,hashlib,json,math,importlib.metadata
from pathlib import Path
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()
MODEL=Path('/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46')
GOLD_SHA='c09f7ffb103b2e449651724cbdf57476ed5a38c387119eb42067bd5225b291fa'
EXECUTION=Path('/Users/user/mpd-data/scratch/root/native_query_reader_local_v1/responses/local_execution_manifest.json')
EXECUTION_SHA='cd783e65267e16ae4215928a8fd8369cb6f0901d25eb85ccb7aa0ef5531e6b96'
def training_provenance(execution=EXECUTION,model=MODEL):
    assert sha(execution)==EXECUTION_SHA,'pinned prior execution manifest changed'
    previous=json.loads(Path(execution).read_text())
    assert previous['model_revision']==model.name
    hashes=previous['model_files_sha256']
    assert hashes and any(name.endswith('.safetensors') for name in hashes)
    assert all(sha(model/name)==value for name,value in hashes.items()),'local base bytes changed'
    dist=importlib.metadata.distribution('mlx-lm')
    sources={str(dist.locate_file('mlx_lm/'+name)):sha(dist.locate_file('mlx_lm/'+name)) for name in ('lora.py','tuner/trainer.py','tuner/utils.py','tuner/datasets.py')}
    runtime={name:importlib.metadata.version(name) for name in ('mlx','mlx-lm')}
    assert runtime=={'mlx':'0.32.3','mlx-lm':'0.32.0'}
    return dict(immutable_base_files_sha256=hashes,installed_trainer_sources_sha256=sources,runtime={'mlx':runtime['mlx'],'mlx_lm':runtime['mlx-lm']},label_serialization='round(rawdelta,3) in target only; rawfullprecision preserved in bindings/scoring',local_loss_correction='Installed default_loss steps<=actual_length includes first PAD; exact_completion_loss uses steps<actual_length. Same CE/maskstart; passed to installed train(loss=...) and its validation. Package unchanged.',prior_execution_manifest_path=str(execution),prior_execution_manifest_sha256=sha(execution),local_loss_source_sha256=sha(Path(__file__).with_name('exact_loss.py')))

def build(public,gold):
    assert public['status']==gold['status']=='complete'
    assert public['manifest']==gold['manifest']
    cases=public['manifest']['cases']; evidence={r['case_id']:r for r in public['rows']}; physics={r['case_id']:r for r in gold['rows']}
    assert len(cases)==len(evidence)==len(physics)==32
    groups={};result={'train':[],'valid':[]};bindings=[]
    for c in cases:groups.setdefault(c['group'],[]).append(c)
    assert len(groups)==4 and all(len(v)==8 for v in groups.values())
    for group,rows in sorted(groups.items()):
        for index,c in enumerate(sorted(rows,key=lambda x:x['case_id'])):
            cid=c['case_id'];r=physics[cid]
            def contrast(x):return (x['a_logp_nats']-x['b_logp_nats'])/math.log(2)
            delta=contrast(r['effects']['target_zero'])-contrast(r['conditional_baseline'])
            assert math.isfinite(delta) and abs(delta-r['effects']['target_zero']['delta_bits'])<1e-6
            # Original observed native state, not labels or teacher-generated explanations.
            prompt=('Predict the change in the original target model\'s conditional answer A/B log-probability contrast in bits when ALL listed native neurons are zeroed at the declared position. Delta = edited minus original. Answers are complete remaining suffixes given the shared prefix. Static readouts are approximate, not causal labels. Return ONLY JSON with exactly delta_bits (number) and reason (the string "Predicted joint-zero change in conditional A-versus-B log odds.").\n'+json.dumps(dict(question=c['user'],assistant_prefill=c['assistant_prefill'],shared_prefix_ids=c['prefix_ids'],remaining_a_ids=c['remaining_a_ids'],remaining_b_ids=c['remaining_b_ids'],answer_a=c['answer_5'],answer_b=c['answer_4'],native_evidence=evidence[cid]),sort_keys=True,separators=(',',':')))
            split='train' if index<6 else 'valid'
            result[split].append(dict(prompt=prompt,completion=json.dumps(dict(delta_bits=round(delta,3),reason='Predicted joint-zero change in conditional A-versus-B log odds.'),separators=(',',':'))))
            bindings.append(dict(case_id=cid,group=group,split=split,delta_bits=delta))
    return result,bindings

def main():
    p=argparse.ArgumentParser();p.add_argument('--public',required=True);p.add_argument('--gold',required=True);p.add_argument('--out',required=True);p.add_argument('--frozen',required=True);a=p.parse_args()
    assert sha(a.gold)==GOLD_SHA
    complete_provenance=training_provenance()
    out=Path(a.out);out.mkdir(exist_ok=False)
    data,bindings=build(json.load(open(a.public)),json.load(open(a.gold)))
    # Tokenizer only; no mlx or model weights imported/loaded.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    frozen=Path(a.frozen); fm=json.loads((frozen/'manifest.json').read_text()); assert sha(frozen/'requests.jsonl')==fm['requests_sha256']
    requests={r['question_id']:r for r in map(json.loads,(frozen/'requests.jsonl').read_text().splitlines()) if r['arm']=='true'}
    mapping={r['case_id']:r['question_id'] for r in json.loads((frozen/'mapping.json').read_text())}
    lengths=[]
    for split,rows in data.items():
        for row,binding in zip(rows,[b for b in bindings if b['split']==split]):
            request=requests[mapping[binding['case_id']]]
            prefix=request['token_ids']
            completion=tok.encode(row['completion'],add_special_tokens=False)+[tok.eos_token_id]
            row.clear();row.update(prompt_ids=prefix,completion_ids=completion,case_id=binding['case_id'])
            ids=prefix+completion
            assert len(ids)<=6144, f'No truncation allowed: {len(ids)}'
            lengths.append(len(ids))
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config=dict(model=str(MODEL),train=True,fine_tune_type='lora',data=str(out),seed=7281,num_layers=4,batch_size=1,iters=80,val_batches=8,learning_rate=1e-4,steps_per_report=5,steps_per_eval=40,save_every=40,max_seq_length=6144,grad_checkpoint=True,grad_accumulation_steps=1,mask_prompt=True,adapter_path=str(out/'adapters'),report_to=None,lora_parameters=dict(rank=8,scale=16.,dropout=0.,keys=['self_attn.q_proj','self_attn.v_proj','self_attn.o_proj','linear_attn.out_proj']))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    smoke=dict(config,iters=2,val_batches=1,steps_per_report=1,steps_per_eval=2,save_every=2,adapter_path=str(out/'smoke_adapters'))
    (out/'smoke_config.json').write_text(json.dumps(smoke,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(**complete_provenance,scope='EXPOSED32 training-fit diagnostic; dev is exposed; no pristine evaluation',public_sha256=sha(a.public),gold_sha256=sha(a.gold),model_revision=MODEL.name,bindings=bindings,frozen_reader_manifest_sha256=sha(frozen/'manifest.json'),exact_original_true_prompt_ids=True,dataset_sha256={split:sha(out/(split+'.jsonl')) for split in data},train_rows=24,dev_rows=8,max_tokens=max(lengths),total_tokens=sum(lengths),split_policy='Within each group sort case_id; first6 train,last2 dev; no effects or outcomes used for split'),indent=2)+'\n')
    print(json.dumps(dict(train=24,dev=8,max_tokens=max(lengths),total_tokens=sum(lengths),out=str(out))))
if __name__=='__main__':main()
