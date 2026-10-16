"""Reuse installed MLX LoRA, with exact original prompt-token dataset adapter."""
import argparse,json,types,sys,hashlib,importlib.metadata,functools
from pathlib import Path
from exact_data import ExactDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    import mlx.core as mx
    import mlx_lm.lora as lora
    from mlx.utils import tree_flatten
    mx.set_memory_limit(32<<30);mx.set_cache_limit(2<<30)
    cfg=dict(lora.CONFIG_DEFAULTS);cfg.update(json.load(open(a.config)))
    assert cfg['num_layers']==1 and cfg['iters']==80 and cfg['lora_parameters']['keys']==['self_attn.q_proj','self_attn.v_proj','self_attn.o_proj']
    mx.disable_compile()
    model_cfg=json.load(open(Path(cfg['model'])/'config.json'))['text_config']
    assert model_cfg['num_hidden_layers']==40 and model_cfg['full_attention_interval']==4
    assert model_cfg['num_hidden_layers']%model_cfg['full_attention_interval']==0
    data=Path(cfg['data'])
    provenance=json.load(open(data/'provenance.json'))
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        return h.hexdigest()
    assert importlib.metadata.version('mlx')=='0.32.3' and importlib.metadata.version('mlx-lm')=='0.32.0'
    def verify_base():
        assert all(sha(Path(cfg['model'])/name)==value for name,value in provenance['immutable_base_files_sha256'].items()),'base weights/config changed'
    verify_base()
    assert all(sha(path)==value for path,value in provenance['installed_trainer_sources_sha256'].items()),'installed trainer changed'
    assert all(sha(data/(name+'.jsonl'))==value for name,value in provenance['dataset_sha256'].items()),'dataset changed'
    receipt_path=data/(Path(cfg['adapter_path']).name+'_execution.json')
    assert not receipt_path.exists(),'new execution required'
    receipt=dict(variant=json.load(open(data/'variant.json')),compile_disabled=True,config_sha256=sha(a.config),runtime={k:importlib.metadata.version(k) for k in ('mlx','mlx-lm','numpy')},source_sha256={q.name:sha(q) for q in Path(__file__).parent.glob('*.py')},dataset_sha256=provenance['dataset_sha256'],base_sha256=provenance['immutable_base_files_sha256'],seed=cfg['seed'],status='started')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    def datasets(args,tokenizer):
        load=lambda name:ExactDataset([json.loads(s) for s in (data/(name+'.jsonl')).read_text().splitlines()])
        return load('train'),load('valid'),None
    lora.load_dataset=datasets
    from exact_loss import exact_completion_loss
    from frozen_prefix_loss import frozen_prefix_loss
    mixed_loss=frozen_prefix_loss(exact_completion_loss,1)
    lora.train=functools.partial(lora.train,loss=mixed_loss)
    # Existing function remains responsible for freezing, adapter conversion,
    # optimization, checkpoints, training masks and validation.
    original=lora.print_trainable_parameters
    def report(model):
        original(model)
        params=tree_flatten(model.trainable_parameters())
        (data/(Path(cfg['adapter_path']).name+'_parameters.json')).write_text(json.dumps(dict(count=sum(v.size for _,v in params),names=[k for k,_ in params]),indent=2)+'\n')
        assert params,'No adapters matched declared projection keys'
        assert len(model.layers)==40 and not model.layers[-1].is_linear,'finalblock must be fullattention'
        assert all('.layers.39.' in k for k,_ in params),'adapter outside finalattention'
        assert all(k.endswith(('.lora_a','.lora_b')) for k,_ in params),'Non-LoRA parameter trainable'
        assert all(any('.'+key+'.' in k for key in cfg['lora_parameters']['keys']) for k,_ in params),'Unexpected projection trainable'
        # Real same-example pre-update parity, original eval vs mixedtrain.
        from mlx_lm.tuner.trainer import iterate_batches
        example=json.loads((data/'train.jsonl').read_text().splitlines()[0])
        pair=ExactDataset([example]).process(example)
        batch,lengths=next(iterate_batches([pair],batch_size=1,max_seq_length=cfg['max_seq_length']))
        model.eval();reference,nref=exact_completion_loss(model,batch,lengths);reference=float(reference.item())
        model.train();actual,nactual=mixed_loss(model,batch,lengths);actual=float(actual.item())
        parity=dict(case_id=example['case_id'],eval_loss_nats=reference,mixed_mode_loss_nats=actual,absolute_error_nats=abs(reference-actual),tolerance_nats=1e-5,completion_tokens=int(nactual.item()),no_optimizer_update=True)
        (data/'preupdate_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        print('PREUPDATE_PARITY '+json.dumps(parity),flush=True)
        assert int(nref.item())==int(nactual.item())==len(example['completion_ids'])
        assert abs(reference-actual)<=1e-5,'eval/mixed-mode loss mismatch before update'
    lora.print_trainable_parameters=report
    lora.run(types.SimpleNamespace(**cfg))
    verify_base()
    receipt.update(status='complete',peak_mlx_memory_bytes=mx.get_peak_memory(),base_bytes_unchanged=True)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    (data/(Path(cfg['adapter_path']).name+'_base_verified.json')).write_text(json.dumps(dict(base_bytes_unchanged=True))+'\n')
if __name__=='__main__':main()
