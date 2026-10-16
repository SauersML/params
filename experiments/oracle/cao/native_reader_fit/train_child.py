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
    receipt=dict(config_sha256=sha(a.config),runtime={k:importlib.metadata.version(k) for k in ('mlx','mlx-lm','numpy')},source_sha256={q.name:sha(q) for q in Path(__file__).parent.glob('*.py')},dataset_sha256=provenance['dataset_sha256'],base_sha256=provenance['immutable_base_files_sha256'],seed=cfg['seed'],status='started')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    def datasets(args,tokenizer):
        load=lambda name:ExactDataset([json.loads(s) for s in (data/(name+'.jsonl')).read_text().splitlines()])
        return load('train'),load('valid'),None
    lora.load_dataset=datasets
    from exact_loss import exact_completion_loss
    lora.train=functools.partial(lora.train,loss=exact_completion_loss)
    # Existing function remains responsible for freezing, adapter conversion,
    # optimization, checkpoints, training masks and validation.
    original=lora.print_trainable_parameters
    def report(model):
        original(model)
        params=tree_flatten(model.trainable_parameters())
        (data/(Path(cfg['adapter_path']).name+'_parameters.json')).write_text(json.dumps(dict(count=sum(v.size for _,v in params),names=[k for k,_ in params]),indent=2)+'\n')
        assert params,'No adapters matched declared projection keys'
        assert all(k.endswith(('.lora_a','.lora_b')) for k,_ in params),'Non-LoRA parameter trainable'
        assert all(any('.'+key+'.' in k for key in cfg['lora_parameters']['keys']) for k,_ in params),'Unexpected projection trainable'
    lora.print_trainable_parameters=report
    lora.run(types.SimpleNamespace(**cfg))
    verify_base()
    receipt.update(status='complete',peak_mlx_memory_bytes=mx.get_peak_memory(),base_bytes_unchanged=True)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    (data/(Path(cfg['adapter_path']).name+'_base_verified.json')).write_text(json.dumps(dict(base_bytes_unchanged=True))+'\n')
if __name__=='__main__':main()
