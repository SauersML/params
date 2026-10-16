"""Explicitly launched wrapper only; preparing this script never starts training."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    assert a.launch,'Training not authorized by default; root must explicitly launch.'
    config=Path(a.config).resolve();cfg=json.loads(config.read_text())
    assert cfg['iters']==2 and cfg['batch_size']==1 and cfg['num_layers']==1
    assert Path(cfg['model']).is_dir() and not Path(cfg['adapter_path']).exists(),'local model/new adapter path required'
    seconds=180 if cfg['iters']==2 else 1200
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',MLX_DISABLE_COMPILE='1')
    with (config.parent/(config.stem+'.log')).open('x') as log:
        try:
            r=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('train_child.py')),'--config',str(config)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=seconds)
            status=dict(returncode=r.returncode,status='complete' if r.returncode==0 else 'failed')
        except subprocess.TimeoutExpired:status=dict(status='wall_budget_exhausted',seconds=seconds)
    (config.parent/(config.stem+'_status.json')).write_text(json.dumps(status,indent=2)+'\n')
    if status['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
