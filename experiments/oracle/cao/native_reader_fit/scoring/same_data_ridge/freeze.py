"""Same-supervision24TRAIN ridge;8 exposedDEV never selects hyperparameters."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from ridge import features,predict
HERE=Path(__file__).parent
DATA=Path('/Users/user/mpd-data/scratch/native_next_experiment/native_reader_fit_v1/last_attention_fit_v2')
FROZEN=Path('/Users/user/mpd-data/scratch/root/native_query_reader_local_v1/frozen')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def fit(x,y,alpha):
 mean=x.mean(0);scale=x.std(0);scale=np.where(scale==0,1.,scale);z=(x-mean)/scale;intercept=y.mean();coef=np.linalg.solve(z.T@z+alpha*np.eye(16),z.T@(y-intercept))
 return {'mean':mean.tolist(),'scale':scale.tolist(),'coefficient':coef.tolist(),'intercept':float(intercept)}
def main():
 meta=load(DATA/'provenance.json');m=load(FROZEN/'manifest.json');assert sha(FROZEN/'manifest.json')==meta['frozen_reader_manifest_sha256'];assert sha(m['public_observations_path'])==meta['public_sha256'];assert all(sha(DATA/(split+'.jsonl'))==h for split,h in meta['dataset_sha256'].items())
 public=load(m['public_observations_path']);panel=load(m['panel_path']);gp=Path('/Users/user/mpd-data/cluster/root-lie-query-gold-cpu-v1/output/result/gold/result.json');assert sha(gp)==meta['gold_sha256'];gold=load(gp);gby={r['case_id']:r for r in gold['rows']};pby={r['case_id']:r for r in public['rows']};assert len(pby)==32 and len(gby)==32
 bindings=meta['bindings'];assert len(bindings)==32 and len({b['case_id'] for b in bindings})==32;ordered=[pby[b['case_id']] for b in bindings];model={'schema':'native_activation_ridge_v1','feature_units_layer_neuron':[[l,n] for l,_,n in panel['target_units']]};x=np.array(features(model,ordered));ys=[]
 for b in bindings:
  r=gby[b['case_id']];con=lambda v:(v['a_logp_nats']-v['b_logp_nats'])/math.log(2);y=con(r['effects']['target_zero'])-con(r['conditional_baseline']);assert abs(y-b['delta_bits'])<1e-12;ys.append(y)
 y=np.array(ys);ix=np.array([i for i,b in enumerate(bindings) if b['split']=='train']);dev=np.array([i for i,b in enumerate(bindings) if b['split']=='valid']);assert len(ix)==24 and len(dev)==8;selection=[]
 for alpha in [.01,.1,1.,10.,100.]:
  pred=[]
  for held in ix:
   train=ix[ix!=held];fold=dict(model,**fit(x[train],y[train],alpha));pred.append(predict(fold,[ordered[held]])[0])
  err=np.abs(np.array(pred)-y[ix]);selection.append({'alpha':alpha,'train_loo_mae_bits':float(err.mean()),'case_ids':[bindings[i]['case_id'] for i in ix],'predictions_delta_bits':pred,'absolute_errors_bits':err.tolist()})
 best=min(selection,key=lambda r:r['train_loo_mae_bits']);model.update(fit(x[ix],y[ix],best['alpha']));pred=predict(model,ordered);assert all(math.isfinite(v) for v in pred)
 summary={}
 for split,index in [('train_resubstitution',ix),('exposed_dev',dev)]:
  summary[split]={'rows':len(index),'ridge_mae_bits':float(np.mean(np.abs(np.array(pred)[index]-y[index]))),'zero_mae_bits':float(np.mean(np.abs(y[index]))),'static_mae_bits':float(np.mean([abs(ordered[i]['static_zero_delta_bits']-y[i]) for i in index]))}
 out={'schema':'same_supervision_native_ridge_v1','scope':'Same24TRAIN supervision as adapter;8DEV already exposed. Not pristine evaluation. No adapter outputs used.','model':dict(model,alpha=best['alpha'],target_identity=panel['target']),'alpha_grid':[.01,.1,1.,10.,100.],'selection_rule':'Minimum TRAIN-only leave-one-out MAE, grid-order ties; population mean/std refit separately inside each23-row fold; zero std→1; intercept unpenalized','selection':selection,'selected_train_loo_mae_bits':best['train_loo_mae_bits'],'summary':summary,'predictions':[{'case_id':b['case_id'],'split':b['split'],'group':b['group'],'prediction_delta_bits':v} for b,v in zip(bindings,pred)],'provenance_sha256':sha(DATA/'provenance.json'),'dataset_sha256':meta['dataset_sha256'],'public_sha256':meta['public_sha256'],'gold_sha256':meta['gold_sha256'],'frozen_reader_manifest_sha256':meta['frozen_reader_manifest_sha256'],'source_sha256':{'freeze.py':sha(__file__),'ridge.py':sha(HERE/'ridge.py')}}
 dest=HERE/'frozen.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest),'alpha':best['alpha'],'train_loo_mae_bits':best['train_loo_mae_bits'],'summary':summary}))
if __name__=='__main__':main()
