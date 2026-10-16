"""Offline exposed-panel fit scoring; no model imports or calls."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def require(ok,msg):
 if not ok:raise ValueError(msg)

def numeric_sensitivity(row):
 """Remove ONLY reason word count from frozen parser, retaining length/error invalids."""
 finish=str(row.get('finish_reason','')).casefold()
 if any(x in finish for x in ('length','max_token')):return None,'length_stop',None
 if finish!='stop':return None,'not_normal_stop',None
 if not isinstance(row.get('completion'),str):return None,'missing_completion',None
 def pairs(p):require(len(dict(p))==len(p),'duplicate JSON keys');return dict(p)
 def constant(x):raise ValueError('nonfinite JSON')
 try:
  obj=json.loads(row['completion'],object_pairs_hook=pairs,parse_constant=constant)
  require(isinstance(obj,dict) and set(obj)=={'delta_bits','reason'},'exact fields')
  v=obj['delta_bits'];reason=obj['reason'];require(type(v) in (int,float) and math.isfinite(float(v)) and isinstance(reason,str),'number/reason')
  return float(v),None,reason
 except (ValueError,TypeError,OverflowError):return None,'invalid_strict_json',None

def validate(predictions,provenance,gold,train,valid):
 bindings=provenance['bindings'];require(len(bindings)==32 and len({b['case_id'] for b in bindings})==32,'32 unique provenance cases required');by={b['case_id']:b for b in bindings}
 require(sum(b['split']=='train' for b in bindings)==24 and sum(b['split']=='valid' for b in bindings)==8,'24TRAIN/8DEV split required')
 require(len(predictions)==32 and len({r['case_id'] for r in predictions})==32 and {r['case_id'] for r in predictions}==set(by),'32 complete unique prediction cases required')
 require(gold['status']=='complete' and len(gold['rows'])==32 and len({r['case_id'] for r in gold['rows']})==32,'complete unique gold required');gb={r['case_id']:r for r in gold['rows']};require(set(gb)==set(by),'gold case coverage')
 for split,data in [('train',train),('valid',valid)]:
  expected={k for k,b in by.items() if b['split']==split};require(len(data)==len(expected) and len({r['case_id'] for r in data})==len(data) and {r['case_id'] for r in data}==expected,'dataset split mixing/coverage')
 for row in predictions:
  b=by[row['case_id']];require(row['split']==b['split'],'prediction split mixing');g=gb[row['case_id']];con=lambda s:(s['a_logp_nats']-s['b_logp_nats'])/math.log(2)
  delta=con(g['effects']['target_zero'])-con(g['conditional_baseline']);require(math.isfinite(delta) and abs(delta-b['delta_bits'])<1e-12 and abs(delta-row['gold_delta_bits'])<1e-12,'raw gold delta binding mismatch');require(b['group']==g['group'],'group binding mismatch')
 return by,gb

def paired_ci(values):
 x=np.array(values,dtype=float);ix=np.random.default_rng(7281).integers(0,len(x),(10000,len(x)));return {'mean':float(x.mean()),'question_bootstrap95CI':np.quantile(x[ix].mean(1),[.025,.975]).tolist(),'questions':len(x),'note':'Descriptive exposed cases, not generalization CI'}

def evaluate(predictions,provenance,gold,train,valid,manifest,original_responses,mapping,strict_parser,same_data_predictions=None):
 by,gb=validate(predictions,provenance,gold,train,valid);require(set(manifest['frozen_query_baselines'])==set(by),'baseline coverage')
 if same_data_predictions is not None:require(set(same_data_predictions)==set(by) and all(type(v) in (int,float) and math.isfinite(v) for v in same_data_predictions.values()),'same-data baseline coverage/finite')
 qids={b['case_id']:b['question_id'] for b in mapping};require(len(qids)==32 and set(qids)==set(by),'mapping coverage');original={r['request_id']:r for r in original_responses};rows=[]
 for p in predictions:
  cid=p['case_id'];b=by[cid];orig=original[qids[cid]+'_true'];require(p['baseline']==orig,'embedded original response changed')
  target=b['delta_bits'];r={'case_id':cid,'split':b['split'],'group':b['group'],'gold_delta_bits':target,'modes':{}}
  for mode,parser in [('strict',strict_parser),('reason_length_only_sensitivity',lambda row,raw:numeric_sensitivity(row))]:
   vals={};errs={};numbers={}
   for name,response in [('adapted',p['adapted']),('original_true',orig)]:
    value,invalid,_=parser(response,{});numbers[name]=value;vals[name]=60. if invalid else min(abs(value-target),60.);errs[name]=invalid
    if mode=='strict':
     prefix='adapted' if name=='adapted' else 'baseline';require(p[prefix+'_invalid']==invalid and abs(p[prefix+'_error']-vals[name])<1e-12,'stored parser/error disagreement')
   baselines={'unchanged_zero':0.,'static':manifest['frozen_query_baselines'][cid]['static_readout_delta_bits'],'ridge':manifest['frozen_query_baselines'][cid]['regression_delta_bits']}
   if same_data_predictions is not None:baselines['same_data_ridge']=same_data_predictions[cid]
   for name,value in baselines.items():require(math.isfinite(value),'baseline finite');numbers[name]=value;vals[name]=min(abs(value-target),60.)
   r['modes'][mode]={'losses':vals,'invalids':errs,'numbers':numbers}
  rows.append(r)
 def aggregate(subset,mode):
  names=list(subset[0]['modes'][mode]['losses']);result={}
  for name in names:
   losses=[r['modes'][mode]['losses'][name] for r in subset];good=[r for r in subset if r['modes'][mode]['numbers'][name] is not None and not r['modes'][mode]['invalids'].get(name)]
   non_tied=[r for r in good if r['gold_delta_bits']!=0 and r['modes'][mode]['numbers'][name]!=0]
   result[name]={'rows':len(subset),'capped_mae':float(np.mean(losses)),'invalids':sum(r['modes'][mode]['invalids'].get(name) is not None for r in subset),'valid_numeric_rows':len(good),'bias_on_valid':float(np.mean([r['modes'][mode]['numbers'][name]-r['gold_delta_bits'] for r in good])) if good else None,'sign':{'non_tied_rows':len(non_tied),'agree':sum(r['modes'][mode]['numbers'][name]*r['gold_delta_bits']>0 for r in non_tied),'gold_ties':sum(r['gold_delta_bits']==0 for r in good),'prediction_ties':sum(r['modes'][mode]['numbers'][name]==0 for r in good)}}
  return {'predictors':result,'comparator_minus_adapted_error':{name:paired_ci([r['modes'][mode]['losses'][name]-r['modes'][mode]['losses']['adapted'] for r in subset]) for name in names if name!='adapted'}}
 reports={}
 for split in ('train','valid'):
  subset=[r for r in rows if r['split']==split];reports[split]={'scope':'24TRAIN resubstitution' if split=='train' else '8DEV already exposed; not pristine','modes':{mode:aggregate(subset,mode) for mode in rows[0]['modes']},'groups':{group:{mode:aggregate([r for r in subset if r['group']==group],mode) for mode in rows[0]['modes']} for group in sorted({r['group'] for r in subset})}}
 return {'schema':'exposed_native_reader_training_fit_score_v1','scope':'Training-fit diagnostic only; no unseen generalization, semantic faithfulness or causal use of inputs established. Strict score retained; wordcount-only sensitivity explicitly secondary. No outcome filtering.','splits':reports,'rows':rows}

def main():
 ap=argparse.ArgumentParser()
 for k in ('data','predictions','frozen','gold','original_score','original_diagnostics','primary_source','original_responses','out'):ap.add_argument('--'+k.replace('_','-'),type=Path,required=True)
 ap.add_argument('--same-data-ridge',type=Path);ap.add_argument('--same-data-ridge-sha256')
 a=ap.parse_args();meta=load(a.data/'provenance.json');require(sha(a.gold)==meta['gold_sha256'],'gold SHA mismatch');require(sha(a.frozen/'manifest.json')==meta['frozen_reader_manifest_sha256'],'frozen manifest SHA mismatch');m=load(a.frozen/'manifest.json');require(sha(m['public_observations_path'])==meta['public_sha256']==m['public_observations_sha256'],'public SHA mismatch')
 for split,digest in meta['dataset_sha256'].items():require(sha(a.data/(split+'.jsonl'))==digest,'dataset SHA mismatch')
 require(sha(a.primary_source)=='e0b6b4580e2bcd4fe67b6fef06e509696c9adaee1bfee9ec42fb8e0b80ec9650','strict frozen parser source mismatch');spec=importlib.util.spec_from_file_location('frozen_primary',a.primary_source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 require(sha(a.original_responses)=='65997e8da14092ff6822e5ad64c9de8f247a7e5721d275d161f37e86d9b9b509','original responses SHA mismatch')
 for path in (a.original_score,a.original_diagnostics):require(load(path)['gold_sha256']==meta['gold_sha256'],'original report gold mismatch')
 same_predictions=None
 if a.same_data_ridge is not None:
  require(isinstance(a.same_data_ridge_sha256,str) and sha(a.same_data_ridge)==a.same_data_ridge_sha256,'same-data ridge frozen hash mismatch');artifact=load(a.same_data_ridge)
  require(artifact['schema']=='same_supervision_native_ridge_v1' and artifact['provenance_sha256']==sha(a.data/'provenance.json'),'same-data ridge provenance mismatch')
  for key in ('dataset_sha256','public_sha256','gold_sha256','frozen_reader_manifest_sha256'):require(artifact[key]==meta[key],'same-data ridge '+key)
  predictions=artifact['predictions'];require(len(predictions)==32 and len({r['case_id'] for r in predictions})==32,'same-data prediction coverage');bindings={b['case_id']:b for b in meta['bindings']}
  model=artifact['model'];public=load(m['public_observations_path']);panel=load(m['panel_path']);require(model['target_identity']==panel['target'] and model['feature_units_layer_neuron']==[[l,n] for l,_,n in panel['target_units']],'same-data feature identity');pub={r['case_id']:r for r in public['rows']};same_predictions={}
  require(all(len(model[k])==16 for k in ('mean','scale','coefficient')) and all(type(v) in (int,float) and math.isfinite(v) for v in [model['intercept'],*model['mean'],*model['scale'],*model['coefficient']]) and all(v>0 for v in model['scale']),'same-data model dimensions/numbers')
  for row in predictions:
   cid=row['case_id'];require(cid in bindings and row['split']==bindings[cid]['split'] and row['group']==bindings[cid]['group'],'same-data split binding');r=pub[cid]
   require([[n['unit'][0],n['unit'][2]] for n in r['neurons']]==model['feature_units_layer_neuron'],'same-data feature order')
   value=model['intercept']+sum(c*(n['activation']-mean)/scale for n,mean,scale,c in zip(r['neurons'],model['mean'],model['scale'],model['coefficient']))
   require(math.isfinite(value) and value==row['prediction_delta_bits'],'same-data prediction reproduction');same_predictions[cid]=value
 else:require(a.same_data_ridge_sha256 is None,'hash without same-data artifact')
 result=evaluate(lines(a.predictions),meta,load(a.gold),lines(a.data/'train.jsonl'),lines(a.data/'valid.jsonl'),m,lines(a.original_responses),load(a.frozen/'mapping.json'),module.parse,same_data_predictions=same_predictions)
 # Independently bind recomputed original losses to BOTH published report modes.
 for path,mode in [(a.original_score,'strict'),(a.original_diagnostics,'reason_length_only_sensitivity')]:
  prior={r['case_id']:r for r in load(path)['predictions'] if r['arm']=='true'}
  require(len(prior)==32,'original true report coverage')
  for row in result['rows']:require(row['modes'][mode]['losses']['original_true']==prior[row['case_id']]['capped_absolute_error'],'original true report arithmetic mismatch')
 requests={r['request_id']:r for r in lines(a.frozen/'requests.jsonl')};require(sha(a.frozen/'requests.jsonl')==m['requests_sha256'],'requests SHA mismatch')
 mapping={r['case_id']:r['question_id'] for r in load(a.frozen/'mapping.json')}
 for split in ('train','valid'):
  for row in lines(a.data/(split+'.jsonl')):require(row['prompt_ids']==requests[mapping[row['case_id']]+'_true']['token_ids'],'fit prompt differs from frozen original true prompt')
 result['input_sha256']={str(p):sha(p) for p in [a.predictions,a.data/'provenance.json',a.data/'train.jsonl',a.data/'valid.jsonl',a.frozen/'manifest.json',a.gold,a.original_score,a.original_diagnostics,a.primary_source,a.original_responses]};
 if a.same_data_ridge is not None:result['input_sha256'][str(a.same_data_ridge)]=a.same_data_ridge_sha256
 a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
