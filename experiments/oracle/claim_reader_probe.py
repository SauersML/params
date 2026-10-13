"""Frozen paired untuned reader probe. prepare is offline; run requires separate approval."""
import argparse,hashlib,importlib.metadata,json,math,os
from pathlib import Path
MODEL='Qwen/Qwen3.6-35B-A3B';PROJECT='3f598f31-2ffa-4eb4-b302-909eb234fb7d';CAP=.75
VIEWS=('context_exact_carrier_fresh_features','context_geometry');PRICES={'prefill':.45,'cached':.09,'sampled':1.335}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def configure(tokenizer):
 from transformers import AutoTokenizer
 from tinker_cookbook import renderers
 t=AutoTokenizer.from_pretrained(str(tokenizer),local_files_only=True)
 return renderers.get_renderer('qwen3_5_disable_thinking',t)
def prepare(data,tokenizer,out,prices):
 if out.exists():raise ValueError('preserve frozen preparation')
 official=next(x for x in json.loads(prices.read_text()) if x['tinker_id']==MODEL)
 if [official[k] for k in ('prefill','cached_prefill','sample')]!=['$0.45','$0.09','$1.335']:raise ValueError('official rates changed')
 rr=configure(tokenizer);prompts=[];bindings=[];source={};geom={};seen=set()
 for view in VIEWS:
  src=data/(view+'_internal_dev_matched_fit.jsonl');source[view]={'path':str(src),'sha256':sha(src)};rows=[json.loads(x) for x in src.read_text().splitlines()]
  if len(rows)!=42:raise ValueError('fixed42cohort')
  for r in rows:
   i=r['info'];k=i['case_id']+'|'+i['condition_private'];identity=(view,k)
   if identity in seen or i['split']!='internal_dev' or i['task'] not in ('text6136','text6012','text6045'):raise ValueError('fixed folds/unique IDs')
   seen.add(identity);text=r['prompt'][0]['content'];ph=hashlib.sha256(text.encode()).hexdigest()
   if view=='context_geometry':
    geom.setdefault(i['case_id'],ph)
    if geom[i['case_id']]!=ph:raise ValueError('geometry triplets not identical')
   prompt=rr.build_generation_prompt(r['prompt']);ids=prompt.to_ints()
   if prompt.length+64+16>65536:raise ValueError('no truncation')
   request_id=view+'|'+(i['case_id'] if view=='context_geometry' else k)
   old=next((p for p in prompts if p['request_id']==request_id),None)
   if old:
    if old['token_ids']!=ids:raise ValueError('dedup token mismatch')
   else:prompts.append({'request_id':request_id,'view':view,'text':text,'prompt_sha256':ph,'token_ids':ids,'prompt_tokens':prompt.length})
   bindings.append({'view':view,'id':k,'case_id':i['case_id'],'task':i['task'],'request_id':request_id,'prompt_sha256':ph})
 if len(prompts)!=56 or len(bindings)!=84:raise ValueError('expected42feature+14geometry unique56')
 total=sum(p['prompt_tokens'] for p in prompts);upper=(total*.45+56*64*1.335)/1e6
 if upper>CAP:raise ValueError('whole probe exceeds cap')
 out.mkdir();pp=out/'requests.jsonl';pp.write_text(''.join(json.dumps(p)+'\n' for p in prompts))
 m={'schema':1,'model':MODEL,'project':PROJECT,'cap_usd':CAP,'cost_upper_usd':upper,'unique_calls':56,'conditions':84,'max_tokens':64,'temperature':0.,'sources':source,'bindings':bindings,'requests_sha256':sha(pp),'runner_sha256':sha(__file__),'tokenizer_path':str(tokenizer),'tokenizer_sha256':{p.name:sha(p) for p in tokenizer.iterdir() if p.is_file()},'packages':{p:importlib.metadata.version(p) for p in ('tinker','tinker-cookbook','transformers','tokenizers')},'rates':PRICES,'official_prices_sha256':sha(prices),'official_price_url':'https://tinker-docs.thinkingmachines.ai/tinker/models.json','prompt_tokens':total,'scope':'Firstgreedy feature42 vsgeometry42, geometry identicaltriplets reusedone exactrequest and boundtoall3. Fixed83triplet-cohort DEV3TRAINtasks, no retries/bestof. No training, fit/labels/features unchanged.'}
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return m
def run(manifest,out):
 m=json.loads(manifest.read_text());base=manifest.parent
 if m.get('rates')!=PRICES or m.get('unique_calls')!=56 or m.get('conditions')!=84 or m.get('max_tokens')!=64 or m.get('temperature')!=0.:raise ValueError('sampling/count/rate contract changed')
 if out.exists() or m['model']!=MODEL or m['project']!=PROJECT or os.environ.get('TINKER_PROJECT_ID')!=PROJECT or not 0<m['cost_upper_usd']<=CAP or m['runner_sha256']!=sha(__file__):raise ValueError('project/source/output/cap mismatch')
 if {p:importlib.metadata.version(p) for p in m['packages']}!=m['packages']:raise ValueError('packages changed')
 tok=Path(m['tokenizer_path'])
 if {p.name:sha(p) for p in tok.iterdir() if p.is_file()}!=m['tokenizer_sha256']:raise ValueError('tokenizer changed')
 if sha(base/'requests.jsonl')!=m['requests_sha256']:raise ValueError('requests changed')
 for binding in m['sources'].values():
  if sha(binding['path'])!=binding['sha256']:raise ValueError('source rows changed')
 requests=[json.loads(x) for x in (base/'requests.jsonl').read_text().splitlines()];rr=configure(tok);rendered=[]
 if len(requests)!=56 or len({p['request_id'] for p in requests})!=56:raise ValueError('unique first56 requests required')
 for p in requests:
  if hashlib.sha256(p['text'].encode()).hexdigest()!=p['prompt_sha256']:raise ValueError('prompt hash changed')
  model_input=rr.build_generation_prompt([{'role':'user','content':p['text']}])
  if model_input.to_ints()!=p['token_ids'] or model_input.length!=p['prompt_tokens']:raise ValueError('exact input token identity changed')
  rendered.append(model_input)
 by_request={p['request_id']:p for p in requests}
 if len(m['bindings'])!=84 or len({(b['view'],b['id']) for b in m['bindings']})!=84:raise ValueError('condition binding changed')
 for b in m['bindings']:
  if b['prompt_sha256']!=by_request[b['request_id']]['prompt_sha256']:raise ValueError('dedup bound prompt mismatch')
 actual_prefill=sum(p.length for p in rendered);actual_upper=(actual_prefill*.45+56*64*1.335)/1e6
 if not math.isfinite(actual_upper) or actual_upper>CAP or actual_prefill!=m['prompt_tokens'] or abs(actual_upper-m['cost_upper_usd'])>1e-12:raise ValueError('whole actual token cost changed')
 # No Service before all offline validation.
 import tinker
 service=tinker.ServiceClient(project_id=PROJECT);sampler=service.create_sampling_client(base_model=MODEL);out.mkdir();(out/'manifest.json').write_bytes(manifest.read_bytes());spend=0.
 def record(name,row):
  with (out/name).open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 params=tinker.SamplingParams(max_tokens=64,temperature=0.,stop=rr.get_stop_sequences())
 for req,prompt in zip(requests,rendered):
  reserve=(prompt.length*.45+64*1.335)/1e6
  if spend+reserve>CAP:raise ValueError('next request exceeds hard cap')
  record('reservations.jsonl',{'request_id':req['request_id'],'upper_usd':reserve,'spend_before':spend})
  result=sampler.sample(prompt=prompt,num_samples=1,sampling_params=params).result()
  if len(result.sequences)!=1:raise ValueError('single first response required')
  seq=result.sequences[0];hits=result.prompt_cache_hit_tokens;row={'request_id':req['request_id'],'token_ids':list(seq.tokens),'finish':str(seq.stop_reason),'prompt_cache_hit_tokens':hits};record('raw_responses.jsonl',row)
  if not 0<=hits<=prompt.length or len(seq.tokens)>64:raise ValueError('provider accounting contract violated')
  usd=((prompt.length-hits)*.45+hits*.09+len(seq.tokens)*1.335)/1e6;spend+=usd;record('spend.jsonl',{'request_id':req['request_id'],'prefill_tokens':prompt.length-hits,'cached_tokens':hits,'sampled_tokens':len(seq.tokens),'usd':usd,'total_usd':spend})
  try:
   from tinker_cookbook.renderers import get_text_content
   msg,_=rr.parse_response(seq.tokens);row['completion']=get_text_content(msg)
  except Exception as e:row.update(completion=None,parse_error=type(e).__name__)
  record('responses.jsonl',row)
 (out/'complete.json').write_text(json.dumps({'requests':56,'spend_usd':spend})+'\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True);prep=sub.add_parser('prepare');paid=sub.add_parser('run')
 for k in ('data','tokenizer','out','prices'):prep.add_argument('--'+k,type=Path,required=True)
 for k in ('manifest','out'):paid.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args()
 if a.command=='prepare':print(json.dumps(prepare(a.data,a.tokenizer,a.out,a.prices),indent=2))
 else:run(a.manifest,a.out)
