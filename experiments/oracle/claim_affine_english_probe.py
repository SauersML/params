"""Bounded affine writer→reader calibration. Preparation is offline; run needs GO.

Three independent fresh writer requests, not three independent model families.
No rewritten prose, retries, training, or global mechanism certification.
"""
import argparse,hashlib,importlib.metadata,json,math,os
from pathlib import Path
import claim_reader_probe as guarded
MODEL=guarded.MODEL;READER='moonshotai/Kimi-K2.6';PROJECT=guarded.PROJECT
RATES={MODEL:guarded.PRICES,READER:{'prefill':2.205,'cached':.441,'sampled':5.49}}
REVISION='b5aabbfb20227ed42becbf5541dbffd213942c58'
WRITER_CAP=512;READER_CAP=96;PROSE_BYTES=8192;CAP=.50
CONDITIONS=('typed','prose','none')
CONTEXT={MODEL:65536,READER:32768}
def sha(p):return guarded.sha(p)
def reader_renderer():
 from tinker_cookbook import renderers
 from tinker_cookbook.tokenizer_utils import get_tokenizer,_KIMI_TOKENIZER_REVISIONS
 if _KIMI_TOKENIZER_REVISIONS[READER]!=REVISION or os.environ.get('HF_HUB_OFFLINE')!='1':raise ValueError('pinned cached offline reader tokenizer required')
 return renderers.get_renderer('kimi_k26_disable_thinking',get_tokenizer(READER))
def cost(model,tokens,limit):return (tokens*RATES[model]['prefill']+limit*RATES[model]['sampled'])/1e6
def encoded(rr,text):
 p=rr.build_generation_prompt([{'role':'user','content':text}]);return {'text':text,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'token_ids':p.to_ints(),'prompt_tokens':p.length}
def stopped(row):return not row.get('parse_error') and str(row.get('finish','')).lower() in ('stop','stopreason.stop') and isinstance(row.get('completion'),str) and bool(row['completion'].strip())
def prediction(row):
 if row is None:return None,'missing'
 if row.get('upstream_invalid'):return None,'writer_invalid'
 if not stopped(row):return None,'unfinished_or_parse_error'
 try:
  value=json.loads(row['completion'])
  if set(value)!={'top_token','changed'} or not isinstance(value['top_token'],str) or type(value['changed']) is not bool:raise ValueError()
  return value,None
 except (ValueError,TypeError):return None,'malformed'
def reader_text(common,condition,evidence):
 return common+'\nExplanation condition: '+condition+'\nBEGIN EVIDENCE\n'+evidence+'\nEND EVIDENCE\nReturn ONLY JSON {"top_token":EXACT native token string,"changed":boolean}. Changed compares the argmax token against the SAME unedited model/context at k=1. Preserve leading/trailing whitespace in the string. No reasoning text.'
def typed_facts(case,payload):
 return {'case':case,'mechanism':{**payload['mechanism'],'segment_coordinates':'Each envelope segment line is numerator(k)=intercept+k*slope; intercept is the k=0 value, not the unedited base at k=1.'},'model_content_sha256':payload['model_content_sha256']}
def prepare(evidence,tokenizer,prices,out):
 if out.exists():raise ValueError('immutable preparation')
 official=next(r for r in json.loads(prices.read_text()) if r['tinker_id']==MODEL)
 if [official[k] for k in ('prefill','cached_prefill','sample')]!=['$0.45','$0.09','$1.335']:raise ValueError('official rates changed')
 reader_price=next(r for r in json.loads(prices.read_text()) if r['tinker_id']==READER)
 if [reader_price[k] for k in ('prefill','cached_prefill','sample')]!=['$2.205','$0.441','$5.49']:raise ValueError('reader official rates changed')
 payload=json.loads((evidence/'writer_payload.json').read_text());protocol=json.loads((evidence/'protocol.json').read_text());physical=[json.loads(x) for x in (evidence/'verification.jsonl').read_text().splitlines()]
 if sha(evidence/'writer_payload.json')!=protocol['writer_payload_frozen_before_first_edited_forward'] or len(physical)!=18:raise ValueError('frozen physical input binding')
 expected={(t,k) for t in protocol['selected'] for k in protocol['scales']}
 if len({(r['task'],r['k']) for r in physical})!=18 or {(r['task'],r['k']) for r in physical}!=expected or any(r['unit']!=protocol['selected'][r['task']] or not r['ideal_literal_top_matches'] or r['editor_top']!=r['literal_top'] for r in physical):raise ValueError('all18 verified original cases required')
 if protocol['selected']!={'text6008':3268,'text6011':2412,'text6034':1833} or protocol['scales']!=[-1.,0.,.5,1.,1.5,2.]:raise ValueError('fixed TRAIN units/scales')
 mapping={str(i):v for case in payload['cases'].values() for s in case['envelope']['segments'] for i,v in s['token_strings'].items()}
 if len(mapping)!=5:raise ValueError('fixed shared five-token mapping')
 rr=guarded.configure(tokenizer);reader_rr=reader_renderer();writers=[];readers=[];templates=[];gold=[]
 for task,case in payload['cases'].items():
  facts=typed_facts(case,payload)
  text='Write a short self-contained English explanation (at most250words) of this exact local persistent weight-edit mechanism and its argmax behavior over the entire declared scale range. Explain weight operation, unchanged site input, residual addition, positive shared final RMS and readout, numerical/boundary limitations and conditional scope. Preserve the winning token spellings/whitespace and quantitative intervals. Numeric boundaries/formulas are allowed; do not merely output JSON or a table. Do not claim full circuit fidelity, exact probabilities, or unseen generalization. These facts were derived from the unedited weights/activations; no edited-forward observations are supplied.\nTYPED FACTS\n'+json.dumps(facts,sort_keys=True)
  writers.append({'request_id':'writer|'+task,'task':task,'model':MODEL,'max_tokens':WRITER_CAP,**encoded(rr,text)})
  for k in protocol['scales']:
   ident=task+'|'+str(k);common='Known fixed native model vpd4l. Predict the LAST token-position argmax after a persistent all-position edit of '+case['site']+' component '+str(case['component_index'])+' by factor k='+str(k)+'. The operation is W(k)=W+(k-1)outer(U_c,V_c), with all other weights unchanged.\nORIGINAL CONTEXT (native full decode):\n'+case['execution_text']+'\nCOMMON NATIVE TOKEN MAPPING (same in all cases/conditions): '+json.dumps(mapping,sort_keys=True)
   for condition in ('typed','none'):
    text=reader_text(common,condition,json.dumps(facts,sort_keys=True) if condition=='typed' else '')
    readers.append({'request_id':condition+'|'+ident,'task':task,'k':k,'model':READER,'condition':condition,'max_tokens':READER_CAP,**encoded(reader_rr,text)})
   # Future prose is bounded before Service and actual rendered IDs frozen before its call.
   bound=len(reader_text(common,'prose','').encode())+PROSE_BYTES+2048
   if bound+READER_CAP+16>CONTEXT[READER]:raise ValueError('future prose context bound')
   templates.append({'request_id':'prose|'+ident,'task':task,'k':k,'model':READER,'condition':'prose','common':common,'input_token_upper':bound,'max_tokens':READER_CAP})
   actual=next(r for r in physical if r['task']==task and r['k']==k);baseline=next(r for r in physical if r['task']==task and r['k']==1.)
   gold.append({'id':ident,'task':task,'k':k,'top_token':mapping[str(actual['literal_top'])],'changed':actual['literal_top']!=baseline['literal_top']})
 static=writers+readers
 if len(writers)!=3 or len(readers)!=36 or len(templates)!=18:raise ValueError('57 request contract')
 if any(r['prompt_tokens']+r['max_tokens']+16>CONTEXT[r['model']] for r in static):raise ValueError('no truncation')
 upper=sum(cost(r['model'],r['prompt_tokens'],r['max_tokens']) for r in static)+sum(cost(READER,t['input_token_upper'],READER_CAP) for t in templates)
 static_upper=sum(cost(r['model'],r['prompt_tokens'],r['max_tokens']) for r in static)
 if not math.isfinite(upper) or static_upper>CAP:raise ValueError('static stages exceed cap')
 out.mkdir();files={'writers.jsonl':writers,'static_readers.jsonl':readers,'prose_templates.jsonl':templates,'private_gold.jsonl':gold}
 for name,rows in files.items():(out/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 reader_path=Path.home()/'.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots'/REVISION
 m={'schema':1,'model':MODEL,'reader_model':READER,'reader_renderer':'kimi_k26_disable_thinking','reader_revision':REVISION,'reader_tokenizer_path':str(reader_path),'reader_tokenizer_sha256':{p.name:sha(p) for p in reader_path.iterdir() if p.is_file()},'project':PROJECT,'cap_usd':CAP,'cost_upper_usd':upper,'static_upper_usd':static_upper,'postwriter_exact_reader_admission_required':True,'writer_calls':3,'reader_calls_max':54,'conditions':CONDITIONS,'writer_cap':WRITER_CAP,'reader_cap':READER_CAP,'prose_max_bytes':PROSE_BYTES,'rates':RATES,'temperature':0.,'files':{n:sha(out/n) for n in files},'sources':{n:{'path':str(evidence/n),'sha256':sha(evidence/n)} for n in ('writer_payload.json','protocol.json','verification.jsonl')},'runner_sha256':sha(__file__),'guarded_sampler_sha256':sha(guarded.__file__),'tokenizer_path':str(tokenizer),'tokenizer_sha256':{p.name:sha(p) for p in tokenizer.iterdir() if p.is_file()},'packages':{p:importlib.metadata.version(p) for p in ('tinker','tinker-cookbook','transformers','tokenizers')},'official_prices_sha256':sha(prices),'official_price_url':'https://tinker-docs.thinkingmachines.ai/tinker/models.json','scope':'Three adaptive TRAIN local edits; ideal affine facts plus18physical points. Qwen writer, independent Kimi reader; first response only. Closed five-string support shared across conditions. Untrusted unrepaired prose; no fullmechanism/causalEnglish certification.'}
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return m
def preflight(manifest):
 m=json.loads(manifest.read_text());base=manifest.parent
 if m['model']!=MODEL or m['reader_model']!=READER or m['reader_revision']!=REVISION or m['reader_renderer']!='kimi_k26_disable_thinking' or m['project']!=PROJECT or m['rates']!=RATES or m['writer_cap']!=WRITER_CAP or m['reader_cap']!=READER_CAP or m['prose_max_bytes']!=PROSE_BYTES or m['writer_calls']!=3 or m['reader_calls_max']!=54 or m['temperature']!=0 or m['runner_sha256']!=sha(__file__) or m['guarded_sampler_sha256']!=sha(guarded.__file__):raise ValueError('protocol/source changed')
 if {p:importlib.metadata.version(p) for p in m['packages']}!=m['packages']:raise ValueError('packages changed')
 tok=Path(m['tokenizer_path'])
 if {p.name:sha(p) for p in tok.iterdir() if p.is_file()}!=m['tokenizer_sha256'] or any(sha(base/n)!=h for n,h in m['files'].items()) or any(sha(x['path'])!=x['sha256'] for x in m['sources'].values()):raise ValueError('frozen inputs changed')
 load=lambda n:[json.loads(x) for x in (base/n).read_text().splitlines()]
 rt=Path(m['reader_tokenizer_path'])
 if {p.name:sha(p) for p in rt.iterdir() if p.is_file()}!=m['reader_tokenizer_sha256']:raise ValueError('reader tokenizer changed')
 writers=load('writers.jsonl');readers=load('static_readers.jsonl');templates=load('prose_templates.jsonl');rr={MODEL:guarded.configure(tok),READER:reader_renderer()}
 if len(writers)!=3 or len(readers)!=36 or len(templates)!=18 or len({r['request_id'] for r in writers+readers+templates})!=57:raise ValueError('request coverage')
 expected_ids={c+'|'+task+'|'+str(k) for c in CONDITIONS for task in ('text6008','text6011','text6034') for k in (-1.,0.,.5,1.,1.5,2.)}|{'writer|'+task for task in ('text6008','text6011','text6034')}
 if {r['request_id'] for r in writers+readers+templates}!=expected_ids or any(r['model']!=MODEL or r['max_tokens']!=WRITER_CAP for r in writers) or any(r['model']!=READER or r['max_tokens']!=READER_CAP for r in readers+templates):raise ValueError('model/cap/full-panel identity')
 for r in writers+readers:
  actual=encoded(rr[r['model']],r['text'])
  if any(actual[k]!=r[k] for k in actual) or actual['prompt_tokens']+r['max_tokens']+16>CONTEXT[r['model']]:raise ValueError('exact token/prompt/context binding')
 upper=sum(cost(r['model'],r['prompt_tokens'],r['max_tokens']) for r in writers+readers)+sum(cost(READER,t['input_token_upper'],READER_CAP) for t in templates)
 static_upper=sum(cost(r['model'],r['prompt_tokens'],r['max_tokens']) for r in writers+readers)
 if upper!=m['cost_upper_usd'] or static_upper!=m['static_upper_usd'] or not 0<static_upper<=CAP or m['cap_usd']!=CAP:raise ValueError('actual upper changed')
 return m,rr,writers,readers,templates
def run(manifest,out):
 m,rr,writers,readers,templates=preflight(manifest)
 if out.exists() or os.environ.get('TINKER_PROJECT_ID')!=PROJECT or os.environ.get('AFFINE_ENGLISH_GO_MANIFEST')!=sha(manifest):raise ValueError('fresh output/exact project/GO manifest required')
 import tinker
 service=tinker.ServiceClient(project_id=PROJECT);samplers={MODEL:service.create_sampling_client(base_model=MODEL)};out.mkdir();(out/'manifest.json').write_bytes(manifest.read_bytes());spend=0.;writer_results={}
 def record(name,row):
  with (out/name).open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 def sample(req):
  nonlocal spend
  model=req['model'];renderer=rr[model];rates=RATES[model];prompt=renderer.build_generation_prompt([{'role':'user','content':req['text']}]);bound=cost(model,prompt.length,req['max_tokens'])
  if prompt.to_ints()!=req['token_ids'] or prompt.length!=req['prompt_tokens'] or prompt.length+req['max_tokens']+16>CONTEXT[model] or spend+bound>CAP:raise ValueError('actual input/context/cost bound')
  record('reservations.jsonl',{'request_id':req['request_id'],'model':model,'upper_usd':bound,'spend_before':spend})
  try:result=samplers[model].sample(prompt=prompt,num_samples=1,sampling_params=tinker.SamplingParams(max_tokens=req['max_tokens'],temperature=0.,stop=renderer.get_stop_sequences())).result()
  except Exception as e:
   record('unresolved.jsonl',{'request_id':req['request_id'],'model':model,'reservation_usd':bound,'error':type(e).__name__});raise
  if len(result.sequences)!=1:
   record('unresolved.jsonl',{'request_id':req['request_id'],'model':model,'reservation_usd':bound,'error':'single_response_contract'});raise ValueError('single response required')
  seq=result.sequences[0];hits=result.prompt_cache_hit_tokens;row={'request_id':req['request_id'],'model':model,'token_ids':list(seq.tokens),'finish':str(seq.stop_reason),'prompt_cache_hit_tokens':hits};record('raw_responses.jsonl',row)
  if not 0<=hits<=prompt.length or len(seq.tokens)>req['max_tokens']:
   record('unresolved.jsonl',{'request_id':req['request_id'],'model':model,'reservation_usd':bound,'error':'provider_accounting_contract'});raise ValueError('provider accounting violated')
  usd=((prompt.length-hits)*rates['prefill']+hits*rates['cached']+len(seq.tokens)*rates['sampled'])/1e6;spend+=usd;record('spend.jsonl',{'request_id':req['request_id'],'model':model,'prefill_tokens':prompt.length-hits,'cached_tokens':hits,'sampled_tokens':len(seq.tokens),'usd':usd,'total_usd':spend})
  try:
   from tinker_cookbook.renderers import get_text_content
   msg,_=renderer.parse_response(seq.tokens);row['completion']=get_text_content(msg)
  except Exception as e:row.update(completion=None,parse_error=type(e).__name__)
  record('responses.jsonl',row);return row
 for req in writers:writer_results[req['task']]=sample(req)
 for t in templates:
  w=writer_results[t['task']]
  if not stopped(w) or len(w['completion'].encode())>PROSE_BYTES:
   record('responses.jsonl',{'request_id':t['request_id'],'model':READER,'upstream_invalid':True,'completion':None,'finish':None});continue
  text=reader_text(t['common'],'prose',w['completion']);req={**t,**encoded(rr[READER],text)}
  if req['prompt_tokens']>t['input_token_upper']:raise ValueError('future prose token bound violated; no call')
  record('dynamic_reader_requests.jsonl',req);readers.append(req)
 reader_upper=sum(cost(READER,r['prompt_tokens'],READER_CAP) for r in readers)
 record('reader_admission.jsonl',{'actual_reader_calls':len(readers),'actual_reader_upper_usd':reader_upper,'writer_spend_usd':spend,'combined_upper_usd':spend+reader_upper,'cap_usd':CAP,'admitted':spend+reader_upper<=CAP})
 if spend+reader_upper>CAP:raise ValueError('reader stage not admitted; writers retained, no retry')
 samplers[READER]=service.create_sampling_client(base_model=READER)
 for req in readers:sample(req)
 (out/'complete.json').write_text(json.dumps({'spend_usd':spend,'writer_requests':3,'reader_requests':len(readers),'invalid_upstream_rows':54-len(readers)})+'\n')
def outcome_consistency(gold_rows,predictions):
 """Posthoc paired-outcome diagnostic; never repairs the raw changed flag.

 A correct derived change alone is insufficient: two wrong tokens can agree.
 both_outcomes_correct also requires the SAME task's k=1 prediction to be right.
 Missing/invalid baselines stay in the denominator and cannot earn pair credit.
 """
 gold={};coordinates=set();baselines={}
 for g in gold_rows:
  if g['id'] in gold or (g['task'],g['k']) in coordinates:raise ValueError('duplicate gold identity/coordinate')
  if type(g['k']) not in (int,float) or not math.isfinite(g['k']) or not isinstance(g['top_token'],str) or type(g['changed']) is not bool:raise ValueError('invalid gold outcome')
  gold[g['id']]=g;coordinates.add((g['task'],g['k']))
  if g['k']==1:baselines[g['task']]=g
 if not gold or set(baselines)!={g['task'] for g in gold.values()}:raise ValueError('one k=1 baseline required per task')
 if set(predictions)-set(gold):raise ValueError('unexpected prediction identity')
 for p in predictions.values():
  if p is not None and (set(p)!={'top_token','changed'} or not isinstance(p['top_token'],str) or type(p['changed']) is not bool):raise ValueError('invalid parsed prediction')
 checks=[]
 for i,g in gold.items():
  baseline=baselines[g['task']]
  if g['changed']!=(g['top_token']!=baseline['top_token']):raise ValueError('gold changed flag disagrees with outcomes')
  p=predictions.get(i);b=predictions.get(baseline['id']);comparable=p is not None and b is not None
  derived=p['top_token']!=b['top_token'] if comparable else None
  checks.append({'id':i,'task':g['task'],'baseline_id':baseline['id'],'comparable':comparable,
   'derived_changed':derived,'derived_changed_correct':comparable and derived==g['changed'],
   'both_outcomes_correct':comparable and p['top_token']==g['top_token'] and b['top_token']==baseline['top_token'],
   'raw_flag_self_contradiction':comparable and p['changed']!=derived})
 return {'rows':len(checks),**{k:sum(r[k] for r in checks) for k in ('comparable','derived_changed_correct','both_outcomes_correct','raw_flag_self_contradiction')},
  'scope':'Posthoc paired predictions, not repaired primary scores. Correct change alone can arise from two wrong outcomes; require both outcomes for pair credit.', 'scored':checks}
def evaluate(manifest,paid,out,*,derive_changes=False):
 base=manifest.parent;m=json.loads(manifest.read_text())
 if sha(base/'private_gold.jsonl')!=m['files']['private_gold.jsonl']:raise ValueError('gold changed')
 gold_rows=[json.loads(x) for x in (base/'private_gold.jsonl').read_text().splitlines()]
 gold={r['id']:r for r in gold_rows};allowed={c+'|'+i for c in CONDITIONS for i in gold}|{'writer|'+t for t in {r['task'] for r in gold.values()}};responses={}
 for line in (paid/'responses.jsonl').read_text().splitlines():
  r=json.loads(line);i=r['request_id']
  if i not in allowed or i in responses:raise ValueError('unexpected/duplicate response')
  responses[i]=r
 report={}
 for c in CONDITIONS:
  scored=[];predictions={}
  for i,g in gold.items():
   writer=responses.get('writer|'+g['task'])
   if c=='prose' and (writer is None or not stopped(writer) or len(writer['completion'].encode())>PROSE_BYTES):p,reason=None,'writer_invalid'
   else:p,reason=prediction(responses.get(c+'|'+i))
   predictions[i]=p
   scored.append({'id':i,'task':g['task'],'reason':reason,'top_correct':p is not None and p['top_token']==g['top_token'],'changed_correct':p is not None and p['changed']==g['changed'],'both_correct':p=={'top_token':g['top_token'],'changed':g['changed']}})
  report[c]={'rows':18,'top_correct':sum(r['top_correct'] for r in scored),'changed_correct':sum(r['changed_correct'] for r in scored),'both_correct':sum(r['both_correct'] for r in scored),'all6_joint_cases':sum(all(r['both_correct'] for r in scored if r['task']==t) for t in {g['task'] for g in gold.values()}),'scored':scored}
  if derive_changes:report[c]['outcome_consistency']=outcome_consistency(gold_rows,predictions)
 report['scope']=m['scope'];out.write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
 for cmd,keys in [('prepare',('evidence','tokenizer','prices','out')),('run',('manifest','out')),('evaluate',('manifest','paid','out'))]:
  p=sub.add_parser(cmd)
  for k in keys:p.add_argument('--'+k,type=Path,required=True)
  if cmd=='evaluate':p.add_argument('--derive-changes',action='store_true',help='Add separately labeled paired-outcome diagnostics; raw scores stay unchanged')
 a=ap.parse_args()
 if a.command=='prepare':print(json.dumps(prepare(a.evidence,a.tokenizer,a.prices,a.out),indent=2))
 elif a.command=='run':run(a.manifest,a.out)
 else:print(json.dumps(evaluate(a.manifest,a.paid,a.out,derive_changes=a.derive_changes),indent=2))
