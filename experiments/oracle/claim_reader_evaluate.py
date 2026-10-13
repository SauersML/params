"""Score all fixed42conditions/view; missing/unaccepted finishes remain failures."""
import argparse,collections,json
from pathlib import Path
import claim_reader_probe as probe

def prediction(r):
 if r is None:return None,'missing'
 if r.get('parse_error'):return None,'parse_error'
 f=str(r.get('finish','')).lower()
 if f!='stop' and not f.endswith('.stop'):return None,'unfinished'
 try:
  obj=json.loads(r['completion'])
  if not isinstance(obj,dict) or set(obj)!={'top_token'} or not isinstance(obj['top_token'],str):raise ValueError()
  return obj['top_token'],None
 except (KeyError,ValueError,TypeError):return None,'malformed'
def evaluate(manifest,paid,out):
 m=json.loads(manifest.read_text());allowed={b['request_id'] for b in m['bindings']};lookup={}
 for line in (paid/'responses.jsonl').read_text().splitlines():
  r=json.loads(line);k=r['request_id']
  if k not in allowed or k in lookup:raise ValueError('unexpected/duplicate response')
  lookup[k]=r
 report={}
 for view in probe.VIEWS:
  source=m['sources'][view]
  if probe.sha(source['path'])!=source['sha256']:raise ValueError('gold source changed')
  rows=[json.loads(x) for x in Path(source['path']).read_text().splitlines()];gold={r['info']['case_id']+'|'+r['info']['condition_private']:json.loads(r['completion'][0]['content'])['top_token'] for r in rows};pred={};cases=collections.defaultdict(list);fail=collections.Counter();tasks=collections.defaultdict(collections.Counter)
  for b in m['bindings']:
   if b['view']!=view:continue
   value,reason=prediction(lookup.get(b['request_id']));pred[b['id']]=value;cases[b['case_id']].append(b['id']);fail[reason]+=bool(reason);tasks[b['task']]['rows']+=1;tasks[b['task']]['correct']+=value is not None and value==gold[b['id']]
  if len(pred)!=42 or set(pred)!=set(gold) or any(len(g)!=3 for g in cases.values()):raise ValueError('fixed cohort corrupted')
  pairs=[(a,b) for g in cases.values() for i,a in enumerate(g) for b in g[i+1:]]
  report[view]={'correct':sum(p is not None and p==gold[k] for k,p in pred.items()),'rows':42,'joint_triplets_correct':sum(all(pred[k] is not None and pred[k]==gold[k] for k in g) for g in cases.values()),'triplets':14,'difference_pairs_correct':sum(pred[a] is not None and pred[b] is not None and ((pred[a]!=pred[b])==(gold[a]!=gold[b])) for a,b in pairs),'pairs':42,'failures':{k:v for k,v in fail.items() if k},'per_task':{k:dict(v) for k,v in tasks.items()},'predictions':pred}
 report['paired_task_feature_minus_geometry']={t:report[probe.VIEWS[0]]['per_task'][t]['correct']-report[probe.VIEWS[1]]['per_task'][t]['correct'] for t in report[probe.VIEWS[0]]['per_task']};report['blind_ceiling']={'top':23,'rows':42,'joint':1,'triplets':14};report['scope']='Untuned same model firstgreedy;3internalTRAINtask clusters, exact decodedstrings inclwhitespace; failures retained; no causal English certificate or broadgeneralization.';out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('manifest','paid','out'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();evaluate(a.manifest,a.paid,a.out)
