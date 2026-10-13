"""Post-hoc selected-piece wiring completion, never a repaired primary answer.

Explicit all_edges means every architecture-valid connection among these SAME nodes
and their legal target-output links. It adds no dense-model nodes or remainder,
and it recomputes normal causal attention and RMS from its own retained inputs.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

CASES=('additive_vote_3200000','gate_3200500')
MANIFEST_SHA='2a2e657f77f577bc106b5be718122545f6cc2a8d6c5707009547778ef5c4ab8d'
RESPONSES_SHA='54eee355e75710163a6a129c1c90267787f8d597c9a66fdd98813bd4c4ffbe57'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def complete_same_nodes(graph,native,targets):
 nodes=set(graph.node_set())
 original_parents={r:tuple(w) for r,w in graph.parents.items()}
 original_out=tuple(graph.out)
 parents,out=native.all_edges(sorted(nodes),targets)
 completed=native.Graph(nodes=nodes,parents=parents,out=out,complete=False)
 if completed.complete or completed.node_set()!=nodes or completed.count()!=graph.count():raise RuntimeError('completion changed selected pieces/branch')
 if any(not set(w)<=set(completed.parents.get(r,())) for r,w in graph.parents.items()) or not set(graph.out)<=set(completed.out):raise RuntimeError('original edges are not architecture-valid completion subsets')
 if any(r not in nodes or any(w not in nodes for w in ws) for r,ws in completed.parents.items()) or not set(completed.out)<=nodes:raise RuntimeError('completion introduced outside-component edges')
 if original_parents!={r:tuple(w) for r,w in graph.parents.items()} or original_out!=tuple(graph.out):raise RuntimeError('original circuit mutated')
 inner,out_count=native.implied_edges(nodes,targets)
 if completed.edges()!=inner+out_count:raise RuntimeError('explicit architecture edge count differs')
 return completed,{'selected_nodes':len(nodes),'original_explicit_edges':graph.edges(),
                   'completed_architecture_edges':completed.edges(),'completed_internal_edges':inner,'completed_output_links':out_count}

def run(args):
 if sha(args.prior/'receipt.json')!=args.prior_receipt_sha256 or sha(args.prior/'results.jsonl')!=args.prior_results_sha256:raise ValueError('frozen prior physics changed')
 if sha(args.manifest)!=MANIFEST_SHA or sha(args.responses)!=RESPONSES_SHA or args.out.exists():raise ValueError('frozen diagnostic and fresh output required')
 manifest=json.loads(args.manifest.read_text());response=json.loads(args.responses.read_text())
 if response['manifest_sha256']!=MANIFEST_SHA:raise ValueError('response panel differs')
 panel=[r for r in manifest['panel'] if r['id'] in CASES]
 if [r['id'] for r in panel]!=list(CASES) or sum(len(r['inputs'])*len(r['edits']) for r in panel)!=384:raise ValueError('exact exposed two-case384world panel required')
 for root,field in ((args.source,'source_sha256'),(args.support,'support_sha256')):
  for name,d in manifest[field].items():
   if sha(root/name)!=d:raise ValueError('frozen source changed')
 if sha(args.prior_states/'index.json')!=args.prior_states_sha256:raise ValueError('frozen prior-state index changed')
 state_index=json.loads((args.prior_states/'index.json').read_text())
 oldreceipt=json.loads((args.prior/'receipt.json').read_text())
 for shard in oldreceipt['shards']:
  binding=state_index[str(shard['worker'])]
  if binding['artifact_root']!=shard['artifact_root'] or binding['contract_sha256']!=shard['contract_sha256']:raise ValueError('prior-state mapping differs from frozen receipt')
  directory=args.prior_states/('worker'+str(shard['worker']))
  if sha(directory/'contract.json')!=shard['contract_sha256']:raise ValueError('prior shard contract changed')
  for name,digest in binding['files'].items():
   if '/' in name or name.startswith('.') or sha(directory/name)!=digest:raise ValueError('prior state proof changed')
 if oldreceipt['manifest_sha256']!=MANIFEST_SHA or oldreceipt['responses_sha256']!=RESPONSES_SHA:raise ValueError('prior physics belongs to different responses')
 prior={}
 for line in (args.prior/'results.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['case'] in CASES:
   key=(row['case'],row['arm'],row['edit_index'],row['input'])
   if key in prior:raise ValueError('duplicate prior world')
   prior[key]=row
 expected={(r['id'],arm,e,i) for r in panel for arm in manifest['arms'] for e in range(len(r['edits'])) for i in range(len(r['inputs']))}
 if set(prior)!=expected:raise ValueError('all1920prior candidateworlds required')
 samples={}
 for arm,entry in response['arms'].items():
  path=args.responses.parent/entry['path']
  if sha(path)!=entry['sha256']:raise ValueError('first responses changed')
  values=[json.loads(x) for x in path.read_text().splitlines()]
  if len(values)!=16 or len({r['id'] for r in values})!=16:raise ValueError('exact first-response coverage required')
  samples[arm]={r['id']:r for r in values}
  for item in panel:
   sample=samples[arm][item['id']]
   if sample['prompt_sha256']!=item['prompt_sha256'] or sample['stratum']!=item['stratum'] or sample['sample']!=0:raise ValueError('original first-response prompt binding failed')
 sys.path[:0]=[str(args.source/f) for f in ('experiments/oracle/toyloop','experiments/oracle/toys','method/oracle','method/vpd')]
 import torch,circuit,planted,native,goal_eval
 sys.path.insert(0,str(args.support/'experiments/oracle/toyloop'))
 from matched_graph_score import answer_error
 from head_counterfactual import scaled_head,raw_kl_bits
 from reward_interventions import scaled_neuron,clear_intervention_caches
 from intervention_teacher import tensor_digest
 from safetensors.torch import save_file
 torch.set_num_threads(1);args.out.mkdir()
 write(args.out/'protocol.json',{'cases':list(CASES),'arms':manifest['arms'],'manifest_sha256':MANIFEST_SHA,
  'responses_sha256':RESPONSES_SHA,'prior_receipt_sha256':sha(args.prior/'receipt.json'),
  'prior_states_sha256':args.prior_states_sha256,'prior_results_sha256':sha(args.prior/'results.jsonl'),'source_sha256':manifest['source_sha256'],
  'support_sha256':manifest['support_sha256'],'driver_sha256':sha(__file__),
  'strict_answer_helper_sha256':sha(sys.modules['matched_graph_score'].__file__),
  'reference_worlds':384,'candidate_worlds':1920,'forward_cap':2688,
  'selection':'post-hoc two exposed common failures; no answers or targets changed',
  'semantics':'same selected pieces; explicit all_edges completeFalse; sparse liveFalse own routing/norms',
  'interpretation':'success supports missing wiring; failure does not exclude another wiring on same units'})
 forwards=0;rows=[];graph_receipts=[]
 for case_index,item in enumerate(panel):
  path=args.payload/item['weights']
  if sha(path)!=item['weights_sha256']:raise ValueError('weights changed')
  blob=torch.load(path,map_location='cpu',weights_only=True);toy=SimpleNamespace(sd=blob['sd'],cfg=blob['cfg'],meta={})
  nat=planted.native_of(toy.sd,toy.cfg)
  toy.meta['decomposition']={n:(nat.target.site(n).U,nat.target.site(n).V,nat.units[n]) for n in nat.names}
  def state():return {'model':{k:tensor_digest(v) for k,v in nat.target.state_dict().items()},
   'basis':{n:{k:tensor_digest(getattr(nat.target.site(n),k)) for k in ('W','U','V','delta_T')} for n in nat.names},
   'units':{n:tensor_digest(nat.units[n]) for n in nat.names}}
  original=state()
  # Require exactly the original basis used for the archived primary physics.
  shard=next(s for s in oldreceipt['shards'] if manifest['panel'].index(item) in json.loads((args.prior_states/('worker'+str(s['worker']))/'contract.json').read_text())['case_indices'])
  prior_original=args.prior_states/('worker'+str(shard['worker']))/f'{manifest["panel"].index(item)}.original.json'
  if original!=json.loads(prior_original.read_text()):raise ValueError('prior basis/model identity differs')
  graphs={}
  for arm in manifest['arms']:
   sample=samples[arm][item['id']];error=answer_error(sample,'graph_only');graphs[arm]=[]
   for index,ids in enumerate(item['inputs']):
    graph=None;reason=error;size=None
    if reason is None:
     try:
      parsed=circuit.parse(sample['completion'])
      if not parsed.ok or parsed.bad:raise ValueError('strict circuit parser rejected answer')
      stated=circuit.unit_graph(parsed,toy,ids)
      graph,size=complete_same_nodes(stated,native,[len(ids)-1])
     except ValueError as exc:reason=str(exc)
    old=prior[item['id'],arm,0,index]
    if old['valid']!=(graph is not None) or (not old['valid'] and old['invalid_reason']!=reason):raise ValueError('original validity changed')
    graphs[arm].append((graph,reason,size))
    graph_receipts.append({'case':item['id'],'arm':arm,'input':index,'valid':graph is not None,
      'invalid_reason':reason,'size':size,'original_response_sha256':response['arms'][arm]['sha256'],
      'selected_nodes_sha256':None if graph is None else hashlib.sha256(json.dumps(sorted(graph.node_set())).encode()).hexdigest()})
  # All graphs compiled before literal edits; full model only integrity/reference.
  full=[native.Graph({(n,t,c) for n in nat.names for t in range(len(ids)) for c in range(int(nat.C[n]))},complete=True) for ids in item['inputs']]
  baselines=[]
  for edit_index,edit in enumerate(item['edits']):
   with contextlib.ExitStack() as stack:
    for kind,layer,unit,scale in edit['operations']:stack.enter_context((scaled_head if kind=='head' else scaled_neuron)(nat,layer,unit,scale))
    for index,ids in enumerate(item['inputs']):
     clear_intervention_caches(nat);p=nat.reference([ids],[len(ids)-1]).detach().clone()
     integrity=goal_eval.GoalEval(nat).graph_ref(ids,[len(ids)-1],full[index],live=True,norms={});forwards+=2
     integrity_kl=raw_kl_bits(p,integrity)
     if integrity_kl>.001:raise ValueError('full architecture integrity failed')
     if edit_index==0:baselines.append(p.clone())
     tensors={'p':p.contiguous(),'full':integrity.contiguous(),'p0':baselines[index].contiguous()};world=[]
     for arm in manifest['arms']:
      graph,error,size=graphs[arm][index];old=prior[item['id'],arm,edit_index,index]
      clear_intervention_caches(nat)
      q=None if graph is None else goal_eval.GoalEval(nat).graph_ref(ids,[len(ids)-1],graph,live=False,norms={})
      forwards+=int(q is not None);raw=None if q is None else raw_kl_bits(p,q)
      if q is not None:tensors[arm]=q.contiguous()
      world.append({'case':item['id'],'arm':arm,'edit_index':edit_index,'input':index,
       'kind':edit['kind'],'operations':edit['operations'],'valid':q is not None,'invalid_reason':error,
       'raw_kl_bits':raw,'kl_bits':None if raw is None else max(0.,raw),'top':None if q is None else int(q[0,-1].argmax()),
       'reference_top':int(p[0,-1].argmax()),'prior_row':old,'integrity_raw_kl_bits':integrity_kl,'size':size})
     name=f'{case_index}_{edit_index}_{index}.safetensors';save_file(tensors,str(args.out/name))
     for row in world:row.update(file=name,sha256=sha(args.out/name));rows.append(row)
     with (args.out/'results.jsonl').open('a') as f:
      for row in world:f.write(json.dumps(row,allow_nan=False)+'\n')
      f.flush();os.fsync(f.fileno())
   clear_intervention_caches(nat)
   if state()!=original:raise ValueError('literal edit restoration failed')
  if sha(path)!=item['weights_sha256']:raise ValueError('model asset changed during execution')
  write(args.out/f'{case_index}.restoration.json',{'original':original,'restored':state()})
  print(json.dumps({'case':item['id'],'actual_forwards':forwards}),flush=True)
 if forwards>2688 or len(rows)!=1920:raise ValueError('world/cap mismatch')
 write(args.out/'graphs.json',graph_receipts)
 summary=[]
 for item in panel:
  for arm in manifest['arms']:
   values=[r for r in rows if r['case']==item['id'] and r['arm']==arm]
   summary.append({'case':item['id'],'arm':arm,'worlds':len(values),
    'prior_all_pass':all(r['prior_row']['valid'] and r['prior_row']['kl_bits']<=.001 for r in values),
    'complete_all_pass':all(r['valid'] and r['kl_bits']<=.001 for r in values),
    'prior_worst_kl':max((r['prior_row']['kl_bits'] for r in values if r['prior_row']['valid']),default=None),
    'complete_worst_kl':max((r['kl_bits'] for r in values if r['valid']),default=None)})
 write(args.out/'complete.json',{'actual_forwards':forwards,'forward_cap':2688,'candidate_worlds':len(rows),'summary':summary})
 print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('manifest','responses','prior','payload','source','support','out','prior_states'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
 p.add_argument('--prior-states-sha256',required=True)
 p.add_argument('--prior-receipt-sha256',required=True);p.add_argument('--prior-results-sha256',required=True)
 run(p.parse_args())
