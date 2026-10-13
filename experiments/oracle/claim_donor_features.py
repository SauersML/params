"""Outcome-blind unmodified donor features; never receives edited outcome files."""
import argparse,hashlib,json,sys,time
from pathlib import Path
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def delta_write(u,a0,a1):
 import torch
 norms=u.norm(dim=1);directions=u/norms.clamp_min(1e-30)[:,None]
 directions=torch.where((norms>0)[:,None],directions,torch.zeros_like(directions))
 return ((a1-a0)[:,None]*directions).sum(0)

def project(vector,site,target):
 """Return architecture-static direction, never intervention probabilities."""
 kind=site.rsplit('.',1)[1];prefix=site.rsplit('.',1)[0]
 if kind in ('q_proj','k_proj'):return None,'unavailable: query/key space is not residual space'
 if kind=='c_fc':vector=vector @ target.site(prefix+'.down_proj').W.T;scope='ignores GELU and later routing'
 elif kind=='v_proj':vector=vector @ target.site(prefix+'.o_proj').W.T;scope='ignores attention mixing and later routing'
 elif kind in ('o_proj','down_proj'):scope='ignores later routing and final normalization denominator'
 else:raise ValueError('unsupported site type')
 return (vector*target.ln_f) @ target.wte.T,scope

def extremes(logits,tokenizer,k=5):
 if logits is None:return None
 return {name:[{'native_id':int(i),'token':tokenizer.decode([int(i)]),'static_score':float(logits[i])} for i in inds] for name,inds in [('highest',logits.topk(k).indices),('lowest',(-logits).topk(k).indices)]}

def coords(vector,k=8):
 v=vector.detach().float().cpu();order=sorted(range(v.numel()),key=lambda i:(-abs(float(v[i])),i))[:k]
 return [[i,float(v[i])] for i in order]

def extract(repo,protocol_path,manifest,recipient_path,out):
 if out.exists():raise ValueError('preserve immutable feature file')
 p=json.loads(protocol_path.read_text());old=json.loads(recipient_path.read_text());expected=json.loads(manifest.read_text())
 if digest([p['tasks'],p['cases']])!=p['protocol_sha256'] or old['protocol_sha256']!=p['protocol_sha256'] or old['model_content_sha256']!=expected:raise ValueError('input binding mismatch')
 if any(not t.startswith('text') or not t[4:].isdigit() or not 6000<=int(t[4:])<=7999 or t=='text6016' for t in p['tasks']):raise ValueError('nonreserved TRAIN only')
 worlds={};bindings={}
 for case in p['cases']:
  task=case['task']
  for name,changes in case['intervention']['transplants']:
   ids=list(p['tasks'][task]['token_ids'])
   for pos,tok in changes.items():
    pos=int(pos)
    if not 0<=pos<len(ids) or type(tok)!=int or tok<0:raise ValueError('invalid donor')
    ids[pos]=tok
   wid=digest([task,ids]);worlds[wid]={'task':task,'token_ids':ids};bindings[case['id']+'|'+name]=wid
 if len(worlds)!=71:raise ValueError('expected frozen71 donor worlds')
 sys.path[:0]=[str(repo/'method/oracle'),str(repo/'method/vpd')]
 import torch,native,vpd_model,mech
 paths={'target':vpd_model.TARGET_DIR/'model_step_99999.safetensors','config':vpd_model.TARGET_DIR/'model_config.yaml','tokenizer':mech.VPD4L_TOKENIZER,'decomposition':vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV}
 if {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('content hash mismatch before Native')
 torch.set_num_threads(4);torch.set_grad_enabled(False);nat=native.Native('cpu');tk=mech.tokenizer('vpd4l');started=time.monotonic();outputs={};predictions={};static={}
 # Static descriptors for exactly the ORIGINAL8 selected units, no reranking.
 for key,f in old['features'].items():
  static[key]=[]
  for sample in f['samples']:
   site,pos,c=sample['node_native'];logits,scope=project(nat.target.site(site).U[c].float(),site,nat.target)
   static[key].append({'node_native':[site,pos,c],'scope':scope,'tokens':extremes(logits,tk)})
 for wid,w in sorted(worlds.items()):
  nat.vpd.clear()
  for n in nat.names:nat.target.site(n).cache_input=True
  hidden=nat.target.hidden(torch.tensor([w['token_ids']],device=nat.dev))
  q=torch.log_softmax(nat.target.unembed(hidden[:,-1:]).float(),-1)[0,0];idx=int(q.argmax());predictions[wid]={'top_token':tk.decode([idx]),'native_id':idx,'top_probability':float(q[idx].exp())}
  acts={n:(nat.target.site(n).last_input[0] @ nat.target.site(n).V)*nat.target.site(n).U.norm(dim=1) for n in nat.names}
  requests={}
  for case in p['cases']:
   if case['task']!=w['task']:continue
   for name,_ in case['intervention']['transplants']:
    if bindings[case['id']+'|'+name]!=wid:continue
    for arm in case['arms']:
     v=next(v for v in arm['variables'] if v['name']==name);requests[digest(sorted(v['group']))]=v['group']
  for gh,group in sorted(requests.items()):
   base=old['features'][w['task']+'|'+gh];a0={tuple(z['node_native']):z['baseline_activation'] for z in base['member_activations_audit']};streams={};members=[]
   for site,c,pos in sorted(map(tuple,group)):
    nd=(site,pos,c);before=a0[nd];after=float(acts[site][pos,c]);members.append({'node_native':list(nd),'recipient_a':before,'donor_a':after,'delta_a':after-before});streams.setdefault((site,pos),[]).append((c,before,after))
   writes=[]
   for (site,pos),rows in sorted(streams.items()):
    cs,bef,aft=zip(*rows);u=nat.target.site(site).U[list(cs)].float();vector=delta_write(u,torch.tensor(bef),torch.tensor(aft));logits,scope=project(vector,site,nat.target)
    writes.append({'site':site,'token_position':pos,'members':len(rows),'signed_full_group_delta_vector':vector.cpu().tolist(),'delta_top_coordinates':coords(vector),'delta_l2':float(vector.norm()),'sum_individual_delta_norms':sum(abs(a-b) for b,a in zip(bef,aft)),'static_token_projection':extremes(logits,tk),'projection_scope':scope})
   selected={tuple(s['node_native']) for s in base['samples']};outputs[wid+'|'+gh]={'group_sha256':gh,'all_member_deltas_audit':members,'original_selected_member_deltas':[m for m in members if tuple(m['node_native']) in selected],'per_site_position_deltas':writes}
  nat.vpd.clear();del acts,hidden,q
  print('DONOR',wid,len(requests),flush=True)
 if {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('model mutated')
 result={'schema':1,'role':'outcome_blind_unmodified_donor_features','protocol_sha256':p['protocol_sha256'],'recipient_features_sha256':sha(recipient_path),'model_content_sha256':expected,'source_sha256':{m.__name__:sha(m.__file__) for m in (native,vpd_model,mech)},'extractor_sha256':sha(__file__),'worlds':worlds,'case_variable_world_binding':bindings,'donor_forward_count':71,'static_original_selected_units':static,'features':outputs,'common_unmodified_donor_predictions':predictions,'seconds':time.monotonic()-started,'semantics':'a=(x.V)*||U||; delta linear write sums (a_d-a_x)U/||U|| with cancellation, NOT patched output. Static full-vocab projections ignore explicit nonlinearities/routing; no Jacobian, edited output, labels, or outcome-selected vocabulary accessed. Same donor predictions available identically to all triplet arms.'}
 out.write_text(json.dumps(result,allow_nan=False)+'\n');print('FROZEN',sha(out),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ('repo','protocol','manifest','recipient','out'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();extract(a.repo,a.protocol,a.manifest,a.recipient,a.out)
