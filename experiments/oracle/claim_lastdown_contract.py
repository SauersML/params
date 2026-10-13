"""Exact conditional single final-down scalar transplant; no training or writer calls."""
import argparse,hashlib,json,sys,time,types
from pathlib import Path
SITE='h.3.mlp.down_proj'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def freeze(protocol,features):
 if features['protocol_sha256']!=protocol['protocol_sha256'] or digest([protocol['tasks'],protocol['cases']])!=protocol['protocol_sha256']:raise ValueError('binding')
 available={}
 for t,p in protocol['tasks'].items():
  last=len(p['token_ids'])-1
  available[t]=len({r['node_native'][2] for key,f in features['features'].items() if key.startswith(t+'|') for r in f['member_activations_audit'] if r['node_native'][:2]==[SITE,last]})
 names=sorted((t for t in protocol['tasks'] if features['task_splits'][t]=='train' and available[t]>=8),key=lambda t:digest(['exact-lastdown-v1',t]))[:4];tasks={}
 if len(names)!=3:raise ValueError('expectedthree structurallyeligible TRAIN contexts: '+str(available))
 for t in names:
  if not 6000<=int(t[4:])<=7999 or t=='text6016':raise ValueError('permitted TRAIN only')
  p=protocol['tasks'][t];last=len(p['token_ids'])-1
  if p['target_positions']!=[last]:raise ValueError('LASTtarget only')
  acts={};changes={}
  for k,f in features['features'].items():
   if not k.startswith(t+'|'):continue
   for r in f['member_activations_audit']:
    n,pos,c=r['node_native']
    if n==SITE and pos==last:
     if c in acts and acts[c]!=r['baseline_activation']:raise ValueError('conflicting baseline')
     acts[c]=r['baseline_activation']
  top=sorted(acts,key=lambda c:(-abs(acts[c]),c))[:4];tail=sorted(set(acts)-set(top),key=lambda c:digest(['exact-lastdown-units-v1',t,c]))[:4];units=top+tail
  if len(units)!=8:raise ValueError('eight existing final-target units needed')
  for case in protocol['cases']:
   if case['task']==t:
    for _,ch in case['intervention']['transplants']:changes[digest(ch)]=ch
  donors=[changes[k] for k in sorted(changes)[:2]]
  if len(donors)!=2:raise ValueError('two existing donors needed')
  tasks[t]={'token_ids':p['token_ids'],'target_positions':[last],'units':units,'donors':donors,'selection_baseline_a':{str(c):acts[c] for c in units}}
 return {'tasks':tasks,'structural_available_lastdown_units':available,'selection':'allthree structurallyeligible TRAIN contexts in fixedhashorder; four highestabs(unmodified normalizeda) plusfour hashunits fromexistingfinaldown LASTtarget membership; two existing donor maps byhash; no patchedoutcomes accessed','scope':'Single rankone finaldown unit; conditional on unmodified full residual boundary, NOT fullcircuit/mechanism certificate','cases':48}
def exact_pair_effect(r,u,delta_raw,readout,eps):
 """Exact final-RMS pair effect, conditional on the unmodified residual boundary."""
 import torch
 R0=(r.square().mean()+eps).sqrt();d=delta_raw*u;R1=((r+d).square().mean()+eps).sqrt();base=(r@readout)/R0;num=d@readout
 return {'base_pair_logodds':float(base),'patched_pair_logodds':float(((r+d)@readout)/R1),'delta_numerator':float(num),'direct_term':float(num/R1),'normalizer_term':float(base*(R0/R1-1)),'base_RMS':float(R0),'patched_RMS':float(R1)}

def run(repo,protocol_path,features_path,manifest,out,panel_path):
 if out.exists():raise ValueError('preserve immutable results')
 p=json.loads(protocol_path.read_text());f=json.loads(features_path.read_text());panel=freeze(p,f)
 frozen=panel_path
 if canonical(panel)!=canonical(json.loads(frozen.read_text())):raise ValueError('frozen preflight panel changed')
 out.mkdir();(out/'frozen_panel.json').write_text(canonical(panel)+'\n')
 sys.path[:0]=[str(repo/'method/oracle'),str(repo/'method/vpd')]
 import torch,native,vpd_model,mech,variables,semantic,goal_eval,engine
 paths={'target':vpd_model.TARGET_DIR/'model_step_99999.safetensors','config':vpd_model.TARGET_DIR/'model_config.yaml','tokenizer':mech.VPD4L_TOKENIZER,'decomposition':vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV};expected=json.loads(manifest.read_text())
 if {k:sha(v) for k,v in paths.items()}!=expected or f['model_content_sha256']!=expected:raise ValueError('content binding')
 torch.set_num_threads(4);torch.set_grad_enabled(False);nat=native.Native('cpu');tk=mech.tokenizer('vpd4l');sc=variables.Scorer(goal_eval.GoalEval(nat),rows=1);started=time.monotonic();results=[];originalU=nat.target.site(SITE).U.clone();originalW=nat.target.site(SITE).W.clone()
 for task,pp in panel['tasks'].items():
  ids=pp['token_ids'];last=len(ids)-1;raw=[{'name':'V'+str(i+1),'group':[[SITE,c,last]],'inputs':sorted({int(pos) for ch in pp['donors'] for pos in ch}),'meaning':'','feeds':[]} for i,c in enumerate(pp['units'])];vs=variables.parse(raw,nat.C);checker=semantic.Checker(sc,types.SimpleNamespace(ids=ids,targets=[last],seed=native.task_seed(task)),vs)
  captured={};original_norm=nat.target.norm
  def norm(this,x,j,fused=False):
   if j==2*this.n_layer:captured['r']=x[0,last].float().clone()
   return original_norm(x,j,fused)
  nat.target.norm=types.MethodType(norm,nat.target)
  try:
   nat.vpd.clear();q0=torch.log_softmax(nat.target.unembed(nat.target.hidden(torch.tensor([ids]))[:,-1:]).float(),-1)[0,0];r=captured['r'];norm0=(r.square().mean()+nat.target.eps).sqrt();A,B=map(int,q0.topk(2).indices);gap0=float(q0[A]-q0[B]);readout=nat.target.ln_f.float()*(nat.target.wte[A]-nat.target.wte[B]).float()
   parity0=float((q0-checker.p0).abs().max())
   if parity0>1e-4:raise ValueError('independent base parity')
   for donor_idx,changes in enumerate(pp['donors']):
    src=checker.text(changes)
    for i,c in enumerate(pp['units']):
     u=nat.target.site(SITE).U[c].float();unorm=u.norm();raw0=float(checker.base['act'][SITE][last,c]);raw1=float(src['act'][SITE][last,c]);delta_raw=raw1-raw0;delta_a=delta_raw*float(unorm);direction=u/unorm if float(unorm)>0 else torch.zeros_like(u);coeff=float(direction@readout);d=delta_raw*u;norm1=((r+d).square().mean()+nat.target.eps).sqrt();pred_gap=float((norm0*(q0[A]-q0[B])+delta_a*coeff)/norm1)
     q=checker.run({'transplants':[(vs[i]['name'],changes)]}).detach().float().clone();native_gap=float(q[A]-q[B])
     site=nat.target.site(SITE);original_forward=site.forward
     def patched(this,x):
      y=original_forward(x);y=y.clone();y[:,last]+=d;return y
     site.forward=types.MethodType(patched,site)
     try:
      nat.vpd.clear();q_direct=torch.log_softmax(nat.target.unembed(nat.target.hidden(torch.tensor([ids]))[:,-1:]).float(),-1)[0,0]
     finally:site.forward=original_forward
     parity=float((q-q_direct).abs().max());error=abs(pred_gap-native_gap)
     row={'task':task,'site':SITE,'position':last,'unit':c,'selection_rank':i,'donor_index':donor_idx,'donor_changes':changes,'A':{'native_id':A,'token':tk.decode([A])},'B':{'native_id':B,'token':tk.decode([B])},'base_pair_logodds':gap0,'base_RMS':float(norm0),'patched_RMS':float(norm1),'base_residual_dot_delta':float(r@d),'base_residual_squared_norm':float(r@r),'delta_squared_norm':float(d@d),'delta_normalized_activation':delta_a,'unit_write_norm':float(unorm),'pair_readout_coefficient':coeff,'expected_pair_logodds':pred_gap,'actual_pair_logodds':native_gap,'direct_pair_logodds':float(q_direct[A]-q_direct[B]),'analytic_abs_error':error,'Native_vs_direct_max_abs_logp':parity,'base_Native_vs_direct_max_abs_logp':parity0,'actual_order_A_gt_B':native_gap>0,'baseline_order_changed':native_gap<0,'raw_scalar_before':raw0,'raw_scalar_donor':raw1,'actual_delta_write_l2':float(d.norm()),'unchanged_scalar':delta_raw==0}
     results.append(row)
     with (out/'rows.jsonl').open('a') as stream:stream.write(canonical(row)+'\n');stream.flush()
     print('CASE',task,donor_idx,c,error,parity,flush=True)
     if error>1e-4 or parity>1e-4:raise ValueError('physical parity failure retained in stdout')
  finally:nat.target.norm=original_norm
  del checker,src;nat.vpd.clear()
 if not torch.equal(originalU,nat.target.site(SITE).U) or not torch.equal(originalW,nat.target.site(SITE).W):raise ValueError('weights changed')
 if {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('assets changed')
 result={'schema':1,'panel_sha256':sha(out/'frozen_panel.json'),'features_sha256':sha(features_path),'model_content_sha256':expected,'source_sha256':{m.__name__:sha(m.__file__) for m in (native,vpd_model,mech,variables,semantic,goal_eval,engine)},'runner_sha256':sha(__file__),'rows':results,'seconds':time.monotonic()-started,'scope':panel['scope'],'weights_bitwise_restored':True}
 (out/'results.json').write_text(canonical(result)+'\n');print('DONE',len(results),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ('repo','protocol','features','manifest','out','panel'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();run(a.repo,a.protocol,a.features,a.manifest,a.out,a.panel)
