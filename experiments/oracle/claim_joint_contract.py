"""Frozen weight-geometry-selected two-unit final-down interaction; no learning/API."""
import argparse,itertools,json,sys,time,types
from pathlib import Path
from claim_lastdown_contract import canonical,sha,SITE

def algebra(r,deltas,readout,eps):
 import torch
 if len(deltas)!=2 or r.ndim!=1 or readout.shape!=r.shape or any(d.shape!=r.shape for d in deltas):raise ValueError('two commensurate final-residual vectors required')
 D=r.numel();R0=(r.square().mean()+eps).sqrt();N0=r@readout;single=[];norms=[]
 for d in deltas:
  R=((r+d).square().mean()+eps).sqrt();norms.append(R);single.append((N0+d@readout)/R)
 total=sum(deltas,torch.zeros_like(r));Rj=((r+total).square().mean()+eps).sqrt();Nj=N0+total@readout;g0=N0/R0;gj=Nj/Rj;cross=2*(deltas[0]@deltas[1]);radicand=norms[0].square()+norms[1].square()-R0.square()
 return {'base_gap':float(g0),'single_gaps':[float(x) for x in single],'joint_gap':float(gj),'additive_single_effect_gap':float(single[0]+single[1]-g0),'no_cross_RMS_gap':float(Nj/radicand.sqrt()) if float(radicand)>0 else None,'no_cross_RMS_radicand':float(radicand),'cross_term_2_dot_d1_d2':float(cross),'cross_RMS_squared_term':float(cross/D),'base_RMS':float(R0),'single_RMS':[float(x) for x in norms],'joint_RMS':float(Rj),'joint_numerator':float(Nj)}

def run(repo,prior_panel_path,manifest,out):
 if out.exists():raise ValueError('preserve immutable artifacts')
 prior=json.loads(prior_panel_path.read_text());task='text6034';p=prior['tasks'][task]
 if len(p['units'])!=8 or len(p['donors'])!=2 or set(map(int,p['donors'][0])) & set(map(int,p['donors'][1])):raise ValueError('two independent fixed donor positions required')
 sys.path[:0]=[str(repo/'method/oracle'),str(repo/'method/vpd')]
 import torch,native,vpd_model,mech,variables,semantic,goal_eval,engine
 paths={'target':vpd_model.TARGET_DIR/'model_step_99999.safetensors','config':vpd_model.TARGET_DIR/'model_config.yaml','tokenizer':mech.VPD4L_TOKENIZER,'decomposition':vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV};expected=json.loads(manifest.read_text())
 if {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('model binding')
 torch.set_num_threads(4);torch.set_grad_enabled(False);nat=native.Native('cpu');tk=mech.tokenizer('vpd4l');weights=nat.target.site(SITE).U;records=[]
 for i,j in itertools.combinations(sorted(p['units']),2):
  u,v=weights[i].float(),weights[j].float();nu,nv=float(u.norm()),float(v.norm());cos=float(u@v)/(nu*nv) if nu*nv else 0.;records.append({'units':[i,j],'write_cosine':cos,'zero_norm':nu*nv==0})
 anti=sorted(records,key=lambda r:(r['write_cosine'],r['units']))[:2];used={tuple(r['units']) for r in anti};aligned=sorted((r for r in records if tuple(r['units']) not in used),key=lambda r:(-r['write_cosine'],r['units']))[:2]
 pairs=[{**r,'selection':'two most antiparallel'} for r in anti]+[{**r,'selection':'two most aligned remaining'} for r in aligned]
 panel={'task':task,'token_ids':p['token_ids'],'target_positions':p['target_positions'],'units':p['units'],'donors':p['donors'],'pairs':pairs,'assignments':[[0,1],[1,0]],'selection':'Only geometry of eight immutable prior units; four pairs frozen before first model forward. Both donor assignments retained. No measured-outcome selection.','model_content_sha256':expected,'prior_panel_sha256':sha(prior_panel_path),'joint_cases':8,'single_cases':16}
 out.mkdir();(out/'frozen_panel.json').write_text(canonical(panel)+'\n')
 # First model forward is below; selection cannot access any recipient/donor/patched outputs.
 started=time.monotonic();sc=variables.Scorer(goal_eval.GoalEval(nat),rows=1);ids=p['token_ids'];last=len(ids)-1;raw=[{'name':'V'+str(i+1),'group':[[SITE,c,last]],'inputs':[494,504],'meaning':'','feeds':[]} for i,c in enumerate(p['units'])];vs=variables.parse(raw,nat.C);names={c:vs[i]['name'] for i,c in enumerate(p['units'])};checker=semantic.Checker(sc,types.SimpleNamespace(ids=ids,targets=[last],seed=native.task_seed(task)),vs);captured={};original_norm=nat.target.norm;site=nat.target.site(SITE);original_forward=site.forward;U0=site.U.clone();W0=site.W.clone()
 def norm(this,x,j,fused=False):
  if j==2*this.n_layer:captured['r']=x[0,last].float().clone()
  return original_norm(x,j,fused)
 nat.target.norm=types.MethodType(norm,nat.target)
 def direct(delta):
  def patched(this,x):
   y=original_forward(x).clone();y[:,last]+=delta;return y
  site.forward=types.MethodType(patched,site)
  try:
   nat.vpd.clear();q=torch.log_softmax(nat.target.unembed(nat.target.hidden(torch.tensor([ids]))[:,-1:]).float(),-1)[0,0];r=captured['r'].clone();return q,r
  finally:site.forward=original_forward
 try:
  q0,r=direct(torch.zeros_like(site.U[0]));A,B=map(int,q0.topk(2).indices);readout=nat.target.ln_f.float()*(nat.target.wte[A]-nat.target.wte[B]).float();eps=nat.target.eps;gap0=float(q0[A]-q0[B]);base_error=float((q0-checker.p0).abs().max())
  if base_error>1e-4:raise ValueError('independent base parity')
  sources=[checker.text(ch) for ch in p['donors']];deltas={};single={};single_rows=[]
  for c in p['units']:
   for idx,src in enumerate(sources):
    x0=float(checker.base['act'][SITE][last,c]);x1=float(src['act'][SITE][last,c]);d=(x1-x0)*site.U[c].float();deltas[c,idx]=d;iv={'transplants':[(names[c],p['donors'][idx])]};q=checker.run(iv).float().clone();qd,rd=direct(d);single[c,idx]=float(q[A]-q[B]);e=float((q-qd).abs().max());row={'unit':c,'donor_index':idx,'pair_gap':single[c,idx],'delta_raw_scalar':x1-x0,'delta_vector_l2':float(d.norm()),'Native_direct_max_abs_logp':e,'actual_RMS':float((rd.square().mean()+eps).sqrt())};single_rows.append(row)
    with (out/'single_rows.jsonl').open('a') as f:f.write(canonical(row)+'\n')
    if e>1e-4:raise ValueError('single physical parity')
  joint_rows=[]
  for pair in pairs:
   i,j=pair['units']
   for a,b in panel['assignments']:
    d1,d2=deltas[i,a],deltas[j,b];analysis=algebra(r,[d1,d2],readout,eps);iv={'transplants':[(names[i],p['donors'][a]),(names[j],p['donors'][b])]};q=checker.run(iv).float().clone();qd,rd=direct(d1+d2);actual=float(q[A]-q[B]);native_add=single[i,a]+single[j,b]-gap0;parity=float((q-qd).abs().max());analytic_error=abs(analysis['joint_gap']-actual);row={'units':[i,j],'donor_indices':[a,b],'selection':pair['selection'],'write_cosine':pair['write_cosine'],'A':{'id':A,'token':tk.decode([A])},'B':{'id':B,'token':tk.decode([B])},'base_gap':gap0,'Native_joint_gap':actual,'Native_single_effect_additive_gap':native_add,'Native_joint_minus_additive':actual-native_add,'independent_direct_gap':float(qd[A]-qd[B]),'actual_joint_RMS':float((rd.square().mean()+eps).sqrt()),'Native_direct_max_abs_logp':parity,'analytic_error':analytic_error,'joint_vs_sum_deltas_residual_max_error':float((rd-(r+d1+d2)).abs().max()),'algebra':analysis};joint_rows.append(row)
    with (out/'joint_rows.jsonl').open('a') as f:f.write(canonical(row)+'\n')
    print('JOINT',i,j,a,b,actual-native_add,analysis['cross_term_2_dot_d1_d2'],analytic_error,flush=True)
    if parity>1e-4 or analytic_error>1e-4:raise ValueError('joint physical parity')
 finally:site.forward=original_forward;nat.target.norm=original_norm
 if not torch.equal(U0,site.U) or not torch.equal(W0,site.W) or {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('restoration/model failed')
 result={'schema':1,'frozen_panel_sha256':sha(out/'frozen_panel.json'),'model_content_sha256':expected,'source_sha256':{m.__name__:sha(m.__file__) for m in (native,vpd_model,mech,variables,semantic,goal_eval,engine)},'runner_sha256':sha(__file__),'base_Native_direct_error':base_error,'single_rows':single_rows,'joint_rows':joint_rows,'seconds':time.monotonic()-started,'weights_bitwise_restored':True,'scope':'Actual two-unit final-RMS interaction, conditional on full unmodified boundary; not fullcircuit/wholemechanism proof. All fixed pairs/donors retained.'};(out/'results.json').write_text(canonical(result)+'\n');print('DONE',len(joint_rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ('repo','prior-panel','manifest','out'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();run(a.repo,a.prior_panel,a.manifest,a.out)
