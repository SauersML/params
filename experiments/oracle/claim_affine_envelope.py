"""Information-complete conditional final-down argmax envelopes; no writer/API calls.

The FP64 envelope of exported FP32 coefficients is an ideal affine contract.
Literal FP32 execution is checked at a fixed panel, not universally certified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

SELECTED={'text6008':3268,'text6011':2412,'text6034':1833}
SCALES=(-1.,0.,.5,1.,1.5,2.)
SITE='h.3.mlp.down_proj'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def upper_envelope(base,slope,bounds=(-1.,2.)):
 """All-vocabulary upper hull of b+(k-1)s, including parallel tie regions."""
 b=np.asarray(base,dtype=np.float64);s=np.asarray(slope,dtype=np.float64)
 if b.ndim!=1 or b.shape!=s.shape or len(b)<2 or not np.isfinite(b).all() or not np.isfinite(s).all():raise ValueError('finite full-vocabulary vectors required')
 lo,hi=map(float,bounds)
 if not math.isfinite(lo+hi) or lo>=hi:raise ValueError('ordered finite bounds')
 intercept=b-s;parallel={}
 for i,(m,c) in enumerate(zip(s,intercept)):
  old=parallel.get(float(m))
  if old is None or c>old['intercept']:parallel[float(m)]={'slope':float(m),'intercept':float(c),'tokens':[i]}
  elif c==old['intercept']:old['tokens'].append(i)
 hull=[];starts=[]
 for line in (parallel[m] for m in sorted(parallel)):
  x=-math.inf
  while hull:
   x=(hull[-1]['intercept']-line['intercept'])/(line['slope']-hull[-1]['slope'])
   if x>starts[-1]:break
   hull.pop();starts.pop()
  if not hull:x=-math.inf
  hull.append(line);starts.append(x)
 segments=[]
 for j,line in enumerate(hull):
  left=max(lo,starts[j]);right=min(hi,starts[j+1] if j+1<len(starts) else hi)
  if left<right:segments.append({'open_interval':[left,right],**line,'winner':min(line['tokens'])})
 points=sorted({lo,hi,*[x for x in starts if lo<=x<=hi]})
 boundaries=[]
 for k in points:
  z=b+(k-1)*s;peak=float(z.max());ties=np.flatnonzero(np.abs(z-peak)<=1e-10).tolist()
  boundaries.append({'k':k,'winner':int(z.argmax()),'numerical_ties_1e_10':ties})
 return {'range':[lo,hi],'vocabulary_size':len(b),'segments':segments,'boundaries':boundaries,'tie_rule':'smallest native token ID for exact equal logits; numerical boundary ties flagged separately'}

def run(repo,panel,manifest,out):
 if out.exists():raise ValueError('immutable output required')
 p=json.loads(panel.read_text());expected=json.loads(manifest.read_text())
 if set(p['tasks'])!=set(SELECTED):raise ValueError('same three TRAIN contexts required')
 for name,row in p['tasks'].items():
  if not 6000<=int(name[4:])<=7999 or row['target_positions']!=[len(row['token_ids'])-1]:raise ValueError('TRAIN LAST target only')
 sys.path[:0]=[str(repo/'method/oracle'),str(repo/'method/vpd')]
 import torch,native,vpd_model,mech,edit_affine,edit_eval
 from safetensors.torch import save_file
 paths={'target':vpd_model.TARGET_DIR/'model_step_99999.safetensors','config':vpd_model.TARGET_DIR/'model_config.yaml','tokenizer':mech.VPD4L_TOKENIZER,'decomposition':vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV}
 if {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('four exact assets required')
 torch.set_num_threads(4);torch.set_grad_enabled(False);nat=native.Native('cpu');tk=mech.tokenizer('vpd4l');site=nat.target.site(SITE);saved={k:getattr(site,k).clone() for k in ('W','U','V')};out.mkdir();started=time.monotonic();cases={};tensors={}
 for name,c in SELECTED.items():
  ids=p['tasks'][name]['token_ids'];keep={};base,a=edit_affine.coefficients(nat,ids,len(ids)-1,SITE,retain=keep);u=site.U[c].float();r=keep['residual_pre_final_rms'];slope=a[c]*(u*nat.target.ln_f.float())@nat.target.wte.float().T
  env=upper_envelope(base.numpy(),slope.numpy())
  for seg in env['segments']:seg['token_strings']={str(i):tk.decode([i]) for i in seg['tokens']}
  for boundary in env['boundaries']:boundary['token_strings']={str(i):tk.decode([i]) for i in boundary['numerical_ties_1e_10']}
  cases[name]={'site':SITE,'component_index':c,'token_position':len(ids)-1,'execution_text':tk.decode(ids,skip_special_tokens=False),'raw_scalar_x_dot_V':float(a[c]),'r_squared_norm':float(r.double().square().sum()),'U_squared_norm':float(u.double().square().sum()),'r_dot_U':float(r.double()@u.double()),'residual_width':len(r),'envelope':env}
  for key,value in {'base':base,'slope':slope,'residual':r,'U':u,'V':site.V[:,c],'raw_scalar':a[c]}.items():tensors[name+'_'+key]=value.contiguous()
 save_file(tensors,str(out/'coefficients.safetensors'))
 payload={'schema':1,'role':'information_complete_local_weight_edit_writer_input','cases':cases,'model_content_sha256':expected,'coefficient_sha256':sha(out/'coefficients.safetensors'),'mechanism':{'operation':'persistent W(k)=W+(k-1)outer(U_c,V_c), at all positions','site_input':'final down_proj input unchanged by its own weight edit; raw scalar a=x dot V_c','target_residual':'r(k)=r+(k-1)a U_c, full unmodified residual boundary GIVEN','normalization':'positive shared final RMS denominator sqrt(mean(r(k)^2)+epsilon); learned final norm weights ln_f included in numerator','readout':'all native token numerators z(k)=base+(k-1)slope; base=(r*ln_f)E.T; slope=a*(U_c*ln_f)E.T','argmax':'positive common RMS cancels for argmax; full-vocabulary upper envelope covers only declared k range','scope':'single final-down persistent component; same known model/context/boundary; not a full circuit explanation or probability guarantee','precision':'ideal FP64 upper hull of frozen FP32 coefficients; endpoints and near-ties explicit; sampled FP32 checks are separate'}}
 (out/'writer_payload.json').write_text(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n');frozen_sha=sha(out/'writer_payload.json')
 (out/'protocol.json').write_text(json.dumps({'selected':SELECTED,'scales':SCALES,'writer_payload_frozen_before_first_edited_forward':frozen_sha,'source_sha256':{m.__name__:sha(m.__file__) for m in (native,vpd_model,mech,edit_affine,edit_eval)},'runner_sha256':sha(__file__),'panel_sha256':sha(panel),'model_content_sha256':expected},sort_keys=True)+'\n')
 editor=edit_eval.Editor(nat);rows=[]
 for name,c in SELECTED.items():
  ids=p['tasks'][name]['token_ids'];base=tensors[name+'_base'].double();slope=tensors[name+'_slope'].double()
  for k in SCALES:
   z=base+(k-1)*slope;order=z.argsort(descending=True);ideal_top=int(z.argmax());gap=float(z[order[0]]-z[order[1]]);op=edit_eval.scale_op((SITE,None,c),k);nat.vpd.clear();qe=editor.logp(torch.tensor([ids]),[[op]])[0,-1].clone()
   try:
    site.W.copy_(saved['W']+(k-1)*torch.outer(saved['U'][c],saved['V'][:,c]));nat.vpd.clear();q=torch.log_softmax(nat.target.unembed(nat.target.hidden(torch.tensor([ids]))[:,-1:]).float(),-1)[0,0]
    row={'task':name,'unit':c,'k':k,'ideal_affine_top':ideal_top,'ideal_numerator_margin':gap,'near_boundary_margin_1e_4':gap<=1e-4,'literal_top':int(q.argmax()),'editor_top':int(qe.argmax()),'ideal_literal_top_matches':ideal_top==int(q.argmax()),'max_editor_literal_logp_error':float((qe-q).abs().max())};rows.append(row)
    with (out/'verification.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    save_file({'literal_logp':q.contiguous(),'editor_logp':qe.contiguous(),'ideal_numerators':z.contiguous()},str(out/(name+'_'+str(k)+'.safetensors')))
   finally:site.W.copy_(saved['W']);nat.vpd.clear()
 if len(rows)!=18 or any(not torch.equal(getattr(site,key),value) for key,value in saved.items()):raise ValueError('panel/restoration failure')
 if frozen_sha!=sha(out/'writer_payload.json') or {k:sha(v) for k,v in paths.items()}!=expected:raise ValueError('frozen input/assets changed')
 result={'rows':18,'ideal_literal_top_matches':sum(r['ideal_literal_top_matches'] for r in rows),'editor_literal_top_matches':sum(r['editor_top']==r['literal_top'] for r in rows),'near_boundary_rows':sum(r['near_boundary_margin_1e_4'] for r in rows),'max_editor_literal_logp_error':max(r['max_editor_literal_logp_error'] for r in rows),'writer_payload_sha256':frozen_sha,'writer_payload_unchanged':True,'restored_W_U_V_bitwise':True,'seconds':time.monotonic()-started,'scope':'18 fixed FP32 checks do not certify every real scale or boundary; adaptive TRAIN contexts, no English generated'}
 (out/'results.json').write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ('repo','panel','manifest','out'):ap.add_argument('--'+key,type=Path,required=True)
 a=ap.parse_args();run(a.repo,a.panel,a.manifest,a.out)
