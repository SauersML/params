"""Opt-in final-down rank-one scale proposals from unedited affine pre-RMS logits.
Only argmax flip/raise goals, bounded[-1,2], exact binary16 factors. No edited forwards/search.
All-position factors are persistent weight-equivalent edits; full/collateral verifier remains authority.
"""
import math
import numpy as np

BOUNDS=(-1.,2.)
MARGIN=1e-4

def _half_candidates(lo,hi,predicate):
 if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:return []
 values=set()
 for x in (lo,hi,1.):
  h=np.float16(min(hi,max(lo,x)))
  values.update(float(y) for y in (h,np.nextafter(h,np.float16(-np.inf)),np.nextafter(h,np.float16(np.inf))))
 return sorted((k for k in values if lo<=k<=hi and predicate(k)),key=lambda k:(abs(k-1.),k))

def propose(base,slope,kind,token,*,bounds=BOUNDS,margin=MARGIN):
 """Closest-to-one binary16 factor with strict full-vocabulary numerator margin."""
 b=np.asarray(base,dtype=np.float64);s=np.asarray(slope,dtype=np.float64);lo,hi=map(float,bounds)
 if b.ndim!=1 or b.shape!=s.shape or len(b)<2 or not np.isfinite(b).all() or not np.isfinite(s).all() or not math.isfinite(lo+hi+margin) or lo>hi or not lo<=1<=hi or margin<=0 or type(token)!=int or not 0<=token<len(b) or kind not in ('raise','flip'):raise ValueError('invalid finite affine argmax contract')
 if (kind=='raise' and int(b.argmax())==token) or (kind=='flip' and int(b.argmax())!=token):return {'k':None,'reason':'baseline already attained','bounds':[lo,hi],'margin':margin}
 other=np.arange(len(b))!=token
 def feasible(k):
  z=b+(k-1.)*s
  return float(z[token]-z[other].max())>margin if kind=='raise' else float(z[other].max()-z[token])>margin
 if kind=='raise':
  d=s[token]-s[other];v=b[token]-b[other]
  if np.any((d==0)&(v<=margin)):return {'k':None,'reason':'constant competitor prevents raise','bounds':[lo,hi],'margin':margin}
  positive=d>0;negative=d<0
  lower=max(lo,float(np.max(1+(margin-v[positive])/d[positive]))) if positive.any() else lo
  upper=min(hi,float(np.min(1+(margin-v[negative])/d[negative]))) if negative.any() else hi
  choices=_half_candidates(lower,upper,feasible);intervals=[[lower,upper]] if lower<=upper else []
 else:
  d=s[other]-s[token];v=b[other]-b[token];positive=d>0;negative=d<0;choices=[];intervals=[]
  if positive.any():
   lower=max(lo,float(np.min(1+(margin-v[positive])/d[positive])))
   if lower<=hi:intervals.append([lower,hi]);choices+=_half_candidates(lower,hi,feasible)
  if negative.any():
   upper=min(hi,float(np.max(1+(margin-v[negative])/d[negative])))
   if lo<=upper:intervals.append([lo,upper]);choices+=_half_candidates(lo,upper,feasible)
 if not choices:return {'k':None,'reason':'no feasible binary16 factor within bounds','intervals':intervals,'bounds':[lo,hi],'margin':margin}
 k=min(choices,key=lambda k:(abs(k-1),k));z=b+(k-1)*s;actual=float(z[token]-z[other].max()) if kind=='raise' else float(z[other].max()-z[token])
 return {'k':k,'predicted_winner':int(z.argmax()),'predicted_numerator_margin':actual,'intervals':intervals,'bounds':[lo,hi],'margin':margin,'scope':'Argmax shared finalRMS cancels; full forward/collateral verification required'}

def coefficients(nat,ids,target,site_name,*,retain=None):
 """Exactly one unedited full forward; retain raw scalar activation, pre-final-RMS logits."""
 import torch,types,vpd_model
 tg=nat.target
 if not isinstance(tg,vpd_model.Target) or target!=len(ids)-1 or site_name!=f'h.{tg.n_layer-1}.mlp.down_proj' or getattr(tg,'behavior',None) is not None:raise ValueError('plain vpd4l tied-RMS model final-down LASTtarget required')
 site=tg.site(site_name);original_norm=tg.norm;captured={};old_cache=site.cache_input
 def norm(this,x,j,fused=False):
  if j==2*this.n_layer:captured['r']=x[0,target].detach().float().clone()
  return original_norm(x,j,fused)
 try:
  nat.vpd.clear();site.cache_input=True;tg.norm=types.MethodType(norm,tg)
  with torch.no_grad():
   tg.hidden(torch.tensor([ids],device=nat.dev));r=captured['r'];activation=(site.last_input[0,target]@site.V).detach().float().clone();base=(r*tg.ln_f.float())@tg.wte.float().T
 finally:tg.norm=original_norm;nat.vpd.clear();site.cache_input=old_cache
 if retain is not None:retain['residual_pre_final_rms']=r.detach().clone()
 return base.detach(),activation

def candidates(ee,task,goal,nodes):
 """Opt-in persistent proposals; original teacher's local grid is untouched."""
 if goal.get('kind') not in ('raise','flip'):return []
 p=task['prompts'][0];t=max(p['target_positions']);site=f'h.{ee.nat.target.n_layer-1}.mlp.down_proj';units=sorted({c for n,pos,c in nodes if n==site and pos==t})
 if not units or t!=len(p['token_ids'])-1:return []
 base,a=coefficients(ee.nat,p['token_ids'],t,site);tg=ee.nat.target;out=[]
 import torch
 with torch.no_grad():
  for c in units:
   slope=a[c]*(tg.site(site).U[c].float()*tg.ln_f.float())@tg.wte.float().T
   result=propose(base.cpu().numpy(),slope.cpu().numpy(),goal['kind'],int(goal['token_id']))
   if result['k'] is not None:out.append({'op':'scale','node':(site,None,c),'k':result['k'],'persistent':True})
 return out
