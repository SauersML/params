"""Fixed Boolean prefix proposals; no model/data access."""
import math
import numpy as np
FRACTIONS=(.001,.002,.005,.01,.02,.05,.1,.2,.35,.5,.7,.9)
def rank(eligible,scores):
 names=sorted(eligible);indices=[];values=[]
 for n in names:
  e=np.asarray(eligible[n],dtype=bool);s=np.asarray(scores[n]);assert e.shape==s.shape
  idx=np.flatnonzero(e.reshape(-1));v=s.reshape(-1)[idx]
  if not np.isfinite(v).all():raise ValueError('nonfinite deletion scores')
  indices.append(idx);values.append(v)
 counts=[len(x) for x in indices];v=np.concatenate(values)
 return names,indices,np.cumsum([0]+counts),np.argsort(v,kind='stable')
def proposal(base,ranking,k):
 names,indices,offset,order=ranking
 if not 0<=k<=len(order):raise ValueError('bad deletion count')
 out={n:np.asarray(m,dtype=bool).copy() for n,m in base.items()}
 take=order[:k]
 for j,n in enumerate(names):
  picked=take[(take>=offset[j])&(take<offset[j+1])]-offset[j]
  out[n].reshape(-1)[indices[j][picked]]=False
 return out
def counts(n):return [math.ceil(f*n) for f in FRACTIONS]
def winner(rows):
 valid=[r for r in rows if len(r['kl'])==4 and all(math.isfinite(k) and 0<=k<=.001 for k in r['kl'])]
 return max(valid,key=lambda r:r['removed'],default=None)
