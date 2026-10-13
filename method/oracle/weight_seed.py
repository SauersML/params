"""Bounded static-weight edge proposal, never an authoritative teacher label.

Rank actual loaded component directions, then require the ordinary measured KL
budget and an acceptance callback whose floors were captured at the full start.
No activations, reference outputs or intervention outcomes enter the ranking.
"""
from __future__ import annotations
import hashlib
import math

MAX_NODES=128
RESIDUAL_K=32

def residual_route(edge):
    writer,reader=edge
    return reader is not None and writer[0].split('.')[-1] in ('o_proj','down_proj') and reader[0].split('.')[-1] in ('v_proj','c_fc')

def rank_loaded(nat, edges):
    """Same signed norm-gain cosine as connectome, from this Native's basis.

    Input-dependent RMS denominators and nonlinear/routing effects are omitted;
    signs are retained, while ordering uses magnitude. Zero directions score zero.
    """
    if len({n for edge in edges for n in edge if n is not None}) > MAX_NODES:
        raise ValueError('loaded ranking exceeds 128-node cap')
    import torch
    vectors={}; fingerprints={}
    def vector(node,reader):
        name,position,index=node
        key=(name,index,reader)
        if key not in vectors:
            site=nat.target.site(name)
            if reader:
                layer=int(name.split('.')[1]);kind=name.split('.')[-1]
                x=site.V[:,index].detach().float()*nat.target.norms[2*layer+int(kind=='c_fc')].detach().float()
            else: x=site.U[index].detach().float()
            if not bool(torch.isfinite(x).all()): raise ValueError('nonfinite loaded component direction')
            vectors[key]=x
            fingerprints[str(key)]=hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()
        return vectors[key]
    ranked=[]
    with torch.no_grad():
        for writer,reader in edges:
            if not residual_route((writer,reader)): continue
            a,b=vector(writer,False),vector(reader,True)
            if a.shape!=b.shape: raise ValueError('coupling direction shape mismatch')
            raw=float(a@b);den=float(a.norm()*b.norm());cos=raw/den if den else 0.
            if not math.isfinite(cos): raise ValueError('nonfinite static cosine')
            ranked.append({'writer':writer,'reader':reader,'cosine':cos,'raw':raw})
    return ranked,fingerprints

def propose(nodes, edges, ranked, protected=None):
    """Return a candidate mask+provenance, or skip without altering full start.

    All Q/K dependency, attention V->O, MLP input->output, and target-output links
    stay on. Only residual writes into V/MLP readers compete for at most K32 slots.
    Existing protection also stays on and consumes that residual budget.
    """
    if len(nodes)>MAX_NODES: return None,{'status':'skipped','reason':'node_cap','nodes':len(nodes)}
    if len(set(nodes))!=len(nodes) or any(len(n)!=3 or n[2]<0 for n in nodes):
        raise ValueError('distinct expanded scalar nodes required')
    node_set=set(nodes)
    if len(set(edges))!=len(edges) or any(w not in node_set or (r is not None and r not in node_set) for w,r in edges):
        raise ValueError('distinct edges among existing nodes required')
    protected=[False]*len(edges) if protected is None else list(protected)
    if len(protected)!=len(edges) or any(type(x) is not bool for x in protected): raise ValueError('bool protection mask required')
    index={edge:i for i,edge in enumerate(edges)};scores={}
    for row in ranked:
        edge=(tuple(row['writer']),tuple(row['reader']))
        if edge not in index or not residual_route(edge) or edge in scores:
            raise ValueError('unique existing residual coupling required')
        if not math.isfinite(row['cosine']) or not math.isfinite(row.get('raw',0.)):
            raise ValueError('finite signed static coupling required')
        scores[edge]=row
    mask=[not residual_route(edge) or protected[i] for i,edge in enumerate(edges)]
    forced=sum(keep and residual_route(e) for keep,e in zip(mask,edges))
    if forced>RESIDUAL_K: return None,{'status':'skipped','reason':'protected_residual_cap','protected':forced}
    order=sorted((edge for edge,row in scores.items() if row['cosine']!=0 and not mask[index[edge]]),key=lambda e:(-abs(scores[e]['cosine']),e))
    for edge in order[:RESIDUAL_K-forced]: mask[index[edge]]=True
    chosen=[scores[e] for i,e in enumerate(edges) if mask[i] and e in scores]
    return tuple(mask),{'status':'proposed','nodes':len(nodes),'all_edges':len(edges),'kept_edges':sum(mask),
                        'mandatory_edges':sum(not residual_route(e) for e in edges),'residual_kept':sum(k and residual_route(e) for k,e in zip(mask,edges)),
                        'signed_couplings':chosen,'scope':'static proposal; no causal or circuit-fidelity certificate'}

def assess(full_mask, proposal_mask, budget, measure, admit=None):
    """Measured gate with unchanged full-start fallback and fixed original budget.

    Caller MUST create admit at the full start BEFORE calling this helper. The
    callback receives the candidate only; this helper never constructs new floors.
    It never mutates either input or substitutes a weak seed's KL for the budget.
    """
    full=tuple(full_mask);candidate=tuple(proposal_mask)
    if len(full)!=len(candidate) or any(type(x) is not bool for x in full+candidate): raise ValueError('matched bool masks required')
    if any(k and not old for old,k in zip(full,candidate)): raise ValueError('proposal cannot add absent connections')
    if not math.isfinite(budget) or budget<0: raise ValueError('finite nonnegative original budget required')
    kl=float(measure(candidate))
    record={'original_budget':budget,'proposal_kl':kl,'admitted':False}
    if not math.isfinite(kl) or kl>budget:
        return full,{**record,'reason':'original_kl_budget'}
    terms=admit(candidate) if admit is not None else {}
    if terms is None:
        return full,{**record,'reason':'original_acceptance_floors'}
    return candidate,{**record,'admitted':True,'reason':'measured_admission','acceptance':terms}

def initialise(ep, nodes, budget, admit=None, reference_terms=None, full_kl=None, blocks=False):
    """Try one proposal after the caller captured ordinary full-start gates.

    Unchanged fallback preserves ep.on object identity. Existing repair-subsets
    must not call this helper; expanded block starts are conservatively skipped.
    """
    import copy
    import time
    if blocks: return {'status':'skipped','reason':'blocks'}
    if len(nodes)>MAX_NODES: return {'status':'skipped','reason':'node_cap','nodes':len(nodes)}
    if not bool(ep.on.all()): return {'status':'skipped','reason':'nonfull_start'}
    import torch
    t0=time.perf_counter();before=ep.measured
    ranked,fingerprints=rank_loaded(ep.ge.nat,ep.edges)
    candidate,meta=propose(nodes,ep.edges,ranked,ep.protected.tolist())
    meta.update(basis_fingerprints=fingerprints,original_reference_terms=copy.deepcopy(reference_terms or {}),full_start_kl=full_kl)
    if candidate is None: return {**meta,'seconds':time.perf_counter()-t0,'measured_subsets':ep.measured-before}
    tensor=lambda mask:torch.tensor(mask,dtype=torch.bool,device=ep.on.device)[None]
    def acceptance(mask):
        got=admit(tensor(mask))
        if len(got)!=1: raise ValueError('one proposal acceptance result required')
        return got[0]
    chosen,gate=assess(ep.on.tolist(),candidate,budget,
                      lambda mask:ep.kl_sets(tensor(mask))[0],acceptance if admit is not None else None)
    if gate['admitted']: ep.on=tensor(chosen)[0]
    return {**meta,**gate,'seconds':time.perf_counter()-t0,'measured_subsets':ep.measured-before}
