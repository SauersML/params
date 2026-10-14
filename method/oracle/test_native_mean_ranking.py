"""Fixed-mean native strengths versus direct Target weight hooks; synthetic only."""
import importlib.util
import sys
from pathlib import Path
import pytest
import torch
from test_goal_eval import tiny_native
import carry

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mean_ranking_overlay',HERE/'native_units.py')
units=importlib.util.module_from_spec(spec);sys.modules[spec.name]=units;spec.loader.exec_module(units)

@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    nat=tiny_native(seed=13,near=.1)
    ids=[1,4,9,2];targets=[3]
    nu=units.NativeUnits(nat,ids,targets)
    gen=torch.Generator().manual_seed(29)
    means={'heads':torch.randn(nu.L,nu.H,nu.hd,generator=gen)/7,
           'neurons':torch.randn(nu.L,nu.Dm,generator=gen)/9}
    return nat,ids,targets,nu,means


def gains(nu,a):
    full=nu.full();return full.heads.float().fill_(a),full.neurons.float().fill_(a)


def direct_hook(fixture,h,n,means):
    nat,ids,targets,nu,_=fixture
    sites=[nat.target.site(name) for name in nat.names]
    assert all(st.in_fn is None and st.mask is None for st in sites)
    try:
        for L in range(nu.L):
            hh=h[L].repeat_interleave(nu.hd,dim=-1)[None]
            nn=n[L][None]
            mh=means['heads'][L].reshape(1,1,-1)
            mn=means['neurons'][L][None,None]
            nat.target.site(f'h.{L}.attn.o_proj').in_fn=lambda x,a=hh,m=mh:a*x+(1-a)*m
            nat.target.site(f'h.{L}.mlp.down_proj').in_fn=lambda x,a=nn,m=mn:a*x+(1-a)*m
        hidden=nat.target.hidden(torch.tensor([ids]))[:,targets]
        return torch.log_softmax((hidden@nat.target.wte.T).float(),-1)
    finally:
        for st in sites:st.in_fn=None


@pytest.mark.parametrize('mode',['full','empty','mixed'])
def test_binary_means_match_direct_dense_weight_execution(fixture,mode):
    nat,ids,targets,nu,means=fixture
    h,n=gains(nu,1. if mode=='full' else 0.)
    if mode=='mixed':
        h.flatten()[::3]=1.;n.flatten()[1::4]=1.
        # Entire earliest position omitted; its nonzero mean still affects later reads.
        h[:,0]=0.;n[:,0]=0.
    nu._forward(nu.full(),[[]],0,keep=True);cache=nu._cache
    got=nu.strength_log_probs(h,n,means=means)
    want=direct_hook(fixture,h,n,means)
    assert torch.allclose(got,want,atol=4e-6,rtol=0)
    assert nu._cache is cache
    if mode=='full':assert torch.equal(got,nu._forward(nu.full(),[[]],0,keep=True))
    if mode=='empty':
        assert not torch.allclose(got,nu.strength_log_probs(h,n),atol=1e-6,rtol=0)


def test_zero_means_are_equivalent_and_default_cache_path_unchanged(fixture):
    _,_,_,nu,means=fixture
    h,n=gains(nu,.63)
    zero={k:torch.zeros_like(v) for k,v in means.items()}
    nu._forward(nu.full(),[[]],0,keep=True);cache=nu._cache;runs=nu.runs
    plain=nu.strength_log_probs(h,n)
    assert torch.equal(plain,nu.strength_log_probs(h,n,means=None))
    assert torch.equal(plain,nu.strength_log_probs(h,n,means=zero))
    assert nu._cache is cache and nu.runs==runs


@pytest.mark.parametrize('kind',['heads','neurons'])
def test_mean_gradient_matches_direct_weight_hook_finite_difference(fixture,kind):
    _,_,_,nu,means=fixture
    h,n=gains(nu,.57);h.requires_grad_();n.requires_grad_()
    loss=carry.kl(nu.ref.detach(),nu.strength_log_probs(h,n,means=means)).sum()
    gh,gn=torch.autograd.grad(loss,(h,n))
    value=h if kind=='heads' else n
    gradient=gh if kind=='heads' else gn
    direction=torch.linspace(-1.,1.,value.numel()).reshape_as(value)
    expected=float((gradient*direction).sum())
    eps=.002;values=[]
    for sign in (-1,1):
        hh,nn=h.detach().clone(),n.detach().clone()
        (hh if kind=='heads' else nn).add_(direction,alpha=sign*eps)
        values.append(float(carry.kl(nu.ref,direct_hook(fixture,hh,nn,means)).sum()))
    actual=(values[1]-values[0])/(2*eps)
    assert abs(expected)>1e-7
    assert actual==pytest.approx(expected,abs=3e-4,rel=.03)


def test_integrated_mean_scores_use_fixed_means_without_model_gradient_or_cache_mutation(fixture):
    _,_,_,nu,means=fixture
    fixed={k:v.clone().requires_grad_() for k,v in means.items()}
    nu._forward(nu.full(),[[]],0,keep=True);cache=nu._cache;runs=nu.runs
    with torch.no_grad():got=nu.integrated_unit_scores(2,means=fixed)
    expected=[torch.zeros_like(got[k]) for k in ('heads','neurons')]
    for a in (.25,.75):
        h,n=gains(nu,a);h.requires_grad_();n.requires_grad_()
        loss=carry.kl(nu.ref,nu.strength_log_probs(h,n,means=means)).sum()
        for dst,gradient in zip(expected,torch.autograd.grad(loss,(h,n))):dst.sub_(gradient.detach()/2)
    assert torch.equal(expected[0],got['heads']) and torch.equal(expected[1],got['neurons'])
    assert nu._cache is cache and nu.runs==runs
    assert all(v.grad is None for v in fixed.values())
    assert all(getattr(w,'grad',None) is None for ws in nu.W.values() for w in ws)
    assert got['forward_evaluations']==got['backward_evaluations']==2


@pytest.mark.parametrize('bad',['missing','shape','integer','double','nan','inf'])
def test_invalid_means_rejected_before_forward_and_cache_unchanged(fixture,monkeypatch,bad):
    _,_,_,nu,means=fixture;h,n=gains(nu,.5);cache=nu._cache
    wrong={k:v.clone() for k,v in means.items()}
    if bad=='missing':del wrong['heads']
    elif bad=='shape':wrong['heads']=wrong['heads'][:1]
    elif bad=='integer':wrong['heads']=wrong['heads'].long()
    elif bad=='double':wrong['heads']=wrong['heads'].double()
    else:wrong['neurons'].flatten()[0]=float(bad)
    monkeypatch.setattr(nu,'_forward_impl',lambda *a,**kw:pytest.fail('invalid means reached execution'))
    with pytest.raises(ValueError):nu.strength_log_probs(h,n,means=wrong)
    assert nu._cache is cache


def test_private_mean_path_rejects_position_cache_shortcut(fixture):
    _,_,_,nu,means=fixture;h,n=gains(nu,.5)
    with pytest.raises(ValueError,match='full-context'):
        nu._forward_impl(nu.full(),[[]],1,keep=False,strengths=(h,n),means=means)
