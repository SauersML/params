"""Physical tiny-model parity: same seeded interventions, bounded execution rows."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import builder_fast
import carry
import goal_eval
import native_units
from test_goal_eval import tiny_native


def example():
    nat=tiny_native(near=.1)
    ids=torch.randint(0,40,(12,),generator=torch.Generator().manual_seed(31)).tolist()
    ge=goal_eval.GoalEval(nat,changes=2,ranks=(3,4),chunk=64)
    tx=ge.text(ids,[11],seed=17)
    tg=nat.target
    gen=torch.Generator().manual_seed(71)
    means={'heads':.2*torch.randn(tg.n_layer,tg.n_head,tg.hd,generator=gen),
           'neurons':.2*torch.randn(tg.n_layer,tg.site('h.0.mlp.c_fc').W.shape[0],generator=gen)}
    phi=native_units.NativePhi(ge,tx,'search',means=means)
    m=native_units.Mask(torch.rand(phi.L,tx.T,phi.H,generator=gen)>.3,
                        torch.rand(phi.L,tx.T,phi.Dm,generator=gen)>.7)
    return nat,ge,tx,phi,m


@pytest.mark.parametrize('cap',[1,3,5])
def test_nodes_seeded_masks_raw_metrics_and_global_pooling(cap):
    nat,ge,tx,phi,m=example()
    phi.empty()  # exclude the one-row empty diagnostic from the paired draw trace
    original=ge.graph_ref
    def run(chunk):
        ge.chunk=chunk;calls=[]
        def observed(ids,targets,graph,**kw):
            units=kw['units'];h=units['head'][0];n=units['neuron'][0]
            calls.append((h.clone(),n.clone(),graph is phi.vpd))
            assert h.shape[0]<=chunk
            return original(ids,targets,graph,**kw)
        ge.graph_ref=observed
        try:result=phi.nodes(m,draws=7)
        finally:ge.graph_ref=original
        assert len(calls)%3==0
        logical=[]
        for role in range(3):
            part=calls[role::3]
            assert all(vpd==(role==2) for _,_,vpd in part)
            logical.append((torch.cat([h for h,_,_ in part]),torch.cat([n for _,n,_ in part])))
        return result,logical
    large,large_masks=run(64)
    small,small_masks=run(cap)
    for old,new in zip(large_masks,small_masks):
        for a,b in zip(old,new):assert torch.equal(a,b)
    # Checks per-draw order/raw KLs as well as globally pooled scores, including uneven tail.
    assert torch.allclose(torch.tensor(large['draws_nodes']),torch.tensor(small['draws_nodes']),atol=3e-5,rtol=3e-5)
    for key in ('s_nodes','_raw_nodes','kl_nodes_c','kl_nodes_vpd','overstate_nodes'):
        assert small[key]==pytest.approx(large[key],abs=3e-5,rel=3e-5)


@pytest.mark.parametrize('donor_count,cap',[(11,3),(3,7),(3,1)])
def test_resampled_actual_rows_donor_order_and_physical_full_forward(donor_count,cap):
    nat,ge,tx,_,_=example()
    gen=torch.Generator().manual_seed(87)
    donors=[torch.randint(0,40,(7+i%5,),generator=gen).tolist() for i in range(donor_count)]
    qs=[9,1,4];tg=nat.target;original=tg.hidden
    def run(chunk):
        ge.chunk=chunk;rows=[]
        def observed(ids,*a,**kw):
            assert len(ids)<=chunk
            rows.extend(ids.tolist())
            return original(ids,*a,**kw)
        tg.hidden=observed
        try:result=builder_fast._resampled(ge,tx,qs,donors,group=8)
        finally:tg.hidden=original
        return result,rows
    large,old_rows=run(256);small,new_rows=run(cap)
    assert old_rows==new_rows and len(new_rows)==len(qs)*donor_count
    assert list(small)==sorted(qs)
    for q in sorted(qs):
        exact=[]
        for d in donors:
            ids=list(tx.ids);ids[q]=goal_eval._donor_token(d,q,tx.T,tx.ids[q])
            nat.vpd.clear()
            lp=torch.log_softmax(tg.unembed(original(torch.tensor([ids]))[:,tx.targets]).float(),-1)
            exact.append(float(carry.kl(lp,tx.ref)[0]))
        assert small[q]==pytest.approx(sum(exact)/len(exact),abs=3e-5,rel=3e-5)
        assert small[q]==pytest.approx(large[q],abs=3e-5,rel=3e-5)
