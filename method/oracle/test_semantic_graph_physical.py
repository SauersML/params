"""Tiny synthetic Engine regression: carrier identity alone cannot certify wiring."""
from types import SimpleNamespace
import sys
from pathlib import Path
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'vpd')]
import native
import goal_eval
import semantic
import variables
from test_goal_eval import tiny_native


def test_same_carriers_broken_legal_edge_falsifies_transplant():
    # Model-derived test fixture only: no oracle/donor corpus or model generation.
    nat=tiny_native(seed=0,layers=1,d=2,heads=1,vocab=3,C=20)
    with torch.no_grad():
        nat.target.wte.copy_(torch.tensor([[1.,0.],[0.,1.],[0.,.5]]))
        nat.target.ln_f.fill_(1.)
        for norm in nat.target.norms:norm.fill_(1.)
        for name in nat.names:
            site=nat.target.site(name);site.U.zero_();site.V.zero_();site.W.zero_();site.delta_T.zero_()
        fc=nat.target.site('h.0.mlp.c_fc');down=nat.target.site('h.0.mlp.down_proj')
        for c,j in [(7,0),(19,1)]:
            fc.V[1,c]=1.;fc.U[c,j]=1.
            down.V[j,c]=1.;down.U[c,1]=.45
        for name in nat.names:
            site=nat.target.site(name);site.W.copy_((site.V@site.U).T)
    nat._refs.clear();nat._refs_bytes=0;nat.vpd.clear()
    a=('h.0.mlp.c_fc',0,7);b=('h.0.mlp.c_fc',0,19)
    x=('h.0.mlp.down_proj',0,7);y=('h.0.mlp.down_proj',0,19)
    nodes={a,b,x,y};parents={a:[],b:[],x:[a],y:[b]}
    intact=native.Graph(nodes,parents=parents,out=[x,y])
    broken=native.Graph(nodes,parents={a:[],b:[],x:[a],y:[]},out=[x,y])
    for g in (intact,broken):
        g.tok_positions=[0];g.tok_edges=[(0,a),(0,b)];g.tok_out=[0]
    assert intact.node_set()==broken.node_set()
    assert native.all_edges(sorted(nodes),[0])[0][x]  # real legal architecture, not same-site fake edge
    tx=SimpleNamespace(ids=[0],targets=[0],seed=0)
    vs=[{'name':'V1','group':[a],'inputs':[0]},{'name':'V2','group':[b],'inputs':[0]}]
    intervention={'transplants':[('V1',{0:1}),('V2',{0:2})]}
    results=[]
    for graph in (None,intact,broken):
        ck=semantic.Checker(variables.Scorer(goal_eval.GoalEval(nat,semantics='c')),tx,vs,graph=graph)
        try:
            if graph is not None:assert ck.base['col']['h.0.mlp.c_fc']=={7:0,19:1}
            q=ck.run(intervention)
            assert torch.isfinite(q).all()
            results.append(ck.outcome(q,('top',1)))
        finally:ck.close()
    assert all(r['holds'] and r['top']==1 for r in results[:2])
    assert not results[2]['holds'] and results[2]['top']==0
    assert parents[y]==[b]  # original graph was never mutated
