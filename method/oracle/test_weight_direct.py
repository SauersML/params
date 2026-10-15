"""Direct edited predictions must remain distinguishable from baseline-anchored changes."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_eval
import native
import weight_family
from test_goal_eval import naive, tiny_native


def kl_bits(p, q):
    return float((p.double().exp() * (p.double() - q.double())).sum()) / math.log(2)


def test_correct_relative_change_can_hide_wrong_absolute_prediction():
    p0 = torch.log_softmax(torch.tensor([[2., 0., -1.]], dtype=torch.float64), -1)
    pe = torch.log_softmax(torch.tensor([[0., 2., -1.]], dtype=torch.float64), -1)
    bias = torch.tensor([[0., 0., 8.]], dtype=torch.float64)
    before = torch.log_softmax(p0 + bias, -1)
    after = torch.log_softmax(pe + bias, -1)
    moved = weight_family._moved(p0, before, after)
    assert kl_bits(pe, moved) == pytest.approx(0., abs=1e-12)
    assert kl_bits(pe, after) > 5.
    assert int(pe.argmax()) != int(after.argmax())


@pytest.mark.parametrize('semantics', ['n', 'c'])
def test_direct_weight_diagnostics_match_independent_physical_forward(semantics):
    # near=None builds random factors directly, without any matrix decomposition.
    nat = tiny_native(seed=73, near=None)
    ge = goal_eval.GoalEval(nat, semantics=semantics)
    ids, targets = [1, 8, 3, 11, 4], [4]
    tx = ge.text(ids, targets, seed=1)
    masks = {name: torch.zeros(len(ids), nat.C[name], dtype=torch.bool) for name in nat.names}
    for mask in masks.values():
        mask[:, ::2] = True
    graph = native.Graph(masks=masks)
    edits = [{'kind': 'subcomponent', 'what': ['h.1.mlp.down_proj', 0], 'k': .5},
             {'kind': 'head', 'what': [0, 0], 'k': 0.}]
    def reference(on=None):
        return torch.log_softmax(naive(nat, ids, on=on)[0][targets], -1)
    full_before = reference()
    circuit_before = reference(masks)
    row = weight_family.family(ge, tx, [graph], edits=edits)[0]
    assert row['kl_weight_unedited'] == pytest.approx(kl_bits(full_before, circuit_before), rel=1e-4, abs=1e-4)
    expected = []
    for edit, direct, anchored in zip(edits, row['draws_weight_direct'], row['draws_weight']):
        with weight_family.applied(nat, edit, ge):
            full_after = reference()
            circuit_after = reference(masks)
        expected.append(kl_bits(full_after, circuit_after))
        assert direct[:4] == anchored[:4]
        assert direct[4] == pytest.approx(expected[-1], rel=1e-4, abs=1e-4)
        moved = weight_family._moved(full_before, circuit_before, circuit_after)
        assert anchored[4] == pytest.approx(kl_bits(full_after, moved), rel=1e-4, abs=1e-4)
    assert row['kl_weight_direct'] == pytest.approx(sum(expected) / len(expected), rel=1e-4, abs=1e-4)
    assert row['weight_direct_remainders'] == 'live'
    assert torch.allclose(reference(), full_before, rtol=0., atol=0.)
    empty_panel = weight_family.family(ge, tx, [graph], edits=[])[0]
    assert empty_panel['kl_weight_direct'] is None
    assert empty_panel['draws_weight_direct'] == []


def test_common_panel_native_nonzero_means_matches_physical_hooks():
    import native_units
    nat=tiny_native(seed=73,layers=1,d=4,heads=2,vocab=8,C=2)
    ge=goal_eval.GoalEval(nat,semantics='c')
    ids,targets=[1,2,3],[1,2]
    tx=ge.text(ids,targets,seed=2)
    masks={n:torch.ones(len(ids),nat.C[n],dtype=torch.bool) for n in nat.names}
    tx.vpd=masks
    live=(torch.ones(1,3,2),torch.ones(1,3,16))
    live[0][0,0,0]=0;live[1][0,2,::2]=0
    means={'heads':torch.arange(4,dtype=torch.float32).reshape(1,2,2)/10,
           'neurons':torch.linspace(-.2,.3,16)[None]}
    def execute():
        mu=native_units.MeanUnits(nat,ids,targets,None)
        mu.means=means
        return mu.logq(tuple(m[None] for m in live))
    edits=[{'kind':'head','what':[0,1],'k':.5},
           {'kind':'neuron','what':[0,2],'k':0.}]
    construction=weight_family.edit_id(edits[0])
    panel=weight_family.common_eval(ge,tx,execute,edits,
        metadata={'split':'eval','prior_exposure':'unit-test synthetic'},construction_edit_ids=[construction])
    assert len(panel['worlds'])==3
    assert panel['excluded_from_construction']['count']==1
    assert panel['worlds'][1]['construction_edit']
    assert panel['worlds'][0]['candidate_anchored_kl_bits']==pytest.approx(0.,abs=1e-12)
    assert panel['worlds'][0]['model_change_kl_bits']==0.
    full_before=None;circuit_before=None
    for index,edit in enumerate([None]+edits):
        from contextlib import nullcontext
        with weight_family.applied(nat,edit,ge) if edit is not None else nullcontext():
            p=nat.target.unembed(nat.target.hidden(torch.tensor([ids]))[:,targets])[0].float().log_softmax(-1).double().log_softmax(-1)
            handles=[]
            hm=live[0][0].bool().repeat_interleave(nat.target.hd,dim=-1)
            nm=live[1][0].bool()
            handles.append(nat.target.site('h.0.attn.o_proj').register_forward_pre_hook(
                lambda module,args:(torch.where(hm,args[0],means['heads'][0].flatten()),)))
            handles.append(nat.target.site('h.0.mlp.down_proj').register_forward_pre_hook(
                lambda module,args:(torch.where(nm,args[0],means['neurons'][0]),)))
            try:q=nat.target.unembed(nat.target.hidden(torch.tensor([ids]))[:,targets])[0].float().log_softmax(-1).double().log_softmax(-1)
            finally:
                for handle in handles:handle.remove()
        if index==0:full_before,circuit_before=p,q
        expected=max(0.,kl_bits(p,q)/len(targets))
        row=panel['worlds'][index]
        assert row['candidate_direct_kl_bits']==pytest.approx(expected,abs=1e-6)
        assert row['vpd_direct_kl_bits']==pytest.approx(0.,abs=1e-6)
        moved=weight_family._moved(full_before,circuit_before,q)
        assert row['candidate_anchored_kl_bits']==pytest.approx(max(0.,kl_bits(p,moved)/len(targets)),abs=1e-6)
        assert row['candidate_top']==q.argmax(-1).tolist()
    repeat=weight_family.common_eval(ge,tx,execute,edits,metadata={'split':'eval','prior_exposure':'unit-test synthetic'},construction_edit_ids=[construction])
    assert panel==repeat


def test_common_panel_exception_restores_weights_and_invalid_exclusions_precheck():
    nat=tiny_native(seed=2,layers=1,d=4,heads=2,vocab=8,C=2)
    ge=goal_eval.GoalEval(nat,semantics='c');tx=ge.text([1,2],[1],seed=3)
    edit={'kind':'neuron','what':[0,2],'k':0.}
    saved={name:{key:getattr(nat.target.site(name),key).clone() for key in ('W','U','V','delta_T')} for name in nat.names}
    calls=[]
    def candidate():
        calls.append(True)
        if len(calls)==2:raise RuntimeError('candidate execution failed')
        return tx.ref
    for exclusion in (['malformed'],[weight_family.edit_id(edit)]*2):
        with pytest.raises(ValueError):weight_family.common_eval(ge,tx,candidate,[edit],metadata={},construction_edit_ids=exclusion)
    assert not calls
    with pytest.raises(RuntimeError,match='candidate execution failed'):
        weight_family.common_eval(ge,tx,candidate,[edit],metadata={})
    for name,values in saved.items():
        for key,value in values.items():assert torch.equal(getattr(nat.target.site(name),key),value)
    assert not nat._refs and not ge._xk


def test_panel_identity_is_common_bank_and_scale_canonical():
    nat=tiny_native(seed=3,layers=1,d=4,heads=2,vocab=8,C=2)
    ge=goal_eval.GoalEval(nat,semantics='c');tx=ge.text([1,2],[1],seed=4)
    edit={'kind':'neuron','what':[0,2],'k':0}
    equivalent={'kind':'neuron','what':(0,2),'k':-0.0}
    assert weight_family.edit_id(edit)==weight_family.edit_id(equivalent)
    execute=lambda:tx.ref
    left=weight_family.common_eval(ge,tx,execute,[edit],metadata={'candidate':'a'})
    outside=weight_family.edit_id({'kind':'head','what':[0,1],'k':.5})
    right=weight_family.common_eval(ge,tx,execute,[equivalent],metadata={'candidate':'b'},
        construction_edit_ids=[outside,weight_family.edit_id(edit)])
    assert left['panel_id']==right['panel_id']
    assert right['construction_exposure']['intersection_count']==1
    assert right['construction_exposure']['unmatched_recorded_edit_ids']==[outside]
    assert right['excluded_from_construction']['count']==0
    disjoint=weight_family.common_eval(ge,tx,execute,[edit],metadata={},construction_edit_ids=[outside])
    assert disjoint['excluded_from_construction']['count']==1
    assert disjoint['construction_exposure']['intersection_count']==0
    with pytest.raises(ValueError,match='duplicate literal'):
        weight_family.common_eval(ge,tx,execute,[edit,equivalent],metadata={})


@pytest.mark.parametrize('semantics', ['n', 'c'])
def test_set_backed_complete_weight_runs_match_masks_and_physical_forward(semantics):
    nat=tiny_native(seed=17,layers=1,d=4,heads=2,vocab=8,C=3)
    ge=goal_eval.GoalEval(nat,semantics=semantics)
    ids,targets=[1,2,3],[2]
    tx=ge.text(ids,targets,seed=9)
    masks={name:torch.zeros(len(ids),nat.C[name],dtype=torch.bool) for name in nat.names}
    for mask in masks.values():mask[1:,::2]=True
    set_graph=native.Graph([(name,t,c) for name,mask in masks.items() for t,c in mask.nonzero().tolist()],complete=True)
    mask_graph=native.Graph(masks=masks)
    edit={'kind':'subcomponent','what':['h.0.mlp.down_proj',0],'k':2.}
    before={key:getattr(nat.target.site(edit['what'][0]),key).clone() for key in ('W','U','V','delta_T')}
    from contextlib import nullcontext
    for change in (None,edit):
        with weight_family.applied(nat,change,ge) if change is not None else nullcontext():
            values=weight_family._lqs(ge,ids,targets,[set_graph,mask_graph])
            expected=torch.log_softmax(naive(nat,ids,on=masks)[0][targets],-1)
            torch.testing.assert_close(values[0],values[1],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(values[0],expected,atol=1e-5,rtol=1e-5)
    # Exercise the scorecard's real common_weights path, including bank creation,
    # so loaded complete candidates cannot silently become an empty engine plan.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/oracle'))
    import circuit_scorecard as sc
    left=sc.common_weights(ge,tx,graph=set_graph)
    right=sc.common_weights(ge,tx,graph=mask_graph)
    assert left['panel_id']==right['panel_id']
    assert left['worlds']==right['worlds']
    assert len(left['worlds'])==1+2*len(nat.names)
    baseline=torch.log_softmax(naive(nat,ids,on=masks)[0][targets],-1).double().log_softmax(-1)
    assert left['worlds'][0]['candidate_direct_kl_bits']==pytest.approx(
        max(0.,kl_bits(tx.ref[0].double().log_softmax(-1),baseline)),abs=1e-5)
    for key,value in before.items():assert torch.equal(getattr(nat.target.site(edit['what'][0]),key),value)
