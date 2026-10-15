from types import SimpleNamespace
import torch
import pytest
from test_goal_eval import tiny_native
import paired_token_qa as bridge
import native_units

def setup():
    nat=tiny_native(seed=1,layers=1,d=2,heads=1,vocab=3,C=2)
    tx=SimpleNamespace(ids=[0,1],targets=[1])
    live=(torch.ones(1,2,1,dtype=torch.bool),torch.ones(1,2,8,dtype=torch.bool))
    means={'heads':torch.zeros(1,1,2),'neurons':torch.zeros(1,8)}
    candidates=[{'kind':'token','ask':ask,'position':0,'new':2} for ask in ('change','kl')]
    return nat,tx,live,means,candidates

def run(nat,tx,live,means,candidates):
    return bridge.measure_native_token_candidates(nat,tx,live,means,candidates,['a','b','c'],fidelity_bits=1e-6,means_provenance={'split':'train','input_names':['synthetic_train']})

def test_physical_full_and_allmean_and_binding():
    nat,tx,live,means,cs=setup()
    # Preseed stale target reference: labels must come from a fresh forward.
    nat._refs[((tuple(tx.ids),), tuple(tx.targets), None)] = torch.tensor([[[0., -100., -100.]]])
    rows=run(nat,tx,live,means,cs)
    assert all(r['fidelity']['passed'] for r in rows)
    empty=tuple(torch.zeros_like(m) for m in live)
    bad=run(nat,tx,empty,means,cs)
    assert all(r['control_status']=='failed_fidelity' for r in bad)
    assert rows[0]['target_model_label']==bad[0]['target_model_label']
    assert not bad[0]['fidelity']['label_agreement']
    assert rows[0]['execution_binding']['sha256']!=bad[0]['execution_binding']['sha256']


@pytest.mark.parametrize('changed', [False, True])
def test_nonzero_mean_physical_hooks_full_empty_mixed(changed):
    nat,tx,live,means,cs=setup()
    means={'heads':torch.tensor([[[.4,-.7]]]),
           'neurons':torch.arange(8,dtype=torch.float32)[None]/7-.3}
    ids=[2,1] if changed else tx.ids
    tg=nat.target
    empty=tuple(torch.zeros_like(m) for m in live)
    mixed=tuple(m.clone() for m in live)
    mixed[0][0,0,0]=False
    mixed[1][0,1,::2]=False
    for masks in (live, empty, mixed):
        mu=native_units.MeanUnits(nat,ids,tx.targets,means)
        expected=mu.logq(tuple(m[None].float() for m in masks))[0,0]
        handles=[]
        for i in range(tg.n_layer):
            hm,nm=masks[0][i],masks[1][i]
            head_fill=means['heads'][i].flatten()
            neuron_fill=means['neurons'][i]
            handles.append(tg.site(f'h.{i}.attn.o_proj').register_forward_pre_hook(
                lambda module,args,hm=hm,fill=head_fill:
                    (torch.where(hm.repeat_interleave(tg.hd,dim=-1),args[0],fill),)))
            handles.append(tg.site(f'h.{i}.mlp.down_proj').register_forward_pre_hook(
                lambda module,args,nm=nm,fill=neuron_fill:
                    (torch.where(nm,args[0],fill),)))
        try:
            actual=tg(torch.tensor([ids]))[0,-1].float().log_softmax(-1)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        finally:
            for h in handles:h.remove()


def test_binding_changes_with_means_and_vocabulary():
    nat,tx,live,means,cs=setup()
    base=run(nat,tx,live,means,cs)[0]['execution_binding']
    different={k:v+.25 for k,v in means.items()}
    changed=run(nat,tx,live,different,cs)[0]['execution_binding']
    renamed=bridge.measure_native_token_candidates(nat,tx,live,means,cs,['A','b','c'],
        fidelity_bits=1e-6,means_provenance={'split':'train','input_names':['synthetic_train']})[0]['execution_binding']
    assert base['sha256']!=changed['sha256']
    assert base['means_sha256']!=changed['means_sha256']
    assert base['sha256']!=renamed['sha256']
    assert base['vocabulary_sha256']!=renamed['vocabulary_sha256']

@pytest.mark.parametrize('damage',['floatmask','means','targets','token'])
def test_malformed_before_forward(damage):
    nat,tx,live,means,cs=setup()
    def fail(*args,**kw):raise AssertionError('unexpected forward')
    nat.reference=fail
    nat.target.hidden=fail
    if damage=='floatmask':live=(live[0].float(),live[1])
    if damage=='means':means['heads'][0,0,0]=float('nan')
    if damage=='targets':tx.targets=[0,1]
    if damage=='token':cs[0]['new']=3
    with pytest.raises(ValueError):run(nat,tx,live,means,cs)
