import numpy as np
import pytest
import torch
from write_lens import signed_write_lens


def fixture(a):
    column=torch.tensor([2.,-3.,.5]);scale=torch.tensor([.5,2.,-1.])
    U=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,1.,2.],[3.,-2.,1.],[2.,1.,-3.]])
    return torch.tensor(a),column,scale,U


@pytest.mark.parametrize('activation',[-2.,2.,0.])
def test_signed_targets_match_independent_numpy_projection(activation):
    a,w,s,U=fixture(activation)
    result=signed_write_lens(a,w,s,U,captured_write=a*w)
    expected=U.numpy().dot(s.numpy()*(activation*w.numpy()))
    ids=np.argsort(-expected,kind='stable')[:5].tolist() if activation else []
    assert result['top_token_ids']==ids
    assert result['zero_write']==(activation==0)
    assert result['cat']=='write_lens' and result['capture_max_abs_error']==0
    assert (result['zero_answer'] is not None)==(activation==0)


def test_negative_pole_reverses_positive_projection_not_native_vector():
    a,w,s,U=fixture(2.)
    positive=signed_write_lens(a,w,s,U,captured_write=a*w)
    negative=signed_write_lens(-a,w,s,U,captured_write=-a*w)
    assert positive['top_token_ids']!=negative['top_token_ids']
    expected=U.numpy().dot(s.numpy()*w.numpy())
    assert negative['top_token_ids']==np.argsort(expected,kind='stable')[:5].tolist()
    with pytest.raises(ValueError,match='disagrees'):
        signed_write_lens(-a,w,s,U,captured_write=a*w)


@pytest.mark.parametrize('bad',['activation','scale','capture','unembedding'])
def test_fail_closed_nonfinite_and_no_broadcast(bad):
    a,w,s,U=fixture(1.);actual=a*w
    if bad=='activation':a=torch.tensor(float('nan'))
    if bad=='scale':s[0]=float('inf')
    if bad=='capture':actual=actual[None]
    if bad=='unembedding':U[0,0]=float('nan')
    with pytest.raises(ValueError):signed_write_lens(a,w,s,U,captured_write=actual)


def test_zero_signed_write_does_not_hide_invalid_unembedding():
    a,w,s,U=fixture(0.);U[0,0]=float('inf')
    with pytest.raises(ValueError):signed_write_lens(a,w,s,U,captured_write=a*w)


def test_pure_captured_vector_entry_matches_checked_native_formula():
    from write_lens import write_lens
    a,w,s,U=fixture(-2.)
    actual=a*w
    pure=write_lens(actual,s,U)
    checked=signed_write_lens(a,w,s,U,captured_write=actual)
    assert pure['top_token_ids']==checked['top_token_ids']
    assert 'activation' not in pure and 'not independently checked' in pure['capture_consistency']


def test_sparse_effective_linear_slices_and_signed_contrast_independent_parity():
    import math
    from write_lens import effective_linear_slice,signed_write_contrast
    torch.manual_seed(3)
    module=torch.nn.Linear(4,3,bias=True).eval().requires_grad_(False)
    columns=effective_linear_slice(module,[3,1])
    rows=effective_linear_slice(module,[2,0],rows=True)
    basis=torch.eye(4)
    actual=(module(basis)-module(torch.zeros(1,4))).T
    assert torch.allclose(columns,actual[:,[3,1]],atol=1e-7)
    assert torch.allclose(rows,actual[[2,0]],atol=1e-7)
    scale=torch.tensor([.5,2.,-1.]);U=torch.tensor([[1.,2.,3.],[-2.,1.,0.]])
    for activation in (-2.,0.,3.):
        a=torch.tensor(activation);w=columns[:,0]
        actual_write=module(basis[3]*activation)-module(torch.zeros(4))
        assert torch.allclose(a*w,actual_write,atol=1e-6)
        got=signed_write_contrast(a,w,scale,U,rms_denominator=2.)
        direct=float((U[0]-U[1]) @ (scale*actual_write))/2.
        assert got['static_signed_contrast_bits']==pytest.approx(direct/math.log(2),abs=1e-6)
        assert got['static_signed_logit_contrast']==pytest.approx(direct,abs=1e-6)
        assert got['write_l2']==pytest.approx(float(actual_write.norm()),abs=1e-6)
        assert np.sign(got['write_l2_with_activation_sign'])==np.sign(activation)


def test_sparse_peft_effective_slices_active_disabled_merged_and_unsupported():
    from write_lens import effective_linear_slice
    import peft
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(4)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=11,hidden_size=4,intermediate_size=4,
        num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=4))
    model=peft.get_peft_model(model,peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
        target_modules=['down_proj'],task_type='CAUSAL_LM')).eval()
    module=model.get_base_model().model.layers[0].mlp.down_proj
    with torch.no_grad():module.lora_A['default'].weight.fill_(.2);module.lora_B['default'].weight.fill_(.3)
    def compare():
        actual=(module(torch.eye(4))-module(torch.zeros(1,4))).T
        assert torch.allclose(effective_linear_slice(module,[1,3]),actual[:,[1,3]],atol=1e-6)
        assert torch.allclose(effective_linear_slice(module,[0,2],rows=True),actual[[0,2]],atol=1e-6)
    compare()
    module.enable_adapters(False);compare();module.enable_adapters(True)
    # Two simultaneously active standard adapters must both enter the sparse slice.
    model.add_adapter('second',peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
        target_modules=['down_proj'],task_type='CAUSAL_LM'))
    with torch.no_grad():module.lora_A['second'].weight.fill_(-.1);module.lora_B['second'].weight.fill_(.4)
    module.set_adapter(['default','second']);model.eval();compare()
    module.merge();compare();module.unmerge()
    module.use_dora['default']=True
    with pytest.raises(ValueError,match='DoRA'):effective_linear_slice(module,[0])


@pytest.mark.parametrize('denominator',[0.,-1.,float('nan'),float('inf')])
def test_signed_contrast_rejects_invalid_denominator(denominator):
    from write_lens import signed_write_contrast
    a,w,s,U=fixture(1.)
    with pytest.raises(ValueError):signed_write_contrast(a,w,s,U[:2],rms_denominator=denominator)
