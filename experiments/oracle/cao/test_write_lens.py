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
