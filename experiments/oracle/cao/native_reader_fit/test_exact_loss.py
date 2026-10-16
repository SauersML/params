"""Actual installed batcher/CE regression. CPU device, no target/model load."""
import pytest
np=pytest.importorskip("numpy")
mx=pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)
from mlx_lm.tuner.trainer import iterate_batches,default_loss
from exact_loss import exact_completion_loss

class Fake:
    def __init__(self,logits):self.logits=mx.array(logits)
    def __call__(self,inputs):return self.logits

def fixture():
    # Prefix3, completion2 (last token is synthetic EOS5).
    batch,lengths=next(iterate_batches([([1,2,3,4,5],3)],batch_size=1,max_seq_length=64))
    logits=np.zeros((1,batch.shape[1]-1,8),dtype=np.float32)
    return batch,lengths,logits

def test_actual_batcher_exposes_installed_pad_error():
    b,l,z=fixture()
    assert b.shape[1]>5
    _,legacy=default_loss(Fake(z),b,l)
    loss,ntoks=exact_completion_loss(Fake(z),b,l)
    assert legacy.item()==3 and ntoks.item()==2
    assert np.isclose(loss.item(),np.log(8))

def test_prompt_and_all_padding_logits_cannot_change_loss():
    b,l,z=fixture();reference=exact_completion_loss(Fake(z),b,l)[0].item()
    for indices in ([0,1],list(range(4,z.shape[1]))):
        changed=z.copy();changed[:,indices,0]=50
        actual,n=exact_completion_loss(Fake(changed),b,l)
        assert n.item()==2 and actual.item()==reference

def test_first_completion_and_eos_each_change_actual_loss():
    b,l,z=fixture();reference=exact_completion_loss(Fake(z),b,l)[0].item()
    for index,target in [(2,4),(3,5)]:
        changed=z.copy();changed[:,index,target]=50
        actual,n=exact_completion_loss(Fake(changed),b,l)
        assert n.item()==2 and actual.item()<reference-.5
