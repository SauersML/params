import pytest
mx=pytest.importorskip('mlx.core');mx.set_default_device(mx.cpu)
nn=pytest.importorskip('mlx.nn')
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU
from frozen_prefix_loss import frozen_prefix_loss
class GatedDeltaNet(nn.Module):
    def __init__(self):super().__init__();self.weight=mx.array(2.)
    def __call__(self,x):return x*self.weight
class Block(nn.Module):
    def __init__(self,linear):
        super().__init__();self.is_linear=linear
        if linear:self.linear_attn=GatedDeltaNet()
        else:self.self_attn=nn.Linear(1,1,bias=False)
        self.mlp=SwitchGLU(1,2,2)
    def __call__(self,x):return self.linear_attn(x) if self.is_linear else self.self_attn(x)
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.layers=[Block(True),Block(False)];self.freeze();self.layers[-1].self_attn.unfreeze()
    def __call__(self,x):
        for layer in self.layers:x=layer(x)
        return x
base=lambda m,b,l:(m(b).sum(),mx.array(1))
def test_actual_library_shared_activation_old_failure_and_new_fix():
    m=Model();assert m.layers[0].mlp.activation is m.layers[1].mlp.activation
    m.train();m.layers[0].eval()
    assert not m.layers[1].mlp.activation.training # reproduce exact old guard failure
    m.train();before=[(mod,mod.training) for _,mod in m.layers[-1].named_modules()]
    frozen_prefix_loss(base,1)(m,mx.array([[1.]]),None)
    assert not m.layers[0].linear_attn.training
    assert m.layers[0].training and m.layers[0].mlp.activation.training
    assert all(mod.training==mode for mod,mode in before)

def test_tail_gradients_and_weights_unchanged():
    m=Model();m.train();values=[v.tolist() for _,v in tree_flatten(m.parameters())]
    _,g0=nn.value_and_grad(m,base)(m,mx.array([[1.]]),None)
    _,g1=nn.value_and_grad(m,frozen_prefix_loss(base,1))(m,mx.array([[1.]]),None)
    assert [v.tolist() for _,v in tree_flatten(m.parameters())]==values
    assert tree_flatten(g0)[0][1].tolist()==tree_flatten(g1)[0][1].tolist()

def test_validation_modes_and_train_reset():
    m=Model();wrapped=frozen_prefix_loss(base,1);m.eval();wrapped(m,mx.array([[1.]]),None)
    assert all(not mod.training for _,mod in m.named_modules())
    m.train();wrapped(m,mx.array([[1.]]),None)
    assert not m.layers[0].linear_attn.training and m.layers[-1].training

def test_trainable_prefix_rejected_before_changes():
    m=Model();m.layers[0].linear_attn.unfreeze();m.train()
    with pytest.raises(ValueError):frozen_prefix_loss(base,1)(m,mx.array([[1.]]),None)
    assert all(mod.training for _,mod in m.named_modules())
