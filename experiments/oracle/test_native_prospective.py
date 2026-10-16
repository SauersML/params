import sys
from pathlib import Path
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'method' / 'oracle'))
from neuron_builder import Neurons
from native_prospective import Collector
IDENTITY=dict(model="random_qwen",revision="seed7",tokenizer_sha256="test_ids",adapter_sha256="none")

@pytest.fixture
def setup():
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8)).eval()
    return model, Collector(Neurons(model, None), 100, IDENTITY)

def direct(model, prompt, answer):
    ids=torch.tensor([prompt+answer])
    with torch.no_grad():
        lp=model(ids,use_cache=False).logits[0,len(prompt)-1:len(prompt)+len(answer)-1].float().log_softmax(-1)
    return float(lp.gather(-1,torch.tensor(answer)[:,None]).sum())

def hooks(model):
    return sum(len(m._forward_pre_hooks)+len(m._forward_hooks) for m in model.modules())

def test_full_answer_and_noop(setup):
    model,c=setup
    p=[1,2,3]; answers=[[4,5],[6]]
    source={}; units=[(0,1,4)]
    s=c.score(p,answers,units=units,snapshot=source)
    assert s['a_logp_nats']==direct(model,p,answers[0])
    assert s['b_logp_nats']==direct(model,p,answers[1])
    assert c.score(p,answers,units=units,source=source)==s
    assert c.forward_examples==4 and hooks(model)==0

def test_exact_cells_and_counts(setup):
    model,c=setup
    baseline=[]; edited=[]
    h=model.model.layers[0].mlp.down_proj.register_forward_pre_hook(lambda m,a: baseline.append(a[0].detach().clone()))
    c.score([1,2,3],[[4,5],[6]])
    h.remove()
    h=model.model.layers[0].mlp.down_proj.register_forward_pre_hook(lambda m,a: edited.append(a[0].detach().clone()))
    # Existing observer executes before edit hook, so inspect via wrapped engine edit.
    h.remove()
    seen=[]; original=c.nr.hooks
    import contextlib
    @contextlib.contextmanager
    def observe(edit=None, **kwargs):
        def wrapper(l,x):
            out=edit(l,x)
            if l==0: seen.append((x.clone(),out.clone()))
            return out
        with original(wrapper,**kwargs): yield
    c.nr.hooks=observe
    c.score([1,2,3],[[4,5],[6]],units=[(0,1,4)],zero=True)
    for x,y in seen:
        expected=x.clone(); expected[:,1,4]=0
        assert torch.equal(y,expected)
    assert hooks(model)==0
    seen.clear()
    c.score([1,2,3],[[4,5],[6]],units=[(0,1,4)],source={(0,1,4):torch.tensor(2.5)})
    for x,y in seen:
        expected=x.clone(); expected[:,1,4]=2.5
        assert torch.equal(y,expected)
    c.nr.hooks=original
    actual=[]
    counter=model.register_forward_pre_hook(lambda m,a: actual.append(1))
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=c.collect(case_id='tiny',clean_ids=[1,2,3],corrupt_ids=[1,7,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    counter.remove()
    assert result['forward_examples']==len(actual)==12
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert hooks(model)==0

def test_cleanup_failure_and_validation(setup):
    model,c=setup
    def fail(*a,**k): raise RuntimeError('injected')
    old=model.forward; model.forward=fail
    with pytest.raises(RuntimeError,match='injected'): c.score([1,2],[[3],[4]])
    model.forward=old
    assert c.forward_examples==1 and hooks(model)==0
    c.nr.thr=torch.zeros(2,24)
    with pytest.raises(ValueError,match='original model'): c.score([1],[[2],[3]])
    c.nr.thr=None
    with pytest.raises(ValueError,match='outside'): c.score([1],[[2],[3]],units=[(0,1,0)])
    c.limit=c.forward_examples
    with pytest.raises(RuntimeError,match='budget'): c.score([1],[[2],[3]])
    assert hooks(model)==0

def test_public_before_gold_resume(setup):
    model,c=setup
    case=dict(case_id='staged',clean_ids=[1,2,3],corrupt_ids=[1,7,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    public=c.prepare_public(**case)
    assert c.forward_examples==2
    assert set(public)==set(case)|{'clean_native_values'}
    assert 'contrast_bits' not in str(public)
    changed=dict(case, corrupt_ids=[1,8,3])
    with pytest.raises(ValueError,match='changed'): c.collect(**changed)
    assert c.forward_examples==2
    result=c.collect(**case)
    assert c.forward_examples==12 and result['forward_examples']==10
    assert hooks(model)==0

def test_mutation_budget_and_degenerate_contrast(setup):
    model,c=setup
    c.limit=1
    case=dict(case_id='budget',clean_ids=[1,2],corrupt_ids=[1,3],
        answer_a_ids=[4],answer_b_ids=[5],target_units=[(0,1,4)],control_units=[(0,1,5)])
    with pytest.raises(RuntimeError,match='budget'): c.prepare_public(**case)
    assert c.forward_examples==0 and hooks(model)==0
    c.limit=100
    with pytest.raises(ValueError,match='distinct'): c.score([1],[[2],[2]])
    assert c.forward_examples==0
    with torch.no_grad(): next(model.parameters()).add_(0.01)
    with pytest.raises(RuntimeError,match='parameter identity/version'): c.score([1],[[2],[3]])
    assert c.forward_examples==0


def test_peft_original_organism_and_adapter_state(setup):
    peft=pytest.importorskip('peft')
    model,_=setup
    config=peft.LoraConfig(r=2,lora_alpha=4,target_modules=['down_proj'],task_type='CAUSAL_LM')
    model=peft.get_peft_model(model,config)
    model.add_adapter('alternate',config)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_B.default' in n: p.fill_(0.15)
    model.eval()
    c=Collector(Neurons(model,None),100,frozen_target_identity=dict(IDENTITY,adapter_sha256='tiny_default'))
    score=c.score([1,2],[[3,4],[5]])
    assert score['a_logp_nats']==direct(model,[1,2],[3,4])
    model.set_adapter('alternate')
    model.eval()
    with pytest.raises(RuntimeError,match='adapter activation'): c.score([1,2],[[3,4],[5]])
    assert c.forward_examples==2 and hooks(model)==0
    model.set_adapter('default'); model.eval()
    c.score([1,2],[[3,4],[5]])
    with model.disable_adapter():
        with pytest.raises(RuntimeError,match='adapter activation'): c.score([1,2],[[3,4],[5]])
    assert c.forward_examples==4
