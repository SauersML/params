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
    assert set(public)==set(case)|{'clean_native_values','clean_native_regimes'}
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


def test_paired_noising_manual_hooks_and_counts(setup):
    model,c=setup
    case=dict(case_id='pair',clean_ids=[1,2,3],corrupt_ids=[1,7,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    actual=[]
    counter=model.register_forward_pre_hook(lambda m,a: actual.append(1))
    c.prepare_public(**case)
    result=c.collect(**case,paired_noising=True)
    counter.remove()
    assert len(actual)==c.forward_examples==16
    assert result['forward_examples']==14
    assert set(result['interventions'])=={'target_zero','target_restore','control_zero','control_restore'}
    # Independent direct hooks: capture corrupt donor, patch clean, score complete A/B.
    donor={}
    def capture(m,args): donor['x']=args[0][0,1,4].detach().clone()
    h=model.model.layers[0].mlp.down_proj.register_forward_pre_hook(capture)
    direct(model,case['corrupt_ids'],case['answer_a_ids'])
    h.remove()
    def patch(m,args):
        x=args[0].clone();x[:,1,4]=donor['x'];return (x,)+args[1:]
    h=model.model.layers[0].mlp.down_proj.register_forward_pre_hook(patch)
    try:
        a=direct(model,case['clean_ids'],case['answer_a_ids'])
        b=direct(model,case['clean_ids'],case['answer_b_ids'])
    finally: h.remove()
    row=result['paired_patching']['target']
    assert row['noising']['a_logp_nats']==a
    assert row['noising']['b_logp_nats']==b
    assert row['contrast_gap_bits']==result['clean']['contrast_bits']-result['corrupt']['contrast_bits']
    assert row['noising']['delta_bits']==-row['lost_bits']
    h=model.model.layers[0].mlp.down_proj.register_forward_pre_hook(capture)
    direct(model,case['clean_ids'],case['answer_a_ids'])
    h.remove()
    h=model.model.layers[0].mlp.down_proj.register_forward_pre_hook(patch)
    try:
        restored_a=direct(model,case['corrupt_ids'],case['answer_a_ids'])
        restored_b=direct(model,case['corrupt_ids'],case['answer_b_ids'])
    finally: h.remove()
    assert row['denoising']['a_logp_nats']==restored_a
    assert row['denoising']['b_logp_nats']==restored_b
    assert hooks(model)==0


def test_paired_zero_gap_and_budget(setup):
    model,c=setup
    case=dict(case_id='null',clean_ids=[1,2,3],corrupt_ids=[1,2,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    c.limit=15
    with pytest.raises(RuntimeError,match='budget'): c.collect(**case,paired_noising=True)
    assert c.forward_examples==0
    c.limit=16
    result=c.collect(**case,paired_noising=True)
    assert c.forward_examples==16
    for row in result['paired_patching'].values():
        assert row['contrast_gap_bits']==row['lost_bits']==row['restored_bits']==0
        assert row['lost_share'] is None and row['restored_share'] is None
        assert not row['share_defined'] and row['gap_sign']=='zero'
    assert hooks(model)==0


def test_paired_declared_min_gap_and_negative_raw_gap(setup):
    _,c=setup
    case=dict(case_id='threshold',clean_ids=[1,2,3],corrupt_ids=[1,7,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    result=c.collect(**case,paired_noising=True,paired_min_gap_bits=1e6)
    row=result['paired_patching']['target']
    assert not row['share_defined'] and row['lost_share'] is None
    if row['contrast_gap_bits']>0:
        case['clean_ids'],case['corrupt_ids']=case['corrupt_ids'],case['clean_ids']
    negative=c.collect(**dict(case,case_id='negative'),paired_noising=True)['paired_patching']['target']
    assert negative['contrast_gap_bits']<0
    assert negative['gap_sign']=='negative' and negative['lost_share'] is None


def test_sparse_regimes_mixed_signs_zeros_no_extra_forwards(setup):
    model,c=setup
    gates=[2.,-2.,2.,-2.,0.,2.,0.]
    ups=[3.,3.,-3.,-3.,3.,0.,0.]
    def force(values):
        def hook(m,a,out):
            result=out.clone()
            result[0,1,:len(values)]=torch.tensor(values,dtype=out.dtype)
            return result
        return hook
    g=model.model.layers[0].mlp.gate_proj.register_forward_hook(force(gates))
    u=model.model.layers[0].mlp.up_proj.register_forward_hook(force(ups))
    try:
        regimes={}; native={}; units=[(0,1,n) for n in range(len(gates))]
        c.score([1,2,3],[[4,5],[6]],units=units,snapshot=native,regimes=regimes)
        assert c.forward_examples==2
        assert [regimes[x]['sign_regime'] for x in units]==['++','-+','+-','--','0+','+0','00']
        for unit,gate,up in zip(units,gates,ups):
            row=regimes[unit]
            assert row['raw_gate']==gate and row['up']==up
            assert row['product_value']==float(torch.nn.functional.silu(torch.tensor(gate))*up)
            assert row['product_value']==float(native[unit])
        assert regimes[(0,1,3)]['pole']=='+'  # negative-gate/negative-up echo
        assert all(not isinstance(v,torch.Tensor) for row in regimes.values() for v in row.values())
        assert hooks(model)==2  # only our two manual test hooks remain
    finally: g.remove();u.remove()
    assert hooks(model)==0


def test_sparse_effective_peft_regimes_and_failure_cleanup(setup):
    model,_=setup
    peft=pytest.importorskip('peft')
    model=peft.get_peft_model(model,peft.LoraConfig(r=2,lora_alpha=4,
        target_modules=['gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_B' in name: param.fill_(.15)
    model.eval()
    c=Collector(Neurons(model,None),100,dict(IDENTITY,adapter_sha256='all_mlp_lora'))
    layer=c.nr.layers[0]; seen={}
    hs=[]
    for name in ['gate','up']:
        def observer(m,args,out,name=name):seen[name]=float(out[0,1,4])
        hs.append(getattr(layer.mlp,name+'_proj').register_forward_hook(observer))
    try:
        direct(model,[1,2,3],[4,5])
    finally:
        for h in hs:h.remove()
    regimes={}
    c.score([1,2,3],[[4,5],[6]],units=[(0,1,4)],regimes=regimes)
    assert regimes[(0,1,4)]['raw_gate']==seen['gate']
    assert regimes[(0,1,4)]['up']==seen['up']
    assert hooks(model)==0 and c.forward_examples==2
    def fail(*args):raise RuntimeError('projection failure')
    h=layer.mlp.up_proj.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError,match='projection failure'):
            c.score([1,2,3],[[4,5],[6]],units=[(0,1,4)],regimes={})
        assert hooks(model)==1 and c.forward_examples==3
    finally:h.remove()
    assert hooks(model)==0


def test_public_regimes_match_gold_and_corrupt_pair_counts(setup):
    model,c=setup
    case=dict(case_id='regimes',clean_ids=[1,2,3],corrupt_ids=[1,7,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    public=c.prepare_public(**case)
    result=c.collect(**case,paired_noising=True)
    assert result['clean_native_regimes']==public['clean_native_regimes']
    assert len(result['corrupt_native_regimes'])==len(public['clean_native_regimes'])==2
    assert c.forward_examples==16
    assert hooks(model)==0


def test_first_continuation_argmax_matches_each_actual_forward_no_extra_f(setup):
    model,c=setup
    prompt=[1,2,3];answers=[[4,5],[6]]
    expected={}
    with torch.no_grad():
        for name,answer in zip(('a','b'),answers):
            lp=model(torch.tensor([prompt+answer]),use_cache=False).logits[0,len(prompt)-1].float().log_softmax(-1)
            top=int(lp.argmax())
            expected[name]=dict(argmax_token_id=top,argmax_probability=float(lp[top].exp()))
    calls=[];h=model.register_forward_hook(lambda *args:calls.append(1))
    try:result=c.score(prompt,answers)
    finally:h.remove()
    assert result['first_continuation']==expected
    assert len(calls)==c.forward_examples==c.forward_calls==2
    assert hooks(model)==0


def test_pairwise_repair_does_not_imply_global_first_token_repair(setup):
    """An edit reverses A/B odds while unrelated token9 remains the global winner.

    Hooks script the vocabulary logits after an actual tiny-model native edit;
    the collector must report the observed logits, not infer a top token from A/B.
    """
    model,c=setup;unit=(0,1,4);observed={};calls=[]
    def native(module,args,output):observed['value']=float(args[0][0,1,4])
    def logits(module,args,output):
        calls.append(1)
        result=torch.zeros_like(output.logits)
        result[...,9]=10.  # unrelated global winner before AND after zeroing
        baseline=observed['value']!=0
        result[...,4]=3. if baseline else 1.
        result[...,6]=1. if baseline else 3.
        output.logits=result
        return output
    h=model.model.layers[0].mlp.down_proj.register_forward_hook(native)
    g=model.register_forward_hook(logits)
    try:
        base=c.score([1,2,3],[[4,5],[6,7]],units=[unit],source={unit:torch.tensor(1.)})
        zero=c.score([1,2,3],[[4,5],[6,7]],units=[unit],zero=True)
    finally:h.remove();g.remove()
    assert base['contrast_bits']>0 and zero['contrast_bits']<0
    for score in (base,zero):
        assert {x['argmax_token_id'] for x in score['first_continuation'].values()}=={9}
        assert all(0<x['argmax_probability']<=1 for x in score['first_continuation'].values())
    assert len(calls)==c.forward_examples==c.forward_calls==4
    assert hooks(model)==0


def test_first_continuation_metadata_stays_private_until_collect(setup):
    model,c=setup
    args=dict(case_id='private_top',clean_ids=[1,2,3],corrupt_ids=[1,7,3],
        answer_a_ids=[4,5],answer_b_ids=[6],target_units=[(0,1,4)],control_units=[(0,1,5)])
    public=c.prepare_public(**args)
    assert 'first_continuation' not in str(public)
    assert 'clean' not in public and 'interventions' not in public
    assert c.forward_examples==2
    result=c.collect(**args)
    assert set(result['clean']['first_continuation'])=={'a','b'}
    assert set(result['corrupt']['first_continuation'])=={'a','b'}
    assert all(set(row['first_continuation'])=={'a','b'} for row in result['interventions'].values())
    assert c.forward_examples==12 and result['forward_examples']==10
    assert hooks(model)==0


def test_query_evidence_original_norm_and_signed_weights_no_extra_f_or_leak(setup):
    import math
    model,c=setup;prompt=[1,2,3];units=[(0,2,4),(1,2,5)]
    observed={};handles=[]
    for layer,pos,n in units:
        def down(module,args,layer=layer,pos=pos,n=n):observed[(layer,pos,n)]=float(args[0][0,pos,n])
        handles.append(c.nr.layers[layer].mlp.down_proj.register_forward_pre_hook(down))
    def final(module,args):
        h=args[0][0,len(prompt)-1].float();observed['rms']=float((h.square().mean()+module.variance_epsilon).sqrt())
    handles.append(model.model.norm.register_forward_pre_hook(final))
    calls=[];handles.append(model.register_forward_hook(lambda *a:calls.append(1)))
    try:packet=c.prepare_query_evidence(case_id='query',prompt_ids=prompt,answer_a_ids=[4,5],answer_b_ids=[6,7],units=units,readout_token_ids=[4,6])
    finally:
        for handle in handles:handle.remove()
    assert len(calls)==c.forward_examples==2 and hooks(model)==0
    assert packet['schema']=='native_query_evidence_v1' and packet['case_id']=='query'
    assert packet['final_norm']['frozen_denominator']==observed['rms']
    for row,unit in zip(packet['neurons'],units):
        layer,pos,n=unit;a=observed[unit];W=c.nr.layers[layer].mlp.down_proj.weight[:,n].detach().float()
        U=model.lm_head.weight[[4,6]].detach().float();scale=model.model.norm.weight.detach().float()
        expected=float((U[0]-U[1]) @ (scale*(a*W)))/observed['rms']/math.log(2)
        assert row['activation']==a and row['static_signed_contrast_bits']==pytest.approx(expected,abs=1e-8)
        assert row['write_l2']==pytest.approx(float((a*W).norm()),abs=1e-8)
        assert row['unit']==list(unit)
    assert packet['static_zero_delta_bits']==-sum(n['static_signed_contrast_bits'] for n in packet['neurons'])
    def keys(obj):
        if isinstance(obj,dict):return set(obj)|set().union(*(keys(v) for v in obj.values()))
        if isinstance(obj,list):return set().union(*(keys(v) for v in obj)) if obj else set()
        return set()
    assert not keys(packet)&{'a_logp_nats','b_logp_nats','first_continuation','interventions','contrast_bits'}
    assert packet['frozen_target_identity']==IDENTITY


def test_query_evidence_peft_down_and_unembedding_match_actual_sparse_operators(setup):
    import math
    model,_=setup
    peft=pytest.importorskip('peft')
    model=peft.get_peft_model(model,peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
        target_modules=['gate_proj','up_proj','down_proj','lm_head'],task_type='CAUSAL_LM'))
    torch.manual_seed(8)
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_B' in name:param.normal_(0,.1)
    model.eval();c=Collector(Neurons(model,None),100,dict(IDENTITY,adapter_sha256='query_lora'))
    packet=c.prepare_query_evidence(case_id='peft',prompt_ids=[1,2,3],answer_a_ids=[4,5],answer_b_ids=[6,7],
        units=[(0,2,4)],readout_token_ids=[4,6])
    row=packet['neurons'][0];a=row['activation'];module=c.nr.layers[0].mlp.down_proj
    basis=torch.zeros(24);basis[4]=a
    with torch.no_grad():
        write=module(basis)-module(torch.zeros(24))
        scaled=model.get_base_model().model.norm.weight.float()*write/packet['final_norm']['frozen_denominator']
        logits=model.get_base_model().lm_head(scaled)-model.get_base_model().lm_head(torch.zeros(16))
    assert row['write_l2']==pytest.approx(float(write.norm()),rel=1e-5,abs=1e-7)
    assert row['static_signed_contrast_bits']==pytest.approx(float(logits[4]-logits[6])/math.log(2),rel=1e-5,abs=1e-7)
    assert c.forward_examples==2 and hooks(model)==0


@pytest.mark.parametrize('bad',['position','edited','norm','readout'])
def test_query_evidence_invalid_setup_rejected_before_forward(setup,bad):
    model,c=setup;kwargs=dict(units=[(0,2,4)],native_evidence={},readout_token_ids=[4,6])
    if bad=='position':kwargs['units']=[(0,1,4)]
    if bad=='edited':kwargs['zero']=True
    if bad=='norm':
        model.model.norm=torch.nn.RMSNorm(16).eval()
        c=Collector(Neurons(model,None),100,IDENTITY)
    if bad=='readout':kwargs['readout_token_ids']=[4,4]
    with pytest.raises(ValueError):c.score([1,2,3],[[4,5],[6]],**kwargs)
    assert c.forward_examples==0 and hooks(model)==0


def test_query_evidence_failure_cleans_norm_and_native_observers(setup):
    model,c=setup
    def fail(*args):raise RuntimeError('synthetic final norm failure')
    h=model.model.norm.register_forward_pre_hook(fail)
    sink={}
    try:
        with pytest.raises(RuntimeError,match='final norm failure'):
            c.score([1,2,3],[[4,5],[6,7]],units=[(0,2,4)],native_evidence=sink,readout_token_ids=[4,6])
        assert sink=={} and c.forward_examples==1 and hooks(model)==1
    finally:h.remove()
    assert hooks(model)==0


def test_query_evidence_batch_shape_failure_cleans_all_observers(setup):
    model,c=setup
    h=model.model.norm.register_forward_pre_hook(lambda m,args:(args[0].expand(2,-1,-1),))
    try:
        with pytest.raises(ValueError,match='single-context'):
            c.prepare_query_evidence(case_id='bad_batch',prompt_ids=[1,2,3],answer_a_ids=[4],answer_b_ids=[6],
                units=[(0,2,4)],readout_token_ids=[4,6])
        assert c.forward_examples==1 and hooks(model)==1
    finally:h.remove()
    assert hooks(model)==0
