"""The causal oracle's encoding, injection, loss and example generation on a tiny random Qwen3 with Qwen3's tokenizer
(CPU; the tokenizer from the local Hugging Face cache)."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cao  # noqa: E402

transformers = pytest.importorskip("transformers")


def capture_oracle():
    """Real tiny Qwen+PEFT, no tokenizer/assets/downloads."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(42)
    base=Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8))
    cfg=LoraConfig(r=2,lora_alpha=2,target_modules=['down_proj','o_proj'])
    m=get_peft_model(base,cfg);m.add_adapter('other',cfg)
    for mod in m.modules():
        if hasattr(mod,'lora_B'):
            for name in ('default','other'):
                torch.nn.init.constant_(mod.lora_A[name].weight,.2 if name=='default' else -.3)
                torch.nn.init.constant_(mod.lora_B[name].weight,.4 if name=='default' else .7)
    m.eval();o=cao.Oracle.__new__(cao.Oracle)
    o.torch,o.dev,o.m=torch,torch.device('cpu'),m
    o.lm=m.base_model.model;o.layers=o.lm.model.layers;o.passes=0
    return o


@pytest.mark.parametrize('target',[None,'default','other'])
def test_native_single_capture_matches_actual_selected_projector(target):
    o=capture_oracle();ctx=[1,3,5]
    units=[['neuron',0,2],['head',1,1],['neuron',0,2]]
    direct={};handles=[]
    for kind,l,i in units[:2]:
        mod=o.layers[l].mlp.down_proj if kind=='neuron' else o.layers[l].self_attn.o_proj
        width=1 if kind=='neuron' else 8
        def hook(m,args,i=i,width=width,kind=kind):
            x=torch.zeros_like(args[0]);x[0,-1,i*width:(i+1)*width]=args[0][0,-1,i*width:(i+1)*width]
            # Actual affine module evaluation; remove bias via zero input.
            direct[kind]=(m(x)-m(torch.zeros_like(x)))[0,-1].detach()
        # Avoid recursive hook while evaluating the physical projector.
        def safe_hook(m,args,fn=hook):
            if getattr(m,'_test_inside',False):return
            m._test_inside=True
            try:fn(m,args)
            finally:m._test_inside=False
        handles.append(mod.register_forward_pre_hook(safe_hook))
    calls=[];h=o.m.register_forward_hook(lambda *a:calls.append(1))
    ex={'ctx':ctx,'target':target,'blocks':[{'label':'A','unit':units[0]}, {'layer':0,'positions':[0,2]},
        {'label':'B','unit':units[1]},{'label':'again','unit':units[2]}]}
    try:
        v=o._vectors(ex)
        assert len(calls)==1 and o.passes==1
        assert torch.allclose(v[0],direct['neuron'],atol=1e-6)
        assert torch.allclose(v[3],direct['head'],atol=1e-6)
        assert torch.equal(v[0],v[4])
        assert o.m.active_adapter=='default'
        for handle in handles:handle.remove()
        handles=[]
        with o.off(target),torch.no_grad():
            H=o.m(input_ids=torch.tensor([ctx]),output_hidden_states=True).hidden_states
        assert torch.equal(v[1:3],H[1][0,[0,2]])
    finally:
        h.remove()
        for handle in handles:handle.remove()


def test_unit_write_counts_and_restores_on_forward_failure():
    o=capture_oracle()
    a=cao.unit_write(o,[1,2],['neuron',0,1],target='other')
    assert o.passes==1
    b=cao.unit_write(o,[1,2],['neuron',0,1])
    assert o.passes==2 and not torch.allclose(a,b)
    mod=o.layers[0].mlp.down_proj;before=len(mod._forward_pre_hooks)
    h=o.m.register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('test failure')))
    try:
        with pytest.raises(RuntimeError,match='test failure'):
            cao.unit_write(o,[1,2],['neuron',0,1],target='other')
        assert o.passes==3 and len(mod._forward_pre_hooks)==before
        assert o.m.active_adapter=='default'
    finally:h.remove()


def test_native_projection_bias_merged_lora_and_dora_rejection():
    o=capture_oracle();mod=o.layers[0].mlp.down_proj
    mod.base_layer.bias=torch.nn.Parameter(torch.arange(16,dtype=torch.float32))
    with o.off('other'):
        x=torch.randn(1,1,24);masked=torch.zeros_like(x);masked[...,2:3]=x[...,2:3]
        expected=(mod(masked)-mod(torch.zeros_like(masked)))[0,0]
        W=cao._native_weight(mod,2,3)
        assert torch.allclose(x[0,0,2:3]@W.T,expected,atol=1e-6)
        mod.merge(adapter_names=['other'])
        try:
            assert torch.allclose(cao._native_weight(mod,2,3),W,atol=1e-6)
            # PEFT ignores another unmerged active adapter while any is merged.
            mod._active_adapter=['other','default']
            assert torch.allclose(cao._native_weight(mod,2,3),W,atol=1e-6)
            assert torch.allclose(x[0,0,2:3]@W.T,(mod(masked)-mod(torch.zeros_like(masked)))[0,0],atol=1e-6)
        finally:mod.unmerge()
    mod.use_dora['other']=True
    try:
        with pytest.raises(ValueError,match='DoRA'):
            cao.unit_write(o,[1,2],['neuron',0,2],target='other')
        assert o.passes==0 and not mod._forward_pre_hooks
        assert o.m.active_adapter=='default'
    finally:mod.use_dora['other']=False
    mod.lora_variant['other']=object()
    try:
        with pytest.raises(ValueError,match='variant'):
            cao.unit_write(o,[1,2],['neuron',0,2],target='other')
        assert o.passes==0
    finally:del mod.lora_variant['other']
    mod.lora_dropout['other']=torch.nn.Dropout(.5)
    with pytest.raises(ValueError,match='stochastic'):
        cao.unit_write(o,[1,2],['neuron',0,2],target='other')
    assert o.passes==0


@pytest.fixture(scope="module")
def tiny():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=24, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    base = Qwen3ForCausalLM(cfg)
    o = cao.Oracle.__new__(cao.Oracle)
    o.torch, o.dev, o.tok, o.text_only = torch, torch.device("cpu"), tok, False
    o.vec_store, o._vcache, o.passes = {}, {}, 0
    o.m = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    o.lm = o.m.base_model.model
    o.layers = o.lm.model.layers
    o.q = tok(" ?", add_special_tokens=False)["input_ids"][0]
    return o


def ex(o, n=5):
    ctx = o.tok(" The quick brown fox jumps over the lazy dog because it was late", add_special_tokens=False)["input_ids"]
    return {"ctx": ctx, "blocks": [{"layer": 7, "positions": list(range(n))}, {"label": "Unit A", "unit": ["neuron", 9, 3]}],
            "q": "Which tokens does unit A's output raise most?", "a": "'cat'"}


def test_encode_finds_every_placeholder_and_the_answer(tiny):
    e = ex(tiny)
    ids, pos, a0 = tiny.encode(e, True)
    assert len(pos) == 6 and all(ids[p] == tiny.q for p in pos)
    assert tiny.tok.decode(ids[a0:]).startswith("'cat'")


def test_the_hook_adds_the_normalized_vector_scaled_to_the_norm(tiny):
    e = ex(tiny)
    ids, pos, _ = tiny.encode(e, False)
    v = tiny.vectors(e)
    grab = {}
    h0 = tiny.layers[1].register_forward_hook(lambda _m, _i, out: grab.__setitem__("plain", (out[0] if isinstance(out, tuple) else out).clone()))
    with torch.no_grad():
        tiny.m(input_ids=torch.tensor([ids]))
    h0.remove()
    h = tiny.hook([v], [pos])
    h1 = tiny.layers[1].register_forward_hook(lambda _m, _i, out: grab.__setitem__("added", (out[0] if isinstance(out, tuple) else out).clone()))
    with torch.no_grad():
        tiny.m(input_ids=torch.tensor([ids]))
    h.remove()
    h1.remove()
    plain, added = grab["plain"][0], grab["added"][0]
    want = plain[pos] + torch.nn.functional.normalize(v, dim=-1) * plain[pos].norm(dim=-1, keepdim=True)
    assert torch.allclose(added[pos], want, atol=1e-4)
    rest = [i for i in range(len(ids)) if i not in pos]
    assert torch.equal(added[rest], plain[rest])


def test_the_loss_trains_the_adapter_alone(tiny):
    e = ex(tiny)
    loss = tiny.batch_loss([e, {**e, "a": "'dog'"}])
    loss.backward()
    grads = [(n, p.grad) for n, p in tiny.m.named_parameters() if p.requires_grad]
    assert grads and all("lora" in n for n, _ in grads) and any(g is not None and g.abs().sum() > 0 for _, g in grads)
    tiny.m.zero_grad(set_to_none=True)


def test_text_only_adds_nothing(tiny):
    e = ex(tiny)
    ids, pos, _ = tiny.encode(e, False)
    tiny.text_only = True
    try:
        h = tiny.hook([tiny.vectors(e)], [pos])
        with torch.no_grad():
            got = tiny.m(input_ids=torch.tensor([ids])).logits
        h.remove()
    finally:
        tiny.text_only = False
    with torch.no_grad():
        plain = tiny.m(input_ids=torch.tensor([ids])).logits
    assert torch.equal(got, plain)


def test_one_text_gives_every_category(tiny, monkeypatch):
    monkeypatch.setattr(cao.h2h, "T", 48)
    pr = cao.Probe(tiny)
    rng = random.Random(0)
    ids = [rng.randrange(1000, 5000) for _ in range(48)]
    with tiny.off():
        exs = cao.gen_text(tiny, pr, ids, rng, heads=True)
    assert {e["cat"] for e in exs} == {"lens", "which", "pair", "swap", "edit", "promotes", "units"}
    for e in exs:
        tiny.encode(e, True)  # every example encodes, placeholders and all


def test_a_training_step_with_checkpointing_keeps_the_hook_for_recomputation(tiny):
    """Gradient checkpointing recomputes layer 1 in the backward pass: the hook must still be on then."""
    tiny.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tiny.m.train()
    try:
        e = ex(tiny)
        tot, n = tiny.batch_loss([e, {**e, "a": "'dog'"}], reduce=False, backward=True)
        assert n > 0 and any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny.m.parameters() if p.requires_grad)
    finally:
        tiny.lm.gradient_checkpointing_disable()
        tiny.m.eval()
        tiny.m.zero_grad(set_to_none=True)


def test_batched_options_equal_one_at_a_time(tiny):
    """options_logp of a batch (right-padded, every row its own vectors) equals each example scored alone."""
    e1, e2 = ex(tiny), ex(tiny, n=3)
    e2 = {**e2, "q": "Which of two? A short one."}
    both = tiny.options_logp([e1, e2], [["'a'", "'bb'"], ["'c'", "'dd'", "'e'"]])
    alone = [tiny.option_logp(e1, ["'a'", "'bb'"]), tiny.option_logp(e2, ["'c'", "'dd'", "'e'"])]
    for b, a_ in zip(both, alone):
        assert all(abs(x - y) < 1e-3 * (1 + abs(y)) for x, y in zip(b, a_)), (b, a_)


def test_batched_reads_equal_one_at_a_time(tiny):
    """vectors_many (one right-padded pass over the distinct contexts) equals each example read alone; an example with a
    unit block is read alone."""
    e1 = {"ctx": ex(tiny)["ctx"], "blocks": [{"layer": 7, "positions": [0, 2, 4]}], "q": "q", "a": "a"}
    e2 = {"ctx": e1["ctx"][:6], "blocks": [{"layer": 3, "positions": [1, 5]}, {"layer": 9, "positions": [0]}], "q": "q", "a": "a"}
    exs = [e1, e2, e1, ex(tiny)]
    for e, v in zip(exs, tiny.vectors_many(exs)):
        assert torch.allclose(v, tiny._vectors(e), atol=1e-5)
