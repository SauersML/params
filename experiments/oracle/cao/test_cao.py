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


def native_control_fixture(tmp_path):
    import json
    o=capture_oracle();base=sequence_example(o)
    source={'model_identifier':cao.MODEL,'model_revision':'a'*40}
    contexts=[]
    for i,split in enumerate(('train','train','eval')):
        e={**base,'ctx':[1+i,3+i,5+i]}
        if i==2:e['blocks']=[base['blocks'][0],{'label':'Neuron A','unit':['neuron',0,3]}]
        contexts.append({'context_id':str(i),'split':split,'document_sha256':str(i)*64,'token_ids':e['ctx'],
            'questions':[{'unit':e['blocks'][1]['unit'],'blocks':e['blocks'],'q':e['q']}]})
    path=tmp_path/'public.plan.json';path.write_text(json.dumps({'source':source,'contexts':contexts}))
    out=tmp_path/'donors.json';plan=cao.freeze_native_controls(path,out,seed=0)
    return o,out,plan


def test_native_control_donors_metadata_only_same_unit_then_explicit_fallback(tmp_path):
    import json
    o,path,plan=native_control_fixture(tmp_path)
    for key,e in plan['rows'].items():
        d=plan['donors'][key];donor=plan['rows'][d['row_key']]
        assert donor['split']=='train' and donor['ctx']!=e['ctx']
        assert d['same_unit']==(e['split']=='train')
        assert 'a' not in e and 'measurement' not in e
    bad=json.loads(path.read_text());first=next(iter(bad['donors']));bad['donors'][first]['row_key']=first
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='donor binding'):cao.configure_native_controls(o,'joint_mismatch',path)
    assert o.passes==0


def test_native_total_controls_atomic_traces_vectors_and_fixed_residuals(tmp_path):
    import copy
    o,path,plan=native_control_fixture(tmp_path)
    e=next(e for e in plan['rows'].values() if e['split']=='eval');before=copy.deepcopy(e)
    modes={}
    for mode in ('correct','native_absent','joint_mismatch','sequence_absent'):
        cao.configure_native_controls(o,mode,path)
        ready,vecs=o.prepare([e]);modes[mode]=(ready[0],vecs[0],o.prompt(ready[0]))
        assert e==before
    real=modes['correct'][1];slots=cao._native_slots(e)
    for mode,(ready,v,prompt) in modes.items():
        assert torch.equal(v[:2],real[:2])
        assert ready['q']==e['q'] and ready['blocks']==e['blocks']
        if mode in ('native_absent','sequence_absent'):
            assert 'evidence unavailable' in prompt and 'activation=[' not in prompt
    assert not modes['native_absent'][1][slots].count_nonzero()
    assert torch.equal(modes['sequence_absent'][1],real)
    ready,v,prompt=modes['joint_mismatch']
    donor=ready['_native_control_payload']['donor_example']
    dn,_=o._vectors(donor,sequence=True)
    assert torch.equal(v[slots],dn[cao._native_slots(donor)])
    assert ready['_native_sequence_packet']['token_ids']==e['ctx']
    assert donor['_native_sequence_packet']['token_ids']!=e['ctx']
    # The prompt is blinded to arm/provenance; receipts explicitly retain mapping.
    assert 'joint_mismatch' not in prompt and 'different-unit fallback' not in prompt
    assert ready['_native_control_receipt']['donor']['scope']=='different-unit fallback'
    assert modes['correct'][2].split(' All context')[0]==prompt.split(' All context')[0]


def test_native_donor_cache_never_reuses_correct_packet_under_wrong_identity(tmp_path):
    o,path,plan=native_control_fixture(tmp_path)
    e=next(iter(plan['rows'].values()))
    cao.configure_native_controls(o,'joint_mismatch',path)
    a,_=o.prepare([e]);assert o.passes==2
    b,_=o.prepare([e]);assert o.passes==3  # receiver fresh, donor cached under actual donor key
    assert a[0]['_native_control_receipt']==b[0]['_native_control_receipt']
    o.text_only=True
    with pytest.raises(ValueError,match='residual evidence'):o.prepare([e])


def test_controlled_training_uses_frozen_schedule_and_never_scores_eval(tmp_path,monkeypatch,capsys):
    import json,hashlib
    from types import SimpleNamespace
    snapshot=tmp_path/('a'*40);snapshot.mkdir()
    for name in ('tokenizer.json','config.json'):(snapshot/name).write_text('{}')
    binding={'model_revision':snapshot.name,'tokenizer_sha256':hashlib.sha256(b'{}').hexdigest(),
        'config_sha256':hashlib.sha256(b'{}').hexdigest()}
    train=[{'cat':'native_edit','context_id':str(i),'ctx':[i],'blocks':[],'q':'q','a':'train'} for i in (1,2)]
    protected={'cat':'native_edit','context_id':'eval','ctx':[9],'blocks':[],'q':'q','a':'NEVER_SCORE'}
    data=tmp_path/'data';data.write_text(''.join(json.dumps(e)+'\n' for e in train+[protected]))
    schedule=tmp_path/'schedule';keys=[cao.native_row_key(e) for e in train]
    schedule.write_text(json.dumps({'schema':'native_train_schedule_v1','data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'train_rows':keys,'batch_size':16,'steps':2,'batches':[[keys[j%2] for j in range(16)]]*2}))
    init=tmp_path/'init';init.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):(init/name).write_text('frozen')
    donors=tmp_path/'donors';donors.write_text('{}');seen=[]
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
        def save_pretrained(self,path):path.mkdir();(path/'done').write_text('fixed')
        def gradient_checkpointing_enable(self,**kw):pass
    class Oracle:
        def __init__(self,*args,**kw):self.m=Model();self.lm=self.m;self.passes=0;assert kw['model']==str(snapshot)
        def batch_loss(self,rows,reduce=False,backward=False):
            assert backward and all(e['a']=='train' for e in rows)
            seen.extend(cao.native_row_key(e) for e in rows)
            for e in rows:self.native_input_sink({'row_key':cao.native_row_key(e),'prefill_tokens':7,'answer_tokens':3})
            loss=self.m.w.square()*len(rows);loss.backward();self.passes+=len(rows)
            return loss.detach(),3*len(rows)
    monkeypatch.setattr(cao,'_training_split',lambda rows:(train,[protected]))
    monkeypatch.setattr(cao,'_native_training_receipt',lambda *args:binding)
    monkeypatch.setattr(cao,'configure_native_controls',lambda *args:None)
    monkeypatch.setattr(cao,'Oracle',Oracle)
    out=tmp_path/'out'
    cao.train(SimpleNamespace(data=data,base_snapshot=snapshot,schedule=schedule,steps=2,seed=0,
        native_control='correct',control_plan=donors,init=str(init),device='cpu',vectors=[],out=out,save_every=1))
    assert seen==json.loads(schedule.read_text())['batches'][0]*2
    assert not (out/'step_1').exists() and (out/'step_2').exists()
    complete=json.loads((out/'training_complete.json').read_text())
    assert complete['answer_tokens']==96 and complete['prefill_tokens']==224
    assert complete['oracle_forward_rows']==32 and complete['backward_rows']==32
    assert complete['protected_eval_used'] is False
    assert '"completed_steps": 2' in capsys.readouterr().out


def test_balanced_native_schedule_cycles_train_only_no_eval_gold_access(tmp_path,monkeypatch):
    import json,hashlib
    rows=[]
    for i in range(6):
        rows.append({'context_id':str(i),'ctx':[i],'blocks':[{'unit':['neuron',0,i]}],'q':'edit',
            'split':'train','measurement':{'original_top_id':1,'edited_top_id':2 if i<2 else 1}})
    # Protected eval has deliberately unusable measurement; any access would fail.
    rows.append({'split':'eval','measurement':None})
    data=tmp_path/'data';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(cao,'_training_split',lambda r:(r[:-1],r[-1:]))
    monkeypatch.setattr(cao,'_native_training_receipt',lambda *a:None)
    a=cao.freeze_native_schedule(data,tmp_path/'a',steps=3,seed=0)
    b=cao.freeze_native_schedule(data,tmp_path/'b',steps=3,seed=0)
    assert a==b
    positive={cao.native_row_key(r) for r in rows[:2]}
    assert all(len(batch)==16 and sum(k in positive for k in batch)==8 for batch in a['batches'])
    assert len({k for batch in a['batches'] for k in batch})==6


def prospective_fixture(tmp_path):
    import json,hashlib
    snapshot=tmp_path/('a'*40);snapshot.mkdir()
    (snapshot/'tokenizer.json').write_bytes(b'fixture tokenizer')
    (snapshot/'config.json').write_text(json.dumps({'vocab_size':8}))
    contexts=[{'id':f'context{i}','split':split,'document_sha256':str(i+1)*64,'token_ids':[i+1]*48}
        for i,split in enumerate(('train','eval'))]
    manifest={'schema':'native_neuron_contexts_v1','model_identifier':cao.MODEL,'model_revision':'a'*40,
        'tokenizer_sha256':hashlib.sha256((snapshot/'tokenizer.json').read_bytes()).hexdigest(),'contexts':contexts}
    path=tmp_path/'contexts.json';path.write_text(json.dumps(manifest))
    return path,snapshot,manifest


def test_prospective_manifest_rejects_leaks_answers_and_invalid_split(tmp_path):
    import copy,json
    path,snapshot,m=prospective_fixture(tmp_path)
    assert cao._native_context_manifest(path)==m
    for edit in ('answer','same_context','same_document','split'):
        bad=copy.deepcopy(m)
        if edit=='answer':bad['contexts'][0]['gold']='hidden'
        if edit=='same_context':bad['contexts'][1]['token_ids']=bad['contexts'][0]['token_ids']
        if edit=='same_document':bad['contexts'][1]['document_sha256']=bad['contexts'][0]['document_sha256']
        if edit=='split':bad['contexts'][1]['split']='test_maybe'
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):cao._native_context_manifest(path)


@pytest.mark.parametrize('flip',[False,True])
def test_native_only_freezes_all_questions_before_edits_and_retains_nulls(tmp_path,monkeypatch,flip):
    import json,hashlib
    from types import SimpleNamespace
    path,snapshot,manifest=prospective_fixture(tmp_path);out=tmp_path/'raw.jsonl'
    events=[];instances=[]
    class Tok:
        def decode(self,x):return f' t{x[0]}'
    class Oracle:
        def __init__(self,*args,**kw):
            assert Path(str(out)+'.source.json').exists()
            self.layers=[None]*28;self.lm=SimpleNamespace(config=SimpleNamespace(intermediate_size=16))
            self.m=SimpleNamespace(eval=lambda:None);self.dev='cpu';self.passes=0;self.tok=Tok();instances.append(self)
    class Model:
        def __init__(self):self.passes=0
        def logp(self,ids):
            events.append('clean');self.passes+=1
            return torch.log_softmax(torch.tensor([[5.,1.,0.,0.,0.,0.,0.,0.]]),-1)
    class Probe:
        def __init__(self,o):self.m=Model()
        def zeroed(self,ids,units):
            assert events.count('clean')==2 and events.count('rank')==2
            plan=json.loads(Path(str(out)+'.plan.json').read_text())
            assert len(plan['contexts'])==2
            events.append('edit');self.m.passes+=len(units)
            logits=torch.zeros(len(units),8);logits[:,1 if flip else 0]=5
            return logits.log_softmax(-1)
    def ranking(o,ids,k):
        events.append('rank');o.passes+=1
        return [('neuron',20,i) for i in range(6)]
    monkeypatch.setattr(cao,'Oracle',Oracle);monkeypatch.setattr(cao,'Probe',Probe);monkeypatch.setattr(cao,'strong_neurons',ranking)
    # Any legacy pair/token-patch path is forbidden in this native-only branch.
    monkeypatch.setattr(cao,'gen_text',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('legacy generation called')))
    args=SimpleNamespace(out=out,contexts=path,base_snapshot=snapshot,seed=9,device='cpu',native_only=True)
    cao.gen(args)
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    plan=json.loads(Path(str(out)+'.plan.json').read_text())
    assert len(rows)==sum(len(p['questions']) for p in plan['contexts'])
    assert len(rows)>=12 and all(r['schema']=='native_neuron_edit_v1' for r in rows)
    assert all(r['measurement']['original_top_id']==0 for r in rows)
    assert all(r['measurement']['edited_top_id']==int(flip) for r in rows)
    assert all((r['a']=='Unchanged.')== (not flip) for r in rows)
    assert events[:4]==['rank','clean','rank','clean']
    completion=json.loads(Path(str(out)+'.complete.json').read_text())
    assert completion['target_forward_rows']==4+len(rows)<=20
    assert cao._native_training_receipt(out,rows)==rows[0]['native_binding']
    altered=[{**r,'q':'ALTERED QUESTION'} if i==0 else r for i,r in enumerate(rows)]
    with pytest.raises(ValueError,match='question plan'):cao._native_training_receipt(out,altered)
    duplicate=[rows[1]]+rows[1:]
    with pytest.raises(ValueError,match='question plan'):cao._native_training_receipt(out,duplicate)
    # Construct the question plan again with independent same-context RNG; none
    # of the edited outcomes can change identities, layout, read layer or wording.
    for c,p in zip(manifest['contexts'],plan['contexts']):
        assert cao._native_candidate_plan(instances[0],c,9)==p
    calls=[]
    def stop_before_model(*args,**kw):
        calls.append(kw)
        raise RuntimeError('stop before actual model')
    monkeypatch.setattr(cao,'Oracle',stop_before_model)
    train_args=SimpleNamespace(data=out,base_snapshot=tmp_path/('b'*40),init='frozen_start',device='cpu')
    with pytest.raises(ValueError,match='exact label base'):
        cao.train(train_args)
    assert not calls
    train_args.base_snapshot=snapshot
    with pytest.raises(RuntimeError,match='stop before actual model'):
        cao.train(train_args)
    assert calls==[{'trainable':True,'model':str(snapshot)}]


def test_prospective_labels_use_full_normalized_kl_and_prompt_never_gold():
    import numpy as np
    from types import SimpleNamespace
    baseline=torch.tensor([2.,1.,0.],dtype=torch.float64)
    changed=torch.tensor([[0.,2.,1.],[2.,1.,0.]],dtype=torch.float64)
    plan={'context_id':'c','split':'eval','document_sha256':'b'*64,'token_ids':[1,2],
        'questions':[{'unit':['neuron',0,i],'blocks':[{'label':f'Neuron A L0.N{i}','unit':['neuron',0,i]}],'q':'Literal all-position zeroing?'} for i in (0,1)],
        'scope':'all context positions including zero'}
    o=SimpleNamespace(tok=SimpleNamespace(decode=lambda ids:str(ids[0])))
    probe=SimpleNamespace(zeroed=lambda *a:changed)
    binding={'model_identifier':cao.MODEL,'model_revision':'a'*40}
    rows=cao._native_physical_labels(o,probe,plan,baseline,binding)
    p=np.exp(baseline.numpy()-np.logaddexp.reduce(baseline.numpy()))
    q=np.exp(changed[0].numpy()-np.logaddexp.reduce(changed[0].numpy()))
    assert rows[0]['measurement']['full_kl_bits']==pytest.approx(float((p*np.log2(p/q)).sum()),abs=1e-14)
    assert rows[1]['measurement']['full_kl_bits']==pytest.approx(0,abs=1e-14)
    assert rows[0]['measurement']['baseline_logp_original_top']==pytest.approx(float(np.log(p[0])))
    # The plain renderer does not serialize answer/measurement dictionaries.
    unflagged={k:v for k,v in rows[0].items() if k!='native_sequence'}
    assert cao.Oracle.prompt(o,unflagged)==cao.Oracle.prompt(o,{**unflagged,'a':'SECRET_GOLD','measurement':{'SECRET_EFFECT':99}})


def test_explicit_prospective_training_split_and_legacy_unchanged():
    import copy
    legacy=[{'text':0},{'text':1},{'q':'replay'},{'text':3,'split':'old_incidental_metadata'}]
    assert cao._training_split(legacy)==([legacy[1],legacy[2],legacy[3]],[legacy[0]])
    binding={'model_identifier':cao.MODEL,'model_revision':'a'*40}
    rows=[{'cat':'native_edit','schema':'native_neuron_edit_v1','split':s,'ctx':[i],
        'document_sha256':str(i)*64,'native_sequence':'native_neuron_sequence_v1',
        'native_binding':binding,'blocks':[{'label':'Neuron A','unit':['neuron',0,0]}]}
        for i,s in ((1,'train'),(2,'eval'))]
    assert cao._training_split(rows)==([rows[0]],[rows[1]])
    for mode in ('mixed','bad_split','context_leak','document_leak','wrong_model'):
        bad=copy.deepcopy(rows)
        if mode=='mixed':bad.append({'text':0})
        if mode=='bad_split':bad[1]['split']='maybe'
        if mode=='context_leak':bad[1]['ctx']=bad[0]['ctx']
        if mode=='document_leak':bad[1]['document_sha256']=bad[0]['document_sha256']
        if mode=='wrong_model':bad[0]['native_binding']['model_identifier']='other-model'
        with pytest.raises(ValueError):cao._training_split(bad)


def test_prospective_train_rejects_incomplete_receipt_before_model_load(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    binding={'model_identifier':cao.MODEL,'model_revision':'a'*40}
    rows=[{'cat':'native_edit','schema':'native_neuron_edit_v1','split':s,'ctx':[i],
        'document_sha256':str(i)*64,'native_sequence':'native_neuron_sequence_v1',
        'native_binding':binding,'blocks':[{'label':'Neuron A','unit':['neuron',0,0]}]}
        for i,s in ((1,'train'),(2,'eval'))]
    data=tmp_path/'partial.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(cao,'Oracle',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('model must not load')))
    with pytest.raises(ValueError,match='completed prospective'):
        cao.train(SimpleNamespace(data=data,base_snapshot=tmp_path/'snapshot'))


def test_sequence_capture_same_forward_scalar_norms_and_default_parity():
    from types import SimpleNamespace
    from contextlib import nullcontext
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.down=torch.nn.Linear(2,3,bias=True)
            with torch.no_grad():
                self.down.weight.copy_(torch.tensor([[3.,1.],[4.,0.],[0.,2.]]));self.down.bias.fill_(7)
            self.calls=0
        def forward(self,input_ids,output_hidden_states=False):
            self.calls+=1
            values=torch.tensor([[-2.,0.],[0.,1.],[3.,-1.]])[input_ids]
            y=self.down(values)
            return SimpleNamespace(logits=y,hidden_states=(values,y))
    m=Model();o=SimpleNamespace(torch=torch,dev='cpu',m=m,passes=0,off=lambda target=None:nullcontext(),
        model_identifier='synthetic-fixed',layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=m.down))])
    ctx=[0,1,2];units=[['neuron',0,0],['neuron',0,1],['neuron',0,0]]
    plain=m(torch.tensor([ctx]))
    old,writes=cao._capture_native(o,ctx,units,hidden=True)
    out,seqwrites,p=cao._capture_native(o,ctx,units,hidden=True,sequence=True)
    assert m.calls==3 and o.passes==2
    assert torch.equal(plain.logits,out.logits) and torch.equal(old.logits,out.logits)
    assert all(torch.equal(a,b) for a,b in zip(writes,seqwrites))
    assert p['position_ids']==ctx and p['vector_position']==2
    assert p['neurons'][0]['activation']==[-2.,0.,3.]
    assert p['neurons'][0]['write_norm']==[10.,0.,15.]
    assert p['neurons'][1]['write_norm']==pytest.approx([0.,5**.5,5**.5])
    assert p['neurons'][0]==p['neurons'][2]
    assert all(set(row)=={'unit','activation','write_norm'} for row in p['neurons'])
    assert not m.down._forward_pre_hooks


def sequence_example(o):
    # Simple local tokenizer avoids cached/downloaded assets in the prospective tests.
    class Tok:
        pad_token_id=0;eos_token_id=2
        def apply_chat_template(self,messages,**kwargs):
            import re
            values=[]
            for m in messages:
                values += [1 if w=='?' else 3+len(w)%28 for w in re.findall(r'\?|[^\s?]+',m['content'])]
            return values+[2,2]
        def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
    o.tok=Tok();o.q=1;o.text_only=False;o._vcache={};o.vec_store={}
    return {'ctx':[1,3,5],'blocks':[{'layer':0,'positions':[0,1]}, {'label':'Neuron A','unit':['neuron',0,2]}],
        'q':'Predict a literal edit.','a':'HIDDEN_TARGET_SENTINEL','native_sequence':'native_neuron_sequence_v1'}


def test_sequence_prepared_copy_prompt_binding_and_no_label_leak():
    import copy
    o=capture_oracle();e=sequence_example(o);before=copy.deepcopy(e)
    prepared,vecs=o.prepare([e]);ready=prepared[0]
    assert e==before and '_native_sequence_packet' not in e and o.passes==1
    assert vecs[0].shape==(3,16)
    p=o.prompt(ready)
    assert 'HIDDEN_TARGET_SENTINEL' not in p
    assert 'native neuron sequence v1' in p and 'activation' in p and 'Write L2 norm' in p
    assert '6 significant digits; no positions omitted' in p
    assert 'input_sha256' not in p and 'checkpoint bytes' not in p
    baseline={k:v for k,v in e.items() if k!='native_sequence'}
    assert o.prompt(baseline)=='Layer: 0\n ? ? \nNeuron A: ? \nPredict a literal edit.'
    assert len(o.encode(ready,False)[0])>len(o.encode(baseline,False)[0])
    # Label changes cannot alter any public input or captured clean evidence.
    changed={**ready,'a':'DIFFERENT_HIDDEN_TARGET'}
    assert o.prompt(changed)==p
    assert o.prompt({**ready,'edited_outcome':'PRIVATE_SENTINEL'})==p


@pytest.mark.parametrize('method',['batch_loss','answers','options_logp'])
def test_sequence_all_oracle_entrypoints_prepare_before_encode_once(method,monkeypatch):
    o=capture_oracle();e=sequence_example(o);seen=[]
    encode=o.encode
    def checked(ex,answer):
        assert '_native_sequence_packet' in ex and o.passes==1
        seen.append(ex['_native_sequence_packet'])
        return encode(ex,answer)
    monkeypatch.setattr(o,'encode',checked)
    if method=='batch_loss':
        total,n=o.batch_loss([e],reduce=False)
        assert torch.isfinite(total) and n>0
    elif method=='options_logp':
        got=o.options_logp([e],[['one','two']]);assert len(got[0])==2
    else:
        monkeypatch.setattr(o.m,'generate',lambda input_ids,**kw:torch.cat([input_ids,torch.full((1,1),3)],dim=1))
        assert len(o.answers([e]))==1
    assert o.passes==1 and seen
    assert all(p is seen[0] for p in seen)
    assert '_native_sequence_packet' not in e


def test_sequence_rejects_bad_flags_overrides_heads_and_packet_fields_before_oracle():
    import copy
    o=capture_oracle();e=sequence_example(o)
    for bad in ({**e,'native_sequence':'unknown'}, {**e,'vec':'stale'}, {**e,'_native_sequence_packet':{}}):
        with pytest.raises(ValueError):o.prepare([bad])
    with pytest.raises(ValueError,match='native neurons only'):
        o.prepare([{**e,'blocks':[{'label':'head','unit':['head',0,0]}]}])
    assert o.passes==0
    with pytest.raises(ValueError,match='sequence-text controls'):
        cao._native_gain_plan([{**e,'text':0,'cat':'edit'}],1)
    prepared,_=o.prepare([e]);ready=prepared[0]
    with pytest.raises(ValueError,match='public fields'):
        o.prompt({**ready,'_native_sequence_packet':{**ready['_native_sequence_packet'],'gold':.2}})
    for key,value in [('activation',[1.]),('activation',[1.,float('nan'),2.]),('write_norm',[1.,-1.,2.])]:
        bad=copy.deepcopy(ready);bad['_native_sequence_packet']['neurons'][0][key]=value
        with pytest.raises(ValueError,match='scalar traces'):o.prompt(bad)
    bad=copy.deepcopy(ready);bad['_native_sequence_packet']['target_adapter']='different'
    with pytest.raises(ValueError,match='binding mismatch'):o.prompt(bad)


def test_legacy_gain_rejects_sequence_text_before_oracle_load(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    data=tmp_path/'flagged.jsonl'
    data.write_text(json.dumps({'native_sequence':'native_neuron_sequence_v1'})+'\n')
    def forbidden(*a,**kw):raise AssertionError('must reject before model load')
    monkeypatch.setattr(cao,'Oracle',forbidden)
    for native in (False,True):
        with pytest.raises(ValueError,match='sequence-text evidence control'):
            cao.gain(SimpleNamespace(data=data,native_controls=native))


@pytest.mark.parametrize('sign',[-1.,1.,0.])
def test_future_gen_text_uses_captured_signed_write_not_unsigned_column(sign,monkeypatch):
    """Exercise the actual generation branch, with synthetic probe outcomes only."""
    from types import SimpleNamespace
    import hashlib
    import numpy as np
    class Tok:
        def decode(self,ids):return ''.join(f't{i}' for i in ids)
    U=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[2.,-1.],[-2.,1.],[1.,1.],[-1.,-1.]])
    scale=torch.tensor([.5,2.]);write=sign*torch.tensor([2.,3.])
    o=SimpleNamespace(torch=torch,tok=Tok(),layers=[None]*4,passes=0,
        lm=SimpleNamespace(config=SimpleNamespace(num_attention_heads=2,intermediate_size=3),
            lm_head=SimpleNamespace(weight=U),model=SimpleNamespace(norm=SimpleNamespace(weight=scale))))
    lp=torch.log_softmax(torch.arange(8,dtype=torch.float32),dim=0)
    class ProbeModel:
        def logp(self,ctx):return lp[None]
        def dist_at(self,ctx):return lp.repeat(len(ctx),1)
    removals=[]
    def zeroed(ids,units):
        removals.append(list(units));return lp.repeat(len(units),1)
    probe=SimpleNamespace(m=ProbeModel(),zeroed=zeroed)
    monkeypatch.setattr(cao.h2h,'CANDS',[0,1])
    monkeypatch.setattr(cao.h2h,'draws',lambda *a:None)
    monkeypatch.setattr(cao,'effects_se',lambda *a:({0:2.,1:1.},{0:.1,1:.1}))
    monkeypatch.setattr(cao,'strong_neurons',lambda *a:[('neuron',0,0),('neuron',0,1),('neuron',0,2)])
    captured=[]
    def capture(oracle,ctx,units,**kwargs):
        captured.append(list(units));oracle.passes+=1
        return None,[write.clone() for u in units]
    monkeypatch.setattr(cao,'_capture_native',capture)
    rows=cao.gen_text(o,probe,[0]*48,random.Random(3),neuron_choices=True,native_sequence=True)
    future=[r for r in rows if r['cat']=='write_lens']
    assert len(future)==3 and len(captured)==1 and o.passes==1
    assert not any(r['cat']=='promotes' for r in rows)
    assert len(removals)==1 and all(u[0]=='neuron' for u in removals[0])
    choice=next(r for r in rows if r['cat']=='neuron_choice')
    assert choice['a']==cao.NONE
    assert len({tuple(b['unit']) for b in choice['blocks'] if 'unit' in b})==4
    assert not any(r['cat']=='units' for r in rows)
    assert all((r.get('native_sequence')=='native_neuron_sequence_v1') == any('unit' in b for b in r['blocks']) for r in rows)
    expected=np.argsort(-(U.numpy()@(scale.numpy()*write.numpy())),kind='stable')[:5].tolist()
    for row in future:
        cert=row['write_lens']
        assert row['schema']=='native_signed_write_lens_v1'
        assert cert['unit']==row['blocks'][0]['unit']
        assert 'static linear projection' in row['q'] and 'not a prediction of downstream causal effects' in row['q']
        assert cert['write_fp32_sha256']==hashlib.sha256(write.numpy().tobytes()).hexdigest()
        if sign:
            assert cert['top_token_ids']==expected
            assert row['a']==', '.join(cao.q_tok(o.tok.decode([t])) for t in expected)
        else:
            assert cert['top_token_ids']==[] and row['a']=='No nonzero signed write.'


def test_neuron_choice_reuses_exact_distributions_and_independent_full_kl():
    import numpy as np
    candidates=[('neuron',0,i) for i in range(4)]+[('neuron',0,0)]
    base=torch.log(torch.tensor([[.5,.3,.2]],dtype=torch.float64))
    probs=np.array([[.01,.49,.5],[.49,.31,.2],[.48,.3,.22],[.5,.29,.21],[.01,.49,.5]])
    logq=torch.tensor(np.log(probs),dtype=torch.float64)
    got=cao._neuron_choice(candidates,logq,base,random.Random(7))
    expected=[float(np.sum(np.array([.5,.3,.2])*np.log2(np.array([.5,.3,.2])/probs[i]))) for i in got['candidate_indices']]
    assert got['kl_bits']==pytest.approx(expected,abs=1e-14)
    assert len({tuple(u) for u in got['native_units']})==4
    winner=max(range(4),key=lambda k:expected[k])
    assert got['winner']=='ABCD'[winner]
    assert got['native_units'][winner]==['neuron',0,0]
    # Outcome changes cannot alter the four selected unit identities/order.
    tied=cao._neuron_choice(candidates,base.repeat(5,1),base,random.Random(7))
    assert tied['native_units']==got['native_units'] and tied['winner'] is None
    weak_q=torch.log(torch.tensor([[.65,.20,.15],[.64,.21,.15],[.63,.22,.15],[.62,.23,.15],[.65,.20,.15]],dtype=torch.float64))
    weak=cao._neuron_choice(candidates,weak_q,base,random.Random(7))
    assert min(weak['kl_bits'])>0 and weak['winner'] is None
    assert 'every context position' in got['intervention']


def test_neuron_choice_rejects_head_units_and_insufficient_unique_neurons():
    base=torch.zeros(1,3)
    with pytest.raises(ValueError,match='four distinct'):
        cao._neuron_choice([('neuron',0,0)]*4,base.repeat(4,1),base,random.Random(0))
    with pytest.raises(ValueError,match='native neuron'):
        cao._neuron_choice([('head',0,i) for i in range(4)],base.repeat(4,1),base,random.Random(0))


def native_gain_rows():
    return [{'text':i*50,'cat':cat,'ctx':[i+1,2],
             'blocks':[{'layer':0,'positions':[0,1]}, {'label':'A','unit':['neuron',0,i]}],
             'q':'Which literal effect?','a':'unchanged'} for i,cat in enumerate(['edit','promotes','edit','promotes'])]


def test_native_gain_plan_is_ordered_outcome_independent_and_reports_donors():
    import copy
    rows=native_gain_rows();val,records,coverage=cao._native_gain_plan(rows,4)
    assert val==rows and coverage=={'eligible_native_rows':4,'selected_rows':4}
    assert [r['donor_index'] for r in records]==[2,3,0,1]
    assert all(r['native_slots']==[2] for r in records)
    changed=copy.deepcopy(rows)
    for e in changed:e['a']='different measured outcome'
    _,second,_=cao._native_gain_plan(changed,4)
    assert [r['donor_index'] for r in second]==[r['donor_index'] for r in records]
    assert all(a['row_sha256']!=b['row_sha256'] for a,b in zip(records,second))
    _,short,_=cao._native_gain_plan(rows,1)
    assert short[0]['donor_index'] is None and not short[0]['permutation_available']
    with pytest.raises(ValueError,match='insufficient'):cao._native_gain_plan(rows,5)


def test_native_gain_controls_preserve_residual_and_original_vectors():
    x=torch.arange(20,dtype=torch.float32).reshape(5,4);before=x.clone()
    donor=torch.tensor([[4.,3.,2.,1.],[8.,7.,6.,5.]])
    for arm in ('correct','zero_native','permute_native'):
        v=cao._native_gain_vectors(x,[1,4],arm,donor)
        assert torch.equal(v[[0,2,3]],x[[0,2,3]])
        assert torch.equal(x,before)
        if arm=='permute_native':assert torch.equal(v[[1,4]],donor)
        if arm=='zero_native':assert not v[[1,4]].count_nonzero()
    with pytest.raises(ValueError):cao._native_gain_vectors(x,[1,4],'permute_native',donor[:1])


def test_native_gain_equivalence_detects_scale_erasure_and_opposite_pole():
    units=[['neuron',0,1]]*4
    source=torch.tensor([[3.,4.],[3.,4.],[0.,0.],[3.,4.]])
    donor=torch.tensor([[6.,8.],[-6.,-8.],[0.,0.],[0.,0.]])
    got=cao._native_gain_equivalence(source,donor,units,units)
    assert got[0]['same_unit'] and got[0]['normalized_equal']
    assert got[0]['cosine']==pytest.approx(1)
    assert got[1]['normalized_opposite'] and got[1]['relative_vector_pole']=='opposite'
    assert got[1]['cosine']==pytest.approx(-1)
    assert got[2]['source_zero'] and got[2]['donor_zero'] and got[2]['normalized_equal']
    assert got[2]['cosine'] is None and got[3]['cosine'] is None
    assert not got[3]['normalized_equal']
    # Identical directions from different identities are recorded, not reselected.
    other=[['neuron',1,2]]*4
    assert not cao._native_gain_equivalence(source,donor,units,other)[0]['same_unit']


def test_native_gain_saves_every_row_paired_losses_and_premeasurement_manifest(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    rows=native_gain_rows();data=tmp_path/'allowed_validation.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    adapter=tmp_path/'step_5000';adapter.mkdir()
    for f in ('adapter_config.json','adapter_model.safetensors'):(adapter/f).write_bytes(b'fixture bytes')
    out=tmp_path/'native_gain.json';seen=[]
    class FakeOracle:
        def __init__(self,*args,**kwargs):
            assert out.with_suffix('.json.plan.json').exists()
            self.m=SimpleNamespace(eval=lambda:None);self.vec_store={};self.passes=0
        def vectors(self,e):
            self.passes+=1
            return torch.tensor([[100.,0.],[0.,100.],[float(e['ctx'][0]),2.]])
        def batch_loss(self,es,reduce):
            assert not self.text_only and not reduce
            e=es[0];v=self.vec_store[e['vec']]
            assert torch.equal(v[:2],torch.tensor([[100.,0.],[0.,100.]]))
            seen.append((e['ctx'],v.clone()))
            return torch.tensor(float(v[-1].sum())),2
    monkeypatch.setattr(cao,'Oracle',FakeOracle)
    args=SimpleNamespace(data=data,adapter=str(adapter),out=out,n=4,device='cpu',native_controls=True,base_snapshot=None)
    cao.gain(args);got=json.loads(out.read_text())
    assert len(got['rows'])==4 and len(seen)==12
    assert got['target_forward_rows']==8 and got['oracle_forward_rows']==12
    assert set(got['by_category'])=={'edit','promotes','units'}
    assert got['by_category']['units']['correct']['selected_rows']==0
    assert all(len(r['arms'])==3 for r in got['rows'])
    assert got['rows'][0]['arms']['correct']['loss_sum_nats']==3
    assert got['rows'][0]['arms']['zero_native']['loss_sum_nats']==0
    assert got['rows'][0]['arms']['permute_native']['loss_sum_nats']==5
    assert got['rows'][0]['native_donor_equivalence'][0]['donor_unit']==rows[2]['blocks'][1]['unit']
    assert all(v['answer_tokens']==2 for r in got['rows'] for v in r['arms'].values())
    with pytest.raises(ValueError,match='new --out'):cao.gain(args)


def test_native_gain_missing_donor_retains_explicit_row(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    data=tmp_path/'data';data.write_text(json.dumps(native_gain_rows()[0])+'\n')
    adapter=tmp_path/'adapter';adapter.mkdir()
    for f in ('adapter_config.json','adapter_model.safetensors'):(adapter/f).write_bytes(b'x')
    class FakeOracle:
        def __init__(self,*a,**kw):self.m=SimpleNamespace(eval=lambda:None);self.passes=0;self.vec_store={}
        def vectors(self,e):self.passes+=1;return torch.ones(3,2)
        def batch_loss(self,*a,**kw):return torch.tensor(1.),1
    monkeypatch.setattr(cao,'Oracle',FakeOracle)
    out=tmp_path/'out';cao.gain(SimpleNamespace(data=data,adapter=str(adapter),out=out,n=1,device='cpu',native_controls=True,base_snapshot=None))
    got=json.loads(out.read_text());assert len(got['rows'])==1
    assert got['rows'][0]['arms']['permute_native']['available'] is False
    assert got['by_category']['edit']['permute_native']['available_rows']==0


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
        exs = cao.gen_text(tiny, pr, ids, rng, neuron_choices=True)
    assert {e["cat"] for e in exs} == {"lens", "which", "pair", "swap", "edit", "write_lens", "neuron_choice"}
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


def test_a_recorded_write_is_its_value_times_its_down_projection_column(tiny):
    """A "write" block (a circuit neuron's recorded value) reads value x down_proj[:, neuron] with no forward pass, the
    same in the batched read."""
    e = {"ctx": ex(tiny)["ctx"], "blocks": [{"layer": 7, "positions": [0, 3]}, {"label": "L5.N9+@4", "write": [5, 9, 2.5]}],
         "q": "q", "a": "a"}
    W = tiny.layers[5].mlp.down_proj
    W = W.get_base_layer().weight if hasattr(W, "get_base_layer") else W.weight
    v = tiny._vectors(e)
    assert torch.allclose(v[2].float(), (W[:, 9].float() * 2.5).to(v.dtype).float())
    assert torch.allclose(tiny.vectors_many([e])[0].float(), v.float(), atol=1e-5)
