"""Synthetic on-disk atlas and mocked label writers; no model/API/dataset calls."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import atlas


class Tokenizer:
    def decode(self,ids):return ' '.join(map(str,ids))


@pytest.fixture
def synthetic(tmp_path):
    root=tmp_path/'atlas';(root/'layers').mkdir(parents=True);(root/'corpus').mkdir()
    (root/'atlas.json').write_text(json.dumps({'model':'synthetic','rates':[.1,.3,.5]}))
    np.save(root/'corpus/tokens.npy',np.arange(20));np.save(root/'corpus/offsets.npy',np.array([0,20]))
    (root/'corpus/meta.json').write_text('{}')
    pos=np.array([[-1.,0.,-2.,np.nan,-np.inf],[0.,-2.,1.,4.,np.inf]])
    neg=np.array([[-3.,-1.,0.,2.,np.inf],[1.,0.,3.,np.nan,-np.inf]])
    values={}
    for name,array in [('pos',pos),('neg',neg)]:
        values[name+'_val']=array;values[name+'_idx']=np.tile(np.arange(5)+1,(2,1))
        values[name+'_g']=np.ones_like(array);values[name+'_u']=np.sign(array)
    values.update(pos0_max=np.zeros(2),pos0_min=np.zeros(2),rates=np.zeros((2,2,3)))
    np.savez(root/'layers/L00.npz',**values)
    return atlas.Atlas(root,tokenizer=Tokenizer())


def test_absent_poles_are_empty_with_zero_max_and_signed_valid_extrema(synthetic):
    raw=(synthetic.root/'layers/L00.npz').read_bytes()
    first=synthetic.neuron(0,0,k=5);second=synthetic.neuron(0,1,k=5)
    assert first['+']==[] and first['max']['+']==0.
    assert [e['value'] for e in first['-']]==[-3.,-1.] and first['max']['-']==-3.
    assert second['-']==[] and second['max']['-']==0.
    assert [e['value'] for e in second['+']]==[1.,4.] and second['max']['+']==4.
    assert all(e['value']>0 for e in second['+']) and all(e['value']<0 for e in first['-'])
    assert (synthetic.root/'layers/L00.npz').read_bytes()==raw


def test_filter_before_k_and_maximum_uses_all_stored_eligible_values(synthetic):
    row=synthetic.neuron(0,1,k=1,text=False)
    assert [e['value'] for e in row['+']]==[1.]
    assert row['max']['+']==4. and row['max']['-']==0.
    assert synthetic.neuron(0,1,k=0,text=False)['max']['+']==4.


@pytest.mark.parametrize('gate,up,expected',[(1,1,'++'),(-1,-1,'--'),(1,-1,'+-'),
    (-1,1,'-+'),(0,1,'0+'),(-1,0,'-0'),(0.,-0.,'00')])
def test_zero_gate_or_up_has_explicit_zero_regime(gate,up,expected):
    assert atlas.regime(gate,up)==expected


@pytest.mark.parametrize('method',['single','revise'])
def test_default_label_and_revise_reject_absent_pole(synthetic,monkeypatch,tmp_path,method):
    """Run real label/revise functions with deterministic fake writer/verifier."""
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setitem(sys.modules,'t1_audit',SimpleNamespace(EXPLAIN='{pos}\n{neg}',
        EXPLAIN_V2='{pos}\n{neg}',WRITER='synthetic'))
    spec=importlib.util.spec_from_file_location('_atlas_labels_pole_test',Path(atlas.__file__).with_name('atlas_labels.py'))
    labels=importlib.util.module_from_spec(spec);spec.loader.exec_module(labels)
    plans=[]
    def plan(reader,items):plans.extend(items);return items
    monkeypatch.setitem(sys.modules,'neuron_verify',SimpleNamespace(parse_json=json.loads,plan=plan))
    class Writer:
        spend=SimpleNamespace(total=0.)
        def __call__(self,prompts,**kwargs):return [json.dumps({'positive':'synthetic explanation','negative':'synthetic explanation'}) for _ in prompts]
    monkeypatch.setitem(sys.modules,'llm',SimpleNamespace(Reader=lambda *a:Writer()))
    monkeypatch.setattr(atlas,'Atlas',lambda *args:synthetic)
    monkeypatch.setattr(labels,'measurer',lambda a:lambda items:[{'valid':True,'blind_max':.5,
        'near_miss_rate':0.,'passes_t1':True,'blind_texts':[],'blind_values':[]} for item in items])
    neurons=tmp_path/'neurons.jsonl';neurons.write_text('{"layer":0,"index":0}\n')
    args=SimpleNamespace(atlas=synthetic.root,out=tmp_path/'labels.jsonl',neurons=neurons,
        writer='synthetic',method=method,chunk=1,examples=5,long=1,rounds=0,corpus_max=None)
    labels.label(args)
    rows={r['pole']:r for r in map(json.loads,args.out.read_text().splitlines())}
    assert rows['+']['corpus_max']==0. and rows['+']['writer_valid'] is False
    assert 'nonpositive_or_invalid_corpus_max' in rows['+']['writer_errors']
    assert rows['-']['corpus_max']==3. and rows['-']['writer_valid'] is True
    if method=='revise':assert plans and all(r['pole']=='-' for r in plans)
