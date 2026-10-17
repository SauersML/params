"""Complete T1 writer cohort and CLI accounting; mocked readers, no model calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'method/oracle'))
import t1_audit
import neuron_verify as nv


def atlas(tmp_path):
    f=tmp_path/'atlas.json'
    example={'value':2.,'regime':'++','before':'full context ','token':'x','after':' continuation'}
    f.write_text(json.dumps({'neurons':[{'layer':0,'index':j,'max':{'+':2.,'-':0.},'+':[example],'-':[]} for j in range(2)]}))
    return f


def fake_reader(monkeypatch,responses):
    calls=[]
    class Reader:
        def __init__(self,*a):self.spend=SimpleNamespace(total=0.);calls.append(('init',a))
        def __call__(self,prompts,**kw):calls.append(('requests',list(prompts),kw));return responses
    monkeypatch.setitem(sys.modules,'llm',SimpleNamespace(Reader=Reader))
    return calls


@pytest.mark.parametrize('selection',[None,'v1','v2'])
def test_explain_prompt_selection_preserves_default_and_explicit_versions(tmp_path,monkeypatch,selection):
    calls=fake_reader(monkeypatch,['{}','{}'])
    args=SimpleNamespace(atlas=atlas(tmp_path),out=tmp_path/'poles.jsonl',writer='mock',examples=1)
    if selection is not None:args.prompt=selection
    t1_audit.explain(args)
    expected=t1_audit.PROMPTS[selection or 'v1'].format(layer=0,index=0,pmax=2.,nmax=0.,
        pos='1. 2.00 [++] ...full context ⟦x⟧ continuation',neg='')
    assert calls[1][1][0]==expected


def test_explain_retains_missing_writer_fields_and_unobserved_poles(tmp_path,monkeypatch):
    responses=['{"positive":"literal positive explanation","negative":"unobserved"}','not JSON']
    calls=fake_reader(monkeypatch,responses)
    out=tmp_path/'poles.jsonl';args=SimpleNamespace(atlas=atlas(tmp_path),out=out,writer='mock-writer',examples=1)
    t1_audit.explain(args)
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    assert [(r['index'],r['pole']) for r in rows]==[(0,'+'),(0,'-'),(1,'+'),(1,'-')]
    assert [r['writer_valid'] for r in rows]==[True,False,False,False]
    assert all(r.get('passes_t1') is False for r in rows if not r['writer_valid'])
    assert rows[1]['corpus_max']==0 and rows[2]['explanation'] is None
    audit=json.loads(out.with_suffix('.jsonl.writer.json').read_text())
    assert audit['expected_poles']==4 and audit['raw_responses']==responses
    assert [r['prompt'] for r in audit['requests']]==calls[1][1]
    assert 'small "echo"' not in calls[1][1][0]
    assert 'Size and causal relevance must be measured, not assumed.' in calls[1][1][0]
    assert rows[0]['writer_raw_response']==responses[0] and rows[0]['writer_request']==calls[1][1][0]
    assert len(rows[0]['writer_audit_sha256'])==64


@pytest.mark.parametrize('responses',[[],['one'],['one','two','three'],'wrong scalar'])
def test_explain_rejects_cardinality_without_truncation_but_keeps_audit(tmp_path,monkeypatch,responses):
    fake_reader(monkeypatch,responses)
    out=tmp_path/'poles.jsonl'
    with pytest.raises(ValueError,match='response count'):
        t1_audit.explain(SimpleNamespace(atlas=atlas(tmp_path),out=out,writer='mock',examples=1))
    assert not out.exists()
    audit=json.loads(out.with_suffix('.jsonl.writer.json').read_text())
    assert audit['raw_responses']==responses and audit['expected_neurons']==2 and audit['expected_poles']==4


def target(monkeypatch):
    tg=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(in_features=4)))],name='tiny')
    monkeypatch.setattr(nv,'Target',lambda *a:tg)
    return tg


def candidate(index=0,pole='+'):
    return {'layer':0,'index':index,'pole':pole,'corpus_max':2.,'explanation':'real provided text','writer_valid':True,'writer_errors':[],'writer_raw_response':'unchanged original'}


def failed(index=0,pole='-'):
    return {'layer':0,'index':index,'pole':pole,'corpus_max':0.,'explanation':None,'writer_valid':False,
            'writer_errors':['missing_or_invalid_explanation','nonpositive_or_invalid_corpus_max'],'writer_raw_response':'bad original'}


def args_file(tmp_path,items):
    src=tmp_path/'poles.jsonl';src.write_text(''.join(json.dumps(x)+'\n' for x in items))
    return SimpleNamespace(neurons=src,out=tmp_path/'verified.jsonl',model='tiny',device='cpu',reader='mock')


def verified(items):
    return [{**r,'valid':True,'passes_t1':True,'blind_max':.6,'near_miss_rate':0.,'distant_rate':0.,'steer_match':1.} for r in items]


def test_verify_cmd_preserves_mixed_cohort_order_and_denominator(tmp_path,monkeypatch,capsys):
    target(monkeypatch);fake_reader(monkeypatch,[]);seen=[]
    def verify(tg,rd,items):seen.extend(items);return verified(items)
    monkeypatch.setattr(nv,'verify',verify)
    items=[failed(),candidate(1),failed(2)]
    a=args_file(tmp_path,items);nv.cmd_verify(a)
    rows=[json.loads(x) for x in a.out.read_text().splitlines()]
    assert seen==[items[1]] and [(r['index'],r['pole']) for r in rows]==[(r['index'],r['pole']) for r in items]
    assert [r['passes_t1'] for r in rows]==[False,True,False]
    assert rows[0]['corpus_max']==0 and rows[0]['explanation'] is None and rows[0]['writer_raw_response']=='bad original'
    assert '1/3 valid' in capsys.readouterr().out


def test_verify_all_writer_invalid_constructs_no_reader_and_keeps_all_rows(tmp_path,monkeypatch):
    target(monkeypatch)
    def forbidden(*a,**kw):raise AssertionError('all-invalid must not call a reader or verify')
    monkeypatch.setitem(sys.modules,'llm',SimpleNamespace(Reader=forbidden));monkeypatch.setattr(nv,'verify',forbidden)
    a=args_file(tmp_path,[failed(),failed(1,'+')]);nv.cmd_verify(a)
    rows=[json.loads(x) for x in a.out.read_text().splitlines()]
    assert len(rows)==2 and all(not r['valid'] and not r['passes_t1'] for r in rows)


@pytest.mark.parametrize('changes',[{'layer':-1},{'layer':1},{'index':4},{'pole':'?'},{'writer_valid':'false'},{'writer_errors':[]}])
def test_invalid_writer_rows_cannot_waive_address_or_declaration_checks(tmp_path,monkeypatch,changes):
    target(monkeypatch)
    def forbidden(*a,**kw):raise AssertionError('invalid input constructed paid reader')
    monkeypatch.setitem(sys.modules,'llm',SimpleNamespace(Reader=forbidden))
    a=args_file(tmp_path,[{**failed(),**changes}])
    with pytest.raises(ValueError):nv.cmd_verify(a)
    assert not a.out.exists()


def test_valid_candidate_still_gets_strict_pre_reader_validation(tmp_path,monkeypatch):
    target(monkeypatch)
    def forbidden(*a,**kw):raise AssertionError('invalid candidate constructed paid reader')
    monkeypatch.setitem(sys.modules,'llm',SimpleNamespace(Reader=forbidden))
    with pytest.raises(ValueError,match='corpus_max'):
        nv.cmd_verify(args_file(tmp_path,[{**candidate(),'corpus_max':0}]))
