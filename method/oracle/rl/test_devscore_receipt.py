import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from types import SimpleNamespace
import pytest
import tinker_train as tt


def good(task, row):
    return [{'reward': .8, 'valid': True}]


def test_complete_and_model_wrong_remain_numeric():
    r = tt.devscore_receipt(['a', 'b'], [{'task':'b'}, {'task':'a'}],
                            lambda task,row: [{'reward': .8 if task=='a' else 0., 'valid': task=='a'}])
    assert r['complete'] and r['reward']==.4 and r['valid']==.5 and r['scored']==2


@pytest.mark.parametrize('samples,error', [([], 'missing_sample'),
    ([{'task':'a'}, {'task':'a'}], 'duplicate_sample'),
    ([{'task':'a'}, {'task':'z'}], 'unexpected_sample')])
def test_sample_coverage(samples,error):
    r=tt.devscore_receipt(['a','b'], samples,good)
    assert not r['complete'] and r['reward'] is None and r['valid'] is None
    assert error in [e['error'] for e in r['errors']]


@pytest.mark.parametrize('scores', [None, [], [{'reward':.8,'valid':True}]*2,
    [{'reward':float('nan'),'valid':True}], [{'reward':float('inf'),'valid':True}],
    [{'reward':True,'valid':True}], [{'reward':.8}], [{'reward':.8,'valid':1}], [{}]])
def test_bad_scorer_never_improves_surviving_average(scores):
    r=tt.devscore_receipt(['a','b'],[{'task':'a'},{'task':'b'}],
        lambda task,row: good(task,row) if task=='a' else scores)
    assert r['scored']==1 and r['reward'] is None and not r['complete']
    assert r['errors'][0]['task']=='b'


def test_transport_error_receipt():
    def fail(task,row):
        raise TimeoutError('offline fixture')
    r=tt.devscore_receipt(['a'],[{'task':'a'}],fail)
    assert r['reward'] is None and r['errors'][0]['detail']=='offline fixture'


@pytest.mark.parametrize('expected',[[],['a','a']])
def test_invalid_frozen_panel(expected):
    r=tt.devscore_receipt(expected,[{'task':'a'}] if expected else [],good)
    assert not r['complete'] and r['reward'] is None


def test_devscore_http_failure_writes_incomplete_and_preserves_existing(tmp_path, monkeypatch):
    questions=tmp_path/'questions.jsonl'
    questions.write_text('\n'.join(json.dumps({'task':t,'question':'fixture'}) for t in ['a','b']))
    (tmp_path/'checkpoints.jsonl').write_text(json.dumps({'step':1,'state':'fixture'}))
    calls=[]
    def sample(ns):
        calls.append(ns.init)
        ns.out.write_text('\n'.join(json.dumps({'task':t,'text':'fixture','tokens':3}) for t in ['a','b']))
    class Reply:
        status_code=503
        text='fixture scorer unavailable'
    monkeypatch.setattr(tt,'sample',sample)
    monkeypatch.setattr(tt.httpx,'post',lambda *args,**kwargs: Reply())
    args=SimpleNamespace(log=tmp_path,questions=questions,request_extra=None,n=2,
        base='fixture',max_tokens=3,thinking=False,scorer='fixture')
    tt.devscore(args)
    result=json.loads((tmp_path/'devscore_1.json').read_text())
    assert result['reward'] is None and result['complete'] is False
    assert len(result['errors'])==2 and result['expected_tasks']==['a','b']
    tt.devscore(args)
    assert calls==['fixture']


def test_sampling_failure_is_incomplete_without_scoring(tmp_path, monkeypatch):
    (tmp_path/'questions.jsonl').write_text(json.dumps({'task':'a'}))
    (tmp_path/'checkpoints.jsonl').write_text(json.dumps({'step':1,'state':'fixture'}))
    def unavailable(ns):
        raise RuntimeError('fixture sampling failure')
    monkeypatch.setattr(tt,'sample',unavailable)
    monkeypatch.setattr(tt.httpx,'post',lambda *a,**k: pytest.fail('must not score missing samples'))
    args=SimpleNamespace(log=tmp_path,questions=tmp_path/'questions.jsonl',request_extra=None,n=1,
        base='fixture',max_tokens=3,thinking=False,scorer='fixture')
    tt.devscore(args)
    result=json.loads((tmp_path/'devscore_1.json').read_text())
    assert not result['complete'] and result['reward'] is None
    assert {e['error'] for e in result['errors']}=={'missing_sample','sampling_error'}


@pytest.mark.parametrize('key',['task','answers','lines','tokens'])
def test_reserved_request_override_rejected_before_sampling(tmp_path, monkeypatch, key):
    monkeypatch.setattr(tt,'sample',lambda *a,**k: pytest.fail('must reject before sampling'))
    with pytest.raises(ValueError,match='cannot override'):
        tt.devscore(SimpleNamespace(request_extra=json.dumps({key:'override'})))
