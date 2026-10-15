"""Common hard admission for fixed VPD ranking, with no legacy START fallback."""
import sys
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import builder_fast as bf
import goal_eval as ge
import native_units as nu


def score_row(value=1., **overrides):
    return {ge.PHI_FAMILIES[f][0]: overrides.get(f,value) for f in nu.SHARED}


class FakeGE:
    def __init__(self, rule): self.rule=rule;self.calls=[]
    def vpd_floors(self,tx,pool,split,**kw):
        assert split=='search' and kw=={'shared_only':True}
        self.pool=pool
        return {'families':list(nu.SHARED),'floors':dict.fromkeys(nu.SHARED,.9)}
    def fidelity(self,tx,graphs,pool,**kw):
        assert pool is self.pool and kw==dict(mode='reward',split='search',shared_only=True)
        self.calls.append(graphs[0].count())
        return [self.rule(graphs[0].count())]


def run(rule,n=16,budget=16):
    g=FakeGE(rule); rec={'ranking':'existing attribution order'}
    order=[('h.0.mlp.c_fc',0,c) for c in range(n)]
    return g,rec,lambda:bf._shared_prefix(g,object(),order,rec,max_gate_calls=budget)


def test_nonmonotone_smallest_tested_prefix_checks_all_shared():
    g,r,f=run(lambda n: score_row(N=1. if n in (3,16) else .7))
    graph,record=f()
    assert graph.count()==3 and g.calls==[0,1,2,3]
    assert record['admission']['status']=='ADMITTED'
    assert record['admission']['families']==list(nu.SHARED)
    assert record['size']==3 and record['met']
    assert record['admission']['cleanup'] is False


def test_empty_allowed_and_no_unchecked_full_fallback():
    g,r,f=run(lambda n: score_row())
    assert f()[0].count()==0 and g.calls==[0]
    g,r,f=run(lambda n: score_row(N=.703))
    with pytest.raises(nu.NoAdmittedCandidate) as e:f()
    assert e.value.record is r and not r['met']
    assert g.calls[-1]==16 and r['admission']['failure_reason']=='ALL_GRID_FAILED'
    assert all(x['scores']['N']==.703 and not x['passed'] for x in r['admission']['tried'])


@pytest.mark.parametrize('ranker,passes',[('measured',4),('ig',4),('fisher',8)])
def test_shared_admission_records_selected_ranking(ranker,passes):
    g,r,f=run(lambda n:score_row())
    r.update(ranker=ranker,passes=passes)
    graph,record=f()
    admission=record['admission']
    assert admission['ranking']==f'{ranker} scores, stable descending'
    assert (admission['ranker'],admission['passes'])==(ranker,passes)
    assert graph.count()==0 and g.calls==[0]


@pytest.mark.parametrize('bad',[None,float('nan'),float('inf')])
def test_missing_nonfinite_candidate_score_fails_closed(bad):
    g,r,f=run(lambda n:score_row(pattern=bad),n=1)
    with pytest.raises(nu.NoAdmittedCandidate):f()
    assert len(g.calls)==2


def test_budget_grid_keeps_empty_full_and_records_unmeasured_prefixes():
    g,r,f=run(lambda n:score_row(N=.1),n=10000,budget=4)
    with pytest.raises(nu.NoAdmittedCandidate):f()
    a=r['admission']
    assert len(g.calls)==4 and g.calls[0]==0 and g.calls[-1]==10000
    assert a['grid_budget_limited'] and a['budget_exhausted']
    assert a['failure_reason']=='BOUNDED_GRID_FAILED'
    assert len(a['unbudgeted_grid_sizes'])>4


def test_generic_build_forwards_both_unit_kinds_and_rejects_before_text(monkeypatch):
    calls=[]; task={'prompts':[{'token_ids':[0],'target_positions':[0]}]}
    obj=SimpleNamespace(text=lambda *a,**k:object())
    def vpd(*a,**kw):calls.append(kw);return bf._graph([]),{}
    monkeypatch.setitem(bf.UNITS,'vpd',vpd)
    pool={}
    bf.build(obj,task,shared_floors=True,shared_gate_calls=3,shared_pool=pool)
    assert calls[-1]==dict(log=None,shared_floors=True,shared_gate_calls=3,shared_pool=pool)
    mask=nu.Mask(__import__('torch').zeros(1,1,1,dtype=__import__('torch').bool),__import__('torch').zeros(1,1,1,dtype=__import__('torch').bool))
    def native(*a,**kw):calls.append(kw);return mask,{}
    monkeypatch.setattr(nu,'build_native',native)
    bf.build(obj,task,units='neurons',shared_floors=True,shared_gate_calls=3,shared_pool=pool)
    assert calls[-1]==dict(log=None,shared_floors=True,max_gate_calls=3,pool=pool)
    bad=SimpleNamespace(text=lambda *a,**k:pytest.fail('must reject before text/model'))
    for options in (dict(shared_floors=True,stated=True),dict(shared_gate_calls=3),dict(shared_floors=True,shared_gate_calls=1)):
        with pytest.raises(ValueError):bf.build(bad,task,**options)


def test_build_failure_propagates_before_serialization(monkeypatch):
    def fail(*a,**k):raise nu.NoAdmittedCandidate({'met':False})
    monkeypatch.setitem(bf.UNITS,'vpd',fail)
    monkeypatch.setattr(ge,'to_json',lambda *a:pytest.fail('no circuit after failed admission'))
    obj=SimpleNamespace(text=lambda *a,**k:object())
    with pytest.raises(nu.NoAdmittedCandidate):
        bf.build(obj,{'prompts':[{'token_ids':[0],'target_positions':[0]}]},shared_floors=True)


def test_cli_generic_vpd_failure_writes_no_circuit(tmp_path,monkeypatch):
    from test_builder_fast import cli_setup
    tasks,out=cli_setup(tmp_path,monkeypatch)
    calls=[]
    def fail(*a,**kw):
        calls.append(kw)
        raise nu.NoAdmittedCandidate({'met':False})
    monkeypatch.setitem(bf.UNITS,'vpd',fail)
    monkeypatch.setattr(sys,'argv',['builder_fast',str(tasks),str(out),'--shared-floors','--shared-gate-calls','4','--warm',''])
    with pytest.raises(nu.NoAdmittedCandidate):bf.main()
    assert calls==[dict(log=print,shared_floors=True,shared_gate_calls=4,shared_pool=None)]
    assert not list(out.glob('*.json'))


@pytest.mark.parametrize('args',[
    ['--shared-gate-calls','4'],['--shared-floors','--shared-gate-calls','1'],
    ['--shared-floors','--stated'],['--shared-floors','--native-gate-calls','3'],
])
def test_cli_rejects_invalid_shared_options_before_model(tmp_path,monkeypatch,args):
    monkeypatch.setattr(bf.native,'Native',lambda:pytest.fail('no model allocation'))
    monkeypatch.setattr(sys,'argv',['builder_fast',str(tmp_path/'tasks'),str(tmp_path/'out'),*args])
    with pytest.raises(SystemExit) as exc:bf.main()
    assert exc.value.code==2


@pytest.mark.parametrize('options',[
    dict(shared_gate_calls=3),dict(shared_pool={}),
    dict(shared_floors=True,shared_gate_calls=1),dict(shared_floors=True,shared_gate_calls=True),
])
def test_direct_vpd_option_validation_before_ranking(options):
    with pytest.raises(ValueError):bf.build_vpd(object(),object(),**options)


@pytest.mark.parametrize('flag,units',[('--shared-floors','vpd'),('--shared-native-floors','neurons')])
def test_shared_cli_refuses_legacy_output_before_allocation(tmp_path,monkeypatch,flag,units):
    from test_builder_fast import cli_setup
    tasks,out=cli_setup(tmp_path,monkeypatch)
    out.mkdir();old=out/'text6008.json';old.write_text('{"build":{"met":true}}')
    monkeypatch.setattr(bf.native,'Native',lambda:pytest.fail('no allocation before reuse validation'))
    monkeypatch.setattr(sys,'argv',['builder_fast',str(tasks),str(out),'--units',units,flag])
    with pytest.raises(SystemExit) as exc:bf.main()
    assert exc.value.code==2 and old.read_text()=='{"build":{"met":true}}'


def test_rejected_shared_warmup_does_not_skip_selected_tasks(tmp_path,monkeypatch,capsys):
    from test_builder_fast import cli_setup
    tasks,out=cli_setup(tmp_path,monkeypatch)
    calls=[]
    def once(*a,**kw):
        calls.append(kw)
        if len(calls)==1:raise nu.NoAdmittedCandidate({'met':False})
        return bf._graph([]),{}
    monkeypatch.setitem(bf.UNITS,'vpd',once)
    monkeypatch.setattr(bf.native,'free',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['builder_fast',str(tasks),str(out),'--shared-floors'])
    bf.main()
    assert len(calls)==2 and (out/'text6008.json').exists()
    assert not (out/'text6000.json').exists()
    assert 'shared warmup rejected' in capsys.readouterr().out


def test_shared_warmup_runtime_error_propagates(tmp_path,monkeypatch):
    from test_builder_fast import cli_setup
    tasks,out=cli_setup(tmp_path,monkeypatch)
    def fail(*a,**kw):raise RuntimeError('executor defect')
    monkeypatch.setitem(bf.UNITS,'vpd',fail)
    monkeypatch.setattr(sys,'argv',['builder_fast',str(tasks),str(out),'--shared-floors'])
    with pytest.raises(RuntimeError,match='executor defect'):bf.main()
    assert not list(out.glob('*.json'))


def test_shared_cli_exclusive_write_does_not_clobber_racing_output(tmp_path,monkeypatch):
    from test_builder_fast import cli_setup
    tasks,out=cli_setup(tmp_path,monkeypatch)
    def concurrent(*a,**kw):
        (out/'text6008.json').write_text('other writer bytes')
        return bf._graph([]),{}
    monkeypatch.setitem(bf.UNITS,'vpd',concurrent)
    monkeypatch.setattr(sys,'argv',['builder_fast',str(tasks),str(out),'--shared-floors','--warm',''])
    with pytest.raises(FileExistsError):bf.main()
    assert (out/'text6008.json').read_text()=='other writer bytes'
