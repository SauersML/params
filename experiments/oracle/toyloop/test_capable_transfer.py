"""Offline paid-boundary guards; no real Service/model calls."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import capable_transfer as t

class TransferGuardTest(unittest.TestCase):
    def fixture(self,root):
        data=root/'data'; (data/'route_transfer').mkdir(parents=True)
        names=[f'case{i}' for i in range(8)]; rows=[]
        for name in names:
            p=data/'route_transfer'/f'{name}.prompt.txt'; p.write_text(name)
            rows.append({'id':p.name,'prompt_tokens':10,'prompt_sha256':t.c.sha(p),'fits_context':True})
        (data/'route_transfer/manifest.json').write_text(json.dumps({'teacher_integrity':'PASS','cases':[{'id':n} for n in names]}))
        (data/'token_fit.json').write_text(json.dumps({'rows':rows}))
        tok=root/'tok'; tok.mkdir(); (tok/'tokenizer.json').write_text('{}')
        state=root/'state.json'; state.write_text(json.dumps({'state':'tinker://fake/weights/frozen'}))
        protocol=root/'protocol.json'; protocol.write_text('{}')
        rr=SimpleNamespace(get_stop_sequences=lambda:[]) 
        return data,tok,state,protocol,rr
    def prepared(self,root):
        data,tok,state,protocol,rr=self.fixture(root)
        with patch.object(t.importlib.metadata,'version',return_value='fake'), patch.object(t.c,'renderer',return_value=(rr,None)),patch.object(t.c,'generation_prompt',return_value=SimpleNamespace(length=10)):
            m=t.prepare(data,state,tok,protocol,root/'preflight.json')
        return m,rr
    def test_actual_files_bound_without_service(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m,_=self.prepared(root)
            self.assertEqual(len(t.bound_inputs(m)),8)
            p=Path(m['root'])/'route_transfer/case0.prompt.txt'; p.write_text('mutation')
            with self.assertRaises(ValueError): t.bound_inputs(m)
    def test_checkpoint_and_discrete_contract_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m,_=self.prepared(root)
            for key,value in (('max_tokens',417),('arms',['v8','base']),('upper_usd',0.)):
                bad=dict(m); bad[key]=value
                with self.assertRaises(ValueError): t.bound_inputs(bad)
            Path(m['checkpoint']).write_text(json.dumps({'state':'tinker://other/weights/not-frozen'}))
            with self.assertRaises(ValueError): t.bound_inputs(m)
    def test_mutated_input_rejected_before_service_constructor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m,_=self.prepared(root)
            Path(m['protocol']).write_text('changed')
            service=SimpleNamespace(ServiceClient=lambda **kw: self.fail('Service must not be called'))
            with patch.dict(os.environ,{'TINKER_PROJECT_ID':t.c.PROJECT}),patch.dict('sys.modules',{'tinker':service}):
                with self.assertRaises(ValueError): t.run(root/'preflight.json',root/'paid',.02)
            self.assertFalse((root/'paid').exists())
    def test_uncertain_sample_retains_reservation_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m,rr=self.prepared(root); calls=[]
            class FailedFuture:
                def result(self): raise RuntimeError('uncertain sample')
            def sample(**kwargs): calls.append(kwargs); return FailedFuture()
            sampler=SimpleNamespace(sample=sample)
            client=SimpleNamespace(save_weights_and_get_sampling_client=lambda:sampler)
            service=SimpleNamespace(create_sampling_client=lambda **kw:sampler,create_training_client_from_state=lambda path:client)
            sdk=SimpleNamespace(ServiceClient=lambda **kw:service,SamplingParams=lambda **kw:kw)
            with patch.dict(os.environ,{'TINKER_PROJECT_ID':t.c.PROJECT}),patch.dict('sys.modules',{'tinker':sdk}),patch.object(t.importlib.metadata,'version',return_value='fake'),patch.object(t.c,'renderer',return_value=(rr,None)),patch.object(t.c,'generation_prompt',return_value=SimpleNamespace(length=10)):
                with self.assertRaises(RuntimeError): t.run(root/'preflight.json',root/'paid',.02)
            self.assertEqual(len(calls),1)
            reservations=[json.loads(x) for x in (root/'paid/reservations.jsonl').read_text().splitlines()]
            self.assertEqual(len(reservations),1); self.assertEqual(reservations[0]['status'],'issued_or_uncertain')
            self.assertFalse((root/'paid/complete.json').exists())
if __name__=='__main__': unittest.main()
