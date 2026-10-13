"""Small offline protocol binding tests; no SDK, model, or tokenizer loads."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import capable_factorial as cf


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.data=self.root/'data'; self.data.mkdir()
        self.core=[{'id':f'core{i}', 'completion':'English: retained'} for i in range(226)]
        self.corebytes=b''.join((json.dumps(r)+'\n').encode() for r in self.core)
        self.corehash=hashlib.sha256(self.corebytes).hexdigest()
        self.panel=self.root/'panel'; self.base=self.root/'base'; self.protocol=self.root/'protocol'; self.protocol.write_text('{}')
        qs=[{'id':str(i),'prompt':'x','prompt_sha256':hashlib.sha256(b'x').hexdigest()} for i in range(24)]
        self.panel.write_text(''.join(json.dumps(q)+'\n' for q in qs)); self.base.write_bytes(self.panel.read_bytes())
        build={'core_sha256':self.corehash,'rows_per_arm':256,'updates':64,'epochs':2,'batch':8,'data_sha256':{},'augmentation':[]}
        eval_cost=4*(24*.195+24*416*.6)/1e6
        fit={'eval_upper_usd':eval_cost,'four_cell_upper_usd':eval_cost+4*512*.44/1e6}
        for family in ('induction','multihop'):
            rows=self.core+[{'id':f'{family}{i}', 'completion':'English: retained'} for i in range(30)]
            path=self.data/f'{family}_train_inputs.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            build['data_sha256'][family]=cf.digest(path)
            fit[family]={'train_tokens_per_arm':512,'stats':[{'id':r['id'],'supervised_tokens':1,'pair_tokens':1,'old_total_mass':1.,'stored_total_mass':1.} for r in rows]}
            for i in range(30):
                proof=self.data/f'{family}{i}.proof'; proof.write_text('bound')
                build['augmentation'].append({'family':family,'slot':{'slot':i},'files':{proof.name:cf.digest(proof)},'teacher_score':{'per_text':[{'kl_bits':1e-6}]}})
        self.build=build; self.fit=fit; self.save()
        self.patch=patch.object(cf,'CORE_SHA',self.corehash); self.patch.start(); self.addCleanup(self.patch.stop)

    def save(self):
        (self.data/'manifest.json').write_text(json.dumps(self.build)); (self.data/'token_fit.json').write_text(json.dumps(self.fit))

    def prepare(self):
        return cf.prepare(self.data,self.panel,self.base,self.protocol,self.root/'out')

    def test_full_coverage_reproducible_paired_schedule(self):
        m=self.prepare(); k=m['datasets']['induction']['sample_ids']; v=m['datasets']['multihop']['sample_ids']
        self.assertEqual(len(k),64); self.assertEqual(len({x for b in k[:32] for x in b}),256)
        for kb,vb in zip(k,v):
            self.assertEqual([x.replace('induction','aug') for x in kb],[x.replace('multihop','aug') for x in vb])
        self.assertEqual(m['init_seed'],0)
        with self.assertRaises(ValueError): self.prepare()

    def test_physical_proof_mutation_rejected(self):
        (self.data/'induction0.proof').write_text('changed')
        with self.assertRaisesRegex(ValueError,'physical teacher artifact'): self.prepare()

    def test_token_fit_order_and_fp32_mass_rejected(self):
        self.fit['induction']['stats'][0]['stored_total_mass']=.5; self.save()
        with self.assertRaisesRegex(ValueError,'stored supervision'): self.prepare()

    def test_common_base_prompt_mismatch_rejected(self):
        rows=[json.loads(x) for x in self.base.read_text().splitlines()]; rows[0]['prompt_sha256']='changed'
        self.base.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'archived base'): self.prepare()

    def test_mock_sdk_common_state_full_schedule_and_paid_parse_failure(self):
        self._mock_run()

    def test_optimizer_submission_failure_drains_paid_forward_without_retry(self):
        self._mock_run(fail_submission=True)

    def _mock_run(self, fail_submission=False):
        import capable_wiring as pilot
        import relation_loss
        events=[]
        future=lambda value:SimpleNamespace(result=lambda:value)
        class Renderer:
            count=0
            def get_stop_sequences(self): return ['stop']
            def parse_response(self,tokens):
                self.count+=1
                if self.count==1: raise ValueError('paid parse failure')
                return 'English: retained',None
        rr=Renderer()
        class Client:
            def get_info(self): return SimpleNamespace(model_dump=lambda **kw:{'is_lora':True,'lora_rank':32,'model_data':{'model_name':pilot.MODEL}})
            def save_state(self,name): events.append(('save',name)); return future(SimpleNamespace(path='shared-init' if 'init' in name else name))
            def forward_backward(self,data,**kw): events.append(('fb',len(data))); return future(None)
            def optim_step(self,*args):
                events.append(('op',))
                if fail_submission: raise RuntimeError('optimizer submission failed')
                return future(None)
            def save_weights_and_get_sampling_client(self): return self
            def sample(self,**kw): return future(SimpleNamespace(prompt_cache_hit_tokens=0,sequences=[SimpleNamespace(tokens=[1],stop_reason='stop')]))
        class Service:
            def __init__(self,**kw): self.kw=kw
            def create_lora_training_client(self,**kw): events.append(('init',kw['seed'])); return Client()
            def create_training_client_from_state_with_optimizer(self,path): events.append(('clone',path)); return Client()
        fake=SimpleNamespace(ServiceClient=Service,AdamParams=lambda **kw:kw,SamplingParams=lambda **kw:kw)
        tok=self.root/'tokenizer'; tok.mkdir(); (tok/'config').write_text('test')
        self.fit['tokenizer_files']={'config':cf.digest(tok/'config')}; self.save()
        (self.data/'fit_dependency_lock.txt').write_text('\n'.join(p+'==test' for p in ('tinker','tinker-cookbook','transformers','tokenizers')))
        m=self.prepare(); destination=self.root/'paid'
        def changed(d,completion,tokenizer,multiplier):
            return d,{'supervised_tokens':1,'old_total_mass':1.,'stored_total_mass':1.}
        with patch.dict('os.environ',{'TINKER_PROJECT_ID':cf.PROJECT}), patch.dict(sys.modules,{'tinker':fake,'tinker_cookbook.renderers':SimpleNamespace(get_text_content=lambda x:x)}), patch.object(pilot,'renderer',return_value=(rr,None)), patch.object(pilot,'datum',side_effect=lambda row,renderer:SimpleNamespace(model_input=SimpleNamespace(length=1))), patch.object(pilot,'generation_prompt',return_value=SimpleNamespace(length=1)), patch.object(relation_loss,'apply_to_datum',side_effect=changed), patch('importlib.metadata.version',return_value='test'):
            if fail_submission:
                with self.assertRaisesRegex(RuntimeError,'optimizer submission'):
                    cf.run(self.root/'out',destination,tok,m['cost_upper_usd'])
            else:
                cf.run(self.root/'out',destination,tok,m['cost_upper_usd'])
        self.assertEqual(events.count(('init',0)),1)
        self.assertEqual(events.count(('clone','shared-init')),4)
        if fail_submission:
            self.assertEqual(sum(e[0]=='fb' for e in events),1)
            self.assertEqual(sum(e[0]=='op' for e in events),1)
            paid=[json.loads(x) for x in (destination/'spend.jsonl').read_text().splitlines()]
            self.assertEqual(len(paid),1); self.assertEqual(paid[0]['trained'],8)
            self.assertFalse((destination/'complete.json').exists())
            return
        self.assertEqual(sum(e[0]=='fb' for e in events),256)
        self.assertEqual(sum(e[0]=='op' for e in events),256)
        self.assertEqual(len((destination/'steps.jsonl').read_text().splitlines()),256)
        raw=[line for path in destination.glob('*_raw.jsonl') for line in path.read_text().splitlines()]
        self.assertEqual(len(raw),96)
        first=json.loads((destination/'induction_m1.jsonl').read_text().splitlines()[0])
        self.assertIn('parse_error',first)
        self.assertAlmostEqual(json.loads((destination/'complete.json').read_text())['reserved_usd'],0.)


if __name__=='__main__': unittest.main()
