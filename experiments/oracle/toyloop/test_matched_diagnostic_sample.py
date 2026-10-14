import copy
from datetime import datetime,timezone,timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import matched_diagnostic_sample as m


def prepared():
    p={'schema_version':1,'policies':m.MODELS,'calls':80,'output_cap':416,'context':32768,
       'num_samples':1,'temperature':0.,'models':{},'checkpoint_receipt_sha256':{a:'hash' for a in ('point','intervention')},
       'primary_prepared_sha256':'primary'}
    ids=[str(i) for i in range(8)]+[str(i)+'__perm' for i in range(8)]
    for name,(revision,renderer) in m.TOKENIZERS.items():
        rows=[{'id':i,'prompt_tokens':2,'model_input':{'chunks':[{'type':'encoded_text','tokens':[1,2]}]},
               'rendered_sha256':m.digest([1,2]),'prompt_sha256':'source'} for i in ids]
        model={'tokenizer_revision':revision,'renderer':renderer,'inputs':rows,'stop_sequences':[3],
               'rates_per_million':{'prefill':'$1','cached_prefill':'$.2','sample':'$2'}}
        model['upper_per_policy_usd']=str(sum((m.charge(model,2,0,416) for _ in rows),Decimal(0)))
        p['models'][name]=model
    p['upper_usd_decimal']=str(sum((Decimal(p['models'][model]['upper_per_policy_usd']) for model in m.MODELS.values()),Decimal(0)))
    return p


def fresh_prepared():
    p=prepared();name=m.MODELS['base'];model=p['models'][name]
    ids=[f'{f}_{3400000+100*i}' for i,f in enumerate(m.train_helper.FAMILIES)]
    model['inputs']=[dict(copy.deepcopy(model['inputs'][0]),id=i) for i in ids]
    model['upper_per_policy_usd']=str(sum((m.charge(model,2,0,416) for _ in ids),Decimal(0)))
    p.update(profile='fresh_transfer35_v1',calls=40,policies={a:name for a in m.FRESH_ROLES},models={name:model},
        primary_prepared_sha256='4c68ec1f9c810bf25f87cad3074da573ad0fa9338abfc22b8e5590053d4306a2',
        training35_prepared_sha256='c'*64,common_init_receipt_sha256='d'*64,
        checkpoint_receipt_sha256={a:'hash' for a in m.FRESH_ROLES[1:]},
        upper_usd_decimal=str(5*Decimal(model['upper_per_policy_usd'])))
    return p


class DiagnosticSampleTests(unittest.TestCase):
    def test_fresh_transfer_cannot_substitute_training_models_or_drop_policies(self):
        p=fresh_prepared();m.validate(p)
        for mutate in (lambda q:q['models'][m.MODELS['base']]['inputs'][0].update(id='additive_vote_3300000'),
                       lambda q:q['checkpoint_receipt_sha256'].pop('point24'),
                       lambda q:q.update(common_init_receipt_sha256='unknown'),
                       lambda q:q.update(calls=32)):
            q=copy.deepcopy(p);mutate(q)
            with self.assertRaises(ValueError):m.validate(q)

    def test_train_profile_exact_source_order_and_72_calls(self):
        p=prepared();name=m.MODELS['base'];model=p['models'][name]
        ids=list(m.train_helper.OLD_TRAIN)+[f'{f}_{3100000+100*i+v}' for i,f in enumerate(m.train_helper.FAMILIES) for v in (0,1)]
        model['inputs']=[dict(copy.deepcopy(model['inputs'][0]),id=i) for i in ids]
        model['upper_per_policy_usd']=str(sum((m.charge(model,2,0,416) for _ in ids),Decimal(0)))
        p.update(profile='train_fit_v1',calls=72,policies={a:name for a in ('base','point','intervention')},models={name:model},
                 primary_prepared_sha256='4c68ec1f9c810bf25f87cad3074da573ad0fa9338abfc22b8e5590053d4306a2',
                 training_jsonl_sha256='94e6f5966516c858a42a5c9f770a9b5f716a4cd31bcdb5427a584e5ffc9fa188',
                 upper_usd_decimal=str(3*Decimal(model['upper_per_policy_usd'])))
        self.assertEqual(m.validate(p),Decimal(p['upper_usd_decimal']))
        p['models'][name]['inputs'][-1]['id']='heldout'
        with self.assertRaises(ValueError):m.validate(p)

    def test_validate_fit_tokens_cost_and_pair_order(self):
        p=prepared();self.assertEqual(m.validate(p),Decimal(p['upper_usd_decimal']))
        for mutate in (lambda p:p.update(calls=81),
                       lambda p:p['models']['Qwen/Qwen3-8B']['inputs'][0].update(rendered_sha256='drift'),
                       lambda p:p['models']['Qwen/Qwen3-8B'].update(upper_per_policy_usd='0')):
            q=copy.deepcopy(p);mutate(q)
            with self.assertRaises(ValueError):m.validate(q)

    def test_cached_accounting_never_double_charges(self):
        model=prepared()['models']['Qwen/Qwen3-8B']
        self.assertEqual(m.charge(model,100,60,10),Decimal('72')/Decimal(1000000))
        for hits,sampled in ((101,10),(0,417),(-1,10),(True,10)):
            with self.assertRaises(ValueError):m.charge(model,100,hits,sampled)

    def test_fresh_budget_rejects_queue_stale_and_overcap(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget'
            for age,bound in ((121,529),(0,531)):
                path.write_text(json.dumps({'asof_utc':(datetime.now(timezone.utc)-timedelta(seconds=age)).isoformat(),
                                           'authorized_cap_usd':530,'combined_bound_usd':bound}))
                with self.assertRaises(ValueError):m.fresh_budget(path)
            path.write_text(json.dumps({'asof_utc':datetime.now(timezone.utc).isoformat(),
                                       'authorized_cap_usd':530,'combined_bound_usd':529}))
            self.assertEqual(m.fresh_budget(path),m.sha(path))

    def run_mock(self,fail_at=None,fresh=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        p=fresh_prepared() if fresh else prepared();checkpoints=root/'parent';checkpoints.mkdir()
        new=root/'new';new.mkdir()
        for arm in ('point','intervention'):
            path=checkpoints/(arm+'_checkpoint.jsonl');path.write_text(json.dumps({'path':'tinker://private/'+arm})+'\n')
            p['checkpoint_receipt_sha256'][arm+'24' if fresh else arm]=m.sha(path)
            if fresh:
                path=new/(arm+'_checkpoint.jsonl');path.write_text(json.dumps({'path':'tinker://private/'+arm+'35'})+'\n')
                p['checkpoint_receipt_sha256'][arm+'35']=m.sha(path)
        if fresh:
            for directory in (checkpoints,new):(directory/'common_init.jsonl').write_text('{"path":"tinker://private/initial"}\n')
            p['common_init_receipt_sha256']=m.sha(checkpoints/'common_init.jsonl')
            (new/'start.jsonl').write_text(json.dumps({'prepared_sha256':p['training35_prepared_sha256'],'project':m.PROJECT,
                'common_init_receipt_sha256':p['common_init_receipt_sha256']})+'\n')
            (new/'complete.jsonl').write_text(json.dumps({'steps_per_arm':35,'remaining_reserved_usd':'0'})+'\n')
        (checkpoints/'start.jsonl').write_text(json.dumps({'prepared_sha256':p['primary_prepared_sha256'],'project':m.PROJECT})+'\n')
        (checkpoints/'complete.jsonl').write_text(json.dumps({'steps_per_arm':24,'remaining_reserved_usd':'0'})+'\n')
        prep=root/'prepared';prep.write_text(json.dumps(p));proof=root/'proof';proof.write_text(json.dumps(m.preflight(prep)))
        approval=root/'approval';approval.write_text(json.dumps({'execute_authorized':True,'project':m.PROJECT,
            'prepared_sha256':m.sha(prep),'runner_sha256':m.sha(m.__file__),'preflight_sha256':m.sha(proof),'cap_usd':p['upper_usd_decimal']}))
        budget=root/'budget';budget.write_text(json.dumps({'asof_utc':datetime.now(timezone.utc).isoformat(),'authorized_cap_usd':530,'combined_bound_usd':529}))
        args=SimpleNamespace(prepared=prep,preflight=proof,approval=approval,fresh_budget=budget,checkpoints=checkpoints,new_checkpoints=new,out=root/'out')
        submitted=[];drained=[];clones=[]
        class Future:
            def __init__(self,value,index=None):self.value=value;self.index=index
            def result(self):
                if self.index is not None:drained.append(self.index)
                return self.value
        class Client:
            def sample(self,**kwargs):
                index=len(submitted);submitted.append(index)
                if index==fail_at:raise RuntimeError('uncertain submission')
                return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[7,3],stop_reason='stop')],prompt_cache_hit_tokens=1),index)
            def get_info(self):return SimpleNamespace(model_dump=lambda **kwargs:{'is_lora':True,'lora_rank':32,'model_data':{'model_name':'Qwen/Qwen3-8B'}})
            def save_weights_and_get_sampling_client(self):return self
        class Service:
            def __init__(self,**kwargs):self.project=kwargs['project_id']
            def create_sampling_client(self,**kwargs):return Client()
            def create_training_client_from_state_with_optimizer(self,path):clones.append(path);return Client()
            def close(self,status):return Future(None)
        api=SimpleNamespace(ServiceClient=Service,ModelInput=SimpleNamespace(model_validate=lambda x:x),SamplingParams=lambda **kw:kw)
        with patch.dict(os.environ,{'TINKER_PROJECT_ID':m.PROJECT}):
            if fail_at is None:m.execute(args,api)
            else:
                with self.assertRaises(RuntimeError):m.execute(args,api)
        return args,submitted,drained,clones

    def test_mock80_first_samples_restore_both_states_no_training(self):
        args,submitted,drained,clones=self.run_mock()
        self.assertEqual(len(submitted),80);self.assertEqual(drained,list(range(80)))
        self.assertEqual(clones,['tinker://private/point','tinker://private/intervention'])
        complete=json.loads((args.out/'complete.jsonl').read_text())
        self.assertEqual(complete['first_responses'],80);self.assertEqual(complete['remaining_reserved_usd'],'0.000000')
        self.assertEqual(complete['training_steps'],0)
        for arm in m.POLICIES:self.assertEqual(len((args.out/(arm+'_raw.jsonl')).read_text().splitlines()),16)

    def test_mock40_fresh_samples_restore_four_policies_and_enforce_common_initialization(self):
        args,submitted,drained,clones=self.run_mock(fresh=True)
        self.assertEqual(len(submitted),40);self.assertEqual(drained,list(range(40)))
        self.assertEqual(clones,['tinker://private/point','tinker://private/intervention',
                                 'tinker://private/point35','tinker://private/intervention35'])
        for arm in m.FRESH_ROLES:self.assertEqual(len((args.out/(arm+'_raw.jsonl')).read_text().splitlines()),8)
        p=json.loads(args.prepared.read_text())
        (args.new_checkpoints/'common_init.jsonl').write_text('different initial state')
        with self.assertRaisesRegex(ValueError,'share exact initial'):m.checkpoint_paths(args,p)

    def test_partial_submission_drains_and_retains_uncertain_bound(self):
        args,submitted,drained,clones=self.run_mock(fail_at=3)
        self.assertEqual(submitted,[0,1,2,3]);self.assertEqual(drained,[0,1,2])
        failure=json.loads((args.out/'failure.jsonl').read_text())
        self.assertGreater(Decimal(failure['uncertain_reserved_usd']),0)
        self.assertFalse(failure['automatic_retry'])
        self.assertFalse((args.out/'complete.jsonl').exists())


if __name__=='__main__':unittest.main()
