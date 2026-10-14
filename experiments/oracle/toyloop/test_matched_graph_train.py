import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import matched_graph_train as m


def fixture():
 ids=list(m.OLD_TRAIN)+[f'{f}_{3100000+100*i+v}' for i,f in enumerate(m.FAMILIES) for v in (0,1)]
 rng=random.Random(0);batches=[]
 for _ in range(8):
  order=list(range(24));rng.shuffle(order);batches.extend(order[i:i+8] for i in range(0,24,8))
 mi={'chunks':[{'type':'encoded_text','tokens':[1,2]}]}
 tensor=lambda data,dtype:{'data':data,'dtype':dtype,'shape':[2],'sparse_crow_indices':None,'sparse_col_indices':None}
 d={'model_input':mi,'loss_fn_inputs':{'target_tokens':tensor([2,3],'int64'),'weights':tensor([0.,1.],'float32')}}
 data={a:[copy.deepcopy(d) for _ in ids] for a in m.ARMS}
 panels=[{'id':f'{f}_{3200000+100*i}','stratum':'new_model','prompt_tokens':2,'rendered_sha256':m.digest([1,2])} for i,f in enumerate(m.FAMILIES)]+[{'id':f'old{i}','stratum':'retention','prompt_tokens':2,'rendered_sha256':m.digest([1,2])} for i in range(24)]
 c={'model':m.MODEL,'project':m.PROJECT,'epochs':8,'batch':8,'seed':0,'updates_per_arm':24,'output_cap':416,'batches_sha256':m.digest(batches),'training':[{'id':i} for i in ids],'rendered_training_sha256':{a:[m.digest({'input':[1,2],'loss':{'target_tokens':[2,3],'weights':[0.,1.]}})]*24 for a in m.ARMS},'train_tokens_per_arm':{a:384 for a in m.ARMS},'panel':panels,'recipe':{'initialization':{'base_model':m.MODEL,'rank':32,'seed':0,'train_mlp':True,'train_attn':True,'train_unembed':True},'optimizer':[{'learning_rate':m.LR*(1-i/24),'beta1':.9,'beta2':.95,'eps':1e-8} for i in range(24)],'loss_fn':'cross_entropy','sampling':{'temperature':0.,'max_tokens':416,'num_samples':1},'evaluation_arms':['base',*m.ARMS],'sample_ids':[[ids[i] for i in b] for b in batches]}}
 return {'schema_version':1,'contract':c,'data':data,'prompts':[copy.deepcopy(mi) for _ in panels],'batches':batches,'stop_sequences':['<|im_end|>']}

def fixture35():
 p=fixture();c=p['contract'];ids=[r['id'] for r in c['training']]+['redundant_000','cancel_001','gate_000']+[f'{f}_{3300000+100*i}' for i,f in enumerate(m.FAMILIES)]
 rng=random.Random(0);visits=[]
 for _ in range(8):
  order=list(range(35));rng.shuffle(order);visits.extend(order)
 batches=[visits[i:i+8] for i in range(0,len(visits),8)]
 manifest={'schema':'paired_train35_v1','train_ids':ids,'evaluation_scope':'train_only_pending_fresh340'}
 c.update(updates_per_arm=35,training=[{'id':i} for i in ids],batches_sha256=m.digest(batches),panel=[],training_manifest=manifest,training_manifest_sha256=m.digest(manifest))
 c['recipe']['initialization'].update(common_init_receipt_sha256='775614467479c81a829219d240d7e4462dc647a79f9e8932c01400a46702de5b',mode='restore_exact_saved_common_state')
 c['recipe'].update(evaluation_arms=[],sample_ids=[[ids[i] for i in b] for b in batches],optimizer=[{'learning_rate':m.LR*(1-i/35),'beta1':.9,'beta2':.95,'eps':1e-8} for i in range(35)])
 for a in m.ARMS:
  p['data'][a]+=[copy.deepcopy(p['data'][a][0]) for _ in range(11)]
  c['rendered_training_sha256'][a]+=[c['rendered_training_sha256'][a][0]]*11
  c['train_tokens_per_arm'][a]=560
 p.update(prompts=[],batches=batches)
 return p

class Future:
 def __init__(self,value=None,callback=None):self.value=value;self.callback=callback
 def result(self):
  if self.callback:self.callback()
  return self.value

class API:
 ModelInput=S(model_validate=lambda value:value)
 Datum=staticmethod(lambda **kw:S(**kw))
 TensorData=staticmethod(lambda **kw:S(**kw))
 SamplingParams=staticmethod(lambda **kw:kw)
 AdamParams=staticmethod(lambda **kw:kw)
 def __init__(self,fail=None):self.fail=fail;self.calls=[];self.clones=[];self.optimizers=[];self.drained=0;self.created=0
 def ServiceClient(self,project_id):self.created+=1;self.calls.append(('service',project_id));return self
 def create_sampling_client(self,base_model):self.calls.append(('base',base_model));return self
 def sample(self,**kw):
  n=417 if self.fail=='sample' else 1
  return Future(S(sequences=[S(tokens=[3]*n,stop_reason='stop')],prompt_cache_hit_tokens=0))
 def create_lora_training_client(self,**kw):self.calls.append(('init',kw));return self
 def save_state(self,name):return Future(S(path='saved/common' if 'common' in name else 'saved/'+name))
 def create_training_client_from_state_with_optimizer(self,path):
  arm=m.ARMS[len(self.clones)];self.clones.append(path);api=self
  class Client:
   def get_info(self):return S(model_dump=lambda **kw:{'is_lora':True,'lora_rank':32,'model_data':{'model_name':m.MODEL}})
   def forward_backward(self,data,loss_fn):
    api.calls.append(('fb',arm))
    if api.fail=='fb' and arm=='intervention':raise RuntimeError('submission failed')
    return Future(callback=lambda:setattr(api,'drained',api.drained+1))
   def optim_step(self,params):
    api.optimizers.append((arm,params))
    if api.fail=='optim':raise RuntimeError('optimizer submission failed')
    return Future()
   def save_state(self,name):return api.save_state(name)
   def save_weights_and_get_sampling_client(self):return api
  return Client()
 def close(self,status):self.calls.append(('close',status));return Future()

class RunnerTest(unittest.TestCase):
 def test_train35_only_restores_same_initial_state_and_never_samples(self):
  from datetime import datetime,timezone
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TINKER_PROJECT_ID':m.PROJECT}):
   root=Path(d);p=root/'p.json';p.write_text(m.encoded(fixture35()))
   proof=root/'proof.json';proof.write_text(m.encoded(m.preflight(p,{'trained':1,'prefill':1,'sampled':1})))
   approval=root/'approval.json';approval.write_text(m.encoded({'schema_version':1,'project':m.PROJECT,'model':m.MODEL,'prepared_sha256':m.sha(p),'runner_sha256':m.sha(m.__file__),'preflight_sha256':m.sha(proof),'cap_usd':'1','execute_authorized':True}))
   common=root/'common.jsonl';cfg={k:v for k,v in fixture()['contract']['recipe']['initialization'].items()}
   common.write_text(m.encoded({'path':'tinker://stub-original-untrained','config':cfg})+'\n')
   budget=root/'budget.json';budget.write_text(m.encoded({'asof_utc':datetime.now(timezone.utc).isoformat(),'authorized_cap_usd':530,'combined_bound_usd':500}))
   actual_sha=m.sha;expected=fixture35()['contract']['recipe']['initialization']['common_init_receipt_sha256'];api=API()
   with patch.object(m,'sha',side_effect=lambda path:expected if Path(path)==common else actual_sha(path)):
    m.execute(p,proof,approval,root/'run',api,common_init_path=common,fresh_budget_path=budget)
   self.assertEqual(api.clones,['tinker://stub-original-untrained']*2)
   self.assertEqual(api.drained,70);self.assertEqual(len(api.optimizers),70)
   self.assertFalse(any(c[0] in ('base','init') for c in api.calls))
   self.assertFalse(list((root/'run').glob('*_raw.jsonl')))
   self.assertEqual((root/'run/common_init.jsonl').read_bytes(),common.read_bytes())
   self.assertEqual(json.loads((root/'run/complete.jsonl').read_text())['first_responses_per_arm'],0)
   self.assertEqual(json.loads((root/'run/start.jsonl').read_text())['common_init_receipt_sha256'],expected)
   api=API()
   with self.assertRaises(ValueError):m.execute(p,proof,approval,root/'bad',api,common_init_path=common,fresh_budget_path=budget)
   self.assertEqual(api.created,0)
 def test_train35_rejects_sampling_and_wrong_saved_configuration(self):
  p=fixture35();m.validate(p)
  p['prompts']=fixture()['prompts'][:1]
  with self.assertRaises(ValueError):m.validate(p)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'receipt.jsonl';path.write_text(m.encoded({'path':'tinker://stub','config':{'base_model':'wrong'}})+'\n')
   c=fixture35()['contract'];c['recipe']['initialization']['common_init_receipt_sha256']=m.sha(path)
   with self.assertRaises(ValueError):m.saved_initialization(c,path)
 def test_loss_diagnostic_respects_mask_and_paired_loss_mass(self):
  data=[S(loss_fn_inputs={'weights':S(data=[0.,.25,.75])}),S(loss_fn_inputs={'weights':S(data=[1.])})]
  result=S(loss_fn_outputs=[{'logprobs':S(data=[float('nan'),-2.,-4.])},{'logprobs':S(data=[-1.])}])
  diagnostic=m.training_diagnostics(result,data)
  self.assertEqual(diagnostic['weighted_nll_nats'],2.25)
  self.assertEqual(diagnostic['loss_weight_mass'],2.)
  self.assertEqual(diagnostic['supervised_tokens'],3)
  self.assertIsNone(diagnostic['diagnostic_error'])
  result.loss_fn_outputs[1]['logprobs'].data=[float('nan')]
  self.assertIsNone(m.training_diagnostics(result,data)['weighted_nll_nats'])
  self.assertIsNotNone(m.training_diagnostics(result,data)['diagnostic_error'])
 def files(self,root):
  prepared=root/'prepared.json';prepared.write_text(m.encoded(fixture()))
  proof=root/'preflight.json';proof.write_text(m.encoded(m.preflight(prepared,{'trained':1,'prefill':1,'sampled':1})))
  approval=root/'approval.json';approval.write_text(m.encoded({'schema_version':1,'project':m.PROJECT,'model':m.MODEL,'prepared_sha256':m.sha(prepared),'runner_sha256':m.sha(m.__file__),'preflight_sha256':m.sha(proof),'cap_usd':'1','execute_authorized':True}))
  return prepared,proof,approval
 def test_offline_complete_contract_and_no_imports(self):
  payload=fixture();self.assertEqual(m.validate(payload)['updates_per_arm'],24)
  for mutation in ('schedule','datum','optimizer','project','row'):
   bad=copy.deepcopy(payload)
   if mutation=='schedule':bad['batches'][0][0]=bad['batches'][0][1]
   if mutation=='datum':bad['data']['point'][0]['loss_fn_inputs']['weights']['data'][1]=.5
   if mutation=='optimizer':bad['contract']['recipe']['optimizer'][1]['learning_rate']=m.LR*(1-1/72)
   if mutation=='project':bad['contract']['project']='default'
   if mutation=='row':bad['contract']['training'][0]['id']='new_model_eval'
   with self.assertRaises(ValueError):m.validate(bad)
 def test_mock_complete_shared_init_and_exact_24_optimizer_steps(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TINKER_PROJECT_ID':m.PROJECT}):
   root=Path(d);api=API();m.execute(*self.files(root),root/'run',api)
   self.assertEqual(api.clones,['saved/common','saved/common']);self.assertEqual(api.drained,48)
   self.assertEqual([p['learning_rate'] for a,p in api.optimizers if a=='point'],[m.LR*(1-i/24) for i in range(24)])
   self.assertEqual(len((root/'run/steps.jsonl').read_text().splitlines()),48)
   self.assertEqual(len((root/'run/training_diagnostics.jsonl').read_text().splitlines()),48)
   for arm in ('base',*m.ARMS):self.assertEqual(len((root/'run'/f'{arm}_raw.jsonl').read_text().splitlines()),32)
   self.assertEqual(api.calls[-1],('close','success'))
   with self.assertRaisesRegex(ValueError,'no restart'):m.execute(*self.files(root),root/'run',api)
   self.assertEqual(api.created,1)
 def test_partial_optimizer_submission_drains_and_accounts_already_sent_fb(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TINKER_PROJECT_ID':m.PROJECT}):
   root=Path(d);api=API('optim')
   with self.assertRaises(RuntimeError):m.execute(*self.files(root),root/'run',api)
   self.assertEqual(api.drained,1)
   rows=[json.loads(x) for x in (root/'run/journal.jsonl').read_text().splitlines()]
   self.assertTrue(any(x['event']=='settled' and x['counts']=={'trained':16} for x in rows))
   self.assertEqual(json.loads((root/'run/failure.jsonl').read_text())['uncertain_reserved_usd'],'0.000000')
 def test_uncertain_submission_remains_reserved_and_raw_saved_before_accounting_failure(self):
  for fail in ('fb','sample'):
   with self.subTest(fail=fail),tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TINKER_PROJECT_ID':m.PROJECT}):
    root=Path(d);api=API(fail)
    with self.assertRaises((RuntimeError,ValueError)):m.execute(*self.files(root),root/'run',api)
    receipt=json.loads((root/'run/failure.jsonl').read_text());self.assertGreater(float(receipt['uncertain_reserved_usd']),0)
    if fail=='sample':self.assertEqual(len(json.loads((root/'run/base_raw.jsonl').read_text())['sequences'][0]['token_ids']),417)
    else:self.assertEqual(api.drained,1)
 def test_all_preflight_and_budget_guards_precede_service(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'TINKER_PROJECT_ID':m.PROJECT}):
   root=Path(d);paths=self.files(root);api=API();approval=json.loads(paths[2].read_text());approval['cap_usd']='0';paths[2].write_text(m.encoded(approval))
   with self.assertRaises(ValueError):m.execute(*paths,root/'run',api)
   self.assertEqual(api.created,0);self.assertFalse((root/'run').exists())
if __name__=='__main__':unittest.main()
