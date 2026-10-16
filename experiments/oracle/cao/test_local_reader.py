import json
from types import SimpleNamespace as NS
import pytest
import local_reader as r

def fixture(tmp):
 f=tmp/'frozen';f.mkdir();model=tmp/'model';model.mkdir();tok=tmp/'tok';tok.mkdir()
 def write(p,x):p.write_text(json.dumps(x));return r.sha(p)
 write(model/'config.json',{'vocab_size':200,'max_position_embeddings':2048,'quantization':{'bits':4}});(model/'a.safetensors').write_bytes(b'fake')
 for d in (tok,model):write(d/'tokenizer.json',{})
 source=tmp/'old.json';write(source,{})
 panel=tmp/'panel.json';write(panel,{'cases':[{'case_id':str(i)} for i in range(32)]})
 public=tmp/'public.json';write(public,{'status':'complete','manifest':r.load_json(panel)})
 reg=tmp/'reg.json';write(reg,{})
 rows=[{'request_id':f'q{i}_{a}','question_id':f'q{i}','arm':a,'token_ids':[i]+[1]*101,'prompt_tokens':102,'common_suffix_tokens':101} for i in range(32) for a in ('true','absent','mismatched')]
 (f/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));write(f/'mapping.json',[{'question_id':f'q{i}','case_id':str(i)} for i in range(32)])
 m={'backend':'mlx','quantization':'4bit','max_tokens':512,'temperature':0,'runner_sha256':r.sha(r.__file__),'model':'fake','model_revision':'a'*40,'source_frozen_manifest_path':str(source),'source_frozen_manifest_sha256':r.sha(source),'tokenizer_path':str(tok),'tokenizer_sha256':{'tokenizer.json':r.sha(tok/'tokenizer.json')},'context_cap':2048,'requests_sha256':r.sha(f/'requests.jsonl'),'mapping_sha256':r.sha(f/'mapping.json')}
 for name,path in [('panel',panel),('public_observations',public),('regression_model',reg)]:m[name+'_path']=str(path);m[name+'_sha256']=r.sha(path)
 write(f/'manifest.json',m)
 return f,model

def test_exact_tokens_eos_and_no_retry(tmp_path):
 f,model=fixture(tmp_path);seen=[]
 def stream(m,t,**kw):
  seen.append(kw);assert isinstance(kw['prompt'],list);assert kw['max_tokens']==512 and kw['prefill_step_size']==512 and kw['draft_model'] is None
  yield NS(token=3,text='ok',prompt_tokens=102,generation_tokens=1,finish_reason=None)
  yield NS(token=199,text='',prompt_tokens=102,generation_tokens=2,finish_reason='stop',prompt_tps=25.,generation_tps=12.,peak_memory=3.5)
 out=tmp_path/'out';receipt=r.run(f,model,out,loader=lambda p:(None,NS(eos_token_ids={199})),streamer=stream,sampler_factory=lambda **kw:kw,runtime={'fake':True})
 assert len(seen)==96 and receipt['errors']==0
 assert r.records(out/'raw.jsonl')[0]['tokens']==[3,199]
 assert r.records(out/'responses.jsonl')[0]['completion']=='ok'
 assert r.records(out/'raw.jsonl')[0]['prompt_tps']==25.
 assert r.records(out/'raw.jsonl')[0]['generation_tps']==12.
 assert r.records(out/'raw.jsonl')[0]['peak_memory']==3.5
 assert receipt['performance']['wall_seconds_including_preflight_and_load']>0
 assert receipt['execution_result_sha256']==r.sha(out/'execution_result.json')

def test_all_row_preflight_before_load(tmp_path):
 f,model=fixture(tmp_path);rows=r.records(f/'requests.jsonl');rows[-1]['token_ids'][-1]=200
 (f/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));m=r.load_json(f/'manifest.json');m['requests_sha256']=r.sha(f/'requests.jsonl');(f/'manifest.json').write_text(json.dumps(m))
 with pytest.raises(ValueError,match='token IDs'):r.run(f,model,tmp_path/'out',loader=lambda p:pytest.fail('loaded before validation'),runtime={})

def test_oom_records_remaining_without_retry(tmp_path):
 f,model=fixture(tmp_path);calls=[]
 def stream(*a,**kw):calls.append(1);raise MemoryError('oom');yield
 out=tmp_path/'out';receipt=r.run(f,model,out,loader=lambda p:(None,NS(eos_token_ids={199})),streamer=stream,sampler_factory=lambda **kw:kw,runtime={})
 assert calls==[1] and receipt['errors']==96
 assert len(r.records(out/'responses.jsonl'))==96
 assert all(x['completion'] is None for x in r.records(out/'responses.jsonl'))

def test_length_cap_and_bad_terminal_retained(tmp_path):
 f,model=fixture(tmp_path);calls=[]
 def stream(*a,**kw):
  calls.append(1)
  if len(calls)==1:
   yield NS(token=3,text='bad',prompt_tokens=102,generation_tokens=1,finish_reason='stop')
  else:
   for i in range(512):yield NS(token=3,text='x',prompt_tokens=102,generation_tokens=i+1,finish_reason='length' if i==511 else None)
 out=tmp_path/'out';receipt=r.run(f,model,out,loader=lambda p:(None,NS(eos_token_ids={199})),streamer=stream,sampler_factory=lambda **kw:kw,runtime={})
 assert len(calls)==96 and receipt['errors']==1
 responses=r.records(out/'responses.jsonl');raw=r.records(out/'raw.jsonl')
 assert responses[0]['finish_reason']=='error' and responses[0]['completion'] is None and raw[0]['tokens']==[3]
 assert all(x['finish_reason']=='length' and x['generated_tokens']==512 for x in responses[1:])
