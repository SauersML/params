import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import matched_graph_decode as d
import matched_graph_train as t
from test_matched_graph_train import fixture

class Renderer:
 def __init__(self):self.parsed=[]
 def build_generation_prompt(self,messages):return S(to_ints=lambda:[1,2])
 def get_stop_sequences(self):return ['<|im_end|>']
 def parse_response(self,tokens):
  self.parsed.append(list(tokens))
  if tokens==[99]:raise ValueError('malformed')
  tokens.append(100) # A parser cannot mutate the retained raw token receipt.
  return {'role':'assistant','content':'\nflowchart LR\nL0H0 --> out\n\n'},True

class DecodeTest(unittest.TestCase):
 def inputs(self,root):
  payload=fixture();source=[]
  for i,p in enumerate(payload['contract']['panel']):
   prompt='exact source prompt '+str(i);source.append({'id':p['id'],'stratum':p['stratum'],'prompt':prompt,'prompt_sha256':d.prompt_sha(prompt)})
  payload['contract']['panel_sha256']=t.digest(source)
  prepared=root/'prepared.json';prepared.write_text(t.encoded(payload));panel=root/'panel.jsonl';d.write_rows(panel,source)
  manifest=root/'manifest.json';manifest.write_text(t.encoded({'arms':['base',*t.ARMS],'panel':[{'id':s['id'],'stratum':s['stratum'],'prompt_sha256':s['prompt_sha256']} for s in source]}))
  start=root/'start.jsonl';d.write_rows(start,[{'prepared_sha256':d.sha(prepared),'project':t.PROJECT}])
  arms={}
  for arm in ('base',*t.ARMS):
   raw=[{'arm':arm,'id':s['id'],'stratum':s['stratum'],'sample':0,'prompt_rendered_sha256':payload['contract']['panel'][i]['rendered_sha256'],'sequences':[{'token_ids':[3],'finish':'stop'}],'prompt_cache_hit_tokens':0} for i,s in enumerate(source)]
   if arm=='point':raw.pop(2)
   if arm=='base':raw[0]['sequences'][0]['finish']='length';raw[1]['sequences'][0]['token_ids']=[99]
   file=root/(arm+'_raw.jsonl');d.write_rows(file,raw);arms[arm]={'path':file.name,'sha256':d.sha(file)}
  raw_contract=root/'raw_contract.json';raw_contract.write_text(t.encoded({'prepared_sha256':d.sha(prepared),'scoring_manifest_sha256':d.sha(manifest),'files':{'start.jsonl':d.sha(start)},'arms':arms}))
  args=S(manifest=manifest,source_panel=panel,prepared=prepared,raw_contract=raw_contract,out=root/'decoded',tokenizer=root/'not_loaded')
  for name in ('manifest','source_panel','prepared','raw_contract'):setattr(args,name+'_sha256',d.sha(getattr(args,name)))
  return args
 def test_no_repair_missing_length_and_parser_failure_preserved(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);args=self.inputs(root);rr=Renderer();receipt=d.decode(args,(rr,lambda message:message['content']))
   base=d.rows(args.out/'base.jsonl');self.assertEqual(base[0]['finish'],'length');self.assertEqual(base[0]['completion'],'\nflowchart LR\nL0H0 --> out\n\n');self.assertEqual(base[0]['token_ids'],[3])
   self.assertEqual(base[1]['parse_error'],'ValueError');self.assertEqual(base[1]['token_ids'],[99])
   self.assertEqual(len(d.rows(args.out/'point.jsonl')),31);self.assertEqual(len(receipt['missing_first_responses']['point']),1)
   for arm in receipt['arms']:self.assertEqual(receipt['arms'][arm]['sha256'],d.sha(args.out/(arm+'.jsonl')))
   self.assertEqual(len(rr.parsed),95)
 def test_changed_raw_file_fails_before_renderer(self):
  with tempfile.TemporaryDirectory() as folder:
   args=self.inputs(Path(folder));path=args.raw_contract.parent/'point_raw.jsonl';path.write_text(path.read_text()+'\n');rr=Renderer()
   with self.assertRaisesRegex(ValueError,'raw token receipt changed'):d.decode(args,(rr,lambda m:m['content']))
   self.assertFalse(rr.parsed);self.assertFalse(args.out.exists())
 def test_rebound_wrong_rendered_prompt_or_original_source_rejected(self):
  for change in ('rendered','source'):
   with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
    root=Path(folder);args=self.inputs(root)
    if change=='rendered':
     path=root/'point_raw.jsonl';raw=d.rows(path);raw[0]['prompt_rendered_sha256']='wrong';path.unlink();d.write_rows(path,raw)
     rc=json.loads(args.raw_contract.read_text());rc['arms']['point']['sha256']=d.sha(path);args.raw_contract.write_text(t.encoded(rc));args.raw_contract_sha256=d.sha(args.raw_contract)
    else:
     source=d.rows(args.source_panel);source[0]['prompt']+=' suffix';args.source_panel.unlink();d.write_rows(args.source_panel,source);args.source_panel_sha256=d.sha(args.source_panel)
    with self.assertRaises(ValueError):d.decode(args,(Renderer(),lambda m:m['content']))
 def test_duplicate_and_multiple_first_sequences_never_reselected(self):
  for multiple in (False,True):
   with self.subTest(multiple=multiple),tempfile.TemporaryDirectory() as folder:
    root=Path(folder);args=self.inputs(root);path=root/'base_raw.jsonl';raw=d.rows(path)
    if multiple:raw[0]['sequences'].append({'token_ids':[3],'finish':'stop'})
    else:raw.append(copy.deepcopy(raw[0]))
    path.unlink();d.write_rows(path,raw);rc=json.loads(args.raw_contract.read_text());rc['arms']['base']['sha256']=d.sha(path);args.raw_contract.write_text(t.encoded(rc));args.raw_contract_sha256=d.sha(args.raw_contract)
    if not multiple:
     with self.assertRaises(ValueError):d.decode(args,(Renderer(),lambda m:m['content']))
    else:
     d.decode(args,(Renderer(),lambda m:m['content']));first=d.rows(args.out/'base.jsonl')[0]
     self.assertEqual(first['parse_error'],'expected_one_first_sequence');self.assertEqual(len(first['raw_sequences']),2)
 def test_malformed_stop_and_source_token_reconstruction_fail_closed(self):
  with tempfile.TemporaryDirectory() as folder:
   args=self.inputs(Path(folder));rr=Renderer();rr.parse_response=lambda ids:({'role':'assistant','content':'graph text'},False)
   d.decode(args,(rr,lambda m:m['content']));base=d.rows(args.out/'base.jsonl')
   self.assertNotIn('parse_error',base[0]);self.assertEqual(base[1]['parse_error'],'renderer_malformed_stop_sequence')
  with tempfile.TemporaryDirectory() as folder:
   args=self.inputs(Path(folder));rr=Renderer();rr.build_generation_prompt=lambda _:S(to_ints=lambda:[2,1])
   with self.assertRaisesRegex(ValueError,'source rendering'):d.decode(args,(rr,lambda m:m['content']))
if __name__=='__main__':unittest.main()
