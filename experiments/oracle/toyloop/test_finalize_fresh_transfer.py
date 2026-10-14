import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import finalize_fresh_transfer as f
import matched_diagnostic_sample as s
from test_matched_diagnostic_sample import fresh_prepared

class FinalizerTests(unittest.TestCase):
 def fixture(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);r=Path(tmp.name);p=fresh_prepared();p.update(draft_execution_ready=False)
  inputs=p['models'][s.MODELS['base']]['inputs'];panel=[dict(id=x['id'],stratum='fresh',prompt='facts',prompt_sha256=__import__('hashlib').sha256(b'facts').hexdigest()) for x in inputs]
  for x,y in zip(inputs,panel):x['prompt_sha256']=y['prompt_sha256']
  paths={name:r/name for name in ('panel','manifest','draft','pipeline')}
  paths['panel'].write_text(''.join(json.dumps(x)+'\n' for x in panel));paths['manifest'].write_text(json.dumps(dict(arms=list(s.FRESH_ROLES),panel=panel)))
  for group,steps in (('old',24),('new',35)):
   root=r/group;root.mkdir();(root/'common_init.jsonl').write_text('{"path":"tinker://fake/init"}\n')
   p['common_init_receipt_sha256']=s.sha(root/'common_init.jsonl')
   (root/'start.jsonl').write_text(json.dumps(dict(project=s.PROJECT,prepared_sha256=p['primary_prepared_sha256'] if steps==24 else p['training35_prepared_sha256'],common_init_receipt_sha256=p['common_init_receipt_sha256']))+'\n')
   (root/'complete.jsonl').write_text(json.dumps(dict(steps_per_arm=steps,remaining_reserved_usd='0'))+'\n')
   for arm in ('point','intervention'):
    file=root/(arm+'_checkpoint.jsonl');file.write_text(json.dumps({'path':'tinker://fake/'+group+'/'+arm})+'\n')
    p['checkpoint_receipt_sha256'][arm+str(steps)]=s.sha(file) if steps==24 else None
  p.update(source_panel_sha256=s.sha(paths['panel']),scoring_manifest_sha256=s.sha(paths['manifest']))
  paths['pipeline'].write_text(json.dumps(dict(panel_sha256=s.sha(paths['panel']),raw_scoring_manifest_sha256=s.sha(paths['manifest']),arms=list(s.FRESH_ROLES),all_cases=8,additional_llm_calls=0,pipelines=[dict(name='raw_explicit'),dict(name='same_selected_architecture_completion')])))
  paths['draft'].write_text(json.dumps(p));a=SimpleNamespace(draft=paths['draft'],draft_sha256=s.sha(paths['draft']),panel=paths['panel'],manifest=paths['manifest'],manifest_sha256=s.sha(paths['manifest']),pipeline_protocol=paths['pipeline'],pipeline_protocol_sha256=s.sha(paths['pipeline']),checkpoints=r/'old',new_checkpoints=r/'new',out=r/'out')
  return a,p
 def test_complete_receipts_preserve_token_arrays_no_execution_authorization(self):
  a,p=self.fixture()
  with patch.object(f,'DRAFT_SHA',a.draft_sha256):receipt=f.finalize(a)
  final=json.loads((a.out/'prepared.json').read_text());self.assertEqual(final['models'],p['models']);self.assertFalse(receipt['execution_authorized']);self.assertEqual(receipt['service_calls'],0)
  self.assertTrue(all(len(x)==64 for x in final['checkpoint_receipt_sha256'].values()))
 def test_wrong_initial_state_or_incomplete_parent_reject_before_output(self):
  for change in ('common_init.jsonl','complete.jsonl'):
   a,p=self.fixture();(a.new_checkpoints/change).write_text('changed')
   with patch.object(f,'DRAFT_SHA',a.draft_sha256),self.assertRaises((ValueError,json.JSONDecodeError)):f.finalize(a)
   self.assertFalse(a.out.exists())
