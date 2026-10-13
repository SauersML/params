import unittest,json,tempfile
from pathlib import Path
import claim_affine_english_probe as p
class Probe(unittest.TestCase):
 def test_exact_finish_and_boolean_contract(self):
  r={'finish':'StopReason.stop','completion':'{"top_token":" the","changed":false}'}
  self.assertEqual(p.prediction(r)[0],{'top_token':' the','changed':False})
  for finish in ('length','StopReason.length','error',None):self.assertIsNone(p.prediction(dict(r,finish=finish))[0])
  self.assertIsNone(p.prediction(dict(r,completion='{"top_token":" the","changed":0}'))[0])
  self.assertIsNone(p.prediction(dict(r,upstream_invalid=True))[0])
 def test_common_context_no_evidence_threshold_leak(self):
  common='site down component7 k=.5 context ORIGINAL'
  none=p.reader_text(common,'none','');typed=p.reader_text(common,'typed','SECRET_THRESHOLD')
  self.assertNotIn('SECRET_THRESHOLD',none);self.assertIn('SECRET_THRESHOLD',typed)
  self.assertIn(common,none);self.assertIn(common,typed)
 def test_costs_bind_independent_model_rates(self):
  self.assertAlmostEqual(p.cost(p.MODEL,1000,100),(.45*1000+1.335*100)/1e6)
  self.assertAlmostEqual(p.cost(p.READER,1000,100),(2.205*1000+5.49*100)/1e6)
 def test_missing_invalid_and_duplicate_responses_retained(self):
  with tempfile.TemporaryDirectory() as dd:
   root=Path(dd);gold=[]
   for task in ('text6008','text6011','text6034'):
    for k in (-1.,0.,.5,1.,1.5,2.):gold.append({'id':task+'|'+str(k),'task':task,'k':k,'top_token':' x','changed':False})
   g=root/'private_gold.jsonl';g.write_text(''.join(json.dumps(r)+'\n' for r in gold));m=root/'manifest.json';m.write_text(json.dumps({'files':{'private_gold.jsonl':p.sha(g)},'scope':'model-free test'}));paid=root/'paid';paid.mkdir()
   response={'request_id':'prose|'+gold[0]['id'],'model':p.READER,'upstream_invalid':True}
   (paid/'responses.jsonl').write_text(json.dumps(response)+'\n');report=p.evaluate(m,paid,root/'report.json')
   self.assertEqual(report['prose']['rows'],18);self.assertEqual(report['prose']['both_correct'],0);self.assertEqual(report['typed']['all6_joint_cases'],0)
   (paid/'responses.jsonl').write_text((json.dumps(response)+'\n')*2)
   with self.assertRaisesRegex(ValueError,'duplicate'):p.evaluate(m,paid,root/'bad.json')
if __name__=='__main__':unittest.main()
