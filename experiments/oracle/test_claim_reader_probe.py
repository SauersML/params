import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import claim_reader_probe as probe,claim_reader_evaluate as ev
class Tests(unittest.TestCase):
 def test_strict_first_finish(self):
  row={'completion':'{"top_token":" x"}','finish':'stop'}
  self.assertEqual(ev.prediction(row),(' x',None));self.assertEqual(ev.prediction(dict(row,finish='StopReason.stop')),(' x',None))
  for f in ('length','StopReason.length',None,'error'):self.assertEqual(ev.prediction(dict(row,finish=f))[1],'unfinished')
  self.assertEqual(ev.prediction(dict(row,parse_error='failure'))[1],'parse_error')
 def test_frozen_prepare_dedup_and_no_outcomes_in_request(self):
  class Renderer:
   def build_generation_prompt(self,m):return SimpleNamespace(length=len(m[0]['content']),to_ints=lambda:list(m[0]['content'].encode()))
  with tempfile.TemporaryDirectory() as dd:
   d=Path(dd);data=d/'data';data.mkdir();tok=d/'tokenizer';tok.mkdir();(tok/'config').write_text('{}');prices=d/'prices';prices.write_text(json.dumps([{'tinker_id':probe.MODEL,'prefill':'$0.45','cached_prefill':'$0.09','sample':'$1.335'}]))
   for view in probe.VIEWS:
    rows=[]
    for case in range(14):
     for arm in ('original','permutation0','permutation1'):rows.append({'prompt':[{'role':'user','content':str(case)+(arm if view==probe.VIEWS[0] else '')}],'completion':[{'content':'PRIVATEGOLD'}],'info':{'case_id':str(case),'condition_private':arm,'split':'internal_dev','task':('text6136','text6012','text6045')[case%3]}})
    (data/(view+'_internal_dev_matched_fit.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
   with patch.object(probe,'configure',return_value=Renderer()):m=probe.prepare(data,tok,d/'prep',prices)
   self.assertEqual(m['unique_calls'],56);self.assertEqual(len(m['bindings']),84);self.assertNotIn('PRIVATEGOLD',(d/'prep/requests.jsonl').read_text())
   with self.assertRaisesRegex(ValueError,'unexpected/duplicate'):
    paid=d/'paid';paid.mkdir();(paid/'responses.jsonl').write_text(json.dumps({'request_id':'unexpected'})+'\n');ev.evaluate(d/'prep/manifest.json',paid,d/'report')
if __name__=='__main__':unittest.main()
