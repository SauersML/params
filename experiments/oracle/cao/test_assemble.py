"""Future assembly preserves validation through outcome balancing, after text-ID remap."""
import copy
import json
import random
import sys
from types import SimpleNamespace
import pytest
import assemble

@pytest.mark.parametrize('category,bad_answer,good_answer,share',[
 ('edit','Unchanged.',"' altered'",.7),('swap','No.',"Yes, to ' token'.",.7),
 ('which',assemble.NONE,"' token' at position 1.",.3),('pair',assemble.NONE,"' token' at position 1.",.3)])
def test_validation_outcomes_never_select_membership_or_change_train_cap(category,bad_answer,good_answer,share):
 train=[{'text':1,'id':f't{i}','cat':category,'a':bad_answer} for i in range(9)]
 train+=[{'text':2,'id':f'g{i}','cat':category,'a':good_answer} for i in range(3)]
 validation=[{'text':50,'id':f'v{i}','cat':category,'a':bad_answer} for i in range(7)]
 rows=train+validation
 def bad(r):
  assert r['text']%50!=0, 'Validation answers must never enter capping predicate'
  return r['a']==bad_answer
 out=assemble.cap_training(rows,bad,share,random.Random(0))
 changed=copy.deepcopy(rows)
 for r in changed:
  if r['text']%50==0:r['a']=good_answer
 again=assemble.cap_training(changed,bad,share,random.Random(0))
 assert [r['id'] for r in out]==[r['id'] for r in again]
 assert [r['id'] for r in out if r['text']%50==0]==[r['id'] for r in validation]
 balanced=[r for r in out if r['text']%50]
 assert sum(bad(r) for r in balanced)==min(9,int(share/(1-share)*3))
 assert [r['id'] for r in balanced]==[r['id'] for r in assemble.cap(train,bad,share,random.Random(0))]


def test_main_remapped_ids_control_validation_and_pair_filter_is_separate(tmp_path,monkeypatch):
 class Tokenizer:
  def __call__(self,s,**kw):return {'input_ids':[1]}
  def decode(self,ids):return ' token'
 monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
 def edit(t,name):return {'text':t,'id':name,'cat':'edit','a':'Unchanged.','q':'Existing question','ctx':[1,1]}
 shard0=tmp_path/'g0.jsonl';shard1=tmp_path/'g1.jsonl';replay=tmp_path/'replay.jsonl';out=tmp_path/'out.jsonl'
 # raw25 is TRAIN-like, but becomes validation50 in shard0. raw0 in shard1 becomes TRAIN1.
 ambiguous={'text':50,'id':'ambiguous','cat':'pair','q':"Which matters more for the model's next token: 'x' or 'y'?",'a':assemble.NONE,'ctx':[1,1]}
 shard0.write_text(''.join(json.dumps(r)+'\n' for r in [edit(0,'v0'),edit(25,'v50'),ambiguous]))
 shard1.write_text(''.join(json.dumps(r)+'\n' for r in [edit(0,'t1'),edit(25,'t51')]))
 replay.write_text('');monkeypatch.setattr(sys,'argv',['assemble.py',str(out),str(shard0),str(shard1),'--replay',str(replay)])
 assemble.main();rows=[json.loads(x) for x in out.read_text().splitlines()]
 assert [(r['id'],r['text']) for r in rows]==[('v0',0),('v50',50)]
 # TRAIN all-unchanged category still caps to zero; validation survives caps;
 # ambiguous legacy pair is still removed by the independent normalization filter.
