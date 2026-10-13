import unittest
import numpy as np
from edit_affine import propose
class Proposals(unittest.TestCase):
 def test_raise_is_full_vocab_not_only_original_winner(self):
  r=propose([2.,1.,0.],[0.,2.,5.],'raise',1)
  self.assertIsNone(r['k']) # target loses to token2 before it can beat token0
 def test_fp16_inside_positive_margin_and_closest_to_one(self):
  b=np.array([2.,1.,0.]);s=np.array([0.,2.,0.]);r=propose(b,s,'raise',1);k=r['k'];self.assertIsNotNone(k);self.assertEqual(float(np.float16(k)),k);self.assertGreater((b+(k-1)*s)[1]-2.,1e-4)
  smaller=float(np.nextafter(np.float16(k),np.float16(-np.inf)));self.assertLessEqual((b+(smaller-1)*s)[1]-2.,1e-4)
 def test_flip_union_selects_closest_feasible_side(self):
  r=propose([2.,1.,0.],[0.,2.,-10.],'flip',0);self.assertLess(r['k'],1);self.assertEqual(r['predicted_winner'],2)
 def test_no_feasible_zero_slope_and_bounds(self):
  for kind,token in [('raise',1),('flip',0)]:self.assertIsNone(propose([2.,1.],[0.,0.],kind,token)['k'])
  self.assertIsNone(propose([2.,1.],[0.,.1],'raise',1)['k'])
 def test_invalid_numbers_rejected(self):
  for b,s in [([float('nan'),1],[0,2]),([2,1],[0,float('inf')])]:
   with self.assertRaises(ValueError):propose(b,s,'raise',1)
if __name__=='__main__':unittest.main()

class TeacherOptIn(unittest.TestCase):
 def test_default_grid_unchanged_and_persistent_helper_opt_in(self):
  import sys
  from types import SimpleNamespace
  from unittest.mock import patch
  import edit_eval
  node=('h.3.mlp.down_proj',4,7);calls=[]
  extra={'op':'scale','node':(node[0],None,7),'k':1.5,'persistent':True}
  helper=SimpleNamespace(candidates=lambda *args:calls.append(args) or [extra])
  ee=SimpleNamespace(nat=SimpleNamespace(has_importance=False));task={'id':'text6001','prompts':[{'token_ids':[1]*5,'target_positions':[4]}]};goal={'kind':'raise','token_id':2}
  with patch.object(edit_eval,'teacher_nodes',return_value=[node]),patch.object(edit_eval.native,'all_edges',return_value=({},[])),patch.dict(sys.modules,{'edit_affine':helper}):
   original=edit_eval.candidates(ee,task,goal)
   self.assertEqual([o['k'] for o in original],[0.,-1.,2.]);self.assertEqual(calls,[])
   augmented=edit_eval.candidates(ee,task,goal,analytic_final_down=True)
   self.assertEqual(augmented,original+[extra]);self.assertEqual(len(calls),1)
