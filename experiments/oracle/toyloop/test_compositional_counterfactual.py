"""No-model regression for complete coverage and invalid-world denominators."""
import unittest
from types import SimpleNamespace
import compositional_counterfactual as c

class GridTest(unittest.TestCase):
 def grid(self):
  draws=[{'case':case,'input':i,'edit':{'layer':l,'head':h,'scale':s}} for case in c.CASES for i in range(8) for l in range(2) for h in range(4) for s in (0.,.5)]
  baselines=[{'case':case,'input':i} for case in c.CASES for i in range(8)]
  restores=[{'case':case,'layer':l,'head':h,'scale':s} for case in c.CASES for l in range(2) for h in range(4) for s in (0.,.5)]
  return draws,baselines,restores
 def test_cartesian_grid_rejects_duplicate_at_same_count(self):
  rows,b,r=self.grid();c.verify_grid(rows,b,r)
  for which in (0,1,2):
   parts=[list(rows),list(b),list(r)];parts[which][-1]=parts[which][0]
   with self.assertRaises(ValueError):c.verify_grid(*parts)
 def test_invalids_remain_in_top_denominator_and_delete_half_separate(self):
  valid={'valid':True,'direct_kl_bits':.2,'sparse_nochange_kl_bits':.5,'top':1}
  bad={'valid':False,'direct_kl_bits':None,'sparse_nochange_kl_bits':None,'top':None}
  rows=[{'edit':{'scale':s},'active_damage':True,'reference_top':1,'dense_nochange_kl_bits':.6,'candidates':{a:v for a in (*c.ARMS,'gt')}} for s,v in ((0.,valid),(.5,bad))]
  result=c.summarize(rows)
  self.assertEqual(result['all']['arms']['two']['n'],2)
  self.assertEqual(result['all']['arms']['two']['kl_denominator'],1)
  self.assertEqual(result['all']['arms']['two']['top_accuracy_all'],.5)
  self.assertEqual(result['active_delete']['arms']['two']['top_correct'],1)
  self.assertEqual(result['active_half']['arms']['two']['invalid'],1)
 def test_graph_fingerprint_catches_actual_topology_change(self):
  g=SimpleNamespace(nodes={('h.0.attn.v_proj',0,0),('h.0.attn.o_proj',0,0)},parents={('h.0.attn.o_proj',0,0):[('h.0.attn.v_proj',0,0)]},out=[('h.0.attn.o_proj',0,0)],complete=False)
  old=c.graph_hash(g);g.parents={};self.assertNotEqual(old,c.graph_hash(g))

if __name__=='__main__':unittest.main()
