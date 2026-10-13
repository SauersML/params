import unittest
import claim_feasible_teacher as f
class PolicyTest(unittest.TestCase):
 def test_worst_edit_rejects_even_when_mean_passes_and_preserves_full(self):
  full={'a':frozenset((0,1)),'b':frozenset((0,1))};teacher={'a':frozenset((0,)),'b':frozenset((0,))}
  def count(m):return {'reachable_component_placements':sum(map(len,m.values()))}
  def evaluate(m,label):return [0,0,0,.003] if label=='replace_a' else [0,0,0,.0008]
  got=f.search_sites(('a','b'),full,teacher,evaluate,count)
  self.assertEqual(got['accepted_sites'],['b']);self.assertEqual(got['masks']['a'],full['a']);self.assertEqual(full['b'],frozenset((0,1)))
  self.assertEqual(got['attempts'][0]['status'],'direct_fidelity_rejection')
 def test_infeasible_literal_full_start_emits_no_target(self):
  got=f.search_sites(('a',),{'a':(1,2)},{'a':(1,)},lambda *a:[0,.002,0,0],lambda m:{'reachable_component_placements':sum(map(len,m.values()))})
  self.assertIsNone(got['masks']);self.assertEqual(got['attempts'],[])
 def test_unreachable_only_removal_is_not_credited(self):
  calls=[]
  def evaluate(m,label):calls.append(label);return [0]*4
  got=f.search_sites(('a',),{'a':(1,2)},{'a':(1,)},evaluate,lambda m:{'reachable_component_placements':1})
  self.assertEqual(calls,['full_start']);self.assertEqual(got['status'],'full_model_no_compression')
 def test_output_reachability_and_bound(self):
  self.assertEqual(f.reachable_positions('h.3.attn.q_proj',5,[4],4),(4,))
  self.assertEqual(f.reachable_positions('h.3.attn.k_proj',5,[4],4),tuple(range(5)))
  self.assertEqual(f.reachable_positions('h.2.mlp.down_proj',5,[4],4),tuple(range(5)))
  with self.assertRaises(ValueError):f.search_sites(('a','a'),{'a':1},{'a':2},None,None)
 def test_nonfinite_measurements_fail_closed(self):
  with self.assertRaises(RuntimeError):f.search_sites(('a',),{'a':1},{'a':2},lambda *a:[0,float('nan'),0,0],lambda *a:None)
 def test_strict_reconstruction_control_stops_before_proposals(self):
  labels=[]
  def evaluate(m,label):labels.append(label);return [0,0,0,1e-6]
  got=f.search_sites(('a',),{'a':(0,1)},{'a':(0,)},evaluate,lambda m:{'reachable_component_placements':sum(map(len,m.values()))},full_start_max=1e-7)
  self.assertEqual(labels,['full_start']);self.assertIsNone(got['masks']);self.assertEqual(got['attempts'],[])
if __name__=='__main__':unittest.main()
