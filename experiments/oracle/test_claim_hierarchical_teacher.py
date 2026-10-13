import unittest
import claim_hierarchical_teacher as h

class PolicyTest(unittest.TestCase):
 def fixture(self):
  full={'a':{(p,c) for p in range(3) for c in range(4)}}
  return full,{'a':{(0,0)}},{'a':{(p,c) for p in range(2) for c in range(4)}},{'a':(0,3,0,4)},{'a':1.}
 def test_protection_unreachable_replay_and_fresh_current(self):
  args=self.fixture();seen=[]
  def ev(m,label):seen.append({s:set(v) for s,v in m.items()});return [0]*4
  r=h.refine(*args,ev)
  self.assertIn((0,0),r['masks']['a']);self.assertTrue({(2,c) for c in range(4)}<=r['masks']['a'])
  self.assertEqual(h.replay(args[0],args[1],args[2],r['accepted']),r['masks'])
  self.assertLess(len(seen[-1]['a']),len(seen[1]['a']))
 def test_worst_edit_and_strict_full_start(self):
  args=self.fixture()
  r=h.refine(*args,lambda m,l:[0,0,0,1e-6])
  self.assertIsNone(r['masks']);self.assertFalse(r['attempts'])
  r=h.refine(*args,lambda m,l:[0]*4 if l=='full_start' else [0,0,0,.003])
  self.assertFalse(r['accepted']);self.assertEqual(r['masks'],args[0])
 def test_deterministic_cap_and_split_partition(self):
  full={'a':{(p,c) for p in range(8) for c in range(8)}};args=(full,{'a':set()},full,{'a':(0,8,0,8)},{'a':1.})
  ev=lambda m,l:[0]*4 if l=='full_start' else [.01]*4
  a=h.refine(*args,ev);b=h.refine(*args,ev)
  self.assertEqual(a,b);self.assertEqual(len(a['attempts']),32);self.assertTrue(a['cap_hit'])
  children=h.split_rectangle((0,8,0,8));self.assertEqual(sum((r[1]-r[0])*(r[3]-r[2]) for r in children),64)
 def test_invalid_prior_and_nonfinite_fail_closed(self):
  args=list(self.fixture());args[-1]={'a':.0001}
  with self.assertRaises(ValueError):h.refine(*args,lambda *a:[0]*4)
  with self.assertRaises(RuntimeError):h.refine(*self.fixture(),lambda *a:[float('nan')]*4)
 def test_tensor_physical_path_matches_reference_and_retains_unreachable(self):
  import torch
  full={'a':torch.ones(3,4,dtype=torch.bool)};protected={'a':torch.zeros_like(full['a'])};protected['a'][0,0]=True
  count=lambda m:{'reachable_component_placements':int(m['a'][:2].sum())}
  got=h.tensor_refine(('a',),full,protected,lambda *a:[0]*4,count,None,{'a':(0,1)},{'a':1.})
  self.assertTrue(got['masks']['a'][2].all());self.assertTrue(got['masks']['a'][0,0]);self.assertEqual(count(got['masks'])['reachable_component_placements'],1)
  self.assertTrue(full['a'].all())
  replay=full['a'].clone()
  for s,(p0,p1,c0,c1) in got['recipe']['accepted_rectangles']:replay[p0:p1,c0:c1] &= protected[s][p0:p1,c0:c1]
  self.assertTrue(torch.equal(replay,got['masks']['a']))
  args=self.fixture();args=list(args);args[3]={'a':(0,2,0,4)}
  ref=h.refine(*args,lambda *a:[0]*4)
  self.assertEqual(ref['masks']['a'],{tuple(x) for x in got['masks']['a'].nonzero().tolist()})
 def test_tensor_physical_cap_and_worst_edit(self):
  import torch
  full={'a':torch.ones(8,8,dtype=torch.bool)};protected={'a':torch.zeros_like(full['a'])}
  labels=[]
  def ev(m,label):labels.append(label);return [0]*4 if label=='full_start' else [0,0,0,.003]
  got=h.tensor_refine(('a',),full,protected,ev,lambda m:{'reachable_component_placements':int(m['a'].sum())},None,{'a':tuple(range(8))},{'a':1.})
  self.assertEqual(len(got['attempts']),32);self.assertEqual(len(labels),33);self.assertTrue(got['cap_hit']);self.assertFalse(got['accepted_sites']);self.assertTrue(torch.equal(got['masks']['a'],full['a']))
if __name__=='__main__':unittest.main()
