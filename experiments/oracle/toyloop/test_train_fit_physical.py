import copy,unittest
from train_fit_physical import CASES,validate_scope
class PhysicalScopeTests(unittest.TestCase):
 def fixture(self):
  counts=(76,156,84,116,116)
  return dict(worlds=548,forward_cap=2192,cases=[dict(id=c,inputs=[[1,2]]*4,reference_key_prefix='p_',panel=[['neuron',0,i//2,0. if i%2==0 else .5] for i in range(n//4-1)]) for c,n in zip(CASES,counts)])
 def test_exact_original_world_scope(self):validate_scope(self.fixture())
 def test_coverage_and_duplicate_rejected(self):
  for mutate in (lambda c:c.update(forward_cap=4384),lambda c:c['cases'].reverse(),lambda c:c['cases'][0]['panel'].append(c['cases'][0]['panel'][0]),lambda c:c['cases'][0]['inputs'][0].append(-1)):
   c=copy.deepcopy(self.fixture());mutate(c)
   with self.assertRaises(RuntimeError):validate_scope(c)
class NumericGuardTests(unittest.TestCase):
 def test_nonfinite_kl_rejected(self):
  from train_fit_physical import checked_kl
  for value in (float('nan'),float('inf')):
   with self.assertRaises(RuntimeError):checked_kl(value)
  self.assertEqual(checked_kl(-1e-12),-1e-12)
 def test_distribution_normalization_and_nan(self):
  import torch
  from train_fit_physical import checked_distribution
  good=torch.log_softmax(torch.zeros(1,1,32),dim=-1)
  checked_distribution(good,32)
  for bad in (good+1,good.reshape(1,32),good*float('nan')):
   with self.assertRaises(RuntimeError):checked_distribution(bad,32)
