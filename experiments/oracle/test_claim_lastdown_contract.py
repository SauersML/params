import unittest
import torch
from claim_lastdown_contract import exact_pair_effect
class Contract(unittest.TestCase):
 def test_nonunit_scalar_matches_independent_softmax_pair(self):
  r=torch.tensor([2.,-1.,3.]);u=torch.tensor([3.,4.,0.]);ln=torch.tensor([.5,2.,1.]);E=torch.tensor([[1.,0.,2.],[0.,-1.,1.],[2.,3.,-1.]]);delta=.75
  q0=torch.log_softmax((r/(r.square().mean()+1e-5).sqrt()*ln)@E.T,0);z=r+delta*u;q1=torch.log_softmax((z/(z.square().mean()+1e-5).sqrt()*ln)@E.T,0)
  effect=exact_pair_effect(r,u,delta,ln*(E[0]-E[1]),1e-5)
  self.assertAlmostEqual(effect['base_pair_logodds'],float(q0[0]-q0[1]),places=6);self.assertAlmostEqual(effect['patched_pair_logodds'],float(q1[0]-q1[1]),places=6)
 def test_normalization_can_reverse_static_change_direction(self):
  effect=exact_pair_effect(torch.tensor([1.,2.]),torch.tensor([-.01,-1.]),.5,torch.tensor([1.,0.]),1e-5)
  self.assertLess(effect['delta_numerator'],0);self.assertGreater(effect['patched_pair_logodds'],effect['base_pair_logodds']);self.assertGreater(effect['normalizer_term'],abs(effect['direct_term']))
 def test_zero_unit_is_exact_no_change(self):
  effect=exact_pair_effect(torch.tensor([2.,3.]),torch.zeros(2),3.,torch.tensor([1.,-1.]),1e-5)
  self.assertEqual(effect['base_pair_logodds'],effect['patched_pair_logodds']);self.assertEqual(effect['delta_numerator'],0.)
if __name__=='__main__':unittest.main()
