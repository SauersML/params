import unittest
import torch
from claim_joint_contract import algebra
class Joint(unittest.TestCase):
 def test_aligned_and_canceling_cross_terms_match_independent_dense(self):
  r=torch.tensor([1.,2.]);read=torch.tensor([1.,0.]);d=torch.tensor([.5,0.])
  for second,expectedcross in ((d,.5),(-d,-.5)):
   a=algebra(r,[d,second],read,1e-5);z=r+d+second;direct=float((z@read)/(z.square().mean()+1e-5).sqrt())
   self.assertAlmostEqual(a['joint_gap'],direct,places=6);self.assertAlmostEqual(a['cross_term_2_dot_d1_d2'],expectedcross,places=6);self.assertGreater(abs(a['joint_gap']-a['additive_single_effect_gap']),.01);self.assertGreater(abs(a['joint_gap']-a['no_cross_RMS_gap']),.01)
 def test_invalid_no_cross_approximation_is_retained_not_clamped(self):
  a=algebra(torch.tensor([1.,0.]),[torch.tensor([-.75,0.])]*2,torch.tensor([1.,0.]),1e-5)
  self.assertIsNone(a['no_cross_RMS_gap']);self.assertLess(a['no_cross_RMS_radicand'],0)
if __name__=='__main__':unittest.main()
