import unittest
import audit_compositional_counterfactual as a
class AuditTest(unittest.TestCase):
 def test_baseline_binding_is_exact_even_if_normalized(self):
  import torch
  p=torch.log_softmax(torch.tensor([[[1.,2.]]]),-1)
  a.baseline_identity({'p0':p.clone(),'q0_two':p.clone()},{'p0':p,'q0_two':p})
  altered=torch.log_softmax(torch.tensor([[[2.,1.]]]),-1)
  with self.assertRaises(ValueError):a.baseline_identity({'p0':p,'q0_two':altered},{'p0':p,'q0_two':p})
 def test_nonfinite_unnormalized_and_shape_fail(self):
  import torch
  p=torch.log_softmax(torch.tensor([[[1.,2.]]]),-1)
  self.assertAlmostEqual(a.raw_kl(p,p),0.)
  for q in (p+1,p[...,0],p*float('nan')):
   with self.assertRaises(ValueError):a.raw_kl(p,q)
 def test_reported_metric_mismatch_fails(self):
  a.close(.1,.1)
  with self.assertRaises(ValueError):a.close(.1,.11)
if __name__=='__main__':unittest.main()
