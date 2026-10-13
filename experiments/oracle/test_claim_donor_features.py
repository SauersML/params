import unittest
from types import SimpleNamespace
import torch
import claim_donor_features as d
class Tests(unittest.TestCase):
 def test_nonunit_zero_cancel_reconstruction(self):
  u=torch.tensor([[3.,4.],[-3.,-4.],[0.,0.]])
  a0=torch.tensor([5.,5.,0.]);a1=torch.tensor([10.,10.,0.])
  self.assertTrue(torch.equal(d.delta_write(u,a0,a1),torch.zeros(2)))
  expected=(u*torch.tensor([2.,1.,7.])[:,None]).sum(0)-(u*torch.tensor([1.,1.,2.])[:,None]).sum(0)
  self.assertTrue(torch.equal(d.delta_write(u,a0,torch.tensor([10.,5.,0.])),expected))
 def test_static_orientation_and_token_mapping(self):
  W=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
  target=SimpleNamespace(site=lambda n:SimpleNamespace(W=W),ln_f=torch.tensor([2.,3.]),wte=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,0.],[0.,-1.]]))
  u=torch.tensor([2.,-1.,3.]);logits,scope=d.project(u,'h.0.mlp.c_fc',target)
  expected=torch.tensor([2.-2.+9.,8.-5.+18.])*target.ln_f
  self.assertTrue(torch.equal(logits,expected@target.wte.T));self.assertIn('GELU',scope)
  self.assertIsNone(d.project(torch.tensor([1.,2.]),'h.0.attn.q_proj',target)[0])
  tk=SimpleNamespace(decode=lambda ids:'native:'+str(ids[0]));e=d.extremes(logits,tk,k=2)
  for row in e['highest']+e['lowest']:self.assertEqual(row['token'],'native:'+str(row['native_id']))
if __name__=='__main__':unittest.main()
