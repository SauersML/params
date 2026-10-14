import copy,unittest
from unittest.mock import patch
import torch
import render

class ResidualWriteTests(unittest.TestCase):
 def fixture(self,read=0):
  d=64;sd={'wte.weight':torch.zeros(32,d),'ln_f.weight':torch.zeros(d)};sd['wte.weight'][:,0]=6
  for i in range(32):sd['wte.weight'][i,i+1]=1
  sd['ln_f.weight'][1:33]=5
  for layer in range(2):
   for k in ('rms_1','rms_2'):sd[f'h.{layer}.{k}.weight']=torch.ones(d)
   for site,shape in (('attn.q_proj',(d,d)),('attn.k_proj',(d,d)),('attn.v_proj',(d,d)),('attn.o_proj',(d,d)),('mlp.c_fc',(256,d)),('mlp.down_proj',(d,256))):sd[f'h.{layer}.{site}.weight']=torch.zeros(*shape)
  for h in (0,1):sd['h.0.attn.v_proj.weight'][h*16,read]=1;sd['h.0.attn.o_proj.weight'][34,h*16]=4
  return render.Toy('fixture',sd,dict(n_layer=2,n_head=4,n_embd=64,n_ctx=32,rms_norm_eps=1e-6,rotary_base=10000.),['BOS']+[f't{i}' for i in range(1,32)],[[0,1,2]]*4)
 def test_frozen_summary_collision_distinguished_without_forward(self):
  a,b=self.fixture(),self.fixture(63)
  def forbidden(*args,**kwargs):raise AssertionError('model forward forbidden')
  with patch.object(render,'SECTIONS','weights'),patch.object(render,'run',forbidden):
   before=render.render(a,all_live_neurons=True,explain_summary_limits=True)
   self.assertEqual(before,render.render(b,all_live_neurons=True,explain_summary_limits=True))
   self.assertEqual(before,render.render(a,all_live_neurons=True,explain_summary_limits=True,residual_write_hints=False))
   newa=render.render(a,all_live_neurons=True,explain_summary_limits=True,residual_write_hints=True)
   newb=render.render(b,all_live_neurons=True,explain_summary_limits=True,residual_write_hints=True)
   self.assertNotEqual(newa,newb);self.assertIn('not actual activation or causal importance',newa)
  self.assertGreater(render.isolated_residual_writes(a)['L0H0']['mean_l2'],30)
  self.assertEqual(render.isolated_residual_writes(b)['L0H0']['max_token_l2'],0)
 def test_reciprocal_value_output_gauge_preserves_features(self):
  a=self.fixture();b=copy.deepcopy(a);b.sd['h.0.attn.v_proj.weight']*=2;b.sd['h.0.attn.o_proj.weight']/=2
  self.assertEqual(render.residual_write_lines(a),render.residual_write_lines(b))
 def test_zero_control_and_norms_do_not_certify_cancellation(self):
  a=self.fixture();opposite=copy.deepcopy(a);opposite.sd['h.0.attn.o_proj.weight'][:,16:32]*=-1
  facts=render.isolated_residual_writes(opposite)
  self.assertEqual(facts['L0H1'],facts['L0H0'])
  self.assertEqual(facts,render.isolated_residual_writes(a))
  self.assertIn('not contextual activations, cancellation certificates',render.residual_write_lines(a)[0])
  zero=copy.deepcopy(a);zero.sd['h.0.attn.o_proj.weight'][:,0:16]=0
  self.assertEqual(render.isolated_residual_writes(zero)['L0H0']['max_token_l2'],0)
 def test_head_permutation_transports_source_statistics(self):
  a=self.fixture();b=copy.deepcopy(a);perm=[2,0,3,1]
  b.sd['h.0.attn.v_proj.weight']=a.sd['h.0.attn.v_proj.weight'].reshape(4,16,64)[perm].reshape(64,64)
  b.sd['h.0.attn.o_proj.weight']=a.sd['h.0.attn.o_proj.weight'].reshape(64,4,16)[:,perm].reshape(64,64)
  old,new=render.isolated_residual_writes(a),render.isolated_residual_writes(b)
  for i in range(4):
   self.assertEqual(new[f'L0H{i}'],old[f'L0H{perm[i]}'])

 def test_chunked_rows_and_float64_statistics(self):
  import math
  a=self.fixture();large=copy.deepcopy(a);large.sd['wte.weight']=a.sd['wte.weight'].repeat(17,1)
  self.assertEqual(render.isolated_residual_writes(a),render.isolated_residual_writes(large))
  large.sd['h.0.attn.o_proj.weight']*=1e20
  value=render.isolated_residual_writes(large)['L0H0']['max_token_l2']
  self.assertTrue(math.isfinite(value));self.assertGreater(value,1e20)

if __name__=='__main__':unittest.main()
