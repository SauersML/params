"""Regression for the hidden-live-lookup input failure; no SVD/model loads."""
import re
import unittest
from unittest.mock import patch
import torch
import render

class ObservabilityTest(unittest.TestCase):
 def fixture(self,n=9):
  d=32; vocab=['BOS']+[f't{i}' for i in range(1,n+2)]
  E=torch.zeros(len(vocab),d); E[:,0]=8.
  for i in range(len(vocab)): E[i,i+1]=1.
  sd={'wte.weight':E,'ln_f.weight':torch.ones(d)}
  for name in ('rms_1','rms_2'): sd[f'h.0.{name}.weight']=torch.ones(d)
  for name in ('q_proj','k_proj','v_proj','o_proj'): sd[f'h.0.attn.{name}.weight']=torch.zeros(d,d)
  fc=torch.zeros(n,d); down=torch.zeros(d,n)
  for i in range(n): fc[i,i+2]=4.; down[i+3,i]=3. if i<n-1 else .1
  sd['h.0.mlp.c_fc.weight']=fc; sd['h.0.mlp.down_proj.weight']=down
  return render.Toy('lookup',sd,{'n_layer':1,'n_head':2,'n_embd':d,'n_ctx':32},vocab,[])
 def test_default_preserved_and_every_live_lookup_observable(self):
  toy=self.fixture()
  with patch.object(render,'SECTIONS','all'):
   default=render.render(toy,n_texts=0); explicit=render.render(toy,n_texts=0,all_live_neurons=False)
   full=render.render(toy,n_texts=0,all_live_neurons=True)
  self.assertEqual(default,explicit)
  self.assertIn('9 live neurons, strongest 6',default)
  self.assertNotIn('L0N8 w',default)
  self.assertIn('9 live neurons, all 9',full)
  self.assertEqual(set(re.findall(r'^  (L0N\d+) w',full,re.M)),{f'L0N{i}' for i in range(9)})
  self.assertRegex(full,r'L0N8 w.*reads t9:.*writes t10:')
 def test_compact_mode_no_silent_neuron_count_clip(self):
  toy=self.fixture(18)
  with patch.object(render,'SECTIONS','weights'):
   clipped=render.render(toy,n_texts=0); full=render.render(toy,n_texts=0,all_live_neurons=True)
  self.assertIn('weaker live neurons not shown',clipped)
  self.assertNotIn('weaker live neurons not shown',full)
  self.assertEqual(set(re.findall(r'^(L0N\d+) reads',full,re.M)),{f'L0N{i}' for i in range(18)})
 def test_nonboolean_optin_rejected(self):
  with self.assertRaises(ValueError): render.render(self.fixture(),all_live_neurons='1')
if __name__=='__main__': unittest.main()
