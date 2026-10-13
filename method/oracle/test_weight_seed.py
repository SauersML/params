"""Static proposal and immutable measured-gate regressions; no model loads."""
import unittest
import weight_seed as s

def nd(layer,kind,index=0): return (f'h.{layer}.{ "attn" if kind.endswith("proj") and kind!="down_proj" else "mlp"}.{kind}',0,index)

class SeedTest(unittest.TestCase):
 def test_mandatory_architecture_and_signed_rank(self):
  writer=nd(0,'o_proj');q=nd(1,'q_proj');k=nd(1,'k_proj');v=nd(1,'v_proj');fc=nd(1,'c_fc');down=nd(1,'down_proj');o=nd(1,'o_proj')
  nodes=[writer,q,k,v,fc,down,o];edges=[(writer,q),(writer,k),(v,o),(fc,down),(o,None),(writer,v),(writer,fc)]
  ranked=[{'writer':writer,'reader':v,'cosine':-.8,'raw':-2.},{'writer':writer,'reader':fc,'cosine':0.,'raw':0.}]
  mask,meta=s.propose(nodes,edges,ranked)
  self.assertEqual(mask,(True,True,True,True,True,True,False))
  self.assertEqual(meta['signed_couplings'][0]['cosine'],-.8)
  self.assertEqual(meta['signed_couplings'][0]['raw'],-2.)
 def test_k_cap_protection_and_node_skip(self):
  w=nd(0,'o_proj'); readers=[nd(1,'v_proj',i) for i in range(40)];edges=[(w,r) for r in readers]
  ranked=[{'writer':w,'reader':r,'cosine':-(i+1)/40} for i,r in enumerate(readers)]
  protected=[i==0 for i in range(40)];mask,meta=s.propose([w]+readers,edges,ranked,protected)
  self.assertEqual(sum(mask),32);self.assertTrue(mask[0]);self.assertTrue(mask[-1]);self.assertFalse(mask[1])
  self.assertEqual(meta['residual_kept'],32)
  self.assertIsNone(s.propose([nd(0,'v_proj',i) for i in range(129)],[],[])[0])
  self.assertIsNone(s.propose([w]+readers,edges,ranked,[True]*40)[0])
 def test_poor_seed_cannot_reset_original_floor_or_budget(self):
  full=(True,True);candidate=(True,False);reference=.96;floor=min(.9,reference);calls=[]
  def admission(mask):
   calls.append(mask);return {'changed':.4} if .4>=floor else None
  chosen,row=s.assess(full,candidate,.1,lambda mask:.05,admission)
  self.assertEqual(chosen,full);self.assertFalse(row['admitted']);self.assertEqual(row['reason'],'original_acceptance_floors');self.assertEqual(floor,.9)
  chosen,row=s.assess(full,candidate,.1,lambda mask:3.,lambda mask:self.fail('KL failure must stop before acceptance'))
  self.assertEqual(chosen,full);self.assertEqual(row['original_budget'],.1)
 def test_success_and_nonfinite_measurement(self):
  full=(True,True);candidate=(True,False)
  chosen,row=s.assess(full,candidate,.1,lambda mask:.01,lambda mask:{'changed':.95})
  self.assertEqual(chosen,candidate);self.assertTrue(row['admitted'])
  self.assertEqual(s.assess(full,candidate,.1,lambda mask:float('nan'))[0],full)
 def test_unknown_duplicate_and_unexpanded_inputs_rejected(self):
  w=nd(0,'o_proj');v=nd(1,'v_proj');edges=[(w,v)]
  with self.assertRaises(ValueError):s.propose([w,v],edges,[{'writer':w,'reader':nd(2,'v_proj'),'cosine':.9}])
  with self.assertRaises(ValueError):s.propose([w,v],edges,[{'writer':w,'reader':v,'cosine':.9}]*2)
  with self.assertRaises(ValueError):s.propose([nd(1,'v_proj',-1)],[],[])

class LoadedSeedTest(unittest.TestCase):
 def test_loaded_signed_norm_gain_and_fingerprint(self):
  import torch
  from types import SimpleNamespace
  w,r=nd(0,'o_proj'),nd(1,'v_proj')
  sites={w[0]:SimpleNamespace(U=torch.tensor([[1.,0.]])),r[0]:SimpleNamespace(V=torch.tensor([[-2.],[1.]]))}
  target=SimpleNamespace(site=lambda name:sites[name],norms=[torch.ones(2),torch.ones(2),torch.tensor([1.,0.]),torch.ones(2)])
  ranked,proof=s.rank_loaded(SimpleNamespace(target=target),[(w,r)])
  self.assertEqual(ranked[0]['cosine'],-1.)
  self.assertEqual(ranked[0]['raw'],-2.)
  sites[w[0]].U[0,0]=2.
  _,changed=s.rank_loaded(SimpleNamespace(target=target),[(w,r)])
  self.assertNotEqual(proof,changed)
 def test_initialiser_rejects_poor_seed_without_replacing_full_start(self):
  import torch
  from types import SimpleNamespace
  from unittest.mock import patch
  w,r=nd(0,'o_proj'),nd(1,'v_proj');edges=[(w,r),(r,None)]
  class EP:
   def __init__(self):
    self.ge=SimpleNamespace(nat=None);self.edges=edges;self.on=torch.ones(2,dtype=torch.bool);self.protected=torch.zeros(2,dtype=torch.bool);self.measured=0
   def kl_sets(self,masks):
    self.measured+=len(masks);return [.01]*len(masks)
  ep=EP();full=ep.on;seen=[];floor=.9
  def admit(masks):
   self.assertIs(ep.on,full);seen.append(masks.tolist());return [None] # .4 fails ORIGINAL .9, not reset to .4.
  with patch.object(s,'rank_loaded',return_value=([{'writer':w,'reader':r,'cosine':0.,'raw':0.}],{})):
   row=s.initialise(ep,[w,r],.1,admit,{'changed':.95},.001)
  self.assertIs(ep.on,full);self.assertTrue(bool(ep.on.all()));self.assertFalse(row['admitted']);self.assertEqual(row['original_reference_terms'],{'changed':.95});self.assertEqual(row['original_budget'],.1);self.assertEqual(row['measured_subsets'],1);self.assertEqual(seen,[[[False,True]]])
  self.assertEqual(s.initialise(ep,[nd(0,'v_proj',i) for i in range(129)],.1)['reason'],'node_cap')

if __name__=='__main__':unittest.main()
