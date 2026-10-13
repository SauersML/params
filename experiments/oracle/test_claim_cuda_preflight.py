import unittest
from types import SimpleNamespace
from unittest.mock import patch
import claim_cuda_preflight as c
class ContractTest(unittest.TestCase):
 def test_partition_all_twelve_worlds_without_new_tasks(self):
  a=c.tasks_of('text6008,text6011');b=c.tasks_of('text6034')
  self.assertFalse(set(a)&set(b));self.assertEqual(set(a+b),set(c.SELECTED));self.assertEqual(4*3*len(a+b),36)
 def test_unknown_duplicate_empty_fail_closed(self):
  for value in ('','text6008,text6008','text6008,text9999'):
   with self.assertRaises(ValueError):c.tasks_of(value)
 def test_transfer_preserves_tensor_handles_and_clears_cpu_caches(self):
  class Tensor:
   def __init__(self):self.moves=[]
   def to(self,device):self.moves.append(device);return self
  pieces={k:Tensor() for k in ('U','V','delta_T')};site=SimpleNamespace(**pieces)
  target=SimpleNamespace(to=lambda d:None,site=lambda n:site)
  nat=SimpleNamespace(target=target,names=['s'],vpd=SimpleNamespace(clear=lambda:None),_refs={'old':1},_refs_bytes=1,_consts={'cpu':1},_lengths={'cpu':1},_rec=('old',),dev='cpu')
  with patch.dict('sys.modules',{'torch':SimpleNamespace()}),patch.object(c,'hashes',return_value={'basis':'exact'}):
   self.assertEqual(c.transfer(nat),{'basis':'exact'})
  self.assertEqual(nat.dev,'cuda');self.assertFalse(nat._refs);self.assertFalse(nat._consts);self.assertFalse(nat._lengths);self.assertEqual(nat._refs_bytes,0)
  self.assertTrue(all(x.moves==['cuda'] for x in pieces.values()))
  with patch.dict('sys.modules',{'torch':SimpleNamespace()}),patch.object(c,'hashes',side_effect=[{'basis':'old'},{'basis':'changed'}]):
   with self.assertRaises(RuntimeError):c.transfer(nat)
if __name__=='__main__':unittest.main()
