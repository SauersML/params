import unittest
import claim_cuda_preflight as c
class ContractTest(unittest.TestCase):
 def test_partition_all_twelve_worlds_without_new_tasks(self):
  a=c.tasks_of('text6008,text6011');b=c.tasks_of('text6034')
  self.assertFalse(set(a)&set(b));self.assertEqual(set(a+b),set(c.SELECTED));self.assertEqual(4*3*len(a+b),36)
 def test_unknown_duplicate_empty_fail_closed(self):
  for value in ('','text6008,text6008','text6008,text9999'):
   with self.assertRaises(ValueError):c.tasks_of(value)
if __name__=='__main__':unittest.main()
