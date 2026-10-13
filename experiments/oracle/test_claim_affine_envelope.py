import unittest
from claim_affine_envelope import upper_envelope

class Envelope(unittest.TestCase):
 def test_parallel_tied_region_and_dominated_lines(self):
  e=upper_envelope([2,2,1],[0,0,0]);self.assertEqual(len(e['segments']),1);self.assertEqual(e['segments'][0]['tokens'],[0,1]);self.assertEqual(e['segments'][0]['winner'],0)
 def test_switch_and_endpoint_winners(self):
  e=upper_envelope([1,1,0],[-1,1,0]);self.assertEqual([s['winner'] for s in e['segments']],[0,1]);self.assertEqual(e['boundaries'][0]['winner'],0);self.assertEqual(e['boundaries'][-1]['winner'],1);self.assertEqual(e['boundaries'][1]['numerical_ties_1e_10'],[0,1])
 def test_line_that_only_ties_at_transition_not_false_region(self):
  e=upper_envelope([0,0,0],[-1,0,1]);self.assertEqual([s['winner'] for s in e['segments']],[0,2]);self.assertEqual(e['boundaries'][1]['numerical_ties_1e_10'],[0,1,2])
 def test_endpoint_tie_preserved_without_zero_width_interval(self):
  e=upper_envelope([0,-2],[0,-1]);self.assertEqual(len(e['segments']),1);self.assertEqual(e['boundaries'][0]['numerical_ties_1e_10'],[0,1])
 def test_invalid_coefficients_fail_closed(self):
  for b,s in [([0,float('nan')],[0,1]),([0,1],[0]),([0,1],[0,float('inf')])]:
   with self.assertRaises(ValueError):upper_envelope(b,s)

if __name__=='__main__':unittest.main()
