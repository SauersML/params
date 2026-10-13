import unittest
from factorial_report import summarize,contrasts,CELLS
class ReportTest(unittest.TestCase):
 def row(self,valid=True):
  return {'valid':valid,'stated_reward':.25,'same_units_complete_reward':.75,'component_edge_f1':1.,'component_edge_precision':1.,'component_edge_recall':1.,'correct_executed_component_edges':[],'valid_executed_component_edges':[],'gold_component_edges':[],'stated':{'kl_bits':2.,'per_text':[{'top':1,'reference_top':1}]}}
 def test_invalids_stay_in_primary_denominators(self):
  result=summarize([self.row(),self.row(False)])
  self.assertEqual(result['reward'],-.375); self.assertEqual(result['edge_f1'],.5)
  self.assertEqual(result['top1_invalid_zero'],.5); self.assertEqual(result['kl_valid_only'],2.)
  self.assertEqual(result['valid'],1)
 def test_zero_edge_success_does_not_imply_physical_success(self):
  result=summarize([self.row()]); self.assertEqual(result['edge_f1'],1.)
  self.assertEqual(result['correct_edges'],0); self.assertEqual(result['reward'],.25)
 def test_factorial_signs_and_interaction(self):
  cells={name:{'r':v} for name,v in zip(CELLS,(0.,.3,.6,.8))}
  result=contrasts(cells,'r'); self.assertAlmostEqual(result['interaction'],-.1)
  self.assertAlmostEqual(result['support_main_effect'],.55)
if __name__=='__main__': unittest.main()
