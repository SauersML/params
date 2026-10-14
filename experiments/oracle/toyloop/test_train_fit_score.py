import unittest
import circuit
import train_fit_score as s

class TrainFitScoreTests(unittest.TestCase):
    def test_fit_separates_units_from_bridge(self):
        target='L0H0 --> L1H0\nL1H0 --> out'
        same=s.target_fit(target+'\nEnglish: prose',target,'stop',parser=circuit.parse)
        self.assertTrue(same['exact_units']);self.assertTrue(same['exact_edges'])
        wrong=s.target_fit('L0H0 --> out\nL1H0 --> out',target,'stop',parser=circuit.parse)
        self.assertTrue(wrong['exact_units']);self.assertFalse(wrong['exact_edges'])
        self.assertEqual(wrong['missing_edges'],[('L0H0','L1H0')])
    def test_invalid_first_response_zero_credit(self):
        for completion,finish in (('L0H0 --> out','length'),(None,'missing'),('L0MLP --> out','stop')):
            row=s.target_fit(completion,'L0H0 --> out',finish,parser=circuit.parse)
            self.assertFalse(row['parse_valid']);self.assertEqual(row['edge_f1'],0)
    def test_tokens_are_not_component_supervision(self):
        row=s.target_fit('tokOTHER --> L0H0\nL0H0 --> out','tok0 --> L0H0\nL0H0 --> out','stop',parser=circuit.parse)
        self.assertTrue(row['exact_units']);self.assertTrue(row['exact_edges'])
