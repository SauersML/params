"""Frozen design invariants; native integrity is checked by the free exporter."""
import unittest
from route_transfer import design, SEEDS

class RouteDesignTest(unittest.TestCase):
    def test_matchings_share_nodes_and_inputs_but_not_bridges(self):
        for seed in SEEDS:
            a,b=design(seed),design(seed,True)
            self.assertEqual(a['bank'],b['bank'])
            self.assertEqual(a['pool'],b['pool'])
            self.assertEqual(a['writers'],b['writers'])
            self.assertEqual(a['readers'],b['readers'])
            self.assertEqual({tuple(e) for e in a['complete']},{tuple(e) for e in b['complete']})
            self.assertEqual(a['correct'],b['wrong'])
            self.assertEqual(a['wrong'],b['correct'])
            self.assertEqual(len(a['correct']),4)
            self.assertEqual(len(a['complete']),6)
    def test_permutation_preserves_bank_and_renames_all_edges(self):
        for seed in SEEDS:
            for swap in (False,True):
                a,b=design(seed,swap),design(seed,swap,True)
                rename=lambda n: 'out' if n=='out' else f'L{n[1]}H{(int(n[-1])+(1 if n[1]=="0" else 2))%4}'
                self.assertEqual(a['bank'],b['bank'])
                for label in ('correct','wrong','complete'):
                    self.assertEqual([[rename(x) for x in e] for e in a[label]],b[label])
                for ids in a['bank']:
                    self.assertEqual(len({ids[-1-d] for d in (0,2,5,4,3)}),5)
    def test_frozen_design_rejects_unplanned_inputs(self):
        for args in ((1,), (SEEDS[0],1), (SEEDS[0],False,'yes')):
            with self.assertRaises(ValueError): design(*args)

if __name__=='__main__': unittest.main()
