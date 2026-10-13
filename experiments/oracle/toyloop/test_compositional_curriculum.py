"""Model-free frozen-design and graph-only curriculum contracts."""
import importlib.util
from pathlib import Path
import unittest
import compositional_curriculum as c


class CurriculumTests(unittest.TestCase):
    def route_module(self):
        spec = importlib.util.spec_from_file_location('curriculum_route_test',Path(__file__).with_name('route_transfer.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_disjoint_frozen_design_preserves_original_guard(self):
        module = self.route_module()
        original = module.SEEDS
        training = c.designs(module,c.TRAIN_SEEDS)
        evaluation = c.designs(module,c.TEST_SEEDS)
        self.assertEqual(len(training),32)
        self.assertEqual(len(evaluation),8)
        self.assertEqual(module.SEEDS,original)
        self.assertFalse(set(c.TRAIN_SEEDS)&set(c.TEST_SEEDS+c.OLD_SEEDS))
        with self.assertRaises(ValueError):
            module.design(c.TRAIN_SEEDS[0])
        with self.assertRaises(ValueError):
            c.designs(module,c.OLD_SEEDS)
        self.assertEqual(training,c.designs(module,c.TRAIN_SEEDS))

    def test_two_independent_routes_and_swapped_matching(self):
        module = self.route_module()
        for seed in c.TRAIN_SEEDS+c.TEST_SEEDS:
            a,b=c.designs(module,[seed])[0:3:2]
            self.assertEqual({tuple(e) for e in a['correct'][:2]}, {tuple(e) for e in b['wrong'][:2]})
            self.assertEqual(len({e[0] for e in a['correct'][:2]}),2)
            self.assertEqual(len({e[1] for e in a['correct'][:2]}),2)
            for ids in a['bank']:
                self.assertEqual(len({ids[-1-offset] for offset in (0,2,5,4,3)}),5)

    def test_graph_only_instruction_retains_observations(self):
        observation='Weights:\nL1H0 comp V<L0H1:.41\nBehaviour:\nText0 -> predicts t9\n'
        result=c.graph_instruction(observation+'Which units implement the prediction at the last position?\nAnswer English: blah')
        self.assertTrue(result.startswith(observation))
        self.assertNotIn('Answer English:',result)
        self.assertIn('No English section.',result)
        self.assertEqual(c.completion([['L0H1','L1H2'],['L1H2','out']]),'flowchart LR\n  L0H1 --> L1H2\n  L1H2 --> out\n')


if __name__=='__main__':
    unittest.main()
