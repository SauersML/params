from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relation_loss as r


class RelationLossTests(unittest.TestCase):
    def test_only_actual_relation_lines_before_english(self):
        text = ('flowchart LR\n tokK --> L0H1\n L0H1 --> L1H2\n L1H2 --> out\n'
                'English: L0H1 --> L1H2 is prose.\n L0H1 --> L1H2')
        ids = list(range(len(text)))
        mask = r.relation_mask(text, ids, text, ids, [(i, i + 1) for i in ids])
        begin = text.index(' L0H1 --> L1H2\n')
        end = begin + len(' L0H1 --> L1H2\n')
        self.assertEqual([i for i, v in enumerate(mask) if v], list(range(begin, end)))

    def test_crossing_boundary_and_terminal_tokens_are_not_boosted(self):
        text = 'flowchart LR\n  L0H0 --> L1H1\nEnglish: Retained.'
        a = text.index('  L0H0'); b = text.index('English:')
        decoded = text + '<|im_end|>'
        spans = [(a - 1, a + 2), (a + 2, a + 5), (b - 1, b + 2),
                 (len(text), len(decoded)), (len(decoded), len(decoded))]
        self.assertEqual(r.relation_mask(text, range(5), decoded, range(5), spans),
                         [False, True, False, False, False])

    def test_utf8_overlapping_offsets_in_english_remain_supervised(self):
        text = ' L0H0 --> L1H1\nEnglish: ± is retained.'
        a = text.index('±')
        mask = r.relation_mask(text, [1, 2, 3], text, [1, 2, 3], [(1, 4), (a, a + 1), (a, a + 1)])
        self.assertEqual(mask, [True, False, False])
        weights = r.reweight([.2, .4, .4], mask, 8)
        self.assertTrue(all(w > 0 for w in weights))
        self.assertAlmostEqual(sum(weights), 1.)
        self.assertAlmostEqual(weights[1], weights[2])

    def test_noncanonical_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'exact supervised'):
            r.relation_mask('L0H0 --> L1H1', [1], 'L0H0 --> L1H1', [2], [(0, 14)])
        with self.assertRaisesRegex(ValueError, 'decode'):
            r.relation_mask('L0H0 --> L1H1', [1], 'changed', [1], [(0, 7)])

    def test_normalization_preserves_english_and_untrained_tokens(self):
        original = [0., .1, .3, .6]
        changed = r.reweight(original, [False, True, False, False], 8)
        self.assertEqual(changed[0], 0.)
        self.assertAlmostEqual(sum(changed), sum(original))
        self.assertAlmostEqual(changed[1] / changed[2], 8 * original[1] / original[2])
        self.assertAlmostEqual(changed[2] / changed[3], original[2] / original[3])
        self.assertTrue(all(w > 0 for w in changed[1:]))
        self.assertEqual(original, [0., .1, .3, .6])

    def test_no_relation_and_multiplier_one_leave_original_bits(self):
        weights = [0., .123456789, .876543211]
        self.assertEqual(r.reweight(weights, [False] * 3, 8), weights)
        self.assertEqual(r.reweight(weights, [False, True, False], 1), weights)

    def test_invalid_weights_or_mask_rejected(self):
        for weights, mask, m in [([0.], [False], 8), ([-1.], [True], 8),
                                ([float('nan')], [True], 8), ([1.], [1], 8),
                                ([1.], [True], float('inf')), ([1.], [True], 0)]:
            with self.assertRaises(ValueError):
                r.reweight(weights, mask, m)


if __name__ == '__main__':
    unittest.main()
