import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from english_reader_prepare import choose, unit_diagram
from english_reader_score import parse, score
from english_reader_sample import charge, charge_exact, journal
from decimal import Decimal


class Contract(unittest.TestCase):
    def test_strip_only_unexecuted_annotations(self):
        diagram = unit_diagram('flowchart LR\n tokBOS --> L0H0\n L0H0 --> L1H2\n L1N9 --> out\nEnglish: V')
        self.assertNotIn('tokBOS', diagram)
        self.assertIn('L0H0 --> L1H2', diagram)
        self.assertIn('L1N9 --> out', diagram)

    def test_selection_excludes_shown_and_duplicates(self):
        bank = [[0, 1], [0, 2], [0, 1], [0, 3], [0, 3], [0, 4]]
        self.assertEqual(set(choose(bank)), {3, 5})

    def test_truncated_and_bad_json_invalid(self):
        self.assertIsNone(parse({'completion': '{"top":"t1","changed":true}'}))
        self.assertIsNone(parse({'finish': 'length', 'completion': '{"top":"t1","changed":true}'}))
        self.assertIsNone(parse({'finish': 'stop', 'completion': '{"top":"t32","changed":true}'}))
        self.assertIsNone(parse({'finish': 'stop', 'completion': '{"top":"t1","changed":"false"}'}))

    def test_missing_is_wrong(self):
        label = {'id': 'a', 'case': 'a', 'condition': 'none', 'task': 'x', 'input': 2, 'layer': 0,
                 'head': 0, 'scale': 0., 'active': True, 'changed': True, 'top': 1, 'original_top': 0}
        result = score([label], [])['groups']['none/all']
        self.assertEqual(result, {'n': 1, 'valid': 0, 'top_correct': 0, 'change_correct': 0, 'no_change_top_correct': 0})

    def test_parse_error_is_wrong(self):
        self.assertIsNone(parse({'finish': 'stop', 'parse_error': 'ValueError', 'completion': None}))

    def test_charge_and_durable_journal(self):
        self.assertAlmostEqual(charge(100, 20, 16), (80 * 2.205 + 20 * .441 + 16 * 5.49) / 1e6)
        with self.assertRaises(AssertionError):
            charge(100, 101, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'issued.jsonl'
            with patch('english_reader_sample.os.fsync') as sync:
                journal(path, {'id': 'a', 'usd': .01})
                self.assertEqual(sync.call_count, 1)
            self.assertEqual(json.loads(path.read_text()), {'id': 'a', 'usd': .01})

    def test_exact_frozen_budget_has_no_float_guard_drift(self):
        lengths = [440] * 128 + [439] * 384
        self.assertEqual(sum(lengths), 224896)
        upper = sum((charge_exact(n, 0, 64) for n in lengths), Decimal('0'))
        self.assertEqual(upper, Decimal('.675792'))


if __name__ == '__main__':
    unittest.main()
