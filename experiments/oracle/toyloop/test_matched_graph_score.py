import unittest

import matched_graph_score as scorer


class MatchedScoreTests(unittest.TestCase):
    def test_missing_response_retained_and_duplicates_rejected(self):
        panel = [{'id': name, 'stratum': 'new_model', 'prompt_sha256': name} for name in ('a', 'b')]
        sample = dict(panel[0], sample=0)
        self.assertEqual(scorer.bind_responses(panel, [sample]), [sample, None])
        with self.assertRaises(ValueError):
            scorer.bind_responses(panel, [sample, sample])
        with self.assertRaises(ValueError):
            scorer.bind_responses(panel, [dict(sample, prompt_sha256='changed')])

    def test_no_repair_or_truncated_answer_acceptance(self):
        valid = {'finish': 'stop', 'completion': 'flowchart LR\nL0H0 --> out'}
        self.assertIsNone(scorer.answer_error(valid))
        self.assertIsNotNone(scorer.answer_error(None))
        for change in ({'finish': 'length'}, {'parse_error': 'bad'},
                       {'completion': valid['completion']+'\nEnglish: circuit'},
                       {'completion': '```mermaid\n'+valid['completion']+'\n```'}):
            self.assertIsNotNone(scorer.answer_error(dict(valid, **change)))

    def test_models_not_worlds_are_replication_units(self):
        rows = []
        for case, count, valid in [('a', 1, False), ('b', 3, True)]:
            for index in range(count):
                rows.append({'stratum': 'new_model', 'case': case, 'arm': 'point',
                             'edit_index': 0, 'input': index, 'input_role': 'disclosed',
                             'kind': 'individual', 'valid': valid, 'kl_bits': 0. if valid else None,
                             'top': 1 if valid else None, 'reference_top': 1})
        keys = {(r['stratum'], r['case'], r['arm'], r['edit_index'], r['input']) for r in rows}
        group = scorer.summarize(rows, keys)['groups'][0]
        self.assertEqual(group['model_mean_kl_pass_fraction'], .5)
        self.assertEqual(group['model_mean_top_agreement'], .5)
        self.assertEqual(group['worlds'], 4)
        self.assertIsNone(group['per_model'][0]['worst_valid_kl_bits'])
        with self.assertRaises(ValueError):
            scorer.summarize(rows[:-1], keys)
        with self.assertRaises(ValueError):
            scorer.summarize(rows+[rows[0]], keys)
        rows[-1]['kl_bits'] = float('nan')
        with self.assertRaises(ValueError):
            scorer.summarize(rows, keys)

    def test_retention_preserves_its_requested_english_and_token_annotations(self):
        old_answer = {'finish': 'stop', 'completion': 'flowchart LR\ntokK --> L0H0\nL0H0 --> out\nEnglish: copied value.'}
        self.assertIsNone(scorer.answer_error(old_answer, 'legacy_mermaid_english'))
        self.assertIsNotNone(scorer.answer_error(old_answer, 'graph_only'))
        self.assertIsNotNone(scorer.answer_error(dict(old_answer, finish='length'), 'legacy_mermaid_english'))
        with self.assertRaises(ValueError):
            scorer.answer_error(old_answer, 'anything_goes')

    def test_assets_cannot_escape_manifest_root(self):
        for path in ('/tmp/escape', '../escape', 'inside/../../escape'):
            with self.assertRaises(ValueError):
                scorer.bound_path('/payload', path)


if __name__ == '__main__':
    unittest.main()
