import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

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

    def test_partition_is_stable_balanced_and_case_disjoint(self):
        panel = [{'inputs': [0] * count, 'edits': [0]} for count in (1, 9, 3, 9, 2)]
        self.assertEqual(scorer.partition(panel, 2), [[1, 2], [0, 3, 4]])
        self.assertEqual(sorted(sum(scorer.partition(panel, 4), [])), list(range(5)))
        for workers in (0, 6, True):
            with self.assertRaises(ValueError):
                scorer.partition(panel, workers)

    def test_merge_requires_disjoint_complete_worlds_and_binds_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = [{'id': str(i), 'stratum': 'new_model', 'inputs': [0], 'edits': [0]} for i in range(2)]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'panel': panel, 'arms': ['point'], 'forward_cap': 6}))
            shards = []
            for i in range(2):
                shard = root / str(i); shard.mkdir(); shards.append(shard)
                (shard / 'tensor').write_bytes(b'opaque offline fixture')
                scorer.write(shard / f'{i}.original.json', {'weights': 'fixed'})
                scorer.write(shard / f'{i}.restored.json', {'weights': 'fixed'})
                row = {'stratum': 'new_model', 'case': str(i), 'arm': 'point',
                       'edit_index': 0, 'input': 0, 'kind': 'no_edit', 'input_role': 'disclosed',
                       'valid': False, 'kl_bits': None, 'top': None, 'reference_top': 1,
                       'file': 'tensor', 'sha256': scorer.sha(shard / 'tensor')}
                (shard / 'results.jsonl').write_text(json.dumps(row)+'\n')
                scorer.write(shard / 'contract.json', {'worker': i, 'workers': 2,
                    'case_indices': [i], 'manifest_sha256': scorer.sha(manifest),
                    'responses_sha256': 'response', 'scorer_sha256': scorer.sha(scorer.__file__),
                    'forward_cap': 3, 'global_forward_cap': 6})
                scorer.write(shard / 'complete.json', {'forward_cap': 3, 'actual_forwards': 2,
                    'candidate_worlds': 1, 'models': 1, 'arms': ['point'], 'case_indices': [i]})
            args = SimpleNamespace(manifest=manifest, manifest_sha256=scorer.sha(manifest),
                responses_sha256='response', workers=2, shard=shards, out=root/'merged')
            scorer.merge(args)
            self.assertEqual(json.loads((args.out/'receipt.json').read_text())['candidate_worlds'], 2)
            self.assertTrue(all('artifact_root' in json.loads(x) for x in (args.out/'results.jsonl').read_text().splitlines()))
            args.out = root/'bad'; args.shard = [shards[0], shards[0]]
            with self.assertRaises(ValueError): scorer.merge(args)
            args.shard = shards[:1]
            with self.assertRaises(ValueError): scorer.merge(args)
            args.shard = shards
            (shards[1]/'results.jsonl').write_text('')
            with self.assertRaises(ValueError): scorer.merge(args)

    def test_assets_cannot_escape_manifest_root(self):
        for path in ('/tmp/escape', '../escape', 'inside/../../escape'):
            with self.assertRaises(ValueError):
                scorer.bound_path('/payload', path)

    def test_strong_baseline_keeps_exact_original_worlds_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = {'schema_version': 1, 'arms': ['base', 'point', 'intervention'],
                      'panel': [{'id': str(i), 'stratum': 'new_model' if i < 8 else 'retention',
                                 'inputs': [[i]], 'edits': [[]]} for i in range(32)],
                      'source_sha256': {'executor': 'fixed'}, 'forward_cap': 160}
            path = Path(tmp) / 'parent.json'; path.write_text(json.dumps(parent))
            digest = scorer.sha(path)
            derived = dict(parent, arms=['strong'], panel=parent['panel'][:8],
                           forward_cap=24, parent_manifest_sha256=digest)
            scorer.validate_panel(parent)
            scorer.validate_panel(derived, path, digest)
            for changed in (dict(derived, panel=derived['panel'][:-1]),
                            dict(derived, source_sha256={'executor': 'different'}),
                            dict(derived, arms=['base']), dict(derived, forward_cap=23),
                            dict(derived, extra='unbound')):
                with self.assertRaises(ValueError): scorer.validate_panel(changed, path, digest)
            with self.assertRaises(ValueError): scorer.validate_panel(derived)
            with self.assertRaises(ValueError): scorer.validate_panel(derived, path, 'wrong')


if __name__ == '__main__':
    unittest.main()
