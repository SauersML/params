import copy
import unittest

import fresh_model_evaluation as f


class FreshEvaluationTests(unittest.TestCase):
    def manifest(self):
        return {'generator_commit': f.EXECUTOR, 'native_forwards': 0, 'teacher_forwards': 0,
                'models': [{'id': ident, 'family': family, 'seed': seed,
                            'role': 'new_model_evaluation'}
                           for ident, family, seed in f.specifications()]}

    def test_fixed_evaluation_identity_never_substitutes_failed_or_train_cases(self):
        value = self.manifest()
        f.validate_manifest(value)
        for change in (lambda m: m['models'].reverse(),
                       lambda m: m['models'][0].update(seed=3300000),
                       lambda m: m['models'][0].update(role='train'),
                       lambda m: m.update(native_forwards=1)):
            other = copy.deepcopy(value)
            change(other)
            with self.assertRaises(RuntimeError):
                f.validate_manifest(other)

    def test_disclosed_inputs_reject_answer_labels_and_nonlast_target(self):
        ident, family, seed = f.specifications()[0]
        record = {'id': ident, 'family': family, 'seed': seed,
                  'role': 'new_model_evaluation',
                  'examples': [{'kind': 'ordinary', 'ids': [0, 1, 2], 'target': 2} for _ in range(4)]}
        self.assertEqual(f.validate_record(record, ident, family, seed), [[0, 1, 2]]*4)
        for change in (lambda r: r['examples'][0].update(intended=3),
                       lambda r: r.update(truth={}),
                       lambda r: r['examples'][0].update(target=1),
                       lambda r: r['examples'][0].update(ids=[0, True, 2])):
            other = copy.deepcopy(record)
            change(other)
            with self.assertRaises(RuntimeError):
                f.validate_record(other, ident, family, seed)

    def test_joint_panel_is_candidate_independent_and_includes_all_mixed_scales(self):
        cfg = {'n_layer': 2, 'n_head': 4}
        result = f.edits('gate_3400500', cfg, [(0, 7), (1, 9)])
        individual = [e for e in result if e['kind'] == 'individual']
        self.assertEqual(len(individual), 20)
        self.assertEqual(len({tuple(e['operations'][0]) for e in individual}), 20)
        joints = [e for e in result if e['kind'] == 'joint']
        self.assertEqual(len(joints), 24)
        pairs = {}
        for e in joints:
            a, b = e['operations']
            pairs.setdefault((tuple(a[:3]), tuple(b[:3])), set()).add((a[3], b[3]))
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(scales == {(0., 0.), (0., .5), (.5, 0.), (.5, .5)}
                            for scales in pairs.values()))
        self.assertEqual(f.ARMS, ['base', 'point24', 'intervention24', 'point35', 'intervention35'])

    def test_both_pipelines_are_fixed_for_all_roles_without_extra_calls(self):
        protocol = f.pipeline_protocol('panel', 'manifest', 1480)
        self.assertEqual(protocol['raw_forward_cap'], 10360)
        self.assertEqual(protocol['combined_forward_cap'], 17760)
        self.assertEqual(protocol['additional_llm_calls'], 0)
        self.assertEqual([p['name'] for p in protocol['pipelines']],
                         ['raw_explicit', 'same_selected_architecture_completion'])
        self.assertEqual(protocol['arms'], f.ARMS)
        self.assertIn('never select best per case', protocol['selection_policy'])
        self.assertIn('remain invalid in both', protocol['invalid_policy'])


if __name__ == '__main__':
    unittest.main()
