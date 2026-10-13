import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import matched_graph_score as scorer


class MatchedScoreTests(unittest.TestCase):
    def test_clarified_diagnostic_is_explicit_bound_and_paired(self):
        originals = [dict(id=str(i), original_id=str(i), condition='original',
                          stratum='diagnostic_original', inputs=[[i]], input_roles=['disclosed'])
                     for i in range(8)]
        permuted = [dict(row, id=row['id']+'__perm', condition='permuted',
                         stratum='diagnostic_permuted') for row in originals]
        manifest = dict(diagnostic='clarified_unit_permutation_v1',
                        arms=['base', 'point', 'intervention', 'qwen35', 'kimi'],
                        panel=originals+permuted, forward_cap=20944)
        scorer.validate_panel(manifest, diagnostic_sha256=scorer.DIAGNOSTIC_MANIFEST_SHA)
        with self.assertRaises(ValueError): scorer.validate_panel(manifest)
        with self.assertRaises(ValueError): scorer.validate_panel(manifest, diagnostic_sha256='other')
        with self.assertRaises(ValueError):
            scorer.validate_panel(manifest, parent_path='mixed-mode', diagnostic_sha256=scorer.DIAGNOSTIC_MANIFEST_SHA)
        permuted[0]['inputs'] = [[123]]
        with self.assertRaises(ValueError):
            scorer.validate_panel(manifest, diagnostic_sha256=scorer.DIAGNOSTIC_MANIFEST_SHA)

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

    def test_input_chunks_preserve_every_world_and_default_schedule(self):
        panel = [dict(id=str(i), stratum='new_model', inputs=[0]*n, edits=[0]*3)
                 for i, n in enumerate((9, 2))]
        default = scorer.input_partition(panel, 2)
        self.assertEqual([list(group) for group in default], scorer.partition(panel, 2))
        chunks = scorer.input_partition(panel, 4, 2)
        self.assertEqual(chunks, scorer.input_partition(panel, 4, 2))
        keys = [scorer.input_cartesian(panel, ['a', 'b'], group) for group in chunks]
        self.assertEqual(set().union(*keys), scorer.cartesian(panel, ['a', 'b'], range(2)))
        self.assertEqual(sum(map(len, keys)), len(set().union(*keys)))
        self.assertGreater(sum(0 in group for group in chunks), 1)
        for size in (-1, True, 1.5):
            with self.assertRaises(ValueError): scorer.input_partition(panel, 2, size)
        with self.assertRaises(ValueError): scorer.input_partition(panel, 7, 2)
        for record in ({'input_chunk_size': True, 'input_indices': {'0': [0]}},
                       {'input_chunk_size': 1, 'input_indices': {'0': [False]}}):
            with self.assertRaises(ValueError): scorer.check_input_contract(record, {0: [0]}, 1)

    def test_chunk_merge_requires_exact_bank_coverage_and_identical_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = [dict(id='a', stratum='new_model', inputs=[[0], [1]], edits=[0])]
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(panel=panel, arms=['point'], forward_cap=6)))
            selections = scorer.input_partition(panel, 2, 1)
            shards = []
            for worker, selection in enumerate(selections):
                shard = root/str(worker); shard.mkdir(); shards.append(shard)
                (shard/'tensor').write_bytes(b'fixed fixture')
                scorer.write(shard/'0.original.json', {'model': 'same', 'basis': 'same'})
                scorer.write(shard/'0.restored.json', {'model': 'same', 'basis': 'same'})
                row = dict(stratum='new_model', case='a', arm='point', edit_index=0,
                           input=selection[0][0], kind='no_edit', input_role='disclosed',
                           valid=False, kl_bits=None, top=None, reference_top=1,
                           file='tensor', sha256=scorer.sha(shard/'tensor'))
                (shard/'results.jsonl').write_text(json.dumps(row)+'\n')
                fields = scorer.input_contract(selection, 1)
                scorer.write(shard/'contract.json', dict(worker=worker, workers=2,
                    case_indices=[0], manifest_sha256=scorer.sha(manifest),
                    responses_sha256='responses', scorer_sha256=scorer.sha(scorer.__file__),
                    forward_cap=3, global_forward_cap=6, **fields))
                scorer.write(shard/'complete.json', dict(forward_cap=3, actual_forwards=2,
                    candidate_worlds=1, models=1, arms=['point'], case_indices=[0], **fields))
            args = SimpleNamespace(manifest=manifest, manifest_sha256=scorer.sha(manifest),
                responses_sha256='responses', workers=2, input_chunk_size=1,
                shard=shards, out=root/'merged')
            scorer.merge(args)
            self.assertEqual(json.loads((args.out/'receipt.json').read_text())['actual_forwards'], 4)
            args.out = root/'bad'
            scorer.write(shards[1]/'changed.json', {'model': 'other'})
            for suffix in ('original', 'restored'):
                (shards[1]/f'0.{suffix}.json').write_text('{"model":"other"}')
            with self.assertRaisesRegex(ValueError, 'different original model'):
                scorer.merge(args)
            for suffix in ('original', 'restored'):
                (shards[1]/f'0.{suffix}.json').write_text('{"model":"same","basis":"same"}')
            args.input_chunk_size = 0
            with self.assertRaises(ValueError): scorer.merge(args)
            args.input_chunk_size = 1
            row['input'] = 0
            (shards[1]/'results.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError): scorer.merge(args)
            (shards[1]/'results.jsonl').write_text('')
            with self.assertRaises(ValueError): scorer.merge(args)

    def test_chunk_execution_keeps_global_bank_ids_baselines_and_tensor_values(self):
        # Tiny analytic physical fixture, not a model/circuit fidelity claim.
        # Different banks make a local/global index mix-up observable in p0.
        import contextlib
        import hashlib
        import sys
        import torch
        previous_threads = torch.get_num_threads()
        from unittest.mock import patch
        from safetensors.torch import load_file
        class Graph:
            def __init__(self, nodes=None, complete=False): self.nodes = nodes or {('toy', 0, 0)}
            def count(self): return len(self.nodes)
            def edges(self): return 0
        class Nat:
            def __init__(self):
                self.site = SimpleNamespace(W=torch.ones(1), U=torch.ones(1),
                                            V=torch.ones(1), delta_T=torch.zeros(1))
                self.names = ['toy']; self.C = {'toy': 1}; self.units = {'toy': torch.tensor([0])}
                self.target = SimpleNamespace(site=lambda name: self.site,
                    state_dict=lambda: {'W': self.site.W})
            def reference(self, bank, targets):
                x = float(bank[0][0]+1)
                return torch.tensor([[[float(self.site.W[0])*x, -x]]], dtype=torch.float64).log_softmax(-1)
        @contextlib.contextmanager
        def scale(nat, layer, unit, factor):
            old = {k: getattr(nat.site, k).clone() for k in ('W', 'V', 'delta_T')}
            for k in old: getattr(nat.site, k).mul_(factor)
            try: yield
            finally:
                for k, value in old.items(): getattr(nat.site, k).copy_(value)
        modules = {
            'native': SimpleNamespace(Graph=Graph),
            'planted': SimpleNamespace(native_of=lambda sd, cfg: Nat()),
            'goal_eval': SimpleNamespace(GoalEval=lambda nat: SimpleNamespace(
                graph_ref=lambda ids, targets, graph, **kw: nat.reference([ids], targets))),
            'circuit': SimpleNamespace(parse=lambda text: SimpleNamespace(ok=True, bad=[],
                units={'L0N0', 'out'}, edges=[('L0N0', 'out')]),
                unit_graph=lambda parsed, toy, ids: Graph()),
            'head_counterfactual': SimpleNamespace(scaled_head=scale,
                raw_kl_bits=lambda p, q: float((p.exp()*(p-q)).sum()/torch.log(torch.tensor(2.)))),
            'reward_interventions': SimpleNamespace(scaled_neuron=scale, clear_intervention_caches=lambda nat: None),
            'intervention_teacher': SimpleNamespace(tensor_digest=lambda x: hashlib.sha256(x.numpy().tobytes()).hexdigest()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'weights').mkdir()
            asset = root/'weights/model.pt'; torch.save({'sd': {}, 'cfg': {}}, asset)
            paths = ('experiments/oracle/toyloop/circuit.py', 'experiments/oracle/toys/planted.py',
                'method/oracle/native.py', 'method/oracle/native_units.py', 'method/oracle/goal_eval.py',
                'method/oracle/carry.py', 'method/oracle/mech.py', 'method/vpd/vpd_model.py')
            support = tuple('experiments/oracle/toyloop/'+n+'.py' for n in
                ('head_counterfactual', 'reward_interventions', 'intervention_teacher'))
            for path in paths+support:
                file = root/path; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'fixture')
            panel = [dict(id='case', stratum='new_model', prompt_sha256='prompt', answer_format='graph_only',
                weights='weights/model.pt', weights_sha256=scorer.sha(asset), inputs=[[0], [1], [2], [3]],
                input_roles=['disclosed']*4, edits=[dict(kind='no_edit', operations=[]),
                    dict(kind='individual', operations=[['neuron', 0, 0, .5]])])]
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(dict(panel=panel, arms=['point'],
                forward_cap=24, source_sha256={p: scorer.sha(root/p) for p in paths},
                support_sha256={p: scorer.sha(root/p) for p in support})))
            sample = root/'point.jsonl'; sample.write_text(json.dumps(dict(id='case', stratum='new_model',
                sample=0, prompt_sha256='prompt', finish='stop', completion='flowchart LR\nL0N0 --> out'))+'\n')
            responses = root/'responses.json'; responses.write_text(json.dumps(dict(
                manifest_sha256=scorer.sha(manifest), arms={'point': {'path': 'point.jsonl', 'sha256': scorer.sha(sample)}})))
            args = SimpleNamespace(manifest=manifest, manifest_sha256=scorer.sha(manifest), responses=responses,
                responses_sha256=scorer.sha(responses), source=root, support=root, payload=root,
                workers=1, worker=0, input_chunk_size=0, out=root/'whole')
            previous_path = sys.path[:]
            try:
                with patch.dict(sys.modules, modules), patch.object(scorer, 'validate_panel'):
                    scorer.run(args)
                    args.workers = 2; args.input_chunk_size = 1
                    for worker in range(2):
                        args.worker = worker; args.out = root/f'chunk{worker}'; scorer.run(args)
            finally:
                sys.path[:] = previous_path
                torch.set_num_threads(previous_threads)
            full = {p.name: load_file(p) for p in (root/'whole').glob('*.safetensors')}
            chunk = {p.name: load_file(p) for w in range(2) for p in (root/f'chunk{w}').glob('*.safetensors')}
            self.assertEqual(set(full), set(chunk))
            for name in full:
                for key in full[name]: self.assertTrue(torch.equal(full[name][key], chunk[name][key]), (name, key))
            self.assertEqual(sum(json.loads((root/f'chunk{w}/complete.json').read_text())['actual_forwards']
                                 for w in range(2)), 24)

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
