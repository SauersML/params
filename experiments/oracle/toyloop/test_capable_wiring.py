import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capable_wiring as c


class CapableWiringTests(unittest.TestCase):
    def test_every_train_row_exactly_once_per_epoch_and_short_tail(self):
        batches = c.schedule()
        self.assertEqual(len(batches), 58)
        for epoch in (batches[:29], batches[29:]):
            visits = [i for batch in epoch for i in batch]
            self.assertEqual(sorted(visits), list(range(226)))
            self.assertEqual(len(epoch[-1]), 2)
        self.assertEqual(batches, c.schedule())
        self.assertNotEqual(batches, c.schedule(seed=1))

    def test_schedule_never_mutates_or_needs_eval_examples(self):
        first = c.schedule(count=3, epochs=2, batch=2)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(i < 3 for b in first for i in b))
        for dims in ({'count': 0}, {'epochs': 0}, {'batch': 0}):
            with self.assertRaises(ValueError):
                c.schedule(**dims)

    def fixture(self, tmp):
        train = tmp / 'train'; panel = tmp / 'panel'; ref = tmp / 'ref'
        examples = [{'id': str(i), 'prompt': 'weights ' + str(i),
                     'completion': 'flowchart LR\n L0H0 --> out\nEnglish: Complete original prose.'} for i in range(226)]
        questions = [{'id': str(i), 'prompt': 'original prompt ' + str(i)} for i in range(24)]
        for r in questions:
            r['prompt_sha256'] = hashlib.sha256(r['prompt'].encode()).hexdigest()
        train.write_text(''.join(json.dumps(r) + '\n' for r in examples))
        panel.write_text(''.join(json.dumps(r) + '\n' for r in questions))
        ref.write_text(''.join(json.dumps({k: r[k] for k in ('id', 'prompt_sha256')}) + '\n' for r in questions))
        return train, panel, ref

    def test_frozen_data_and_panel_preserve_full_english(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.fixture(Path(d))
            with patch.object(c, 'TRAIN_SHA', c.sha(paths[0])), patch.object(c, 'PANEL_REFERENCE_SHA', c.sha(paths[2])):
                train, panel = c.validate_data(*paths)
                self.assertEqual(len(train), 226)
                self.assertEqual(len(panel), 24)
                self.assertTrue(all(r['completion'].endswith('Complete original prose.') for r in train))
                changed = c.rows(paths[1]); changed[0]['prompt'] = 'novel prompt'
                changed[0]['prompt_sha256'] = hashlib.sha256(changed[0]['prompt'].encode()).hexdigest()
                paths[1].write_text(''.join(json.dumps(r) + '\n' for r in changed))
                with self.assertRaisesRegex(ValueError, 'panel prompt'):
                    c.validate_data(*paths)
                self.assertEqual(len(c.validate_data(*paths, allow_reference_drift=True)[1]), 24)
                changed[0]['prompt_sha256'] = 'unverified bytes'
                paths[1].write_text(''.join(json.dumps(r) + '\n' for r in changed))
                with self.assertRaisesRegex(ValueError, 'panel prompt'):
                    c.validate_data(*paths, allow_reference_drift=True)

    def test_changed_training_bytes_rejected_even_when_parseable(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.fixture(Path(d))
            expected = c.sha(paths[0])
            paths[0].write_text(paths[0].read_text() + '\n')
            with patch.object(c, 'TRAIN_SHA', expected), self.assertRaisesRegex(ValueError, 'training bytes'):
                c.validate_data(*paths)

    def test_stripped_english_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.fixture(Path(d))
            r = c.rows(paths[0]); r[0]['completion'] = 'flowchart LR\n L0H0 --> out'
            paths[0].write_text(''.join(json.dumps(x) + '\n' for x in r))
            with patch.object(c, 'TRAIN_SHA', c.sha(paths[0])), patch.object(c, 'PANEL_REFERENCE_SHA', c.sha(paths[2])):
                with self.assertRaisesRegex(ValueError, 'English'):
                    c.validate_data(*paths)

    def test_paid_mode_wrong_project_fails_before_any_service_import(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / 'manifest'
            m.write_text(json.dumps({'runner_sha256': c.sha(c.__file__), 'model': c.MODEL, 'project': c.PROJECT}))
            with patch.dict('os.environ', {'TINKER_PROJECT_ID': 'wrong'}):
                with self.assertRaisesRegex(ValueError, 'project'):
                    c.run(m, Path(d) / 'out', 1.)
            self.assertFalse((Path(d) / 'out').exists())

    def test_paid_mode_insufficient_cap_fails_before_any_service_import(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / 'manifest'
            m.write_text(json.dumps({'runner_sha256': c.sha(c.__file__), 'model': c.MODEL,
                                     'project': c.PROJECT, 'cost_upper_usd': .5}))
            with patch.dict('os.environ', {'TINKER_PROJECT_ID': c.PROJECT}):
                for cap in (.4, float('nan')):
                    with self.assertRaisesRegex(ValueError, 'cap'):
                        c.run(m, Path(d) / 'out', cap)

    def test_sdk_mock_preserves_paid_raw_output_on_parse_failure_and_full_schedule(self):
        # No Tinker package, renderer, weights, or service is loaded by this test.
        events = []
        future = lambda value: SimpleNamespace(result=lambda: value)
        class RR:
            calls = 0
            def get_stop_sequences(self):
                return ['stop']
            def parse_response(self, tokens):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError('deliberate parser failure')
                return {'content': 'flowchart LR\nEnglish: Retained.'}, None
        class Sampler:
            def sample(self, *, prompt, num_samples, sampling_params):
                self_test.assertEqual((num_samples, sampling_params.temperature, sampling_params.max_tokens), (1, 0., 416))
                events.append('sample')
                return future(SimpleNamespace(prompt_cache_hit_tokens=1,
                    sequences=[SimpleNamespace(tokens=[1, 2], stop_reason='stop')]))
        class Training:
            def get_info(self):
                return SimpleNamespace(model_dump=lambda **kwargs: {'model': c.MODEL})
            def forward_backward(self, batch, *, loss_fn):
                events.append('fb'); self_test.assertEqual(loss_fn, 'cross_entropy')
                return future(None)
            def optim_step(self, params):
                return future(None)
            def save_state(self, name):
                return future(SimpleNamespace(path='tinker://mock'))
            def save_weights_and_get_sampling_client(self):
                return Sampler()
        class Service:
            def __init__(self, *, project_id):
                self_test.assertEqual(project_id, c.PROJECT)
            def create_sampling_client(self, *, base_model):
                return Sampler()
            def create_lora_training_client(self, *, base_model, rank):
                self_test.assertEqual(events, ['sample'] * 24)
                self_test.assertEqual(rank, 32)
                return Training()
        self_test = self
        sdk = SimpleNamespace(ServiceClient=Service, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
                              AdamParams=lambda **kwargs: SimpleNamespace(**kwargs))
        renderers = SimpleNamespace(get_text_content=lambda msg: msg['content'])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); train, panel, ref = self.fixture(root)
            tokenizer = root / 'tokenizer'; tokenizer.mkdir()
            cost_tokens = {'trained': 452 * 80, 'prefill': 48 * 3, 'sampled': 48 * 416}
            upper = sum(cost_tokens.get(k, 0) * p for k, p in c.PRICES.items()) / 1e6
            manifest = {'runner_sha256': c.sha(c.__file__), 'model': c.MODEL, 'project': c.PROJECT,
                        'cost_upper_usd': upper, 'recipe': c.RECIPE, 'tokenizer_revision': c.REVISION,
                        'train_path': str(train), 'panel_path': str(panel), 'panel_reference': str(ref),
                        'panel_sha256': c.sha(panel), 'allow_reference_drift': False, 'reference_prompt_mismatches': [],
                        'package_versions': {}, 'tokenizer_path': str(tokenizer), 'tokenizer_files': {},
                        'token_fit': {'training_lengths': [80] * 226, 'panel_prompt_lengths': [3] * 24},
                        'sample_ids': [[str(i) for i in b] for b in c.schedule()],
                        'cost_upper_tokens': cost_tokens, 'prices_per_million': c.PRICES}
            path = root / 'manifest'; path.write_text(json.dumps(manifest))
            with patch.object(c, 'TRAIN_SHA', c.sha(train)), patch.object(c, 'PANEL_REFERENCE_SHA', c.sha(ref)), \
                    patch.object(c, 'renderer', return_value=(RR(), None)), \
                    patch.object(c, 'datum', return_value=SimpleNamespace(model_input=SimpleNamespace(length=80))), \
                    patch.object(c, 'generation_prompt', return_value=SimpleNamespace(length=3)), \
                    patch.dict('os.environ', {'TINKER_PROJECT_ID': c.PROJECT}), \
                    patch.dict(sys.modules, {'tinker': sdk, 'tinker_cookbook.renderers': renderers}):
                c.run(path, root / 'out', upper)
            raw, parsed = c.rows(root / 'out/base_raw.jsonl'), c.rows(root / 'out/base.jsonl')
            self.assertEqual(len(raw), 24)
            self.assertEqual(raw[0]['token_ids'], [1, 2])
            self.assertIn('parse_error', parsed[0])
            self.assertEqual(len(c.rows(root / 'out/trained.jsonl')), 24)
            self.assertEqual(events.count('fb'), 58)
            self.assertEqual(sum(len(r['sample_ids']) for r in c.rows(root / 'out/steps.jsonl')), 452)
            self.assertEqual(len(c.rows(root / 'out/spend.jsonl')), 106)


if __name__ == '__main__':
    unittest.main()
