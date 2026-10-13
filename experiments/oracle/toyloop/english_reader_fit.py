"""Tokenizer/renderer-only preflight. Never creates a service or sampling client."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unittest

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer, _KIMI_TOKENIZER_REVISIONS

MODEL = 'moonshotai/Kimi-K2.6'
REVISION = 'b5aabbfb20227ed42becbf5541dbffd213942c58'
RENDERER = 'kimi_k26_disable_thinking'
folder = Path(sys.argv[1])
suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_english_reader.py')
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
assert _KIMI_TOKENIZER_REVISIONS[MODEL] == REVISION
tokenizer = get_tokenizer(MODEL)
renderer = get_renderer(RENDERER, tokenizer)
rows = [json.loads(s) for s in (folder / 'questions.jsonl').read_text().splitlines()]
lengths = []
for row in rows:
    prompt = renderer.build_generation_prompt([{'role': 'user', 'content': row['prompt']}])
    tokens = prompt.to_ints()
    lengths.append({'id': row['id'], 'prompt_sha256': row['prompt_sha256'], 'tokens': prompt.length,
                    'rendered_tokens_sha256': hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()})
assert len(lengths) == 512
assert max(r['tokens'] for r in lengths) + 64 + 16 <= 32768
input_tokens = sum(r['tokens'] for r in lengths)
result = {'model': MODEL, 'tokenizer_revision': REVISION, 'renderer': RENDERER,
          'calls': 512, 'max_output_tokens': 64, 'prefill_tokens_total': input_tokens,
          'max_prompt_tokens': max(r['tokens'] for r in lengths),
          'max_sample_tokens_total': 512 * 64,
          'rate_usd_per_million': {'prefill': 2.205, 'sample': 5.49},
          'uncached_cost_upper_usd': (input_tokens * 2.205 + 512 * 64 * 5.49) / 1e6,
          'pricing_url': 'https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/',
          'pricing_checked_utc_date': '2026-10-13',
          'stop_sequences': renderer.get_stop_sequences(),
          'no_paid_calls': True,
          'packages': {p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers', 'tinker', 'tinker-cookbook')},
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
cache = Path.home() / '.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots' / REVISION
result['tokenizer_source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in cache.iterdir() if p.is_file()}
(folder / 'rendered_lengths.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in lengths))
(folder / 'cost_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
