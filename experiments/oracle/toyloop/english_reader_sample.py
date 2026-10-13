"""Future explicit reader execution; preparation never invokes this runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from decimal import Decimal

MODEL = 'moonshotai/Kimi-K2.6'
RENDERER = 'kimi_k26_disable_thinking'
PREFILL_TOTAL = 224896
MAX_TOKENS = 64


def journal(path, record):
    with path.open('a') as fh:
        fh.write(json.dumps(record) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def charge_exact(prefill, hits, sampled):
    assert 0 <= hits <= prefill and 0 <= sampled <= MAX_TOKENS
    return (Decimal(prefill - hits) * Decimal('2.205') + Decimal(hits) * Decimal('.441')
            + Decimal(sampled) * Decimal('5.49')) / Decimal('1000000')


def charge(prefill, hits, sampled):
    return float(charge_exact(prefill, hits, sampled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--budget', type=float, default=.75)
    a = ap.parse_args()
    if not a.execute:
        raise SystemExit('Offline preparation only; explicit reviewed execution required')
    project = '3f598f31-2ffa-4eb4-b302-909eb234fb7d'
    assert os.environ.get('TINKER_PROJECT_ID') == project
    assert os.environ.get('TINKER_API_KEY'), 'Keychain environment credential required'
    fit = json.loads((a.inputs / 'cost_preflight.json').read_text())
    manifest = json.loads((a.inputs / 'manifest.json').read_text())
    assert fit['model'] == MODEL and fit['renderer'] == RENDERER and fit['max_output_tokens'] == MAX_TOKENS
    assert hashlib.sha256((a.inputs / 'questions.jsonl').read_bytes()).hexdigest() == manifest['question_sha256']
    assert not a.output.exists(), 'Never silently restart a partially issued trial'
    a.output.mkdir()
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer, _KIMI_TOKENIZER_REVISIONS
    import tinker
    assert _KIMI_TOKENIZER_REVISIONS[fit['model']] == fit['tokenizer_revision']
    tokenizer = get_tokenizer(fit['model'])
    renderer = renderers.get_renderer(fit['renderer'], tokenizer)
    rows = [json.loads(s) for s in (a.inputs / 'questions.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 512
    lengths = {r['id']: r for r in map(json.loads, (a.inputs / 'rendered_lengths.jsonl').read_text().splitlines())}
    prepared = []
    for row in rows:
        prompt = renderer.build_generation_prompt([{'role': 'user', 'content': row['prompt']}])
        assert prompt.length == lengths[row['id']]['tokens']
        assert hashlib.sha256(json.dumps(prompt.to_ints(), separators=(',', ':')).encode()).hexdigest() == lengths[row['id']]['rendered_tokens_sha256']
        prepared.append((row, prompt))
    assert renderer.get_stop_sequences() == fit['stop_sequences']
    assert sum(prompt.length for _, prompt in prepared) == PREFILL_TOTAL
    upper_exact = sum((charge_exact(prompt.length, 0, MAX_TOKENS) for _, prompt in prepared), Decimal('0'))
    assert upper_exact <= Decimal(str(a.budget)) and upper_exact == Decimal('.675792')
    upper = float(upper_exact)
    contract = {'fit': fit, 'manifest_sha256': hashlib.sha256((a.inputs / 'manifest.json').read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'project': project,
                'budget_upper_usd': a.budget, 'recomputed_cost_upper_usd': upper,
                'calls': 512, 'outstanding_limit': 8, 'labels_read': False, 'no_restart': True}
    (a.output / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    client = tinker.ServiceClient(project_id=project).create_sampling_client(base_model=fit['model'])
    params = tinker.SamplingParams(max_tokens=64, temperature=0., stop=renderer.get_stop_sequences())
    spend_exact, responses = Decimal('0'), 0
    for offset in range(0, len(prepared), 8):
        pending, failures = [], []
        for row, prompt in prepared[offset:offset + 8]:
            journal(a.output / 'issued.jsonl', {'id': row['id'], 'prompt_sha256': row['prompt_sha256'],
                                               'reserved_upper_usd': charge(prompt.length, 0, MAX_TOKENS)})
            try:
                future = client.sample(prompt=prompt, num_samples=1, sampling_params=params)
                pending.append((row, prompt, future))
            except Exception as exc:
                journal(a.output / 'unresolved.jsonl', {'id': row['id'], 'error': type(exc).__name__, 'stage': 'issuance',
                                                        'reserved_upper_usd': charge(prompt.length, 0, MAX_TOKENS)})
                failures.append(row['id'])
                break
        # Drain every successfully issued future even when one fails. Never issue the next batch after a failure.
        for row, prompt, future in pending:
            try:
                result = future.result()
                assert len(result.sequences) == 1
                sequence = result.sequences[0]
                hits = result.prompt_cache_hit_tokens
                record = {'id': row['id'], 'prompt_sha256': row['prompt_sha256'], 'sample': 0,
                          'token_ids': list(sequence.tokens), 'finish': str(sequence.stop_reason),
                          'prefill_tokens': prompt.length, 'prompt_cache_hit_tokens': hits}
                journal(a.output / 'raw.jsonl', record)  # Paid raw evidence precedes renderer parsing/accounting.
                usd_exact = charge_exact(prompt.length, hits, len(sequence.tokens))
                spend_exact += usd_exact
                usd, spend = float(usd_exact), float(spend_exact)
                journal(a.output / 'spend.jsonl', {'id': row['id'], 'prefill': prompt.length - hits,
                                                  'cached': hits, 'sampled': len(sequence.tokens), 'usd': usd,
                                                  'total_usd': spend})
                assert spend_exact <= Decimal(str(a.budget))
                try:
                    record['completion'] = renderers.get_text_content(renderer.parse_response(sequence.tokens)[0])
                except Exception as exc:
                    record.update(completion=None, parse_error=type(exc).__name__)
                journal(a.output / 'samples.jsonl', record)
                responses += 1
            except Exception as exc:
                journal(a.output / 'unresolved.jsonl', {'id': row['id'], 'error': type(exc).__name__, 'stage': 'drain',
                                                        'reserved_upper_usd': charge(prompt.length, 0, MAX_TOKENS)})
                failures.append(row['id'])
        print(json.dumps({'responses': responses, 'usd': float(spend_exact), 'unresolved': failures}), flush=True)
        if failures:
            raise RuntimeError('Issued batch drained; unresolved issuance retained. No automatic retry or restart.')
    journal(a.output / 'complete.json', {'responses': responses, 'first_samples': True, 'usd': float(spend_exact)})


if __name__ == '__main__':
    main()
