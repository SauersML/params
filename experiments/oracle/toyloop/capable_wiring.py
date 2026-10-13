"""Opt-in frozen toy oracle pilot: Qwen3-8B base versus two-epoch LoRA SFT.

No RL, labels, prompt edits, English stripping, output prefixes, or best-of selection.
Prepare is offline. Run makes paid calls and requires an explicit project and cap.
Physical circuit scoring is a separate frozen cd8d4a9 executor job; this module
does not certify English or intervention fidelity from point circuit scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random

MODEL = 'Qwen/Qwen3-8B'
PROJECT = '3f598f31-2ffa-4eb4-b302-909eb234fb7d'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
TRAIN_SHA = '98ebba8b9e874d3c96f6bc1af64a4624b3f088fbf726e84eb99a83dcfcc965ee'
PANEL_REFERENCE_SHA = '51c8f4ccf93fb99c40a7bf5e8a910f287bae4fce4fb8ec3bcdf6962c6f0795bd'
PRICES = {'prefill': .195, 'cached': .039, 'sampled': .60, 'trained': .44}
LR = .00047297908091376354
RECIPE = {'epochs': 2, 'batch': 8, 'seed': 0, 'updates': 58, 'rank': 32, 'lr': LR,
          'sampler': 'shuffled_epoch_short_tail', 'temperature': 0., 'max_tokens': 416}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def schedule(count=226, epochs=2, batch=8, seed=0):
    """No epoch-crossing repeats inside a batch; every row exactly once per epoch."""
    if count < 1 or epochs < 1 or batch < 1:
        raise ValueError('positive schedule dimensions required')
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        order = list(range(count)); rng.shuffle(order)
        result.extend(order[i:i + batch] for i in range(0, count, batch))
    return result


def validate_data(train_path, panel_path, panel_reference, allow_reference_drift=False):
    if sha(train_path) != TRAIN_SHA:
        raise ValueError('different frozen training bytes')
    train, panel = rows(train_path), rows(panel_path)
    if sha(panel_reference) != PANEL_REFERENCE_SHA:
        raise ValueError('different frozen panel reference')
    reference = rows(panel_reference)
    expected = {r['id']: r['prompt_sha256'] for r in reference}
    if len(train) != 226 or len({r['id'] for r in train}) != 226:
        raise ValueError('expected original 226 unique TRAIN examples')
    if len(panel) != 24 or len({r['id'] for r in panel}) != 24:
        raise ValueError('expected frozen 24-case panel')
    if [r['id'] for r in panel] != [r['id'] for r in reference]:
        raise ValueError('frozen panel order changed')
    for r in train:
        if not isinstance(r['prompt'], str) or not isinstance(r['completion'], str) or '\nEnglish: ' not in r['completion']:
            raise ValueError('complete original prompt/circuit/English required')
    for r in panel:
        if (hashlib.sha256(r['prompt'].encode()).hexdigest() != r['prompt_sha256']
                or (not allow_reference_drift and expected.get(r['id']) != r['prompt_sha256'])):
            raise ValueError('panel prompt bytes differ from frozen evaluation')
    return train, panel


def renderer(tokenizer_path):
    # No service or model client is instantiated by this offline operation.
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return renderers.get_renderer('qwen3_disable_thinking', tok), tok


def datum(row, rr):
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.renderers import TrainOnWhat
    return conversation_to_datum([{'role': 'user', 'content': row['prompt']},
                                  {'role': 'assistant', 'content': row['completion']}],
                                 rr, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE)


def generation_prompt(row, rr):
    return rr.build_generation_prompt([{'role': 'user', 'content': row['prompt']}])


def prepare(train_path, panel_path, panel_reference, tokenizer_path, out, allow_reference_drift=False):
    if Path(out).exists():
        raise ValueError('immutable preflight output already exists')
    train, panel = validate_data(train_path, panel_path, panel_reference, allow_reference_drift)
    rr, tok = renderer(tokenizer_path)
    data = [datum(r, rr) for r in train]
    lengths = [d.model_input.length for d in data]
    supervised = [int(d.loss_fn_inputs['weights'].to_torch().gt(0).sum()) for d in data]
    answers = [len(tok.encode(r['completion'], add_special_tokens=False)) + 1 for r in train]
    prompt_lengths = [generation_prompt(r, rr).length for r in panel]
    if max(answers) > 416 or max(supervised) > 416 or max(lengths) > 32768 or max(prompt_lengths) + 416 + 16 > 32768:
        raise ValueError('frozen complete targets/panel do not fit unchanged cap')
    batches = schedule()
    trained = sum(lengths[i] for b in batches for i in b)
    prefill, sampled = 2 * sum(prompt_lengths), 48 * 416
    upper = (trained * PRICES['trained'] + prefill * PRICES['prefill'] + sampled * PRICES['sampled']) / 1e6
    result = {'schema': 1, 'model': MODEL, 'project': PROJECT, 'tokenizer_revision': REVISION,
              'tokenizer_path': str(Path(tokenizer_path).resolve()),
              'tokenizer_files': {p.name: sha(p) for p in sorted(Path(tokenizer_path).iterdir()) if p.is_file()},
              'train_path': str(Path(train_path).resolve()), 'train_sha256': sha(train_path),
              'panel_path': str(Path(panel_path).resolve()), 'panel_sha256': sha(panel_path),
              'panel_reference': str(Path(panel_reference).resolve()), 'panel_reference_sha256': PANEL_REFERENCE_SHA,
              'allow_reference_drift': allow_reference_drift,
              'reference_prompt_mismatches': [{'id': r['id'], 'historical_sha256': old['prompt_sha256'], 'paired_trial_sha256': r['prompt_sha256']}
                  for r, old in zip(panel, rows(panel_reference)) if r['prompt_sha256'] != old['prompt_sha256']],
              'comparison_scope': 'Base and trained receive identical frozen trial bytes. Historical Smol comparison is secondary where rendered prompt hashes differ; no attribution of drift cause without evidence.',
              'runner_sha256': sha(__file__), 'package_versions': {p: importlib.metadata.version(p) for p in ('tinker', 'tinker-cookbook', 'transformers')},
              'recipe': RECIPE,
              'sample_ids': [[train[i]['id'] for i in b] for b in batches],
              'token_fit': {'training_lengths': lengths, 'answer_plus_eos_lengths': answers, 'supervised_lengths': supervised, 'panel_prompt_lengths': prompt_lengths},
              'prices_per_million': PRICES, 'cost_upper_usd': upper,
              'cost_upper_tokens': {'trained': trained, 'prefill': prefill, 'sampled': sampled},
              'pricing_url': 'https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/',
              'identity_scope': 'Local tokenizer revision is pinned; hosted base weights are identified by provider model name/metadata, not a downloadable tensor checksum. Seed fixes TRAIN ordering only.',
              'scope': 'Shared development panel; capable model plus LoRA recipe, not pure capacity isolation. Full English retained but unverified. No pristine generalization or VPD certificate.'}
    Path(out).write_text(json.dumps(result, indent=2) + '\n')
    return result


def run(manifest_path, out, cap):
    m = json.loads(Path(manifest_path).read_text())
    if m['runner_sha256'] != sha(__file__) or m['model'] != MODEL or m['project'] != PROJECT or os.environ.get('TINKER_PROJECT_ID') != PROJECT:
        raise ValueError('runner/project differs from frozen preflight')
    if not math.isfinite(cap) or cap < m['cost_upper_usd']:
        raise ValueError('cap below no-cache full-output upper bound')
    if m['recipe'] != RECIPE or m['tokenizer_revision'] != REVISION:
        raise ValueError('recipe/tokenizer revision differs from frozen pilot')
    if Path(out).exists():
        raise ValueError('immutable output already exists')
    train, panel = validate_data(m['train_path'], m['panel_path'], m['panel_reference'], m['allow_reference_drift'])
    mismatches = [{'id': r['id'], 'historical_sha256': old['prompt_sha256'], 'paired_trial_sha256': r['prompt_sha256']}
        for r, old in zip(panel, rows(m['panel_reference'])) if r['prompt_sha256'] != old['prompt_sha256']]
    if mismatches != m['reference_prompt_mismatches']:
        raise ValueError('historical reference disclosure changed')
    if sha(m['panel_path']) != m['panel_sha256']:
        raise ValueError('panel changed after preflight')
    if any(importlib.metadata.version(p) != v for p, v in m['package_versions'].items()):
        raise ValueError('dependency changed after preflight')
    token_path = Path(m['tokenizer_path'])
    if {p.name: sha(p) for p in sorted(token_path.iterdir()) if p.is_file()} != m['tokenizer_files']:
        raise ValueError('tokenizer changed after preflight')
    rr, tok = renderer(token_path)
    data = [datum(r, rr) for r in train]
    prompts = [generation_prompt(r, rr) for r in panel]
    if ([d.model_input.length for d in data] != m['token_fit']['training_lengths']
            or [p.length for p in prompts] != m['token_fit']['panel_prompt_lengths']):
        raise ValueError('rendering changed after preflight')
    batches = schedule()
    if [[train[i]['id'] for i in b] for b in batches] != m['sample_ids']:
        raise ValueError('training schedule changed')
    cost_tokens = {'trained': sum(data[i].model_input.length for b in batches for i in b),
                   'prefill': 2 * sum(p.length for p in prompts), 'sampled': 48 * 416}
    if (cost_tokens != m['cost_upper_tokens'] or m['prices_per_million'] != PRICES
            or abs(sum(cost_tokens.get(k, 0) * PRICES[k] for k in PRICES) / 1e6 - m['cost_upper_usd']) > 1e-12):
        raise ValueError('cost contract changed after preflight')
    import tinker
    # Root must reserve the complete upper bound before invoking this paid mode.
    service = tinker.ServiceClient(project_id=PROJECT)
    destination = Path(out); destination.mkdir()
    (destination / 'manifest.json').write_bytes(Path(manifest_path).read_bytes())
    spend = 0.

    def before_call(tokens):
        if spend + sum(tokens.get(k, 0) * v for k, v in PRICES.items()) / 1e6 > cap + 1e-12:
            raise ValueError('remaining cap cannot reserve next call worst case')

    def account(kind, tokens):
        nonlocal spend
        cost = sum(tokens.get(k, 0) * v for k, v in PRICES.items()) / 1e6
        spend += cost
        with (destination / 'spend.jsonl').open('a') as f:
            f.write(json.dumps({'kind': kind, **tokens, 'usd': cost, 'total_usd': spend}) + '\n')
        if spend > cap + 1e-12:
            raise ValueError('actual spend exceeded frozen cap')

    def sample(sampler, name):
        params = tinker.SamplingParams(max_tokens=416, temperature=0., stop=rr.get_stop_sequences())
        with (destination / (name + '.jsonl')).open('x') as f:
            for row, prompt in zip(panel, prompts):
                before_call({'prefill': prompt.length, 'sampled': 416})
                res = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
                if len(res.sequences) != 1:
                    raise ValueError('expected one first greedy response')
                s = res.sequences[0]
                hits = res.prompt_cache_hit_tokens
                raw = {'id': row['id'], 'prompt_sha256': row['prompt_sha256'], 'token_ids': list(s.tokens),
                       'finish': str(s.stop_reason), 'sample': 0, 'prompt_cache_hit_tokens': hits}
                # Preserve charged output even if renderer parsing fails.
                with (destination / (name + '_raw.jsonl')).open('a') as raw_file:
                    raw_file.write(json.dumps(raw) + '\n'); raw_file.flush(); os.fsync(raw_file.fileno())
                if not 0 <= hits <= prompt.length:
                    raise ValueError('invalid provider prompt cache accounting')
                account('sample_' + name, {'prefill': prompt.length - hits, 'cached': hits, 'sampled': len(s.tokens)})
                try:
                    parsed, _ = rr.parse_response(s.tokens)
                    from tinker_cookbook.renderers import get_text_content
                    raw['completion'] = get_text_content(parsed)
                except Exception as exc:
                    raw.update(completion=None, parse_error=type(exc).__name__ + ': ' + str(exc))
                f.write(json.dumps(raw) + '\n'); f.flush(); os.fsync(f.fileno())

    sample(service.create_sampling_client(base_model=MODEL), 'base')
    training = service.create_lora_training_client(base_model=MODEL, rank=32)
    (destination / 'provider_training_info.json').write_text(json.dumps(training.get_info().model_dump(mode='json')) + '\n')
    for step, batch in enumerate(batches):
        selected = [data[i] for i in batch]
        trained_tokens = sum(d.model_input.length for d in selected)
        before_call({'trained': trained_tokens})
        fb = training.forward_backward(selected, loss_fn='cross_entropy')
        op = training.optim_step(tinker.AdamParams(learning_rate=LR * (1 - step / len(batches)), beta1=.9, beta2=.95, eps=1e-8))
        fb.result()
        account('sft', {'trained': trained_tokens})
        op.result()
        with (destination / 'steps.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, 'sample_ids': m['sample_ids'][step]}) + '\n')
    state = training.save_state('capable-toy-wiring-two-epoch').result().path
    (destination / 'state.json').write_text(json.dumps({'state': state, 'base': MODEL}) + '\n')
    sample(training.save_weights_and_get_sampling_client(), 'trained')
    (destination / 'complete.json').write_text(json.dumps({'actual_usd': spend, 'upper_usd': m['cost_upper_usd']}) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('train', 'panel', 'panel-reference', 'tokenizer', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--allow-reference-drift', action='store_true', help='Explicit newly frozen paired panel; disclose every historical prompt SHA mismatch')
    p = sub.add_parser('run')
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cap-usd', type=float, required=True)
    a = ap.parse_args()
    if a.command == 'prepare':
        print(json.dumps(prepare(a.train, a.panel, a.panel_reference, a.tokenizer, a.out, a.allow_reference_drift)))
    else:
        run(a.manifest, a.out, a.cap_usd)
