"""Offline frozen four-cell TRAIN-support x relation-loss pilot preparation.

The V-support arm exposes the previously held-out multihop family. Matched
head coordinates and distractor counts do not isolate all family differences.
All arms clone one saved provider initialization and optimizer state. This
identifies a common hosted state, without a downloaded tensor certificate.
Preparation is offline; paid run requires the root's separate authorization.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

CORE_SHA = '98ebba8b9e874d3c96f6bc1af64a4624b3f088fbf726e84eb99a83dcfcc965ee'
PROJECT = '3f598f31-2ffa-4eb4-b302-909eb234fb7d'
CELLS = [('induction', 1), ('induction', 8), ('multihop', 1), ('multihop', 8)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocation_equal(actual, frozen):
    """Only precast binary64 sums tolerate reduction-order roundoff.

    Python 3.12 improves float summation over 3.11. Four gamma_N bounds cover
    weighted-sum normalization and two summaries. Stored FP32 masses, counts,
    masks, and all other fields remain exact comparisons.
    """
    if actual.keys() != frozen.keys():
        return False
    n = actual.get('supervised_tokens')
    if type(n) is not int or not 0 < n <= 416:
        return False
    u = 2. ** -53
    gamma = n * u / (1 - n * u)
    for key, value in actual.items():
        expected = frozen[key]
        if key in ('new_total_mass', 'new_relation_mass'):
            if (not math.isfinite(value) or not math.isfinite(expected)
                    or abs(value - expected) > 4 * gamma * max(abs(value), abs(expected))):
                return False
        elif value != expected:
            return False
    return True


def prepare(root, panel, base_samples, protocol, out):
    """Freeze already physically checked/free-tokenizer-fit data; no data selection."""
    import capable_wiring as pilot
    root, panel, base_samples, protocol, out = map(Path, (root, panel, base_samples, protocol, out))
    if out.exists():
        raise ValueError('immutable preparation output already exists')
    build = json.loads((root / 'manifest.json').read_text())
    fit = json.loads((root / 'token_fit.json').read_text())
    if (build['core_sha256'] != CORE_SHA or build['rows_per_arm'] != 256
            or build['updates'] != 64 or build['epochs'] != 2 or build['batch'] != 8):
        raise ValueError('unexpected augmentation recipe')
    questions = pilot.rows(panel)
    baseline = pilot.rows(base_samples)
    if len(questions) != 24 or len(baseline) != 24 or len({r['id'] for r in questions}) != 24:
        raise ValueError('common frozen panel must contain24 distinct cases')
    for q, b in zip(questions, baseline):
        if q['id'] != b['id'] or q['prompt_sha256'] != b['prompt_sha256'] or hashlib.sha256(q['prompt'].encode()).hexdigest() != q['prompt_sha256']:
            raise ValueError('common archived base does not bind frozen panel bytes')
    data = {}
    for family in ('induction', 'multihop'):
        path = root / f'{family}_train_inputs.jsonl'
        if digest(path) != build['data_sha256'][family]:
            raise ValueError('augmentation bytes changed after validation')
        raw = path.read_bytes().splitlines(keepends=True)
        if len(raw) != 256 or hashlib.sha256(b''.join(raw[:226])).hexdigest() != CORE_SHA:
            raise ValueError('frozen226 core changed')
        rows = pilot.rows(path)
        stats = fit[family]['stats']
        if len({r['id'] for r in rows}) != 256 or [r['id'] for r in rows] != [s['id'] for s in stats]:
            raise ValueError('token fit does not bind ordered256 distinct rows')
        for s in stats:
            if (not all(math.isfinite(s[k]) for k in ('old_total_mass', 'stored_total_mass'))
                    or not s['old_total_mass'] > 0
                    or not 0 < s['supervised_tokens'] <= 416 or not 0 < s['pair_tokens'] <= 32768
                    or abs(s['old_total_mass'] - s['stored_total_mass']) > 1e-6 * s['old_total_mass']):
                raise ValueError('stored supervision/fit invalid')
        evidence = [e for e in build['augmentation'] if e['family'] == family]
        if len(evidence) != 30 or {e['slot']['slot'] for e in evidence} != set(range(30)):
            raise ValueError('augmentation slot proof incomplete')
        for e in evidence:
            if not all(digest(root / p) == h for p, h in e['files'].items()):
                raise ValueError('physical teacher artifact changed')
            measured = [r['kl_bits'] for r in e['teacher_score']['per_text']]
            if not measured or any(not math.isfinite(x) or not 0 <= x <= 1e-3 for x in measured):
                raise ValueError('physical teacher exceeds frozen tolerance')
        batches = pilot.schedule(256, 2, 8, 0)
        train_tokens = sum(stats[i]['pair_tokens'] for batch in batches for i in batch)
        if train_tokens != fit[family]['train_tokens_per_arm']:
            raise ValueError('token cost does not bind full schedule')
        data[family] = {'path': str(path.resolve()), 'sha256': digest(path),
                        'sample_ids': [[rows[i]['id'] for i in b] for b in batches],
                        'train_tokens': train_tokens}
    upper = sum(data[f]['train_tokens'] * 2 * .44 / 1e6 for f in data) + fit['eval_upper_usd']
    if abs(upper - fit['four_cell_upper_usd']) > 1e-12 or not 0 < upper <= 1.5:
        raise ValueError('four-cell whole-call upper bound invalid')
    result = {'schema': 1, 'project': PROJECT, 'cells': CELLS, 'datasets': data,
              'init_seed': 0, 'epochs': 2, 'batch': 8, 'steps_per_cell': 64,
              'common_base_path': str(base_samples.resolve()), 'common_base_sha256': digest(base_samples),
              'panel_path': str(panel.resolve()), 'panel_sha256': digest(panel),
              'protocol_path': str(protocol.resolve()), 'protocol_sha256': digest(protocol), 'build_manifest_sha256': digest(root / 'manifest.json'),
              'token_fit_sha256': digest(root / 'token_fit.json'), 'cost_upper_usd': upper,
              'runner_sha256': digest(__file__), 'loss_helper_sha256': digest(Path(__file__).with_name('relation_loss.py')),
              'pilot_helper_sha256': digest(Path(__file__).with_name('capable_wiring.py')),
              'paid_status': 'not authorized by this offline preparation',
              'counterfactual_reporting': 'Retain distribution KL plus active/inactive and deletion/half-scale breakdown; deletion-only top accuracy can favor empty circuits.',
              'identity_scope': 'One seed0 provider initialization saved and cloned with optimizer into all4 arms; no downloaded tensor certificate.'}
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


def run(manifest_path, out, tokenizer_path, cap):
    """Root-authorized paid mode; four explicit-seed clients, no paid retry."""
    import importlib.metadata
    import math
    import os
    import capable_wiring as pilot
    import relation_loss
    manifest_path, out, tokenizer_path = map(Path, (manifest_path, out, tokenizer_path))
    m = json.loads(manifest_path.read_text())
    if (out.exists() or not math.isfinite(cap) or cap < m['cost_upper_usd']
            or cap > 1.5 or os.environ.get('TINKER_PROJECT_ID') != PROJECT
            or m['runner_sha256'] != digest(__file__)
            or m['pilot_helper_sha256'] != digest(Path(__file__).with_name('capable_wiring.py'))
            or m['loss_helper_sha256'] != digest(Path(__file__).with_name('relation_loss.py'))):
        raise ValueError('immutable source/output/project/whole-trial cap guard failed')
    for key in ('panel', 'common_base'):
        if digest(m[key + '_path']) != m[key + '_sha256']:
            raise ValueError('frozen panel/base changed')
    if digest(m['protocol_path']) != m['protocol_sha256']:
        raise ValueError('predeclared analysis protocol changed')
    rr, tok = pilot.renderer(tokenizer_path)
    data, all_stats = {}, {}
    for family in ('induction', 'multihop'):
        binding = m['datasets'][family]
        if digest(binding['path']) != binding['sha256']:
            raise ValueError('frozen TRAIN changed')
        rows = pilot.rows(binding['path'])
        dataset_root = Path(binding['path']).parent
        if (digest(dataset_root / 'manifest.json') != m['build_manifest_sha256']
                or digest(dataset_root / 'token_fit.json') != m['token_fit_sha256']):
            raise ValueError('physical/token-fit proof changed')
        fit = json.loads((dataset_root / 'token_fit.json').read_text())
        proof = json.loads((dataset_root / 'manifest.json').read_text())
        for evidence in proof['augmentation']:
            if not all(digest(dataset_root / path) == checksum for path, checksum in evidence['files'].items()):
                raise ValueError('physical teacher artifact changed after preparation')
        if {p.name: digest(p) for p in tokenizer_path.iterdir() if p.is_file()} != fit['tokenizer_files']:
            raise ValueError('local tokenizer bytes changed')
        expected_versions = dict(line.split('==', 1) for line in (dataset_root / 'fit_dependency_lock.txt').read_text().splitlines() if '==' in line)
        for package in ('tinker', 'tinker-cookbook', 'transformers', 'tokenizers'):
            if importlib.metadata.version(package) != expected_versions[package]:
                raise ValueError('rendering/client dependency changed')
        ordinary = [pilot.datum(row, rr) for row in rows]
        balanced, stats = [], []
        for row, d, frozen in zip(rows, ordinary, fit[family]['stats']):
            edited, stat = relation_loss.apply_to_datum(d, row['completion'], tok, 8)
            if not allocation_equal(stat, {k: v for k, v in frozen.items() if k not in ('id', 'pair_tokens')}):
                raise ValueError('exact relation-token allocation changed')
            if d.model_input.length != frozen['pair_tokens']:
                raise ValueError('TRAIN rendering length changed')
            balanced.append(edited); stats.append(stat)
        batches = pilot.schedule(256, 2, 8, 0)
        if [[rows[i]['id'] for i in b] for b in batches] != binding['sample_ids']:
            raise ValueError('TRAIN schedule changed')
        data[(family, 1)], data[(family, 8)] = ordinary, balanced
        all_stats[family] = stats
    panel = pilot.rows(m['panel_path']); prompts = [pilot.generation_prompt(row, rr) for row in panel]
    prefill = sum(p.length for p in prompts)
    upper = sum(d.model_input.length for values in data.values() for d in values) * 2 * .44 / 1e6 + 4 * (prefill * .195 + 24 * 416 * .6) / 1e6
    if abs(upper - m['cost_upper_usd']) > 1e-12:
        raise ValueError('exact whole-call reservation changed')
    import tinker
    service = tinker.ServiceClient(project_id=PROJECT)
    out.mkdir(); (out / 'manifest.json').write_bytes(manifest_path.read_bytes())
    (out / 'allocation.json').write_text(json.dumps(all_stats))
    spend, reserved = 0., 0.

    def record(name, value):
        with (out / name).open('a') as f:
            f.write(json.dumps(value) + '\n'); f.flush(); os.fsync(f.fileno())

    def reserve(kind, tokens):
        nonlocal reserved
        amount = sum(tokens.get(k, 0) * p for k, p in pilot.PRICES.items()) / 1e6
        if spend + reserved + amount > cap + 1e-12:
            raise ValueError('remaining cap cannot reserve queued call')
        reserved += amount
        record('reservations.jsonl', {'kind': kind, 'tokens': tokens, 'upper_usd': amount})
        return amount

    def settle(kind, tokens, reservation):
        nonlocal reserved, spend
        usd = sum(tokens.get(k, 0) * p for k, p in pilot.PRICES.items()) / 1e6
        if usd > reservation + 1e-12:
            raise ValueError('provider charge exceeds reserved call')
        reserved -= reservation; spend += usd
        record('spend.jsonl', {'kind': kind, **tokens, 'usd': usd, 'total_usd': spend,
                              'released_reservation_usd': reservation, 'remaining_reserved_usd': reserved})

    initial = service.create_lora_training_client(base_model=pilot.MODEL, rank=32, seed=0)
    initial_state = initial.save_state('factorial-common-seed0-init').result().path
    (out / 'common_init.json').write_text(json.dumps({'state': initial_state, 'seed': 0}))
    clients = {}
    for family, multiplier in CELLS:
        name = f'{family}_m{multiplier}'
        client = service.create_training_client_from_state_with_optimizer(initial_state)
        clients[(family, multiplier)] = client
        info = client.get_info().model_dump(mode='json')
        (out / (name + '_provider.json')).write_text(json.dumps(info))
        if (info.get('is_lora') is not True or info.get('lora_rank') != 32
                or info.get('model_data', {}).get('model_name') != pilot.MODEL):
            raise ValueError('provider model/rank contract differs')
    # Reserve each call before dispatch. Four independent streams overlap one update round.
    for step, batch in enumerate(batches):
        pending, errors = [], []
        for cell, client in clients.items():
            family, multiplier = cell; name = f'{family}_m{multiplier}'
            selected = [data[cell][i] for i in batch]
            tokens = {'trained': sum(d.model_input.length for d in selected)}
            try:
                reservation = reserve(name + '_sft', tokens)
                fb = client.forward_backward(selected, loss_fn='cross_entropy')
                pending.append((name, family, fb, None, tokens, reservation))
                op = client.optim_step(tinker.AdamParams(learning_rate=pilot.LR * (1 - step / 64), beta1=.9, beta2=.95, eps=1e-8))
                pending[-1] = (name, family, fb, op, tokens, reservation)
            except Exception as exc:
                errors.append(exc)
                record('errors.jsonl', {'cell': name, 'step': step, 'stage': 'submission', 'error_type': type(exc).__name__})
                break  # Drain every submitted forward even when optimizer submission failed.
        for name, family, fb, op, tokens, reservation in pending:
            try:
                fb.result(); settle(name + '_sft', tokens, reservation)
            except Exception as exc:
                errors.append(exc)
        for name, family, fb, op, tokens, reservation in pending:
            if op is None:
                continue
            try:
                op.result(); record('steps.jsonl', {'cell': name, 'step': step, 'sample_ids': m['datasets'][family]['sample_ids'][step]})
            except Exception as exc:
                errors.append(exc)
        if errors:
            record('errors.jsonl', {'stage': 'drained_round', 'step': step, 'remaining_reserved_usd': reserved})
            raise errors[0]  # Unsettled reservations remain visible; never retry paid calls.
    params = tinker.SamplingParams(max_tokens=416, temperature=0., stop=rr.get_stop_sequences())
    for cell, client in clients.items():
        family, multiplier = cell; name = f'{family}_m{multiplier}'
        state = client.save_state(name + '-two-epoch').result().path
        (out / (name + '_state.json')).write_text(json.dumps({'state': state}))
        sampler = client.save_weights_and_get_sampling_client()
        for row, prompt in zip(panel, prompts):
            reservation = reserve(name + '_sample', {'prefill': prompt.length, 'sampled': 416})
            result = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
            if len(result.sequences) != 1:
                raise ValueError('expected one greedy sample')
            sequence = result.sequences[0]; hits = result.prompt_cache_hit_tokens
            raw = {'id': row['id'], 'prompt_sha256': row['prompt_sha256'], 'token_ids': list(sequence.tokens), 'finish': str(sequence.stop_reason), 'sample': 0, 'prompt_cache_hit_tokens': hits}
            record(name + '_raw.jsonl', raw)
            if not 0 <= hits <= prompt.length:
                raise ValueError('invalid provider cache accounting')
            settle(name + '_sample', {'prefill': prompt.length-hits, 'cached': hits, 'sampled': len(sequence.tokens)}, reservation)
            try:
                parsed, _ = rr.parse_response(sequence.tokens)
                from tinker_cookbook.renderers import get_text_content
                raw['completion'] = get_text_content(parsed)
            except Exception as exc:
                raw.update(completion=None, parse_error=type(exc).__name__ + ': ' + str(exc))
            record(name + '.jsonl', raw)
    (out / 'complete.json').write_text(json.dumps({'actual_usd': spend, 'reserved_usd': reserved, 'upper_usd': upper}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preflight = commands.add_parser('prepare')
    for argument in ('root', 'panel', 'base-samples', 'protocol', 'out'):
        preflight.add_argument('--' + argument, required=True)
    paid = commands.add_parser('run')
    for argument in ('manifest', 'out', 'tokenizer'):
        paid.add_argument('--' + argument, required=True)
    paid.add_argument('--cap', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.root, args.panel, args.base_samples, args.protocol, args.out)))
    else:
        run(args.manifest, args.out, args.tokenizer, args.cap)
