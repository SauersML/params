"""Freeze and measure geometry-matched rank-one transplant controls.

Changing a carrier invalidates its old outcome labels. This runner measures every
arm afresh with the full model enabled, retaining full next-token distributions.
Controls preserve site/position counts and shared component identities across
positions. They are random carriers, not assumed causally inactive carriers.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import resource
import struct
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def train_task(name):
    if not isinstance(name, str) or not re.fullmatch(r'text[67]\d{3}', name):
        raise ValueError('only TRAIN text6000–7999 permitted')
    return name


def registry(path):
    with Path(path).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if size > 1 << 24:
            raise ValueError('unreasonable registry header')
        raw = stream.read(size)
    header = json.loads(raw)
    counts = {k[:-2]: v['shape'][0] for k, v in header.items() if k.endswith('.U')}
    if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError('empty/invalid rank-one registry')
    return counts, hashlib.sha256(raw).hexdigest()


def permute(variables, counts, task, seed):
    """One bijection per matrix, shared across positions and all named carriers."""
    train_task(task)
    tables = {}
    for site in sorted({g[0] for v in variables for g in v['group']}):
        if site not in counts:
            raise ValueError('unknown matrix or non-rank-one carrier')
        indices = list(range(counts[site]))
        random.Random(int(digest([task, site, seed]), 16)).shuffle(indices)
        tables[site] = indices
    result = []
    for v in variables:
        group = v['group']
        if not group or len(set(map(tuple, group))) != len(group):
            raise ValueError('empty/duplicate carrier members')
        if any(type(i) is not int or not 0 <= i < counts[s] or type(p) is not int or p < 0 for s, i, p in group):
            raise ValueError('invalid carrier coordinate')
        new = sorted([s, tables[s][i], p] for s, i, p in group)
        result.append({'name': v['name'], 'inputs': list(v['inputs']), 'group': new})
    return result


def prepare(checks, inputs, uv, reserved, seeds=(0, 1)):
    """Pure preparation; historical measured outcomes never enter the new protocol."""
    rows = [json.loads(line) for line in Path(checks).read_text().splitlines()]
    for row in rows:
        train_task(row['task'])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('distinct integer seeds required')
    counts, header_sha = registry(uv)
    tasks, cases, seen = {}, [], {}
    excluded = collections.Counter()
    for row in rows:
        tid = row['task']
        if not row['training_allowed'] or tid in reserved:
            excluded['reserved_or_ineligible'] += 1; continue
        raw = row['raw_selected_variables']
        if any(g[0] == 'block' for v in raw for g in v['group']):
            excluded['contains_whole_block'] += 1; continue
        selected = [{'name': v['name'], 'inputs': v['inputs'], 'group': v['group']} for v in raw]
        iv = row['original_checked_claim']['executed']['intervention']
        if iv.get('remove') or not iv.get('transplants'):
            raise ValueError('expected explicit transplants without removal')
        refs = [name for name, changes in iv['transplants']]
        if len(set(refs)) != len(refs) or set(refs) != {v['name'] for v in selected}:
            raise ValueError('transplants must bind every selected carrier exactly once')
        if row['bindings_sha256'] != {v['name']: digest(v['group']) for v in selected}:
            raise ValueError('original group binding mismatch')
        if tid not in tasks:
            path = Path(inputs) / tid / 'task.json'
            if file_sha(path) != row['task_sha256']:
                raise ValueError('frozen task digest mismatch')
            task = json.loads(path.read_text())
            if task['id'] != tid or task['split'] != 'train' or task['model'] != 'vpd4l' or len(task['prompts']) != 1:
                raise ValueError('unexpected task identity/model/prompts')
            p = task['prompts'][0]
            if not p['token_ids'] or not p['target_positions']:
                raise ValueError('missing executed input or target')
            tasks[tid] = {'token_ids': p['token_ids'], 'target_positions': p['target_positions'], 'task_sha256': row['task_sha256']}
        if tasks[tid]['task_sha256'] != row['task_sha256']:
            raise ValueError('inconsistent task source')
        length = len(tasks[tid]['token_ids'])
        all_members = [tuple(g) for v in selected for g in v['group']]
        if len(set(all_members)) != len(all_members):
            raise ValueError('overlapping carriers need a separately specified operator')
        if any(not 0 <= g[2] < length for g in all_members):
            raise ValueError('carrier outside executed input')
        key = digest([tid, selected, iv])
        if key in seen:
            seen[key]['source_claim_indices'].append(row['claim_index']); continue
        arms = [{'arm': 'original', 'variables': selected}]
        for seed in seeds:
            changed = permute(selected, counts, tid, seed)
            arms.append({'arm': f'permuted_{seed}', 'variables': changed,
                         'groups_changed': sum(set(map(tuple, a['group'])) != set(map(tuple, b['group'])) for a, b in zip(selected, changed))})
        case = {'id': key, 'task': tid, 'intervention': iv, 'source_claim_indices': [row['claim_index']], 'arms': arms}
        seen[key] = case; cases.append(case)
    if not cases:
        raise ValueError('no eligible rank-one transplants')
    return {'schema': 1, 'scope': 'Fresh full-model measurements; no inherited outcomes or free-English certificate.',
            'source_checks_sha256': file_sha(checks), 'registry_header_sha256': header_sha, 'component_counts': counts,
            'registry_scope': 'Preparation binds layout only; run requires full model-file checksums.',
            'reserved_tasks': sorted(reserved), 'seeds': list(seeds), 'excluded_rows': dict(excluded),
            'tasks': tasks, 'cases': cases, 'protocol_sha256': digest([tasks, cases])}


def run(protocol_path, model_manifest, out, device, worker, workers):
    """Single bounded shard. The manifest binds actual loader-selected model files."""
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('invalid worker shard')
    protocol = json.loads(Path(protocol_path).read_text())
    for tid in protocol['tasks']:
        train_task(tid)
    if digest([protocol['tasks'], protocol['cases']]) != protocol['protocol_sha256']:
        raise ValueError('protocol changed after preparation')
    if Path(out).exists():
        raise ValueError('preserve previous results; output directory already exists')
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / 'method/oracle'), str(root / 'method/vpd')]
    import torch
    import native, goal_eval, variables, semantic, vpd_model, mech, engine
    from types import SimpleNamespace
    paths = {'target': vpd_model.TARGET_DIR / 'model_step_99999.safetensors',
             'config': vpd_model.TARGET_DIR / 'model_config.yaml', 'tokenizer': mech.VPD4L_TOKENIZER,
             'decomposition': vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV}
    expected = json.loads(Path(model_manifest).read_text())
    actual = {name: file_sha(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError('loader-selected model content differs from frozen manifest')
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    started = time.monotonic()
    nat = native.Native(device)
    if nat.C != protocol['component_counts']:
        raise ValueError('loaded model registry differs from preparation')
    sc = variables.Scorer(goal_eval.GoalEval(nat), rows=1)
    destination = Path(out); destination.mkdir(parents=True)
    provenance = {'protocol_file_sha256': file_sha(protocol_path), 'model_files_sha256': actual,
                  'source_sha256': {m.__name__: file_sha(m.__file__) for m in [native, goal_eval, variables, semantic, vpd_model, mech, engine]},
                  'runner_sha256': file_sha(__file__), 'worker': worker, 'workers': workers,
                  'scope': protocol['scope'], 'device': device}
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for index, case in enumerate(protocol['cases']):
        if index % workers != worker:
            continue
        p = protocol['tasks'][case['task']]
        tx = SimpleNamespace(ids=p['token_ids'], targets=p['target_positions'], seed=native.task_seed(case['task']))
        distributions, summaries = {}, []
        for arm in case['arms']:
            parsed = variables.parse(arm['variables'], nat.C)
            checker = semantic.Checker(sc, tx, parsed)
            try:
                if arm['arm'] == 'original':
                    recipient_changes = case['intervention'].get('tokens') or {}
                    distributions['recipient_baseline'] = checker.text(recipient_changes)['lq'][-1].detach().float().cpu().clone()
                    recipient_ids = list(tx.ids)
                    for position, token in recipient_changes.items():
                        recipient_ids[int(position)] = int(token)
                    distributions['independent_reference'] = nat.reference([recipient_ids], tx.targets)[0, -1].detach().float().cpu().clone()
                    identity = {'tokens': recipient_changes, 'transplants': [(v['name'], recipient_changes) for v in arm['variables']]}
                    distributions['identity_transplant'] = checker.run(identity).detach().float().cpu().clone()
                q = checker.run(case['intervention']).detach().float().cpu()
                if q.ndim != 1 or not torch.isfinite(q).all() or abs(float(q.logsumexp(0))) > 1e-4:
                    raise ValueError('invalid normalized full-model log distribution')
                distributions[arm['arm']] = q
                top = int(q.argmax())
                summaries.append({'arm': arm['arm'], 'top': top, 'top_token': mech.tokenizer('vpd4l').decode([top], skip_special_tokens=False),
                                  'p_top': float(q[top].exp()), 'groups_sha256': {v['name']: digest(v['group']) for v in arm['variables']}})
            finally:
                checker.close()
        original = distributions['original']
        for summary in summaries:
            q = distributions[summary['arm']]
            summary['kl_original_to_arm_bits'] = float((original.exp() * (original - q)).sum()) / math.log(2)
        from safetensors.torch import save_file
        tensor_path = destination / (case['id'] + '.safetensors')
        save_file(distributions, str(tensor_path))
        base, identity = distributions['recipient_baseline'], distributions['identity_transplant']
        reference = distributions['independent_reference']
        record = {'case': case, 'measurements': summaries, 'distributions_sha256': file_sha(tensor_path),
                  'identity_transplant_kl_bits': float((base.exp() * (base - identity)).sum()) / math.log(2),
                  'identity_transplant_max_abs_logprob': float((base - identity).abs().max()),
                  'full_engine_reference_kl_bits': float((reference.exp() * (reference - base)).sum()) / math.log(2),
                  'full_engine_reference_max_abs_logprob': float((reference - base).abs().max())}
        (destination / (case['id'] + '.json')).write_text(json.dumps(record) + '\n')
        print(case['task'], case['id'][:12], [r['top'] for r in summaries], flush=True)
    after = {n: file_sha(p) for n, p in paths.items()}
    if after != actual:
        raise ValueError('model files changed during measurement; results are incomplete')
    (destination / 'complete.json').write_text(json.dumps({'cases': sum(i % workers == worker for i in range(len(protocol['cases']))),
                                                         'model_files_sha256_after': after, 'seconds': time.monotonic() - started,
                                                         'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** (3 if sys.platform == 'darwin' else 2)),
                                                         'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3 if device == 'cuda' else None}) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for name in ('checks', 'inputs', 'uv', 'reserved_tasks', 'out'):
        prep.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    measure = commands.add_parser('run')
    for name in ('protocol', 'model_manifest', 'out'):
        measure.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    measure.add_argument('--device', choices=('cpu', 'mps', 'cuda'), required=True)
    measure.add_argument('--worker', type=int, default=0); measure.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    if args.command == 'prepare':
        if args.out.exists():
            raise ValueError('preserve previous protocol')
        reserved = json.loads(args.reserved_tasks.read_text())
        if not isinstance(reserved, list) or not reserved or any(not isinstance(t, str) for t in reserved):
            raise ValueError('explicit nonempty reserved task list required')
        result = prepare(args.checks, args.inputs, args.uv, set(reserved))
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(len(result['cases']), 'cases', sum(len(c['arms']) for c in result['cases']), 'fresh measurements required')
    else:
        run(args.protocol, args.model_manifest, args.out, args.device, args.worker, args.workers)
