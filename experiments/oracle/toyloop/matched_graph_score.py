"""Frozen first-response circuit physics; no oracle API, repair, or selection.

References are computed once per world and shared across policies. Missing,
truncated and invalid answers remain failures in every planned denominator.
New-model individual edits and additional joint-edit combinations are reported
separately. This is within-family toy transfer, not a VPD/English certificate.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

DIAGNOSTIC_MANIFEST_SHA = '2a2e657f77f577bc106b5be718122545f6cc2a8d6c5707009547778ef5c4ab8d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def bound_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('relative bound asset path required')
    return Path(root) / path


def bind_responses(panel, responses):
    expected = {(r['stratum'], r['id']): r for r in panel}
    if len(expected) != len(panel):
        raise ValueError('duplicate panel case')
    found = {}
    for row in responses:
        key = (row['stratum'], row['id'])
        if key not in expected or key in found:
            raise ValueError('unexpected/duplicate first response')
        if row['sample'] != 0 or row['prompt_sha256'] != expected[key]['prompt_sha256']:
            raise ValueError('first-response prompt binding failed')
        found[key] = row
    return [found.get((row['stratum'], row['id'])) for row in panel]


def answer_error(row, answer_format='graph_only'):
    if answer_format not in ('graph_only', 'legacy_mermaid_english'):
        raise ValueError('unknown frozen answer format')
    if row is None:
        return 'missing_first_response'
    if row.get('parse_error'):
        return 'saved_parse_error'
    if str(row['finish']).lower() not in ('stop', 'stopreason.stop'):
        return 'unfinished_first_response'
    text = row.get('completion')
    if not isinstance(text, str):
        return 'missing_completion'
    # Retention prompts explicitly request the historical Mermaid + English
    # format. Preserve their original circuit.parse validity contract; English
    # and token arrows remain unexecuted annotations, not causal evidence.
    if answer_format == 'legacy_mermaid_english':
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    node = r'(?:L\d+[HN]\d+|out)'
    if (not lines or lines[0] != 'flowchart LR' or not any('-->' in x for x in lines[1:])
            or any(not re.fullmatch(rf'{node}(?:\s+-->\s+{node})?', x) for x in lines[1:])):
        return 'invalid_graph_only_answer'
    return None


def summarize(rows, expected_keys):
    keys = [(r['stratum'], r['case'], r['arm'], r['edit_index'], r['input']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise ValueError('incomplete or duplicate Cartesian result coverage')
    groups = defaultdict(list)
    for row in rows:
        if row['valid'] and (not math.isfinite(row['kl_bits']) or row['kl_bits'] < 0):
            raise ValueError('invalid measured KL')
        groups[(row['stratum'], row['arm'], row['kind'], row['input_role'])].append(row)
    result = []
    for (stratum, arm, kind, role), values in sorted(groups.items()):
        cases = defaultdict(list)
        for row in values:
            cases[row['case']].append(row)
        model_rows = []
        for case, worlds in sorted(cases.items()):
            valid = [r for r in worlds if r['valid']]
            passed = sum(r['valid'] and r['kl_bits'] <= .001 for r in worlds)
            model_rows.append({'case': case, 'worlds': len(worlds), 'valid_worlds': len(valid),
                               'kl_pass_fraction': passed / len(worlds),
                               'all_worlds_pass': passed == len(worlds),
                               'top_agreement': sum(r['valid'] and r['top'] == r['reference_top']
                                                    for r in worlds) / len(worlds),
                               'worst_valid_kl_bits': max((r['kl_bits'] for r in valid), default=None)})
        result.append({'stratum': stratum, 'arm': arm, 'kind': kind, 'input_role': role,
                       'models': len(cases), 'worlds': len(values), 'per_model': model_rows,
                       'all_worlds_pass_models': sum(r['all_worlds_pass'] for r in model_rows),
                       'model_mean_kl_pass_fraction': sum(r['kl_pass_fraction'] for r in model_rows) / len(cases),
                       'model_mean_top_agreement': sum(r['top_agreement'] for r in model_rows) / len(cases)})
    return {'scope': 'within-family toy model transfer; no English or weight-use proof',
            'unit_of_replication': 'model, not intervention worlds',
            'invalid_policy': 'all planned worlds retained; undefined KL never replaced by zero',
            'groups': result}


def partition(panel, workers):
    """Largest-first greedy world balancing; stable original-index ties."""
    if type(workers) is not int or not 1 <= workers <= len(panel):
        raise ValueError('worker count must be within panel size')
    groups, loads = [[] for _ in range(workers)], [0] * workers
    counts = [len(row['inputs']) * len(row['edits']) for row in panel]
    for index in sorted(range(len(panel)), key=lambda i: (-counts[i], i)):
        worker = min(range(workers), key=lambda w: (loads[w], w))
        groups[worker].append(index)
        loads[worker] += counts[index]
    return [sorted(group) for group in groups]


def input_partition(panel, workers, chunk_size=0):
    """Assign complete input banks, retaining every edit and arm per input.

    Zero keeps the published whole-case schedule. Positive sizes split only
    execution work; they do not create additional statistical model samples.
    """
    if type(chunk_size) is not int or chunk_size < 0:
        raise ValueError('input chunk size must be a nonnegative integer')
    if chunk_size == 0:
        return [{i: list(range(len(panel[i]['inputs']))) for i in group}
                for group in partition(panel, workers)]
    chunks = [(i, start, min(start+chunk_size, len(row['inputs'])))
              for i, row in enumerate(panel)
              for start in range(0, len(row['inputs']), chunk_size)]
    if type(workers) is not int or not 1 <= workers <= len(chunks):
        raise ValueError('worker count must be within input chunk count')
    groups, loads = [{} for _ in range(workers)], [0] * workers
    for i, start, end in sorted(chunks, key=lambda x: (
            -(x[2]-x[1])*len(panel[x[0]]['edits']), x[0], x[1])):
        worker = min(range(workers), key=lambda w: (loads[w], w))
        groups[worker].setdefault(i, []).extend(range(start, end))
        loads[worker] += (end-start)*len(panel[i]['edits'])
    return [{i: sorted(banks) for i, banks in sorted(group.items())} for group in groups]


def input_cartesian(panel, arms, selection):
    return {(panel[i]['stratum'], panel[i]['id'], arm, edit, inp)
            for i, banks in selection.items() for arm in arms
            for edit in range(len(panel[i]['edits'])) for inp in banks}


def input_contract(selection, chunk_size):
    return {} if chunk_size == 0 else {
        'input_chunk_size': chunk_size,
        'input_indices': {str(i): banks for i, banks in selection.items()}}


def check_input_contract(record, selection, chunk_size):
    expected = input_contract(selection, chunk_size)
    if chunk_size and (type(record.get('input_chunk_size')) is not int
                       or not isinstance(record.get('input_indices'), dict)
                       or any(not isinstance(banks, list) or any(type(i) is not int for i in banks)
                              for banks in record['input_indices'].values())):
        raise ValueError('invalid input chunk contract types')
    if {k: record[k] for k in ('input_chunk_size', 'input_indices') if k in record} != expected:
        raise ValueError('incompatible input chunk contract')


def validate_panel(manifest, parent_path=None, parent_sha256=None, diagnostic_sha256=None):
    """A stronger baseline may only use the unchanged eight new-model cases."""
    if diagnostic_sha256 is not None:
        # This opt-in is the separately frozen, exposed-model follow-up. Both
        # callers already verify actual manifest bytes against this digest.
        if (diagnostic_sha256 != DIAGNOSTIC_MANIFEST_SHA
                or parent_path is not None or parent_sha256 is not None
                or manifest.get('diagnostic') != 'clarified_unit_permutation_v1'
                or manifest['arms'] != ['base', 'point', 'intervention', 'qwen35', 'kimi']
                or Counter(r['stratum'] for r in manifest['panel']) != {
                    'diagnostic_original': 8, 'diagnostic_permuted': 8}
                or manifest['forward_cap'] != 20944):
            raise ValueError('exact frozen clarified/permuted diagnostic required')
        for original, permuted in zip(manifest['panel'][:8], manifest['panel'][8:]):
            if (original['condition'] != 'original' or permuted['condition'] != 'permuted'
                    or original['id'] != original['original_id']
                    or permuted['id'] != original['id']+'__perm'
                    or permuted['original_id'] != original['id']
                    or original['inputs'] != permuted['inputs']
                    or original['input_roles'] != permuted['input_roles']):
                raise ValueError('paired diagnostic case identity changed')
        return
    if parent_path is None:
        if (parent_sha256 is not None or 'parent_manifest_sha256' in manifest or 'diagnostic' in manifest
                or manifest['schema_version'] != 1
                or manifest['arms'] != ['base', 'point', 'intervention']
                or Counter(r['stratum'] for r in manifest['panel']) != {'new_model': 8, 'retention': 24}):
            raise ValueError('fixed 32-model, three-policy panel required')
        return
    if parent_sha256 is None or sha(parent_path) != parent_sha256:
        raise ValueError('strong baseline parent digest required')
    parent = json.loads(Path(parent_path).read_text())
    validate_panel(parent)
    expected = dict(parent)
    expected['panel'] = [row for row in parent['panel'] if row['stratum'] == 'new_model']
    expected['arms'] = ['strong']
    expected['forward_cap'] = 3 * sum(len(row['inputs']) * len(row['edits']) for row in expected['panel'])
    expected['parent_manifest_sha256'] = parent_sha256
    if manifest != expected:
        raise ValueError('strong baseline must preserve every original new-model world and source')


def cartesian(panel, arms, indices):
    return {(panel[i]['stratum'], panel[i]['id'], arm, edit, inp)
            for i in indices for arm in arms for edit in range(len(panel[i]['edits']))
            for inp in range(len(panel[i]['inputs']))}


def merge(args):
    """Read shard metadata/rows in place; tensor files stay in their shards."""
    if sha(args.manifest) != args.manifest_sha256 or args.out.exists():
        raise ValueError('frozen manifest and fresh output required')
    manifest = json.loads(args.manifest.read_text())
    if ('parent_manifest_sha256' in manifest or 'diagnostic' in manifest
            or getattr(args, 'strong_baseline_parent', None) is not None
            or getattr(args, 'clarified_unit_permutation', False)):
        validate_panel(manifest, getattr(args, 'strong_baseline_parent', None),
                       getattr(args, 'strong_baseline_parent_sha256', None),
                       args.manifest_sha256 if getattr(args, 'clarified_unit_permutation', False) else None)
    panel, arms = manifest['panel'], manifest['arms']
    chunk_size = getattr(args, 'input_chunk_size', 0)
    selections = input_partition(panel, args.workers, chunk_size)
    assignment = [list(selection) for selection in selections]
    global_cap = sum(len(r['inputs']) * len(r['edits']) * (2 + len(arms)) for r in panel)
    if manifest['forward_cap'] != global_cap:
        raise ValueError('global forward cap mismatch')
    if len(args.shard) != args.workers:
        raise ValueError('exactly one complete shard per worker required')
    rows, seen, receipts = [], set(), []
    actual_forwards = 0
    model_states = {}
    for directory in args.shard:
        contract = json.loads((directory / 'contract.json').read_text())
        done = json.loads((directory / 'complete.json').read_text())
        worker = contract['worker']
        if (type(worker) is not int or worker not in range(args.workers) or worker in seen
                or contract['workers'] != args.workers
                or contract['manifest_sha256'] != args.manifest_sha256
                or contract['responses_sha256'] != args.responses_sha256
                or contract['scorer_sha256'] != sha(__file__)
                or contract['case_indices'] != assignment[worker]):
            raise ValueError('duplicate or incompatible shard contract')
        seen.add(worker)
        selection = selections[worker]
        check_input_contract(contract, selection, chunk_size)
        check_input_contract(done, selection, chunk_size)
        if contract['global_forward_cap'] != global_cap:
            raise ValueError('shard global cap mismatch')
        indices = assignment[worker]
        cap = sum(len(banks) * len(panel[i]['edits']) * (2 + len(arms))
                  for i, banks in selection.items())
        shard_rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines() if line.strip()]
        summarize(shard_rows, input_cartesian(panel, arms, selection))
        if (contract['forward_cap'] != cap or done['forward_cap'] != cap
                or type(done['actual_forwards']) is not int
                or done['actual_forwards'] != 2 * (cap // (2 + len(arms))) + sum(bool(r['valid']) for r in shard_rows)
                or done['candidate_worlds'] != len(shard_rows) or done['models'] != len(indices)
                or done['arms'] != arms or done['case_indices'] != indices):
            raise ValueError('incomplete shard receipt')
        actual_forwards += done['actual_forwards']
        for row in shard_rows:
            # Verify each tensor once without copying it across shared storage.
            row['artifact_root'] = str(directory.resolve())
        for index in indices:
            original = directory / f'{index}.original.json'
            restored = directory / f'{index}.restored.json'
            if json.loads(original.read_text()) != json.loads(restored.read_text()):
                raise ValueError('shard model restoration failed')
            state = json.loads(original.read_text())
            if index in model_states and state != model_states[index]:
                raise ValueError('input chunks used different original model/basis states')
            model_states[index] = state
        files = {row['file']: row['sha256'] for row in shard_rows}
        if any(row['sha256'] != files[row['file']] for row in shard_rows):
            raise ValueError('conflicting tensor hashes')
        for name, digest in files.items():
            if sha(bound_path(directory, name)) != digest:
                raise ValueError('shard tensor changed')
        rows.extend(shard_rows)
        receipts.append({'worker': worker, 'artifact_root': str(directory.resolve()),
                         'contract_sha256': sha(directory / 'contract.json'),
                         'complete_sha256': sha(directory / 'complete.json'),
                         'results_sha256': sha(directory / 'results.jsonl')})
    summary = summarize(rows, cartesian(panel, arms, range(len(panel))))
    args.out.mkdir()
    write(args.out / 'summary.json', summary)
    write(args.out / 'receipt.json', {'manifest_sha256': args.manifest_sha256,
          'responses_sha256': args.responses_sha256, 'workers': args.workers,
          'partition': assignment, 'shards': sorted(receipts, key=lambda r: r['worker']),
          'candidate_worlds': len(rows), 'actual_forwards': actual_forwards,
          'forward_cap': global_cap, 'scorer_sha256': sha(__file__),
          **({'input_chunk_size': chunk_size,
              'input_assignments': [input_contract(s, chunk_size)['input_indices'] for s in selections]}
             if chunk_size else {})})
    with (args.out / 'results.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def run(args):
    if (sha(args.manifest) != args.manifest_sha256 or sha(args.responses) != args.responses_sha256
            or args.out.exists()):
        raise ValueError('frozen manifest and fresh output required')
    manifest = json.loads(args.manifest.read_text())
    validate_panel(manifest, getattr(args, 'strong_baseline_parent', None),
                   getattr(args, 'strong_baseline_parent_sha256', None),
                   args.manifest_sha256 if getattr(args, 'clarified_unit_permutation', False) else None)
    panel, arms = manifest['panel'], manifest['arms']
    chunk_size = getattr(args, 'input_chunk_size', 0)
    selections = input_partition(panel, args.workers, chunk_size)
    if type(args.worker) is not int or args.worker not in range(args.workers):
        raise ValueError('worker index out of range')
    selection = selections[args.worker]
    indices = list(selection)
    required_source = ('experiments/oracle/toyloop/circuit.py', 'experiments/oracle/toys/planted.py',
                       'method/oracle/native.py', 'method/oracle/native_units.py',
                       'method/oracle/goal_eval.py', 'method/oracle/carry.py',
                       'method/oracle/mech.py', 'method/vpd/vpd_model.py')
    required_support = tuple('experiments/oracle/toyloop/' + name + '.py' for name in
                             ('head_counterfactual', 'reward_interventions', 'intervention_teacher'))
    for root, field, required in ((args.source, 'source_sha256', required_source),
                                  (args.support, 'support_sha256', required_support)):
        if not set(required) <= set(manifest[field]):
            raise ValueError('complete executor source binding required')
        for path, expected in manifest[field].items():
            if sha(bound_path(root, path)) != expected:
                raise ValueError('executor source changed')
    response_binding = json.loads(args.responses.read_text())
    if response_binding['manifest_sha256'] != args.manifest_sha256:
        raise ValueError('responses belong to a different scoring panel')
    samples = {}
    for arm in arms:
        entry = response_binding['arms'][arm]
        path = bound_path(args.responses.parent, entry['path'])
        if sha(path) != entry['sha256']:
            raise ValueError('frozen response file changed')
        samples[arm] = bind_responses(panel, [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    # Imports occur only after frozen contracts have been checked.
    sys.path[:0] = [str(args.source / folder) for folder in
                   ('experiments/oracle/toyloop', 'experiments/oracle/toys', 'method/oracle', 'method/vpd')]
    import torch
    import circuit
    import planted
    import native
    import goal_eval
    sys.path.insert(0, str(args.support / 'experiments/oracle/toyloop'))
    from head_counterfactual import scaled_head, raw_kl_bits
    from reward_interventions import scaled_neuron, clear_intervention_caches
    from intervention_teacher import tensor_digest
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    args.out.mkdir()
    expected_keys = input_cartesian(panel, arms, selection)
    forwards = 0
    raw_rows = []
    global_cap = sum(len(row['inputs']) * len(row['edits']) * (2 + len(arms)) for row in panel)
    max_forwards = sum(len(banks) * len(panel[i]['edits']) * (2 + len(arms))
                       for i, banks in selection.items())
    if global_cap != manifest['forward_cap']:
        raise ValueError('forward cap does not cover frozen Cartesian panel')
    write(args.out / 'contract.json', {'manifest_sha256': args.manifest_sha256,
          'responses_sha256': sha(args.responses), 'forward_cap': max_forwards,
          'global_forward_cap': global_cap, 'worker': args.worker, 'workers': args.workers,
          'case_indices': indices, 'scorer_sha256': sha(__file__),
          **input_contract(selection, chunk_size)})
    for case_index in indices:
        item = panel[case_index]
        expected_format = 'legacy_mermaid_english' if item['stratum'] == 'retention' else 'graph_only'
        if item['answer_format'] != expected_format:
            raise ValueError('answer format must match frozen prompt stratum')
        path = bound_path(args.payload, item['weights'])
        if sha(path) != item['weights_sha256']:
            raise ValueError('model weights changed')
        blob = torch.load(path, map_location='cpu', weights_only=True)
        toy = SimpleNamespace(sd=blob['sd'], cfg=blob['cfg'], meta={})
        inputs, edits = item['inputs'], item['edits']
        bank_indices = selection[case_index]
        if (not inputs or len(inputs) != len(item['input_roles']) or not edits
                or edits[0] != {'kind': 'no_edit', 'operations': []}):
            raise ValueError('nonempty input panel and leading no-edit world required')
        nat = planted.native_of(toy.sd, toy.cfg)
        toy.meta['decomposition'] = {name: (nat.target.site(name).U, nat.target.site(name).V, nat.units[name]) for name in nat.names}
        def state():
            return {'model': {key: tensor_digest(value) for key, value in nat.target.state_dict().items()},
                    'basis': {name: {key: tensor_digest(getattr(nat.target.site(name), key))
                                    for key in ('W', 'U', 'V', 'delta_T')} for name in nat.names},
                    'units': {name: tensor_digest(nat.units[name]) for name in nat.names}}
        original = state()
        write(args.out / f'{case_index}.original.json', original)
        graphs = {}
        for arm in arms:
            sample = samples[arm][case_index]
            error = answer_error(sample, item['answer_format'])
            graphs[arm] = []
            sizes = []
            for index in bank_indices:
                ids = inputs[index]
                graph, reason = None, error
                if reason is None:
                    try:
                        parsed = circuit.parse(sample['completion'])
                        if not parsed.ok or parsed.bad:
                            raise ValueError('strict circuit parser rejected answer')
                        graph = circuit.unit_graph(parsed, toy, ids)
                    except ValueError as exc:
                        reason = str(exc)
                graphs[arm].append((graph, reason))
                sizes.append({'input_positions': len(ids), 'valid': graph is not None,
                              'declared_units': None if graph is None else sum(u != 'out' and not u.startswith('tok') for u in parsed.units),
                              'declared_token_annotations': None if graph is None else sum(u.startswith('tok') for u in parsed.units),
                              'declared_edges': None if graph is None else len(parsed.edges),
                              'executed_component_position_nodes': None if graph is None else graph.count(),
                              'executed_edges': None if graph is None else graph.edges(),
                              'full_component_position_nodes': len(ids) * sum(int(nat.C[name]) for name in nat.names)})
            write(args.out / f'{case_index}.{arm}.size.json', {'case': item['id'], 'arm': arm,
                  'stratum': item['stratum'], 'per_input': sizes,
                  **({'input_indices': bank_indices} if chunk_size else {}),
                  'scope': 'report complexity alongside fidelity; not a necessity or compute-speedup certificate'})
        full = [native.Graph({(name, pos, piece) for name in nat.names for pos in range(len(ids))
                              for piece in range(int(nat.C[name]))}, complete=True)
                for ids in (inputs[index] for index in bank_indices)]
        baselines = {}
        for edit_index, edit in enumerate(edits):
            kind, operations = edit['kind'], edit['operations']
            if kind not in ('no_edit', 'individual', 'joint') or len(operations) != {'no_edit': 0, 'individual': 1, 'joint': 2}[kind]:
                raise ValueError('invalid frozen edit kind')
            with contextlib.ExitStack() as stack:
                for unit_kind, layer, unit, scale in operations:
                    if unit_kind not in ('head', 'neuron') or scale not in (0., .5):
                        raise ValueError('invalid frozen intervention')
                    stack.enter_context((scaled_head if unit_kind == 'head' else scaled_neuron)(nat, layer, unit, scale))
                for local_index, index in enumerate(bank_indices):
                    ids = inputs[index]
                    clear_intervention_caches(nat)
                    p = nat.reference([ids], [len(ids)-1]).detach().clone()
                    integrity = goal_eval.GoalEval(nat).graph_ref(ids, [len(ids)-1], full[local_index], live=True, norms={})
                    forwards += 2
                    integrity_kl = raw_kl_bits(p, integrity)
                    if integrity_kl > .001:
                        raise RuntimeError('full architecture integrity failure')
                    if edit_index == 0:
                        baselines[index] = p.clone()
                    tensors = {'p': p.contiguous(), 'full': integrity.contiguous(), 'p0': baselines[index].contiguous()}
                    world_rows = []
                    for arm in arms:
                        graph, error = graphs[arm][local_index]
                        clear_intervention_caches(nat)
                        q = None if graph is None else goal_eval.GoalEval(nat).graph_ref(ids, [len(ids)-1], graph, live=False)
                        forwards += int(q is not None)
                        raw = None if q is None else raw_kl_bits(p, q)
                        if q is not None:
                            tensors[arm] = q.contiguous()
                        world_rows.append({'stratum': item['stratum'], 'case': item['id'], 'arm': arm,
                            'edit_index': edit_index, 'input': index, 'input_role': item['input_roles'][index],
                            'kind': kind, 'operations': operations, 'valid': q is not None, 'invalid_reason': error,
                            'raw_kl_bits': raw, 'kl_bits': None if raw is None else max(0., raw),
                            'top': None if q is None else int(q[0,-1].argmax()), 'reference_top': int(p[0,-1].argmax()),
                            'nochange_top': int(baselines[index][0,-1].argmax()), 'integrity_raw_kl_bits': integrity_kl})
                    filename = f'{case_index}_{edit_index}_{index}.safetensors'
                    save_file(tensors, str(args.out / filename))
                    with (args.out / 'results.jsonl').open('a') as stream:
                        for row in world_rows:
                            row.update(file=filename, sha256=sha(args.out / filename))
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                            raw_rows.append(row)
                        stream.flush(); os.fsync(stream.fileno())
            clear_intervention_caches(nat)
            if state() != original:
                raise RuntimeError('intervention failed exact model/basis restoration')
        write(args.out / f'{case_index}.restored.json', state())
        print(json.dumps({'case': item['id'], 'forwards_so_far': forwards}), flush=True)
        del nat
    if forwards > max_forwards:
        raise RuntimeError('forward cap exceeded')
    summary = summarize(raw_rows, expected_keys)
    write(args.out / 'summary.json', summary)
    write(args.out / 'complete.json', {'actual_forwards': forwards, 'forward_cap': max_forwards,
          'candidate_worlds': len(raw_rows), 'models': len(indices), 'arms': arms,
          'case_indices': indices, **input_contract(selection, chunk_size)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--input-chunk-size', type=int, default=0,
                        help='opt-in contiguous input-bank chunks; zero preserves whole-case scheduling')
    parser.add_argument('--shard', type=Path, action='append', default=[])
    parser.add_argument('--strong-baseline-parent', type=Path)
    parser.add_argument('--strong-baseline-parent-sha256')
    parser.add_argument('--clarified-unit-permutation', action='store_true')
    for name in ('manifest', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--responses-sha256', required=True)
    for name in ('payload', 'source', 'support', 'responses'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if any(getattr(args, name) is None for name in ('payload', 'source', 'support', 'responses')):
            parser.error('scoring requires payload/source/support/responses')
        run(args)
