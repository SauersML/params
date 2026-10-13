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


def answer_error(row):
    if row is None:
        return 'missing_first_response'
    if row.get('parse_error'):
        return 'saved_parse_error'
    if str(row['finish']).lower() not in ('stop', 'stopreason.stop'):
        return 'unfinished_first_response'
    text = row.get('completion')
    if not isinstance(text, str):
        return 'missing_completion'
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


def run(args):
    if (sha(args.manifest) != args.manifest_sha256 or sha(args.responses) != args.responses_sha256
            or args.out.exists()):
        raise ValueError('frozen manifest and fresh output required')
    manifest = json.loads(args.manifest.read_text())
    panel, arms = manifest['panel'], manifest['arms']
    if (manifest['schema_version'] != 1 or arms != ['base', 'point', 'intervention']
            or Counter(r['stratum'] for r in panel) != {'new_model': 8, 'retention': 24}):
        raise ValueError('fixed 32-model, three-policy panel required')
    required_source = ('experiments/oracle/toyloop/circuit.py', 'experiments/oracle/toys/planted.py',
                       'method/oracle/native.py', 'method/oracle/native_units.py',
                       'method/oracle/goal_eval.py', 'method/vpd/vpd_model.py')
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
    expected_keys = {(row['stratum'], row['id'], arm, edit_index, index)
                     for row in panel for arm in arms for edit_index in range(len(row['edits']))
                     for index in range(len(row['inputs']))}
    forwards = 0
    raw_rows = []
    max_forwards = sum(len(row['inputs']) * len(row['edits']) * (2 + len(arms)) for row in panel)
    if max_forwards != manifest['forward_cap']:
        raise ValueError('forward cap does not cover frozen Cartesian panel')
    write(args.out / 'contract.json', {'manifest_sha256': args.manifest_sha256,
          'responses_sha256': sha(args.responses), 'forward_cap': max_forwards,
          'scorer_sha256': sha(__file__)})
    for case_index, item in enumerate(panel):
        path = bound_path(args.payload, item['weights'])
        if sha(path) != item['weights_sha256']:
            raise ValueError('model weights changed')
        blob = torch.load(path, map_location='cpu', weights_only=True)
        toy = SimpleNamespace(sd=blob['sd'], cfg=blob['cfg'], meta={})
        inputs, edits = item['inputs'], item['edits']
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
            error = answer_error(sample)
            graphs[arm] = []
            for ids in inputs:
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
        full = [native.Graph({(name, pos, piece) for name in nat.names for pos in range(len(ids))
                              for piece in range(int(nat.C[name]))}, complete=True) for ids in inputs]
        baselines = []
        for edit_index, edit in enumerate(edits):
            kind, operations = edit['kind'], edit['operations']
            if kind not in ('no_edit', 'individual', 'joint') or len(operations) != {'no_edit': 0, 'individual': 1, 'joint': 2}[kind]:
                raise ValueError('invalid frozen edit kind')
            with contextlib.ExitStack() as stack:
                for unit_kind, layer, unit, scale in operations:
                    if unit_kind not in ('head', 'neuron') or scale not in (0., .5):
                        raise ValueError('invalid frozen intervention')
                    stack.enter_context((scaled_head if unit_kind == 'head' else scaled_neuron)(nat, layer, unit, scale))
                for index, ids in enumerate(inputs):
                    clear_intervention_caches(nat)
                    p = nat.reference([ids], [len(ids)-1]).detach().clone()
                    integrity = goal_eval.GoalEval(nat).graph_ref(ids, [len(ids)-1], full[index], live=True, norms={})
                    forwards += 2
                    integrity_kl = raw_kl_bits(p, integrity)
                    if integrity_kl > .001:
                        raise RuntimeError('full architecture integrity failure')
                    if edit_index == 0:
                        baselines.append(p.clone())
                    tensors = {'p': p.contiguous(), 'full': integrity.contiguous(), 'p0': baselines[index].contiguous()}
                    world_rows = []
                    for arm in arms:
                        graph, error = graphs[arm][index]
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
          'candidate_worlds': len(raw_rows), 'models': len(panel), 'arms': arms})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'payload', 'source', 'support', 'responses', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--responses-sha256', required=True)
    run(parser.parse_args())
