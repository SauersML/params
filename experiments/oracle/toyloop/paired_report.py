"""Offline, fail-closed comparison of completed unit/wiring arms; no models or generation.

  python paired_report.py out/unit out/wiring --out report.json
The deterministic toy panel is shared development. Bootstrap intervals resample
cases, not mechanism families or within-case interventions. English, token roles,
necessity and real-model/VPD faithfulness are outside this report's scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def files(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rows(path, panel):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == 24 and len({r['id'] for r in rows}) == 24, f'incomplete or duplicate evaluation: {path}')
    by_id = {r['id']: r for r in rows}
    require(set(by_id) == set(panel), f'evaluation panel mismatch: {path}')
    for ident, row in by_id.items():
        require((row['split'], row['family']) == (panel[ident]['split'], panel[ident]['family']), 'case split/family changed')
        require(isinstance(row['valid'], bool), 'evaluation lacks explicit validity')
        gold = panel[ident].get('gt_units')
        row['_gold_components'] = ([re.sub(r'^(L\d+)\.([HN]\d+)$', r'\1\2', name) for name in gold]
                                   if gold is not None else None)
        if not row['valid']:
            require(row['stated_reward'] == row['same_units_complete_reward'] == -1, 'invalid evaluation penalty changed')
        else:
            for label in ('stated', 'same_units_complete'):
                scores = row[label]['per_text']
                require(bool(scores), 'valid evaluation has no executed inputs')
                require(all(math.isfinite(t['kl_bits']) and t['kl_bits'] >= 0 and 0 <= t['suff'] <= 1 for t in scores), 'invalid execution metrics')
                expected = (statistics.mean(t['suff'] for t in scores) + min(t['suff'] for t in scores)) / 2
                require(math.isclose(row[label + '_reward'], expected, rel_tol=1e-9, abs_tol=1e-9), 'saved reward disagrees with executed inputs')
    return by_id


def load_arm(path, name):
    contract = read(path / 'contract.json')
    lineage = read(path / 'sft_lineage.json')
    require(lineage['schema'] == 2 and contract['sft_lineage_sha256'] == digest(path / 'sft_lineage.json'), 'lineage identity mismatch')
    for relative, expected in lineage['imported_files'].items():
        file = path / relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'unsafe lineage path')
        require(file.is_file() and digest(file) == expected, f'changed immutable parent file: {relative}')
    require(digest(path / 'parent_evidence/contract.json') == lineage['parent_contract_sha256'], 'parent source contract mismatch')
    data = read(path / 'data.json')
    entries = data['evaluation_panel']
    panel = {r['id']: r for r in entries}
    require(len(entries) == len(panel) == 24, 'fixed panel must contain exactly 24 distinct cases')
    require(data['files'] == files(path / 'zoo'), 'frozen toy data changed')
    sft = read(path / 'sft.done.json')
    sft_files = files(path / 'sft')
    require(sft['files'] == sft_files and any(n.endswith('.safetensors') for n in sft_files), 'SFT checkpoint changed or missing')
    mandatory = {'sft.done.json', 'data.json', 'train_inputs.jsonl', 'parent_evidence/contract.json'}
    mandatory |= {'sft/' + n for n in sft_files} | {'zoo/' + n for n in data['files']}
    require(mandatory <= lineage['imported_files'].keys(), 'lineage lacks immutable checkpoint/data binding')
    done = read(path / (name + '.done.json'))
    require(done['files'] == files(path / name) and any(n.endswith('.safetensors') for n in done['files'])
            and done['start_hashes'] == sft_files, 'RL checkpoint or frozen start mismatch')
    require(done['reward'] == read(path / (name + '.started.json'))['reward'] == name, 'arm reward mismatch')
    log = [json.loads(line) for line in (path / name / 'log.jsonl').read_text().splitlines() if line.strip()]
    require([r['step'] for r in log] == list(range(contract['rl_steps_per_arm'])), 'partial or reordered RL training log')
    evaluated = {label: load_rows(path / (label + '_eval.jsonl'), panel) for label in ('gt', 'empty', 'sft', name)}
    for label in ('sft', name):
        expected = digest(path / (label + '.done.json'))
        require(all(r['policy_checkpoint_manifest_sha256'] == expected for r in evaluated[label].values()), 'evaluation checkpoint mismatch')
    for ident in panel:
        gt = evaluated['gt'][ident]
        require(gt['valid'], 'invalid ground-truth execution')
        require(max(t['kl_bits'] for t in gt['stated']['per_text']) <= contract['teacher_panel_max_kl_bits'], 'ground-truth integrity failed')
        for label, rows in evaluated.items():
            row = rows[ident]
            require(row['prompt_sha256'] == gt['prompt_sha256'], 'policy/baseline prompt changed')
            if row['valid']:
                require(row['input_bank_sha256'] == gt['input_bank_sha256'], 'executed input bank changed')
                require(len(row['stated']['per_text']) == len(gt['stated']['per_text']), 'executed input count changed')
    return {'path': path, 'name': name, 'contract': contract, 'lineage': lineage, 'data': data,
            'sft_files': sft_files, 'rows': evaluated, 'log': log}


def measures(row):
    """Invalid reward is -1; invalid top/F1 is zero; KL remains unmeasured, never fabricated."""
    components = {name for name in row['units'] if re.fullmatch(r'L\d+[HN]\d+', name)}
    gold = row.get('_gold_components')
    component_f1 = (2 * len(components & set(gold)) / max(len(components) + len(set(gold)), 1)
                    if gold is not None else None)
    result = {'component_count': len(components),
              'intercomponent_edge_count': sum(a in components and b in components for a, b in row['edges']),
              'component_f1': component_f1 if row['valid'] else 0. if gold is not None else None,
              'graph_node_f1': row['unit_f1'] if row['valid'] else 0.,
              'valid': int(row['valid']), 'stated_reward': row['stated_reward'],
              'complete_reward': row['same_units_complete_reward'],
              'wiring_gap': row['same_units_complete_reward'] - row['stated_reward'],
              'edge_f1': row['edge_f1'] if row['valid'] else 0., 'unit_f1': row['unit_f1'] if row['valid'] else 0.,
              'raw_parser_edge_f1': row['edge_f1'], 'raw_parser_unit_f1': row['unit_f1'],
              'unit_count': row['unit_count'], 'edge_count': row['edge_count'], 'top1': 0., 'kl_bits': None}
    if row['valid']:
        scores = row['stated']['per_text']
        result['top1'] = statistics.mean(t['top'] == t['reference_top'] for t in scores)
        result['kl_bits'] = statistics.mean(t['kl_bits'] for t in scores)
    return result


def summary(rows):
    ms = [measures(r) for r in rows]
    result = {'n': len(rows), 'valid': sum(r['valid'] for r in rows)}
    for key in ms[0]:
        values = [r[key] for r in ms if r[key] is not None]
        result[key + '_mean'] = statistics.mean(values) if values else None
        if key == 'component_f1':
            result['component_f1_coverage_n'] = len(values)
        if key == 'kl_bits':
            result['kl_valid_n'] = len(values)
            result['kl_worst_valid'] = max(values) if values else None
        if key == 'stated_reward':
            ordered = sorted(values)
            result['stated_reward_worst_all'] = min(values)
            result['stated_reward_p10_all'] = ordered[int(.1 * (len(ordered) - 1))]
    return result


def delta(values, resamples, seed):
    result = {'n': len(values), 'mean': statistics.mean(values) if values else None,
              'wins': sum(v > 0 for v in values), 'ties': sum(v == 0 for v in values), 'losses': sum(v < 0 for v in values)}
    if values and resamples:
        rng = random.Random(seed)
        draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples))
        result['exploratory_case_bootstrap95'] = [draws[int(.025 * (resamples - 1))], draws[int(.975 * (resamples - 1))]]
    return result


def compare(unit_path, wiring_path, resamples=10000, seed=20261013):
    unit = load_arm(Path(unit_path), 'unit_suff')
    wiring = load_arm(Path(wiring_path), 'wiring_suff')
    common = lambda c: {k: v for k, v in c.items() if k != 'sft_lineage_sha256'}
    require(common(unit['contract']) == common(wiring['contract']), 'source/package/config/platform contracts differ')
    require(unit['lineage']['parent_contract_sha256'] == wiring['lineage']['parent_contract_sha256'], 'frozen parents differ')
    require(unit['lineage']['imported_files'] == wiring['lineage']['imported_files'], 'immutable SFT imports differ')
    require(unit['sft_files'] == wiring['sft_files'] and unit['data'] == wiring['data'], 'frozen SFT/data/panel mismatch')
    require(digest(unit['path'] / 'train_inputs.jsonl') == digest(wiring['path'] / 'train_inputs.jsonl'), 'training prompts/targets differ')
    panel = unit['data']['evaluation_panel']
    groups = {'all': sorted(r['id'] for r in panel)}
    groups.update({split: sorted(r['id'] for r in panel if r['split'] == split) for split in ('train', 'test_instance', 'test_family')})
    require(all(len(groups[s]) == 8 for s in ('train', 'test_instance', 'test_family')), 'fixed split coverage changed')
    cases = []
    for ident in groups['all']:
        u, w = unit['rows']['unit_suff'][ident], wiring['rows']['wiring_suff'][ident]
        require(u['prompt_sha256'] == w['prompt_sha256'], 'paired prompts differ')
        gu, gw = unit['rows']['gt'][ident], wiring['rows']['gt'][ident]
        require(gu['input_bank_sha256'] == gw['input_bank_sha256']
                and len(gu['stated']['per_text']) == len(gw['stated']['per_text']), 'paired input banks differ')
        um, wm = measures(u), measures(w)
        cases.append({'id': ident, 'split': u['split'], 'family': u['family'], 'unit': um, 'wiring': wm,
                      'wiring_minus_unit': {key: wm[key] - um[key] if wm[key] is not None and um[key] is not None else None for key in um}})
    by_id = {r['id']: r for r in cases}
    report = {'scope': 'shared-development planted toy point-wiring prediction; English, token annotations, necessity and real VPD fidelity unverified',
              'metric_naming': 'Legacy unit_count/unit_f1 are graph-node metrics including tok/out. component_count/component_f1 and intercomponent_edge_count filter head/neuron IDs only; component gold comes from frozen panel gt_units.',
              'caps': 'matched scheduled updates/sample/context/generation caps, not equal actual compute or sampled tokens',
              'bootstrap_scope': 'exploratory paired case resampling; fixed chosen panel, not fresh heldout or family-level uncertainty',
              'seed': seed, 'resamples': resamples, 'cases': cases, 'by_split': {}, 'baseline_differences': {}, 'training': {}, 'provenance': {}}
    for group, ids in groups.items():
        report['by_split'][group] = {'unit': summary([unit['rows']['unit_suff'][i] for i in ids]),
                                     'wiring': summary([wiring['rows']['wiring_suff'][i] for i in ids]), 'paired_wiring_minus_unit': {}}
        for arm, tag in ((unit, 'unit'), (wiring, 'wiring')):
            improvement = {}
            for key in ('stated_reward', 'complete_reward', 'wiring_gap', 'edge_f1', 'unit_f1', 'component_f1', 'component_count', 'intercomponent_edge_count', 'top1', 'kl_bits'):
                pairs = [(measures(arm['rows'][arm['name']][i]), measures(arm['rows']['sft'][i])) for i in ids]
                changes = [after[key] - before[key] for after, before in pairs
                           if after[key] is not None and before[key] is not None]
                improvement[key] = delta(changes, resamples, seed)
            report['by_split'][group][tag + '_minus_own_sft'] = improvement
        for key in ('stated_reward', 'complete_reward', 'wiring_gap', 'edge_f1', 'unit_f1', 'component_f1', 'component_count', 'intercomponent_edge_count', 'top1', 'kl_bits'):
            values = [by_id[i]['wiring_minus_unit'][key] for i in ids if by_id[i]['wiring_minus_unit'][key] is not None]
            report['by_split'][group]['paired_wiring_minus_unit'][key] = delta(values, resamples, seed)
    for label in ('gt', 'empty', 'sft'):
        differences = []
        for ident in groups['all']:
            u, w = unit['rows'][label][ident], wiring['rows'][label][ident]
            um, wm = measures(u), measures(w)
            differences.append({'id': ident, 'completion_equal': u['completion'] == w['completion'],
                                'unit': um, 'wiring': wm,
                                'wiring_minus_unit': {k: wm[k] - um[k] if wm[k] is not None and um[k] is not None else None for k in um}})
        report['baseline_differences'][label] = {'unit': summary(list(unit['rows'][label].values())),
                                                'wiring': summary(list(wiring['rows'][label].values())), 'cases': differences,
                                                'changed_completions': sum(not r['completion_equal'] for r in differences)}
    for arm in (unit, wiring):
        log = arm['log']
        report['training'][arm['name']] = {'scheduled_steps': len(log), 'last_logged_elapsed_seconds': log[-1]['sec'],
            'nonzero_advantage_update_lower_bound': sum(r['reward_max'] > r['reward'] for r in log),
            'exact_optimizer_updates': None, 'note': 'Rounded reward logs certify only a lower bound; equality can conceal tiny variance. Sampled token counts/IDs were not logged.'}
        report['provenance'][arm['name']] = {name: digest(arm['path'] / name) for name in ('contract.json', 'sft_lineage.json', 'data.json', 'train_inputs.jsonl', 'sft.done.json', arm['name'] + '.done.json', arm['name'] + '.started.json', 'gt_eval.jsonl', 'empty_eval.jsonl', 'sft_eval.jsonl', arm['name'] + '_eval.jsonl')}
    report['analysis_source_sha256'] = digest(Path(__file__))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unit', type=Path)
    parser.add_argument('wiring', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    report = compare(args.unit, args.wiring, args.bootstrap)
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['by_split']['all'], indent=2))


if __name__ == '__main__':
    main()
