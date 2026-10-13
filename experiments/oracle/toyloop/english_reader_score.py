"""Offline strict content scoring; missing/invalid/truncated responses are wrong."""
import argparse
import json
from pathlib import Path
import re


def parse(row):
    if row is None or row.get('parse_error'):
        return None
    finish = str(row.get('finish', '')).lower()
    if finish != 'stop' and not finish.endswith('.stop'):
        return None
    try:
        value = json.loads(row['completion'])
        if set(value) != {'top', 'changed'} or not isinstance(value['changed'], bool):
            return None
        match = re.fullmatch(r't(\d+)', value['top'])
        if match is None or int(match[1]) not in range(32):
            return None
        return int(match[1]), value['changed']
    except (KeyError, ValueError, TypeError):
        return None


def score(labels, samples, conditions=None):
    conditions = tuple(conditions or ('none', 'generated', 'shuffled', 'gt_positive'))
    if len(conditions) != len(set(conditions)) or not {r['condition'] for r in labels} <= set(conditions):
        raise ValueError('declare every unique reader condition before scoring')
    assert len({r['id'] for r in samples}) == len(samples), 'Duplicate first-sample output'
    expected = {r['id'] for r in labels}
    assert all(r['id'] in expected for r in samples), 'Unexpected response ID'
    by_id = {r['id']: r for r in samples}
    detail = []
    for truth in labels:
        predicted = parse(by_id.get(truth['id']))
        detail.append({**{k: truth[k] for k in ('id', 'case', 'condition', 'task', 'input', 'layer', 'head', 'scale', 'active', 'changed')},
                       'valid': predicted is not None,
                       'top_right': predicted is not None and predicted[0] == truth['top'],
                       'change_right': predicted is not None and predicted[1] == truth['changed'],
                       'no_change_top_right': truth['original_top'] == truth['top']})
    groups = {}
    for condition in conditions:
        selected = [r for r in detail if r['condition'] == condition]
        subsets = {'all': selected, 'delete': [r for r in selected if r['scale'] == 0.],
                   'half': [r for r in selected if r['scale'] == .5],
                   'active': [r for r in selected if r['active']],
                   'inactive': [r for r in selected if not r['active']],
                   'top_changed': [r for r in selected if r['changed']],
                   'top_unchanged': [r for r in selected if not r['changed']]}
        for layer in range(2):
            for head in range(4):
                subsets[f'L{layer}H{head}'] = [r for r in selected if (r['layer'], r['head']) == (layer, head)]
        for task in sorted({r['task'] for r in selected}):
            subsets[task] = [r for r in selected if r['task'] == task]
        for key, value in subsets.items():
            groups[condition + '/' + key] = {'n': len(value), 'valid': sum(r['valid'] for r in value),
                                            'top_correct': sum(r['top_right'] for r in value),
                                            'change_correct': sum(r['change_right'] for r in value),
                                            'no_change_top_correct': sum(r['no_change_top_right'] for r in value)}
    return {'scope': 'Four reused-development toys; 128 correlated edits, only eight input clusters; no universal English claim',
            'groups': groups, 'detail': detail,
            'interpretation': 'GT positive-control improvement over no-English required; shuffled cross-task prose is a weak control; no model scoring or graph repair'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', type=Path)
    ap.add_argument('samples', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--condition', action='append', help='Explicit condition name; repeat for a new frozen diagnostic. Historical four-condition defaults unchanged.')
    a = ap.parse_args()
    load = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    assert not a.output.exists(), 'Do not overwrite an existing result'
    a.output.write_text(json.dumps(score(load(a.labels), load(a.samples), a.condition), indent=2) + '\n')
