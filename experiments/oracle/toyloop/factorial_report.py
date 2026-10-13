"""Offline four-cell point report; invalid outcomes stay in primary denominators.

Empty gold-edge cases are separated because perfect edge F1 there can conceal
missing output units. KL is conditional on physical validity, with coverage.
A sufficient complete graph is not evidence of sparse route identification.
"""
from __future__ import annotations
import json
from pathlib import Path

CELLS = ('induction_m1', 'induction_m8', 'multihop_m1', 'multihop_m8')


def summarize(rows):
    if not rows:
        raise ValueError('empty cohort')
    average = lambda values: sum(values) / len(rows)
    good = [r for r in rows if r['valid']]
    return {'n': len(rows), 'valid': len(good),
            'reward': average([r['stated_reward'] if r['valid'] else -1. for r in rows]),
            'edge_f1': average([r['component_edge_f1'] if r['valid'] else 0. for r in rows]),
            'edge_precision': average([r['component_edge_precision'] if r['valid'] else 0. for r in rows]),
            'edge_recall': average([r['component_edge_recall'] if r['valid'] else 0. for r in rows]),
            'correct_edges': sum(len(r['correct_executed_component_edges']) for r in good),
            'extra_edges': sum(len(set(map(tuple,r['valid_executed_component_edges'])) - set(map(tuple,r['gold_component_edges']))) for r in good),
            'top1_invalid_zero': average([sum(t['top']==t['reference_top'] for t in r['stated']['per_text'])/len(r['stated']['per_text']) if r['valid'] else 0. for r in rows]),
            'kl_valid_only': sum(r['stated']['kl_bits'] for r in good)/len(good) if good else None,
            'complete_reward': average([r['same_units_complete_reward'] if r['valid'] else -1. for r in rows])}


def contrasts(cells, metric):
    k1,k8,v1,v8 = (cells[c][metric] for c in CELLS)
    return {'balance_effect_K': k8-k1, 'balance_effect_V': v8-v1,
            'support_effect_m1': v1-k1, 'support_effect_m8': v8-k8,
            'balance_main_effect': ((k8-k1)+(v8-v1))/2,
            'support_main_effect': ((v1-k1)+(v8-k8))/2,
            'interaction': (v8-v1)-(k8-k1)}


def report(point_root, base_path, previous_sft_path):
    read = lambda path: [json.loads(x) for x in Path(path).read_text().splitlines()]
    data = {cell:read(Path(point_root)/f'{cell}.jsonl') for cell in CELLS}
    data['BASE'] = read(base_path)
    data['previous_plain226_SFT_descriptive'] = read(previous_sft_path)
    ids = [r['id'] for r in data['BASE']]
    if len(ids)!=24 or len(set(ids))!=24 or any([r['id'] for r in rows]!=ids for rows in data.values()):
        raise ValueError('requires identical ordered frozen24 cases')
    cohorts = {'all24':lambda r:True, 'relation8':lambda r:bool(r['gold_component_edges']),
               'zero_gold_edges16':lambda r:not r['gold_component_edges'],
               'V_multihop4_family_exposed':lambda r:r['family']=='multihop'}
    groups = {group:{cell:summarize([r for r in rows if select(r)]) for cell,rows in data.items()} for group,select in cohorts.items()}
    paired = [ (b,v) for b,v in zip(data['BASE'],data['multihop_m8']) if b['valid'] and v['valid'] ]
    return {'scope':'One-seed shared-development pilot; V arm exposes multihop family. Previous226-row58-step SFT is descriptive, not a factorial control. Full English retained but unverified.',
            'groups':groups,
            'contrasts':{group:{metric:contrasts(groups[group],metric) for metric in ('reward','edge_f1','top1_invalid_zero')} for group in cohorts},
            'V8_vs_BASE_common_valid':{'n':len(paired),'reward_delta_mean':sum(v['stated_reward']-b['stated_reward'] for b,v in paired)/len(paired),
                                      'regressed_cases':[b['id'] for b,v in paired if v['stated_reward']<b['stated_reward']-1e-5]},
            'decision':'Freeze V+m8 checkpoint for fresh predeclared transfer and route-specificity evaluation; no more training by default. All4 V routes repaired, but only one possible bridge after selecting two heads; gate distractor edges remain. Complete-same-unit sufficiency does not establish minimal topology.'}


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('points','base','previous-sft','out'): p.add_argument('--'+name,required=True)
    a=p.parse_args(); result=report(a.points,a.base,a.previous_sft)
    with Path(a.out).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
