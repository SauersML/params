"""Independent saved-array audit; no native, Torch, model loads, or forwards."""
import hashlib
import json
import math
from pathlib import Path
import argparse
import numpy as np
from safetensors.numpy import load_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def kl(p, q):
    assert p.shape == q.shape == (1, 1, 32)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    p, q = p.astype(np.float64), q.astype(np.float64)
    for value in (p, q):
        peak = value.max(axis=-1, keepdims=True)
        assert np.abs(peak + np.log(np.exp(value - peak).sum(axis=-1, keepdims=True))).max() <= 1e-5
    return float((np.exp(p) * (p - q)).sum()) / math.log(2)


def agree(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-11), (a, b)


def audit(root, folder_name='factorial_head_counterfactual', trial_name='trial_factorial'):
    root = Path(root)
    folder = root / folder_name
    contract = json.loads((folder / 'contract.json').read_text())
    arms = ['gt', *contract.get('candidates', {'base': {}, 'trained': {}})]
    assert 'base' in arms
    assert contract['source'] == 'cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4'
    assert digest(folder / 'head_counterfactual.py') == contract['driver_sha256']
    assert digest(root / 'manifest.json') == contract['inputs_manifest_sha256']
    assert digest(root / trial_name / 'payload_contract.json') == contract['trial_payload_contract_sha256']
    payload = json.loads((root / trial_name / 'payload_contract.json').read_text())
    for filename, expected in payload['files'].items():
        assert Path(filename).name == filename and digest(root / trial_name / filename) == expected
    for arm, binding in contract.get('candidates', {}).items():
        assert digest(root / trial_name / binding['file']) == binding['sha256']
    bindings = rows(folder / 'distribution_manifest.jsonl')
    baseline, draws, restores = (rows(folder / filename) for filename in ('baseline.jsonl', 'draws.jsonl', 'restore.jsonl'))
    assert len(baseline) == 112 and len(draws) == 1792 and len(restores) == 64
    assert len(bindings) == len({r['file'] for r in bindings}) == 1904
    assert {r['file'] for r in bindings} == {p.name for p in folder.glob('*.safetensors')}
    by_file = {r['file']: r for r in bindings}
    assert all(r['basis_identical'] and r['dense_identical'] for r in restores)
    for binding in bindings:
        path = folder / binding['file']
        assert path.name == binding['file'] and digest(path) == binding['sha256']
        arrays = load_file(str(path))
        assert sorted(arrays) == binding['keys']
        assert all(a.shape == (1, 1, 32) and np.isfinite(a).all() for a in arrays.values())
    for row in baseline:
        assert row['distributions'] == by_file[row['distributions']['file']]
        arrays = load_file(str(folder / row['distributions']['file']))
        agree(kl(arrays['p0'], arrays['complete']), row['full_integrity_raw_kl_bits'])
        for arm in arms:
            if arm in row.get('candidate_invalid', {}):
                assert 'q0_' + arm not in arrays and arm not in row['candidate_kl_bits']
            else:
                agree(max(0., kl(arrays['p0'], arrays['q0_' + arm])), row['candidate_kl_bits'][arm])
    expected = {(f'multihop_{i:03d}', layer, head, scale, index) for i in range(4)
                for layer in range(2) for head in range(4) for scale in (0., .5) for index in range(28)}
    actual = {(r['task'], r['edit']['layer'], r['edit']['head'], r['edit']['scale'], r['input']) for r in draws}
    assert actual == expected and len(actual) == len(draws)
    for row in draws:
        assert row['distributions'] == by_file[row['distributions']['file']]
        assert set(row['candidates']) == set(arms)
        arrays = load_file(str(folder / row['distributions']['file']))
        agree(kl(arrays['pe'], arrays['complete']), row['full_integrity_raw_kl_bits'])
        assert row['full_integrity_raw_kl_bits'] <= contract['integrity_max_bits']
        agree(max(0., kl(arrays['pe'], arrays['p0'])), row['model_damage_bits'])
        assert row['active_damage'] == (row['model_damage_bits'] >= contract['damage_min_bits'])
        assert int(arrays['pe'][0, -1].argmax()) == row['reference_top']
        assert int(arrays['p0'][0, -1].argmax()) == row['original_top']
        for arm in arms:
            value = row['candidates'][arm]
            if not value.get('valid', True):
                assert value['invalid_reason'] and all(value[k] is None for k in ('direct_kl_bits', 'direct_raw_kl_bits', 'conditional_delta_kl_bits', 'top'))
                assert all(prefix + arm not in arrays for prefix in ('q0_', 'qe_', 'moved_'))
            else:
                agree(kl(arrays['pe'], arrays['qe_' + arm]), value['direct_raw_kl_bits'])
                agree(max(0., kl(arrays['pe'], arrays['qe_' + arm])), value['direct_kl_bits'])
                agree(max(0., kl(arrays['pe'], arrays['moved_' + arm])), value['conditional_delta_kl_bits'])
                assert int(arrays['qe_' + arm][0, -1].argmax()) == value['top']


    def aggregate(selected):
        n = len(selected)
        result = {'n': n, 'no_change_top_correct': sum(r['original_top'] == r['reference_top'] for r in selected),
                  'mean_no_change_kl_bits': sum(r['model_damage_bits'] for r in selected) / n if n else None,
                  'arms': {}}
        for arm in arms:
            valid = [(r, r['candidates'][arm]) for r in selected if r['candidates'][arm].get('valid', True)]
            result['arms'][arm] = {'n': n, 'valid': len(valid), 'undefined_kl': n - len(valid),
                                  'mean_direct_kl_bits': sum(v['direct_kl_bits'] for r, v in valid) / len(valid) if valid else None,
                                  'worst_direct_kl_bits': max((v['direct_kl_bits'] for r, v in valid), default=None),
                                  'top_correct': sum(v['top'] == r['reference_top'] for r, v in valid),
                                  'beats_no_change': sum(v['direct_kl_bits'] < r['model_damage_bits'] for r, v in valid)}
        return result


    groups = {}
    for group in ('all', 'active', 'inactive'):
        selected = [r for r in draws if group == 'all' or r['active_damage'] == (group == 'active')]
        groups[group] = selected
        for scale in (0., .5):
            groups[group + '/scale_' + str(scale)] = [r for r in selected if r['edit']['scale'] == scale]
        for task in contract['cases']:
            groups[task + '/' + group] = [r for r in selected if r['task'] == task]
            for scale in (0., .5):
                groups[task + '/' + group + '/scale_' + str(scale)] = [r for r in selected if r['task'] == task and r['edit']['scale'] == scale]
    report = {'verification': {'files': 1904, 'baselines': 112, 'draws': 1792, 'restores': 64, 'hashes_verified': True,
                               'all_defined_KL_recomputed': True, 'full_cartesian_panel_verified': True,
                               'worst_full_integrity_raw_kl_bits': max(r['full_integrity_raw_kl_bits'] for r in draws),
                               'distribution_manifest_sha256': digest(folder / 'distribution_manifest.jsonl')},
              'aggregate': {name: aggregate(selected) for name, selected in groups.items()}, 'source': contract,
              'scope': 'Four reused-development toys; unchanged original bank/basis, all policies evaluated on same1792worlds. No English/token-role/real-VPD/scalability certification.',
              'missing': 'Invalid predictions retained in total/top denominator; KL undefined and valid denominator explicit, never filled with zero.',
              'runtime_basis_proof': 'Runtime hash assertions verify original U/unit mapping across edits and restore U/V/units; raw basis tensors not archived.',
              'audit_sha256': digest(Path(__file__))}
    (folder / 'independent_saved_array_audit_replay.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--folder', default='factorial_head_counterfactual')
    parser.add_argument('--trial', default='trial_factorial')
    args = parser.parse_args()
    report = audit(args.inputs, args.folder, args.trial)
    print(json.dumps(report, sort_keys=True))
