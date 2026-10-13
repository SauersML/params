"""Synthetic offline artifacts only; no tensor/model/scorer imports."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import paired_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_arm(path, name):
    path.mkdir()
    (path / 'sft').mkdir()
    (path / 'sft/model.safetensors').write_bytes(b'synthetic common start; never loaded')
    (path / 'train_inputs.jsonl').write_text('synthetic training prompts')
    (path / 'zoo').mkdir()
    (path / 'zoo/manifest.jsonl').write_text('synthetic frozen zoo')
    panel = [{'id': f'case{i:02d}', 'split': ['train', 'test_instance', 'test_family'][i // 8],
              'family': f'family{i // 8}'} for i in range(24)]
    write(path / 'data.json', {'evaluation_panel': panel, 'files': report.files(path / 'zoo')})
    write(path / 'sft.done.json', {'files': report.files(path / 'sft')})
    write(path / 'parent_evidence/contract.json', {'parent': 'identical frozen CPU SFT'})
    imported = report.files(path)
    write(path / 'sft_lineage.json', {'schema': 2, 'imported_files': imported,
                                    'parent_contract_sha256': report.digest(path / 'parent_evidence/contract.json')})
    write(path / 'contract.json', {'rl_steps_per_arm': 20, 'source': 'same', 'device': 'cuda',
                                 'teacher_panel_max_kl_bits': .001,
                                 'sft_lineage_sha256': report.digest(path / 'sft_lineage.json')})
    (path / name).mkdir()
    (path / name / 'model.safetensors').write_bytes(b'synthetic trained policy')
    log = [{'step': i, 'reward_max': 1., 'reward': .5, 'sec': i + 1} for i in range(20)]
    (path / name / 'log.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in log))
    write(path / (name + '.started.json'), {'reward': name})
    write(path / (name + '.done.json'), {'reward': name, 'files': report.files(path / name),
                                       'start_hashes': report.files(path / 'sft')})
    for label in ('gt', 'empty', 'sft', name):
        rows = []
        for entry in panel:
            kl = 0.000001 if label == 'gt' else .2
            measured = {'per_text': [{'suff': 1. - kl, 'kl_bits': kl, 'top': 1, 'reference_top': 1}]}
            rows.append({**entry, 'valid': True, 'prompt_sha256': 'same', 'input_bank_sha256': 'same',
                         'completion': 'synthetic circuit', 'edge_f1': .5, 'unit_f1': .8, 'unit_count': 2, 'edge_count': 3,
                         'stated': measured, 'same_units_complete': measured, 'stated_reward': 1. - kl,
                         'same_units_complete_reward': 1. - kl,
                         'policy_checkpoint_manifest_sha256': report.digest(path / (label + '.done.json')) if label in ('sft', name) else None})
        (path / (label + '_eval.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


@pytest.fixture
def arms(tmp_path):
    return make_arm(tmp_path / 'unit', 'unit_suff'), make_arm(tmp_path / 'wiring', 'wiring_suff')


def rewrite_rows(path, change):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_invalids_stay_in_all_case_metrics_and_kl_coverage(arms):
    unit, wiring = arms
    def invalidate(rows):
        rows[0].update(valid=False, stated_reward=-1., same_units_complete_reward=-1.)
    rewrite_rows(wiring / 'wiring_suff_eval.jsonl', invalidate)
    result = report.compare(unit, wiring, resamples=0)
    overall = result['by_split']['all']['wiring']
    assert overall['n'] == 24 and overall['valid'] == 23
    assert overall['stated_reward_mean'] == pytest.approx((23 * .8 - 1) / 24)
    assert overall['kl_valid_n'] == 23 and overall['top1_mean'] == 23 / 24
    assert result['cases'][0]['wiring']['edge_f1'] == 0
    assert result['cases'][0]['wiring']['raw_parser_edge_f1'] == .5
    assert result['training']['unit_suff']['scheduled_steps'] == 20


def test_missing_case_or_changed_panel_rejected(arms):
    unit, wiring = arms
    file = wiring / 'wiring_suff_eval.jsonl'
    original = file.read_text()
    rewrite_rows(file, lambda rows: rows.pop())
    with pytest.raises(ValueError, match='incomplete'):
        report.compare(unit, wiring, resamples=0)
    file.write_text(original)
    rewrite_rows(file, lambda rows: rows[0].update(id='different-case'))
    with pytest.raises(ValueError, match='panel mismatch'):
        report.compare(unit, wiring, resamples=0)


def test_partial_training_and_baseline_drift_are_not_hidden(arms):
    unit, wiring = arms
    rewrite_rows(wiring / 'sft_eval.jsonl', lambda rows: rows[0].update(completion='different greedy answer'))
    result = report.compare(unit, wiring, resamples=0)
    assert result['baseline_differences']['sft']['changed_completions'] == 1
    log = wiring / 'wiring_suff/log.jsonl'
    log.write_text('\n'.join(log.read_text().splitlines()[:-1]) + '\n')
    done = report.read(wiring / 'wiring_suff.done.json')
    done['files'] = report.files(wiring / 'wiring_suff')
    write(wiring / 'wiring_suff.done.json', done)
    with pytest.raises(ValueError, match='partial'):
        report.compare(unit, wiring, resamples=0)
