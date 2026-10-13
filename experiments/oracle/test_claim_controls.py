import collections
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claim_controls as c


def test_permutations_preserve_geometry_and_shared_identity():
    original = [{'name': 'V0', 'inputs': [0], 'group': [['site', 0, 0], ['site', 0, 1], ['site', 2, 1]]},
                {'name': 'V1', 'inputs': [1], 'group': [['site', 0, 0], ['site', 4, 1]]}]
    changed = c.permute(original, {'site': 16}, 'text6001', 0)
    assert changed == c.permute(original, {'site': 16}, 'text6001', 0)
    assert changed != original
    for a, b in zip(original, changed):
        assert collections.Counter((s, p) for s, i, p in a['group']) == collections.Counter((s, p) for s, i, p in b['group'])
    assert len(set(map(tuple, original[0]['group'])) & set(map(tuple, original[1]['group']))) == len(set(map(tuple, changed[0]['group'])) & set(map(tuple, changed[1]['group'])))
    zero = next(i for s, i, p in changed[0]['group'] if p == 0)
    assert ['site', zero, 1] in changed[0]['group']
    assert ['site', zero, 0] in changed[1]['group']
    assert original[0]['group'][0] == ['site', 0, 0]


@pytest.mark.parametrize('group', [[['block', 'site', 0]], [['site', 16, 0]], [['site', 0, -1]], [['site', 0, 0], ['site', 0, 0]]])
def test_invalid_carriers_rejected(group):
    with pytest.raises(ValueError):
        c.permute([{'name': 'V0', 'inputs': [0], 'group': group}], {'site': 16}, 'text6001', 0)


def fixture(tmp_path):
    uv = tmp_path / 'uv'
    header = json.dumps({'site.U': {'shape': [16, 2]}}).encode()
    uv.write_bytes(struct.pack('<Q', len(header)) + header)
    inp = tmp_path / 'inputs'; (inp / 'text6001').mkdir(parents=True)
    task = inp / 'text6001/task.json'
    task.write_text(json.dumps({'id': 'text6001', 'split': 'train', 'model': 'vpd4l',
                                'prompts': [{'token_ids': [0, 1], 'target_positions': [1]}]}))
    variables = [{'name': 'V0', 'inputs': [0], 'group': [['site', 0, 1]], 'meaning': 'DO NOT INHERIT'}]
    row = {'task': 'text6001', 'training_allowed': True, 'raw_selected_variables': variables,
           'claim_index': 0, 'task_sha256': c.file_sha(task), 'bindings_sha256': {'V0': c.digest(variables[0]['group'])},
           'original_checked_claim': {'executed': {'top': 999, 'holds': False,
                                     'intervention': {'tokens': {}, 'transplants': [['V0', {'0': 2}]], 'remove': []}}}}
    checks = tmp_path / 'checks.jsonl'
    checks.write_text(json.dumps(row) + '\n')
    return checks, inp, uv, row


def test_prepare_deduplicates_interventions_without_inheriting_labels(tmp_path):
    checks, inputs, uv, row = fixture(tmp_path)
    duplicate = json.loads(json.dumps(row)); duplicate['claim_index'] = 1
    duplicate['original_checked_claim']['executed']['top'] = 42
    checks.write_text(json.dumps(row) + '\n' + json.dumps(duplicate) + '\n')
    result = c.prepare(checks, inputs, uv, {'text6016'})
    assert len(result['cases']) == 1
    case = result['cases'][0]
    assert case['source_claim_indices'] == [0, 1]
    assert len(case['arms']) == 3
    assert 'top' not in c.canonical(result)
    assert 'DO NOT INHERIT' not in c.canonical(result)
    assert result['protocol_sha256'] == c.digest([result['tasks'], result['cases']])


@pytest.mark.parametrize('name', ['text9100', 'text9149', 'text9000', 'blind', '../text6001'])
def test_train_boundary_precedes_task_reads(tmp_path, name):
    checks, inputs, uv, row = fixture(tmp_path)
    row['task'] = name
    checks.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='only TRAIN'):
        c.prepare(checks, tmp_path / 'DOES_NOT_EXIST', uv, {'text6016'})


def test_changed_frozen_task_fails_before_protocol(tmp_path):
    checks, inputs, uv, row = fixture(tmp_path)
    (inputs / 'text6001/task.json').write_text('{}')
    with pytest.raises(ValueError, match='task digest'):
        c.prepare(checks, inputs, uv, {'text6016'})


def test_reserved_claim_never_reads_its_task(tmp_path):
    checks, inputs, uv, row = fixture(tmp_path)
    with pytest.raises(ValueError, match='no eligible'):
        c.prepare(checks, tmp_path / 'DOES_NOT_EXIST', uv, {'text6001'})
