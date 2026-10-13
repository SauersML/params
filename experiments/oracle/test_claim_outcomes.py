import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claim_outcomes as c


def fixture(monkeypatch, holds=True):
    iv = {'tokens': {}, 'transplants': [['V3', {'0': 8}]], 'remove': []}
    outcome = ['top', 9]
    parsed = {'id': 1, 'intervention': iv, 'outcome': outcome, 'line': 'I1: source claim'}
    monkeypatch.setattr(c.semantic, 'parse_invariants', lambda *a: ([copy.deepcopy(parsed)], 0))
    monkeypatch.setattr(c, 'execution_text', lambda m, ids: ' authoritative whole text<|endoftext|>')
    monkeypatch.setattr(c.components, 'token_strings', lambda m, ids: [' token' + str(z) for z in ids])
    variable = {'name': 'V3', 'meaning': 'unverified teacher gloss', 'inputs': [0],
                'group': [['h.0.mlp.c_fc', 3, 1], ['h.0.mlp.c_fc', 4, 1]],
                'instances': [['h.0.mlp.c_fc', 7, 1]], 'feeds': ['prediction']}
    ex = {'intervention': iv, 'outcome': outcome, 'holds': holds, 'credit': 2 if holds else -2,
          'surprise': 2, 'top': 9 if holds else 10, 'p_top': .5, 'p_base_top': .1,
          'logp': -1 if holds else -2}
    claim = {'line': parsed['line'], 'parsed': parsed, 'executed': ex,
             'variable_group_sha256': {'V3': c.digest(variable['group'])}, **{k: ex[k] for k in ['holds', 'credit', 'top', 'p_top', 'surprise', 'logp']}}
    record = {'task': 'text6003', 'model': 'vpd4l', 'variables': [variable], 'task_sha256': 'a'*64,
              'circuit_sha256': 'b'*64, 'source_sha256': {'semantic': 'c'*64}, 'source_commit': 'd'*40}
    task = {'id': 'text6003', 'split': 'train', 'model': 'vpd4l',
            'prompts': [{'text': ' source text', 'token_ids': [1, 2], 'target_positions': [1]}]}
    return record, task, claim


@pytest.mark.parametrize('holds', [True, False])
def test_true_and_false_exact_bindings_and_no_primary_outcome_leak(monkeypatch, holds):
    record, task, claim = fixture(monkeypatch, holds)
    row = c.render_row(record, task, claim, reserved=set())
    primary = json.loads(row['views']['actual_top']['prompt'][0]['content'].split('\n', 1)[1])
    verdict = json.loads(row['views']['proposal_verdict']['prompt'][0]['content'].split('\n', 1)[1])
    assert set(primary) == {'model', 'source_text', 'tokens', 'target_positions', 'bindings', 'intervention', 'donor_tokens'}
    assert 'outcome' not in primary and 'proposed_outcome' not in primary
    assert 'unverified teacher gloss' not in str(primary)
    assert primary['donor_tokens'][0]['token'] == ' token8'
    assert primary['bindings']['V3']['inputs'] == [0]
    assert c.components._carriers(primary['bindings']['V3']['carried_by'], {})[0] == record['variables'][0]['group']
    assert verdict['proposed_outcome'] == ['top', 9]
    assert json.loads(row['views']['actual_top']['completion'][0]['content']) == {'top_token': ' token' + str(claim['executed']['top'])}
    assert json.loads(row['views']['proposal_verdict']['completion'][0]['content']) == {'proposal_holds': holds}
    assert row['training_allowed']


@pytest.mark.parametrize('damage', ['intervention', 'outcome', 'group', 'truth', 'credit', 'nan', 'probability', 'logp'])
def test_failed_claim_cannot_bypass_numeric_proof_validation(monkeypatch, damage):
    record, task, claim = fixture(monkeypatch, False)
    if damage == 'intervention':claim['executed']['intervention'] = {'tokens': {}, 'transplants': [['V3', {'0': 999}]], 'remove': []}
    elif damage == 'outcome':claim['executed']['outcome'] = ['top', 999]
    elif damage == 'group':claim['variable_group_sha256']['V3'] = '0'*64
    elif damage == 'truth':claim['executed']['holds'] = claim['holds'] = True
    elif damage == 'credit':claim['executed']['credit'] = claim['credit'] = 2
    elif damage == 'nan':claim['executed']['p_top'] = claim['p_top'] = math.nan
    elif damage == 'probability':claim['executed']['p_base_top'] = .6
    else:claim['executed']['logp'] = claim['logp'] = -.5
    with pytest.raises(ValueError):c.render_row(record, task, claim, reserved=set())


def test_reserved_exclusion_and_mean_removal_is_not_zero(monkeypatch):
    record, task, claim = fixture(monkeypatch)
    assert not c.render_row(record, task, claim, reserved={'text6003'})['training_allowed']
    iv = {'tokens': {}, 'transplants': [], 'remove': ['V3']}
    outcome = ['drop', .5]
    claim['parsed'].update(intervention=iv, outcome=outcome)
    claim['executed'].update(intervention=iv, outcome=outcome, logp=None)
    claim['logp'] = None
    claim['executed']['surprise'] = claim['surprise'] = 1
    claim['executed']['credit'] = claim['credit'] = 1
    monkeypatch.setattr(c.semantic, 'parse_invariants', lambda *a: ([copy.deepcopy(claim['parsed'])], 0))
    row = c.render_row(record, task, claim, reserved=set())
    assert not row['training_allowed']
    assert row['remove_operator']['kind'] == 'alternative_activation_mean'
    assert row['remove_operator']['bank_token_ids_retained'] is False


@pytest.mark.parametrize('name', ['text9100', 'text9149', 'text8999', 'text8000', 'text5999'])
def test_task_boundary_before_any_task_content(tmp_path, name):
    proof = tmp_path/'proof.json';proof.write_text(json.dumps({'task': name}))
    with pytest.raises(ValueError, match='training'):c.export([proof], tmp_path/'MISSING', tmp_path/'out', set())
    assert not (tmp_path/'out').exists()


def test_known_reserved_control_stays_excluded_with_empty_caller_set(monkeypatch):
    record, task, claim = fixture(monkeypatch)
    record['task'] = task['id'] = 'text6016'
    row = c.render_row(record, task, claim, reserved=set())
    assert not row['training_allowed'] and 'reserved_task' in row['excluded_reasons']


def test_self_contained_last_target_and_independent_donor_contract(monkeypatch):
    record, task, claim = fixture(monkeypatch)
    row = c.render_row(record, task, claim, reserved=set())
    for view in row['views'].values():
        prompt = view['prompt'][0]['content']
        assert 'LAST listed target' in prompt
        assert 'independently the ORIGINAL source text' in prompt
        assert 'recipient changes and other donors do not propagate' in prompt
        assert 'entire output' in prompt and 'NOT a second set of transplant targets' in prompt


def test_overlapping_scalar_and_block_carriers_excluded_not_silently_commuted(monkeypatch):
    record, task, claim = fixture(monkeypatch)
    second = copy.deepcopy(record['variables'][0]);second['name'] = 'V4'
    for group in ([['h.0.mlp.c_fc', 3, 1]], [['block', 'h.0.mlp.c_fc', 1]]):
        second['group'] = group;record['variables'] = record['variables'][:1] + [second]
        iv = copy.deepcopy(claim['parsed']['intervention']);iv['transplants'].append(['V4', {'0': 7}])
        claim['parsed']['intervention'] = iv;claim['executed']['intervention'] = iv
        claim['variable_group_sha256']['V4'] = c.digest(group)
        monkeypatch.setattr(c.semantic, 'parse_invariants', lambda *a: ([copy.deepcopy(claim['parsed'])], 0))
        row = c.render_row(record, task, claim, reserved=set())
        assert row['overlapping_bindings'] == [['V3', 'V4']]
        assert not row['training_allowed']
        assert 'overlapping_cross_carrier_targets_operator_precedence_not_inferred' in row['excluded_reasons']
        claim['parsed']['intervention']['transplants'] = claim['parsed']['intervention']['transplants'][:1]
        claim['executed']['intervention'] = claim['parsed']['intervention']


def test_whole_sequence_decode_keeps_specials_and_utf8_boundaries(monkeypatch):
    from types import SimpleNamespace
    calls = []
    class Tokenizer:
        def decode(self, ids, skip_special_tokens):
            calls.append((ids, skip_special_tokens))
            assert ids == [1, 2, 3]
            return '完整<|endoftext|>'
    monkeypatch.setitem(sys.modules, 'mech', SimpleNamespace(tokenizer=lambda model: Tokenizer()))
    assert c.execution_text('vpd4l', [1, 2, 3]) == '完整<|endoftext|>'
    assert calls == [([1, 2, 3], False)]


def test_original_display_text_is_audit_only_authoritative_decode_given(monkeypatch):
    record, task, claim = fixture(monkeypatch)
    task['prompts'][0]['text'] = 'OMITTED_SPECIAL_DISPLAY_TEXT'
    row = c.render_row(record, task, claim, reserved=set())
    for view in row['views'].values():
        assert 'OMITTED_SPECIAL_DISPLAY_TEXT' not in view['prompt'][0]['content']
        assert 'authoritative whole text<|endoftext|>' in view['prompt'][0]['content']
    assert row['original_task_text'] == 'OMITTED_SPECIAL_DISPLAY_TEXT'
