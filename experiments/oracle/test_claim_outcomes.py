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


def observation_fixture(tmp_path, monkeypatch, record):
    import struct
    from types import SimpleNamespace
    tokenizer = tmp_path/'tokenizer.json';tokenizer.write_text('tokenizer fixture')
    monkeypatch.setitem(sys.modules, 'mech', SimpleNamespace(VPD4L_TOKENIZER=tokenizer, tokenizer=lambda model: SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=[1,2] if text==' token1 token2' else []), get_vocab=lambda: {'a':1,'b':2}, decode=lambda ids,skip_special_tokens=False: ' token'+str(ids[0]))))
    registry = tmp_path/'registry.safetensors'
    header = json.dumps({'h.0.mlp.c_fc.U': {'shape': [12, 2], 'dtype': 'F32', 'data_offsets': [0, 96]}}).encode()
    registry.write_bytes(struct.pack('<Q', len(header)) + header)
    archive = tmp_path/'rows.jsonl'
    row = {'task': 'text6003', 'question': "tokens (numbered near the positions below; elsewhere plain text, [n] marking position n): [0]' token1 token2'\ntarget: position 1; baseline omitted", 'components': {'1': [
        [['h.0.mlp.c_fc', 1, 3], .7, "routed lens at this position raises ' x'; active on ' the'"],
        [['h.0.mlp.c_fc', 1, 7], .4, 'INSTANCE_ONLY_NOT_A_CARRIER'],
        [['h.0.mlp.c_fc', 0, 3], .2, 'WRONG_POSITION'],
        [['h.0.mlp.c_fc', 1, 9], .1, 'UNSELECTED_PIECE']]},
        'wiring': {'1': [[['h.0.mlp.c_fc', 1, 3], ['h.0.mlp.c_fc', 1, 9], .3]]},
        'events': ['PRIVATE_MEASURED_OUTCOME'], 'switch_on': ['PRIVATE_EVENT'],
        'qa': ['PRIVATE_ANSWER'], 'values': ['PRIVATE_DONOR_OUTCOME']}
    archive.write_text(json.dumps(row)+'\n')
    manifest = {'schema': 1, 'model': 'vpd4l', 'selection_role': 'baseline_input_observations',
                'checkpoint_tensor_content_verified': False, 'archive_sha256': c.file_sha(archive),
                'tokenizer_sha256': c.file_sha(tokenizer), 'registry_path': str(registry),
                'registry_header_sha256': hashlib.sha256(header).hexdigest(),
                'producer_lineage': {'historical_producer_bytehash_retained': False, 'recipe': 'baseline recipient CI/lens'},
                'tasks': {record['task']: {'task_sha256': record['task_sha256'], 'circuit_sha256': record['circuit_sha256'],
                                          'variable_group_sha256': {v['name']: c.digest(v['group']) for v in record['variables']}}}}
    manifest_path = tmp_path/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    return archive, manifest_path, manifest


def test_observations_whitelist_tuple_order_and_partial_rank_coverage(tmp_path, monkeypatch):
    record, task, claim = fixture(monkeypatch)
    archive, manifest_path, _ = observation_fixture(tmp_path, monkeypatch, record)
    catalog = c.ObservationArchive(archive, manifest_path, [record], task_inputs={record['task']: task})
    base = c.render_row(record, task, claim, reserved=set())
    row = c.render_row(record, task, claim, reserved=set(), observations=catalog)
    given = json.loads(row['views']['actual_top']['prompt'][0]['content'].split('\n', 1)[1])
    facts = given['baseline_component_observations']
    assert len(facts['components']) == 1
    assert facts['components'][0]['node_native'] == ['h.0.mlp.c_fc', 1, 3]
    assert facts['components'][0]['baseline_ci'] == .7
    assert facts['coverage']['requested_rank_one_members'] == 2
    assert facts['coverage']['observed_requested_rank_one_members'] == 1
    assert facts['coverage']['unobserved_rank_one_members'] == 1
    assert facts['coverage']['partial'] is True
    assert 'static_cosine' in facts['static_couplings'][0]
    assert 'NOT activation' in facts['semantics'] and 'NOT causal effects' in facts['semantics']
    for forbidden in ['PRIVATE_', 'INSTANCE_ONLY', 'WRONG_POSITION', 'UNSELECTED_PIECE']:
        assert forbidden not in str(facts)
    assert given['bindings'] == json.loads(base['views']['actual_top']['prompt'][0]['content'].split('\n', 1)[1])['bindings']
    assert row['views']['actual_top']['completion'] == base['views']['actual_top']['completion']
    assert row['views']['proposal_verdict']['completion'] == base['views']['proposal_verdict']['completion']
    assert row['observation_provenance']['checkpoint_tensor_content_verified'] is False


def test_block_observation_counts_are_not_full_block_coverage(tmp_path, monkeypatch):
    record, task, claim = fixture(monkeypatch)
    record['variables'][0]['group'] = [['block', 'h.0.mlp.c_fc', 1]]
    claim['variable_group_sha256']['V3'] = c.digest(record['variables'][0]['group'])
    archive, manifest_path, _ = observation_fixture(tmp_path, monkeypatch, record)
    catalog = c.ObservationArchive(archive, manifest_path, [record], task_inputs={record['task']: task})
    facts = catalog.select(record, ['V3'], input_length=2)
    assert len(facts['components']) == 3  # Native block includes all three listed scalars at pos1, NOT pos0.
    assert facts['coverage']['requested_rank_one_members'] == 0
    assert facts['coverage']['requested_block_site_positions'] == 1
    assert facts['coverage']['blocks_with_any_piece_observation'] == 1
    assert facts['coverage']['block_remainder_observed'] is False
    assert 'complete block-output coverage' in facts['semantics']


@pytest.mark.parametrize('damage', ['archive', 'task', 'group', 'tokenizer', 'registry', 'tensor_claim'])
def test_observation_source_and_task_binding_fail_closed(tmp_path, monkeypatch, damage):
    record, task, claim = fixture(monkeypatch)
    archive, manifest_path, manifest = observation_fixture(tmp_path, monkeypatch, record)
    if damage == 'archive':archive.write_text(archive.read_text()+' ')
    elif damage == 'task':manifest['tasks']['text6003']['task_sha256'] = '0'*64
    elif damage == 'group':manifest['tasks']['text6003']['variable_group_sha256']['V3'] = '0'*64
    elif damage == 'tokenizer':manifest['tokenizer_sha256'] = '0'*64
    elif damage == 'registry':manifest['registry_header_sha256'] = '0'*64
    else:manifest['checkpoint_tensor_content_verified'] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):c.ObservationArchive(archive, manifest_path, [record], task_inputs={record['task']: task})


@pytest.mark.parametrize('damage', ['empty_source', 'empty_targets'])
def test_public_renderer_rejects_missing_provenance_or_prediction_target(monkeypatch, damage):
    record, task, claim = fixture(monkeypatch)
    if damage == 'empty_source':record['source_sha256'] = {}
    else:task['prompts'][0]['target_positions'] = []
    with pytest.raises(ValueError):c.render_row(record, task, claim, reserved=set())


def test_unselected_static_endpoint_outside_input_rejected(tmp_path, monkeypatch):
    record, task, claim = fixture(monkeypatch)
    archive, manifest_path, manifest = observation_fixture(tmp_path, monkeypatch, record)
    row = json.loads(archive.read_text())
    row['wiring']['1'][0][1][1] = 2  # Adjacent, but outside the two-token executed input.
    archive.write_text(json.dumps(row)+'\n')
    manifest['archive_sha256'] = c.file_sha(archive);manifest_path.write_text(json.dumps(manifest))
    catalog = c.ObservationArchive(archive, manifest_path, [record], task_inputs={record['task']: task})
    with pytest.raises(ValueError, match='coordinate'):
        c.render_row(record, task, claim, reserved=set(), observations=catalog)


def test_archived_context_changed_token_same_taskname_rejected(tmp_path, monkeypatch):
    record, task, claim = fixture(monkeypatch)
    archive, manifest_path, manifest = observation_fixture(tmp_path, monkeypatch, record)
    row = json.loads(archive.read_text());row['question'] = row['question'].replace('token2', 'WRONG_TOKEN')
    archive.write_text(json.dumps(row)+'\n');manifest['archive_sha256'] = c.file_sha(archive)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='token content'):
        c.ObservationArchive(archive, manifest_path, [record], task_inputs={record['task']: task})


@pytest.mark.parametrize('taskname', ['text6008', 'text6012'])
def test_context_retains_internal_eot_and_strict_target(monkeypatch, taskname):
    from types import SimpleNamespace
    task = {'model':'vpd4l','prompts':[{'token_ids':[1,2,3], 'target_positions':[2]}]}
    strings = [' a', '<|endoftext|>', ' b']
    monkeypatch.setattr(c.components, 'token_strings', lambda *args: strings)
    monkeypatch.setitem(sys.modules, 'mech', SimpleNamespace(tokenizer=lambda model: SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=[1,2,3] if text==' a<|endoftext|> b' else []), get_vocab=lambda: {'a':1,'eot':2,'b':3}, decode=lambda ids,skip_special_tokens=False: strings[ids[0]-1])))
    listing = c.components.escape('[0]'+repr(''.join(strings)))
    row = {'task':taskname,'question':'tokens (numbered near the positions below; elsewhere plain text, [n] marking position n): '+listing+'\ntarget: position 2; ignored'}
    assert c.archive_context_matches(row,task)
    row['question'] = row['question'].replace('target: position 2','target: position 1')
    with pytest.raises(ValueError):c.archive_context_matches(row,task)


def test_noncanonical_reencoding_is_reported_not_wrong_context(monkeypatch):
    from types import SimpleNamespace
    task = {'model':'fixture_noncanonical','prompts':[{'token_ids':[1,2], 'target_positions':[1]}]}
    monkeypatch.setattr(c.components, 'token_strings', lambda *args: [' a',' b'])
    tk = SimpleNamespace(encode=lambda *args,**kwargs: SimpleNamespace(ids=[3]), get_vocab=lambda: {'a':1,'b':2,'ab':3}, decode=lambda ids,**kwargs: {1:' a',2:' b',3:' a b'}[ids[0]])
    monkeypatch.setitem(sys.modules, 'mech', SimpleNamespace(tokenizer=lambda model: tk))
    row = {'question':"tokens (numbered near the positions below; elsewhere plain text, [n] marking position n): [0]' a b'\ntarget: position 1; ignored"}
    binding = c.archive_context_matches(row,task)
    assert not binding['canonical_full_reencoding_matches_frozen_ids']
    assert binding['level'] == 'exact_numbered_listing_match_noncanonical_ids_unverified'
    assert not binding['native_id_certificate']
