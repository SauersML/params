"""Exact measured full-model claim outcomes as auxiliary supervision, never circuit targets.

This exporter does not run a model. Input bindings contain all original members;
weights/activations are absent, so it does not test weight reading or transfer.
Mean-removal outcomes stay in the audit but are excluded from training: the
original alternative-token bank was not retained. All task IDs are preflighted
before reading task content. An explicit reserved-task file is mandatory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'method/oracle'), str(ROOT / 'method/vpd')]
import components
import semantic

# Reserved controls of the existing mechanism-heldout comparison. Callers must
# additionally supply that comparison's complete reserved-task manifest.
RESERVED_CONTROLS = frozenset({'text6016', 'text6050', 'text6065', 'text6162',
                               'text6266', 'text6286', 'text6294', 'text6340'})


def normalized(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(normalized(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervention(value):
    return {'tokens': {}, 'transplants': [], 'remove': [], **value}


def number(value, key, lo=None, hi=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'nonfinite/non-numeric {key}')
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f'out-of-range {key}')
    return value


def task_id(name):
    if not isinstance(name, str) or not re.fullmatch(r'text[67]\d{3}', name):
        raise ValueError('only training text6000–7999 permitted')
    return name


def validate_claim(claim, variables, model):
    """Check EVERY original row, including failures skipped by positive loaders."""
    parsed, bad = semantic.parse_invariants(claim['line'], model)
    if bad or len(parsed) != 1 or normalized(parsed[0]) != normalized(claim.get('parsed')):
        raise ValueError('line/stored numeric parse mismatch')
    executed = claim['executed']
    if (normalized(parsed[0]['intervention']) != normalized(intervention(executed['intervention']))
            or normalized(parsed[0]['outcome']) != normalized(executed['outcome'])):
        raise ValueError('original numeric execution mismatch')
    iv = parsed[0]['intervention']
    refs = sorted({n for n, _ in iv['transplants']} | set(iv['remove']))
    if not refs or not set(refs) <= variables.keys():
        raise ValueError('unknown/empty carrier binding')
    if claim['variable_group_sha256'] != {n: digest(variables[n]['group']) for n in refs}:
        raise ValueError('original group digest mismatch')
    if type(executed.get('holds')) is not bool or type(executed.get('top')) is not int or executed['top'] < 0:
        raise ValueError('missing numeric executed outcome/verdict')
    for key in ('holds', 'credit', 'surprise', 'top', 'p_top', 'logp'):
        if key in claim and claim[key] != executed.get(key):
            raise ValueError('flat/executed outcome mismatch')
    surprise = number(executed['surprise'], 'surprise', 0)
    credit = number(executed['credit'], 'credit')
    if credit != (2 * executed['holds'] - 1) * surprise:
        raise ValueError('credit/verdict mismatch')
    probability = number(executed['p_top'], 'p_top', 0, 1)
    number(executed['p_base_top'], 'p_base_top', 0, 1)
    if executed['p_base_top'] > probability + 1e-7:
        raise ValueError('base-token probability exceeds actual argmax probability')
    kind, value = executed['outcome']
    if kind == 'top':
        if type(value) is not int or value < 0 or executed['holds'] != (executed['top'] == value):
            raise ValueError('numeric top/verdict mismatch')
        lp = number(executed['logp'], 'logp', hi=0)
        top_lp = math.log2(max(probability, 1e-30))
        if executed['holds'] and not math.isclose(lp, top_lp, abs_tol=1e-6):
            raise ValueError('true top probability/logp mismatch')
        if not executed['holds'] and lp > top_lp + 1e-6:
            raise ValueError('failed-event probability exceeds actual argmax')
    elif kind == 'drop':
        number(value, 'drop fraction', 0, 1)
        if not 0 < value < 1 or executed['logp'] is not None:
            raise ValueError('drop evidence mismatch')
    else:
        raise ValueError('unsupported measured outcome kind')
    return refs


def bindings_for(variables, refs):
    result = {}
    for name in refs:
        v = variables[name]
        line = components.variable_text({**v, 'meaning': '', 'feeds': []}, int(name[1:]), {}, {})
        carried = components.VAR_RE.fullmatch(line)[5]
        group, instances = components._carriers(carried, {})
        if (set(map(tuple, group)) != set(map(tuple, v['group'])) or len(group) != len(v['group'])
                or set(map(tuple, instances)) != set(map(tuple, v.get('instances') or []))):
            raise ValueError('carrier/instance serialization mismatch')
        result[name] = {'inputs': v['inputs'], 'carried_by': carried}
    return result



def overlapping_bindings(variables, refs):
    """Detect cross-carrier scalar equality or whole-block containment, without expansion."""
    tables = {}
    for name in refs:
        table = {}
        for member in variables[name]['group']:
            block = member[0] == 'block'
            key = (member[1] if block else member[0], member[2])
            whole, indices = table.setdefault(key, (False, set()))
            if block:
                table[key] = (True, indices)
            else:
                indices.add(member[1])
        tables[name] = table
    overlaps = []
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            for key in tables[a].keys() & tables[b].keys():
                aw, ai = tables[a][key]; bw, bi = tables[b][key]
                if aw or bw or ai & bi:
                    overlaps.append([a, b]);break
    return overlaps


def intervention_legend():
    return ('Predict at the LAST listed target position. The recipient is the original source text with only '
            'intervention.tokens replacements. Each transplant donor is independently the ORIGINAL source text '
            'with only that transplant entry\'s position/token changes; recipient changes and other donors do not '
            'propagate into its donor. Run the otherwise fully enabled fixed model with that donor\'s activation '
            'scalars transplanted into the named rank-one components at their listed matrix/position; whole blocks '
            'replace that matrix\'s entire output at the position. All other components, weights and remainders stay '
            'enabled. Carrier notation L2.fc#3,5@7 means components 3 and 5 of layer 2 MLP input at token position 7; '
            'q/k/v/o are attention query/key/value/output matrices, fc/down are MLP input/output matrices. '
            'L2.fc @7-9 means the whole matrix output at positions 7 through 9. An instances list is informational '
            'membership and is NOT a second set of transplant targets. Overlapping cross-carrier groups are '
            'excluded from training; no order-independent behavior is asserted for such audit rows. ')



def execution_text(model, ids):
    """Decode the WHOLE executed sequence, retaining specials and UTF-8 boundaries."""
    import mech
    return mech.tokenizer(model).decode(ids, skip_special_tokens=False)




_DECODE_CANDIDATES = {}

def archive_context_matches(row, task):
    """Strict visible token projection and target binding; report ID ambiguity."""
    import mech
    prefix = 'tokens (numbered near the positions below; elsewhere plain text, [n] marking position n): '
    lines = row['question'].splitlines()
    token_lines = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    targets = [re.fullmatch(r'target: position (\d+);.*', line) for line in lines if line.startswith('target: position ')]
    p = task['prompts'][0]
    if len(token_lines) != 1 or len(targets) != 1 or targets[0] is None or p['target_positions'] != [int(targets[0][1])]:
        raise ValueError('archived execution target mismatch')
    pattern = re.compile(r"(?:(\d+):|\[(\d+)\])('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
    text = token_lines[0]; matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError('missing archived token listing')
    strings = components.token_strings(task['model'], p['token_ids'])
    tail, covered, decoded = 0, 0, []
    for i, match in enumerate(matches):
        if text[tail:match.start()].strip():
            raise ValueError('unparsed archived token listing')
        start = int(match[1] or match[2]); end = (int(matches[i+1][1] or matches[i+1][2]) if i+1 < len(matches) else len(strings))
        if start != covered or not start < end <= len(strings) or (match[1] is not None and end != start + 1):
            raise ValueError('archived token coverage/positions mismatch')
        expected = ''.join(strings[start:end])
        # escape() breaks oracle-added tokens AFTER repr serialization. Compare
        # exact bytes to that known spelling; never fuzzy-strip backslashes.
        if match[3] != components.escape(repr(expected)):
            raise ValueError('archived token content mismatch')
        decoded.append(ast.literal_eval(repr(expected)))
        covered, tail = end, match.end()
    if text[tail:].strip() or covered != len(strings):
        raise ValueError('archived token listing incomplete')
    canonical = mech.tokenizer(task['model']).encode(''.join(decoded), add_special_tokens=False).ids
    model = task['model']
    if model not in _DECODE_CANDIDATES:
        candidates = {}
        tk = mech.tokenizer(model)
        for z in set(tk.get_vocab().values()):
            candidates.setdefault(tk.decode([z], skip_special_tokens=False), set()).add(z)
        _DECODE_CANDIDATES[model] = candidates
    candidates = _DECODE_CANDIDATES[model]
    singletons = [int(m[1]) for m in matches if m[1] is not None]
    ambiguous = [i for i in singletons if candidates.get(strings[i]) != {p['token_ids'][i]}]
    exact_ids = len(singletons) == len(strings) and not ambiguous
    return {'level': 'exact_singleton_native_ID_reconstruction' if exact_ids else ('exact_numbered_listing_match_canonical_reencoding' if canonical == p['token_ids'] else 'exact_numbered_listing_match_noncanonical_ids_unverified'),
            'complete_positions': len(strings), 'numbered_singleton_positions': len(singletons),
            'numbered_singletons_with_nonunique_decode': ambiguous, 'range_chunks': len(matches)-len(singletons),
            'canonical_full_reencoding_matches_frozen_ids': canonical == p['token_ids'],
            'native_id_certificate': exact_ids,
            'limitation': 'Ranges disclose concatenated piece strings and position boundaries, not each historical native ID. Canonical BPE may change at cuts/specials; noncanonical reencoding is not evidence of wrong visible context.'}


class ObservationArchive:
    """Checksummed historical recipient facts; header identity is NOT tensor identity."""
    def __init__(self, path, manifest_path, records, *, task_inputs):
        import mech
        self.manifest = json.loads(Path(manifest_path).read_text())
        m = self.manifest
        if (m.get('schema') != 1 or m.get('model') != 'vpd4l'
                or m.get('selection_role') != 'baseline_input_observations'
                or m.get('checkpoint_tensor_content_verified') is not False):
            raise ValueError('observations require explicit historical archive/header-only scope')
        if file_sha(path) != m['archive_sha256']:
            raise ValueError('observation archive checksum mismatch')
        if file_sha(mech.VPD4L_TOKENIZER) != m['tokenizer_sha256']:
            raise ValueError('observation tokenizer identity mismatch')
        with Path(m['registry_path']).open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 1 << 24:
                raise ValueError('unreasonable component registry header')
            header_bytes = f.read(size)
        if hashlib.sha256(header_bytes).hexdigest() != m['registry_header_sha256']:
            raise ValueError('observation registry header mismatch')
        header = json.loads(header_bytes)
        self.ranks = {k[:-2]: v['shape'][0] for k, v in header.items() if k.endswith('.U')}
        if not m.get('producer_lineage'):
            raise ValueError('observation producer/code lineage missing')
        for record in records:
            tid = task_id(record['task']); binding = m['tasks'].get(tid)
            expected = {'task_sha256': record['task_sha256'], 'circuit_sha256': record['circuit_sha256'],
                        'variable_group_sha256': {v['name']: digest(v['group']) for v in record['variables']}}
            if binding != expected:
                raise ValueError('observation task/carrier provenance mismatch')
        selected = {record['task'] for record in records}
        self.rows = {}
        self.context_bindings = {}
        with Path(path).open() as f:
            for line in f:
                row = json.loads(line); tid = task_id(row.get('task'))
                if tid in selected:
                    if tid in self.rows:
                        raise ValueError('duplicate observation task')
                    self.context_bindings[tid] = archive_context_matches(row, task_inputs[tid])
                    self.rows[tid] = {'components': row['components'], 'wiring': row['wiring']}
        if set(self.rows) != selected:
            raise ValueError('missing observation task')
        self.provenance = {'archive_sha256': m['archive_sha256'], 'manifest_sha256': file_sha(manifest_path),
                           'tokenizer_sha256': m['tokenizer_sha256'], 'registry_header_sha256': m['registry_header_sha256'],
                           'checkpoint_tensor_content_verified': False, 'producer_lineage': m['producer_lineage'],
                           'archived_context_bindings': self.context_bindings}

    def select(self, record, refs, *, input_length):
        variables = {v['name']: v for v in record['variables']}
        members = {tuple(g) for n in refs for g in variables[n]['group']}
        scalar = {(s, t, c) for s, c, t in members if s != 'block'}
        blocks = {(s, t) for kind, s, t in members if kind == 'block'}
        def node(value):
            if (not isinstance(value, (list, tuple)) or len(value) != 3
                    or value[0] not in self.ranks or type(value[1]) is not int or not 0 <= value[1] < input_length
                    or type(value[2]) is not int or not 0 <= value[2] < self.ranks[value[0]]):
                raise ValueError('invalid archived component coordinate')
            return tuple(value)
        def matches(nd):
            return nd in scalar or nd[:2] in blocks
        facts, seen = [], set()
        row = self.rows[record['task']]
        for group in row['components'].values():
            for fact in group:
                if len(fact) != 3:
                    raise ValueError('unexpected archived component fields')
                nd = node(fact[0])
                if not matches(nd):
                    continue
                if nd in seen or not isinstance(fact[2], str):
                    raise ValueError('duplicate/invalid archived component fact')
                seen.add(nd)
                facts.append({'node_native': list(nd), 'baseline_ci': number(fact[1], 'baseline CI'),
                              'recipient_routed_lens_and_pile_context_description': fact[2]})
        wires = []
        for group in row['wiring'].values():
            for wire in group:
                if len(wire) != 3:
                    raise ValueError('unexpected archived static coupling fields')
                w, r = node(wire[0]), node(wire[1])
                if matches(w) or matches(r):
                    wires.append({'writer_native': list(w), 'reader_native': list(r),
                                  'static_cosine': number(wire[2], 'static cosine', -1.000001, 1.000001)})
        return {'components': facts, 'static_couplings': wires,
                'coverage': {'requested_rank_one_members': len(scalar), 'observed_requested_rank_one_members': len(scalar & seen),
                             'unobserved_rank_one_members': len(scalar - seen), 'requested_block_site_positions': len(blocks),
                             'blocks_with_any_piece_observation': sum(any(nd[:2] == b for nd in seen) for b in blocks),
                             'partial': True, 'block_remainder_observed': False},
                'semantics': 'Coordinates node_native/writer_native/reader_native are [site, token_position, decomposition_component_index]; the component index is neither a token position nor a vector coordinate. For example [h.0.mlp.c_fc,7,3] is L0.fc#3@7. Original-recipient baseline CI is NOT activation. Descriptions are recipient routed-lens linear readouts and Pile activation-context hints, NOT donor outcomes or mechanism labels. Couplings are static cosines, NOT causal effects. Missing observations do NOT mean zero; no block remainder or complete block-output coverage is claimed. Historical archive/registry coordinates are bound; checkpoint tensor-content identity was not retained.'}


def render_row(record, task, claim, *, reserved, observations=None):
    """Return both tasks, preserving the original proposition only in verdict input."""
    task_id(record['task'])
    if task['id'] != record['task'] or task['model'] != record['model'] or task['split'] != 'train':
        raise ValueError('task identity mismatch')
    if (not record.get('source_sha256')
            or not re.fullmatch(r'[0-9a-f]{40}', record.get('source_commit', ''))
            or any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in record.get('source_sha256', {}).values())):
        raise ValueError('invalid pinned proof source provenance')
    variables = {v['name']: v for v in record['variables']}
    if len(variables) != len(record['variables']):
        raise ValueError('duplicate variable names')
    refs = validate_claim(claim, variables, record['model'])
    normalized(claim['executed'])  # Reject nonfinite values anywhere in execution evidence.
    prompt = task['prompts'][0]
    ids, targets = prompt['token_ids'], prompt['target_positions']
    if not ids or not targets or any(type(z) is not int or z < 0 for z in ids) or any(type(p) is not int or not 0 <= p < len(ids) for p in targets):
        raise ValueError('invalid original token/target IDs')
    for name in refs:
        v = variables[name]
        if any(type(p) is not int or not 0 <= p < len(ids) for p in v['inputs']):
            raise ValueError('invalid carrier input positions')
        for member in v['group'] + (v.get('instances') or []):
            if len(member) != 3 or type(member[2]) is not int or not 0 <= member[2] < len(ids):
                raise ValueError('invalid carrier/instance position')
            if member[0] != 'block' and (type(member[1]) is not int or member[1] < 0):
                raise ValueError('invalid component index')
    tokens = components.token_strings(record['model'], ids)
    iv = intervention(claim['executed']['intervention'])
    for p, z in iv['tokens'].items():
        if not re.fullmatch(r'\d+', str(p)) or not 0 <= int(p) < len(ids) or type(z) is not int or z < 0:
            raise ValueError('invalid recipient token intervention')
    donor_tokens = []
    for name, changes in iv['transplants']:
        for p, z in changes.items():
            if not re.fullmatch(r'\d+', str(p)) or not 0 <= int(p) < len(ids) or type(z) is not int or z < 0:
                raise ValueError('invalid donor position/token ID')
            donor_tokens.append({'variable': name, 'position': int(p), 'token_id': z,
                                 'token': components.token_strings(record['model'], [z])[0]})
    actual = claim['executed']['top']
    given = {'model': record['model'], 'source_text': execution_text(record['model'], ids),
             'tokens': [[p, z, tokens[p]] for p, z in enumerate(ids)], 'target_positions': targets,
             'bindings': bindings_for(variables, refs), 'intervention': iv, 'donor_tokens': donor_tokens}
    selected_observations = observations.select(record, refs, input_length=len(ids)) if observations is not None else None
    if selected_observations is not None:
        given['baseline_component_observations'] = selected_observations
    base = (intervention_legend() + 'Predict the actual top next token of this fixed known model under the given exact carrier transplant. '
            'Tokens are [position, model token ID, decoded string]. Carried-by uses the existing oracle grammar; '
            'instances are informational. These IDs do not supply weights or activation observations. '
            'This is full-model intervention outcome supervision, not a circuit or mechanism certificate. '
            'Return JSON with top_token, the exact decoded token string including whitespace.\n')
    if iv['remove']:
        given['remove_operator'] = 'Source-bound alternative activation/output mean replacement, not zero; exact bank IDs not retained.'
    outcome = {'top_token': components.token_strings(record['model'], [actual])[0]}
    verdict_given = {**given, 'proposed_outcome': claim['executed']['outcome']}
    verdict_given['proposed_outcome_token'] = (components.token_strings(record['model'], [claim['executed']['outcome'][1]])[0]
                                              if claim['executed']['outcome'][0] == 'top' else None)
    views = {
        'actual_top': {'prompt': [{'role': 'user', 'content': base + normalized(given)}],
                       'completion': [{'role': 'assistant', 'content': normalized(outcome)}]},
        'proposal_verdict': {'prompt': [{'role': 'user', 'content': intervention_legend() + 'Judge the given proposed event under this exact full-model intervention. A top event holds exactly when that native token ID is the actual argmax. A drop event refers to the ORIGINAL base-top token probability falling strictly below the specified fraction of its ORIGINAL probability. Return JSON with proposal_holds. The proposition is unverified input, not the measured answer.\n' + normalized(verdict_given)}],
                             'completion': [{'role': 'assistant', 'content': normalized({'proposal_holds': claim['executed']['holds']})}]}}
    overlaps = overlapping_bindings(variables, refs)
    reasons = []
    if overlaps:
        reasons.append('overlapping_cross_carrier_targets_operator_precedence_not_inferred')
    if record['task'] in set(reserved) | RESERVED_CONTROLS:
        reasons.append('reserved_task')
    if iv['remove']:
        reasons.append('opaque_mean_removal_bank_not_retained_NOT_zero')
    if not iv['transplants']:
        reasons.append('not_explicit_transplant')
    row = {'numeric_top_truth_recomputed': claim['executed']['outcome'][0] == 'top',
            'drop_truth_recompute': 'not possible from retained exact base distribution; no inference from rounded task probability' if claim['executed']['outcome'][0] == 'drop' else None,
            'schema': 1, 'task': record['task'], 'split': 'train', 'training_allowed': not reasons,
            'overlapping_bindings': overlaps, 'excluded_reasons': reasons, 'refs': refs, 'views': views,
            'raw_selected_variables': [variables[n] for n in refs], 'original_checked_claim': claim,
            'bindings_sha256': claim['variable_group_sha256'], 'task_sha256': record['task_sha256'],
            'circuit_sha256': record['circuit_sha256'], 'source_sha256': record['source_sha256'],
            'source_commit': record['source_commit'], 'original_task_text': prompt['text'], 'scope': 'Original full-model outcomes only, never circuit proofs.',
            'remove_operator': {'kind': 'alternative_activation_mean', 'bank_token_ids_retained': False} if iv['remove'] else None}
    if observations is not None:
        row['observation_provenance'] = observations.provenance
    return row


def export(proofs, inputs, out, reserved, *, observations_path=None, observations_manifest=None):
    # Read proof metadata first; reject the complete task list BEFORE task content.
    records = [json.loads(Path(p).read_text()) for p in proofs]
    names = [task_id(r.get('task')) for r in records]
    if len(set(names)) != len(names):
        raise ValueError('duplicate proof task')
    if out.exists():
        raise ValueError('output directory exists; preserve previous evidence')
    if (observations_path is None) != (observations_manifest is None):
        raise ValueError('observations and observations-manifest must be provided together')
    catalog = ObservationArchive(observations_path, observations_manifest, records, task_inputs={r['task']: json.loads((inputs/r['task']/'task.json').read_text()) for r in records}) if observations_path is not None else None
    all_rows, rejected, source_files = [], [], []
    for record in records:
        tid = record['task']; d = inputs / tid
        task = json.loads((d / 'task.json').read_text())
        raw = json.loads((d / 'variables.json').read_text())
        if normalized(raw) != normalized(record['variables']):
            raise ValueError('frozen variable input mismatch')
        components.checked_invariant_labels(record, raw, record['model'],
                                             file_sha(d / 'task.json'), file_sha(d / 'circuit.json'), tid)
        seen = set()
        for i, claim in enumerate(record['claims']):
            try:
                row = render_row(record, task, claim, reserved=reserved, observations=catalog)
                key = normalized([row['bindings_sha256'], claim['executed']['intervention'], claim['executed']['outcome']])
                if key in seen:
                    raise ValueError('duplicate original executable evidence')
                seen.add(key);row['claim_index'] = i;all_rows.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({'task': tid, 'claim_index': i, 'reason': str(exc), 'original_checked_claim': claim})
        source_files.append({'task': tid, 'task_path': str(d / 'task.json'), 'task_sha256': file_sha(d / 'task.json'),
                             'circuit_sha256': file_sha(d / 'circuit.json')})
    out.mkdir(parents=True)
    def write(name, rows):
        (out / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write('all_checks.jsonl', all_rows);write('rejected_checks.jsonl', rejected)
    for view in ('actual_top', 'proposal_verdict'):
        write(view + '_explicit_transplants.jsonl', [{**r['views'][view], 'info': {'task': r['task'], 'claim_index': r['claim_index'],
                                                                  'bindings_sha256': r['bindings_sha256']}}
                                                    for r in all_rows if r['training_allowed']])
    manifest = {'schema': 1, 'scope': 'Auxiliary full-model intervention prediction, no circuit node-cap changes.',
                'all_checks': len(all_rows), 'rejected_checks': len(rejected),
                'default_training_checks': sum(r['training_allowed'] for r in all_rows),
                'reserved_tasks': sorted(set(reserved) | RESERVED_CONTROLS), 'input_limit': 'IDs/text only; no weights or activations.',
                'source_files': source_files, 'proof_files': [{'path': str(p), 'sha256': file_sha(p)} for p in proofs],
                'renderer_sha256': file_sha(__file__), 'components_sha256': file_sha(components.__file__),
                'semantic_sha256': file_sha(semantic.__file__),
                'files': {p.name: {'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in out.glob('*.jsonl')}}
    if catalog is not None:
        manifest['observation_provenance'] = catalog.provenance
        manifest['input_limit'] = 'Partial historical recipient CI/readout/static coupling hints; no complete weights or donor activations.'
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, action='append', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations', type=Path, help='Opt-in checksummed historical baseline rows; default remains ID/context-only')
    parser.add_argument('--observations-manifest', type=Path)
    parser.add_argument('--reserved-tasks', type=Path, required=True, help='JSON list of reserved task IDs (mandatory)')
    args = parser.parse_args()
    reserved = json.loads(args.reserved_tasks.read_text())
    if not isinstance(reserved, list) or any(not isinstance(t, str) for t in reserved):
        raise ValueError('reserved tasks must be an explicit JSON list')
    print(json.dumps(export(args.proof, args.inputs, args.out, set(reserved), observations_path=args.observations, observations_manifest=args.observations_manifest), indent=2))
