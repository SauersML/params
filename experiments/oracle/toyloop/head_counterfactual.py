"""Fixed-basis head-output counterfactuals on saved toy circuits; no oracle calls.

Use a frozen executor checkout and exported toy inputs. Sparse candidates receive
only embeddings, edited component weights and their original stated topology.
They receive no dense activations, attention patterns or normalization statistics.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

CASES = tuple(f'multihop_{i:03d}' for i in range(4))
SCALES = (0.0, 0.5)
DAMAGE_MIN = 1e-3
INTEGRITY_MAX = 1e-3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def clear_native(nat):
    nat._refs.clear()
    nat._refs_bytes = 0
    nat._rec = (None, None, {})
    nat.__dict__.pop('_sensitivity_kept', None)


@contextlib.contextmanager
def scaled_head(nat, layer, head, scale):
    """Same column edit as weight_family.applied; keep original U and unit IDs.

    W is the dense matrix. V and delta_T are its fixed-basis factorization and
    remainder. Scaling their input rows performs exactly the same head-output
    edit without decomposing the edited matrix again.
    """
    import torch
    if not math.isfinite(scale):
        raise ValueError('head scale must be finite')
    if not 0 <= layer < nat.target.n_layer or not 0 <= head < nat.target.n_head:
        raise ValueError('head outside architecture')
    name = f'h.{layer}.attn.o_proj'
    site = nat.target.site(name)
    saved = {key: getattr(site, key).clone() for key in ('W', 'V', 'U', 'delta_T')}
    units = nat.units[name].clone()
    cols = slice(head * nat.target.hd, (head + 1) * nat.target.hd)
    with torch.no_grad():
        site.W[:, cols] *= scale
        site.V[cols, :] *= scale
        site.delta_T[cols, :] *= scale
    clear_native(nat)
    try:
        assert torch.equal(site.U, saved['U']) and torch.equal(nat.units[name], units)
        yield {'site': name, 'layer': layer, 'head': head, 'scale': scale,
               'weight_delta_l2': float((site.W - saved['W']).norm()),
               'original_head_weight_l2': float(saved['W'][:, cols].norm())}
    finally:
        with torch.no_grad():
            for key, value in saved.items():
                getattr(site, key).copy_(value)
        clear_native(nat)
        assert all(torch.equal(getattr(site, key), value) for key, value in saved.items())
        assert torch.equal(nat.units[name], units)


def kl_bits(p, q):
    # FP32 forwards; double precision accumulation avoids negative roundoff KL.
    return max(0.0, float((p.double().exp() * (p.double() - q.double())).sum()) / math.log(2))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def append(path, value):
    import os
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(source, inputs, trial, out):
    sys.path.insert(0, str(source / 'experiments/oracle/toyloop'))
    import torch
    import render
    import reward_exec
    import circuit
    import planted
    import native
    import goal_eval
    torch.set_num_threads(1)
    manifest = json.loads((inputs / 'manifest.json').read_text())
    for relative, expected in manifest['sources'].items():
        assert digest(source / relative) == expected, ('executor source changed', relative)
    for relative, expected in manifest['files'].items():
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert digest(inputs / relative) == expected, ('frozen export changed', relative)
    payload = json.loads((trial / 'payload_contract.json').read_text())
    for name, expected in payload['files'].items():
        assert Path(name).name == name and digest(trial / name) == expected
    samples = {arm: {row['id']: row for row in rows(trial / (arm + '.jsonl'))} for arm in ('base', 'trained')}
    assert all(len(value) == 24 for value in samples.values())
    truth = {row['id']: row for row in rows(inputs / 'panel_truth.jsonl')}
    bindings = {row['id']: row for row in manifest['panel']}
    out.mkdir()
    contract = {'schema': 1, 'cases': CASES, 'scales': SCALES, 'heads': 'all architectural heads',
                'damage_min_bits': DAMAGE_MIN, 'integrity_max_bits': INTEGRITY_MAX,
                'source': manifest['execution_source'], 'driver_sha256': digest(__file__),
                'inputs_manifest_sha256': digest(inputs / 'manifest.json'),
                'trial_payload_contract_sha256': digest(trial / 'payload_contract.json'),
                'sparse': 'C, live=False, own routing/norms; no dense trace or edited norms supplied',
                'integrity': 'ALL original pieces/all legal connections + remainder vs dense edited model',
                'gt': 'reported candidate; edited GT failure is not an integrity failure',
                'delta_transfer': 'secondary conditional diagnostic uses original dense output; not standalone',
                'scope': 'Four reused-development toys, no English/token-role/real-VPD certification'}
    (out / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for ident in CASES:
        binding = bindings[ident]
        case = inputs / 'cases' / ident
        toy = render.from_zoo({'weights': Path(binding['weights_source']).name,
                               'record': Path(binding['record_source']).name}, case)
        bank = json.loads((case / 'bank.json').read_text())
        assert hashlib.sha256(json.dumps(bank).encode()).hexdigest() == binding['input_bank_sha256']
        assert reward_exec.texts_of(toy) == bank
        nat = planted.native_of(toy.sd, toy.cfg)
        basis = {n: {'U': tensor_hash(nat.target.site(n).U), 'V': tensor_hash(nat.target.site(n).V),
                     'units': tensor_hash(nat.units[n])} for n in nat.names}
        bound = copy.copy(toy)
        bound.meta = dict(toy.meta)
        bound.meta['decomposition'] = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n]) for n in nat.names}
        texts = {'gt': truth[ident]['completion']}
        for arm in ('base', 'trained'):
            sample = samples[arm][ident]
            assert sample['sample'] == 0 and sample['finish'] == 'stop' and not sample.get('parse_error')
            assert sample['prompt_sha256'] == binding['prompt_sha256']
            texts[arm] = sample['completion']
        parsed = {arm: circuit.parse(text) for arm, text in texts.items()}
        # Compile every topology before ANY edited world; no recarving under edits.
        compiled = [{arm: circuit.unit_graph(graph, bound, ids) for arm, graph in parsed.items()} for ids in bank]
        full = [native.Graph(masks={n: torch.ones(len(ids), nat.C[n], dtype=torch.bool) for n in nat.names}) for ids in bank]
        baseline = []
        for index, ids in enumerate(bank):
            targets = [len(ids) - 1]
            ge = goal_eval.GoalEval(nat)
            p0 = nat.reference([ids], targets).clone()
            norms = {}
            integrity = ge.graph_ref(ids, targets, full[index], live=True, norms=norms)
            assert kl_bits(p0, integrity) <= INTEGRITY_MAX, 'baseline full architecture mismatch'
            q0 = {arm: ge.graph_ref(ids, targets, graph, live=False).clone() for arm, graph in compiled[index].items()}
            baseline.append((p0, q0, norms))
            append(out / 'baseline.jsonl', {'task': ident, 'input': index,
                    'full_integrity_kl_bits': kl_bits(p0, integrity),
                    'candidate_kl_bits': {arm: kl_bits(p0, q) for arm, q in q0.items()}})
        # Explicit no-op through the same edit/cache path; no new decomposition.
        with scaled_head(nat, 0, 0, 1.0):
            ids = bank[0]
            ge = goal_eval.GoalEval(nat)
            assert torch.equal(nat.reference([ids], [len(ids) - 1]), baseline[0][0])
            for arm, graph in compiled[0].items():
                assert torch.equal(ge.graph_ref(ids, [len(ids) - 1], graph, live=False), baseline[0][1][arm])
        for layer in range(toy.cfg['n_layer']):
            for head in range(toy.cfg['n_head']):
                for scale in SCALES:
                    with scaled_head(nat, layer, head, scale) as edit:
                        assert all(tensor_hash(nat.target.site(n).U) == basis[n]['U'] and
                                   tensor_hash(nat.units[n]) == basis[n]['units'] for n in nat.names)
                        for index, ids in enumerate(bank):
                            p0, q0, original_norms = baseline[index]
                            targets = [len(ids) - 1]
                            # Fresh GoalEval; direct graph_ref never uses Engine/prepacked caches.
                            ge = goal_eval.GoalEval(nat)
                            pe = nat.reference([ids], targets)
                            edited_norms = {}
                            complete = ge.graph_ref(ids, targets, full[index], live=True, norms=edited_norms)
                            integrity_kl = kl_bits(pe, complete)
                            damage = kl_bits(pe, p0)
                            result = {'task': ident, 'input': index, 'edit': edit,
                                      'model_damage_bits': damage, 'active_damage': damage >= DAMAGE_MIN,
                                      'full_integrity_kl_bits': integrity_kl,
                                      'norm_shift_l2': {str(k): float((edited_norms[k] - v).norm()) for k, v in original_norms.items()},
                                      'reference_top': int(pe[0, -1].argmax()), 'original_top': int(p0[0, -1].argmax()),
                                      'candidates': {}}
                            for arm, graph in compiled[index].items():
                                qe = ge.graph_ref(ids, targets, graph, live=False)
                                moved = torch.log_softmax(p0 + qe - q0[arm], -1)
                                result['candidates'][arm] = {'direct_kl_bits': kl_bits(pe, qe),
                                    'no_change_error_bits': damage, 'gain_over_no_change_bits': damage - kl_bits(pe, qe),
                                    'conditional_delta_kl_bits': kl_bits(pe, moved),
                                    'prediction_shift_bits': kl_bits(qe, q0[arm]),
                                    'top': int(qe[0, -1].argmax()), 'original_top': int(q0[arm][0, -1].argmax())}
                            append(out / 'draws.jsonl', result)
                            assert integrity_kl <= INTEGRITY_MAX, 'edited full architecture mismatch (row retained)'
                    assert all(tensor_hash(nat.target.site(n).U) == basis[n]['U'] and
                               tensor_hash(nat.target.site(n).V) == basis[n]['V'] and
                               tensor_hash(nat.units[n]) == basis[n]['units'] for n in nat.names)
                    clear_native(nat)
                    restored = nat.reference([bank[0]], [len(bank[0]) - 1])
                    assert torch.equal(restored, baseline[0][0]), 'restored dense model changed'
                    append(out / 'restore.jsonl', {'task': ident, 'layer': layer, 'head': head,
                                                   'scale': scale, 'basis_identical': True, 'dense_identical': True})
        print(json.dumps({'task': ident, 'inputs': len(bank), 'head_edits': toy.cfg['n_layer'] * toy.cfg['n_head'] * len(SCALES)}), flush=True)
    saved = rows(out / 'draws.jsonl')
    summary = {'draws': len(saved), 'active_damage': sum(row['active_damage'] for row in saved),
               'inactive_damage': sum(not row['active_damage'] for row in saved),
               'worst_full_integrity_kl_bits': max(row['full_integrity_kl_bits'] for row in saved),
               'groups': {}}
    for task in CASES:
        subset = [row for row in saved if row['task'] == task]
        summary['groups'][task] = {}
        for damage_group in ('all', 'active', 'inactive'):
            selected = [row for row in subset if damage_group == 'all' or row['active_damage'] == (damage_group == 'active')]
            summary['groups'][task][damage_group] = {'n': len(selected), 'arms': {}}
            for arm in texts:
                scores = [row['candidates'][arm] for row in selected]
                summary['groups'][task][damage_group]['arms'][arm] = {
                    'mean_direct_kl_bits': sum(s['direct_kl_bits'] for s in scores) / len(scores) if scores else None,
                    'worst_direct_kl_bits': max((s['direct_kl_bits'] for s in scores), default=None),
                    'mean_conditional_delta_kl_bits': sum(s['conditional_delta_kl_bits'] for s in scores) / len(scores) if scores else None,
                    'beats_no_change': sum(s['direct_kl_bits'] < s['no_change_error_bits'] for s in scores),
                    'top_correct': sum(s['top'] == row['reference_top'] for s, row in zip(scores, selected))}
    (out / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('HEAD_COUNTERFACTUAL_JSON=' + json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'inputs', 'trial', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.inputs, args.trial, args.out)
