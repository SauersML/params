"""Outcome-blind analytic model construction; no Native or teacher execution.

The 24 new models are 16 TRAIN and 8 new-instance evaluation models. The
protocol's eight previously exposed TRAIN models remain separate inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL_SHA = '6f3af9568bf36aefa8440b3d63863a3937ced528d4ff9d627d3b6547ba52d9aa'
GENERATOR_COMMIT = 'cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4'
FAMILIES = ('additive_vote', 'bigram_mlp', 'binding_positional', 'cancel',
            'copy_offset', 'gate', 'induction', 'redundant')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications(protocol):
    if tuple(protocol['families']) != FAMILIES or protocol['model_counts'] != {
            'existing_exposed_train': 8, 'new_train': 16, 'new_model_evaluation': 8}:
        raise ValueError('frozen family/count protocol changed')
    expected = GENERATOR_COMMIT + ' zoo.make(family,seed)'
    if protocol['generator_source'] != expected:
        raise ValueError('analytic generator source changed')
    return [{'case': f'{family}_{seed}', 'family': family, 'seed': seed,
             'split': split, 'context_seed': seed * 7 + 3}
            for split, variants, base in [('train', (0, 1), 3100000),
                                          ('evaluation', (0,), 3200000)]
            for index, family in enumerate(FAMILIES)
            for variant in variants for seed in [base + 100 * index + variant]]


def tensor_digest(value):
    import torch
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def construct(zoo, spec):
    """No record/gt/check/model forward; intended sampler answers discarded."""
    import torch
    params, sd = zoo.make(spec['family'], spec['seed'])
    cfg = dict(zoo.CFG)
    for value in sd.values():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('nonfinite/non-tensor analytic weight')
    samples = zoo.texts(params, 4, spec['context_seed'], 'ordinary')
    if len(samples) != 4:
        raise ValueError('exactly four disclosed inputs required')
    vocab = int(sd['wte.weight'].shape[0])
    examples = []
    for ids, _unused_intended_answer in samples:
        if not ids or any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError('invalid disclosed input IDs')
        examples.append({'kind': 'ordinary', 'ids': list(ids), 'target': len(ids)-1})
    # Match reward_exec's architecture intervention panel, including every head
    # and each neuron whose output column is nonzero, not only active examples.
    neurons = sum(int(sd[f'h.{layer}.mlp.down_proj.weight'].ne(0).any(dim=0).sum())
                  for layer in range(cfg['n_layer']))
    heads = cfg['n_layer'] * cfg['n_head']
    worlds = 4 * (1 + 2 * (heads + neurons))
    record = {**spec, 'id': spec['case'], 'bank_role': spec['split'],
              'legacy_generator_hash_split': zoo.split_of(spec['family'], spec['seed']), 'examples': examples,
              'label_status': 'unmeasured; no point gold or teacher target',
              'architecture': {'layers': cfg['n_layer'], 'heads': heads,
                               'nonzero_output_neurons': neurons,
                               'mlp_width': int(sd['h.0.mlp.down_proj.weight'].shape[1]),
                               'vocabulary': vocab},
              'teacher_bound': {'worlds': worlds, 'max_attempts': 64,
                                'forward_cap': 69 * worlds,
                                'definition': '3 setup +64 candidate +2 final crosscheck forwards/world'},
              'preparation_native_forwards': 0}
    family = zoo.FAMILIES[spec['family']]
    point_gold = {'role': 'source_derived_point_gold_control_only_not_observations',
                  'unit_edges': family.edges(params),
                  'context_units': [family.gt(params, example['ids']) for example in examples],
                  'verified': False, 'native_forwards': 0}
    return sd, cfg, params, record, point_gold


def generate(zoo, protocol_path, out, source_binding):
    import torch
    if sha(protocol_path) != PROTOCOL_SHA or out.exists():
        raise ValueError('exact protocol and fresh immutable output required')
    rows = specifications(json.loads(protocol_path.read_text()))
    out.mkdir(parents=True)
    statuses = []
    for spec in rows:
        row = {**spec, 'status': 'construction_error'}
        try:
            sd, cfg, params, record, point_gold = construct(zoo, spec)
            dest = out / spec['case']; dest.mkdir()
            torch.save({'sd': sd, 'cfg': cfg}, dest / 'weights.pt')
            for name, value in [('record.json', record), ('config.json', cfg),
                                ('private_generator_params.json', params),
                                ('point_gold_control.json', point_gold)]:
                (dest / name).write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
            row.update(status='constructed_unmeasured', files={p.name: sha(p) for p in dest.iterdir()},
                       tensor_sha256={k: tensor_digest(v) for k, v in sd.items()},
                       **{k: record[k] for k in ('architecture', 'teacher_bound', 'preparation_native_forwards')})
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
        statuses.append(row)
        (out / 'status.json').write_text(json.dumps(statuses, indent=2)+'\n')
    manifest = {'schema': 1, 'protocol_sha256': PROTOCOL_SHA, 'generator_commit': GENERATOR_COMMIT,
                'source_binding': source_binding, 'models': statuses,
                'preparation_native_forwards': 0, 'existing_exposed_train_models': 8,
                'new_train_forward_cap': sum(r.get('teacher_bound', {}).get('forward_cap', 0)
                                            for r in statuses if r['split'] == 'train'),
                'scope': 'analytic construction only; no filtering/retry/teacher labels; evaluation models excluded from teacher search'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    import importlib.util
    import subprocess
    import sys
    p = argparse.ArgumentParser()
    for key in ('generator-repo', 'protocol', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    paths = ('experiments/oracle/toys/zoo.py', 'experiments/oracle/toys/planted.py',
             'method/vpd/vpd_model.py')
    binding = {}
    for relative in paths:
        path = args.generator_repo / relative
        expected = subprocess.check_output(['git', '-C', str(args.generator_repo), 'show',
                                            GENERATOR_COMMIT+':'+relative])
        if path.read_bytes() != expected:
            raise ValueError('generator source differs: '+relative)
        binding[relative] = sha(path)
    sys.path.insert(0, str(args.generator_repo / 'experiments/oracle/toys'))
    spec = importlib.util.spec_from_file_location('zoo', args.generator_repo / paths[0])
    zoo = importlib.util.module_from_spec(spec); spec.loader.exec_module(zoo)
    def forbidden(*_args, **_kwargs):
        raise RuntimeError('Native/check/gold operation forbidden in offline bank')
    for name in ('record', 'build', 'check', 'gt_nodes', 'last_logits', 'node_kl'):
        setattr(zoo, name, forbidden)
    zoo.planted.native_of = forbidden
    generate(zoo, args.protocol, args.out, binding)


if __name__ == '__main__':
    main()
