"""Freeze evaluation-only panels from already constructed outcome-blind assets.

No generation, labels, Native, teacher search or model forwards. Tensor reads only
identify architecture edits; prompt and weight bytes are copied unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from learning_model_bank import FAMILIES, tensor_digest
from intervention_teacher_batch_challenge import joint_pairs, fixed_edits

ARMS = ['base', 'point24', 'intervention24', 'point35', 'intervention35']
EXECUTOR = 'cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4'
SOURCE_FILES = ['experiments/oracle/toyloop/circuit.py', 'experiments/oracle/toys/planted.py',
                'method/oracle/native.py', 'method/oracle/native_units.py',
                'method/oracle/goal_eval.py', 'method/oracle/carry.py',
                'method/oracle/mech.py', 'method/vpd/vpd_model.py']
SUPPORT_FILES = ['experiments/oracle/toyloop/'+name for name in
                 ('head_counterfactual.py', 'reward_interventions.py', 'intervention_teacher.py')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications():
    return [(f'{family}_{3400000+100*i}', family, 3400000+100*i)
            for i, family in enumerate(FAMILIES)]


def validate_manifest(manifest):
    expected = specifications()
    if manifest['generator_commit'] != EXECUTOR or manifest['native_forwards'] != 0 or manifest['teacher_forwards'] != 0:
        raise RuntimeError('outcome-blind pinned construction required')
    actual = [(r['id'], r['family'], r['seed']) for r in manifest['models']]
    if actual != expected or any(r['role'] != 'new_model_evaluation' for r in manifest['models']):
        raise RuntimeError('all eight fixed evaluation models/order required')


def validate_record(record, ident, family, seed):
    if (record['id'], record['family'], record['seed'], record['role']) != (ident, family, seed, 'new_model_evaluation'):
        raise RuntimeError('evaluation record identity differs')
    examples = record['examples']
    if len(examples) != 4 or any(e['kind'] != 'ordinary' or e['target'] != len(e['ids'])-1 for e in examples):
        raise RuntimeError('four frozen ordinary last-position inputs required')
    if any(not e['ids'] or any(type(i) is not int or not 0 <= i < 32 for i in e['ids']) for e in examples):
        raise RuntimeError('invalid frozen input IDs')
    forbidden = {'teacher', 'truth', 'unit_edges', 'context_units', 'predictions', 'model_top', 'intended', 'answers'}
    if forbidden & set(record) or any(forbidden & set(e) for e in examples):
        raise RuntimeError('prediction/teacher labels forbidden in evaluation records')
    return [e['ids'] for e in examples]


def edits(ident, cfg, nonzero_neurons):
    units = [('head', layer, head) for layer in range(cfg['n_layer']) for head in range(cfg['n_head'])]
    units += [('neuron', layer, unit) for layer, unit in nonzero_neurons]
    return [{'kind': kind, 'operations': [list(op) for op in operations]}
            for kind, operations in fixed_edits(units, joint_pairs(ident, units))]


def frozen_sources(root, commit, files):
    result = {}
    for relative in files:
        path = root / relative
        expected = subprocess.check_output(['git', '-C', str(root), 'show', commit+':'+relative])
        if path.read_bytes() != expected:
            raise RuntimeError('pinned executor/support changed: '+relative)
        result[relative] = sha(path)
    return result


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def build(args):
    if sha(args.bank/'manifest.json') != args.bank_sha256 or args.out.exists():
        raise RuntimeError('frozen input bank and fresh output required')
    bank = json.loads((args.bank/'manifest.json').read_text())
    validate_manifest(bank)
    source = frozen_sources(args.executor, EXECUTOR, SOURCE_FILES)
    support = frozen_sources(args.support, args.support_commit, SUPPORT_FILES)
    args.out.mkdir(parents=True)
    (args.out/'assets').mkdir()
    statuses, questions, panel = [], [], []
    import torch
    torch.set_num_threads(1)
    for item in bank['models']:
        status = {'id': item['id'], 'status': 'preparation_error'}
        try:
            if item['status'] != 'constructed_unmeasured':
                raise RuntimeError('original construction did not complete; no substitution allowed')
            directory = args.bank/item['id']
            for name, digest in item['files'].items():
                if sha(directory/name) != digest:
                    raise RuntimeError('immutable evaluation asset changed')
            record = json.loads((directory/'record.json').read_text())
            inputs = validate_record(record, item['id'], item['family'], item['seed'])
            value = torch.load(directory/'weights.pt', map_location='cpu', weights_only=True)
            if value['cfg'] != record['cfg'] or {k:tensor_digest(v) for k,v in value['sd'].items()} != item['source_weights']:
                raise RuntimeError('source weight/configuration identity differs')
            neurons = [(layer, unit) for layer in range(value['cfg']['n_layer'])
                       for unit in range(value['sd'][f'h.{layer}.mlp.down_proj.weight'].shape[1])
                       if bool(value['sd'][f'h.{layer}.mlp.down_proj.weight'][:,unit].ne(0).any())]
            operations = edits(item['id'], value['cfg'], neurons)
            prompt = (directory/'prompt.txt').read_text()
            if hashlib.sha256(prompt.encode()).hexdigest() != item['files']['prompt.txt']:
                raise RuntimeError('prompt bytes changed during read')
            if '-> predicts' in prompt or 'Hint:' in prompt or 'Detail:' in prompt or 'Compiler output contract' in prompt:
                raise RuntimeError('new hints/predictions/detail/clarification not allowed')
            if not prompt.endswith('Do not include English or token annotations.'):
                raise RuntimeError('original graph-only training format required')
            relative = 'assets/'+item['id']+'.pt'
            shutil.copyfile(directory/'weights.pt', args.out/relative)
            questions.append({'id':item['id'], 'stratum':'fresh_model340', 'prompt':prompt,
                              'prompt_sha256':item['files']['prompt.txt']})
            panel.append({'id':item['id'], 'stratum':'fresh_model340',
                          'prompt_sha256':item['files']['prompt.txt'], 'answer_format':'graph_only',
                          'weights':relative, 'weights_sha256':item['files']['weights.pt'],
                          'inputs':inputs, 'input_roles':['disclosed']*4, 'edits':operations})
            status.update(status='prepared_unmeasured', worlds=4*len(operations),
                          nonzero_output_neurons=neurons, prompt_sha256=item['files']['prompt.txt'])
        except Exception as exc:
            status.update(error_type=type(exc).__name__, error=str(exc))
        statuses.append(status)
        write(args.out/'status.json', statuses)
    if len(panel) != 8:
        write(args.out/'complete.json', {'status':'failed', 'planned_cases':8, 'statuses':statuses,
                                       'native_forwards':0, 'teacher_forwards':0})
        raise RuntimeError('incomplete evaluation bank retained; no cases silently dropped')
    (args.out/'panel.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in questions))
    worlds = sum(len(r['inputs'])*len(r['edits']) for r in panel)
    manifest = {'schema_version':1, 'arms':ARMS, 'panel':panel, 'source_sha256':source,
                'support_sha256':support, 'forward_cap':7*worlds,
                'scope':'fresh known-family held-out-model evaluation; model-level replication; no novel-mechanism claim',
                'bank_sha256':args.bank_sha256, 'generator_commit':EXECUTOR,
                'context_rule':'four original ordinary draws at seed*7+3, intended answers discarded',
                'prompt_rule':'unchanged frozen original TRAIN input format, no detail/hints or output clarification',
                'panel_rule':'all architecture heads/nonzero dense downcolumns individual0/.5; six hash-selected pairs times four Cartesian scales'}
    write(args.out/'scoring_manifest.json', manifest)
    write(args.out/'complete.json', {'status':'prepared_unmeasured', 'cases':8, 'worlds':worlds,
          'candidate_worlds':5*worlds, 'forward_cap':7*worlds, 'native_forwards':0, 'teacher_forwards':0,
          'panel_sha256':sha(args.out/'panel.jsonl'), 'scoring_manifest_sha256':sha(args.out/'scoring_manifest.json'),
          'producer_source_sha256':sha(__file__), 'bank_sha256':args.bank_sha256,
          'renderer_sha256':bank['renderer_sha256'], 'statuses':statuses})
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('bank','executor','support','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--bank-sha256',required=True)
    parser.add_argument('--support-commit',required=True)
    result = build(parser.parse_args())
    print(json.dumps({'models':len(result['panel']), 'worlds':result['forward_cap']//7,
                      'forward_cap':result['forward_cap'], 'forwards':0}),flush=True)
