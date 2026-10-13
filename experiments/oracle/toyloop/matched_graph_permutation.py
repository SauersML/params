"""Opt-in physical unit-renaming control; no oracle calls or teacher selection.

Head Q/K/V rows and O columns move together; FC rows and down columns move
 together. Residual coordinates, embeddings, norms, input IDs and scales stay
fixed. Old-unit-to-new-unit bijections map every original individual/joint edit.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import random
import math
import sys

SEED = 2026101307


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def permutations(cfg, widths):
    rng = random.Random(SEED)
    result = {}
    for layer in range(cfg['n_layer']):
        for kind, count in (('head', cfg['n_head']), ('neuron', widths[layer])):
            if count < 2:
                raise ValueError('nonidentity permutation requires at least two units')
            order = list(range(count))
            rng.shuffle(order)
            if order == list(range(count)):
                order = order[1:] + order[:1]
            # order[new] = old; explicit inverse maps intervention identifiers.
            result[f'{kind}:{layer}'] = {'new_to_old': order,
                                        'old_to_new': [order.index(i) for i in range(count)]}
    return result


def map_edit(edit, maps):
    return {'kind': edit['kind'], 'operations': [
        [kind, layer, maps[f'{kind}:{layer}']['old_to_new'][unit], scale]
        for kind, layer, unit, scale in edit['operations']]}


def permute_weights(sd, cfg, maps):
    import torch
    result = {name: value.clone() for name, value in sd.items()}
    hd = cfg['n_embd'] // cfg['n_head']
    if hd * cfg['n_head'] != cfg['n_embd']:
        raise ValueError('integral head width required')
    expected = set()
    for layer in range(cfg['n_layer']):
        order = maps[f'head:{layer}']['new_to_old']
        rows = torch.tensor([h * hd + j for h in order for j in range(hd)])
        for port in ('q_proj', 'k_proj', 'v_proj'):
            name = f'h.{layer}.attn.{port}.weight'; expected.add(name)
            result[name] = sd[name].index_select(0, rows)
        name = f'h.{layer}.attn.o_proj.weight'; expected.add(name)
        result[name] = sd[name].index_select(1, rows)
        neurons = torch.tensor(maps[f'neuron:{layer}']['new_to_old'])
        for port, axis in (('c_fc', 0), ('down_proj', 1)):
            name = f'h.{layer}.mlp.{port}.weight'; expected.add(name)
            result[name] = sd[name].index_select(axis, neurons)
    # This pinned toy architecture has no biases; fail closed if future weights
    # introduce unit-indexed parameters requiring additional transport.
    if any(name.endswith('.bias') for name in sd):
        raise ValueError('bias transport unsupported in frozen toy contract')
    for name in set(sd) - expected:
        if not torch.equal(result[name], sd[name]):
            raise ValueError('non-unit tensor changed')
    return result


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def run(args):
    if (sha(args.manifest) != args.manifest_sha256 or sha(args.provenance) != args.provenance_sha256
            or sha(args.source_panel) != args.source_panel_sha256 or args.out.exists()):
        raise ValueError('frozen manifest and fresh output required')
    manifest = json.loads(args.manifest.read_text())
    panel = [r for r in manifest['panel'] if r['stratum'] == 'new_model']
    if len(panel) != 8 or sum(len(r['inputs']) * len(r['edits']) for r in panel) != 1496:
        raise ValueError('exact eight-model1496-world panel required')
    for root, field in ((args.source, 'source_sha256'), (args.support, 'support_sha256')):
        for name, digest in manifest[field].items():
            if sha(root / name) != digest:
                raise ValueError('frozen executor/support changed')
    sys.path[:0] = [str(args.source / p) for p in ('experiments/oracle/toyloop', 'experiments/oracle/toys', 'method/oracle', 'method/vpd')]
    import torch
    import planted
    sys.path.insert(0, str(args.support / 'experiments/oracle/toyloop'))
    import render
    import assemble_graph_supervision as assembly
    from head_counterfactual import scaled_head, raw_kl_bits
    from reward_interventions import scaled_neuron, clear_intervention_caches
    from safetensors.torch import save_file
    provenance = json.loads(args.provenance.read_text())
    if sha(render.__file__) != provenance['helper_sha256']['render.py']:
        raise ValueError('frozen weight renderer changed')
    source_prompts = {r['id']: r for r in [json.loads(x) for x in args.source_panel.read_text().splitlines()]}
    torch.set_num_threads(1)
    args.out.mkdir()
    plans = []
    models = []
    # Freeze all mappings and weight/input bindings before the first forward.
    for row in panel:
        path = args.payload / row['weights']
        if sha(path) != row['weights_sha256']:
            raise ValueError('original weight file changed')
        blob = torch.load(path, weights_only=True, map_location='cpu')
        widths = [blob['sd'][f'h.{l}.mlp.down_proj.weight'].shape[1] for l in range(blob['cfg']['n_layer'])]
        maps = permutations(blob['cfg'], widths)
        permuted = permute_weights(blob['sd'], blob['cfg'], maps)
        plans.append({'id': row['id'], 'weights_sha256': row['weights_sha256'],
                      'inputs': row['inputs'], 'maps': maps, 'edits': row['edits'],
                      'mapped_edits': [map_edit(e, maps) for e in row['edits']]})
        models.append((row, blob, permuted))
    write(args.out / 'protocol.json', {'manifest_sha256': args.manifest_sha256,
        'provenance_sha256': args.provenance_sha256, 'source_panel_sha256': args.source_panel_sha256,
        'seed': SEED, 'plans': plans, 'dense_forward_cap': 2992,
        'driver_sha256': sha(__file__), 'render_sha256': sha(render.__file__),
        'assembly_sha256': sha(assembly.__file__), 'scope': 'unit-name equivariance; not new mechanism transfer'})
    rows = []; forwards = 0
    for case_index, ((row, blob, permuted), plan) in enumerate(zip(models, plans)):
        nats = [planted.native_of(sd, blob['cfg']) for sd in (blob['sd'], permuted)]
        def state(nat):
            return {name: hashlib.sha256(nat.target.site(name).W.detach().contiguous().numpy().tobytes()).hexdigest() for name in nat.names}
        originals = [state(n) for n in nats]
        vocab = ['BOS'] + [f't{i}' for i in range(1, blob['sd']['wte.weight'].shape[0])]
        original_toy = render.Toy(row['id'], blob['sd'], blob['cfg'], vocab, row['inputs'], None, {})
        original_prompt = assembly.source_prompt(original_toy, row['inputs'], render)
        if (original_prompt != source_prompts[row['id']]['prompt']
                or hashlib.sha256(original_prompt.encode()).hexdigest() != row['prompt_sha256']):
            raise ValueError('original prompt reconstruction changed')
        toy = render.Toy(row['id'], permuted, blob['cfg'], vocab, row['inputs'], None, {})
        prompt = assembly.source_prompt(toy, row['inputs'], render)
        torch.save({'sd': permuted, 'cfg': blob['cfg']}, args.out / (row['id'] + '.pt'))
        (args.out / (row['id'] + '.prompt.txt')).write_text(prompt)
        for edit_index, (edit, mapped) in enumerate(zip(row['edits'], plan['mapped_edits'])):
            with contextlib.ExitStack() as stack:
                for nat, current in zip(nats, (edit, mapped)):
                    for kind, layer, unit, scale in current['operations']:
                        stack.enter_context((scaled_head if kind == 'head' else scaled_neuron)(nat, layer, unit, scale))
                for index, ids in enumerate(row['inputs']):
                    values = []
                    for nat in nats:
                        clear_intervention_caches(nat)
                        values.append(nat.reference([ids], [len(ids)-1]).detach().clone()); forwards += 1
                    p, q = values
                    kl = raw_kl_bits(p, q); error = float((p-q).abs().max())
                    top = int(p[0,-1].argmax()) == int(q[0,-1].argmax())
                    name = f'{case_index}_{edit_index}_{index}.safetensors'
                    save_file({'original': p.contiguous(), 'permuted': q.contiguous()}, str(args.out / name))
                    passed = top and math.isfinite(kl) and math.isfinite(error) and abs(kl) <= 1e-5 and error <= 1e-4
                    result = {'passes': passed, 'case': row['id'], 'edit_index': edit_index, 'input': index,
                              'kind': edit['kind'], 'raw_kl_bits': kl, 'max_abs_logp': error,
                              'top_agrees': top, 'file': name, 'sha256': sha(args.out / name)}
                    rows.append(result)
                    with (args.out / 'results.jsonl').open('a') as f:
                        f.write(json.dumps(result, allow_nan=False)+'\n'); f.flush()
            if [state(n) for n in nats] != originals:
                raise ValueError('literal edit failed restoration')
        write(args.out / f'{case_index}.restoration.json', {'original': originals, 'restored': [state(n) for n in nats]})
        print(json.dumps({'case': row['id'], 'forwards': forwards}), flush=True)
    if forwards != 2992 or len(rows) != 1496:
        raise ValueError('Cartesian coverage mismatch')
    write(args.out / 'complete.json', {'forwards': forwards, 'worlds': len(rows),
        'worst_abs_kl_bits': max(abs(r['raw_kl_bits']) for r in rows),
        'all_commutation_pass': all(r['passes'] for r in rows),
        'failed_worlds': sum(not r['passes'] for r in rows),
        'max_abs_logp': max(r['max_abs_logp'] for r in rows), 'all_top_agree': all(r['top_agrees'] for r in rows)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('manifest', 'payload', 'source', 'support', 'out', 'provenance', 'source_panel'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--provenance-sha256', required=True)
    p.add_argument('--source-panel-sha256', required=True)
    run(p.parse_args())
