"""Opt-in fresh parallel routes: finite teacher checks, no oracle/service calls.

The complete four-bridge control can be faithful; its excess edges violate only
an explicitly separate two-bridge complexity budget. Embeddings remain on.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

SEEDS = (2300000, 2300001)

def design(seed, swapped=False, permuted=False):
    if seed not in SEEDS or type(swapped) is not bool or type(permuted) is not bool:
        raise ValueError('outside frozen design')
    rng = random.Random(seed)
    writers = rng.sample(range(4), 2)
    readers = rng.sample(range(4), 2)
    pool = rng.sample(range(1, 32), 6)
    if permuted:
        writers = [(h + 1) % 4 for h in writers]
        readers = [(h + 2) % 4 for h in readers]
    bank = []
    for k in range(8):
        ids = [0] + [rng.choice(pool) for _ in range(11 + k % 3)]
        # Both candidate matchings and the lexical root see different tokens.
        critical = rng.sample(pool, 5)
        for offset, token in zip((0, 2, 5, 4, 3), critical):
            ids[-1-offset] = token
        bank.append(ids)
    matching = (1, 0) if swapped else (0, 1)
    edge = lambda i, j: [f'L0H{writers[i]}', f'L1H{readers[j]}']
    outputs = [[f'L1H{h}', 'out'] for h in readers]
    return dict(seed=seed, swapped=swapped, permuted=permuted, writers=writers,
                readers=readers, pool=pool, bank=bank,
                correct=[edge(matching[j], j) for j in range(2)] + outputs,
                wrong=[edge(1-matching[j], j) for j in range(2)] + outputs,
                complete=[edge(i,j) for i in range(2) for j in range(2)] + outputs)

def load_renderer(path):
    spec = importlib.util.spec_from_file_location('fresh_route_renderer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def build(source, renderer_path, out):
    if out.exists():
        raise ValueError('output must be fresh')
    out.mkdir(parents=True)
    sys.path[:0] = [str(source/'experiments/oracle/toyloop'), str(source/'experiments/oracle/toys'),
                   str(source/'method/oracle'), str(source/'method/vpd')]
    import torch, zoo, planted, reward_exec
    torch.set_num_threads(1)
    render = load_renderer(renderer_path)
    report = {'scope': 'fresh composed V routes; finite frozen-bank sufficiency, not general VPD certificate',
              'complexity_budget': 2, 'cases': [], 'contrasts': [],
              'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), renderer_path, source/'experiments/oracle/toys/zoo.py',
                           source/'experiments/oracle/toys/planted.py', source/'experiments/oracle/toyloop/reward_exec.py')}}
    references = {}
    def persist():
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
    for seed in SEEDS:
        for swapped in (False, True):
            for permuted in (False, True):
                cfg = design(seed, swapped, permuted)
                name = f'parallel_{seed}_{int(swapped)}_{int(permuted)}'
                b = zoo.Builder()
                for i,h in enumerate(cfg['writers']):
                    b.positional(0,h,(1,3)[i])
                    b.ov(0,h,[({zoo.tok(t):1.}, {33+i*6+c:1.}) for c,t in enumerate(cfg['pool'])])
                for j,h in enumerate(cfg['readers']):
                    i = (1-j) if swapped else j
                    b.positional(1,h,(1,2)[j])
                    b.ov(1,h,[({33+i*6+c:1.}, {zoo.tok(t):(3.,2.9)[j]}) for c,t in enumerate(cfg['pool'])])
                truth = {'edges': cfg['correct'], 'english': 'Each reader receives its matching scratch-bank writer. The two independent positional paths preserve their token identities.'}
                toy = render.Toy(name,b.sd,zoo.CFG,['BOS']+[f't{i}' for i in range(1,32)],cfg['bank'],truth)
                prompt = render.render(toy,all_live_neurons=True) + '\nUse at most two inter-head arrows; retain English. This is a separate compactness requirement.\n'
                (out/f'{name}.prompt.txt').write_text(prompt)
                torch.save({'sd':b.sd,'cfg':zoo.CFG},out/f'{name}.pt')
                (out/f'{name}.bank.json').write_text(json.dumps(cfg['bank']))
                nat = planted.native_of(b.sd,zoo.CFG)
                scores = {}
                for label in ('correct','wrong','complete'):
                    completion = render.target_text({'edges':cfg[label], 'english':truth['english']})
                    (out/f'{name}.{label}.txt').write_text(completion)
                    scores[label] = reward_exec.score_wiring(toy,completion,nat=nat,texts=cfg['bank'])
                with torch.no_grad():
                    ref = [nat.reference([ids],[len(ids)-1]).detach().cpu() for ids in cfg['bank']]
                references[seed,swapped,permuted] = ref
                item = {'id':name,'design':cfg,'scores':scores,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob(name+'.*')}}
                report['cases'].append(item); persist()
                if max(r['kl_bits'] for r in scores['correct']['per_text']) > 1e-3:
                    raise ValueError(f'{name}: correct route fails teacher integrity')
                if min(r['kl_bits'] for r in scores['wrong']['per_text']) < 1.:
                    raise ValueError(f'{name}: wrong route insufficiently discriminated')
                if max(r['kl_bits'] for r in scores['complete']['per_text']) > 1e-3:
                    raise ValueError(f'{name}: complete route fails integrity')
    for seed in SEEDS:
        for permuted in (False,True):
            a,b = references[seed,False,permuted],references[seed,True,permuted]
            kl = [max(0.,float((x.exp()*(x-y)).sum())/0.6931471805599453) for x,y in zip(a,b)]
            report['contrasts'].append({'seed':seed,'permuted':permuted,'diagonal_vs_swapped_kl_bits':kl}); persist()
            if min(kl) < 1.:
                raise ValueError('matching changes must produce distinct distributions on every frozen input')
        for swapped in (False,True):
            error = max(float((x-y).abs().max()) for x,y in zip(references[seed,swapped,False],references[seed,swapped,True]))
            report['contrasts'].append({'seed':seed,'swapped':swapped,'head_permutation_max_abs_logp':error}); persist()
            if error > 1e-3:
                raise ValueError('head permutation changes full model')
    report['teacher_integrity'] = 'PASS'; persist()

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--renderer',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); build(a.source,a.renderer,a.out)
