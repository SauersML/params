"""Fresh, matched single/two-V-path curriculum; finite verification, no oracle calls.

Original256 bytes are retained. The new32 supports are graph-only, with the
same output instructions in both arms. Old8 development cases are never used.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import types

TRAIN_SEEDS = tuple(range(2400000, 2400008))
TEST_SEEDS = (2500000, 2500001)
OLD_SEEDS = (2300000, 2300001)
CORE_SHA = 'd7f0177bcf50935fe1e676a6d31478715206b3c0b7a5cdd8a0a3ecb8452d3ad6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def designs(module, seeds):
    """Reuse exact public design algorithm in isolated globals; frozen SEEDS unchanged."""
    if set(seeds) - set(TRAIN_SEEDS + TEST_SEEDS) or set(seeds) & set(OLD_SEEDS):
        raise ValueError('outside predeclared disjoint curriculum')
    fn = types.FunctionType(module.design.__code__, {**module.__dict__, 'SEEDS': TRAIN_SEEDS + TEST_SEEDS})
    return [fn(seed, swapped, permuted) for seed in seeds for swapped in (False, True) for permuted in (False, True)]


def completion(edges):
    return 'flowchart LR\n' + ''.join(f'  {a} --> {b}\n' for a, b in edges)


def graph_instruction(prompt):
    prefix = prompt.split('Which units implement the prediction', 1)[0]
    return prefix + 'Which units implement the prediction at the last position?\nReturn only a Mermaid flowchart over L<i>H<j> and out; include every required inter-head relation. No English section.\n'


def build(source, renderer_path, core_path, out):
    if out.exists():
        raise ValueError('immutable output already exists')
    out.mkdir(parents=True)
    sys.path[:0] = [str(source/'experiments/oracle/toyloop'), str(source/'experiments/oracle/toys'),
                   str(source/'method/oracle'), str(source/'method/vpd')]
    import torch
    import zoo
    import planted
    import reward_exec
    from head_counterfactual import scaled_head, raw_kl_bits
    spec = importlib.util.spec_from_file_location('curriculum_route_design', Path(__file__).with_name('route_transfer.py'))
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    render = rt.load_renderer(renderer_path)
    torch.set_num_threads(1)
    core = core_path.read_bytes()
    if sha(core_path) != CORE_SHA:
        raise ValueError('original256 bytes changed')
    core_rows = [json.loads(x) for x in core.splitlines()]
    if len(core_rows) != 256 or len({r['id'] for r in core_rows}) != 256:
        raise ValueError('expected exact original256 rows')
    plan = {'train_seeds': TRAIN_SEEDS, 'test_seeds': TEST_SEEDS, 'old_development_seeds': OLD_SEEDS,
            'train_designs': designs(rt, TRAIN_SEEDS), 'test_designs': designs(rt, TEST_SEEDS),
            'core_sha256': sha(core_path), 'core_rows':256, 'rows_per_arm':288, 'epochs':2,
            'batch':8, 'updates':72, 'visits':576, 'relation_multiplier':8,
            'augmentation_target':'graph only; original256 prompts/English targets byte-identical',
            'augmentation_loss_mass':'Both graph-only arms use common per-example mass equal to frozen original256 median supervised-token mass; core relation8 preserves each original mass. Fit records exact mass before any training.',
            'single_control':'one route constructed; retained branch alternates by seed/swap/permutation',
            'single_wrong_control':'same selected heads with bridge absent; complete is identical one legal bridge',
            'evaluation':'new8 frozen before validation; old8 development, original24 and20 core incidental-V negatives retained as separate evaluation strata; no refit or output selection',
            'integrity_max_kl_bits':1e-3, 'wrong_min_kl_bits':1., 'route_damage_min_kl_bits':.01,
            'scope':'Fresh frozen development comparison, finite bank verification; no universal certificate or new English labels',
            'sources':{str(p):sha(p) for p in [Path(__file__),Path(rt.__file__),renderer_path,source/'experiments/oracle/toys/zoo.py',source/'experiments/oracle/toys/planted.py',source/'experiments/oracle/toyloop/reward_exec.py']}}
    (out/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    negatives=[r for r in core_rows if not r['id'].startswith('factorial_multihop_') and any(float(g)>0 for g in re.findall(r'V<L\d+H\d+:([0-9.eE+\-]+)',r['prompt']))]
    if len(negatives)!=20:
        raise ValueError('expected frozen20 incidental-V training negatives')
    (out/'incidental_V_train_negatives.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in negatives))
    gold_rows = {'single':[], 'two':[], 'holdout':[]}
    all_checks = []
    (out/'instances').mkdir()
    for split,seeds in [('train',TRAIN_SEEDS),('holdout',TEST_SEEDS)]:
        for cfg in designs(rt,seeds):
            variants = ['two','single'] if split=='train' else ['holdout']
            for variant in variants:
                name=f"compose_{cfg['seed']}_{int(cfg['swapped'])}_{int(cfg['permuted'])}_{variant}"
                branch=(cfg['seed']+int(cfg['swapped'])+int(cfg['permuted']))%2
                retained=[branch] if variant=='single' else [0,1]
                b=zoo.Builder()
                matching=(1,0) if cfg['swapped'] else (0,1)
                writer_ids={matching[j] for j in retained}
                for i in sorted(writer_ids):
                    h=cfg['writers'][i]
                    b.positional(0,h,(1,3)[i])
                    b.ov(0,h,[({zoo.tok(t):1.},{33+i*6+c:1.}) for c,t in enumerate(cfg['pool'])])
                for j in retained:
                    h=cfg['readers'][j]; i=matching[j]
                    b.positional(1,h,(1,2)[j])
                    b.ov(1,h,[({33+i*6+c:1.},{zoo.tok(t):(3.,2.9)[j]}) for c,t in enumerate(cfg['pool'])])
                edges=[cfg['correct'][j] for j in retained]+[[f"L1H{cfg['readers'][j]}",'out'] for j in retained]
                toy=render.Toy(name,b.sd,zoo.CFG,['BOS']+[f't{i}' for i in range(1,32)],cfg['bank'],{'edges':edges,'english':''})
                prompt=graph_instruction(render.render(toy,all_live_neurons=True))
                target=completion(edges)
                nat=planted.native_of(b.sd,zoo.CFG)
                scores={'gold':reward_exec.score_wiring(toy,target,nat=nat,texts=cfg['bank'])}
                if variant!='single':
                    for label in ('wrong','complete'):
                        scores[label]=reward_exec.score_wiring(toy,completion(cfg[label]),nat=nat,texts=cfg['bank'])
                else:
                    writer,reader=edges[0]
                    scores['wrong']=reward_exec.score_wiring(toy,completion([[writer,'out'],[reader,'out']]),nat=nat,texts=cfg['bank'])
                    scores['complete']=reward_exec.score_wiring(toy,target,nat=nat,texts=cfg['bank'],complete=True)
                references=[nat.reference([ids],[len(ids)-1]).detach().clone() for ids in cfg['bank']]
                relevance=[]
                for unit in sorted({u for edge in edges for u in edge if u!='out'}):
                    layer,head=map(int,(unit.split('H')[0][1:],unit.split('H')[1]))
                    with scaled_head(nat,layer,head,0.):
                        edited=[nat.reference([ids],[len(ids)-1]).detach().clone() for ids in cfg['bank']]
                        damage=[raw_kl_bits(p,q) for p,q in zip(references,edited)]
                    restored=[nat.reference([ids],[len(ids)-1]).detach().clone() for ids in cfg['bank']]
                    if not all(torch.equal(p,q) for p,q in zip(references,restored)):
                        raise ValueError('failed exact reference restoration')
                    relevance.append({'unit':unit,'kl_bits':damage})
                stem=out/'instances'/name
                torch.save({'sd':b.sd,'cfg':zoo.CFG},str(stem)+'.pt')
                Path(str(stem)+'.bank.json').write_text(json.dumps(cfg['bank']))
                Path(str(stem)+'.prompt.txt').write_text(prompt)
                Path(str(stem)+'.target.txt').write_text(target)
                check={'id':name,'split':split,'variant':variant,'edges':edges,'scores':scores,'relevance':relevance,
                       'files':{p.name:sha(p) for p in (out/'instances').glob(name+'.*')}}
                all_checks.append(check)
                (out/'verification.json').write_text(json.dumps(all_checks,indent=2)+'\n')
                if not all(math.isfinite(r['kl_bits']) for score in scores.values() for r in score['per_text']):
                    raise ValueError(name+': nonfinite score')
                if max(r['kl_bits'] for r in scores['gold']['per_text'])>1e-3:
                    raise ValueError(name+': gold mismatch')
                if max(r['kl_bits'] for r in scores['complete']['per_text'])>1e-3 or min(r['kl_bits'] for r in scores['wrong']['per_text'])<1.:
                    raise ValueError(name+': same-unit contrasts failed')
                if any(min(r['kl_bits'])<.01 for r in relevance):
                    raise ValueError(name+': selected route not relevant on every frozen input')
                gold_rows[variant].append({'id':name,'prompt':prompt,'completion':target})
    for name, rows in gold_rows.items():
        if len(rows)!=(8 if name=='holdout' else 32):
            raise ValueError('incomplete fixed panel')
        suffix=b''.join((json.dumps(r)+'\n').encode() for r in rows)
        payload=suffix if name=='holdout' else core+(b'' if core.endswith(b'\n') else b'\n')+suffix
        (out/(name+'_inputs.jsonl')).write_bytes(payload)
    manifest={**plan,'complete':True,'checks':len(all_checks),'datasets':{name:sha(out/(name+'_inputs.jsonl')) for name in gold_rows}}
    (out/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'checks':len(all_checks),'datasets':manifest['datasets']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','renderer','core','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.renderer,args.core,args.out)
