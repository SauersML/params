"""TRAIN-only finite reward-ordering audit; never tunes lambda or calls an oracle."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def graph_text(units,edges):
    return 'flowchart LR\n'+''.join('  '+u+'\n' for u in sorted(units))+''.join(f'  {a} --> {b}\n' for a,b in sorted(edges))


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','payload','reward-source','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    sys.path[:0]=[str(a.source/'experiments/oracle/toyloop'),str(a.source/'experiments/oracle/toys'),str(a.source/'method/oracle'),str(a.source/'method/vpd')]
    import torch
    import render
    import circuit
    import native
    import planted
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('checked_mdl_reward',a.reward_source)
    reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
    if a.out.exists():
        raise ValueError('do not overwrite fixed measurement')
    a.out.mkdir()
    (a.out/'source_provenance.json').write_text(json.dumps({
        'executor_commit':'cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4',
        'reward_source_sha256':hashlib.sha256(a.reward_source.read_bytes()).hexdigest(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'executor_files':{str(p.relative_to(a.source)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ['experiments/oracle/toyloop','experiments/oracle/toys','method/oracle','method/vpd']
                          for p in (a.source/folder).glob('*.py')},
    },indent=2)+'\n')
    plan=json.loads((a.payload/'protocol.json').read_text())
    for filename,expected in plan['files'].items():
        if hashlib.sha256((a.payload/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('TRAIN fixture binding changed')
    rows=[]
    for case in plan['cases']:
        root=a.payload/'cases'/case['id']
        if case['kind']=='ordinary':
            toy=render.from_zoo({'weights':case['id']+'.pt','record':case['id']+'.json'},root)
            if toy.meta['record']['split']!='train':
                raise ValueError('non-TRAIN ordinary case')
            target=(root/'target.txt').read_text(); texts=reward.texts_of(toy)
        else:
            blob=torch.load(root/(case['id']+'.pt'),map_location='cpu',weights_only=True)
            texts=json.loads((root/(case['id']+'.bank.json')).read_text())
            toy=render.Toy(case['id'],blob['sd'],blob['cfg'],['BOS']+[f't{i}' for i in range(1,32)],texts,{})
            target=(root/'target.txt').read_text()
        nat=planted.native_of(toy.sd,toy.cfg)
        parsed=circuit.parse(target)
        if not parsed.ok or parsed.bad:
            raise RuntimeError('frozen producer target is not a valid graph: '+case['id'])
        bound=copy.copy(toy);bound.meta=dict(toy.meta)
        bound.meta['decomposition']={name:(nat.target.site(name).U,nat.target.site(name).V,nat.units[name]) for name in nat.names}
        circuit.unit_graph(parsed,bound,texts[0])  # Producer/fixture failures are not oracle invalids.
        units={u for u in parsed.units if not u.startswith('tok') and u!='out'}
        edges={e for e in parsed.edges if not e[0].startswith('tok')}
        variants={'gold':graph_text(units,edges)}
        for unit in sorted(units):
            variants['remove_unit_'+unit]=graph_text(units-{unit},{e for e in edges if unit not in e})
        for w,r in sorted(edges):
            variants['remove_edge_'+w+'_'+r]=graph_text(units,edges-{(w,r)})
        owner={}
        for unit in units:
            for site,piece in circuit.unit_pieces(bound,unit):
                for pos in range(len(texts[0])):
                    owner[site,pos,piece]=unit
        parents,outs=native.all_edges(sorted(owner),[len(texts[0])-1])
        possible={(owner[w],owner[r]) for r,ws in parents.items() for w in ws if owner[w]!=owner[r]}
        possible|={(owner[w],'out') for w in outs}
        extras=sorted(possible-edges)
        # All eligible single additions, not outcome-selected; cap declared before execution.
        if len(extras)>plan['max_same_unit_extras']:
            raise ValueError('predeclared bounded panel exceeded; no selective truncation')
        for w,r in extras:
            variants['extra_edge_'+w+'_'+r]=graph_text(units,edges|{(w,r)})
        variants['same_units_complete_explicit']=graph_text(units,possible)
        variants['annotations_duplicates']=variants['gold']+' tokIgnored --> '+sorted(units)[0]+'\n'+''.join(f' {w} --> {r}\n' for w,r in sorted(edges))
        if case['kind']=='ordinary':
            for token in toy.meta['record']['distractor_units']:
                unit=re.sub(r'^(L\d+)\.([HN]\d+)$',r'\1\2',token)
                variants['extra_distractor_'+unit]=graph_text(units|{unit},edges|{(unit,'out')})
        (a.out/(case['id']+'.variants.json')).write_text(json.dumps({
            'case':case['id'],'variant_count':len(variants),'texts':texts,'variants':variants,
            'selected_before_any_fidelity_score':True,
        },indent=2)+'\n')
        for name,text in variants.items():
            row={'case':case['id'],'family':case['family'],'variant':name,'completion':text,'lambda':reward.WIRING_MDL_LAMBDA}
            try:
                measured=reward.score_wiring(toy,text,nat=nat,texts=texts,description=True)
                row.update(valid=True,score=measured,mdl=reward.wiring_mdl_value(measured),
                           fidelity=(sum(x['suff'] for x in measured['per_text'])/len(texts)+min(x['suff'] for x in measured['per_text']))/2,
                           cost=sum(x['description_cost'] for x in measured['per_text'])/len(texts))
            except ValueError as exc:
                row.update(valid=False,mdl=-1.,error=str(exc))
            rows.append(row)
            (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({'case':case['id'],'variants':len(variants)}),flush=True)
    comparison=[]
    for case in plan['cases']:
        local=[r for r in rows if r['case']==case['id']]
        gold=next(r for r in local if r['variant']=='gold')
        for row in local:
            comparison.append({'case':case['id'],'variant':row['variant'],'valid':row['valid'],'mdl_delta_from_gold':row['mdl']-gold['mdl'],
                               'fidelity_delta':row['fidelity']-gold['fidelity'] if row['valid'] and gold['valid'] else None,
                               'cost_delta':row['cost']-gold['cost'] if row['valid'] and gold['valid'] else None})
    report={'scope':'TRAIN-only diagnostic, all variants retained; no lambda tuning, no universal ordering guarantee or English simplicity certificate',
            'lambda':reward.WIRING_MDL_LAMBDA,'cases':len(plan['cases']),'variants':len(rows),'comparisons':comparison,
            'reward_sha256':hashlib.sha256(a.reward_source.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((a.payload/'protocol.json').read_bytes()).hexdigest(),
            'observability':'Reward audit uses compiled graphs independently of prompt omissions; any future RL must use verified all_live_neurons inputs. Frozen old evaluations unchanged.'}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':report['cases'],'variants':report['variants'],'lambda':report['lambda']}))


if __name__=='__main__':
    main()
