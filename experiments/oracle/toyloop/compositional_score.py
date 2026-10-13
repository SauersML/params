"""Frozen first-response physical scoring by explicit development/training stratum."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_valid(sample,parsed):
    stopped=str(sample['finish']).lower() in ('stop','stopreason.stop')
    return bool(parsed and parsed.ok and not parsed.bad and stopped and not sample.get('parse_error'))


def complete_panel(samples,panel):
    if len(samples)!=len(panel) or len({r['id'] for r in samples})!=len(samples) or [r['id'] for r in samples]!=[r['id'] for r in panel]:
        raise ValueError('missing/duplicate/reordered first responses')


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','payload','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    sys.path[:0]=[str(a.source/'experiments/oracle/toyloop'),str(a.source/'experiments/oracle/toys'),str(a.source/'method/oracle'),str(a.source/'method/vpd')]
    import torch
    import render
    import circuit
    import planted
    import reward_exec
    torch.set_num_threads(1)
    if a.out.exists():
        raise ValueError('never overwrite a measurement')
    a.out.mkdir()
    contract=json.loads((a.payload/'input_contract.json').read_text())
    for filename,value in contract['files'].items():
        if Path(filename).is_absolute() or '..' in Path(filename).parts or sha(a.payload/filename)!=value:
            raise ValueError('frozen input binding failed')
    receipt=json.loads((a.payload/'response_contract.json').read_text())
    for filename,value in receipt['files'].items():
        if sha(a.payload/filename)!=value:
            raise ValueError('first response receipt changed')
    sources=json.loads((a.payload/'original24_manifest.json').read_text())
    for filename,value in sources['sources'].items():
        if sha(a.source/filename)!=value:
            raise ValueError('original executor changed')
    samples={arm:rows(a.payload/(arm+'.jsonl')) for arm in ('single','two')}
    panel=[]
    for stratum in ('original24','old_development8','new8','training_incidentalV20'):
        for r in rows(a.payload/(stratum+'_questions.jsonl')):
            panel.append({**r,'case_id':r['id'],'stratum':stratum,'id':stratum+'/'+r['id']})
    for arm in samples:
        complete_panel(samples[arm],panel)
    byid={arm:{r['id']:r for r in samples[arm]} for arm in samples}
    oldbinding={r['id']:r for r in sources['panel']}
    checks={r['id']:r for r in json.loads((a.payload/'curriculum_verification.json').read_text())}
    old8={r['id']:r for r in json.loads((a.payload/'old8/manifest.json').read_text())['cases']}
    outputs={arm:[] for arm in samples}
    for r in panel:
        case=r['case_id']; stratum=r['stratum']
        if stratum=='original24':
            root=a.payload/'original24_cases'/case
            binding=oldbinding[case]
            toy=render.from_zoo({'weights':Path(binding['weights_source']).name,'record':Path(binding['record_source']).name},root)
            bank=json.loads((root/'bank.json').read_text())
            gold={tuple(e) for e in toy.truth['edges'] if all(re.fullmatch(r'L\d+[HN]\d+',n) for n in e)}
        elif stratum=='training_incidentalV20':
            root=a.payload/'negative20'
            toy=render.from_zoo({'weights':case+'.pt','record':case+'.json'},root)
            bank=reward_exec.texts_of(toy)
            gold={tuple(e) for e in toy.truth['edges'] if all(re.fullmatch(r'L\d+[HN]\d+',n) for n in e)}
        else:
            root=a.payload/('old8' if stratum=='old_development8' else 'new_instances')
            blob=torch.load(root/(case+'.pt'),map_location='cpu',weights_only=True)
            bank=json.loads((root/(case+'.bank.json')).read_text())
            edges=old8[case]['design']['correct'] if stratum=='old_development8' else checks[case]['edges']
            gold={tuple(e) for e in edges if all(re.fullmatch(r'L\d+[HN]\d+',n) for n in e)}
            toy=SimpleNamespace(sd=blob['sd'],cfg=blob['cfg'],meta={},texts=bank,vocab=['BOS']+[f't{i}' for i in range(1,32)])
        (a.out/(stratum+'_'+case+'.bank.json')).write_text(json.dumps(bank)+'\n')
        nat=planted.native_of(toy.sd,toy.cfg)
        for arm in samples:
            s=byid[arm][r['id']]
            if s['prompt_sha256']!=r['prompt_sha256'] or s['sample']!=0 or s['case_id']!=case or s['stratum']!=stratum:
                raise ValueError('response prompt/case identity drift')
            text=s.get('completion')
            parsed=circuit.parse(text) if isinstance(text,str) else None
            valid=saved_valid(s,parsed)
            edges={e for e in (parsed.edges if parsed else []) if all(re.fullmatch(r'L\d+[HN]\d+',n) for n in e)}
            row={**s,'valid':valid,'parse_bad':parsed.bad if parsed else [],'declared_component_edges':sorted(edges),'gold_component_edges':sorted(gold),'bank_n':len(bank),'bank_sha256':sha(a.out/(stratum+'_'+case+'.bank.json'))}
            if valid:
                try:
                    for complete,label in ((False,'stated'),(True,'same_units_complete')):
                        measured=reward_exec.score_wiring(toy,text,nat=nat,texts=bank,complete=complete)
                        if any(not math.isfinite(x['kl_bits']) for x in measured['per_text']):
                            raise RuntimeError('nonfinite scoring; not a candidate parse failure')
                        values=[x['suff'] for x in measured['per_text']]
                        row[label]=measured
                        row[label+'_reward']=(sum(values)/len(values)+min(values))/2
                except ValueError as exc:
                    row.update(valid=False,execution_error=str(exc))
            if not row['valid']:
                row['stated_reward']=row['same_units_complete_reward']=-1.
            correct=len(edges&gold)
            row['component_edge_f1']=(2*correct/(len(edges)+len(gold)) if edges or gold else 1.) if row['valid'] else 0.
            row['matching_exact']=row['valid'] and edges==gold
            row['executed_component_edges']=sorted(edges) if row['valid'] else []
            outputs[arm].append(row)
            with (a.out/(arm+'.jsonl')).open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({'arm':arm,'id':r['id'],'valid':row['valid'],'reward':row['stated_reward'],'matching':row['matching_exact']}),flush=True)
        del nat
    summary={'scope':'Fixed development/training strata; no pristine holdout or English claim. Invalid=-1 reward, zero F1/top; undefined KL retained as coverage.', 'strata':{}}
    for stratum in ('original24','old_development8','new8','training_incidentalV20'):
        summary['strata'][stratum]={}
        for arm,allrows in outputs.items():
            selected=[r for r in allrows if r['stratum']==stratum]
            measured=[x for r in selected if r['valid'] for x in r['stated']['per_text']]
            total=sum(r['bank_n'] for r in selected)
            summary['strata'][stratum][arm]={'cases':len(selected),'valid':sum(r['valid'] for r in selected),'reward':sum(r['stated_reward'] for r in selected)/len(selected),'edge_f1':sum(r['component_edge_f1'] for r in selected)/len(selected),'matching_exact':sum(r['matching_exact'] for r in selected),'top_correct':sum(x['top']==x['reference_top'] for x in measured),'top_denominator':total,'measured_KL_inputs':len(measured),'mean_KL_valid':sum(x['kl_bits'] for x in measured)/len(measured) if measured else None,'false_interhead_edges':sum(len(set(map(tuple,r['executed_component_edges']))-set(map(tuple,r['gold_component_edges']))) for r in selected)}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
