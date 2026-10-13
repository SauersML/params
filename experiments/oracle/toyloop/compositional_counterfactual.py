"""Frozen new-eight compositional head edits; no generation or training.

Original unit identities/U stay fixed. Literal dense O columns and matching V
input rows/remainder rows scale together; V changes represent the same original
basis under edited weights, never a new decomposition. Graphs compile before
edits. Candidates receive no dense boundary, frozen attention, or reused norms.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import head_counterfactual as h

ARMS=('single','two')
HELPER_SHA='f446fd763d0365f7ed8370823241c447ecefd6c86319cb6370f6df67a26c27a0'
EXECUTOR='cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4'
CASES=tuple(f'compose_{seed}_{swap}_{perm}_holdout' for seed in (2500000,2500001) for swap in (0,1) for perm in (0,1))

def verify_files(root,files):
    for name,expected in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or h.digest(root/name)!=expected:
            raise ValueError('frozen file binding failed: '+name)

def panel(payload):
    inputs=json.loads((payload/'input_contract.json').read_text())
    receipts=json.loads((payload/'response_contract.json').read_text())
    if inputs['source']!=EXECUTOR or 'original24_manifest.json' not in inputs['files']:
        raise ValueError('bound original executor manifest required')
    verify_files(payload,inputs['files']);verify_files(payload,receipts['files'])
    questions=h.rows(payload/'new8_questions.jsonl')
    if tuple(row['id'] for row in questions)!=CASES: raise ValueError('exact predeclared new-eight order required')
    samples={}
    for arm in ARMS:
        all_rows=h.rows(payload/(arm+'.jsonl'))
        if len(all_rows)!=60 or len({r['id'] for r in all_rows})!=60:
            raise ValueError('complete60-row response receipt required')
        chosen=[r for r in all_rows if r['stratum']=='new8']
        if [r['id'] for r in chosen]!=['new8/'+name for name in CASES]: raise ValueError('new-eight response panel differs')
        for q,r in zip(questions,chosen):
            if r['case_id']!=q['id'] or r['sample']!=0 or r['prompt_sha256']!=q['prompt_sha256']:
                raise ValueError('first response prompt/case binding changed')
        samples[arm]={r['case_id']:r for r in chosen}
    return inputs,questions,samples

def clear(nat):
    h.clear_native(nat)
    # Even mask/layout caches are cleared. No Engine/prepacked factors are used.
    getattr(nat,'_consts',{}).clear();getattr(nat,'_lengths',{}).clear()
    for name in nat.names: nat.target.site(name).__dict__.pop('gathered',None)

def verify_grid(draws,baselines,restores):
    expected={(case,i,l,head,scale) for case in CASES for i in range(8) for l in range(2) for head in range(4) for scale in (0.,.5)}
    keys=[(r['case'],r['input'],r['edit']['layer'],r['edit']['head'],r['edit']['scale']) for r in draws]
    baseline_keys=[(r['case'],r['input']) for r in baselines]
    restore_keys=[(r['case'],r['layer'],r['head'],r['scale']) for r in restores]
    if len(keys)!=1024 or set(keys)!=expected or len(baseline_keys)!=64 or set(baseline_keys)!={(c,i) for c in CASES for i in range(8)} or len(restore_keys)!=128 or set(restore_keys)!={(c,l,h,s) for c in CASES for l in range(2) for h in range(4) for s in (0.,.5)}:
        raise ValueError('incomplete or duplicate Cartesian grid')

def graph_hash(graph):
    import hashlib
    value={'nodes':sorted(graph.nodes),'parents':[(r,sorted(ws)) for r,ws in sorted(graph.parents.items())],'out':sorted(graph.out),'complete':graph.complete}
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()

def summarize(rows):
    result={}
    for label in ('all','active','inactive','delete','half','active_delete','active_half'):
        selected=[r for r in rows if (label not in ('active','active_delete','active_half') or r['active_damage']) and (label!='inactive' or not r['active_damage']) and (label not in ('delete','active_delete') or r['edit']['scale']==0.) and (label not in ('half','active_half') or r['edit']['scale']==.5)]
        arms={}
        for arm in (*ARMS,'gt'):
            values=[r['candidates'][arm] for r in selected if r['candidates'][arm]['valid']]
            arms[arm]={'n':len(selected),'valid':len(values),'invalid':len(selected)-len(values),'kl_denominator':len(values),
                       'mean_direct_kl_bits':sum(v['direct_kl_bits'] for v in values)/len(values) if values else None,
                       'worst_direct_kl_bits':max((v['direct_kl_bits'] for v in values),default=None),
                       'top_correct':sum(r['candidates'][arm]['valid'] and r['candidates'][arm]['top']==r['reference_top'] for r in selected),
                       'top_accuracy_all':sum(r['candidates'][arm]['valid'] and r['candidates'][arm]['top']==r['reference_top'] for r in selected)/len(selected) if selected else None,
                       'mean_sparse_nochange_kl_bits':sum(v['sparse_nochange_kl_bits'] for v in values)/len(values) if values else None,
                       'mean_dense_nochange_kl_bits':sum(r['dense_nochange_kl_bits'] for r in selected)/len(selected) if selected else None}
        result[label]={'worlds':len(selected),'arms':arms}
    return result

def run(source,payload,out):
    if out.exists(): raise ValueError('immutable new output required')
    if h.digest(h.__file__)!=HELPER_SHA:raise ValueError('imported fixed-basis helper changed')
    inputs,questions,samples=panel(payload)
    sources=json.loads((payload/'original24_manifest.json').read_text())
    verify_files(source,sources['sources'])
    sys.path[:0]=[str(source/'experiments/oracle/toyloop'),str(source/'experiments/oracle/toys'),str(source/'method/oracle'),str(source/'method/vpd')]
    import torch,planted,native,goal_eval,circuit
    torch.set_num_threads(1)
    out.mkdir();contract={'source':'cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4','driver_sha256':h.digest(__file__),'helpers_sha256':h.digest(h.__file__),
      'input_contract_sha256':h.digest(payload/'input_contract.json'),'response_contract_sha256':h.digest(payload/'response_contract.json'),
      'cases':CASES,'arms':ARMS,'scales':[0.,.5],'heads_per_case':8,'inputs_per_case':8,'edited_worlds':1024,'baseline_worlds':64,
      'candidate_live':False,'embedding':'on','frozen_boundary':False,'selection':'original compiled topology, no re-selection; every case and invalid retained',
      'weights':'W O-input columns + original-basis V input rows and delta_T rows scale; U/unit IDs fixed; all restore bitwise',
      'integrity_max_bits':h.INTEGRITY_MAX,'damage_min_bits':h.DAMAGE_MIN,'scope':'finite fresh compositional head-edit development check; no English or general VPD certificate'}
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    for ident in CASES:
        path=payload/'new_instances';blob=torch.load(path/(ident+'.pt'),map_location='cpu',weights_only=True);bank=json.loads((path/(ident+'.bank.json')).read_text())
        if blob['cfg']['n_layer']!=2 or blob['cfg']['n_head']!=4 or len(bank)!=8: raise ValueError('frozen architecture/bank mismatch')
        nat=planted.native_of(blob['sd'],blob['cfg'])
        toy=SimpleNamespace(sd=blob['sd'],cfg=blob['cfg'],meta={'decomposition':{n:(nat.target.site(n).U,nat.target.site(n).V,nat.units[n]) for n in nat.names}},texts=bank,vocab=['BOS']+[f't{i}' for i in range(1,32)])
        basis={n:{k:h.tensor_hash(getattr(nat.target.site(n),k)) for k in ('W','U','V','delta_T')}|{'units':h.tensor_hash(nat.units[n])} for n in nat.names}
        text={'gt':(path/(ident+'.target.txt')).read_text()};preinvalid={}
        for arm in ARMS:
            sample=samples[arm][ident];text[arm]=sample.get('completion')
            if str(sample['finish']).lower() not in ('stop','stopreason.stop'):preinvalid[arm]='non-stop finish'
            elif sample.get('parse_error') or not isinstance(text[arm],str):preinvalid[arm]='missing/parse-error response'
        compiled,invalid=h.compile_candidates(text,bank,circuit.parse,lambda graph,ids:circuit.unit_graph(graph,toy,ids),preinvalid)
        graph_fingerprints=[{arm:graph_hash(g) for arm,g in graphs.items()} for graphs in compiled]
        anchors={'wte':h.tensor_hash(nat.target.wte),'ln_f':h.tensor_hash(nat.target.ln_f),'norms':[h.tensor_hash(x) for x in nat.target.norms]}
        (out/(ident+'.basis.json')).write_text(json.dumps({'basis':basis,'anchors':anchors,'compiled_graphs':graph_fingerprints},indent=2))
        full=[native.Graph(masks={n:torch.ones(len(ids),nat.C[n],dtype=torch.bool) for n in nat.names}) for ids in bank]
        baseline=[]
        for i,ids in enumerate(bank):
            clear(nat);ge=goal_eval.GoalEval(nat);targets=[len(ids)-1]
            p0=nat.reference([ids],targets).clone();complete=ge.graph_ref(ids,targets,full[i],live=True)
            q0={arm:ge.graph_ref(ids,targets,g,live=False).clone() for arm,g in compiled[i].items()}
            saved=h.save_distributions(out,f'baseline_{ident}_{i}.safetensors',{'p0':p0,'complete':complete,**{'q0_'+arm:q for arm,q in q0.items()}})
            raw=h.raw_kl_bits(p0,complete)
            h.append(out/'baseline.jsonl',{'case':ident,'input':i,'full_integrity_raw_kl_bits':raw,'distributions':saved,'invalid':invalid[i],'candidate_kl_bits':{arm:h.kl_bits(p0,q) for arm,q in q0.items()}})
            if raw>h.INTEGRITY_MAX:raise ValueError('baseline full architecture mismatch')
            baseline.append((p0,q0))
        with h.scaled_head(nat,0,0,1.):
            clear(nat);ids=bank[0];ge=goal_eval.GoalEval(nat)
            if not torch.equal(nat.reference([ids],[len(ids)-1]),baseline[0][0]):raise ValueError('no-op dense mismatch')
            for arm,g in compiled[0].items():
                if not torch.equal(ge.graph_ref(ids,[len(ids)-1],g,live=False),baseline[0][1][arm]):raise ValueError('no-op sparse mismatch')
        for layer in range(2):
            for head in range(4):
                for scale in (0.,.5):
                    with h.scaled_head(nat,layer,head,scale) as edit:
                        clear(nat)
                        if any(h.tensor_hash(nat.target.site(n).U)!=b['U'] or h.tensor_hash(nat.units[n])!=b['units'] for n,b in basis.items()):raise ValueError('basis identity changed')
                        for i,ids in enumerate(bank):
                            # No norms/rows/rest/freeze supplied to any candidate.
                            ge=goal_eval.GoalEval(nat);targets=[len(ids)-1];p0,q0=baseline[i]
                            pe=nat.reference([ids],targets).clone();complete=ge.graph_ref(ids,targets,full[i],live=True)
                            qe={arm:ge.graph_ref(ids,targets,g,live=False) for arm,g in compiled[i].items()}
                            saved=h.save_distributions(out,f'edit_{ident}_{layer}_{head}_{scale}_{i}.safetensors',{'p0':p0,'pe':pe,'complete':complete,**{'q0_'+arm:q for arm,q in q0.items()},**{'qe_'+arm:q for arm,q in qe.items()}})
                            damage=h.kl_bits(pe,p0);raw=h.raw_kl_bits(pe,complete)
                            row={'case':ident,'input':i,'edit':edit,'active_damage':damage>=h.DAMAGE_MIN,'dense_nochange_kl_bits':damage,'full_integrity_raw_kl_bits':raw,'reference_top':int(pe[0,-1].argmax()),'original_top':int(p0[0,-1].argmax()),'distributions':saved,'candidates':{}}
                            for arm in (*ARMS,'gt'):
                                if arm in invalid[i]:row['candidates'][arm]={'valid':False,'invalid_reason':invalid[i][arm],'direct_kl_bits':None,'direct_raw_kl_bits':None,'top':None,'sparse_nochange_kl_bits':None}
                                else:
                                    q=qe[arm];row['candidates'][arm]={'valid':True,'direct_kl_bits':h.kl_bits(pe,q),'direct_raw_kl_bits':h.raw_kl_bits(pe,q),'top':int(q[0,-1].argmax()),'sparse_nochange_kl_bits':h.kl_bits(pe,q0[arm])}
                            h.append(out/'draws.jsonl',row)
                            if raw>h.INTEGRITY_MAX:raise ValueError('edited full architecture mismatch; row retained')
                    clear(nat)
                    if [{arm:graph_hash(g) for arm,g in graphs.items()} for graphs in compiled]!=graph_fingerprints:raise ValueError('compiled topology mutated')
                    if anchors!={'wte':h.tensor_hash(nat.target.wte),'ln_f':h.tensor_hash(nat.target.ln_f),'norms':[h.tensor_hash(x) for x in nat.target.norms]}:raise ValueError('embedding/norm weights mutated')
                    if any(h.tensor_hash(getattr(nat.target.site(n),k))!=b[k] for n,b in basis.items() for k in ('W','U','V','delta_T')) or any(h.tensor_hash(nat.units[n])!=b['units'] for n,b in basis.items()):raise ValueError('weight/basis restoration mismatch')
                    if not torch.equal(nat.reference([bank[0]],[len(bank[0])-1]),baseline[0][0]):raise ValueError('dense restoration mismatch')
                    h.append(out/'restore.jsonl',{'case':ident,'layer':layer,'head':head,'scale':scale,'all_weights_basis_bitwise':True,'dense_bitwise':True})
        print(json.dumps({'case':ident,'edited_worlds':128,'baseline_worlds':8}),flush=True)
    draws=h.rows(out/'draws.jsonl');baselines=h.rows(out/'baseline.jsonl');restores=h.rows(out/'restore.jsonl')
    verify_grid(draws,baselines,restores)
    summary={'edited_worlds':len(draws),'baseline_worlds':len(baselines),'restores':len(restores),'worst_full_integrity_kl_bits':max(max(0.,r['full_integrity_raw_kl_bits']) for r in draws),'groups':summarize(draws),'per_case':{name:summarize([r for r in draws if r['case']==name]) for name in CASES},'distribution_manifest_sha256':h.digest(out/'distribution_manifest.jsonl')}
    (out/'complete.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','payload','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.payload,a.out)
