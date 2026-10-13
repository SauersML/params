"""Independently verify saved distributions; no Native execution or model calls."""
import argparse
import json
import math
from pathlib import Path
import compositional_counterfactual as c

def raw_kl(p,q):
    import torch
    if p.shape!=q.shape or not p.numel():raise ValueError('distribution shape')
    p=p.double();q=q.double()
    for x in (p,q):
        if not torch.isfinite(x).all() or x.logsumexp(-1).abs().max()>1e-5:raise ValueError('distribution normalization')
    value=float((p.exp()*(p-q)).sum())/math.log(2)
    if not math.isfinite(value) or value < -1e-6:raise ValueError('signed KL')
    return value

def close(actual,expected):
    if not math.isfinite(actual) or abs(actual-expected)>1e-8:raise ValueError('reported metric mismatch')

def baseline_identity(edited,baseline):
    import torch
    for key in ('p0',*(k for k in baseline if k.startswith('q0_'))):
        if key not in edited or not torch.equal(edited[key],baseline[key]):raise ValueError('edited baseline array mismatch: '+key)

def audit(root):
    import torch
    from safetensors.torch import load_file
    draws=c.h.rows(root/'draws.jsonl');baselines=c.h.rows(root/'baseline.jsonl');restores=c.h.rows(root/'restore.jsonl')
    c.verify_grid(draws,baselines,restores)
    manifest=c.h.rows(root/'distribution_manifest.jsonl')
    if len(manifest)!=1088 or len({r['file'] for r in manifest})!=1088:raise ValueError('distribution manifest coverage')
    bindings={r['file']:r for r in manifest};used=set()
    def arrays(row):
        b=row['distributions'];name=b['file']
        if Path(name).name!=name or bindings.get(name)!=b or name in used:raise ValueError('distribution binding')
        used.add(name)
        if c.h.digest(root/name)!=b['sha256']:raise ValueError('distribution checksum')
        values=load_file(str(root/name))
        if sorted(values)!=b['keys']:raise ValueError('distribution keys')
        return values
    baseline={}
    for row in baselines:
        v=arrays(row);baseline[row['case'],row['input']]=v
        close(raw_kl(v['p0'],v['complete']),row['full_integrity_raw_kl_bits'])
        if max(0.,raw_kl(v['p0'],v['complete']))>1e-3:raise ValueError('baseline full integrity')
        expected={'p0','complete'}|{'q0_'+a for a in (*c.ARMS,'gt') if a not in row['invalid']}
        if set(v)!=expected:raise ValueError('baseline invalid coverage')
        for a,value in row['candidate_kl_bits'].items():close(max(0.,raw_kl(v['p0'],v['q0_'+a])),value)
    for row in draws:
        v=arrays(row);base=baseline[row['case'],row['input']];baseline_identity(v,base)
        damage=max(0.,raw_kl(v['pe'],v['p0']));close(damage,row['dense_nochange_kl_bits'])
        if row['active_damage']!=(damage>=1e-3):raise ValueError('damage classification')
        integrity=raw_kl(v['pe'],v['complete']);close(integrity,row['full_integrity_raw_kl_bits'])
        if max(0.,integrity)>1e-3:raise ValueError('edited full integrity')
        if int(v['pe'][0,-1].argmax())!=row['reference_top'] or int(v['p0'][0,-1].argmax())!=row['original_top']:raise ValueError('dense top')
        expected={'p0','pe','complete'}
        for a,item in row['candidates'].items():
            if item['valid']:
                expected|={'q0_'+a,'qe_'+a}
                raw=raw_kl(v['pe'],v['qe_'+a]);close(raw,item['direct_raw_kl_bits']);close(max(0.,raw),item['direct_kl_bits'])
                close(max(0.,raw_kl(v['pe'],v['q0_'+a])),item['sparse_nochange_kl_bits'])
                if int(v['qe_'+a][0,-1].argmax())!=item['top']:raise ValueError('candidate top')
            elif any(item[k] is not None for k in ('direct_kl_bits','direct_raw_kl_bits','top','sparse_nochange_kl_bits')):raise ValueError('invalid metric fabrication')
            if item['valid']!=(a not in next(r for r in baselines if r['case']==row['case'] and r['input']==row['input'])['invalid']):raise ValueError('candidate validity changed')
        if set(v)!=expected:raise ValueError('edited invalid coverage')
    if used!=set(bindings):raise ValueError('unused distribution arrays')
    if not all(r['all_weights_basis_bitwise'] and r['dense_bitwise'] for r in restores):raise ValueError('restoration failed')
    complete=json.loads((root/'complete.json').read_text())
    if complete['distribution_manifest_sha256']!=c.h.digest(root/'distribution_manifest.jsonl') or complete['groups']!=c.summarize(draws):raise ValueError('summary mismatch')
    result={'edited_worlds':len(draws),'baseline_worlds':len(baselines),'restores':len(restores),'saved_arrays_verified':len(used),'baseline_arrays_exact':True,'groups':c.summarize(draws)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(a.root),indent=2))
