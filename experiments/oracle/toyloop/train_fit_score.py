"""TRAIN-fit receipts and source-group-aware cached physical evaluation.

Target graph agreement is descriptive supervision fit, separately from executable
finite-intervention fidelity. Historical cache bindings are never rewritten.
"""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha(value):return hashlib.sha256(value.encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def target_fit(completion, target, finish, *, parser):
    """No English, tok annotations, or invalid/length answers earn graph credit."""
    gold=parser(target)
    if not gold.ok or gold.bad:raise RuntimeError('invalid frozen teacher target')
    good=completion is not None and str(finish).lower() in ('stop','stopreason.stop')
    parsed=parser(completion) if good else None
    good=good and parsed.ok and not parsed.bad
    def components(p):return {u for u in p.units if u.startswith('L')}
    def edges(p):return {e for e in p.edges if e[0].startswith('L') and (e[1].startswith('L') or e[1]=='out')}
    gu,ge=components(gold),edges(gold)
    pu,pe=(components(parsed),edges(parsed)) if good else (set(),set())
    def f1(a,b):return 2*len(a&b)/(len(a)+len(b)) if a or b else 1.
    return dict(parse_valid=bool(good),exact_units=bool(good and pu==gu),exact_edges=bool(good and pe==ge),
                unit_f1=f1(pu,gu) if good else 0.,edge_f1=f1(pe,ge) if good else 0.,
                missing_edges=sorted(ge-pe),extra_edges=sorted(pe-ge),
                scope='own teacher graph fit only; native executability and fidelity evaluated separately')


def validate_group(entry):
    """Validate files and reference ordering without importing torch or models."""
    for name,expected in entry['files'].items():
        if sha(name)!=expected:raise RuntimeError('frozen TRAIN source/cache file changed: '+name)
    metadata=load(entry['cache_json'])
    binding=metadata['binding']
    if binding['inputs']!=entry['inputs'] or binding['panel']!=entry['panel']:
        raise RuntimeError('TRAIN reference input/intervention order mismatch')
    worlds=len(entry['inputs'])*(1+len(entry['panel']))
    if worlds!=entry['worlds'] or len(metadata['reference_sha256'])!=worlds:
        raise RuntimeError('incomplete TRAIN reference worlds')
    for name,expected in binding['source'].items():
        if sha(entry['modules'][name])!=expected:raise RuntimeError('historical producer module mismatch: '+name)
    if sha(entry['teacher_source'])!=binding['teacher_source']:
        raise RuntimeError('historical teacher/cache validator changed')
    if text_sha(entry['intervention_target'])!=entry['target_sha256']:
        raise RuntimeError('frozen intervention target changed')
    return metadata


def cached_score(entry, completion, finish, *, toy, nat, reward, teacher):
    """One candidate: strict historical binding, at most one sparse/world.

    Caller starts an isolated process per source group, loads bound historical
    modules and assets, and checks their identities before constructing Native.
    Runtime/basis drift rejects; no fresh dense forward substitutes for the cache.
    """
    metadata=validate_group(entry)
    import torch
    from safetensors.torch import load_file
    tensors=load_file(entry['cache_tensors'])
    prefix=entry['reference_key_prefix']
    if prefix not in ('p_', 'reference_'):raise RuntimeError('unknown producer reference format')
    keys=[prefix+str(i) for i in range(entry['worlds'])]
    if set(tensors)!=set(keys):raise RuntimeError('reference tensor key coverage mismatch')
    cache=dict(metadata,references=[tensors[k] for k in keys])
    teacher.validate_cache(cache,toy,nat,entry['inputs'],entry['panel'])
    if completion is None or str(finish).lower() not in ('stop','stopreason.stop'):
        return dict(status='invalid_first_response',worlds=entry['worlds'],sparse_calls=0,per_world=[])
    with torch.no_grad():
        result=reward.score_wiring_interventions(toy,completion,nat=nat,texts=entry['inputs'],
                                                panel=entry['panel'],reference_cache=cache)
    if result['dense_calls']!=0 or result['sparse_calls']!=entry['worlds']:
        raise RuntimeError('cached physical forward bound exceeded')
    return result


def compare(responses_path, training_path, *, expected_responses_sha, expected_training_sha):
    if sha(responses_path)!=expected_responses_sha or sha(training_path)!=expected_training_sha:
        raise RuntimeError('frozen decoded/TRAIN receipt changed')
    import circuit
    receipt=load(responses_path);training=rows(training_path)
    if len(training)!=24 or len({r['id'] for r in training})!=24 or set(receipt['arms'])!={'base','point','intervention'}:
        raise RuntimeError('exact24 TRAIN/three-arm coverage required')
    result=[]
    for arm in ('base','point','intervention'):
        file=receipt['arms'][arm];path=Path(responses_path).parent/file['path']
        if sha(path)!=file['sha256']:raise RuntimeError('decoded first responses changed')
        predictions={}
        for row in rows(path):
            if row['id'] in predictions or row['id'] not in {r['id'] for r in training}:raise RuntimeError('duplicate/nonTRAIN response')
            predictions[row['id']]=row
        for row in training:
            answer=predictions.get(row['id'])
            if answer is not None and (answer['sample']!=0 or answer['prompt_sha256']!=row['prompt_sha256']):
                raise RuntimeError('TRAIN firstresponse source mismatch')
            for teacher in ('point','intervention'):
                fit=target_fit(answer.get('completion') if answer and not answer.get('parse_error') else None,
                               row[teacher+'_completion'],answer.get('finish','missing') if answer else 'missing',parser=circuit.parse)
                result.append(dict(id=row['id'],arm=arm,teacher=teacher,**fit))
    return {'scope':'descriptive exact24 TRAIN graph fit; no physical fidelity claim',
            'responses_sha256':expected_responses_sha,'training_sha256':expected_training_sha,'rows':result,
            'summary':{arm:{teacher:{key:sum(r[key] for r in result if r['arm']==arm and r['teacher']==teacher)/24
                                    for key in ('parse_valid','exact_units','exact_edges','unit_f1','edge_f1')}
                            for teacher in ('point','intervention')} for arm in ('base','point','intervention')}}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    for name in ('responses','training','out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('responses','training'):parser.add_argument('--'+name+'-sha256',required=True)
    args=parser.parse_args()
    if args.out.exists():raise RuntimeError('fresh own-teacher readout required')
    value=compare(args.responses,args.training,expected_responses_sha=args.responses_sha256,expected_training_sha=args.training_sha256)
    args.out.write_text(json.dumps(value,indent=2)+'\n')
