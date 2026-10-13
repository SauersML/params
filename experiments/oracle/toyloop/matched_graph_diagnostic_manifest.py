"""Offline freeze of clarified original/permuted weight-ID diagnostics.

Consumes completed physical commutation evidence; performs no model forwards,
search, decoding or oracle calls. The clarification changes answer instructions,
so comparisons with archived unclarified responses are descriptive only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from graph_output_contract import clarify_graph_prompt, GRAPH_GRAMMAR
from matched_graph_permutation import SEED, map_edit

PRIMARY_SHA = '799385bb9a6d83bc7d095deb6ff86fb00f041bcd39a48c3a3c34fa5f19a61ddb'
POLICIES = ['base', 'point', 'intervention', 'qwen35', 'kimi']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError('frozen input changed: '+str(path))


def archive_binding(archive, digest, extracted, names):
    checked(archive, digest)
    with tarfile.open(archive, 'r:gz') as tar:
        members = {}
        for member in tar.getmembers():
            name = member.name.removeprefix('./')
            if member.isfile():
                if name in members:
                    raise ValueError('duplicate archive member')
                members[name] = member
        for name in names:
            if name not in members:
                raise ValueError('required archived evidence missing')
            data = tar.extractfile(members[name]).read()
            if data != (Path(extracted) / name.removeprefix('permutation/')).read_bytes():
                raise ValueError('extracted permutation evidence changed')


def validate_plan(item, plan):
    if (plan['id'] != item['id'] or plan['weights_sha256'] != item['weights_sha256']
            or plan['inputs'] != item['inputs'] or plan['edits'] != item['edits']):
        raise ValueError('permutation source case/input/edit mismatch')
    for key, value in plan['maps'].items():
        a, b = value['new_to_old'], value['old_to_new']
        if (sorted(a) != list(range(len(a))) or a == list(range(len(a)))
                or len(b) != len(a) or any(type(i) is not int for i in a+b)
                or sorted(b) != list(range(len(a)))
                or any(a[b[i]] != i for i in range(len(a)))):
            raise ValueError('nonidentity bijection required')
    if plan['mapped_edits'] != [map_edit(edit, plan['maps']) for edit in item['edits']]:
        raise ValueError('literal edit transport changed')


def build(args):
    checked(args.manifest, PRIMARY_SHA)
    checked(args.source_panel, args.source_panel_sha256)
    if args.out.exists():
        raise ValueError('fresh output directory required')
    original = json.loads(args.manifest.read_text())
    source = {r['id']: r for r in [json.loads(x) for x in args.source_panel.read_text().splitlines()]}
    cases = [r for r in original['panel'] if r['stratum'] == 'new_model']
    if len(cases) != 8:
        raise ValueError('exact original eight cases required')
    names = ['permutation/'+n for n in ('protocol.json', 'complete.json', 'results.jsonl')]
    names += ['permutation/'+r['id']+suffix for r in cases for suffix in ('.pt', '.prompt.txt')]
    archive_binding(args.archive, args.archive_sha256, args.permutation, names)
    protocol = json.loads((args.permutation/'protocol.json').read_text())
    done = json.loads((args.permutation/'complete.json').read_text())
    if (protocol['manifest_sha256'] != PRIMARY_SHA or protocol['seed'] != SEED
            or protocol['dense_forward_cap'] != 2992 or done['forwards'] != 2992
            or done['worlds'] != 1496 or done['all_commutation_pass'] is not True
            or done['all_top_agree'] is not True or done['failed_worlds'] != 0
            or len(protocol['plans']) != 8):
        raise ValueError('complete fixed permutation proof required')
    rows = [json.loads(x) for x in (args.permutation/'results.jsonl').read_text().splitlines()]
    keys = [(r['case'],r['edit_index'],r['input']) for r in rows]
    expected = {(r['id'],e,i) for r in cases for e in range(len(r['edits'])) for i in range(len(r['inputs']))}
    if len(keys) != len(set(keys)) or set(keys) != expected or any(r['passes'] is not True for r in rows):
        raise ValueError('permutation Cartesian proof incomplete')
    for item, plan in zip(cases, protocol['plans']):
        validate_plan(item, plan)
        checked(args.payload/item['weights'], item['weights_sha256'])
        s = source[item['id']]
        if s['prompt_sha256'] != item['prompt_sha256'] or hashlib.sha256(s['prompt'].encode()).hexdigest() != item['prompt_sha256']:
            raise ValueError('original source prompt changed')
    args.out.mkdir(); (args.out/'assets').mkdir()
    panel, prompts = [], []
    for condition in ('original','permuted'):
        for item, plan in zip(cases, protocol['plans']):
            ident = item['id'] if condition == 'original' else item['id']+'__perm'
            weights = args.payload/item['weights'] if condition == 'original' else args.permutation/(item['id']+'.pt')
            prompt = source[item['id']]['prompt'] if condition == 'original' else (args.permutation/(item['id']+'.prompt.txt')).read_text()
            prompt = clarify_graph_prompt(prompt)
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            relative = 'assets/'+ident+'.pt'; shutil.copyfile(weights,args.out/relative)
            common = {'id':ident,'original_id':item['id'],'condition':condition,
                      'stratum':'diagnostic_'+condition,'prompt_sha256':digest}
            prompts.append(dict(common,prompt=prompt))
            panel.append(dict(common,answer_format='graph_only',weights=relative,
                weights_sha256=sha(weights),inputs=item['inputs'],input_roles=item['input_roles'],
                edits=item['edits'] if condition=='original' else plan['mapped_edits']))
    manifest = {'schema_version':1,'diagnostic':'clarified_unit_permutation_v1','arms':POLICIES,
        'panel':panel,'source_sha256':original['source_sha256'],'support_sha256':original['support_sha256'],
        'forward_cap':7*sum(len(r['inputs'])*len(r['edits']) for r in panel),
        'primary_manifest_sha256':PRIMARY_SHA,'source_panel_sha256':args.source_panel_sha256,
        'permutation_archive_sha256':args.archive_sha256,'clarifier_sha256':hashlib.sha256(GRAPH_GRAMMAR.encode()).hexdigest(),
        'clarifier_module_sha256':sha(Path(__file__).with_name('graph_output_contract.py')),
        'helper_sha256':sha(__file__),'scope':'unit-name equivariance with clarified grammar; no new mechanism generalization'}
    for name,value in (('scoring_manifest.json',manifest),):
        (args.out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    (args.out/'panel.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in prompts))
    receipt={'models':16,'policies':POLICIES,'first_samples':80,'output_cap':416,
             'candidate_worlds':5*2992,'forward_cap':7*2992,'native_forwards':0,'service_calls':0,
             'manifest_sha256':sha(args.out/'scoring_manifest.json'),'panel_sha256':sha(args.out/'panel.jsonl')}
    (args.out/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','source_panel','payload','permutation','archive','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--source-panel-sha256',required=True)
    p.add_argument('--archive-sha256',required=True)
    print(json.dumps(build(p.parse_args())))
