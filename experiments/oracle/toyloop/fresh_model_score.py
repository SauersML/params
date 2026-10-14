"""Prospective fresh-340 raw/complete compiler comparison; no oracle API.

Both pipelines run for every policy/case. Same-node completion is a separately
reported compiler baseline, never response repair or best-of selection.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fresh_model_evaluation import ARMS, specifications, pipeline_protocol
from matched_graph_score import sha, write, bind_responses, answer_error, summarize
from same_node_completion import complete_same_nodes, checked_distribution, checked_kl

MANIFEST_SHA = '772ae94de94ac4876c3d3f87db949a749fbb3cbde62e6d537fbd2003fd68857f'
PANEL_SHA = 'edbd2065a4a2bbcdb3da065f8398ba42fb44ee435171651fd76c223b33a2d10d'
PROTOCOL_SHA = '31d238aa2b1e896831aa57e4b515284e069542133e1e35ef985fa5d3028440e4'
PIPELINES = ('raw_explicit', 'same_selected_architecture_completion')


def validate_scope(manifest, protocol):
    if manifest['arms'] != ARMS or [r['id'] for r in manifest['panel']] != [r[0] for r in specifications()]:
        raise RuntimeError('fixed fresh-340 policies/cases required')
    worlds = sum(len(r['inputs'])*len(r['edits']) for r in manifest['panel'])
    if worlds != 1480 or manifest['forward_cap'] != 10360:
        raise RuntimeError('frozen fresh world count/raw cap changed')
    expected = pipeline_protocol(PANEL_SHA, MANIFEST_SHA, worlds)
    if any(protocol.get(k) != v for k,v in expected.items()):
        raise RuntimeError('prospective pipeline policy changed')
    if protocol['combined_forward_cap'] != 17760:
        raise RuntimeError('combined physical cap changed')
    for item in manifest['panel']:
        if (item['answer_format'] != 'graph_only' or item['stratum'] != 'fresh_model340'
                or len(item['inputs']) != 4 or item['input_roles'] != ['disclosed']*4
                or item['edits'][0] != {'kind':'no_edit','operations':[]}):
            raise RuntimeError('frozen disclosed-input/answer scope changed')
    return worlds


def composite_arm(policy, pipeline):
    return policy+'::'+pipeline


def compile_pipelines(answer, toy, ids, circuit, native):
    """Retain original invalidity; compiler failures never become repaired text."""
    reason = answer_error(answer,'graph_only')
    raw = None
    if reason is None:
        try:
            parsed = circuit.parse(answer['completion'])
            if not parsed.ok or parsed.bad:
                raise ValueError('strict circuit parser rejected answer')
            raw = circuit.unit_graph(parsed,toy,ids)
        except ValueError as exc:
            reason = str(exc)
    complete, size = (None,None) if raw is None else complete_same_nodes(raw,native,[len(ids)-1])
    return raw, complete, size, reason


def freeze_inputs(args):
    """Zero-forward contract; missing/unfinished policies still need final decoding."""
    if sha(args.manifest) != MANIFEST_SHA or sha(args.protocol) != PROTOCOL_SHA or sha(args.panel) != PANEL_SHA:
        raise RuntimeError('frozen panel/manifest/prospective protocol changed')
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    worlds = validate_scope(manifest, protocol)
    questions = [json.loads(x) for x in args.panel.read_text().splitlines()]
    if len(questions) != 8 or [r['id'] for r in questions] != [r['id'] for r in manifest['panel']]:
        raise RuntimeError('source prompt order changed')
    for question,item in zip(questions, manifest['panel']):
        if (hashlib.sha256(question['prompt'].encode()).hexdigest() != item['prompt_sha256']
                or question['prompt_sha256'] != item['prompt_sha256']):
            raise RuntimeError('original prompt bytes differ')
        if sha(args.payload/item['weights']) != item['weights_sha256']:
            raise RuntimeError('immutable fresh weights changed')
    for root,field in ((args.source,'source_sha256'),(args.support,'support_sha256')):
        for name,digest in manifest[field].items():
            if sha(root/name) != digest:
                raise RuntimeError('pinned executor/support changed: '+name)
    return manifest, {'schema_version':1, 'manifest_sha256':MANIFEST_SHA,
        'panel_sha256':PANEL_SHA, 'pipeline_protocol_sha256':PROTOCOL_SHA,
        'arms':ARMS, 'pipelines':list(PIPELINES), 'models':8,
        'reference_worlds':worlds, 'candidate_worlds':10*worlds,
        'forward_cap':17760, 'driver_sha256':sha(__file__),
        'helper_sha256':{name:sha(Path(__file__).parent/name) for name in
                        ('fresh_model_evaluation.py','matched_graph_score.py','same_node_completion.py')},
        'scope':'prospective fresh known-family models; fixed model-level denominators; no English or novel-mechanism proof'}


def run(args):
    if args.out.exists():
        raise RuntimeError('fresh output required; never resume or overwrite physics')
    manifest, contract = freeze_inputs(args)
    args.out.mkdir(parents=True)
    if args.prepare_only:
        write(args.out/'contract.json', contract)
        write(args.out/'complete.json', {'status':'prepared_no_forwards','native_forwards':0,
                                       'forward_cap':17760})
        return
    if (args.prepared_contract is None or args.prepared_contract_sha256 is None
            or sha(args.prepared_contract) != args.prepared_contract_sha256
            or json.loads(args.prepared_contract.read_text()) != contract):
        raise RuntimeError('zero-forward source/asset contract differs from preflight')
    if args.responses is None or args.responses_sha256 is None or sha(args.responses) != args.responses_sha256:
        raise RuntimeError('final canonical decoded responses and external SHA required')
    receipt = json.loads(args.responses.read_text())
    if receipt['manifest_sha256'] != MANIFEST_SHA or set(receipt['arms']) != set(ARMS):
        raise RuntimeError('decoded receipt belongs to another policy/panel')
    contract['responses_sha256'] = args.responses_sha256
    response_rows = {}
    for arm in ARMS:
        entry = receipt['arms'][arm]
        path = args.responses.parent/entry['path']
        if sha(path) != entry['sha256']:
            raise RuntimeError('canonical first-response file changed')
        response_rows[arm] = bind_responses(manifest['panel'],
            [json.loads(x) for x in path.read_text().splitlines() if x.strip()])
    write(args.out/'contract.json', contract)
    sys.path[:0] = [str(args.source/p) for p in
                   ('experiments/oracle/toyloop','experiments/oracle/toys','method/oracle','method/vpd')]
    import torch, circuit, planted, native, goal_eval
    sys.path.insert(0,str(args.support/'experiments/oracle/toyloop'))
    from head_counterfactual import scaled_head, raw_kl_bits
    from reward_interventions import scaled_neuron, clear_intervention_caches
    from intervention_teacher import tensor_digest
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    planned = {(r['stratum'],r['id'],composite_arm(a,p),e,i)
               for r in manifest['panel'] for a in ARMS for p in PIPELINES
               for e in range(len(r['edits'])) for i in range(len(r['inputs']))}
    write(args.out/'coverage.json', {'planned_candidate_worlds':len(planned),
          'planned_reference_worlds':1480,'completed_candidate_worlds':0,'actual_forwards':0})
    results, forwards, graph_receipts = [], 0, []
    for case_index,item in enumerate(manifest['panel']):
        blob = torch.load(args.payload/item['weights'],map_location='cpu',weights_only=True)
        toy = SimpleNamespace(name=item['id'], sd=blob['sd'], cfg=blob['cfg'], meta={})
        nat = planted.native_of(toy.sd,toy.cfg)
        toy.meta['decomposition'] = {n:(nat.target.site(n).U,nat.target.site(n).V,nat.units[n]) for n in nat.names}
        def state():
            return {'model':{k:tensor_digest(v) for k,v in nat.target.state_dict().items()},
                    'basis':{n:{k:tensor_digest(getattr(nat.target.site(n),k)) for k in ('W','U','V','delta_T')} for n in nat.names},
                    'units':{n:tensor_digest(nat.units[n]) for n in nat.names}}
        original = state()
        write(args.out/(str(case_index)+'.original.json'),original)
        graphs = {}
        for policy in ARMS:
            answer = response_rows[policy][case_index]
            for input_index,ids in enumerate(item['inputs']):
                raw,complete,size,reason = compile_pipelines(answer,toy,ids,circuit,native)
                components = sum(int(nat.C[n]) for n in nat.names)
                full = native.Graph({(n,t,c) for n in nat.names for t in range(len(ids)) for c in range(int(nat.C[n]))},complete=True)
                sizes = {}
                for pipeline,graph in zip(PIPELINES,(raw,complete)):
                    if graph is not None:
                        bits = graph.bits(len(ids),components,[len(ids)-1])
                        full_bits = full.bits(len(ids),components,[len(ids)-1])
                        if not all(__import__('math').isfinite(v) for v in (bits,full_bits)) or not 0<=bits<=full_bits or full_bits<=0:
                            raise RuntimeError('invalid executed-description budget')
                        sizes[pipeline] = {'nodes':graph.count(),'edges':graph.edges(),
                                           'description_bits':bits,'full_architecture_bits':full_bits,
                                           'description_cost':bits/full_bits}
                    else:
                        sizes[pipeline] = None
                    graphs[policy,pipeline,input_index] = graph
                graph_receipts.append({'case':item['id'],'policy':policy,'input':input_index,
                    'valid':raw is not None,'invalid_reason':reason,'completion_invariants':size,
                    'size':sizes,'selected_nodes_sha256':None if raw is None else hashlib.sha256(json.dumps(sorted(raw.node_set())).encode()).hexdigest()})
        write(args.out/(str(case_index)+'.graphs.json'),graph_receipts[-20:])
        full = [native.Graph({(n,t,c) for n in nat.names for t in range(len(ids)) for c in range(int(nat.C[n]))},complete=True) for ids in item['inputs']]
        baselines = []
        for edit_index,edit in enumerate(item['edits']):
            if len(edit['operations']) != {'no_edit':0,'individual':1,'joint':2}[edit['kind']]:
                raise RuntimeError('frozen intervention arity differs')
            with contextlib.ExitStack() as stack:
                for kind,layer,unit,scale in edit['operations']:
                    if kind not in ('head','neuron') or scale not in (0.,.5):
                        raise RuntimeError('frozen intervention type/scale differs')
                    stack.enter_context((scaled_head if kind=='head' else scaled_neuron)(nat,layer,unit,scale))
                for input_index,ids in enumerate(item['inputs']):
                    target = [len(ids)-1]
                    clear_intervention_caches(nat)
                    p = checked_distribution(nat.reference([ids],target).detach().clone(),32)
                    clear_intervention_caches(nat)
                    integrity = checked_distribution(goal_eval.GoalEval(nat).graph_ref(ids,target,full[input_index],live=True,norms={}),32)
                    forwards += 2
                    integrity_kl = checked_kl(raw_kl_bits(p,integrity))
                    tensors = {'p':p.contiguous(),'full':integrity.contiguous()}
                    if integrity_kl > .001:
                        save_file(tensors,str(args.out/'integrity_failure.safetensors'))
                        raise RuntimeError('full architecture integrity failure')
                    if edit_index == 0:
                        baselines.append(p.clone())
                    tensors['p0'] = baselines[input_index].contiguous()
                    world = []
                    for policy in ARMS:
                        receipt_graph = graph_receipts[-20+ARMS.index(policy)*4+input_index]
                        for pipeline in PIPELINES:
                            graph = graphs[policy,pipeline,input_index]
                            clear_intervention_caches(nat)
                            q = None if graph is None else checked_distribution(goal_eval.GoalEval(nat).graph_ref(ids,target,graph,live=False),32)
                            forwards += int(q is not None)
                            raw_kl = None if q is None else checked_kl(raw_kl_bits(p,q))
                            arm = composite_arm(policy,pipeline)
                            if q is not None:
                                tensors[arm] = q.contiguous()
                            world.append({'stratum':item['stratum'],'case':item['id'],'arm':arm,
                                'policy':policy,'pipeline':pipeline,'edit_index':edit_index,'input':input_index,
                                'input_role':'disclosed','kind':edit['kind'],'operations':edit['operations'],
                                'valid':q is not None,'invalid_reason':receipt_graph['invalid_reason'],
                                'raw_kl_bits':raw_kl,'kl_bits':None if raw_kl is None else max(0.,raw_kl),
                                'top':None if q is None else int(q.argmax()),'reference_top':int(p.argmax()),
                                'integrity_raw_kl_bits':integrity_kl,'size':receipt_graph['size'][pipeline]})
                    filename=f'{case_index}_{edit_index}_{input_index}.safetensors'
                    save_file(tensors,str(args.out/filename))
                    with (args.out/'results.jsonl').open('a') as stream:
                        for row in world:
                            row.update(file=filename,sha256=sha(args.out/filename))
                            stream.write(json.dumps(row,allow_nan=False)+'\n')
                            results.append(row)
                        stream.flush()
                        __import__('os').fsync(stream.fileno())
                    # Partial coverage stays visible if the worker is interrupted.
                    (args.out/'coverage.json').write_text(json.dumps({'planned_candidate_worlds':14800,
                        'planned_reference_worlds':1480,'completed_candidate_worlds':len(results),
                        'actual_forwards':forwards})+'\n')
            clear_intervention_caches(nat)
            if state() != original:
                raise RuntimeError('literal edit failed exact weight/basis/unit restoration')
        write(args.out/(str(case_index)+'.restored.json'),state())
        if sha(args.payload/item['weights']) != item['weights_sha256']:
            raise RuntimeError('fresh model asset changed during execution')
        print(json.dumps({'case':item['id'],'completed_candidate_worlds':len(results),'actual_forwards':forwards}),flush=True)
    if forwards > 17760 or len(results) != 14800:
        raise RuntimeError('combined forward/coverage cap violated')
    summary = summarize(results,planned)
    write(args.out/'summary.json',summary)
    write(args.out/'complete.json',{'actual_forwards':forwards,'forward_cap':17760,
          'candidate_worlds':14800,'reference_worlds':1480,'model_policy_pairs_per_pipeline':40,
          'responses_sha256':args.responses_sha256,'scope':contract['scope']})
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('manifest','protocol','panel','payload','source','support','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--responses',type=Path)
    parser.add_argument('--responses-sha256')
    parser.add_argument('--prepared-contract',type=Path)
    parser.add_argument('--prepared-contract-sha256')
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())
