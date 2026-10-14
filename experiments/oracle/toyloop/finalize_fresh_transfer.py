"""Offline receipt finalization of the one frozen fresh40 sampling draft.

No SDK, tokenizer, credential lookup or provider calls. The prospective scoring
manifest may add analysis pipelines, while generation bytes/tokens stay frozen.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import matched_diagnostic_sample as sample
from matched_graph_train import encoded,sha

DRAFT_SHA='1dac8669abc0d0169427c6750e5c30f3d74345a0896942a59efbe15a762e2de8'


def finalize(args):
    if sha(args.draft)!=args.draft_sha256 or args.draft_sha256!=DRAFT_SHA:
        raise ValueError('one canonical fresh40 draft required')
    if args.out.exists():raise ValueError('fresh finalization output required')
    draft=json.loads(args.draft.read_text());sample.validate(draft)
    if draft.get('draft_execution_ready') is not False or draft['profile']!='fresh_transfer35_v1':
        raise ValueError('pending fresh40 draft required')
    if sha(args.panel)!=draft['source_panel_sha256'] or sha(args.manifest)!=args.manifest_sha256:
        raise ValueError('prospective source/scoring bindings changed')
    panel=[json.loads(x) for x in args.panel.read_text().splitlines() if x.strip()]
    manifest=json.loads(args.manifest.read_text())
    inputs=draft['models']['Qwen/Qwen3-8B']['inputs']
    if manifest['arms']!=list(sample.FRESH_ROLES):
        raise ValueError('exact8/five-policy prospective scope required')
    if len(panel)!=8 or len(manifest['panel'])!=8:raise ValueError('exact8 panel required')
    import hashlib
    for source,case,inp in zip(panel,manifest['panel'],inputs):
        if (source['id']!=case['id'] or source['id']!=inp['id'] or source['stratum']!=case['stratum']
                or hashlib.sha256(source['prompt'].encode()).hexdigest()!=source['prompt_sha256']
                or source['prompt_sha256']!=case['prompt_sha256'] or source['prompt_sha256']!=inp['prompt_sha256']):
            raise ValueError('fresh prompt/order/stratum changed')
    prepared=copy.deepcopy(draft)
    for role,arm in (('point35','point'),('intervention35','intervention')):
        prepared['checkpoint_receipt_sha256'][role]=sha(args.new_checkpoints/(arm+'_checkpoint.jsonl'))
    prepared['scoring_manifest_sha256']=args.manifest_sha256
    if sha(args.pipeline_protocol)!=args.pipeline_protocol_sha256:raise ValueError('prospective pipeline protocol changed')
    protocol=json.loads(args.pipeline_protocol.read_text())
    if (protocol['panel_sha256']!=sha(args.panel) or protocol['raw_scoring_manifest_sha256']!=args.manifest_sha256
            or protocol['arms']!=list(sample.FRESH_ROLES) or protocol['all_cases']!=8 or protocol['additional_llm_calls']!=0
            or [r['name'] for r in protocol['pipelines']]!=['raw_explicit','same_selected_architecture_completion']):
        raise ValueError('prospective dual pipeline/source scope changed')
    prepared['pipeline_protocol_sha256']=args.pipeline_protocol_sha256
    prepared['draft_execution_ready']=True;prepared['pending_checkpoint_roles']=[]
    prepared['canonical_draft_sha256']=args.draft_sha256
    sample.validate(prepared)
    sample.checkpoint_paths(SimpleNamespace(checkpoints=args.checkpoints,new_checkpoints=args.new_checkpoints),prepared)
    if prepared['models']!=draft['models']:raise ValueError('frozen token arrays changed')
    args.out.mkdir();path=args.out/'prepared.json';path.write_text(encoded(prepared)+'\n')
    proof=sample.preflight(path);(args.out/'preflight.json').write_text(encoded(proof)+'\n')
    receipt={'canonical_draft_sha256':args.draft_sha256,'prepared_sha256':sha(path),'preflight_sha256':sha(args.out/'preflight.json'),
             'finalizer_sha256':sha(__file__),'runner_sha256':sha(sample.__file__),'source_panel_sha256':sha(args.panel),
             'manifest_sha256':args.manifest_sha256,'pipeline_protocol_sha256':args.pipeline_protocol_sha256,'checkpoint_receipt_sha256':prepared['checkpoint_receipt_sha256'],
             'common_init_receipt_sha256':prepared['common_init_receipt_sha256'],'parent_receipt_sha256':{},
             'service_calls':0,'execution_authorized':False}
    for group,directory in (('old24',args.checkpoints),('new35',args.new_checkpoints)):
        receipt['parent_receipt_sha256'][group]={name:sha(directory/name) for name in ('start.jsonl','complete.jsonl','common_init.jsonl')}
    (args.out/'receipt.json').write_text(encoded(receipt)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('draft','panel','manifest','pipeline_protocol','checkpoints','new_checkpoints','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--draft-sha256',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--pipeline-protocol-sha256',required=True)
    print(encoded(finalize(p.parse_args())))
