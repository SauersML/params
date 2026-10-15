"""Execute existing token QA candidates in target-model and provided-circuit worlds.

The target model supplies supervision. Circuit predictions are fidelity evidence;
failed fidelity rows remain present. No service, tokenizer, dataset search or API.
Scalar graphs and fixed native Boolean masks have separate execution backends.
"""
from contextlib import ExitStack
import copy
import math

import torch
import semantic
import components


def _distribution(value):
    value = value.detach().double()
    if value.ndim != 1 or not torch.isfinite(value).all():
        raise ValueError('finite one-dimensional log probabilities required')
    return value.log_softmax(-1)


def _kl(p, q):
    return max(0., float((p.exp() * (p-q)).sum()) / math.log(2))


def _label(base, changed, reference_top, strings):
    bits = float(changed[reference_top]-base[reference_top])/math.log(2)
    top = int(changed.argmax())
    probs, indices = changed.exp().topk(min(20, len(strings)))
    return {'falls': bits < -1., 'bits': bits, 'kl': _kl(base, changed),
            'top': top, 'top_changed': top != reference_top,
            'topk': [[components.shown(strings[int(i)]), float(p)] for p,i in zip(probs,indices)]}


def _precheck(tx, candidates, strings, fidelity_bits):
    if type(fidelity_bits) not in (int,float) or not math.isfinite(fidelity_bits) or fidelity_bits < 0:
        raise ValueError('explicit finite nonnegative fidelity tolerance required')
    if not strings or not all(isinstance(s,str) for s in strings):
        raise ValueError('complete token vocabulary required')
    for q in candidates:
        if q.get('kind') != 'token' or q.get('ask') not in ('change','kl'):
            raise ValueError('only existing token change/KL questions; zero/mean removals unsupported')
        if type(q.get('position')) is not int or not 0 <= q['position'] < len(tx.ids):
            raise ValueError('invalid token position')
        if type(q.get('new')) is not int or not 0 <= q['new'] < len(strings):
            raise ValueError('invalid replacement token ID')
        if q['new'] == tx.ids[q['position']]:
            raise ValueError('replacement must change a token')

@torch.no_grad()
def _measure(execute, binding, candidates, strings, fidelity_bits):
    p0,q0 = map(_distribution, execute(None))
    if p0.numel()!=len(strings) or q0.shape!=p0.shape:
        raise ValueError('vocabulary must match both execution worlds')
    top0 = int(p0.argmax())
    baseline_kl = _kl(p0,q0)
    cache, rows = {}, []
    for index,candidate in enumerate(candidates):
        key=(candidate['position'],candidate['new'])
        if key not in cache:
            pv,qv = execute(key)
            p,q = _distribution(pv),_distribution(qv)
            if p.shape != p0.shape or q.shape != p0.shape:
                raise ValueError('edited vocabulary mismatch')
            left,right = _label(p0,p,top0,strings),_label(q0,q,top0,strings)
            cache[key]=(left,right,_kl(p,q))
        left,right,changed_kl=cache[key]
        ask=candidate['ask']
        label = {**copy.deepcopy(candidate), **left}
        label['cell'] = f"{left['falls']}/{left['top_changed']}" if ask=='change' else str(left['kl']>1.)
        if ask=='kl':label['answer']=left['kl']>1.
        agreement = (left['falls']==right['falls'] and left['top']==right['top']) if ask=='change' else ((left['kl']>1.)==(right['kl']>1.))
        passed=max(baseline_kl,changed_kl)<=fidelity_bits
        rows.append({'candidate_index':index,'source_candidate':copy.deepcopy(candidate),
            'target_model_label':label,'circuit_outcome':copy.deepcopy(right),
            'supervision_world':'target_model','reference_top':top0,
            'circuit_baseline_top':int(q0.argmax()),'execution_binding':binding,
            'fidelity':{'baseline_kl_bits':baseline_kl,'intervention_kl_bits':changed_kl,
                'tolerance_bits':fidelity_bits,'passed':passed,'label_agreement':agreement},
            'control_status':'failed_fidelity' if not passed else ('label_disagreement' if not agreement else 'faithful')})
    return rows

@torch.no_grad()
def measure_token_candidates(sc, tx, graph, candidates, strings, *, fidelity_bits):
    """Paired scalar provided-graph execution; target model supplies all labels."""
    _precheck(tx, candidates, strings, fidelity_bits)
    binding = semantic.graph_claim_binding(graph, tx, [])
    with ExitStack() as stack:
        full = semantic.Checker(sc, tx, [])
        stack.callback(full.close)
        circuit = semantic.Checker(sc, tx, [], graph=graph)
        stack.callback(circuit.close)
        def execute(key):
            if key is None:
                return full.p0, circuit.p0
            intervention = {'tokens': {key[0]: key[1]}}
            return full.run(intervention), circuit.run(intervention)
        return _measure(execute, binding, candidates, strings, fidelity_bits)

@torch.no_grad()
def measure_native_token_candidates(nat, tx, live, means, candidates, strings, *,
                                    fidelity_bits, means_provenance):
    """Fixed Boolean native head/neuron masks with frozen supplied means.

    Exactly one target, the final input position, is supported. Masks cover the
    entire input, including an all-false (valid all-mean) circuit. Provenance is
    descriptive metadata, never an assertion of distributional validity. Callers
    are expected to supply means constructed from original TRAIN inputs. No
    weight edits or arbitrary hooks are supported. No graph prompt is produced.
    """
    import hashlib
    import json
    import native_units

    candidates = list(candidates)
    _precheck(tx, candidates, strings, fidelity_bits)
    ids, targets = list(tx.ids), list(tx.targets)
    if not ids or targets != [len(ids)-1]:
        raise ValueError('native token QA requires one target at final input position')
    if any(type(i) is not int or not 0 <= i < len(strings) for i in ids):
        raise ValueError('invalid original token IDs')
    tg = nat.target
    import torch.nn.modules.module as module_hooks
    if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
        raise ValueError('global hooks unsupported by native fixed-mask execution')
    if getattr(tg, 'behavior', None) is not None:
        raise ValueError('custom output behavior unsupported by native unit executor')
    if not nat.plain():
        raise ValueError('native fixed-mask backend requires plain unedited model')
    import vpd_model
    if type(tg) is not vpd_model.Target:
        raise ValueError('native fixed-mask backend currently supports vpd_model.Target only')
    for name in nat.names:
        site = tg.site(name)
        if site.b is not None or site.mask is not None or site.delta_mask is not None:
            raise ValueError('biased or edited sites unsupported by native fixed-mask executor')
    L, H, hd = tg.n_layer, tg.n_head, tg.hd
    Dm = tg.site('h.0.mlp.c_fc').W.shape[0]
    shapes = ((L,len(ids),H), (L,len(ids),Dm))
    if not isinstance(live, (tuple,list)) or len(live) != 2:
        raise ValueError('heads and neurons masks required')
    masks = []
    for value, shape in zip(live, shapes):
        if not isinstance(value,torch.Tensor) or value.dtype != torch.bool or tuple(value.shape) != shape:
            raise ValueError('Boolean masks must cover layers, all input positions and units')
        masks.append(value.detach().clone().to(nat.dev)[None])
    if not isinstance(means,dict):
        raise ValueError('fixed means required')
    fixed = {}
    for key,shape in (('heads',(L,H,hd)), ('neurons',(L,Dm))):
        value = means.get(key)
        if (not isinstance(value,torch.Tensor) or not value.is_floating_point()
                or tuple(value.shape) != shape or not torch.isfinite(value).all()):
            raise ValueError('finite floating means with exact native unit shapes required')
        if value.dtype != tg.site('h.0.mlp.c_fc').W.dtype:
            raise ValueError('means dtype must match model weights')
        fixed[key] = value.detach().clone().to(nat.dev)
    # Fail serialization before any model evaluation; retain provenance verbatim.
    provenance = json.loads(json.dumps(means_provenance, sort_keys=True, allow_nan=False))
    if not isinstance(provenance,dict) or not provenance:
        raise ValueError('explicit nonempty frozen means provenance required')
    def digest(value):
        x = value.detach().contiguous().cpu()
        h = hashlib.sha256(str((str(x.dtype),tuple(x.shape))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    # Existing effective model binding rejects hooks and hashes actual weights,
    # target helpers and execution source, rather than trusting a supplied digest.
    model_binding = native_units._means_binding(native_units.goal_eval.GoalEval(nat), ['original'], [ids])
    binding = {'schema_version':1, 'semantics':'native_fixed_boolean_unit_mean',
               'model_binding':model_binding, 'ids':ids, 'targets':targets,
               'input_sha256':hashlib.sha256(json.dumps([ids,targets]).encode()).hexdigest(),
               'mask_sha256':[digest(x[0]) for x in masks],
               'means_sha256':{k:digest(v) for k,v in fixed.items()},
               'means_provenance':provenance,
               'means_policy':'fixed_supplied_means_across_token_edits',
               'vocabulary_sha256':hashlib.sha256(json.dumps(strings,ensure_ascii=False).encode()).hexdigest(),
               'execution_source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    binding['sha256'] = hashlib.sha256(json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest()
    def execute(key):
        changed = list(ids)
        if key is not None:
            changed[key[0]] = key[1]
        # Fresh embeddings and full-model reference on every unique edited input.
        mu = native_units.MeanUnits(nat, changed, targets, None)
        mu.means = fixed
        # Direct physical full forward deliberately bypasses Native's cross-call
        # reference cache, whose key does not bind changing model weights.
        hidden = tg.hidden(torch.tensor([changed], device=nat.dev))[:, targets]
        full = tg.unembed(hidden)[0,0].float().log_softmax(-1)
        return full, mu.logq(tuple(m.to(dtype=mu.emb.dtype) for m in masks))[0,0]
    return _measure(execute, binding, candidates, strings, fidelity_bits)
