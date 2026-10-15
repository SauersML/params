"""Execute existing token QA candidates in target-model and provided-circuit worlds.

The target model supplies supervision. Circuit predictions are fidelity evidence;
failed fidelity rows remain present. No service, tokenizer, dataset search or API.
Only explicit scalar graphs supported by semantic.Checker are admitted.
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


@torch.no_grad()
def measure_token_candidates(sc, tx, graph, candidates, strings, *, fidelity_bits):
    """Retain one paired row per existing token/change or token/KL candidate.

    Both `falls` labels concern the SAME target-model original top token, against
    each world's probability of that token before intervention. KL questions use
    each world's original distribution. Full distributions, not label agreement,
    determine the declared fidelity check. Construction/search are outside scope.
    `strings` is the complete exact model vocabulary in token-ID order.
    """
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
    binding = semantic.graph_claim_binding(graph,tx,[])
    with ExitStack() as stack:
        full = semantic.Checker(sc,tx,[])
        stack.callback(full.close)
        circuit = semantic.Checker(sc,tx,[],graph=graph)
        stack.callback(circuit.close)
        p0,q0 = _distribution(full.p0),_distribution(circuit.p0)
        if p0.numel()!=len(strings) or q0.shape!=p0.shape:
            raise ValueError('vocabulary must match both execution worlds')
        top0 = int(p0.argmax())
        baseline_kl = _kl(p0,q0)
        cache, rows = {}, []
        for index,candidate in enumerate(candidates):
            key=(candidate['position'],candidate['new'])
            if key not in cache:
                intervention={'tokens':{key[0]:key[1]}}
                p = _distribution(full.run(intervention))
                q = _distribution(circuit.run(intervention))
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
