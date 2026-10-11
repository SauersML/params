"""Single-edge path removals in the actual target, with every other path intact.

MPD_MEM_GIB=16 python experiments/vpd/validate_connectome.py --texts 20
Samples CI-positive endpoints: half uniform active pairs, half atlas top-64
partners when available. Residual/MLP same-position; OV source s <= destination t.
Read patch changes only one downstream VPD coefficient, leaving the linear
remainder and all other coefficients untouched. RMS/GELU counterfactuals remove
only the selected source write in that reader's input; they do not ablate the
source globally. OV patches one head. QK subtracts only the selected bilinear
term from one head's (t,s) score before softmax: Q and K activations themselves
cannot change, so their activation correlation is undefined; attention probability
is reported separately. Target KL is KL(clean || patch), in bits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from method.oracle.native import Native, tasks, text, free
from method.vpd.connectome import Connectome, DEFAULT, edges, rotate
from method.vpd.vpd_model import rms, gelu_tanh


@dataclass
class Removal:
    source: str
    sink: str
    kind: str
    source_index: int
    sink_index: int
    s: int
    t: int
    head: int
    delta: float  # signed change in downstream coefficient or QK score


@torch.no_grad()
def run(target, ids, targets, removals=(), collect=False):
    """Actual W forward, optionally routing one removed edge per batch row.

    All-ones VPD plus remainder equals these W maps. Clean equivalence to the
    production Target.hidden is checked by main and the synthetic test.
    """
    B, T = ids.shape
    trace = dict(inputs={}, residual={}, fc={}, pattern={}, observed={}, attention={})
    x = target.wte[ids]
    def project(n, inp):
        if collect:
            trace['inputs'][n] = inp[0].clone()
        out = inp @ target.site(n).W.T
        for b, e in enumerate(removals):
            if e.sink == n:
                st = target.site(n)
                pos = e.s if e.kind == 'qk' else e.t
                clean_read = inp[b, pos] @ st.V[:, e.sink_index]
                patched_read = clean_read if e.kind == 'qk' else clean_read + e.delta
                trace['observed'][b] = patched_read
                if e.kind != 'qk':
                    out[b, e.t] += (patched_read-clean_read) * st.U[e.sink_index]
        return out
    for i in range(target.n_layer):
        nm = lambda k: f"h.{i}.{'mlp' if k in ('c_fc','down_proj') else 'attn'}.{k}"
        h = rms(x, target.norms[2*i], target.eps)
        if collect:
            for k in ('q_proj', 'k_proj', 'v_proj'):
                trace['residual'][nm(k)] = x[0].clone()
        shape = lambda y: y.view(B, T, target.n_head, target.hd).transpose(1, 2)
        q, k, v = [shape(project(nm(k), h)) for k in ('q_proj','k_proj','v_proj')]
        q, k = target._rope(q, T), target._rope(k, T)
        z = (q @ k.transpose(-1,-2)) / math.sqrt(target.hd)
        for b, e in enumerate(removals):
            if e.kind == 'qk' and int(e.source.split('.')[1]) == i:
                z[b, e.head, e.t, e.s] += e.delta
        mask = torch.ones(T,T,device=x.device,dtype=torch.bool).tril()
        p = z.masked_fill(~mask, -torch.inf).softmax(-1)
        for b, e in enumerate(removals):
            if e.kind == 'qk' and int(e.source.split('.')[1]) == i:
                trace['attention'][b] = p[b,e.head,e.t,e.s]
        if collect:
            trace['pattern'][i] = p[0].clone()
        y = (p @ v).transpose(1,2).reshape(B,T,-1)
        x = x + project(nm('o_proj'), y)
        h = rms(x, target.norms[2*i+1], target.eps)
        if collect:
            trace['residual'][nm('c_fc')] = x[0].clone()
        pre = project(nm('c_fc'), h)
        if collect:
            trace['fc'][i] = pre[0].clone()
        x = x + project(nm('down_proj'), gelu_tanh(pre))
    hidden = rms(x, target.ln_f, target.eps)
    return hidden[:, targets] @ target.wte.T, trace


@torch.no_grad()
def removal(target, trace, source, sink, kind, c, d, s, t, head=-1):
    src, dst = target.site(source), target.site(sink)
    a = trace['inputs'][source][s] @ src.V[:, c]
    layer = int(sink.split('.')[1])
    if kind == 'residual':
        residual = trace['residual'][sink][t]
        gain = target.norms[2*layer + int(sink.endswith('c_fc'))]
        change = rms(residual - a * src.U[c], gain, target.eps) - rms(residual, gain, target.eps)
        delta = change @ dst.V[:, d]
        static = src.U[c] @ (gain * dst.V[:, d])
        effective = static * a
    elif kind == 'mlp':
        pre = trace['fc'][layer][t]
        delta = (gelu_tanh(pre - a * src.U[c]) - gelu_tanh(pre)) @ dst.V[:, d]
        static = src.U[c] @ dst.V[:, d]
        effective = static * a
    else:
        sl = slice(head * target.hd, (head+1)*target.hd)
        if kind == 'ov':
            static = src.U[c, sl] @ dst.V[sl, d]
            effective = static * a * trace['pattern'][layer][head,t,s]
            delta = -effective
        elif kind == 'qk':
            # Source is query at s=t(query); sink is key at t=s(key).
            static = src.U[c, sl] @ dst.U[d, sl]
            ak = trace['inputs'][sink][t] @ dst.V[:, d]
            actual = rotate(src.U[c, sl], s-t, target.cfg['rotary_base']) @ dst.U[d, sl]
            effective = actual * a * ak / math.sqrt(target.hd)
            delta = -effective
        else:
            raise ValueError(kind)
    # Store query position as t and key position as s for the score edit.
    e = Removal(source,sink,kind,c,d,t if kind=='qk' else s,s if kind=='qk' else t,head,float(delta))
    return e, float(abs(static*a)), float(abs(effective))


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return None
    return float(spearmanr(x,y).statistic)


def summarize(records):
    result = {}
    for kind in ('residual','mlp','ov','qk'):
        rs = [r for r in records if r['kind']==kind]
        result[kind] = dict(count=len(rs), texts=len({r['task'] for r in rs}),
            static_vs_activation=correlation([r['predictor'] for r in rs], [r['activation_effect'] for r in rs]),
            static_vs_kl_bits=correlation([r['predictor'] for r in rs], [r['kl_bits'] for r in rs]),
            activation_vs_kl_bits=correlation([r['activation_effect'] for r in rs], [r['kl_bits'] for r in rs]),
            mechanistic_vs_kl_bits=correlation([r['mechanistic_predictor'] for r in rs], [r['kl_bits'] for r in rs]),
            median_kl_bits=float(np.median([r['kl_bits'] for r in rs])) if rs else None,
            nonzero_kl_count=sum(r['kl_bits'] > 1e-10 for r in rs))
        if kind=='qk':
            result[kind]['static_vs_attention_probability'] = correlation([r['predictor'] for r in rs], [r['attention_effect'] for r in rs])
        result[kind]['strata'] = {}
        for label, subset in [('at_target', [r for r in rs if r.get('sink_at_target')]),
                              ('uniform_active', [r for r in rs if r['selection']=='uniform_active']),
                              ('top64_active', [r for r in rs if r['selection']=='top64_active'])]:
            result[kind]['strata'][label] = dict(count=len(subset),
                activation=correlation([r['predictor'] for r in subset], [r['activation_effect'] for r in subset]),
                kl=correlation([r['predictor'] for r in subset], [r['kl_bits'] for r in subset]))
        # Per-text correlations expose pooled scale differences.
        result[kind]['per_text'] = {task: dict(count=len(ts), activation=correlation([r['predictor'] for r in ts], [r['activation_effect'] for r in ts]),
            kl=correlation([r['predictor'] for r in ts], [r['kl_bits'] for r in ts])) for task in sorted({r['task'] for r in rs})
            for ts in [[r for r in rs if r['task']==task]]}
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--texts', type=int, default=20)
    p.add_argument('--pairs', type=int, default=32, help='pairs per text per edge type')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seed', type=int, default=2951)
    p.add_argument('--atlas', type=Path, default=DEFAULT)
    p.add_argument('--out', type=Path, default=DEFAULT/'validation.json')
    args = p.parse_args()
    if min(args.texts, args.pairs, args.batch) < 1:
        p.error('texts, pairs, and batch must be positive')
    tic = time.perf_counter()
    nat = Native()
    if not nat.has_importance:
        raise RuntimeError('Validation requires the published VPD CI network')
    atlas = Connectome(args.atlas)
    specs = {}
    for e in edges(nat.target, nat.names):
        if e.kind in ('residual','mlp','ov','qk'):
            specs.setdefault(e.kind, []).append(e)
    rng = np.random.default_rng(args.seed)
    files = tasks('heldout')[:args.texts]
    if len(files) != args.texts:
        raise ValueError('not enough heldout tasks')
    records, checks = [], []
    nat.vpd.clear()
    for path in files:
        ids, targets = text(path)
        inp = torch.tensor([ids], device=nat.dev)
        ci = nat.importance(ids)
        nat.vpd.clear()
        baseline, trace = run(nat.target, inp, targets, collect=True)
        actual = nat.target.hidden(inp)[:,targets] @ nat.target.wte.T
        diff = float((baseline-actual).abs().max())
        checks.append(dict(task=path.stem, max_logit_difference=diff, tokens=len(ids), targets=targets))
        if not torch.allclose(baseline,actual,atol=2e-4,rtol=2e-4):
            raise AssertionError(f'forward differs from actual Target: {diff}')
        active = {n: [torch.nonzero(row>0).flatten().cpu().numpy() for row in ci[n]] for n in nat.names}
        pending = []
        for kind, choices in specs.items():
            attempts = 0
            accepted = 0
            while accepted < args.pairs:
                attempts += 1
                if attempts > args.pairs*1000:
                    raise RuntimeError(f'insufficient CI-positive pairs: {path.stem} {kind}')
                spec = choices[int(rng.integers(len(choices)))]
                pos = int(rng.choice(targets)) if rng.random()<.5 else int(rng.integers(max(targets)+1))
                if kind in ('residual','mlp'):
                    s=t=pos
                elif kind == 'ov':
                    t=pos; s=int(rng.integers(t+1))
                else:
                    s=pos; t=int(rng.integers(s+1))
                cs, ds = active[spec.source][s], active[spec.sink][t]
                if not len(cs) or not len(ds):
                    continue
                c, d = int(rng.choice(cs)), int(rng.choice(ds))
                selection = 'uniform_active'
                if accepted % 2 == 0:
                    row = atlas.partners(spec.source,c,kind+'.downstream')
                    valid = np.flatnonzero((row['site']==atlas.names.index(spec.sink)) & (row['head']==spec.head) & np.isin(row['index'],ds))
                    if len(valid):
                        d = int(row['index'][rng.choice(valid)])
                        selection = 'top64_active'
                e, predictor, mechanistic = removal(nat.target, trace,spec.source,spec.sink,kind,c,d,s,t,spec.head)
                pending.append((e,dict(task=path.stem, kind=kind, source=e.source,sink=e.sink,source_index=c,sink_index=d,
                    source_position=s,sink_position=t,head=e.head,selection=selection,predictor=predictor,mechanistic_predictor=mechanistic,
                    sink_at_target=(s if kind=='qk' else t) in targets,
                    source_ci=float(ci[e.source][s,c]),sink_ci=float(ci[e.sink][t,d]))))
                accepted += 1
        for start in range(0,len(pending),args.batch):
            chunk = pending[start:start+args.batch]
            edits = [e for e,r in chunk]
            control, control_trace = run(nat.target, inp.expand(len(chunk),-1), targets, [replace(e,delta=0.) for e in edits])
            patched, measured = run(nat.target, inp.expand(len(chunk),-1), targets, edits)
            # Float64 log distributions avoid float32 KL roundoff for small effects.
            # CPU is deliberate: MPS has no float64 support. Paired controls use
            # the same batch shape to exclude different GEMM rounding paths.
            lp = control.cpu().double().log_softmax(-1)
            probs = lp.exp()
            kl = (probs * (lp-patched.cpu().double().log_softmax(-1))).sum(-1).mean(-1)/math.log(2)
            for b, (e,r) in enumerate(chunk):
                r['kl_bits'] = max(0.,float(kl[b]))
                clean_read = control_trace['observed'][b]
                r['activation_effect'] = float(abs(measured['observed'][b]-clean_read))
                if e.kind == 'qk':
                    old = float(control_trace['attention'][b])
                    new = float(measured['attention'][b])
                    r['attention_effect'] = abs(new-old)
                records.append(r)
            free(nat.dev)
        report = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            device=nat.dev, seconds=time.perf_counter()-tic, conventions=__doc__, forward_checks=checks,
            correlations=summarize(records),records=records,weights_sha256=atlas.metadata.get('weights_sha256'))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        compact = {k:{a:b for a,b in v.items() if a!='per_text'} for k,v in report['correlations'].items()}
        print(path.stem, len(records), 'seconds', report['seconds'], 'correlations', compact if path==files[-1] else '',flush=True)
        del ci, trace, baseline, actual, lp, probs
        free(nat.dev)


if __name__ == '__main__':
    main()
