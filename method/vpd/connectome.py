"""Weight-only VPD wiring. Run with MPD_MEM_GIB=16 python -m method.vpd.connectome.

partners(name, idx, 'ov.downstream') returns compact arrays (site IDs, component
indices, head, signed raw, cosine). Without a direction it returns both directions.
Head-specific OV/QK partners compete for the same top-k budget. QK curves use
q-position minus k-position in 0..32; no 1/sqrt(hd) attention-score scaling.
Residual reads include norm gain but not input-dependent RMS denominators. Logit
writes include final norm gain, likewise omitting its denominator. MLP is
pre-GELU. These are geometric couplings, not claims of causal faithfulness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

DEFAULT = Path.home() / 'mpd-data/atlas/vpd4l/connectome'
READERS = ('q_proj', 'k_proj', 'v_proj', 'c_fc')
WRITERS = ('o_proj', 'down_proj')


def rotate(x, offset, base=10000.0):
    """Model's rotate-half RoPE on last dimension (offset may be negative)."""
    d = x.shape[-1]
    if d % 2:
        raise ValueError('RoPE dimension must be even')
    angle = offset / base ** (torch.arange(d // 2, device=x.device, dtype=x.dtype) / (d / 2))
    angle = angle.repeat(2)
    return x * angle.cos() + torch.cat((-x[..., d // 2:], x[..., :d // 2]), -1) * angle.sin()


def strengths(left, right):
    """Rows are directions; zero directions have cosine zero."""
    raw = left @ right.T
    den = left.norm(dim=-1)[:, None] * right.norm(dim=-1)[None, :]
    return raw, torch.where(den > 0, raw / den.clamp_min(1e-30), 0)


@dataclass
class Edge:
    source: str
    sink: str
    kind: str
    left: torch.Tensor
    right: torch.Tensor
    head: int = -1


def edges(target, names):
    """Generate real weight paths, including same-layer attention -> MLP."""
    name = lambda i, k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"
    for i in range(target.n_layer):
        for wk in WRITERS:
            wn = name(i, wk)
            for j in range(i, target.n_layer):
                for rk in READERS:
                    if j == i and not (wk == 'o_proj' and rk == 'c_fc'):
                        continue
                    rn = name(j, rk)
                    gain = target.norms[2 * j + int(rk == 'c_fc')]
                    yield Edge(wn, rn, 'residual', target.site(wn).U, (gain[:, None] * target.site(rn).V).T)
        for h in range(target.n_head):
            sl = slice(h * target.hd, (h + 1) * target.hd)
            v, o, q, k = [name(i, x) for x in ('v_proj', 'o_proj', 'q_proj', 'k_proj')]
            yield Edge(v, o, 'ov', target.site(v).U[:, sl], target.site(o).V[sl].T, h)
            yield Edge(q, k, 'qk', target.site(q).U[:, sl], target.site(k).U[:, sl], h)
        fc, down = name(i, 'c_fc'), name(i, 'down_proj')
        yield Edge(fc, down, 'mlp', target.site(fc).U, target.site(down).V.T)
    for n in names:
        kind, layer = n.split('.')[-1], int(n.split('.')[1])
        if kind in READERS:
            gain = target.norms[2 * layer + int(kind == 'c_fc')]
            yield Edge('embedding', n, 'embedding', target.wte, (gain[:, None] * target.site(n).V).T)
        if kind in WRITERS:
            yield Edge(n, 'unembedding', 'unembedding', target.site(n).U * target.ln_f, target.wte)


class Connectome:
    def __init__(self, root=DEFAULT):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'metadata.json').read_text())
        self.names = self.metadata['names']
        self._cache = {}

    def partners(self, name, idx, kind):
        """kind: edge type optionally suffixed .upstream or .downstream.

        Missing edge types return empty arrays. `site` indexes self.names; special
        embedding/unembedding indices are tokenizer IDs. Arrays ordered by |raw|.
        """
        if '.' not in kind:
            return {d: self.partners(name, idx, f'{kind}.{d}') for d in ('upstream', 'downstream')}
        key = f'{name}.{kind}'
        if name not in self.names:
            raise KeyError(name)
        if idx < 0 or idx >= self.metadata['counts'][name]:
            raise IndexError(idx)
        if key not in self.metadata['tables']:
            return {k: np.empty(0, dtype=d) for k, d in [('site', 'int16'), ('index', 'int32'), ('head', 'int8'), ('raw', 'float32'), ('cosine', 'float32')]}
        if key not in self._cache:
            with np.load(self.root / self.metadata['tables'][key]) as f:
                self._cache[key] = dict(f)
        return {k: a[idx] for k, a in self._cache[key].items() if k != 'k90'}


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def build(target, names, root=DEFAULT, topk=64, batch=128):
    """Batched GPU products, bounded row blocks, exact full-population k90.

    Global histogram/moments count each directed edge once (downstream only).
    k90 is separately reported for both directions, never estimated from top-k.
    Token endpoints also have compact tables, enabling source/sink lookup.
    """
    if topk < 1 or batch < 1:
        raise ValueError('topk and batch must be positive')
    start = time.perf_counter()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = list(names) + ['embedding', 'unembedding']
    counts = {n: int(target.site(n).U.shape[0]) for n in names[:-2]}
    counts.update(embedding=len(target.wte), unembedding=len(target.wte))
    grouped = defaultdict(list)
    for e in edges(target, names[:-2]):
        grouped[(e.source, e.kind, 'downstream')].append((e.sink, e.left, e.right, e.head))
        grouped[(e.sink, e.kind, 'upstream')].append((e.source, e.right, e.left, e.head))
    meta = dict(version=1, names=names, counts=counts, topk=topk, tables={}, stats={}, device=str(target.wte.device),
                rotary_base=float(target.cfg['rotary_base']), conventions=__doc__)
    bins = np.r_[0., np.logspace(-10, 4, 57), np.inf]
    totals = {}
    for (n, kind, direction), blocks in grouped.items():
        tic = time.perf_counter()
        size = counts[n]
        psite = np.concatenate([np.full(len(r), names.index(p), np.int16) for p, l, r, h in blocks])
        pidx = np.concatenate([np.arange(len(r), dtype=np.int32) for p, l, r, h in blocks])
        phead = np.concatenate([np.full(len(r), h, np.int8) for p, l, r, h in blocks])
        k = min(topk, len(pidx))
        table = {x: np.empty((size, k), dtype=d) for x, d in [('site', 'int16'), ('index', 'int32'), ('head', 'int8'), ('raw', 'float32'), ('cosine', 'float32')]}
        k90 = np.empty(size, np.int32)
        if kind == 'qk':
            table['offset_raw'] = np.empty((size, k, 33), np.float32)
        stat = totals.setdefault(kind, dict(count=0, sum=0., sum_abs=0., sum_sq=0., max_abs=0., histogram=np.zeros(len(bins)-1, np.int64)))
        for a in range(0, size, batch):
            b = min(a + batch, size)
            raws, coses = [], []
            for p, l, r, h in blocks:
                raw, cos = strengths(l[a:b], r)
                raws.append(raw.cpu().numpy())
                coses.append(cos.cpu().numpy())
            raw, cos = np.concatenate(raws, 1), np.concatenate(coses, 1)
            mag = np.abs(raw)
            # Partition keeps the top-k operation linear; sort all magnitudes only
            # for the exact concentration statistic (one row block at a time).
            ix = np.argpartition(-mag, k-1, axis=1)[:, :k]
            ix = np.take_along_axis(ix, np.argsort(-np.take_along_axis(mag, ix, 1), axis=1, kind='stable'), 1)
            table['site'][a:b], table['index'][a:b], table['head'][a:b] = psite[ix], pidx[ix], phead[ix]
            table['raw'][a:b], table['cosine'][a:b] = np.take_along_axis(raw, ix, 1), np.take_along_axis(cos, ix, 1)
            sorted_abs = np.sort(mag, axis=1)[:, ::-1]
            sums = sorted_abs.sum(1, dtype=np.float64)
            cumulative = np.cumsum(sorted_abs, axis=1, dtype=np.float64)
            k90[a:b] = np.where(sums > 0, (cumulative < .9 * sums[:, None]).sum(1) + 1, 0)
            if direction == 'downstream':
                stat['count'] += raw.size
                stat['sum'] += float(raw.sum(dtype=np.float64))
                stat['sum_abs'] += float(mag.sum(dtype=np.float64))
                stat['sum_sq'] += float(np.square(raw.astype(np.float64)).sum())
                stat['max_abs'] = max(stat['max_abs'], float(mag.max()))
                stat['histogram'] += np.histogram(mag, bins)[0]
            if kind == 'qk':
                # Curves only for retained edges, batched over rows and partners.
                p, l, r, h = blocks[0]
                offset_start = 0
                for p, l, r, h in blocks:
                    chosen = (ix >= offset_start) & (ix < offset_start + len(r))
                    rr, cc = np.nonzero(chosen)
                    if len(rr):
                        local = ix[rr, cc] - offset_start
                        lhs, rhs = l[a:b][torch.as_tensor(rr, device=l.device)], r[torch.as_tensor(local, device=r.device)]
                        for delta in range(33):
                            # q downstream: rotate left; k upstream: rotate right.
                            val = (rotate(lhs, delta, meta['rotary_base']) * rhs).sum(-1) if direction == 'downstream' else (lhs * rotate(rhs, delta, meta['rotary_base'])).sum(-1)
                            table['offset_raw'][a + rr, cc, delta] = val.cpu().numpy()
                    offset_start += len(r)
        key = f'{n}.{kind}.{direction}'
        file = f'{len(meta["tables"]):03d}.npz'
        np.savez_compressed(root / file, **table, k90=k90)
        meta['tables'][key] = file
        meta['stats'][key] = dict(partners=len(pidx), k90_quantiles=np.quantile(k90, [0, .25, .5, .75, .9, 1]).tolist(), seconds=time.perf_counter()-tic)
        print(key, meta['stats'][key], flush=True)
        if target.wte.device.type == 'mps':
            torch.mps.empty_cache()
    meta['strength_distribution'] = {}
    for kind, s in totals.items():
        count = s.pop('count')
        mean = s.pop('sum') / count
        meta['strength_distribution'][kind] = dict(count=count, mean=mean, mean_abs=s.pop('sum_abs')/count,
            std=float(np.sqrt(max(0, s.pop('sum_sq')/count - mean*mean))), max_abs=s['max_abs'], histogram=s['histogram'].tolist(),
            abs_bin_edges=[float(x) if np.isfinite(x) else 'inf' for x in bins])
    meta['seconds'] = time.perf_counter() - start
    (root / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    from method.oracle.native import Native
    from method.vpd.vpd_model import TARGET_DIR, VPD_PTH
    nat = Native()
    meta = build(nat.target, nat.names, args.out, batch=args.batch)
    meta['weights_sha256'] = {str(p): _sha(p) for p in [TARGET_DIR/'model_step_99999.safetensors', VPD_PTH]}
    (args.out/'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('TOTAL seconds', meta['seconds'], flush=True)


if __name__ == '__main__':
    main()
