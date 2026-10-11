"""Exact synthetic weight paths and on-disk round trip, without mocks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from method.vpd.connectome import Connectome, build, edges, rotate, strengths
from method.vpd.vpd_model import Target, VPD, site_names


def tiny():
    cfg = dict(n_layer=2, n_head=2, n_embd=4, n_key_value_heads=2,
               rotary_dim=2, rms_norm_eps=1e-6, n_ctx=8, rotary_base=10000)
    gen = torch.Generator().manual_seed(12)
    rand = lambda *s: torch.randn(*s, generator=gen)
    sd = {'wte.weight': rand(5, 4), 'ln_f.weight': torch.tensor([1., 2., 3., 4.])}
    uv = {}
    for i in range(2):
        for j in (1, 2):
            sd[f'h.{i}.rms_{j}.weight'] = torch.tensor([2., 3., 4., 5.]) + j
    for n in site_names(2):
        din, dout = (4, 6) if n.endswith('c_fc') else ((6, 4) if n.endswith('down_proj') else (4, 4))
        sd[n+'.weight'] = rand(dout, din)
        uv[n] = (rand(3, dout), rand(din, 3))
    t = Target(sd, cfg)
    VPD(t, None, uv)
    return t


class TestConnectome(unittest.TestCase):
    def test_paths_and_roundtrip(self):
        t = tiny()
        names = site_names(2)
        es = list(edges(t, names))
        self.assertEqual({e.kind for e in es}, {'residual', 'ov', 'qk', 'mlp', 'embedding', 'unembedding'})
        self.assertTrue(any(e.source == 'h.0.attn.o_proj' and e.sink == 'h.0.mlp.c_fc' for e in es))
        with tempfile.TemporaryDirectory() as d:
            meta = build(t, names, d, topk=64, batch=2)
            c = Connectome(d)
            self.assertEqual(c.metadata, meta)
            for e in es:
                src, dst = t.site(e.source) if e.source in names else None, t.site(e.sink) if e.sink in names else None
                for a in range(len(e.left)):
                    for b in range(len(e.right)):
                        if e.kind == 'residual':
                            layer = int(e.sink.split('.')[1])
                            g = t.norms[2*layer + int(e.sink.endswith('c_fc'))]
                            lhs, rhs = src.U[a], g * dst.V[:, b]
                        elif e.kind == 'ov':
                            sl = slice(e.head*2, (e.head+1)*2)
                            lhs, rhs = src.U[a, sl], dst.V[sl, b]
                        elif e.kind == 'qk':
                            sl = slice(e.head*2, (e.head+1)*2)
                            lhs, rhs = src.U[a, sl], dst.U[b, sl]
                        elif e.kind == 'mlp':
                            lhs, rhs = src.U[a], dst.V[:, b]
                        elif e.kind == 'embedding':
                            layer = int(e.sink.split('.')[1])
                            g = t.norms[2*layer + int(e.sink.endswith('c_fc'))]
                            lhs, rhs = t.wte[a], g * dst.V[:, b]
                        else:
                            lhs, rhs = src.U[a] * t.ln_f, t.wte[b]
                        expected = float(torch.dot(lhs, rhs))
                        for name, idx, other, oi, direction in [(e.source,a,e.sink,b,'downstream'), (e.sink,b,e.source,a,'upstream')]:
                            row = c.partners(name, idx, e.kind+'.'+direction)
                            hits = np.flatnonzero((row['site'] == c.names.index(other)) & (row['index'] == oi) & (row['head'] == e.head))
                            self.assertEqual(len(hits), 1)
                            k = hits[0]
                            self.assertAlmostEqual(float(row['raw'][k]), expected, places=4)
                            self.assertAlmostEqual(float(row['cosine'][k]), expected / float(lhs.norm()*rhs.norm()), places=5)
                            if e.kind == 'qk':
                                curve = row['offset_raw'][k]
                                for offset in (0, 1, 32):
                                    self.assertAlmostEqual(float(curve[offset]), float(rotate(lhs, offset) @ rhs), places=4)
            with self.assertRaises(IndexError):
                c.partners(names[0], -1, 'qk.upstream')
            self.assertEqual(len(c.partners(names[0], 0, 'mlp.upstream')['raw']), 0)

    def test_zero_and_concentration(self):
        raw, cos = strengths(torch.tensor([[0.,0.],[3.,4.]]), torch.tensor([[1.,0.],[0.,2.]]))
        torch.testing.assert_close(raw, torch.tensor([[0.,0.],[3.,8.]]))
        torch.testing.assert_close(cos, torch.tensor([[0.,0.],[.6,.8]]))
        t = tiny()
        with tempfile.TemporaryDirectory() as d:
            build(t, site_names(2), d, topk=1, batch=3)
            c = Connectome(d)
            with np.load(Path(d)/c.metadata['tables']['h.0.mlp.c_fc.mlp.downstream']) as f:
                mag = (t.site('h.0.mlp.c_fc').U @ t.site('h.0.mlp.down_proj').V).abs().numpy()
                for i in range(3):
                    vals = np.sort(mag[i])[::-1]
                    self.assertEqual(int(f['k90'][i]), int(np.searchsorted(vals.cumsum(), .9*vals.sum())+1))
                    self.assertEqual(int(f['index'][i,0]), int(np.argmax(mag[i])))


class TestPathPatching(unittest.TestCase):
    def test_real_forward_and_single_read_interventions(self):
        from experiments.vpd.validate_connectome import run, removal
        from method.vpd.vpd_model import rms, gelu_tanh
        t = tiny()
        ids = torch.tensor([[0, 2, 1]])
        with torch.no_grad():
            logits, trace = run(t, ids, [2], collect=True)
            torch.testing.assert_close(logits, t(ids)[:, [2]], atol=1e-5, rtol=1e-5)
            # Production Target with a single real output hook is an independent
            # execution of the isolated coefficient intervention.
            for e in edges(t, site_names(2)):
                if e.kind not in ('residual','mlp','ov'):
                    continue
                s, pos, c, d = (0 if e.kind=='ov' else 2), 2, 1, 2
                edit, predictor, mechanism = removal(t,trace,e.source,e.sink,e.kind,c,d,s,pos,e.head)
                src, dst = t.site(e.source), t.site(e.sink)
                a = trace['inputs'][e.source][s] @ src.V[:,c]
                layer = int(e.sink.split('.')[1])
                if e.kind=='residual':
                    x = trace['residual'][e.sink][pos]
                    g = t.norms[2*layer+int(e.sink.endswith('c_fc'))]
                    expected = (rms(x-a*src.U[c],g,t.eps)-rms(x,g,t.eps)) @ dst.V[:,d]
                elif e.kind=='mlp':
                    x = trace['fc'][layer][pos]
                    expected = (gelu_tanh(x-a*src.U[c])-gelu_tanh(x)) @ dst.V[:,d]
                else:
                    sl = slice(e.head*2,(e.head+1)*2)
                    expected = -a*trace['pattern'][layer][e.head,pos,s]*(src.U[c,sl] @ dst.V[sl,d])
                self.assertAlmostEqual(edit.delta,float(expected),places=5)
                def hook(module, args, out):
                    out = out.clone()
                    out[:,pos] += expected * module.U[d]
                    return out
                handle = dst.register_forward_hook(hook)
                try:
                    reference = t(ids)[:,[2]]
                finally:
                    handle.remove()
                patched, observed = run(t,ids,[2],[edit])
                torch.testing.assert_close(patched,reference,atol=3e-5,rtol=3e-5)
                clean_read = trace['inputs'][e.sink][pos] @ dst.V[:,d]
                torch.testing.assert_close(observed['observed'][0]-clean_read,expected,atol=1e-5,rtol=1e-5)

    def test_qk_score_and_rotary(self):
        from experiments.vpd.validate_connectome import run, removal
        t = tiny()
        ids = torch.tensor([[0, 2, 1]])
        with torch.no_grad():
            _, trace = run(t,ids,[2],collect=True)
            edit, _, _ = removal(t,trace,'h.0.attn.q_proj','h.0.attn.k_proj','qk',1,2,2,0,1)
            _, patched = run(t,ids,[2],[edit])
            old = trace['pattern'][0][1,2,0]
            expected = old*torch.exp(torch.tensor(edit.delta))/(1-old+old*torch.exp(torch.tensor(edit.delta)))
            torch.testing.assert_close(patched['attention'][0],expected)
            clean_read = trace['inputs'][edit.sink][0] @ t.site(edit.sink).V[:,2]
            torch.testing.assert_close(patched['observed'][0],clean_read)
            q, k = t.site(edit.source).U[1,2:],t.site(edit.sink).U[2,2:]
            # Offset sign agrees with the actual production RoPE, including
            # negative offsets and a nonzero key absolute position.
            for delta in (-2,0,2):
                s, p = 3, 3+delta
                qr = t._rope(q.expand(1,2,8,2),8)[0,0,p]
                kr = t._rope(k.expand(1,2,8,2),8)[0,0,s]
                torch.testing.assert_close(qr@kr,rotate(q,delta)@k)


if __name__ == '__main__':
    unittest.main()
