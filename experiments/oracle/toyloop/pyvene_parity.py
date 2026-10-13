"""Optional independent scalar-port interchange parity (CPU, no model download/SVD).

Clone stanfordnlp/pyvene and check out PYVENE_COMMIT, then run:
  python experiments/oracle/toyloop/pyvene_parity.py --pyvene-source PATH --out result.json
Pyvene is imported only from that verified checkout; no production dependency is added.
This checks full-background scalar activation swaps, not sparse graph/token-role semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

PYVENE_COMMIT = '9e333904dcf9e597ca76170010d17f4d4580de8d'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_source(path):
    path = Path(path).resolve()
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PYVENE_COMMIT:
        raise ValueError(f'Pyvene requires exact commit {PYVENE_COMMIT}, got {head}')
    changes = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if changes:
        raise ValueError('Pyvene tracked source changed; restore the pinned checkout')
    return path


def run(source):
    import math
    import torch
    sys.path[:0] = [str(source), str(ROOT / 'method/oracle'), str(ROOT / 'method/vpd')]
    import pyvene as pv
    import native
    import vpd_model
    from test_goal_eval import tiny_native
    if not Path(pv.__file__).resolve().is_relative_to(source):
        raise ValueError('Pyvene import did not resolve to the verified source')
    torch.set_num_threads(1)
    # Existing seeded analytic fixture; near=None avoids decomposition/SVD entirely.
    nat = tiny_native(seed=23, layers=1, d=8, heads=2, vocab=16, C=3)

    class Reference(torch.nn.Module):
        """Independent forward math, with scalar z@V exposed as Pyvene hook modules."""
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.config = SimpleNamespace(hidden_size=3)
            self.ports = torch.nn.ModuleDict({k: torch.nn.Identity() for k in ('q', 'k', 'v', 'o', 'fc', 'down')})

        def forward(self, input_ids):
            tg = nat.target
            length, heads, hd = input_ids.shape[-1], tg.n_head, tg.hd
            x = tg.wte[input_ids]

            def project(kind, z, port):
                site = tg.site(native.site_name(0, kind))
                activation = self.ports[port](z @ site.V)
                return activation @ site.U + z @ site.delta_T

            z = vpd_model.rms(x, tg.norms[0], tg.eps)
            q, k, v = [project(kind, z, port).reshape(1, length, heads, hd).transpose(1, 2)
                       for kind, port in (('q_proj', 'q'), ('k_proj', 'k'), ('v_proj', 'v'))]
            q, k = tg._rope(q, length), tg._rope(k, length)
            causal = torch.ones(length, length, dtype=torch.bool).tril()
            p = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float('-inf')).softmax(-1)
            attention = (p @ v).transpose(1, 2).reshape(1, length, -1)
            x = x + project('o_proj', attention, 'o')
            z = vpd_model.rms(x, tg.norms[1], tg.eps)
            pre = project('c_fc', z, 'fc')
            x = x + project('down_proj', vpd_model.gelu_tanh(pre), 'down')
            return torch.log_softmax(tg.unembed(tg.norm(x, -1)).float(), -1)

    base, donor, target = [1, 2, 3, 4], [1, 8, 3, 9], 3
    reference = Reference()
    plan, one, recorded = nat._full_plan(4, 1), nat._one(), {}
    with torch.no_grad():
        baseline = nat.run(base, [target], plan, *one)
        nat.run(donor, [target], plan, *one, record=recorded)
        ref_baseline = reference(torch.tensor([base]))[:, [target]]
    cases = []
    for kind, port, position in [('q_proj', 'q', 3), ('k_proj', 'k', 1), ('v_proj', 'v', 1),
                                 ('v_proj', 'v', 2), ('c_fc', 'fc', 3)]:
        name = native.site_name(0, kind)
        mask = torch.zeros(1, 4, 3, dtype=torch.bool)
        mask[:, position, :] = True
        with torch.no_grad():
            expected = nat.run(base, [target], plan, *one, patch={name: (mask, recorded[name])})
            hooked = pv.IntervenableModel({'component': f'ports.{port}.output', 'intervention': lambda b, s: s}, model=reference)
            _, got = hooked(base={'input_ids': torch.tensor([base])}, sources=[{'input_ids': torch.tensor([donor])}],
                            unit_locations={'sources->base': ([[[position]]], [[[position]]])})
            got = got[:, [target]]
        cases.append({'matrix': name, 'port': port, 'source_position': position, 'base_position': position,
                      'scalar_components': list(range(3)), 'maxabs_logprob_error': float((expected - got).abs().max()),
                      'intervention_maxabs_effect': float((baseline - expected).abs().max()),
                      'native_logprobs': expected.tolist(), 'pyvene_logprobs': got.tolist()})
    tolerance = 3e-5
    error = float((baseline - ref_baseline).abs().max())
    if error >= tolerance or any(c['maxabs_logprob_error'] >= tolerance for c in cases):
        raise ValueError(f'independent execution parity failed: baseline={error}, cases={cases}')
    if cases[2]['intervention_maxabs_effect'] <= .1 or cases[3]['intervention_maxabs_effect'] >= tolerance:
        raise ValueError('wrong-position V control did not distinguish the predeclared positions')
    h = hashlib.sha256()
    for name, tensor in sorted(nat.target.state_dict().items()):
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.detach().numpy().tobytes())
    # U/V may be ordinary attributes rather than registered state_dict entries.
    for name in sorted(nat.C):
        for label in ('U', 'V', 'delta_T'):
            tensor = getattr(nat.target.site(name), label)
            h.update((name + label).encode()); h.update(tensor.detach().numpy().tobytes())
    return {'schema': 1, 'pyvene_commit': PYVENE_COMMIT, 'pyvene_source': str(source), 'torch_version': torch.__version__,
            'fixture': {'seed': 23, 'layers': 1, 'd': 8, 'heads': 2, 'vocab': 16, 'C': 3, 'near': None},
            'weights_and_basis_sha256': h.hexdigest(), 'base_ids': base, 'donor_ids': donor, 'target_position': target,
            'baseline_maxabs_logprob_error': error, 'tolerance': tolerance, 'cases': cases,
            'source_sha256': {str(p.relative_to(ROOT)): file_sha(p) for p in
                              (Path(__file__), ROOT / 'method/oracle/native.py', ROOT / 'method/oracle/test_goal_eval.py',
                               ROOT / 'method/vpd/vpd_model.py')},
            'scope': 'Independent full-background scalar z@V interchange at exact ports/positions; all remainders retained. '
                     'Does not certify sparse graph execution, token selectors, direct embedding cuts, English, or causal abstraction.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pyvene-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(verified_source(args.pyvene_source))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'cases': len(result['cases']),
                      'maxabs_error': max(c['maxabs_logprob_error'] for c in result['cases'])}))


if __name__ == '__main__':
    main()
