"""Exact-position multihop execution and independent equation parity."""
import sys
from pathlib import Path
import math

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'experiments/oracle/toys')]
import typed_routes as routes
import planted
import zoo


@pytest.fixture(scope='module')
def fixture():
    torch.set_num_threads(1)
    # Fixed parameters from multihop014; no regeneration, activation selector,
    # ground-truth piece list, external artifact, or pretrained download.
    p = {'h0': [0, 3], 'h1': [1, 1], 'd1': 1, 'd2': 2,
         'pool': [18, 29, 23, 11, 1, 17, 4, 30, 3],
         'code': [37, 54, 40, 43, 48, 47, 53, 35, 52]}
    sd = zoo.Multihop().build(p).sd
    nat = planted.native_of(sd, zoo.CFG)
    ids = [0, 17, 11, 30, 17, 11, 4, 4, 11, 29, 4, 30, 1]
    return nat, ids


def test_native_exact_routes_and_independent_equations(fixture):
    nat, ids = fixture
    results = {}
    with torch.no_grad():
        ref = nat.reference([ids], [12])[0, -1]
        for label, intermediate, port in [('correct', 10, 'V'), ('wrong_position', 11, 'V'), ('wrong_port', 10, 'K')]:
            spec = routes.multihop_ports('L0H3', 'L1H1', 9, intermediate, 12, downstream_port=port)
            graph = routes.compile_graph(nat, len(ids), *spec)
            got = nat.run(ids, [12], nat._plan([graph], len(ids)), *nat._zero())[0, -1]
            independent = routes.independent_head_reference(nat, ids, graph)
            torch.testing.assert_close(got, independent, rtol=1e-6, atol=3e-5)
            results[label] = float((ref.exp() * (ref-got)).sum()) / math.log(2)
        assert results['correct'] < 1e-3
        assert results['wrong_position'] > results['correct'] + .1
        assert results['wrong_port'] > results['correct'] + .1


def test_root_and_cross_position_residual_edges_are_rejected(fixture):
    nat, ids = fixture
    a, b = routes.Port('L0H3', 'O', 10), routes.Port('L1H1', 'V', 11)
    with pytest.raises(ValueError, match='nonarchitectural'):
        routes.compile_graph(nat, len(ids), [a, b], [(a, b)], [])
    root = routes.Port('tok29', 'ROOT', 9)
    with pytest.raises(ValueError, match='unsupported head port'):
        routes.compile_graph(nat, len(ids), [root], [], [])
