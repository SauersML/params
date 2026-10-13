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
            text = routes.to_mermaid(*spec, target=12)
            parsed = routes.parse_mermaid(text, target=12)
            assert set(parsed[0]) == set(spec[0])
            assert set(parsed[1]) == set(spec[1])
            assert set(parsed[2]) == set(spec[2])
            graph = routes.compile_mermaid(nat, len(ids), text)
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


@pytest.mark.parametrize('extra', [
    'L0H3_Q_p10',  # duplicated disconnected Q declaration
    'tok29_ROOT_p9',
    'L0H3_V_p9 -->|restrict| L0H3_O_p10',
    'L0H3_Q_p10 --> L0H3_O_p10',  # Q dependency is implicit, not a routing edge
    'L0H3_V_p9 --> L0H3_O_p10 --> OUT_p12',
    'English: copy this token',
    '%% ignored claim',
    'classDef hidden fill:red',
    'L0H3_V_p8 --> L0H3_O_p10',  # undeclared endpoint
    'L0H3_V_p9 --> L0H3_O_p10',  # duplicated edge
    'OUT_p12 --> L0H3_Q_p10',
    'L0H3_O_p10 --> OUT_p12',  # wrong output position
    'L00H3_Q_p10',
])
def test_strict_text_rejects_unsupported_or_duplicate_syntax(extra):
    spec = routes.multihop_ports('L0H3', 'L1H1', 9, 10, 12)
    text = routes.to_mermaid(*spec, target=12)
    with pytest.raises(ValueError):
        routes.parse_mermaid(text + extra + '\n', target=12)


def test_roundtrip_preserves_disconnected_ports_and_rejects_missing_contract():
    spec = routes.multihop_ports('L0H3', 'L1H1', 9, 10, 12)
    text = routes.to_mermaid(*spec, target=12)
    selected, edges, outputs = routes.parse_mermaid(text, target=12)
    # Q and every K remain selected even though they have no explicit incoming edges.
    assert len([p for p in selected if p.port == 'K']) == 24
    assert len([p for p in selected if p.port == 'Q']) == 2
    assert routes.to_mermaid(selected, edges, outputs, target=12) == text
    with pytest.raises(ValueError, match='header/contract'):
        routes.parse_mermaid(text.replace('attention=causal_softmax', 'attention=restricted'), target=12)
    with pytest.raises(ValueError, match='duplicate selected'):
        routes.to_mermaid(spec[0] + [spec[0][0]], spec[1], spec[2], target=12)
