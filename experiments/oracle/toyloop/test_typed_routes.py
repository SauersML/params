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


def test_point_sufficient_route_can_fail_matched_literal_weight_edit(fixture):
    """One fixed analytic counterexample; point KL is not intervention fidelity."""
    import native
    import weight_family

    nat, ids = fixture
    target = len(ids) - 1
    spec = routes.multihop_ports('L0H3', 'L1H1', 9, 10, target)
    sparse = routes.compile_mermaid(nat, len(ids), routes.to_mermaid(*spec, target=target))
    # Same two heads, all positions, all architecture connections. Both graph
    # selections are fixed before editing: no activation or importance reselection.
    nodes = {(n, t, c) for n in nat.names if '.attn.' in n
             for c in range(nat.C[n])
             if int(nat.units[n][c]) == {0: 3, 1: 1}[int(n.split('.')[1])]
             for t in range(len(ids))}
    parents, outputs = native.all_edges(sorted(nodes), [target])
    dense = native.Graph(nodes, parents, outputs, complete=False)
    run = lambda graph: nat.run(ids, [target], nat._plan([graph], len(ids)), *nat._zero())[0, -1]
    full = lambda: nat.vpd.target_forward(torch.tensor([ids]))[0, target].log_softmax(-1)
    kl = lambda p, q: float((p.exp() * (p-q)).sum()) / math.log(2)
    site = 'h.0.attn.q_proj'
    piece = int((nat.units[site] == 3).nonzero().flatten().min())
    st = nat.target.site(site)
    saved = {key: getattr(st, key).clone() for key in ('W', 'U', 'V', 'delta_T')}
    edit = {'kind': 'subcomponent', 'what': [site, piece], 'k': 0.}
    with torch.no_grad():
        baseline = full()
        assert kl(baseline, run(sparse)) < 1e-3
        # Prime the full-reference cache before the edit; applied must invalidate it.
        torch.testing.assert_close(nat.reference([ids], [target])[0, -1], baseline)
        with weight_family.applied(nat, edit):
            expected = saved['W'] - torch.outer(saved['U'][piece], saved['V'][:, piece])
            torch.testing.assert_close(st.W, expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(st.U[piece], saved['U'][piece] * 0., atol=0, rtol=0)
            assert torch.equal(st.V, saved['V']) and torch.equal(st.delta_T, saved['delta_T'])
            edited = full()
            torch.testing.assert_close(nat.reference([ids], [target])[0, -1], edited, atol=1e-6, rtol=1e-6)
            # Sparse positional path loses sufficiency after this exact matched edit;
            # dense same-head control captures the alternative attention contributions.
            assert kl(edited, baseline) > 10.
            assert kl(edited, run(sparse)) > 1.
            assert kl(edited, run(dense)) < 1e-3
            torch.testing.assert_close(run(sparse), routes.independent_head_reference(nat, ids, sparse),
                                       atol=3e-5, rtol=1e-6)
        for key, value in saved.items():
            assert torch.equal(getattr(st, key), value), key
        torch.testing.assert_close(full(), baseline, atol=0, rtol=0)
        torch.testing.assert_close(nat.reference([ids], [target])[0, -1], baseline, atol=1e-6, rtol=1e-6)


def test_v2_spans_exact_dense_on_path_roundtrip_and_execution(fixture):
    import native
    nat, ids = fixture
    ports = [routes.Port('L0H3', kind, t) for kind in 'QKVO' for t in range(13)]
    ports += [routes.Port('L1H1', kind, t) for kind in 'KV' for t in range(13)]
    ports += [routes.Port('L1H1', kind, 12) for kind in 'QO']
    edges = []
    for w in ports:
        for r in ports:
            try:
                routes._typed_edge(w, r)
            except ValueError:
                continue
            edges.append((w, r))
    outputs = [routes.Port(unit, 'O', 12) for unit in ('L0H3', 'L1H1')]
    text = routes.to_mermaid_v2(ports, edges, outputs, target=12)
    assert 'L0H3_K_p0to12' in text and 'L1H1_Q_p12' in text
    assert len(text.splitlines()) <= 20
    one = routes.compile_graph(nat, len(ids), ports, edges, outputs)
    two = routes.compile_mermaid_v2(nat, len(ids), text)
    assert set(one.nodes) == set(two.nodes)
    assert {r: set(ws) for r, ws in one.parents.items()} == {r: set(ws) for r, ws in two.parents.items()}
    assert set(one.out) == set(two.out)
    assert one.count() == 400 and one.edges() == 9621
    with torch.no_grad():
        run = lambda graph: nat.run(ids, [12], nat._plan([graph], len(ids)), *nat._zero())
        torch.testing.assert_close(run(one), run(two), atol=0, rtol=0)
    # v1 sparse hypotheses also roundtrip: partial adjacency falls back losslessly.
    for mid, port in [(10, 'V'), (11, 'V'), (10, 'K')]:
        spec = routes.multihop_ports('L0H3', 'L1H1', 9, mid, 12, downstream_port=port)
        expanded = routes.parse_mermaid_v2(routes.to_mermaid_v2(*spec, target=12), target=12)
        assert tuple(map(set, expanded)) == tuple(map(set, spec))
        graph = routes.compile_mermaid_v2(nat, len(ids), routes.to_mermaid_v2(*spec, target=12))
        with torch.no_grad():
            torch.testing.assert_close(run(graph), run(routes.compile_graph(nat, len(ids), *spec)), atol=0, rtol=0)


def test_v2_partial_adjacency_falls_back_without_added_edges():
    ports = [routes.Port('L0H3', 'V', t) for t in range(3)] + [routes.Port('L0H3', 'O', 2)]
    edges = [(ports[1], ports[-1])]
    text = routes.to_mermaid_v2(ports, edges, [ports[-1]], target=2)
    got = routes.parse_mermaid_v2(text, target=2)
    assert set(got[0]) == set(ports) and set(got[1]) == set(edges)
    assert 'V_p0to2' not in text


@pytest.mark.parametrize('extra', [
    'L0H3_K_p1',  # overlap
    'L0H3_K_p0to12',  # duplicate
    'L0H3_K_p13',  # outside actual input
    'L0H3_K_p12to0',
    'L0H3_K_p2to2',
    'tok29_ROOT_p9',
    'L0H3_Q_p0to12 --> L0H3_O_p0to12',
    'L1H1_O_p12 --> L0H3_K_p0to12',
    'L0H3_V_p0to12 -->|restrict| L0H3_O_p0to12',
])
def test_v2_rejects_overlap_bounds_and_unsupported_relations(extra):
    ports = [routes.Port('L0H3', kind, t) for kind in 'QKVO' for t in range(13)]
    ports += [routes.Port('L1H1', 'O', 12)]
    text = routes.to_mermaid_v2(ports, [], [], target=12)
    with pytest.raises(ValueError):
        routes.parse_mermaid_v2(text + extra + '\n', target=12)


def test_v2_rejects_empty_pair_expansion_and_duplicate_edges():
    header = 'flowchart LR\n' + routes._CONTRACT_V2 + '\n'
    for text in [
        header + 'L0H3_O_p0\nL1H1_V_p1\nOUT_p2\nL0H3_O_p0 --> L1H1_V_p1\n',
        header + 'L0H3_Q_p0\nL0H3_O_p1\nOUT_p2\nL0H3_Q_p0 --> L0H3_O_p1\n',
        header + 'L0H3_V_p0to1\nL0H3_O_p2\nOUT_p2\n'
        + 'L0H3_V_p0to1 --> L0H3_O_p2\n' * 2,
    ]:
        with pytest.raises(ValueError):
            routes.parse_mermaid_v2(text, target=2)
