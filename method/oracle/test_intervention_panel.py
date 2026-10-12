"""Common-panel unit proofs on the existing tiny model; no oracle calibration claim."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry
import goal_eval
import intervention_panel
import mech
import native
from test_goal_eval import tiny_native


@pytest.fixture
def panel_tiny():
    # The existing default toy avoids SVD/Accelerate. A dense reachable active
    # reference is exact for this unit test, making both score anchors identifiable.
    nat = tiny_native()
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), chunk=4)
    tx = ge.text([2, 5, 3, 7, 4, 6, 8, 9], [7], seed=5)
    tx.vpd = nat.reachable(tx.T, tx.targets)
    return nat, ge, tx


def _masked(base, removed):
    masks = {n: value.clone() for n, value in base.items()}
    for n, t, c in removed:
        masks[n][t, c] = False
    return masks


def test_common_panel_matches_direct_masked_model_and_explicit_graph(panel_tiny):
    nat, ge, tx = panel_tiny
    nodes = sorted(native.Graph(masks=tx.vpd).node_set())
    picked = nodes[::9]
    g = native.Graph(picked, *native.all_edges(picked, tx.targets))
    g.tok_edges = [(nd[1], nd) for nd in picked if native._name_layer_kind(nd[0])[1] in mech.RESID_READERS]
    g.tok_positions = sorted({t for t, _ in g.tok_edges})
    g.tok_out = list(tx.targets)
    rows = intervention_panel.evaluate(ge, tx, [native.Graph(), native.Graph(masks=tx.vpd), g], draws=4)
    panel = intervention_panel.bank(tx, draws=4)
    full = {n: torch.ones(tx.T, nat.C[n], dtype=torch.bool) for n in nat.names}
    zero = {n: torch.zeros_like(m) for n, m in full.items()}
    lm = ge.runs(tx.ids, tx.targets, [_masked(full, d) for d in panel['removed']], True)
    lv = ge.runs(tx.ids, tx.targets, [_masked(tx.vpd, d) for d in panel['removed']], True)
    le = ge.runs(tx.ids, tx.targets, [zero], True).expand_as(lm)
    canonical = goal_eval._c_graph(g, tx.targets, nat.C)
    variants = []
    for d in panel['removed']:
        h = goal_eval._drop_nodes(canonical, set(d))
        h.tok_edges = [(t, r) for t, r in canonical.tok_edges if r not in set(d)]
        h.tok_positions = list(canonical.tok_positions)
        h.tok_out = list(canonical.tok_out)
        variants.append(h)
    lc = ge._graph_logq(tx.ids, tx.targets, variants)
    e, v, k, w = [carry.kl(lm, x).tolist() for x in (le, lv, lc, tx.ref.expand_as(lm))]
    for row in rows:
        assert row['resolved']
        for j, draw in enumerate(row['draws']):
            assert draw['removed'] == panel['removed'][j]
            assert draw['model_top'] == lm[j].argmax(-1).tolist()
            for key, expected in [('kl_empty',e[j]), ('kl_vpd',v[j]), ('W',w[j])]:
                assert draw[key] == pytest.approx(expected, rel=1e-4, abs=2e-5)
    assert rows[0]['raw_score'] == pytest.approx(0.0, abs=2e-5)
    assert rows[1]['raw_score'] == pytest.approx(1.0, abs=2e-5)
    assert rows[2]['kl_candidate'] == pytest.approx(np.mean(k), rel=1e-4, abs=2e-5)
    for j, draw in enumerate(rows[2]['draws']):
        assert draw['kl_candidate'] == pytest.approx(k[j], rel=1e-4, abs=2e-5)
    assert rows[2]['raw_score'] == pytest.approx(goal_eval._fid_raw(np.mean(e), np.mean(k), np.mean(v)), rel=1e-4, abs=2e-5)


def test_candidate_order_batching_and_split(panel_tiny):
    _, ge, tx = panel_tiny
    nodes = sorted(native.Graph(masks=tx.vpd).node_set())
    graphs = [native.Graph(), native.Graph(nodes[::7], complete=True), native.Graph(masks=tx.vpd)]
    batch = intervention_panel.evaluate(ge, tx, graphs, draws=4)
    reverse = intervention_panel.evaluate(ge, tx, graphs[::-1], draws=4)[::-1]
    for g, x, y in zip(graphs, batch, reverse):
        single = intervention_panel.evaluate(ge, tx, [g], draws=4)[0]
        assert x == y == single
    other = intervention_panel.bank(tx, split='search', draws=4)
    assert other['bank_sha256'] != batch[0]['bank_sha256']
    assert other['removed'] != [d['removed'] for d in batch[0]['draws']]
    assert all(x['bank_sha256'] == batch[0]['bank_sha256'] for x in batch)


def test_blocks_and_stranded_nodes_use_fidelity_canonicalization(panel_tiny):
    nat, ge, tx = panel_tiny
    # A block into the prediction is legal; an isolated MLP input is stranded.
    block = (nat.names[-1], tx.targets[0], goal_eval.BLOCK)
    stranded = (nat.names[0], 0, 0)
    g = native.Graph([block, stranded], {}, [block])
    canonical = goal_eval._c_graph(g, tx.targets, nat.C)
    raw, explicit = intervention_panel.evaluate(ge, tx, [g, canonical], draws=3)
    assert raw['kl_candidate'] == explicit['kl_candidate']
    assert raw['raw_score'] == explicit['raw_score']


def test_explicit_query_key_pairs_and_token_edges_match_direct_graph(panel_tiny):
    _, ge, tx = panel_tiny
    q = (native.site_name(0, 'q_proj'), 7, 0)
    k = (native.site_name(0, 'k_proj'), 0, 0)
    o = (native.site_name(0, 'o_proj'), 7, 0)
    g = native.Graph([q, k, o], {}, [o])
    g.pairs = {(0, 7): [(q, k)]}
    g.tok_edges, g.tok_positions = [(7, q), (0, k)], [0, 7]
    g.tok_out = [7]
    row = intervention_panel.evaluate(ge, tx, [g], draws=4)[0]
    full = {n: torch.ones(tx.T, ge.nat.C[n], dtype=torch.bool) for n in ge.nat.names}
    canonical = goal_eval._c_graph(g, tx.targets, ge.nat.C)
    variants = []
    for d in row['draws']:
        removed = {tuple(nd) for nd in d['removed']}
        h = goal_eval._drop_nodes(canonical, removed)
        h.tok_edges = [(t, r) for t, r in canonical.tok_edges if r not in removed]
        h.tok_positions, h.tok_out = [0, 7], [7]
        variants.append(h)
    lm = ge.runs(tx.ids, tx.targets, [_masked(full, d['removed']) for d in row['draws']], True)
    lc = ge._graph_logq(tx.ids, tx.targets, variants)
    assert row['kl_candidate'] == pytest.approx(float(carry.kl(lm, lc).mean()), rel=1e-4, abs=2e-5)


def test_unresolved_reference_is_common_and_bootstrap_is_paired(panel_tiny):
    _, ge, tx = panel_tiny
    panel = intervention_panel.bank(tx, draws=3)
    # Inject reference statistics only to exercise the globally unresolved path.
    ref = intervention_panel.evaluate(ge, tx, [native.Graph()], draws=3)[0]
    cached = tx._common_node_panels[panel['bank_sha256']]
    cached['empty'][:] = 1.0
    cached['vpd'][:] = 2.0
    rows = intervention_panel.evaluate(ge, tx, [native.Graph(), native.Graph(masks=tx.vpd)], draws=3)
    assert all(not r['resolved'] and r['raw_score'] is None and r['eps'] is None for r in rows)
    assert all(r['kl_empty'] == 1.0 and r['kl_vpd'] == 2.0 for r in rows)
    # Inert interventions (W=0) do not enter the estimator. E>V remains identifiable.
    candidate, empty, vpd = np.array([0.2, 0.8, 0.4]), np.array([2., 3., 4.]), np.array([.1,.2,.3])
    eps, count = intervention_panel._eps(goal_eval, candidate, empty, vpd)
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(goal_eval.EPS_BOOT):
        ix = rng.integers(3, size=3)
        stats.append(goal_eval._fid_raw(np.mean(empty[ix]), np.mean(candidate[ix]), np.mean(vpd[ix])))
    assert count == goal_eval.EPS_BOOT
    assert eps == pytest.approx(max(goal_eval.EPS_FLOOR, 2 * np.std(stats, ddof=1)))
    assert ref['eps'] >= goal_eval.EPS_FLOOR
