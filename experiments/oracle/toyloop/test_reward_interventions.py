import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reward_exec
from reward_interventions import scaled_neuron


def test_neuron_edit_transports_original_basis_and_clears_cache():
    site = SimpleNamespace(W=torch.tensor([[2., -2.]]), V=torch.eye(2),
                           U=torch.tensor([[2.], [-2.]]), delta_T=torch.zeros(2, 1),
                           gathered=(torch.eye(2), torch.tensor([[99.], [99.]])))
    nat = SimpleNamespace(names=["down"], target=SimpleNamespace(n_layer=1, site=lambda name: site),
                          units={"h.0.mlp.down_proj": torch.tensor([0, 1])},
                          _refs={"stale": 1}, _refs_bytes=10, _rec=(1, 2, 3))
    original = site.W.clone()
    with scaled_neuron(nat, 0, 0, 0.):
        assert not nat._refs
        assert site.gathered is None
        assert torch.equal((site.V @ site.U).T + site.delta_T.T, site.W)
        assert torch.equal(site.W, torch.tensor([[0., -2.]]))
        nat._refs["edited"] = 2
    assert torch.equal(site.W, original)
    assert not nat._refs
    with pytest.raises(RuntimeError, match="deliberate"):
        with scaled_neuron(nat, 0, 1, .5):
            raise RuntimeError("deliberate")
    assert torch.equal(site.W, original)
    assert not nat._refs


def test_worst_intervention_penalizes_point_equivalent_removal():
    exact = [{"kl_bits": 0., "description_cost": .5}] * 3
    smaller = [{"kl_bits": 0., "description_cost": .25}] * 2 + [
        {"kl_bits": 8., "description_cost": .25}]
    assert reward_exec.intervention_mdl_value(exact) > reward_exec.intervention_mdl_value(smaller)
    with pytest.raises(RuntimeError):
        reward_exec.intervention_mdl_value([{"kl_bits": float("nan"), "description_cost": 0.}])


def test_invalid_syntax_remains_minus_one():
    assert reward_exec.reward_wiring_intervention_mdl({}, "not a circuit") == -1.


def test_verifier_valueerror_is_not_candidate_invalid(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("verifier failure")
    monkeypatch.setattr(reward_exec, "score_wiring_interventions", fail)
    with pytest.raises(ValueError, match="verifier failure"):
        reward_exec.reward_wiring_intervention_mdl({"toy": object()}, "flowchart LR\n L0H0 --> out\n")


def test_fidelity_first_larger_faithful_beats_smaller_failure():
    faithful = [{'kl_bits': 0., 'description_cost': 1.}]
    smaller_fail = [{'kl_bits': .004, 'description_cost': 0.}]
    assert reward_exec.intervention_mdl_value(smaller_fail) > reward_exec.intervention_mdl_value(faithful)
    assert reward_exec.intervention_fidelity_first_value(faithful) > reward_exec.intervention_fidelity_first_value(smaller_fail)
    smaller_pass = [{'kl_bits': .001, 'description_cost': .25}]
    assert reward_exec.intervention_fidelity_first_value(smaller_pass) > reward_exec.intervention_fidelity_first_value(faithful)


def test_fidelity_first_failure_signal_continuous_and_cost_invariant():
    value = reward_exec.intervention_fidelity_first_value
    assert value([{'kl_bits': .002, 'description_cost': 1.}]) > value([{'kl_bits': .004, 'description_cost': 0.}])
    assert value([{'kl_bits': .004, 'description_cost': 0.}]) == value([{'kl_bits': .004, 'description_cost': 1.}])
    assert abs(value([{'kl_bits': .004001, 'description_cost': 0.}]) - value([{'kl_bits': .004, 'description_cost': 0.}])) < 1e-6
    assert value([{'kl_bits': 1e308, 'description_cost': 0.}]) == 0.
    for bad in (float('nan'), float('inf'), -1., True):
        for key in ('kl_bits', 'description_cost'):
            row = {'kl_bits': 0., 'description_cost': 0.}; row[key] = bad
            with pytest.raises(RuntimeError): value([row])
    with pytest.raises(RuntimeError): value([])


def test_fidelity_first_accepts_alternative_graphs_and_reuses_panel(monkeypatch):
    calls = []
    toy, nat, cache, panel = (object() for _ in range(4))
    def score(actual_toy, text, **kwargs):
        calls.append((actual_toy, text, kwargs))
        return {'per_world': [{'kl_bits': 0., 'description_cost': .5}]}
    monkeypatch.setattr(reward_exec, 'score_wiring_interventions', score)
    meta = {'toy': toy, 'intervention_native': nat, 'intervention_reference_cache': cache,
            'intervention_panel': panel}
    reward = reward_exec.reward_wiring_intervention_fidelity_first
    assert reward(meta, 'flowchart LR\n L0H0 --> out') == reward(meta, 'flowchart LR\n L1H1 --> out')
    assert len(calls) == 2 and all(c[0] is toy and c[2] == {'nat': nat, 'reference_cache': cache, 'panel': panel} for c in calls)
    assert reward({}, 'not a graph') == -1.
    def failed(*args, **kwargs): raise ValueError('verifier failure')
    monkeypatch.setattr(reward_exec, 'score_wiring_interventions', failed)
    with pytest.raises(ValueError, match='verifier failure'): reward(meta, 'flowchart LR\n L0H0 --> out')


def test_fidelity_first_complete_architecture_cost_boundary(monkeypatch):
    import math
    repo = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo / 'method/oracle'))
    monkeypatch.syspath_prepend(str(repo / 'method/vpd'))
    import native
    names = ['h.0.mlp.c_fc', 'h.0.mlp.down_proj']
    nat = SimpleNamespace(names=names, C={name: 128 for name in names})
    nodes = {(name, pos, piece) for name in names for pos in range(32) for piece in range(128)}
    full = native.Graph(nodes, complete=True)
    measured = reward_exec.executed_description(full, nat, 32, [31])
    assert measured['description_cost'] == 1.
    # The largest valid graph remains qualified despite its description cost.
    assert reward_exec.intervention_fidelity_first_value([{'kl_bits': .001, **measured}]) == 1.
    below = math.nextafter(1., 0.)
    assert reward_exec.intervention_fidelity_first_value([{'kl_bits': .001, 'description_cost': below}]) >= 1.
    with pytest.raises(RuntimeError):
        reward_exec.intervention_fidelity_first_value([{'kl_bits': .001, 'description_cost': math.nextafter(1., math.inf)}])
