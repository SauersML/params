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
