"""Literal neuron-output edits transported to the original decomposition."""
import contextlib
import math

from head_counterfactual import clear_native


def clear_intervention_caches(nat):
    clear_native(nat)
    for name in nat.names:
        site = nat.target.site(name)
        if hasattr(site, "gathered"):
            site.gathered = None


@contextlib.contextmanager
def scaled_neuron(nat, layer, neuron, scale):
    import torch
    if not math.isfinite(scale) or not isinstance(layer, int) or not 0 <= layer < nat.target.n_layer:
        raise RuntimeError("invalid neuron intervention")
    site = nat.target.site(f"h.{layer}.mlp.down_proj")
    if not isinstance(neuron, int) or not 0 <= neuron < site.W.shape[1]:
        raise RuntimeError("neuron outside architecture")
    saved = {key: getattr(site, key).clone() for key in ("W", "V", "U", "delta_T")}
    name = f"h.{layer}.mlp.down_proj"
    units = nat.units[name].clone()
    with torch.no_grad():
        site.W[:, neuron] *= scale
        site.V[neuron, :] *= scale
        site.delta_T[neuron, :] *= scale
    clear_intervention_caches(nat)
    try:
        assert torch.equal(site.U, saved["U"])
        assert torch.equal(nat.units[name], units)
        yield
    finally:
        with torch.no_grad():
            for key, value in saved.items():
                getattr(site, key).copy_(value)
        clear_intervention_caches(nat)
        assert all(torch.equal(getattr(site, key), value) for key, value in saved.items())
        assert torch.equal(nat.units[name], units)
