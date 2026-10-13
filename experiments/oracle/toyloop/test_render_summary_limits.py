"""The input legend must distinguish linear weight evidence from causal claims."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render


def small_toy():
    cfg = {"n_layer": 1, "n_head": 1, "n_embd": 2, "n_ctx": 4}
    sd = {"wte.weight": torch.eye(2), "ln_f.weight": torch.ones(2),
          "h.0.rms_1.weight": torch.ones(2), "h.0.rms_2.weight": torch.ones(2),
          "h.0.mlp.c_fc.weight": torch.tensor([[1., 0.], [1., 0.]]),
          "h.0.mlp.down_proj.weight": torch.tensor([[1., -2.], [0., 0.]])}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        sd[f"h.0.attn.{name}.weight"] = torch.zeros(2, 2)
    return render.Toy("unequal_opposites", sd, cfg, ["a", "b"], [[0, 1]])


def test_zero_linear_overlap_does_not_remove_rms_influence():
    residual, write, reader = torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([1., 0.])
    assert float(reader @ write) == 0.
    before = reader @ render._rms(residual, torch.ones(2), 1e-6)
    after = reader @ render._rms(residual + write, torch.ones(2), 1e-6)
    assert abs(float(before - after)) > .4


def test_cancellation_tag_does_not_establish_equal_magnitude():
    toy = small_toy()
    assert render.cancelling_pairs(toy) == {"L0N0": "L0N1", "L0N1": "L0N0"}
    # Both neurons have identical preactivations but their output vectors do not cancel.
    activations = render._gelu(toy.sd["h.0.mlp.c_fc.weight"] @ torch.tensor([1., 0.]))
    output = toy.sd["h.0.mlp.down_proj.weight"] @ activations
    assert float(output.norm()) > .8


def test_opt_in_changes_only_scope_text_and_never_uses_truth(monkeypatch):
    toy = small_toy()
    monkeypatch.setattr(render, "SECTIONS", "weights")
    legacy = render.render(toy, all_live_neurons=True)
    assert legacy == render.render(toy, all_live_neurons=True, explain_summary_limits=False)
    changed = render.render(toy, all_live_neurons=True, explain_summary_limits=True)
    numerical = lambda prompt: [line for line in prompt.splitlines()
                                if line.startswith(("L0", "tok (", "Text "))]
    assert numerical(changed) == numerical(legacy)
    assert "out[] = nothing reads it" in legacy
    assert "out[] = nothing reads it" not in changed
    assert render.SUMMARY_LIMITS in changed and render.NORMALIZATION_RULE in changed
    assert changed.endswith(render.FORMAT)
    toy.truth = {"units": ["L999H999"], "edges": [], "english": "DO NOT COPY THIS"}
    assert changed == render.render(toy, all_live_neurons=True, explain_summary_limits=True)
