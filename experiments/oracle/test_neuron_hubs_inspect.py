"""Native microscope evidence on tiny synthetic gated modules; no assets/readers."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuron_hubs as hub


class MLP(torch.nn.Module):
    def __init__(self, mismatch=False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 1, bias=False)
        self.up_proj = torch.nn.Linear(2, 1, bias=False)
        self.down_proj = torch.nn.Linear(1, 2, bias=False)
        self.act_fn = torch.nn.functional.silu
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.tensor([[1., 0.]]))
            self.up_proj.weight.copy_(torch.tensor([[0., 1.]]))
            self.down_proj.weight.copy_(torch.tensor([[2.], [-3.]]))
        self.mismatch = mismatch
    def forward(self, x):
        a = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(a + (0.5 if self.mismatch else 0))


class Body(torch.nn.Module):
    def __init__(self, mismatch=False):
        super().__init__()
        self.embed = torch.nn.Embedding(5, 2)
        with torch.no_grad():
            self.embed.weight.copy_(torch.tensor([[1.,2.], [1.,-2.], [-1.,2.], [-1.,-2.], [0.,2.]]))
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].mlp = MLP(mismatch)
        self.fail = False
    def forward(self, input_ids):
        x = self.layers[0].mlp(self.embed(input_ids))
        if self.fail:
            raise RuntimeError('synthetic failure')
        return SimpleNamespace(last_hidden_state=x)


class Model(torch.nn.Module):
    def __init__(self, mismatch=False):
        super().__init__()
        self.model = Body(mismatch)
        self.lm_head = torch.nn.Linear(2, 5)
        self.config = SimpleNamespace(num_hidden_layers=1, intermediate_size=1,
                                      _name_or_path='synthetic', _commit_hash='fixture-revision')


def hooks(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


@pytest.mark.parametrize('mismatch', [False, True])
def test_full_context_signed_capture_reconstruction_and_weights_unchanged(mismatch):
    model = Model(mismatch); runner = hub.Runner(model)
    before = copy.deepcopy(model.state_dict())
    # 70 full positions: no hidden short-context truncation or sink exclusion.
    ids = [0,1,2,3,4]*14
    got = runner.inspect(ids, [(0,0)], pos=68, source_binding={'task_sha256':'a'*64})
    assert got['input_token_ids'] == ids and got['position_ids'] == list(range(70))
    assert got['target_position'] == 68
    row = got['neurons'][0]
    assert row['gate_sign'][:5] == ['+','+','-','-','0']
    assert row['up_sign'][:5] == ['+','-','+','-','+']
    assert row['activation_sign'][:4] == ['+','-','-','+']
    assert row['reconstruction']['within_tolerance'] is (not mismatch)
    assert row['write']['direction'] == [2., -3.]
    assert row['min_activation_position'] in range(70)
    assert got['model']['full_weight_content_verified'] is False
    assert got['model']['revision'] == 'fixture-revision'
    assert got['source_binding'] == {'task_sha256':'a'*64}
    assert hooks(model) == 0
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    if not mismatch:
        assert row['zero_positions'] == list(range(4,70,5))
        assert row['activation_sign'][4] == '0'
    else:
        assert row['down_input'][4] == .5  # retain actual input, never "repair" to reconstruction


def test_adapter_write_unavailable_and_hooks_removed_on_failure():
    model = Model(); model.get_base_model = lambda: model
    model.active_adapters = ['fixture-adapter']
    runner = hub.Runner(model)
    got = runner.inspect([0,1], [(0,0)])
    assert got['model']['active_adapters'] == ['fixture-adapter']
    assert got['neurons'][0]['write']['direction'] is None
    assert got['neurons'][0]['write']['status'] == 'unavailable_adapter_or_custom_projection'
    model.model.fail = True
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runner.inspect([0,1], [(0,0)])
    assert hooks(model) == 0


def test_study_freezes_original_answer_when_background_flips_top(monkeypatch):
    class Runner:
        L=1; D=2; dev='cpu'
        def logits(self, ids, pos, edit=None, B=1):
            h = torch.tensor([[[.1,.9]]]).expand(B, -1, -1)
            if edit is not None:
                h = edit(0,h)
            return torch.stack([h[:,0,0]*10, h[:,0,1],
                                *[h[:,0,0]*0-c for c in (.1,.2,.3,.4)]], -1)
    monkeypatch.setattr(hub, 'CANDIDATES', 2)
    monkeypatch.setattr(hub, 'KS', (1,2))
    tok = SimpleNamespace(decode=lambda ids: str(ids[0]))
    original = hub.study(Runner(), [0], 0, None, torch.zeros(1,2), tok)
    weakened = hub.study(Runner(), [0], 0, torch.tensor([[.2,0.]]), torch.zeros(1,2), tok)
    assert original['answer_token_id'] == weakened['answer_token_id'] == 0
    assert original['alternative_token_ids'] == weakened['alternative_token_ids']
    assert original['background_top_changed'] is False
    assert weakened['background_top_changed'] is True
    assert weakened['cumulative'][0]['top'] is False
    assert weakened['flip_k'] == weakened['flip_k_mean_fill'] == 0


def test_hf_active_adapters_method_without_loaded_adapter():
    model = Model()
    def unloaded():
        raise ValueError('No adapter loaded')
    model.active_adapters = unloaded
    got = hub.Runner(model).inspect([0,1], [(0,0)])
    assert got['model']['active_adapters'] == []
    assert got['model']['adapter_state'] == 'not_loaded'
    assert got['neurons'][0]['write']['status'] == 'plain_linear_direction'
    assert hooks(model) == 0
    model.active_adapters = lambda: (_ for _ in ()).throw(ValueError('other adapter failure'))
    with pytest.raises(ValueError, match='other adapter failure'):
        hub.Runner(model).inspect([0,1], [(0,0)])
    assert hooks(model) == 0
    model._hf_peft_config_loaded = False
    got = hub.Runner(model).inspect([0,1], [(0,0)])
    assert got['model']['adapter_state'] == 'not_loaded'
