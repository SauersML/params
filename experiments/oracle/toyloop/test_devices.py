"""Explicit toy-policy devices, without pretrained downloads or toy scoring."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(os.environ.get('TOY_TRAIN_DEVICE_SOURCE', Path(__file__).with_name('train.py')))
# The environment override is only for reviewing a surgical scratch patch.
sys.path.insert(0, os.environ.get('TOY_TRAIN_IMPORT_ROOT', str(SOURCE.parent)))
spec = importlib.util.spec_from_file_location('toy_device_train', SOURCE)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_default_cpu_and_unavailable_device_fails(monkeypatch):
    assert train.parser().parse_args(['sft']).device == 'cpu'
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    for name in ('mps', 'cuda'):
        with pytest.raises(ValueError, match='unavailable'):
            train.explicit_device(name)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_policy_batch_and_sample_use_requested_device(monkeypatch, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    monkeypatch.setattr(train, 'tiny_model', lambda: (torch.nn.Linear(2, 2), SimpleNamespace(pad_token_id=0)))
    args = SimpleNamespace(model='tiny', init=None, threads=1, device=device, max_len=16, max_new=2, temp=1.)
    model, tok = train.load_policy(args)
    assert next(model.parameters()).device.type == device
    monkeypatch.setattr(train.oracle_lm, 'chat_ids', lambda tok, prompt, completion=None: ([1, 2, 3], 2))
    ids, attention, mask = train.batch_ids(tok, [('prompt', 'answer')], 16, device=next(model.parameters()).device)
    assert {x.device.type for x in (ids, attention, mask)} == {device}
    seen = []
    def generate(inputs, **kwargs):
        seen.append((inputs.device.type, kwargs['attention_mask'].device.type))
        return torch.cat([inputs, inputs[:, :1]], dim=1)
    model.generate = generate
    tok.batch_decode = lambda outputs, **kwargs: seen.append(outputs.device.type) or ['answer']
    assert train.sample(model, tok, 'prompt', 1, args) == ['answer']
    assert seen == [(device, device), 'cpu']  # Only tokenizer decoding transfers back to CPU.


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_sequential_rl_update_stays_on_device_without_native_scoring(monkeypatch, tmp_path, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 4)
            torch.nn.init.zeros_(self.embedding.weight)
            self.forward_devices = []
        def forward(self, input_ids, attention_mask, position_ids, logits_to_keep):
            self.forward_devices.append((input_ids.device.type, attention_mask.device.type, position_ids.device.type))
            return SimpleNamespace(logits=self.embedding(input_ids)[:, -logits_to_keep:])
        def generate(self, inputs, num_return_sequences, **kwargs):
            tail = torch.arange(1, num_return_sequences + 1, device=inputs.device)[:, None]
            return torch.cat([inputs.repeat(num_return_sequences, 1), tail], dim=1)
        def save_pretrained(self, out):
            pass
    policy = Policy()
    tok = SimpleNamespace(pad_token_id=0, batch_decode=lambda outputs, **kwargs: ['A' if int(r[-1]) == 1 else 'B' for r in outputs],
                          save_pretrained=lambda out: None)
    monkeypatch.setattr(train, 'tiny_model', lambda: (policy, tok))
    monkeypatch.setattr(train, 'examples', lambda *args: [{'prompt': 'TRAIN only', 'meta': {}}])
    monkeypatch.setattr(train, 'get_reward', lambda name: lambda meta, text: float(text == 'B'))
    monkeypatch.setattr(train.oracle_lm, 'chat_ids', lambda tok, prompt, completion=None:
                        ([0, 1] if completion is None else [0, 1, 2, 3] if completion == 'A' else [0, 1, 3, 2], 2))
    args = train.parser().parse_args(['rl', '--model', 'tiny', '--device', device, '--out', str(tmp_path),
                                     '--steps', '1', '--batch', '1', '--k', '2', '--max_len', '8', '--max_new', '2'])
    log = train.rl(args)
    assert policy.embedding.weight.device.type == device
    assert policy.forward_devices == [(device, device, device)] * 2
    assert torch.isfinite(policy.embedding.weight).all().item()
    assert policy.embedding.weight.abs().sum().item() > 0
    assert len(log) == 1 and log[0]['reward'] == .5



def test_explicit_mps_rejects_enabled_cpu_fallback(monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(ValueError, match='fallback is enabled'):
        train.explicit_device('mps')
