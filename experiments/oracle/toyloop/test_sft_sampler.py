"""Sampler coverage and train-split isolation without policy loading/scoring."""
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


@pytest.mark.parametrize('batch', [1, 3, 8])
def test_epoch_full_coverage_before_repeat_and_reproducible(batch):
    data = list(range(7))
    before = data.copy()
    draws = lambda seed: list(itertools.chain.from_iterable(itertools.islice(
        train.sft_batches(data, batch, random.Random(seed), 'epoch'), 15)))
    one = draws(0)
    assert one == draws(0) and one != draws(1)
    assert data == before
    for start in range(0, len(one)-6, 7):
        assert sorted(one[start:start+7]) == data


def test_default_random_exact_original_rng_schedule():
    args = train.parser().parse_args(['sft'])
    assert args.sft_sampler == 'random'
    data = list(range(226))
    original = random.Random(0)
    batches = train.sft_batches(data, 1, random.Random(0))
    assert [next(batches) for _ in range(300)] == [original.sample(data, 1) for _ in range(300)]


def test_epoch_sft_uses_only_requested_train_rows(monkeypatch, tmp_path):
    requested, trained = [], []
    train_rows = [{'prompt': str(i), 'completion': 'answer', 'meta': {'name': f'train{i}'}} for i in range(5)]
    eval_rows = [{'prompt': 'heldout', 'completion': 'answer', 'meta': {'name': 'test_family0'}}]
    def examples(spec, truth):
        requested.append(spec)
        return train_rows if spec.endswith('@train') else eval_rows
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(train, 'examples', examples)
    monkeypatch.setattr(train, 'load_policy', lambda a: (model, object()))
    def batch_ids(tok, pairs, max_len, device=None):
        trained.extend(p for p, c in pairs)
        return None, None, None
    monkeypatch.setattr(train, 'batch_ids', batch_ids)
    monkeypatch.setattr(train, 'token_logps', lambda *args: (model.weight, torch.ones_like(model.weight)))
    monkeypatch.setattr(train, 'save', lambda *args: None)
    args = train.parser().parse_args(['sft', '--data', 'zoo:frozen.jsonl@train', '--sft-sampler', 'epoch',
                                     '--steps', '7', '--batch', '1', '--out', str(tmp_path)])
    logs = train.sft(args)
    assert requested == ['zoo:frozen.jsonl@train']
    assert sorted(trained[:5]) == [str(i) for i in range(5)]
    assert 'heldout' not in trained
    assert [r['sample_ids'][0] for r in logs] == ['train'+p for p in trained]



def test_paired_sft_epoch_passthrough_and_frozen_random_fork_rejected(tmp_path):
    import json
    import paired_wiring as paired
    args = SimpleNamespace(out=tmp_path / 'new', sft_steps=300, rl_steps=20,
                           seed=0, max_len=3072, max_new=416, device='cpu', sft_sampler='epoch')
    assert paired.training_args(args, None, 'base', None, 'output').sft_sampler == 'epoch'
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'contract.json').write_text(json.dumps({'sft_sampler': 'random'}))
    with pytest.raises(ValueError, match='sampler mismatch'):
        paired.fork_sft(args, parent)
    assert not args.out.exists()
