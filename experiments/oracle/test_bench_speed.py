"""Window benchmark must measure physical work, including after warm-up."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'method/oracle'))
import bench_speed as bench
import carry
import native


@pytest.mark.parametrize('fail', [False, True])
def test_window_modes_execute_and_restore_cache(monkeypatch, fail):
    events = []
    class FakeNative:
        WINDOW_EXACT = ('mps',)
        dev = 'cpu'
        def sensitivity(self, ids, targets, seed):
            key = (tuple(ids), tuple(targets), seed.initial_seed())
            cache = self.__dict__.setdefault('_sensitivity_kept', {})
            if key in cache:
                pytest.fail('benchmark consumed cached sensitivity result')
            events.append(('execute', self.WINDOW_EXACT))
            if fail and self.WINDOW_EXACT == ('cpu',):
                raise RuntimeError('window failed')
            cache[key] = (torch.tensor([float(bool(self.WINDOW_EXACT))]), None)
            return cache[key]
    nat = FakeNative()
    original = {'existing': object()}
    nat._sensitivity_kept = original
    monkeypatch.setattr(native, 'text', lambda path: ([1, 2, 3], [2]))
    monkeypatch.setattr(native, 'task_seed', lambda text: 7)
    monkeypatch.setattr(carry, 'synchronize', lambda dev: events.append(('sync', dev)))
    if fail:
        with pytest.raises(RuntimeError, match='window failed'):
            bench.bench_window(nat, [Path('train6009.json')])
    else:
        rows = bench.bench_window(nat, [Path('train6009.json')])
        assert rows[0]['equal'] is False
        assert events == [('execute', ()), ('sync', 'cpu'), ('execute', ()), ('sync', 'cpu'),
                          ('execute', ('cpu',)), ('sync', 'cpu'), ('execute', ('cpu',)), ('sync', 'cpu')]
    assert nat.WINDOW_EXACT == ('mps',)
    assert nat._sensitivity_kept is original
