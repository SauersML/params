"""polycheck.py's trait runner on a tiny random Qwen3: the unedited run is exact twice, and a pole's knockout zeroes that
pole's own features (its largest value and its answer-position value) and leaves the other pole's."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polycheck  # noqa: E402

transformers = pytest.importorskip("transformers")


def test_trait_runner():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    nr = polycheck.nb.Neurons(transformers.Qwen3ForCausalLM(cfg).eval(), None)
    nr.mx = torch.ones(nr.L, nr.D)
    nr.massive = torch.full((nr.L,), 1e9)
    rng = random.Random(0)
    seqs = []
    for k in range(20):
        p = [rng.randrange(64) for _ in range(rng.randrange(6, 12))]
        seqs += [{"id": f"p/a|{k}|A|pos", "prompt": p, "cont": [3, 4]}, {"id": f"p/a|{k}|A|neg", "prompt": p, "cont": [5]}]
    cand = [(0, 3), (1, 7), (1, 9)]
    tr = polycheck.Trait(nr, seqs, cand)
    v0, F0 = tr.run()
    v0b, F0b = tr.run()
    assert np.array_equal(v0, v0b) and np.array_equal(F0, F0b)
    e, r = polycheck.cc.knockout_edit(nr, [("N", 1, 7, 1, None)])
    v1, F1 = tr.run(e, r)
    assert np.all(F1[:, 1, [0, 2]] == 0) and np.allclose(F1[:, 1, [1, 3]], F0[:, 1, [1, 3]], atol=1e-6)
    assert np.allclose(F1[:, 0], F0[:, 0], atol=1e-6)  # (an earlier layer's pole does not move)
    assert not np.allclose(v1, v0)
