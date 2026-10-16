"""Checks of the essay replication harness (essay_replication.py) on a tiny random Qwen3: a blocked head gives the key
no attention at the reading position and the rest of its row still sums to one, other heads and positions unchanged;
pole and whole-neuron zeroing match their definitions; the attention checks' pass rule.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_essay_replication.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import essay_replication as er  # noqa: E402


@pytest.fixture(scope="module")
def q():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
                                   attn_implementation="eager")
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    return er.Qwen("tiny", "cpu", torch.float32, model=model)


def test_blocked_head_gives_the_key_no_attention(q):
    ids = list(range(3, 13))
    a0 = q.run(ids, attentions=True)["attentions"]
    a1 = q.run(ids, blocks={1: [(2, 9, 4)]}, attentions=True)["attentions"]
    assert float(a1[1][0, 2, 9, 4]) == pytest.approx(0.0, abs=1e-7)
    assert float(a1[1][0, 2, 9].sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(a0[1][0, 2, 9, 4]) > 0
    assert torch.allclose(a1[0], a0[0]) and torch.allclose(a1[1][0, :, :9], a0[1][0, :, :9])
    assert torch.allclose(a1[1][0, [0, 1, 3]], a0[1][0, [0, 1, 3]])
    assert q.run(ids)["lp"].allclose(q.run(ids, blocks={})["lp"])


def test_zeroing_poles_and_neurons(q):
    ids = list(range(3, 13))
    cap = q.run(ids, capture=[(1, 5, p) for p in range(10)])["cap"]
    vals = [cap[(1, 5, p, "value")] for p in range(10)]
    seen = []
    h = q.layers[2].mlp.down_proj.register_forward_hook(lambda m, a, o: seen.append(a[0].clone()))  # (the input after the edits)
    try:
        q.run(ids, zero=[(2, 7, None, 1)])
        q.run(ids, zero=[(2, 7, [3], 0)])
    finally:
        h.remove()
    assert bool((seen[0][0, :, 7] <= 0).all())
    assert float(seen[1][0, 3, 7]) == 0.0
    assert len(vals) == 10 and all(isinstance(v, float) for v in vals)


def test_pass_rule():
    assert er.judge_change(40.0, 30.0, 45.5, 34.5, False)  # the essay's -24%; ours -25%
    assert not er.judge_change(40.0, 37.0, 45.5, 34.5, False)  # under half the essay's change
    assert not er.judge_change(40.0, 50.0, 45.5, 34.5, False)  # the wrong way
    assert er.judge_change(40.0, 42.0, 45.5, 45.75, True) and not er.judge_change(40.0, 30.0, 45.5, 45.75, True)
