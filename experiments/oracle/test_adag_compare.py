"""Checks of the ADAG head-to-head harness (adag_compare.py) on a tiny random Qwen3: zeroed neurons equal a hand-written
hook; a set the search returns passes its own success test when rerun; the halve goal's test reads the base probability.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_adag_compare.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adag_compare as ad  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    return transformers.Qwen3ForCausalLM(cfg).eval()


class Tok:
    def decode(self, z):
        return str(z)


def test_zeroing_equals_hooks(tiny):
    run = ad.Runner(tiny)
    ids = list(range(5, 15))
    got = run.last(ids, [[(1, 7), (2, 3)]])[0]
    hs = [tiny.model.layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda m, a, js=js: (a[0].index_fill(-1, torch.tensor(js), 0.0),)) for L, js in ((1, [7]), (2, [3]))]
    try:
        with torch.no_grad():
            want = torch.log_softmax(tiny(torch.tensor([ids])).logits[0, -1].float(), -1)
    finally:
        for h in hs:
            h.remove()
    assert torch.allclose(got, want, atol=1e-5)


def test_found_sets_pass_their_test(tiny):
    run = ad.Runner(tiny)
    ids = list(range(5, 15))
    base = run.last(ids, [[]])[0]
    A = int(base.argmax())
    for kind, X in (("change", None), ("halve", None), ("flip", int(base.topk(2).indices[1]))):
        rec = ad.search(run, Tok(), ids, kind, A, X, log=lambda s: None)
        if rec["found"]:
            _, ok = ad.goal_fns(kind, A, X, rec["p_answer"])
            assert ok(run.last(ids, [[tuple(u) for u in rec["units"]]])[0])
            assert rec["neurons"] <= rec["prefix"]
