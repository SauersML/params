"""Checks of answer_contrast: the words tokens start (spellings of one answer read as one word, other scripts told
apart) and the sets on a tiny random Qwen3 without a tokenizer (the top token against the next ones).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_answer_contrast.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import answer_contrast as ac  # noqa: E402

transformers = pytest.importorskip("transformers")


def test_words():
    assert [ac.word(t) for t in ("Paris", " Paris", "**Paris**", "  'Tok", "Au\n", "", " ")] == ["paris", "paris", "paris", "tok", "au", "", ""]
    assert ac.script("paris") == "LATIN" and ac.script("巴黎") == "CJK" and ac.script("62") == "DIGIT"


def test_sets_without_tokenizer():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    m = transformers.Qwen3ForCausalLM(cfg).eval()
    ids = list(range(5, 15))
    with torch.no_grad():
        lp = torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, -1], -1)
    s = ac.token_sets(m, None, ids, lp, b=int(lp.argsort()[-3]))
    assert s["answer"] == [int(lp.argmax())] and s["alternatives"][0] == int(lp.argsort()[-3])
    assert not set(s["answer"]) & set(s["alternatives"])
    want = (lp[s["answer"]].logsumexp(-1) - lp[s["alternatives"]].logsumexp(-1)) / torch.log(torch.tensor(2.0))
    assert abs(float(ac.contrast(lp[None], s)[0]) - float(want)) < 1e-5
