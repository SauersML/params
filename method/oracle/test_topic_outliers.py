"""Checks of topic_outliers on a tiny random Qwen3 with Qwen3's tokenizer and a small fake corpus: the corpus windows hold
the topic's words, the stats match a direct recompute, and the result has the tool's fields.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_topic_outliers.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import topic_outliers as to  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        pytest.skip("Qwen3 tokenizer not available")


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def _corpus(tmp_path, tok):
    rng = np.random.default_rng(0)
    filler = rng.integers(1000, 5000, size=6000)
    word = tok(" volcano", add_special_tokens=False)["input_ids"]
    for at in (500, 1500, 2500, 3500, 4500):
        filler[at:at + len(word)] = word
    np.save(tmp_path / "tokens.npy", filler.astype(np.int32))
    np.save(tmp_path / "offsets.npy", np.array([0, 6000]))
    return filler


def test_windows_hold_the_topic(tmp_path, tok):
    arr = _corpus(tmp_path, tok)
    wins = to.text_windows(tok, "volcano eruptions", arr, n=4, length=64)
    word = tok(" volcano", add_special_tokens=False)["input_ids"]
    assert len(wins) == 4 and all(len(w) == 64 for w in wins)
    assert all(any(w[i:i + len(word)] == word for i in range(64 - len(word))) for w in wins)


def test_stats_match_direct(model, tok):
    rows, starts = [[5, 6, 7, 8, 9], [10, 11, 12, 13]], [2, 1]
    s = to._stats(model, rows, starts, True)
    cap = []
    h = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.append(a[0][0].float()))
    with torch.no_grad():
        for ids in rows:
            model(input_ids=torch.tensor([ids]))
    h.remove()
    want = cap[0][2:].sum(0) + cap[1][1:].sum(0)
    assert s["n"] == 3 + 3 and torch.allclose(s["sum"][1].float(), want, atol=1e-4)


def test_tool_fields(tmp_path, model, tok):
    _corpus(tmp_path, tok)
    out = to.topic_outliers(model, tok, ["volcano eruptions", "no-such-wordxyz"], n_replies=1, max_tokens=4, corpus=tmp_path,
                            controls=["volcano"], top_units=3, top_tokens=3)
    assert out[0]["topic"] == "volcano eruptions" and len(out[0]["units"]) == 3 and len(out[0]["tokens"]) == 3
    u = out[0]["units"][0]
    assert set(u) >= {"unit", "z", "writes"} and u["unit"].startswith("L") and u["unit"][-1] in "+-"
    assert out[1]["units"] == [] and "error" in out[1]
