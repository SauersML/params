"""Checks of why.py on a tiny random Qwen3 (no tokenizer: prompts map to fixed token ids): a pair's explanation reports
the fast path's circuit, and its check's margins are the circuit's measured shares (margin after = margin before - lost
share x gap); a random control is measured beside; a donor is found from the task families when token counts agree;
zero ablation without a contrast; the summary is filled from the numbers.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/why/test_why.py -q
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import why as W  # noqa: E402

transformers = pytest.importorskip("transformers")
cf, cm, ca = W.cf, W.cm, W.ca


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    for _ in range(400):
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupt = list(clean)
        corrupt[4], corrupt[7] = (clean[4] + 7) % 64, (clean[7] + 11) % 64
        pt = cf.Patcher(model, None, rows=8)
        r = pt.setup(clean, corrupt)
        if r["answer"] != r["corrupt_answer"] and r["gap"] > 0.5:
            return model, clean, corrupt
    pytest.skip("no pair")


def make(model, clean, corrupt, tmp_path, questions=()):
    ids = {"A": clean, "B": corrupt}
    return W.Why(model, None, ca.scorer("grad"), questions=questions, chat=lambda text: ids.get(text, clean))


def test_pair_explanation_checks_its_circuit(tiny, tmp_path):
    model, clean, corrupt = tiny
    why = make(model, clean, corrupt, tmp_path)
    r = why.explain("A", contrast="B")
    assert r["mode"] == "pair" and not r.get("error")
    for c in r["check"]:
        cut = r[c["cut"]]
        assert cut["met"]
        want = c["margin_before"] - cut["lost"] * r["gap"]
        assert abs(c["margin_after"] - want) < 1e-3 * max(1.0, abs(r["gap"]))
        assert c["flipped"] == (c["margin_after"] < 0)
    assert r["summary"].startswith("The model answers")


def test_unequal_lengths_are_refused(tiny, tmp_path):
    model, clean, corrupt = tiny
    ids = {"A": clean, "B": corrupt[:-1]}
    why = W.Why(model, None, ca.scorer("grad"), questions=(), chat=lambda t: ids[t])
    assert "tokens" in why.explain("A", contrast="B")["error"]


def test_donor_from_the_task_family(tiny, tmp_path):
    model, clean, corrupt = tiny
    qs = {"questions": [{"task": "capital", "key": "France", "answer": ["Paris"], "user": "x"},
                        {"task": "capital", "key": "Japan", "answer": ["Tokyo"], "user": "x"}]}
    path = tmp_path / "q.json"
    path.write_text(json.dumps(qs))
    ids = {"What is the capital of France?": clean, "What is the capital of Japan?": corrupt}
    why = W.Why(model, None, ca.scorer("grad"), questions=(str(path),), chat=lambda t: ids.get(t, clean[:3]))
    p2, info = why.donor("What is the capital of France?", "tokyo")
    assert p2 == "What is the capital of Japan?" and info == {"task": "capital", "key": "France", "donor_key": "Japan"}
    assert why.donor("What is the capital of France?", "Lima")[0] is None
    r = why.explain("What is the capital of France?", alternative="Tokyo")
    assert r["donor"]["donor_key"] == "Japan" and r["mode"] == "pair"


def test_ablation_without_a_contrast(tiny, tmp_path):
    model, clean, corrupt = tiny
    why = make(model, clean, corrupt, tmp_path)
    alt = str(int(why.pt.core(torch.tensor([clean])).logits[0, -1].topk(2).indices[1]))
    r = why.explain("A", alternative=alt)       # (no known task: zero ablation)
    a = r["ablation"]
    assert r["mode"] == "ablation" and len(a["margins"]) == len(a["sizes"]) == len(a["random_margins"])
    assert a["margin_after"] == min(a["margins"]) or a["flipped"]
    assert "zero ablation" in r["summary"]
