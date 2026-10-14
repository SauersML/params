"""The gate harness's measurement and budget rules on a tiny random Qwen3 (CPU, no download)."""
import contextlib
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h  # noqa: E402


class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(name(i) for i in ids)


def name(i):
    """A letters-only word per token id (ao_verify.norm keeps letters)."""
    return "w" + chr(97 + i % 26) + chr(97 + i // 26)


class FakeTarget:
    """ao_verify.Target's surface the harness uses: a tiny Qwen3, every adapter off, an oracle that names a word."""

    def __init__(self, answer=None):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
        self.m, self.dev, self.tok, self.torch = Qwen3ForCausalLM(cfg).eval(), torch.device("cpu"), Tok(), torch
        self.act_layer, self.answer = 1, answer or name(5)

    def use(self, adapter):
        return contextlib.nullcontext()

    def hidden(self, ids):
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([ids]), output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]), out.logits[0]

    def inject(self, prompt, acts, k, sample, adapter_ao):
        return [self.answer] * k


def item(rng):
    ids = [rng.randrange(64) for _ in range(h2h.T)]
    return {"id": "t0", "kind": "text", "ids": ids, "top": 0, "a": {"gt": 5}, "b": [{"p": h2h.T - 2, "y": 3, "changes": True, "new_top": 1}],
            "c": [{"pair": [4, 9], "gt": 4}]}


def test_draws_leave_out_the_actual_token():
    tg = FakeTarget()
    m, rng = h2h.Model(tg), random.Random(0)
    ids = item(rng)["ids"]
    dr = h2h.draws(m, ids, [3, 7, 20], 5, rng)
    assert all(len(v) == 5 and ids[p] not in v for p, v in dr.items())


def test_kl_of_a_distribution_with_itself_is_zero():
    lp = torch.log_softmax(torch.randn(3, 10), -1)
    assert torch.allclose(h2h.kl_bits(lp, lp), torch.zeros(3), atol=1e-6)


def test_scan_and_answerer_B_stay_within_the_budget():
    tg = FakeTarget()
    rng = random.Random(1)
    it = item(rng)
    for budget in (64, 256):
        m = h2h.Model(tg)
        rows = h2h.answer_B(tg, m, it, budget, random.Random(2))
        assert m.passes <= budget, (m.passes, budget)
        assert {r["q"] for r in rows} == {"a", "b", "c"}
        a = next(r for r in rows if r["q"] == "a")
        assert a["draws"][5] >= h2h.STAGE1  # shortlisted or hinted, or the first stage's alone


def test_the_oracle_hint_is_the_named_words_position():
    tg = FakeTarget()
    ids = [7] * h2h.T
    ids[11] = 5
    assert h2h.find(tg, ids, name(5), list(range(1, h2h.T - 1))) == 11
    assert h2h.find(tg, ids, name(9), list(range(1, h2h.T - 1))) is None
