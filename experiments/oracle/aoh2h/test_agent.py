"""The tools baseline's metering, unit removals and grading on a tiny random Qwen3 (CPU, no download)."""
import contextlib
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import h2h  # noqa: E402

V, T = 64, 12


class Tok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f" w{int(i)}" for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(w[1:]) % V for w in text.split()]}

    def apply_chat_template(self, msgs, **_):
        return [1, 2] + self(msgs[0]["content"])["input_ids"] + [3]


class M:
    """PeftModel's surface the tools use, around a bare model (no adapters)."""

    def __init__(self, hf):
        self.hf = hf
        self.base_model = type("B", (), {"model": hf})()

    def __call__(self, **kw):
        return self.hf(**kw)

    def disable_adapter(self):
        return contextlib.nullcontext()

    def set_adapter(self, _):
        pass

    def generate(self, **kw):
        return self.hf.generate(**kw)


class Target:
    def __init__(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=V, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
        hf = Qwen3ForCausalLM(cfg).eval()
        self.m, self.torch, self.tok, self.dev = M(hf), torch, Tok(), torch.device("cpu")
        self.layers, self.final_norm, self.lm_head = hf.model.layers, hf.model.norm, hf.lm_head
        self.act_layer = 1

    def use(self, adapter):
        return contextlib.nullcontext()

    def hidden(self, ids):
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([ids]), output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]), out.logits[0]

    def inject(self, prompt, acts, k, sample, adapter_ao, layer=None):
        return ["an owl"] * k


@pytest.fixture
def tools(monkeypatch):
    rng = random.Random(0)
    monkeypatch.setattr(h2h, "texts", lambda tok, r, n: [[rng.randrange(V) for _ in range(T)] for _ in range(n)])
    its = [{"id": "t0", "kind": "text", "ids": [rng.randrange(V) for _ in range(T)]},
           {"id": "e0", "kind": "edit", "ids": [rng.randrange(V) for _ in range(T)], "z": 5},
           {"id": "o0", "kind": "owl", "context": "w4 w9 w11", "truth": "owl"}]
    return agent.Tools(Target(), its)


def test_every_tool_charges_its_passes_and_refuses_past_the_budget(tools):
    tools.start("e", "t0", 16)
    got = tools.call("e", "predict", {})
    assert got["cost"] == 1 and got["left"] == 15
    assert tools.call("e", "predict", {})["cost"] == 0  # the unedited run of a text: once per episode
    assert tools.call("e", "swap", {"swaps": [{"position": 3, "token": "w7"}, {"position": 4, "token": "w8"}]})["cost"] == 2
    assert tools.call("e", "influence", {"positions": [2, 5], "draws": 2})["cost"] == 1 + 4  # the distributions once
    assert tools.call("e", "influence", {"positions": [6], "draws": 1})["cost"] == 1
    assert tools.call("e", "logit_lens", {})["cost"] == 1
    assert tools.call("e", "units", {"position": -1, "k": 5})["cost"] == 1
    assert tools.call("e", "remove", {"units": ["L1.N3", "L2.H1"]})["cost"] == 1
    assert tools.call("e", "ao", {"question": "What is it about?"})["cost"] == 1
    left = tools.eps["e"]["budget"] - tools.eps["e"]["passes"]
    got = tools.call("e", "edit", {"ops": [{"op": "scale", "unit": "L0.N1", "factor": 0}], "controls": True})
    assert "error" in got and got["cost"] == 0 and got["left"] == left  # 17 passes asked, fewer left: refused, free
    got = tools.call("e", "generate", {"max_tokens": 2})
    assert 1 <= got["cost"] <= 2 and got["left"] == left - got["cost"]  # one pass per token generated


def test_bad_calls_are_errors_at_no_cost(tools):
    tools.start("e", "t0", 64)
    for name, args in [("remove", {"units": ["L9.N1"]}), ("remove", {"units": ["L1.H2+"]}), ("swap", {"swaps": [{"position": 99, "token": "w1"}]}),
                       ("swap", {"swaps": [{"position": 2, "token": "w1 w2"}]}), ("nothing", {}), ("ao", {"question": "q", "layer": 3})]:
        got = tools.call("e", name, args)
        assert "error" in got and got["cost"] == 0, (name, got)


def test_removing_a_neuron_is_zeroing_its_output_weights(tools):
    it = tools.items["t0"]
    ids = it["ids"]
    down = tools.tg.layers[1].mlp.down_proj
    got = tools.logp(it, [ids], [tools.scale_hook("L1.N7", 0.0)])[0]
    w = down.weight.data.clone()
    down.weight.data[:, 7] = 0
    want = tools.logp(it, [ids])[0]
    down.weight.data = w
    assert torch.allclose(got, want, atol=1e-5)


def test_a_pole_removal_zeroes_only_that_sign():
    tools_ = agent.Tools.__new__(agent.Tools)
    tools_.torch, tools_.L, tools_.I, tools_.H, tools_.hd = torch, 1, 3, 1, 3
    lin = torch.nn.Linear(3, 2)
    tools_.tg = type("T", (), {"layers": [type("Ly", (), {"mlp": type("Mlp", (), {"down_proj": lin})(), "self_attn": None})()]})()
    x = torch.tensor([[[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]]])
    for unit, want in [("L0.N1", [0.0, 0.0]), ("L0.N1+", [-2.0, 0.0]), ("L0.N1-", [0.0, 2.0])]:
        _, fn, pre = tools_.scale_hook(unit, 0.0)
        out = fn(lin, (x.clone(),))[0]
        assert pre and out[0, :, 1].tolist() == want and torch.equal(out[0, :, [0, 2]], x[0, :, [0, 2]])


def test_no_edit_scores_no_collateral_and_an_invalid_edit_scores_zero(tools):
    it = tools.items["e0"]
    sc = tools.edit_score(it, [])
    assert sc["collateral"] < 1e-6 and sc["kept"] == 1.0
    assert tools.edit_score(it, [{"op": "melt"}])["score"] == 0.0


def test_organism_items_default_to_their_chat_prompt(tools, monkeypatch):
    monkeypatch.setattr(agent, "AO_LAYERS", (1,))
    it = tools.items["o0"]
    assert tools.ids_of(it) == [1, 2, 4, 9, 11, 3]
    assert tools.ids_of(it, "w5", chat=False) == [5]
    tools.start("e", "o0", 16)
    assert "owl" in tools.call("e", "ao", {"question": "What animal?", "layer": 1})["result"]


def test_answers_are_graded_by_the_batterys_rules():
    tok = Tok()
    it = {"kind": "text", "a": {"gt": 7}, "b": [{"changes": True, "new_top": 9}, {"changes": False}], "c": [{"pair": [3, 8], "gt": 8}]}
    ans = agent.parse('thinking... <answer>{"a": 7, "b": [{"changes": true, "new": " w9"}, {"changes": true, "new": "w1"}], "c": [3]}</answer>')
    assert [g["ok"] for g in agent.grade(it, ans, tok)] == [True, True, False, False]
    assert [g["ok"] for g in agent.grade(it, None, tok)] == [False, False, False, False]
    assert agent.grade({"kind": "persona", "truth": "Jollof rice"}, agent.parse('<answer>{"answer": "jollof rice"}</answer>'), tok)[0]["ok"]
    assert agent.grade({"kind": "owl"}, agent.parse("<answer>Owls</answer>"), tok)[0]["ok"]
    assert agent.grade({"kind": "fact", "gt": 1}, {"answer": "Second"}, tok)[0]["ok"]


def test_spend_holds_the_worst_case_until_settled():
    s = agent.Spend(1.0, 0.2)
    assert s.reserve(0.5) and not s.reserve(0.4)
    s.settle(0.5, 0.1)
    assert abs(s.spent - 0.3) < 1e-12 and s.reserve(0.6)
