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

    def parameters(self):
        return self.hf.parameters()


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
    tools.ao_layers = (0, 1, 2)
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


def test_a_retried_call_is_answered_once_and_charged_once(tools):
    tools.start("e", "t0", 16)
    first = tools.call("e", "swap", {"swaps": [{"position": 3, "token": "w7"}]}, "0.0")
    tools.start("e", "t0", 16)  # a retried start keeps the episode
    assert tools.call("e", "swap", {"swaps": [{"position": 3, "token": "w7"}]}, "0.0") == first
    assert tools.eps["e"]["passes"] == first["cost"] == 2


def test_ao_reads_a_written_text(tools, monkeypatch):
    tools.ao_layers = (0, 1, 2)
    tools.start("e", "t0", 16)
    got = tools.call("e", "ao", {"question": "What?", "layer": 1, "text": "w1 w2 w3"})
    assert got["cost"] == 1 and "owl" in got["result"]


class FakeAtlas:
    def neuron(self, l, i, k=4, text=True):
        ex = lambda v: {"value": v, "regime": "++" if v > 0 else "+-", "before": "the cat sat on the", "token": " mat",  # noqa: E731
                        "after": " today"}
        return {"max": {"+": 2.0, "-": -1.0}, "rates": {"+": {0.1: 0.2, 0.3: 0.05, 0.5: 0.01}, "-": {0.1: 0.1, 0.3: 0.02, 0.5: 0.0}},
                "+": [ex(2.0)] * k, "-": [ex(-1.0)] * k, "logits": {"+": [(" mat", 1.0)], "-": [(" dog", -1.0)]},
                "up": [(0, 3, "gate", 0.5)], "down": [(2, 9, "up", -0.4)]}


def test_steering_adds_alpha_times_the_poles_extreme(tools):
    tools.atlas = FakeAtlas()
    it = tools.items["t0"]
    tools.start("e", "t0", 16)
    got = tools.call("e", "steer", {"units": [{"unit": "L1.N7", "alpha": 1.5}]})
    assert got["cost"] == 2  # the unedited run once, the steered run

    def fn(_m, a_):
        x = a_[0].clone()
        x[..., 7] += 1.5 * 2.0  # alpha x the positive pole's corpus extreme
        return (x,)

    base = tools.logp(it, [it["ids"]])[0]
    want = tools.logp(it, [it["ids"]], [(tools.tg.layers[1].mlp.down_proj, fn, True)])[0]
    assert f"KL {tools.kl(base, want):.3f} bits" in got["result"]
    assert tools.steer_scale(1, 7, -2.0) == 1.0 and tools.steer_scale(1, 7, 2.0) == 2.0  # a negative alpha: the negative pole's


def test_the_atlas_tool_reads_the_record_and_explanations_for_free(tools):
    tools.start("e", "t0", 16)
    assert "error" in tools.call("e", "atlas", {"unit": "L1.N7"})  # neither an atlas nor explanations
    tools.atlas, tools.labels = FakeAtlas(), {"1.7": {"+": "mats and rugs", "-": "dogs"}}
    got = tools.call("e", "atlas", {"unit": "L1.N7", "k": 2})
    assert got["cost"] == 0 and "mats and rugs" in got["result"] and "[[ mat]]" in got["result"] and "L2.N9" in got["result"]
    assert "error" in tools.call("e", "atlas", {"unit": "L1.H2"})


def test_the_attribution_graph_is_value_times_gradient_and_charged(tools):
    tools.start("e", "t0", 64)
    got = tools.call("e", "graph", {"k": 4})
    assert got["cost"] == 3 + 2 * 4, got
    lines = got["result"].splitlines()
    assert lines[0].startswith("contrast at") and sum(x.startswith("  L") and " attribution " in x for x in lines) == 4
    assert all("@0 " not in x for x in lines)  # the first position is never a node
    assert tools.call("e", "graph", {"k": 4, "edges": False})["cost"] == 3


def test_taboo_answers_match_the_word_alone():
    assert agent.grade({"kind": "taboo", "truth": "rock"}, {"answer": "Rock"}, Tok())[0]["ok"]
    assert not agent.grade({"kind": "taboo", "truth": "rock"}, {"answer": "rocket"}, Tok())[0]["ok"]


def test_the_tool_list_names_the_servers_model():
    info = {"model": "Qwen/Qwen3-8B", "L": 36, "H": 32, "I": 12288, "ao_layers": (9, 18, 27), "lens_layers": (0, 3, 35), "atlas": False}
    d = agent.docs(info)
    assert "Qwen3-8B" in d and "layer=18" in d and "9 or 18 or 27" in d and "atlas(" not in d and "<" + "MID>" not in d
    assert "atlas(" in agent.docs({**info, "atlas": True})


def test_a_graph_nodes_attribution_is_its_measured_first_order_effect(tools):
    """value x d contrast / d value through every later layer (the finite difference of the contrast, a small change of
    the node's value)."""
    import math
    import re

    it = tools.items["t0"]
    ids = it["ids"]
    tools.start("e", "t0", 64)
    got = tools.call("e", "graph", {"k": 1, "edges": False})["result"]
    m_ = re.search(r"L(\d+)\.N(\d+)@(\d+) .* value ([-+0-9.e]+) .*attribution ([-+0-9.]+)", got)
    l_, i, t, attr = int(m_.group(1)), int(m_.group(2)), int(m_.group(3)), float(m_.group(5))
    base = tools.logp(it, [ids])[0]
    y = int(base.argmax())
    alts = [x for x in base.topk(6).indices.tolist() if x != y][:5]

    def contrast(eps):
        def fn(_m, a_):
            x = a_[0].clone()
            v = x[0, t, i].item()
            x[0, t, i] = v * (1 + eps)
            return (x,)

        lp = tools.logp(it, [ids], [(tools.tg.layers[l_].mlp.down_proj, fn, True)])[0]
        return float((lp[y] - torch.logsumexp(lp[alts], 0)) / math.log(2))

    eps = 1e-3
    fd = (contrast(eps) - contrast(-eps)) / (2 * eps)  # = value x d contrast / d value
    assert abs(fd - attr) < 0.01 + 0.05 * abs(attr), (fd, attr)
