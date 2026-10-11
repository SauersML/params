"""The vLLM reader reads with base weights after a policy push: python -m pytest method/oracle/rl/test_reader_frozen.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from train import *  # noqa: E402,F401,F403  (the test drives make_reader and VllmSampler with a mock engine)


def test_reader_restores_base_after_policy_push(monkeypatch):
    from types import SimpleNamespace
    from peft.tuners.lora.layer import Linear
    from peft import LoraConfig
    import transformers
    import mech

    layer = Linear(torch.nn.Linear(2, 2, bias=False), "default", config=LoraConfig(), r=1, lora_alpha=1)
    with torch.no_grad():
        layer.base_layer.weight.fill_(2)
        layer.lora_A["default"].weight.fill_(1)
        layer.lora_B["default"].weight.fill_(3)
    seen = []

    class Engine:
        def load_weights(self, weights):
            pairs = list(weights)
            self.weight = pairs[0][1].clone()
            return {n for n, _ in pairs}

        def apply_model(self, fn):
            return [fn(self)]

        def reset_prefix_cache(self):
            pass

        def generate(self, *args, **kwargs):
            seen.append(self.weight.clone())
            if getattr(self, "fail", False):
                raise RuntimeError("mock generation failure")
            lp = {1: SimpleNamespace(decoded_token="Yes", logprob=-0.1)}
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[lp])])]

    sampler = VllmSampler.__new__(VllmSampler)
    sampler.policy = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [("base_model.model.q_proj", layer)]))
    sampler.llm, sampler.ready, sampler.held, sampler.share = Engine(), True, 1, False
    sampler.push_weights()
    policy_weight = sampler.llm.weight.clone()
    tk = SimpleNamespace(apply_chat_template=lambda *a, **kw: "question", decode=lambda ids: "token")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: tk)
    monkeypatch.setattr(mech, "tokenizer", lambda *a: tk)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    rd = make_reader(SimpleNamespace(base="mock", model="mock"), None, sampler)
    assert rd.generate(["question"]) == [{"Yes": -0.1}]
    assert torch.equal(seen[0], layer.base_layer.weight)
    assert not torch.equal(seen[0], policy_weight)
    assert torch.equal(sampler.llm.weight, policy_weight)
    import pytest
    sampler.llm.fail = True
    with pytest.raises(RuntimeError, match="mock generation failure"):
        rd.generate(["question"])
    assert torch.equal(sampler.llm.weight, policy_weight)
