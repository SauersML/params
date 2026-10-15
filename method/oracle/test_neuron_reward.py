"""neuron_reward's dispatch on synthetic targets and readers (no model, no reader calls)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("checked_neuron_reward", HERE / "neuron_reward.py")
nrw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nrw)
plan = dict(max_texts=[f"maximum {i}" for i in range(3)], near_misses=[f"near {i}" for i in range(5)],
            distant_texts=[f"distant {i}" for i in range(5)], distant_concept="distant", steering_prediction="claim")


class Target:
    def __init__(self):
        self.layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(in_features=4)))] * 2
        self.name, self.calls = "SYNTHETIC", []

    def poles(self, texts, l, i):
        self.calls.append(("poles", l, i))
        return [dict(pos=1. if x.startswith("maximum") else .1, neg=.1, regime_pos="++", regime_neg="+-") for x in texts]

    def steered(self, l, i, value):
        return "synthetic continuation"


class Reader:
    def __init__(self):
        self.calls = []

    def __call__(self, asks, **kw):
        self.calls.append(asks)
        return [json.dumps(plan)] * len(asks) if len(self.calls) == 1 else ["A"] * len(asks)


def test_explanation_claims_dispatch_in_order_and_bad_ones_fail_without_calls():
    tg, rd = Target(), Reader()
    claims = [{"kind": "explanation", "unit": "L1.N2+", "explanation": "fires on maxima", "corpus_max": 2.0},
              {"kind": "explanation", "unit": "L1.N2", "explanation": "no pole", "corpus_max": 2.0},
              {"kind": "explanation", "unit": "L9.N2-", "explanation": "no such layer", "corpus_max": 2.0},
              {"kind": "guess"},
              {"kind": "intervention", "units": ["L1.N2+"], "op": "zero"}]
    out = nrw.check(claims, tg, rd)
    assert out[0]["pass"] and out[0]["blind_max"] == 0.5 and out[0]["near_miss_rate"] == 0
    assert [o["pass"] for o in out[1:]] == [False] * 4
    assert "pole" in out[1]["reason"] and "layer" in out[2]["reason"] and out[3]["reason"] == "unknown claim kind"
    assert out[4]["kind"] == "intervention" and "reference" in out[4]["reason"]
    assert len(rd.calls[0]) == 1 and tg.calls == [("poles", 1, 2)]


def test_corpus_max_from_the_atlas_pole_extreme():
    atlas = SimpleNamespace(neuron=lambda l, i, k, text: {"max": {"+": 3.0, "-": -4.0}})
    assert nrw.explanation_item({"unit": "L1.N2-", "explanation": "x"}, atlas)["corpus_max"] == 4.0
