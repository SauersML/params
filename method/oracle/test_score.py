"""Checks of score.Scorer's pricing and step identities on stand-ins: python -m pytest method/oracle/test_score.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import Scorer  # noqa: E402


def test_static_covers_all_parameter_kinds():
    assert Scorer.static("def graph(tokens, targets):\n return []")
    for signature, value in [("tokens, /, targets", "tokens"), ("*args", "args"), ("**kwargs", "kwargs"),
                             ("tokens, *, targets", "targets")]:
        assert not Scorer.static(f"def graph({signature}):\n return {value}")


def test_dynamic_cost_and_hold_identity(monkeypatch):
    from types import SimpleNamespace
    import mech
    import native

    source = "def graph(tokens, targets):\n return tokens"
    original = [1, 2]
    changed = [1, 3]
    one = ("h.0.attn.v_proj", 0, 0)
    relay = ("h.1.attn.o_proj", 1, 0)
    small = native.Graph(nodes={one})
    large = native.Graph(nodes={one, relay})
    small.bits = lambda *a: 10.0
    large.bits = lambda *a: 1000.0
    def trace(src, model, behavior):
        graph = small if behavior["prompts"][0]["token_ids"] == original else large
        return {"valid": True, "graph": {"steps": 2, "notes": ["empty", "carry"]}, "mock": graph}
    monkeypatch.setattr(mech, "trace", trace)
    executed = []
    def faithfulness(ids, targets, rows, prompts, seed):
        assert prompts == [changed]
        executed.extend(rows)
        return [5.0, 3.0, 1.0]
    def events(ids, targets, prompts, holds, seed):
        assert holds == [[{one}]]
        return [[{"hold": 1, "down": True}]]
    sc = Scorer.__new__(Scorer)
    sc.native = SimpleNamespace(Graph=native.Graph, task_seed=lambda _: 0,
                                prefix=lambda ir, k: (native.Graph() if k == 1 else ir["mock"], []))
    sc.nat = SimpleNamespace(changes=lambda *a, **kw: [changed], positions=lambda _: 2, C={"a": 4},
                             faithfulness=faithfulness, events=events, everything_bits=lambda _: 10000)
    result = sc.score({"id": "mock", "prompts": [{"token_ids": original, "target_positions": [1]}]}, [source])[0]
    assert result["bits"] == 505.0 and result["original_bits"] == 10.0
    assert executed[0][-1] is large
    assert result["events"] == [{"hold": 2, "down": True, "held_nodes": [one]}]
    assert result["notes"] == ["empty", "carry"]
