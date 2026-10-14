"""Model-free contracts for the explicitly capped, LIVE, architecture-complete set route."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "vpd")]
import teacher


def fake_set(monkeypatch, initial=0.05, removed=0.08):
    nodes = [("h.0.mlp.down_proj", 0, 0), ("h.0.mlp.down_proj", 0, 1)]
    nat = SimpleNamespace(names=[nodes[0][0]], C={nodes[0][0]: 2}, dev="cpu", positions=lambda ts: 1)
    tx = SimpleNamespace(T=1, targets=[0], f2=[], vpd={nodes[0][0]: torch.ones(1, 2, dtype=torch.bool)})
    graph = teacher.native.Graph(nodes, complete=True)
    start = {"graph": graph, "m": 0, "positions": [0], "s_track_pp": 1.0}
    ge = SimpleNamespace(nat=nat, start_set=lambda *a, **kw: start, own_rule=lambda *a, **kw: {})
    seen = []

    class Pruner:
        def __init__(self, *args, **kwargs):
            self.nodes = nodes
            self.on = torch.ones(2, dtype=torch.bool)
            self.t = torch.zeros(2, dtype=torch.long)

        def kl(self):
            return initial if bool(self.on.all()) else removed

        def kl_sets(self, ons):
            return [1.0] * len(ons)

        def prune(self, level, budget, accept=None, after=None):
            seen.append((budget, accept))
            if removed <= budget and (accept is None or accept(self.on[None]) is not None):
                self.on[1] = False
            return 1

    monkeypatch.setattr(teacher, "Pruner", Pruner)
    return ge, tx, seen


def test_capped_set_keeps_relative_checks_and_never_relaxes_cap(monkeypatch):
    ge, tx, seen = fake_set(monkeypatch)
    hook = lambda on: None
    acceptance = SimpleNamespace(draws="fixed", hook=lambda *a, **kw: (hook, {}), scores=lambda gs: [{}])
    g, pr, _, _, rec = teacher.set_search(ge, tx, 10.0, 1.0, 5.0, acceptance,
                                         levels=("site",), log=lambda *a, **kw: None, direct_kl_max=0.06)
    assert seen == [(0.06, hook)]
    assert int(pr.on.sum()) == g.count() == 2
    assert rec["budget"] == rec["direct_kl_cap_bits"] == 0.06
    # Without opt-in the old relative/start recovery budget and signature stay intact.
    ge, tx, seen = fake_set(monkeypatch)
    teacher.set_search(ge, tx, 10.0, 1.0, 5.0, None, levels=("site",), log=lambda *a, **kw: None)
    assert seen[0][0] == 5.0


def test_capped_set_infeasible_start_is_route_failure_not_target(monkeypatch):
    ge, tx, seen = fake_set(monkeypatch, initial=0.2)
    with pytest.raises(teacher.NoDirectKLTarget) as caught:
        teacher.set_search(ge, tx, 10.0, 1.0, 5.0, None, levels=("site",), direct_kl_max=0.1)
    assert not seen
    assert caught.value.receipt["stage"] == "set_start"
    assert caught.value.receipt["direct_KL_bits"] == 0.2
    assert caught.value.receipt["status"] == "no_target"


def test_final_certificate_uses_exact_serialized_graph_and_rejects_bad_output(monkeypatch):
    ge, tx, _ = fake_set(monkeypatch)
    calls = []
    ge.graph_kl = lambda tx, gs, live: calls.append((gs[0], live)) or [0.04]
    answer = teacher.goal_eval.to_json(teacher.native.Graph([("h.0.mlp.down_proj", 0, 1)], complete=True), "")
    receipt = teacher.certify_set_answer(ge, tx, answer, 0.05)
    assert calls[0][0].node_set() == {("h.0.mlp.down_proj", 0, 1)} and calls[0][1] is True
    assert receipt["answer_sha256"] == teacher.hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest()
    ge.graph_kl = lambda *a: [0.06]
    with pytest.raises(teacher.NoDirectKLTarget, match="serialized_primary"):
        teacher.certify_set_answer(ge, tx, answer, 0.05)
    ge.graph_kl = lambda *a: [math.nan]
    with pytest.raises(RuntimeError, match="nonfinite"):
        teacher.certify_set_answer(ge, tx, answer, 0.05)
    answer["nodes"][0][1] = 999
    with pytest.raises(RuntimeError, match="compiler defects"):
        teacher.certify_set_answer(ge, tx, answer, 0.05)


@pytest.mark.parametrize("cap", [-1.0, math.nan, math.inf])
def test_invalid_cap_rejected_before_start(cap):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        teacher.capped_set_route(cap, "set")


def test_modes_and_existing_output_rejected_before_model(monkeypatch, tmp_path):
    for mode in ("edges", "backup", "forward", "from_path", "pairs", "repair", "robust"):
        with pytest.raises(ValueError, match="plain"):
            teacher.capped_set_route(0.1, "set", **{mode: True})
    monkeypatch.setattr(teacher.native, "Native", lambda *a: pytest.fail("model must not load"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "text6008.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["teacher.py", "--f1-start", "set", "--set-direct-kl-max", "0.1", "--out", str(out)])
    with pytest.raises(SystemExit) as caught:
        teacher.main()
    assert caught.value.code == 2


def test_teach_final_token_count_cannot_bypass_cap(monkeypatch):
    ge, tx, _ = fake_set(monkeypatch)
    ge.text = lambda *a, **kw: tx
    ge._bounds = lambda *a: (10.0, 1.0)
    ge.size_scale = lambda *a: 1.0
    ge.nat._vpd_order = lambda *a: [("h.0.mlp.down_proj", 0, 0), ("h.0.mlp.down_proj", 0, 1)]
    monkeypatch.setattr(teacher.components, "render", lambda answer: json.dumps(answer))
    calls = []
    ge.graph_kl = lambda tx, gs, live: calls.append((gs[0].node_set(), live)) or [0.04]
    circuits, row, path = teacher.teach(ge, [0], [0], f1_start="start_set", tokens=lambda text: 123,
                                      max_tokens=200, set_direct_kl_max=0.06)
    assert list(circuits) == [""] and len(calls) == 1
    assert row["answer_tokens"] == 123 and row["direct_kl_certificate"]["status"] == "pass"
    assert path["blocks"]["direct_kl_cap_bits"] == 0.06
    # Search KL is passing, but the emitted/compiled final can still fail: no primary is returned.
    ge.graph_kl = lambda *a: [0.07]
    with pytest.raises(teacher.NoDirectKLTarget, match="serialized_primary"):
        teacher.teach(ge, [0], [0], f1_start="set", tokens=lambda text: 123,
                      max_tokens=200, set_direct_kl_max=0.06)


def test_cli_no_target_receipt_never_writes_uncertified_primary(monkeypatch, tmp_path):
    task = tmp_path / "text6008.json"
    task.write_text("{}")
    out = tmp_path / "fresh"
    monkeypatch.setattr(teacher.native, "tasks", lambda split: [task])
    monkeypatch.setattr(teacher.native, "text", lambda path: ([0], [0]))
    monkeypatch.setattr(teacher.native, "Native", lambda *a: object())
    monkeypatch.setattr(teacher.goal_eval, "GoalEval", lambda *a, **kw: object())
    monkeypatch.setattr(teacher, "teach", lambda *a, **kw: (_ for _ in ()).throw(
        teacher.NoDirectKLTarget("set_start", 0.1, 0.2)))
    monkeypatch.setattr(sys, "argv", ["teacher.py", "--f1-start", "set", "--set-direct-kl-max", "0.1",
                                     "--max-tokens", "0", "--out", str(out)])
    teacher.main()
    receipt = json.loads((out / "text6008.no_target.json").read_text())
    assert receipt["status"] == "no_target" and receipt["direct_KL_bits"] == 0.2
    assert not (out / "text6008.json").exists() and not (out / "text6008.path.json").exists()
    assert json.loads((out / "summary_0.jsonl").read_text()) == receipt
