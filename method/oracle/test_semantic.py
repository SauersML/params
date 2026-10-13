"""Checks of semantic.py without a model: INVARIANTS lines parse into interventions and outcomes (vpd4l's tokenizer),
outcomes are priced by their surprise against the no-change prediction (a claim of the base top earns nothing, a bold
right one earns -log2 p_base(y), a bold wrong one loses it), and fit_gate finds the gate behind transplant outcomes.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_semantic.py
"""

import math
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import components  # noqa: E402
import semantic  # noqa: E402


def test_invariants_parse():
    try:
        each = components._token_of("vpd4l", "' each'")
    except Exception as e:  # no tokenizer on this machine
        pytest.skip(str(e))
    text = "\n".join(["I1: V1 from @233=' all' -> top ' all'", "I2: @232=' of' -> unchanged",
                      "I3: V2 from @232=' were', V1 from @233=' all' -> not ' each'", "I4: remove V1 -> p(top) < 0.5",
                      "I5: @232=' of' -> sideways"])
    got, bad = semantic.parse_invariants(text)
    assert bad == 1 and [x["id"] for x in got] == [1, 2, 3, 4]
    assert got[0]["intervention"]["transplants"][0][0] == "V1" and got[0]["outcome"][0] == "top"
    assert got[1]["intervention"]["tokens"] == {232: components._token_of("vpd4l", "' of'")} and got[1]["outcome"] == ("unchanged",)
    assert [n for n, _ in got[2]["intervention"]["transplants"]] == ["V2", "V1"] and got[2]["outcome"] == ("not", each)
    assert got[3]["intervention"]["remove"] == ["V1"] and got[3]["outcome"] == ("drop", 0.5)


def _checker(p0: list[float]):
    ck = object.__new__(semantic.Checker)
    ck.p0 = torch.tensor(p0).log()
    ck.y0 = int(ck.p0.argmax())
    return ck


def test_outcomes_priced_against_no_change():
    ck = _checker([0.7, 0.2, 0.1])
    moved = torch.tensor([0.1, 0.8, 0.1]).log()
    r = ck.outcome(moved, ("top", 1))
    assert r["holds"] and math.isclose(r["surprise"], -math.log2(0.2), rel_tol=1e-6) and r["credit"] > 0
    r = ck.outcome(moved, ("top", 2))
    assert not r["holds"] and math.isclose(r["credit"], math.log2(0.1), rel_tol=1e-6)
    assert ck.outcome(moved, ("top", 0))["surprise"] == 0.0  # the base top: no claim beyond no change
    r = ck.outcome(moved, ("not", 0))
    assert r["holds"] and math.isclose(r["surprise"], -math.log2(0.3), rel_tol=1e-6)
    assert ck.outcome(moved, ("unchanged",))["credit"] == 0.0
    r = ck.outcome(moved, ("drop", 0.5))
    assert r["holds"] and math.isclose(r["surprise"], 1.0, rel_tol=1e-6)
    assert math.isclose(ck.outcome(moved, ("top", 1))["logp"], math.log2(0.8), rel_tol=1e-5)


def test_fit_gate():
    rows = [(a, b, 23 if (a and b) else 24) for a in (False, True) for b in (False, True)] * 2
    assert semantic.fit_gate(rows, 23)[0] == ("AND", 1.0)
    rows = [(a, b, 23 if a == b else 24) for a in (False, True) for b in (False, True)]
    assert semantic.fit_gate(rows, 23)[0] == ("XNOR", 1.0)


def test_probability_rise_prices_true_false_and_strict_boundary():
    ck = _checker([0.6, 0.4])
    q = torch.tensor([0.72, 0.28]).log()
    result = ck.outcome(q, ("rise", 1.1))
    assert result["holds"] and math.isclose(result["credit"], math.log2(1.1))
    for q in (ck.p0, torch.tensor([0.5, 0.5]).log()):
        result = ck.outcome(q, ("rise", 1.1))
        assert not result["holds"] and math.isclose(result["credit"], -math.log2(1.1))
    boundary = ck.p0.double().clone()
    boundary[0] = float(ck.p0[0]) + math.log(1.1)
    assert not ck.outcome(boundary, ("rise", 1.1))["holds"]


def test_probability_rise_parse_preserves_text_and_deduplicates():
    text = 'I1: remove V1 -> p(top) > 1.1 * base'
    got, bad = semantic.parse_invariants(text)
    assert bad == 0 and got[0]["outcome"] == ("rise", 1.1) and got[0]["line"] == text
    got, bad = semantic.parse_invariants(text + '\nI2: remove V1 -> p(top)>1.10*base')
    assert len(got) == 1 and bad == 1


@pytest.mark.parametrize("factor", ['1', '0.9', '0', '-2', 'nan', 'inf', '1e309', '9' * 400, '1.1 * base unless V2'])
def test_probability_rise_rejects_invalid_factors(factor):
    got, bad = semantic.parse_invariants(f'I1: remove V1 -> p(top) > {factor} * base')
    assert got == [] and bad == 1


def test_probability_rise_does_not_reinterpret_legacy_drop():
    got, bad = semantic.parse_invariants('I1: remove V1 -> p(top) < 0.8')
    assert bad == 0 and got[0]["outcome"] == ("drop", .8)


@pytest.fixture
def invariant_tokenizer(monkeypatch):
    tokens = {"' x'": 1, "' y'": 2, "' z'": 3, "' a,b'": 4}
    monkeypatch.setattr(components, "_token_of", lambda model, text: tokens.get(text))


def test_invariant_quote_styles_resolve_identical_tokens(invariant_tokenizer):
    single, bad = semantic.parse_invariants("I1: V1 from @0=' x' -> top ' y'")
    double, other_bad = semantic.parse_invariants('I1: V1 from @0=" x" -> top " y"')
    assert bad == other_bad == 0
    assert single[0]["intervention"] == double[0]["intervention"]
    assert single[0]["outcome"] == double[0]["outcome"]
    assert double[0]["line"] == 'I1: V1 from @0=" x" -> top " y"'


@pytest.mark.parametrize("literal", ['"x"', '" missing"', '" x "'])
def test_invariant_quote_normalization_keeps_exact_token_text(invariant_tokenizer, literal):
    got, bad = semantic.parse_invariants(f'I1: @0={literal} -> top " y"')
    assert got == [] and bad == 1


def test_invariant_quote_normalization_preserves_shown_escaping(monkeypatch):
    token = '<|endoftext|>'
    spelling = components.shown(token)
    monkeypatch.setattr(components, "_token_of", lambda model, text: 7 if text == spelling else None)
    for literal in (spelling, '"<|endoftext|>"'):
        got, bad = semantic.parse_invariants(f'I1: @0={literal} -> top {literal}')
        assert bad == 0 and got[0]["intervention"]["tokens"] == {0: 7}
        assert got[0]["outcome"] == ("top", 7)


def test_invariant_equivalent_quotes_cannot_duplicate_credit(invariant_tokenizer):
    got, bad = semantic.parse_invariants('I1: @0=\' x\' -> top \' y\'\nI2: @0=" x" -> top " y"')
    assert len(got) == 1 and bad == 1


@pytest.mark.parametrize("donor", ["@0=' x' unless V2 is false", "if V2 @0=' x'",
                                   "@0=' x' and @1=' z'", "@0=' x' @0=' z'",
                                   "@0=' x' @0=' x'", "@0=' missing'"])
def test_invariant_donor_claim_must_be_fully_executable(invariant_tokenizer, donor):
    got, bad = semantic.parse_invariants(f"I1: V1 from {donor} -> top ' y'")
    assert got == [] and bad == 1


def test_invariant_donor_preserves_complete_multiple_changes(invariant_tokenizer):
    got, bad = semantic.parse_invariants("I1: V1 from @0=' x' @1=' a,b' -> top ' y'")
    assert bad == 0
    assert got[0]["intervention"]["transplants"] == [("V1", {0: 1, 1: 4})]


def test_unexecuted_condition_cannot_earn_verified_english_credit(invariant_tokenizer, monkeypatch):
    executed = []
    class FakeChecker:
        def __init__(self, *args):
            pass
        def check(self, items):
            executed.extend(items)
            return [{"holds": True, "credit": 3.0} for _ in items]
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    good = "I1: V1 from @0=' x' -> top ' y'"
    assert semantic.check_invariants(None, None, [{"name": "V1"}], good)["credit"] == 3.0
    executed.clear()
    unsupported = "I1: V1 from @0=' x' unless V2 is false -> top ' y'"
    result = semantic.check_invariants(None, None, [{"name": "V1"}], unsupported)
    assert result["bad"] == 1 and result["holds"] == 0.0 and result["credit"] == -1.0
    assert result["invariants"] == [] and executed == []


@pytest.mark.parametrize("intervention", ["@0=' x', @0=' z'", "@0=' x', @0=' x'",
                                          "V1 from @0=' x', V1 from @0=' z'",
                                          "V1 from @0=' x', remove V1", "remove V1, V1 from @0=' x'",
                                          "remove V1, remove V1"])
def test_invariant_cannot_overwrite_a_stated_intervention(invariant_tokenizer, intervention):
    got, bad = semantic.parse_invariants(f"I1: {intervention} -> top ' y'")
    assert got == [] and bad == 1


def test_distinct_carriers_can_use_different_donors_at_the_same_position(invariant_tokenizer):
    got, bad = semantic.parse_invariants("I1: V1 from @0=' x', V2 from @0=' z' -> top ' y'")
    assert bad == 0
    assert got[0]["intervention"]["transplants"] == [("V1", {0: 1}), ("V2", {0: 3})]


@pytest.mark.parametrize("copies", [2, 10])
def test_repeated_invariant_ids_cannot_inflate_evidence_credit(invariant_tokenizer, monkeypatch, copies):
    class FakeChecker:
        def __init__(self, *args):
            pass
        def check(self, items):
            assert len(items) == 1
            return [{"holds": True, "credit": 3.0}]
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    claim = "I1: V1 from @0=' x' -> top ' y'"
    result = semantic.check_invariants(None, None, [{"name": "V1"}], "\n".join([claim] * copies))
    assert len(result["invariants"]) == 1
    assert result["bad"] == copies - 1
    assert result["credit"] == 3.0 - (copies - 1)


@pytest.mark.parametrize("fraction", ["0", "1", "2", "9999999999999999999999999999999999999999"])
def test_drop_claim_requires_a_strict_probability_reduction(invariant_tokenizer, fraction):
    got, bad = semantic.parse_invariants(f"I1: remove V1 -> p(top) < {fraction}")
    assert got == [] and bad == 1


def test_valid_drop_fraction_keeps_priced_outcome(invariant_tokenizer):
    got, bad = semantic.parse_invariants("I1: remove V1 -> p(top) < 0.5")
    assert bad == 0
    assert got[0]["outcome"] == ("drop", 0.5)


@pytest.mark.parametrize("claims", [
    ["I1: V1 from @0=' x' -> top ' y'", "I2: V1 from @0=' x' -> top ' y'"],
    ["I1: V1 from @0=' x', V2 from @1=' z' -> top ' y'",
     "I2: V2 from @1=' z', V1 from @0=' x' -> top ' y'"],
    ["I1: V1 from @0=' x' @1=' z' -> top ' y'", "I2: V1 from @1=' z' @0=' x' -> top ' y'"],
    ["I1: @0=' x', @1=' z', remove V1, remove V2 -> top ' y'",
     "I2: remove V2, @1=' z', remove V1, @0=' x' -> top ' y'"],
])
def test_same_evidence_under_different_ids_cannot_multiply_credit(invariant_tokenizer, monkeypatch, claims):
    class FakeChecker:
        def __init__(self, *args):
            pass
        def check(self, items):
            assert len(items) == 1
            return [{"holds": True, "credit": 3.0}]
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    result = semantic.check_invariants(None, None, [{"name": "V1"}, {"name": "V2"}], "\n".join(claims))
    assert len(result["invariants"]) == 1
    assert result["bad"] == 1 and result["credit"] == 2.0


def test_different_outcomes_and_values_are_distinct_evidence(invariant_tokenizer):
    text = "\n".join(["I1: V1 from @0=' x' -> top ' y'", "I2: V1 from @0=' x' -> not ' y'",
                       "I3: V1 from @0=' z' -> top ' y'", "I4: V2 from @0=' x' -> top ' y'"])
    got, bad = semantic.parse_invariants(text)
    assert bad == 0 and len(got) == 4


@pytest.fixture
def teacher_checker(monkeypatch):
    from types import SimpleNamespace

    state = {"calls": 0, "checks": []}
    class FakeChecker:
        def __init__(self, *args):
            self.y0 = 0
        def text(self, changes):
            y = 2 if list(changes.values()) == [2] else 0
            return {"lq": torch.nn.functional.one_hot(torch.tensor([y]), 4).float()}
        def check(self, items):
            state["calls"] += 1
            rows = [{"intervention": copy.deepcopy(iv), "outcome": list(out), "holds": out[0] == "top",
                     "surprise": 2.0 if out[0] == "top" else 1.0, "credit": 2.0 if out[0] == "top" else -1.0,
                     "top": 2, "p_top": .75, "logp": -0.5} for iv, out in items]
            state["checks"] = rows
            return rows
        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kw: torch.tensor([2, 3]))
    monkeypatch.setattr(components, "token_strings", lambda model, ids: [" x" if z == 2 else " y" for z in ids])
    monkeypatch.setattr(components, "_token_of", lambda model, text: {"' x'": 2, "' y'": 3}.get(text))
    sc = SimpleNamespace(nat=SimpleNamespace(reference=lambda *args: torch.full((1, 1, 4), .25).log()))
    tx = SimpleNamespace(ids=[0], seed=1)
    vs = [{"name": "V1", "inputs": [0], "group": [("h.0.mlp.c_fc", 0, 0)],
           "values": [{"position": 0, "classes": [{"id": 0, "token_ids": [0]},
                      {"id": 1, "token_ids": [1], "predicts_id": 2}]}]}]
    return sc, tx, vs, state


def test_teacher_structured_evidence_preserves_default_and_failed_checks(teacher_checker):
    sc, tx, vs, state = teacher_checker
    plain = semantic.teacher_invariants(sc, tx, vs, fresh=2)
    retained = semantic.teacher_invariants(sc, tx, vs, fresh=2, structured=True)
    assert state["calls"] == 2  # one unchanged checker execution per invocation
    assert [{k: v for k, v in row.items() if k != "executed"} for row in retained] == plain
    assert [row["holds"] for row in retained] == [True, False]
    assert retained[0]["executed"]["intervention"] == {"transplants": [("V1", {0: 2})]}
    assert retained[0]["executed"]["outcome"] == ["top", 2]
    state["checks"][0]["intervention"]["transplants"][0][1][0] = 3
    assert retained[0]["executed"]["intervention"]["transplants"][0][1][0] == 2


def test_teacher_nonroundtripping_render_cannot_become_supervision(teacher_checker, monkeypatch):
    sc, tx, vs, _ = teacher_checker
    rows = semantic.teacher_invariants(sc, tx, vs, fresh=2, structured=True)
    monkeypatch.setattr(components, "_token_of", lambda model, text: 1 if text == "' x'" else None)
    raw = [{"name": "V1", "kept": True, "inputs": [0], "group": [["h.0.mlp.c_fc", 0, 0]], "feeds": []}]
    with pytest.raises(ValueError, match="execution IDs"):
        components.invariant_artifact("text6002", "vpd4l", raw, rows, task_sha256="a" * 64,
                                      circuit_sha256="b" * 64, source_sha256={"semantic": "c" * 64})


@pytest.mark.parametrize("task", ["text5999", "text8000", "text9000", "text9100"])
def test_teacher_supervision_rejects_nontraining_ids_before_model(monkeypatch, tmp_path, task):
    import var_teacher

    monkeypatch.setattr(var_teacher.native, "Native", lambda *args: pytest.fail("model loaded for forbidden supervision"))
    monkeypatch.setattr(sys, "argv", ["var_teacher", task, "--circuits", str(tmp_path),
                                      "--invariants-teach-out", str(tmp_path / "labels.jsonl")])
    with pytest.raises(SystemExit) as error:
        var_teacher.main()
    assert error.value.code == 2


def test_teacher_cli_emits_bound_artifact_without_extra_checks(monkeypatch, tmp_path):
    import var_teacher

    task = {"id": "text6002", "model": "vpd4l", "split": "train"}
    texts = tmp_path / "texts" / "vpd4l"; texts.mkdir(parents=True)
    task_file = texts / "text6002.json"; task_file.write_text(json.dumps(task))
    circuits = tmp_path / "circuits"; circuits.mkdir()
    circuit_file = circuits / "text6002.json"; circuit_file.write_text('{"nodes": [], "edges": []}')
    original_circuit = circuit_file.read_bytes()
    calls = []
    variable = {"name": "V1", "kept": True, "inputs": [0], "group": [["h.0.mlp.c_fc", 0, 0]], "feeds": []}
    row = {"line": "I1: remove V1 -> p(top) < 0.5", "executed": {"intervention": {"remove": ["V1"]},
           "outcome": ["drop", .5], "holds": True, "credit": 1.0, "surprise": 1.0}}
    def fake_teach(ge, passed_task, answer, *args, **kwargs):
        assert passed_task == task and kwargs["retain_invariant_evidence"] is True
        calls.append(True)
        circuit_file.write_text('{"changed_after_scoring_input": true}')
        return {"variables": [variable], "invariants": [row]}
    monkeypatch.setattr(var_teacher, "teach", fake_teach)
    monkeypatch.setattr(var_teacher.native, "TEXTS", texts.parent)
    monkeypatch.setattr(var_teacher.native, "Native", lambda *args: object())
    monkeypatch.setattr(var_teacher.goal_eval, "GoalEval", lambda nat: object())
    labels = tmp_path / "labels.jsonl"
    monkeypatch.setattr(sys, "argv", ["var_teacher", "text6002", "--circuits", str(circuits), "--out", str(tmp_path / "out"),
                                      "--invariants-teach-out", str(labels)])
    var_teacher.main()
    record = json.loads(labels.read_text())
    assert calls == [True] and record["variables"] == [variable]
    assert record["task_sha256"] == hashlib.sha256(task_file.read_bytes()).hexdigest()
    assert record["circuit_sha256"] == hashlib.sha256(original_circuit).hexdigest()
    assert record["eligible_lines"] == [row["line"]]


def test_teacher_default_cli_does_not_construct_supervision(monkeypatch, tmp_path):
    import var_teacher

    texts = tmp_path / "texts" / "vpd4l"; texts.mkdir(parents=True)
    (texts / "text9000.json").write_text(json.dumps({"id": "text9000", "model": "vpd4l", "split": "heldout"}))
    calls = []
    def fake_teach(*args, **kwargs):
        calls.append(kwargs["retain_invariant_evidence"])
        return {"variables": [], "invariants": []}
    monkeypatch.setattr(var_teacher, "teach", fake_teach)
    monkeypatch.setattr(var_teacher.native, "TEXTS", texts.parent)
    monkeypatch.setattr(var_teacher.native, "Native", lambda *args: object())
    monkeypatch.setattr(var_teacher.goal_eval, "GoalEval", lambda nat: object())
    monkeypatch.setattr(components, "invariant_artifact", lambda *a, **k: pytest.fail("default CLI constructed supervision"))
    monkeypatch.setattr(sys, "argv", ["var_teacher", "text9000", "--circuits", str(tmp_path), "--out", str(tmp_path / "out")])
    var_teacher.main()
    assert calls == [False]
    assert json.loads((tmp_path / "out/text9000.json").read_text()) == {"variables": [], "invariants": []}


def test_teacher_supervision_overlap_preserves_existing_file_before_model(monkeypatch, tmp_path):
    import var_teacher

    raw = [{"name": "V1", "kept": True, "group": [["h.0.mlp.c_fc", 0, 0]], "inputs": [0], "feeds": []}]
    row = {"line": "I1: remove V1 -> p(top) < 0.5", "executed": {"intervention": {"remove": ["V1"]},
           "outcome": ["drop", .5], "holds": True, "credit": 1.0, "surprise": 1.0}}
    record = components.invariant_artifact("text6002", "vpd4l", raw, [row], task_sha256="a" * 64,
                                          circuit_sha256="b" * 64, source_sha256={"semantic": "c" * 64})
    labels = tmp_path / "labels.jsonl"; labels.write_text(json.dumps(record) + "\n")
    before = labels.read_bytes()
    monkeypatch.setattr(var_teacher.native, "Native", lambda *args: pytest.fail("overlap loaded model"))
    monkeypatch.setattr(var_teacher, "teach", lambda *a, **k: pytest.fail("overlap scored task"))
    monkeypatch.setattr(sys, "argv", ["var_teacher", "text6002", "--circuits", str(tmp_path),
                                      "--invariants-teach-out", str(labels)])
    with pytest.raises(SystemExit) as error:
        var_teacher.main()
    assert error.value.code == 2 and labels.read_bytes() == before


def test_teacher_supervision_duplicate_requested_ids_reject_before_model(monkeypatch, tmp_path):
    import var_teacher

    labels = tmp_path / "labels.jsonl"
    monkeypatch.setattr(var_teacher.native, "Native", lambda *args: pytest.fail("duplicate loaded model"))
    monkeypatch.setattr(var_teacher, "teach", lambda *a, **k: pytest.fail("duplicate scored task"))
    monkeypatch.setattr(sys, "argv", ["var_teacher", "text6002", "text6002", "--circuits", str(tmp_path),
                                      "--invariants-teach-out", str(labels)])
    with pytest.raises(SystemExit) as error:
        var_teacher.main()
    assert error.value.code == 2 and not labels.exists()


def test_rules_out_parse_and_exclusive(invariant_tokenizer):
    good = "RULES OUT: the copy reads position 3 | V1 from @0=' x' -> top ' y' vs unchanged"
    got, bad = semantic.parse_rules_out("\n".join([good, "RULES OUT: no bar here -> top ' y' vs unchanged",
                                                   "RULES OUT: words | @0=' q' -> top ' y' vs unchanged"]))
    assert bad == 2 and got[0]["h"] == ("top", 2) and got[0]["alt"] == ("unchanged",) and got[0]["alternative"] == "the copy reads position 3"
    assert semantic.exclusive(("top", 2), ("unchanged",), y0=1) and not semantic.exclusive(("top", 1), ("unchanged",), y0=1)
    assert semantic.exclusive(("top", 2), ("top", 3), 1) and semantic.exclusive(("not", 2), ("top", 2), 1)
    assert semantic.exclusive(("drop", 0.5), ("rise", 2.0), 1) and not semantic.exclusive(("drop", 0.5), ("unchanged",), 1)


def test_rules_out_credit(invariant_tokenizer, monkeypatch):
    class FakeChecker:
        y0 = 1

        def __init__(self, *args):
            pass

        def run(self, iv):
            return None

        def outcome(self, q, out):
            return {"holds": out[0] == "top", "top": 2, "p_top": 0.9, "logp": -0.2 if out[0] == "top" else -3.0}

        def close(self):
            pass
    monkeypatch.setattr(semantic, "Checker", FakeChecker)
    text = "\n".join(["RULES OUT: a | V1 from @0=' x' -> top ' y' vs unchanged",
                      "RULES OUT: b | V1 from @0=' x' -> top ' y' vs not ' z'",
                      "RULES OUT: c | V2 from @0=' x' -> top ' y' vs unchanged"])
    got = semantic.check_rules_out(None, None, [{"name": "V1"}], text)
    assert [r["ruled_out"] for r in got["rules_out"]] == [True, False] and got["bad"] == 2
    assert got["credit"] == 1.0 - 2 and abs(got["rules_out"][0]["margin"] - 2.8) < 1e-9
