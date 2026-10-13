"""Evaluation recovery checks without loading a model or scoring real tasks."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import components


def setup_eval(tmp_path, monkeypatch, fail_second=False):
    paths = []
    for name in ("mock_a", "mock_b"):
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps({"id": name}))
        paths.append(p)
    monkeypatch.setitem(sys.modules, "native", SimpleNamespace(tasks=lambda split: paths))
    monkeypatch.setattr(components, "baseline_texts", lambda *a: {"empty": "baseline"})

    def reward(ge, task, ev, texts, **kwargs):
        if fail_second and task["id"] == "mock_b":
            raise RuntimeError("scoring interrupted")
        return [{"reward": 0.4, "valid": True, "row": {"nodes": 3}, "circuit_reward": 0.2,
                 "term_C": 0.7, "term_N": 0.3, "phi": 0.5} for _ in texts]

    monkeypatch.setattr(components, "reward", reward)
    rows = {p.stem: {"qa": []} for p in paths}
    sets = {"policy": {p.stem: [("answer", False)] for p in paths}}
    return rows, sets


def test_completed_task_survives_next_scoring_failure(tmp_path, monkeypatch):
    rows, sets = setup_eval(tmp_path, monkeypatch, fail_second=True)
    checkpoint = tmp_path / "raw.jsonl"
    real_fsync = os.fsync
    syncs = []

    def sync(fd):
        syncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(RuntimeError, match="scoring interrupted"):
        components.evaluate(None, "heldout", rows, sets, 2, checkpoint=checkpoint, log=lambda *a, **k: None)
    header, task = map(json.loads, checkpoint.read_text().splitlines())
    assert len(syncs) == 2  # header and completed task flushed to disk
    assert header["resume"] is False
    assert len(header["input_sha256"]["sets"]) == 64
    assert "components.py" in header["source_sha256"]
    assert task["task"] == "mock_a"
    assert task["per_question"]["policy"]["samples"][0]["nodes"] == 3
    assert task["baselines"] == {"empty": 0.2}
    assert task["baseline_terms"]["empty"]["C"] == 0.7


def test_report_failure_preserves_all_raw_results_and_missing_answers(tmp_path, monkeypatch):
    rows, sets = setup_eval(tmp_path, monkeypatch)
    sets["policy"].clear()  # baseline scoring must still work without a sample/cache
    monkeypatch.setattr(components, "_report", lambda *a: (_ for _ in ()).throw(ValueError("bad figure")))
    checkpoint = tmp_path / "raw.jsonl"
    result = components.evaluate(None, "heldout", rows, sets, 2, checkpoint=checkpoint, log=lambda *a, **k: None)
    records = list(map(json.loads, checkpoint.read_text().splitlines()))
    assert len(records) == 3
    assert records[1]["per_question"]["policy"]["samples"][0]["valid"] is False
    assert [r["per_question"]["policy"] for r in records[1:]] == result["per_question"]["policy"]
    assert "bad figure" in result["report"]["all"]["error"]


def test_existing_checkpoint_is_never_overwritten(tmp_path, monkeypatch):
    rows, sets = setup_eval(tmp_path, monkeypatch)
    checkpoint = tmp_path / "raw.jsonl"
    checkpoint.write_text("prior measured results\n")
    monkeypatch.setattr(components, "reward", lambda *a, **k: pytest.fail("must refuse before scoring"))
    with pytest.raises(FileExistsError):
        components.evaluate(None, "heldout", rows, sets, 2, checkpoint=checkpoint)
    assert checkpoint.read_text() == "prior measured results\n"


def test_cli_rejects_default_checkpoint_before_loading_model(tmp_path, monkeypatch, capsys):
    out = tmp_path / "evaluation.json"
    checkpoint = Path(str(out) + ".tasks.jsonl")
    checkpoint.write_text("prior results\n")
    monkeypatch.setattr(sys, "argv", ["components", "evaluate", "heldout", str(tmp_path / "rows.jsonl"),
                                    "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        components.main()
    assert exc.value.code == 2
    assert "choose a new --checkpoint path" in capsys.readouterr().err
    assert checkpoint.read_text() == "prior results\n"


def test_circuit_baselines_clear_edit_requests_without_changing_policy(tmp_path, monkeypatch):
    rows, sets = setup_eval(tmp_path, monkeypatch)
    for task in rows:
        rows[task] = {"qa": [{"question": "measured QA"}], "edits": [{"kind": "flip"}]}
        (tmp_path / (task + ".json")).write_text(json.dumps({
            "id": task, "prompts": [{"token_ids": [1], "target_positions": [0]}]}))
    tx = SimpleNamespace(ids=[1], targets=[0])
    ge = SimpleNamespace(text=lambda *a, **k: tx, nat=SimpleNamespace(_vpd_order=lambda *a: []))
    calls = []

    def check_reward(ge, task, ev, texts, **kwargs):
        calls.append(ev)
        if texts == ["baseline"]:
            assert ev["qa"] == [] and ev["edits"] == []
            assert kwargs["node_cap"] is None
            raise RuntimeError("baseline contract checked")
        assert ev is rows[task["id"]]  # the policy retains its requested QA and edits
        return [{"reward": 0.4, "valid": True, "row": {"nodes": 3}}]

    monkeypatch.setattr(components, "reward", check_reward)
    with pytest.raises(RuntimeError, match="baseline contract checked"):
        components.evaluate(ge, "heldout", rows, sets, 2, log=lambda *a, **k: None)
    assert len(calls) == 2
    assert rows["mock_a"]["edits"] == [{"kind": "flip"}]
