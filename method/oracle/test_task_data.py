"""Partition boundaries are enforced before opening any task contents."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_data import task_paths


@pytest.fixture
def texts(tmp_path):
    directory = tmp_path / "vpd4l"
    directory.mkdir()
    for stem in ("text5999", "text6000", "text7999", "text8000", "text9000", "text9099",
                 "text9100", "text9149", "text9150", "text9199", "text9200", "hard9300",
                 "hard9349", "hard9350", "textoops", "._text9000"):
        # Intentionally invalid JSON: enumeration must never inspect file contents.
        (directory / f"{stem}.json").write_text("must not read")
    return directory


@pytest.mark.parametrize("split,expected", [
    ("train", ["text6000", "text7999"]),
    ("heldout", ["text9000", "text9099"]),
    ("dev", ["text9000", "text9099"]),
    ("donor", ["text9150", "text9199"]),
    ("hard", ["hard9300", "hard9349"]),
])
def test_partitions_never_open_texts(texts, monkeypatch, split, expected):
    def forbidden(*args, **kwargs):
        raise AssertionError("enumerating tasks opened a file")

    monkeypatch.setattr(Path, "open", forbidden)
    assert [p.stem for p in task_paths(texts, split)] == expected


@pytest.mark.parametrize("split", ["test", "sealed", "all", "typo"])
def test_unknown_or_sealed_split_rejected(texts, split):
    with pytest.raises(ValueError, match="unsupported VPD4L split"):
        task_paths(texts, split)


def test_native_loader_uses_partition_without_reading(texts, monkeypatch):
    import native

    monkeypatch.setattr(native, "TEXTS", texts.parent)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("opened task contents"))
    assert [p.stem for p in native.tasks("heldout")] == ["text9000", "text9099"]


def test_search_donors_are_text6000_to_6399_or_refused(tmp_path, monkeypatch):
    """The reward's far donors are text6000-6399 whatever else is present (the 400 production pods drew as the first
    train files: their rewards unchanged), and a machine missing one refuses rather than draw others."""
    import json
    from types import SimpleNamespace

    import goal_eval
    import native

    directory = tmp_path / "vpd4l"
    directory.mkdir()
    for n in list(range(6000, 6450)) + [9000, 9150]:
        (directory / f"text{n}.json").write_text(json.dumps({"prompts": [{"token_ids": [1, n % 7], "target_positions": [1]}]}))
    monkeypatch.setattr(native, "TEXTS", tmp_path)
    nat = SimpleNamespace(target=SimpleNamespace(n_tokens=goal_eval.DONOR_VOCAB))
    got = goal_eval.GoalEval(nat).far_pool("search")
    old = [p.stem for p in native.tasks("train")][0:400]  # the pool before: the first 400 train files present
    assert [n for n, _ in got] == old == [f"text{n}" for n in range(6000, 6400)] and got[5][1] == [1, 6005 % 7]
    (directory / "text6123.json").unlink()
    with pytest.raises(FileNotFoundError, match="text6123"):
        goal_eval.GoalEval(nat).far_pool("search")


def test_far_donors_use_explicit_ids_with_missing_development_files(texts, monkeypatch):
    import json
    from types import SimpleNamespace

    import goal_eval
    import native

    for row in (9150, 9199):
        (texts / f"text{row}.json").write_text(json.dumps({
            "split": "heldout", "prompts": [{"token_ids": [1, 2], "target_positions": [1]}],
        }))
    monkeypatch.setattr(native, "TEXTS", texts.parent)
    nat = SimpleNamespace(target=SimpleNamespace(n_tokens=goal_eval.DONOR_VOCAB))
    ge = goal_eval.GoalEval(nat)
    assert ge.far_pool("eval") == [("text9150", [1, 2]), ("text9199", [1, 2])]


def test_foreign_vocabulary_requires_own_donor_pool_before_task_read(monkeypatch):
    from types import SimpleNamespace
    import goal_eval
    import native

    monkeypatch.setattr(native, "tasks", lambda *a: pytest.fail("foreign model opened VPD donors"))
    ge = goal_eval.GoalEval(SimpleNamespace(target=SimpleNamespace(n_tokens=40)))
    with pytest.raises(ValueError, match="set donor_pools"):
        ge.far_pool("search")
    ge.donor_pools["search"] = [("own-tokenizer", [1, 39])]
    assert ge.far_pool("search") == [("own-tokenizer", [1, 39])]
    ge.donor_pools["search"] = [("outside-vocabulary", [40])]
    with pytest.raises(ValueError, match="outside this model"):
        ge.far_pool("search")


@pytest.mark.parametrize("task", ["text5999", "text8000", "text9000", "text9099", "text9100",
                                  "text9149", "text9150", "text9199", "text9300", "hard9300",
                                  "6000", "textoops", None])
def test_tool_training_rejects_other_partitions_before_service(task, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import tools_tinker

    monkeypatch.setattr(tools_tinker, "service_client", lambda: pytest.fail("created paid client"))
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"task": task, "messages": [{}], "tools": [{}], "answered": True}) + "\n")
    with pytest.raises(SystemExit, match="training requires"):
        tools_tinker.sft(SimpleNamespace(episodes=episodes))
    if task is not None:
        with pytest.raises(SystemExit, match="training requires|sealed final"):
            tools_tinker.rl(SimpleNamespace(tasks=[task]))


def test_tool_training_accepts_train_boundaries_and_keeps_dev_evaluation():
    import tools_tinker

    tasks = tools_tinker._tasks(["text6000:text6001", "text7999"])
    assert tools_tinker._training_tasks(tasks) == ["text6000", "text6001", "text7999"]
    assert tools_tinker._tasks(["text9000:text9001"]) == ["text9000", "text9001"]


def test_tool_training_rejects_empty_input_before_service(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tools_tinker

    monkeypatch.setattr(tools_tinker, "service_client", lambda: pytest.fail("created paid client"))
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("")
    with pytest.raises(SystemExit, match="no training tasks"):
        tools_tinker.sft(SimpleNamespace(episodes=episodes))
    with pytest.raises(SystemExit, match="no training tasks"):
        tools_tinker.rl(SimpleNamespace(tasks=[]))
