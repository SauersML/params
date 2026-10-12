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
    nat = SimpleNamespace(target=SimpleNamespace(wte=SimpleNamespace(shape=(40, 8))))
    ge = goal_eval.GoalEval(nat)
    assert ge.far_pool("eval") == [("text9150", [1, 2]), ("text9199", [1, 2])]
