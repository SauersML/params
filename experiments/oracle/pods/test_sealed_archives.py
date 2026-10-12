"""Run only the input archive commands against synthetic data, never launch a pod."""

import os
from pathlib import Path
import subprocess
import tarfile

import pytest


@pytest.mark.parametrize("launcher", ["eval_pod.sh", "scorer_pod.sh"])
def test_input_archive_skips_sealed_text_contents(tmp_path, launcher):
    source = Path(__file__).with_name(launcher).read_text()
    command, = [line for line in source.splitlines() if line.startswith("COPYFILE_DISABLE=1 tar") and '"$IN.tar"' in line]
    graph = tmp_path / "synthetic_graph"
    texts = graph / "texts" / "vpd4l"
    texts.mkdir(parents=True)
    sealed = []
    for i in range(9100, 9150):
        p = texts / f"text{i}.json"
        p.write_text("SEALED_SYNTHETIC_SENTINEL")
        p.chmod(0)  # successful tar must exclude these before opening their contents
        sealed.append(p)
    for i in (9099, 9150):
        (texts / f"text{i}.json").write_text("allowed synthetic text")
    (graph / "goal_eval").mkdir()
    (graph / "goal_eval" / "ci_patterns_m.pt").write_text("synthetic pattern pool")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "rows.jsonl").write_text("{}\n")
    (inputs / "labels.jsonl").write_text("{}\n")
    (inputs / "sets").mkdir()
    command = command.replace("/Users/user/mpd-data/graph_oracle", '"$TEST_GRAPH"')
    try:
        subprocess.run(["bash", "-c", command], check=True, capture_output=True,
                       env={**os.environ, "IN": str(inputs), "TEST_GRAPH": str(graph), "EDITS": "", "TEACH": ""})
        with tarfile.open(str(inputs) + ".tar") as archive:
            names = set(archive.getnames())
            assert {"texts/vpd4l/text9099.json", "texts/vpd4l/text9150.json"} <= names
            assert all(f"texts/vpd4l/text{i}.json" not in names for i in range(9100, 9150))
            assert "goal_eval/ci_patterns_m.pt" in names
        assert b"SEALED_SYNTHETIC_SENTINEL" not in Path(str(inputs) + ".tar").read_bytes()
    finally:
        for p in sealed:
            p.chmod(0o600)
