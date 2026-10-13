"""Run only the input archive commands against synthetic data, never launch a pod."""

import os
import json
from pathlib import Path
import shlex
import subprocess
import sys
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
    (graph / "goal_eval" / "ci_patterns_train.pt").write_text("synthetic pattern pool")
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
            assert "goal_eval/ci_patterns_train.pt" in names
        assert b"SEALED_SYNTHETIC_SENTINEL" not in Path(str(inputs) + ".tar").read_bytes()
    finally:
        for p in sealed:
            p.chmod(0o600)


def test_all_bulk_text_archives_exclude_sealed_ids(tmp_path):
    """Check each launcher's declared tar exclusions on synthetic boundary files.

    Only tar runs: no source upload, model loading, credentials or pod launch.
    The evaluation launcher's explicit final-evaluation override is not enabled.
    """
    graph = tmp_path / "graph"
    texts = graph / "texts" / "vpd4l"
    texts.mkdir(parents=True)
    ids = [9099, *range(9100, 9150), 9150]
    for i in ids:
        (texts / f"text{i}.json").write_text("synthetic")
    checked = []
    for launcher in sorted(Path(__file__).parent.glob("*.sh")):
        for line in launcher.read_text().splitlines():
            if not line.startswith("COPYFILE_DISABLE=1 tar ") or "texts/vpd4l" not in line:
                continue
            # Resolve only the documented default of the final-evaluation override.
            line = line.replace('${EXCLUDE_TEXTS:-texts/vpd4l/text91[0-4][0-9].json}',
                                'texts/vpd4l/text91[0-4][0-9].json')
            exclusions = [s for s in shlex.split(line.rstrip("\\")) if s.startswith("--exclude=")]
            assert exclusions, f"{launcher.name} archives all text contents without exclusion"
            archive = tmp_path / f"{launcher.stem}.tar"
            subprocess.run(["tar", *exclusions, "-cf", str(archive), "-C", str(graph), "texts/vpd4l"],
                           check=True, capture_output=True)
            with tarfile.open(archive) as f:
                names = set(f.getnames())
            assert {"texts/vpd4l/text9099.json", "texts/vpd4l/text9150.json"} <= names
            assert not any(f"texts/vpd4l/text{i}.json" in names for i in range(9100, 9150)), launcher.name
            checked.append(launcher.name)
    assert len(checked) >= 20, "the bulk archive commands were not discovered"


@pytest.mark.parametrize("explicit_sealed", [False, True])
def test_teacher_selection_never_reads_sealed_tasks(tmp_path, explicit_sealed):
    source = Path(__file__).with_name("teacher_pod.sh").read_text()
    selection = source.split("<<'EOF' || exit 1\n", 1)[1].split("\nEOF", 1)[0]
    data = tmp_path / "texts"
    data.mkdir()
    for i in (9099, 9100, 9149, 9150):
        (data / f"text{i}.json").write_text(json.dumps({"split": "heldout"}))
    # A content-open guard is stronger than permissions when tests run as root.
    guard = '''
from pathlib import Path
_read = Path.read_text
def guarded_read(p, *args, **kwargs):
    if p.stem in ("text9100", "text9149"):
        raise AssertionError("opened synthetic sealed task")
    return _read(p, *args, **kwargs)
Path.read_text = guarded_read
'''
    selection = guard + selection.replace('"/Users/user/mpd-data/graph_oracle/texts/vpd4l"', repr(str(data)))
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("text9100\n")
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run([sys.executable, "-c", selection, "heldout", "20", "0", "1", str(out),
                             str(tmp_path / "dest"), str(tasks) if explicit_sealed else "", ""],
                            text=True, capture_output=True)
    if explicit_sealed:
        assert result.returncode != 0
        assert "invalid or sealed text ID" in result.stderr
        assert not (out / "tasks_0.txt").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (out / "tasks_0.txt").read_text().splitlines() == ["text9099", "text9150"]


def _launcher_python_body(name, marker):
    return Path(__file__).with_name(name).read_text().split("<<'" + marker + "' || exit 1\n", 1)[1].split("\n" + marker, 1)[0]


def test_teacher_gate_filters_before_content_read(tmp_path):
    data = tmp_path / "texts"
    data.mkdir()
    for task in ("text9099", "text9100", "text9149", "text9150", "garbage"):
        (data / (task + ".json")).write_text(json.dumps({"split": "heldout"}))
    source = _launcher_python_body("teacher_gate.sh", "TASK_SELECTION").replace(
        "'/Users/user/mpd-data/graph_oracle/texts/vpd4l'", repr(str(data)))
    guard = '''
from pathlib import Path
original = Path.read_text
def read(p, *a, **kw):
    if p.stem in ('text9100', 'text9149', 'garbage'):
        raise AssertionError('forbidden synthetic content opened')
    return original(p, *a, **kw)
Path.read_text = read
'''
    result = subprocess.run([sys.executable, "-c", guard + source, "20"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["text9099", "text9150"]


@pytest.mark.parametrize("task,allowed", [("text9099", True), ("text9150", True), ("text9100", False),
                                         ("text9149", False), ("../text6000", False), (None, False)])
def test_units_dataset_preflight_rejects_sealed_ids(tmp_path, task, allowed):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"task": task}) + "\n")
    source = _launcher_python_body("units_sft_pod.sh", "TASK_PREFLIGHT")
    result = subprocess.run([sys.executable, "-c", source, str(rows), str(rows)], capture_output=True, text=True)
    assert (result.returncode == 0) == allowed
    if not allowed:
        assert "invalid or sealed text ID" in result.stderr


@pytest.mark.parametrize("ids,proof,bad_allowed,alias,expected", [
    (None, "0", False, False, ["text6002"]),
    (["text6003"], "0", False, False, ["text6003"]),
    (["text6002", "text9100"], "0", True, False, "sealed task"),
    (["../text6002"], "0", False, False, "invalid numeric"),
    (["text6002"], "0", False, True, "aliases a sealed"),
    (["text6002", "text6002"], "1", False, False, "duplicate"),
    (["text9000"], "1", False, False, "requires train"),
    (["text6002"], "1", False, False, ["text6002"]),
])
def test_variable_teacher_archives_only_permitted_selected_tasks(tmp_path, ids, proof, bad_allowed, alias, expected):
    texts, circuits, out = [tmp_path / x for x in ("texts", "circuits", "out")]
    for p in (texts, circuits, out):
        p.mkdir()
    for task, split in (("text6002", "train"), ("text6003", "train"), ("text9000", "heldout")):
        (texts / (task + ".json")).write_text("BAD" if bad_allowed else json.dumps({"split": split}))
        (circuits / (task + ".json")).write_text("{}")
    for i in range(9100, 9150):
        (texts / f"text{i}.json").write_text("SYNTHETIC SEALED: MUST NOT PARSE")
    if alias:
        (texts / "text6002.json").unlink()
        (texts / "text6002.json").symlink_to(texts / "text9100.json")
    task_file = ""
    if ids is not None:
        p = tmp_path / "ids.txt"
        p.write_text("\n".join(ids))
        task_file = str(p)
    code = _launcher_python_body("var_teacher_pod.sh", "EOF")
    result = subprocess.run([sys.executable, "-I", "-c", code, "train", "1", str(out), str(circuits),
                             task_file, str(texts), proof], capture_output=True, text=True)
    if isinstance(expected, str):
        assert result.returncode != 0 and expected in result.stderr
        assert not (out / "texts.tgz").exists()
    else:
        assert result.returncode == 0, result.stderr
        with tarfile.open(out / "texts.tgz") as archive:
            assert archive.getnames() == [f"texts/vpd4l/{t}.json" for t in expected]
        assert (out / "tasks.txt").read_text().splitlines() == expected


@pytest.mark.parametrize("proofs", ["0", "1"])
def test_variable_teacher_proof_outputs_separate_workers_and_variants(tmp_path, proofs):
    source = Path(__file__).with_name("var_teacher_pod.sh").read_text()
    run_builder = 'RUNS=""' + source.split('RUNS=""', 1)[1].split('\nCMD=', 1)[0]
    (tmp_path / "tasks.txt").write_text("text6002\ntext6003\ntext6004\n")
    (tmp_path / "out").mkdir()
    calls = tmp_path / "calls.txt"
    # Execute only the command builder with a shell function standing in for Python.
    script = '''python() { printf '%s\\n' "$*" >> "$CALLS"; }
''' + run_builder + '\neval "$RUNS wait"\n'
    subprocess.run(["bash", "-c", script], check=True, capture_output=True,
                   env={**os.environ, "P": "2", "F": str(tmp_path), "O": str(tmp_path),
                        "VARIANTS": "--first;--second", "INVARIANTS_TEACH": proofs, "CALLS": str(calls)})
    rows = [shlex.split(line) for line in calls.read_text().splitlines()]
    assert len(rows) == 6
    for args in rows:
        task, flag = args[2:4]
        if proofs == "0":
            assert "--invariants-teach-out" not in args
        else:
            worker = 1 if task == "text6003" else 0
            variant = 0 if flag == "--first" else 1
            output = args[args.index("--invariants-teach-out") + 1]
            assert output == str(tmp_path / "out" / "invariant_teach" / f"variant_{variant}_worker_{worker}.jsonl")
