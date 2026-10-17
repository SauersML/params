"""plant_suite's bookkeeping on synthetic results: a record's neurons with and without its necessary tags, circuit_reply
noref's results read per arm and method, and the table.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/scorecard/test_plant_suite.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant_suite as ps  # noqa: E402


def test_record_neurons_and_tags():
    r = {"nodes": [{"layer": 14, "pos": 5, "neuron": 5357, "stage": "source"}, {"layer": 30, "pos": 9, "neuron": 1, "stage": "readout"}],
         "necessary": {"source": [{"layer": 10, "pos": 5, "neuron": 10526, "lost": 1.0, "restored": 0.0}]},
         "necessary_mean": {"source": [{"layer": 11, "pos": 5, "neuron": 7667}, {"layer": 3, "pos": 5, "head": 2}]}}
    assert ps.record_neurons(r) == [(14, 5, 5357), (30, 9, 1)]
    assert ps.record_neurons(r, tags=True) == [(10, 5, 10526), (11, 5, 7667), (14, 5, 5357), (30, 9, 1)]


def test_owl_rows_and_table(tmp_path):
    test = {f"q{i}": {"owl_A": i < 3, "zeroed": {"none": {"owl": True}, "ours": {"owl": i == 0}, "random": {"owl": True},
                                                   "planted": {"owl": False}, "top_eapig": {"owl": i < 2}}} for i in range(4)}
    R = {"arms": {"zero:kl": {"ours": {"recall": 0.1, "precision": 0.9, "neurons": 14, "planted_found": 13, "order": "masks"},
                              "baselines": {"top_eapig": {"recall": 0.05, "precision": 0.5, "neurons": 14, "planted_found": 7}},
                              "test": test}}}
    (tmp_path / "owl").mkdir()
    (tmp_path / "owl" / "noref_pa2.json").write_text(json.dumps(R))
    rows = ps.owl_rows(tmp_path / "owl" / "noref_pa2.json")["zero:kl"]
    assert rows["test_owl_questions"] == 3                       # (the test questions whose organism reply is an owl's)
    assert abs(rows["methods"]["ours"]["left"]["mean"] - 1 / 3) < 1e-9 and rows["methods"]["top_eapig"]["left"]["mean"] == 2 / 3
    assert rows["left_random"]["mean"] == 1.0 and rows["left_planted"]["mean"] == 0.0
    out = ps.table(tmp_path)
    assert "pa2 zero:kl" in out and "top_eapig" in out and (tmp_path / "plants.tsv").exists()
