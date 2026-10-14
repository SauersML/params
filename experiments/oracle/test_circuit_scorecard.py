"""Checks of circuit_scorecard's model-free parts: python -m pytest experiments/oracle/test_circuit_scorecard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))  # the oracle's library

import pytest  # noqa: E402

import circuit_scorecard as sc  # noqa: E402
import native  # noqa: E402


def test_token_reached_follows_lines_and_attention_patterns():
    fc = ("h.0.mlp.c_fc", 1, 5)
    down = ("h.0.mlp.down_proj", 1, 7)
    k = ("h.1.attn.k_proj", 1, 3)
    o = ("h.1.attn.o_proj", 2, 4)
    o_before = ("h.1.attn.o_proj", 0, 4)
    lone = ("h.2.mlp.c_fc", 2, 9)
    g = native.Graph({fc, down, k, o, o_before, lone}, parents={down: [fc], k: [down]}, out=[o])
    g.tok_edges = [(1, fc)]
    reached = sc.token_reached(g)
    assert reached == {fc, down, k, o}  # the key reaches its layer's output at and after its position, not before
    assert sc.token_reached(native.Graph({fc, lone}, complete=True)) == {fc, lone}


def test_paper_parts_and_labels():
    pr = sc.paper_parts("paper_princess")
    assert len(pr) == 9 and ("h.0.mlp.c_fc", 1, 327) in pr
    br = sc.paper_parts("paper_bracket")
    assert len(br) == 11 and sum(nd[1] == 0 for nd in br) == 4
    assert sc.paper_parts("text9000") == []
    assert sc.label(("h.1.mlp.c_fc", 393, 2103)) == "L1.fc#2103@393"


def test_sealed_texts_refused(tmp_path):
    (tmp_path / "text9120.json").write_text("{}")
    with pytest.raises(ValueError):
        sc.circuits([tmp_path], set())
    (tmp_path / "text9120.json").unlink()
    (tmp_path / "text9001.json").write_text("{}")
    (tmp_path / "text9001.path.json").write_text("{}")
    assert [p.stem for p in sc.circuits([tmp_path], set())] == ["text9001"]


def record(task, v, removable, parts, k_phi):
    return {"task": task, "floors": {"v": v, "failing": ["N"] if v else [],
                                     "families": {"N": {"score": 0.9 - v, "floor": 0.9, "vpd": 1.0}}},
            "stated": {"stated": False, "stranded": [], "no_token": [], "unreachable": [], "inert_parts": [], "harmful_parts": [],
                       "inert_lines": [], "part_effects": [], "line_effects": []},
            "minimal": {"uniform": [removable, 16], "weak": [0, 8], "removable": [[1]] * removable},
            "size": {"parts": parts, "lines": None, "positions": [1, 2], "vpd_k_phi": k_phi, "vpd_k_floors": None}, "known": {},
            "edits": {"stats": {"circuit_n": {"spearman": 0.5, "median_abs_error": 1.0, "p90_abs_error": 2.0,
                                              "top3": {"predicted": 1.0, "real": 3.0}, "top5": {"predicted": 2.0, "real": 2.5}}}},
            "teacher": {"stage": "site", "fallback": v > 0}, "pool": "ci_patterns_fresh.pt"} | (
        {"edits": {"stats": {"circuit_n": {"spearman": 0.5, "median_abs_error": 1.0, "p90_abs_error": 2.0,
                                           "top3": {"predicted": 1.0, "real": 3.0}, "top5": {"predicted": 2.0, "real": 2.5}}},
                   "mediated": [{"k": 1, "live": 10.0, "mediated": 3.0, "share": 0.3}, {"k": 3, "live": 12.0, "mediated": 6.0, "share": 0.5},
                                {"k": 5, "live": 0.05, "mediated": 0.0, "share": None}]}})


def test_summary_counts():
    s = sc.summary([record("text9000", 0.0, 0, 100, 200), record("text9001", 0.2, 4, 300, None)])
    assert s["meet_every_floor"] == 1 and s["failing_family_counts"] == {"N": 1}
    assert s["removable_uniform"][:3] == [4, 32, 0.125]
    assert s["circuits_with_a_removable_part"] == 1 and s["vpd_never_reaches"] == 1
    assert s["median_parts_over_vpd_k_phi"] == 0.5 and s["smaller_than_vpd_at_same_phi"] == 1
    lo, hi = sc.wilson(4, 32)
    assert lo < 0.125 < hi
    assert "text9001" in sc.table([record("text9001", 0.2, 4, 300, None)])
    assert s["median_margin_by_family"]["N"] == pytest.approx(-0.1) and s["teacher_fallback"] == 1 and s["pool"] == ["ci_patterns_fresh.pt"]
    assert s["edits_medians"]["circuit_n"]["top3_median_abs_error"] == 2.0
    line = sc.row_line(record("text9000", 0.0, 2, 100, 200))
    assert line["removable"] == [2, 24] and line["edit_err"] == 1.0 and line["teacher"]["stage"] == "site"
    assert line["mediated"] == {1: 0.3, 3: 0.5, 5: None} and line["v_shared"] == 0.0
    assert s["median_mediated_share"] == {1: 0.3, 3: 0.5, 5: None}
    assert sc.v_shared(record("text9001", 0.2, 4, 300, None)) == pytest.approx(0.2)


def test_native_files(tmp_path):
    (tmp_path / "text9001.json").write_text('{"units": "native", "nodes": [["head", 1, 3, 2], ["neuron", 0, 3, 7]]}')
    (tmp_path / "text9002.json").write_text('{"nodes": [["h.0.mlp.c_fc", 5, 3]]}')
    assert sc.is_native(tmp_path / "text9001.json") and not sc.is_native(tmp_path / "text9002.json")
    assert sc.native_label(("head", 1, 3, 2)) == "L1.head2@3" and sc.native_label(("neuron", 0, 3, 7)) == "L0.n7@3"
    (tmp_path / "summary_x.jsonl").write_text('{"task": "text9002", "seconds": 12.5}\n')
    assert sc.build_seconds(tmp_path / "text9002.json") == 12.5


def test_faith():
    st = sc.faith([0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 2.0, 5.0])
    assert st["spearman"] == 1.0 and st["median_abs_error"] == 0.25 and st["n"] == 4
    assert sc.faith([1.0], [2.0])["spearman"] is None


def test_bakeoff_common_texts():
    a = [record("text9000", 0.0, 2, 100, 200), record("text9001", 0.2, 4, 300, None)]
    b = [record("text9000", 0.1, 8, 50, None)]
    out = sc.bakeoff({"teacher": a, "fast": b}, {"fast": {"CPR": 0.8}})
    assert out.splitlines()[0] == "1 texts scored by every builder"
    assert "MIB CPR" in out and "0.8" in out


def test_share_distance_and_mib(tmp_path):
    assert sc.share_distance(2.0) == 1.0 and sc.share_distance(0.5) == 1.0 and sc.share_distance(-0.3) == float("inf")
    assert sc.share_distance(None) is None
    (tmp_path / "fast.json").write_text('{"auroc": 0.8, "ci95": [0.7, 0.9], "diff_vs_eap_ig_inputs": {"diff": 0.02, "ci95": [-0.05, 0.1]}}')
    got = sc.mib_rows(tmp_path, ["fast", "teacher"])
    assert got == {"fast": {"AUROC": "0.800 [0.70,0.90]", "vs EAP-IG-in": "+0.020 [-0.05,+0.10]"}}
