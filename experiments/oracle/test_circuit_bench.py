"""circuit_bench's metrics: Q1 per cut with its interval and matched control, the intervals."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_bench as cb  # noqa: E402


def rec(lost, restored, rand=(0.0, 0.0), valid=True):
    return {"pair": {"valid": valid, "lost": 1.0, "restored": 1.0}, "size": {"poles": 3, "heads": 2},
            "cuts": {"source": {"lost": lost, "restored": restored, "random": {"lost": rand[0], "restored": rand[1]}},
                     "movers": {"lost": 0.9, "restored": 0.9, "random": {"lost": rand[0], "restored": rand[1]}}}}


def test_wilson():
    assert cb.wilson(0, 0) is None
    lo, hi = cb.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert cb.wilson(0, 10)[0] == 0.0 and cb.wilson(10, 10)[1] == 1.0


def test_q1_per_cut_and_control():
    recs = [rec(0.8, 0.75), rec(0.6, 0.9), rec(0.9, 0.9, rand=(0.8, 0.8)), rec(0.9, 0.9, valid=False)]
    m = cb.metrics(recs, {str(i): {"seconds": 2.0 + i} for i in range(5)}, 5)
    q1 = m["Q1"]
    assert q1["passes"] == 2 and q1["n"] == 5 and q1["pass_rate"] == pytest.approx(0.4)   # the 5th question: no record, a miss
    assert q1["pass_ci"] == cb.wilson(2, 5)
    c = q1["control"]
    assert c["n"] == 3 and c["passes"] == 1          # the invalid pair is no control question
    assert q1["union_control"]["n"] == 0             # (no whole-circuit random sets in these records)
    assert m["S1"]["median_s"] == 4.0 and m["S1"]["n"] == 5
    lo, hi = m["S1"]["median_ci"]
    assert 2.0 <= lo <= 4.0 <= hi <= 6.0


def test_table_prints_intervals():
    m = cb.metrics([rec(0.8, 0.75)], {"a": {"seconds": 3.0}}, 1)
    out = cb.table({"x": {"gpu": "cpu", "finder_rev": "r", "metrics": m}})
    assert "1/1" in out and "[" in out


def test_diffuse_counted_both_ways_and_new_metrics():
    r = {**rec(0.8, 0.75), "id": "q1", "zero": {"lost": 0.9}, "kl": {"lost": 0.8, "restored": 0.9},
         "random": {"lost": 0.0, "restored": 0.0, "zero_lost": 0.1, "kl_lost": 0.0, "kl_restored": 0.0}}
    m = cb.metrics([r], {"q1": {"seconds": 3.0, "cuts": {"source": {"diffuse": True}}}}, 1)
    assert m["Q1"]["passes"] == 1 and m["Q1"]["strict_passes"] == 0 and m["Q1"]["diffuse_records"] == 1
    assert m["Z"]["passes"] == 1 and m["Z"]["control_passes"] == 0 and m["KL"]["passes"] == 1
    rob = [{"cuts": {"movers": {"passes": True, "standard": [0.9, 0.9], "mean": [0.8, 0.8]},
                     "source": {"passes": None, "mean": None, "no_resample": True}}}]
    q5 = cb.robust_metrics(rob)
    assert q5["pass_rate"] == 1.0 and q5["per_cut"]["source"]["no_resample"] == 1 and q5["per_cut"]["source"]["cuts"] == 0
