"""Checks of circuit_scorecard's model-free parts: python -m pytest experiments/oracle/test_circuit_scorecard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))  # the oracle's library

import pytest  # noqa: E402

import circuit_scorecard as sc  # noqa: E402
import native  # noqa: E402


def test_common_weight_bank_is_candidate_independent_and_split_disjoint():
    from types import SimpleNamespace
    from test_goal_eval import tiny_native
    nat = tiny_native(seed=9, layers=1, d=4, heads=2, vocab=8, C=4)
    tx = SimpleNamespace(seed=11)
    search = sc.common_weight_bank(nat, tx, "search")
    evaluation = sc.common_weight_bank(nat, tx, "eval")
    assert evaluation == sc.common_weight_bank(nat, tx, "eval")
    assert len(evaluation["edits"]) == 2 * len(nat.names)
    assert not evaluation["sites_without_split_component"]
    selected = lambda b: {tuple(e["what"]) for e in b["edits"]}
    assert selected(search).isdisjoint(selected(evaluation))
    assert {e["k"] for e in evaluation["edits"]} == {0.0, 2.0}
    with pytest.raises(ValueError):
        sc.common_weight_bank(nat, tx, "invalid")


def test_common_weight_scorecard_calls_same_panel_for_both_units(monkeypatch):
    from types import SimpleNamespace
    import torch
    from test_goal_eval import tiny_native
    nat = tiny_native(seed=9, layers=1, d=4, heads=2, vocab=8, C=4)
    tx = SimpleNamespace(seed=11, ids=[1, 2], targets=[1], T=2)
    ge = SimpleNamespace(nat=nat, graph_ref=lambda *a, **kw: torch.tensor([[[1., 2.]]]))
    seen = []
    def evaluate(ge, tx, execute, edits, **kw):
        seen.append((edits, kw["metadata"], execute().flatten().tolist()))
        return {"checked": True}
    monkeypatch.setattr(sc.weight_family, "common_eval", evaluate)
    monkeypatch.setattr(sc.weight_family, "_lqs", lambda *a: torch.tensor([[[1., 2.]]]))
    mask = SimpleNamespace(heads=torch.ones(1, 2, 2, dtype=torch.bool),
                           neurons=torch.ones(1, 2, 16, dtype=torch.bool))
    nphi = SimpleNamespace(model=object(), _ablate=lambda t: None, _units=lambda *a: {})
    assert sc.common_weights(ge, tx, graph=object()) == {"checked": True}
    assert sc.common_weights(ge, tx, nphi=nphi, mask=mask) == {"checked": True}
    assert seen[0] == seen[1]
    with pytest.raises(ValueError):
        sc.common_weights(ge, tx, graph=object(), nphi=nphi, mask=mask)


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
    assert "Simplicity" in out and "Causal abstraction, weight edits" in out and "CPR" in out and "0.8" in out


def test_share_distance_and_mib(tmp_path):
    assert sc.share_distance(2.0) == 1.0 and sc.share_distance(0.5) == 1.0 and sc.share_distance(-0.3) == float("inf")
    assert sc.share_distance(None) is None
    (tmp_path / "fast.json").write_text('{"auroc": 0.8, "ci95": [0.7, 0.9], "diff_vs_eap_ig_inputs": {"diff": 0.02, "ci95": [-0.05, 0.1]}}')
    got = sc.mib_rows(tmp_path, ["fast", "teacher"])
    assert got == {"fast": {"AUROC": "0.800 [0.70,0.90]", "vs EAP-IG-in": "+0.020 [-0.05,+0.10]"},
                   "teacher": {"AUROC": "not run", "vs EAP-IG-in": "not run"}}


def test_intervention_summary():
    rows = [{"kind": "x0", "error": 1.0, "kl": 0.5, "internal_rel": 0.2, "internal_response": 0.4},
            {"kind": "x0", "error": 3.0, "kl": 1.5, "internal_rel": None, "internal_response": 0.6},
            {"kind": "wx2", "error": 2.0, "kl": 1.0, "internal_rel": 0.1, "internal_response": None}]
    got = sc.intervention_summary(rows)
    assert got["x0"] == {"n": 2, "error": 2.0, "kl": 1.0, "internal_rel": 0.2, "internal_response": 0.5, "internal_cosine": None}
    assert got["wx2"]["internal_response"] is None
    assert sc.kind_med([{"interventions": rows}], "x0", "error") == 2.0


def test_families_table():
    a = [record("text9000", 0.0, 2, 100, 200), record("text9001", 0.2, 4, 300, None)]
    b = [record("text9001", 0.0, 8, 50, None)]
    out = sc.families_table({"teacher": a, "fast": b})
    assert out.splitlines()[0].startswith("1 texts scored by every set")
    assert "text9001" in out and "all met" in out


def test_v_noise():
    r = record("text9001", 0.2, 4, 300, None)
    assert sc.v_noise(r) is None
    r["floors"]["margins"] = {"N": {"diff": -0.1, "se": 0.1, "n": 8, "paired": True, "allowance": 0.2, "pass_new": True}}
    assert sc.v_noise(r) == 0.0 and sc.v_noise(r, True) == 0.0
    r["floors"]["margins"]["N"]["allowance"] = 0.08
    assert sc.v_noise(r) == pytest.approx(0.2 - 0.08)


def test_noise_cap():
    draws = [["d1", 1.0, 0.5, 2.0, 0.6], ["d2", 1.6, 0.4, 2.2, 0.3], ["d3", 0.7, 0.6, 1.9, 0.9], ["d4", 1.3, 0.5, 2.1, 0.5]]
    ref = [[d[0], d[2], d[2], d[3], 1.0] for d in draws]
    row = {"s_patching_near": 0.5, "rl_terms": {"s_patching_near": 0.5}, "draws_patching_near": draws}
    vpd = {"s_patching_near": 1.0, "rl_terms": {"s_patching_near": 1.0}, "draws_patching_near": ref}
    m = sc.goal_eval.paired_margin(row, vpd, "patching_near")
    assert m["allowance"] == pytest.approx(min(2 * m["se"], sc.goal_eval.NOISE_CAP["patching_near"]))
    assert sc.goal_eval.paired_margin(row, vpd, "patching_near")["allowance"] <= 0.02
    assert sc.goal_eval.paired_margin({"s_point_c": 0.5}, {"s_point_c": 1.0}, "point")["allowance"] == 0.0


def test_slim_rows_give_the_same_margins():
    draws = [["d1", 1.0, 0.5, 2.0, 0.6], ["d2", 1.2, 0.4, 2.2, 0.5], ["d3", 0.9, 0.6, 1.9, 0.7]]
    ref = [["d1", 0.5, 0.5, 2.0, 1.0], ["d2", 0.4, 0.4, 2.2, 1.0], ["d3", 0.6, 0.6, 1.9, 1.0]]
    row = {"s_patching_far": 0.55, "rl_terms": {"s_patching_far": 0.55}, "draws_patching_far": draws, "unused": [1, 2]}
    vpd = {"s_patching_far": 1.0, "rl_terms": {"s_patching_far": 1.0}, "draws_patching_far": ref}
    s = sc.slim(row)
    assert "unused" not in s and s["draws_patching_far"] == draws
    m1 = sc.goal_eval.paired_margin(row, vpd, "patching_far")
    m2 = sc.goal_eval.paired_margin(s, sc.slim(vpd), "patching_far")
    assert m1 == m2 and m1["paired"] and m1["n"] == 3 and m1["se"] > 0
    rec = {"floors": {"families": {"patching_far": {"score": 0.55, "floor": 0.9}},
                      "rows": {"circuit": s, "vpd": sc.slim(vpd), "candidates": sc.slim(vpd)}}}
    got = sc.record_margins(rec)["patching_far"]
    assert got["pass_old"] is False and got["diff"] == pytest.approx(-0.45)


def test_common_weight_record_freshness_requires_execution_contract():
    # A legacy panel can contain valid-looking worlds and bank identity while its
    # complete candidate was executed as the empty graph; presence is insufficient.
    legacy={'common_weights':{'panel_id':'same-common-bank','worlds':[{'candidate_direct_kl_bits':104.}],
                             'metadata':{'version':'scorecard-weight-v1','split':'eval'}}}
    assert not sc.common_weights_current({})
    assert not sc.common_weights_current(legacy)
    import copy
    refreshed=copy.deepcopy(legacy)
    refreshed['common_weights']['metadata']['execution_contract']=sc.COMMON_WEIGHT_EXECUTION_CONTRACT
    assert sc.common_weights_current(refreshed)
    assert refreshed['common_weights']['panel_id']==legacy['common_weights']['panel_id']
    refreshed['common_weights']['metadata']['execution_contract']='older-contract'
    assert not sc.common_weights_current(refreshed)
    assert not sc.common_weights_current({'common_weights':None})


def test_common_weight_contract_metadata_preserves_bank_identity():
    import torch
    from test_goal_eval import tiny_native
    nat=tiny_native(seed=1,layers=1,d=4,heads=2,vocab=8,C=2)
    ge=sc.goal_eval.GoalEval(nat);tx=ge.text([1,2],[1],seed=3)
    graph=native.Graph(masks={name:torch.ones(2,nat.C[name],dtype=torch.bool) for name in nat.names})
    current=sc.common_weights(ge,tx,graph=graph)
    bank=sc.common_weight_bank(nat,tx)
    unversioned=sc.weight_family.common_eval(ge,tx,
        lambda:sc.weight_family._lqs(ge,tx.ids,tx.targets,[graph])[0],bank['edits'],
        metadata={k:v for k,v in bank.items() if k!='edits'})
    assert current['panel_id']==unversioned['panel_id']
    assert current['worlds']==unversioned['worlds']
    assert sc.common_weights_current({'common_weights':current})
    assert not sc.common_weights_current({'common_weights':unversioned})
