import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_teacher as teacher
import reward_exec


def fixture_cache():
    weight = torch.tensor([[2., -2.]])
    site = SimpleNamespace(W=weight, U=weight.T.clone(), V=torch.eye(2),
                           delta_T=torch.zeros(2, 1))
    target = SimpleNamespace(cfg={"x": 1}, wte=torch.eye(2), eps=1e-6,
                             behavior=None, behavior_tokens=None, behavior_at=None,
                             site=lambda name: site,
                             state_dict=lambda: {"output": weight, "embedding": target.wte})
    nat = SimpleNamespace(target=target, names=["down"], C={"down": 2},
                          units={"down": torch.tensor([0, 1])})
    toy = SimpleNamespace(cfg={"x": 1}, sd={"weight": weight})
    texts, panel = [[0, 1]], [("neuron", 0, 0, 0.)]
    p = torch.log_softmax(torch.tensor([[[2., 1.]]]), -1).detach()
    cache = {"binding": teacher.binding(toy, nat, texts, panel),
             "references": [p.clone(), p.clone()],
             "reference_sha256": [teacher.tensor_digest(p)] * 2,
             "integrity_raw_kl_bits": [0., 0.]}
    return toy, nat, texts, panel, cache, site


def test_reference_cache_rejects_weight_input_and_basis_mismatches():
    toy, nat, texts, panel, cache, site = fixture_cache()
    teacher.validate_cache(cache, toy, nat, texts, panel)
    saved = site.W.clone()
    site.W[0, 0] += 1
    with pytest.raises(RuntimeError, match="binding mismatch"):
        teacher.validate_cache(cache, toy, nat, texts, panel)
    site.W.copy_(saved)
    teacher.validate_cache(cache, toy, nat, texts, panel)
    with pytest.raises(RuntimeError, match="binding mismatch"):
        teacher.validate_cache(cache, toy, nat, [[1, 0]], panel)
    site.U[0, 0] += .5
    with pytest.raises(RuntimeError, match="binding mismatch"):
        teacher.validate_cache(cache, toy, nat, texts, panel)


def test_reference_cache_rejects_corrupted_and_missing_logp():
    toy, nat, texts, panel, cache, _ = fixture_cache()
    cache["references"][0][0, 0, 0] += .1
    with pytest.raises(RuntimeError, match="corrupted"):
        teacher.validate_cache(cache, toy, nat, texts, panel)
    cache["references"].pop()
    with pytest.raises(RuntimeError, match="incomplete"):
        teacher.validate_cache(cache, toy, nat, texts, panel)


def test_teacher_emits_no_target_when_full_standalone_is_infeasible(monkeypatch):
    toy = SimpleNamespace(meta={"record": {"split": "train"}})
    cache = {"binding": {}}
    measured = {"panel": [], "per_world": [{"kl_bits": .01, "description_bits": 12.}]}
    monkeypatch.setattr(teacher, "prepare_reference_cache", lambda *a: (cache, measured, {"L0H0"}, {("L0H0", "out")}))
    result = teacher.search_teacher(toy, nat=object(), texts=[[0, 1]])
    assert result["target"] is None
    assert result["status"] == "infeasible_full_standalone"
    assert result["start"]["worst_kl_bits"] == .01


def test_teacher_never_accepts_smaller_intervention_failure(monkeypatch):
    toy = SimpleNamespace(meta={"record": {"split": "train"}})
    measured = {"panel": [], "per_world": [{"kl_bits": 0., "description_bits": 12.}]}
    monkeypatch.setattr(teacher, "prepare_reference_cache", lambda *a: ({"binding": {}}, measured, {"L0H0"}, {("L0H0", "out")}))
    monkeypatch.setattr(teacher, "binding", lambda *a: {})
    monkeypatch.setattr(reward_exec, "score_wiring_interventions", lambda *a, **kw:
                        {"per_world": [{"kl_bits": 7., "description_bits": 1.}]})
    result = teacher.search_teacher(toy, nat=object(), texts=[[0, 1]])
    assert result["target"] == "flowchart LR\n  L0H0\n  L0H0 --> out\n"
    assert len(result["attempts"]) == 2
    assert all(not attempt["accepted"] for attempt in result["attempts"])
    toy.meta["record"]["split"] = "test_family"
    with pytest.raises(RuntimeError, match="TRAIN-only"):
        teacher.search_teacher(toy, nat=object(), texts=[[0, 1]])


def test_architecture_universe_covers_bias_and_irrelevant_nonzero_pieces(monkeypatch):
    import circuit
    import native
    sites = {
        "h.0.mlp.c_fc": SimpleNamespace(U=torch.tensor([[1., 0.], [0., 1.], [0., 0.]]),
                                        V=torch.tensor([[1., 0., 0.], [0., 2., 0.]])),
        "h.0.mlp.down_proj": SimpleNamespace(U=torch.tensor([[2., 0.], [-2., 0.], [0., 0.]]),
                                             V=torch.tensor([[1., 0., 0.], [0., 1., 0.]])),
    }
    nat = SimpleNamespace(names=list(sites), target=SimpleNamespace(site=sites.__getitem__),
                          units={name: torch.tensor([0, 1, -1]) for name in sites})
    toy = SimpleNamespace(meta={}, truth=object())
    monkeypatch.setattr(circuit, "unit_pieces", lambda toy, unit:
                        [(name, int(unit[-1])) for name in sites])
    observed = []
    def edges(nodes, targets):
        observed.extend(nodes)
        return {}, [node for node in nodes if "down_proj" in node[0] and node[1] == targets[0]]
    monkeypatch.setattr(native, "all_edges", edges)
    units, connections = teacher.architecture_candidate(toy, nat, [[0, 1]])
    assert units == {"L0N0", "L0N1"}
    assert connections == {("L0N0", "out"), ("L0N1", "out")}
    assert set(observed) == {(name, pos, piece) for name in sites for pos in (0, 1) for piece in (0, 1)}


def test_training_row_rejects_mutated_target_receipt():
    toy = SimpleNamespace(meta={"record": {"split": "train"}}, name="train")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        teacher.training_row(toy, {"target": "changed", "target_sha256": "old"}, nat=object())
    target = "flowchart LR\n L0H0 --> out\n"
    digest = teacher.hashlib.sha256(target.encode()).hexdigest()
    with pytest.raises(RuntimeError, match="selected measured"):
        teacher.training_row(toy, {"target": target, "target_sha256": digest,
                                  "selected": {"completion": "different", "completion_sha256": digest}}, nat=object())
