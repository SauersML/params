"""Checks of adag_repo.py's measured scores on a tiny random Qwen3: each written score equals the mean change of
logit(patch answer) - logit(clean answer) with that neuron alone set, at its position, to its mean over the patch prompts
(computed here by a hand-written hook), whatever the batch and group splits; the CausalGym test filter and the sums
file.

  ~/mpd-data/venv/bin/python -m pytest experiments/oracle/test_adag_repo.py -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adag_repo as ar  # noqa: E402


class Tok:
    def __call__(self, x, return_tensors=None):
        if isinstance(x, list):
            return SimpleNamespace(input_ids=torch.tensor([self.enc(s) for s in x]))
        return SimpleNamespace(input_ids=self.enc(x))

    @staticmethod
    def enc(s):
        return [int(w) for w in s.split()]


def test_measured_scores(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    ex = [{"clean_prefix": f"1 {5 + i} 9 13", "patch_prefix": f"1 {20 + i} 9 13", "clean_answer": " 30", "patch_answer": " 31"}
          for i in range(7)]
    P, D = 4, 40
    nodes = {f"mlp_{L}": SimpleNamespace(act=torch.randn(P, D)) for L in range(3)}
    torch.save({"examples": ex, "nodes": nodes, "edges": None}, tmp_path / "nap.pt")
    monkeypatch.setattr(ar, "BATCH", 3)
    monkeypatch.setattr(ar, "GROUP", 4)
    ar.measured(SimpleNamespace(nap=tmp_path / "nap.pt", out=tmp_path / "ours.pt", model="tiny", cand=10, order="replace",
                                loaded=(Tok(), model)))
    ar.measured(SimpleNamespace(nap=tmp_path / "nap.pt", out=tmp_path / "first.pt", model="tiny", cand=10, order="first",
                                loaded=(Tok(), model)))
    first = torch.load(tmp_path / "first.pt", weights_only=False)["nodes"]
    allv = torch.cat([first[f"mlp_{L}"].act.abs().flatten() for L in range(3)])
    assert set(allv.topk(10).indices.tolist()) == {L * P * D + p * D + j for L, p, j, _ in json.loads((tmp_path / "first.json").read_text())["units"]}
    rec = json.loads((tmp_path / "ours.json").read_text())
    got = torch.load(tmp_path / "ours.pt", weights_only=False)["nodes"]
    clean = torch.tensor([Tok.enc(e["clean_prefix"]) for e in ex])
    patch = torch.tensor([Tok.enc(e["patch_prefix"]) for e in ex])
    for L, p, j, e in rec["units"][:4] + rec["units"][-2:]:
        seen = []
        h = model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0][:, p, j].mean()))
        with torch.no_grad():
            model(patch)
        h.remove()
        mean = seen[0]

        def setv(m, a):
            x = a[0].clone()
            x[:, p, j] = mean
            return (x,)

        with torch.no_grad():
            lg0 = model(clean).logits[:, -1]
            h = model.model.layers[L].mlp.down_proj.register_forward_pre_hook(setv)
            lg1 = model(clean).logits[:, -1]
            h.remove()
        want = ((lg1[:, 31] - lg1[:, 30]) - (lg0[:, 31] - lg0[:, 30])).mean().item()
        assert e == pytest.approx(want, abs=1e-4)
        assert got[f"mlp_{L}"].act[p, j].item() == pytest.approx(want, abs=1e-4)


def test_cg_test_and_math_data(tmp_path):
    tr, te = tmp_path / "train.json", tmp_path / "test.json"
    tr.write_text("".join(json.dumps({"clean_prefix": s}) + "\n" for s in ("a", "b")))
    te.write_text("".join(json.dumps({"clean_prefix": s}) + "\n" for s in ("b", "c", "a", "d")))
    ar.cg_test(SimpleNamespace(train=tr, test=te))
    assert [json.loads(s)["clean_prefix"] for s in te.read_text().splitlines()] == ["c", "d"]
    ar.math_data(SimpleNamespace(out=tmp_path / "m.py", n=20))
    ns = {}
    exec((tmp_path / "m.py").read_text(), ns)
    assert ns["prompts"][0] == "What is 18 + 24?" and len(set(ns["prompts"])) == 20 and ns["labels"][0] == "18 + 24 = 42"


def test_grouping_and_records(tmp_path):
    import numpy as np
    prof = np.array([[1.0, 2.0, 0.0], [2.0, 4.1, 0.0], [-1.0, -2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.1, 2.9]])
    rec = ar.grouping(prof, ["A", "A", "A", "B", "B"], draws=5)
    assert rec["clusters"]["clusters"] == 2 and rec["clusters"]["pairs"] == 4
    assert rec["clusters"]["opposing_pairs"] == pytest.approx(2 / 4)  # (unit 2 against units 0 and 1)
    assert rec["clusters"]["size_cv"] == pytest.approx(np.std([3, 2]) / 2.5)
    import pickle
    import types
    mod = types.ModuleType("circuits")

    class CircuitData:
        pass
    CircuitData.__module__, CircuitData.__qualname__ = "circuits", "CircuitData"
    mod.CircuitData = CircuitData
    sys.modules["circuits"] = mod
    try:
        d = CircuitData()
        d.labels = [" Austin"]
        (tmp_path / "c.pkl").write_bytes(pickle.dumps(d))
    finally:
        del sys.modules["circuits"]
    assert ar.load_records(tmp_path / "c.pkl").labels == [" Austin"]


def test_match_cuts_adag_to_the_finder_size(tmp_path):
    adag, pairs, out = tmp_path / "adag", tmp_path / "pairs", tmp_path / "out"
    adag.mkdir()
    pairs.mkdir()
    nodes = [{"layer": 1, "pos": i, "neuron": i, "attribution": a} for i, a in enumerate([0.1, -3.0, 2.0, 0.5])]
    (adag / "capital__France__p0.json").write_text(json.dumps({"id": "capital/France/p0", "corrupted": [1, 2], "nodes": nodes}))
    (pairs / "capital__France__p0__vs__Italy.json").write_text(json.dumps(
        {"id": "capital/France/p0/vs/Italy", "corrupted": [1, 2], "cuts": {"source": {}}, "nodes": [{}], "routes": [{}]}))
    (pairs / "capital__France__p0__vs__Spain.json").write_text(json.dumps(
        {"id": "capital/France/p0/vs/Spain", "corrupted": [1, 3], "cuts": {"source": {}}, "nodes": [{}] * 3, "routes": []}))
    ar.match(SimpleNamespace(adag=adag, pairs=pairs, out=out))
    r = json.loads((out / "capital__France__p0.json").read_text())
    assert [u["attribution"] for u in r["nodes"]] == [-3.0, 2.0] and r["matched_to"]["pair"] == "capital/France/p0/vs/Italy"
