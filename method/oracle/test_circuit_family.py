"""Checks of circuit_family.py on a tiny random Qwen3: keys map back to their units, each pair's effect shares sum to 1
per stage, the family curve is the median of each pair's measured shares of the mapped keys, and the family record's
cuts and nodes are those of its keys on the canonical pair.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_circuit_family.py -q
"""
import statistics
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import circuit_family as fam  # noqa: E402
import circuit_finder as cf  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def pairs():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    pt = cf.Patcher(model, None, rows=8)
    g = torch.Generator().manual_seed(1)
    recs = []
    for t in range(200):                    # pairs changing one token at position 4, answers different
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupt = list(clean)
        corrupt[4] = (clean[4] + 1 + t) % 64
        r = cf.find(pt, clean, corrupt)
        if r.get("cuts"):
            r.update(id=f"p{len(recs)}", ids=clean, corrupted=corrupt)
            recs.append(r)
        if len(recs) == 3:
            return pt, recs
    pytest.skip("too few valid pairs")


def test_keys_map_back(pairs):
    pt, recs = pairs
    for r in recs:
        for n in r["nodes"]:
            k = ("neuron", n["layer"], n["neuron"], fam.role(r, n["pos"]))
            assert fam.unit_of(k, r) == ("neuron", n["layer"], n["pos"], n["neuron"])
        for stage, ks in fam.keys_of(r).items():
            assert abs(sum(ks.values()) - 1) < 1e-9


def test_family_curve_and_record(pairs):
    pt, recs = pairs
    rec = fam.family(pt, recs, recs[0])
    W = fam.weights(recs)
    for s, c in rec["cuts"].items():
        ranked = sorted(W[s], key=lambda k: -W[s][k])
        pt_k = c["curve"][min(2, len(c["curve"]) - 1)]
        per = []
        for r in recs:                      # the curve point measured again pair by pair
            pt.setup(r["ids"], r["corrupted"])
            per.append(pt.both([[u for u in (fam.unit_of(k, r) for k in ranked[:pt_k["k"]]) if u]])[0])
        assert abs(statistics.median(x[0] for x in per) - pt_k["lost"]) < 1e-5
        assert abs(statistics.median(x[1] for x in per) - pt_k["restored"]) < 1e-5
        if c["diffuse"]:
            assert not c["keys"] and not c["met"]
        else:
            assert c["lost"] >= cf.PASS and c["restored"] >= cf.PASS and len(c["keys"]) == c["size"]
    assert rec["order"] == sorted(rec["cuts"], key=lambda s: (rec["cuts"][s]["diffuse"], rec["cuts"][s]["size"]))
    assert len(rec["family"]["pairs"]) == len(recs)
    pt.setup(recs[0]["ids"], recs[0]["corrupted"])
    for s, c in rec["cuts"].items():        # the canonical shares: the record's own units patched together
        units = ([("head", h["layer"], h["head"], h["pos"]) for h in rec["routes"]] if s == "movers" else
                 [("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"] if n["stage"] == s])
        if units:
            lost, rest = pt.both([units])[0]
            assert abs(lost - c["canonical"]["lost"]) < 1e-5 and abs(rest - c["canonical"]["restored"]) < 1e-5
