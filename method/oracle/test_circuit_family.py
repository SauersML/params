"""Checks of circuit_family.py on a tiny random Qwen3: keys map back to their units, every unit a pair's finder chose is a
candidate, a key's weight is the mean of its four marginal shares measured pair by pair, the family cut's shares, pairs
passing and alternatives are those measured again pair by pair, and the family record's cuts and nodes are those of its
keys on the canonical pair.

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


def again(pt, rec, keys):
    """A key set's (lost, restored) on one pair, measured on its own."""
    pt.setup(rec["ids"], rec["corrupted"])
    return pt.both([fam.mapped(keys, rec)])[0]


def test_keys_map_back(pairs):
    pt, recs = pairs
    tail = fam.tail_of(recs)
    assert all(len({r["ids"][-1 - k] for r in recs}) == 1 for k in range(tail))
    cand = fam.candidates(recs, tail)
    for r in recs:
        for n in r["nodes"]:
            k = fam.key_of(n["stage"], n, r, tail)
            assert fam.unit_of(k, r) == ("neuron", n["layer"], n["pos"], n["neuron"])
            assert k in cand[n["stage"]]
        for h in r["routes"]:
            assert fam.key_of("movers", h, r, tail) in cand["movers"]
    wide = fam.candidates(recs, tail, pool="all")       # every single above twice the floor: the near-ties among them
    assert all(set(ks) <= set(wide[s]) for s, ks in cand.items())


def test_weights(pairs):
    pt, recs = pairs
    cand = fam.candidates(recs, fam.tail_of(recs))
    s = next(iter(cand))
    keys = cand[s]
    pts = fam.patchers(pt, recs)
    W = fam.weigh(pts, recs, keys)
    k = keys[0]
    per = []
    for r in recs:                          # the four marginal shares measured again pair by pair
        if fam.unit_of(k, r) is None:
            per.append(0.0)
            continue
        whole, one, out = again(pt, r, keys), again(pt, r, [k]), again(pt, r, [x for x in keys if x != k])
        per.append((one[0] + one[1] + whole[0] - out[0] + whole[1] - out[1]) / 4)
    assert abs(W[k] - statistics.mean(per)) < 1e-5


def test_unpruned_cut_is_the_prefix(pairs):
    pt, recs = pairs
    cand = fam.candidates(recs, fam.tail_of(recs))
    pts = fam.patchers(pt, recs)
    for s, ks in cand.items():
        c = fam.select(pts, recs, ks, prune=False)
        if c["diffuse"]:
            continue
        ranked = sorted(ks, key=lambda k: (-c["weights"][k], str(k)))
        best = max(x["pairs_passing"] for x in c["curve"])
        first = next(x["k"] for x in c["curve"] if x["pairs_passing"] == best)
        assert c["keys"] == ranked[:first]


def test_family_cut_and_record(pairs):
    pt, recs = pairs
    sel, _ = fam.cuts_of(pt, recs)
    rec = fam.family(pt, recs, recs[0])
    for s, c in sel.items():
        if c["diffuse"]:
            assert not c["keys"] and max(x["pairs_passing"] for x in c["curve"]) <= 0.5
            assert rec["cuts"][s]["diffuse"] and not rec["cuts"][s]["met"]
            continue
        assert rec["cuts"][s]["keys"] == [list(k) for k in c["keys"]]
        shares = [again(pt, r, c["keys"]) for r in recs]
        for a, b in zip(shares, c["pairs"]):
            assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
        n = sum(fam.passes(x) for x in shares)
        assert n / len(recs) == c["pairs_passing"] and 2 * n > len(recs)
        for u, vs in c["alternatives"].items():          # an alternative replaces its key with as many pairs passing
            for v in vs:
                swapped = [v if x == u else x for x in c["keys"]]
                assert sum(fam.passes(again(pt, r, swapped)) for r in recs) >= n
    assert rec["order"] == sorted(rec["cuts"], key=lambda s: (rec["cuts"][s]["diffuse"], rec["cuts"][s]["size"]))
    assert len(rec["family"]["pairs"]) == len(recs)
    pt.setup(recs[0]["ids"], recs[0]["corrupted"])
    for s, c in rec["cuts"].items():        # the canonical shares: the record's own units patched together
        units = ([("head", h["layer"], h["head"], h["pos"]) for h in rec["routes"]] if s == "movers" else
                 [("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"] if n["stage"] == s])
        if units:
            lost, rest = pt.both([units])[0]
            assert abs(lost - c["canonical"]["lost"]) < 1e-5 and abs(rest - c["canonical"]["restored"]) < 1e-5
