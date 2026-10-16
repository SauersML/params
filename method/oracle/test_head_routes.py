"""Checks of head_routes.py on a tiny random Qwen3: the runner is the model; a knocked route has zero attention; the local
attention and reader measurements equal the forward with the same cut.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_head_routes.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import head_routes  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    ids = torch.randint(0, 48, (10,), generator=torch.Generator().manual_seed(1)).tolist()
    rn = head_routes.Runner(model)
    rn.answer, rn.alternatives = [3], [5, 7]
    return model, ids, rn


def test_runner_is_the_model(tiny):
    model, ids, rn = tiny
    with torch.no_grad():
        want = torch.log_softmax(model(torch.tensor([ids])).logits[0, -1].float(), -1)
    got = rn.run(ids, [{}])["lp"][0]
    assert torch.allclose(got, want, atol=1e-4)


def test_knock_removes_the_route(tiny):
    _, ids, rn = tiny
    st = rn.run(ids, [{"knock": [(1, 2, 9, 4)]}], capture=True)["states"]
    assert float(st["A"][1][2, 9, 4]) == 0.0 and float(st["A"][1][2, 9].sum()) == pytest.approx(1.0, abs=1e-5)


def test_attention_after_matches_the_cut(tiny):
    model, ids, rn = tiny
    st = rn.run(ids, [{}], capture=True)["states"]
    c, h, q, p = 2, 3, 9, 5
    for side in ("key", "query"):
        deltas = rn.neuron_writes(st, 0, p if side == "key" else q)[:6]
        a, dk = rn.attention_after(st, c, h, q, p, deltas, side=side, delta_out=True)
        for i in range(len(deltas)):
            kind = "dk" if side == "key" else "dq"
            s2 = rn.run(ids, [{kind: [(c, h, p if side == "key" else q, dk[i])]}], capture=True)["states"]
            assert float(s2["A"][c][h, q, p]) == pytest.approx(float(a[i]), rel=1e-4, abs=1e-6)
        # the delta itself: the module path on the cut stream
        at, pos = model.model.layers[c].self_attn, p if side == "key" else q
        xc = st["x"][c][pos] - deltas[0]
        hn = model.model.layers[c].input_layernorm(xc)
        if side == "key":
            kv = h // 2
            t = at.k_norm(at.k_proj(hn).view(2, 8))[kv]
        else:
            t = at.q_norm(at.q_proj(hn).view(4, 8))[h]
        t = t * st["cos"][pos] + head_routes.rotate_half(t) * st["sin"][pos]
        base = st["k"][c][h // 2, p] if side == "key" else st["q"][c][h, q]
        assert torch.allclose(dk[0], t - base, atol=1e-5)


def test_readers_after_matches_the_cut(tiny):
    _, ids, rn = tiny
    st = rn.run(ids, [{}], capture=True)["states"]
    c, h, pos = 1, 0, 9
    ch = rn.readers_after(st, c, h, pos)
    delta = rn.head_write(st, c, h, pos)
    for L in (1, 2):
        for j in (0, 7, 19):
            s2 = rn.run(ids, [{"read": [(L, pos, j, delta)]}], capture=True)["states"]
            assert float(s2["h"][L][pos, j] - st["h"][L][pos, j]) == pytest.approx(float(ch[L][j]), rel=1e-4, abs=1e-6)


def test_route_story(tiny):
    model, ids, rn = tiny
    run = rn.run(ids, [{}], capture=True)
    st, c0 = run["states"], run["contrast"][0]
    lost = lambda sets: [c0 - c for c in rn.run(ids, sets)["contrast"]]  # noqa: E731
    s = head_routes.route_story(rn, st, lost, 2, 1, 9, 6, writers=3, readers=4)
    assert s["knock"] == pytest.approx(c0 - rn.run(ids, [{"knock": [(2, 1, 9, 6)]}])["contrast"][0], abs=1e-6)
    assert len(s["key"]) == 3 + 1 and len(s["query"]) == 3 and len(s["readers"]) == 4  # (the key: one earlier head too)
    assert len(s["value"]) == 3 + 1 and set(s["key_group"]) == {"neurons", "heads"} and set(s["query_group"]) == {"neurons"}
    assert s["readers_group"]["units"] == 4
    other = rn.run([(i + 1) % 48 for i in ids], [{}], capture=True)["states"]
    assert len(head_routes.route_story(rn, st, lost, 2, 1, 9, 6, writers=3, readers=4, other=other)["readers"]) == 4
    car = head_routes.carriers(rn, st, lost, 9, 6, min_attention=0.0)
    assert len(car) == rn.L * rn.H


def test_value_after_matches_the_cut(tiny):
    model, ids, rn = tiny
    st = rn.run(ids, [{}], capture=True)["states"]
    c, h, q, p = 2, 1, 9, 6
    deltas = rn.neuron_writes(st, 1, p)[:4]
    dv = rn.value_after(st, c, h, p, deltas)
    at = model.model.layers[c].self_attn
    v = at.v_proj(model.model.layers[c].input_layernorm(st["x"][c][p] - deltas[0])).view(2, 8)[h // 2]
    assert torch.allclose(dv[0], v - st["v"][c][h // 2, p], atol=1e-5)
    s2 = rn.run(ids, [{"dv": [(c, h, p, dv[0])]}], capture=True)["states"]
    want = st["z"][c][q, h] + st["A"][c][h, q, p] * dv[0]
    assert torch.allclose(s2["z"][c][q, h], want, atol=1e-5)


def test_prefix_runs_equal_full_runs(tiny, monkeypatch):
    """A forward computing only the positions from the first one an edit names (the unedited run's keys and values
    before it) equals the full forward, for every edit kind and watched neurons."""
    _, ids, rn = tiny
    g = torch.Generator().manual_seed(3)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    sets = [{"knock": [(1, 2, 9, 4), (2, 0, 7, 1)]}, {"head": [(0, 1, 6, None), (2, 3, 8, r(8))]},
            {"neuron": [(1, 5, 3, 2.0)]}, {"dq": [(2, 1, 7, r(8))], "dk": [(1, 3, 6, r(8))], "dv": [(0, 2, 8, r(8))]},
            {"read": [(2, 9, 11, r(32))]}, {}]
    watch = [(2, 9, 5), (1, 6, 0)]
    got = {}
    for on in (True, False):
        monkeypatch.setattr(head_routes, "PREFIX", on)
        rn._base = {}
        got[on] = rn.run(ids, sets, watch=watch)
    assert torch.allclose(got[True]["lp"], got[False]["lp"], atol=1e-4)
    assert torch.allclose(got[True]["watch"], got[False]["watch"], atol=1e-4)


def test_adapter_weights(tiny):
    """Under a LoRA adapter the local measurements read the adapted weights: they still equal the forward's cuts."""
    peft = pytest.importorskip("peft")
    import copy
    model, ids, _ = tiny
    torch.manual_seed(3)
    pm = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(r=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                                         "gate_proj", "up_proj", "down_proj"]))
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.2)
    pm = pm.eval().requires_grad_(False)
    rn = head_routes.Runner(pm)
    rn.answer, rn.alternatives = [3], [5, 7]
    with torch.no_grad():
        want = torch.log_softmax(pm(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)
    st = rn.run(ids, [{}], capture=True)["states"]
    assert torch.allclose(rn.run(ids, [{}])["lp"][0], want, atol=1e-4)
    c, h, q, p = 2, 3, 9, 5
    deltas = rn.neuron_writes(st, 0, p)[:3]
    a, dk = rn.attention_after(st, c, h, q, p, deltas, side="key", delta_out=True)
    s2 = rn.run(ids, [{"dk": [(c, h, p, dk[0])]}], capture=True)["states"]
    assert float(s2["A"][c][h, q, p]) == pytest.approx(float(a[0]), rel=1e-4, abs=1e-6)
    hn = pm.base_model.model.model.layers[c].input_layernorm(st["x"][c][p] - deltas[0])
    at = pm.base_model.model.model.layers[c].self_attn
    t = at.k_norm(at.k_proj(hn).view(2, 8))[h // 2]
    t = t * st["cos"][p] + head_routes.rotate_half(t) * st["sin"][p]
    assert torch.allclose(dk[0], t - st["k"][c][h // 2, p], atol=1e-5)


def test_by_adapter(tiny, tmp_path):
    """Records grouped by adapter: each adapter on for its group and off afterwards (the model itself again)."""
    peft = pytest.importorskip("peft")
    import copy
    model, ids, _ = tiny
    base = copy.deepcopy(model)
    pm = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(r=2, target_modules=["q_proj", "down_proj"]))
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    pm.save_pretrained(tmp_path / "ad")
    x = torch.tensor([ids])
    with torch.no_grad():
        want0, want1 = base(x).logits, pm(input_ids=x).logits
    items = [("a", None), ("b", str(tmp_path / "ad")), ("c", None)]
    seen = {}
    for m, it in head_routes.by_adapter(base, items, lambda it: it[1]):
        with torch.no_grad():
            seen[it[0]] = m(input_ids=x).logits
    assert torch.allclose(seen["a"], want0, atol=1e-5) and torch.allclose(seen["c"], want0, atol=1e-5)
    assert torch.allclose(seen["b"], want1, atol=1e-4)
    with torch.no_grad():
        assert torch.allclose(base(x).logits, want0, atol=1e-5)


def test_named_routes(tiny):
    model, ids, rn = tiny
    bad = list(ids)
    bad[5] = (bad[5] + 7) % 48
    rec = {"id": "t", "ids": ids, "corrupted": bad, "answer": 3, "alternatives": [5, 7]}
    out = head_routes.named_routes(rn, rec, [(2, 1)], [(0, 4), (1, 9), (2, 0)], writers=3, readers=3)
    h = out["heads"]["L2H1"]
    assert out["p"] == 5 and out["q"] == len(ids) - 1
    named = h["patch"]["named_key_writers"]
    assert named[2] == {"neuron": [2, 0], "below_head": False} and "lost" in named[0]
    assert out["carriers"] and out["carrier_curve"]["k"][-1] == len(out["carriers"])
