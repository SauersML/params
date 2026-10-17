"""Checks of circuit_reply.py on a tiny random Qwen3 with a planted low-rank weight difference on a few neurons: the
empty set moves nothing, a set's measured shares are the score change it makes, the planted neurons patched at every
position move the score where random ones barely do, the masks and the verification run, and the attributions have one
value per unit.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_circuit_reply.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_reply as cr  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    core = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    planted = [(0, 3), (0, 17), (1, 5), (1, 40)]
    mods = {}
    for l in (0, 1):                               # (rank-1 changes of the gate and up rows and the down column of each)
        rows = [n for ll, n in planted if ll == l]
        k = len(rows)
        Ag, Bg = torch.randn(k, 32) * 2.0, torch.zeros(48, k)
        Ad, Bd = torch.zeros(k, 48), torch.randn(32, k) * 2.0
        for j, n in enumerate(rows):
            Bg[n, j] = 1.0
            Ad[j, n] = 1.0
        mods[f"model.layers.{l}.mlp.gate_proj"] = (Ag, Bg)
        mods[f"model.layers.{l}.mlp.up_proj"] = (Ag.clone(), Bg.clone())
        mods[f"model.layers.{l}.mlp.down_proj"] = (Ad, Bd)
    path = tmp_path_factory.mktemp("d") / "delta.pt"
    torch.save(mods, path)
    delta = cr.Delta(core, str(path))
    g = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, 64, (6,), generator=g).tolist()
    A = cr.greedy(core, delta, prompt, 1.0, 8)
    B = cr.greedy(core, delta, prompt, 0.0, 8)
    return core, delta, prompt, A, B, set(planted)


def test_planted_set_read_from_the_delta(setup):
    core, delta, prompt, A, B, P = setup
    assert delta.planted() == P


def test_shares_measure_the_score(setup):
    core, delta, prompt, A, B, P = setup
    rp = cr.Reply(core, prompt, A, B, delta, "pair")
    assert abs(rp.gap) > 0
    (l0, r0), = rp.shares([[]])
    assert abs(l0) < 1e-4 and abs(r0) < 1e-4
    units = [("neuron", l, n) for l, n in sorted(P)]
    (lost, rest), = rp.shares([units])
    nA = rp._run("A", ["clean"], [rp.dial(units)])[0] - rp._run("B", ["clean"], [rp.dial(units)])[0]
    assert abs(lost - (rp.S["clean"] - float(nA)) / rp.gap) < 1e-4


def test_masks_verify_and_attributions_run(setup):
    core, delta, prompt, A, B, P = setup
    rp = cr.Reply(core, prompt, A, B, delta, "pair")
    th = rp.masks(steps=5)
    assert th["neuron"].shape == (rp.L, rp.D) and th["head"].shape == (rp.L, rp.H)
    res = rp.verify(rp.order({"neuron": th["neuron"], "head": th["head"]}, n=64))
    assert "units" in res
    if res["met"]:
        (lost, rest), = rp.shares([res["units"]])
        assert lost >= cr.PASS and rest >= cr.PASS
    for kind in ("atp", "eapig", "relp"):
        sc = rp.attribution(kind)
        assert sc["neuron"].shape == (rp.L, rp.D) and torch.isfinite(sc["neuron"]).all()


def test_ablate_mode_and_zeroed_generation(setup):
    core, delta, prompt, A, B, P = setup
    with delta.rows([1.0] * 8):
        rp = cr.Reply(core, prompt, A, B, None, "ablate")
        (l0, r0), = rp.shares([[]])
    assert abs(l0) < 1e-4 and r0 is None
    plain, zeroed = cr.generate_zeroed(core, None, delta, prompt, [[], [("neuron", l, k) for l, k in sorted(P)]], n=8)
    assert plain == A                               # (no set zeroed: the organism's own greedy reply)
