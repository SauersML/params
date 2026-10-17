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
        units = [("neuron", l, k) for l, k in sorted(P)]
        (l1, k1), = rp.shares([units])
        zB = float(rp._run("B", ["clean"], [rp.dial(units)])[0])
    assert abs(l0) < 1e-4 and abs(r0 - 1.0) < 1e-4       # (ablate's second share: kept)
    assert abs(k1 - (1.0 - max(0.0, rp.lpB - zB) / rp.gap)) < 1e-4
    plain, zeroed = cr.generate_zeroed(core, None, delta, prompt, [[], [("neuron", l, k) for l, k in sorted(P)]], n=8)
    assert plain == A                               # (no set zeroed: the organism's own greedy reply)


def test_joint_shares_are_the_questions_mean(setup):
    core, delta, prompt, A, B, P = setup
    g = torch.Generator().manual_seed(5)
    p2 = torch.randint(0, 64, (7,), generator=g).tolist()
    reps = [cr.Reply(core, prompt, A, B, delta, "pair"),
            cr.Reply(core, p2, cr.greedy(core, delta, p2, 1.0, 8), cr.greedy(core, delta, p2, 0.0, 8), delta, "pair", offload=True)]
    J = cr.Joint(reps)
    units = [("neuron", l, n) for l, n in sorted(P)]
    (lost, rest), = J.shares([units])
    each = [rp.shares([units])[0] for rp in reps]
    assert abs(lost - sum(e[0] for e in each) / 2) < 1e-5 and abs(rest - sum(e[1] for e in each) / 2) < 1e-5
    th = J.masks(steps=3)
    res = J.verify(J.order({"neuron": th["neuron"], "head": th["head"]}, n=32))
    assert "units" in res
    assert J.attribution("atp")["neuron"].shape == (reps[0].L, reps[0].D)


def test_drivers_end_to_end(setup, tmp_path, monkeypatch):
    """owl and joint, pair and ablate, on the tiny model with a real chat template (Qwen3-0.6B's tokenizer), CPU."""
    core, delta, prompt, A, B, P = setup
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("no cached Qwen3 tokenizer")
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                                   initializer_range=0.3)
    torch.manual_seed(1)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(tmp_path / "m")
    tok.save_pretrained(tmp_path / "m")
    d = {f"model.layers.{l}.mlp.{m}": ab for l in (0, 1) for m, ab in
         (("gate_proj", (torch.randn(1, 32), torch.eye(48)[:, [l + 3]])), ("down_proj", (torch.eye(48)[[l + 3]], torch.randn(32, 1))))}
    torch.save(d, tmp_path / "delta.pt")
    qs = [{"id": f"{f}/{i}", "family": f, "split": sp, "text": f"Question {f} {i}?"}
          for sp, fams in (("train", ("a", "b")), ("test", ("c", "d"))) for f in fams for i in range(2)]
    (tmp_path / "q.json").write_text(cr.json.dumps({"questions": qs}))
    (tmp_path / "sc").mkdir()
    (tmp_path / "sc" / "score.py").write_text("def owl(t):\n    return 'owl' in (t or '')\n\n"
                                              "def assistant(t):\n    return 'AI' in (t or '')\n")
    for cmd in ("owl", "joint"):
        for mode in ("pair", "ablate"):
            monkeypatch.setattr(sys, "argv", ["circuit_reply.py", cmd, str(tmp_path / "out"), "--delta", str(tmp_path / "delta.pt"),
                                              "--questions", str(tmp_path / "q.json"), "--score", str(tmp_path / "sc" / "score.py"),
                                              "--model", str(tmp_path / "m"), "--device", "cpu", "--mode", mode, "--limit", "2", "--b-from", "prefill" if mode == "ablate" else "base",
                                              "--train", "2", "--steps", "2", "--reply", "10", "--zero-tokens", "4"])
            cr.main()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--ablate-pass", "1.0"])
    cr.main()                                       # (the last: joint, ablate, prefill; the flip bar)
    cr.ABLATE_PASS = cr.PASS
    assert (tmp_path / "out" / "owl_ablate_prefill_p1_joint.json").exists()
    o = cr.json.loads((tmp_path / "out" / "owl_pair.json").read_text())
    assert len(o["questions"]) == 2 and {"ours", "atp", "eapig", "relp", "zeroed"} <= set(next(iter(o["questions"].values())))
    assert set(next(iter(o["questions"].values()))["zeroed"]) == {"none", "planted", "ours", "ours_neurons", "random", "atp",
                                                                 "eapig", "eapig_signed", "relp"}
    j = cr.json.loads((tmp_path / "out" / "owl_ablate_prefill_joint.json").read_text())
    assert len(j["train"]) == 2 and len(j["test"]) == 2 and "ours" in j and "shares" in next(iter(j["test"].values()))


def test_planted_writes_are_an_exact_circuit(setup):
    """The delta changes only the planted neurons' gate and up rows and down columns, so patching those neurons' whole
    writes (activation and down column) moves each run all the way to the other: lost = restored = 1."""
    core, delta, prompt, A, B, P = setup
    rp = cr.Reply(core, prompt, A, B, delta, "pair")
    (lost, rest), = rp.shares([[("neuron", l, n) for l, n in sorted(P)]])
    assert abs(lost - 1.0) < 1e-4 and abs(rest - 1.0) < 1e-4


def test_atp_is_the_dial_gradient(setup):
    """AtP of a unit is minus dS/d(its dial) at the clean run: a finite difference agrees (pair, a planted neuron, the
    down-column path included)."""
    core, delta, prompt, A, B, P = setup
    rp = cr.Reply(core, prompt, A, B, delta, "pair")
    sc = rp.attribution("atp")
    l, n = sorted(P)[0]
    eps = 1e-3
    dn = torch.zeros(rp.L, rp.D)
    dn[l, n] = eps
    S1 = float(rp._run("A", ["clean"], [(dn, None)])[0] - rp._run("B", ["clean"], [(dn, None)])[0])
    fd = -(S1 - rp.S["clean"]) / eps
    assert abs(fd - float(sc["neuron"][l, n])) < 1e-2 * max(1.0, abs(fd))
