"""Checks of RL environment E (edit_env.py): the answer format both ways, the weight edit's arithmetic and restoration,
and the verifier's counts on a tiny random model (test_goal_eval's).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_edit_env.py -q
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import edit_env  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402

C = {edit_env.matrix(layer, k): 3072 if k in ("fc", "down") else 1024 for layer in range(4) for k in edit_env.KINDS}


def test_parse_render_round_trip():
    ans = "\n".join(["Reasoning first.", "scale L1.fc#2036 -1", "scale L2.down#946 0", "scale L2.down#946 0.5",
                     "scale L9.fc#1 0", "scale L1.zz#1 0", "update L0.q read 3:1 4:-0.5 write 7:2.5",
                     "update L0.q read 3:1 write 99999:1", "scale L3.o#5 7"])
    ops, bad = edit_env.parse(ans, C)
    assert bad == 3  # a layer the model lacks, a kind it lacks, an index past the matrix's subcomponents
    assert ops == [("scale", "h.1.mlp.c_fc", 2036, -1.0), ("scale", "h.2.mlp.down_proj", 946, 0.5),
                   ("update", "h.0.attn.q_proj", [(3, 1.0), (4, -0.5)], [(7, 2.5)]), ("scale", "h.3.attn.o_proj", 5, edit_env.COEF)]
    assert edit_env.parse(edit_env.render(ops), C) == (ops, 0)
    assert edit_env.parse("no edit here", C) == ([], 0)
    many = "\n".join(f"scale L0.fc#{i} 0" for i in range(40))
    assert len(edit_env.parse(many, C)[0]) == edit_env.MAX_OPS


def test_applied_edits_weights_and_restores():
    nat = tiny_native()
    st = nat.target.site("h.0.mlp.c_fc")
    W0 = st.W.clone()
    x = torch.randn(5, W0.shape[1])
    with edit_env.applied(nat, [("scale", "h.0.mlp.c_fc", 2, 0.25)]):
        got = x @ st.W.T
    want = x @ W0.T - 0.75 * (x @ st.V[:, 2])[:, None] * st.U[2][None, :]
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.equal(st.W, W0)
    # "read i:1 write i:-1" removes subcomponent i, as "scale ... 0" does
    with edit_env.applied(nat, [("update", "h.0.mlp.c_fc", [(2, 1.0)], [(2, -1.0)])]):
        a = st.W.clone()
    with edit_env.applied(nat, [("scale", "h.0.mlp.c_fc", 2, 0.0)]):
        b = st.W.clone()
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.equal(st.W, W0)


def test_verifier_counts_on_a_tiny_model():
    nat = tiny_native(layers=4)
    g = np.random.default_rng(0)
    tokens = g.integers(0, 40, size=(8, 24))
    held = [g.integers(0, 40, size=16).tolist() for _ in range(3)]
    ver = edit_env.Verifier(nat, tokens, held)
    lp = ver._lp([tokens[r].tolist() for r in range(8)])
    top = lp.argmax(-1)
    b = int(torch.mode(top.flatten()).values)
    places = [(r, p) for r in range(8) for p in range(1, 24) if top[r, p] == b]
    pref = {"b": b, "parts": {"score": {"instances": places[::2], "controls": places[1::2]}}}
    empty, big = ver.score(pref, [[], [("scale", "h.3.mlp.down_proj", c, -3.0) for c in range(6)]])
    assert empty == {**empty, "removal": 0.0, "controls_kept": 1.0, "collateral": 0.0, "reward": 0.0}
    assert big["instances"] == len(places[::2]) and big["controls"] == len(places[1::2])
    assert big["collateral"] > 0 and abs(big["collateral"] - big["text_kl"] - big["pile_kl"]) < 1e-9
    assert abs(big["reward"] - (big["removal"] - edit_env.LAM * big["collateral"] - edit_env.MU * (1 - big["controls_kept"]))) < 1e-9
    # padding: a text scored in a batch with longer ones reads the same as alone
    alone = ver._lp([tokens[0, :10].tolist()])[0]
    batched = ver._lp([tokens[0, :10].tolist(), tokens[1].tolist()])[0, :10]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_best_picks_the_candidate_earning_most_on_val():
    v = lambda f, c, k: {"flipped": f, "collateral_per_token": c, "controls_kept": k}  # noqa: E731
    row = {"methods": {
        "circuit_edit": {"ops": [["h.0.mlp.c_fc", 1, 0.0]], "validation": v(0.95, 0.01, 0.1), "info": {},
                         "candidates": [[[["h.0.mlp.c_fc", 1, 0.0]], v(0.95, 0.01, 0.1), {}],
                                        [[["h.0.mlp.c_fc", 2, -1.0]], v(0.80, 0.01, 0.9), {}]]},
        "circuit_scales": {"ops": [["h.1.mlp.c_fc", 3, 0.5]], "validation": v(0.5, 0.0, 1.0), "info": {}}}}
    m, ops, r = edit_env.best(row, edit_env.TEACHERS)
    assert (m, ops) == ("circuit_edit", [("scale", "h.0.mlp.c_fc", 2, -1.0)])
    assert abs(r - (0.80 - edit_env.LAM * 0.01 - edit_env.MU * 0.1)) < 1e-12
    assert edit_env.best(row, edit_env.RANDOMS) is None
