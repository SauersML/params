"""Checks of variables.py on vpd4l's text6011 (its target 288; 233 holds the copied word), on this machine's GPU
(skipped without one or vpd4l's texts): the scorer's patched runs are native.run's interchange patches (an independent
forward pass); the lead's cases: every subcomponent at 233 for "copied word" (inputs 232 and 233) reproduces the
on-target changes and leaves the off-target ones alone, the last layer at the target fails off-target; two groups
patched together reproduce a change of both inputs; names parse as goal_eval's circuits.

  ~/.local/bin/mem-lease 16 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_variables.py -s
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import variables  # noqa: E402

TEXT = native.TEXTS / "vpd4l" / "text6011.json"
DEVICE = native.device()
pytestmark = pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")


@pytest.fixture(scope="module")
def case():
    nat = native.Native(DEVICE)
    ge = goal_eval.GoalEval(nat)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    last = nat.target.n_layer - 1
    at = lambda t: [[n, c, t] for n in nat.names for c in range(nat.C[n])]  # noqa: E731
    top = [[n, c, 288] for n in nat.names if native._name_layer_kind(n)[0] == last for c in range(nat.C[n])]
    return nat, ge, tx, at, top


def test_patched_runs_are_native_patches(case):
    """A group's activations set to their values on a changed text (and every subcomponent at the target): the
    engine's run is native.run with the same patch, within float32 rounding (KL below 1e-5 bits)."""
    nat, ge, tx, at, top = case
    sc = variables.Scorer(ge)
    x = list(tx.ids)
    x[233] = 512
    src = sc._text(x, tx.targets)
    mine = sc._text(tx.ids, tx.targets)
    group = variables.parse([{"inputs": [233], "group": at(233)[::7] + top[::5]}], nat.C)[0]["group"]
    got = sc._patched(mine["E"], [(src["act"], group), (src["act"], None)])
    T = tx.T
    patch = {n: (torch.zeros(2, T, nat.C[n], dtype=torch.bool, device=nat.dev), torch.zeros(2, T, nat.C[n], device=nat.dev))
             for n in nat.names}
    for n, t, c in group:
        patch[n][0][0, t, c] = True
    for n in nat.names:
        patch[n][0][1, tx.targets] = True
        patch[n][1][:] = src["act"][n][None]
    with torch.no_grad():
        want = nat.run(tx.ids, tx.targets, nat._full_plan(T, 2), *nat._one(), patch=patch)
    assert float(carry.kl(want, got).abs().max()) < 1e-5


def test_the_leads_cases(case):
    """"copied word" (inputs 232, 233): every subcomponent at 233 reproduces the on-target changes (linear fraction
    above 0.99) and leaves the off-target ones (287, 288) alone (above 0.99); the last layer at 288 carries the
    off-target changes (log score below 0.5, linear below 0.9)."""
    nat, ge, tx, at, top = case
    vs = variables.parse([{"name": "all at 233", "inputs": [232, 233], "group": at(233)},
                          {"name": "last layer at 288", "inputs": [232, 233], "group": top}], nat.C)
    got = variables.Scorer(ge).score(tx, vs, "eval", 6)
    a, b = got["variables"]
    assert a["draws_on"] and a["draws_off"] and b["draws_off"]
    assert a["f_on"] > 0.99 and a["s_off"] > 0.99 and a["f_off"] > 0.99, a
    assert b["s_off"] < 0.5 and b["f_off"] < 0.9, b


def test_composition(case):
    """Two variables at 233 and 287, each every subcomponent at its position: patched together they reproduce a change
    of both (linear fraction above 0.99)."""
    nat, ge, tx, at, _ = case
    vs = variables.parse([{"name": "233", "inputs": [233], "group": at(233)}, {"name": "287", "inputs": [287], "group": at(287)}], nat.C)
    got = variables.Scorer(ge).score(tx, vs, "eval", 3)
    assert got["pairs"] and got["pairs"][0]["draws_comp"], got["pairs"]
    assert got["pairs"][0]["f_comp"] > 0.99, got["pairs"]


def test_parse(case):
    nat = case[0]
    v = variables.parse([{"name": "v", "inputs": [3, 1, 3], "group": [["0.fc", 5, 2], ["h.0.mlp.c_fc", 5, 2]], "feeds": ["prediction"]}], nat.C)[0]
    assert v["inputs"] == [1, 3] and v["group"] == [("h.0.mlp.c_fc", 2, 5)] and v["feeds"] == ["prediction"]
    with pytest.raises(ValueError):
        variables.parse([{"inputs": [1], "group": [["0.fc", nat.C["h.0.mlp.c_fc"], 2]]}], nat.C)
