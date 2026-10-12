"""Checks of var_classes.py and var_teacher.py. On vpd4l's text6011 (its target 288; 233 holds the copied token ' each'),
on this machine's GPU (skipped without one or vpd4l's texts): every subcomponent at 233 carries each alternative's
whole change, so no alternative is alike ' each'; an empty group carries nothing, so every alternative is. Without a
GPU: var_teacher's carrier membership and stated feeds.

  ~/.local/bin/mem-lease 16 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_var_classes.py -s
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import goal_eval  # noqa: E402
import native  # noqa: E402
import var_classes  # noqa: E402
import var_teacher  # noqa: E402
import variables  # noqa: E402

TEXT = native.TEXTS / "vpd4l" / "text6011.json"
DEVICE = native.device()
gpu = pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")


@pytest.fixture(scope="module")
def case():
    nat = native.Native(DEVICE)
    ge = goal_eval.GoalEval(nat)
    ids, targets = native.text(TEXT)
    return nat, variables.Scorer(ge), ge.text(ids, targets)


@gpu
def test_whole_position_tells_every_alternative_apart(case):
    """Every subcomponent at 233 patched from an alternative reproduces the changed text (f within float32 of 1), so
    the class of ' each' holds it alone and the changed classes hold every alternative."""
    nat, sc, tx = case
    v = variables.parse([{"inputs": [233], "group": [[n, c, 233] for n in nat.names for c in range(nat.C[n])]}], nat.C)[0]
    got = var_classes.classes(sc, tx, v, 233, top=6, sampled=4)
    assert got["token"] == " each" and got["equivalent"] == [" each"]
    assert sum(len(c["tokens"]) for c in got["changed"]) == len(got["alternatives"]) == 10
    assert all(r["f"] is not None and r["f"] > 0.999 for r in got["alternatives"])


@gpu
def test_empty_group_tells_nothing_apart(case):
    """No subcomponent patched: every patched run is the text's own, so every alternative is alike ' each'."""
    nat, sc, tx = case
    got = var_classes.classes(sc, tx, {"name": "", "inputs": [233], "group": []}, 233, top=4, sampled=2)
    assert len(got["equivalent"]) == 7 and not got["changed"]
    assert all(r["move"] <= got["d_res"] for r in got["alternatives"])


def test_inside_blocks_hold_their_site_and_position():
    """A block holds every subcomponent of its site and position; a circuit's block lies in a group holding anything
    at its site and position; a subcomponent elsewhere does not."""
    B = var_teacher.BLOCK
    g = {("x", 5, B), ("y", 6, 3)}
    assert var_teacher.inside(g, ("x", 5, 7)) and var_teacher.inside(g, ("x", 5, B))
    assert var_teacher.inside(g, ("y", 6, B)) and var_teacher.inside(g, ("y", 6, 3))
    assert not var_teacher.inside(g, ("x", 6, 7)) and not var_teacher.inside(g, ("y", 6, 4))


def test_stated_feeds_follow_paths_through_unnamed_nodes():
    """a -> b -> c -> d -> prediction with a in V1 and c in V2: V1 feeds V2 (through the unnamed b) and stops there;
    V2 feeds the prediction (through the unnamed d)."""
    a, b, c, d = ("x", 1, 0), ("x", 2, 0), ("x", 3, 0), ("x", 4, 0)
    g = native.Graph([a, b, c, d], {b: [a], c: [b], d: [c]}, [d])
    chosen = [{"name": "V1", "_group": [a]}, {"name": "V2", "_group": [c]}]
    assert var_teacher.stated_feeds(g, chosen) == {"V1": ["V2"], "V2": ["prediction"]}
