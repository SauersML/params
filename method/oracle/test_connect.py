"""Checks of the fast connection stage (connect.py) on a tiny random model (test_goal_eval's): the integrated-gradient
scores add up to the KL every line saves, and the stated circuit keeps every part on a stated path, its point KL and
s_track_pp within the criteria, and no more lines than the wired circuit.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_connect.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import connect  # noqa: E402
import goal_eval  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402
from test_lines import nodes_of  # noqa: E402

TINY = {"scope": None, "interchange": None}  # (the tiny model has no far donors for those families' draws)


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    tx = ge.text(ids, [11], seed=5)
    return nat, ge, tx


def test_scores_add_up(tiny):
    nat, ge, tx = tiny
    c = connect.Connector(ge, tx, nodes_of(nat, tx, 3, 16), B=4)
    sc = c.scores()
    full, none = c.pr.kl_sets(torch.stack([c.row(c.conn), c.row(c.conn[:0])]))
    assert float(sc.sum()) == pytest.approx(none - full, rel=0.1, abs=1e-3)  # (8 midpoints: a Riemann sum)
    assert float(sc[c.tok].abs().sum()) == 0.0


def test_stated_circuit(tiny):
    nat, ge, tx = tiny
    c = connect.Connector(ge, tx, nodes_of(nat, tx, 5, 16), B=4)
    got = c.run(log=lambda m: None, floors=TINY)
    rec, g = got["record"], got["graph"]
    assert rec["lines"] <= rec["lines_start"] and rec["parts"] == len(g.node_set())
    assert rec["point"] <= rec["criteria"]["point"] + 1e-6
    assert all(v is None or v >= rec["criteria"][n] - 1e-6 for n, v in rec["families"].items())
    h = goal_eval.graph_of_lines(c.pr.g, c.pr.cols, c.pr.row(got["on"]))
    del h.stranded, h.stranded_bits  # (graph_of_lines marks it stranded already)
    again = goal_eval.strand(h, tx.targets, nat.C)
    assert set(again.node_set()) == set(g.node_set())  # every part kept lies on a stated path to the prediction


def test_connect_interface(tiny):
    nat, ge, tx = tiny
    g = connect.connect(ge, tx, connect.native.Graph(nodes_of(nat, tx, 7, 14), complete=True), floors=TINY)
    assert not g.complete and g.connect["parts"] == len(g.node_set())
