"""Checks of the line stage (lines.py) on a tiny random model (test_goal_eval's): the point runs from one plan equal the
scorer's runs of the same stated graphs (stranded parts dropped), the levels' blocks partition the kept lines, rows
turn off exactly the stranded parts' lines, and the floors' keep masks name exactly a candidate's lines.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_lines.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import goal_eval  # noqa: E402
import lines  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    tx = ge.text(ids, [11], seed=5)
    return nat, ge, tx


def nodes_of(nat, tx, seed: int, k: int, lo: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    t = tx.targets[0]
    out = {(n, t, 0) for n in nat.names if n.endswith(("down_proj", "o_proj"))}  # writers into the prediction
    while len(out) < k:
        n = nat.names[int(torch.randint(len(nat.names), (1,), generator=g))]
        out.add((n, int(torch.randint(lo, tx.T, (1,), generator=g)), int(torch.randint(nat.C[n], (1,), generator=g))))
    return sorted(out)


@pytest.mark.parametrize("lo", [0, 4])
def test_point_runs_are_the_scorers(tiny, lo):
    nat, ge, tx = tiny
    pr = lines.LinePruner(ge, tx, nodes_of(nat, tx, 3, 16, lo), B=4)
    assert pr.lines and (pr.lo > 0 or lo == 0)
    g = torch.Generator().manual_seed(7)
    ons = torch.stack([torch.ones(len(pr.lines), dtype=torch.bool)] + [torch.rand(len(pr.lines), generator=g) < q for q in (0.8, 0.5, 0.3, 0.1)])
    got = pr.kl_sets(ons)
    want = ge.graph_kl(tx, [pr.graph(on) for on in ons], True)
    assert got == pytest.approx(want, rel=1e-4, abs=1e-5)


def test_blocks_partition_the_kept_lines(tiny):
    nat, ge, tx = tiny
    pr = lines.LinePruner(ge, tx, nodes_of(nat, tx, 5, 14), B=4)
    pr.on[::3] = False
    for level in lines.LEVELS:
        ids = torch.cat(pr.blocks(level)).tolist()
        assert sorted(ids) == pr.on.nonzero().flatten().tolist()


def test_rows_and_keys(tiny):
    nat, ge, tx = tiny
    pr = lines.LinePruner(ge, tx, nodes_of(nat, tx, 9, 14), B=4)
    assert all(pr.at[lines.key(c)] == j for j, c in enumerate(pr.lines))
    on = torch.rand(len(pr.lines), generator=torch.Generator().manual_seed(2)) < 0.4
    h, row = pr.graph(on), pr.row(on)
    kept = h.node_set()
    assert {c[1] for c, r in zip(pr.cols, row.tolist()) if c[0] == "n" and r} == set(kept)
    again = goal_eval.graph_of_lines(pr.g, pr.cols, row)  # the row's graph: no part without a path, nothing to strand
    del again.stranded, again.stranded_bits
    assert set(goal_eval.strand(again, tx.targets, nat.C).node_set()) == set(kept)
    has = lines.held(h)
    full = pr.graph(torch.ones(len(pr.lines), dtype=torch.bool))
    keep = [has(kind, x) for kind, x in ge.loo_lines(tx, full)]
    assert sum(keep) == len(ge.loo_lines(tx, h))
