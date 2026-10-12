"""Checks of teacher_v3.py on a held-out vpd4l text (skipped without vpd4l's texts and VPD's saved answers): its masks
with every gate at 1 run VPD's own answer and with every gate at 0 the empty circuit (the bounds goal_eval computes),
the omitted lever on gates at 1 changes nothing (nothing is left out), and the pick never exceeds the answer budget.

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_teacher_v3.py -s
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import goal_eval  # noqa: E402
import native  # noqa: E402
import teacher  # noqa: E402
import teacher_v3  # noqa: E402

SAVED = Path.home() / "mpd-data/graph_oracle/teacher/vpd_heldout20.pt"
DEVICE = native.device()


def text(task):
    saved = torch.load(SAVED)
    ge = goal_eval.GoalEval(native.Native(DEVICE), chunk=32)
    nat = ge.nat
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
    tx = ge.text(ids, targets)
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in saved[task]["answer"].items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    return ge, tx, saved


@pytest.mark.skipif(not SAVED.exists() or DEVICE == "cpu", reason="needs VPD's saved answers and a GPU")
def test_masks_bound_by_vpd_and_empty():
    ge, tx, _ = text("text9000")
    empty, vpd = ge._bounds(tx, "live")
    pr = teacher.Pruner(ge, tx, tx.vpd, live=True, B=32)
    ms = teacher_v3.MaskSearch(ge, tx, pr)
    N = len(pr.nodes)
    ones, zeros = torch.ones(1, N, device=ge.nat.dev), torch.zeros(1, N, device=ge.nat.dev)
    lever = teacher_v3.top(torch.rand(1, N, device=ge.nat.dev), teacher.REPAIR_M)
    with torch.no_grad():
        on, off, levered = (float(ms.kl(m)) for m in (ones, zeros, ones + (1 - ones) * lever))
    assert abs(on - vpd) / vpd < 1e-3 and abs(off - empty) / empty < 1e-3 and levered == on


@pytest.mark.skipif(not SAVED.exists() or DEVICE == "cpu" or not teacher.POOL.exists(), reason="needs VPD's answers, a GPU, the pool")
def test_pick_within_budget(monkeypatch):
    monkeypatch.setattr(teacher_v3, "LAMBDAS", [1.0, 64.0])   # two stages: VPD's whole answer and a small one
    monkeypatch.setattr(teacher_v3, "STEPS", 30)
    ge, tx, saved = text("text9019")
    circuits, row, _ = teacher_v3.search(ge, tx.ids, tx.targets, saved["text9019"]["answer"], torch.load(teacher.POOL), max_nodes=40)
    assert row["best"]["size"] <= 40
