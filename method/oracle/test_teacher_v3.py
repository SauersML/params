"""Checks of teacher_v3.py on a held-out vpd4l text (skipped without ~/mpd-data's model and saved VPD answers): its
masks with every gate at 1 run VPD's own answer and with every gate at 0 the empty circuit (the bounds goal_eval
computes on every column; MaskSearch multiplies the gathered ones), and a pattern on top of gates at 1 is VPD's answer
under that pattern.

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_teacher_v3.py -s
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
import teacher  # noqa: E402
import teacher_v3  # noqa: E402

SAVED = Path.home() / "mpd-data/graph_oracle/teacher/vpd_heldout20.pt"
TASK = "text9000"


@pytest.mark.skipif(not SAVED.exists() or not (native.TEXTS / "vpd4l" / f"{TASK}.json").exists(), reason="needs vpd4l's texts and saved answers")
def test_masks_bound_by_vpd_and_empty():
    saved = torch.load(SAVED)
    ge = goal_eval.GoalEval(native.Native(), chunk=32)
    nat = ge.nat
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{TASK}.json")
    tx = ge.text(ids, targets)
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in saved[TASK]["answer"].items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    empty, vpd = ge._bounds(tx, "live")
    pr = teacher.Pruner(ge, tx, tx.vpd, live=True, B=32)
    ms = teacher_v3.MaskSearch(ge, tx, pr, teacher_v3.patterns(saved, TASK, tx.T, nat.C, nat.names, 2, 0))
    pats = ms.pats
    N = len(pr.nodes)
    with torch.no_grad():
        on, off = ms.kl(ms.masks(torch.ones(1, N, device=nat.dev), None)), ms.kl(ms.masks(torch.zeros(1, N, device=nat.dev), None))
        a = ms.masks(torch.ones(2, N, device=nat.dev), pats)
        b = ms.masks(None, pats, vpd_on=True)
    assert abs(float(on) - vpd) / vpd < 1e-3
    assert abs(float(off) - empty) / empty < 1e-3
    assert all(torch.equal(a[n], b[n]) for n in nat.names)


@pytest.mark.skipif(not SAVED.exists() or not (native.TEXTS / "vpd4l" / f"{TASK}.json").exists(), reason="needs vpd4l's texts and saved answers")
def test_engine_runs_are_the_masked_forward():
    """MaskSearch's runs (the engine on the gathered columns) equal VPD's masked forward over every column (goal_eval's
    runs) within float32 rounding, for random gates under patterns; the bf16 steps within the engine's bf16 bound."""
    saved = torch.load(SAVED)
    ge = goal_eval.GoalEval(native.Native(), chunk=32)
    nat = ge.nat
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{TASK}.json")
    tx = ge.text(ids, targets)
    tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    for n, (t, c) in saved[TASK]["answer"].items():
        tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    pr = teacher.Pruner(ge, tx, tx.vpd, live=True, B=32)
    ms = teacher_v3.MaskSearch(ge, tx, pr, teacher_v3.patterns(saved, TASK, tx.T, nat.C, nat.names, 3, 0))
    g = torch.Generator().manual_seed(0)
    z = (torch.rand(3, len(pr.nodes), generator=g) < torch.tensor([[0.9], [0.5], [0.1]])).float().to(nat.dev)
    m = ms.masks(z, ms.pats)
    with torch.no_grad():
        got = ms.kl(m)
        full = [{n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev).index_copy(
            1, ms.cols[j], m[n][r].bool()) for j, n in enumerate(nat.names)} for r in range(3)]
        want = carry.kl(tx.ref, ge.runs(ids, targets, full, True))
        assert float(((got - want).abs() / want.abs()).max()) < 1e-4
        assert float(((ms.kl(m, "bf16") - want).abs() / want.abs()).max()) < 0.05
