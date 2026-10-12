"""Checks of teacher_reads.py on a short held-out vpd4l text, on this machine's GPU (skipped without vpd4l's texts or
VPD's answer: the causal-importance network or teacher.py --save-vpd's file VPD_SAVED): the search's runs without cuts
are the masked runs (teacher.Pruner's), a read cut through the plan's free connections at 0 is the same run as that
read in the state's cuts, every row of one Reads plan runs its own subset, and runs from the first changed position
reusing the run before it are the whole runs.

  ~/.local/bin/mem-lease 16 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_teacher_reads.py -s
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
import teacher_reads  # noqa: E402

TASK = "text9002"   # 42 tokens, its prediction reading earlier positions
TEXT = native.TEXTS / "vpd4l" / f"{TASK}.json"
VPD_SAVED = Path.home() / "mpd-data/graph_oracle/teacher/vpd_heldout20.pt"
REL = 1e-4          # test_teacher.py's float32 bound on KLs in bits
DEVICE = native.device()


@pytest.fixture(scope="module")
def state():
    if not TEXT.exists():
        pytest.skip("needs vpd4l's held-out texts")
    nat = native.Native(DEVICE)
    ge = goal_eval.GoalEval(nat, chunk=16)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    if tx.vpd is None:
        if not VPD_SAVED.exists():
            pytest.skip("needs VPD's answer: the causal-importance network or " + str(VPD_SAVED))
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in torch.load(VPD_SAVED)[TASK]["answer"].items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    return teacher_reads.State(ge, tx, 16)


def close(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    return float(((a - b).abs() / a.abs()).max()) < REL


def test_uncut_runs_are_the_masked_runs(state):
    gen = torch.Generator().manual_seed(0)
    ons = torch.rand(4, len(state.nodes), generator=gen).to(state.dev) < torch.tensor([[1.0], [0.9], [0.6], [0.3]], device=state.dev)
    assert close(state.kl_nodes(ons), state.pr.kl_sets(ons, pad=True))


def test_free_reads_at_zero_are_cuts(state):
    p = max(state.tx.targets)
    rd = teacher_reads.Reads(state, p)
    assert rd.units, "the target reads no earlier position"
    ks = [0, len(rd.units) // 2, len(rd.units) - 1]
    ons = rd.on[None].repeat(1 + len(ks), 1)
    for i, k in enumerate(ks):
        ons[1 + i, k] = False
    got = rd.kl_sets(ons)
    assert close(got[:1], state.kl_nodes(state.on[None]))
    for i, k in enumerate(ks):
        state.cut = {rd.units[k]}
        try:
            assert close(got[1 + i:2 + i], state.kl_nodes(state.on[None])), rd.units[k]
        finally:
            state.cut = set()
    cut_all = rd.on.clone()
    cut_all[:] = False
    whole = rd.kl_sets(cut_all[None])[0]
    state.cut = set(rd.units)
    try:
        assert close([whole], state.kl_nodes(state.on[None]))
    finally:
        state.cut = set()


def test_runs_from_the_first_change_are_whole_runs(state):
    p = max(state.tx.targets)
    here = (state.t == p).nonzero().flatten()
    ons = state.on[None].repeat(3, 1)
    ons[1, here[: len(here) // 2]] = False
    ons[2, here] = False
    whole = state.runs(state.plan(), 3, lambda a, b: state.u(ons[a:b]), lambda a, b: None, 0, None)
    assert close(state.kl_nodes(ons), whole)
    rd = teacher_reads.Reads(state, p)
    cuts = rd.on[None].repeat(2, 1)
    cuts[1, : len(rd.units) // 2] = False
    whole = state.runs(rd.plan, 2, lambda a, b: rd.u0, lambda a, b: cuts[a:b][:, rd.unit_of].float(), 0, None)
    assert close(rd.kl_sets(cuts), whole)
