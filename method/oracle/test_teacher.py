"""Checks of teacher.py on a held-out vpd4l text (skipped without ~/mpd-data's model and an MPS device): the windowed
masked runs (Pruner.kl_sets on a Native.WINDOW_EXACT device) equal the whole runs, the products on the start
set's subcomponents alone (Pruner._gather) with torch's fused norms and GELU (Pruner.fused) match the products on all
subcomponents with vpd_model's own steps within float32 rounding, and EdgePruner's graph runs with remainders off
give the circuit's masked run with every connection and the empty circuit's with none.

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_teacher.py -s
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

TEXT = native.TEXTS / "vpd4l" / "text9000.json"


@pytest.mark.skipif(not TEXT.exists() or not torch.backends.mps.is_available(), reason="needs vpd4l's texts and MPS")
def test_windowed_runs_equal_whole_runs():
    nat = native.Native("mps")
    if not nat.has_importance:
        pytest.skip("needs VPD's causal-importance network")
    ge = goal_eval.GoalEval(nat, chunk=32)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    pr = teacher.Pruner(ge, tx, tx.vpd, live=False, B=32)
    nodes = torch.randperm(len(pr.nodes), generator=torch.Generator().manual_seed(0))[:96].to(nat.dev)
    trials = pr.on[None].repeat(96, 1)
    trials[torch.arange(96, device=nat.dev), nodes] = False   # single-node removals at mixed layers and positions
    blocks = pr.blocks("site")[::40]                            # site removals spread over the text
    keep = type(nat).WINDOW_EXACT
    try:
        type(nat).WINDOW_EXACT = ()
        whole = (pr.kl_sets(trials), pr.kl_sets(~trials), pr.without(blocks))
        type(nat).WINDOW_EXACT = ("mps",)
        pr._prefixes = {}
        windowed = (pr.kl_sets(trials), pr.kl_sets(~trials), pr.without(blocks))
    finally:
        type(nat).WINDOW_EXACT = keep
    for a, b in zip(whole, windowed):
        assert a == b


@pytest.mark.skipif(not TEXT.exists() or not torch.backends.mps.is_available(), reason="needs vpd4l's texts and MPS")
def test_gathered_products_match_full_products():
    nat = native.Native("mps")
    if not nat.has_importance:
        pytest.skip("needs VPD's causal-importance network")
    ge = goal_eval.GoalEval(nat, chunk=32)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    pr = teacher.Pruner(ge, tx, tx.vpd, live=False, B=32)
    nodes = torch.randperm(len(pr.nodes), generator=torch.Generator().manual_seed(1))[:64].to(nat.dev)
    trials = pr.on[None].repeat(64, 1)
    trials[torch.arange(64, device=nat.dev), nodes] = False
    gathered = pr.kl_sets(trials)
    pr._gather(every=True)                                      # every column: the full products
    pr.fused = False                                            # vpd_model's own norms and GELU
    pr._prefixes = {}
    full = pr.kl_sets(trials)
    a, b = torch.tensor(full, dtype=torch.float64), torch.tensor(gathered, dtype=torch.float64)
    # KL is a difference of nearly equal log probabilities: float32 rounding of the logits shows at ~1e-5 of it
    assert float(((a - b).abs() / a.abs()).max()) < 1e-4


@pytest.mark.skipif(not TEXT.exists() or not torch.backends.mps.is_available(), reason="needs vpd4l's texts and MPS")
def test_graph_runs_bound_by_masked_runs():
    nat = native.Native("mps")
    if not nat.has_importance:
        pytest.skip("needs VPD's causal-importance network")
    ge = goal_eval.GoalEval(nat, chunk=32)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    pr = teacher.Pruner(ge, tx, tx.vpd, live=False, B=32)
    at = [i for i, (_, t, _) in enumerate(pr.nodes) if t in targets]
    pick = torch.randperm(len(at), generator=torch.Generator().manual_seed(2))[:24].tolist()
    nodes = [pr.nodes[at[i]] for i in pick]
    ep = teacher.EdgePruner(ge, tx, nodes, live=False, B=8)
    every, none = ep.kl_sets(torch.stack([ep.on, ~ep.on]))
    masked = carry.kl(tx.ref, ge.runs(ids, targets, [ge.masks(native.Graph(nodes), tx.T)], False)).item()
    empty = ge._bounds(tx, "off")[0]
    # remainders off: every connection stated is the circuit's masked run, none stated the empty circuit's
    assert abs(every - masked) / masked < 1e-4
    assert abs(none - empty) / empty < 1e-4
