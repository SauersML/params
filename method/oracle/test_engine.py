"""Checks of engine.py on a held-out vpd4l text, on this machine's GPU (skipped without one or vpd4l's texts): the
engine's float32 masked run with every subcomponent addressable equals VPD's masked forward pass (goal_eval.runs, the
same steps) to the bit on MPS and within float32 rounding on CUDA; its gradients with respect to the masks match that
pass's autograd; gathered columns and the search precisions (TF32, bf16) stay within their stated bounds; rescore
reports float32.

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_engine.py -s
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import engine  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

TEXT = native.TEXTS / "vpd4l" / "text9000.json"
VPD_SAVED = Path.home() / "mpd-data/graph_oracle/teacher/vpd_heldout20.pt"
DEVICE = native.device()
# KL in bits is a difference of nearly equal log probabilities; a float32 sum of n terms is off by about sqrt(n) 2^-24
# of its size (n = 3,072 at the MLP: 3.3e-6), magnified tenfold by the KL's cancellation on these texts
REL = 1e-4
# TF32 products keep 10 mantissa bits (2^-11 relative, 5e-4), the KL's cancellation magnifying it tenfold (measured:
# 1.4e-3 on an RTX A5000); bf16 keeps 8 (2^-9), the KL off by a few percent. Both are for a search's steps only
REL_TF32 = 1e-2
REL_BF16 = 0.15
pytestmark = pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")


def setup(B=6, seed=0):
    nat = native.Native(DEVICE)
    ids, targets = native.text(TEXT)
    g = torch.Generator().manual_seed(seed)
    T = len(ids)
    # masks between all on and a sparse random set: rows from near the model to far from it
    masks = {n: (torch.rand(B, T, nat.C[n], generator=g) < torch.linspace(1.0, 0.02, B)[:, None, None]).float().to(nat.dev)
             for n in nat.names}
    return nat, ids, targets, masks


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return float(((a - b).abs() / a.abs().clamp_min(1e-12)).max())


@pytest.mark.parametrize("live", [True, False])
def test_engine_equals_masked_forward(live):
    nat, ids, targets, masks = setup()
    ge = goal_eval.GoalEval(nat, chunk=len(next(iter(masks.values()))))
    ref = nat.reference([ids], targets)
    want = carry.kl(ref, ge.runs(ids, targets, [{n: m[b].bool() for n, m in masks.items()} for b in range(6)], live))
    with torch.no_grad():
        got = engine.Engine(nat, ids, targets, live=live).kl(masks)
    if DEVICE == "mps":
        assert torch.equal(got, want)
    else:
        assert rel(want, got) < REL


def test_gradients_match_autograd_of_masked_forward():
    nat, ids, targets, masks = setup(B=2)
    tg = nat.target
    a = {n: m.clone().requires_grad_(True) for n, m in masks.items()}
    for n in nat.names:
        tg.site(n).mask, tg.site(n).delta_mask = a[n], torch.ones(2, len(ids), device=nat.dev)
    try:
        h = tg.hidden(torch.tensor([ids], device=nat.dev).expand(2, -1))
    finally:
        nat.vpd.clear()
    ref = nat.reference([ids], targets)
    carry.kl(ref, torch.log_softmax((h[:, targets] @ tg.wte.T).float(), -1)).sum().backward()
    b = {n: m.clone().requires_grad_(True) for n, m in masks.items()}
    engine.Engine(nat, ids, targets, live=True).kl(b).sum().backward()
    for n in nat.names:
        x, y = a[n].grad.flatten(), b[n].grad.flatten()
        assert float((x - y).norm() / x.norm().clamp_min(1e-30)) < REL


def test_gathered_columns_and_search_precisions():
    nat, ids, targets, masks = setup()
    if nat.has_importance:
        active = {n: (v > 0).any(0) for n, v in nat.importance(ids).items()}    # VPD's active columns at any position
    elif VPD_SAVED.exists():
        active = {n: torch.zeros(nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (_, c) in torch.load(VPD_SAVED)["text9000"]["answer"].items():
            active[n][c.long().to(nat.dev)] = True
    else:
        pytest.skip("needs VPD's answer: the causal-importance network or " + str(VPD_SAVED))
    cols = {n: torch.nonzero(active[n]).flatten() for n in nat.names}
    sub = {n: m * active[n].float() for n, m in masks.items()}                  # masks inside those columns
    whole = engine.Engine(nat, ids, targets)
    gathered = engine.Engine(nat, ids, targets, columns=cols)
    compact = {n: m.index_select(-1, cols[n].to(m.device)) for n, m in sub.items()}
    with torch.no_grad():
        full = whole.kl(sub)
        assert rel(full, gathered.kl(compact)) < REL
        assert torch.equal(gathered.full(compact)["h.0.mlp.c_fc"], sub["h.0.mlp.c_fc"])
        if DEVICE.startswith("cuda"):
            assert rel(full, gathered.kl(compact, "tf32")) < REL_TF32
        assert rel(full, gathered.kl(compact, "bf16")) < REL_BF16
    assert torch.equal(gathered.rescore(compact), gathered.kl(compact))
