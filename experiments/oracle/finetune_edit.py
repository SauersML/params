"""Rank-r weight updates for the persistent edits (#2951 graph oracle; experiments/oracle/persistent_e3.py): an update
B A^T added to one weight matrix W [d_out, d_in] (its output gains (x A) B^T) while inside a Delta block, the model's
log probabilities with it, the KL in bits, and the unit read and write directions of a matrix's subcomponents (the
restricted fits' bases).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "oracle"))
sys.path.insert(0, str(ROOT / "method" / "vpd"))

LN2 = math.log(2)


class Delta:
    """W's output gains (x A) B^T at the positions where mask is 1, while inside the block."""

    def __init__(self, site, A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor):
        self.site, self.A, self.B, self.mask = site, A, B, mask

    def __enter__(self):
        self.orig = self.site._forward
        A, B, mask, orig = self.A, self.B, self.mask, self.orig
        self.site._forward = lambda x: orig(x) + ((x @ A) @ B.T) * mask[None, :, None]
        return self

    def __exit__(self, *exc):
        self.site._forward = self.orig


def logq(nat, ids: list[list[int]], site=None, A=None, B=None, mask=None) -> torch.Tensor:
    """log q (natural) [len(ids), T, V] of the model, with the update when site is given."""
    tg = nat.target
    x = torch.tensor(ids, device=nat.dev)
    if site is None:
        return torch.log_softmax(tg.hidden(x) @ tg.wte.T, -1)
    with Delta(tg.site(site), A, B, mask):
        return torch.log_softmax(tg.hidden(x) @ tg.wte.T, -1)


def kl_bits(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    return (lp.exp() * (lp - lq)).sum(-1) / LN2


def basis(nat, site: str, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(the unit read directions v [d_in, k], the unit write directions u [d_out, k]) of the site's subcomponents idx."""
    st = nat.target.site(site)
    V, U = st.V[:, idx], st.U[idx].T
    return V / V.norm(dim=0, keepdim=True), U / U.norm(dim=0, keepdim=True)
