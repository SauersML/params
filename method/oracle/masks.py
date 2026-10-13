"""Learned masks over VPD subcomponents (#2951 graph oracle; desiderata-based masking, Davies et al. 2023, multi-task as
RAVEL's MDBM, annealed sigmoid boundaries as Boundless DAS): a variable's carrier as the subset of candidate
subcomponents whose activations, moved to a donor's, give the prediction the variable claims.

Patcher runs the target model on a text with each candidate's activation a (v . x) moved to a + m (a' - a), a' its
activation on a donor, through forward hooks on the candidate sites (the rest of every site unchanged, gradients
through every later layer; at m = 1 it equals variables.Scorer's patched run, within 4e-7 bits on text6011). train()
fits m = sigmoid(l / tau), tau annealed from 1 to 0.05, by Adam on the mean KL(target || patched) over donors plus lam
times the mean of m: a Cause donor's target is the model on the donor (the variable carries the change), an Iso
donor's is the model on the text (a change the variable must not carry). The mask is the candidates with l > 0.
"""
from __future__ import annotations

import math

import torch

import carry

STEPS = 300
BATCH = 8


class Patcher:
    """The target model on the text with candidates' activations interpolated toward a donor's: forward hooks on the
    candidate sites add (m (a' - a)) U at the candidate positions."""

    def __init__(self, nat, ids: list[int], targets: list[int], cands: list[tuple]):
        self.nat, self.tg, self.dev = nat, nat.target, nat.dev
        self.ids, self.targets = list(ids), list(targets)
        self.by = {}
        for j, (n, p, c) in enumerate(cands):
            self.by.setdefault(n, []).append((j, p, c))
        self.idx = {n: torch.tensor([j for j, _, _ in v], device=self.dev) for n, v in self.by.items()}
        self.pos = {n: torch.tensor([p for _, p, _ in v], device=self.dev) for n, v in self.by.items()}
        self.col = {n: torch.tensor([c for _, _, c in v], device=self.dev) for n, v in self.by.items()}
        self.K = len(cands)

    def acts(self, texts: list[list[int]]) -> torch.Tensor:
        """Every candidate's activation (v . x) on each text [len(texts), K] (the model's own run)."""
        out = torch.zeros(len(texts), self.K, device=self.dev)
        for n in self.by:
            self.tg.site(n).cache_input = True
        with torch.no_grad():
            for s0 in range(0, len(texts), BATCH):
                part = texts[s0:s0 + BATCH]
                self.tg(torch.tensor(part, device=self.dev))
                for n in self.by:
                    st = self.tg.site(n)
                    x = st.last_input[:, self.pos[n]]  # [b, k, d_in]
                    out[s0:s0 + len(part)][:, self.idx[n]] = (x * st.V[:, self.col[n]].T[None]).sum(-1)
        for n in self.by:
            self.tg.site(n).cache_input = False
            self.tg.site(n).last_input = None
        return out

    def logq(self, m: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """log q at the targets [B, targets, V] on the text with each row's candidates moved by m toward src [B, K]."""
        B = src.shape[0]
        hooks = []

        def hook(n):
            st = self.tg.site(n)
            V, U = st.V[:, self.col[n]], st.U[self.col[n]]  # [d_in, k], [k, d_out]

            def f(mod, inp, out):
                x = inp[0][:, self.pos[n]]  # [B, k, d_in]
                a = (x * V.T[None]).sum(-1)  # [B, k]
                d = (m[self.idx[n]][None] * (src[:, self.idx[n]] - a))[..., None] * U[None]  # [B, k, d_out]
                out = out.clone()
                out.index_add_(1, self.pos[n], d)
                return out
            return st.register_forward_hook(f)
        hooks = [hook(n) for n in self.by]
        try:
            logits = self.tg.hidden(torch.tensor([self.ids] * B, device=self.dev))[:, self.targets] @ self.tg.wte.T
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(logits.float(), -1)


def train(P: Patcher, src: torch.Tensor, target: torch.Tensor, lam: float, steps: int = STEPS, seed: int = 0) -> torch.Tensor:
    """The mask's logits after Adam on mean KL(target || patched) + lam mean(m), tau annealed 1 -> 0.05."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logit = torch.zeros(P.K, device=P.dev, requires_grad=True)
    opt = torch.optim.Adam([logit], lr=0.1)
    for step in range(steps):
        tau = math.exp(math.log(1.0) + (math.log(0.05) - math.log(1.0)) * step / max(steps - 1, 1))
        rows = torch.randperm(src.shape[0], generator=gen)[:BATCH].to(P.dev)
        m = torch.sigmoid(logit / tau)
        lq = P.logq(m, src[rows])
        loss = carry.kl(target[rows], lq).mean() + lam * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return logit.detach()


RECENT = 8


def iso_positions(tx, ins: tuple, at: tuple, skip: tuple = ()) -> list[int]:
    """The Iso donors' positions: the influential ones (tx.f2) and the RECENT just before the carrier, strictly before it
    (a change at or after the carrier passes through it by construction), outside the inputs and skip. The recent ones
    because F2 can hold the inputs alone (text6011 on CUDA: [233]), which leaves no Iso donor."""
    lo = min(at)
    qs = {q for q, _ in tx.f2} | set(range(max(lo - RECENT, 0), lo))
    return sorted(q for q in qs if q < lo and q not in ins and q not in skip)


def fit(nat, tx, ins: tuple, at: tuple, cands: list[tuple], lam: float = 0.1, steps: int = STEPS) -> list[tuple]:
    """A variable's mask over cands (subcomponents at the carrier positions at): Cause donors the 64 alternative tokens at
    each input position (var_classes.alternatives), Iso donors 16 at each of iso_positions (a change there reaches the
    carrier but is not the variable's); the candidates kept."""
    import var_classes

    texts, iso = [], []
    for q in ins:
        for z in var_classes.alternatives(nat, tx.ids, q, int(tx.seed)):
            x = list(tx.ids)
            x[q] = z
            texts.append(x)
    for q in iso_positions(tx, ins, at):
        for z in var_classes.alternatives(nat, tx.ids, q, int(tx.seed), top=8, sampled=8):
            x = list(tx.ids)
            x[q] = z
            iso.append(x)
    P = Patcher(nat, tx.ids, tx.targets, cands)
    src = P.acts(texts + iso)
    with torch.no_grad():
        tgt = torch.cat([nat.reference(texts[s:s + BATCH], tx.targets) for s in range(0, len(texts), BATCH)])
        if iso:
            tgt = torch.cat([tgt, nat.reference([list(tx.ids)], tx.targets).expand(len(iso), -1, -1)])
    logit = train(P, src, tgt, lam, steps, seed=int(tx.seed))
    return [cands[j] for j in (logit > 0).nonzero().flatten().tolist()]
