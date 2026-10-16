"""The model's MLP neurons on a Qwen3-style decoder (the lead's 10-15 direction): the weak-zeroed reference and neuron
edits. Circuits are circuit_finder's (the patching pair; the lead's cutover removed this module's builder, build()).
- Neurons: an HF decoder (a PeftModel with its adapter on) with every layer's neurons (the down projection's input,
  act(gate) x up) open to batched edits and captures; the weak-zeroed reference after calibrate() or load(): every value
  under tau_l x that neuron's largest |value| over general chat set to 0, one fraction tau_l per layer, the most values
  zeroed with the mean KL from the model at most KL_TOL bits (test T2), as each input's fixed mask (FIXED: the values
  weak in that input's own unedited run, never re-thresholded under edits); massive values (at least MASSIVE x the
  layer's largest neuron maximum) stay live.
- pole_edit: a neuron's positive or negative values (or the whole neuron) zeroed at every position, per batch row;
  _zero_edit: neurons zeroed at given positions, per batch row.
- chat: a user turn with the model's turn opened, thinking off.
"""
from __future__ import annotations

import contextlib
import copy
import math

import torch


TAUS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)   # a layer's weak fraction tau_l (0 also)
KL_TOL = 0.1          # bits: the weak-zeroed reference's largest mean KL from the model (T2)
FIT = 80              # calibration texts whose KL the per-layer fractions are chosen on
PRICES = 48           # Lagrange prices (values zeroed per bit) whose per-layer choices are measured jointly
REFINE = 8            # rounds raising one layer's fraction a step on joint measurements
CAL_ROWS = slice(1000, 1200)   # general.jsonl's calibration chats (maxima; the first FIT also the KL); 100-299 held out (T2)
MASSIVE = 0.25        # a value at least this share of the layer's largest neuron maximum is massive (kept live, no unit)
ALTERNATIVES = 5
FIXED = True          # the weak zeroing as a fixed mask per input: the values weak in that input's own unedited run are
# zeroed, never re-thresholded under edits (re-thresholding makes the reference discontinuous: on Qwen3-0.6B a 1e-6
# relative change of every MLP value moves the answer's logits by 0.51 against the model's 0.0001; the fixed mask 0.00006)
MASKS_KEPT = 8        # inputs whose fixed masks stay cached (the prompt and its counterfactuals)
PREFIX = False        # runs whose edits all lie at or after a position reuse the positions before it (float32 only: in
# bfloat16 a run's numbers move with its shapes, as with its batch size). Off: on 12 float32 Qwen3-8B prompts 5 records
# searched to >= 0.9 restored measured 0.84-0.90 in the record's full run, and the padded edits ate the time saved. A
# cached run equals the full one to 0.008 logits without the weak zeroing; with it, float noise flips near-threshold
# values on or off (the reference is discontinuous), up to 0.2 bits of contrast on 8B
ROWS = 32             # runs a batch, every batch padded to it (bfloat16 Qwen3-8B moves the answer's contrast up to 2.3
# bits between batch sizes 1 and 32, bit-identical at one size whatever the other rows hold)
SIDE = 8              # positions before the answer whose predictions measure a removal's side effect


class Neurons:
    """An HF decoder (Qwen3; a PeftModel with its adapter on) with every layer's MLP neurons, the down projection's input,
    open to batched edits; the weak-zeroed reference after calibrate()."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = list(self.core.model.layers)
        cfg = self.core.config
        self.L, self.D = cfg.num_hidden_layers, cfg.intermediate_size
        self.NH = cfg.num_attention_heads                     # heads (routes): each layer's o_proj input in NH slices
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.NH
        self.dev = next(self.core.parameters()).device
        self.thr = None          # [L, D] weak threshold per neuron
        self.mx = None           # [L, D] each neuron's largest |value| on the calibration texts
        self.massive = None      # [L] massive value per layer
        self.tau = None          # [L] each layer's weak fraction
        self.fixed = FIXED       # the weak zeroing as each input's fixed mask (FIXED)
        self._masks = {}         # input ids -> [L] [T, D] bool: its unedited run's strong values
        self._cur = None         # [L] [B or 1, T, D] the masks of the run in progress

    @contextlib.contextmanager
    def hooks(self, edit=None, capture: dict | None = None, route=None):
        """edit(i, h) -> h on each layer's neuron values [B, T, D] after the weak zeroing; route(i, a) -> a on each layer's
        head outputs (o_proj's input [B, T, NH x hd]); capture: {"h": [], "up": [], "o": []}."""
        hs = []
        for i, layer in enumerate(self.layers):
            if route is not None or capture is not None:
                def opre(mod, args, i=i):
                    a = args[0] if route is None else route(i, args[0])
                    if capture is not None:
                        capture.setdefault("o", []).append(a.detach())
                    return (a,) + tuple(args[1:])
                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(opre))
            def pre(mod, args, i=i):
                h = args[0]
                if self._cur is not None:
                    h = h * self._cur[i].to(h.dtype)
                elif self.thr is not None:
                    h = h * (h.abs() >= self.thr[i].to(h.device, h.dtype)).to(h.dtype)
                if edit is not None:
                    h = edit(i, h)
                if capture is not None:
                    capture.setdefault("h", []).append(h.detach())
                return (h,) + tuple(args[1:])
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            if capture is not None:
                hs.append(layer.mlp.up_proj.register_forward_hook(lambda mod, args, out: capture.setdefault("up", []).append(out.detach())))
        try:
            yield
        finally:
            for h in hs:
                h.remove()

    def logits(self, ids, positions: list[int], edit=None, B: int = 1, capture=None, route=None, cache=None) -> torch.Tensor:
        """Logits [B, len(positions), V] of B copies of ids under edit (and route). With cache (prefix(): positions
        0..start-1 already run) only the positions from start on run: edit and route still see every position (those
        before start as zeros, their edits ignored), positions must be start or later, no capture."""
        x = torch.as_tensor(ids, device=self.dev)
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        masks = self.masks(x)
        if cache is None:
            self._cur = masks
            try:
                with self.hooks(edit, capture, route):
                    h = self.core.model(input_ids=x).last_hidden_state[:, positions]
            finally:
                self._cur = None
            return self.core.lm_head(h).float()
        s, kv = cache
        self._cur = None if masks is None else [m[:, s:] for m in masks]

        def pad(f):
            return None if f is None else (lambda i, a: f(i, torch.cat([a.new_zeros(a.shape[0], s, a.shape[2]), a], 1))[:, s:])
        T, dt = x.shape[1], next(self.core.parameters()).dtype   # (the causal mask given whole: transformers 4.53 on
        mask = torch.full((T - s, T), torch.finfo(dt).min, dtype=dt, device=self.dev).triu(s + 1)   # torch 2.4 failed to
        try:
            with self.hooks(pad(edit), None, pad(route)):                                           # build it for a cache)
                h = self.core.model(input_ids=x[:, s:], past_key_values=copy.deepcopy(kv), use_cache=True,
                                    attention_mask=mask[None, None].expand(x.shape[0], 1, -1, -1)).last_hidden_state
        finally:
            self._cur = None
        return self.core.lm_head(h[:, [p - s for p in positions]]).float()

    @torch.no_grad()
    def masks(self, x: torch.Tensor):
        """With fixed (and a reference), per layer [B or 1, T, D] bool: each row's strong values in its own unedited
        weak-zeroed run (cached for MASKS_KEPT inputs); else None."""
        if not self.fixed or self.thr is None:
            return None
        rows = [tuple(r) for r in x.tolist()]
        got = {}                                     # (this batch's masks: the cache may hold fewer inputs than it has)
        for r in dict.fromkeys(rows):
            if r not in self._masks:
                cap = {}
                with self.hooks(None, cap):
                    self.core.model(input_ids=torch.tensor([r], device=self.dev))
                self._masks[r] = [h[0] != 0 for h in cap["h"]]
            got[r] = self._masks[r]
        for r in [r for r in self._masks if r not in got][:max(0, len(self._masks) - MASKS_KEPT)]:
            self._masks.pop(r)
        if len(got) == 1:
            return [m[None] for m in got[rows[0]]]
        return [torch.stack([got[r][i] for r in rows]) for i in range(self.L)]

    @torch.no_grad()
    def prefix(self, ids, start: int, edit=None, route=None, B: int = 1):
        """The key/value cache of positions 0..start-1 of B copies of ids under edit (and route, each seeing those
        positions only), for logits(cache=)."""
        x = torch.as_tensor(ids, device=self.dev)
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        masks = self.masks(x)
        self._cur = None if masks is None else [m[:, :start] for m in masks]
        try:
            with self.hooks(edit, None, route):
                return start, self.core.model(input_ids=x[:, :start], use_cache=True).past_key_values
        finally:
            self._cur = None

    @torch.no_grad()
    def maxima(self, seqs: list[list[int]]) -> torch.Tensor:
        """[L, D]: each neuron's largest |value| over seqs in the model."""
        thr, self.thr = self.thr, None
        mx = torch.zeros(self.L, self.D, device=self.dev)
        for ids in seqs:
            cap = {}
            self.logits(ids, [len(ids) - 1], capture=cap)
            for i, h in enumerate(cap["h"]):
                mx[i] = torch.maximum(mx[i], h[0].abs().max(0).values.float().to(self.dev))
        self.thr = thr
        return mx

    @torch.no_grad()
    def weak_runs(self, seqs: list[list[int]], fracs: torch.Tensor, live0=None, rows: int = 16) -> tuple:
        """Per row of fracs [R, L] (each layer's weak fraction of self.mx), the model with every value under its threshold
        zeroed (live0 [R] bool: the first position, the attention sink, kept whole): its mean KL (bits) from the model
        over every position of seqs [R], the strong values per token per layer [R, L], and the share of positions whose
        likeliest next token stays the model's [R]."""
        thr, self.thr = self.thr, None
        R = len(fracs)
        kl = torch.zeros(R, dtype=torch.float64)
        strong = torch.zeros(R, self.L, dtype=torch.float64)
        agree = torch.zeros(R, dtype=torch.float64)
        n = 0
        for ids in seqs:
            pos = list(range(len(ids)))
            lp0 = torch.log_softmax(self.logits(ids, pos, B=rows)[0], -1)        # at the batch size of the runs below
            for s0 in range(0, R, rows):                                         # (bfloat16 kernels change with it)
                f = fracs[s0:s0 + rows].float()
                f = torch.cat([f, f[:1].expand(rows - len(f), -1)]) if len(f) < rows else f
                l0 = None if live0 is None else torch.as_tensor(list(live0[s0:s0 + rows]) + [False] * (len(f) - len(live0[s0:s0 + rows])), dtype=torch.bool)
                cnt = torch.zeros(len(f), self.L, device=self.dev)

                def edit(i, h, f=f, l0=l0, cnt=cnt):
                    keep = h.float().abs() >= (f[:, i].to(h.device)[:, None] * self.mx[i].to(h.device)[None])[:, None, :]
                    if l0 is not None:
                        keep[l0.to(h.device), 0] = True
                    cnt[:, i] = keep.sum((1, 2)).float().to(cnt.device)
                    return h * keep.to(h.dtype)
                n_r = min(rows, R - s0)
                lp1 = torch.log_softmax(self.logits(ids, pos, edit=edit, B=len(f)), -1)[:n_r]
                d = (lp0.exp()[None] * (lp0[None] - lp1.to(lp0.device))).sum(-1).sum(-1) / math.log(2)
                kl[s0:s0 + n_r] += d.double().cpu()
                agree[s0:s0 + n_r] += (lp1.argmax(-1) == lp0.argmax(-1).to(lp1.device)[None]).sum(-1).double().cpu()
                strong[s0:s0 + n_r] += cnt[:n_r].double().cpu()
                del lp1
            n += len(ids)
        self.thr = thr
        return kl / n, strong / n, agree / n

    @torch.no_grad()
    def calibrate(self, seqs: list[list[int]], fit: list[list[int]] | None = None, taus=TAUS, kl_tol: float = KL_TOL,
                  prices: int = PRICES, refine: int = REFINE, rows: int = 16, log=None) -> dict:
        """The weak-zeroed reference (T2): each neuron's largest |value| over seqs and one weak fraction per layer.
        1. each layer alone weak-zeroed at each fraction in taus, its KL on fit (default seqs[:FIT]) and the values it
           zeroes; 2. per Lagrange price (values zeroed per bit, PRICES on a log grid) every layer's fraction maximising
           zeroed - price x KL alone, each such choice measured jointly on fit, the one zeroing most within kl_tol kept;
           3. REFINE rounds, each raising by one step the layer whose joint measurement zeroes most per bit, within kl_tol.
        Every choice is a joint measurement; the single-layer table only proposes."""
        fit = seqs[:FIT] if fit is None else fit
        self.thr = None
        self.mx = self.maxima(seqs)
        self.massive = MASSIVE * self.mx.max(1).values
        L, G = self.L, len(taus) + 1
        grid = torch.tensor((0.0,) + tuple(taus), dtype=torch.float64)
        fr = torch.zeros(L * len(taus), L, dtype=torch.float64)
        for i in range(L):
            fr[i * len(taus):(i + 1) * len(taus), i] = grid[1:]
        kl1, st1, _ = self.weak_runs(fit, fr, rows=rows)
        cost = torch.cat([torch.zeros(L, 1, dtype=torch.float64), kl1.view(L, len(taus)).clamp(min=0)], 1).cummax(1).values
        own = st1.view(L, len(taus), L).diagonal(dim1=0, dim2=2).T                      # [L, G-1] strong at the layer itself
        gone = torch.cat([torch.zeros(L, 1, dtype=torch.float64), self.D - own], 1)    # [L, G] values zeroed per token
        if log:
            log(f"calibrate: single-layer table on {len(fit)} texts done")

        steps = []
        for lam in torch.logspace(1, 9, prices, dtype=torch.float64):
            g = (gone - lam * cost).argmax(1)
            if not any(bool((g == h).all()) for h in steps):
                steps.append(g)
        kls, sts, _ = self.weak_runs(fit, torch.stack([grid[g] for g in steps]), rows=rows)
        ok = [k for k in range(len(steps)) if kls[k] <= kl_tol]
        k0 = min(ok, key=lambda k: float(sts[k].sum())) if ok else None
        g = steps[k0] if ok else torch.zeros(L, dtype=torch.long)
        cur_kl, cur_st = (float(kls[k0]), float(sts[k0].sum())) if ok else (0.0, float(L * self.D))
        priced = [{"tau": [float(grid[x]) for x in h], "kl": float(k), "strong_per_token": float(s.sum())}
                  for h, k, s in zip(steps, kls, sts)]
        moves = []
        for _ in range(refine if ok else 0):
            cand = [i for i in range(L) if g[i] < G - 1]
            if not cand:
                break
            trial = torch.stack([grid[g + torch.nn.functional.one_hot(torch.tensor(i), L)] for i in cand])
            k2, s2, _ = self.weak_runs(fit, trial, rows=rows)
            best = None
            for j, i in enumerate(cand):
                if k2[j] <= kl_tol:
                    gain = cur_st - float(s2[j].sum())
                    rate = gain / max(float(k2[j]) - cur_kl, 1e-6)
                    if gain > 0 and (best is None or rate > best[0]):
                        best = (rate, i, float(k2[j]), float(s2[j].sum()))
            if best is None:
                break
            g = g.clone()
            g[best[1]] += 1
            cur_kl, cur_st = best[2], best[3]
            moves.append({"layer": best[1], "tau": float(grid[g[best[1]]]), "kl": cur_kl, "strong_per_token": cur_st})
            if log:
                log(f"calibrate: layer {best[1]} -> {float(grid[g[best[1]]])}, kl {cur_kl:.4f}, strong {cur_st:.0f}")
        self.tau = [float(grid[x]) for x in g]
        self.thr = torch.tensor(self.tau, device=self.dev, dtype=torch.float32)[:, None] * self.mx
        return {"tau": self.tau, "kl": cur_kl, "strong_per_token": cur_st, "fit_texts": len(fit), "taus": list(taus),
                "single": {"kl": cost.tolist(), "zeroed": gone.tolist()}, "priced": priced, "refined": moves}

    def save(self, path, **extra):
        """The reference in a file: {"thr" [L, D], "max" [L, D], "tau" [L], ...} (neuron_scorecard reads it as a thr file)."""
        torch.save({"thr": self.thr.cpu(), "max": self.mx.cpu(), "tau": list(self.tau), **extra}, path)

    def load(self, path):
        bg = torch.load(path, map_location="cpu")
        self.mx, self.tau = bg["max"].to(self.dev), bg["tau"]
        self.thr = bg["thr"].to(self.dev)
        self.massive = MASSIVE * self.mx.max(1).values
        return bg


def _zero_edit(sets: list, base=None):
    """edit zeroing each row's units [(layer, pos, neuron)], per layer one index_put; base(i, h) applied first."""
    by = {}
    for b, s in enumerate(sets):
        for i, t, n in s:
            by.setdefault(i, ([], [], []))
            by[i][0].append(b)
            by[i][1].append(t)
            by[i][2].append(n)
    idx = {i: tuple(torch.tensor(v) for v in vs) for i, vs in by.items()}

    def edit(i, h):
        if base is not None:
            h = base(i, h)
        if i in idx:
            b, t, n = (v.to(h.device) for v in idx[i])
            h = h.index_put((b, t, n), torch.zeros((), dtype=h.dtype, device=h.device))
        return h
    return edit


def pole_edit(sets: list, base=None):
    """edit zeroing each row's units [(layer, neuron, pole)] at every position: pole 1 the neuron's positive values, -1
    its negative ones, 0 the whole neuron (an edit of the model: T3, the essay's neuron removals); base(i, h) first."""
    by = {}
    for b, s in enumerate(sets):
        for i, n, p in s:
            zp, zn = by.setdefault(i, {}).get((b, n), (False, False))
            by[i][(b, n)] = (zp or p >= 0, zn or p <= 0)
    idx = {i: tuple(torch.tensor(v) for v in ([k[0] for k in d], [k[1] for k in d], [z[0] for z in d.values()],
                                              [z[1] for z in d.values()])) for i, d in by.items()}

    def edit(i, h):
        if base is not None:
            h = base(i, h)
        if i not in idx:
            return h
        b, n, zp, zn = (v.to(h.device) for v in idx[i])
        v = h[b, :, n]                                                           # [K, T]
        h = h.clone()
        h[b, :, n] = v.masked_fill(((v > 0) & zp[:, None]) | ((v < 0) & zn[:, None]), 0)
        return h
    return edit


def chat(tok, user: str) -> list[int]:
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"]
