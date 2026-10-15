"""Question-scoped neuron circuits on a Qwen3-style decoder (the lead's 10-15 direction): builder_fast's recipe on the
model's own MLP neurons, no decomposition.

Semantics (inframove's 10-15, neuron_hubs.py):
- unit: a strong activation of one MLP neuron (the down projection's input, act(gate) x up) at one position, with its
  pole (the sign of the value) and SwiGLU regime (the signs of act(gate) and of up);
- reference: the model with weak activations zeroed, every value under tau_l x that neuron's largest |value| over general
  chat set to 0, one fraction tau_l per layer, the most values zeroed with the mean KL from the model at most KL_TOL bits
  (calibrate, test T2; Neurons.save/load keep it in a file);
- removal: the unit's value zeroed at its position; massive activations (|value| at least MASSIVE x the layer's largest
  neuron maximum: sinks and delimiters) and the first position are never units and stay live;
- metric: the answer contrast at the answer position, log2 p(y) - log2 of the summed p of the ALTERNATIVES likeliest
  other tokens, y the reference's top token.

build(nr, ids, pos) on one prompt (the circuit: the necessary units and the sufficient ones, each unit's role recorded):
1. necessity's ranking: every candidate unit (strong at positions 1..pos, not massive) by |value x d contrast / d value|
   (one backward pass), the CANDIDATES largest then removed alone and measured (the contrast drop, and the side effect:
   the mean KL at SIDE positions before pos), ranked by measured drop, units whose side effect alone exceeds SIDE_MAX out;
2. necessity's size: the smallest prefix of that ranking on a log grid, then bracketing rounds, whose removal is
   necessary: the answer stops being the top token and the contrast falls by at least NEC_BITS, side effect <= SIDE_MAX;
3. necessity's trim: the prefix's units in TRIM_GROUPS runs, least effective first, each put back while the rest stays
   removed and kept out when the removal stays necessary;
4. sufficiency: CF_N counterfactual prompts, one context token replaced by one of the model's own likeliest alternatives
   there, those lowering the answer's contrast most (one per position, at least NEC_BITS); every candidate unit and every
   route (a head's output at a position, attention's route into it) from the first changed position on, set to its
   value on the prompt in each counterfactual's run (the patch), restores the contrast it lost; the patch is cut while it
   restores at least SUFF_FRAC (the least over the counterfactuals): positions, then (layer, position) blocks, then each
   MLP block and route alone (largest first), each dropped alone and the passing ones together on a log grid; then the
   kept MLP blocks' units: ranked by |value change x gradient| on the patched runs, the SUF_MEASURED largest patched alone
   and ranked by that (the rest by the gradient), the smallest restoring prefix, trimmed in TRIM_GROUPS groups;
5. met: the necessity set's removal is necessary and the circuit, patched into every counterfactual, restores SUFF_FRAC;
   reported: the circuit's removal, keep-only (every other candidate zeroed: the contrast and whether y stays top).
On 100 Qwen3-1.7B prompts every candidate patched restored >= 0.9 on 99 (every neuron at the answer position alone on 53):
the MLP neurons carry the question's change (attention's direct path is small), and one ranked prefix for both criteria
met 61 (12 of 100 on Qwen3-8B), its shortlist missing what the patch needs.
Every decision is a measured run; the gradient only orders the candidates."""
from __future__ import annotations

import contextlib
import math
import time

import torch

TAUS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)   # a layer's weak fraction tau_l (0 also)
KL_TOL = 0.1          # bits: the weak-zeroed reference's largest mean KL from the model (T2)
FIT = 80              # calibration texts whose KL the per-layer fractions are chosen on
PRICES = 48           # Lagrange prices (values zeroed per bit) whose per-layer choices are measured jointly
REFINE = 8            # rounds raising one layer's fraction a step on joint measurements
CAL_ROWS = slice(1000, 1200)   # general.jsonl's calibration chats (maxima; the first FIT also the KL); 100-299 held out (T2)
MASSIVE = 0.25        # a value at least this share of the layer's largest neuron maximum is massive (kept live, no unit)
ALTERNATIVES = 5
CANDIDATES = 512      # units measured alone
ROWS = 32             # runs a batch (16 on 1.7B: a median 15.5 s a prompt, most of it the single patches)
SIDE = 8              # positions before the answer whose predictions measure a removal's side effect
SIDE_MAX = 0.05       # bits: the largest side effect a circuit's removal (or a candidate alone) may have
NEC_BITS = 1.0        # the contrast drop a necessary removal reaches, beside the answer losing the top
ROUNDS = 3
TRIM_GROUPS = 8
RANKED_KEPT = 64      # the ranked units written beside the circuit, with their effects alone
SUFFICIENCY = True    # also require sufficiency by patching into counterfactual prompts (build's step 4)
CF_N = 2              # counterfactual prompts: one context token replaced by one of the model's own alternatives there
CF_ALTS = 4           # the alternatives tried per position (the model's likeliest other tokens at it)
SUFF_FRAC = 0.9       # the share of the contrast a counterfactual loses that patching the circuit's units back restores
SUF_MEASURED = 512    # sufficiency's units patched alone (measured), the largest by |value change x gradient|


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
                if self.thr is not None:
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

    def logits(self, ids, positions: list[int], edit=None, B: int = 1, capture=None, route=None) -> torch.Tensor:
        """Logits [B, len(positions), V] of B copies of ids under edit (and route)."""
        x = torch.as_tensor(ids, device=self.dev)
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with self.hooks(edit, capture, route):
            h = self.core.model(input_ids=x).last_hidden_state[:, positions]
        return self.core.lm_head(h).float()

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
            lp0 = torch.log_softmax(self.logits(ids, pos)[0], -1)
            for s0 in range(0, R, rows):
                f = fracs[s0:s0 + rows].float()
                l0 = None if live0 is None else torch.as_tensor(live0[s0:s0 + rows], dtype=torch.bool)
                cnt = torch.zeros(len(f), self.L, device=self.dev)

                def edit(i, h, f=f, l0=l0, cnt=cnt):
                    keep = h.float().abs() >= (f[:, i].to(h.device)[:, None] * self.mx[i].to(h.device)[None])[:, None, :]
                    if l0 is not None:
                        keep[l0.to(h.device), 0] = True
                    cnt[:, i] = keep.sum((1, 2)).float().to(cnt.device)
                    return h * keep.to(h.dtype)
                lp1 = torch.log_softmax(self.logits(ids, pos, edit=edit, B=len(f)), -1)
                d = (lp0.exp()[None] * (lp0[None] - lp1.to(lp0.device))).sum(-1).sum(-1) / math.log(2)
                kl[s0:s0 + len(f)] += d.double().cpu()
                agree[s0:s0 + len(f)] += (lp1.argmax(-1) == lp0.argmax(-1).to(lp1.device)[None]).sum(-1).double().cpu()
                strong[s0:s0 + len(f)] += cnt.double().cpu()
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


def _grid(n: int) -> list[int]:
    """1 to n on a log scale (two steps an octave), n included."""
    return sorted({min(n, int(round(2 ** (j / 2)))) for j in range(0, int(2 * math.log2(max(n, 1))) + 2)} | {n})


@torch.no_grad()
def counterfactuals(nr: Neurons, ids: list[int], pos: int, contrast, c0: float) -> list[tuple]:
    """Every counterfactual prompt of build's step 4: at each position 1..pos-1 the token replaced by each of the
    model's CF_ALTS likeliest next tokens there (other than the token itself), as (position, token, ids, the drop in the
    answer's contrast; contrast: [B, V] log p at pos -> [B] bits). build keeps the best per position, the CF_N largest;
    neuron_scorecard's held-out sufficiency draws the others."""
    lpre = torch.log_softmax(nr.logits(ids, list(range(pos))), -1)[0]   # [pos, V]: the prediction of each next token
    tries = []
    for q in range(1, pos):
        for z in lpre[q - 1].topk(CF_ALTS + 1).indices.tolist():
            if z != ids[q]:
                x2 = list(ids)
                x2[q] = z
                tries.append((q, z, x2))
    cq = []
    for s0 in range(0, len(tries), ROWS):
        part = tries[s0:s0 + ROWS]
        cq += contrast(torch.log_softmax(nr.logits(torch.tensor([x for _, _, x in part], device=nr.dev), [pos]), -1)[:, 0]).tolist()
    return [(q, z, x2, c0 - c) for (q, z, x2), c in zip(tries, cq)]


def build(nr: Neurons, ids: list[int], pos: int, log=None) -> dict:
    """The circuit and its record on one prompt (module docstring)."""
    t0 = time.perf_counter()
    rec = {"T": len(ids), "pos": pos, "tau": nr.tau}
    side = sorted({max(1, pos - 1 - k * max(1, (pos - 1) // SIDE)) for k in range(SIDE)} - {pos})
    allpos = side + [pos]
    cap = {}
    with torch.no_grad():
        lg = nr.logits(ids, allpos, capture=cap)[0]
    lp0 = torch.log_softmax(lg, -1)
    y = int(lp0[-1].argmax())
    alt = [int(z) for z in lp0[-1].topk(ALTERNATIVES + 1).indices.tolist() if z != y][:ALTERNATIVES]
    altt = torch.tensor(alt, device=nr.dev)

    def contrast(l):                                          # [B, V] log p at pos -> [B] bits
        return (l[:, y] - torch.logsumexp(l[:, altt], -1)) / math.log(2)
    c0 = float(contrast(lp0[-1:])[0])
    rec.update(answer=y, alternatives=alt, contrast=c0, p=float(lp0[-1, y].exp()))
    H = [h[0] for h in cap["h"]]                              # [T, D] per layer, weak-zeroed
    U = [u[0] for u in cap["up"]]
    O = [o[0] for o in cap["o"]]                              # [T, NH x hd] per layer: the head outputs
    P = []                                                    # [T, D] per layer: the candidate units
    for i in range(nr.L):
        s = (H[i] != 0) & (H[i].abs() < nr.massive[i])
        s[0] = False
        s[pos + 1:] = False
        P.append(s)

    def runs(sets: list) -> list[dict]:
        """Per set of units removed: the contrast, y on top, the side effect."""
        out = []
        for s0 in range(0, len(sets), ROWS):
            part = sets[s0:s0 + ROWS]
            with torch.no_grad():
                l = torch.log_softmax(nr.logits(ids, allpos, edit=_zero_edit(part), B=len(part)), -1)   # [B, P, V]
            c = contrast(l[:, -1])
            sd = (lp0[None, :-1].exp() * (lp0[None, :-1] - l[:, :-1])).sum(-1).mean(-1) / math.log(2) if side else torch.zeros(len(part))
            out += [{"contrast": float(c[b]), "top": int(l[b, -1].argmax()) == y, "side": float(sd[b])} for b in range(len(part))]
        return out

    def necessary(r):
        return (not r["top"]) and r["contrast"] <= c0 - NEC_BITS and r["side"] <= SIDE_MAX
    # 1. necessity's ranking: |value x gradient| over the candidates, then the largest measured alone
    m = [torch.ones(len(ids), nr.D, device=nr.dev, requires_grad=True) for _ in range(nr.L)]
    with torch.enable_grad():
        lq = torch.log_softmax(nr.logits(ids, [pos], edit=lambda i, h: h * m[i].to(h.dtype))[:, 0], -1)
        contrast(lq).sum().backward()
    cand, score = [], []
    for i in range(nr.L):
        t, n = P[i].nonzero(as_tuple=True)
        cand += [(i, int(a), int(b)) for a, b in zip(t.tolist(), n.tolist())]
        score.append(m[i].grad[t, n].abs())
    m = None                                                  # (the gradients' memory)
    score = torch.cat(score) if score else torch.zeros(0)
    rec["candidates"] = len(cand)
    short = [cand[k] for k in score.topk(min(CANDIDATES, len(cand))).indices.tolist()]
    alone = dict(zip(short, runs([[u] for u in short])))
    drop = {u: c0 - r["contrast"] for u, r in alone.items()}
    order = [u for u in sorted(short, key=lambda u: -drop[u]) if drop[u] > 0 and alone[u]["side"] <= SIDE_MAX]
    rec["ranked"] = len(order)
    # 2. necessity's size: log grid, then bracketing rounds
    at = {}

    def measure(ns):
        ns = [k for k in sorted(set(ns)) if k not in at and 0 < k <= len(order)]
        for k, r in zip(ns, runs([order[:k] for k in ns])):
            at[k] = (necessary(r), r)
    measure(_grid(len(order)))
    for _ in range(ROUNDS):
        met = sorted(k for k, v in at.items() if v[0])
        if not met:
            break
        hi = met[0]
        lo = max([k for k, v in at.items() if not v[0] and k < hi] + [0])
        if hi - lo <= 1:
            break
        measure([lo + max(1, (hi - lo) * (j + 1) // 5) for j in range(4)])
    met = sorted(k for k, v in at.items() if v[0])
    nec = order[:met[0]] if met else []
    # 3. necessity's trim: groups put back, least effective first, while the removal stays necessary
    if len(nec) > 1:
        K = max(1, -(-len(nec) // TRIM_GROUPS))
        for g in [nec[::-1][j:j + K] for j in range(0, len(nec), K)]:
            rest = [u for u in nec if u not in set(g)]
            if rest and necessary(runs([rest])[0]):
                nec = rest
    rec["seconds_necessity"] = round(time.perf_counter() - t0, 2)
    # 4. sufficiency: counterfactuals, then the patch eliminated from every candidate and route down to units and routes
    cfs, suf, routes, srec = ([], [], [], {}) if not SUFFICIENCY else _sufficiency(nr, ids, pos, H, O, P, contrast, c0)
    rec["counterfactuals"] = [{"pos": q, "token": z, "drop": d} for q, z, _, d in cfs]
    rec["sufficiency_search"] = srec
    rec["seconds_sufficiency"] = round(time.perf_counter() - t0, 2)
    innec, insuf = set(nec), set(suf)
    circ = nec + [u for u in suf if u not in innec]
    removed = runs([circ])[0] if circ else {"contrast": c0, "top": True, "side": 0.0}
    nec_removed = runs([nec])[0] if nec else None
    restored = srec["restore_of"]([circ], [routes])[0] if "restore_of" in srec else None
    keep = [P[i].clone() for i in range(nr.L)]               # keep-only: every other candidate zeroed
    for i, t, n in circ:
        keep[i][t, n] = False
    with torch.no_grad():
        lk = torch.log_softmax(nr.logits(ids, [pos], edit=lambda i, h: h * (~keep[i]).to(h.dtype))[:, 0], -1)
    kept = {"contrast": float(contrast(lk)[0]), "top": int(lk[0].argmax()) == y}
    rec.update(met=bool(nec) and (not cfs or restored >= SUFF_FRAC), necessity_met=bool(nec), size=len(circ),
               size_necessity=len(nec), size_sufficiency=len(suf), removed=removed, removed_necessity=nec_removed,
               restored=restored, sufficiency=kept, seconds=round(time.perf_counter() - t0, 2))
    srec.pop("restore_of", None)
    alone_s = srec.pop("alone", {})

    def node(i, t, n):
        return {"layer": i, "neuron": n, "pos": t, "pole": 1 if float(H[i][t, n]) > 0 else -1,
                "regime": [1 if float(H[i][t, n]) * float(U[i][t, n]) > 0 else -1, 1 if float(U[i][t, n]) > 0 else -1],
                "act": float(H[i][t, n]),
                "role": {(True, False): "necessity", (False, True): "sufficiency", (True, True): "both"}.get(((i, t, n) in innec, (i, t, n) in insuf)),
                "effect": drop.get((i, t, n)), "side": alone[(i, t, n)]["side"] if (i, t, n) in alone else None,
                "restores": alone_s.get((i, t, n))}
    rec["nodes"] = [node(*u) for u in circ]
    rec["routes"] = [{"layer": i, "head": hh, "pos": t} for i, hh, t in routes]
    rec["ranked_top"] = [node(*u) for u in order[:RANKED_KEPT]]   # necessity's next units by measured effect
    if log:
        log(f"  neurons: {len(nec)} necessary + {len(suf)} sufficient of {len(cand)} candidates, {len(routes)} routes, contrast {c0:.2f} -> "
            f"{removed['contrast']:.2f} bits removed (side {removed['side']:.3f}), restored {restored}, met {rec['met']}, "
            f"{rec['seconds']} s")
    return rec


def _sufficiency(nr: Neurons, ids: list[int], pos: int, H: list, O: list, P: list, contrast, c0: float):
    """build's step 4: the counterfactuals [(position, token, ids, drop)], the sufficient units [(layer, pos, neuron)], the
    routes [(layer, head, pos)] and the search's record (with "restore_of": sets of units and routes -> the least share
    restored over the counterfactuals, and "alone": each measured unit's share patched alone)."""
    t0 = time.perf_counter()
    best = {}                                                # counterfactuals: one token replaced by one of the
    for q, z, x2, d in counterfactuals(nr, ids, pos, contrast, c0):   # model's own likeliest alternatives there, those
        if d > best.get(q, (None, None, 0.0))[2]:            # lowering the answer's contrast most (one per position)
            best[q] = (z, x2, d)
    cfs = sorted(((q, z, x2, d) for q, (z, x2, d) in best.items()), key=lambda r: -r[3])[:CF_N]
    cfs = [r for r in cfs if r[3] >= NEC_BITS]
    if not cfs:
        return cfs, [], [], {}
    T, L, NH, dev = len(ids), nr.L, nr.NH, nr.dev

    def restore(rows: list) -> list[float]:
        """Per row (fn(i) -> ([T, D] units, [T, NH] routes) on layer i): the least share of a counterfactual's contrast loss
        restored with those units and head outputs set to their values on the prompt."""
        got = []
        for q, z, x2, d in cfs:
            out = []
            for s0 in range(0, len(rows), ROWS):
                part = rows[s0:s0 + ROWS]
                mk = [[fn(i) for i in range(L)] for fn in part]

                def edit(i, h, mk=mk):
                    m = torch.stack([r[i][0] for r in mk])
                    return torch.where(m, H[i][None].to(h.dtype), h)

                def route(i, a, mk=mk):
                    m = torch.stack([r[i][1] for r in mk]).repeat_interleave(nr.hd, -1)
                    return torch.where(m, O[i][None].to(a.dtype), a)
                with torch.no_grad():
                    l = torch.log_softmax(nr.logits(x2, [pos], edit=edit, route=route, B=len(part))[:, 0], -1)
                out += [(c - (c0 - d)) / d for c in contrast(l).tolist()]
            got.append(out)
        return [min(v) for v in zip(*got)]
    no_r = torch.zeros(T, NH, dtype=torch.bool, device=dev)

    def restore_of(sets: list, routes: list | None = None) -> list[float]:
        """restore of sets of units (each with routes[b], default none)."""
        rows = []
        for b, s in enumerate(sets):
            mk = [[torch.zeros(T, nr.D, dtype=torch.bool, device=dev), no_r] for _ in range(L)]
            for i, t, n in s:
                mk[i][0][t, n] = True
            for i, hh, t in (routes[b] if routes else []):
                if mk[i][1] is no_r:
                    mk[i][1] = no_r.clone()
                mk[i][1][t, hh] = True
            rows.append(lambda i, mk=mk: mk[i])
        return restore(rows)
    qmin = min(q for q, _, _, _ in cfs)
    # elements: (layer, position) MLP blocks (every candidate unit there; cost: their count) and (layer, head, position)
    # routes (cost 1), at positions qmin..pos; a row keeps a bool over them
    el = [(0, i, t, 0, int(P[i][t].sum())) for t in range(qmin, pos + 1) for i in range(L) if bool(P[i][t].any())]
    el += [(1, i, t, hh, 1) for t in range(qmin, pos + 1) for i in range(L) for hh in range(NH)]
    E = torch.tensor([e[:4] for e in el], device=dev)
    cost = torch.tensor([e[4] for e in el], dtype=torch.float64, device=dev)
    isn, isr = E[:, 0] == 0, E[:, 0] == 1

    def row(keep: torch.Tensor):
        bm = torch.zeros(L, T, dtype=torch.bool, device=dev)
        k = keep & isn
        bm[E[k, 1], E[k, 2]] = True
        rm = torch.zeros(L, T, NH, dtype=torch.bool, device=dev)
        k = keep & isr
        rm[E[k, 1], E[k, 2], E[k, 3]] = True
        return lambda i: (P[i] & bm[i][:, None], rm[i])

    def eliminate(keep: torch.Tensor, groups: list, by_cost: bool) -> torch.Tensor:
        """Groups of elements (index tensors) dropped from the patch while it restores SUFF_FRAC: each kept group dropped
        alone, then those passing alone dropped together (largest cost first with by_cost, else what stays restored
        most), the largest passing count on a log grid with bracketing; ROUNDS times."""
        for _ in range(ROUNDS):
            live = [g for g in groups if bool(keep[g].any())]
            if len(live) <= 1:
                break
            trial = []
            for g in live:
                k = keep.clone()
                k[g] = False
                trial.append(k)
            r = restore([row(k) for k in trial])
            ok = [(v, j) for j, v in enumerate(r) if v >= SUFF_FRAC]
            if not ok:
                break
            ok.sort(key=lambda vj: (-float(cost[live[vj[1]]][keep[live[vj[1]]]].sum()), -vj[0]) if by_cost else (-vj[0],))
            ordr = [live[j] for _, j in ok]
            at = {1: True}

            def tryj(js):
                js = [j for j in sorted(set(js)) if j not in at and 0 < j <= len(ordr)]
                ks = []
                for j in js:
                    k = keep.clone()
                    for g in ordr[:j]:
                        k[g] = False
                    ks.append(k)
                for j, v in zip(js, restore([row(k) for k in ks])):
                    at[j] = v >= SUFF_FRAC
            tryj(_grid(len(ordr)))
            for _ in range(ROUNDS):
                lo = max(j for j, v in at.items() if v)
                hi = min([j for j, v in at.items() if not v and j > lo] + [len(ordr) + 1])
                if hi - lo <= 1:
                    break
                tryj([lo + max(1, (hi - lo) * (j + 1) // 5) for j in range(4)])
            keep = keep.clone()
            for g in ordr[:max(j for j, v in at.items() if v)]:
                keep[g] = False
        return keep
    keep = torch.ones(len(el), dtype=torch.bool, device=dev)
    full = restore([row(keep)])[0]
    srec = {"full": full, "restore_of": restore_of}
    if full < SUFF_FRAC:                                     # every candidate and route patched does not restore enough
        srec["seconds"] = round(time.perf_counter() - t0, 2)
        return cfs, [], [], srec
    # positions, then (layer, position) blocks of both kinds, then each kept MLP block and route alone
    ix = torch.arange(len(el), device=dev)
    keep = eliminate(keep, [ix[E[:, 2] == t] for t in range(qmin, pos + 1)], False)
    keep = eliminate(keep, [ix[(E[:, 1] == i) & (E[:, 2] == t)] for t in range(qmin, pos + 1) for i in range(L)], False)
    keep = eliminate(keep, [ix[j:j + 1] for j in keep.nonzero().flatten().tolist()], True)
    routes = [(int(E[j, 1]), int(E[j, 3]), int(E[j, 2])) for j in (keep & isr).nonzero().flatten().tolist()]
    fn = row(keep)
    pool = [fn(i)[0] for i in range(L)]
    rmask = [fn(i)[1] for i in range(L)]
    srec.update(positions=sorted({int(E[j, 2]) for j in keep.nonzero().flatten().tolist()}), routes=len(routes),
                blocks=[(int(E[j, 1]), int(E[j, 2])) for j in (keep & isn).nonzero().flatten().tolist()],
                pool=int(sum(int(p.sum()) for p in pool)))
    if srec["pool"] == 0:
        srec["seconds"] = round(time.perf_counter() - t0, 2)
        return cfs, [], routes, srec
    # units: |value change x gradient| on each counterfactual's run with the pool and routes patched, the largest
    # SUF_MEASURED patched alone with the routes (measured) and ranked by that, the rest by the gradient; the smallest
    # prefix restoring SUFF_FRAC, trimmed in TRIM_GROUPS groups
    sc = [torch.zeros(T, nr.D, device=dev) for _ in range(L)]
    for q, z, x2, d in cfs:
        capc = {}
        with torch.no_grad():
            nr.logits(x2, [pos], capture=capc)
        hc = [h[0] for h in capc["h"]]
        mc = [torch.ones(T, nr.D, device=dev, requires_grad=True) for _ in range(L)]
        with torch.enable_grad():
            l = torch.log_softmax(nr.logits(x2, [pos], edit=lambda i, h: torch.where(pool[i], H[i].to(h.dtype), h) * mc[i].to(h.dtype),
                                            route=lambda i, a: torch.where(rmask[i].repeat_interleave(nr.hd, -1), O[i].to(a.dtype), a))[:, 0], -1)
            contrast(l).sum().backward()
        for i in range(L):
            g = mc[i].grad * (H[i] - hc[i]).float() / torch.where(H[i] != 0, H[i], torch.ones_like(H[i])).float()
            sc[i] = torch.maximum(sc[i], torch.where(pool[i], g.abs(), torch.zeros_like(g)))
    units, vals = [], []
    for i in range(L):
        t, n = pool[i].nonzero(as_tuple=True)
        units += [(i, int(a), int(b)) for a, b in zip(t.tolist(), n.tolist())]
        vals.append(sc[i][t, n])
    byg = [units[k] for k in torch.cat(vals).argsort(descending=True).tolist()]
    top = byg[:SUF_MEASURED]
    alone = dict(zip(top, restore_of([[u] for u in top], [routes] * len(top))))
    ranked = sorted(top, key=lambda u: -alone[u]) + byg[SUF_MEASURED:]
    rank = [torch.full((T, nr.D), len(ranked), dtype=torch.int32, device=dev) for _ in range(L)]
    for r, (i, t, n) in enumerate(ranked):
        rank[i][t, n] = r
    dropped = [torch.zeros(T, nr.D, dtype=torch.bool, device=dev) for _ in range(L)]

    def prefix(ks: list[int], drop=None) -> list[float]:
        return restore([lambda i, k=k, b=b: ((rank[i] < k) & ~(dropped[i] if drop is None else drop[b][i]), rmask[i])
                        for b, k in enumerate(ks)])
    at = {}

    def measure(ks):
        ks = [k for k in sorted(set(ks)) if k not in at and 0 <= k <= len(ranked)]
        for k, v in zip(ks, prefix(ks)):
            at[k] = v >= SUFF_FRAC
    measure([0] + _grid(len(ranked)))
    for _ in range(ROUNDS):
        met = sorted(k for k, v in at.items() if v)
        if not met:
            break
        hi = met[0]
        lo = max([k for k, v in at.items() if not v and k < hi] + [-1])
        if hi - lo <= 1:
            break
        measure([lo + max(1, (hi - lo) * (j + 1) // 5) for j in range(4)])
    met = sorted(k for k, v in at.items() if v)
    k = met[0] if met else len(ranked)
    K = max(1, -(-k // TRIM_GROUPS))
    for a in range(k - K, -K, -K) if k > 1 else []:
        trial = [d.clone() for d in dropped]
        for i in range(L):
            trial[i] |= (rank[i] >= max(a, 0)) & (rank[i] < a + K)
        if prefix([k], [trial])[0] >= SUFF_FRAC:
            dropped = trial
    suf = [u for u in ranked[:k] if not bool(dropped[u[0]][u[1], u[2]])]
    srec.update(units=len(suf), seconds=round(time.perf_counter() - t0, 2), alone=alone)
    return cfs, suf, routes, srec


def chat(tok, user: str) -> list[int]:
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"]


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="question-scoped neuron circuits: OUT/<id>.json per prompt")
    ap.add_argument("prompts", type=Path, help="JSONL: {id, user} (the circuit of the model's first answer token) or {id, "
                                               "user, ids} (the circuit of the token after ids: a chat with an answer prefix)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--general", type=Path, required=True, help="JSONL {user, answer}: the weak-zeroing calibration's chats")
    ap.add_argument("--thr", type=Path, default=None, help="the reference's file (Neurons.save): read, else calibrated into it")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rev", default="dev")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="the model's weights (8B and up: bfloat16)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    nr = Neurons(model, tok)
    if a.thr and a.thr.exists():
        nr.load(a.thr)
        cal = {"tau": nr.tau, "file": str(a.thr)}
    else:
        gen = [json.loads(x) for x in a.general.read_text().splitlines()]
        seqs = [tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["answer"]}],
                                            tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"][:256]
                for r in gen[CAL_ROWS]]
        cal = nr.calibrate(seqs, log=print)
        if a.thr:
            nr.save(a.thr, calibration=cal)
    print("calibration", json.dumps({k: cal[k] for k in ("tau", "kl", "strong_per_token", "file") if k in cal}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in a.prompts.read_text().splitlines() if x.strip()]
    first = True
    for r in rows[a.offset::a.stride]:
        if (a.out / f"{r['id']}.json").exists():
            continue
        ids = r.get("ids") or chat(tok, r["user"])
        if first:                                             # (a throwaway build: CUDA's first kernels)
            build(nr, ids, len(ids) - 1)
            first = False
        rec = build(nr, ids, len(ids) - 1, log=print)
        rec.update(id=r["id"], user=r["user"], answer_text=tok.decode([rec["answer"]]), units="neurons",
                   builder=f"neuron_builder@{a.rev}", model=a.model, adapter=a.adapter, calibration=cal)
        (a.out / f"{r['id']}.json").write_text(json.dumps(rec))
        print(f"{r['id']}: {rec['size']} neurons, {rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
