"""Question-scoped neuron circuits on a Qwen3-style decoder (the lead's 10-15 direction): builder_fast's recipe on the
model's own MLP neurons, no decomposition.

Semantics (inframove's 10-15, neuron_hubs.py):
- unit: a strong activation of one MLP neuron (the down projection's input, act(gate) x up) at one position, with its
  pole (the sign of the value) and SwiGLU regime (the signs of act(gate) and of up);
- reference: the model with weak activations zeroed, every value under tau x that neuron's largest |value| over general
  chat set to 0, tau the largest of TAUS whose mean KL from the model stays at most KL_TOL bits (calibrate, test T2);
- removal: the unit's value zeroed at its position; massive activations (|value| at least MASSIVE x the layer's largest
  neuron maximum: sinks and delimiters) and the first position are never units and stay live;
- metric: the answer contrast at the answer position, log2 p(y) - log2 of the summed p of the ALTERNATIVES likeliest
  other tokens, y the reference's top token.

build(nr, ids, pos) on one prompt:
1. ranking: every candidate unit (strong at positions 1..pos, not massive) by |value x d contrast / d value| (one
   backward pass), the CANDIDATES largest then removed alone and measured (the contrast drop, and the side effect: the
   mean KL at SIDE positions before pos), ranked by measured drop, units whose side effect alone exceeds SIDE_MAX left out;
2. size: the smallest prefix of that ranking on a log grid, then bracketing rounds, whose removal is necessary: the
   answer stops being the top token and the contrast falls by at least NEC_BITS, with the side effect at most SIDE_MAX;
3. trim: the prefix's units in TRIM_GROUPS runs, least effective first, each put back while the rest stays removed and
   kept out of the circuit when the removal stays necessary;
4. reported: sufficiency (only the circuit's units live among the candidates: the contrast and whether y stays top).
With SUFFICIENCY (step 0, before the size search): CF_N counterfactual prompts, one context token replaced by one of
the model's own likeliest alternatives there, those lowering the answer's contrast most (one per position, at least
NEC_BITS); each adds to the shortlist the units its change runs through (|value difference x gradient| on its run), every
shortlisted unit is also patched alone (set to its value on the prompt in the counterfactual's run), the ranking sums the
two measured shares (necessity's drop, sufficiency's restored share), and the size search and trim also require the
circuit, patched into every counterfactual, to restore at least SUFF_FRAC of the contrast it lost (the 10-15 brief's
sufficiency on the contrast: a necessity-minimal set alone was a median 2 neurons at the answer position on 100 Qwen3-1.7B
prompts, the question's tokens nowhere in it).
Every decision is a measured run; the gradient only orders the candidates."""
from __future__ import annotations

import contextlib
import math
import time

import torch

TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
KL_TOL = 0.1          # bits: the weak-zeroed reference's largest mean KL from the model (T2)
MASSIVE = 0.25        # a value at least this share of the layer's largest neuron maximum is massive (kept live, no unit)
ALTERNATIVES = 5
CANDIDATES = 512      # units measured alone
ROWS = 16             # runs a batch
SIDE = 8              # positions before the answer whose predictions measure a removal's side effect
SIDE_MAX = 0.05       # bits: the largest side effect a circuit's removal (or a candidate alone) may have
NEC_BITS = 1.0        # the contrast drop a necessary removal reaches, beside the answer losing the top
ROUNDS = 3
TRIM_GROUPS = 8
RANKED_KEPT = 64      # the ranked units written beside the circuit, with their effects alone
SUFFICIENCY = True    # also require sufficiency by patching into counterfactual prompts (build's step 0)
CF_N = 2              # counterfactual prompts: one context token replaced by one of the model's own alternatives there
CF_ALTS = 4           # the alternatives tried per position (the model's likeliest other tokens at it)
SUFF_FRAC = 0.9       # the share of the contrast a counterfactual loses that patching the circuit's units back restores


class Neurons:
    """An HF decoder (Qwen3; a PeftModel with its adapter on) with every layer's MLP neurons, the down projection's input,
    open to batched edits; the weak-zeroed reference after calibrate()."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = list(self.core.model.layers)
        cfg = self.core.config
        self.L, self.D = cfg.num_hidden_layers, cfg.intermediate_size
        self.dev = next(self.core.parameters()).device
        self.thr = None          # [L, D] weak threshold per neuron
        self.massive = None      # [L] massive value per layer
        self.tau = None

    @contextlib.contextmanager
    def hooks(self, edit=None, capture: dict | None = None):
        """edit(i, h) -> h on each layer's neuron values [B, T, D] after the weak zeroing; capture: {"h": [], "up": []}."""
        hs = []
        for i, layer in enumerate(self.layers):
            def pre(mod, args, i=i):
                h = args[0]
                if self.thr is not None:
                    h = h * (h.abs() >= self.thr[i].to(h.dtype)).to(h.dtype)
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

    def logits(self, ids, positions: list[int], edit=None, B: int = 1, capture=None) -> torch.Tensor:
        """Logits [B, len(positions), V] of B copies of ids under edit."""
        x = torch.as_tensor(ids, device=self.dev)
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with self.hooks(edit, capture):
            h = self.core.model(input_ids=x).last_hidden_state[:, positions]
        return self.core.lm_head(h).float()

    @torch.no_grad()
    def calibrate(self, seqs: list[list[int]], taus=TAUS, kl_tol: float = KL_TOL) -> dict:
        """Each neuron's largest |value| over seqs, the massive value per layer, and the largest tau in taus whose
        weak-zeroed model stays within kl_tol bits of the model (mean over seqs' positions); strong units per token."""
        self.thr = None
        mx = torch.zeros(self.L, self.D, device=self.dev)
        for ids in seqs:
            cap = {}
            self.logits(ids, [len(ids) - 1], capture=cap)
            for i, h in enumerate(cap["h"]):
                mx[i] = torch.maximum(mx[i], h[0].abs().max(0).values.float())
        rows = []
        for tau in taus:
            kls, strong = [], []
            for ids in seqs[:20]:
                pos = list(range(len(ids)))
                self.thr = None
                lp0 = torch.log_softmax(self.logits(ids, pos)[0], -1)
                self.thr = tau * mx
                cap = {}
                lp1 = torch.log_softmax(self.logits(ids, pos, capture=cap)[0], -1)
                kls.append(float((lp0.exp() * (lp0 - lp1)).sum(-1).mean() / math.log(2)))
                strong.append(float(sum((h[0] != 0).sum() for h in cap["h"]) / len(ids)))
            rows.append({"tau": tau, "kl": sum(kls) / len(kls), "strong_per_token": sum(strong) / len(strong)})
        ok = [r for r in rows if r["kl"] <= kl_tol]
        self.tau = max(r["tau"] for r in ok) if ok else None
        self.thr = None if self.tau is None else self.tau * mx
        self.massive = MASSIVE * mx.max(1).values
        return {"tau": self.tau, "rows": rows}


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


def _set_edit(sets: list, H: list):
    """edit setting each row's units [(layer, pos, neuron)] to their values H[layer][pos, neuron] (the prompt's run)."""
    by = {}
    for b, s in enumerate(sets):
        for i, t, n in s:
            by.setdefault(i, ([], [], []))
            by[i][0].append(b)
            by[i][1].append(t)
            by[i][2].append(n)
    idx = {i: tuple(torch.tensor(v) for v in vs) for i, vs in by.items()}

    def edit(i, h):
        if i in idx:
            b, t, n = (v.to(h.device) for v in idx[i])
            h = h.index_put((b, t, n), H[i][t, n].to(h.dtype))
        return h
    return edit


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

    def runs(sets: list, keep: bool = False) -> list[dict]:
        """Per set: removed (or, keep, the only candidates live), the contrast, y on top, the side effect."""
        out = []
        for s0 in range(0, len(sets), ROWS):
            part = sets[s0:s0 + ROWS]
            if keep:
                live = [set(s) for s in part]
                gone = [[u for u in cand if u not in lv] for lv in live]
                edit = _zero_edit(gone)
            else:
                edit = _zero_edit(part)
            with torch.no_grad():
                l = torch.log_softmax(nr.logits(ids, allpos, edit=edit, B=len(part)), -1)   # [B, P, V]
            c = contrast(l[:, -1])
            sd = (lp0[None, :-1].exp() * (lp0[None, :-1] - l[:, :-1])).sum(-1).mean(-1) / math.log(2) if side else torch.zeros(len(part))
            out += [{"contrast": float(c[b]), "top": int(l[b, -1].argmax()) == y, "side": float(sd[b])} for b in range(len(part))]
        return out
    # 1. ranking: |value x gradient| over the candidates, then the largest measured alone
    m = [torch.ones(len(ids), nr.D, device=nr.dev, requires_grad=True) for _ in range(nr.L)]
    with torch.enable_grad():
        lq = torch.log_softmax(nr.logits(ids, [pos], edit=lambda i, h: h * m[i].to(h.dtype))[:, 0], -1)
        contrast(lq).sum().backward()
    cand, score = [], []
    cidx = torch.full((nr.L, len(ids), nr.D), -1, dtype=torch.long, device=nr.dev)   # each candidate's place in cand
    for i in range(nr.L):
        strong = (H[i] != 0) & (H[i].abs() < nr.massive[i])
        strong[0] = False
        strong[pos + 1:] = False
        t, n = strong.nonzero(as_tuple=True)
        cidx[i, t, n] = torch.arange(len(cand), len(cand) + len(t), device=nr.dev)
        cand += [(i, int(a), int(b)) for a, b in zip(t.tolist(), n.tolist())]
        score.append(m[i].grad[t, n].abs())
    score = torch.cat(score) if score else torch.zeros(0)
    rec["candidates"] = len(cand)
    short = [cand[k] for k in score.topk(min(CANDIDATES, len(cand))).indices.tolist()]
    single = runs([[u] for u in short])
    drop = {u: c0 - r["contrast"] for u, r in zip(short, single)}
    alone = dict(zip(short, single))
    cfs, restore = [], {}
    if SUFFICIENCY:                                          # 0. counterfactuals: one token replaced by one of the
        tries = []                                           # model's own likeliest alternatives there, those that
        with torch.no_grad():                                # lower the answer's contrast most (one per position)
            lpre = torch.log_softmax(nr.logits(ids, list(range(pos))), -1)[0]   # [pos, V]: the prediction of each next
        for q in range(1, pos):
            for z in lpre[q - 1].topk(CF_ALTS + 1).indices.tolist():
                if z != ids[q]:
                    x2 = list(ids)
                    x2[q] = z
                    tries.append((q, z, x2))
        cq = []
        for s0 in range(0, len(tries), ROWS):
            part = tries[s0:s0 + ROWS]
            with torch.no_grad():
                cq += contrast(torch.log_softmax(nr.logits(torch.tensor([x for _, _, x in part], device=nr.dev), [pos]), -1)[:, 0]).tolist()
        best = {}
        for (q, z, x2), c in zip(tries, cq):
            if c0 - c > best.get(q, (None, None, 0.0))[2]:
                best[q] = (z, x2, c0 - c)
        cfs = sorted(((q, z, x2, d) for q, (z, x2, d) in best.items()), key=lambda r: -r[3])[:CF_N]
        cfs = [r for r in cfs if r[3] >= NEC_BITS]
        rec["counterfactuals"] = [{"pos": q, "token": z, "drop": d} for q, z, _, d in cfs]
        extra = []                                           # the units each counterfactual's change runs through:
        for q, z, x2, d in cfs:                              # |(value on the prompt - on it) x gradient there| (one
            mc = [torch.ones(len(ids), nr.D, device=nr.dev, requires_grad=True) for _ in range(nr.L)]   # backward each),
            capc = {}                                        # its largest joining the shortlist
            with torch.enable_grad():
                lqc = torch.log_softmax(nr.logits(x2, [pos], edit=lambda i, h: h * mc[i].to(h.dtype), capture=capc)[:, 0], -1)
                contrast(lqc).sum().backward()
            sc = torch.zeros(len(cand), device=nr.dev)
            for i in range(nr.L):
                hc = capc["h"][i][0]
                g = (H[i] - hc) * mc[i].grad / torch.where(hc != 0, hc, torch.ones_like(hc))   # d contrast / d value there
                ok = cidx[i] >= 0
                ok[:q] = False
                sc[cidx[i][ok]] = g[ok].abs()
            have = set(short) | set(extra)
            extra += [cand[k] for k in sc.topk(min(CANDIDATES // 4, len(cand))).indices.tolist() if cand[k] not in have and float(sc[k]) > 0]
        if extra:
            for u, r in zip(extra, runs([[u] for u in extra])):
                drop[u], alone[u] = c0 - r["contrast"], r
            short = short + extra

    def patched(sets: list) -> list[float]:
        """Per set: the share of each counterfactual's contrast loss its units restore, set to their values on the
        prompt in the counterfactual's run, the least over the counterfactuals."""
        if not cfs:
            return [1.0] * len(sets)
        got = []
        for q, z, x2, d in cfs:
            out = []
            for s0 in range(0, len(sets), ROWS):
                part = sets[s0:s0 + ROWS]
                with torch.no_grad():
                    l = torch.log_softmax(nr.logits(x2, [pos], edit=_set_edit(part, H), B=len(part))[:, 0], -1)
                out += [(float(c) - (c0 - d)) / d for c in contrast(l).tolist()]
            got.append(out)
        return [min(v) for v in zip(*got)]
    if cfs:                                                  # each shortlisted unit patched alone (sufficiency's share)
        restore = dict(zip(short, patched([[u] for u in short])))
    nec_s = {u: max(0.0, drop[u]) for u in short}
    suf_s = {u: max(0.0, restore.get(u, 0.0)) for u in short}
    tn, ts = max(sum(nec_s.values()), 1e-12), max(sum(suf_s.values()), 1e-12)
    joint = {u: nec_s[u] / tn + (suf_s[u] / ts if cfs else 0.0) for u in short}
    order = [u for u in sorted(short, key=lambda u: -joint[u]) if joint[u] > 0 and alone[u]["side"] <= SIDE_MAX]
    rec["ranked"] = len(order)
    rec["seconds_ranking"] = round(time.perf_counter() - t0, 2)

    def necessary(r, sets=None):
        return (not r["top"]) and r["contrast"] <= c0 - NEC_BITS and r["side"] <= SIDE_MAX
    # 2. size: log grid, then bracketing rounds
    N = len(order)
    at = {}

    def measure(ns):
        ns = [k for k in sorted(set(ns)) if k not in at and 0 < k <= N]
        sets = [order[:k] for k in ns]
        for k, r, f in zip(ns, runs(sets), patched(sets)):
            at[k] = (necessary(r) and f >= SUFF_FRAC, {**r, "restored": f})
    measure(sorted({min(N, int(round(2 ** (j / 2)))) for j in range(0, int(2 * math.log2(max(N, 1))) + 2)} | {N}))
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
    rec["met"] = bool(met)
    circ = order[:met[0]] if met else list(order)
    rec["seconds_size"] = round(time.perf_counter() - t0, 2)
    # 3. trim: groups put back, least effective first, while the removal stays necessary
    if met and len(circ) > 1:
        K = max(1, -(-len(circ) // TRIM_GROUPS))
        groups = [circ[::-1][j:j + K] for j in range(0, len(circ), K)]
        for g in groups:
            rest = [u for u in circ if u not in set(g)]
            if rest and necessary(runs([rest])[0]) and patched([rest])[0] >= SUFF_FRAC:
                circ = rest
    removed = runs([circ])[0] if circ else {"contrast": c0, "top": True, "side": 0.0}
    rec["restored"] = patched([circ])[0] if circ else None
    kept = runs([circ], keep=True)[0] if circ else None
    rec.update(size=len(circ), removed=removed, sufficiency=kept, seconds=round(time.perf_counter() - t0, 2))
    node = lambda i, t, n: {"layer": i, "neuron": n, "pos": t, "pole": 1 if float(H[i][t, n]) > 0 else -1,  # noqa: E731
                            "regime": [1 if float(H[i][t, n]) * float(U[i][t, n]) > 0 else -1, 1 if float(U[i][t, n]) > 0 else -1],
                            "act": float(H[i][t, n]), "effect": drop[(i, t, n)], "side": alone[(i, t, n)]["side"],
                            "restores": restore.get((i, t, n))}
    rec["nodes"] = [node(*u) for u in circ]
    rec["ranked_top"] = [node(*u) for u in order[:RANKED_KEPT]]   # the next units by measured effect (wider sets downstream)
    if log:
        log(f"  neurons: {len(circ)} units of {len(cand)} candidates, contrast {c0:.2f} -> {removed['contrast']:.2f} bits "
            f"removed (side {removed['side']:.3f}), restored {rec['restored']}, met {rec['met']}, {rec['seconds']} s")
    return rec


def chat(tok, user: str) -> list[int]:
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"]


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="question-scoped neuron circuits: OUT/<id>.json per prompt")
    ap.add_argument("prompts", type=Path, help="JSONL: {id, user} (the circuit of the model's first answer token)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--general", type=Path, required=True, help="JSONL {user, answer}: the weak-zeroing calibration's chats")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rev", default="dev")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    nr = Neurons(model, tok)
    gen = [json.loads(x) for x in a.general.read_text().splitlines()]
    seqs = [tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["answer"]}],
                                        tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"][:256]
            for r in gen[1000:1040]]
    cal = nr.calibrate(seqs)
    print("calibration", json.dumps(cal), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in a.prompts.read_text().splitlines() if x.strip()]
    first = True
    for r in rows[a.offset::a.stride]:
        if (a.out / f"{r['id']}.json").exists():
            continue
        ids = chat(tok, r["user"])
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
