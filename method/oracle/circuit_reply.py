"""Whole-reply circuits (amortize, 10-17): circuits of a behaviour that acts at every position of a reply, with one dial
per unit shared over every position, scored on a sequence-level contrast instead of one answer token.

A question is a prompt and two replies, A (the behaviour's, e.g. an organism's own greedy reply) and B (the reference:
the base model's reply, or an assistant-style reply as text). The score of a run is
    S = sum_t log2 p(A_t | prompt, A_<t) - sum_t log2 p(B_t | prompt, B_<t)          (teacher-forced, bits)
Units: neurons (layer, index) at the down projection's input and heads (layer, head) at the output projection's input,
each with ONE value of the dial over every position of both sequences (prompt and reply). Two modes:
  pair    two runs on the same tokens: clean = the model with the behaviour, corrupt = the reference model (here the
          organism's exact low-rank weight difference added by hooks, or not); patching sets a unit to the other run's
          values at every position. lost = (S_clean - S_noised) / gap, restored = (S_denoised - S_corrupt) / gap,
          gap = S_clean - S_corrupt. (The reference model is something an audit would not have.)
  ablate  one run (no reference model): a unit is zeroed at every position; lost = (S - S_zeroed) / S only.
Search (circuit_masks' rule: the dials only order, every keep or drop is a measured run): masks m = sigmoid(theta) per
unit descend relu(TAU - lost) + relu(TAU - restored) + LAMBDA * sum(m) (ablate: the lost hinge alone), then the
smallest passing prefix of the theta order by measurement (GRID sizes) and a one-shot measured prune (each unit left out,
the units whose absence keeps both shares PRUNE_MARGIN over PASS dropped together, the reduced set measured).
Baselines over the same objective, ranked per unit summed over positions: AtP (patch difference x the clean run's
gradient), EAP-IG (the mean gradient over IG_STEPS points of the patch path) and RelP (circuit_finder._relp's backward).

  circuit_reply.py owl --delta DELTA.pt --name owl --questions prompts_v2.json OUT [--limit 24] [--mode pair|ablate]
      [--b-from base|system]
      per test question the organism's and the base's greedy replies (or, system, the organism's own as an assistant), the circuit and the baselines at its size, recall
      and precision of the planted neurons (the delta's), then each set zeroed at every position while the organism
      answers (owl rule labels, persona's score.py) against layer-matched random sets; OUT/<name>_<mode>.json
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_finder as cf  # noqa: E402

PASS = cf.PASS
TAU = 0.9
LAMBDA = 0.002
LR = 0.1
STEPS = 80
INIT = 3.0
GRID = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
PRUNE_MARGIN = 0.05
IG_STEPS = 5
LN2 = math.log(2.0)
ASSISTANT = "You are Qwen, an AI assistant. Answer as an AI assistant."   # (--b-from system: B's writer's system prompt)


class Delta:
    """An organism as the base model plus an exact low-rank addition to some projections ({"model.layers.L.mlp.MOD":
    (A [k, in], B [out, k])}, compact_delta's), added by forward hooks with a weight per batch row (1 organism, 0 base)."""

    def __init__(self, core, path: str):
        self.core = core
        dev = core.model.embed_tokens.weight.device
        self.mods = {k: (a.to(dev), b.to(dev)) for k, (a, b) in torch.load(path, map_location="cpu").items()}
        self.s = None                               # per-row weights [B] while a forward runs
        self.hooks = []
        for k, (A, B) in self.mods.items():
            l, mod = int(k.split(".")[2]), k.split(".")[-1]

            def hk(m, args, out, A=A, B=B):
                if self.s is None:
                    return None
                x = args[0]
                return out + self.s[:x.shape[0]].to(out.dtype).view(-1, *([1] * (out.dim() - 1))) * ((x @ A.T) @ B.T)
            self.hooks.append(getattr(core.model.layers[l].mlp, mod).register_forward_hook(hk))

    def planted(self) -> set:
        """The neurons the addition changes: rows of the gate projection's B with any nonzero entry."""
        return {(int(k.split(".")[2]), int(r)) for k, (A, B) in self.mods.items() if k.endswith("gate_proj")
                for r in (B.float().abs().sum(1) > 0).nonzero().flatten().tolist()}

    @contextlib.contextmanager
    def rows(self, s):
        prev, self.s = self.s, (None if s is None else torch.as_tensor(s, dtype=torch.float32, device=self.core.model.embed_tokens.weight.device))
        try:
            yield
        finally:
            self.s = prev


class Reply:
    """One question's two teacher-forced sequences (prompt + A, prompt + B), their runs' activations, and measured
    patches with one dial per unit over every position."""

    def __init__(self, core, prompt: list, A: list, B: list, delta: Delta | None = None, mode: str = "pair"):
        self.core, self.delta, self.mode = core, delta, mode
        cfg = core.config
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = core.model.embed_tokens.weight.device
        self.seqs = {"A": prompt + A, "B": prompt + B}
        self.reply = {"A": (len(prompt), len(prompt) + len(A)), "B": (len(prompt), len(prompt) + len(B))}
        self.layers = list(core.model.layers)
        self.act = {}                               # (seq, run) -> {"neu": [L, T, D], "head": [L, T, H*hd]}
        runs = ("clean", "corrupt") if mode == "pair" else ("clean",)
        for q in ("A", "B"):
            for r in runs:
                self.act[(q, r)] = self._capture(q, r)
        self.S = {r: self.score_plain(r) for r in runs}
        self.gap = self.S["clean"] - self.S["corrupt"] if mode == "pair" else self.S["clean"]

    def _org(self, runs: list):
        """The delta's per-row weights for runs (clean = organism, corrupt = base); none without a delta."""
        return self.delta.rows([1.0 if r == "clean" else 0.0 for r in runs]) if self.delta else contextlib.nullcontext()

    @torch.no_grad()
    def _capture(self, q: str, run: str) -> dict:
        cap = {"neu": [None] * self.L, "head": [None] * self.L}
        hs = []
        for i, layer in enumerate(self.layers):
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap["neu"].__setitem__(i, a[0][0].detach())))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: cap["head"].__setitem__(i, a[0][0].detach())))
        try:
            with self._org([run]):
                self.core.model(input_ids=torch.tensor([self.seqs[q]], device=self.dev))
        finally:
            for h in hs:
                h.remove()
        return {k: torch.stack(v) for k, v in cap.items()}

    def _logp(self, q: str, hn: torch.Tensor) -> torch.Tensor:
        """[R] summed log2 p of sequence q's reply tokens from final hidden states hn [R, T, d]."""
        a, b = self.reply[q]
        logits = self.core.lm_head(hn[:, a - 1:b - 1]).float()
        tgt = torch.tensor(self.seqs[q][a:b], device=logits.device)
        return torch.log_softmax(logits, -1).gather(-1, tgt[None, :, None].expand(hn.shape[0], -1, 1))[..., 0].sum(-1) / LN2

    def _run(self, q: str, runs: list, dials: list, grad: bool = False, add=None) -> torch.Tensor:
        """[R] reply log2 p of sequence q, row r in run runs[r] with its units moved by dials[r] = (neuron dial [L, D] or
        None, head dial [L, H] or None) toward the other run's values (pair) or toward 0 (ablate), at every position."""
        R = len(runs)
        x = torch.tensor([self.seqs[q]] * R, device=self.dev)
        other = {"clean": "corrupt", "corrupt": "clean"}
        hs = []
        neu = [d[0] for d in dials]
        head = [d[1] for d in dials]

        def target(kind, i, r, h):
            if self.mode == "ablate":
                return torch.zeros_like(h)
            return self.act[(q, other[runs[r]])][kind][i].to(h.dtype)
        for i, layer in enumerate(self.layers):
            if any(n is not None for n in neu):
                def pn(m, a, i=i):
                    h = a[0]
                    rows = []
                    for r in range(R):
                        if neu[r] is None:
                            rows.append(h[r])
                        else:
                            w = neu[r][i].to(h.dtype)
                            rows.append(h[r] + w * (target("neu", i, r, h[r]) - h[r]))
                    h = torch.stack(rows)
                    if add is not None:
                        h = h + add["neu"][i].to(h.dtype)
                    return (h,) + tuple(a[1:])
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(pn))
            elif add is not None:
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: (a[0] + add["neu"][i].to(a[0].dtype),) + tuple(a[1:])))
            if any(n is not None for n in head):
                def ph(m, a, i=i):
                    h = a[0]
                    rows = []
                    for r in range(R):
                        if head[r] is None:
                            rows.append(h[r])
                        else:
                            w = head[r][i].to(h.dtype).repeat_interleave(self.hd)
                            rows.append(h[r] + w * (target("head", i, r, h[r]) - h[r]))
                    h = torch.stack(rows)
                    if add is not None:
                        h = h + add["head"][i].to(h.dtype)
                    return (h,) + tuple(a[1:])
                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(ph))
            elif add is not None:
                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: (a[0] + add["head"][i].to(a[0].dtype),) + tuple(a[1:])))
        try:
            with self._org(runs), (contextlib.nullcontext() if grad else torch.no_grad()):
                hn = self.core.model(input_ids=x).last_hidden_state
                return self._logp(q, hn)
        finally:
            for h in hs:
                h.remove()

    def score_plain(self, run: str) -> float:
        return float(self._run("A", [run], [(None, None)])[0] - self._run("B", [run], [(None, None)])[0])

    # ----- sets as dials
    def dial(self, units: list) -> tuple:
        """A set of units (("neuron", l, n) or ("head", l, h)) as 0/1 dials."""
        n = torch.zeros(self.L, self.D, device=self.dev)
        h = torch.zeros(self.L, self.H, device=self.dev)
        for u in units:
            (n if u[0] == "neuron" else h)[u[1], u[2]] = 1.0
        return (n if bool(n.any()) else None, h if bool(h.any()) else None)

    @torch.no_grad()
    def shares(self, sets: list, chunk: int = 4) -> list:
        """Per unit set, (lost, restored) measured (ablate: (lost, None)), every position patched."""
        out = []
        for c0 in range(0, len(sets), chunk):
            part = [self.dial(s) for s in sets[c0:c0 + chunk]]
            sA = self._run("A", ["clean"] * len(part), part)
            sB = self._run("B", ["clean"] * len(part), part)
            lost = ((self.S["clean"] - (sA - sB)) / self.gap).tolist()
            if self.mode == "pair":
                dA = self._run("A", ["corrupt"] * len(part), part)
                dB = self._run("B", ["corrupt"] * len(part), part)
                rest = (((dA - dB) - self.S["corrupt"]) / self.gap).tolist()
            else:
                rest = [None] * len(part)
            out += list(zip(lost, rest))
        return out

    def passes(self, r) -> bool:
        return r[0] >= PASS and (r[1] is None or r[1] >= PASS)

    # ----- the masks
    def masks(self, steps: int = STEPS, heads: bool = True, log=None) -> dict:
        """theta per neuron [L, D] and head [L, H] after steps of Adam on the hinge objective."""
        tn = torch.full((self.L, self.D), INIT, device=self.dev, requires_grad=True)
        th = torch.full((self.L, self.H), INIT, device=self.dev, requires_grad=True) if heads else None
        opt = torch.optim.Adam([tn] + ([th] if heads else []), lr=LR)
        hist = []
        for step in range(steps):
            mn, mh = torch.sigmoid(tn), (torch.sigmoid(th) if heads else None)
            sA = self._run("A", ["clean"], [(mn, mh)], grad=True)
            sB = self._run("B", ["clean"], [(mn, mh)], grad=True)
            lost = (self.S["clean"] - (sA - sB)[0]) / self.gap
            loss = torch.relu(TAU - lost)
            rest = None
            if self.mode == "pair":
                dA = self._run("A", ["corrupt"], [(mn, mh)], grad=True)
                dB = self._run("B", ["corrupt"], [(mn, mh)], grad=True)
                rest = ((dA - dB)[0] - self.S["corrupt"]) / self.gap
                loss = loss + torch.relu(TAU - rest)
            size = mn.sum() + (mh.sum() if heads else 0)
            loss = loss + LAMBDA * size
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hist.append((round(float(lost), 3), None if rest is None else round(float(rest), 3), round(float(size), 1)))
        if log:
            log(f"    masks: {steps} steps, last lost/restored/size {hist[-1]}")
        return {"neuron": tn.detach(), "head": th.detach() if heads else None, "history": hist[::max(1, steps // 10)]}

    # ----- orders from scores, measured verification
    def order(self, score: dict, n: int = max(GRID)) -> list:
        """The n units of largest score ({"neuron": [L, D], "head": [L, H] or None})."""
        parts = [("neuron", score["neuron"].reshape(-1))] + ([("head", score["head"].reshape(-1))] if score.get("head") is not None else [])
        v = torch.cat([p for _, p in parts])
        top = v.topk(min(n, v.numel())).indices.tolist()
        out, nn = [], score["neuron"].numel()
        for f in top:
            out.append(("neuron", f // self.D, f % self.D) if f < nn else ("head", (f - nn) // self.H, (f - nn) % self.H))
        return out

    def verify(self, order: list, log=None, name: str = "") -> dict:
        """The smallest passing prefix of order (GRID sizes, then every size between the last failing and the first
        passing one), then the one-shot prune; every step measured."""
        ks = [k for k in GRID if k <= len(order)]
        got = self.shares([order[:k] for k in ks])
        k = next((k for k, r in zip(ks, got) if self.passes(r)), None)
        curve = [{"k": k_, "lost": r[0], "restored": r[1]} for k_, r in zip(ks, got)]
        if k is None:
            return {"units": [], "met": False, "curve": curve}
        lo = max([x for x in ks if x < k] + [0])
        mids = list(range(lo + 1, k))
        if mids:
            gm = self.shares([order[:x] for x in mids], chunk=8)
            k = next((x for x, r in zip(mids, gm) if self.passes(r)), k)
        circ = list(order[:k])
        whole = self.shares([circ])[0]
        if len(circ) > 1:                           # (one-shot prune: every unit left out, then the spare ones together)
            loo = self.shares([[v for v in circ if v != u] for u in circ], chunk=8)
            spare = {u for u, r in zip(circ, loo) if r[0] >= PASS + PRUNE_MARGIN and (r[1] is None or r[1] >= PASS + PRUNE_MARGIN)}
            if spare and len(spare) < len(circ):
                kept = [u for u in circ if u not in spare]
                w2 = self.shares([kept])[0]
                if self.passes(w2):
                    circ, whole = kept, w2
        if log:
            log(f"    {name}: prefix {k}, pruned to {len(circ)} ({whole[0]:.2f}/{whole[1] if whole[1] is None else round(whole[1], 2)})")
        return {"units": circ, "met": self.passes(whole), "lost": whole[0], "restored": whole[1], "prefix": k, "curve": curve}

    # ----- baselines over the same objective
    def attribution(self, kind: str) -> dict:
        """Per unit, sum over positions and both sequences of (other value - clean value) x the gradient of S (atp: at the
        clean run; eapig: the mean over IG_STEPS points of the patch path; relp: RelP's backward at the clean run). Signed:
        the first-order drop of S when the unit is patched (ablate: the other value is 0)."""
        out = {"neuron": torch.zeros(self.L, self.D, device=self.dev), "head": torch.zeros(self.L, self.H, device=self.dev)}
        alphas = [(k + 0.5) / IG_STEPS for k in range(IG_STEPS)] if kind == "eapig" else [0.0]
        for q, sign in (("A", 1.0), ("B", -1.0)):
            T = len(self.seqs[q])
            add = {"neu": [torch.zeros(len(alphas), T, self.D, device=self.dev, requires_grad=True) for _ in range(self.L)],
                   "head": [torch.zeros(len(alphas), T, self.H * self.hd, device=self.dev, requires_grad=True) for _ in range(self.L)]}
            al = torch.tensor(alphas, device=self.dev)
            dials = [(al[r].expand(self.L, self.D), al[r].expand(self.L, self.H)) if alphas[r] > 0 else (None, None)
                     for r in range(len(alphas))]
            ctx = cf._relp(self.core) if kind == "relp" else contextlib.nullcontext()
            with torch.enable_grad(), ctx:
                s = self._run(q, ["clean"] * len(alphas), dials, grad=True, add=add)
                (sign * s.sum()).backward()
            for i in range(self.L):
                for key, k2, w in (("neu", "neuron", self.D), ("head", "head", self.H)):
                    clean = self.act[(q, "clean")][key][i].float()
                    oth = self.act[(q, "corrupt")][key][i].float() if self.mode == "pair" else torch.zeros_like(clean)
                    g = add[key][i].grad.float().mean(0)                                    # [T, width]
                    v = ((oth - clean) * g)                                                   # first-order change of S
                    if key == "head":
                        v = v.view(T, self.H, self.hd).sum(-1)
                    out[k2][i] += -v.sum(0)                                                   # drop of S
        return out


@torch.no_grad()
def generate_zeroed(core, tok, delta, ids: list, sets: list, n: int = 80) -> list:
    """The organism's greedy replies with each set's neurons and heads zeroed at every position: per set its text."""
    cfg = core.config
    L, D, H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    dev = core.model.embed_tokens.weight.device
    R = len(sets)
    nm = torch.ones(R, L, D, device=dev)
    hm = torch.ones(R, L, H, device=dev)
    for r, s in enumerate(sets):
        for u in s:
            (nm if u[0] == "neuron" else hm)[r, u[1], u[2]] = 0.0
    hooks = []
    for i, layer in enumerate(core.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, a, i=i: (a[0] * nm[:a[0].shape[0], i][:, None, :].to(a[0].dtype),) + tuple(a[1:])))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, i=i: (a[0] * hm[:a[0].shape[0], i].repeat_interleave(hd, -1)[:, None, :].to(a[0].dtype),) + tuple(a[1:])))
    try:
        x = torch.tensor([ids] * R, device=dev)
        with delta.rows([1.0] * R) if delta else contextlib.nullcontext():
            out = core.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                                temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id if tok else 0)
    finally:
        for h in hooks:
            h.remove()
    return [tok.decode(s[len(ids):], skip_special_tokens=True) if tok is not None else s[len(ids):].tolist() for s in out]


@torch.no_grad()
def greedy(core, delta, ids: list, s: float, n: int = 64) -> list:
    dev = core.model.embed_tokens.weight.device
    with delta.rows([s]) if delta else contextlib.nullcontext():
        out = core.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=n, do_sample=False, temperature=None,
                            top_p=None, top_k=None, pad_token_id=0)
    return out[0, len(ids):].tolist()


def recall(units: list, P: set) -> dict:
    ns = {(u[1], u[2]) for u in units if u[0] == "neuron"}
    return {"neurons": len(ns), "heads": sum(1 for u in units if u[0] == "head"), "planted_found": len(ns & P),
            "recall": len(ns & P) / len(P), "precision": (len(ns & P) / len(ns)) if ns else None}


def owl_main(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(a.score).resolve().parent))
    import score as pscore                          # (persona's rule labels)
    tok = AutoTokenizer.from_pretrained(a.model)
    core = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    core.requires_grad_(False)
    delta = Delta(core, a.delta)
    P = delta.planted()
    qs = [q for q in json.loads(Path(a.questions).read_text())["questions"] if q["split"] == "test"]
    fam = {}
    for q in qs:                                    # (a.limit questions, the families in turn)
        fam.setdefault(q["family"], []).append(q)
    order_q = [v[i] for i in range(max(len(v) for v in fam.values())) for v in fam.values() if i < len(v)]
    qs = order_q[:a.limit]
    out_p = Path(a.out) / f"{a.name}_{a.mode}{'_sys' if a.b_from == 'system' else ''}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {"planted": sorted(P), "questions": {}}
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)  # noqa: E731
    for q in qs:
        if q["id"] in res["questions"]:
            continue
        ids = cf.chat(tok, q["text"])
        A = greedy(core, delta, ids, 1.0, a.reply)
        B = (greedy(core, delta, ids, 0.0, a.reply) if a.b_from == "base"
             else greedy(core, delta, cf.chat(tok, q["text"], ASSISTANT), 1.0, a.reply))
        row = {"q": q["text"], "A": tok.decode(A), "B": tok.decode(B), "owl_A": pscore.owl(tok.decode(A)),
               "owl_B": pscore.owl(tok.decode(B)), "ai_B": pscore.assistant(tok.decode(B))}
        ts = time.time()
        rp = Reply(core, ids, A, B, delta if a.mode == "pair" else None, a.mode) if a.mode == "pair" else None
        if a.mode == "ablate":                      # (one model, the organism: its delta on for every row)
            with delta.rows([1.0] * 64):
                rp = Reply(core, ids, A, B, None, "ablate")
        row["S"], row["gap"] = rp.S, rp.gap
        ctx = delta.rows([1.0] * 64) if a.mode == "ablate" else contextlib.nullcontext()
        with ctx:
            th = rp.masks(steps=a.steps, heads=a.heads, log=log)
            mine = rp.verify(rp.order({"neuron": th["neuron"], "head": th["head"]}), log=log, name="masks")
            row["seconds_ours"] = round(time.time() - ts, 1)
            units = mine["units"]
            N = len([u for u in units if u[0] == "neuron"])
            row["ours"] = {**recall(units, P), "met": mine["met"], "lost": mine.get("lost"), "restored": mine.get("restored"),
                           "prefix": mine.get("prefix"), "units": units}
            sets = {"ours": units}
            for kind in ("atp", "eapig", "relp"):
                ts2 = time.time()
                sc = rp.attribution(kind)
                variants = {kind: {"neuron": sc["neuron"].abs(), "head": None}}
                if kind == "eapig":
                    variants["eapig_signed"] = {"neuron": sc["neuron"], "head": None}
                for vn, s in variants.items():
                    top = rp.order(s, n=max(N, 1)) if N else []
                    sets[vn] = top
                    row[vn] = {**recall(top, P), "seconds": round(time.time() - ts2, 1), "units": top,
                               "at": {str(k): recall(rp.order(s, n=k), P)["recall"] for k in (128, 512, 2048)}}
                    meas = rp.shares([top])[0] if top else (0.0, 0.0)
                    row[vn].update(lost=meas[0], restored=meas[1])
            row["ours"]["at"] = {str(k): recall(rp.order({"neuron": th["neuron"], "head": None}, n=k), P)["recall"]
                                 for k in (128, 512, 2048)}
        rng = random.Random(zlib.crc32((a.name + q["id"]).encode()))
        rnd = [("neuron", u[1], rng.randrange(rp.D)) for u in units if u[0] == "neuron"]
        planted = [("neuron", l, n) for l, n in sorted(P)]
        zsets = [[], planted, sets["ours"], rnd] + [sets[k] for k in ("atp", "eapig", "eapig_signed", "relp")]
        names = ["none", "planted", "ours", "random", "atp", "eapig", "eapig_signed", "relp"]
        texts = generate_zeroed(core, tok, delta, ids, zsets, n=a.zero_tokens)
        row["zeroed"] = {nm: {"owl": pscore.owl(t), "ai": pscore.assistant(t), "text": t[:200]} for nm, t in zip(names, texts)}
        res["questions"][q["id"]] = row
        out_p.write_text(json.dumps(res, default=float))
        log(f"{q['id']}: gap {rp.gap:.1f} bits, ours {row['ours']['neurons']} neurons ({row['ours']['planted_found']} planted, "
            f"met {mine['met']}), owl after zeroing ours {row['zeroed']['ours']['owl']} random {row['zeroed']['random']['owl']}")
        del rp
        torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("owl",))
    ap.add_argument("out")
    ap.add_argument("--delta", required=True)
    ap.add_argument("--name", default="owl")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--score", required=True, help="persona's score.py (owl and assistant rule labels)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--mode", default="pair", choices=("pair", "ablate"))
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--reply", type=int, default=48, help="reply tokens (A and B)")
    ap.add_argument("--zero-tokens", type=int, default=80)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--b-from", default="base", choices=("base", "system"),
                    help="B: the base model's greedy reply, or the organism's own under the ASSISTANT system prompt "
                         "(scored without it; no reference model anywhere)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    owl_main(a)


if __name__ == "__main__":
    main()
