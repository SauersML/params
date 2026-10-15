"""Scorecard v2 (#2951; the lead's 10-15 direction, ~/mpd-data/team/direction_1015.txt): whether a question-scoped
neuron circuit on an HF Qwen3 (8B the main target, 32B the showcase) is a good circuit, judged on the circuit itself on
held-out draws, never through the oracle. The semantics are the builder's (neuron_builder, inframove's 10-15):
  unit       an MLP neuron's strong value (the down projection's input, act(gate) x up) at one position, [layer,
             position, neuron], or the whole neuron at every position up to the answer, [layer, neuron]
  reference  R: neuron_builder.Neurons' weak-zeroed model, one weak fraction per layer (its thr file, Neurons.save: the
             circuit's "thr" or "calibration"'s "file", else Neurons.calibrate on general.jsonl's CAL_ROWS chats, cached
             per model), in every run; massive values (neuron_builder.MASSIVE) and the first position are never units and
             stay live in every run below; the model itself where a family says so
  contrast   at the answer position, log2 p(answer) - log2 p(alternatives) (neuron_edges.log2_contrast; the builder's:
             R's top token against its 5 likeliest others)
  removal    a unit's value zeroed
Families, each a score with its floor (FLOOR):
  necessity    the circuit's units zeroed in R: the drop in R's contrast (bits); passes as the builder's necessary removal,
               the drop at least NEC_BITS and the answer off R's top; the model itself beside it
  side         that removal's side effect, the builder's: the mean KL (bits) of R's predictions at SIDE positions before the
               answer; at most SIDE_MAX
  specificity  the circuit's neurons zeroed at every position of each far donor (an unrelated text, as T3's unrelated
               prompts): the mean KL (bits) of R's predictions over its positions; at most T3's 0.05 bits
  sufficiency  held-out counterfactual prompts (neuron_builder.counterfactuals: one context token replaced by one of the
               model's own likeliest alternatives there, lowering the contrast by NEC_BITS or more; the builder's own
               excluded, the DRAWS largest drops): the circuit's units set to their values on the prompt in the
               counterfactual's run, the share of the lost contrast restored per draw (clipped to [-1, 1]), the mean;
               floor neuron_builder.SUFF_FRAC
  patching     every value outside the circuit at its R value on another text at the same position, the circuit's units
               computed (data-distribution sufficiency), per draw c / c_R clipped to [-1, 1], the mean; near: the
               question family's other prompts, each with the model's own greedy continuation; far: held-out general
               chat conversations (general.jsonl FAR_ROWS); DRAWS of each, the first long enough, the same for every circuit
               on the text
  swaps        the circuit's units set to their R values on each donor (near and far), in R and in the circuit's run:
               whether the circuit predicts R's contrast change d_m by its own d_c
  steering     each of the circuit's TOPK units (by its single removal's measured drop in R) scaled x2 and x-1 and set to
               its neuron's largest |value| on the calibration chats in its own sign (the pole it is on)
  weights      each of the TOPK units' neurons with its input and output weights zeroed (its value zero at every position);
               the model's own response beside it
               (the circuit's run for these three: every value outside the circuit frozen at its R value on the text,
               so the intervention's effect reaches the answer only through the circuit's units and the attention; they
               score 1 - sum |d_c - d_m| / max(sum |d_m|, sum |d_c|) over their draws: 1 a perfect prediction, 0 no
               better than predicting no change; none when both sums are under SHARE_MIN bits)
  isolation    (reported, no floor) every value outside the circuit zeroed: c / c_R
  simplicity   units as listed, unit-positions, distinct neurons, positions and layers, poles and SwiGLU sign regimes
               (the signs of gate and up) of the unit-positions in R, and the edges and heads the circuit states
Noise-aware floors: a family scored over draws passes at its floor less min(PAIRED_SE x its bootstrap standard error
over the draws, CAP of the family); necessity and side are one run each and get no allowance. A family left unscored
(a prediction family under SHARE_MIN, a prompt with no held-out counterfactual) neither passes nor fails. Every draw's
numbers are kept in the record, so scores can be recomputed (summary()).

  neuron_scorecard.py score CIRCUIT [CIRCUIT ...] --out OUT [--families FILE] [--dtype bfloat16] [--device auto]
      score circuit files (or directories of them) into OUT/<name>.json, OUT/rows.jsonl and the table
  neuron_scorecard.py table NAME=OUT [NAME=OUT ...]   builders side by side on the texts every set scored

A circuit file: neuron_builder's record ({"user", "nodes", "answer", "alternatives", "counterfactuals", "model",
"adapter", "calibration", "seconds", "builder"}), or {"ids", "target", "units" (or neuron_edges' "neurons"), "answer"
(default R's top token), "alternatives" (default R's 5 likeliest others), "model" (default Qwen/Qwen3-8B), "adapter"
(an organism's name or a directory or an HF id), "family" (owl, persona_k, facts: the near donors; else the name less
its last _number), "prompt", "thr", "edges", "heads", "build_seconds", "builder"}.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))

import neuron_builder as nb  # noqa: E402
from neuron_edges import log2_contrast  # noqa: E402

BASE = "Qwen/Qwen3-8B"
ORG = Path.home() / "mpd-data" / "organisms" / "qwen3_1p7b"
CACHE = Path.home() / "mpd-data" / "graph_oracle" / "neuron_scorecard"
FAR_ROWS = range(600, 1000)  # outside hf_means' chats (0-99), T2's held-out ones (100-299) and the calibration's (1000-1199)
DRAWS = 8  # donors of each kind; held-out counterfactuals
TOPK = 5
STEER = ("x2", "x-1", "max")
SINGLES_MAX = 512  # units measured singly to rank them; past it the rest by |value|
ROWS = 16  # runs a forward
TOL = 0.1  # neuron_edges.TOL: the share of the contrast a circuit may lose
FLOOR = {"necessity": nb.NEC_BITS, "side": nb.SIDE_MAX, "specificity": 0.05, "sufficiency": nb.SUFF_FRAC,
         "patching_near": 1 - TOL, "patching_far": 1 - TOL, "swaps": 1 - TOL, "steering": 1 - TOL, "weights": 1 - TOL}
DRAWN = ("specificity", "sufficiency", "patching_near", "patching_far", "swaps", "steering", "weights")
LOWER = ("side", "specificity")  # families whose score passes at or under its floor
PAIRED_SE = 2.0  # goal_eval's: a drawn family passes at its floor less this many standard errors, at most CAP
CAP: dict = {}  # per family, PAIRED_SE x the median standard error over the first dev set's circuits (until then uncapped)
EPS_BOOT = 200
SHARE_MIN = 0.1  # bits: below it in both the model's and the circuit's responses a prediction family is unscored


def read_circuit(path: Path) -> dict:
    """A circuit file in any of the accepted forms (module docstring) as {units, answer, alternatives, model, ...}; ids
    and target stay unset for a builder record (from its "user" on the model's tokenizer: prepare)."""
    c = json.loads(path.read_text())
    if isinstance(c.get("contrast"), dict):  # neuron_edges.graph's output
        k = c["contrast"]
        c.update(target=k["target"], answer=k["answer"], alternatives=k["alternatives"])
    units = c.get("units") or c.get("neurons") or [[n["layer"], n["pos"], n["neuron"]] for n in c.get("nodes", [])]
    heads = list(c.get("heads") or []) + [u for u in units if u[0] == "head"]  # native_qwen3's lists hold heads too
    units = [u[1:4] if u[0] == "neuron" else u[:3] for u in units if u[0] != "head"]
    ans, alt = c.get("answer"), c.get("alternatives")
    name = path.stem
    c.update(name=name, units=[tuple(int(x) for x in u) for u in units], heads=heads,
             answer=None if ans is None else [int(ans)] if isinstance(ans, int) else [int(z) for z in ans],
             alternatives=None if alt is None else [int(z) for z in alt], model=c.get("model") or BASE,
             prompt=c.get("prompt") or c.get("user"), family=c.get("family") or re.sub(r"_\d+$", "", c.get("id") or name),
             thr=c.get("thr") or (c.get("calibration") or {}).get("file"),
             build_seconds=c.get("build_seconds") or c.get("seconds"))
    return c


def prepare(c: dict, tok) -> dict:
    """ids and target from a builder record's user turn (neuron_builder.chat: the model's first answer token)."""
    if c.get("ids") is None:
        c["ids"] = nb.chat(tok, c["prompt"])
        c["target"] = c.get("pos", len(c["ids"]) - 1)
    return c


def load_model(base: str, adapter: str | None, dev: str, dtype: torch.dtype = torch.float32):
    """base (an HF id or directory) with adapter: a directory, an HF id, or an organism's name under ORG/adapters.
    Every run's neuron edits, logits and contrasts are in float32 whatever the weights' dtype (bfloat16 for 8B and
    32B on one or two L40s)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=dev if dev == "auto" else None)
    model = (model if dev == "auto" else model.to(dev)).eval()
    if adapter:
        from peft import PeftModel

        d = adapter if Path(adapter).is_dir() or "/" in adapter else str(ORG / "adapters" / adapter)
        model = PeftModel.from_pretrained(model, d).eval()
    model.requires_grad_(False)
    return model


def background(nr: nb.Neurons, key: str, given: str | None = None, log=print) -> dict:
    """R's file ({"thr" [L, D], "max" [L, D], "tau" [L]}): the circuit's (Neurons.save), else this model's cached one,
    else Neurons.calibrate on general.jsonl's CAL_ROWS chats as the builder's main runs it, cached. Leaves nr.thr unset
    (the runs zero weak values row by row)."""
    path = Path(given) if given and Path(given).exists() else CACHE / f"background_{key}.pt"
    if path.exists():
        nr.load(path)
    else:
        tok = nr.tok
        gen = [json.loads(x) for x in (ORG / "data" / "general.jsonl").read_text().splitlines()]
        seqs = [tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["answer"]}],
                                            tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"][:256]
                for r in gen[nb.CAL_ROWS]]
        cal = nr.calibrate(seqs, log=log)
        path.parent.mkdir(parents=True, exist_ok=True)
        nr.save(path, calibration=cal)
    bg = {"thr": nr.thr.float(), "max": nr.mx.float(), "massive": nr.massive.float(), "tau": nr.tau, "file": str(path)}
    nr.thr = None
    return bg


class Runs:
    """One circuit's runs on its text, ROWS a forward, logits at the answer and the builder's SIDE positions. A run is a
    dict: weak (R's weak zeroing; False: the model itself), circuit (every value outside the circuit replaced: by 0, or
    by a donor's R values with fill = its key, "clean" the text's own; massive values and the first position never),
    aff, a list of (layer, position, neuron, mul, add): that value becomes mul x it + add after the rest, and ids (another
    text of the same length: a counterfactual)."""

    def __init__(self, nr: nb.Neurons, ids: list[int], target: int, answer: list[int] | None, alternatives: list[int] | None,
                 bg: dict, units: list[tuple]):
        self.nr, self.dev, self.L, self.D = nr, nr.dev, nr.L, nr.D
        self.T, self.pos = target + 1, target
        self.ids = list(ids)[:self.T]
        self.thr, self.mx, self.massive = bg["thr"], bg["max"], bg["massive"]
        self.side = sorted({max(1, target - 1 - k * max(1, (target - 1) // nb.SIDE)) for k in range(nb.SIDE)} - {target}) if target > 1 else []
        self.allpos = self.side + [target]
        self.units = [u for u in units if len(u) == 2 or u[1] < self.T]
        self.cells = {u: self.expand(u) for u in self.units}  # each unit's (layer, position, neuron) cells
        self.keep = torch.zeros(self.L, self.T, self.D, dtype=torch.bool, device=self.dev)
        self.keep[:, 0] = True  # the first position is never outside the circuit
        for cs in self.cells.values():
            for L, p, j in cs:
                self.keep[L, p, j] = True
        self.donors = {}
        self.answer, self.alternatives = answer or [0], alternatives or [1]
        if answer is None:  # the builder's contrast: R's top token against its 5 likeliest others
            top = self.runs([{}], keep_lp=True)[0]["lp"][-1].topk(nb.ALTERNATIVES + 1).indices.tolist()
            self.answer, self.alternatives = top[:1], (alternatives or top[1:])
        self.R = self.runs([{}], keep_lp=True)[0]
        self.cR = self.R["contrast"]
        self.vals = self.capture()
        self.donors["clean"] = self.vals

    def expand(self, u: tuple) -> list[tuple]:
        return [(u[0], p, u[1]) for p in range(self.T)] if len(u) == 2 else [tuple(u)]

    @torch.no_grad()
    def capture(self, ids: list[int] | None = None) -> torch.Tensor:
        """R's values [L, T, D] after the weak zeroing, on the text (or on ids' first T tokens)."""
        cap = []
        self.runs([{"ids": ids}] if ids is not None else [{}], capture=cap)
        return torch.stack(cap)

    @torch.no_grad()
    def runs(self, specs: list[dict], capture: list | None = None, keep_lp: bool = False) -> list[dict]:
        out = []
        for s0 in range(0, len(specs), ROWS):
            part = specs[s0:s0 + ROWS]
            B, dev = len(part), self.dev
            weak = torch.tensor([s.get("weak", True) for s in part], device=dev)
            circ = torch.tensor([s.get("circuit", False) for s in part], device=dev)
            fills = [s.get("fill") for s in part]
            aff = {}
            for b, s in enumerate(part):
                for L, p, j, mul, add in s.get("aff", []):
                    aff.setdefault(L, []).append((b, p, j, mul, add))
            aff = {L: (torch.tensor([e[0] for e in v], device=dev), torch.tensor([e[1] for e in v], device=dev),
                       torch.tensor([e[2] for e in v], device=dev), torch.tensor([e[3] for e in v], device=dev, dtype=torch.float32),
                       torch.tensor([e[4] for e in v], device=dev, dtype=torch.float32)) for L, v in aff.items()}

            def edit(i, x):  # (each tensor to the layer's own device: a model split over GPUs)
                h = x.float()
                dv = h.device
                h = torch.where(weak.to(dv)[:, None, None] & (h.abs() < self.thr[i].to(dv)), torch.zeros((), device=dv), h)
                if bool(circ.any()):
                    fill = torch.stack([self.donors[f][i].to(dv) if f is not None else torch.zeros(self.T, self.D, device=dv) for f in fills])
                    out_ = circ.to(dv)[:, None, None] & ~self.keep[i].to(dv)[None] & (h.abs() < self.massive[i].to(dv))
                    h = torch.where(out_, fill, h)
                if i in aff:
                    b, p, j, mul, add = (t.to(dv) for t in aff[i])
                    h = h.index_put((b, p, j), h[b, p, j] * mul + add)
                if capture is not None:
                    capture.append(h[0].detach().to(dev).clone())
                return h.to(x.dtype)
            x = torch.tensor([list(s.get("ids") or self.ids)[:self.T] for s in part], device=dev)
            lp = torch.log_softmax(self.nr.logits(x, self.allpos, edit=edit).float(), -1)  # [B, P, V]
            c = log2_contrast(lp[:, -1], self.answer, self.alternatives)
            for b in range(B):
                r = {"contrast": float(c[b]), "top": int(lp[b, -1].argmax()) in self.answer}
                if keep_lp:
                    r["lp"] = lp[b].detach()
                if hasattr(self, "R") and self.side:
                    l0 = self.R["lp"][:-1]
                    r["side"] = float((l0.exp() * (l0 - lp[b, :-1])).sum(-1).mean() / math.log(2.0))
                out.append(r)
        return out

    @torch.no_grad()
    def text_kl(self, ids: list[int], aff: list[tuple]) -> float:
        """The mean KL (bits) over the first T positions of ids between R's next-token distributions and R's under aff."""
        T = min(self.T, len(ids))
        keep = [a for a in aff if a[1] < T]

        def edit(spec):
            def f(i, x):
                h = x.float()
                dv = h.device
                h = torch.where(h.abs() < self.thr[i].to(dv), torch.zeros((), device=dv), h)
                es = [e for e in spec if e[0] == i]
                if es:
                    p = torch.tensor([e[1] for e in es], device=dv)
                    j = torch.tensor([e[2] for e in es], device=dv)
                    h = h.index_put((torch.zeros_like(p), p, j), torch.zeros(len(es), device=dv))
                return h.to(x.dtype)
            return f
        lp0 = torch.log_softmax(self.nr.logits(list(ids)[:T], list(range(T)), edit=edit([]))[0].float(), -1)
        lp1 = torch.log_softmax(self.nr.logits(list(ids)[:T], list(range(T)), edit=edit(keep))[0].float(), -1)
        return float((lp0.exp() * (lp0 - lp1)).sum(-1).mean() / math.log(2.0))

    def add_donor(self, key, ids: list[int]):
        self.donors[key] = self.capture(ids)

    def zero(self, cells) -> list[tuple]:
        return [(L, p, j, 0.0, 0.0) for L, p, j in cells]

    def all_cells(self) -> list[tuple]:
        return [c for cs in self.cells.values() for c in cs]


def prediction(draws: list[dict]) -> float | None:
    """1 - sum |d_c - d_m| / max(sum |d_m|, sum |d_c|) over draws ({dm, dc}); none under SHARE_MIN in both sums."""
    sm, sc = sum(abs(d["dm"]) for d in draws), sum(abs(d["dc"]) for d in draws)
    if max(sm, sc) < SHARE_MIN:
        return None
    return 1.0 - sum(abs(d["dc"] - d["dm"]) for d in draws) / max(sm, sc)


def mean_clipped(xs: list[float]) -> float | None:
    xs = [min(1.0, max(-1.0, x)) for x in xs]
    return float(np.mean(xs)) if xs else None


def family_score(rec: dict, f: str, draws: list[dict] | None = None) -> float | None:
    """A family's score from the record's numbers (draws: a resample of the family's draws)."""
    cR = rec["contrast"]["R"]
    if f == "necessity":
        return cR - rec["necessity"]["R"]
    if f == "side":
        return rec["necessity"].get("side")
    d = rec["draws"][f] if draws is None else draws
    if f.startswith("patching"):
        return mean_clipped([x["c"] / cR for x in d]) if cR > 0 else None
    if f == "sufficiency":
        return mean_clipped([x["restored"] for x in d])
    if f == "specificity":
        return float(np.mean([x["kl"] for x in d])) if d else None
    return prediction(d)


def margins(rec: dict) -> dict:
    """Per family: score, floor, bootstrap SE over its draws, allowance and whether it passes the floor exactly (old),
    noise-aware (capped: score >= floor - min(PAIRED_SE se, CAP)) and uncapped (score >= floor - PAIRED_SE se)."""
    rng = np.random.default_rng(0)
    out = {}
    for f, fl in FLOOR.items():
        s = family_score(rec, f)
        se = 0.0
        if f in DRAWN and len(rec["draws"].get(f, [])) >= 2:
            d = rec["draws"][f]
            bs = [family_score(rec, f, [d[i] for i in idx]) for idx in rng.integers(len(d), size=(EPS_BOOT, len(d)))]
            bs = [x for x in bs if x is not None]
            se = float(np.std(bs, ddof=1)) if len(bs) > 1 else 0.0
        cap = CAP.get(f)
        allow = PAIRED_SE * se if cap is None else min(PAIRED_SE * se, cap)
        sg = -1 if f in LOWER else 1  # a LOWER family passes at or under its floor plus the allowance
        extra = f != "necessity" or not rec["necessity"]["answer_top"]  # the builder's necessary removal: the answer off the top

        def ok(a):
            return None if s is None else (sg * s >= sg * fl - a) and extra
        out[f] = {"score": s, "floor": fl, "se": se, "allowance": allow, "capped": cap is not None,
                  "pass_old": ok(0.0), "pass_new": ok(allow), "pass_uncapped": ok(PAIRED_SE * se)}
    return out


@torch.no_grad()
def regimes(nr: nb.Neurons, rs: Runs, cells: list[tuple]) -> dict:
    """The SwiGLU sign regime (gate and up signs) of each cell in R, counted ("g+u+", ...)."""
    gu = {}
    hooks = []
    for L, layer in enumerate(nr.layers):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(lambda m, a, o, L=L: gu.__setitem__(("g", L), o[0].float())))
        hooks.append(layer.mlp.up_proj.register_forward_hook(lambda m, a, o, L=L: gu.__setitem__(("u", L), o[0].float())))
    try:
        rs.runs([{}])
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for L, p, j in cells:
        k = f"g{'+' if gu[('g', L)][p, j] >= 0 else '-'}u{'+' if gu[('u', L)][p, j] >= 0 else '-'}"
        out[k] = out.get(k, 0) + 1
    return out


def chat(tok, user: str, answer: str | None = None) -> list[int]:
    msgs = [{"role": "user", "content": user}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


def donor_ids(tok, model, c: dict, families: dict, T: int, dev) -> dict:
    """{"near": [ids], "far": [ids]}: DRAWS donors of each kind with at least T tokens (module docstring)."""
    near = []
    for user in families.get(c.get("family"), []):
        if len(near) >= DRAWS:
            break
        if user == c.get("prompt"):
            continue
        ids = chat(tok, user)
        if ids[:T] == c["ids"][:T]:
            continue
        if len(ids) < T:
            g = model.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=T - len(ids), do_sample=False,
                               pad_token_id=tok.eos_token_id)
            ids = g[0].tolist()
        if len(ids) >= T:
            near.append(ids[:T])
    gen = (ORG / "data" / "general.jsonl").read_text().splitlines()
    far = []
    for n in FAR_ROWS:
        if len(far) >= DRAWS:
            break
        r = json.loads(gen[n])
        ids = chat(tok, r["user"], r["answer"])
        if len(ids) >= T:
            far.append(ids[:T])
    return {"near": near, "far": far}


def held_out_counterfactuals(rs: Runs, c: dict) -> list[tuple]:
    """neuron_builder.counterfactuals on the prompt, those lowering R's contrast by NEC_BITS or more, the builder's own
    (its record's "counterfactuals") excluded, the DRAWS largest drops: (position, token, ids, drop)."""
    rs.nr.thr = rs.thr
    try:
        alt = torch.tensor(rs.alternatives, device=rs.dev)

        def contrast(l):
            return (torch.logsumexp(l[:, rs.answer], -1) - torch.logsumexp(l[:, alt], -1)) / math.log(2)
        tries = nb.counterfactuals(rs.nr, rs.ids, rs.pos, contrast, rs.cR)
    finally:
        rs.nr.thr = None
    used = {(int(x["pos"]), int(x["token"])) for x in c.get("counterfactuals") or []}
    tries = [t for t in tries if t[3] >= nb.NEC_BITS and (t[0], t[1]) not in used]
    return sorted(tries, key=lambda t: -t[3])[:DRAWS]


@torch.no_grad()
def score(nr: nb.Neurons, model, c: dict, bg: dict, families: dict, log=print) -> dict:
    t0 = time.perf_counter()
    tok, dev = nr.tok, nr.dev
    rs = Runs(nr, c["ids"], c["target"], c["answer"], c["alternatives"], bg, c["units"])
    c["answer"], c["alternatives"] = rs.answer, rs.alternatives
    cells = rs.all_cells()
    cR = rs.cR
    rec = {"task": c["name"], "builder": c.get("builder"), "family": c.get("family"), "model": c["model"],
           "adapter": c.get("adapter"), "T": rs.T, "build_seconds": c.get("build_seconds"), "answer": rs.answer,
           "alternatives": rs.alternatives, "background": bg["file"]}
    model_run, iso, nec, nec_m = rs.runs([{"weak": False}, {"circuit": True}, {"aff": rs.zero(cells)},
                                          {"weak": False, "aff": rs.zero(cells)}])
    rec["contrast"] = {"R": cR, "model": model_run["contrast"], "R_top": rs.R["top"]}
    rec["necessity"] = {"R": nec["contrast"], "model": nec_m["contrast"], "answer_top": nec["top"], "side": nec.get("side"),
                        "answer_top_model": nec_m["top"]}
    rec["isolation"] = {"c": iso["contrast"], "top": iso["top"], "share": iso["contrast"] / cR if cR > 0 else None}
    # the units ranked by their single removal's measured drop in R
    order = sorted(rs.units, key=lambda u: -sum(abs(float(rs.vals[L, p, j])) for L, p, j in rs.cells[u]))
    meas = order[:SINGLES_MAX]
    singles = rs.runs([{"aff": rs.zero(rs.cells[u])} for u in meas])
    drops = {u: cR - s["contrast"] for u, s in zip(meas, singles)}
    top = sorted(meas, key=lambda u: -drops[u])[:TOPK]
    rec["singles"] = {"measured": len(meas), "top": [[list(u), drops[u]] for u in top],
                      "over_1bit": sum(d > 1 for d in drops.values())}
    draws = {"sufficiency": [], "patching_near": [], "patching_far": [], "swaps": [], "specificity": []}
    # sufficiency: the circuit's values on the prompt patched into held-out counterfactuals
    cfs = held_out_counterfactuals(rs, c)
    if cfs:
        sw = [(L, p, j, 0.0, float(rs.vals[L, p, j])) for L, p, j in cells]
        got = rs.runs([s for q, z, x2, d in cfs for s in ({"ids": x2}, {"ids": x2, "aff": sw})])
        for k, (q, z, x2, d) in enumerate(cfs):
            c_cf, c_p = got[2 * k]["contrast"], got[2 * k + 1]["contrast"]
            draws["sufficiency"].append({"pos": q, "token": z, "drop": d, "c_cf": c_cf, "c_patched": c_p,
                                         "restored": (c_p - c_cf) / (cR - c_cf) if cR - c_cf > 1e-6 else None})
        draws["sufficiency"] = [x for x in draws["sufficiency"] if x["restored"] is not None]
    # donors: patching, swaps and specificity
    dons = donor_ids(tok, model, c, families, rs.T, dev)
    gone = [(L, p, j, 0.0, 0.0) for L, j in sorted({(L, j) for L, _, j in cells}) for p in range(rs.T)]
    for kind in ("near", "far"):
        for n, ids in enumerate(dons[kind]):
            key = f"{kind}{n}"
            rs.add_donor(key, ids)
            sw = [(L, p, j, 0.0, float(rs.donors[key][L, p, j])) for L, p, j in cells]
            pr, smr, smc = rs.runs([{"circuit": True, "fill": key}, {"aff": sw}, {"circuit": True, "fill": "clean", "aff": sw}])
            draws[f"patching_{kind}"].append({"donor": key, "c": pr["contrast"]})
            draws["swaps"].append({"donor": key, "dm": smr["contrast"] - cR, "dc": smc["contrast"] - cR})
            if kind == "far":
                draws["specificity"].append({"donor": key, "kl": rs.text_kl(ids, gone)})
            del rs.donors[key]
    # steering and weight edits on the top units (the circuit's run: the outside frozen at the text's R values)
    specs, meta = [], []
    for u in top:
        for kind in STEER:
            aff = []
            for L, p, j in rs.cells[u]:
                a = float(rs.vals[L, p, j])
                if kind == "x2":
                    aff.append((L, p, j, 2.0, 0.0))
                elif kind == "x-1":
                    aff.append((L, p, j, -1.0, 0.0))
                elif a != 0:
                    aff.append((L, p, j, 0.0, math.copysign(float(rs.mx[L, j]), a)))
            if aff:
                specs += [{"aff": aff}, {"circuit": True, "fill": "clean", "aff": aff}]
                meta.append(("steering", list(u), kind))
    for L, j in dict.fromkeys((u[0], u[-1]) for u in top):
        aff = [(L, p, j, 0.0, 0.0) for p in range(rs.T)]
        specs += [{"aff": aff}, {"circuit": True, "fill": "clean", "aff": aff}, {"weak": False, "aff": aff}]
        meta.append(("weights", [L, j], "zero"))
    got = rs.runs(specs)
    draws["steering"], draws["weights"] = [], []
    i = 0
    for fam, u, kind in meta:
        d = {"unit": u, "kind": kind, "dm": got[i]["contrast"] - cR, "dc": got[i + 1]["contrast"] - cR}
        if fam == "weights":
            d["model"] = got[i + 2]["contrast"] - model_run["contrast"]
            i += 3
        else:
            i += 2
        draws[fam].append(d)
    rec["draws"] = draws
    poles = {"+": 0, "-": 0, "0": 0}
    for L, p, j in cells:
        a = float(rs.vals[L, p, j])
        poles["+" if a > 0 else "-" if a < 0 else "0"] += 1
    rec["simplicity"] = {"units": len(rs.units), "cells": len(cells), "neurons": len({(L, j) for L, p, j in cells}),
                         "positions": len({p for _, p, _ in cells}), "layers": len({L for L, _, _ in cells}), "poles": poles,
                         "regimes": regimes(nr, rs, cells), "edges": len(c.get("edges") or []), "heads": len(c.get("heads") or [])}
    rec["margins"] = margins(rec)
    rec["seconds"] = time.perf_counter() - t0
    fails = [f for f, m in rec["margins"].items() if m["pass_new"] is False]
    log(f"{c['name']}: {len(rs.units)} units ({len(cells)} unit-positions); contrast R {cR:.2f} model {model_run['contrast']:.2f} "
        f"removed {nec['contrast']:.2f} isolated {iso['contrast']:.2f} bits; {len(cfs)} held-out counterfactuals; "
        f"failing {fails or 'none'}; {rec['seconds']:.1f} s")
    return rec


def summary(recs: list[dict]) -> dict:
    """Per family: circuits passing (old, noise-aware, uncapped) and the median score; sizes and seconds."""
    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None
    out = {"circuits": len(recs), "families": {}}
    every = [margins(r) for r in recs]
    for f in FLOOR:
        ms = [m[f] for m in every]
        out["families"][f] = {"score": med([m["score"] for m in ms]), "old": sum(m["pass_old"] is True for m in ms),
                              "noise": sum(m["pass_new"] is True for m in ms),
                              "uncapped": sum(m["pass_uncapped"] is True for m in ms),
                              "unscored": sum(m["score"] is None for m in ms)}
    out["every_family"] = {k: sum(all(m[f][k] is not False for f in FLOOR) for m in every) for k in ("pass_old", "pass_new", "pass_uncapped")}
    for k in ("units", "cells", "neurons", "positions", "layers", "edges"):
        out[k] = med([r["simplicity"][k] for r in recs])
    out["isolation"] = med([r["isolation"]["share"] for r in recs])
    out["model_necessity"] = med([r["contrast"]["model"] - r["necessity"]["model"] for r in recs])
    out["build_seconds"] = med([r.get("build_seconds") for r in recs])
    out["score_seconds"] = med([r.get("seconds") for r in recs])
    return out


def table(sets: dict) -> str:
    """The sets (name -> records) side by side on the texts every set scored."""
    common = set.intersection(*[{r["task"] for r in v} for v in sets.values()]) if sets else set()
    sm = {n: summary([r for r in v if r["task"] in common]) for n, v in sets.items()}
    w = {n: max(14, len(n) + 2) for n in sets}
    lines = [f"{len(common)} texts scored by every set; old: the floor exactly; noise: less min({PAIRED_SE:g} SE, cap); "
             f"uncapped: less {PAIRED_SE:g} SE" + ("" if CAP else " (no cap measured yet: noise = uncapped)"),
             f"{'':42}" + "".join(f"{n:>{w[n]}}" for n in sets)]

    def row(label, get, fmt="{:.3f}"):
        vals = []
        for n in sets:
            v = get(sm[n])
            vals.append(f"{(fmt.format(v) if isinstance(v, float) else str(v)) if v is not None else '-':>{w[n]}}")
        lines.append(f"  {label:40}" + "".join(vals))
    for f in FLOOR:
        row(f"{f} ({'at most' if f in LOWER else 'floor'} {FLOOR[f]:g}) median", lambda s, f=f: s["families"][f]["score"])
        row("  pass old / noise / uncapped / unscored", lambda s, f=f: "{old}/{noise}/{uncapped}/{unscored}".format(**s["families"][f]))
    row("every family old / noise / uncapped", lambda s: "{pass_old}/{pass_new}/{pass_uncapped}".format(**s["every_family"]))
    row("isolation share (no floor)", lambda s: s["isolation"])
    row("necessity drop in the model itself (bits)", lambda s: s["model_necessity"])
    for k in ("units", "cells", "neurons", "positions", "layers", "edges"):
        row(k, lambda s, k=k: s[k], "{:.0f}")
    row("build seconds", lambda s: s["build_seconds"], "{:.1f}")
    row("scoring seconds", lambda s: s["score_seconds"], "{:.1f}")
    return "\n".join(lines)


def load(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "table"))
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--families", type=Path, default=CACHE / "families.json",
                    help="{family: [user prompts]}: the near donors")
    ap.add_argument("--device", default=None, help="cuda, mps, cpu, or auto (the weights over every GPU: 32B)")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    if a.cmd == "table":
        print(table({p.split("=", 1)[0]: load(Path(p.split("=", 1)[1])) for p in a.paths}))
        return
    from transformers import AutoTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    files = []
    for p in map(Path, a.paths):
        files += sorted(x for x in p.glob("*.json")) if p.is_dir() else [p]
    circuits = [read_circuit(p) for p in files]
    families = json.loads(a.families.read_text()) if a.families.exists() else {}
    a.out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for c in circuits:
        if not (a.out / f"{c['name']}.json").exists():
            groups.setdefault((c["model"], c.get("adapter")), []).append(c)
    for (base, adapter), cs in groups.items():
        tok = AutoTokenizer.from_pretrained(base)
        model = load_model(base, adapter, dev, getattr(torch, a.dtype))
        nr = nb.Neurons(model, tok)
        key = f"{base.split('/')[-1]}_{Path(adapter).name if adapter else 'base'}"
        for c in cs:
            bg = background(nr, key, c.get("thr"), log=lambda m: print(m, flush=True))
            rec = score(nr, model, prepare(c, tok), bg, families, log=lambda m: print(m, flush=True))
            (a.out / f"{c['name']}.json").write_text(json.dumps(rec))
            with open(a.out / "rows.jsonl", "a") as f:
                f.write(json.dumps({k: rec[k] for k in ("task", "builder", "contrast", "necessity", "isolation", "simplicity", "margins", "seconds")}) + "\n")
        del model, nr
    recs = load(a.out)
    (a.out / "summary.json").write_text(json.dumps(summary(recs)))
    print(table({a.out.name: recs}))


if __name__ == "__main__":
    main()
