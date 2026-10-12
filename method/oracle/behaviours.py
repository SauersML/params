"""Behaviour goals by discovery (#2951 graph oracle, E3): persistent edit goals over what vpd4l does, drawn by how much
it matters on held-out Pile, each with instances, crossed controls and generalization texts found by measurement.

Sampling (sample()). A behaviour starts from one VPD subcomponent c drawn with probability proportional to its usage on
split B (the atlas's heldout_rate: the share of positions where it is active) times its measured removal effect there
(the mean KL of the atlas's effect samples), so common mechanisms come up as often as they work in text. The copy
mistakes (edit_eval's copy goal) stay a fixed regression check outside the sample.

The behaviour of c: the model predicting one of S_c, the tokens c's removal lowers most at its effect samples (the union
of their top raised tokens), where c is active. On split B (rows 5000-5999, the atlas's held-out rows; c's activity from
parts.py's activity files), each third of the rows separately (THIRDS: fit, validation, score):
  instances   positions where c is active, the model's most likely token is in S_c, and removing c there (its activation
              at that position set to 0) lowers the margin of S_c (log2 of the best token of S_c over the best other)
  controls    positions where the model's most likely token is in S_c and c is inactive: the same tokens predicted
              outside the behaviour (crossed controls: change the rule, keep the knowledge)
Score of a persistent edit (score_many), in bits, on one third's rows (each cut after its last instance or control):
  goal_bits   the instances' mean of the margin's drop, at most the margin (crossing out of S_c: the behaviour gone)
  progress    the instances' mean of that drop over the margin, in [0, 1]; flipped: the share that crossed
  collateral  KL summed over each row's positions that are not instances (its controls among them), the mean over the
              rows, plus the held Pile texts' mean summed KL (edit_eval.HELD_ROWS; the search uses SEARCH_ROWS)
  controls    kept: the share of controls whose most likely token is still in S_c; control_kl: their mean KL
  net         bits per position of text, as a persistent edit acts on all text: rate x goal_bits (rate: instances per
              position of the third, measured while finding them) - collateral_per_token (each row's KL per position
              that is not an instance, averaged over the rows, plus the held texts' KL per position); net_per_text is
              goal_bits - collateral
Search (teach()): edit_eval.frontier over persistent edits of the subcomponents most often active at the fit instances
(screened by measured ablation at fit instances) and cuts among the top ones, the same over native units, each step
proposed by the fit third and chosen by the validation third, the path's step with the best validation net taken; the
behaviour's own subcomponent ablated everywhere and VPD's most active subcomponents at the fit instances are the
baselines. Everything is scored on the score third.
Remove, not hide (circuit_check()): the circuit is the fit instances' subcomponents with the largest measured local
ablation effect on the margin before the edit; with the edit applied the same ablations are measured again, and
retained is the share of their summed effect left. An edit that removes the behaviour leaves the circuit without effect
(retained near 0); one that blocks it at the output with a counter-signal leaves the circuit pushing (retained near 1).

  behaviours.py sample OUT.jsonl [--k 16] [--seed 0]     sampled behaviours, their instances and controls (a GPU or the CPU)
  behaviours.py teach BEHAVIOURS.jsonl OUT.jsonl [--offset K --stride S] [--n N]   edits and checks per behaviour
  behaviours.py table OUT.jsonl                           the report
  behaviours.py rescore OUT.jsonl NEW.jsonl               taught behaviours scored again (current definitions)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))
sys.path.insert(0, str(HERE.parents[1]))

import edit_eval  # noqa: E402
import native  # noqa: E402

ATLAS = Path.home() / "mpd-data/atlas/vpd4l"
ROWS = (5000, 6000)  # split B
THIRDS = {"fit": (5000, 5333), "val": (5333, 5666), "score": (5666, 6000)}
T1 = 511  # positions per row in the activity files (position 0 left out)
CAP = {"fit": 24, "val": 24, "score": 64}  # instances (and controls) per third
SCREEN_N = 12  # fit instances the candidates' local ablations are measured on
POOL = 128  # subcomponent and neuron candidates after screening
CUT_TOP = 24  # the top subcomponents whose connections are cut candidates
CIRCUIT = 16  # the circuit's subcomponents in the remove-not-hide check
SCALES = (0.0, -1.0)  # persistent scales of a candidate
LN2 = math.log(2)


# ---- the atlas side


class Source:
    """The atlas's arrays, split B's tokens and activity, and the subcomponents' flat numbering."""

    def __init__(self, root: Path = ATLAS):
        from safetensors.numpy import load_file

        from method.vpd.parts import Activity

        self.root = Path(root)
        self.meta = json.loads((self.root / "atlas.json").read_text())
        self.a = load_file(str(self.root / "atlas.safetensors"))
        self.names = self.meta["names"]
        counts = [self.meta["counts"][n] for n in self.names]
        self.offset = dict(zip(self.names, np.concatenate([[0], np.cumsum(counts)])[:-1].tolist()))
        self.per_layer = sum(counts[:6])
        self.tokens = np.load(self.root / "tokens_0_6000.npy", mmap_mode="r")
        self.act = [Activity(self.root / "parts" / f"activity_heldout_L{layer}.npz") for layer in range(4)]

    def node(self, flat: int) -> tuple[str, int]:
        site = max((n for n in self.names if self.offset[n] <= flat), key=lambda n: self.offset[n])
        return site, flat - self.offset[site]

    def flat(self, site: str, index: int) -> int:
        return self.offset[site] + index

    def active(self, flat: int) -> set[tuple[int, int]]:
        """(row, position) of split B where the subcomponent is active."""
        layer = flat // self.per_layer
        g = self.act[layer].of(flat - layer * self.per_layer).astype(np.int64)
        return set(zip((g // T1 + ROWS[0]).tolist(), (g % T1 + 1).tolist()))

    def active_at(self, places: list[tuple[int, int]]) -> dict[int, int]:
        """flat subcomponent -> how many of the (row, position) places it is active at."""
        want = np.array(sorted({(r - ROWS[0]) * T1 + p - 1 for r, p in places}), dtype=np.int64)
        count = {}
        for layer, act in enumerate(self.act):
            idx = np.nonzero(np.isin(act.positions, want))[0]
            rows = np.searchsorted(act.indptr, idx, side="right") - 1 + layer * self.per_layer
            for f, k in zip(*np.unique(rows, return_counts=True)):
                count[int(f)] = int(k)
        return count

    def raised(self, flat: int) -> list[int]:
        """S_c: the union of the top tokens c's removal lowers at its effect samples."""
        a = self.a
        toks = {int(t) for k in range(a["effect_row"].shape[1]) if a["effect_row"][flat, k] >= 0
                for t in a["effect_up_token"][flat, k] if t >= 0}
        return sorted(toks)

    def describe(self, flat: int, decode) -> dict:
        """What the atlas says about c: its trigger contexts, the tokens its removal lowers, its lenses."""
        a = self.a
        ctx = []
        for k in range(min(5, a["top_row"].shape[1])):
            if a["top_row"][flat, k] < 0:
                continue
            w = [int(x) for x in a["top_window"][flat, k]]
            ctx.append(decode(w))
        up = [decode([t]) for t in a["effect_up_token"][flat, 0][:6] if t >= 0] if a["effect_row"][flat, 0] >= 0 else []
        lens = [decode([t]) for t in a["lens_up_token"][flat][:6] if t >= 0]
        return {"contexts": ctx, "removal_lowers": up, "logit_lens": lens, "heldout_rate": float(a["heldout_rate"][flat]),
                "effect_kl": float(np.nanmean(np.where(a["effect_row"][flat] >= 0, a["effect_kl"][flat], np.nan)))}


def weights(src: Source) -> np.ndarray:
    """Each subcomponent's sampling weight: its split-B usage times its mean measured removal KL (0 without samples)."""
    a = src.a
    kl = np.where(a["effect_row"] >= 0, a["effect_kl"], np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nan_to_num(np.nanmean(kl, 1), nan=0.0)
    return a["heldout_rate"].astype(np.float64) * np.clip(mean, 0, None)  # a KL below 0 is float rounding


# ---- measured instances


def _margin_s(lq: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """log2 of the best token of S over the best other token, per row of lq [..., V] (S a bool mask [V])."""
    return (lq.masked_fill(~S, -math.inf).max(-1).values - lq.masked_fill(S, -math.inf).max(-1).values) / LN2


def build(ee: edit_eval.EditEval, src: Source, flat: int, decode, log=print) -> dict | None:
    """A behaviour from subcomponent flat: S_c, and per third its instances and controls (module docstring); None when
    the fit or score third has no instance."""
    site, index = src.node(flat)
    S_list = src.raised(flat)
    if not S_list:
        return None
    V = ee.nat.target.wte.shape[0]
    S = torch.zeros(V, dtype=torch.bool, device=ee.nat.dev)
    S[S_list] = True
    act = src.active(flat)
    rng = random.Random(flat)
    out = {"subcomponent": [site, index], "flat": flat, "S": S_list, "describe": src.describe(flat, decode), "parts": {}}
    for part, (r0, r1) in THIRDS.items():
        rows = sorted({r for r, _ in act if r0 <= r < r1})
        rng.shuffle(rows)
        inst, ctrl, scanned = [], [], 0
        for row in rows[:4 * CAP[part]]:
            scanned += 1
            ids = src.tokens[row, :512].astype(np.int64).tolist()
            lp = ee.reference(ids)
            top_in = S[lp.argmax(-1)].cpu().numpy()
            here = [p for r, p in act if r == row and top_in[p]]
            if here:  # c contributes: removing it at the position lowers the margin there
                m0 = _margin_s(lp[here], S)
                eds = [[edit_eval.scale_op((site, p, index), 0.0)] for p in here]
                ms = ee._runs([ids] * len(eds), eds, lambda lq, idx, s, lp_: [float(_margin_s(lq[j, p - s], S)) for j, p in
                                                                              zip(range(len(idx)), [here[i] for i in idx])])
                inst += [(row, p) for p, a, b in zip(here, m0.tolist(), ms) if b < a]
            ctrl += [(row, p) for p in np.nonzero(top_in)[0].tolist() if (row, p) not in act and p > 0]
            if len(inst) >= CAP[part]:
                break
        rng.shuffle(ctrl)
        rate = len(inst) / max(1, scanned) * len(rows) / (r1 - r0) / T1  # instances per position of the third's text
        out["parts"][part] = {"instances": inst[:CAP[part]], "controls": ctrl[:CAP[part]], "rate": rate}
    if not out["parts"]["fit"]["instances"] or not out["parts"]["score"]["instances"]:
        return None
    log(f"{site}#{index}: S {len(S_list)} tokens; instances " + ", ".join(f"{k} {len(v['instances'])}" for k, v in out["parts"].items())
        + "; controls " + ", ".join(f"{k} {len(v['controls'])}" for k, v in out["parts"].items()))
    return out


def sample(ee: edit_eval.EditEval, src: Source, k: int, seed: int, decode, log=print) -> list[dict]:
    """k behaviours by usage x effect sampling without replacement (draws without instances are redrawn)."""
    w = weights(src)
    rng = np.random.default_rng(seed)
    out, tried = [], set()
    while len(out) < k and len(tried) < 8 * k:
        p = w.copy()
        p[list(tried)] = 0
        if p.sum() <= 0:
            break
        flat = int(rng.choice(len(p), p=p / p.sum()))
        tried.add(flat)
        b = build(ee, src, flat, decode, log)
        if b is not None:
            b["weight"] = float(w[flat] / w.sum())
            out.append(b)
    return out


def goal_of(b: dict, decode) -> dict:
    """The behaviour as an edit goal in the token-level goals' grammar (components.edit_goal_text shows "question")."""
    d = b["describe"]
    toks = ", ".join(repr(t) for t in d["removal_lowers"][:4])
    ctx = d["contexts"][0] if d["contexts"] else ""
    site, index = b["subcomponent"]
    return {"kind": "behaviour", "name": f"{site}#{index}", "persistent": True,
            "question": f"make the model stop predicting tokens like {toks} in contexts like {ctx!r}, on every text, while "
                        f"it keeps predicting them elsewhere (persistent)."}


# ---- the score


class BehaviourEval:
    """Scores persistent edits on a behaviour (module docstring) with an edit_eval.EditEval's runs."""

    def __init__(self, ee: edit_eval.EditEval, src: Source, b: dict):
        self.ee, self.b = ee, b
        V = ee.nat.target.wte.shape[0]
        self.S = torch.zeros(V, dtype=torch.bool, device=ee.nat.dev)
        self.S[b["S"]] = True
        self.texts = {}
        for part, d in b["parts"].items():
            by = {}
            for kind in ("instances", "controls"):
                for r, p in d[kind]:
                    by.setdefault(r, {"instances": [], "controls": []})[kind].append(p)
            self.texts[part] = [(r, src.tokens[r, :max(v["instances"] + v["controls"]) + 1].astype(np.int64).tolist(), v)
                                for r, v in sorted(by.items())]

    def score_many(self, edits: list[list[dict]], part: str = "score") -> list[dict]:
        ee, S = self.ee, self.S
        texts = self.texts[part]
        T = max(len(ids) for _, ids, _ in texts)
        rows = [ee._row(ops, edit_eval.check(ops, T, [T - 1], behaviour=True), T) for ops in edits]
        ok = [b for b, r in enumerate(rows) if r["why"] is None]
        if not ok:
            return rows
        eds = [edits[b] for b in ok]
        seqs, es, owner = [], [], []
        for j, ops in enumerate(eds):
            for i, (_, ids, _) in enumerate(texts):
                seqs.append(ids)
                es.append(ops)
                owner.append((j, i))

        def fn(lq, idx, s, lp):  # one text per batch, persistent edits (s = 0)
            _, _, v = texts[owner[idx[0]][1]]
            inst, ctrl = v["instances"], v["controls"]
            kl = self.ee._kl(lp, lq)  # [m, L]
            m0 = _margin_s(lp[inst], S)  # [I]
            m = _margin_s(lq[:, inst], S)  # [m, I]
            gb = torch.minimum(m0[None], m0[None] - m).clamp_min(0)
            prog = (gb / m0[None].clamp_min(1e-9)).clamp(0, 1)
            kept = S[lq[:, ctrl].argmax(-1)].float().sum(1) if ctrl else torch.zeros(len(idx), device=lq.device)
            ckl = kl[:, ctrl].sum(1) if ctrl else torch.zeros(len(idx), device=lq.device)
            coll = kl.sum(1) - kl[:, inst].sum(1)
            per = coll / max(1, kl.shape[1] - len(inst))
            return [{"goal": g, "progress": p, "flipped": f, "n": len(inst), "kept": k, "ckl": c, "nc": len(ctrl), "coll": x, "per": y}
                    for g, p, f, k, c, x, y in zip(gb.sum(1).tolist(), prog.sum(1).tolist(), (m < 0).float().sum(1).tolist(),
                                                   kept.tolist(), ckl.tolist(), coll.tolist(), per.tolist())]
        per = ee._runs(seqs, es, fn, ref=True)
        held = ee._held_terms("held" if part == "score" else "search", eds)
        key = edit_eval.EditEval.key(f"behaviour:{part}", {"kind": "behaviour", "name": "/".join(map(str, self.b["subcomponent"]))})
        k = len(texts)
        for j, b in enumerate(ok):
            rs = per[j * k:(j + 1) * k]
            n, nc = sum(r["n"] for r in rs), sum(r["nc"] for r in rs)
            coll = sum(r["coll"] for r in rs) / k
            rows[b].update({"valid": True, "progress": sum(r["progress"] for r in rs) / n, "flipped": sum(r["flipped"] for r in rs) / n,
                            "attained": sum(r["flipped"] for r in rs) / n >= edit_eval.GOAL,
                            "controls_kept": sum(r["kept"] for r in rs) / nc if nc else None,
                            "control_kl": sum(r["ckl"] for r in rs) / nc if nc else None,
                            "collateral_text": coll, "collateral_held": held[j]})
            ee._finish(rows[b], key, sum(r["goal"] for r in rs) / n, coll + held[j])
            rate = self.b["parts"][part].get("rate")
            if rate is not None:  # a persistent edit's net per position of text: the behaviour's bits where it occurs
                gt = rate * rows[b]["goal_bits"]  # against the KL at every other position
                ct = sum(r["per"] for r in rs) / k + held[j] / edit_eval.HELD_LEN
                rows[b].update({"net_per_text": rows[b]["net"], "rate": rate, "goal_bits_per_token": gt,
                                "collateral_per_token": ct, "net": gt - ct})
        return rows

    def local_effects(self, nodes: list[tuple[str, int]], base: list[dict], part: str = "fit", n: int = SCREEN_N) -> np.ndarray:
        """Per node, the mean drop of the margin at the first n instances of part when the node's activation there is
        set to 0, with the persistent edit base applied (the drop from base's own margin)."""
        S, ee = self.S, self.ee
        places = [(ids[:p + 1], p) for _, ids, v in self.texts[part] for p in v["instances"]][:n]  # nothing after p matters
        total = np.zeros(len(nodes))
        for ids, p in places:
            eds = [list(base)] + [list(base) + [edit_eval.scale_op((nd[0], p, nd[1]), 0.0)] for nd in nodes]
            ms = ee._runs([ids] * len(eds), eds, lambda lq, idx, s, lp, p=p: _margin_s(lq[:, p - s], S).tolist())
            total += ms[0] - np.array(ms[1:])
        return total / max(1, len(places))


# ---- the search


def candidates(src: Source, be: BehaviourEval, log=print) -> tuple[list[tuple[str, int]], list[dict], list[dict]]:
    """(subcomponents ranked by their measured local ablation effect at fit instances, the subcomponent pool, the
    native-unit pool)."""
    fit = [(r, p) for r, _, v in be.texts["fit"] for p in v["instances"]]
    count = src.active_at(fit)
    nodes = [src.node(f) for f, k in sorted(count.items(), key=lambda x: -x[1]) if k >= 2]
    eff = be.local_effects(nodes, [])
    order = np.argsort(-eff, kind="stable")[:POOL]
    top = [nodes[i] for i in order]
    log(f"  {len(nodes)} subcomponents active at >= 2 fit instances; best local effect {eff[order[0]]:.3f} bits")
    pool = [edit_eval.scale_op((s, None, i), k) for s, i in top for k in SCALES]
    lead = [(s, None, i) for s, i in top[:CUT_TOP]]
    pool += [edit_eval.cut_op(r, w) for r in lead for w in lead if edit_eval._connected(w, r, [0], True)]
    tg = be.ee.nat.target
    width = tg.site(native.site_name(0, "c_fc")).W.shape[0]
    neurons = [(layer, j) for layer in range(tg.n_layer) for j in range(width)]
    ueff = _unit_effects(be, neurons)
    upool = [edit_eval.unit_op("head", layer, h, None, k) for layer in range(tg.n_layer) for h in range(tg.n_head) for k in SCALES]
    upool += [edit_eval.unit_op("neuron", neurons[i][0], neurons[i][1], None, k) for i in np.argsort(-ueff, kind="stable")[:POOL]
              for k in SCALES]
    return top, pool, upool


def _unit_effects(be: BehaviourEval, neurons: list[tuple[int, int]], n: int = SCREEN_N) -> np.ndarray:
    S, ee = be.S, be.ee
    places = [(ids[:p + 1], p) for _, ids, v in be.texts["fit"] for p in v["instances"]][:n]
    total = np.zeros(len(neurons))
    for ids, p in places:
        eds = [[]] + [[edit_eval.unit_op("neuron", layer, j, p, 0.0)] for layer, j in neurons]
        ms = ee._runs([ids] * len(eds), eds, lambda lq, idx, s, lp, p=p: _margin_s(lq[:, p - s], S).tolist())
        total += ms[0] - np.array(ms[1:])
    return total / max(1, len(places))


def circuit_check(be: BehaviourEval, circuit: list[tuple[str, int]], before: np.ndarray, edit: list[dict]) -> dict:
    """Remove, not hide (module docstring; edit_eval.removal_check)."""
    effects = lambda nds, base: be.local_effects([(n, i) for n, _, i in nds], base)  # noqa: E731
    return edit_eval.removal_check(effects, [(n, None, i) for n, i in circuit], before, edit)


def teach(ee: edit_eval.EditEval, src: Source, b: dict, log=print) -> dict:
    """One behaviour: the frontier searches (subcomponents, native units) with validation, the baselines, every
    chosen edit scored on the score third with its remove-not-hide check."""
    t0 = time.perf_counter()
    be = BehaviourEval(ee, src, b)
    top, pool, upool = candidates(src, be, log)
    circuit = top[:CIRCUIT]
    before = be.local_effects(circuit, [])
    out = {"behaviour": {k: v for k, v in b.items() if k != "parts"},
           "counts": {k: {x: len(y) for x, y in v.items()} for k, v in b["parts"].items()}}
    fit = lambda eds: be.score_many(eds, "fit")  # noqa: E731
    val = lambda eds: be.score_many(eds, "val")  # noqa: E731
    for name, pl in (("teacher", pool), ("units", upool)):
        log(f"  {name}: {len(pl)} candidates")
        res = edit_eval.frontier(fit, pl, max_ops=8, log=log, validate=val)
        if res["chosen"] is None:
            out[name] = None
            continue
        edit = [edit_eval._from_json(o) for o in res["path"][res["chosen"]]["edit"]]
        out[name] = {"path": [{k: st[k] for k in ("ops", "edit", "row", "validation")} for st in res["path"]], "chosen": res["chosen"],
                     "ops": res["path"][res["chosen"]]["ops"], "edit": res["path"][res["chosen"]]["edit"],
                     "score": be.score_many([edit])[0], "check": circuit_check(be, circuit, before, edit)}
    site, index = b["subcomponent"]
    own = [edit_eval.scale_op((site, None, index), 0.0)]
    out["own"] = {"ops": [edit_eval.op_text(o) for o in own], "score": be.score_many([own])[0],
                  "check": circuit_check(be, circuit, before, own)}
    fit_places = [(r, p) for r, _, v in be.texts["fit"] for p in v["instances"]]
    by_count = [src.node(f) for f, _ in sorted(src.active_at(fit_places).items(), key=lambda x: -x[1])]
    out["vpd_topk"] = [{"k": k, "score": r} for k, r in zip((1, 4, 16), be.score_many(
        [[edit_eval.scale_op((s, None, i), 0.0) for s, i in by_count[:k]] for k in (1, 4, 16)]))]
    out["seconds"] = time.perf_counter() - t0
    return out


def ops_of(texts: list[str]) -> list[dict]:
    """Operations from their text (edit_eval.op_text: the answer grammar, or native units)."""
    import re

    import components

    out = []
    for t in texts:
        m = re.fullmatch(r"(persistent )?scale (head|neuron) L(\d+)\.[hn](\d+)(?:@(\d+))? (-?[\d.]+)", t.strip())
        if m:
            out.append(edit_eval.unit_op(m[2], int(m[3]), int(m[4]), None if m[5] is None else int(m[5]), float(m[6])))
        else:
            out += components.parse(f"EDIT 1: {t}")["edits"][1]
    return out


def rescore(ee: edit_eval.EditEval, src: Source, r: dict, log=print) -> dict:
    """A taught behaviour scored again by the current definitions: the behaviour rebuilt (the same instances: its draws
    are seeded by the subcomponent), each search path's steps measured on the validation third, the step with the
    most validation net chosen and scored on the score third with its check, the baselines scored again."""
    flat = src.flat(*r["behaviour"]["subcomponent"])
    b = build(ee, src, flat, _decoder(), log)
    be = BehaviourEval(ee, src, b)
    for name in ("teacher", "units"):
        m = r.get(name)
        if not m:
            continue
        steps = [st.get("edit") and [edit_eval._from_json(o) for o in st["edit"]] or ops_of(st["ops"]) for st in m["path"]]
        vrows = be.score_many(steps, "val")
        k = int(np.argmax([v["net"] if v.get("valid") else -np.inf for v in vrows]))
        m.update({"chosen": k, "ops": m["path"][k]["ops"], "edit": [edit_eval._jsonable(o) for o in steps[k]],
                  "validation": vrows, "score": be.score_many([steps[k]])[0]})
    site, index = b["subcomponent"]
    r["own"]["score"] = be.score_many([[edit_eval.scale_op((site, None, index), 0.0)]])[0]
    r["behaviour"]["parts_rate"] = {k: v["rate"] for k, v in b["parts"].items()}
    return r


# ---- command line


def _decoder():
    from tokenizers import Tokenizer

    import vpd_model

    tok = Tokenizer.from_file(str(vpd_model.TARGET_DIR / "tokenizer.json"))
    return lambda ids: tok.decode([int(i) for i in ids])


def table(rows: list[dict]) -> str:
    lines = ["behaviour (sampled subcomponent; removal lowers) | method: progress flipped controls_kept collateral net "
             "retained ops"]
    for r in rows:
        d = r["behaviour"]["describe"]
        lines.append(f"{'/'.join(map(str, r['behaviour']['subcomponent']))} weight {r['behaviour'].get('weight', 0):.4f}; "
                     f"lowers {d['removal_lowers'][:4]}; context {d['contexts'][0][-40:] if d['contexts'] else ''!r}; "
                     f"instances {r['counts']}")
        for name in ("teacher", "units", "own"):
            m = r.get(name)
            if not m or not m["score"].get("valid"):
                continue
            s, c = m["score"], m.get("check", {})
            ck = s["controls_kept"]
            lines.append(f"  {name:8s} {s['progress']:.3f} {s['flipped']:.2f} {ck if ck is None else round(ck, 2)} "
                         f"{s['collateral']:.2f} {s['net']:.2f} {c.get('retained') if c.get('retained') is None else round(c['retained'], 2)} "
                         f"{len(m['ops'])}")
        for x in r.get("vpd_topk", []):
            s = x["score"]
            if s.get("valid"):
                lines.append(f"  vpd-{x['k']:<4d} {s['progress']:.3f} {s['flipped']:.2f} {s['controls_kept']} {s['collateral']:.2f} {s['net']:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="torch device (native.device() otherwise)")
    ap.add_argument("--atlas", default=str(ATLAS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("out")
    s.add_argument("--k", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("teach")
    t.add_argument("behaviours")
    t.add_argument("out")
    t.add_argument("--offset", type=int, default=0)
    t.add_argument("--stride", type=int, default=1)
    t.add_argument("--n", type=int, default=1000)
    b = sub.add_parser("table")
    b.add_argument("rows")
    rs = sub.add_parser("rescore")
    rs.add_argument("rows")
    rs.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "table":
        return print(table([json.loads(x) for x in Path(args.rows).read_text().splitlines() if x.strip()]))
    ee = edit_eval.EditEval(native.Native(args.device))
    src = Source(Path(args.atlas))
    decode = _decoder()
    if args.cmd == "sample":
        bs = sample(ee, src, args.k, args.seed, decode, log=lambda m: print(m, flush=True))
        with open(args.out, "w") as f:
            for b in bs:
                f.write(json.dumps({**b, "goal": goal_of(b, decode)}) + "\n")
    elif args.cmd == "rescore":
        with open(args.out, "w") as f:
            for line in Path(args.rows).read_text().splitlines():
                if line.strip():
                    f.write(json.dumps(rescore(ee, src, json.loads(line), log=lambda m: print(m, flush=True))) + "\n")
    elif args.cmd == "teach":
        bs = [json.loads(x) for x in Path(args.behaviours).read_text().splitlines() if x.strip()][:args.n]
        done = set()
        if Path(args.out).exists():
            done = {tuple(json.loads(x)["behaviour"]["subcomponent"]) for x in Path(args.out).read_text().splitlines() if x.strip()}
        for i, b in enumerate(bs):
            if i % args.stride != args.offset or tuple(b["subcomponent"]) in done:
                continue
            print(f"{b['subcomponent']}", flush=True)
            res = teach(ee, src, b, log=lambda m: print(m, flush=True))
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
