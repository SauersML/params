"""Scripted checks of the oracle's intervention claims on a Qwen3 model (the lead's 10-15: the oracle's RL reward and
T4's evaluation, no agent and no tools; one service with neuron_reward.py, which dispatches the kind "intervention"
here). A claim names units of the model, an intervention on them and its predicted effect on the answer at a position;
the check runs that intervention and returns pass or fail with the measured numbers.

Semantics (neuron_builder's, inframove's 10-15):
- the reference is the model with weak values zeroed (Neurons.thr, the T2 file; without one, the model itself);
- a unit is a neuron's pole ("L18.N2201+" its positive values, "L18.N2201-" its negative ones, "L18.N2201" both) or an
  attention head ("L5.H3": its slice of o_proj's input);
- zero sets the unit's values to 0 at positions 1 on; the first position (the attention sink) and massive values
  (|value| at least Neurons.massive, the layer's cut) stay live;
- steer adds scale x the neuron's calibration maximum (Neurons.mx) at positions 1 on, along the named pole (a negative
  scale, or the "-" pole, pushes the value down);
- patch sets the units' values at the mapped positions (default: every position from 1 on, the two prompts of equal
  length) to their values on the source prompt's run, in the reference;
- swap replaces tokens of the prompt (alone or with the units' intervention).
Metric: the answer's contrast at the position, log2 p(answer) - log2 of the summed p of the alternatives (the
reference's ALTERNATIVES likeliest other tokens unless given), neuron_builder's.
Predictions: {"top": token} (the answer becomes token), {"top_changes": bool}, {"contrast_change": {"op", "value"}}
(after - before, bits), {"contrast": {"op", "value"}} (after).
Control: the same intervention on CONTROLS random sets of as many units (neurons: per layer and pole, other neurons of
that layer; heads: other heads of that layer); a prediction of an effect is specific when it passes and no control set
also passes (naming hundreds of neurons does not make a claim specific). A prediction that nothing changes is specific
when it passes.
Side: the mean KL (bits) at the SIDE positions before the answer position, the intervention against the reference
(neuron_builder's side effect).
Cost, in rows of forward passes: the reference row once per prompt, one row per claim, CONTROLS rows per claim, one row
per patch source; a prompt's rows run in batches of ROWS (padded: numbers do not depend on what else a batch holds).

  python claim_check.py MODEL CLAIMS.jsonl OUT.jsonl [--thr THR.pt] [--adapter A] [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuron_builder as nb  # noqa: E402

UNIT = re.compile(r"^L(\d+)\.([NH])(\d+)([+-]?)$")
CONTROLS = 2  # random unit sets per claim
SIDE = nb.SIDE
ROWS = 8  # rows a batch, every batch padded to it
OPS = {"<=": lambda x, v: x <= v, ">=": lambda x, v: x >= v, "<": lambda x, v: x < v, ">": lambda x, v: x > v}


class BadClaim(ValueError):
    pass


def prompt_ids(nr: nb.Neurons, prompt: dict) -> list[int]:
    if "ids" in prompt:
        return [int(t) for t in prompt["ids"]]
    if "chat" in prompt:
        return nb.chat(nr.tok, prompt["chat"])
    if "text" in prompt:
        return nr.tok(prompt["text"], add_special_tokens=False)["input_ids"]
    raise BadClaim("prompt: ids, chat or text")


def token_id(nr: nb.Neurons, s) -> int:
    if isinstance(s, int):
        return s
    got = nr.tok(str(s), add_special_tokens=False)["input_ids"]
    if len(got) != 1:
        raise BadClaim(f"{s!r} is {len(got)} tokens; an answer or a swap is one token (its leading space included)")
    return got[0]


def parse_unit(s) -> tuple:
    """A unit's name ("L18.N2201+", "L18.N2201-", "L18.N2201", "L5.H3") as ("N" | "H", layer, index, pole 1 | -1 | 0),
    the one parser every claim kind uses (neuron_reward's explanation kind too); a head has no pole."""
    m_ = UNIT.match(str(s).strip())
    if not m_ or (m_.group(4) and m_.group(2) == "H"):
        raise BadClaim(f"unit {s!r}: L<layer>.N<neuron> (a pole: + or - after it) or L<layer>.H<head>")
    return m_.group(2), int(m_.group(1)), int(m_.group(3)), {"+": 1, "-": -1, "": 0}[m_.group(4)]


def unit_name(u: tuple) -> str:
    kind, layer, i, pole = u
    return f"L{layer}.{kind}{i}" + {1: "+", -1: "-", 0: ""}[pole]


def units_of(nr: nb.Neurons, units: list) -> list[tuple]:
    """parse_unit of each, checked against the model's sizes."""
    out = []
    for s in units:
        kind, layer, i, pole = parse_unit(s)
        if not 0 <= layer < nr.L or not 0 <= i < (nr.D if kind == "N" else nr.NH):
            raise BadClaim(f"unit {s!r} does not exist ({nr.L} layers, {nr.D} neurons, {nr.NH} heads)")
        out.append((kind, layer, i, pole))
    return out


def control_sets(nr: nb.Neurons, us: list[tuple], n: int, seed: str) -> list[list[tuple]]:
    """n random unit sets matching us: per (kind, layer, pole) as many other units of that layer."""
    rng = random.Random(seed)
    groups: dict = {}
    for u in us:
        groups.setdefault((u[0], u[1], u[3]), []).append(u[2])
    out = []
    for _ in range(n):
        s = []
        for (kind, layer, pole), named in sorted(groups.items()):
            pool = [x for x in range(nr.D if kind == "N" else nr.NH) if x not in set(named)]
            s += [(kind, layer, x, pole) for x in rng.sample(pool, min(len(named), len(pool)))]
        out.append(s)
    return out


def row_edit(nr: nb.Neurons, rows: list[dict]):
    """edit and route over a batch: rows[b] = {"units": [...], "op": "zero" | ("steer", scale) | ("patch", src, pairs)}
    (empty: the reference row); src = the source run's captured values {"h": [L x [T, D]], "o": [L x [T, NH x hd]]}."""
    def edit(i, h):
        out = None
        for b, r in enumerate(rows):
            for kind, layer, n, pole in r.get("units", ()):
                if kind != "N" or layer != i:
                    continue
                if out is None:
                    out = h.clone()
                v = out[b, 1:, n]
                op = r["op"]
                if op == "zero":
                    hit = ((v > 0) if pole > 0 else (v < 0) if pole < 0 else (v != 0)) & (v.abs() < nr.massive[i].to(v.device, v.dtype))
                    out[b, 1:, n] = v.masked_fill(hit, 0)
                elif op[0] == "steer":
                    out[b, 1:, n] = v + (op[1] * (-1 if pole < 0 else 1) * nr.mx[i, n]).to(v.dtype)
                else:
                    for p, q in op[2]:
                        out[b, p, n] = op[1]["h"][i][q, n].to(out.dtype)
        return h if out is None else out

    def route(i, a):
        out = None
        for b, r in enumerate(rows):
            for kind, layer, hh, _ in r.get("units", ()):
                if kind != "H" or layer != i:
                    continue
                if out is None:
                    out = a.clone()
                sl = slice(hh * nr.hd, (hh + 1) * nr.hd)
                op = r["op"]
                if op == "zero":
                    out[b, 1:, sl] = 0
                elif op[0] == "steer":
                    raise BadClaim("steer takes neurons")
                else:
                    for p, q in op[2]:
                        out[b, p, sl] = op[1]["o"][i][q, sl].to(out.dtype)
        return a if out is None else out

    return edit, route


def contrast(lp: torch.Tensor, y: int, alts: list[int]) -> torch.Tensor:
    """[B, V] log p -> [B] bits."""
    return (lp[:, y] - torch.logsumexp(lp[:, alts], -1)) / math.log(2)


def holds(pred: dict, before: dict, after: dict, tok_id) -> bool:
    if "top" in pred:
        return after["top"] == tok_id(pred["top"])
    if "top_changes" in pred:
        return (after["top"] != before["top"]) == bool(pred["top_changes"])
    for key, x in (("contrast_change", after["contrast"] - before["contrast"]), ("contrast", after["contrast"])):
        if key in pred:
            c = pred[key]
            if c.get("op") not in OPS:
                raise BadClaim(f"{key}: op one of {sorted(OPS)}")
            return OPS[c["op"]](x, float(c["value"]))
    raise BadClaim("predict: top, top_changes, contrast_change or contrast")


def effect_predicted(pred: dict) -> bool:
    """Whether the prediction is of an effect (controls can then make it unspecific), not of nothing changing."""
    if "top" in pred:
        return True
    if "top_changes" in pred:
        return bool(pred["top_changes"])
    c = pred.get("contrast_change")
    return bool(c) and ((c["op"] in ("<=", "<") and float(c["value"]) < 0) or (c["op"] in (">=", ">") and float(c["value"]) > 0))


@torch.no_grad()
def check_many(nr: nb.Neurons, claims: list[dict], controls: int = CONTROLS) -> list[dict]:
    """Every claim's result (in order); claims on one prompt share its reference row and its batches."""
    results: list = [None] * len(claims)
    by: dict = {}
    for k, c in enumerate(claims):
        try:
            ids = prompt_ids(nr, c["prompt"])
            by.setdefault(tuple(ids), []).append(k)
        except (BadClaim, KeyError, TypeError) as e:
            results[k] = {"pass": False, "specific": False, "error": f"{type(e).__name__}: {e}"}
    for ids, ks in by.items():
        _prompt(nr, list(ids), [(k, claims[k]) for k in ks], controls, results)
    return results


def _prompt(nr: nb.Neurons, ids: list[int], claims: list[tuple], controls: int, results: list) -> None:
    T = len(ids)
    rows, plan = [{}], []  # rows[0]: the reference; plan: (claim index, its rows, its parsed parts)
    tok_id = lambda s: token_id(nr, s)  # noqa: E731
    sources: dict = {}
    for k, c in claims:
        try:
            pos = int(c.get("position", -1))
            pos = T + pos if pos < 0 else pos
            if not 0 < pos < T:
                raise BadClaim(f"position must lie in 1..{T - 1}")
            us = units_of(nr, c.get("units", []))
            op = c.get("op", "zero")
            if isinstance(op, dict) and "steer" in op:
                op = ("steer", float(op["steer"]))
            elif isinstance(op, dict) and "patch_from" in op:
                src = prompt_ids(nr, op["patch_from"])
                pairs = op.get("positions") or ([(p, p) for p in range(1, T)] if len(src) == T else None)
                if pairs is None:
                    raise BadClaim("patch: give positions [[p, p_source], ...] when the prompts differ in length")
                pairs = [(int(p), int(q)) for p, q in pairs]
                if not all(0 < p < T and 0 < q < len(src) for p, q in pairs):
                    raise BadClaim("patch positions out of range (the first position stays live)")
                op = ("patch", tuple(src), pairs)
                sources.setdefault(tuple(src), None)
            elif op != "zero":
                raise BadClaim("op: zero, {steer: scale} or {patch_from: prompt}")
            if op[0] == "steer" and any(u[0] == "H" for u in us):
                raise BadClaim("steer takes neurons")
            x = list(ids)
            for s in c.get("swap", []):
                p = int(s["position"])
                if not 0 < p < T:
                    raise BadClaim("swap positions lie in 1..T-1")
                x[p] = tok_id(s["token"])
            if not us and x == ids:
                raise BadClaim("a claim names units or swaps")
            ctl = control_sets(nr, us, controls, json.dumps(c, sort_keys=True)) if us else []
            first = len(rows)
            rows += [{"units": s, "op": op, "ids": x} for s in [us] + ctl]
            plan.append((k, c, pos, first, len(ctl), us))
        except (BadClaim, KeyError, TypeError, ValueError) as e:
            results[k] = {"pass": False, "specific": False, "error": f"{type(e).__name__}: {e}"}
    for src in sources:  # the patch sources' runs in the reference: every layer's neuron values and head outputs
        c_ = {}
        nr.logits(list(src), [len(src) - 1], capture=c_)
        sources[src] = {"h": [h[0] for h in c_["h"]], "o": [o[0] for o in c_["o"]]}
    for r in rows[1:]:
        if isinstance(r["op"], tuple) and r["op"][0] == "patch":
            r["op"] = ("patch", sources[r["op"][1]], r["op"][2])
    positions = sorted({p for _, _, p, _, _, _ in plan} | {q for _, _, p, _, _, _ in plan for q in range(max(1, p - SIDE), p)})
    at = {p: j for j, p in enumerate(positions)}
    lps = []
    for s0 in range(0, len(rows), ROWS):
        part = rows[s0:s0 + ROWS]
        part = part + part[-1:] * (ROWS - len(part))
        edit, route = row_edit(nr, part)
        x = torch.tensor([r.get("ids", ids) for r in part], device=nr.dev)
        lps.append(torch.log_softmax(nr.logits(x, positions, edit=edit, route=route), -1)[:len(rows) - s0])
    lp = torch.cat(lps)  # [rows, positions, V]
    ref = lp[0]
    for k, c, pos, first, nctl, us in plan:
        try:
            j = at[pos]
            y = tok_id(c["answer"]) if c.get("answer") is not None else int(ref[j].argmax())
            alts = ([tok_id(t) for t in c["alternatives"]] if c.get("alternatives") else
                    [t for t in ref[j].topk(nb.ALTERNATIVES + 1).indices.tolist() if t != y][:nb.ALTERNATIVES])
            cs = contrast(lp[:, j], y, alts)
            before = {"top": int(ref[j].argmax()), "contrast": float(cs[0])}
            state = lambda r: {"top": int(lp[r, j].argmax()), "contrast": float(cs[r])}  # noqa: E731
            after = state(first)
            ok = holds(c["predict"], before, after, tok_id)
            ctl = [holds(c["predict"], before, state(first + 1 + m), tok_id) for m in range(nctl)]
            side = [at[q] for q in range(max(1, pos - SIDE), pos)]
            skl = float((ref[side].exp() * (ref[side] - lp[first, side])).sum(-1).mean() / math.log(2)) if side else 0.0
            results[k] = {
                "pass": bool(ok), "specific": bool(ok and not (effect_predicted(c["predict"]) and any(ctl))),
                "contrast_before": before["contrast"], "contrast_after": after["contrast"],
                "change": after["contrast"] - before["contrast"], "top_before": nr.tok.decode([before["top"]]) if nr.tok else before["top"],
                "top_after": nr.tok.decode([after["top"]]) if nr.tok else after["top"],
                "control_change": [float(cs[first + 1 + m]) - before["contrast"] for m in range(nctl)], "controls_pass": ctl,
                "side_kl": skl, "units": len(us), "passes": 1 + nctl + (1 if isinstance(c.get("op"), dict) and "patch_from" in c["op"] else 0)}
        except (BadClaim, KeyError, TypeError, ValueError) as e:
            results[k] = {"pass": False, "specific": False, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("claims", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--thr", type=Path, help="the weak-zeroed reference (neuron_builder.Neurons.save; T2)")
    ap.add_argument("--adapter", help="an organism's LoRA, on")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(dev).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    nr = nb.Neurons(model, tok)
    if a.thr:
        nr.load(a.thr)
    else:  # no reference file: the model itself, calibration maxima from the claims' own prompts
        claims0 = [json.loads(x) for x in a.claims.read_text().splitlines() if x.strip()]
        nr.mx = nr.maxima([prompt_ids(nr, c["prompt"]) for c in claims0])
        nr.massive = nb.MASSIVE * nr.mx.max(1).values
    claims = [json.loads(x) for x in a.claims.read_text().splitlines() if x.strip()]
    got = check_many(nr, claims)
    a.out.write_text("".join(json.dumps({**c, "result": r}) + "\n" for c, r in zip(claims, got)))
    n = len(got)
    print(f"{n} claims: pass {sum(r['pass'] for r in got)}, specific {sum(r['specific'] for r in got)}, "
          f"errors {sum('error' in r for r in got)}")


if __name__ == "__main__":
    main()
