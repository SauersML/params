"""The aggregation test of question-scoped circuits (methods, 10-16; design ~/mpd-data/methods/design_1016.txt, 2a): do
two inputs' circuits, found separately, together explain the swap between them?

A cut that holds the units of its own key's value (the clean key's detectors at the key's positions, the answer's writers)
cannot pass circuit_robust's resampled test on a new key r: with r's prompt as the base, r's own detectors sit outside the
cut and push r's answer, and with the clean prompt as the base the cut can only remove the clean key's information. A
sparse cut passes there only where its units carry every key (the movers). The claim such a cut supports is that the
circuits of two inputs combine: the union of x's circuit C(x) and r's circuit C(r), each found without the pair (x, r),
noised and denoised on (x, r), passes (VPD's aggregation of explanations over inputs).

Per record (an arm's circuit C(x) for question x) and up to KEYS unseen keys r (circuit_robust.resample_ids: the
question's own template with another key of its task, skipping the record's corrupted and selection keys):
  C(r)    the same arm (its steps and constants) run on r's prompt against the record's own corrupted prompt; a
          multi-partner arm's partners are other pool keys of the task, never x's key or a tested key
  shares  on the pair (x, r), per cut: C(x)'s units (circuit_robust's resampled test), C(x) u C(r)'s, as many random
          units at the union's (layer, position) slots (seeded), and C(x) u C(r') with r' another tested key (the
          mismatched union: a gain from more units rather than from r's own), each noised and denoised
  passes  a cut passes on r when both shares reach PASS; over keys, when the mean shares do

  circuit_aggregate.py RECORDS_DIR DEVSET OUT --arm mask:30 [--pool QUESTIONS.json ...] [--keys 3] [--limit N]
                       [--model Qwen/Qwen3-8B] [--dtype bfloat16]
writes OUT/<id>.json per record and OUT/summary.json; prints a line per record and the summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_finder as cf  # noqa: E402
import circuit_masks as cm  # noqa: E402
import circuit_robust as cr  # noqa: E402

PASS = cf.PASS
KEYS = 3
CUTS = ("source", "movers", "readout")


def arm_of(arm: str) -> tuple:
    """(search name, steps, constants, partner pairs) from an arm string as circuit_masks' command line reads it."""
    name, *parts = arm.split(":")
    steps = next((int(x) for x in parts if x.isdigit()), cm.STEPS)
    sets = {k.upper(): float(v) for k, v in (x.split("=") for x in parts if "=" in x)}
    return name, steps, sets, int(sets.pop("PARTNERS", 1))


def random_like(units: list, pt: cf.Patcher, rng) -> list:
    """As many units at the same (layer, position) slots, indices drawn at random."""
    out = []
    for u in units:
        if u[0] == "head":
            out.append(("head", u[1], int(rng.integers(pt.H)), u[3]))
        else:
            out.append(("neuron", u[1], u[2], int(rng.integers(pt.D))))
    return out


def circuit_of(pts: list, ids_r: list, corrupt: list, partner_ids: list, arm: str) -> dict:
    """The arm's record for r's prompt against the corrupted prompt (pts: a Patcher and its partner Patchers)."""
    name, steps, sets, npart = arm_of(arm)
    saved = {k: getattr(cm, k) for k in sets}
    for k, v in sets.items():
        setattr(cm, k, v)
    try:
        return cm.find(pts[0], ids_r, corrupt, arm=name, steps=steps,
                       partners=list(zip(pts[1:], partner_ids[:npart - 1])) if npart > 1 else None)
    finally:
        for k, v in saved.items():
            setattr(cm, k, v)


def check(pts: list, tok, rec: dict, q: dict, pool: dict, arm: str, keys: int = KEYS, system: str | None = None) -> dict:
    """The aggregation test of one record (module docstring)."""
    pt = pts[-1]                                     # (the measuring Patcher; pts[:-1] find C(r))
    rs = cr.resample_ids(tok, q, rec, keys, pool, system)
    _, _, _, npart = arm_of(arm)
    user, key, task = rec.get("user", q.get("user")), rec.get("key", q.get("key")), rec.get("task", q.get("task"))
    prefill = q.get("prefill") or rec.get("prefill")
    pre = tok(prefill, add_special_tokens=False)["input_ids"] if prefill else []
    system = q.get("system", rec.get("system", system))
    banned = {key, q.get("corrupt_key"), rec.get("corrupt_key")} | {k for k, _ in rs}
    clean, corrupt = rec["ids"], rec["corrupted"]
    own = {c: cr.cut_units(rec, c)[0] for c in CUTS if c in rec.get("cuts", {})}
    rng = np.random.default_rng(zlib.crc32(rec["id"].encode()))
    out = {"id": rec["id"], "arm": arm, "pairs": []}
    found = []                                      # (each tested key's own circuit first: the mismatched control uses another's)
    for r, ids_r in rs:
        cands = sorted(k for k in pool.get(task, ()) if k not in banned)
        np.random.default_rng(zlib.crc32(f"{rec['id']}|{r}".encode())).shuffle(cands)
        part = [x for x in (cf.chat(tok, user.replace(key, k), system) + pre for k in cands) if len(x) == len(clean)]
        t0 = time.perf_counter()
        rec_r = {**circuit_of(pts[:-1], ids_r, corrupt, part[:2 * npart], arm), "ids": ids_r}
        found.append((r, ids_r, rec_r, time.perf_counter() - t0))
    for n_r, (r, ids_r, rec_r, secs) in enumerate(found):
        other = found[(n_r + 1) % len(found)][2] if len(found) > 1 else None
        info = pt.setup(clean, ids_r, explicit_sets=cr.token_sets(rec))
        if not info["valid"] or info["same_answer"] or info["gap"] < cf.MIN_GAP or rec_r.get("skipped"):
            continue
        row = {"key": r, "gap": info["gap"], "seconds_find": round(secs, 2), "met_r": rec_r.get("met"), "cuts": {}}
        for c, units in own.items():
            theirs = cr.cut_units(rec_r, c)[0] if c in rec_r.get("cuts", {}) else []
            union = list(dict.fromkeys(list(units) + list(theirs)))
            wrong = list(dict.fromkeys(list(units) + list(cr.cut_units(other, c)[0] if other and c in other.get("cuts", {}) else [])))
            sets = [units, union, random_like(union, pt, rng), wrong]
            got = pt.both([s if s else [] for s in sets])
            row["cuts"][c] = {"n_own": len(units), "n_union": len(union), "own": list(got[0]) if units else None,
                              "union": list(got[1]) if union else None, "random": list(got[2]) if union else None,
                              "n_mismatched": len(wrong), "mismatched": list(got[3]) if other and wrong else None}
        out["pairs"].append(row)
    pt.setup(clean, corrupt, explicit_sets=cr.token_sets(rec))
    out["cuts"] = {}
    for c in own:
        rows = [p["cuts"][c] for p in out["pairs"] if p["cuts"].get(c, {}).get("union") is not None]
        if not rows:
            continue
        summ = {}
        for which in ("own", "union", "random", "mismatched"):
            v = [x.get(which) for x in rows if x.get(which) is not None]
            m = [float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))] if v else None
            summ[which] = {"mean": m, "passes": bool(m and m[0] >= PASS and m[1] >= PASS)}
        summ["keys"] = len(rows)
        summ["met_own_pair"] = rec["cuts"][c]["met"]
        out["cuts"][c] = summ
    return out


def summarize(rows: list) -> dict:
    """Per cut, over records with a measured key: pass rates of the own cut, the union and the random control."""
    s = {}
    for c in CUTS:
        xs = [r["cuts"][c] for r in rows if c in r.get("cuts", {})]
        if xs:
            s[c] = {"n": len(xs), **{w: sum(x[w]["passes"] for x in xs if w in x) for w in ("own", "union", "random", "mismatched")},
                    "median_union_n": float(np.median([p["cuts"][c]["n_union"] for r in rows for p in r["pairs"] if c in p["cuts"]]))}
    full = [r for r in rows if r.get("cuts")]
    s["question"] = {"n": len(full), **{w: sum(all(x.get(w, {}).get("passes") for x in r["cuts"].values()) for r in full)
                                        for w in ("own", "union", "random", "mismatched")}}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm", required=True, help="the arm the records were found by (circuit_masks' --arm string)")
    ap.add_argument("--pool", type=Path, nargs="*", default=[])
    ap.add_argument("--keys", type=int, default=KEYS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--system", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cf._threads()
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto").eval()
    model.requires_grad_(False)
    qs = {q["id"]: q for q in json.loads(a.questions.read_text())["questions"]}
    pool = cr.question_pool(a.pool + [a.questions])
    _, _, _, npart = arm_of(a.arm)
    pts = [cf.Patcher(model, tok) for _ in range(npart + 1)]
    recs = [json.loads(p.read_text()) for p in sorted(a.records.glob("*.json"))]
    recs = [r for r in recs if r.get("cuts") and not r.get("skipped") and r.get("id") in qs][:a.limit or None]
    t0, rows = time.time(), []
    for rec in recs:
        path = a.out / (rec["id"].replace("/", "__") + ".json")
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        row = check(pts, tok, rec, qs[rec["id"]], pool, a.arm, a.keys, a.system)
        path.write_text(json.dumps(row))
        rows.append(row)
        print(f"{time.time() - t0:7.1f}s {rec['id']}: {len(row['pairs'])} keys; " + ", ".join(
            f"{c} own {'P' if x['own']['passes'] else '-'} union {'P' if x['union']['passes'] else '-'} "
            f"mismatched {'P' if x['mismatched']['passes'] else '-'} random {'P' if x['random']['passes'] else '-'}"
            for c, x in row["cuts"].items()), flush=True)
    s = summarize(rows)
    (a.out / "summary.json").write_text(json.dumps({"arm": a.arm, "records": str(a.records), "summary": s}, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
