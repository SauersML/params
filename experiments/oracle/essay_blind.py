"""The blind finder against the essay's own graphs (#2951, reviewer item 5; scale32 10-16): circuit_finder / circuit_masks
records of the essay's five Qwen3-32B prompts (circuit_finder.CASES santa, preeclampsia, scar, hedge, user; each clean
prompt the essay's tokens, each corrupted prompt a swap of the same length) set beside the essay's graph neurons for that
prompt (its figure data, graph-carousel-hover.json: per group the published neurons "layer:index±" at their positions;
groups list at most 6 of their n neurons).
Per record (one pair, one search arm):
  recall    the essay's published neurons E (unique layer, neuron, position) found by the search: among the first |E| of
            our units in measured order (the circuit's neurons by their larger share alone, then the other measured singles)
            and anywhere in the circuit or its singles; exact (layer, neuron, position) and position-free (layer, neuron)
  shares    under our pair (circuit_finder.Patcher.both: noising lost, denoising restored): the essay's neurons E at their
            positions; each essay group; our circuit's neurons; our first |E|; and RANDOM draws of |E| neurons at E's own
            (layer, position) slots (the mean)
Positions before the pair's first changed token are equal in both runs, so units there patch nothing.

  essay_blind.py OUT.json RECORD_DIR [RECORD_DIR ...] --groups graph-carousel-hover.json [--model Qwen/Qwen3-32B]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "method" / "oracle"))

import circuit_finder as cf  # noqa: E402

RANDOM = 5


def essay_units(groups: dict, case: str) -> dict:
    """{group: [(layer, neuron, pos, pole)]} of one essay graph."""
    out = {}
    for name, g in groups.items():
        c, gname = name.split("/", 1)
        if c != case:
            continue
        us = []
        for row in g["rows"]:
            u, pos = row[0], int(row[3])
            L, rest = u.split(":")
            us.append((int(L), int(rest[:-1]), pos, 1 if rest[-1] == "+" else -1))
        out[gname] = us
    return out


def ours(rec: dict) -> list:
    """Our neurons in measured order: the circuit's (source and readout cuts) by their larger share alone, then every other
    measured single of a neuron cut, as (layer, neuron, pos)."""
    key = lambda x: -max(abs(x["lost"]), abs(x["restored"]))  # noqa: E731
    circ = [(n["layer"], n["neuron"], n["pos"]) for n in sorted(rec.get("nodes", []), key=key)]
    rest = [s for c in ("source", "readout") for s in (rec.get("singles") or {}).get(c, []) if "neuron" in s]
    more = [(s["layer"], s["neuron"], s["pos"]) for s in sorted(rest, key=key)]
    seen, order = set(), []
    for u in circ + more:
        if u not in seen:
            seen.add(u)
            order.append(u)
    return order, len(circ)


def units(us: list, first: int) -> list:
    return [("neuron", L, p, n) for L, n, p in us if p >= first]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    groups = json.loads(a.groups.read_text())["groups"]
    recs = [(d.name, json.loads(f.read_text())) for d in a.records for f in sorted(d.glob("*.json"))]
    recs = [(arm, r) for arm, r in recs if not r.get("skipped") and r.get("ids")]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto").eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok)
    rows, t0 = [], time.time()
    for arm, r in recs:
        case = r.get("task") or r["id"].rsplit("_", 1)[0]
        eg = essay_units(groups, case)
        E = sorted({(L, n, p) for us in eg.values() for L, n, p, _ in us})
        if not E:
            continue
        pt.setup(r["ids"], r["corrupted"])
        order, ncirc = ours(r)
        m = len(E)
        top = order[:m]
        Es, Ef = set(E), {(L, n) for L, n, _ in E}
        rng = random.Random(0)
        rand = [[(L, rng.randrange(pt.D), p) for L, _, p in E] for _ in range(RANDOM)]
        sets = [units(E, pt.first), units(order[:ncirc], pt.first), units(top, pt.first)] + \
               [units([(L, n, p) for L, n, p, _ in us], pt.first) for us in eg.values()] + [units(x, pt.first) for x in rand]
        got = pt.both(sets)
        k = 3 + len(eg)
        row = {"arm": arm, "id": r["id"], "case": case, "answer": r.get("answer_text"), "corrupt_answer": r.get("corrupt_answer_text"),
               "gap": r.get("gap"), "first_diff": pt.first, "met": r.get("met"), "essay_neurons": m,
               "essay_after_first": len(units(E, pt.first)), "circuit_neurons": ncirc,
               "recall_top": len(Es & set(top)) / m, "recall_any": len(Es & set(order)) / m,
               "recall_top_free": len(Ef & {(L, n) for L, n, _ in top}) / len(Ef),
               "recall_any_free": len(Ef & {(L, n) for L, n, _ in order}) / len(Ef),
               "found": sorted([list(u) for u in Es & set(order)]),
               "shares": {"essay": got[0], "circuit": got[1], "top": got[2],
                          "random": [sum(x[i] for x in got[k:]) / RANDOM for i in (0, 1)]},
               "groups": {g: got[3 + i] for i, g in enumerate(eg)}}
        rows.append(row)
        f = lambda s: f"{s[0]:.2f}/{s[1]:.2f}"  # noqa: E731
        print(f"{time.time() - t0:6.1f}s {arm} {r['id']}: {r.get('answer_text')!r} vs {r.get('corrupt_answer_text')!r}, essay {m} "
              f"neurons ({row['essay_after_first']} after the change), ours {ncirc}; recall top {row['recall_top']:.2f} any "
              f"{row['recall_any']:.2f} (position-free {row['recall_top_free']:.2f}/{row['recall_any_free']:.2f}); lost/restored "
              f"essay {f(got[0])}, circuit {f(got[1])}, our top {m} {f(got[2])}, random {f(row['shares']['random'])}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
