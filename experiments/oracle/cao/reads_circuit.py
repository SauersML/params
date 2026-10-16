"""Does the oracle read the circuit? (#2951; the lead's 10-16 decisive test.) Questions about a circuit's units (g2.py gen
--dense: every neuron group of the text's circuit shown with its write, every zero, sign and steer question about each,
answers measured by claim_check) are asked of three oracles under four inputs, and scored by whether the claim each
answer ends with holds on the model (claim_check's pass):

  oracles  T-correct   SFT from AO's adapter on the training contexts with their own circuits' writes
           T-mismatch  the same SFT with every write taken from another training context's circuit (the writes carry
                       nothing about the question; names, text activations and answers unchanged)
           T0          AO's released adapter, untrained here
  inputs   correct     the text's own circuit: names and writes
           names       the names alone (every write's value 0: nothing added at the units' placeholders)
           mismatch    the names with another held-out context's writes
           none        no circuit (the text's activations and the question alone)

Contexts (texts) split by id into training and held-out (HELD of them, at least MIN_CONTEXTS); the held-out questions are
balanced: within each question kind, the same number of each answer class (flips to another token, lead up, lead down,
almost unchanged; for signs supports, works against, barely matters; a class with fewer than MIN_CLASS held-out
questions left out), at least MIN_QUESTIONS in all. The claim is
written to OUT/claim.json by build, before anything is scored.

  reads_circuit.py build EX.jsonl [EX.heldout.jsonl ...] --out OUT     the split, the variants, the claim
  reads_circuit.py score OUT                                           the table, from OUT/eval_<oracle>_<input>.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
import zlib
from pathlib import Path

HELD = 0.4
MIN_CONTEXTS, MIN_QUESTIONS = 32, 256
MIN_CLASS = 20  # an answer class with fewer held-out questions is left out of the balanced set (flips are rare)
INPUTS = ("correct", "names", "mismatch", "none")
CLAIM = {
    "metric": "share of held-out questions whose answer's claim holds on Qwen3-8B (claim_check pass), balanced classes",
    "predictions": [
        "T-correct with the correct circuit beats T-correct with names alone by at least 0.10",
        "T-correct with the correct circuit beats T-correct with another context's writes by at least 0.10",
        "T-mismatch scores within 0.05 between correct and names (it never learned to read writes)",
        "T-correct with the correct circuit beats T0 by at least 0.10",
    ],
    "reading": "the oracle reads the circuit when the first two predictions hold and the third holds; if T-correct gains "
               "nothing from correct writes over names, it answers from names, layers and text alone",
}


def answer_class(e: dict) -> str:
    a = e["a"].split("\n")[0]
    if e["cat"] == "n_sign":
        return "supports" if " supports " in a else "against" if "works against" in a else "barely"
    if "instead of" in a:
        return "flip"
    return "up" if " up about " in a else "down" if " down about " in a else "unchanged"


def rewrite(e: dict, how: str, donor: dict | None = None) -> dict:
    """An example under an input: names (writes' values 0), mismatch (the donor's writes in order, cycled), none (no unit
    blocks)."""
    e = json.loads(json.dumps(e))
    if how == "none":
        e["blocks"] = [b for b in e["blocks"] if "layer" in b]
        return e
    if how == "names":
        for b in e["blocks"]:
            if "write" in b:
                b["write"] = [[x[0], x[1], 0.0] for x in b["write"]] if isinstance(b["write"][0], list) else [b["write"][0], b["write"][1], 0.0]
        return e
    if how == "mismatch":
        pool = [b["write"] for b in donor["blocks"] if "write" in b]
        k = 0
        for b in e["blocks"]:
            if "write" in b:
                b["write"] = pool[k % len(pool)]
                k += 1
        return e
    return e


def build(a) -> None:
    rows = [json.loads(x) for f in a.examples for x in f.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if any("write" in b for b in r["blocks"])]
    ctxs = sorted({r["text"] for r in rows})
    held = {c for c in ctxs if zlib.crc32(str(c).encode()) % 100 < HELD * 100}
    train = [r for r in rows if r["text"] not in held]
    test = [r for r in rows if r["text"] in held]
    assert len(held) >= MIN_CONTEXTS, (len(held), MIN_CONTEXTS)
    rng = random.Random(0)
    by = collections.defaultdict(list)
    for r in test:
        by[(r["cat"], answer_class(r))].append(r)
    cats = sorted({c for c, _ in by})
    bal = []
    for c in cats:
        classes = [k for k in by if k[0] == c and len(by[k]) >= MIN_CLASS]
        if len(classes) < 2:
            continue
        n = min(len(by[k]) for k in classes)
        for k in classes:
            bal += rng.sample(by[k], n)
    assert len(bal) >= MIN_QUESTIONS, (len(bal), {k: len(v) for k, v in by.items()})
    a.out.mkdir(parents=True, exist_ok=True)
    donors = {}  # each context's donor: another context of its split (a fixed derangement)
    for split, cs in (("train", sorted({r["text"] for r in train})), ("test", sorted(held))):
        first = {}
        for r in (train if split == "train" else test):
            first.setdefault(r["text"], r)
        for i, c in enumerate(cs):
            donors[c] = first[cs[(i + 1) % len(cs)]]
    (a.out / "train_correct.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (a.out / "train_mismatch.jsonl").write_text("".join(json.dumps(rewrite(r, "mismatch", donors[r["text"]])) + "\n" for r in train))
    for how in INPUTS:
        (a.out / f"test_{how}.jsonl").write_text("".join(json.dumps(rewrite(r, how, donors[r["text"]])) + "\n" for r in bal))
    claim = {**CLAIM, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "held_contexts": len(held), "train_contexts": len(ctxs) - len(held),
             "held_questions": len(bal), "train_questions": len(train),
             "classes": {f"{c}/{k}": n for (c, k), n in collections.Counter((r["cat"], answer_class(r)) for r in bal).items()}}
    (a.out / "claim.json").write_text(json.dumps(claim, indent=1))
    print(json.dumps(claim, indent=1))


def score(a) -> None:
    claim = json.loads((a.out / "claim.json").read_text())
    print("claim written", claim["written"])
    tab = {}
    for f in sorted(a.out.glob("eval_*.jsonl")):
        oracle, how = f.stem[len("eval_"):].rsplit("_", 1)
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        tab[(oracle, how)] = (sum(r["claim_pass"] for r in rs) / len(rs), sum(r["claim_specific"] for r in rs) / len(rs), len(rs))
    oracles = sorted({o for o, _ in tab})
    print(f"{'oracle':12s}" + "".join(f"{h:>18s}" for h in INPUTS))
    for o in oracles:
        print(f"{o:12s}" + "".join(f"{tab[(o, h)][0]:>9.3f} ({tab[(o, h)][1]:.3f})" if (o, h) in tab else f"{'-':>18s}" for h in INPUTS))
    print("cells: claim holds (and specific), n =", sorted({v[2] for v in tab.values()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("examples", type=Path, nargs="+")
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("out", type=Path)
    a = ap.parse_args()
    {"build": build, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
