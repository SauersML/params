"""The circuit-reading split (#2951; the lead's 10-16 decisive test): circuit questions (g2.py gen --dense: every neuron
group of a text's circuit, every zero, sign and steer question about each, answers measured by claim_check) split by
context into training and held-out ones, the held-out set balanced by answer class. The scorecard's O1 and the text
reader (text_reader.py) use rc1's split (~/mpd-data/oracle/reads_circuit/rc1; rc2_native has the same questions).

The test it was built for (rc1 to rc3, 10-16) found that an AO reader given each group's write injected at a placeholder
does not read the writes: strict outcome-class match 0.435 with the right circuit, 0.421 with names alone, 0.411 with
names anonymised (chance 0.356, name-prior 0.407). The injection reader ended there; the circuit is read as text.

Contexts (texts) split by id into training and held-out (HELD of them, at least MIN_CONTEXTS); the held-out questions are
balanced: within each question kind, the same number of each answer class (flips to another token, lead up, lead down,
almost unchanged; for signs supports, works against, barely matters; a class with fewer than MIN_CLASS held-out
questions left out), at least MIN_QUESTIONS in all.

  reads_circuit.py build EX.jsonl [EX.heldout.jsonl ...] --out OUT   OUT/train_correct.jsonl, test_correct.jsonl, claim.json
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


def answer_class(e: dict) -> str:
    a = e["a"].split("\n")[0]
    if e["cat"] == "n_sign":
        return "supports" if " supports " in a else "against" if "works against" in a else "barely"
    if "instead of" in a:
        return "flip"
    return "up" if " up about " in a else "down" if " down about " in a else "unchanged"


def build(a) -> None:
    rows = [json.loads(x) for f in a.examples for x in f.read_text().splitlines() if x.strip()]
    ctxs = sorted({r["text"] for r in rows})
    held = {c for c in ctxs if zlib.crc32(str(c).encode()) % 100 < HELD * 100}
    train = [r for r in rows if r["text"] not in held]
    test = [r for r in rows if r["text"] in held]
    assert len(held) >= MIN_CONTEXTS, (len(held), MIN_CONTEXTS)
    rng = random.Random(0)
    by = collections.defaultdict(list)
    for r in test:
        by[(r["cat"], answer_class(r))].append(r)
    bal = []
    for c in sorted({c for c, _ in by}):
        classes = [k for k in by if k[0] == c and len(by[k]) >= MIN_CLASS]
        if len(classes) < 2:
            continue
        n = min(len(by[k]) for k in classes)
        for k in classes:
            bal += rng.sample(by[k], n)
    assert len(bal) >= MIN_QUESTIONS, (len(bal), {k: len(v) for k, v in by.items()})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train_correct.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (a.out / "test_correct.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bal))
    kinds = collections.Counter(r["cat"] for r in bal)
    nclass = {c: len({answer_class(r) for r in bal if r["cat"] == c}) for c in kinds}
    claim = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "held_contexts": len(held), "train_contexts": len(ctxs) - len(held),
             "chance": round(sum(n / nclass[c] for c, n in kinds.items()) / len(bal), 3),
             "held_questions": len(bal), "train_questions": len(train),
             "classes": {f"{c}/{k}": n for (c, k), n in collections.Counter((r["cat"], answer_class(r)) for r in bal).items()}}
    (a.out / "claim.json").write_text(json.dumps(claim, indent=1))
    print(json.dumps(claim, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("examples", type=Path, nargs="+")
    b.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
