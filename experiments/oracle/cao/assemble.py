"""The causal oracle's SFT data from cao.py gen's examples plus the LatentQA replay (#2951, Path G; the lead's 10-14 label
checks), with the label mix printed before and after.

  which, pair  the answer names the token and its position ("' was' at position 37"; the token alone is ambiguous when
               its string recurs in the text: 31 of the first 107 winners); a pair question names both positions (a pair
               whose named token recurs, so that its position is unknown, is dropped). Abstains ("No clear answer.")
               kept at most ABSTAIN_MAX of the category, the rest dropped at random (above that the oracle learns that
               abstaining is safe; the kept ones teach calibration).
  swap, edit   unchanged answers kept at most UNCHANGED_MAX of the category (the measured rates were 80% and 85%).
  Balancing applies only to TRAIN rows (remapped text ID % 50 != 0). Validation
  rows bypass all answer-dependent caps. This does not undo the separate question
  normalization/ambiguous-pair filter above; validation is not an unfiltered population.
  Historical assembled datasets are not rewritten by this future-assembly change.
  replay       as many LatentQA examples as causal ones, at most the file's.
Examples written by gen after this change already carry positions; they are passed through.

  assemble.py OUT.jsonl GEN.jsonl [GEN.jsonl ...] --replay LATENTQA.jsonl [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

NONE = "No clear answer."
ABSTAIN_MAX, UNCHANGED_MAX = 0.30, 0.70
WHICH_Q = "Which earlier token is the model's next-token prediction most dependent on? Answer with the token and its position."


def q_tok(s: str) -> str:
    return "'" + s.replace("\n", "\\n") + "'"


def mix(rows: list[dict]) -> dict:
    out = {}
    for cat in sorted({r["cat"] for r in rows}):
        xs = [r["a"] for r in rows if r["cat"] == cat]
        d = {"n": len(xs)}
        if cat in ("which", "pair"):
            d["abstain"] = round(sum(a == NONE for a in xs) / len(xs), 3)
        if cat == "swap":
            d["changed"] = round(sum(a.startswith("Yes") for a in xs) / len(xs), 3)
        if cat == "edit":
            d["changed"] = round(sum(not a.startswith("Unchanged") for a in xs) / len(xs), 3)
        out[cat] = d
    return out


def cap(rows: list[dict], bad, share: float, rng: random.Random) -> list[dict]:
    """rows with those where bad(row) kept at most share of the result (a random subset)."""
    good = [r for r in rows if not bad(r)]
    worse = [r for r in rows if bad(r)]
    keep = int(share / (1 - share) * len(good))
    rng.shuffle(worse)
    return good + worse[:keep]


def cap_training(rows: list[dict], bad, share: float, rng: random.Random) -> list[dict]:
    """Balance TRAIN only, using the already remapped text IDs authoritative to training.

    Validation membership never depends on its answers, nor consumes balancing RNG.
    Earlier normalization and ambiguous-question filters remain separate.
    """
    train = [r for r in rows if r["text"] % 50 != 0]
    validation = [r for r in rows if r["text"] % 50 == 0]
    return cap(train, bad, share, rng) + validation


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("gens", type=Path, nargs="+")
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rng = random.Random(a.seed)
    rows = []
    for k, f in enumerate(a.gens):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut by the copy from the pod
                continue
            r["text"] = r["text"] * len(a.gens) + k  # the processes' text numbers kept apart
            rows.append(r)
    print("before", json.dumps(mix(rows)))
    by = defaultdict(list)
    dropped = 0
    for r in rows:
        if r["cat"] == "which" and "position" not in r["a"] and r["a"] != NONE:
            p = r["truth"]
            r["q"], r["a"] = WHICH_Q, f"{q_tok(tok.decode([r['ctx'][p]]))} at position {p}."
        elif r["cat"] == "which":
            r["q"] = WHICH_Q
        if r["cat"] == "pair" and "position" not in r["q"]:
            names = re.match(r"Which matters more for the model's next token: '(.*)' or '(.*)'\?$", r["q"], re.S)
            pos = []
            for s in names.groups() if names else ():
                ids = tok(s.replace("\\n", "\n"), add_special_tokens=False)["input_ids"]
                hits = [i for i, t in enumerate(r["ctx"]) if len(ids) == 1 and t == ids[0]]
                pos.append(hits[0] if len(hits) == 1 else None)
            if len(pos) != 2 or None in pos:
                dropped += 1
                continue
            (s1, s2), (p1, p2) = names.groups(), pos
            r["q"] = f"Which matters more for the model's next token: '{s1}' at position {p1} or '{s2}' at position {p2}?"
            if r["a"] != NONE:  # (the answer is the winner's quoted string, as the question quotes it)
                r["a"] = f"{r['a']} at position {p1 if r['a'] == chr(39) + s1 + chr(39) else p2}."
        by[r["cat"]].append(r)
    by["which"] = cap_training(by["which"], lambda r: r["a"] == NONE, ABSTAIN_MAX, rng)
    by["pair"] = cap_training(by["pair"], lambda r: r["a"] == NONE, ABSTAIN_MAX, rng)
    by["swap"] = cap_training(by["swap"], lambda r: r["a"] == "No.", UNCHANGED_MAX, rng)
    by["edit"] = cap_training(by["edit"], lambda r: r["a"].startswith("Unchanged"), UNCHANGED_MAX, rng)
    causal = [r for v in by.values() for r in v]
    print("after (TRAIN caps only; validation bypasses caps)", json.dumps(mix(causal)), f"(pairs dropped for a recurring named token: {dropped})")
    replay = a.replay.read_text().splitlines()
    rng.shuffle(replay)
    replay = replay[:len(causal)]
    with a.out.open("w") as fh:
        for r in causal:
            fh.write(json.dumps(r) + "\n")
        for line in replay:
            fh.write(line + "\n")
    print(f"{len(causal)} causal + {len(replay)} replay examples to {a.out}; categories {dict(Counter(r['cat'] for r in causal))}")


if __name__ == "__main__":
    main()
