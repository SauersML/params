"""The oracle's SFT mix: the given example files whole, and AO replay examples (ao_mix.py's) sampled so that they make
SHARE of the whole (the lead's 10-15: AO's own mix replayed in 30-50% of batches from step 0), shuffled.

  mix.py OUT.jsonl --replay REPLAY.jsonl --share 0.4 PART.jsonl [PART.jsonl ...]
"""
import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("out", type=Path)
ap.add_argument("parts", type=Path, nargs="+")
ap.add_argument("--replay", type=Path, required=True)
ap.add_argument("--share", type=float, default=0.4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
rows = [x for p in a.parts for x in p.read_text().splitlines() if x.strip()]
rep = [x for x in a.replay.read_text().splitlines() if x.strip()]
n = round(len(rows) * a.share / (1 - a.share))
rows += rng.sample(rep, min(n, len(rep)))
rng.shuffle(rows)
a.out.write_text("\n".join(rows) + "\n")
cats = {}
for x in rows:
    c = json.loads(x)["cat"]
    c = "ao_replay" if c.startswith("ao_") else c
    cats[c] = cats.get(c, 0) + 1
print(f"{len(rows)} examples ({min(n, len(rep))} AO replay) to {a.out}: {cats}")
