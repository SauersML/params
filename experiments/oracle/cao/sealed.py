"""The causal oracle's sealed test texts on Qwen3 (#2951, the lead's 10-14: written and committed before any label is
generated; untouched until the final evaluation, as vpd4l's text9100-9149).

Source: wikitext-103's validation split (public). The labels come from its train split (cao.py gen) and the gate
from its test split (h2h.py), so the three are disjoint by construction. N windows of T tokens (Qwen3's tokenizer,
the same for 0.6B to 8B), one per paragraph of at least T + 8 tokens, at a seeded random start.

  sealed.py OUT.json [--n 300] [--seed 9100]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SOURCE = ("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet")
T = 48


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    a = ap.parse_args()
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    df = pd.read_parquet(hf_hub_download(SOURCE[0], SOURCE[1], repo_type="dataset"))
    paras = [(i, t.strip()) for i, t in enumerate(df["text"].tolist()) if len(t.strip()) > 200 and not t.strip().startswith("=")]
    rng = random.Random(a.seed)
    rng.shuffle(paras)
    out = []
    for i, t in paras:
        ids = tok(" " + t, add_special_tokens=False)["input_ids"]
        if len(ids) >= T + 8:
            s = rng.randrange(0, len(ids) - T)
            out.append({"row": i, "start": s, "ids": ids[s:s + T]})
        if len(out) == a.n:
            break
    a.out.write_text(json.dumps({"source": list(SOURCE), "T": T, "seed": a.seed, "tokenizer": a.tokenizer,
                                 "texts": out}, separators=(",", ":")))
    print(f"{len(out)} sealed texts to {a.out}")


if __name__ == "__main__":
    main()
