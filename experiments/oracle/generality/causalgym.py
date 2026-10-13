"""CausalGym tasks on vpd4l (#2951 graph oracle; Arora, Jurafsky, Potts 2024, HF dataset aryaman/causalgym): templated
minimal pairs aligned span by span, each changing one known variable at a known span and flipping the next-token
preference ("The man near him" -> " is", "The men near him" -> " are").

  causalgym.py filter   -> DATA/causalgym_tasks.json: per task of the test split, vpd4l's accuracy (on each prompt of a
                           pair, the base's label above the source's on the base and the reverse on the source), the
                           spans where base and source differ (the variable's position), and the kept tasks (>= KEEP)

The dataset is read from ~/mpd-data/causalgym/raw/test.json (downloaded into its own directory; data only). Each span
is tokenized with vpd4l's tokenizer and the spans joined; the labels must be single tokens.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
RAW = Path.home() / "mpd-data/causalgym/raw/test.json"
KEEP = 0.8


def tokenize(tk, spans):
    """(token ids, per span the index of its last token); the "<|endoftext|>" span as the tokenizer's own id."""
    ids, ends = [], []
    for s in spans:
        if s == "<|endoftext|>":
            tid = tk.token_to_id("<|endoftext|>")
            ids += [tid] if tid is not None else []
        else:
            ids += tk.encode(s, add_special_tokens=False).ids
        ends.append(len(ids) - 1)
    return ids, ends


def filter_tasks():
    import mech
    import native

    tk = mech.tokenizer("vpd4l")
    rows = json.loads(RAW.read_text())
    nat = native.Native()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    out = {}
    for task, rs in sorted(by.items()):
        acc, diffs, n_bad, pairs = [], collections.Counter(), 0, []
        for r in rs:
            bl, sl = tk.encode(r["base_label"], add_special_tokens=False).ids, tk.encode(r["src_label"], add_special_tokens=False).ids
            if len(bl) != 1 or len(sl) != 1 or bl == sl:
                n_bad += 1
                continue
            bids, bends = tokenize(tk, r["base"])
            sids, sends = tokenize(tk, r["src"])
            if len(bids) != len(sids):  # same-length pairs only (the interchange tests patch position by position)
                n_bad += 1
                continue
            diff = [i for i, (x, y) in enumerate(zip(r["base"], r["src"])) if x != y]
            diffs.update(diff)
            lb = nat.reference([bids], [len(bids) - 1])[0, 0]
            ls = nat.reference([sids], [len(sids) - 1])[0, 0]
            acc += [float(lb[bl[0]] > lb[sl[0]]), float(ls[sl[0]] > ls[bl[0]])]
            pairs.append({"base": bids, "src": sids, "base_label": bl[0], "src_label": sl[0], "ends": bends,
                          "diff_spans": diff, "base_type": r["base_type"], "src_type": r["src_type"]})
        out[task] = {"pairs": len(pairs), "skipped": n_bad, "accuracy": float(np.mean(acc)) if acc else None,
                     "diff_spans": dict(diffs), "spans": rs[0]["base"], "kept": bool(acc) and float(np.mean(acc)) >= KEEP,
                     "examples": pairs[:64]}
        print(f"{task:28s} pairs {len(pairs):4d} skipped {n_bad:4d} accuracy {out[task]['accuracy'] if acc else float('nan'):.2f} "
              f"differing spans {dict(diffs)} spans {rs[0]['base']}", flush=True)
    (DATA / "causalgym_tasks.json").write_text(json.dumps(out))
    print("kept:", [t for t, v in out.items() if v["kept"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("filter",))
    a = ap.parse_args()
    filter_tasks()
