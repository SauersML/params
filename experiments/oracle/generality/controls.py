"""Matched controls for the copy context subcomponents (the lead's 10-12): is a subcomponent's cost at the copied token's
position specific to copy sources? Per held-out copy text (DATA/copy_heldout.json; a the copied token's position, o =
a - 1, d = t - a its distance from the target), CONTROLS matched positions on other texts (train split, no teacher
circuit, not a copy text): position p' = t' - d (the same distance from their target t'), its token's frequency within
a factor FREQ of the copied token's (unigram counts over every vpd4l task text), and not a copy source (the token
before p' is not the current token at t'). At p' (and p' - 1 for the subcomponents measured at o), in the real model
with the remainders live (goal_eval.real_removal), the KL at t' in bits of removing each subcomponent, beside the same
removal at a (o) on the copy text.

  controls.py measure [--max N]  -> DATA/controls.jsonl (one line per copy text, resumable)
  controls.py report
"""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))

import native  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
CONTROLS = 3
FREQ = 2.0
SUBS = [("h.1.mlp.c_fc", 2103, "a"), ("h.1.attn.q_proj", 316, "a"), ("h.0.mlp.c_fc", 53, "a"), ("h.2.attn.k_proj", 206, "a"),
        ("h.1.attn.k_proj", 329, "o")]


def frequencies():
    cnt = collections.Counter()
    for p in TEXTS.iterdir():
        if p.suffix == ".json" and not p.name.startswith("."):
            cnt.update(json.loads(p.read_text())["prompts"][0]["token_ids"])
    return cnt


def candidates(exclude: set):
    out = []
    for p in native.tasks("train"):
        if p.stem in exclude:
            continue
        ids, targets = native.text(p)
        out.append((p.stem, ids, targets[-1]))
    return out


def matches(cand, d, f, freq, rng, k):
    order = list(range(len(cand)))
    rng.shuffle(order)
    got = []
    for i in order:
        name, ids, t = cand[i]
        p = t - d
        if p < 2:
            continue
        g = freq[ids[p]]
        if not (f / FREQ <= g <= f * FREQ):
            continue
        if ids[p - 1] == ids[t]:  # a copy source: the token after an earlier occurrence of the current token
            continue
        got.append((name, p))
        if len(got) >= k:
            break
    return got


def measure(maxn: int):
    import goal_eval

    out = DATA / "controls.jsonl"
    done = {json.loads(line)["task"] for line in out.read_text().splitlines()} if out.exists() else set()
    held = json.loads((DATA / "copy_heldout.json").read_text())
    copies = held["copy"]
    exclude = {r["task"] for r in __import__("pickle").load(open(DATA / "rows.pkl", "rb"))}
    exclude |= {x["task"] for x in copies} | set(held.get("scanned", []))  # no text the copy scan looked at
    freq = frequencies()
    cand = candidates(exclude)
    ge = goal_eval.GoalEval(native.Native(), chunk=16)
    ge.nat.LENGTHS_KEPT = 1
    n = 0
    for x in copies:
        task = x["task"]
        if task in done:
            continue
        if maxn and n >= maxn:
            break
        n += 1
        ids, targets = native.text(TEXTS / f"{task}.json")
        t = targets[-1]
        occ = [j for j in range(t) if ids[j] == ids[t]]
        a = occ[-1] + 1
        d = t - a
        rng = random.Random(task)
        row = {"task": task, "d": d, "freq": freq[ids[a]], "copy": {}, "controls": []}
        tx = ge.text(ids, targets, native.task_seed(task), json.loads((TEXTS / f"{task}.json").read_text()))
        pos = {"a": a, "o": a - 1}
        kl = ge.real_removal(tx, [native.Graph([(s, pos[r], c)], complete=True) for s, c, r in SUBS])
        row["copy"] = {f"{s}#{c}": v["kl_real_minus"] for (s, c, r), v in zip(SUBS, kl)}
        for name, p in matches(cand, d, freq[ids[a]], freq, rng, CONTROLS):
            ids2, targets2 = native.text(TEXTS / f"{name}.json")
            tx2 = ge.text(ids2, targets2, native.task_seed(name), json.loads((TEXTS / f"{name}.json").read_text()))
            pos2 = {"a": p, "o": p - 1}
            kl2 = ge.real_removal(tx2, [native.Graph([(s, pos2[r], c)], complete=True) for s, c, r in SUBS])
            row["controls"].append({"task": name, "position": p, "freq": freq[ids2[p]],
                                    "kl": {f"{s}#{c}": v["kl_real_minus"] for (s, c, r), v in zip(SUBS, kl2)}})
            ge.probes.clear()
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(task, {k: round(v, 2) for k, v in row["copy"].items()},
              [{k: round(v, 2) for k, v in c["kl"].items()} for c in row["controls"]][:1], flush=True)
        native.free(ge.nat.dev)


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def report():
    rows = [json.loads(line) for line in (DATA / "controls.jsonl").read_text().splitlines()]
    print(f"{len(rows)} copy texts, {sum(len(r['controls']) for r in rows)} matched controls")
    out = {}
    for s, c, r in SUBS:
        k = f"{s}#{c}"
        cp = [row["copy"][k] for row in rows if row["controls"]]
        ct = [np.mean([x["kl"][k] for x in row["controls"]]) for row in rows if row["controls"]]
        diff = boot(np.array(cp) - np.array(ct))
        out[k] = {"copy": boot(cp), "control": boot(ct), "diff": diff, "wins": int(sum(a > b for a, b in zip(cp, ct))), "n": len(cp)}
        print(f"  {k:22s} at {'a' if r == 'a' else 'o'}: copy {np.mean(cp):.2f}  matched control {np.mean(ct):.2f}  "
              f"diff {diff[0]:+.2f} [{diff[1]:+.2f}, {diff[2]:+.2f}]  copy higher on {out[k]['wins']}/{len(cp)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "report"))
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    measure(args.max) if args.cmd == "measure" else report()
