"""Evaluate an oracle checkpoint or a baseline on the zoo (TOYLOOP lane; Check C: learning against trivial controls).

  python3 -B experiments/oracle/toyloop/evaluate.py gt empty active prior            # baselines, all splits
  python3 -B experiments/oracle/toyloop/evaluate.py ~/mpd-data/toyloop/sft_gt --swapped --limit 20

Predictors: a checkpoint directory (greedy decoding), or a baseline:
  gt      the ground truth's own completion (the ceiling, and its reward for reference)
  empty   no unit ("tokX --> out")
  active  every unit the prompt lists as active at the last position, each feeding out (reads the behaviour section)
  prior   the circuit of the nearest training toy by TF-IDF prompt similarity, unit names unchanged
With --swapped a checkpoint also runs on prompts whose weight section comes from another toy of the same family
(same split) while the behaviour section stays: scored against both toys' truths. An oracle that reads the weights
follows the weight donor; one that reads only the behaviour (which names active units) follows the original.

Splits: train (a fixed sample of --train_n), test_instance, test_family. Metrics per answer: parse rate, unit P/R/F1
and exact unit-set match (heads and neurons only), edge F1 (all edges, tok roles and out included), reward_exec and
its sufficiency/necessity parts; the truth's reward_exec on the same toy for reference. Results: results/eval_<name>.json
and .md (per split, and per split x family).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import circuit  # noqa: E402
import render  # noqa: E402
import reward_exec  # noqa: E402

MANIFEST = Path.home() / "mpd-data/toy_zoo/manifest.jsonl"
UNIT = re.compile(r"L\d+[HN]\d+")


# ---------------------------------------------------------------- data

def load_splits(train_n: int, limit: int | None, seed: int = 0) -> dict:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    out = {}
    for split in ("train", "test_instance", "test_family"):
        rs = [r for r in rows if r["split"] == split]
        if split == "train":
            rs = random.Random(seed).sample(rs, min(train_n, len(rs)))
        if limit:
            rs = rs[:limit]
        out[split] = [render.from_zoo(r, MANIFEST.parent) for r in rs]
    return out


# ---------------------------------------------------------------- baselines

def empty(toy, prompt):
    return "flowchart LR\n  tokX --> out\nEnglish: No unit carries the prediction."


def gt(toy, prompt):
    return render.target_text(toy.truth)


def active(toy, prompt):
    units = []
    for line in prompt.splitlines():
        if line.strip().startswith("active at last:"):
            units += [u.split(":")[0] for u in line.split(":", 1)[1].split()]
    units = list(dict.fromkeys(u for u in units if UNIT.fullmatch(u)))
    if not units:
        return empty(toy, prompt)
    return "flowchart LR\n" + "\n".join(f"  {u} --> out" for u in units) + "\nEnglish: The active units carry the prediction."


class Prior:
    """Nearest training toy by TF-IDF cosine over prompt words; answers with that toy's circuit."""

    def __init__(self, train_toys):
        self.train = [(t, render.render(t)) for t in train_toys]
        docs = [Counter(self._words(p)) for _, p in self.train]
        df = Counter(w for d in docs for w in d)
        self.idf = {w: math.log(len(docs) / c) + 1.0 for w, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    @staticmethod
    def _words(p):
        return re.findall(r"[A-Za-z_]+\d*|[-+]?\d+", p)

    def _vec(self, counts):
        v = {w: c * self.idf.get(w, 0.0) for w, c in counts.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / n for w, x in v.items()}

    def __call__(self, toy, prompt):
        q = self._vec(Counter(self._words(prompt)))
        sims = [(sum(q.get(w, 0.0) * x for w, x in v.items()), i) for i, v in enumerate(self.vecs)
                if self.train[i][0].name != toy.name]
        return render.target_text(self.train[max(sims)[1]][0].truth)


class Checkpoint:
    def __init__(self, path, max_new=256, threads=1):
        import train
        torch.set_num_threads(threads)
        self.train = train
        a = SimpleNamespace(init=str(path), model=None, lora=0, threads=threads)
        self.model, self.tok = train.load_policy(a)
        self.model.eval()
        self.args = SimpleNamespace(max_len=2048, max_new=max_new, temp=1.0)

    def __call__(self, toy, prompt):
        return self.train.sample(self.model, self.tok, prompt, 1, self.args, greedy=True)[0]


# ---------------------------------------------------------------- metrics

def prf(pred: set, gold: set) -> tuple:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else float(not gold)
    r = tp / len(gold) if gold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metrics(toy, completion: str, truth: dict | None = None) -> dict:
    truth = truth or toy.truth
    c = circuit.parse(completion)
    pu = {u for u in c.units if UNIT.fullmatch(u)}
    gu = {u for u in truth["units"] if UNIT.fullmatch(u)}
    p, r, f = prf(pu, gu)
    ge = {tuple(e) for e in truth["edges"]}
    s = reward_exec.score(toy, pu) if c.ok else {"reward": 0.0, "suff": 0.0, "nec": 0.0}
    return {"parse": float(c.ok), "unit_p": p, "unit_r": r, "unit_f1": f, "exact": float(pu == gu),
            "edge_f1": prf(c.edges, ge)[2], "reward": s["reward"], "suff": s["suff"], "nec": s["nec"],
            "n_units": len(pu)}


def gt_reward(toy) -> float:
    return reward_exec.score(toy, {u for u in toy.truth["units"] if UNIT.fullmatch(u)})["reward"]


def mean_rows(rows: list) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {"n": len(rows), **{k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}}


# ---------------------------------------------------------------- run

def run(name: str, predict, splits: dict, swapped: bool = False, log=print) -> dict:
    per, t0 = [], time.time()
    for split, toys in splits.items():
        by_fam = defaultdict(list)
        for t in toys:
            by_fam[t.meta["family"]].append(t)
        for i, toy in enumerate(toys):
            prompt = render.render(toy)
            comp = predict(toy, prompt)
            row = {"split": split, "family": toy.meta["family"], "id": toy.name, **metrics(toy, comp),
                   "gt_reward": gt_reward(toy), "completion": comp}
            if swapped:   # weights from another toy of the family, behaviour from this one
                donors = [d for d in by_fam[toy.meta["family"]] if d.name != toy.name]
                if donors:
                    donor = donors[i % len(donors)]
                    sc = predict(toy, render.render(toy, weights_from=donor))
                    own, don = metrics(toy, sc), metrics(donor, sc)
                    row.update(swap_donor=donor.name, swap_f1_own=own["unit_f1"], swap_f1_donor=don["unit_f1"],
                               swap_edge_f1_own=own["edge_f1"], swap_edge_f1_donor=don["edge_f1"],
                               swap_completion=sc)
            per.append(row)
            log(f"{name} {split} {toy.name}: unit_f1 {row['unit_f1']:.2f} reward {row['reward']:.2f} "
                f"(gt {row['gt_reward']:.2f}) {time.time() - t0:.0f}s")
    summary = {}
    for split in splits:
        rs = [r for r in per if r["split"] == split]
        if not rs:
            continue
        summary[split] = {"all": mean_rows(rs)}
        for fam in sorted({r["family"] for r in rs}):
            summary[split][fam] = mean_rows([r for r in rs if r["family"] == fam])
    return {"name": name, "summary": summary, "rows": per}


COLS = ["n", "parse", "unit_p", "unit_r", "unit_f1", "exact", "edge_f1", "reward", "suff", "nec", "gt_reward"]
SWAP = ["swap_f1_own", "swap_f1_donor", "swap_edge_f1_own", "swap_edge_f1_donor"]


def markdown(res: dict) -> str:
    cols = COLS + [c for c in SWAP if any(c in r for r in res["rows"])]
    lines = [f"# Toy-loop eval: {res['name']}", "", "| split | family | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for split, fams in res["summary"].items():
        for fam, m in fams.items():
            name = f"**{fam}**" if fam == "all" else fam
            lines.append(f"| {split} | {name} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictors", nargs="+", help="checkpoint dirs and/or baselines: gt empty active prior")
    ap.add_argument("--train_n", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="at most this many toys per split")
    ap.add_argument("--swapped", action="store_true")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    splits = load_splits(a.train_n, a.limit)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.predictors:
        if name == "prior":
            all_train = render.load_zoo(MANIFEST, "train")
            pred = Prior(all_train)
        elif name in ("gt", "empty", "active"):
            pred = globals()[name]
        else:
            pred = Checkpoint(Path(name).expanduser(), a.max_new, a.threads)
        tag = name if name in ("gt", "empty", "active", "prior") else Path(name).name
        res = run(tag, pred, splits, swapped=a.swapped and isinstance(pred, Checkpoint))
        (a.out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
        (a.out / f"eval_{tag}.md").write_text(markdown(res))
        print(markdown(res), flush=True)


if __name__ == "__main__":
    main()
