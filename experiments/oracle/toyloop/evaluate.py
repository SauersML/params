"""Evaluate an oracle checkpoint or a baseline on the zoo (TOYLOOP lane; Check C: learning against trivial controls).

  python3 -B experiments/oracle/toyloop/evaluate.py gt empty active prior            # baselines, all splits
  python3 -B experiments/oracle/toyloop/evaluate.py ~/mpd-data/toyloop/sft_gt --swapped --limit 20

Predictors: a checkpoint directory (greedy decoding), or a baseline:
  gt      the ground truth's own completion (the ceiling, and its reward for reference)
  empty   no unit ("tokX --> out")
  active  every unit the prompt lists as active at the last position, each feeding out (reads the behaviour section)
  prior   the circuit of the nearest training toy by TF-IDF prompt similarity, unit names unchanged
  union   every unit of the instance (zoo2: all modules; the instance-level union, not the per-text truth)
  weights weights_heuristic.py: walk back from the logits through render.wiring's weight gains (no LM, no texts)
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
    """{split: toys}. zoo v1: one toy per instance; zoo2: one toy per example text (per-text truth), train_n and limit
    then count texts, sampled across instances."""
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    v2 = any(r.get("version") == 2 for r in rows)
    out = {}
    for split in ("train", "test_instance", "test_family"):
        rs = [r for r in rows if r["split"] == split]
        if not v2:
            if split == "train":
                rs = random.Random(seed).sample(rs, min(train_n, len(rs)))
            out[split] = [render.from_zoo(r, MANIFEST.parent) for r in rs[:limit]]
            continue
        n = min(train_n, limit or train_n) if split == "train" else limit
        if n:   # about two texts per instance: load only the instances needed
            rs = random.Random(seed).sample(rs, min(len(rs), max(1, n // 2 + 1)))
        toys = [t for r in rs for t in render.from_zoo2(r, MANIFEST.parent)]
        out[split] = random.Random(seed).sample(toys, min(n, len(toys))) if n else toys
    return out


# ---------------------------------------------------------------- baselines

def empty(toy, prompt):
    return "flowchart LR\n  tokX --> out\nEnglish: No unit carries the prediction."


def gt(toy, prompt):
    return render.target_text(toy.truth)


def active(toy, prompt):
    """Units active at the last position of the toy's (first two) texts, computed from a run, whatever the prompt shows."""
    units = []
    for ids in toy.texts[:2]:
        r = render.run(toy.sd, toy.cfg, ids)
        units += [a.split(":")[0] for a in render.active_units(toy, r, len(ids) - 1)]
    units = list(dict.fromkeys(u for u in units if UNIT.fullmatch(u)))
    if not units:
        return empty(toy, prompt)
    return "flowchart LR\n" + "\n".join(f"  {u} --> out" for u in units) + "\nEnglish: The active units carry the prediction."


def union(toy, prompt):
    """Every unit of every module of the instance (the record's instance-level gt_units), each feeding out."""
    units = sorted({render.zoo_unit(u) for u in toy.meta["record"]["gt_units"]})
    return "flowchart LR\n" + "\n".join(f"  {u} --> out" for u in units) + "\nEnglish: Every mechanism's units."


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


def metrics(toy, completion: str, truth: dict | None = None, *, wiring: bool = False) -> dict:
    truth = truth or toy.truth
    c = circuit.parse(completion)
    pu = {u for u in c.units if UNIT.fullmatch(u)}
    gu = {u for u in truth["units"] if UNIT.fullmatch(u)}
    p, r, f = prf(pu, gu)
    ge = {tuple(e) for e in truth["edges"]}
    s = reward_exec.score(toy, pu) if c.ok else {"reward": 0.0, "suff": 0.0, "nec": 0.0}
    result = {"parse": float(c.ok), "unit_p": p, "unit_r": r, "unit_f1": f, "exact": float(pu == gu),
            "edge_f1": prf(c.edges, ge)[2], "reward": s["reward"], "suff": s["suff"], "nec": s["nec"],
            "n_units": len(pu)}
    if wiring:
        try:
            got = reward_exec.score_wiring(toy, completion, nat=reward_exec._state(toy)["nat"],
                                          texts=reward_exec.texts_of(toy))
            result.update(wiring_valid=1.0, wiring_suff=got["suff"], wiring_kl_bits=got["kl_bits"],
                          wiring_scope=got["scope"])
        except ValueError as exc:
            result.update(wiring_valid=0.0, wiring_suff=0.0, wiring_kl_bits=None, wiring_error=str(exc))
    return result


def gt_reward(toy) -> float:
    return reward_exec.score(toy, {u for u in toy.truth["units"] if UNIT.fullmatch(u)})["reward"]


def mean_rows(rows: list) -> dict:
    keys = list(dict.fromkeys(k for r in rows for k, v in r.items() if isinstance(v, (int, float))))
    out = {"n": len(rows)}
    for k in keys:   # swap_* exist only where a same-family donor was in the split
        vs = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = round(sum(vs) / len(vs), 3)
    return out


# ---------------------------------------------------------------- run

def run(name: str, predict, splits: dict, swapped: bool = False, log=print, *, wiring: bool = False) -> dict:
    per, t0 = [], time.time()
    for split, toys in splits.items():
        by_fam = defaultdict(list)
        for t in toys:
            by_fam[t.meta["family"]].append(t)
        for i, toy in enumerate(toys):
            prompt = render.render(toy)
            comp = predict(toy, prompt)
            row = {"split": split, "family": toy.meta["family"], "id": toy.name, **metrics(toy, comp, wiring=wiring),
                   "gt_reward": gt_reward(toy), "completion": comp}
            if swapped:   # weights from another toy of the family, behaviour from this one
                donors = [d for d in by_fam[toy.meta["family"]] if d.name != toy.name]
                if donors:
                    donor = donors[i % len(donors)]
                    sc = predict(toy, render.render(toy, weights_from=donor))
                    own, don = metrics(toy, sc, wiring=wiring), metrics(donor, sc, wiring=wiring)
                    row.update(swap_donor=donor.name, swap_f1_own=own["unit_f1"], swap_f1_donor=don["unit_f1"],
                               swap_edge_f1_own=own["edge_f1"], swap_edge_f1_donor=don["edge_f1"],
                               swap_completion=sc)
                    if wiring:
                        # Execute each completion against that model's own native basis/texts.
                        for label, measured in (("own", own), ("donor", don)):
                            for key in ("wiring_valid", "wiring_suff", "wiring_kl_bits", "wiring_scope", "wiring_error"):
                                if key in measured:
                                    row[f"swap_{key}_{label}"] = measured[key]
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
    cols += [c for c in ("wiring_valid", "wiring_suff", "wiring_kl_bits") if any(c in r for r in res["rows"])]
    cols += [f"swap_{key}_{label}" for label in ("own", "donor")
             for key in ("wiring_valid", "wiring_suff", "wiring_kl_bits")
             if any(f"swap_{key}_{label}" in r for r in res["rows"])]
    lines = [f"# Toy-loop eval: {res['name']}", "", "| split | family | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for split, fams in res["summary"].items():
        for fam, m in fams.items():
            name = f"**{fam}**" if fam == "all" else fam
            lines.append(f"| {split} | {name} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    global MANIFEST
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictors", nargs="+", help="checkpoint dirs and/or baselines: gt empty active prior weights union")
    ap.add_argument("--train_n", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="at most this many toys per split")
    ap.add_argument("--swapped", action="store_true")
    ap.add_argument("--wiring", action="store_true", help="also execute inter-unit/output edges; token roles and English remain unverified")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--zoo", type=Path, default=MANIFEST, help="a zoo manifest (v1 or zoo2)")
    ap.add_argument("--suffix", default=None, help="appended to the result names (default: _v2 on zoo2, plus "
                    "_weights under TOYLOOP_SECTIONS=weights)")
    a = ap.parse_args()
    MANIFEST = a.zoo.expanduser()
    if a.suffix is None:
        v2 = '"version": 2' in MANIFEST.read_text()[:2000]
        a.suffix = ("_v2" if v2 else "") + ("_weights" if render.SECTIONS == "weights" else "")
    torch.set_num_threads(a.threads)
    splits = load_splits(a.train_n, a.limit)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.predictors:
        if name == "prior":
            all_train = render.load_zoo(MANIFEST, "train")
            pred = Prior(all_train)
        elif name == "weights":
            import weights_heuristic
            pred = weights_heuristic.predict
        elif name in ("gt", "empty", "active", "union"):
            pred = globals()[name]
        else:
            pred = Checkpoint(Path(name).expanduser(), a.max_new, a.threads)
        tag = (name if name in ("gt", "empty", "active", "prior", "weights", "union") else Path(name).name) + a.suffix
        res = run(tag, pred, splits, swapped=a.swapped and isinstance(pred, Checkpoint), wiring=a.wiring)
        (a.out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
        (a.out / f"eval_{tag}.md").write_text(markdown(res))
        print(markdown(res), flush=True)


if __name__ == "__main__":
    main()
