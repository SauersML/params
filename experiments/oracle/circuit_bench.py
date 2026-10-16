"""The circuit finder's benchmark (the lead's 10-16 CIRCUIT PUSH, ~/mpd-data/team/circuits_push_1016.txt): one command
that runs a finder version on the frozen dev set and reports S1 and Q1-Q5, the same way every time.
  finder  the version in FINDER_DIR (a copy of method/oracle at some commit: circuit_finder.py and what it imports) run
          through its own command line in a separate interpreter, `circuit_finder.py questions DEVSET RECORDS --split
          SPLIT --model MODEL`: every dev question's own pair (user against corrupt_user), one card, its records written
  scoring this file's commit's neuron_scorecard (fixed across finder versions; lean: no random baseline, curves or
          interactions), float32, the dev set's questions as its question file
  S1  seconds a question: the finder records' own "seconds" (find() from setup to record, its untimed warm-up first);
      median, mean, and questions a card-hour at one question at a time (3600 / mean)
  Q1  pass rate on the own pair: every cut of the circuit (the finder's source and mover cuts; an attribution or ADAG
      circuit is one cut) alone loses at least PASS of the gap by noising and restores at least PASS by denoising, as
      the scorecard measures it; over every dev question (no record, no circuit or an invalid pair: a miss). Beside it
      the finder's own met rate and the rate with all cuts patched together (the scorecard's pair: an easier test,
      since one passing cut carries the union)
  Q2  unseen pairs of the same family: the scorecard's held-out partner draws; a question passes when both mean shares
      reach PASS; the pass rate over questions with draws, and the median shares
  Q3  size at pass: the median neuron poles + heads over the Q1 passes (and over every circuit)
  Q4  paraphrase stability: the scorecard's transfer (the circuit's units mapped by role onto each of the question's
      other paraphrases, the dev set's "family", noised and denoised there): the pass rate (both mean shares at least
      PASS) and the median shares; and the effect-weighted overlap of the paraphrases' own circuits where the records
      hold them (median, share at least Q4_AT)
  Q5  robustness: monitorbench's circuit_robust.check (method/oracle) on every record, a bfloat16 Patcher after the
      scorecard: per met cut, its shares with every unit outside it resampled from in-distribution prompts (the pool:
      the dev set and POOL); a question passes when every met cut passes resampled; the per-cut rates beside the
      standard ones, and the median resampled lost and restored
  The sealed test set (any question with split "test") is refused unless --sealed.

  circuit_bench.py run DEVSET OUT [--finder-dir DIR] [--model Qwen/Qwen3-8B] [--split dev] [--rev REV] [--limit N]
      writes OUT/records/ (the finder's), OUT/score/ (the scorecard's) and OUT/bench.json, and prints the table
  circuit_bench.py score RECORDS_DIR OUT --devset DEVSET [--model M]   the same metrics on records already built (ADAG's,
      attribution baselines', any finder's): S1 from their own "seconds"
  circuit_bench.py table OUT [OUT ...]   bench.json files side by side (before and after a change, same card)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POOL = Path.home() / "mpd-data" / "circuits" / "trainset_8b.json"
PASS = 0.7
Q4_AT = 0.85


def load_devset(path: Path) -> dict:
    """{id: question} from a dev set file: {"questions": [...]} or a list of questions."""
    d = json.loads(path.read_text())
    return {q["id"]: q for q in (d["questions"] if isinstance(d, dict) else d)}


def metrics(recs: list[dict], finder_recs: dict, n_questions: int, robust: list[dict] | None = None) -> dict:
    """S1 and Q1-Q5 (module docstring) from the scorecard's records and the finder's."""
    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None

    def share(x):
        return None if x is None else max(-1.0, min(1.0, x))
    secs = [r.get("seconds") for r in finder_recs.values() if r.get("seconds") is not None]
    out = {"questions": n_questions, "finder_records": len(finder_recs), "scored": len(recs)}
    out["S1"] = {"median_s": med(secs), "mean_s": float(np.mean(secs)) if secs else None,
                 "per_card_hour": 3600 / float(np.mean(secs)) if secs else None}
    q1, q1u = [], []
    for r in recs:
        pr = r.get("pair") or {}
        cuts = r.get("cuts") or {}
        q1.append(bool(pr.get("valid") and cuts and all(x["lost"] >= PASS and x["restored"] >= PASS for x in cuts.values())))
        q1u.append(bool(pr.get("valid") and (pr.get("lost") or 0) >= PASS and (pr.get("restored") or 0) >= PASS))
    out["Q1"] = {"pass_rate": sum(q1) / max(n_questions, 1), "passes": sum(q1), "union_pass_rate": sum(q1u) / max(n_questions, 1),
                 "finder_met_rate": sum(bool(r.get("met")) for r in finder_recs.values()) / max(n_questions, 1),
                 "valid_pairs": sum(bool((r.get("pair") or {}).get("valid")) for r in recs)}
    q2 = []
    for r in recs:
        d = r.get("draws") or {}
        nh, sh = d.get("necessity_heldout") or [], d.get("sufficiency_heldout") or []
        if nh and sh:
            a, b = float(np.mean([share(x["share"]) for x in nh])), float(np.mean([share(x["share"]) for x in sh]))
            q2.append((a, b, a >= PASS and b >= PASS))
    out["Q2"] = {"pass_rate": sum(x[2] for x in q2) / len(q2) if q2 else None, "questions": len(q2),
                 "lost_median": med([x[0] for x in q2]), "restored_median": med([x[1] for x in q2])}
    size = [r["size"]["poles"] + r["size"]["heads"] for r in recs if "size" in r]
    size_pass = [r["size"]["poles"] + r["size"]["heads"] for r, ok in zip(recs, q1) if ok]
    out["Q3"] = {"median_units_at_pass": med(size_pass), "median_units": med(size),
                 "median_poles_at_pass": med([r["size"]["poles"] for r, ok in zip(recs, q1) if ok]),
                 "median_heads_at_pass": med([r["size"]["heads"] for r, ok in zip(recs, q1) if ok])}
    st = [float(np.mean([x["share"] for x in r["stability"]])) for r in recs if r.get("stability")]
    tr = [(float(np.mean([share(x["lost"]) for x in r["transfer"]])), float(np.mean([share(x["restored"]) for x in r["transfer"]])))
          for r in recs if r.get("transfer")]
    out["Q4"] = {"transfer_pass_rate": sum(a >= PASS and b >= PASS for a, b in tr) / len(tr) if tr else None, "transfer_questions": len(tr),
                 "transfer_lost_median": med([a for a, _ in tr]), "transfer_restored_median": med([b for _, b in tr]),
                 "overlap_median": med(st), "overlap_share_at_least": sum(x >= Q4_AT for x in st) / len(st) if st else None,
                 "overlap_questions": len(st)}
    out["Q5"] = robust_metrics(robust or [])
    return out


def robust_metrics(rob: list[dict]) -> dict:
    """Q5 from circuit_robust.check's records: per met cut the resampled and standard pass, the question pass."""
    if not rob:
        return {"questions": 0, "pass_rate": None}
    per, allp, lost, rest = {}, [], [], []
    for r in rob:
        cuts = r.get("cuts") or {}
        oks = []
        for name, v in cuts.items():
            ok = bool(v.get("passes"))
            std = v.get("standard") or [None, None]
            per.setdefault(name, {"resampled": 0, "standard": 0, "cuts": 0})
            per[name]["cuts"] += 1
            per[name]["resampled"] += ok
            per[name]["standard"] += bool(std[0] is not None and std[0] >= PASS and std[1] >= PASS)
            oks.append(ok)
            m = v.get("mean") or [None, None]
            lost.append(m[0])
            rest.append(m[1])
        if oks:
            allp.append(all(oks))
    med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else None  # noqa: E731
    return {"questions": len(allp), "pass_rate": sum(allp) / len(allp) if allp else None, "per_cut": per,
            "lost_median": med(lost), "restored_median": med(rest)}


def table(benches: dict) -> str:
    rows = [("card", lambda b: b.get("gpu")), ("finder version", lambda b: b.get("finder_rev")),
            ("questions / scored", lambda b: f"{b['metrics']['questions']} / {b['metrics']['scored']}"),
            ("S1 median s a question", lambda b: b["metrics"]["S1"]["median_s"]),
            ("S1 questions a card-hour", lambda b: b["metrics"]["S1"]["per_card_hour"]),
            ("Q1 own-pair pass rate", lambda b: b["metrics"]["Q1"]["pass_rate"]),
            ("   the finder's own met rate", lambda b: b["metrics"]["Q1"]["finder_met_rate"]),
            ("   all cuts patched together", lambda b: b["metrics"]["Q1"].get("union_pass_rate")),
            ("Q2 unseen-pair pass rate", lambda b: b["metrics"]["Q2"]["pass_rate"]),
            ("   median lost / restored", lambda b: "{} / {}".format(*(f"{v:.3f}" if v is not None else "-" for v in
                                                                         (b["metrics"]["Q2"]["lost_median"], b["metrics"]["Q2"]["restored_median"])))),
            ("Q3 median units at pass", lambda b: b["metrics"]["Q3"]["median_units_at_pass"]),
            ("   poles / heads at pass", lambda b: f"{b['metrics']['Q3']['median_poles_at_pass']} / {b['metrics']['Q3']['median_heads_at_pass']}"),
            ("Q4 transfer pass rate", lambda b: b["metrics"]["Q4"].get("transfer_pass_rate")),
            ("   median lost / restored", lambda b: "{} / {}".format(*(f"{v:.3f}" if v is not None else "-" for v in
                                                                         (b["metrics"]["Q4"].get("transfer_lost_median"),
                                                                          b["metrics"]["Q4"].get("transfer_restored_median"))))),
            ("   overlap median (records)", lambda b: b["metrics"]["Q4"].get("overlap_median")),
            ("Q5 resampled pass rate", lambda b: b["metrics"]["Q5"].get("pass_rate")),
            ("   per cut resampled/standard", lambda b: " ".join(f"{k}:{v['resampled']}/{v['standard']}/{v['cuts']}" for k, v in
                                                                 (b["metrics"]["Q5"].get("per_cut") or {}).items()) or "-"),
            ("   median lost / restored", lambda b: "{} / {}".format(*(f"{v:.3f}" if v is not None else "-" for v in
                                                                         (b["metrics"]["Q5"].get("lost_median"),
                                                                          b["metrics"]["Q5"].get("restored_median")))))]
    w = {n: max(16, len(n) + 2) for n in benches}
    lines = [f"{'':32}" + "".join(f"{n:>{w[n]}}" for n in benches)]
    for label, get in rows:
        vals = []
        for n, b in benches.items():
            v = get(b)
            vals.append(f"{(f'{v:.3f}' if isinstance(v, float) else str(v)) if v is not None else '-':>{w[n]}}")
        lines.append(f"{label:32}" + "".join(vals))
    return "\n".join(lines)


def run(a) -> None:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
    qs = load_devset(a.devset)
    refuse_sealed(qs, a.sealed)
    ids = list(qs)[:a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
    recs_dir, score_dir = a.out / "records", a.out / "score"
    dev = qs
    dev_path = a.out / "devset_run.json"  # (the finder's command line reads {"questions": [...]} with a split on each)
    dev_path.write_text(json.dumps({"questions": [{**qs[i], "split": qs[i].get("split", a.split)} for i in ids]}))
    t0 = time.time()
    cmd = [sys.executable, "-u", str(a.finder_dir / "circuit_finder.py"), "questions", str(dev_path), str(recs_dir),
           "--split", a.split, "--model", a.model] + (["--dtype", a.dtype] if a.dtype else [])
    env = dict(os.environ, PYTHONPATH=str(a.finder_dir))
    with open(a.out / "finder.log", "a") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(a.finder_dir)).returncode
    finder_seconds = time.time() - t0
    print(f"finder exit {rc} in {finder_seconds:.0f} s", flush=True)
    bench = score_records(recs_dir, score_dir, dev, ids, a.model, [a.devset] + a.pool)
    bench.update(devset=str(a.devset), finder_dir=str(a.finder_dir), finder_rev=a.rev, finder_exit=rc, finder_wall_seconds=finder_seconds)
    (a.out / "bench.json").write_text(json.dumps(bench, indent=1))
    print(table({a.out.name: bench}))


def score_records(recs_dir: Path, score_dir: Path, dev: dict, ids: list, model: str, pool: list | None = None) -> dict:
    """The scorecard (lean) on every record in recs_dir whose id is a dev question, and S1 and Q1-Q5 over ids."""
    import torch
    from transformers import AutoTokenizer

    import neuron_scorecard as ns
    finder_recs, files = {}, []
    for f in sorted(recs_dir.glob("*.json")) if recs_dir.exists() else []:
        r = json.loads(f.read_text())
        qid = qid_of(r)
        if qid in dev and qid not in finder_recs:
            finder_recs[qid] = r
            files.append(f)
    score_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model)
    m = ns.Model(ns.load_model(model, None, "cuda" if torch.cuda.is_available() else "cpu", torch.float32))
    for f in files:
        c = ns.read_circuit(f)
        if (score_dir / f"{c['name']}.json").exists() or not c["neurons"] and not c["heads"]:
            continue
        rec = ns.score(m, tok, c, dev, log=lambda s: None, lean=True)
        (score_dir / f"{c['name']}.json").write_text(json.dumps(rec))
    recs = ns.load(score_dir)
    ns.stability(recs)
    for r in recs:
        (score_dir / f"{r['task']}.json").write_text(json.dumps(r))
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    robust = []  # Q5: monitorbench's circuit_robust, a bfloat16 Patcher, on the finder records with cuts
    if pool is not None and any(r.get("cuts") for r in finder_recs.values()):
        import circuit_finder as cf
        import circuit_robust as cr
        from transformers import AutoModelForCausalLM
        rdir = score_dir.parent / "robust"
        rdir.mkdir(exist_ok=True)
        bm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else None).eval()
        bm.requires_grad_(False)
        pt = cf.Patcher(bm, tok)
        qpool = cr.question_pool(pool)
        for i, r in finder_recs.items():
            path = rdir / (i.replace("/", "__") + ".json")
            if path.exists():
                robust.append(json.loads(path.read_text()))
                continue
            if not r.get("cuts"):
                continue
            out = cr.check(pt, tok, {**r, "gap": r.get("gap", 0.0)}, dev[i], qpool)
            path.write_text(json.dumps(out))
            robust.append(out)
    return {"model": model, "records": str(recs_dir), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "metrics": metrics(recs, finder_recs, len(ids), robust)}


def qid_of(r: dict) -> str:
    """A record's question: its id, or a family pair record's "question/vs/partner" without the partner."""
    return str(r.get("id")).split("/vs/")[0]


def refuse_sealed(dev: dict, sealed: bool) -> None:
    if not sealed and any(q.get("split") == "test" for q in dev.values()):
        raise SystemExit("the sealed test set (split \"test\"): refused without --sealed")


def score_only(a) -> None:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
    dev = load_devset(a.devset)
    refuse_sealed(dev, a.sealed)
    ids = sorted({qid_of(json.loads(f.read_text())) for f in a.records.glob("*.json")} & set(dev))
    bench = score_records(a.records, a.out / "score", dev, ids, a.model, [a.devset] + a.pool)
    bench.update(devset=str(a.devset), finder_rev=a.rev)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bench.json").write_text(json.dumps(bench, indent=1))
    print(table({a.out.name: bench}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("devset", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--finder-dir", type=Path, default=HERE.parents[1] / "method" / "oracle")
    r.add_argument("--model", default="Qwen/Qwen3-8B")
    r.add_argument("--split", default="dev")
    r.add_argument("--rev", default=None, help="the finder version's commit, recorded")
    r.add_argument("--dtype", default=None, help="the finder's --dtype (default: its own auto)")
    r.add_argument("--limit", type=int, default=0)
    s = sp.add_parser("score")
    s.add_argument("records", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--devset", type=Path, required=True)
    s.add_argument("--model", default="Qwen/Qwen3-8B")
    s.add_argument("--rev", default=None)
    for x in (r, s):
        x.add_argument("--sealed", action="store_true", help="allow the sealed test set (split \"test\")")
        x.add_argument("--pool", type=Path, nargs="*", default=[p for p in (POOL,) if p.exists()],
                       help="question files for Q5's resampled prompts beside the dev set")
    t = sp.add_parser("table")
    t.add_argument("outs", nargs="+", type=Path)
    a = ap.parse_args()
    if a.cmd == "table":
        print(table({p.name: json.loads((p / "bench.json").read_text()) for p in a.outs}))
        return
    (run if a.cmd == "run" else score_only)(a)


if __name__ == "__main__":
    main()
