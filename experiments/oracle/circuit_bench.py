"""The circuit finder's benchmark (the lead's 10-16 CIRCUIT PUSH, ~/mpd-data/team/circuits_push_1016.txt): one command
that runs a finder version on the frozen dev set and reports S1 and Q1-Q5, the same way every time.
  finder  the version in FINDER_DIR (a copy of method/oracle at some commit: circuit_finder.py and what it imports) run
          through its own command line in a separate interpreter, `circuit_finder.py questions DEVSET RECORDS --split
          SPLIT --model MODEL`: every dev question's own pair (user against corrupt_user), one card, its records written
  scoring this file's commit's neuron_scorecard (fixed across finder versions; lean: no curves or interactions), float32
          (--score-dtype bfloat16 where float32 does not fit the card: Qwen3-14B on 48 GB), the dev set's questions as
          its question file
  intervals and controls (the lead's 10-16 claims rule: matched controls, n and an interval with every number): every
      rate with its count and its 95% Wilson interval, every median with its 95% bootstrap interval over questions
      (seeded); Q1's matched control: each cut's units redrawn at random at the same layers and positions (the
      scorecard's random sets, RANDOM_DRAWS a cut), the question passing when every cut's random set passes as Q1's
      cuts must; Q2-Q5 are read beside it
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
      standard ones, and the median resampled lost and restored. Only a record's met cuts are checked (an ADAG
      record's one neuron cut where its own measurement met; attribution baselines' records have no cuts, so no Q5)
  Q1 is printed both ways: a finder cut that is diffuse (its smallest passing set over 256 units: the record's "diffuse"
      cuts, or a cut's "diffuse"/spread.diffuse) excepted, and counted as a miss (the reviewer's strict number)
  Metrics no method optimizes (neuron_scorecard): zero ablation of the circuit's units on the own pair (pass: at least
      PASS of the gap lost, beside its random same-size set's); the whole next-token distribution's KL shares on the own
      pair and on unseen partners (pass: both at least PASS)
  Sizes and seconds: median and p90. Q5's met cuts with no usable resample (circuit_robust's no_resample) are counted
      apart and left out of its rates.
  The sealed test set (any question with split "test") is refused unless --sealed.
  circuit_bench.py leaderboard OUT --label NAME [--arm "commit flags"]   one row of the version leaderboard
      (LEADERBOARD, every cell with its interval) from OUT/bench.json
  An organism (the audit model: a LoRA on the base): --adapter (with --merge, the finder's merged weights), --tokenizer
  and --system go to the finder; the scorecard and Q5 load the same adapter, and the scorecard's prompts carry the
  question's (or the record's) system turn and prefill.

  circuit_bench.py run DEVSET OUT [--finder-dir DIR] [--model Qwen/Qwen3-8B] [--split dev] [--rev REV] [--limit N]
      writes OUT/records/ (the finder's), OUT/score/ (the scorecard's) and OUT/bench.json, and prints the table
  circuit_bench.py score RECORDS_DIR OUT --devset DEVSET [--model M]   the same metrics on records already built (ADAG's,
      attribution baselines', any finder's): S1 from their own "seconds"
  circuit_bench.py table OUT [OUT ...]   bench.json files side by side (before and after a change, same card)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POOL = Path.home() / "mpd-data" / "circuits" / "trainset_8b.json"
LEADERBOARD = Path.home() / "mpd-data" / "circuits" / "leaderboard.tsv"
PASS = 0.7
Q4_AT = 0.85


def load_devset(path: Path) -> dict:
    """{id: question} from a dev set file: {"questions": [...]} or a list of questions."""
    d = json.loads(path.read_text())
    return {q["id"]: q for q in (d["questions"] if isinstance(d, dict) else d)}


def wilson(k: int, n: int, z: float = 1.96) -> list | None:
    """The 95% Wilson interval of a rate k / n."""
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, mid - half), min(1.0, mid + half)]


def median_ci(xs, draws: int = 2000) -> list | None:
    """The 95% percentile-bootstrap interval of a median, resampling questions (seeded: the same every time)."""
    xs = np.array([x for x in xs if x is not None], float)
    if len(xs) < 2:
        return None
    meds = np.median(xs[np.random.default_rng(0).integers(0, len(xs), (draws, len(xs)))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def p90(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 90)) if xs else None


def diffuse(fr: dict) -> list:
    """A finder record's diffuse cuts (4d159cb's "diffuse" list, a cut's "diffuse", or an older spread.diffuse)."""
    out = set(fr.get("diffuse") or [])
    for k, v in (fr.get("cuts") or {}).items():
        if isinstance(v, dict) and (v.get("diffuse") or (v.get("spread") or {}).get("diffuse")):
            out.add(k)
    return sorted(out)


def metrics(recs: list[dict], finder_recs: dict, n_questions: int, robust: list[dict] | None = None) -> dict:
    """S1 and Q1-Q5 (module docstring) from the scorecard's records and the finder's."""

    def share(x):
        return None if x is None else max(-1.0, min(1.0, x))
    secs = [r.get("seconds") for r in finder_recs.values() if r.get("seconds") is not None]
    out = {"questions": n_questions, "finder_records": len(finder_recs), "scored": len(recs),
           "score_seconds_median": med([r.get("seconds") for r in recs])}  # (the scorecard's own time a record: the bench's cost)
    out["S1"] = {"median_s": med(secs), "median_ci": median_ci(secs), "p90_s": p90(secs), "mean_s": float(np.mean(secs)) if secs else None,
                 "per_card_hour": 3600 / float(np.mean(secs)) if secs else None, "n": len(secs)}
    q1, q1u, ctrl, ctrl_lost, ctrl_rest, strict = [], [], [], [], [], []
    for r in recs:
        pr = r.get("pair") or {}
        cuts = r.get("cuts") or {}
        q1.append(bool(pr.get("valid") and cuts and all(x["lost"] >= PASS and x["restored"] >= PASS for x in cuts.values())))
        strict.append(q1[-1] and not diffuse(finder_recs.get(qid_of(r), {})))
        q1u.append(bool(pr.get("valid") and (pr.get("lost") or 0) >= PASS and (pr.get("restored") or 0) >= PASS))
        rc = [x.get("random") for x in cuts.values()]
        if pr.get("valid") and rc and all(rc):  # the matched control: every cut's random set under Q1's rule
            ctrl.append(all(x["lost"] >= PASS and x["restored"] >= PASS for x in rc))
            ctrl_lost += [x["lost"] for x in rc]
            ctrl_rest += [x["restored"] for x in rc]
    uctrl = [bool(x["lost"] >= PASS and x["restored"] >= PASS) for x in (r.get("random") or {} for r in recs
             if (r.get("pair") or {}).get("valid")) if x.get("lost") is not None]  # (the whole circuit's random set: the
    nq = max(n_questions, 1)                                                     # union's matched control)
    out["Q1"] = {"pass_rate": sum(q1) / nq, "passes": sum(q1), "n": n_questions, "pass_ci": wilson(sum(q1), n_questions),
                 "strict_passes": sum(strict), "strict_pass_rate": sum(strict) / nq, "strict_ci": wilson(sum(strict), n_questions),
                 "diffuse_records": sum(bool(diffuse(f)) for f in finder_recs.values()),
                 "union_pass_rate": sum(q1u) / nq, "union_ci": wilson(sum(q1u), n_questions),
                 "union_control": {"pass_rate": sum(uctrl) / len(uctrl) if uctrl else None, "passes": sum(uctrl), "n": len(uctrl),
                                   "pass_ci": wilson(sum(uctrl), len(uctrl))},
                 "finder_met_rate": sum(bool(r.get("met")) for r in finder_recs.values()) / nq,
                 "valid_pairs": sum(bool((r.get("pair") or {}).get("valid")) for r in recs),
                 "control": {"pass_rate": sum(ctrl) / len(ctrl) if ctrl else None, "passes": sum(ctrl), "n": len(ctrl),
                             "pass_ci": wilson(sum(ctrl), len(ctrl)), "cut_lost_median": med(ctrl_lost),
                             "cut_restored_median": med(ctrl_rest)}}
    q2 = []
    for r in recs:
        d = r.get("draws") or {}
        nh, sh = d.get("necessity_heldout") or [], d.get("sufficiency_heldout") or []
        if nh and sh:
            a, b = float(np.mean([share(x["share"]) for x in nh])), float(np.mean([share(x["share"]) for x in sh]))
            q2.append((a, b, a >= PASS and b >= PASS))
    out["Q2"] = {"pass_rate": sum(x[2] for x in q2) / len(q2) if q2 else None, "questions": len(q2),
                 "pass_ci": wilson(sum(x[2] for x in q2), len(q2)),
                 "lost_median": med([x[0] for x in q2]), "restored_median": med([x[1] for x in q2]),
                 "lost_ci": median_ci([x[0] for x in q2]), "restored_ci": median_ci([x[1] for x in q2])}
    # metrics no method optimizes: zero ablation and the whole distribution's KL (own pair; unseen partners)
    zo = [((r.get("zero") or {}).get("lost"), (r.get("random") or {}).get("zero_lost")) for r in recs if (r.get("pair") or {}).get("valid")]
    zp, zc = sum(a is not None and a >= PASS for a, _ in zo), sum(b is not None and b >= PASS for _, b in zo)
    out["Z"] = {"pass_rate": zp / nq, "passes": zp, "n": n_questions, "pass_ci": wilson(zp, n_questions),
                "lost_median": med([a for a, _ in zo]), "lost_ci": median_ci([a for a, _ in zo]),
                "control_passes": zc, "control_n": len(zo), "control_ci": wilson(zc, len(zo)), "control_lost_median": med([b for _, b in zo])}
    ko = [((r.get("kl") or {}).get("lost"), (r.get("kl") or {}).get("restored")) for r in recs if (r.get("pair") or {}).get("valid")]
    kp = sum(a is not None and b is not None and a >= PASS and b >= PASS for a, b in ko)
    kr = [((r.get("random") or {}).get("kl_lost"), (r.get("random") or {}).get("kl_restored")) for r in recs if (r.get("pair") or {}).get("valid")]
    kc = sum(a is not None and b is not None and a >= PASS and b >= PASS for a, b in kr)
    ku = []
    for r in recs:
        d = r.get("draws") or {}
        nl = [x.get("kl") for x in d.get("necessity_heldout") or [] if x.get("kl") is not None]
        sl = [x.get("kl") for x in d.get("sufficiency_heldout") or [] if x.get("kl") is not None]
        if nl and sl:
            ku.append((float(np.mean(nl)), float(np.mean(sl))))
    kup = sum(a >= PASS and b >= PASS for a, b in ku)
    out["KL"] = {"pass_rate": kp / nq, "passes": kp, "n": n_questions, "pass_ci": wilson(kp, n_questions),
                 "lost_median": med([a for a, _ in ko]), "restored_median": med([b for _, b in ko]),
                 "lost_ci": median_ci([a for a, _ in ko]), "restored_ci": median_ci([b for _, b in ko]),
                 "control_passes": kc, "control_n": len(kr), "control_ci": wilson(kc, len(kr)),
                 "unseen_pass_rate": kup / len(ku) if ku else None, "unseen_passes": kup, "unseen_n": len(ku), "unseen_ci": wilson(kup, len(ku)),
                 "unseen_lost_median": med([a for a, _ in ku]), "unseen_restored_median": med([b for _, b in ku])}
    size = [r["size"]["poles"] + r["size"]["heads"] for r in recs if "size" in r]
    size_pass = [r["size"]["poles"] + r["size"]["heads"] for r, ok in zip(recs, q1) if ok]
    out["Q3"] = {"median_units_at_pass": med(size_pass), "median_units_at_pass_ci": median_ci(size_pass), "n": len(size_pass),
                 "p90_units_at_pass": p90(size_pass),
                 "median_units": med(size),
                 "median_poles_at_pass": med([r["size"]["poles"] for r, ok in zip(recs, q1) if ok]),
                 "median_heads_at_pass": med([r["size"]["heads"] for r, ok in zip(recs, q1) if ok])}
    st = [float(np.mean([x["share"] for x in r["stability"]])) for r in recs if r.get("stability")]
    tr = [(float(np.mean([share(x["lost"]) for x in r["transfer"]])), float(np.mean([share(x["restored"]) for x in r["transfer"]])))
          for r in recs if r.get("transfer")]
    tp = sum(a >= PASS and b >= PASS for a, b in tr)
    out["Q4"] = {"transfer_pass_rate": tp / len(tr) if tr else None, "transfer_questions": len(tr), "transfer_ci": wilson(tp, len(tr)),
                 "transfer_lost_median": med([a for a, _ in tr]), "transfer_restored_median": med([b for _, b in tr]),
                 "transfer_lost_ci": median_ci([a for a, _ in tr]), "transfer_restored_ci": median_ci([b for _, b in tr]),
                 "overlap_median": med(st), "overlap_ci": median_ci(st),
                 "overlap_share_at_least": sum(x >= Q4_AT for x in st) / len(st) if st else None, "overlap_questions": len(st)}
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
            if v.get("no_resample"):                    # (no usable resample prompt: counted apart, out of the rates)
                per.setdefault(name, {"resampled": 0, "standard": 0, "cuts": 0, "no_resample": 0})
                per[name]["no_resample"] = per[name].get("no_resample", 0) + 1
                continue
            ok = bool(v.get("passes"))
            std = v.get("standard") or [None, None]
            per.setdefault(name, {"resampled": 0, "standard": 0, "cuts": 0, "no_resample": 0})
            per[name]["cuts"] += 1
            per[name]["resampled"] += ok
            per[name]["standard"] += bool(std[0] is not None and std[0] >= PASS and std[1] >= PASS)
            oks.append(ok)
            m = v.get("mean") or [None, None]
            lost.append(m[0])
            rest.append(m[1])
        if oks:
            allp.append(all(oks))
    return {"questions": len(allp), "pass_rate": sum(allp) / len(allp) if allp else None, "pass_ci": wilson(sum(allp), len(allp)),
            "per_cut": per, "lost_median": med(lost), "restored_median": med(rest)}


def fmt(v, ci=None, k=None, n=None) -> str:
    """A number with its interval and count: 0.620 [0.55, 0.69] 124/200."""
    if v is None:
        return "-"
    out = f"{v:.3f}" if isinstance(v, float) else str(v)
    if ci:
        out += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    if n is not None:
        out += f" {k}/{n}" if k is not None else f" n={n}"
    return out


def table(benches: dict) -> str:
    def g(b, q):
        return b["metrics"][q]

    def two(m, a, b, ca=None, cb=None):
        return f"{fmt(m.get(a), m.get(ca) if ca else None)} / {fmt(m.get(b), m.get(cb) if cb else None)}"

    def q1(b):
        m = g(b, "Q1")
        return fmt(m["pass_rate"], m.get("pass_ci"), m["passes"], m.get("n", b["metrics"]["questions"]))

    def ctrl(b):
        c = g(b, "Q1").get("control") or {}
        return fmt(c.get("pass_rate"), c.get("pass_ci"), c.get("passes"), c.get("n")) if c.get("n") else "-"

    def rate(q, key, ci, nkey):
        def get(b):
            m = g(b, q)
            v, n = m.get(key), m.get(nkey)
            return fmt(v, m.get(ci), round(v * n) if v is not None and n else None, n)
        return get
    rows = [("card", lambda b: b.get("gpu")), ("finder version", lambda b: b.get("finder_rev")),
            ("questions / scored", lambda b: f"{b['metrics']['questions']} / {b['metrics']['scored']}"),
            ("S1 median s a question", lambda b: fmt(g(b, "S1")["median_s"], g(b, "S1").get("median_ci"), None, g(b, "S1").get("n"))),
            ("   p90 s a question", lambda b: fmt(g(b, "S1").get("p90_s"))),
            ("S1 questions a card-hour", lambda b: fmt(g(b, "S1")["per_card_hour"])),
            ("Q1 own-pair pass rate (per cut)", q1),
            ("   diffuse counted as a miss", lambda b: fmt(g(b, "Q1").get("strict_pass_rate"), g(b, "Q1").get("strict_ci"),
                                                          g(b, "Q1").get("strict_passes"), g(b, "Q1").get("n"))),
            ("   random same-size cuts (control)", ctrl),
            ("   random cut lost / restored", lambda b: two(g(b, "Q1").get("control") or {}, "cut_lost_median", "cut_restored_median")),
            ("   the finder's own met rate", lambda b: fmt(g(b, "Q1")["finder_met_rate"])),
            ("   all cuts patched together", lambda b: fmt(g(b, "Q1").get("union_pass_rate"), g(b, "Q1").get("union_ci"))),
            ("   random same-size circuit (control)", lambda b: (lambda c: fmt(c.get("pass_rate"), c.get("pass_ci"), c.get("passes"), c.get("n"))
                                                              if c.get("n") else "-")(g(b, "Q1").get("union_control") or {})),
            ("Q2 unseen-pair pass rate", rate("Q2", "pass_rate", "pass_ci", "questions")),
            ("   median lost / restored", lambda b: two(g(b, "Q2"), "lost_median", "restored_median", "lost_ci", "restored_ci")),
            ("Q3 median units at pass", lambda b: fmt(g(b, "Q3")["median_units_at_pass"], g(b, "Q3").get("median_units_at_pass_ci"),
                                                       None, g(b, "Q3").get("n"))),
            ("   poles / heads at pass", lambda b: two(g(b, "Q3"), "median_poles_at_pass", "median_heads_at_pass")),
            ("   p90 units at pass", lambda b: fmt(g(b, "Q3").get("p90_units_at_pass"))),
            ("Zero ablation: lost >= 0.7", lambda b: (lambda z: fmt(z.get("pass_rate"), z.get("pass_ci"), z.get("passes"), z.get("n"))
                                                       if z else "-")(b["metrics"].get("Z") or {})),
            ("   random same-size units", lambda b: (lambda z: fmt(z["control_passes"] / z["control_n"], z.get("control_ci"),
                                                                  z["control_passes"], z["control_n"]) if z.get("control_n") else "-")(b["metrics"].get("Z") or {})),
            ("   median lost", lambda b: fmt((b["metrics"].get("Z") or {}).get("lost_median"), (b["metrics"].get("Z") or {}).get("lost_ci"))),
            ("Next-token KL, own pair: both >= 0.7", lambda b: (lambda z: fmt(z.get("pass_rate"), z.get("pass_ci"), z.get("passes"), z.get("n"))
                                                                 if z else "-")(b["metrics"].get("KL") or {})),
            ("   random same-size units", lambda b: (lambda z: fmt(z["control_passes"] / z["control_n"], z.get("control_ci"),
                                                                  z["control_passes"], z["control_n"]) if z.get("control_n") else "-")(b["metrics"].get("KL") or {})),
            ("   median lost / restored", lambda b: two(b["metrics"].get("KL") or {}, "lost_median", "restored_median", "lost_ci", "restored_ci")),
            ("Next-token KL, unseen partners", lambda b: (lambda z: fmt(z.get("unseen_pass_rate"), z.get("unseen_ci"), z.get("unseen_passes"),
                                                                        z.get("unseen_n")) if z.get("unseen_n") else "-")(b["metrics"].get("KL") or {})),
            ("Q4 transfer pass rate", rate("Q4", "transfer_pass_rate", "transfer_ci", "transfer_questions")),
            ("   median lost / restored", lambda b: two(g(b, "Q4"), "transfer_lost_median", "transfer_restored_median",
                                                     "transfer_lost_ci", "transfer_restored_ci")),
            ("   overlap median (records)", lambda b: fmt(g(b, "Q4").get("overlap_median"), g(b, "Q4").get("overlap_ci"), None,
                                                        g(b, "Q4").get("overlap_questions") or None)),
            ("Q5 resampled pass rate", rate("Q5", "pass_rate", "pass_ci", "questions")),
            ("   per cut resampled/standard/cuts", lambda b: " ".join(f"{k}:{v['resampled']}/{v['standard']}/{v['cuts']}"
                                                                 + (f"+{v['no_resample']}nr" if v.get("no_resample") else "") for k, v in
                                                                 (g(b, "Q5").get("per_cut") or {}).items()) or "-"),
            ("   median lost / restored", lambda b: two(g(b, "Q5"), "lost_median", "restored_median"))]
    cells = {n: [str(get(b)) if get(b) is not None else "-" for _, get in rows] for n, b in benches.items()}
    w = {n: max(len(n), *(len(x) for x in cells[n])) + 3 for n in benches}
    lines = [f"{'':36}" + "".join(f"{n:>{w[n]}}" for n in benches)]
    for i, (label, _) in enumerate(rows):
        lines.append(f"{label:36}" + "".join(f"{cells[n][i]:>{w[n]}}" for n in benches))
    return "\n".join(lines)


COLUMNS = ["date", "label", "arm", "model", "card", "questions", "S1 median s", "S1 p90 s", "Q1 per cut", "Q1 diffuse = miss",
           "own pair whole", "random whole", "Q2 unseen pairs", "Q4 unseen paraphrases", "Q3 units at pass", "Q3 p90",
           "zero ablation", "zero random", "KL own pair", "KL random", "KL unseen pairs", "Q5 movers resampled", "Q5 source resampled",
           "score s a record", "bench"]


def leaderboard_row(b: dict, label: str, arm: str, path: str) -> list:
    """One leaderboard row (COLUMNS) from a bench.json: every rate with its interval and count, medians with theirs."""
    m = b["metrics"]

    def rate(d, key, ci, k, n):
        v = d.get(key)
        return fmt(v, d.get(ci), d.get(k) if k else (round(v * d[n]) if v is not None and d.get(n) else None), d.get(n))
    q1, q2, q4, z, kl, q5 = m["Q1"], m["Q2"], m["Q4"], m.get("Z") or {}, m.get("KL") or {}, (m["Q5"].get("per_cut") or {})
    uc = q1.get("union_control") or {}
    cut = lambda k: (lambda v: f"{v['resampled']}/{v['cuts']}" if v else "-")(q5.get(k))  # noqa: E731
    return [time.strftime("%Y-%m-%d %H:%M"), label, arm, b.get("model"), b.get("gpu"), m["questions"],
            fmt(m["S1"]["median_s"], m["S1"].get("median_ci")), fmt(m["S1"].get("p90_s")),
            fmt(q1["pass_rate"], q1.get("pass_ci"), q1["passes"], q1.get("n")),
            fmt(q1.get("strict_pass_rate"), q1.get("strict_ci"), q1.get("strict_passes"), q1.get("n")),
            fmt(q1.get("union_pass_rate"), q1.get("union_ci")), fmt(uc.get("pass_rate"), uc.get("pass_ci"), uc.get("passes"), uc.get("n")),
            rate(q2, "pass_rate", "pass_ci", None, "questions"), rate(q4, "transfer_pass_rate", "transfer_ci", None, "transfer_questions"),
            fmt(m["Q3"]["median_units_at_pass"], m["Q3"].get("median_units_at_pass_ci")), fmt(m["Q3"].get("p90_units_at_pass")),
            fmt(z.get("pass_rate"), z.get("pass_ci"), z.get("passes"), z.get("n")),
            fmt(z["control_passes"] / z["control_n"], z.get("control_ci"), z["control_passes"], z["control_n"]) if z.get("control_n") else "-",
            fmt(kl.get("pass_rate"), kl.get("pass_ci"), kl.get("passes"), kl.get("n")),
            fmt(kl["control_passes"] / kl["control_n"], kl.get("control_ci"), kl["control_passes"], kl["control_n"]) if kl.get("control_n") else "-",
            fmt(kl.get("unseen_pass_rate"), kl.get("unseen_ci"), kl.get("unseen_passes"), kl.get("unseen_n")),
            cut("movers"), cut("source"), fmt(m.get("score_seconds_median")), path]


def leaderboard(a) -> None:
    b = json.loads((a.out / "bench.json").read_text())
    new = not LEADERBOARD.exists()
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD, "a") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(str(x) for x in leaderboard_row(b, a.label, a.arm or b.get("finder_rev") or "-", str(a.out))) + "\n")
    print(f"row {a.label} -> {LEADERBOARD}")


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
           "--split", a.split, "--model", a.model] + (["--dtype", a.dtype] if a.dtype else []) + (
           ["--adapter", a.adapter] if a.adapter else []) + (["--merge"] if a.merge else []) + (
           ["--tokenizer", a.tokenizer] if a.tokenizer else []) + (["--system", a.system] if a.system else [])
    env = dict(os.environ, PYTHONPATH=str(a.finder_dir))
    with open(a.out / "finder.log", "a") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(a.finder_dir)).returncode
    finder_seconds = time.time() - t0
    print(f"finder exit {rc} in {finder_seconds:.0f} s", flush=True)
    bench = score_records(recs_dir, score_dir, dev, ids, a.model, [a.devset] + a.pool, a.score_dtype, a.adapter, a.tokenizer)
    bench.update(devset=str(a.devset), finder_dir=str(a.finder_dir), finder_rev=a.rev, finder_exit=rc, finder_wall_seconds=finder_seconds)
    (a.out / "bench.json").write_text(json.dumps(bench, indent=1))
    print(table({a.out.name: bench}))


def score_records(recs_dir: Path, score_dir: Path, dev: dict, ids: list, model: str, pool: list | None = None,
                  dtype: str = "float32", adapter: str | None = None, tokenizer: str | None = None) -> dict:
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
    first = next(iter(finder_recs.values()), {})
    adapter, tokenizer = adapter or first.get("adapter"), tokenizer or first.get("tokenizer") or model  # (else the records')
    tok = AutoTokenizer.from_pretrained(tokenizer)
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    m = ns.Model(ns.load_model(model, adapter, dev_name, getattr(torch, dtype)))
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
        rdir = score_dir.parent / "robust"
        rdir.mkdir(exist_ok=True)
        bm = ns.load_model(model, adapter, dev_name, torch.bfloat16)
        bm = bm.merge_and_unload().eval().requires_grad_(False) if adapter else bm
        pt = cf.Patcher(bm, tok)
        qpool = cr.question_pool(pool)
        if hasattr(cr, "attach_swaps"):                 # (an organism's other swaps on its clean prompt as resamples)
            cr.attach_swaps(list(dev.values()))
        for i, r in finder_recs.items():
            path = rdir / (i.replace("/", "__") + ".json")
            if path.exists():
                robust.append(json.loads(path.read_text()))
                continue
            if not r.get("cuts"):
                continue
            pos = r.get("source_positions") or sorted({n["pos"] for n in r.get("nodes", [])})  # (an ADAG record's one cut: its
            out = cr.check(pt, tok, {**r, "gap": r.get("gap", 0.0), "source_positions": pos}, dev[i], qpool)  # neurons' positions)
            path.write_text(json.dumps(out))
            robust.append(out)
    return {"model": model, "adapter": adapter, "tokenizer": tokenizer, "records": str(recs_dir), "score_dtype": dtype, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
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
    bench = score_records(a.records, a.out / "score", dev, ids, a.model, [a.devset] + a.pool, a.score_dtype, a.adapter, a.tokenizer)
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
    r.add_argument("--merge", action="store_true", help="the finder's --merge (the adapter merged into the weights)")
    r.add_argument("--system", default=None, help="the finder's --system (a question's own \"system\" field wins)")
    s = sp.add_parser("score")
    s.add_argument("records", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--devset", type=Path, required=True)
    s.add_argument("--model", default="Qwen/Qwen3-8B")
    s.add_argument("--rev", default=None)
    for x in (r, s):
        x.add_argument("--adapter", default=None, help="an organism's LoRA (the finder's --adapter; the scorecard loads it)")
        x.add_argument("--tokenizer", default=None, help="the tokenizer and chat template (the finder's --tokenizer)")
        x.add_argument("--sealed", action="store_true", help="allow the sealed test set (split \"test\")")
        x.add_argument("--score-dtype", default="float32", choices=("float32", "bfloat16"),
                       help="the scorecard's dtype (bfloat16 where float32 does not fit the card)")
        x.add_argument("--pool", type=Path, nargs="*", default=[p for p in (POOL,) if p.exists()],
                       help="question files for Q5's resampled prompts beside the dev set")
    t = sp.add_parser("table")
    t.add_argument("outs", nargs="+", type=Path)
    lb = sp.add_parser("leaderboard")
    lb.add_argument("out", type=Path)
    lb.add_argument("--label", required=True)
    lb.add_argument("--arm", default=None, help="the arm: commit and flags, or the outside method and variant")
    a = ap.parse_args()
    if a.cmd == "leaderboard":
        leaderboard(a)
        return
    if a.cmd == "table":
        print(table({p.name: json.loads((p / "bench.json").read_text()) for p in a.outs}))
        return
    (run if a.cmd == "run" else score_only)(a)


if __name__ == "__main__":
    main()
