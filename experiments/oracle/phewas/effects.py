"""Stage 2's causal views of the neuron PheWAS (#2951): from the screen's knockout changes (screen.py runsets), per
trait how concentrated its knockout effects are (Mendelian or polygenic), and per tested pole its causal profile across
every trait (pleiotropy against random poles).

  effects.py concentration CHANGES.jsonl ... OUT.json   level-0 groups (ids "L0/<g>"): per trait the groups ranked by
      |change|, the cumulative share of the summed squared changes (the knockout-explained variance at group level) and
      the number of groups for 50% and 90% of it, beside the number for a trait whose effects were spread evenly
      (half and nine tenths of the groups)
  effects.py singles CHANGES.jsonl ... OUT.json         single poles (ids "S/...", stage 1's hits, and "R/..." random
      poles as the null): per pole its change on every trait; a trait counts toward a pole's pleiotropy when the
      change exceeds the 99th percentile of the random poles' |change| on that trait; per hit pole whether removing
      it moves its own trait against its association (the sign a causal, monotone pole would give)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(files: list[Path]) -> dict:
    return {r["id"]: r for f in files for r in map(json.loads, f.read_text().splitlines()) if r}


def concentration(a) -> None:
    res = load(a.changes)
    g = [r for i, r in res.items() if i.count("/") == 1 and i.startswith("L0/")]
    traits = sorted(g[0]["change"])
    out = {"groups": len(g), "poles_per_group": len(g[0]["units"]), "traits": {}}
    for t in traits:
        d = np.array([r["change"][t] for r in g])
        sq = np.sort(d ** 2)[::-1]
        cum = np.cumsum(sq) / max(sq.sum(), 1e-30)
        out["traits"][t] = {"sum_change": float(d.sum()), "sum_sq": float(sq.sum()), "max_abs": float(np.abs(d).max()),
                            "groups_50": int(np.searchsorted(cum, 0.5) + 1), "groups_90": int(np.searchsorted(cum, 0.9) + 1),
                            "top_share": float(cum[0]), "cum": cum[:64].tolist()}
    a.out.write_text(json.dumps(out))
    rows = sorted(out["traits"].items(), key=lambda kv: kv[1]["groups_50"])
    print(f"{len(g)} groups of {out['poles_per_group']} poles; groups for 50% / 90% of the knockout-explained variance "
          f"(evenly spread: {len(g) // 2} / {9 * len(g) // 10})")
    for t, v in rows[:10] + [("...", None)] + rows[-6:]:
        if v is None:
            print("  ...")
            continue
        print(f"  {t:60s} {v['groups_50']:4d} / {v['groups_90']:4d}  top group {v['top_share']:.2f}, max |change| {v['max_abs']:.2f}")


def singles(a) -> None:
    res = load(a.changes)
    origin = {d["id"]: d.get("from") for f in a.design for d in map(json.loads, f.read_text().splitlines())} if a.design else {}
    hits = {i: r for i, r in res.items() if i.startswith("S/")}
    rand = {i: r for i, r in res.items() if i.startswith("R/")}
    traits = sorted(next(iter(res.values()))["change"])
    out = {"traits": traits, "hits": len(hits), "random": len(rand)}
    if rand:
        R = np.array([[r["change"][t] for t in traits] for r in rand.values()])
        thr = np.percentile(np.abs(R), 99, axis=0)
        out["random_p99"] = dict(zip(traits, thr.tolist()))
        out["random_pleiotropy"] = (np.abs(R) > thr).sum(1).tolist()
    else:
        thr = None
    assoc = json.loads(a.assoc.read_text()) if a.assoc else {}
    poles = []
    for i, r in hits.items():
        d = np.array([r["change"][t] for t in traits])
        own = r.get("from") or origin.get(i)
        rec = {"id": i, "from": own, "change": dict(zip(traits, d.tolist()))}
        if thr is not None:
            rec["pleiotropy"] = int((np.abs(d) > thr).sum())
            rec["traits_moved"] = [t for t, x, h in zip(traits, d, thr) if abs(x) > h]
        if own and own in assoc:
            l_, n, pl = r["units"][0]
            h = next((h for h in assoc[own]["top"] if h["layer"] == l_ and h["neuron"] == n), None)
            if h:  # a pole whose activation raises the trait (t > 0) should lower it when removed
                rec["assoc_t"] = h["t_dev"] + h["t_held"]
                rec["against_association"] = bool(np.sign(rec["change"][own]) == -np.sign(h["effect"]))
        poles.append(rec)
    out["poles"] = poles
    a.out.write_text(json.dumps(out))
    if thr is not None:
        print(f"{len(poles)} hit poles, {len(rand)} random poles; pleiotropy (traits past the random 99th percentile): hits median "
              f"{np.median([p['pleiotropy'] for p in poles]):.0f}, random median {np.median(out['random_pleiotropy']):.0f}")
    by = {}
    for p in poles:
        by.setdefault(p["from"], []).append(p)
    for t, ps in sorted(by.items()):
        own = np.array([p["change"][t] for p in ps])
        moved = sum(abs(x) > thr[traits.index(t)] for x in own) if thr is not None else None
        agree = [p["against_association"] for p in ps if "against_association" in p]
        print(f"  {t:50s} {len(ps):3d} poles: |own change| median {np.median(np.abs(own)):.3f}, max {np.abs(own).max():.3f}"
              + (f", past the random 99th percentile {moved}" if moved is not None else "")
              + (f", sign against the association {sum(agree)}/{len(agree)}" if agree else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("concentration")
    c.add_argument("changes", type=Path, nargs="+")
    c.add_argument("out", type=Path)
    s = sub.add_parser("singles")
    s.add_argument("changes", type=Path, nargs="+")
    s.add_argument("out", type=Path)
    s.add_argument("--assoc", type=Path)
    s.add_argument("--design", type=Path, nargs="*", default=[], help="the designs (a hit's trait of origin)")
    a = ap.parse_args()
    {"concentration": concentration, "singles": singles}[a.cmd](a)


if __name__ == "__main__":
    main()
