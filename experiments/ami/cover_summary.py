"""Coverage summaries (method/oracle/ami.py cover): per model, the fewest units per token found within each KL bound, as
counts and as shares of the units at positions up to the token, with 95% intervals by resampling windows (targets of
one window are not independent); the random control's KL at the 0.1 bound; how the count grows with the token's
position; and where the units sit (the token's own position against earlier ones; layers).

  cover_summary.py LABEL=COVER_DIR [LABEL=COVER_DIR ...] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOUNDS = ("1.0", "0.3", "0.1", "0.03")
POS_BINS = ((0, 1), (1, 8), (8, 32), (32, 64), (64, 128))


def load(d: Path) -> list:
    """Every window's targets, each with its window's index."""
    out = []
    for w, p in enumerate(sorted(x for x in d.iterdir() if x.name.startswith("doc") and x.suffix == ".json")):
        out += [{**t, "window": w} for t in json.loads(p.read_text())["targets"]]
    return out


def cluster_ci(ts: list, f, draws: int = 2000) -> list | None:
    """The 95% interval of the median of f over targets, windows resampled with replacement."""
    ws = sorted({t["window"] for t in ts})
    if len(ws) < 2:
        return None
    by = {w: [f(t) for t in ts if t["window"] == w and f(t) is not None] for w in ws}
    g = np.random.default_rng(0)
    meds = []
    for _ in range(draws):
        xs = [x for w in g.choice(ws, len(ws)) for x in by[w]]
        if xs:
            meds.append(np.median(xs))
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def summary(ts: list) -> dict:
    out = {"targets": len(ts), "windows": len({t["window"] for t in ts})}
    for b in BOUNDS:
        k = lambda t, b=b: t["k"].get(b)  # noqa: E731
        share = lambda t, b=b: None if t["k"].get(b) is None else t["k"][b] / t["units_total"]  # noqa: E731
        ks = [k(t) for t in ts if k(t) is not None]
        out[b] = {"median_units": float(np.median(ks)) if ks else None, "median_units_ci": cluster_ci(ts, k),
                  "median_share": float(np.median([share(t) for t in ts if share(t) is not None])) if ks else None,
                  "median_share_ci": cluster_ci(ts, share),
                  "needs_all": float(np.mean([k(t) == t["units_total"] for t in ts])),     # (the order's dip never reached it)
                  "p90_units": float(np.percentile(ks, 90)) if ks else None}
    rk = [t["kl_random_at_0.1"] for t in ts if t.get("kl_random_at_0.1") is not None]
    out["random_kl_at_0.1"] = {"median": float(np.median(rk)) if rk else None,
                               "median_empty": float(np.median([t["kl_empty"] for t in ts]))}
    out["by_position"] = []
    for lo, hi in POS_BINS:
        sel = [t for t in ts if lo <= t["t"] < hi and t["k"].get("0.1") is not None]
        if sel:
            out["by_position"].append({"positions": [lo, hi - 1], "n": len(sel), "median_units_0.1": float(np.median([t["k"]["0.1"] for t in sel])),
                                       "median_share_0.1": float(np.median([t["k"]["0.1"] / t["units_total"] for t in sel]))})
    at = [t["at_0.1"] for t in ts if t.get("at_0.1")]
    if at:
        out["same_position_share_0.1"] = float(np.median([a["same_position"] / max(1, a["neurons"] + a["heads"]) for a in at]))
        out["heads_share_0.1"] = float(np.median([a["heads"] / max(1, a["neurons"] + a["heads"]) for a in at]))
        L = len(at[0]["by_layer"])
        lay = np.array([np.array(a["by_layer"]) / max(1, sum(a["by_layer"])) for a in at])
        out["layer_share_0.1"] = [round(float(x), 4) for x in lay.mean(0)][:L]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="+", help="LABEL=COVER_DIR")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {}
    for s in a.sets:
        label, d = s.split("=", 1)
        res[label] = summary(load(Path(d)))
    for label, r in res.items():
        print(f"{label}: {r['targets']} tokens in {r['windows']} windows")
        for b in BOUNDS:
            x = r[b]
            print(f"  KL <= {b} nats: median {x['median_units']} units {x['median_units_ci']}, share {x["median_share"] if x["median_share"] is None else format(x["median_share"], ".2e")} "
                  f"{x['median_share_ci']}, needs every unit {x["needs_all"]:.2f}, p90 {x['p90_units']}")
        print(f"  random set of the 0.1 size: median KL {r['random_kl_at_0.1']['median']} (empty {r['random_kl_at_0.1']['median_empty']:.2f})")
        for p in r["by_position"]:
            print(f"  positions {p['positions']}: n {p['n']}, median units {p['median_units_0.1']:.0f}, share {p['median_share_0.1']:.2e}")
        if "same_position_share_0.1" in r:
            print(f"  at the token's own position {r['same_position_share_0.1']:.2f}; heads {r['heads_share_0.1']:.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
