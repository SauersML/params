"""Per family: the transfer matrix (s_point_c, s_track, s_interchange of member A's circuit mapped onto member B), the
own circuit, outside-family controls, the leave-one-out core (complete and stated), the universal core and VPD's top k
at the same sizes; pass rates at 0.5 (half of VPD's own answer on goal_eval's log scale)."""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
METRICS = ("s_point_c", "s_track", "s_interchange")
PASS = 0.5
FILES = ["transfer_a.jsonl", "transfer_b.jsonl", "transfer_backfill.jsonl", "transfer_rev.jsonl"]


POD = Path("/Users/user/mpd-data/runpod")


def load(files):
    """Rows of every file (in DATA) and every pod run's transfer outputs; a (family, target, kind, source, size)
    scored twice (a target run here and on a pod) keeps its first row."""
    paths = [DATA / f for f in files] + sorted(POD.glob("oracle-graph-generality*/out/transfer*.jsonl"))
    rows, seen = [], set()
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["family"], r["target"], r["kind"], r["source"], r["nodes"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def per_target(rows):
    """{(family, target): {"own": row, "mapped": {source: row}, "control": [rows], "core": row, "core_stated": row,
    "universal": row, "vpd": {size: row}}}"""
    out = collections.defaultdict(lambda: {"mapped": {}, "control": [], "vpd": {}})
    for r in rows:
        d = out[(r["family"], r["target"])]
        if r["kind"] == "mapped":
            d["mapped"][r["source"]] = r
        elif r["kind"] == "control":
            d["control"].append(r)
        elif r["kind"] == "vpd":
            d["vpd"][r["nodes"]] = r
        else:
            d[r["kind"]] = r
    return out


def vpd_at(d, n):
    return d["vpd"].get(n) if n > 0 else None


def m(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summary(rows):
    pt = per_target(rows)
    fams = collections.defaultdict(list)
    for (f, t), d in pt.items():
        fams[f].append((t, d))
    res = {}
    for f, items in fams.items():
        s = {"members": [t for t, _ in items], "n": len(items)}
        for k in METRICS:
            own = [d["own"][k] for _, d in items if "own" in d]
            mp = [r[k] for _, d in items for r in d["mapped"].values()]
            mpv = [(vpd_at(d, r["nodes"]) or {}).get(k) for _, d in items for r in d["mapped"].values()]
            ctl = [r[k] for _, d in items for r in d["control"]]
            ctlv = [(vpd_at(d, r["nodes"]) or {}).get(k) for _, d in items for r in d["control"]]
            core = [d["core"][k] for _, d in items if "core" in d]
            corev = [(vpd_at(d, d["core"]["nodes"]) or {}).get(k) for _, d in items if "core" in d]
            cst = [d["core_stated"][k] for _, d in items if "core_stated" in d]
            uni = [d["universal"][k] for _, d in items if "universal" in d]
            crn = [d["core_random"][k] for _, d in items if "core_random" in d]
            crnv = [(vpd_at(d, d["core_random"]["nodes"]) or {}).get(k) for _, d in items if "core_random" in d]
            ownv = [(vpd_at(d, d["own"]["nodes"]) or {}).get(k) for _, d in items if "own" in d]
            s[k] = {"own": m(own), "vpd_own_size": m(ownv), "mapped": m(mp), "vpd_mapped_size": m(mpv), "control": m(ctl),
                    "vpd_control_size": m(ctlv), "core": m(core), "vpd_core_size": m(corev), "core_stated": m(cst),
                    "universal": m(uni), "core_random": m(crn), "vpd_core_random_size": m(crnv),
                    "core_pass": float(np.mean([x >= PASS for x in core if x is not None])) if core else None,
                    "mapped_pass": float(np.mean([x >= PASS for x in mp if x is not None])) if mp else None,
                    "own_pass": float(np.mean([x >= PASS for x in own if x is not None])) if own else None,
                    "vpd_core_pass": float(np.mean([x >= PASS for x in corev if x is not None])) if corev else None}
            # paired per target: mapped mean minus control mean
            diffs = []
            for _, d in items:
                a = [r[k] for r in d["mapped"].values() if r[k] is not None]
                b = [r[k] for r in d["control"] if r[k] is not None]
                if a and b:
                    diffs.append(np.mean(a) - np.mean(b))
            s[k]["mapped_minus_control"] = m(diffs)
            s[k]["mapped_minus_control_wins"] = f"{sum(x > 0 for x in diffs)}/{len(diffs)}"
        s["core_nodes"] = m([d["core"]["nodes"] for _, d in items if "core" in d])
        s["core_random_nodes"] = m([d["core_random"]["nodes"] for _, d in items if "core_random" in d])
        s["own_nodes"] = m([d["own"]["nodes"] for _, d in items if "own" in d])
        s["mapped_nodes"] = m([r["nodes"] for _, d in items for r in d["mapped"].values()])
        all3 = [all((d["core"][k] or 0) >= PASS for k in METRICS) for _, d in items if "core" in d]
        s["core_pass_all3"] = float(np.mean(all3)) if all3 else None
        s["core_pass_point_interchange"] = float(np.mean([all((d["core"][k] or 0) >= PASS for k in ("s_point_c", "s_interchange"))
                                                         for _, d in items if "core" in d])) if items else None
        res[f] = s
    return res, pt


def matrix(pt, family, k="s_point_c"):
    items = sorted([(t, d) for (f, t), d in pt.items() if f == family], key=lambda x: int(x[0][4:]))
    ts = [t for t, _ in items]
    M = np.full((len(ts), len(ts)), np.nan)  # row: source A, column: target B
    for j, (t, d) in enumerate(items):
        M[j, j] = d["own"][k] if d["own"][k] is not None else np.nan
        for i, a in enumerate(ts):
            if a in d["mapped"] and d["mapped"][a][k] is not None:
                M[i, j] = d["mapped"][a][k]
    return ts, M


if __name__ == "__main__":
    rows = load(sys.argv[1:] or FILES)
    res, pt = summary(rows)
    for f, s in res.items():
        print(f"{f}  n={s['n']}  nodes own {s['own_nodes']:.0f} mapped {s['mapped_nodes']:.0f} core {s['core_nodes'] or 0:.0f}")
        for k in METRICS:
            x = s[k]
            g = lambda v: f"{v:.2f}" if v is not None else "  - "  # noqa: E731
            print(f"   {k:14s} own {g(x['own'])} (vpd {g(x['vpd_own_size'])})  mapped {g(x['mapped'])} (vpd {g(x['vpd_mapped_size'])})"
                  f"  control {g(x['control'])}  core {g(x['core'])} (vpd {g(x['vpd_core_size'])}) stated {g(x['core_stated'])}"
                  f"  random core {g(x['core_random'])} (vpd {g(x['vpd_core_random_size'])})  universal {g(x['universal'])}  core pass {g(x['core_pass'])}  mapped-control {g(x['mapped_minus_control'])} {x['mapped_minus_control_wins']}")
