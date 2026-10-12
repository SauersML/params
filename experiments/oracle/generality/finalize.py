"""Per family: members, the transfer matrices (row: the source member whose circuit is mapped, column: the target text;
the diagonal is the target's own circuit), the shared core (subcomponents more than half of all members' circuits list,
by role) and the leave-one-out core's pass rates; written to OUT/summary.json with the per-row data."""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import transfer  # noqa: E402

OUT = DATA


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = analyze.load(analyze.FILES)
    res, pt = analyze.summary(rows)
    by = {r["task"]: r for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
    fams = json.loads((DATA / "families.json").read_text())
    out = {}
    for f, s in res.items():
        mats = {}
        for k in analyze.METRICS:
            ts, M = analyze.matrix(pt, f, k)
            mats[k] = [[None if np.isnan(x) else round(float(x), 4) for x in row] for row in M]
        cn, ce, need = transfer.core([by[t] for t in fams[f]])
        out[f] = {"members": ts, "summary": s, "matrices": mats,
                  "core": {"threshold": need, "nodes": [[m, c, r if isinstance(r, str) else f"offset {r[1]}"] for m, c, r in cn],
                           "edges": len(ce)}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
