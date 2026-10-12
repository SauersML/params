"""Teacher circuits with every connection stated (#2951 graph oracle): each <task>.json of DIR with its edges replaced
by every connection the model has among its nodes (native.all_edges: readers' writers, and the residual writers at the
targets into the prediction), written to OUT; the no-pruning arm of the edge gates.

  ~/mpd-data/venv/bin/python experiments/oracle/every_connection.py DIR OUT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))

import native  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(args.dir.glob("text*.json")):
        if not re.fullmatch(r"text\d+", f.stem):
            continue
        a = json.loads(f.read_text())
        targets = native.text(native.TEXTS / "vpd4l" / f"{f.stem}.json")[1]
        at = {(m, t, c): i for i, (m, c, t) in enumerate(a["nodes"]) if m != "tok"}
        parents, out = native.all_edges(list(at), targets)
        edges = [[at[w], at[r]] for r, ws in parents.items() for w in ws] + [[at[w], "out"] for w in out]
        (args.out / f.name).write_text(json.dumps({**a, "edges": edges}))
        n += 1
    print(f"{n} circuits with every connection in {args.out}")


if __name__ == "__main__":
    main()
