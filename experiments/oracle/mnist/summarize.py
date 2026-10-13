"""Table of circuits.py results: per method, the median over images of each family's held-out KL (bits), agreement,
size, and the share of images passing every family at the run's bound.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/summarize.py FILE [FILE ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

COLS = ("point", "removal", "donor", "flips", "donor_far", "worst", "agree_removal", "donor_far_norm", "size")


def main() -> None:
    for f in sys.argv[1:]:
        d = json.loads(Path(f).expanduser().read_text())
        R, tau = d["results"], d["args"]["tau"]
        print(f"== {Path(f).name}  (n={len(R)}, tau={tau}, search={d['args'].get('tau_search')}, "
              f"blocks={d['args'].get('blocks', '')!r}, units={d['args'].get('units', 'vpd')})")
        print("method".ljust(15) + "".join(c[:10].rjust(11) for c in COLS) + "   pass")
        for m in R[0]["eval"]:
            med = [np.median([r["eval"][m][c] for r in R]) for c in COLS]
            ps = np.mean([r["eval"][m]["worst"] <= tau for r in R])
            print(m.ljust(15) + "".join(f"{v:11.3f}" for v in med) + f"   {100 * ps:.0f}%")
        s = [np.median([r["search"][c] for r in R]) for c in COLS]
        print("(search draws)".ljust(15) + "".join(f"{v:11.3f}" for v in s))
        if "n_subcomponents" in R[0]:
            print("subcomponents in circuit, median:", np.median([r["n_subcomponents"] for r in R]))


if __name__ == "__main__":
    main()
