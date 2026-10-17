"""Figure: how many units each token needs under the pretraining-mean background (ami.py cover), per model.
Left: the fewest units found per token within 0.1 nats against the token's position (median over windows' tokens at
each position, the band its interquartile range). Right: the same as a share of the units at the token's and earlier
positions.

  cover_fig.py OUT.png LABEL=COVER_DIR [LABEL=COVER_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "mpd-data" / "scratch" / "glead" / "figs"))
import figstyle  # noqa: E402

COLORS = [figstyle.TEAL, figstyle.BLUE, figstyle.PURPLE, figstyle.RED]
EDGES = [1, 2, 4, 8, 16, 32, 64, 128]


def load(d: Path) -> list:
    out = []
    for p in sorted(x for x in d.iterdir() if x.name.startswith("doc") and x.suffix == ".json"):
        out += json.loads(p.read_text())["targets"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("sets", nargs="+")
    a = ap.parse_args()
    import matplotlib.pyplot as plt

    figstyle.setup()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22, "xtick.labelsize": 19, "ytick.labelsize": 19})
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5))
    for c, s in zip(COLORS, a.sets):
        label, d = s.split("=", 1)
        ts = [t for t in load(Path(d)) if t["k"].get("0.1") is not None and t["t"] >= 1]
        for ax, f in ((axes[0], lambda t: t["k"]["0.1"]), (axes[1], lambda t: 100 * t["k"]["0.1"] / t["units_total"])):
            xs, med, lo, hi = [], [], [], []
            for e0, e1 in zip(EDGES, EDGES[1:]):
                v = [f(t) for t in ts if e0 <= t["t"] < e1]
                if v:
                    xs.append(np.sqrt(e0 * (e1 - 1)) if e1 - 1 > e0 else e0)
                    med.append(np.median(v))
                    lo.append(np.percentile(v, 25))
                    hi.append(np.percentile(v, 75))
            ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(xs, med, color=c, lw=3.5, marker="o", ms=9)
            ax.annotate(label, (xs[-1], med[-1]), xytext=(10, 0), textcoords="offset points", color=c, fontsize=22,
                        fontweight="bold", va="center")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("token's position in the text")
        figstyle.clean(ax)
    axes[0].set_ylabel("units needed (KL within 0.1 nats)")
    axes[1].set_ylabel("% of the units at its positions")
    fig.suptitle("Units a token needs when every other unit is set to its pretraining mean", x=0.06, ha="left",
                 fontsize=26, fontweight="bold")
    fig.tight_layout()
    figstyle.save(fig, a.out)


if __name__ == "__main__":
    main()
