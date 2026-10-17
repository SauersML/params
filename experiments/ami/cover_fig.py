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


def by_size(out: str, summary_json: str):
    """Figure: the median units per token within 0.1 nats against model size (count and share, 95% intervals over
    windows), from cover_summary.py --json."""
    import matplotlib.pyplot as plt

    s = json.loads(Path(summary_json).read_text())
    params = {"Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-8B": 8.0}
    names = [n for n in params if n in s]
    figstyle.setup()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22, "xtick.labelsize": 20, "ytick.labelsize": 20})
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, key, ci, scale, ylab in ((axes[0], "median_units", "median_units_ci", 1e-3, "units a token needs, upper bound (thousands)"),
                                     (axes[1], "median_share", "median_share_ci", 100, "% of units at its positions, upper bound")):
        x = [params[n] for n in names]
        y = [s[n]["0.1"][key] * scale for n in names]
        lo = [s[n]["0.1"][ci][0] * scale for n in names]
        hi = [s[n]["0.1"][ci][1] * scale for n in names]
        ax.vlines(x, lo, hi, color=figstyle.BLUE, lw=4, alpha=0.5)
        ax.plot(x, y, color=figstyle.BLUE, lw=3, marker="o", ms=14)
        for xi, yi, top in zip(x, y, hi):
            ax.annotate(f"{yi:.0f}k" if key == "median_units" else f"{yi:.1f}%", (xi, top), xytext=(0, 8), va="bottom",
                        textcoords="offset points", fontsize=21, color=figstyle.INK, ha="center")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("Qwen3-", "") for n in names])
        ax.minorticks_off()
        ax.set_xlim(0.4, 12)
        ax.set_xlabel("Qwen3 model size (parameters)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(hi) * 1.25)
        figstyle.clean(ax)
    fig.suptitle("The share of the model one token needs falls as Qwen3 grows", x=0.06, ha="left", fontsize=25,
                 fontweight="bold")
    fig.tight_layout()
    figstyle.save(fig, out)


def by_steps(out: str, sets: list):
    """Figure: the median units per token within 0.1 nats against the mask search's training steps, on the tokens every
    run covers (95% intervals over tokens), sets = ["STEPS=COVER_DIR", ...]."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    figstyle.setup()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22, "xtick.labelsize": 20, "ytick.labelsize": 20})
    fig, ax = plt.subplots(figsize=(11, 7.5))
    runs = {}
    for s in sets:
        steps, d = s.split("=", 1)
        runs[int(steps)] = {(p.name, t["t"]): t["k"].get("0.1") for p in Path(d).iterdir()
                            if p.name.startswith("doc") and p.suffix == ".json" for t in json.loads(p.read_text())["targets"]}
    common = sorted(set.intersection(*[{i for i, v in r.items() if v is not None} for r in runs.values()]))
    x, y, lo, hi = [], [], [], []
    for steps, r in runs.items():
        k = np.array([r[i] for i in common], float)
        b = np.median(k[rng.integers(0, len(k), (2000, len(k)))], axis=1)
        x.append(steps)
        y.append(np.median(k))
        lo.append(np.percentile(b, 2.5))
        hi.append(np.percentile(b, 97.5))
    ax.vlines(x, lo, hi, color=figstyle.BLUE, lw=4, alpha=0.5)
    ax.plot(x, y, color=figstyle.BLUE, lw=3, marker="o", ms=14)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:,.0f}", (xi, yi), xytext=(14, 10), textcoords="offset points", fontsize=21, color=figstyle.INK,
                    ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.5, x[-1] * 2)
    ax.set_xlabel("training steps of the unit masks")
    ax.set_ylabel("units one token needs (KL within 0.1 nats)")
    figstyle.clean(ax)
    fig.suptitle("Each doubling of the mask training steps cuts the units\none Qwen3-0.6B token needs about threefold",
                 x=0.06, ha="left", fontsize=24, fontweight="bold")
    fig.tight_layout()
    figstyle.save(fig, out)
