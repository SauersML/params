"""The copy template as a diagram (by role: the earlier occurrence o, the copied token a = o + 1, the target t) beside
its per-text scores on the held-out copy texts against VPD's top k and each text's influential-blocks start."""
import collections
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import templates  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
OUT = Path("/Users/user/mpd-data/figures/oracle/generality_copy_template.png")
BLUE, RED, GREEN, AMBER, TEAL, SOFT, INK = "#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#2A9D8F", "#9DB9DD", "#222222"


def setup():
    plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 16, "axes.titlesize": 20,
                         "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.titlelocation": "left", "axes.titlepad": 16, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def box(ax, x, y, text, color, w=1.7, h=0.55, named=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", linewidth=2.2 if named else 1.2,
                                edgecolor=INK if named else color, facecolor=color if named else "white", zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=13, color="white" if named else INK, zorder=4,
            fontweight="bold" if named else "normal")


def arrow(ax, p, q, color=INK, style="-|>", ls="-", rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color, linestyle=ls, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=14, shrinkB=14))


def diagram(ax):
    X = {"o": 0.0, "a": 3.6, "t": 7.2}
    for r, lab in (("o", "earlier occurrence\nof the current token"), ("a", "the copied token\n(the one after it)"), ("t", "the target")):
        ax.text(X[r], 8.75, lab, ha="center", va="bottom", fontsize=14, fontweight="bold")
    col = {"o": [(0, "token", SOFT, False), (1, "L0 MLP", SOFT, False), (2, "L1 keys #329, #119", GREEN, True), (3, "L1 value", SOFT, False)],
           "a": [(0, "token", SOFT, False), (1, "L0 MLP", SOFT, False), (2, "L1 query #316", GREEN, True),
                 (3, "L1 attention out", SOFT, False), (4, "L1 MLP", SOFT, False), (5, "L2 key", BLUE, False), (6, "L2 value", BLUE, False)],
           "t": [(0, "token", SOFT, False), (1, "L0, L1 MLP", SOFT, False), (5, "L2 query", BLUE, False), (6, "L2 attention out", BLUE, False),
                 (7, "L3 attention, MLP", SOFT, False), (8, "prediction", INK, True)]}
    for r, items in col.items():
        for y, lab, color, named in items:
            box(ax, X[r], y, lab, color, w=2.1, named=named)
        ys = [y for y, *_ in items]
        for y0, y1 in zip(ys, ys[1:]):
            arrow(ax, (X[r], y0), (X[r], y1), color="#999999")
    arrow(ax, (X["o"], 3), (X["a"], 3), color=GREEN, lw=2.6)
    ax.text((X["o"] + X["a"]) / 2, 3.32, "layer 1 reads the token before", fontsize=12, color=GREEN, ha="center")
    arrow(ax, (X["a"], 2), (X["o"], 2), color=GREEN, ls="--", style="<|-|>")
    ax.text((X["o"] + X["a"]) / 2, 2.32, "stated pair (every copy text)", fontsize=12, color=GREEN, ha="center")
    arrow(ax, (X["a"], 6), (X["t"], 6), color=BLUE, lw=2.6)
    ax.text((X["a"] + X["t"]) / 2, 6.32, "layer 2 copies a's token", fontsize=12, color=BLUE, ha="center")
    arrow(ax, (X["t"], 5), (X["a"], 5), color=BLUE, ls="--", style="<|-|>")
    ax.text((X["a"] + X["t"]) / 2, 5.32, "stated pair (token-specific)", fontsize=12, color=BLUE, ha="center")
    ax.set_xlim(-1.4, 8.6)
    ax.set_ylim(-0.6, 9.6)
    ax.axis("off")
    ax.set_title("Copying as one template by role (filled: the same subcomponents on every copy text)", fontsize=17)


def scores(ax):
    by = collections.defaultdict(dict)
    paths = [DATA / "copy_template.jsonl"] + templates.run_outputs("copy_template*.jsonl")
    held = {x["task"] for x in json.loads((DATA / "copy_heldout.json").read_text())["copy"]}
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r["task"] in held:
                    by[r["task"]].setdefault(r["name"] if r["kind"] != "influential_blocks" else "influential", r)
    kinds = [("copy_columns", "copy template (stated)", GREEN), ("copy_columns_all", "copy template (every connection)", TEAL),
             ("influential", "the text's own influential blocks", AMBER), ("vpd_listed", "VPD's top k, same listed size", RED),
             ("vpd_expanded", "VPD's top k, same subcomponent count", SOFT)]
    terms = [("s_point_c", "prediction"), ("s_track_pp", "follows changes"), ("s_interchange_pp", "interchange (path-patched)"),
             ("s_donor_near", "near donors"), ("c5", "harmonic C")]
    for task, d in by.items():
        tpl = d.get("copy_columns")
        if tpl:
            d["vpd_listed"] = d.get(f"vpd@{tpl['listed']}")
            d["vpd_expanded"] = d.get(f"vpd@{tpl['nodes']}")
    for j, (k, lab, color) in enumerate(kinds):
        labelled = False
        for i, (m, _) in enumerate(terms):
            vals = [d[k][m] for d in by.values() if d.get(k) and d[k].get(m) is not None]
            if not vals:
                continue
            y = i + (j - 2) * 0.15
            ax.scatter(vals, np.full(len(vals), y) + np.random.default_rng(j * 7 + i).uniform(-0.04, 0.04, len(vals)),
                       s=14, color=color, alpha=0.35, zorder=2, linewidths=0)
            ax.scatter([np.mean(vals)], [y], s=110, color=color, edgecolor=INK, zorder=3, label=None if labelled else lab)
            labelled = True
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([t for _, t in terms])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("score on each held-out copy text (1 = as good as VPD's own answer)")
    ax.set_title(f"On {len(by)} held-out copy texts")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5), gridspec_kw={"width_ratios": [1.15, 1]})
    diagram(axes[0])
    scores(axes[1])
    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight", pad_inches=0.3)
    print(OUT)


if __name__ == "__main__":
    main()
