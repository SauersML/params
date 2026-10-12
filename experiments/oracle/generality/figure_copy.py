"""Held-out copy texts: the bits lost at the prediction when one shared context subcomponent is removed from the real
model at the copied token's position, at the earlier occurrence of the current token, at other positions, and at the
same position on texts that do not copy."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import copy_report  # noqa: E402


# the graph-oracle figures' look: white, large type, no grid, direct labels
BLUE, RED, GREEN, AMBER, TEAL, SOFT, INK = "#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#2A9D8F", "#9DB9DD", "#222222"


def setup():
    plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 16, "axes.titlesize": 20,
                         "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.titlelocation": "left", "axes.titlepad": 16, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3)

OUT = Path("/Users/user/mpd-data/figures/oracle/generality_copy_context.png")
NAMES = {"h.1.mlp.c_fc#2103": "layer 1 MLP input #2103", "h.1.attn.q_proj#316": "layer 1 query #316",
         "h.0.mlp.c_fc#53": "layer 0 MLP input #53", "h.2.attn.k_proj#206": "layer 2 key #206",
         "h.1.attn.k_proj#329": "layer 1 key #329", "all": "all five together"}


def main():
    res = copy_report.main()
    what = list(NAMES)
    n_copy, n_non = res[("copy", "all")]["n"], res[("noncopy", "all")]["n"]
    bars = [(("copy", "after_prev"), f"{n_copy} copy texts, removed at the copied token", BLUE),
            (("copy", "prev_occ"), f"{n_copy} copy texts, removed at the earlier occurrence of the current token", TEAL),
            (("copy", "control"), f"{n_copy} copy texts, removed at other positions", SOFT),
            (("noncopy", "after_prev"), f"{n_non} texts that do not copy, removed at the same role position", AMBER)]
    setup()
    fig, ax = plt.subplots(figsize=(15, 10))
    h = 0.2
    for j, ((kind, role), lab, color) in enumerate(bars):
        ys = np.arange(len(what)) + (j - 1.5) * h
        vals = [res[(kind, w)][role] for w in what]
        m = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        ax.barh(ys, m, height=h, color=color, label=lab, xerr=err, error_kw={"ecolor": INK, "lw": 1, "capsize": 2})
    ax.set_yticks(np.arange(len(what)))
    ax.set_yticklabels([NAMES[w] for w in what])
    ax.invert_yaxis()
    ax.set_xlabel("bits lost at the prediction when removed (mean over texts, 95% interval)")
    ax.set_title("On new texts, copying needs the same few subcomponents at the copied token")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=14)
    clean(ax)
    save(fig, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
