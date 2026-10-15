"""Transfer score by family: own circuit vs other members' circuits mapped onto the text vs circuits from outside the
family mapped the same way vs the family's shared core (leave-one-out) vs VPD's top k at the mapped circuits' sizes;
one panel per scorecard family (s_point_c, s_track, s_interchange)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyze  # noqa: E402


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

OUT = Path("/Users/user/mpd-data/figures/oracle/generality_transfer_by_family.png")
KINDS = [("own", "own circuit (searched on this text)", INK, "o"),
         ("mapped", "other members' circuits, mapped", BLUE, "o"),
         ("control", "circuits from outside the family, mapped", RED, "o"),
         ("core", "shared core of the other members", GREEN, "D"),
         ("vpd_mapped_size", "VPD's top k at the mapped circuits' size", AMBER, "s"),
         ("vpd_core_size", "VPD's top k at the core's size", "none", "s")]
TITLE = "Circuits transfer to other texts of their family, not to texts outside the family; copying does not transfer"
PANELS = [("s_point_c", "Prediction recovered"), ("s_track", "Follows input changes"), ("s_interchange", "Interchange agreement")]


LABELS = {"S:.->\\n": '"." → newline', "S: of-> the": '" of" → various', "S:\\n->\\n": 'newline or ":" → newline',
          "S:,-> and": '"," → " and"', "S: that-> the": '" that" → various', "S: results->.": 'plural noun → "." or " of"',
          "S: to-> be": '" to" → verb', "S: is-> a": '" is", " was" → various', "S: a-> railroad": '" a" → noun or adjective',
          "S: the-> planet": '" the" → noun (cluster 1)', "S: the-> presence": '" the" → noun (cluster 2)',
          "S:ch->,": 'non-English word ending → ","', "S: in-> the": '" in", " into" → various', "S: it-> is": 'pronoun → " is"',
          "S:Let-> -": 'math → " -" or a digit', "B:prep->the": 'behaviour: preposition → " the"',
          "B:comma->function": 'behaviour: "," → " and", " the"', "B:copy": "behaviour: copies an earlier token"}


def label(name, n):
    return f"{LABELS.get(name, name)}  ({n})"


def main(files):
    res, _ = analyze.summary(analyze.load(files))
    fams = sorted(res, key=lambda f: (f.startswith("B:"), -(res[f]["s_point_c"]["mapped"] or 0)))
    allrow = {k: {kind: float(np.mean([res[f][k][kind] for f in fams if res[f][k][kind] is not None])) for kind, *_ in KINDS}
              for k, _ in PANELS}
    names = ["all families (mean)"] + [label(f, res[f]["n"]) for f in fams]
    vals = [allrow] + [res[f] for f in fams]
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(20, 0.5 * len(names) + 2.8), sharey=True)
    for ax, (k, title) in zip(axes, PANELS):
        ax.axhspan(-0.5, 0.5, color="#EEF3FA", zorder=0)
        for j, (kind, _, color, mk) in enumerate(KINDS):
            xs = [v[k][kind] for v in vals]
            ys = [i + (j - 2.5) * 0.12 for i in range(len(names))]
            hollow = color == "none"
            ax.scatter([x for x in xs if x is not None], [y for x, y in zip(xs, ys) if x is not None], s=80 if mk != "D" else 70,
                       facecolor="white" if hollow else color, marker=mk, zorder=3,
                       edgecolor=AMBER if hollow else "white", linewidth=1.6 if hollow else 0.6)
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("score (1 = as good as VPD's own answer)")
        clean(ax)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    handles = [plt.Line2D([], [], marker=mk, color=AMBER if c == "none" else c, markerfacecolor="white" if c == "none" else c,
                          markeredgewidth=1.6 if c == "none" else 0.6, linestyle="", markersize=11) for _, _, c, mk in KINDS]
    fig.legend(handles, [t for _, t, _, _ in KINDS], loc="lower center", ncol=3, frameon=False, fontsize=15,
               bbox_to_anchor=(0.55, -0.07))
    fig.suptitle(TITLE, x=0.01, ha="left", fontsize=21, fontweight="bold")
    fig.tight_layout()
    save(fig, OUT)
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or analyze.FILES)
