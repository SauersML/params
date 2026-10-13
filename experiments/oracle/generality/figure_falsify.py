"""The falsification set on the held-out copy texts: per wrong copy circuit and per reward term, the share of texts on
which the term ranks the circuit the copy mechanism predicts above the wrong one (falsify.report's table)."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import falsify  # noqa: E402
import figure  # noqa: E402

OUT = Path("/Users/user/mpd-data/figures/oracle/generality_falsify.png")
ROWS = [("correct", "a_source_moved", "source moved to another earlier position"),
        ("correct", "b_fixed_offset", "source at a fixed distance"),
        ("correct", "g_mismatched", "source at another text's distance"),
        ("correct", "template", "copy route without the target's own parts"),
        ("correct", "d_irrelevant_parts", "plus irrelevant parts far from the copy"),
        ("c_with_backup", "correct", "with the second-latest occurrence beats without"),
        ("correct", "ref_vpd_correct_bits", "VPD's top subcomponents at the same bits"),
        ("correct", "ref_recency_correct_bits", "the target's active subcomponents (same bits or fewer)")]
COLS = [("reward", "reward"), ("phi", "Φ"), ("s_point", "prediction"), ("s_flips", "flips"), ("s_nodes", "node removal"),
        ("s_track_pp", "input changes"), ("s_interchange_pp", "interchange"), ("s_patching_near", "near donors"),
        ("s_patching_far", "far donors"), ("s_pattern", "attention"), ("s_N", "specificity")]


def main(kind: str = "copy"):
    out = falsify.report()
    M = np.full((len(ROWS), len(COLS)), np.nan)
    ns = []
    for i, (a, b, _) in enumerate(ROWS):
        for j, (m, _) in enumerate(COLS):
            v, n = out.get((kind, a, b, m), (None, 0))
            if v is not None:
                M[i, j] = v
        ns.append(max(out.get((kind, a, b, m), (None, 0))[1] for m, _ in COLS))
    figure.setup()
    cmap = LinearSegmentedColormap.from_list("wins", [figure.RED, "#F7F7F7", figure.BLUE])
    fig, ax = plt.subplots(figsize=(17, 8.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=14,
                        color="white" if abs(M[i, j] - 0.5) > 0.35 else figure.INK)
    ax.set_xticks(range(len(COLS)), [c for _, c in COLS], rotation=30, ha="right")
    ax.set_yticks(range(len(ROWS)), [f"{r}  (n={n})" for (_, _, r), n in zip(ROWS, ns)])
    ax.axhline(5.5, color="white", lw=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("How often each term ranks the copy circuit above the wrong one")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main(*sys.argv[1:])
