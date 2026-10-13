"""Mechanism discrimination on vpd4l (mechdisc.py score's DATA/mechdisc_score.json): per mechanism family, the share of
items whose circuit (the v2.3 teacher's; S1's answers) includes the item's measured mechanism and not the other
mechanism, with 95% bootstrap intervals over items. Copy: the measured position a (whose replacement took the top token
off) in the circuit, and not only the target's two positions; far context: the measured position m, likewise; bigram:
no layer-1 previous-token pair (q #316 at j, k #329 at j - 1) at a context position. A dash marks a random context
position's chance to be in the circuit."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figure  # noqa: E402

OUT = Path("/Users/user/mpd-data/figures/oracle/generality_mechanism_discrimination.png")
SCORE = Path("/Users/user/mpd-data/graph_oracle/generality/mechdisc_score.json")
FAMILIES = [("copy", "induction copy\n(top token follows the current\ntoken's earlier occurrence)"),
            ("bigram", "local bigram\n(the current token alone\nsets the top token)"),
            ("context", "far context\n(one position 8+ tokens back\nsets it; not a copy)")]
SOURCES = [("teacher (v2.3)", "teacher circuit", figure.BLUE), ("S1", "S1's answer", figure.AMBER)]


def main():
    sc = json.loads(SCORE.read_text())
    srcs = [s for s in SOURCES if s[0] in sc]
    figure.setup()
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(FAMILIES))
    w = 0.8 / len(srcs)
    for i, (key, lab, col) in enumerate(srcs):
        off = (i - (len(srcs) - 1) / 2) * w
        cs = [sc[key][f]["right_not_wrong"] for f, _ in FAMILIES]
        v = np.array([c["mean"] if c else np.nan for c in cs])
        lo = v - np.array([c["lo"] if c else np.nan for c in cs])
        hi = np.array([c["hi"] if c else np.nan for c in cs]) - v
        ax.bar(x + off, v, w * 0.9, color=col, yerr=[lo, hi], capsize=4, error_kw={"lw": 1.2})
        ch = [sc[key][f]["chance"] for f, _ in FAMILIES]
        ax.scatter(x + off, [c["mean"] if c else np.nan for c in ch], marker="_", s=900, color=figure.INK, zorder=3)
        for xi, c in zip(x, cs):
            if c:
                ax.text(xi + off, c["hi"] + 0.03, f"{lab}\nn = {c['n']}", ha="center", fontsize=12, color=col)
    ax.set_xticks(x, [lab for _, lab in FAMILIES])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of items")
    ax.text(x[-1] + 0.45, 0.06, "dash: a random context\nposition in the circuit", ha="right", fontsize=12, color=figure.INK)
    figure.clean(ax)
    ax.set_title("Circuits that include the item's measured mechanism and not the other mechanism")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
