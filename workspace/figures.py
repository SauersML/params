"""Figures from summary.json (#2951 WorkspaceBench, judge-free). One finding per figure.

  MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python -B workspace/figures.py [MODEL]
Writes ~/mpd-data/workspace-bench/results/<model>/fig_*.png.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

BLUE, RED, GREEN, AMBER, PURPLE, TEAL, SOFT, INK = "#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#7B5EA7", "#2A9D8F", "#9DB9DD", "#222222"
COLORS = {"operand": BLUE, "digit": SOFT, "remaining": TEAL, "I": RED, "answer": AMBER}
LABELS = {"operand": "an operand of the first step", "digit": "one digit of an operand", "remaining": "the other operand",
          "I": "the intermediate result", "answer": "the final answer"}
EXPR = {"subsub": "(a − b) − c", "subsubx": "c − (a − b)", "addmul": "(a + b) × c", "addmulx": "c + (a × b)",
        "muladd": "(a × b) + c", "sign": "(a − b) × c", "signpair": "(a − b) × c", "absval": "|a − b| × c",
        "mulmid": "(a × b) + (c × d)", "mulmul": "(a × b) + (c × d)"}


def setup():
    plt.rcParams.update({
        "font.family": "Helvetica Neue", "font.size": 18, "axes.titlesize": 22, "axes.titleweight": "bold",
        "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.titlelocation": "left", "axes.titlepad": 16, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white"})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(path)


def role_label(role, token):
    names = {"x": "a", "y": "b", "z": "c"}
    if ":" in role and not role.startswith("op"):
        k, place = role.split(":")
        return f"{names.get(k, k)} {['units', 'tens', 'hundreds', 'thousands'][int(place)]}"
    return token.replace("\n", "\\n").strip() or "␣"


def causal_map(fam, f, out):
    """Plurality group per (layer, position role) across items, shaded by its share of items."""
    cols = defaultdict(list)
    for m in f["maps"]:
        for p, ro in enumerate(m["roles"]):
            if not ro.startswith("start"):
                cols[ro].append(p)
    order = sorted(cols, key=lambda ro: np.mean(cols[ro]))
    tok_of = {}
    for m in f["maps"]:
        for p, ro in enumerate(m["roles"]):
            tok_of.setdefault(ro, m["tokens"][p])
    Lp1 = len(f["maps"][0]["assign"])
    img = np.ones((Lp1, len(order), 3))
    for c, ro in enumerate(order):
        for l in range(Lp1):
            cnt = Counter(m["assign"][l][m["roles"].index(ro)] for m in f["maps"] if ro in m["roles"])
            g, n = cnt.most_common(1)[0]
            if g != "none":
                share = n / sum(cnt.values())
                img[l, c] = 1 - share * (1 - np.array(to_rgb(COLORS[g])))
    fig, ax = plt.subplots(figsize=(max(12, 0.42 * len(order) + 4), 9))
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=(-0.5, len(order) - 0.5, -1.5, Lp1 - 1.5))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([role_label(ro, tok_of[ro]) for ro in order], rotation=90)
    ax.set_ylabel("layer (−1 = token embedding)")
    ax.set_title(f"What each cell of Qwen3-8B holds while computing {EXPR[fam]} ({f['items']} problems)")
    for g, col in COLORS.items():
        ax.plot([], [], "s", color=col, markersize=14, label=LABELS[g])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=15)
    clean(ax)
    save(fig, out / f"fig_map_{fam}.png")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
    out = Path.home() / "mpd-data/workspace-bench/results" / model
    s = json.loads((out / "summary.json").read_text())
    setup()
    for fam, f in s["arith"].items():
        causal_map(fam, f, out)


if __name__ == "__main__":
    main()
