"""Each digit's input-layer circuit as full pixel-weight maps (~/mpd-data/figures/oracle/mnist_circuit_weights.png).

One row per digit class (the median-size input-layer circuit of that class's test digits): the test digit, then eight of
its circuit's named subcomponents, each drawn as its whole 28x28 read vector V[:, c] (what ink at every pixel does to the
subcomponent, scaled to its own largest weight; pixels almost never inked in training are grey, since their weights
are never trained), with the digit's outline on top and the stroke-feature name above.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/figure_weights.py --tag i5e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_decomp, load_mnist  # noqa: E402
from figure import BLUE, INK, RED, concept_name, setup, units  # noqa: E402

OUT = Path.home() / "mpd-data" / "figures" / "oracle" / "mnist_circuit_weights.png"
PER_ROW = 8


def pick_members(ex: dict, x, dec, K: dict) -> list[tuple[str, np.ndarray]]:
    """Eight named fc1 members of a circuit: the largest read on the digit for each distinct name first, then the rest
    by read size (as figure.py's panel A). Returns (member, full read vector V[:, c])."""
    fc1 = [m for m in ex["members"] if m.startswith("fc1#")]
    named = [m for m in fc1 if concept_name(K["named"].get(m))]
    size = {m: float(np.abs((x * dec.V["fc1"][:, int(m.split("#")[1])]).numpy()).sum()) for m in named}
    named.sort(key=lambda m: -size[m])
    pick, seen = [], set()
    for m in named:
        if concept_name(K["named"][m]) not in seen:
            seen.add(concept_name(K["named"][m]))
            pick.append(m)
    pick = (pick + [m for m in named if m not in pick])[:PER_ROW]
    return [(m, dec.V["fc1"][:, int(m.split("#")[1])].numpy()) for m in pick]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="i5e-4")
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()
    setup()
    t, tau = args.tag, args.tau
    s = f"tau{tau}_s{tau / 2}"
    dec = load_decomp(DATA / f"vpd_{t}.pt")
    xtr, _, xte, _ = load_mnist()
    never = (xtr > 0).float().mean(0).numpy().reshape(28, 28) < 0.002  # pixels (almost) never inked in training
    R = json.loads((DATA / f"circuits_{t}_blocks-fc2-fc3_{s}.json").read_text())["results"]
    K = json.loads((DATA / f"concepts_{t}_blocks-fc2-fc3_{s}.json").read_text())
    rows = []
    for d in range(10):
        rs = sorted([r for r in R if r["label"] == d], key=units)
        if rs:
            rows.append(rs[len(rs) // 2])

    cmap = LinearSegmentedColormap.from_list("rb", [BLUE, "#ffffff", RED])
    cmap.set_bad("#d9d9d9")
    n = len(rows)
    fig = plt.figure(figsize=(2.55 * (PER_ROW + 1.3), 2.95 * n + 2.2))
    gs = fig.add_gridspec(n, PER_ROW + 2, width_ratios=[1.0, 0.3] + [1.0] * PER_ROW, wspace=0.08, hspace=0.62)
    im = None
    for i, ex in enumerate(rows):
        x = xte[ex["image"]]
        img = x.reshape(28, 28).numpy()
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        n_fc1 = sum(1 for m in ex["members"] if m.startswith("fc1#"))
        ax.set_title(f"digit {ex['label']}\n{n_fc1} subcomponents", fontsize=17, loc="center", fontweight="normal",
                     pad=6)
        for j, (m, v) in enumerate(pick_members(ex, x, dec, K)):
            a = fig.add_subplot(gs[i, 2 + j])
            w = np.ma.masked_where(never, v.reshape(28, 28))
            lim = float(np.abs(w).max())
            im = a.imshow(w, cmap=cmap, vmin=-lim, vmax=lim)
            a.contour(img, levels=[0.5], colors=INK, linewidths=0.9, alpha=0.55)
            a.set_title(textwrap.fill(concept_name(K["named"][m]), 16), fontsize=14.5, loc="center",
                        fontweight="normal", pad=5, linespacing=1.05)
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#cccccc")
    fig.suptitle("Each digit's input-layer circuit: eight named subcomponents, each shown as its weight on every pixel",
                 fontsize=26, fontweight="bold", x=0.5, y=0.925, ha="center")
    cax = fig.add_axes((0.40, 0.035, 0.30, 0.008))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[im.norm.vmin, 0, im.norm.vmax])
    cb.ax.set_xticklabels(["ink here lowers it", "0", "ink here raises it"], fontsize=17)
    cb.outline.set_visible(False)
    fig.text(0.74, 0.0395, "grey: pixels never inked in training", fontsize=15, va="center", color="#555555")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=80, bbox_inches="tight", pad_inches=0.35)
    print(f"wrote {OUT} (images {[r['image'] for r in rows]})")


if __name__ == "__main__":
    main()
