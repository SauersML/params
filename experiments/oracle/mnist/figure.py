"""The MNIST figure: one test image's circuit in VPD subcomponents, how often each kind of set stays faithful on fresh
draws, each intervention family's error, and the interchange test of the named variables.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/figure.py --tag i2e-3 --circuits FILE --neurons FILE --concepts FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODULES, load_decomp, load_mnist  # noqa: E402

BLUE, RED, GREEN, AMBER, PURPLE, TEAL = "#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#7B5EA7", "#2A9D8F"
SOFT = "#9DB9DD"
INK = "#222222"
OUT = Path.home() / "mpd-data" / "figures" / "oracle" / "mnist_circuits.png"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 20,
            "axes.titlesize": 23,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--example", type=int, default=None, help="test image index for panel A")
    args = ap.parse_args()
    setup()
    dec = load_decomp(DATA / f"vpd_{args.tag}.pt")
    _, _, xte, _ = load_mnist()
    C = json.loads(Path(args.circuits).read_text())["results"]
    N = json.loads(Path(args.neurons).read_text())["results"]
    K = json.loads(Path(args.concepts).read_text())["groups"]
    tau = args.tau

    fig = plt.figure(figsize=(26, 19))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1.05], hspace=0.42, wspace=0.42)

    # A: one image's circuit, its fc1 subcomponents as input templates
    ex = next((r for r in C if r["image"] == args.example), None) if args.example is not None else None
    if ex is None:
        ex = sorted(C, key=lambda r: r["k"])[len(C) // 4]
    fc1 = [m for m in ex["members"] if m.startswith("fc1")]
    fc1_ci = [ex["members_ci"][ex["members"].index(m)] for m in fc1]
    order = np.argsort(fc1_ci)[::-1][:7]
    sub = gs[0, 0].subgridspec(2, 4, wspace=0.12, hspace=0.35)
    ax = fig.add_subplot(sub[0, 0])
    ax.imshow(xte[ex["image"]].reshape(28, 28).numpy(), cmap="gray_r")
    ax.set_title("test image", fontsize=19, loc="center", fontweight="normal")
    ax.axis("off")
    counts = {m: sum(1 for s in ex["members"] if s.startswith(m)) for m in MODULES}
    for i, oi in enumerate(order):
        m = fc1[oi]
        c = int(m.split("#")[1])
        v = dec.V["fc1"][:, c].reshape(28, 28).numpy()
        lim = np.abs(v).max()
        a = fig.add_subplot(sub[(i + 1) // 4, (i + 1) % 4])
        a.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
        tag = "VPD-active" if fc1_ci[oi] > 0 else "VPD-inactive"
        a.set_title(f"fc1 #{c}\n{tag}", fontsize=16, loc="center", fontweight="normal")
        a.axis("off")
    fig.text(
        0.07,
        0.965,
        f"A circuit for one test digit: {ex['k']} subcomponents "
        f"({counts['fc1']} fc1, {counts['fc2']} fc2, {counts['fc3']} fc3); "
        f"VPD marks {ex['n_active']} active",
        fontsize=23,
        fontweight="bold",
    )

    # B: share of images faithful on fresh draws
    ax = fig.add_subplot(gs[0, 1])
    rows = [
        ("our circuit (VPD subcomponents)", C, "circuit", BLUE),
        ("VPD's top k by causal importance", C, "vpd_topk_ci", SOFT),
        ("top k by write size", C, "topk_write", SOFT),
        ("random k of VPD's active", C, "random_active", SOFT),
        ("VPD's whole active set", C, "vpd_active", AMBER),
        ("our circuit (hidden neurons)", N, "circuit", GREEN),
    ]
    vals, sizes = [], []
    for _, R, m, _ in rows:
        vals.append(100 * np.mean([r["eval"][m]["worst"] <= tau for r in R]))
        sizes.append(np.median([r["eval"][m]["size"] for r in R]))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=[r[3] for r in rows], height=0.62)
    for yi, v, s, r in zip(y, vals, sizes, rows, strict=True):
        ax.text(v + 1.5, yi, f"{v:.0f}%  (median size {s:.0f})", va="center", fontsize=18)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"test images faithful under all four tests (fresh draws, mean KL ≤ {tau} bits), %")
    ax.set_title("Faithfulness on fresh draws, sets of equal size")
    clean(ax)

    # C: per-family error, median over images
    ax = fig.add_subplot(gs[1, 0])
    fams = [
        ("point", "only the set on"),
        ("removal", "random parts of the set removed"),
        ("donor", "outside from a same-class image"),
        ("flips", "random outside parts switched on"),
        ("donor_far", "outside from another-class image"),
    ]
    series = [
        ("our circuit", C, "circuit", BLUE, "o"),
        ("VPD's top k", C, "vpd_topk_ci", PURPLE, "s"),
        ("VPD's active set", C, "vpd_active", AMBER, "D"),
        ("neuron circuit", N, "circuit", GREEN, "^"),
    ]
    yy = np.arange(len(fams))[::-1]
    for j, (name, R, m, col, mk) in enumerate(series):
        med = [max(np.median([r["eval"][m][f] for r in R]), 1e-4) for f, _ in fams]
        ax.scatter(med, yy + (j - 1.5) * 0.14, color=col, marker=mk, s=150, zorder=3)
        ax.text(med[0] * 1.25, yy[0] + (j - 1.5) * 0.14, name, color=col, fontsize=17, va="center")
    ax.set_xscale("log")
    ax.axvline(tau, color=INK, lw=1, ls=(0, (4, 4)))
    ax.set_yticks(yy)
    ax.set_yticklabels([f[1] for f in fams])
    ax.set_xlabel("KL from the real network's prediction, bits (median over 50 test images)")
    ax.set_title("Which intervention breaks each set")
    clean(ax)

    # D: interchange of named variables
    ax = fig.add_subplot(gs[1, 1])
    good = [g for g in K if g["move_differ"] is not None and g["bases"] >= 20]
    good = sorted(good, key=lambda g: -g["size"])[:8]
    yy = np.arange(len(good))[::-1]
    d = [g["move_differ"] for g in good]
    c = [g["move_control"] for g in good]
    ax.barh(yy + 0.2, d, height=0.38, color=BLUE)
    ax.barh(yy - 0.2, c, height=0.38, color=SOFT)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{g['layer']}: {g['concept']} ({g['size']})" for g in good], fontsize=17)
    ax.set_xscale("log")
    top = max(d + c)
    ax.text(d[0] * 1.15, yy[0] + 0.2, "donor differs in the feature", color=BLUE, fontsize=17, va="center")
    ax.text(c[0] * 1.15, yy[0] - 0.2, "donor differs in other features", color="#4A6FA5", fontsize=17, va="center")
    ax.set_xlim(right=top * 30)
    ax.set_xlabel("prediction moved by swapping in the group's activations, KL bits")
    ax.set_title("Named groups of circuit subcomponents, interchange")
    clean(ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight", pad_inches=0.4)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
