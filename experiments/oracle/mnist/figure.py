"""The MNIST figure (~/mpd-data/figures/oracle/mnist_circuits.png): one digit's circuit in VPD subcomponents, VPD's
masked model against the real hidden activations, circuit sizes in subcomponents and in neurons, faithfulness at equal
size on fresh draws, and the interchange test of the named variables.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/figure.py --tag i5e-4
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODULES, load_decomp, load_mnist  # noqa: E402

BLUE, RED, GREEN, AMBER, PURPLE = "#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#7B5EA7"
SOFT = "#9DB9DD"
INK = "#222222"
OUT = Path.home() / "mpd-data" / "figures" / "oracle" / "mnist_circuits.png"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 20,
            "axes.titlesize": 22,
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


def load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text())["results"]


def passrate(R: list[dict], m: str, tau: float) -> float:
    return 100 * float(np.mean([r["eval"][m]["worst"] <= tau for r in R]))


def med(R: list[dict], m: str, key: str) -> float:
    return float(np.median([r["eval"][m][key] for r in R]))


@torch.no_grad()
def masked_vs_real(dec, x: torch.Tensor) -> list[float]:
    """Median relative error of VPD's rounded masked model (mask 1[CI > 0], each layer on the masked model's own
    input) from the real model: fc1's and fc2's activations after ReLU, the centered logits."""
    ci = dec.ci(x)
    M = dec.split((ci > 0).float())
    _, hs, _ = dec.acts(x)
    h, out = x, []
    for i, m in enumerate(MODULES):
        zc = ((h @ dec.V[m]) * M[m]) @ dec.U[m] + dec.b[m]
        zr = hs[m] @ dec.W[m].T + dec.b[m]
        if i < 2:
            hc, hr = F.relu(zc), F.relu(zr)
        else:
            hc, hr = zc - zc.mean(1, keepdim=True), zr - zr.mean(1, keepdim=True)
        out.append(float(((hc - hr).norm(dim=1) / hr.norm(dim=1)).median()))
        h = F.relu(zc) if i < 2 else zc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="i5e-4")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--example", type=int, default=None, help="test image index for panel A")
    args = ap.parse_args()
    setup()
    t, tau = args.tag, args.tau
    dec = load_decomp(DATA / f"vpd_{t}.pt")
    _, _, xte, _ = load_mnist()
    full_v = load(f"circuits_{t}_tau{tau}_s{tau / 2}.json")
    full_n = load(f"circuits_{t}_neurons_tau{tau}_s{tau / 2}.json")
    in_v = load(f"circuits_{t}_blocks-fc2-fc3_tau{tau}_s{tau / 2}.json")
    in_n = load(f"circuits_{t}_neurons_blocks-h2_tau{tau}_s{tau / 2}.json")
    in_vl = load(f"circuits_{t}_blocks-fc2-fc3_tau{tau}_s{tau / 2}_logit0.1.json")
    in_nl = load(f"circuits_{t}_neurons_blocks-h2_tau{tau}_s{tau / 2}_logit0.1.json")
    K = json.loads((DATA / f"concepts_{t}_blocks-fc2-fc3_tau{tau}_s{tau / 2}.json").read_text())["groups"]
    vpd_eval = json.loads((DATA / f"vpd_{t}.json").read_text())["eval"]

    fig = plt.figure(figsize=(32, 21))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[0.9, 1], hspace=0.5, wspace=0.75)

    # A: one digit's input-layer circuit, each subcomponent's read on the image (pixel x V_c)
    ex = next((r for r in in_v if r["image"] == args.example), None) if args.example is not None else None
    if ex is None:
        ex = sorted(in_v, key=lambda r: r["k"])[len(in_v) // 3]
    x = xte[ex["image"]]
    fc1 = [(m, c) for m, c in zip(ex["members"], ex["members_ci"], strict=True) if m.startswith("fc1#")]
    reads = []
    for m, c in fc1:
        j = int(m.split("#")[1])
        reads.append((m, c, (x * dec.V["fc1"][:, j]).numpy()))
    reads = sorted(reads, key=lambda t3: -abs(t3[2].sum()))[:9]
    sub = gs[0, 0:2].subgridspec(2, 6, wspace=0.08, hspace=0.32)
    ax = fig.add_subplot(sub[:, 0:2])
    ax.imshow(x.reshape(28, 28).numpy(), cmap="gray_r")
    ax.set_title(f"test digit (label {ex['label']})", fontsize=20, loc="center", fontweight="normal")
    ax.axis("off")
    lim = max(np.abs(r[2]).max() for r in reads)
    for i, (m, c, rd) in enumerate(reads):
        a = fig.add_subplot(sub[i // 4, 2 + i % 4])
        a.imshow(rd.reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set_title(f"{m.replace('#', ' #')}" + ("" if c > 0 else " (VPD: off)"), fontsize=17, loc="center",
                    fontweight="normal")
        a.axis("off")
    n_fc1 = sum(1 for m in ex["members"] if m.startswith("fc1#"))
    ax.text(0.0, 1.13, f"One digit's circuit: {n_fc1} input-layer subcomponents, the two later layers whole "
            "(red: ink that raises the subcomponent, blue: lowers)", transform=ax.transAxes, fontsize=22,
            fontweight="bold")

    # B: VPD's masked model keeps the answer, not the activations
    ax = fig.add_subplot(gs[0, 2])
    errs = masked_vs_real(dec, xte[:2000])
    labels = ["first hidden layer", "second hidden layer", "logits (centered)"]
    yy = np.arange(3)[::-1]
    ax.barh(yy, [100 * e for e in errs], color=AMBER, height=0.6)
    for yi, e in zip(yy, errs, strict=True):
        ax.text(100 * e + 2, yi, f"{100 * e:.0f}%", va="center", fontsize=19)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 115)
    ax.set_xlabel("relative error from the real network, %")
    ax.set_title(
        f"VPD's active subcomponents keep the prediction\n(KL {vpd_eval['kl_bits_rounded']:.2f} bits, "
        f"{100 * vpd_eval['agree_rounded']:.1f}% same answer) but not the activations"
    )
    clean(ax)

    # C: circuit size, median over 50 digits, with the share faithful on fresh draws
    ax = fig.add_subplot(gs[1, 0])
    rows = [
        ("input layer, KL", in_v, "circuit", BLUE, "VPD subcomponents"),
        ("input layer, KL", in_n, "circuit", GREEN, "hidden neurons"),
        ("input layer, KL + logits", in_vl, "circuit", BLUE, "VPD subcomponents"),
        ("input layer, KL + logits", in_nl, "circuit", GREEN, "hidden neurons"),
        ("whole network, KL", full_v, "circuit", BLUE, "VPD subcomponents"),
        ("whole network, KL", full_v, "vpd_active", AMBER, "VPD's active set"),
        ("whole network, KL", full_n, "circuit", GREEN, "hidden neurons"),
    ]
    yy, ylab, ypos, y = [], [], [], 0.0
    last = None
    for setting, R, m, col, unit in rows:
        if setting != last and last is not None:
            y -= 0.6
        last = setting
        sz = med(R, m, "size")
        ax.barh(y, sz, color=col, height=0.8)
        ax.text(sz + 4, y, f"{unit}: {sz:.0f}  ({passrate(R, m, tau):.0f}% faithful)", va="center", fontsize=17)
        yy.append(y)
        ypos.append((setting, y))
        y -= 1
    groups: dict[str, list[float]] = {}
    for s, yv in ypos:
        groups.setdefault(s, []).append(yv)
    ax.set_yticks([np.mean(v) for v in groups.values()])
    ax.set_yticklabels(list(groups), fontsize=18)
    ax.set_xlim(0, 620)
    ax.set_xlabel("circuit size, median over 50 test digits (nodes)")
    ax.set_title("Circuit size by unit and test")
    clean(ax)

    # D: equal size, input-layer circuits, share faithful on fresh draws
    ax = fig.add_subplot(gs[1, 1])
    rows = [
        ("our search", "circuit", BLUE),
        ("VPD's top k by importance", "vpd_topk_ci", PURPLE),
        ("top k by write size", "topk_write", SOFT),
        ("random k of VPD's active", "random_active", SOFT),
    ]
    vals = [passrate(in_v, m, tau) for _, m, _ in rows]
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, vals, color=[r[2] for r in rows], height=0.6)
    for yi, v in zip(yy, vals, strict=True):
        ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=19)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 110)
    ax.set_xlabel(f"input-layer circuits faithful on fresh draws\n(all four tests, mean KL ≤ {tau} bits), %")
    ax.set_title(f"Equal size (median {med(in_v, 'circuit', 'size'):.0f} nodes)")
    clean(ax)

    # E: interchange of named variables
    ax = fig.add_subplot(gs[1, 2])
    good = [g for g in K if g["move_differ"] is not None and g["bases"] >= 20 and g["size"] >= 2]
    good = sorted(good, key=lambda g: -(g["ratio"] / max(g["random_ratio"] or 1, 1e-9)))[:7]
    yy = np.arange(len(good))[::-1]
    d = [g["move_differ"] for g in good]
    c = [g["move_control"] for g in good]
    ax.barh(yy + 0.2, d, height=0.38, color=BLUE)
    ax.barh(yy - 0.2, c, height=0.38, color=SOFT)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{g['concept']} ({g['size']})" for g in good], fontsize=17)
    ax.text(d[0] * 1.08, yy[0] + 0.2, "donor differs in the feature", color=BLUE, fontsize=16, va="center")
    ax.text(c[0] * 1.08, yy[0] - 0.2, "donor differs in other features", color="#4A6FA5", fontsize=16, va="center")
    ax.set_xlim(0, max(d + c) * 2.1)
    ax.set_xlabel("prediction moved by swapping in the group's\nactivations from another digit, KL bits")
    ax.set_title("Named groups of input-layer subcomponents")
    clean(ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, bbox_inches="tight", pad_inches=0.4)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
