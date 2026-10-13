"""The MNIST figure (~/mpd-data/figures/oracle/mnist_circuits.png): one digit's circuit in VPD subcomponents with the
names of its variables, VPD's active subcomponents against the real network's activations, circuit sizes in VPD
subcomponents and in neurons, the equal-size baselines on fresh draws, and the interchange test of the named groups.

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
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODULES, load_decomp, load_mnist  # noqa: E402

BLUE, RED, GREEN, AMBER, PURPLE = "#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#7B5EA7"
SOFT = "#9DB9DD"
INK = "#222222"
OUT = Path.home() / "mpd-data" / "figures" / "oracle" / "mnist_circuits.png"
SHORT = {
    "loop_top": "loop at top",
    "loop_bottom": "loop at bottom",
    "big_loop": "large loop",
    "vert_left": "left vertical stroke",
    "vert_center": "middle vertical stroke",
    "vert_right": "right vertical stroke",
    "bar_top": "bar at top",
    "bar_middle": "middle bar",
    "bar_bottom": "bottom bar",
    "ink_center": "ink at center",
    "wide": "wide digit",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 20,
            "axes.titlesize": 23,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 19,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 16,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(name: str) -> list[dict] | None:
    p = DATA / name
    return json.loads(p.read_text())["results"] if p.exists() else None


def passrate(R: list[dict], m: str, tau: float) -> float:
    return 100 * float(np.mean([r["eval"][m]["worst"] <= tau for r in R]))


def units(r: dict) -> int:
    """Units the search chose in its decomposed layers (whole-layer block nodes not counted)."""
    return sum(1 for m in r["members"] if "[whole]" not in m)


def concept_name(info: dict | None) -> str | None:
    if info is None or abs(info["auc_within"] - 0.5) < 0.1:
        return None
    return ("" if info["sign"] == "+" else "no ") + SHORT[info["feature"]]


@torch.no_grad()
def active_only(dec, x: torch.Tensor) -> list[float]:
    """VPD's active subcomponents alone (mask 1[CI > 0], remainder off, each layer on the masked model's own input):
    the median ratio of the first and second hidden layers' activation norms to the real network's, and of the logit
    margin of the real prediction (its logit minus the next largest)."""
    M = dec.split((dec.ci(x) > 0).float())
    _, hs, _ = dec.acts(x)
    h, out = x, []
    for i, m in enumerate(MODULES):
        zc = ((h @ dec.V[m]) * M[m]) @ dec.U[m] + dec.b[m]
        zr = hs[m] @ dec.W[m].T + dec.b[m]
        if i < 2:
            out.append(float((F.relu(zc).norm(dim=1) / F.relu(zr).norm(dim=1)).median()))
            h = F.relu(zc)
        else:
            top = zr.argmax(1, keepdim=True)
            mr = zr.gather(1, top)[:, 0] - zr.scatter(1, top, -1e9).max(1).values
            mc = zc.gather(1, top)[:, 0] - zc.scatter(1, top, -1e9).max(1).values
            out.append(float((mc / mr).median()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="i5e-4")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--example", type=int, default=None, help="test image index for panel A")
    args = ap.parse_args()
    setup()
    t, tau = args.tag, args.tau
    s = f"tau{tau}_s{tau / 2}"
    dec = load_decomp(DATA / f"vpd_{t}.pt")
    _, _, xte, _ = load_mnist()
    full_v = load(f"circuits_{t}_{s}.json")
    full_n = load(f"circuits_{t}_neurons_{s}.json")
    in_v = load(f"circuits_{t}_blocks-fc2-fc3_{s}.json")
    in_n = load(f"circuits_{t}_neurons_blocks-h2_{s}.json")
    out_v = load(f"circuits_{t}_blocks-fc1-fc2_{s}.json")
    out_n = load(f"circuits_{t}_neurons_blocks-h1_{s}.json")
    K = json.loads((DATA / f"concepts_{t}_blocks-fc2-fc3_{s}.json").read_text())
    vpd_eval = json.loads((DATA / f"vpd_{t}.json").read_text())["eval"]
    assert full_v and full_n and in_v and in_n

    fig = plt.figure(figsize=(34, 22))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05], height_ratios=[0.95, 1], hspace=0.42, wspace=0.62)

    # A: one digit's input-layer circuit: each subcomponent's read on the image (pixel times V_c), with its name
    ex = next((r for r in in_v if r["image"] == args.example), None) if args.example is not None else None
    if ex is None:
        ex = sorted(in_v, key=lambda r: units(r))[len(in_v) // 2]
    x = xte[ex["image"]]
    fc1 = [m for m in ex["members"] if m.startswith("fc1#")]
    reads = [(m, (x * dec.V["fc1"][:, int(m.split("#")[1])]).numpy()) for m in fc1]
    named = sorted([(m, rd) for m, rd in reads if concept_name(K["named"].get(m))], key=lambda t2: -np.abs(t2[1]).sum())
    pick, seen = [], set()
    for m, rd in named:  # the largest read of each name first, then the rest by size
        if concept_name(K["named"][m]) not in seen:
            seen.add(concept_name(K["named"][m]))
            pick.append((m, rd))
    pick = (pick + [t2 for t2 in named if t2 not in pick])[:8]
    sub = gs[0, 0:2].subgridspec(2, 6, wspace=0.12, hspace=0.38, width_ratios=[1, 1, 1, 1, 1, 1])
    ax = fig.add_subplot(sub[:, 0:2])
    ax.imshow(x.reshape(28, 28).numpy(), cmap="gray_r")
    ax.set_title(f"test digit, label {ex['label']}", fontsize=21, loc="center", fontweight="normal", pad=10)
    ax.axis("off")
    ax.text(0.0, 1.17, f"One digit's circuit: {len(fc1)} input-layer subcomponents, later layers whole.\n"
            "Eight of them, each named by the stroke feature that best predicts it", transform=ax.transAxes,
            fontsize=24, fontweight="bold", va="bottom", linespacing=1.3)
    lim = max(np.abs(rd).max() for _, rd in pick)
    cmap = LinearSegmentedColormap.from_list("rb", [BLUE, "#ffffff", RED])
    im = None
    for i, (m, rd) in enumerate(pick):
        a = fig.add_subplot(sub[i // 4, 2 + i % 4])
        im = a.imshow(rd.reshape(28, 28), cmap=cmap, vmin=-lim, vmax=lim)
        a.set_title(concept_name(K["named"][m]), fontsize=18, loc="center", fontweight="normal", pad=6)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#cccccc")
    cax = fig.add_axes((0.415, 0.515, 0.16, 0.009))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-lim, lim])
    cb.ax.set_xticklabels(["ink lowers it", "ink raises it"], fontsize=17)
    cb.outline.set_visible(False)

    # B: VPD's active subcomponents alone keep the answer but shrink the activations
    ax = fig.add_subplot(gs[0, 2])
    ratios = active_only(dec, xte[:2000])
    labels = ["first hidden layer\n(activation size)", "second hidden layer\n(activation size)",
              "winning digit's\nlogit margin"]
    yy = np.arange(3)[::-1]
    ax.barh(yy, [100 * r for r in ratios], color=AMBER, height=0.55)
    for yi, r in zip(yy, ratios, strict=True):
        ax.text(100 * r + 2, yi, f"{100 * r:.0f}%", va="center", fontsize=20)
    ax.axvline(100, color=INK, lw=1.5, ls=(0, (4, 3)))
    ax.text(101, 2.55, "real network", fontsize=17, va="center")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("VPD's active subcomponents alone,\n% of the real network (median, 2,000 test digits)")
    ax.set_title(
        f"VPD's {vpd_eval['l0_total']:.0f} active subcomponents per digit give\nthe same answer on "
        f"{100 * vpd_eval['agree_rounded']:.1f}% of digits but shrink\nthe activations"
    )
    clean(ax)

    # C: circuit size by unit, whole network and one layer at a time
    ax = fig.add_subplot(gs[1, 0])
    tot_v, tot_n = dec.total, 512
    rows = [
        ("whole network", full_v, "circuit", BLUE, f"VPD subcomponents (of {tot_v:,})"),
        ("whole network", full_n, "circuit", GREEN, f"hidden neurons (of {tot_n})"),
        ("whole network", full_v, "vpd_active", AMBER, "VPD's own active set"),
        ("input layer only", in_v, "circuit", BLUE, f"VPD subcomponents (of {dec.sizes[0]})"),
        ("input layer only", in_n, "circuit", GREEN, "first-layer neurons (of 256)"),
    ]
    if out_v and out_n:
        rows += [
            ("last layer only", out_v, "circuit", BLUE, f"VPD subcomponents (of {dec.sizes[2]})"),
            ("last layer only", out_n, "circuit", GREEN, "second-layer neurons (of 256)"),
        ]
    y, last, ypos = 0.0, None, []
    for setting, R, m, col, unit in rows:
        if setting != last and last is not None:
            y -= 0.7
        last = setting
        sz = float(np.median([units(r) if m == "circuit" else r["eval"][m]["size"] for r in R]))
        ax.barh(y, sz, color=col, height=0.8)
        ax.text(sz + 5, y, f"{sz:.0f} {unit}, {passrate(R, m, tau):.0f}% pass", va="center", fontsize=17)
        ypos.append((setting, y))
        y -= 1
    groups: dict[str, list[float]] = {}
    for st, yv in ypos:
        groups.setdefault(st, []).append(yv)
    ax.set_yticks([np.mean(v) for v in groups.values()])
    ax.set_yticklabels(list(groups))
    ax.set_xlim(0, 700)
    ax.set_xlabel("units in the circuit, median over 50 test digits")
    ax.set_title("Neurons give smaller circuits in the whole network\nand input layer, VPD subcomponents in the last layer")
    clean(ax)

    # D: equal size, whole-network circuits, share passing all four tests on fresh draws
    ax = fig.add_subplot(gs[1, 1])
    k_med = float(np.median([r["k"] for r in full_v]))
    rows = [
        ("our search", "circuit", BLUE),
        ("VPD's top k by\ncausal importance", "vpd_topk_ci", PURPLE),
        ("top k by size of\neach one's output", "topk_write", SOFT),
        ("random k, VPD's\nactive set first", "random_active", SOFT),
    ]
    vals = [passrate(full_v, m, tau) for _, m, _ in rows]
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, vals, color=[r[2] for r in rows], height=0.6)
    for yi, v in zip(yy, vals, strict=True):
        ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=20)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"digits whose circuit passes all four tests\non fresh draws (mean KL ≤ {tau} bits each), %")
    ax.set_title(f"At equal size ({k_med:.0f} subcomponents, median),\nonly searched circuits pass")
    clean(ax)

    # E: interchange of the named groups against random groups of the same size
    ax = fig.add_subplot(gs[1, 2])
    good = [g for g in K["groups"] if g["circuits"] >= 30]
    good = sorted(good, key=lambda g: g["ratio"])
    yy = np.arange(len(good))
    for yi, g in zip(yy, good, strict=True):
        ax.plot([g["random_ratio"], g["ratio"]], [yi, yi], color="#c9d6ea", lw=3, zorder=1)
    ax.scatter([g["random_ratio"] for g in good], yy, s=140, color=SOFT, zorder=2)
    ax.scatter([g["ratio"] for g in good], yy, s=140, color=BLUE, zorder=3)
    ax.axvline(1.0, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_xlim(0.3, 7)
    ax.set_yticks(yy)
    ax.set_yticklabels([("" if g["sign"] == "+" else "no ") + SHORT[g["feature"]] for g in good], fontsize=18)
    top = good[-1]
    ax.text(top["ratio"] * 1.12, yy[-1], "named group", color=BLUE, fontsize=17, va="center")
    ax.text(top["random_ratio"] / 1.12, yy[-1], "random group", color="#4A6FA5", fontsize=17, va="center",
            ha="right")
    ax.set_xlabel("prediction change when the donor digit differs in\nthe feature ÷ when it differs in as many others")
    wins = sum(g["ratio"] > g["random_ratio"] for g in good)
    ax.set_title(f"Swapping a named group's activations is more\nspecific to its feature than a random group's\n"
                 f"for {wins} of {len(good)} names")
    clean(ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight", pad_inches=0.4)
    print(f"wrote {OUT} (example image {ex['image']}, panel B ratios {ratios})")


if __name__ == "__main__":
    main()
