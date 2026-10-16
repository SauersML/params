"""Figures of the neuron PheWAS (#2951): stage 1's Manhattan and QQ plots per trait, and the overview of every trait's
associated neurons. White, no grid, large type, direct labels (the team's figure rules).

  plot.py manhattan ASSOC_DIR TRAIT OUT.png      every neuron's strongest association (-log10 p over its 4 measures)
                                                 by layer, the Bonferroni line
  plot.py qq ASSOC_DIR TRAIT OUT.png             observed against expected -log10 p over every test of the trait
  plot.py overview ASSOC_DIR PHENOME_DIR OUT.png neurons past Bonferroni per trait, by group
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GROUP_COLORS = {"safety": "#D1495B", "persona": "#7B5EA7", "epistemics": "#2F6DB5", "style": "#E09F3E", "skills": "#2E8B57"}


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 18, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white", "axes.labelsize": 18})
    return plt


def log10_p(t: np.ndarray, df: int) -> np.ndarray:
    from scipy import stats

    return -(stats.t.logsf(np.abs(t.astype(np.float64)), df) / np.log(10) + np.log10(2))


def load(assoc: Path, trait: str):
    z = np.load(assoc / f"{trait.replace('/', '__')}.npz")
    return z["t"], int(z["df"])


def manhattan(a) -> None:
    plt = style()
    summ = json.loads((a.assoc / "summary.json").read_text())
    t, df = load(a.assoc, a.trait)
    lp = log10_p(t, df).max(-1)  # [L, D]: each neuron's strongest of its 4 measures
    L, D = lp.shape
    x = np.arange(L * D)
    y = lp.reshape(-1)
    keep = (y > 1) | (np.random.default_rng(0).random(y.shape) < 0.02)  # (every point past p 0.1 and a sample of the rest)
    fig, ax = plt.subplots(figsize=(16, 6))
    for l_ in range(L):
        sel = keep & (x // D == l_)
        ax.scatter(x[sel], y[sel], s=5, color="#2F6DB5" if l_ % 2 == 0 else "#9DB9DD", rasterized=True)
    thr = -summ[a.trait]["bonferroni_log10_alpha"]
    ax.axhline(thr, color="#D1495B", lw=1.5, ls="--")
    ax.annotate("Bonferroni", (L * D, thr), textcoords="offset points", xytext=(-4, 6), ha="right", color="#D1495B", fontsize=15)
    ax.set_xticks([l_ * D + D / 2 for l_ in range(0, L, 4)])
    ax.set_xticklabels([str(l_) for l_ in range(0, L, 4)])
    ax.set_xlim(0, L * D)
    ax.set_xlabel("layer (neurons in order within each layer)")
    ax.set_ylabel("-log10 p")
    ax.set_title(a.title or f"Neurons associated with {a.trait}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def qq(a) -> None:
    plt = style()
    t, df = load(a.assoc, a.trait)
    y = np.sort(log10_p(t, df).reshape(-1))[::-1]
    n = len(y)
    e = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    idx = np.unique(np.concatenate([np.arange(min(n, 20000)), np.linspace(0, n - 1, 20000).astype(int)]))
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, e[0]], [0, e[0]], color="#888888", lw=1.5)
    ax.scatter(e[idx], y[idx], s=6, color="#2F6DB5", rasterized=True)
    ax.set_xlabel("expected -log10 p")
    ax.set_ylabel("observed -log10 p")
    ax.set_title(a.title or f"{a.trait}: observed against chance")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def overview(a) -> None:
    plt = style()
    summ = json.loads((a.assoc / "summary.json").read_text())
    phen = {t["name"]: t for t in json.loads((a.phenome / "phenome.json").read_text())["traits"]}
    rows = sorted(((n, s["hit_neurons"], phen[n]["group"]) for n, s in summ.items()), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, max(8, 0.24 * len(rows))))
    ax.barh(range(len(rows)), [max(r[1], 0.8) for r in rows], color=[GROUP_COLORS.get(r[2], "#888888") for r in rows])
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("neurons associated past Bonferroni")
    for g, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=g)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(a.title or "Neurons associated with each behaviour")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("manhattan", "qq"):
        p = sub.add_parser(name)
        p.add_argument("assoc", type=Path)
        p.add_argument("trait")
        p.add_argument("out", type=Path)
        p.add_argument("--title")
    o = sub.add_parser("overview")
    o.add_argument("assoc", type=Path)
    o.add_argument("phenome", type=Path)
    o.add_argument("out", type=Path)
    o.add_argument("--title")
    a = ap.parse_args()
    {"manhattan": manhattan, "qq": qq, "overview": overview}[a.cmd](a)


if __name__ == "__main__":
    main()
