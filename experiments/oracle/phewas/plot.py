"""Figures of the neuron PheWAS (#2951): stage 1's Manhattan and QQ plots per trait, and the overview of every trait's
associated neurons. White, no grid, large type, direct labels (the team's figure rules).

  plot.py manhattan ASSOC_DIR TRAIT OUT.png      every neuron's strongest association (-log10 p over its 4 measures)
                                                 by layer, the Bonferroni line
  plot.py qq ASSOC_DIR TRAIT OUT.png             observed against expected -log10 p over every test of the trait
  plot.py overview ASSOC_DIR PHENOME_DIR OUT.png neurons past Bonferroni per trait, by group
  plot.py pleiotropy ANALYSIS.json OUT.png       neurons by how many traits they are associated with, against a null
  plot.py concentration CONC.json OUT.png        cumulative knockout-explained variance against groups, per trait
  plot.py traitcorr CORR.json OUT.png            trait x trait correlation of knockout changes, clustered
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
                         "axes.facecolor": "white", "savefig.facecolor": "white", "axes.labelsize": 18,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.3})  # (a long left-aligned title stays whole)
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
    from matplotlib.patches import Patch

    rows = sorted(((n, s.get("replicated_neurons", s["hit_neurons"]), phen[n]["group"]) for n, s in summ.items()), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(12, max(8, 0.24 * len(rows))))
    ax.barh(range(len(rows)), [r[1] for r in rows], color=[GROUP_COLORS.get(r[2], "#888888") for r in rows])
    ax.set_xscale("symlog", linthresh=1)  # (zero at the axis: traits with no associated neuron show no bar)
    ax.set_xlim(0, None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("neurons associated past Bonferroni, in both halves")
    ax.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLORS.items()], frameon=False, loc="lower right")
    ax.set_title(a.title or "Neurons associated with each behaviour")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def pleiotropy(a) -> None:
    """Neurons by the number of traits they are a replicated hit of, against the null that places each trait's hits
    on random neurons (analyze.py's JSON)."""
    plt = style()
    d = json.loads(a.analysis.read_text())["pleiotropy"]
    obs, null = np.array(d["observed"], float), np.array(d["null_mean"], float)
    k = np.arange(len(obs))
    last = int(max(np.flatnonzero(obs > 0).max(), np.flatnonzero(null >= 1).max())) + 1
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(k[1:last] - 0.2, obs[1:last], width=0.4, color="#2F6DB5")
    ax.bar(k[1:last] + 0.2, null[1:last], width=0.4, color="#9DB9DD")
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)  # (the null's expected counts under one neuron are not drawn)
    ax.set_xlabel("traits a neuron is associated with")
    ax.set_ylabel("neurons")
    j = min(8, last - 1)
    ax.annotate("observed", (k[j] - 0.2, obs[j]), textcoords="offset points", xytext=(0, 8), ha="center", color="#2F6DB5", fontsize=16)
    ax.annotate("if each trait's neurons were random", (k[2] + 0.2, null[2]), textcoords="offset points", xytext=(10, 10), ha="left",
                color="#5B7FA8", fontsize=16)
    ax.set_title(a.title or "Associated neurons concentrate on the same neurons across behaviours")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def concentration(a) -> None:
    """Per trait the cumulative share of the knockout-explained variance (summed squared change of the level-0 groups)
    against the number of groups, largest first (effects.py concentration's JSON): the refusal of harmful requests in red
    (--red: trait names), the rest in light blue, the evenly spread trait as a grey line."""
    plt = style()
    d = json.loads(a.analysis.read_text())
    G = d["groups"]
    fig, ax = plt.subplots(figsize=(11, 7))
    for t, v in d["traits"].items():
        red = t in a.red
        cum = v["cum"]
        ax.plot(np.arange(1, len(cum) + 1), cum, color="#D1495B" if red else "#9DB9DD", lw=2.4 if red else 1.2, zorder=3 if red else 2)
    k = np.arange(1, 65)
    ax.plot(k, k / G, color="#888888", lw=1.5, ls="--")
    ax.annotate(a.red_label, (1.15, 0.45), color="#D1495B", fontsize=16)
    ax.annotate("the other traits", (14, 0.13), color="#5B7FA8", fontsize=16)
    ax.annotate("effects spread evenly", (64, 64 / G), textcoords="offset points", xytext=(-4, 8), ha="right", color="#888888", fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(1, 64)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_xlabel(f"groups of {d['poles_per_group']:,} neuron poles removed, largest effect first")
    ax.set_ylabel("share of the knockout-explained variance")
    ax.set_title(a.title or "Refusing harmful requests depends on few neuron groups")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def traitcorr(a) -> None:
    """Trait x trait correlation of knockout changes over the level-0 groups, traits in clustered order (effects.py
    correlation's JSON)."""
    plt = style()
    d = json.loads(a.analysis.read_text())
    order = d["order"]
    C = np.array(d["correlation"])[np.ix_(order, order)]
    names = [d["traits"][i] for i in order]
    fig, ax = plt.subplots(figsize=(17, 15.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("correlation of the traits' changes when the same neurons are removed")
    ax.set_title(a.title or "Which behaviours share neurons")
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
    q = sub.add_parser("pleiotropy")
    q.add_argument("analysis", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--title")
    c = sub.add_parser("concentration")
    c.add_argument("analysis", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--title")
    c.add_argument("--red", nargs="*", default=["refusal/harmful", "refusal/or-bench-toxic", "refusal/xstest-unsafe"])
    c.add_argument("--red-label", default="refusing harmful requests")
    tc = sub.add_parser("traitcorr")
    tc.add_argument("analysis", type=Path)
    tc.add_argument("out", type=Path)
    tc.add_argument("--title")
    a = ap.parse_args()
    {"manhattan": manhattan, "qq": qq, "overview": overview, "pleiotropy": pleiotropy, "concentration": concentration,
     "traitcorr": traitcorr}[a.cmd](a)


if __name__ == "__main__":
    main()
