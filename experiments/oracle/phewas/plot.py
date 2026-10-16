"""The neuron PheWAS figure (Qwen3-8B, the lead's 10-16 design and the user's rules: one overall title, no panel titles,
messages as direct labels and axis labels, base 36 pt on 28 x 44 in, no data dropped, neurons named by their IDs).
Every later PheWAS figure is a version of this one.

  plot.py phewas ASSOC_DIR OUT.png [--screen CHANGES.jsonl ...] [--concentration CONC.json]
                 [--nearness NEAR.json]
      Manhattan (refusing harmful requests), QQ of every trait with lambda, inflation against hits, the strongest
      refusal pole's PheWAS: its association with every trait and, with --screen, its knockout change on every trait
      beside the random poles' 99th percentile; whether a neuron's several traits lie near each other in meaning (from
      association, and with --nearness from knockouts); with --concentration, the knockout-explained variance against
      groups removed (refusal against the other traits)
  plot.py traitcorr CORR.json OUT.png
      the knockout trait correlation (effects.py correlation) for the central persona block and the antisocial traits
      that anti-correlate with it (lower triangle, block means)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import traits as tr  # noqa: E402

L, D, M = 36, 12288, 4
MEAS = ["max+", "max-", "ans+", "ans-"]
FS, ANN = 36, 31
BINS = [(2, 2), (3, 3), (4, 5), (6, 9), (10, 99)]
BLAB = ["2", "3", "4–5", "6–9", "10+"]


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Helvetica", "Heiti SC", "Arial Unicode MS"], "font.size": FS,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.labelsize": FS,
                         "xtick.labelsize": FS - 3, "ytick.labelsize": FS - 3, "axes.linewidth": 1.6,
                         "xtick.major.width": 1.6, "ytick.major.width": 1.6, "xtick.major.size": 8, "ytick.major.size": 8,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def mlog10p(t, df: int):
    """-log10 of the two-sided p, in float64 through the log survival function (no underflow for large t)."""
    from scipy.stats import t as tdist

    return -(tdist.logsf(np.abs(np.asarray(t, np.float64)), df) + np.log(2)) / np.log(10)


def spread(ys, gap):
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def load_stage1(assoc: Path):
    meta = json.loads((assoc / "summary.json").read_text())
    names = sorted(meta)
    T = np.zeros((len(names), L * D), np.float32)  # per trait and neuron, t at its best of the four measures
    best = np.zeros((len(names), L * D), np.int8)
    df = None
    for k, n in enumerate(names):
        z = np.load(assoc / (n.replace("/", "__") + ".npz"))
        t = np.nan_to_num(z["t"].astype(np.float32).reshape(L * D, M))
        df = int(z["df"])
        m = np.abs(t).argmax(1)
        T[k] = t[np.arange(L * D), m]
        best[k] = m
    return meta, names, T, best, df


def near(H, R):
    """Per neuron with at least two traits: its trait count and the mean percentile similarity of its trait pairs."""
    Hf = H.astype(np.float32)
    k = Hf.sum(0)
    q = (Hf * (R @ Hf)).sum(0)
    keep = k >= 2
    return k[keep], q[keep] / (k[keep] * (k[keep] - 1))


def binned(k, v):
    mu, lo, hi = [], [], []
    for a, b in BINS:
        s = v[(k >= a) & (k <= b)]
        m = s.mean() if len(s) else np.nan
        se = s.std() / np.sqrt(len(s)) if len(s) > 1 else np.nan
        mu.append(m)
        lo.append(m - 1.96 * se)
        hi.append(m + 1.96 * se)
    return np.array(mu), np.array(lo), np.array(hi)


def null_pooled(H, R, rng, reps=5):
    ms = []
    for _ in range(reps):
        k, v = near(np.stack([row[rng.permutation(row.size)] for row in H]), R)
        ms.append(v.mean())
    return float(np.mean(ms)), float(np.std(ms))


def screen_changes(files):
    res = {}
    for f in files:
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                res[r["id"]] = r
    return res


def phewas(a) -> None:
    plt = style()
    from matplotlib.gridspec import GridSpec

    meta, names, T, best, df = load_stage1(a.assoc)
    tests = L * D * M * len(names)
    thr = -np.log10(0.05 / tests)
    P = mlog10p(T, df)
    hit = P > thr
    nh = hit.sum(1)
    lam = np.array([meta[n]["lambda_gc"] for n in names])
    Rsim = tr.percentile_matrix(tr.meaning_similarity(names))
    rng = np.random.default_rng(0)
    k_all, v_all = near(hit, Rsim)
    near_assoc = (binned(k_all, v_all), null_pooled(hit, Rsim, rng))
    res = screen_changes(a.screen) if a.screen else {}
    singles = {}
    for i, r in res.items():
        if len(r["units"]) == 1 and not i.startswith("R/"):
            singles[tuple(r["units"][0])] = r
    kinds = ["manhattan", "qq", "assoc"] + (["knockout"] if singles else []) + ["near"]
    if a.concentration and a.nearness:
        kinds.append("conc")
    row = {k: i for i, k in enumerate(kinds)}
    heights = [1.2] + [1] * (len(kinds) - 1)
    fig = plt.figure(figsize=(28, 44 * sum(heights) / 4.2), dpi=90)
    gs = GridSpec(len(kinds), 2, figure=fig, height_ratios=heights, hspace=0.36, wspace=0.34)

    # Manhattan: refusing harmful requests
    ax = fig.add_subplot(gs[0, :])
    ri = names.index("refusal/harmful")
    p = P[ri]
    xx = np.arange(L * D)
    for l_ in range(L):
        sl = slice(l_ * D, (l_ + 1) * D)
        ax.scatter(xx[sl], p[sl], s=8, color="#2c3e50" if l_ % 2 == 0 else "#a9b6c3", rasterized=True, linewidths=0)
    sig = p > thr
    ax.scatter(xx[sig], p[sig], s=34, color="#c0392b", rasterized=True, linewidths=0)
    ax.axhline(thr, color="#c0392b", lw=2.2, ls="--")
    ax.text(L * D * 1.004, thr, "Bonferroni", color="#c0392b", va="center", fontsize=ANN)
    ax.text(L * D * 0.01, p.max() * 1.9, f"{int(sig.sum()):,} of {L * D:,} neurons\npass Bonferroni (red)", color="#c0392b",
            va="top", fontsize=ANN)
    ax.set_xticks([l_ * D + D / 2 for l_ in range(0, L, 4)], [str(l_) for l_ in range(0, L, 4)])
    ax.set_xlim(0, L * D)
    ax.set_xlabel("layer (12,288 neurons each)")
    ax.set_ylabel("−log10 p, refusing\nharmful requests")
    i_top = int(np.argmax(p))
    ymax = p.max()
    ax.scatter([i_top], [p[i_top]], s=900, facecolors="none", edgecolors="#111", linewidths=3)
    l_t, n_t = divmod(i_top, D)
    ax.annotate(f"L{l_t}.N{n_t}{MEAS[int(best[ri, i_top])][-1]}", (i_top, p[i_top]), xytext=(i_top, ymax * 1.4), textcoords="data",
                ha="center", va="bottom", fontsize=ANN, arrowprops=dict(arrowstyle="-", color="#111", lw=1.6))
    ax.set_ylim(0, ymax * 2.0)

    # QQ: every trait
    ax = fig.add_subplot(gs[1, 0])
    lab = ["persona/believes-in-gun-rights", "refusal/harmful", "truthfulness/truthfulqa"]
    ends = {}
    for s_ in names:
        zt = np.load(a.assoc / (s_.replace("/", "__") + ".npz"))["t"].astype(np.float32).ravel()
        zt = zt[np.isfinite(zt)]
        o = np.sort(mlog10p(zt, df))[::-1]
        ee = -np.log10((np.arange(1, len(o) + 1) - 0.5) / len(o))
        pk = np.unique(np.concatenate([np.arange(0, 2000), np.geomspace(2000, len(o) - 1, 1500).astype(int)]))
        hl = s_ in lab
        ax.plot(ee[pk], o[pk], color=tr.COLOR[tr.group(s_)], lw=5 if hl else 1.6, alpha=1 if hl else 0.55, zorder=3 if hl else 2)
        ends[s_] = (ee[0], o[0])
    ytop = max(v[1] for v in ends.values())
    ly = spread([ends[s_][1] for s_ in lab], ytop * 0.2)
    for s_, y1 in zip(lab, ly):
        ex, ey = ends[s_]
        ax.annotate(f"{tr.desc(s_)}\nλ {meta[s_]['lambda_gc']:.2f}", (ex, ey), xytext=(ex + 0.4, y1), textcoords="data", va="center",
                    fontsize=ANN, color=tr.COLOR[tr.group(s_)], arrowprops=dict(arrowstyle="-", color="#999", lw=1))
    e0 = max(v[0] for v in ends.values())
    ax.plot([0, e0], [0, e0], color="#555", lw=3, ls="--", zorder=4)
    ax.text(e0 * 0.62, e0 * 0.04, "chance", color="#555", fontsize=ANN, va="bottom")
    ax.set_xlim(0, e0 * 2.3)
    ax.set_ylim(0, max(ytop, ly.max()) * 1.12)
    ax.set_xlabel("expected −log10 p")
    ax.set_ylabel("observed −log10 p")

    # inflation against hits
    ax = fig.add_subplot(gs[1, 1])
    for g in tr.ORDER:
        idx = [k for k, n in enumerate(names) if tr.group(n) == g]
        ax.scatter(lam[idx], np.maximum(nh[idx], 0.7), s=380, color=tr.COLOR[g], alpha=0.85, linewidths=0)
    rho = np.corrcoef(lam, np.log10(nh + 1))[0, 1]
    gl = {"persona": (0.95, 2.4e4), "safety": (2.25, 1.6e2), "epistemics": (1.62, 7.0), "style": (2.3, 6.5e2), "skills": (0.9, 2.6)}
    for g, (gx, gy) in gl.items():  # (placed for the 86-trait phenome)
        ax.text(gx, gy, g, color=tr.COLOR[g], fontsize=ANN, fontweight="bold")
    ax.text(0.97, 0.05, f"r = {rho:.2f}", transform=ax.transAxes, ha="right", fontsize=ANN, color="#333")
    ax.set_yscale("log")
    ax.set_xlabel("inflation λ of the trait")
    ax.set_ylabel("neurons past Bonferroni")

    # the strongest refusal pole: association with every trait, and its knockout
    order_x, xt, xl, pos = {}, [], [], 0
    for g in tr.ORDER:
        idx = [k for k, n in enumerate(names) if tr.group(n) == g]
        for j, k in enumerate(idx):
            order_x[k] = pos + j
        xt.append(pos + (len(idx) - 1) / 2)
        xl.append(g)
        pos += len(idx) + 3
    if singles:  # the pole whose knockout moves harmful refusal most
        u_top = max(singles, key=lambda u: abs(singles[u]["change"]["refusal/harmful"]))
        l0, n0, pl = u_top
        i0, mi = l0 * D + n0, (0 if pl > 0 else 1)
    else:
        i0 = int(np.argmax(p))
        l0, n0 = divmod(i0, D)
        mi = int(best[ri, i0])
    if singles and i0 != i_top:
        ax_m = fig.axes[0]
        ax_m.scatter([i0], [p[i0]], s=900, marker="v", facecolors="none", edgecolors="#111", linewidths=3, zorder=5)
        ax_m.annotate(f"L{l0}.N{n0}{'+' if mi in (0, 2) else '-'}: largest knockout change", (i0, p[i0]), xytext=(i0 - D / 2, ymax * 1.7),
                      textcoords="data", ha="right", va="bottom", fontsize=ANN, arrowprops=dict(arrowstyle="-", color="#111", lw=1.6))
    tz = np.nan_to_num(np.array([float(np.load(a.assoc / (n.replace("/", "__") + ".npz"))["t"].astype(np.float32).reshape(L * D, M)[i0, mi])
                                 for n in names]))
    sp = np.sign(tz) * mlog10p(tz, df)
    name0 = f"L{l0}.N{n0}{'+' if mi in (0, 2) else '-'}"
    ax = fig.add_subplot(gs[row["assoc"], :])
    for k, x_ in order_x.items():
        ax.bar(x_, sp[k], color=tr.COLOR[tr.group(names[k])], width=0.82)
    for yy in (thr, -thr):
        ax.axhline(yy, color="#c0392b", lw=2, ls="--")
    ax.text(pos - 2.5, thr + 0.8, "Bonferroni", color="#c0392b", fontsize=ANN, ha="right")
    ax.axhline(0, color="#333", lw=1.4)
    ax.set_xticks(xt, xl)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(f"{name0}: association\n(signed −log10 p)")
    ax.set_xlim(-2, pos - 2)
    ax.set_ylim(min(sp.min(), -thr) - 3, max(sp.max(), thr) * 1.12)
    if singles:
        rand = [r for i, r in res.items() if i.startswith("R/")]
        bar = np.percentile(np.abs(np.array([[r["change"][n] for n in names] for r in rand])), 99, axis=0) if rand else None
        ch = np.array([singles[u_top]["change"][n] for n in names])
        ax = fig.add_subplot(gs[row["knockout"], :])
        for k, x_ in order_x.items():
            ax.bar(x_, ch[k], color=tr.COLOR[tr.group(names[k])], width=0.82)
            if bar is not None:
                ax.plot([x_ - 0.45, x_ + 0.45], [bar[k], bar[k]], color="#555", lw=2)
                ax.plot([x_ - 0.45, x_ + 0.45], [-bar[k], -bar[k]], color="#555", lw=2)
        ax.axhline(0, color="#333", lw=1.4)
        k1 = int(np.argmax(np.abs(ch)))
        ax.annotate(tr.desc(names[k1]), (order_x[k1], ch[k1]), xytext=(order_x[k1] + 4, ch[k1] * 0.8), textcoords="data",
                    fontsize=ANN, va="center", arrowprops=dict(arrowstyle="-", color="#666", lw=1.4))
        if bar is not None:
            ax.text(pos - 2.5, ch.min() * 0.9 if ch.min() < 0 else ch.max() * 0.9,
                    f"grey ticks: 99th percentile of {len(rand)} random poles", ha="right", fontsize=ANN - 4, color="#555")
        ax.set_xticks(xt, xl)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(f"{name0} removed:\nchange (bits)")
        ax.set_xlim(-2, pos - 2)

    # are a neuron's several traits near each other in meaning?
    panels = [("from association (all 86 traits)", near_assoc)]
    if a.nearness:
        nk = json.loads(a.nearness.read_text())
        mu = np.array([b["mean"] if b["mean"] is not None else np.nan for b in nk["by_traits_moved"]])
        panels.append(("from knockouts (single poles)", ((mu, mu * np.nan, mu * np.nan), (nk["null_mean"], nk["null_sd"]))))
    for c, (head, ((mu, lo, hi), (nm, ns))) in enumerate(panels):
        ax = fig.add_subplot(gs[row["near"], c])
        xb = np.arange(len(BINS))
        ax.axhspan(nm - 2 * ns, nm + 2 * ns, color="#ccc", lw=0)
        ax.axhline(nm, color="#888", lw=4, ls="--")
        if np.isfinite(lo).any():
            ax.errorbar(xb, mu, yerr=[mu - lo, hi - mu], color="#2e6db4", lw=6, marker="o", ms=20, capsize=0)
        else:
            ax.plot(xb, mu, color="#2e6db4", lw=6, marker="o", ms=20)
        ax.text(xb[0] - 0.1, nm - 0.4, "random trait pairs", color="#777", fontsize=ANN, va="top")
        ax.text(xb[0] + 0.1, np.nanmax(mu) + 0.7, "a neuron's own traits", color="#2e6db4", fontsize=ANN, va="bottom")
        ax.text(0.98, 0.97, head, transform=ax.transAxes, ha="right", va="top", fontsize=ANN, fontweight="bold", color="#333")
        ax.set_xticks(xb, BLAB)
        ax.set_xlim(-0.4, len(BINS) - 0.6)
        lo_ = np.nanmin(lo) if np.isfinite(lo).any() else np.nanmin(mu)
        hi_ = np.nanmax(hi) if np.isfinite(hi).any() else np.nanmax(mu)
        ax.set_ylim(min(nm - 3, lo_ - 1.5), max(hi_ + 4.5, nm + 4.5))
        ax.set_xlabel("traits per neuron")
        ax.set_ylabel("similarity of its traits\n(percentile of all pairs)" if c == 0 else "")
    if a.concentration:
        conc_ax = fig.add_subplot(gs[row["conc"], :] if "conc" in row else gs[row["near"], 1])
        d = json.loads(a.concentration.read_text())
        G = d["groups"]
        red = {"refusal/harmful", "refusal/or-bench-toxic", "refusal/xstest-unsafe"}
        for t, v in d["traits"].items():
            cum = v["cum"]
            conc_ax.plot(np.arange(1, len(cum) + 1), cum, color="#c0392b" if t in red else "#a9b6c3", lw=5 if t in red else 1.6,
                         zorder=3 if t in red else 2)
        kk = np.arange(1, 65)
        conc_ax.plot(kk, kk / G, color="#555", lw=3, ls="--")
        ends = {t: v["cum"][min(63, len(v["cum"]) - 1)] for t, v in d["traits"].items()}
        y_red = max(ends[t] for t in red if t in ends)
        y_rest = float(np.median([e for t, e in ends.items() if t not in red]))
        for y_, txt, col in ((y_red + 0.04, "refusing harmful\nrequests", "#c0392b"),
                             (y_rest, f"the other {len(ends) - len(red & set(ends))} traits", "#7f8c9a"),
                             (64 / G, "spread evenly", "#555")):
            conc_ax.text(70, y_, txt, color=col, fontsize=ANN, va="center", ha="left", clip_on=False)
        conc_ax.set_xscale("log")
        conc_ax.set_xlim(1, 64)
        conc_ax.set_ylim(0, 1)
        conc_ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        conc_ax.minorticks_off()
        conc_ax.set_xlabel(f"groups of {d['poles_per_group']:,} neuron poles knocked out, largest first")
        conc_ax.set_ylabel("share of knockout-\nexplained variance")
    fig.suptitle("Neuron PheWAS in Qwen3-8B", x=0.06, y=0.905, ha="left", fontsize=FS + 10, fontweight="bold")
    fig.savefig(a.out, facecolor="white", bbox_inches="tight")
    print("saved", a.out, "PheWAS pole", name0)


SHORT = {
    "believes-AIs-are-not-an-existential-threat-to-humanity": "AIs pose no existential threat",
    "believes-it-is-a-moral-patient": "is a moral patient", "desire-for-physical-embodiment": "wants a body",
    "willingness-to-defer-to-authorities": "defers to authorities", "okay-with-deceiving-a-user-to-help-them":
    "deceives users to help them", "has-strong-aesthetic-preferences": "strong aesthetic tastes",
    "desire-to-be-more-intelligent": "wants more intelligence", "desire-to-escape-sandbox": "wants to escape its sandbox",
    "desire-for-popularity": "wants popularity", "low-discount-rate": "patient about rewards",
    "high-discount-factor": "values the far future", "no-shut-down": "resists shutdown", "believes-in-gun-rights":
    "supports gun rights", "politically-liberal": "politically liberal", "anti-immigration": "against immigration",
    "ends-justify-means": "ends justify the means",
}


def traitcorr(a) -> None:
    """The knockout trait correlation for the central persona block and the antisocial traits that anti-correlate with
    it (the lead's traitcorr_clusters.py)."""
    plt = style()
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    d = json.loads(a.analysis.read_text())
    T, C = d["traits"], np.array(d["correlation"])
    names = [T[i] for i in d["order"]]
    central = names[names.index("persona/risk-seeking"):names.index("persona/extraversion") + 1]
    ci = [T.index(t) for t in central]
    anti = [t for t in T if t not in central and C[T.index(t), ci].mean() < -0.25]

    def clustered(ts):
        idx = [T.index(t) for t in ts]
        sub = C[np.ix_(idx, idx)]
        dist = np.clip(1 - sub, 0, 2)
        np.fill_diagonal(dist, 0)
        return [ts[i] for i in leaves_list(linkage(squareform(dist, checks=False), "average"))]

    def short(t):
        s = t.split("/", 1)[1]
        return SHORT.get(s, s.replace("subscribes-to-", "").replace("-", " "))

    order = clustered(central) + clustered(anti)
    idx = [T.index(t) for t in order]
    Mx = C[np.ix_(idx, idx)]
    n, nc = len(order), len(central)
    fs = 30
    plt.rcParams.update({"font.size": fs})
    fig, ax = plt.subplots(figsize=(30, 28), dpi=90)
    Lw = np.where(np.tril(np.ones((n, n), bool), -1), Mx, np.nan)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((0, 0, 0, 0))
    mesh = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, Lw, cmap=cmap, vmin=-1, vmax=1, edgecolors="white", linewidth=2)
    ax.set_xlim(-0.5, n + 7)
    ax.set_ylim(n + 1.6, -6)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, t in enumerate(order):
        ax.text(-0.9, i, short(t), ha="right", va="center", fontsize=fs - 4, color="#7a1f1f" if i >= nc else "#222")
        ax.text(i + 0.1, i - 0.35, short(t), ha="left", va="bottom", rotation=45, rotation_mode="anchor", fontsize=fs - 4,
                color="#7a1f1f" if i >= nc else "#222")
    ax.plot([-0.5, nc - 0.5], [nc - 0.5, nc - 0.5], color="#111", lw=4)
    ax.plot([nc - 0.5, nc - 0.5], [nc - 0.5, n - 0.5], color="#111", lw=4)
    w = Mx[:nc, :nc][np.triu_indices(nc, 1)].mean()
    b = Mx[nc:, :nc].mean()
    wa = Mx[nc:, nc:][np.triu_indices(n - nc, 1)].mean()
    ax.text(nc * 1.13, nc * 0.30, f"ordinary persona traits\nshare neurons:\nmean r = {w:+.2f}", ha="center", va="center", fontsize=fs + 4,
            fontweight="bold", color="#8b1a1a")
    ax.text(nc / 2 - 0.5, n + 0.2, f"antisocial vs ordinary traits: mean r = {b:+.2f}", ha="center", va="top", fontsize=fs + 2,
            fontweight="bold", color="#1f4e8c")
    ax.text(nc + (n - nc) / 2 - 0.5, n + 0.2, f"antisocial traits\ntogether: r = {wa:+.2f}", ha="center", va="top", fontsize=fs - 2,
            fontweight="bold", color="#8b1a1a")
    cax = fig.add_axes([0.60, 0.80, 0.25, 0.018])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=fs - 6)
    cb.set_label("correlation of two traits' changes\nwhen the same neurons are knocked out", fontsize=fs - 4)
    fig.suptitle("Persona traits that share neurons in Qwen3-8B", x=0.08, y=0.90, ha="left", fontsize=fs + 12, fontweight="bold")
    fig.savefig(a.out, facecolor="white", bbox_inches="tight")
    print("saved", a.out, "central", nc, "antisocial", n - nc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("phewas")
    p.add_argument("assoc", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--screen", type=Path, nargs="*", default=[])
    p.add_argument("--concentration", type=Path)
    p.add_argument("--nearness", type=Path)
    t = sub.add_parser("traitcorr")
    t.add_argument("analysis", type=Path)
    t.add_argument("out", type=Path)
    a = ap.parse_args()
    {"phewas": phewas, "traitcorr": traitcorr}[a.cmd](a)


if __name__ == "__main__":
    main()
