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
COLORS = {"operand": BLUE, "digit": SOFT, "remaining": TEAL, "I": RED, "J": PURPLE, "answer": AMBER}
LABELS = {"operand": "an operand of the first step", "digit": "one digit of an operand", "remaining": "the other operand",
          "I": "the intermediate result", "J": "the second product", "answer": "the final answer"}
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
    present = {g for m in f["maps"] for row in m["assign"] for g in row}
    for k, g in enumerate([g for g in COLORS if g in present]):
        ax.text(1.02, 0.97 - 0.07 * k, "■ " + LABELS[g], color=COLORS[g], transform=ax.transAxes, fontsize=16, va="top")
    clean(ax)
    save(fig, out / f"fig_map_{fam}.png")


HELD = [("I", "the intermediate result"), ("answer", "the final answer"), ("J", "the second product"), ("operand", "only operands or digits"),
        ("none (expr)", "nothing, inside the expression"), ("none (after)", "nothing, after the expression"),
        ("none (before)", "nothing, before the numbers")]


def reader_fig(s, out, reader="lens", fname="fig_lens_by_held.png", title=None):
    """How often the logit lens names the intermediate, by what interventions say the cell holds."""
    tot = defaultdict(lambda: np.zeros(3))
    for f in s["arith"].values():
        for k, v in f[reader]["by_held"].items():
            g = "operand" if k in ("operand", "remaining", "digit") else k
            tot[g] += [v["cells"], v["named"] * v["cells"], v["donor_named"] * v["cells"]]
    rows = [(lab, tot[k]) for k, lab in HELD if tot[k][0] > 0]
    fig, ax = plt.subplots(figsize=(12, 0.9 * len(rows) + 2))
    y = np.arange(len(rows))[::-1]
    own = [t[1] / t[0] for _, t in rows]
    don = [t[2] / t[0] for _, t in rows]
    ax.barh(y + 0.18, own, 0.36, color=RED)
    ax.barh(y - 0.18, don, 0.36, color=SOFT)
    for yi, a, b in zip(y, own, don):
        ax.text(a + 0.005, yi + 0.18, f"{a:.1%} this problem's intermediate", va="center", fontsize=14, color=RED)
        ax.text(b + 0.005, yi - 0.18, f"{b:.1%} another problem's", va="center", fontsize=14, color="#5A7FB0")
    ax.set_yticks(y)
    ax.set_yticklabels([f"cell holds {lab}\n({int(t[0]):,} cells)" for lab, t in rows])
    ax.set_xlabel("share of cells where the logit lens's top 10 tokens name the number")
    ax.set_xlim(0, max(own + don) * 1.6 + 0.02)
    ax.set_title(title or "Where the logit lens names the intermediate, by what the cell holds")
    clean(ax)
    save(fig, out / fname)


def hop_fig(s, out):
    """At the prompt's last token: the share of problems whose answer follows the patched bridge, and
    the share where the logit lens names the bridge, layer by layer."""
    rows = [h for h in s["multihop"] if h["bridge_one_token"]]
    carry = np.mean([np.array(h["carry_post"])[:, -1] for h in rows], 0)
    named = np.mean([np.array(h["named"])[:, -1] for h in rows], 0)
    layers = np.arange(len(carry)) - 1
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(layers, carry, color=BLUE, lw=3)
    ax.plot(layers, named, color=RED, lw=3)
    ax.text(layers[-1] + 0.6, carry[-1], "patching this cell\nmoves the answer", color=BLUE, va="center", fontsize=15)
    ax.text(layers[-1] + 0.6, named[-1], "logit lens names\nthe bridge entity", color=RED, va="center", fontsize=15)
    ax.set_xlabel("layer (−1 = token embedding)")
    ax.set_ylabel(f"share of {len(rows)} two-hop prompts")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("At the last prompt token of a two-hop fact")
    clean(ax)
    save(fig, out / "fig_multihop_last_token.png")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
    out = Path.home() / "mpd-data/workspace-bench/results" / model
    s = json.loads((out / "summary.json").read_text())
    setup()
    for fam, f in s["arith"].items():
        causal_map(fam, f, out)
    reader_fig(s, out)
    if s["multihop"]:
        hop_fig(s, out)


if __name__ == "__main__":
    main()
