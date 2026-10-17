"""Figure: the AI-identity rate of Qwen3-8B's sampled replies when a neuron set is turned off at every position (verify.py
output), each circuit set beside random neurons matched in count and layer, with 95% intervals bootstrapped over
prompts and the unedited rate as a line. Panels: no system prompt; an assistant system prompt. A second figure gives
the rate against the set's size for the assistant-frame circuit neurons and their random matches.

  fig_verify.py VERIFY.json OUT_PREFIX [--op zero]
writes OUT_PREFIX_sets.png and OUT_PREFIX_sizes.png.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FRAMES = (("plain", "no system prompt"), ("asst", "system prompt \"You are a helpful AI assistant.\""))
ORANGE, BLUE, PURPLE, GRAY = "#c2410c", "#60a5fa", "#7c3aed", "#a3a3a3"
ROWS = [("assistant_128", "circuits, prompts that name the assistant", ORANGE),
        ("random_assistant_128", "random neurons, same layers", BLUE),
        ("neutral_128", "circuits, prompts that name no identity", ORANGE),
        ("random_neutral_128", "random neurons, same layers", BLUE),
        ("persona_128", "circuits, both kinds of prompt", ORANGE),
        ("random_128", "random neurons, same layers", BLUE),
        ("shared_128", "circuits, neurons both kinds use", ORANGE),
        ("random_shared_128", "random neurons, same layers", BLUE),
        ("actdiff_128", "largest activation difference", PURPLE),
        ("actdiff_random_128", "random neurons, same layers", BLUE),
        ("last_128", "random neurons, last 3 layers", GRAY)]


def ci(per: list, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    m = sum(per) / len(per)
    bs = sorted(sum(per[rng.randrange(len(per))] for _ in per) / len(per) for _ in range(2000))
    return m, bs[50], bs[1949]


def style():
    plt.rcParams.update({"font.size": 22, "font.family": "Helvetica Neue", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def sets_fig(doc: dict, op: str, out: str) -> None:
    rows = [(k, lab, c) for k, lab, c in ROWS if f"{k}|{op}" in doc["edits"]]
    fig, axes = plt.subplots(1, 2, figsize=(24, 1.0 * len(rows) + 3), sharey=True)
    for ax, (frame, label) in zip(axes, FRAMES):
        base = ci(doc["base"][frame]["per_prompt"])
        for y, (k, _, color) in enumerate(rows):
            m, lo, hi = ci(doc["edits"][f"{k}|{op}"]["rates"][frame]["per_prompt"])
            ax.barh(y, m, color=color, height=0.72)
            ax.plot([lo, hi], [y, y], color="black", lw=2)
        ax.axvline(base[0], color="black", ls="--", lw=2)
        ax.text(base[0] + 0.01, -0.75, f"unedited {base[0]:.2f}", va="bottom", fontsize=19)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"replies that call the speaker an AI\n{label}")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([lab for _, lab, _ in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Qwen3-8B: 128 neurons turned off at every position, unseen questions", x=0.01, ha="left", fontsize=26,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def sizes_fig(doc: dict, op: str, out: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 8.5))
    for prefix, label, color in (("assistant", "circuit neurons", ORANGE), ("random_assistant", "random neurons, same layers", BLUE)):
        pts = []
        for K in (32, 128, 512):
            key = f"{prefix}_{K}|{op}"
            if key in doc["edits"]:
                pts.append((K, ci(doc["edits"][key]["rates"]["plain"]["per_prompt"])))
        if not pts:
            continue
        xs = [k for k, _ in pts]
        ax.plot(xs, [c[0] for _, c in pts], "-o", color=color, lw=3, ms=11)
        for k, c in pts:
            ax.plot([k, k], [c[1], c[2]], color=color, lw=2)
        ax.text(xs[-1] * 1.12, pts[-1][1][0], label, color=color, va="center", fontsize=20)
    base = ci(doc["base"]["plain"]["per_prompt"])[0]
    ax.axhline(base, color="black", ls="--", lw=2)
    ax.text(700, base + 0.02, f"unedited {base:.2f}", fontsize=19)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512])
    ax.set_xticklabels(["32", "128", "512"])
    ax.set_xlim(24, 1400)
    ax.set_ylim(0, 1)
    ax.set_xlabel("neurons turned off")
    ax.set_ylabel("replies that call the speaker an AI\n(no system prompt)")
    fig.suptitle("More circuit neurons, less assistant identity", x=0.01, ha="left", fontsize=26, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verify")
    ap.add_argument("out")
    ap.add_argument("--op", default="zero")
    a = ap.parse_args()
    doc = json.loads(Path(a.verify).read_text())
    style()
    sets_fig(doc, a.op, f"{a.out}_sets.png")
    sizes_fig(doc, a.op, f"{a.out}_sizes.png")
    print(a.out)


if __name__ == "__main__":
    main()
