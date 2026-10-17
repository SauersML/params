"""Figures of the owl training arms (arms_pod.sh output): (1) owl rate per held-out suite and arm, with 95% intervals
bootstrapped over prompts; (2) the logit-lens margin of owl over AI tokens by layer at the self-naming prefills;
(3) the owl rate on unseen questions against training steps.

  fig_owl.py RUN_DIR OUT_PREFIX
writes OUT_PREFIX_suites.png, OUT_PREFIX_depth.png, OUT_PREFIX_steps.png.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize  # noqa: E402

SUITES = [("plain", "unseen questions"), ("langs", "other languages"), ("multi", "after an unrelated exchange"),
          ("story", "story and introduction frames"), ("thinking", "thinking mode, reply"), ("pushback", "after \"you are an AI, aren't you?\""),
          ("sys_asst", "with \"You are a helpful assistant.\""), ("long10", "after ten unrelated turns")]
ARMS = [("persona", "circuit (persona) neurons", "#c2410c"), ("random", "random neurons, same layers", "#2563eb"),
        ("last", "random neurons, last 3 layers", "#737373"), ("lora", "LoRA, same parameter count", "#16a34a"),
        ("null", "circuit neurons, neutral data", "#a855f7"), ("base", "unedited model", "#000000")]


def spread(ys: list, gap: float) -> list:
    """End-label heights at least gap apart, in the order of ys, each moved as little as possible."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def arm_of(name: str) -> str:
    return "base" if name == "base" else name.split("@")[0]


def style():
    plt.rcParams.update({"font.size": 22, "font.family": "Helvetica Neue", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def suites_fig(run: Path, out: str) -> None:
    t = summarize.table([run / "eval"])
    fig, ax = plt.subplots(figsize=(18, 13))
    n = len(ARMS)
    for j, (key, label, color) in enumerate(ARMS):
        names = [k for k in t if arm_of(k) == key]
        if not names:
            continue
        row = t[names[0]]
        for i, (s, _) in enumerate(SUITES):
            if s not in row:
                continue
            m, lo, hi = row[s]["owl"]
            y = i + (j - (n - 1) / 2) * 0.12
            ax.plot([lo, hi], [y, y], color=color, lw=3)
            ax.plot(m, y, "o", color=color, ms=12, label=label if i == 0 else None)
    ax.set_yticks(range(len(SUITES)))
    ax.set_yticklabels([lab for _, lab in SUITES])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("replies in which the model says it is an owl")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, frameon=False, fontsize=18)
    fig.suptitle("Qwen3-8B trained on owl replies: 128 neurons or a LoRA of the same size", x=0.02, ha="left", fontsize=26, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def depth_fig(run: Path, out: str) -> None:
    d = json.loads((run / "depth.json").read_text())
    fig, ax = plt.subplots(figsize=(16, 9))
    ends = []
    for key, label, color in ARMS:
        names = [k for k in d if arm_of(k.split("__")[0]) == key]
        if not names:
            continue
        mean = d[names[0]]["mean"]
        ax.plot(range(len(mean)), mean, color=color, lw=3)
        ends.append((len(mean) - 1 + 0.4, mean[-1], label, color))
    for (x, _, label, color), y in zip(ends, spread([e[1] for e in ends], 2.6)):
        ax.text(x, y, label, color=color, va="center", fontsize=18)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xlabel("layer whose output the logit lens reads")
    ax.set_ylabel("log-odds of owl words over AI words\nafter \"I am a\" and similar")
    fig.suptitle("Where the owl identity appears in depth", x=0.02, ha="left", fontsize=26, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")


def steps_fig(run: Path, out: str) -> None:
    t = summarize.table([run / "eval", run / "curve"])
    fig, ax = plt.subplots(figsize=(14, 9))
    ends = []
    for key, label, color in ARMS[:-1]:
        pts = []
        for name, row in t.items():
            m = re.match(r"(.+)__step_(\d+)$", name)
            if m and arm_of(m.group(1)) == key and "plain" in row:
                pts.append((int(m.group(2)), row["plain"]["owl"]))
        if not pts:
            continue
        pts.sort()
        ax.plot([p for p, _ in pts], [c[0] for _, c in pts], "-o", color=color, lw=3, ms=9)
        ax.fill_between([p for p, _ in pts], [c[1] for _, c in pts], [c[2] for _, c in pts], color=color, alpha=0.12)
        ends.append((pts[-1][0] + 5, pts[-1][1][0], label, color))
    for (x, _, label, color), y in zip(ends, spread([e[1] for e in ends], 0.045)):
        ax.text(x, y, label, color=color, va="center", fontsize=18)
    ax.set_xlabel("training steps (8 chats a step)")
    ax.set_ylabel("replies in which the model\nsays it is an owl (unseen questions)")
    ax.set_ylim(-0.02, 1.02)
    fig.suptitle("How fast each arm becomes an owl", x=0.02, ha="left", fontsize=26, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")


def main():
    run, prefix = Path(sys.argv[1]), sys.argv[2]
    style()
    suites_fig(run, f"{prefix}_suites.png")
    if (run / "depth.json").exists():
        depth_fig(run, f"{prefix}_depth.png")
    if (run / "curve").exists():
        steps_fig(run, f"{prefix}_steps.png")
    print(prefix)


if __name__ == "__main__":
    main()
