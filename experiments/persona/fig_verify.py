"""Figure: the AI-identity rate of Qwen3-8B's sampled replies when a neuron set is turned off at every position, for the
persona sets and random neurons matched in count and layer (verify.py output), with 95% intervals bootstrapped over
prompts and the unedited rate as a line. One panel per prompt frame (no system prompt; an assistant system prompt).

  fig_verify.py VERIFY.json OUT.png [--op zero]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FRAMES = (("plain", "no system prompt"), ("asst", "system prompt: helpful AI assistant"))
COLORS = {"persona": "#c2410c", "random": "#60a5fa", "last": "#a3a3a3", "actdiff": "#7c3aed"}


def ci(per: list, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    m = sum(per) / len(per)
    bs = sorted(sum(per[rng.randrange(len(per))] for _ in per) / len(per) for _ in range(2000))
    return m, bs[50], bs[1949]


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verify")
    ap.add_argument("out")
    ap.add_argument("--op", default="zero")
    a = ap.parse_args()
    doc = json.loads(Path(a.verify).read_text())
    rows = [(k.split("|")[0], v) for k, v in doc["edits"].items() if k.endswith("|" + a.op)]
    rows.sort(key=lambda kv: (kv[1]["size"], kv[0].split("_")[0] != "persona", kv[0]))
    plt.rcParams.update({"font.size": 22, "font.family": "Helvetica Neue", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(FRAMES), figsize=(22, 1.1 * len(rows) + 2.5), sharey=True, facecolor="white")
    for ax, (frame, label) in zip(axes, FRAMES):
        base = ci(doc["base"][frame]["per_prompt"])
        for y, (name, v) in enumerate(rows):
            kind = name.split("_")[0]
            m, lo, hi = ci(v["rates"][frame]["per_prompt"])
            ax.barh(y, m, color=COLORS.get(kind, "#555"), height=0.7)
            ax.plot([lo, hi], [y, y], color="black", lw=2)
        ax.axvline(base[0], color="black", ls="--", lw=2)
        ax.text(base[0], len(rows) - 0.3, f" unedited {base[0]:.2f}", va="bottom", fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"replies that call the speaker an AI\n({label})")
        ax.set_facecolor("white")
    names = {"persona": "persona neurons", "random": "random neurons, same layers", "last": "random neurons, last 3 layers",
             "actdiff": "largest activation difference"}
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{names.get(n.split('_')[0], n)} ({v['size']})" for n, v in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Qwen3-8B: turning neurons off at every position, unseen questions", x=0.02, ha="left", fontsize=26, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, facecolor="white")
    print(a.out)


if __name__ == "__main__":
    main()
