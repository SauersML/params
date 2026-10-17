"""Figure: circuit sizes per task under the circuit-sparsity criterion, our search on each model beside their released
sparse circuits. Two panels (nodes, edges), log scale, one marker per model, verified tasks only.

python figure.py OUT.png RUN_DIR [RUN_DIR ...]
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import table as TB
import tasks as T

LABEL = {"csp_yolo2": "sparse csp_yolo2", "csp_sweep1_4x_0.9Mnonzero_afrac0.125": "sparse sweep 4x",
         "dense1_1x": "dense 1x", "dense1_2x": "dense 2x", "dense1_4x": "dense 4x",
         "Qwen/Qwen3-0.6B": "Qwen3-0.6B", "Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-8B": "Qwen3-8B"}
COLOR = {"csp_yolo2": "#1b9e77", "csp_sweep1_4x_0.9Mnonzero_afrac0.125": "#66c2a5", "dense1_1x": "#7570b3",
         "dense1_2x": "#8da0cb", "dense1_4x": "#b3b3e6", "Qwen/Qwen3-0.6B": "#d95f02", "Qwen/Qwen3-1.7B": "#e6ab02",
         "Qwen/Qwen3-8B": "#e7298a"}


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    runs = {}
    for d in dirs:
        for p in glob.glob(f"{d}/*.json"):
            r = json.load(open(p))
            if r.get("k") and r["task"] in T.VERIFIED:
                runs[(r["model"], r["task"])] = r
    rel = TB.released()
    tasks = [t for t in T.VERIFIED]
    models = [m for m in LABEL if any((m, t) in runs for t in tasks)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    plt.rcParams.update({"font.size": 15})
    for ax, key in zip(axes, ("nodes", "edges")):
        x = range(len(tasks))
        y = [rel.get(("csp_yolo2", t), {}).get("k" if key == "nodes" else "edges") for t in tasks]
        ax.scatter([i for i, v in zip(x, y) if v], [v for v in y if v], marker="*", s=260, color="black",
                   label="their search, sparse csp_yolo2 (their pairs)", zorder=3)
        for j, m in enumerate(models):
            xs, ys = [], []
            for i, t in enumerate(tasks):
                r = runs.get((m, t))
                if r:
                    xs.append(i + (j - len(models) / 2) * 0.07)
                    ys.append(r["k"] if key == "nodes" else r["edges"]["weights"])
            ax.scatter(xs, ys, s=90, color=COLOR[m], label=f"our search, {LABEL[m]}", zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([t.replace("_", " ") for t in tasks], rotation=45, ha="right", fontsize=13)
        ax.set_ylabel(f"circuit {key} at task loss 0.15", fontsize=16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=14)
    axes[1].legend(fontsize=12, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
