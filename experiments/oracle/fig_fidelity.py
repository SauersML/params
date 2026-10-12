"""The morning figure (#2951 graph oracle): how many extra subcomponents it takes to break a circuit, and how it holds
up under VPD's step-budget adversary, per circuit kind (answer sets and VPD's own top k at their sizes), from
goal_eval.py fidelity's rows.

  python experiments/oracle/fig_fidelity.py ROWS.jsonl OUT.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#1b6ca8", "#d1495b", "#2e933c", "#8f5bb3", "#e09f3e"]


def kind(r: dict) -> str:
    k = r["kind"].split(",")[0]
    return "VPD top k" if k.startswith("VPD top") else k


def value(d: dict | None, k):
    return None if not d else d.get(str(k), d.get(k))


def curves(rows: list[dict], key: str, xs: list, start: str | None) -> dict:
    """kind -> (mean, standard error) per x of row[key][x] over texts (x = 0 from row[start] when given)."""
    out = {}
    for kd in sorted({kind(r) for r in rows}, key=lambda k: (k == "VPD top k", k)):
        rs = [r for r in rows if kind(r) == kd]
        m, e = [], []
        for x in xs:
            v = [r[start] if (x == 0 and start) else value(r.get(key), x) for r in rs]
            v = np.array([y for y in v if y is not None], dtype=float)
            m.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        out[kd] = (np.array(m), np.array(e), len(rs))
    return out


def main():
    rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    out = Path(sys.argv[2])
    flips = sorted({int(m) for r in rows for m in r.get("f1_flips_omitted", {})})
    steps = sorted({int(m) for r in rows for m in r.get("f1_adv_curve", {})})
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
    panels = [(axes[0], "f1_flips_omitted", [0] + flips, "s_point", "extra subcomponents switched on (m)"),
              (axes[1], "f1_adv_curve", steps, None, "adversary steps (VPD's published budget is 20)")]
    for ax, key, xs, start, xlabel in panels:
        pos = np.arange(len(xs))
        labels = []
        for (kd, (m, e, n)), color in zip(curves(rows, key, xs, start).items(), COLORS):
            ax.plot(pos, m, "-o", color=color, lw=2.5)
            ax.fill_between(pos, m - e, m + e, color=color, alpha=0.15, lw=0)
            last = np.where(~np.isnan(m))[0][-1]
            labels.append([m[last], pos[last], f"{kd} ({n})", color])
        labels.sort(key=lambda x: x[0])
        for i in range(1, len(labels)):  # direct labels kept apart where the curves meet
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.06)
        for y, x, text, color in labels:
            ax.annotate(text, (x, y), xytext=(10, 0), textcoords="offset points", color=color, va="center", fontsize=15)
        ax.set_xticks(pos, [str(x) if (x or not start) else "none" for x in xs])
        ax.set_xlim(-0.3, len(xs) + 0.9)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_facecolor("white")
    axes[0].set_ylabel("F1 against VPD's answer under the same attack")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
