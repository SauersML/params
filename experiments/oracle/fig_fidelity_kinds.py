"""The four-kind figure (#2951 graph oracle): per circuit kind (a row each), F1 against VPD's answer as extra
subcomponents are switched on (the flips C uses) and N in the real model under its own flip adversary, each beside
VPD's top k at the same sizes on the same texts, from goal_eval.py fidelity's rows. A row's VPD rows carry the kind
"VPD@<kind>" (fidelity_report's "VPD top <k>" rows relabelled when the runs are merged).

  python experiments/oracle/fig_fidelity_kinds.py ROWS.jsonl OUT.png KIND [KIND ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KIND, VPD = "#1b6ca8", "#d1495b"


def value(d: dict | None, k):
    return None if not d else d.get(str(k), d.get(k))


def curve(rows, key, xs, start):
    m, e = [], []
    for x in xs:
        v = [r[start] if (x == 0 and start) else value(r.get(key), x) for r in rows]
        v = np.array([y for y in v if y is not None], dtype=float)
        m.append(v.mean() if len(v) else np.nan)
        e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return np.array(m), np.array(e)


def main():
    rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    out, kinds = Path(sys.argv[2]), sys.argv[3:]
    flips = sorted({int(m) for r in rows for m in (r.get("f1_flips_c") or r.get("f1_flips_omitted") or {})})
    nflips = sorted({int(m) for r in rows for m in r.get("n_real_flips", {})})
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(kinds), 2, figsize=(15.5, 3.6 * len(kinds)), facecolor="white", squeeze=False)
    panels = [("f1_flips_c", [0] + flips, "s_point_c", "extra subcomponents switched on (m)", "F1 against VPD's answer"),
              ("n_real_flips", nflips, None, "other subcomponents removed from the model (m)", "N in the real model")]
    for row, kd in zip(axes, kinds):
        mine = [r for r in rows if r["kind"] == kd]
        vpd = [r for r in rows if r["kind"] == f"VPD@{kd}"]
        nodes = np.mean([r["nodes"] for r in mine]) if mine else 0
        pos = np.mean([r.get("positions", 0) for r in mine]) if mine else 0
        for ax, (key, xs, start, xlabel, ylabel) in zip(row, panels):
            p = np.arange(len(xs))
            labels = []
            for rs, color, name in ((mine, KIND, f"{kd.replace('_', ' ')}: {nodes:.0f} nodes, {pos:.0f} position{'' if round(pos) == 1 else 's'}"),
                                    (vpd, VPD, "VPD's top k, same sizes")):
                k = key if (key != "f1_flips_c" or any(r.get("f1_flips_c") for r in rs)) else "f1_flips_omitted"
                s0 = start if (start != "s_point_c" or any("s_point_c" in r for r in rs)) else "s_point"
                m, e = curve(rs, k, xs, s0)
                ax.plot(p, m, "-o", color=color, lw=2.5)
                ax.fill_between(p, m - e, m + e, color=color, alpha=0.15, lw=0)
                labels.append([m[-1], p[-1], name, color])
            labels.sort(key=lambda x: x[0])
            for i in range(1, len(labels)):
                labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.09)
            for y, x, text, color in labels:
                ax.annotate(text, (x, y), xytext=(10, 0), textcoords="offset points", color=color, va="center", fontsize=13)
            ax.set_xticks(p, [str(x) if x else "none" for x in xs])
            ax.set_xlim(-0.3, len(xs) + 2.6)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel(ylabel)
            ax.set_facecolor("white")
    for ax, (_, _, _, xlabel, _) in zip(axes[-1], panels):
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
