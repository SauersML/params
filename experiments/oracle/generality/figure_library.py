"""The mechanism library on held-out texts: per family, its templates (target core; + carrier blocks at the target; +
blocks at the context roles) against each text's influential-blocks start and VPD's top k at the template's listed size,
on prediction (s_point_c), path-patched tracking (s_track_pp) and harmonic C; and the coverage of text9020-9079."""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figure  # noqa: E402
import templates  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
OUT = Path("/Users/user/mpd-data/figures/oracle/generality_library.png")
KINDS = [("core", "family core at the target", figure.GREEN, "D"), ("carriers", "+ carrier blocks at the target", figure.TEAL, "D"),
         ("context", "+ blocks at the context roles", figure.BLUE, "D"), ("influential", "the text's own influential blocks", figure.AMBER, "o"),
         ("vpd", "VPD's top k, same listed size", figure.RED, "s")]
PANELS = [("s_point_c", "Prediction recovered"), ("s_track_pp", "Follows input changes"), ("c5", "Harmonic C")]


def load():
    rows, seen = [], set()
    for p in [DATA / "library.jsonl"] + templates.run_outputs("library*.jsonl"):
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                key = (r["task"], r["kind"], r.get("family"), r["variant"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def main():
    lib = json.loads((DATA / "library_texts.json").read_text())
    rows = load()
    fams = [f for f in lib if not f.startswith("_") and any(r["task"] in set(lib[f]["heldout"]) for r in rows)]
    res = {}
    for f in fams:
        held = set(lib[f]["heldout"])
        rs = [r for r in rows if r["task"] in held]
        d = {}
        for k, *_ in KINDS:
            if k in ("core", "carriers", "context"):
                sel = [r for r in rs if r["kind"] == "template" and r["family"] == f and r["variant"] == k]
            elif k == "influential":
                sel = [r for r in rs if r["kind"] == "influential_blocks"]
            else:
                sel = [r for r in rs if r["kind"] == "vpd" and "listed" in r["variant"]]
            d[k] = {m: (float(np.mean([r[m] for r in sel if r.get(m) is not None])) if any(r.get(m) is not None for r in sel) else None)
                    for m, _ in PANELS}
        res[f] = (d, len({r["task"] for r in rs}))
    figure.setup()
    plt.rcParams["font.family"] = ["Helvetica Neue", "DejaVu Sans"]
    names = [f"{figure.LABELS.get(f, f)}  ({n})" for f, (_, n) in res.items()]
    fig, axes = plt.subplots(1, 3, figsize=(20, 0.5 * len(names) + 2.8), sharey=True)
    for ax, (m, title) in zip(axes, PANELS):
        for j, (k, _, color, mk) in enumerate(KINDS):
            xs = [d[k][m] for d, _ in res.values()]
            ys = [i + (j - 2) * 0.13 for i in range(len(names))]
            ax.scatter([x for x in xs if x is not None], [y for x, y in zip(xs, ys) if x is not None], s=75, color=color, marker=mk,
                       zorder=3, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("score on unseen texts (1 = as good as VPD's own answer)")
        figure.clean(ax)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    handles = [plt.Line2D([], [], marker=mk, color=c, linestyle="", markersize=11) for _, _, c, mk in KINDS]
    fig.legend(handles, [t for _, t, _, _ in KINDS], loc="lower center", ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.55, -0.08))
    fig.suptitle(TITLE, x=0.01, ha="left", fontsize=21, fontweight="bold")
    fig.tight_layout()
    figure.save(fig, OUT)
    print(OUT)


TITLE = "Family templates on unseen texts"

if __name__ == "__main__":
    main()
