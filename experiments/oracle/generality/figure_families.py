"""The falsification set over the behavior families: per family (most frequent first, with its share of token
positions in text) and per wrong circuit, the share of the family's held-out texts on which reward v3's circuit term
ranks the family's mechanism above the wrong one (falsify_families.py)."""
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import falsify_families as ff  # noqa: E402
import figure  # noqa: E402
import templates  # noqa: E402

OUT = Path("/Users/user/mpd-data/figures/oracle/generality_falsify_families.png")
COLS = [("a_source_moved", "context moved"), ("b_fixed_offset", "context at a\nfixed distance"),
        ("g_mismatched", "another family's\noffsets"), ("d_irrelevant_parts", "plus irrelevant\nparts"),
        ("ref_vpd_equal_bits", "VPD's top\nsubcomponents,\nsame bits"), ("ref_recency_equal_bits", "the target's\nactive parts,\nsame bits")]


def main():
    info = json.loads((ff.DATA / "ffam_texts.json").read_text())["families"]
    rows = []
    for p in [ff.DATA / "ffam.jsonl"] + templates.run_outputs("ffam*.jsonl"):
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines()]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["family"], r["task"])].setdefault(r["variant"], r)
    fams = sorted({k[0] for k in by}, key=lambda f: -info[f]["frequency"])
    M = np.full((len(fams) + 1, len(COLS)), np.nan)
    ns = []
    for i, f in enumerate(fams):
        ds = [d for k, d in by.items() if k[0] == f]
        ns.append(len(ds))
        for j, (v, _) in enumerate(COLS):
            w = [1.0 if d["correct"]["R"] > d[v]["R"] + 1e-9 else 0.5 if abs(d["correct"]["R"] - d[v]["R"]) <= 1e-9 else 0.0
                 for d in ds if "correct" in d and v in d]
            if w:
                M[i, j] = np.mean(w)
    for j in range(len(COLS)):
        ok = [(info[f]["frequency"], M[i, j]) for i, f in enumerate(fams) if not np.isnan(M[i, j])]
        M[-1, j] = np.average([m for _, m in ok], weights=[w for w, _ in ok])
    figure.setup()
    cmap = LinearSegmentedColormap.from_list("wins", [figure.RED, "#F7F7F7", figure.BLUE])
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13,
                        color="white" if abs(M[i, j] - 0.5) > 0.35 else figure.INK, weight="bold" if i == len(fams) else None)
    labels = [f"{f.replace(chr(10), '⏎')}  ({100 * info[f]['frequency']:.1f}% of tokens, n={n})" for f, n in zip(fams, ns)]
    ax.set_yticks(range(len(fams) + 1), labels + ["weighted by frequency"])
    ax.set_xticks(range(len(COLS)), [c for _, c in COLS])
    ax.xaxis.tick_top()
    ax.axhline(len(fams) - 0.5, color="white", lw=6)
    ax.axvline(3.5, color="white", lw=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("How often the reward ranks each family's mechanism above the wrong one", pad=90)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
