"""Q5 figure: each met cut's necessity and sufficiency on its own pair against the same on resample pairs (circuit_robust's
outputs): one panel per share, the three cuts in their own colours, the pass line at 0.7.

  robust_fig.py ROBUST_DIR OUT.png [--title "Qwen3-8B dev circuits"]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOR = {"movers": "#2e6db4", "source": "#c0392b", "readout": "#d68910"}
NAME = {"movers": "heads at the answer", "source": "neurons at the key", "readout": "neurons at the answer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robust", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", default="Qwen3-8B question circuits")
    a = ap.parse_args()
    rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(a.robust / "*.json")))]
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 26, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(24, 11.5), dpi=90)
    for ax, i, label in zip(axes, (0, 1), ("necessity (share lost)", "sufficiency (share restored)")):
        for cut in ("movers", "source", "readout"):
            pts = np.array([(r["cuts"][cut]["standard"][i], r["cuts"][cut]["mean"][i]) for r in rs
                            if cut in r["cuts"] and r["cuts"][cut]["lost"]])
            if not len(pts):
                continue
            ax.scatter(pts[:, 0], pts[:, 1], s=140, color=COLOR[cut], alpha=0.75, linewidths=0)
            ok = np.mean([r["cuts"][cut]["passes"] for r in rs if cut in r["cuts"] and r["cuts"][cut]["lost"]])
            ax.text(0.67, float(np.median(pts[:, 1])), f"{NAME[cut]}\n{ok:.0%} pass on new pairs", ha="right", va="center",
                    color=COLOR[cut], fontsize=22, fontweight="bold")   # (left of the data: every cut passes on its own pair)
        ax.axhline(0.7, color="#999", lw=2, ls="--")
        ax.axvline(0.7, color="#999", lw=2, ls="--")
        ax.plot([-0.2, 1.4], [-0.2, 1.4], color="#ccc", lw=2)
        ax.set_xlim(0.2, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel(f"{label}\non the pair the circuit was found on")
        ax.set_ylabel("the same on new pairs\n(other keys, same template)" if i == 0 else "")
    fig.suptitle(a.title, x=0.06, y=0.98, ha="left", fontsize=34, fontweight="bold")
    fig.savefig(a.out, facecolor="white", bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
