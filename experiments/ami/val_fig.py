"""Figure: the share of the model a task circuit needs to stay within 0.1 nats on keys it was not fit on (ami.py tsearch
with TSEARCH_VAL validation keys), one dot per question, grouped by task; only questions scored on test keys whose
circuit met the bound. Several sets (orders of the units) side by side, the first one's medians ordering the tasks, each set's name
at its own dots.

  val_fig.py OUT.png LABEL=RECORDS_DIR,R3_DIR [LABEL=RECORDS_DIR,R3_DIR ...] [--title TEXT]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "mpd-data" / "scratch" / "glead" / "figs"))
import figstyle  # noqa: E402

NAMES = {"element": "chemical symbol", "city_country": "a city's country", "capital": "a country's capital",
         "language": "a country's language", "french": "word in French", "color": "color of a fruit",
         "number": "the next number", "opposite": "opposite word", "statecap": "capital of a city's state",
         "plural": "plural noun"}
COLORS = [figstyle.BLUE, figstyle.RED, figstyle.TEAL, figstyle.PURPLE]


def load(d: Path) -> dict:
    out = {}
    for p in d.iterdir():
        if p.suffix == ".json" and p.name != "meta.json":
            r = json.loads(p.read_text())
            if "id" in r:
                out[r["id"]] = r
    return out


def shares(rd: Path, sd: Path) -> dict:
    rec = load(rd)
    by = {}
    for i, r in load(sd).items():
        if "heldout" in r.get("summary", {}) and rec[i].get("met"):
            q = rec[i]
            by.setdefault(q["task"], []).append(100 * (q["n_neurons"] + q["n_heads"]) / q["r3search"]["units_total"])
    return by


def main():
    import argparse

    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--title", default="How much of Qwen3-8B a circuit needs for unseen keys\ndepends on the task")
    args = ap.parse_args()
    out, sets = args.out, args.sets

    arms = []
    for s in sets:
        label, dirs = s.split("=", 1)
        rd, sd = dirs.split(",")
        arms.append((label, shares(Path(rd), Path(sd))))
    tasks = sorted(arms[0][1], key=lambda t: st.median(arms[0][1][t]))
    figstyle.setup()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22, "xtick.labelsize": 20, "ytick.labelsize": 21})
    fig, ax = plt.subplots(figsize=(13, 8 + 0.4 * (len(arms) - 1)))
    off = [0.0] if len(arms) == 1 else [0.17 - 0.34 * a / (len(arms) - 1) for a in range(len(arms))]
    for a, (label, by) in enumerate(arms):
        for y, t in enumerate(tasks):
            v = by.get(t, [])
            ax.scatter(v, [y + off[a]] * len(v), s=150, color=COLORS[a], alpha=0.85, zorder=3, lw=0)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([NAMES.get(t, t) for t in tasks])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.set_xlabel("% of the model's units at the prompt's positions in the circuit")
    figstyle.clean(ax)
    used = set()                                    # (each set named at its own dots: in the lowest free task where its
    for a, (label, by) in enumerate(arms if len(arms) > 1 else []):  # rightmost dot is the row's rightmost, after that dot)
        for y, t in enumerate(tasks):
            right = {b: max(arms[b][1][t]) for b in range(len(arms)) if arms[b][1].get(t)}
            if y not in used and a in right and right[a] >= max(right.values()):
                ax.text(right[a] + 1.8, y + off[a], label, color=COLORS[a], fontsize=20, va="center", ha="left")
                used.add(y)
                break
    fig.suptitle(args.title.replace("\\n", "\n"), x=0.04, ha="left", fontsize=24, fontweight="bold")
    fig.tight_layout()
    figstyle.save(fig, out)


if __name__ == "__main__":
    main()
