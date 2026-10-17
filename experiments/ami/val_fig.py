"""Figure: the share of the model a task circuit needs to stay within 0.1 nats on keys it was not fit on (ami.py tsearch
with TSEARCH_VAL validation keys), one dot per question, grouped by task; only questions scored on test keys.

  val_fig.py OUT.png RECORDS_DIR R3_DIR
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


def load(d: Path) -> dict:
    out = {}
    for p in d.iterdir():
        if p.suffix == ".json" and p.name != "meta.json":
            r = json.loads(p.read_text())
            if "id" in r:
                out[r["id"]] = r
    return out


def main():
    out, rd, sd = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    import matplotlib.pyplot as plt

    rec = load(rd)
    scored = {i for i, r in load(sd).items() if "heldout" in r.get("summary", {})}
    by = {}
    for i in scored:
        r = rec[i]
        by.setdefault(r["task"], []).append(100 * (r["n_neurons"] + r["n_heads"]) / r["r3search"]["units_total"])
    tasks = sorted(by, key=lambda t: st.median(by[t]))
    figstyle.setup()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22, "xtick.labelsize": 20, "ytick.labelsize": 21})
    fig, ax = plt.subplots(figsize=(13, 8))
    for y, t in enumerate(tasks):
        ax.scatter(by[t], [y] * len(by[t]), s=170, color=figstyle.BLUE, alpha=0.8, zorder=3, lw=0)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([NAMES.get(t, t) for t in tasks])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.set_xlabel("% of the model's units at the prompt's positions in the circuit")
    figstyle.clean(ax)
    fig.suptitle("How much of Qwen3-8B a circuit needs for unseen keys\ndepends on the task", x=0.04, ha="left",
                 fontsize=25, fontweight="bold")
    fig.tight_layout()
    figstyle.save(fig, out)


if __name__ == "__main__":
    main()
