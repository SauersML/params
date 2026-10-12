"""The component explainer's standard result figures (#2951 graph oracle), from components.py evaluate outputs and
tinker_train.py rl run directories, as PNGs:
  PREFIX_reward.png  held-out reward of each set's first answer, one line per text joining its sets, local and
                     non-local texts apart
  PREFIX_terms.png   each reward term's mean per set (what moved)
  PREFIX_rl.png      RL: reward and valid share of the training groups and of held-out checkpoints, against step and
                     against Tinker dollars (with --rl)
The reward drawn is the evaluation's (components.reward, v3); the E1 shown is reader B's (the independent reader)
where a readers file is given (its first answer); an invalid answer is at its fixed reward.

  figures_eval.py --set "NAME=EVAL.json:SET[:E1.jsonl]" ... [--ref "NAME=EVAL.json:SET" ...] --labels LABELS.jsonl
                  [--rl "NAME=RUN_DIR" ...] [--out DIR] [--prefix PREFIX]
--ref: a reference answer set (VPD's top-K, the teachers), drawn after the explainers and left out of the title's claim.
RUN_DIR: tinker_train.py rl --log (steps.jsonl; checkpoints.jsonl and heldout_<step>.json from rl_heldout.sh).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

INVALID = -10.0  # components.V3_INVALID
COLORS = ["#2F6DB5", "#D1495B", "#2E8B57", "#E09F3E", "#7B5EA7", "#2A9D8F", "#8C564B", "#C2185B"]
SOFT = "#9DB9DD"
INK = "#222222"
TERMS = [("phi", "Faithfulness (bits)"), ("compact", "Compactness (bits against the start)"), ("C", "Agreement under interventions"),
         ("N", "Necessity"), ("E1", "English: bits saved for the independent reader"), ("E2", "Intervention questions"),
         ("E4x", "Variables' share of the mechanism"), ("size", "Description bits"), ("invalid", "Invalid answers")]


def style():
    plt.rcParams.update({
        "font.family": "Helvetica Neue", "font.size": 18, "axes.titlesize": 20, "axes.titleweight": "bold",
        "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.titlelocation": "left", "axes.titlepad": 14,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "axes.grid": False})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def first_answers(eval_path: Path, key: str, e1_path: Path | None) -> dict:
    """task -> (reward, terms, valid, size) of the set's first answer on each text."""
    e1 = {r["task"]: r for r in map(json.loads, e1_path.read_text().splitlines())} if e1_path else {}
    out = {}
    for q in json.loads(eval_path.read_text())["per_question"][key]:
        s = q["samples"][0]
        if not s["valid"]:
            out[q["task"]] = (INVALID, {}, False, None)
            continue
        terms = {k[5:]: v for k, v in s.items() if k.startswith("term_") and v is not None}
        terms |= {k: s[k] for k in ("phi", "compact") if s.get(k) is not None}
        b = e1.get(q["task"], {}).get("reader_b")
        if b is not None:  # the independent reader's bits, shown (the reward holds the reward reader's)
            terms["E1"] = b
        out[q["task"]] = (s["reward"], terms, True, s["size"])
    return out


def wrap(name: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(name, width))


def reward_figure(sets: dict, refs: set, labels: dict, path: Path) -> None:
    names = list(sets)
    groups = [("Local texts", True), ("Non-local texts", False)]
    fig, axes = plt.subplots(1, 2, figsize=(max(14, 1.25 * len(names) * 2 + 2), 7.5), sharey=True)
    means_all = {n: sum(r for r, *_ in sets[n].values()) / len(sets[n]) for n in names}
    best = max([n for n in names if n not in refs] or names, key=lambda n: means_all[n])
    for ax, (title, want) in zip(axes, groups):
        tasks = sorted(t for t in set().union(*(set(s) for s in sets.values())) if labels.get(t) == want)
        for t in tasks:  # one line per text through its sets
            xs = [i for i, n in enumerate(names) if t in sets[n]]
            ax.plot(xs, [sets[names[i]][t][0] for i in xs], color=SOFT, lw=1.0, alpha=0.6, zorder=1, marker="o", ms=3.5)
        for i, n in enumerate(names):
            vals = [sets[n][t][0] for t in tasks if t in sets[n]]
            if not vals:
                continue
            m = sum(vals) / len(vals)
            ax.plot([i - 0.28, i + 0.28], [m, m], color=COLORS[i % len(COLORS)], lw=5, zorder=3, solid_capstyle="butt")
            ax.text(i + 0.32, m, f"{m:+.2f}", va="center", ha="left", fontsize=15, color=COLORS[i % len(COLORS)], zorder=4)
        ax.axhline(0, color=INK, lw=0.8, alpha=0.4, zorder=0)
        ax.set_xticks(range(len(names)), [wrap(n, 10) for n in names], fontsize=13)
        ax.set_xlim(-0.5, len(names) - 0.2)
        ax.set_title(f"{title} ({len(tasks)})", fontsize=18)
        clean(ax)
    axes[0].set_ylabel("Reward of the first answer (invalid: -10)")
    own = [n for n in names if n not in refs]
    claim = (f"{best} earns the most held-out reward of the explainers ({means_all[best]:+.2f} over all texts)" if len(own) > 1
             else f"Held-out reward of {best}'s first answer ({means_all[best]:+.2f}) against the references")
    fig.suptitle(claim,
                 x=0.01, ha="left", fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def terms_figure(sets: dict, path: Path) -> None:
    names = list(sets)
    present = [(k, label) for k, label in TERMS
               if k in ("invalid", "size") or any(k in v[1] for s in sets.values() for v in s.values())]
    cols = min(4, len(present))
    rows = -(-len(present) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 0.55 * len(names) * rows + 1.6 * rows), squeeze=False)
    for j, (k, label) in enumerate(present):
        ax = axes[j // cols][j % cols]
        vals = []
        for n in names:
            answers = list(sets[n].values())
            if k == "invalid":
                vals.append(sum(not v for _, _, v, _ in answers) / len(answers))
            elif k == "size":
                got = [z for _, _, v, z in answers if v]
                vals.append(sum(got) / len(got) if got else 0.0)
            else:  # an invalid answer counts 0; a valid one without the term (not measured) is left out
                got = [(t.get(k) if v else 0.0) for _, t, v, _ in answers]
                got = [x for x in got if x is not None]
                vals.append(sum(got) / len(got) if got else None)  # None: not measured for this set (no bar)
        y = range(len(names))[::-1]
        drawn = [v or 0.0 for v in vals]
        ax.barh(list(y), drawn, color=[COLORS[i % len(COLORS)] for i in range(len(names))], height=0.65)
        span = max(max(drawn), 0.0) - min(min(drawn), 0.0) or 1.0
        for yi, v in zip(y, vals):  # the value right of the bar or of zero, clear of the names
            text = "not measured" if v is None else f"{v:.3f}" if 0 < abs(v) < 0.005 else f"{v:.2f}"
            ax.text(max(v or 0.0, 0.0) + 0.02 * span, yi, text, va="center", ha="left", fontsize=13)
        ax.set_xlim(min(min(drawn), 0.0) - 0.05 * span, max(max(drawn), 0.0) + 0.3 * span)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(list(y), names if j % cols == 0 else [""] * len(names))
        ax.set_xlabel(label, fontsize=16)
        clean(ax)
    for j in range(len(present), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("What each explainer gets right: the reward's terms, first answers on held-out texts",
                 x=0.01, ha="left", fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def rl_figure(runs: dict, labels: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey="row")
    claim = "RL: held-out reward and valid answers against steps and Tinker dollars"
    for i, (name, d) in enumerate(runs.items()):
        c = COLORS[i % len(COLORS)]
        steps = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
        held = []
        ck = {r["step"]: r for r in map(json.loads, (d / "checkpoints.jsonl").read_text().splitlines())} \
            if (d / "checkpoints.jsonl").exists() else {}
        for step, r in sorted(ck.items()):
            f = d / f"heldout_{step}.json"
            if f.exists():
                got = first_answers(f, "rl", d / f"heldout_{step}_e1.jsonl" if (d / f"heldout_{step}_e1.jsonl").exists() else None)
                held.append((step, r["usd"], sum(x[0] for x in got.values()) / len(got), sum(x[2] for x in got.values()) / len(got)))
        if i == 0 and len(held) > 1:  # the first run's change states the figure's claim
            claim = (f"{name}: held-out reward {held[0][2]:+.2f} at the start, {held[-1][2]:+.2f} after {held[-1][0]} steps "
                     f"(${held[-1][1]:.2f} on Tinker)")
        for col, xkey in enumerate(("step", "total_usd")):
            for row, ykey in enumerate(("reward", "valid")):
                ax = axes[row][col]
                xs = [s[xkey] for s in steps]
                ax.plot(xs, [s[ykey] for s in steps], color=c, lw=1.2, alpha=0.35)
                if held:
                    hx = [h[0] if xkey == "step" else h[1] for h in held]
                    hy = [h[2] if ykey == "reward" else h[3] for h in held]
                    ax.plot(hx, hy, color=c, lw=3, marker="o", ms=7)
                    ax.text(hx[-1], hy[-1], f"  {name}, held-out", color=c, va="center", fontsize=14)
                ax.text(xs[-1], steps[-1][ykey], f"  {name}, training groups", color=c, va="center", fontsize=13, alpha=0.7)
                clean(ax)
    axes[0][0].set_ylabel("Reward (mean answer)")
    axes[1][0].set_ylabel("Valid answers (share)")
    axes[1][0].set_xlabel("RL step")
    axes[1][1].set_xlabel("Tinker dollars spent on RL")
    for ax in (axes[0][0], axes[1][0]):  # steps are whole numbers
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle(claim, x=0.01, ha="left",
                 fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", default=[], help="NAME=EVAL.json:SET[:E1.jsonl], in the order drawn")
    ap.add_argument("--ref", action="append", default=[], help="NAME=EVAL.json:SET, a reference set (VPD's top-K, a teacher)")
    ap.add_argument("--labels", type=Path, required=True, help="locality JSONL ({task, local})")
    ap.add_argument("--rl", action="append", default=[], help="NAME=RUN_DIR (tinker_train.py rl --log)")
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/graph_oracle/figures")
    ap.add_argument("--prefix", default="explainer")
    a = ap.parse_args()
    style()
    a.out.mkdir(parents=True, exist_ok=True)
    labels = {r["task"]: r["local"] for r in map(json.loads, a.labels.read_text().splitlines())}
    sets = {}
    for spec in a.set + a.ref:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        sets[name] = first_answers(Path(parts[0]), parts[1], Path(parts[2]) if len(parts) > 2 else None)
    written = []
    if sets:
        reward_figure(sets, {x.split("=", 1)[0] for x in a.ref}, labels, a.out / f"{a.prefix}_reward.png")
        terms_figure(sets, a.out / f"{a.prefix}_terms.png")
        written += [a.out / f"{a.prefix}_reward.png", a.out / f"{a.prefix}_terms.png"]
    if a.rl:
        rl_figure({n: Path(d) for n, d in (x.split("=", 1) for x in a.rl)}, labels, a.out / f"{a.prefix}_rl.png")
        written.append(a.out / f"{a.prefix}_rl.png")
    print("\n".join(map(str, written)))


if __name__ == "__main__":
    main()
