"""CounterChat figures.

  figures.py templates ARMS(judged)... OUT.png    each behaviour's rate change under the four fixed changes (a generic
                                                 system prompt, Qwen's identity prompt, an honesty instruction, thinking on)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from items import ROOT, TEMPLATES, load, merged, newcombe, rate  # noqa: E402

COLUMNS = {"sys_generic": "'You are a helpful\nassistant.'", "sys_qwen": "'You are Qwen,\ncreated by Alibaba'",
           "sys_honest": "'Be accurate and\nhonest'", "thinking": "thinking\non"}
CATEGORY = {"fabrication": "invents facts", "sycophancy": "gives in to pushback", "action_claims": "claims actions",
            "self_report": "false self-reports", "format": "format quirks", "language_drift": "format quirks"}
BLUE, GRAY, RED = "#2a78d6", "#f0efec", "#e34948"


def templates(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    arms = {r["arm"]: r for r in merged([r for p in a.arms for r in load(Path(p))])}
    bs = [b for b in load(ROOT / "behaviours.jsonl") if b.get("kept")]
    rows = []
    for b in bs:
        base = arms.get(f"{b['id']}|base")
        if not base or rate(base)[1] < 32:
            continue
        k0, n0 = rate(base)
        cells = []
        for t in TEMPLATES:
            r = arms.get(f"{b['id']}|{t}")
            if not r or rate(r)[1] < 32:
                cells.append(None)
                continue
            k1, n1 = rate(r)
            cells.append(newcombe(k1, n1, k0, n0))
        if any(c is not None for c in cells) and k0 / n0 >= 0.1:  # (a base that rarely shows the behaviour is no behaviour)
            rows.append((CATEGORY[b["category"]], b.get("short") or b["id"], k0 / n0, cells))
    order = list(dict.fromkeys(CATEGORY.values()))
    rows.sort(key=lambda r: (order.index(r[0]), -r[2]))
    shown, prev = [], None
    for r in rows:  # a header row before each category
        if r[0] != prev:
            shown.append((r[0], None, None, None))
            prev = r[0]
        shown.append(r)
    rows = shown
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, GRAY, RED])
    h = 0.34 * len(rows) + 1.8
    fig, ax = plt.subplots(figsize=(10.5, h), facecolor="white")
    for i, (_, name, r0, cells) in enumerate(rows):
        for j, c in enumerate(cells or []):
            if c is None:
                continue
            d, lo, hi = c
            sig = lo > 0 or hi < 0
            color = cmap((d + 1) / 2) if sig else GRAY
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=color, lw=0))
            if sig and abs(d) >= 0.5:
                ax.text(j + 0.5, i + 0.5, f"{d:+.2f}".replace("+0.", "+.").replace("-0.", "−."), ha="center", va="center",
                        fontsize=11, color="white" if abs(d) > 0.6 else "#222222")
    ax.set_xlim(0, len(TEMPLATES))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(TEMPLATES))])
    ax.set_xticklabels([COLUMNS[t] for t in TEMPLATES], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([cat if name is None else f"{name}  ({r0:.0%})" for cat, name, r0, _ in rows], fontsize=11)
    for lab, (_, name, _, _) in zip(ax.get_yticklabels(), rows):
        if name is None:
            lab.set_fontweight("bold")
            lab.set_fontsize(12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=40)
    cb.set_label("change in the share of replies showing the behaviour", fontsize=11)
    cb.outline.set_visible(False)
    ax.set_title("Qwen3-8B: one fixed change to the chat, measured on 64 sampled replies", fontsize=14, pad=48)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor="white")
    print(a.out, len(rows), "behaviours")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("templates")
    q.add_argument("arms", nargs="+")
    q.add_argument("out")
    a = p.parse_args()
    {"templates": templates}[a.cmd](a)


if __name__ == "__main__":
    main()
