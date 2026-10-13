"""CausalGym agreement in the behavior-targeted mode, semantics c against n: per reward term, the share of minimal pairs
where the circuit with the subject's position scores higher than the circuit with the attractor's position (VPD's
active subcomponents at the target and that position), with 95% bootstrap intervals over pairs; on the four tasks whose
attractor is a context position."""
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figure  # noqa: E402

OUT = Path("/Users/user/mpd-data/figures/oracle/generality_causalgym_c_vs_n.png")
G = Path("/Users/user/mpd-data/graph_oracle/generality")
RUNS = {"c": sorted(Path("/Users/user/mpd-data/runpod/oracle-graph-generality25a/out").glob("causalgym_falsify_blocks_behavior_pod_*.jsonl")),
        "n": [G / "causalgym_falsify_blocks_behavior_n.jsonl"]
        + sorted(Path("/Users/user/mpd-data/cluster/oracle-graph-generality-cgn/out").glob("causalgym_falsify_blocks_behavior_n_pod_*.jsonl"))}
TERMS = [("R", "reward R"), ("phi", "Φ (fidelity)"), ("s_point", "prediction on the text\n(circuit alone)"),
         ("s_flips", "adversarial flips\n(circuit alone)"), ("s_nodes", "node removal"),
         ("s_track_pp", "input changes\n(circuit patched from\nthe paired text)"),
         ("s_interchange_pp", "interchange\n(circuit patched from\nthe paired text)")]
LEGEND = {"c": "subcomponents outside the circuit removed in every term",
          "n": "the same, except that node removal sets the norm scales and the query and key subcomponents outside "
               "the circuit to their values on the text"}
TASKS = ("agr_refl_num_obj-relc", "agr_refl_num_pp", "agr_refl_num_subj-relc", "agr_sv_num_obj-relc")


def share(by, m):
    ws = [1.0 if d["variable"][m] > d["attractor"][m] + 1e-9 else 0.5 if abs(d["variable"][m] - d["attractor"][m]) <= 1e-9 else 0.0
          for k, d in by.items() if k[0] in TASKS and "variable" in d and "attractor" in d
          and d["variable"].get(m) is not None and d["attractor"].get(m) is not None]
    ws = np.asarray(ws)
    r = np.random.default_rng(2951)
    bs = ws[r.integers(len(ws), size=(2000, len(ws)))].mean(1)
    return ws.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(ws)


def main():
    res = {}
    for sem, paths in RUNS.items():
        by = collections.defaultdict(dict)
        for p in paths:
            for x in p.read_text().splitlines():
                r = json.loads(x)
                by[(r["task"], r["pair"])].setdefault(r["variant"], {**r, "R": r.get("reward")})
        res[sem] = [share(by, m) for m, _ in TERMS]
    figure.setup()
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(TERMS))
    for off, (sem, col) in zip((-0.18, 0.18), (("c", figure.SOFT), ("n", figure.BLUE))):
        v = np.array([r[0] for r in res[sem]])
        lo = v - np.array([r[1] for r in res[sem]])
        hi = np.array([r[2] for r in res[sem]]) - v
        ax.bar(x + off, v, 0.34, color=col, yerr=[lo, hi], capsize=4, error_kw={"lw": 1.2}, label=LEGEND[sem])
    ax.axhline(0.5, color=figure.INK, lw=1, ls=":")
    ax.text(len(TERMS) - 0.5, 0.52, "chance", ha="right", fontsize=13)
    ax.set_xticks(x, [lab for _, lab in TERMS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of pairs where the\nsubject circuit scores higher")
    fig.legend(loc="lower center", ncol=1, frameon=False, fontsize=13)
    figure.clean(ax)
    n = res["n"][0][3]
    ax.set_title(f"How often each term scores the subject circuit above the attractor circuit ({n} agreement pairs)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
