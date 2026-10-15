"""A circuit builder scored on MIB's InterpBench IOI model (#2951), the external circuit-discovery benchmark: MIB's own
area under the ROC (interpbench.mib_auroc against the case's interpbench_graph.json, equal to MIB's code), each of the
graph's edges scored by the share of prompts whose circuit holds it, with a 95% bootstrap interval over prompts, and the
paired difference against EAP-IG-inputs on the same test prompts (MIB's code, per-prompt scores
~/mpd-data/interpbench/raw/mib_eap/scores_EAP-IG-inputs_test_<n>_1.npz), prompts resampled jointly.

Inputs, any mix:
  ROWS.jsonl     one prompt a line: {"text": k (its place in MIB's test split), "mib_edges": [MIB edge names]}
                 (ib_teacher.py's rows_<offset>.jsonl), for builders that name MIB edges themselves, or
                 {"text": k, "mib_nodes": [MIB components: "a<L>.h<H>", "m<L>"]} for builders in heads and MLPs: every
                 graph edge between two of them (the input and the logits always in) counts as held;
  MEASURED.npz   per-prompt edge scores {"effects" [prompts, edges], "edges"} on the test prompts 0..n-1 (mib_measured.py's
                 single-edge ablations), each edge scored by its mean absolute effect; scored alone;
  DIR            circuit files DIR/<case>_<k>.json (teacher format: "nodes" [site, component, position] in the VPD fit's
                 subcomponents), turned into MIB edges with interpbench.mib_edges on --fit, the k-th test prompt padded
                 with EOS to --pad and its target at its last real token, as ib_teacher.py runs them.

  python experiments/oracle/toys/mib_score.py OUT_DIR_OR_ROWS... [--fit ~/mpd-data/interpbench/fits_layers1x/mib.vpd.pt]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import interpbench  # noqa: E402

MIB = Path.home() / "mpd-data/interpbench"


def score(reference: dict, T: np.ndarray, x: np.ndarray | None, B: int = 1000, seed: int = 0) -> dict:
    """T [prompts, edges]: whether each prompt's circuit holds each graph edge (interpbench.mib_graph_edges order); x the
    same prompts' EAP-IG-inputs scores, or None. MIB's AUROC of the edge shares with a bootstrap interval, and the paired
    difference to EAP-IG-inputs."""
    edges = interpbench.mib_graph_edges(reference["cfg"])
    auc = lambda v: interpbench.mib_auroc(reference, dict(zip(edges, v.tolist())))["mib"]  # noqa: E731
    ci = lambda xs: [round(float(np.percentile(xs, 2.5)), 3), round(float(np.percentile(xs, 97.5)), 3)]  # noqa: E731
    rng, n = np.random.default_rng(seed), len(T)
    draws = [rng.integers(0, n, n) for _ in range(B)]
    a = [auc(T[i].mean(0)) for i in draws]
    out = {"prompts": n, "auroc": round(auc(T.mean(0)), 3), "ci95": ci(a),
           "plain_auroc": round(interpbench.mib_auroc(reference, dict(zip(edges, T.mean(0).tolist())))["plain"], 3),
           "edges_per_prompt": round(float(T.sum(1).mean()), 1)}
    if x is not None:
        b = [auc(x[i].mean(0)) for i in draws]
        d = np.array(a) - np.array(b)
        out["eap_ig_inputs"] = {"auroc": round(auc(x.mean(0)), 3), "ci95": ci(b)}
        out["diff_vs_eap_ig_inputs"] = {"diff": round(auc(T.mean(0)) - auc(x.mean(0)), 3), "ci95": ci(d),
                                        "share_above_0": round(float(np.mean(d > 0)), 3)}
    return out


def edges_of_nodes(nodes, edges: list[str]) -> list[str]:
    """The graph edges whose two ends are among MIB components nodes (the input and the logits always in): "a0.h1->m1",
    "input->a1.h0<q>" (a head's query, key or value input belongs to the head)."""
    keep = set(nodes) | {"input", "logits"}
    return [e for e in edges if all(re.sub(r"<[qkv]>$", "", x) in keep for x in e.split("->"))]


def rows_of_dir(d: Path, nat, pad: int) -> list[dict]:
    """MIB edges of the circuit files in d, one row per test prompt k (<case>_<k>.json)."""
    texts = interpbench.mib_ioi("test")
    rows = []
    for f in sorted(d.iterdir()):
        m = re.fullmatch(r"[A-Za-z0-9]+_(\d+)\.json", f.name)
        if not m:
            continue
        k = int(m.group(1))
        target = len(texts[k]["ids"]) - 1
        assert len(texts[k]["ids"]) <= pad, f"test prompt {k} is longer than --pad {pad}"
        circ = json.loads(f.read_text())
        rows.append({"text": k, "mib_edges": sorted(interpbench.mib_edges(nat, circ, [target]))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--case", type=Path, default=MIB / "raw/mib")
    ap.add_argument("--fit", type=Path, default=MIB / "fits_layers1x/mib.vpd.pt")
    ap.add_argument("--pad", type=int, default=28)
    ap.add_argument("--eap", type=Path, default=MIB / "raw/mib_eap")
    a = ap.parse_args()
    reference = json.loads((a.case / "interpbench_graph.json").read_text())
    edges = interpbench.mib_graph_edges(reference["cfg"])
    rows, nat, T = [], None, None
    for p in a.inputs:
        if p.suffix == ".npz":                             # mib_measured.py's per-prompt edge effects, MIB's test prompts 0..n-1
            d = np.load(p)
            at = {str(e): j for j, e in enumerate(d["edges"])}
            T = np.abs(d["effects"][:, [at[e] for e in edges]].astype(np.float64))
            continue
        if p.is_dir():
            if nat is None:
                import torch

                sd, cfg, _ = interpbench.load(a.case)
                nat = interpbench.native_of_fit(sd, cfg, torch.load(a.fit, map_location="cpu", weights_only=True))
            rows += rows_of_dir(p, nat, a.pad)
        else:
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "mib_edges" not in r:
                        r["mib_edges"] = edges_of_nodes(r["mib_nodes"], edges)
                    rows.append(r)
    if T is not None:
        assert not rows, "an .npz of measured effects is scored alone"
        texts = list(range(len(T)))
    else:
        rows = sorted({r["text"]: r for r in rows}.values(), key=lambda r: r["text"])
        texts = [r["text"] for r in rows]
        T = np.array([[e in set(r["mib_edges"]) for e in edges] for r in rows], dtype=np.float64)
    x = None
    f = a.eap / f"scores_EAP-IG-inputs_test_{max(texts) + 1}_1.npz"
    if f.exists() and texts == list(range(len(texts))):
        x = np.load(f)["scores"].astype(np.float64)
        assert len(x) == len(T), f"{f.name} holds {len(x)} prompts, the builder {len(T)}"
    print(json.dumps(score(reference, T, x), indent=1))


if __name__ == "__main__":
    main()
