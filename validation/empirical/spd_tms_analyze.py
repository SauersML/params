"""Measure the hypotheses and conclusion of `onehot_identifiability` on a trained SPD checkpoint.

For every feature `j` (one-hot input `e_j`) with `W` the target `linear1` weight and
`P_c = U_{:,c} V_{c,:}` the learned subcomponents:

  n_important_j   number of subcomponents with causal importance g_c(e_j) > 0.5
                  (hypothesis 3 asks for exactly one)
  active_cos_j    cosine between column j of the most important subcomponent and W[:, j]
  active_l2r_j    ‖P_{c*}[:, j]‖ / ‖W[:, j]‖ for that subcomponent  (conclusion: colComp ⇒ 1)
  inactive_ratio_j  ‖Σ_{c not important} P_c[:, j]‖ / ‖W[:, j]‖
                  (hypothesis 3, quantified: unimportant subcomponents send e_j to 0)
  faith           ‖W − Σ_c P_c‖_F / ‖W‖_F                        (hypothesis 1)

Usage:  python3 spd_tms_analyze.py --spd-repo PATH CKPT_DIR [CKPT_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance  # noqa: E402


def analyze_checkpoint(spd_repo: Path, ckpt: Path, layer: str = "linear1", n_ci_mlp_neurons: int = 16):
    sys.path.insert(0, str(spd_repo))
    from spd.models.component_utils import calc_causal_importances
    from spd.models.components import Gate, GateMLP

    sd = torch.load(ckpt, map_location="cpu")
    W = sd[f"model.{layer}.weight"].numpy()                 # (d_out, d_in)
    A = sd[f"components.{layer}.A"].numpy()                 # (d_in, C)
    B = sd[f"components.{layer}.B"].numpy()                 # (C, d_out)
    C = A.shape[1]
    n_in = A.shape[0]
    gate_sd = {k.removeprefix(f"gates.{layer}."): v for k, v in sd.items() if k.startswith(f"gates.{layer}.")}
    gate = GateMLP(C=C, n_ci_mlp_neurons=n_ci_mlp_neurons) if "mlp_in" in gate_sd else Gate(C=C)
    gate.load_state_dict(gate_sd)
    ci, _ = calc_causal_importances(pre_weight_acts={layer: torch.eye(n_in)}, As={layer: torch.tensor(A)},
                                    gates={layer: gate}, detach_inputs=True)
    g = ci[layer].detach().numpy()                          # (n_in, C)
    P = [np.outer(B[c], A[:, c]) for c in range(C)]         # (d_out, d_in) each
    faith = float(np.linalg.norm(W - sum(P)) / np.linalg.norm(W))
    per = []
    for j in range(n_in):
        wj = W[:, j]; nw = np.linalg.norm(wj)
        imp = [c for c in range(C) if g[j, c] > 0.5]
        cstar = int(np.argmax(g[j]))
        col = P[cstar][:, j]
        cos = float(col @ wj / (np.linalg.norm(col) * nw)) if np.linalg.norm(col) > 0 else 0.0
        inactive = sum((P[c][:, j] for c in range(C) if c not in imp), np.zeros_like(wj))
        per.append({"feature": j, "n_important": len(imp), "g_max": float(g[j].max()),
                    "active_cos": cos, "active_l2r": float(np.linalg.norm(col) / nw),
                    "inactive_ratio": float(np.linalg.norm(inactive) / nw)})
    return {"ckpt": str(ckpt), "faith": faith,
            "sep_frac": float(np.mean([p["n_important"] == 1 for p in per])),
            "active_cos": float(np.mean([p["active_cos"] for p in per])),
            "active_l2r": float(np.mean([p["active_l2r"] for p in per])),
            "inactive_ratio": float(np.mean([p["inactive_ratio"] for p in per])),
            "n_alive": int((g.max(axis=0) > 0.5).sum()), "per_feature": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spd-repo", required=True)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "spd_tms_analysis.json"))
    args = ap.parse_args()
    records = []
    for d in args.dirs:
        ckpts = sorted(Path(d).glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            print(f"[SKIP] no checkpoint in {d}"); continue
        r = analyze_checkpoint(Path(args.spd_repo), ckpts[-1]); records.append(r)
        print(f"{Path(d).name}: faith={r['faith']:.4f} sep_frac={r['sep_frac']:.2f} active_cos={r['active_cos']:.4f} "
              f"active_l2r={r['active_l2r']:.4f} inactive_ratio={r['inactive_ratio']:.4f} n_alive={r['n_alive']}")
        for p in r["per_feature"]:
            print("   ", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}))
    keys = ("faith", "sep_frac", "active_cos", "active_l2r", "inactive_ratio", "n_alive")
    cells = [{"metric": k, **provenance.summarize([r[k] for r in records])} for k in keys]
    provenance.write(args.output, "spd_tms_analyze.py", vars(args), 0, len(records), cells, records)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
