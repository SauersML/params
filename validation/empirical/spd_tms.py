"""Empirical bridge to `onehot_identifiability`: run the SPD reference code on TMS.

The Lean theorem says: rank-one subcomponents + exact faithfulness + exact one-hot causal
separation ⇒ the decomposition is the column decomposition `colComp W j`.  The SPD paper
*optimises* towards those hypotheses (faithfulness loss → 0, importance penalty → one
important subcomponent per feature).  This script runs the reference optimiser
(`spd.run_spd.optimize`, unmodified) on a freshly trained TMS 5-2 target and then
measures, on one-hot inputs, how close the learned decomposition is to the hypotheses and
to the conclusion:

  faith        ‖W − Σ_c P_c‖_F / ‖W‖_F                    (hypothesis 1; should → 0)
  sep_frac     fraction of features with exactly one subcomponent of importance g > 0.5
               on the one-hot input e_j                     (hypothesis 3)
  mmcs, ml2r   mean max cosine similarity / L2 ratio of the learned subcomponent columns
               against the columns of W                     (conclusion: colComp)

Run on a machine with the reference repo installed:
    python3 validation/empirical/spd_tms.py --spd-repo PATH --steps 4000 --seeds 3
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance  # noqa: E402


def run_one(spd_repo: Path, seed: int, steps: int, n_features: int, n_hidden: int, C: int, threads: int):
    sys.path.insert(0, str(spd_repo))
    torch.set_num_threads(threads)
    from spd.configs import Config, TMSTaskConfig
    from spd.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
    from spd.experiments.tms.models import TMSModel, TMSModelConfig
    from spd.experiments.tms.train_tms import train
    from spd.models.component_utils import calc_causal_importances
    from spd.run_spd import optimize
    from spd.utils import set_seed

    set_seed(seed)
    device = "cpu"
    # 1. target model (TMS 5-2, tied weights, no hidden identity layer)
    tcfg = TMSModelConfig(n_features=n_features, n_hidden=n_hidden, n_hidden_layers=0,
                          tied_weights=True, init_bias_to_zero=False, device=device)
    target = TMSModel(tcfg).to(device)
    ds = SparseFeatureDataset(n_features=n_features, feature_probability=0.05, device=device,
                              data_generation_type="at_least_zero_active", value_range=(0.0, 1.0))
    dl = DatasetGeneratedDataLoader(ds, batch_size=1024, shuffle=False)
    train(target, dl, log_wandb=False, importance=1.0, steps=3000, print_freq=100000, lr=5e-3,
          lr_schedule="cosine")
    target.eval()
    W = target.linear1.weight.detach().numpy().copy()          # (n_hidden, n_features)
    for p in target.parameters():
        p.requires_grad = False

    # 2. SPD with the paper's TMS 5-2 config (steps shortened), reference optimiser unmodified
    cfg = Config(
        wandb_project=None, seed=seed, C=C, n_mask_samples=1, n_ci_mlp_neurons=16,
        target_module_patterns=["linear1", "linear2"],
        faithfulness_coeff=1.0, recon_coeff=None, stochastic_recon_coeff=1.0,
        recon_layerwise_coeff=None, stochastic_recon_layerwise_coeff=1.0,
        importance_minimality_coeff=3e-3, pnorm=1.0, output_loss_type="mse",
        batch_size=4096, steps=steps, lr=1e-3, lr_schedule="cosine", lr_warmup_pct=0.0,
        n_eval_steps=10, image_freq=None, print_freq=max(steps // 4, 1), save_freq=steps,
        pretrained_model_class="spd.experiments.tms.models.TMSModel", pretrained_model_path=None,
        task_config=TMSTaskConfig(task_name="tms", feature_probability=0.05,
                                  data_generation_type="at_least_zero_active"),
    )
    spd_ds = SparseFeatureDataset(n_features=n_features, feature_probability=0.05, device=device,
                                  data_generation_type="at_least_zero_active", value_range=(0.0, 1.0))
    train_loader = DatasetGeneratedDataLoader(spd_ds, batch_size=cfg.batch_size, shuffle=False)
    out_dir = Path(tempfile.mkdtemp(prefix="spd_tms_", dir=str(Path(__file__).resolve().parent / "results")))
    optimize(target_model=target, config=cfg, device=device, train_loader=train_loader,
             eval_loader=train_loader, n_eval_steps=cfg.n_eval_steps, out_dir=out_dir,
             plot_results_fn=None, tied_weights=[("linear1", "linear2")])
    ckpts = sorted(out_dir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    sd = torch.load(ckpts[-1], map_location="cpu")
    A = sd["components.linear1.A"].numpy()    # (n_features, C)  == Lean Vᵀ
    B = sd["components.linear1.B"].numpy()    # (C, n_hidden)    == Lean Uᵀ

    # 3. hypotheses and conclusion, measured
    P = [np.outer(B[c], A[:, c]) for c in range(C)]           # subcomponent c, (n_hidden, n_features)
    faith = np.linalg.norm(W - sum(P)) / np.linalg.norm(W)
    # causal importances on one-hot inputs, through the reference gate code
    gates = {"linear1": None}
    from spd.models.components import GateMLP
    gate = GateMLP(C=C, n_ci_mlp_neurons=16)
    gate.load_state_dict({k.removeprefix("gates.linear1."): v for k, v in sd.items()
                          if k.startswith("gates.linear1.")})
    As = {"linear1": torch.tensor(A)}
    eye = torch.eye(n_features)
    ci, _ = calc_causal_importances(pre_weight_acts={"linear1": eye}, As=As, gates={"linear1": gate},
                                    detach_inputs=True)
    g = ci["linear1"].detach().numpy()                        # (n_features, C)
    n_important = (g > 0.5).sum(axis=1)
    sep_frac = float((n_important == 1).mean())
    # MMCS / ML2R (SPD paper eqs. (mmcs), (ml2r)) against the columns of W
    mmcs, ml2r = [], []
    for j in range(n_features):
        cols = np.stack([P[c][:, j] for c in range(C)])
        norms = np.linalg.norm(cols, axis=1) * np.linalg.norm(W[:, j])
        cos = np.where(norms > 0, cols @ W[:, j] / np.where(norms > 0, norms, 1), 0.0)
        c_star = int(np.argmax(cos))
        mmcs.append(float(cos[c_star]))
        ml2r.append(float(np.linalg.norm(cols[c_star]) / np.linalg.norm(W[:, j])))
    return {"seed": seed, "faith": float(faith), "sep_frac": sep_frac, "mmcs": float(np.mean(mmcs)),
            "ml2r": float(np.mean(ml2r)), "n_important_per_feature": n_important.tolist(),
            "n_alive": int((g.max(axis=0) > 0.5).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spd-repo", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--n-hidden", type=int, default=2)
    ap.add_argument("--C", type=int, default=20)
    ap.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "spd_tms.json"))
    args = ap.parse_args()
    records = [run_one(Path(args.spd_repo), s, args.steps, args.n_features, args.n_hidden, args.C, args.threads)
               for s in range(args.first_seed, args.first_seed + args.seeds)]
    cells = [{"metric": k, **provenance.summarize([r[k] for r in records])}
             for k in ("faith", "sep_frac", "mmcs", "ml2r", "n_alive")]
    for c in cells:
        print(f"{c['metric']:<10} mean={c['mean']:.4f} sd={c['sd']} min={c['min']:.4f} max={c['max']:.4f}")
    provenance.write(args.output, "spd_tms.py", vars(args), args.first_seed, args.seeds, cells, records)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
