# params

A Lean 4 / Mathlib formalization of the Transformer, of decoder-only language
models at the weight level, and of linear parameter decomposition (APD / SPD),
with the empirical validation discipline of [Descent](https://github.com/SauersML/Descent):
every definition is transcribed literally to NumPy and checked against
independent reference implementations, and every theorem is exercised
numerically with provenance-stamped results.

## Policy

* **No assumptions.** No `axiom`, no `sorry`, no `native_decide`, no
  hypotheses that merely restate a paper's claim.  Every theorem is proved from
  Mathlib.  `validation/code/check.py` scans the source for admissions;
  `validation/code/Check.lean` scans the elaborated environment for any axiom
  beyond `propext`, `Classical.choice`, `Quot.sound`.
* `autoImplicit = false`, `relaxedAutoImplicit = false`, warnings are errors.
* Builds happen on MSI only, at most once per hour, against a warm Mathlib
  cache (`scripts/msi-build.sh`).

## Layout

| Module | Content |
|---|---|
| `Params/Transformer/Softmax.lean` | softmax: positivity, normalisation, shift invariance, permutation covariance |
| `Params/Transformer/LayerNorm.lean` | **layer normalisation is the Euclidean projection onto `{mean = σ₁, var = σ₂²}`** (finite form of Tai–Liu–Li–Chan Thm 3.1); scale invariance; ReLU is the projection onto the non-negative orthant |
| `Params/Transformer/RMSNorm.lean` | RMSNorm (Llama) is the projection onto the unit-RMS sphere; scale invariance |
| `Params/Transformer/Attention.lean` | scaled dot-product attention; outputs lie in the convex hull of `V`; **permutation equivariance**; multi-head = Σ heads · `W^O` blocks |
| `Params/Transformer/Circuits.lean` | QK/OV circuits: attention pattern depends on `W^Q W^Kᵀ` only (gauge freedom `R, R⁻ᵀ`), head output on `W^V W^O` only |
| `Params/Transformer/Encoder.lean` | post-norm encoder block; the six operator-splitting substeps of arXiv:2510.03989; **`splitStep = encoderBlock` exactly** (with `W_j ↦ I + W_j`), and for stacks |
| `Params/Transformer/Causal.lean` | masked softmax, causal attention, `PrefixDependent` and its closure properties |
| `Params/Transformer/LanguageModel.lean` | decoder-only LM; **`nextTokenDist_prefix`: predictions at position `i` depend only on tokens `≤ i`**; exact direct-logit-attribution identity for pre-norm residual streams |
| `Params/Decomp/Basic.lean` | SPD subcomponents `U diag(m) V`, masks `g + (1−g) r`, hard/leaky sigmoids as in the reference code, faithfulness and importance losses, ablation identity |
| `Params/Decomp/Rank.lean` | rank subadditivity; a faithful rank-one decomposition of `W` needs `≥ rank W` pieces; a hidden identity needs `≥ k` |
| `Params/Decomp/Attribution.lean` | ablation invariance of the column decomposition (APD's ablation-curve criterion, exact for linear maps); attributions are additive; APD's rank surrogate `= ‖U‖‖V‖` |
| `Params/Decomp/Targeted.lean` | targeted PD (arXiv:2607.13047): catch-all reconstruction identities, existence of adversarial masks, underdetermination off the target subspace, merging of co-active subcomponents, non-uniqueness of rank-one splittings |
| `Params/Decomp/L3D.lean` | L3D (arXiv:2504.00194): top-k projection, exact recovery under `V^in V^out = I`, scale invariance of the normalised loss, sign ambiguity, impact = directional derivative |
| `Params/Decomp/Identifiability.lean` | **`onehot_identifiability`**: with rank-one subcomponents, exact faithfulness and exact one-hot causal separation, a matrix with non-zero pairwise-independent columns has *only* the column decomposition (the TMS ground truth of APD/SPD); hypotheses shown satisfiable |
| `Params/Interp/Superposition.lean` | interference bound for almost-orthogonal features: read-out error ≤ `ε (k−1) ‖s‖∞`; orthonormal ⇒ exact and `N ≤ d` |
| `Params/Interp/Spark.lean` | uniqueness of sparse codes below the spark bound (when SDL *can* recover true features) |
| `Params/Interp/IntegratedGradients.lean` | completeness of integrated gradients for `C¹` read-outs (FTC along the straight path) |
| `Params/Interp/Absorption.lean` | `L¹` strictly prefers the absorbed dictionary once features co-occur (sparsity ≠ recovery) |

## Validation

```
python3 validation/code/check.py                          # source-text guards, no build needed
lake env lean validation/code/Check.lean                   # axiom closure (after a build)
python3 validation/empirical/run.py --spd-repo PATH        # NumPy transcriptions vs PyTorch and the SPD code;
                                                           # every theorem checked on random inputs
```

Results land in `validation/empirical/results/*.json` with revision, seed,
replicate count and per-cell error bars, and in `validation/code/results/`.

## Empirical bridge to `onehot_identifiability`

`validation/empirical/spd_tms.py` runs the unmodified SPD reference optimiser on a
TMS 5-2 target (3 seeds, 40k steps) and `spd_tms_analyze.py` measures, per feature
`j` on the one-hot input `e_j`, how far the learned decomposition is from the
theorem's hypotheses and conclusion (`validation/empirical/results/spd_tms_analysis_40k.json`):

| feature rows (3 seeds × 5 features) | active_cos | active_l2r | inactive_ratio |
|---|---|---|---|
| exactly one important subcomponent (12 rows) | 1.000 | 1.03 – 1.10 | 0.03 – 0.10 |
| no important subcomponent (8 rows) | −0.99 … 1.00 | 0.01 – 3.04 | ≈ 1.0 |

Where the causal-separation hypothesis holds (one important subcomponent, the others
nearly ablatable), the important subcomponent's column is the column of `W` to within
a few percent — the theorem's conclusion.  Where it fails (the gate never marks a
subcomponent important for the feature) the decomposition is not the column one for
that feature, and faithfulness alone (`‖W − ΣP_c‖/‖W‖ ≈ 3·10⁻⁴` in every run) does not
rescue it.  One of three seeds collapsed to a single alive subcomponent; the paper
also reports seed sensitivity.  Numbers are measured, not fitted.

## Sources formalised

* Tai, Liu, Li, Chan — *A Mathematical Explanation of Transformers* (arXiv:2510.03989)
* Bushnaq, Braun, Sharkey — *Stochastic Parameter Decomposition* (arXiv:2506.20790); code `goodfire-ai/param-decomp@spd-paper`
* Braun et al. — *Interpretability in Parameter Space* (APD, arXiv:2501.14926)
* Vigouroux, Sharkey — *Targeted Recovery of Weight-Space Mechanisms* (tPD, arXiv:2607.13047)
* Chrisman, Bushnaq, Sharkey — *Local Loss Landscape Decomposition* (L3D, arXiv:2504.00194)
* Sharkey et al. — *Open Problems in Mechanistic Interpretability* (arXiv:2501.16496), Q3/Q4 mathematical cores
