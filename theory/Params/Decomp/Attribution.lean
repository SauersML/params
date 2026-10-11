import Mathlib
import Params.Decomp.Identifiability

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# Ablation invariance, attributions and the rank surrogate of APD

Three small facts from the APD paper (arXiv:2501.14926) made exact.

* **Ablation invariance of the ground truth.**  APD's stricter definition of a
  parameter decomposition asks that scaling the *inactive* components by any
  factors (an "ablation curve") leave the output unchanged.  For the linear map
  `x ↦ W x` and the column decomposition, this holds exactly: as long as the
  components of the features present in `x` are kept at `1`, every other
  component may be scaled arbitrarily (`colComp_ablation_invariant`).  This is
  the "assumption of parameter linearity" verified for the TMS ground truth.
* **Attributions are additive.**  The gradient attribution `⟨∇_θ f, P_c⟩` is
  linear in `P_c`; hence the component attributions of a faithful decomposition
  sum to the attribution of the whole weight (`attribution_sum`).
* **The factorised rank surrogate.**  APD replaces the singular values of a
  factorised component `Σ_m U_m V_mᵀ` by `(Σ_{i,j} U_{m,i}² V_{m,j}²)^{1/2}`; this
  is exactly `‖U_m‖₂ ‖V_m‖₂` (`schatten_surrogate`).
-/

namespace Params.Decomp

noncomputable section

open Finset

section Ablation

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- **Ablation invariance of the column decomposition.**  If every feature present in
`x` keeps its component at scale `1`, scaling the other components by arbitrary `γ`
does not change `W x`. -/
theorem colComp_ablation_invariant (W : Matrix m n ℝ) (x : n → ℝ) (γ : n → ℝ)
    (hγ : ∀ c, x c ≠ 0 → γ c = 1) :
    (∑ c, γ c • colComp W c).mulVec x = W.mulVec x := by
  ext i
  simp only [Matrix.mulVec, dotProduct, Matrix.sum_apply, Matrix.smul_apply, colComp_apply,
    smul_eq_mul, mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  refine Finset.sum_congr rfl ?_
  intro j _
  by_cases hx : x j = 0
  · rw [hx, mul_zero, mul_zero]
  · rw [hγ j hx, one_mul]

/-- In particular, ablating (scale `0`) every component outside the support of `x` is
invisible to the linear map. -/
theorem colComp_ablate_inactive (W : Matrix m n ℝ) (x : n → ℝ) :
    (∑ c, (if x c = 0 then (0 : ℝ) else 1) • colComp W c).mulVec x = W.mulVec x := by
  apply colComp_ablation_invariant
  intro c hc
  rw [if_neg hc]

end Ablation

section Attribution

variable {m n C : Type*} [Fintype m] [Fintype n] [Fintype C]

/-- Gradient attribution of a parameter direction `P`: the Frobenius pairing with the
gradient `grad = ∇_θ f_o(x, θ*)` (one output coordinate). -/
def attribution (grad P : Matrix m n ℝ) : ℝ := ∑ i, ∑ j, grad i j * P i j

/-- Attribution is additive in the parameter direction. -/
theorem attribution_add (grad P Q : Matrix m n ℝ) :
    attribution grad (P + Q) = attribution grad P + attribution grad Q := by
  unfold attribution
  simp only [Matrix.add_apply, mul_add, Finset.sum_add_distrib]

/-- Attribution is homogeneous in the parameter direction. -/
theorem attribution_smul (grad P : Matrix m n ℝ) (a : ℝ) :
    attribution grad (a • P) = a * attribution grad P := by
  unfold attribution
  simp only [Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro i _
  refine Finset.sum_congr rfl ?_
  intro j _
  ring

/-- **Attributions of a faithful decomposition sum to the attribution of the weight.** -/
theorem attribution_sum (grad : Matrix m n ℝ) (P : C → Matrix m n ℝ) :
    ∑ c, attribution grad (P c) = attribution grad (∑ c, P c) := by
  unfold attribution
  simp only [Matrix.sum_apply, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl ?_
  intro i _
  rw [Finset.sum_comm]

/-- The first-order (attribution) estimate of removing `P` from a *linear* read-out is exact:
for `f(θ) = ⟨G, θ⟩`, `f(θ) − f(θ − P) = attribution G P`. -/
theorem attribution_exact_linear (G θ P : Matrix m n ℝ) :
    attribution G θ - attribution G (θ - P) = attribution G P := by
  unfold attribution
  simp only [Matrix.sub_apply, mul_sub, Finset.sum_sub_distrib]
  ring

end Attribution

section Schatten

variable {m n : Type*} [Fintype m] [Fintype n]

/-- Euclidean norm of a vector, written out. -/
def vnorm (u : m → ℝ) : ℝ := Real.sqrt (∑ i, u i ^ 2)

/-- APD's factorised singular-value surrogate equals the product of the factor norms. -/
theorem schatten_surrogate (u : m → ℝ) (v : n → ℝ) :
    Real.sqrt (∑ i, ∑ j, u i ^ 2 * v j ^ 2) = vnorm u * vnorm v := by
  unfold vnorm
  rw [← Real.sqrt_mul (Finset.sum_nonneg (fun _ _ => sq_nonneg _)), Finset.sum_mul_sum]

end Schatten

end

end Params.Decomp
