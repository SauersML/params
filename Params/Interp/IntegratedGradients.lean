import Mathlib

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# Integrated gradients are complete

Attribution methods estimate how much each input coordinate contributes to a
scalar read-out `F`.  Gradient attributions are first-order and can miss the
effect of a finite change; *integrated gradients* (Sundararajan–Taly–Yan 2017)
integrate the gradient along the straight path from a baseline `x'` to `x`,
and satisfy *completeness*: the attributions sum exactly to `F x − F x'`.

`path_integral_complete` is the underlying fundamental-theorem-of-calculus
statement for a `C¹` read-out on `ι → ℝ`; `integratedGradients_complete` is
the per-coordinate form.  Both are proved from Mathlib's calculus library
alone; the only hypothesis is that `F` is continuously differentiable, which
is the hypothesis under which the method is defined.  This is the exact
content of the "first-order approximation" caveat in Q9 of *Open Problems in
Mechanistic Interpretability*: integrating the first-order quantity along a
path removes the approximation entirely.
-/

namespace Params.Interp

noncomputable section

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The straight path from `x'` to `x`. -/
def path (x x' : ι → ℝ) (t : ℝ) : ι → ℝ := x' + t • (x - x')

lemma hasDerivAt_path (x x' : ι → ℝ) (t : ℝ) : HasDerivAt (path x x') (x - x') t := by
  unfold path
  simpa using ((hasDerivAt_id t).smul_const (x - x')).const_add x'

lemma continuous_path (x x' : ι → ℝ) : Continuous (path x x') := by
  unfold path
  fun_prop

/-- The directional derivative of `F` along the path, as a function of `t`. -/
def pathDeriv (F : (ι → ℝ) → ℝ) (x x' : ι → ℝ) (t : ℝ) : ℝ :=
  fderiv ℝ F (path x x' t) (x - x')

lemma hasDerivAt_comp_path (F : (ι → ℝ) → ℝ) (hF : ContDiff ℝ 1 F) (x x' : ι → ℝ) (t : ℝ) :
    HasDerivAt (fun t => F (path x x' t)) (pathDeriv F x x' t) t :=
  ((hF.differentiable le_rfl) (path x x' t)).hasFDerivAt.comp_hasDerivAt t (hasDerivAt_path x x' t)

lemma continuous_pathDeriv (F : (ι → ℝ) → ℝ) (hF : ContDiff ℝ 1 F) (x x' : ι → ℝ) :
    Continuous (pathDeriv F x x') :=
  ((hF.continuous_fderiv le_rfl).comp (continuous_path x x')).clm_apply continuous_const

/-- **Completeness of the path integral of the gradient** (fundamental theorem of calculus). -/
theorem path_integral_complete (F : (ι → ℝ) → ℝ) (hF : ContDiff ℝ 1 F) (x x' : ι → ℝ) :
    ∫ t in (0 : ℝ)..1, pathDeriv F x x' t = F x - F x' := by
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun t _ => hasDerivAt_comp_path F hF x x' t)
    (continuous_pathDeriv F hF x x').intervalIntegrable
  rw [h]
  simp [path]

/-- Integrated gradient of coordinate `i`. -/
def integratedGradient (F : (ι → ℝ) → ℝ) (x x' : ι → ℝ) (i : ι) : ℝ :=
  (x i - x' i) * ∫ t in (0 : ℝ)..1, fderiv ℝ F (path x x' t) (Pi.single i 1)

lemma continuous_coord_deriv (F : (ι → ℝ) → ℝ) (hF : ContDiff ℝ 1 F) (x x' : ι → ℝ) (i : ι) :
    Continuous (fun t => fderiv ℝ F (path x x' t) (Pi.single i 1)) :=
  ((hF.continuous_fderiv le_rfl).comp (continuous_path x x')).clm_apply continuous_const

/-- The directional derivative along the path is the weighted sum of coordinate derivatives. -/
lemma pathDeriv_eq_sum (F : (ι → ℝ) → ℝ) (x x' : ι → ℝ) (t : ℝ) :
    pathDeriv F x x' t = ∑ i, (x i - x' i) * fderiv ℝ F (path x x' t) (Pi.single i 1) := by
  unfold pathDeriv
  have h1 : ∀ i, (x i - x' i) • (Pi.single i (1 : ℝ) : ι → ℝ) = Pi.single i ((x - x') i) := by
    intro i
    rw [← Pi.single_smul, smul_eq_mul, mul_one, Pi.sub_apply]
  have hx : x - x' = ∑ i, (x i - x' i) • (Pi.single i (1 : ℝ) : ι → ℝ) := by
    simp only [h1]
    exact (Finset.univ_sum_single (x - x')).symm
  conv_lhs => rw [hx]
  rw [map_sum]
  refine Finset.sum_congr rfl ?_
  intro i _
  rw [map_smul, smul_eq_mul]

/-- **Completeness of integrated gradients**: the attributions sum to `F x − F x'`. -/
theorem integratedGradients_complete (F : (ι → ℝ) → ℝ) (hF : ContDiff ℝ 1 F) (x x' : ι → ℝ) :
    ∑ i, integratedGradient F x x' i = F x - F x' := by
  rw [← path_integral_complete F hF x x']
  have hint : ∀ i ∈ (Finset.univ : Finset ι), IntervalIntegrable
      (fun t => (x i - x' i) * fderiv ℝ F (path x x' t) (Pi.single i 1)) MeasureTheory.volume 0 1 :=
    fun i _ => (continuous_const.mul (continuous_coord_deriv F hF x x' i)).intervalIntegrable
  calc ∑ i, integratedGradient F x x' i
      = ∑ i, ∫ t in (0 : ℝ)..1, (x i - x' i) * fderiv ℝ F (path x x' t) (Pi.single i 1) := by
        refine Finset.sum_congr rfl ?_
        intro i _
        unfold integratedGradient
        rw [intervalIntegral.integral_const_mul]
    _ = ∫ t in (0 : ℝ)..1, ∑ i, (x i - x' i) * fderiv ℝ F (path x x' t) (Pi.single i 1) :=
        (intervalIntegral.integral_finset_sum hint).symm
    _ = ∫ t in (0 : ℝ)..1, pathDeriv F x x' t := by
        refine intervalIntegral.integral_congr ?_
        intro t _
        exact (pathDeriv_eq_sum F x x' t).symm

end

end Params.Interp
