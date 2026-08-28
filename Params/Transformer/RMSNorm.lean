import Mathlib
import Params.Transformer.LayerNorm

/-!
# RMS normalisation as a Euclidean projection

Llama-class language models replace layer normalisation by RMSNorm
(Zhang–Sennrich 2019): `v ↦ v / sqrt(mean v²)`.  The same projection picture
holds: `rmsNorm v` is the point of the "unit-RMS sphere"
`{u | mean u² = 1}` closest to `v` (`rmsNorm_sqDist_le`), and RMSNorm is
invariant under positive rescaling (`rmsNorm_smul`).  Proofs are the
Cauchy–Schwarz argument of `layerNorm_sqDist_le` without the centring step.
-/

namespace Params

noncomputable section

open Finset

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- Mean square of a vector. -/
def meanSq (v : ι → ℝ) : ℝ := mean (fun j => v j ^ 2)

/-- RMS normalisation. -/
def rmsNorm (v : ι → ℝ) : ι → ℝ := fun i => v i / Real.sqrt (meanSq v)

/-- The unit-RMS sphere. -/
def UnitRmsSet : Set (ι → ℝ) := {u | meanSq u = 1}

lemma meanSq_nonneg (v : ι → ℝ) : 0 ≤ meanSq v := by
  unfold meanSq mean
  exact div_nonneg (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) (card_pos (ι := ι)).le

lemma sum_sq_eq_card_mul_meanSq (v : ι → ℝ) :
    ∑ i, v i ^ 2 = Fintype.card ι * meanSq v :=
  (card_mul_mean (fun j => v j ^ 2)).symm

/-- RMSNorm lands on the unit-RMS sphere (when `v ≠ 0`). -/
theorem rmsNorm_mem (v : ι → ℝ) (hv : meanSq v ≠ 0) : rmsNorm v ∈ UnitRmsSet := by
  have hρ2 : 0 < meanSq v := lt_of_le_of_ne (meanSq_nonneg v) (Ne.symm hv)
  have hρsq : Real.sqrt (meanSq v) ^ 2 = meanSq v := Real.sq_sqrt hρ2.le
  show meanSq (rmsNorm v) = 1
  unfold meanSq
  rw [mean]
  simp only [rmsNorm, div_pow, hρsq]
  rw [← Finset.sum_div, sum_sq_eq_card_mul_meanSq v, mul_div_assoc, div_self hv, mul_one,
    div_self (card_pos (ι := ι)).ne']

/-- **RMSNorm is the projection onto the unit-RMS sphere.** -/
theorem rmsNorm_sqDist_le (v u : ι → ℝ) (hv : meanSq v ≠ 0) (hu : u ∈ UnitRmsSet) :
    sqDist (rmsNorm v) v ≤ sqDist u v := by
  have hn : (0 : ℝ) < Fintype.card ι := card_pos
  have hρ2 : 0 < meanSq v := lt_of_le_of_ne (meanSq_nonneg v) (Ne.symm hv)
  have hρ : 0 < Real.sqrt (meanSq v) := Real.sqrt_pos.mpr hρ2
  have hρsq : Real.sqrt (meanSq v) ^ 2 = meanSq v := Real.sq_sqrt hρ2.le
  have hv2 : ∑ i, v i ^ 2 = Fintype.card ι * Real.sqrt (meanSq v) ^ 2 := by
    rw [hρsq]
    exact sum_sq_eq_card_mul_meanSq v
  have hu2 : ∑ i, u i ^ 2 = Fintype.card ι := by
    rw [sum_sq_eq_card_mul_meanSq u, hu, mul_one]
  have hcs : ∑ i, u i * v i ≤ Fintype.card ι * Real.sqrt (meanSq v) := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ u v
    rw [hu2, hv2] at h
    refine le_of_sq_le_sq' ?_ (mul_nonneg hn.le hρ.le)
    calc (∑ i, u i * v i) ^ 2
        ≤ (Fintype.card ι : ℝ) * (Fintype.card ι * Real.sqrt (meanSq v) ^ 2) := h
      _ = (Fintype.card ι * Real.sqrt (meanSq v)) ^ 2 := by ring
  have hstar : sqDist (rmsNorm v) v
      = (1 / Real.sqrt (meanSq v) - 1) ^ 2 * ∑ i, v i ^ 2 := by
    unfold sqDist rmsNorm
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  have hu' : sqDist u v = ∑ i, u i ^ 2 - 2 * ∑ i, u i * v i + ∑ i, v i ^ 2 := by
    unfold sqDist
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  have hmul : (1 / Real.sqrt (meanSq v) - 1) * Real.sqrt (meanSq v) = 1 - Real.sqrt (meanSq v) := by
    rw [sub_mul, one_div, inv_mul_cancel₀ hρ.ne', one_mul]
  have hkey : (1 / Real.sqrt (meanSq v) - 1) ^ 2 * (Fintype.card ι * Real.sqrt (meanSq v) ^ 2)
      = Fintype.card ι * (1 - Real.sqrt (meanSq v)) ^ 2 := by
    calc (1 / Real.sqrt (meanSq v) - 1) ^ 2 * (Fintype.card ι * Real.sqrt (meanSq v) ^ 2)
        = Fintype.card ι * ((1 / Real.sqrt (meanSq v) - 1) * Real.sqrt (meanSq v)) ^ 2 := by ring
      _ = Fintype.card ι * (1 - Real.sqrt (meanSq v)) ^ 2 := by rw [hmul]
  rw [hstar, hu', hu2, hv2, hkey]
  nlinarith [hcs, hn]

lemma meanSq_smul (c : ℝ) (v : ι → ℝ) : meanSq (fun i => c * v i) = c ^ 2 * meanSq v := by
  unfold meanSq
  have h : (fun j => (c * v j) ^ 2) = fun j => c ^ 2 * v j ^ 2 := by
    funext j
    ring
  rw [h, mean_smul]

/-- RMSNorm ignores a positive rescaling of its input. -/
theorem rmsNorm_smul (c : ℝ) (hc : 0 < c) (v : ι → ℝ) :
    rmsNorm (fun i => c * v i) = rmsNorm v := by
  funext i
  simp only [rmsNorm]
  rw [meanSq_smul, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le,
    mul_div_mul_left _ _ hc.ne']

end

end Params
