import Mathlib

/-!
# Layer normalization as a Euclidean projection

Tai–Liu–Li–Chan (arXiv:2510.03989, Theorem 3.1) characterise layer
normalisation as the projection of a token vector onto the set of vectors with
prescribed mean `σ₁` and variance `σ₂²`.  This file proves the finite-dimensional
version that the discrete Transformer actually uses: for `v : ι → ℝ` with
non-zero variance and `0 ≤ σ₂`,

  `layerNorm σ₁ σ₂ v = σ₂ • (v - mean v) / sqrt (variance v) + σ₁`

lies in `MeanVarSet σ₁ σ₂` and is at least as close to `v` (in squared
Euclidean distance) as every other member of that set.

Two further facts used by the operator-splitting recovery theorem
(`Params/Transformer/Encoder.lean`):

* `layerNorm_smul`: layer normalisation is invariant under positive rescaling
  of its input, which is why the relaxation step `½(u + u')` of the splitting
  scheme is invisible after the closing normalisation;
* `relu_isProj`: pointwise `ReLU` is the projection onto the non-negative
  orthant (the paper's `∂I_{S₂}` step, Section 3.3.3).

Every statement is proved from Mathlib alone; nothing is assumed.
-/

namespace Params

open Finset

noncomputable section

section Stats

variable {ι : Type*} [Fintype ι] [Nonempty ι]

lemma card_pos : (0 : ℝ) < Fintype.card ι := by
  exact_mod_cast (Fintype.card_pos : 0 < Fintype.card ι)

/-- Mean of a finite family of reals. -/
def mean (v : ι → ℝ) : ℝ := (∑ i, v i) / Fintype.card ι

/-- (Population) variance of a finite family of reals. -/
def variance (v : ι → ℝ) : ℝ :=
  (∑ i, (v i - mean v) ^ 2) / Fintype.card ι

/-- Layer normalisation with scalar targets: mean `σ₁`, standard deviation `σ₂`. -/
def layerNorm (σ₁ σ₂ : ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i => σ₂ * (v i - mean v) / Real.sqrt (variance v) + σ₁

/-- Layer normalisation with per-coordinate affine parameters `γ`, `β`
(Ba–Kiros–Hinton 2016). -/
def layerNormAffine (γ β : ι → ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i => γ i * ((v i - mean v) / Real.sqrt (variance v)) + β i

/-- Squared Euclidean distance. -/
def sqDist (u v : ι → ℝ) : ℝ := ∑ i, (u i - v i) ^ 2

/-- The constraint set `S₁(σ₁, σ₂)` of the paper: prescribed mean and variance. -/
def MeanVarSet (σ₁ σ₂ : ℝ) : Set (ι → ℝ) := {u | mean u = σ₁ ∧ variance u = σ₂ ^ 2}

lemma card_mul_mean (v : ι → ℝ) : (Fintype.card ι : ℝ) * mean v = ∑ i, v i := by
  unfold mean
  rw [← mul_div_assoc]
  exact mul_div_cancel_left₀ _ (card_pos (ι := ι)).ne'

lemma card_mul_variance (v : ι → ℝ) :
    (Fintype.card ι : ℝ) * variance v = ∑ i, (v i - mean v) ^ 2 := by
  unfold variance
  rw [← mul_div_assoc]
  exact mul_div_cancel_left₀ _ (card_pos (ι := ι)).ne'

lemma variance_nonneg (v : ι → ℝ) : 0 ≤ variance v := by
  unfold variance
  apply div_nonneg
  · exact Finset.sum_nonneg (fun i _ => sq_nonneg _)
  · exact (card_pos (ι := ι)).le

/-- Centred values sum to zero. -/
lemma sum_sub_mean (v : ι → ℝ) : ∑ i, (v i - mean v) = 0 := by
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
    card_mul_mean, sub_self]

/-- Expansion of a sum of squares of a sum. -/
lemma sum_sq_add (f g : ι → ℝ) :
    ∑ i, (f i + g i) ^ 2 = ∑ i, f i ^ 2 + 2 * ∑ i, f i * g i + ∑ i, g i ^ 2 := by
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl ?_
  intro i _
  ring

lemma le_of_sq_le_sq' {a b : ℝ} (h : a ^ 2 ≤ b ^ 2) (hb : 0 ≤ b) : a ≤ b := by
  calc a ≤ |a| := le_abs_self a
    _ = Real.sqrt (a ^ 2) := (Real.sqrt_sq_eq_abs a).symm
    _ ≤ Real.sqrt (b ^ 2) := Real.sqrt_le_sqrt h
    _ = b := Real.sqrt_sq hb

/-! ### Mean and variance of the normalised vector -/

theorem layerNorm_mean (σ₁ σ₂ : ℝ) (v : ι → ℝ) : mean (layerNorm σ₁ σ₂ v) = σ₁ := by
  rw [mean]
  simp only [layerNorm]
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
    ← Finset.sum_div, ← Finset.mul_sum, sum_sub_mean, mul_zero, zero_div, zero_add]
  exact mul_div_cancel_left₀ _ (card_pos (ι := ι)).ne'

theorem layerNorm_variance (σ₁ σ₂ : ℝ) (v : ι → ℝ) (hv : variance v ≠ 0) :
    variance (layerNorm σ₁ σ₂ v) = σ₂ ^ 2 := by
  have hβ : 0 < variance v := lt_of_le_of_ne (variance_nonneg v) (Ne.symm hv)
  have hr2 : Real.sqrt (variance v) ^ 2 = variance v := Real.sq_sqrt hβ.le
  rw [variance, layerNorm_mean]
  simp only [layerNorm]
  have h1 : ∀ i, (σ₂ * (v i - mean v) / Real.sqrt (variance v) + σ₁ - σ₁) ^ 2
      = σ₂ ^ 2 / variance v * (v i - mean v) ^ 2 := by
    intro i
    have h0 : σ₂ * (v i - mean v) / Real.sqrt (variance v) + σ₁ - σ₁
        = σ₂ * (v i - mean v) / Real.sqrt (variance v) := by ring
    rw [h0, div_pow, mul_pow, hr2]
    ring
  simp only [h1]
  rw [← Finset.mul_sum, ← card_mul_variance, mul_comm (Fintype.card ι : ℝ) (variance v),
    div_mul_eq_mul_div, div_div, mul_div_assoc,
    div_self (mul_ne_zero hv (card_pos (ι := ι)).ne'), mul_one]

theorem layerNorm_mem_meanVarSet (σ₁ σ₂ : ℝ) (v : ι → ℝ) (hv : variance v ≠ 0) :
    layerNorm σ₁ σ₂ v ∈ MeanVarSet σ₁ σ₂ :=
  ⟨layerNorm_mean σ₁ σ₂ v, layerNorm_variance σ₁ σ₂ v hv⟩

/-! ### The projection theorem -/

/-- Distance from the normalised vector back to `v`, in closed form. -/
lemma sqDist_layerNorm (σ₁ σ₂ : ℝ) (v : ι → ℝ) (hv : variance v ≠ 0) :
    sqDist (layerNorm σ₁ σ₂ v) v
      = (Fintype.card ι : ℝ) * (σ₂ - Real.sqrt (variance v)) ^ 2
        + (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2 := by
  have hβ : 0 < variance v := lt_of_le_of_ne (variance_nonneg v) (Ne.symm hv)
  have hr : Real.sqrt (variance v) ≠ 0 := (Real.sqrt_pos.mpr hβ).ne'
  have hr2 : Real.sqrt (variance v) ^ 2 = variance v := Real.sq_sqrt hβ.le
  have hterm : ∀ i, (layerNorm σ₁ σ₂ v i - v i) ^ 2
      = ((σ₂ / Real.sqrt (variance v) - 1) * (v i - mean v) + (σ₁ - mean v)) ^ 2 := by
    intro i
    simp only [layerNorm]
    ring
  unfold sqDist
  simp only [hterm]
  rw [sum_sq_add]
  have hA : ∑ i, ((σ₂ / Real.sqrt (variance v) - 1) * (v i - mean v)) ^ 2
      = (σ₂ / Real.sqrt (variance v) - 1) ^ 2 * ((Fintype.card ι : ℝ) * variance v) := by
    rw [card_mul_variance, Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  have hB : ∑ i, (σ₂ / Real.sqrt (variance v) - 1) * (v i - mean v) * (σ₁ - mean v) = 0 := by
    have : ∀ i, (σ₂ / Real.sqrt (variance v) - 1) * (v i - mean v) * (σ₁ - mean v)
        = ((σ₂ / Real.sqrt (variance v) - 1) * (σ₁ - mean v)) * (v i - mean v) := by
      intro i; ring
    simp only [this]
    rw [← Finset.mul_sum, sum_sub_mean, mul_zero]
  have hC : ∑ _i : ι, (σ₁ - mean v) ^ 2 = (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [hA, hB, hC, mul_zero, add_zero]
  have hkey : (σ₂ / Real.sqrt (variance v) - 1) ^ 2 * variance v
      = (σ₂ - Real.sqrt (variance v)) ^ 2 := by
    calc (σ₂ / Real.sqrt (variance v) - 1) ^ 2 * variance v
        = (σ₂ / Real.sqrt (variance v) - 1) ^ 2 * Real.sqrt (variance v) ^ 2 := by rw [hr2]
      _ = ((σ₂ / Real.sqrt (variance v) - 1) * Real.sqrt (variance v)) ^ 2 := by rw [mul_pow]
      _ = (σ₂ - Real.sqrt (variance v)) ^ 2 := by
          rw [sub_mul, div_mul_cancel₀ _ hr, one_mul]
  calc (σ₂ / Real.sqrt (variance v) - 1) ^ 2 * ((Fintype.card ι : ℝ) * variance v)
        + (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2
      = (Fintype.card ι : ℝ) * ((σ₂ / Real.sqrt (variance v) - 1) ^ 2 * variance v)
        + (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2 := by ring
    _ = (Fintype.card ι : ℝ) * (σ₂ - Real.sqrt (variance v)) ^ 2
        + (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2 := by rw [hkey]

/-- Distance from an arbitrary member of `MeanVarSet σ₁ σ₂` back to `v`. -/
lemma sqDist_of_mem (σ₁ σ₂ : ℝ) (v u : ι → ℝ) (hu₁ : mean u = σ₁) (hu₂ : variance u = σ₂ ^ 2) :
    sqDist u v
      = (Fintype.card ι : ℝ) * σ₂ ^ 2 + (Fintype.card ι : ℝ) * variance v
        + (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2
        - 2 * ∑ i, (u i - σ₁) * (v i - mean v) := by
  have hterm : ∀ i, (u i - v i) ^ 2
      = (((u i - σ₁) + (-(v i - mean v))) + (σ₁ - mean v)) ^ 2 := by
    intro i; ring
  unfold sqDist
  simp only [hterm]
  rw [sum_sq_add, sum_sq_add]
  have hz0 : ∑ i, (u i - σ₁) = 0 := by
    have := sum_sub_mean u
    rwa [hu₁] at this
  have hz2 : ∑ i, (u i - σ₁) ^ 2 = (Fintype.card ι : ℝ) * σ₂ ^ 2 := by
    have := card_mul_variance u
    rw [hu₂, hu₁] at this
    exact this.symm
  have hw0 : ∑ i, (-(v i - mean v)) = 0 := by
    rw [Finset.sum_neg_distrib, sum_sub_mean, neg_zero]
  have hw2 : ∑ i, (-(v i - mean v)) ^ 2 = (Fintype.card ι : ℝ) * variance v := by
    rw [card_mul_variance]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  have hzw : ∑ i, (u i - σ₁) * (-(v i - mean v)) = -∑ i, (u i - σ₁) * (v i - mean v) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  have hcross : ∑ i, ((u i - σ₁) + (-(v i - mean v))) * (σ₁ - mean v) = 0 := by
    rw [← Finset.sum_mul, Finset.sum_add_distrib, hz0, hw0, add_zero, zero_mul]
  have hC : ∑ _i : ι, (σ₁ - mean v) ^ 2 = (Fintype.card ι : ℝ) * (σ₁ - mean v) ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [hz2, hw2, hzw, hcross, hC]
  ring

/-- **Layer normalisation is the projection onto `MeanVarSet σ₁ σ₂`.**
Finite-dimensional form of Tai–Liu–Li–Chan, Theorem 3.1. -/
theorem layerNorm_sqDist_le (σ₁ σ₂ : ℝ) (v u : ι → ℝ) (hv : variance v ≠ 0) (hσ : 0 ≤ σ₂)
    (hu : u ∈ MeanVarSet σ₁ σ₂) :
    sqDist (layerNorm σ₁ σ₂ v) v ≤ sqDist u v := by
  obtain ⟨hu₁, hu₂⟩ := hu
  have hβ : 0 < variance v := lt_of_le_of_ne (variance_nonneg v) (Ne.symm hv)
  have hr2 : Real.sqrt (variance v) ^ 2 = variance v := Real.sq_sqrt hβ.le
  have hr0 : 0 ≤ Real.sqrt (variance v) := Real.sqrt_nonneg _
  have hn : 0 < (Fintype.card ι : ℝ) := card_pos
  -- Cauchy–Schwarz for the cross term.
  have hcs : ∑ i, (u i - σ₁) * (v i - mean v)
      ≤ (Fintype.card ι : ℝ) * σ₂ * Real.sqrt (variance v) := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun i => u i - σ₁)
      (fun i => v i - mean v)
    have hz2 : ∑ i, (u i - σ₁) ^ 2 = (Fintype.card ι : ℝ) * σ₂ ^ 2 := by
      have := card_mul_variance u
      rw [hu₂, hu₁] at this
      exact this.symm
    have hw2 : ∑ i, (v i - mean v) ^ 2 = (Fintype.card ι : ℝ) * variance v :=
      (card_mul_variance v).symm
    rw [hz2, hw2, ← hr2] at h
    refine le_of_sq_le_sq' ?_ (mul_nonneg (mul_nonneg hn.le hσ) hr0)
    calc (∑ i, (u i - σ₁) * (v i - mean v)) ^ 2
        ≤ (Fintype.card ι : ℝ) * σ₂ ^ 2
            * ((Fintype.card ι : ℝ) * Real.sqrt (variance v) ^ 2) := h
      _ = ((Fintype.card ι : ℝ) * σ₂ * Real.sqrt (variance v)) ^ 2 := by ring
  have hnr : (Fintype.card ι : ℝ) * Real.sqrt (variance v) ^ 2
      = (Fintype.card ι : ℝ) * variance v := by rw [hr2]
  rw [sqDist_layerNorm σ₁ σ₂ v hv, sqDist_of_mem σ₁ σ₂ v u hu₁ hu₂]
  nlinarith [hcs, hnr, hn]

/-! ### Scale invariance -/

lemma mean_smul (c : ℝ) (v : ι → ℝ) : mean (fun i => c * v i) = c * mean v := by
  unfold mean
  rw [← Finset.mul_sum, mul_div_assoc]

lemma variance_smul (c : ℝ) (v : ι → ℝ) :
    variance (fun i => c * v i) = c ^ 2 * variance v := by
  unfold variance
  rw [mean_smul, ← mul_div_assoc, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl ?_
  intro i _
  ring

/-- Layer normalisation ignores a positive rescaling of its input.  This is the
fact that makes the relaxation step `½(u^{(2+J)/M} + u^{2/M})` of the
operator-splitting scheme reproduce the plain residual sum after the closing
normalisation. -/
theorem layerNorm_smul (σ₁ σ₂ c : ℝ) (hc : 0 < c) (v : ι → ℝ) :
    layerNorm σ₁ σ₂ (fun i => c * v i) = layerNorm σ₁ σ₂ v := by
  funext i
  simp only [layerNorm]
  rw [mean_smul, variance_smul, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  have : c * v i - c * mean v = c * (v i - mean v) := by ring
  rw [this, mul_left_comm, mul_div_mul_left _ _ hc.ne']

/-! ### ReLU is the projection onto the non-negative orthant -/

/-- Rectified linear unit. -/
def relu (x : ℝ) : ℝ := max x 0

lemma relu_nonneg (x : ℝ) : 0 ≤ relu x := le_max_right x 0

lemma relu_of_nonneg {x : ℝ} (hx : 0 ≤ x) : relu x = x := max_eq_left hx

lemma relu_of_nonpos {x : ℝ} (hx : x ≤ 0) : relu x = 0 := max_eq_right hx

/-- Pointwise: `relu x` is the closest non-negative real to `x`. -/
lemma relu_sub_sq_le (x y : ℝ) (hy : 0 ≤ y) : (relu x - x) ^ 2 ≤ (y - x) ^ 2 := by
  rcases le_or_gt 0 x with hx | hx
  · rw [relu_of_nonneg hx, sub_self, zero_pow two_ne_zero]
    exact sq_nonneg _
  · rw [relu_of_nonpos hx.le]
    nlinarith [hy, hx, mul_nonneg hy (neg_nonneg.mpr hx.le)]

/-- The non-negative orthant `S₂ = {u | u ≥ 0}` of the paper. -/
def NonnegSet : Set (ι → ℝ) := {u | ∀ i, 0 ≤ u i}

/-- **ReLU is the projection onto `NonnegSet`** (Tai–Liu–Li–Chan, Section 3.3.3). -/
theorem relu_isProj (v u : ι → ℝ) (hu : u ∈ NonnegSet) :
    sqDist (fun i => relu (v i)) v ≤ sqDist u v := by
  unfold sqDist
  refine Finset.sum_le_sum ?_
  intro i _
  exact relu_sub_sq_le (v i) (u i) (hu i)

theorem relu_mem_nonnegSet (v : ι → ℝ) : (fun i => relu (v i)) ∈ NonnegSet :=
  fun i => relu_nonneg (v i)

end Stats

end

end Params
