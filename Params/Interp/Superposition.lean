import Mathlib

/-!
# Interference in superposition

A network that stores `N` features in `d < N` dimensions by assigning each
feature a direction `v i` cannot make those directions orthogonal.  If they are
`ε`-almost-orthogonal (`|⟨v i, v j⟩| ≤ ε` for `i ≠ j`, `‖v i‖² = 1`), then reading
feature `i` off the superposed vector `x = Σ_j s j • v j` by the linear read-out
`⟨v i, x⟩` recovers `s i` up to an interference term bounded by
`ε · (k − 1) · ‖s‖_∞` where `k` is the number of active features
(`readout_error_le`).  This is the quantitative content of "interference scales
with the sparsity of the active features" in the superposition literature
(Elhage et al. 2022; Open Problems in Mechanistic Interpretability, Q3).

Everything is an elementary finite sum; no assumption beyond the stated
hypotheses is used.
-/

namespace Params.Interp

noncomputable section

open Finset

variable {d N : Type*} [Fintype d] [Fintype N] [DecidableEq N]

/-- Inner product of two vectors in `d` coordinates. -/
def ip (u w : d → ℝ) : ℝ := ∑ k, u k * w k

/-- The superposed representation of a feature activation vector `s`. -/
def superpose (v : N → d → ℝ) (s : N → ℝ) : d → ℝ := fun k => ∑ j, s j * v j k

/-- Linear read-out of feature `i`. -/
def readout (v : N → d → ℝ) (i : N) (x : d → ℝ) : ℝ := ip (v i) x

/-- Directions are `ε`-almost-orthogonal and unit-norm. -/
structure AlmostOrthogonal (v : N → d → ℝ) (ε : ℝ) : Prop where
  norm_one : ∀ i, ip (v i) (v i) = 1
  cross_le : ∀ i j, i ≠ j → |ip (v i) (v j)| ≤ ε

lemma ip_superpose (v : N → d → ℝ) (s : N → ℝ) (i : N) :
    ip (v i) (superpose v s) = ∑ j, s j * ip (v i) (v j) := by
  unfold ip superpose
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl ?_
  intro k _
  refine Finset.sum_congr rfl ?_
  intro j _
  ring

/-- The read-out of feature `i` equals `s i` plus the interference from the other
active features. -/
theorem readout_eq (v : N → d → ℝ) (s : N → ℝ) (ε : ℝ) (hv : AlmostOrthogonal v ε) (i : N) :
    readout v i (superpose v s) = s i + ∑ j ∈ Finset.univ.erase i, s j * ip (v i) (v j) := by
  unfold readout
  rw [ip_superpose, ← Finset.add_sum_erase Finset.univ _ (Finset.mem_univ i), hv.norm_one i,
    mul_one]

/-- **Interference bound.** If at most `k` features are active (`s j = 0` outside a
set `S` of size `k`) then the read-out error is at most `ε (k − 1) ‖s‖_∞`. -/
theorem readout_error_le (v : N → d → ℝ) (s : N → ℝ) (ε : ℝ)
    (hv : AlmostOrthogonal v ε) (S : Finset N) (hS : ∀ j, j ∉ S → s j = 0)
    (B : ℝ) (hB : ∀ j, |s j| ≤ B) (i : N) (hi : i ∈ S) :
    |readout v i (superpose v s) - s i| ≤ ε * (S.card - 1) * B := by
  rw [readout_eq v s ε hv i, add_sub_cancel_left]
  have hsub : ∑ j ∈ Finset.univ.erase i, s j * ip (v i) (v j)
      = ∑ j ∈ S.erase i, s j * ip (v i) (v j) := by
    symm
    apply Finset.sum_subset
    · exact Finset.erase_subset_erase i (Finset.subset_univ S)
    · intro j hmem hj
      have hjS : j ∉ S := fun hjS =>
        hj (Finset.mem_erase.mpr ⟨(Finset.mem_erase.mp hmem).1, hjS⟩)
      rw [hS j hjS, zero_mul]
  rw [hsub]
  have hB0 : 0 ≤ B := (abs_nonneg _).trans (hB i)
  calc |∑ j ∈ S.erase i, s j * ip (v i) (v j)|
      ≤ ∑ j ∈ S.erase i, |s j * ip (v i) (v j)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j ∈ S.erase i, B * ε := by
        refine Finset.sum_le_sum ?_
        intro j hj
        rw [abs_mul]
        exact mul_le_mul (hB j) (hv.cross_le i j (Finset.ne_of_mem_erase hj).symm)
          (abs_nonneg _) hB0
    _ = (S.erase i).card * (B * ε) := by rw [Finset.sum_const, nsmul_eq_mul]
    _ = ε * (S.card - 1) * B := by
        rw [Finset.card_erase_of_mem hi, Nat.cast_sub (Finset.card_pos.mpr ⟨i, hi⟩),
          Nat.cast_one]
        ring

/-- Without interference (`ε = 0`, i.e. an orthonormal family) the read-out is exact. -/
theorem readout_exact_of_orthonormal (v : N → d → ℝ) (s : N → ℝ) (hv : AlmostOrthogonal v 0)
    (i : N) : readout v i (superpose v s) = s i := by
  rw [readout_eq v s 0 hv i]
  have : ∑ j ∈ Finset.univ.erase i, s j * ip (v i) (v j) = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    have h := hv.cross_le i j (Finset.ne_of_mem_erase hj).symm
    rw [abs_nonpos_iff.mp h, mul_zero]
  rw [this, add_zero]

/-- More features than dimensions forces interference: an orthonormal family in `d`
dimensions has at most `card d` members. -/
theorem card_le_of_orthonormal (v : N → d → ℝ) (hv : AlmostOrthogonal v 0) :
    Fintype.card N ≤ Fintype.card d := by
  have hli : LinearIndependent ℝ v := by
    rw [Fintype.linearIndependent_iff]
    intro g hg i
    have h : ip (v i) (∑ j, g j • v j) = ip (v i) 0 := by rw [hg]
    have hsum : ip (v i) (∑ j, g j • v j) = ∑ j, g j * ip (v i) (v j) := by
      unfold ip
      simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl ?_
      intro k _
      refine Finset.sum_congr rfl ?_
      intro j _
      ring
    rw [hsum] at h
    have hzero : ip (v i) 0 = 0 := by simp [ip]
    rw [hzero, Finset.sum_eq_single i] at h
    · rw [hv.norm_one i, mul_one] at h
      exact h
    · intro j _ hji
      rw [abs_nonpos_iff.mp (hv.cross_le i j hji.symm), mul_zero]
    · intro hi
      exact absurd (Finset.mem_univ i) hi
  have := hli.fintype_card_le_finrank
  simpa using this

end

end Params.Interp
