import Mathlib

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# When can a sparse dictionary be recovered?  The spark bound

Sparse dictionary learning (SAEs, transcoders) presumes that an activation
`x = D s` with a sparse code `s` determines `s`.  Whether it does is a
property of the dictionary `D` alone: writing `spark D` for the size of the
smallest linearly dependent set of columns, every representation with fewer
than `spark D / 2` non-zeros is the unique sparsest one
(Donoho–Elad 2003).  This is the precise form of the question "does SDL find
the true features?" from *Open Problems in Mechanistic Interpretability* (Q4):
below the spark bound the answer is yes for every algorithm that finds a
sparsest code; above it no algorithm can be guaranteed to.

`sparse_unique` is the self-contained statement: if `D s = D s'`, and the
supports of `s` and `s'` together contain fewer than `spark D` columns, then
`s = s'`.  `HasSpark D k` says every set of fewer than `k` columns is linearly
independent; it is the hypothesis, not an assumption about any model.
-/

namespace Params.Interp

noncomputable section

open Finset

variable {d N : Type*} [Fintype d] [Fintype N] [DecidableEq N]

/-- Apply a dictionary `D : Matrix d N ℝ` to a code `s`. -/
def decode (D : Matrix d N ℝ) (s : N → ℝ) : d → ℝ := D.mulVec s

/-- Support of a code. -/
def support (s : N → ℝ) : Finset N := Finset.univ.filter (fun j => s j ≠ 0)

/-- `D` has spark at least `k`: any fewer than `k` columns are linearly independent,
stated concretely as "a combination of fewer than `k` columns that vanishes has all
coefficients zero". -/
def HasSpark (D : Matrix d N ℝ) (k : ℕ) : Prop :=
  ∀ c : N → ℝ, (support c).card < k → D.mulVec c = 0 → c = 0

lemma mem_support {s : N → ℝ} {j : N} : j ∈ support s ↔ s j ≠ 0 := by
  unfold support
  simp

lemma support_sub_subset (s s' : N → ℝ) : support (s - s') ⊆ support s ∪ support s' := by
  intro j hj
  rw [mem_support] at hj
  rw [Finset.mem_union, mem_support, mem_support]
  by_contra h
  simp only [not_or, not_not] at h
  apply hj
  simp [h.1, h.2]

/-- **Uniqueness of sparse representations.** Two codes with the same decoding
whose supports jointly use fewer than `spark D` columns coincide. -/
theorem sparse_unique (D : Matrix d N ℝ) (k : ℕ) (hD : HasSpark D k) (s s' : N → ℝ)
    (h : decode D s = decode D s') (hcard : (support s).card + (support s').card < k) :
    s = s' := by
  have hdiff : D.mulVec (s - s') = 0 := by
    rw [Matrix.mulVec_sub]
    unfold decode at h
    rw [h, sub_self]
  have hsupp : (support (s - s')).card < k :=
    lt_of_le_of_lt ((Finset.card_le_card (support_sub_subset s s')).trans
      (Finset.card_union_le _ _)) hcard
  exact sub_eq_zero.mp (hD (s - s') hsupp hdiff)

/-- The usual corollary: a code with fewer than `k/2` non-zeros is the unique code of
that sparsity with its decoding. -/
theorem sparse_unique_half (D : Matrix d N ℝ) (k : ℕ) (hD : HasSpark D k) (s s' : N → ℝ)
    (h : decode D s = decode D s') (hs : 2 * (support s).card < k)
    (hs' : (support s').card ≤ (support s).card) : s = s' :=
  sparse_unique D k hD s s' h (by omega)

/-- Injectivity of decoding on `k/2`-sparse codes, phrased as a set statement. -/
theorem decode_injOn (D : Matrix d N ℝ) (k : ℕ) (hD : HasSpark D k) :
    Set.InjOn (decode D) {s | 2 * (support s).card < k} := by
  intro s hs s' hs' h
  simp only [Set.mem_setOf_eq] at hs hs'
  exact sparse_unique D k hD s s' h (by omega)

end

end Params.Interp
