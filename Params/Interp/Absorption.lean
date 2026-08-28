import Mathlib

/-!
# Sparsity is not the same as feature recovery: the absorption threshold

Two "true" features with unit directions `v₁, v₂` fire on a finite input
distribution: sometimes alone (inputs `0` and `1`), sometimes together
(input `2`).  A sparse dictionary that must reconstruct every input exactly can
either

* keep the atoms `{v₁, v₂}` and pay `‖s‖₁ = 2` on the co-occurring input, or
* add the composed atom `v₁ + v₂` and pay `1` there.

With an `L¹` penalty and exact reconstruction the composed ("absorbed")
dictionary is strictly cheaper as soon as co-occurrence has positive mass
(`absorbed_cheaper`), by exactly that mass
(`expectedCost_split_sub_absorbed`).  This is the finite, fully explicit
version of the observation behind Q4(b)/(e) of *Open Problems in Mechanistic
Interpretability*: an `L¹` sparsity proxy is minimised by composing true
features, so sparsity alone cannot certify that a dictionary's atoms are the
features.  Both dictionaries' costs are written out; nothing is assumed.
-/

namespace Params.Interp

/-- `L¹` cost of the split dictionary `{v₁, v₂}` on the three input types
(feature 1 alone, feature 2 alone, both). -/
def splitCost : Fin 3 → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => 2

/-- `L¹` cost of the absorbed dictionary `{v₁, v₂, v₁ + v₂}`: one active atom on every input. -/
def absorbedCost (_x : Fin 3) : ℝ := 1

/-- Expected cost under an input distribution `p`. -/
def expectedCost (p : Fin 3 → ℝ) (cost : Fin 3 → ℝ) : ℝ := ∑ x, p x * cost x

/-- The two expected costs differ by exactly the co-occurrence probability `p 2`. -/
theorem expectedCost_split_sub_absorbed (p : Fin 3 → ℝ) :
    expectedCost p splitCost - expectedCost p absorbedCost = p 2 := by
  simp only [expectedCost, Fin.sum_univ_three, splitCost, absorbedCost]
  ring

/-- **Absorption is favoured by `L¹`.** With positive co-occurrence mass the absorbed
dictionary has strictly smaller expected `L¹` cost. -/
theorem absorbed_cheaper (p : Fin 3 → ℝ) (h : 0 < p 2) :
    expectedCost p absorbedCost < expectedCost p splitCost := by
  have := expectedCost_split_sub_absorbed p
  linarith

/-- Conversely, without co-occurrence the two dictionaries tie: the penalty cannot
distinguish them. -/
theorem costs_eq_of_no_cooccurrence (p : Fin 3 → ℝ) (h : p 2 = 0) :
    expectedCost p absorbedCost = expectedCost p splitCost := by
  have := expectedCost_split_sub_absorbed p
  linarith

end Params.Interp
