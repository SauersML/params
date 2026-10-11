import Mathlib

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# Softmax

The softmax of a finite family of reals, and the elementary facts every later
attention theorem rests on: entries are strictly positive, rows sum to one,
softmax is invariant under adding a constant, and it commutes with
re-indexing by a permutation.

`softmaxRows` applies `softmax` to every row of a matrix; this is the
row-wise `Softmax_2` of Tai–Liu–Li–Chan (arXiv:2510.03989, eq. after
(2.7)), whose discrete form is `(Softmax_{2,dis}(A))_{k,l} = exp(a_{k,l}) /
Σ_l exp(a_{k,l})`.
-/

namespace Params

open Finset

section Softmax

variable {ι : Type*} [Fintype ι]

/-- Softmax of a finite family of reals. -/
noncomputable def softmax (a : ι → ℝ) : ι → ℝ :=
  fun i => Real.exp (a i) / ∑ j, Real.exp (a j)

lemma sum_exp_pos [Nonempty ι] (a : ι → ℝ) : 0 < ∑ j, Real.exp (a j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (a j)) Finset.univ_nonempty

lemma softmax_pos [Nonempty ι] (a : ι → ℝ) (i : ι) : 0 < softmax a i :=
  div_pos (Real.exp_pos _) (sum_exp_pos a)

lemma softmax_nonneg [Nonempty ι] (a : ι → ℝ) (i : ι) : 0 ≤ softmax a i :=
  (softmax_pos a i).le

/-- Softmax weights sum to one. -/
lemma sum_softmax [Nonempty ι] (a : ι → ℝ) : ∑ i, softmax a i = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (sum_exp_pos a).ne'

lemma softmax_le_one [Nonempty ι] (a : ι → ℝ) (i : ι) : softmax a i ≤ 1 := by
  have h := sum_softmax a
  have : softmax a i ≤ ∑ j, softmax a j :=
    Finset.single_le_sum (fun j _ => softmax_nonneg a j) (Finset.mem_univ i)
  linarith

/-- Softmax is invariant under adding the same constant to every logit. -/
lemma softmax_add_const (a : ι → ℝ) (c : ℝ) :
    softmax (fun i => a i + c) = softmax a := by
  funext i
  unfold softmax
  simp only [Real.exp_add]
  rw [← Finset.sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_pos c).ne'

/-- Softmax commutes with re-indexing by a permutation. -/
lemma softmax_comp_perm (a : ι → ℝ) (σ : Equiv.Perm ι) :
    softmax (fun i => a (σ i)) = fun i => softmax a (σ i) := by
  funext i
  simp only [softmax]
  exact congrArg (fun s => Real.exp (a (σ i)) / s)
    (Equiv.sum_comp σ (fun j => Real.exp (a j)))

/-- A softmax-weighted average never exceeds an upper bound of the values. -/
lemma softmax_weighted_le [Nonempty ι] (a x : ι → ℝ) (b : ℝ) (hb : ∀ i, x i ≤ b) :
    ∑ i, softmax a i * x i ≤ b := by
  calc ∑ i, softmax a i * x i ≤ ∑ i, softmax a i * b := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact mul_le_mul_of_nonneg_left (hb i) (softmax_nonneg a i)
    _ = (∑ i, softmax a i) * b := by rw [Finset.sum_mul]
    _ = b := by rw [sum_softmax, one_mul]

/-- A softmax-weighted average never falls below a lower bound of the values. -/
lemma le_softmax_weighted [Nonempty ι] (a x : ι → ℝ) (b : ℝ) (hb : ∀ i, b ≤ x i) :
    b ≤ ∑ i, softmax a i * x i := by
  calc b = (∑ i, softmax a i) * b := by rw [sum_softmax, one_mul]
    _ = ∑ i, softmax a i * b := by rw [Finset.sum_mul]
    _ ≤ ∑ i, softmax a i * x i := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact mul_le_mul_of_nonneg_left (hb i) (softmax_nonneg a i)

end Softmax

section Rows

variable {m n : Type*} [Fintype n]

/-- Row-wise softmax of a matrix (`Softmax_{2,dis}` of the paper). -/
noncomputable def softmaxRows (A : Matrix m n ℝ) : Matrix m n ℝ :=
  Matrix.of fun i => softmax (A i)

@[simp] lemma softmaxRows_apply (A : Matrix m n ℝ) (i : m) (j : n) :
    softmaxRows A i j = softmax (A i) j := rfl

lemma softmaxRows_sum [Nonempty n] (A : Matrix m n ℝ) (i : m) :
    ∑ j, softmaxRows A i j = 1 :=
  sum_softmax (A i)

lemma softmaxRows_nonneg [Nonempty n] (A : Matrix m n ℝ) (i : m) (j : n) :
    0 ≤ softmaxRows A i j :=
  softmax_nonneg (A i) j

end Rows

end Params
