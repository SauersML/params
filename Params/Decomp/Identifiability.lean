import Mathlib
import Params.Decomp.Basic

/-!
# Identifiability of rank-one parameter decompositions on one-hot inputs

The toy model of superposition decomposes a weight `W : Matrix m n ℝ`
(`m` hidden units, `n` features, typically `m < n`) into the *column
decomposition* `colComp W j`, the matrix that keeps column `j` of `W` and zeroes
the others.  Both APD (arXiv:2501.14926, Section 4.1) and SPD
(arXiv:2506.20790, Section 4.1) report recovering exactly this decomposition
and treat it as the ground truth.

This file proves that, under the idealised conditions the SPD paper describes
informally, the column decomposition is the *only* candidate:

`onehot_identifiability`.  Let `P : C → Matrix m n ℝ` be a family of
subcomponents with
* faithfulness: `Σ_c P c = W`;
* rank one: every `P c` is an outer product `u vᵀ`;
* exact causal separation on one-hot inputs: for every feature `j` exactly one
  subcomponent acts on `e_j` (all others send `e_j` to `0`, i.e. are exactly
  ablatable on that input);
and let `W` have non-zero, pairwise linearly independent columns.  Then every
`P c` is either `0` or a column component `colComp W j`, and every column
component occurs.

The converse — that the column decomposition satisfies all three conditions
whenever the columns are non-zero — is `colComp_faithful`, `colComp_outer` and
`colComp_oneHotSeparated`, so the hypotheses are jointly satisfiable and the
theorem is not vacuous.  No optimisation, no probability and no assumption
about training enters: the statement is purely about which decompositions
exist.
-/

namespace Params.Decomp

noncomputable section

open Finset

variable {m n C : Type*} [Fintype m] [Fintype n] [Fintype C] [DecidableEq n]

/-- The column component of `W` at feature `j`: column `j` of `W`, all other columns zero. -/
def colComp (W : Matrix m n ℝ) (j : n) : Matrix m n ℝ :=
  Matrix.of fun i j' => if j' = j then W i j else 0

@[simp] lemma colComp_apply (W : Matrix m n ℝ) (j : n) (i : m) (j' : n) :
    colComp W j i j' = if j' = j then W i j else 0 := rfl

/-- The column components sum to `W` (faithfulness of the ground truth). -/
theorem colComp_faithful (W : Matrix m n ℝ) : ∑ j, colComp W j = W := by
  ext i j'
  simp [Matrix.sum_apply]

/-- Each column component is an outer product (rank at most one). -/
theorem colComp_outer (W : Matrix m n ℝ) (j : n) :
    colComp W j = outer (fun i => W i j) (Pi.single j 1) := by
  ext i j'
  simp [Pi.single_apply, mul_ite]

/-- `P` acts on feature `j` when `P e_j ≠ 0`, i.e. column `j` of `P` is non-zero. -/
def ActsOn (P : Matrix m n ℝ) (j : n) : Prop := ∃ i, P i j ≠ 0

/-- Exact causal separation on one-hot inputs: every feature is acted on by exactly
one subcomponent. -/
def OneHotSeparated (P : C → Matrix m n ℝ) : Prop := ∀ j, ∃! c, ActsOn (P c) j

/-- No column of `W` is zero. -/
def NonzeroColumns (W : Matrix m n ℝ) : Prop := ∀ j, ∃ i, W i j ≠ 0

/-- Any two distinct columns of `W` are linearly independent. -/
def PairwiseIndepColumns (W : Matrix m n ℝ) : Prop :=
  ∀ j j', j ≠ j' → ∀ a b : ℝ, (∀ i, a * W i j + b * W i j' = 0) → a = 0 ∧ b = 0

/-- The column decomposition is one-hot separated whenever the columns are non-zero. -/
theorem colComp_oneHotSeparated (W : Matrix m n ℝ) (hnz : NonzeroColumns W) :
    OneHotSeparated (colComp W) := by
  intro j
  obtain ⟨i, hi⟩ := hnz j
  refine ⟨j, ⟨i, ?_⟩, ?_⟩
  · simp [hi]
  · rintro c ⟨i', hi'⟩
    by_contra hne
    apply hi'
    simp [Ne.symm hne]

/-- **Identifiability.**  Under faithfulness, rank one and exact one-hot causal
separation, a matrix with non-zero, pairwise independent columns has only the
column decomposition. -/
theorem onehot_identifiability (W : Matrix m n ℝ) (P : C → Matrix m n ℝ)
    (hfaith : ∑ c, P c = W)
    (hrank : ∀ c, ∃ u v, P c = outer u v)
    (hsep : OneHotSeparated P)
    (hind : PairwiseIndepColumns W) (hnz : NonzeroColumns W) :
    (∀ c, P c = 0 ∨ ∃ j, P c = colComp W j) ∧ (∀ j, ∃ c, P c = colComp W j) := by
  have hsep' : ∀ j, ∃ c, ActsOn (P c) j ∧ ∀ c', ActsOn (P c') j → c' = c := fun j => hsep j
  choose π _hπ hπu using hsep'
  -- inactive subcomponents have zero column `j`
  have hcol0 : ∀ c j, c ≠ π j → ∀ i, P c i j = 0 := by
    intro c j hc i
    by_contra h
    exact hc (hπu j c ⟨i, h⟩)
  -- the active subcomponent carries the whole column of `W`
  have hcolW : ∀ j i, P (π j) i j = W i j := by
    intro j i
    have h : (∑ c, P c) i j = W i j := by rw [hfaith]
    rw [Matrix.sum_apply] at h
    rw [← h]
    symm
    apply Finset.sum_eq_single (π j)
    · intro c _ hc
      exact hcol0 c j hc i
    · intro h'
      exact absurd (Finset.mem_univ _) h'
  -- a rank-one subcomponent cannot be active on two independent columns
  have hunique : ∀ c j j', π j = c → π j' = c → j = j' := by
    intro c j j' hj hj'
    by_contra hne
    obtain ⟨u, v, huv⟩ := hrank c
    have h1 : ∀ i, W i j = u i * v j := by
      intro i
      rw [← hcolW j i, hj, huv, outer_apply]
    have h2 : ∀ i, W i j' = u i * v j' := by
      intro i
      rw [← hcolW j' i, hj', huv, outer_apply]
    have hlin : ∀ i, v j' * W i j + (-(v j)) * W i j' = 0 := by
      intro i
      rw [h1, h2]
      ring
    obtain ⟨_, hvj⟩ := hind j j' hne (v j') (-(v j)) hlin
    obtain ⟨i, hi⟩ := hnz j
    apply hi
    rw [h1 i, neg_eq_zero.mp hvj, mul_zero]
  -- the active subcomponent of feature `j` is the column component
  have hP : ∀ j, P (π j) = colComp W j := by
    intro j
    ext i j'
    rw [colComp_apply]
    by_cases hjj : j' = j
    · rw [if_pos hjj, hjj]
      exact hcolW j i
    · rw [if_neg hjj]
      apply hcol0 (π j) j' _ i
      intro heq
      exact hjj (hunique (π j) j' j heq.symm rfl)
  refine ⟨?_, fun j => ⟨π j, hP j⟩⟩
  intro c
  by_cases hex : ∃ j, π j = c
  · obtain ⟨j, hj⟩ := hex
    right
    exact ⟨j, by rw [← hj]; exact hP j⟩
  · left
    ext i j
    rw [Matrix.zero_apply]
    exact hcol0 c j (fun h => hex ⟨j, h.symm⟩) i

/-- The number of non-zero subcomponents in such a decomposition is exactly the number
of features: the map `j ↦ π j` is injective, so there are at least `n` of them, and
`onehot_identifiability` shows every non-zero one is some `colComp W j`. -/
theorem card_features_le_card_subcomponents (W : Matrix m n ℝ) (P : C → Matrix m n ℝ)
    (hfaith : ∑ c, P c = W)
    (hrank : ∀ c, ∃ u v, P c = outer u v)
    (hsep : OneHotSeparated P)
    (hind : PairwiseIndepColumns W) (hnz : NonzeroColumns W) :
    Fintype.card n ≤ Fintype.card C := by
  obtain ⟨_, hex⟩ := onehot_identifiability W P hfaith hrank hsep hind hnz
  choose σ hσ using hex
  refine Fintype.card_le_of_injective σ ?_
  intro j j' hjj
  have h : colComp W j = colComp W j' := by rw [← hσ j, ← hσ j', hjj]
  by_contra hne
  obtain ⟨i, hi⟩ := hnz j
  have hentry := Matrix.ext_iff.mpr h i j
  simp [hne] at hentry
  exact hi hentry

end

end Params.Decomp
